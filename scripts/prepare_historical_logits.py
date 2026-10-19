"""Full retained historical A/B logits, separate from capped MMD embeddings.

audit is metadata/saved-array only. prepare requires explicit inference flag,
CPU frozen prefix CNNs; fits temperature from A only, never computes Q/U/weights.
"""
import argparse
import io
import json
from datetime import datetime, timezone
import numpy as np
import torch
from scripts.frozen_encoder_extraction import (
    Evidence, ROOT, G2, PLAN, PLAN_HASH, RUNS, PREFIX, VIEWS, SOURCE_MAPPING_HASH,
    HISTORICAL_MANIFEST_HASH, csv_rows, json_bytes, sha256, require, read_bound,
    validate_mapping, image, load_encoder, state_hash, preprocessing,
)
from scripts.calibrate_sbsmi import fit_temperature, T_MIN, T_MAX

G2_HASH='c1cd9cabb29c16c60f88e5355dca0f0e0086fcbaf86578235e94fb55f92c45ba'
FINAL_GROUPS='data/derived/balanced_phase1_audit/20261013_233825_921873/members.csv'
FINAL_GROUP_HASH='1aa9020a353b6bd36ddb2a10ffa32bb1ca34d0339d6e3ca2ece4a4fdad8625e0'
MONTHS={'E1':('2019-11','2019-12'),'E2':('2019-12','2020-01'),'Final':('2020-02','2020-03')}


def validate_rows(rows,case,stage,mapping):
    require(case in MONTHS and stage in ('A','B'),'Forbidden case/stage')
    ids=[r['sha'] for r in rows]
    require(bool(rows) and len(set(ids))==len(ids),'Empty/duplicate selection')
    month=MONTHS[case][0 if stage=='A' else 1]
    for row in rows:
        dt=datetime.fromisoformat(row['timestamp'])
        require(dt.utcoffset() is not None and dt.utcoffset().total_seconds()==0,'UTC required')
        require(dt.strftime('%Y-%m')==month and row['stage']==stage,'Chronology mismatch')
        require(row['family'] in mapping and row['retained']=='True' and bool(row['group_id']),'Retention/group/mapping mismatch')


def selection(case,evidence):
    require(case in MONTHS,'Forbidden case')
    pc=evidence.js(PLAN+'/checksums.json',PLAN_HASH)
    mapping=evidence.js(PLAN+'/'+case+'/class_mapping.json',pc[case+'/class_mapping.json'])
    support=evidence.js(PLAN+'/'+case+'/support.json',pc[case+'/support.json'])
    validate_mapping(mapping,{'E1':32,'E2':42,'Final':51}[case])
    source=evidence.js('docs/class_mapping.json',SOURCE_MAPPING_HASH)['class_to_idx']
    canonical=csv_rows(evidence.read('data/manifests/train_validation_manifest.csv',HISTORICAL_MANIFEST_HASH))
    canonical={r['sha']:r for r in canonical}
    if case=='Final':
        require(mapping==source,'Final mapping mismatch')
        members=csv_rows(evidence.read(FINAL_GROUPS,FINAL_GROUP_HASH))
        rows=[]
        for r in members:
            require(r['month'] in MONTHS[case],'Nonhistorical group member')
            if r['retained']=='True':
                rows.append(dict(sha=r['sha'],timestamp=r['timestamp'],family=r['family'],
                    stage='A' if r['month']=='2020-02' else 'B',retained='True',
                    group_id=r['group_id'],content_sha256=r['disarmed_content_sha256']))
    else:
        gc=evidence.js(G2+'/checksums.json',G2_HASH)
        def bound(name):return evidence.read(f'{G2}/{case}/{name}',gc[case+'/'+name])
        require(mapping==json.loads(bound('class_mapping.json')),'G2 mapping mismatch')
        members=csv_rows(bound('AB_10_members.csv')); by_id={r['sha']:r for r in members}
        require(len(by_id)==len(members),'Duplicate graph vertices')
        rows=[]
        for stage in ('A','B'):
            kept=csv_rows(bound(stage+'_keep.csv'))
            for r in kept:
                require(r['decision']=='KEEP' and r['threshold']=='0','Primary retention mismatch')
                m=by_id[r['sha']]
                require(all(r[k]==m[k] for k in ('sha','family','timestamp','stage')),'Group identity mismatch')
                rows.append(dict(m))
            require({r['sha'] for r in kept}=={r['sha'] for r in members if r['stage']==stage and r['retained']=='True'},'Missing/extra retained graph IDs')
    stages={s:[r for r in rows if r['stage']==s] for s in ('A','B')}
    for stage,rr in stages.items():
        validate_rows(rr,case,stage,mapping)
        for r in rr:
            require(all(r[k]==canonical[r['sha']][k] for k in ('timestamp','family')),'Historical source mismatch')
        for family in support:
            ff=[r for r in rr if r['family']==family]
            require(len(ff)>=20 and len({r['group_id'] for r in ff})>=5,'Frozen support failed')
    require(not ({r['sha'] for r in stages['A']} & {r['sha'] for r in stages['B']}),'A/B identity overlap')
    return stages,mapping,source,support


def validate_bundle(arrays,rows,mapping):
    names=[c for c in sorted(mapping,key=mapping.get)]
    require(arrays['sample_identifiers'].tolist()==[r['sha'] for r in rows],'IDs/order mismatch')
    require(arrays['class_names'].tolist()==names,'Class order mismatch')
    labels=arrays['labels'];logits=arrays['logits']
    require(labels.dtype.kind in 'iu' and labels.tolist()==[mapping[r['family']] for r in rows],'Label mismatch')
    require(logits.shape==(len(rows),len(mapping)) and logits.dtype.kind=='f' and np.isfinite(logits).all(),'Invalid logits')


def infer_logits(model,batch,k):
    require(all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),'Model must be frozen eval')
    require(batch.dtype==torch.float32 and batch.ndim==4 and tuple(batch.shape[1:])==(1,64,64)
            and torch.isfinite(batch).all().item() and batch.min()>=0 and batch.max()<=1,'Invalid image')
    with torch.inference_mode(): out=model(batch)
    require(out.shape==(len(batch),k) and torch.isfinite(out).all().item() and not out.requires_grad,'Invalid logits output')
    return out.cpu().numpy().copy()


def fit_on_a(bundle,rows,mapping,fitter=fit_temperature):
    require(all(r['stage']=='A' for r in rows),'Temperature fitting requires A')
    validate_bundle(bundle,rows,mapping)
    t,boundary=fitter(torch.tensor(bundle['logits'],dtype=torch.float64),torch.tensor(bundle['labels'],dtype=torch.int64))
    require(np.isfinite(t) and T_MIN<=t<=T_MAX,'Invalid temperature')
    return dict(temperature=float(t),optimum_at_boundary=bool(boundary),bounds=[T_MIN,T_MAX],
                objective='standard unweighted NLL on complete retained A',fit_ids=[r['sha'] for r in rows])


def static_checkpoint(case,view,mapping,evidence):
    run,h=RUNS[case,view];base=run if case=='Final' else f'{PREFIX}/{case}/{view}/{run}'
    evidence.read(base+'/epoch_100.pt',h)  # Hash bytes only; no torch.load in audit.
    config=evidence.js(base+'/config.json');_,config_id=preprocessing(view,evidence)
    if case!='Final':
        c=evidence.js(base+'/COMPLETE.json');s=evidence.js(base+'/epoch_100.json')
        require(c['selected']==s and s['sha256']==h and s['epoch']==100,'Incomplete checkpoint')
        require(c['config_sha256']==sha256(json_bytes(config)),'Config checksum mismatch')
        binding=config['dataset']
        require(config['case']==case and config['view']==view and binding['cutoff_exclusive']==MONTHS[case][0], 'Wrong prefix cutoff')
        require(binding['class_to_idx']==mapping and binding['num_classes']==len(mapping) and binding['config_id']==config_id,'Checkpoint binding mismatch')
    else:
        require(config['class_to_idx']==mapping and config['num_classes']==51 and config['dataset_config_sha256']==config_id,'Final binding mismatch')
    return dict(path=base+'/epoch_100.pt',sha256=h,K=len(mapping),config_id=config_id)


def audit(evidence):
    registry=evidence.js('docs/baseline_artifacts.json')['entries']
    results=[]
    for case in MONTHS:
        stages,mapping,source,support=selection(case,evidence)
        observed={}; views=[]
        for view in VIEWS:
            cp=static_checkpoint(case,view,mapping,evidence)
            record=dict(view=view,checkpoint=cp)
            if case=='Final':
                entry=next(e for e in registry if e['view']==view)
                require(entry['checkpoint_sha256']==cp['sha256'],'Registry checkpoint mismatch')
                t=evidence.js(entry['temperature_file'],entry['temperature_file_sha256'])
                require(t['fit_month']=='2020-02' and t['fit_samples']==len(stages['A']) and t['class_to_idx']==mapping
                        and t['checkpoint_sha256']==cp['sha256'] and t['dataset_config_sha256']==cp['config_id']
                        and not t['future_test_used'] and T_MIN<=t['temperature']<=T_MAX,'Temperature provenance mismatch')
                for stage,month in zip(('A','B'),MONTHS[case]):
                    ev=next(e for e in entry['evaluations'] if e['month']==month)
                    rep=evidence.js(ev['report'],ev['report_sha256'])
                    require(rep['checkpoint_sha256']==cp['sha256'] and rep['class_to_idx']==mapping and rep['month']==month
                            and rep['selection']['threshold']==0 and rep['dataset_config_sha256']==cp['config_id'],'Prediction provenance mismatch')
                    path=str(__import__('pathlib').PurePosixPath(ev['report']).parent/'predictions.npz')
                    blob=evidence.read(path,ev['predictions_sha256'])
                    with np.load(io.BytesIO(blob),allow_pickle=False) as arrays:
                        # Bind saved order by IDs to immutable membership; no metrics read/recomputed.
                        ids=arrays['sample_identifiers'].tolist();rr={r['sha']:r for r in stages[stage]}
                        require(len(ids)==len(rr) and set(ids)==set(rr),'Saved prediction sample set mismatch')
                        ordered=[rr[s] for s in ids];validate_bundle(arrays,ordered,mapping)
                        if stage in observed: require(observed[stage]==ids,'Cross-view order mismatch')
                        observed[stage]=ids
                    if stage=='A':
                        require(t['calibration_source']['predictions_sha256']==ev['predictions_sha256'] and
                                t['calibration_source']['report_sha256']==ev['report_sha256'],'A calibration source mismatch')
                record.update(status='AVAILABLE',temperature=t['temperature'],temperature_file=entry['temperature_file'])
            else:
                # Explicit directory filename inventory, not registry-based absence assumption.
                base=ROOT/PREFIX/case/view
                candidates=[p.relative_to(ROOT).as_posix() for p in base.rglob('*') if p.is_file() and
                            any(word in p.name.lower() for word in ('logit','predict','temperature','calibrat'))]
                record.update(status='MISSING' if not candidates else 'UNKNOWN_CANDIDATES',candidates=candidates)
            views.append(record)
        results.append(dict(case=case,K=len(mapping),A=len(stages['A']),B=len(stages['B']),
                            B_support=sum(r['family'] in support for r in stages['B']),support=len(support),views=views))
    return results


def prepare(case,evidence,output):
    require(case in ('E1','E2'),'Final must reuse existing predictions and temperatures')
    stages,mapping,source,support=selection(case,evidence)
    (output/'selection.json').write_bytes(json_bytes(dict(case=case,stages=stages,class_to_idx=mapping,support=support)))
    results=[];content_by_stage={}
    for view in VIEWS:
        model,cp=load_encoder(case,view,mapping,evidence);before=state_hash(model)
        folder=output/view;folder.mkdir()
        for stage in ('A','B'):
            rows=stages[stage];parts=[];contents=[]
            for i in range(0,len(rows),32):
                items=[image(r,view,source,evidence) for r in rows[i:i+32]]
                require([h for _,h in items]==[r['content_sha256'] for r in rows[i:i+32]],'Content/group provenance mismatch')
                parts.append(infer_logits(model,torch.stack([x for x,_ in items]),len(mapping)))
                contents.extend(h for _,h in items)
            if stage in content_by_stage: require(contents==content_by_stage[stage],'Cross-view content mismatch')
            content_by_stage[stage]=contents
            bundle=dict(logits=np.concatenate(parts),labels=np.array([mapping[r['family']] for r in rows],dtype=np.int64),
                        sample_identifiers=np.array([r['sha'] for r in rows]),class_names=np.array(sorted(mapping,key=mapping.get)))
            validate_bundle(bundle,rows,mapping)
            file=folder/(stage+'.npz');np.savez(file,**bundle)
            if stage=='A':
                temperature=fit_on_a(bundle,rows,mapping)
                temperature.update(case=case,view=view,checkpoint=cp,class_to_idx=mapping,
                                   fit_month=MONTHS[case][0],A_sha256=sha256(file.read_bytes()))
                (folder/'temperature.json').write_bytes(json_bytes(temperature))
            print(f'{case}/{view}/{stage}: saved {bundle["logits"].shape}',flush=True)
        require(state_hash(model)==before and all(p.grad is None for p in model.parameters()),'Model state mutation')
        results.append(dict(view=view,checkpoint=cp,state_unchanged=True,temperature=temperature['temperature']))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('audit')
    p=sub.add_parser('prepare');p.add_argument('--case',choices=['E1','E2'],required=True);p.add_argument('--execute-inference',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare' and not args.execute_inference: parser.error('Explicit --execute-inference required')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    e=Evidence()
    for path in ('scripts/prepare_historical_logits.py','scripts/frozen_encoder_extraction.py',
                 'scripts/calibrate_sbsmi.py','scripts/episode_training_dataset.py','tests/test_historical_logits.py'): e.read(path)
    output=ROOT/'data/derived/historical_logits_preparation'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    (output/'STARTED.json').write_bytes(json_bytes(vars(args)))
    try:
        result=audit(e) if args.command=='audit' else prepare(args.case,e,output)
        for path,h in e.hashes.items():read_bound(ROOT,path,h)
        (output/'receipt.json').write_bytes(json_bytes(dict(status='PASS',command=vars(args),results=result,
            input_source_hashes=e.hashes,torch=str(torch.__version__),numpy=np.__version__,device='cpu',real_U=False,Q_computed=False,future_access=False)))
        (output/'checksums.json').write_bytes(json_bytes({p.relative_to(output).as_posix():sha256(p.read_bytes()) for p in output.rglob('*') if p.is_file()}))
        print('Receipt:',output/'receipt.json')
        if args.command=='audit':
            for row in result: print(row['case'],row['A'],row['B'],[(v['view'],v['status']) for v in row['views']])
    except BaseException as exc:
        (output/'FAILED.json').write_bytes(json_bytes(dict(error=type(exc).__name__,message=str(exc))))
        raise


if __name__=='__main__':main()
