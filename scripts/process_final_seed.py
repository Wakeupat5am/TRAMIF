"""Explicit seed-specific historical CPU processing; never opens C/Future.

No work on import. Real execution needs separate owner approval and an explicit
operator flag. Uses existing formulas and frozen reference/matched manifests.
"""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import io
import json
from pathlib import Path
import numpy as np
import torch
from scripts import train_final_seed as trainer
from scripts.frozen_encoder_extraction import Evidence,ROOT,VIEWS,PLAN,PLAN_HASH,plan,image,features,state_hash,json_bytes,sha256,require,preprocessing,csv_rows
from scripts.prepare_historical_logits import selection,infer_logits,fit_on_a,validate_bundle
from scripts.compute_historical_quality import quality
from scripts.mmd_measurement_preparation import fit_train_normalizer,fit_train_bandwidth,reference_pairs
from scripts.descriptive_mmd import class_conditional_mmd2
from scripts.freeze_historical_quality_weights import quality_weights
from scripts.evaluate_historical_episode import fixed_qd_weights

OUTPUT='data/derived/final_seed_processing'
FREEZE='data/derived/pre_future_package/20261017_034204_340305/package.json'
FREEZE_HASH='d163106db3669d8df9739d30070775e30f91f969957104a04ebbbf228affd14e'
SOURCES=('scripts/process_final_seed.py','scripts/frozen_encoder_extraction.py',
 'scripts/prepare_historical_logits.py','scripts/compute_historical_quality.py',
 'scripts/mmd_measurement_preparation.py','scripts/descriptive_mmd.py',
 'scripts/freeze_historical_quality_weights.py','scripts/evaluate_historical_episode.py',
 'scripts/uniform_fusion.py','scripts/calibrate_sbsmi.py','tests/test_final_seed_processing.py')


def code_hashes():
    return {**trainer.source_hashes(),**{p:sha256((ROOT/p).read_bytes()) for p in SOURCES}}


def binding_check(actual,expected):
    require(actual==expected,'Cross-seed/view/checkpoint binding mismatch')


def validate_embeddings(rows,ids,array,width):
    expected=[r['sha'] for r in rows]
    require(len(set(ids))==len(ids) and ids==expected,'Embedding IDs/order mismatch')
    require(array.shape==(len(rows),width) and array.dtype.kind=='f' and np.isfinite(array).all(),'Embedding shape/nonfinite mismatch')


def compute_view(stages,mapping,support,bundles,reference,ref_matrix,matched,ab_matrices,fitter=None):
    """Pure numerical composition. No re-selection, cap changes or encoder fitting."""
    for s in ('A','B'):validate_bundle(bundles[s],stages[s],mapping)
    t=fit_on_a(bundles['A'],stages['A'],mapping,**({} if fitter is None else dict(fitter=fitter)))
    q,losses,mask,_=quality(bundles['B'],stages['B'],mapping,support,t['temperature'])
    families=tuple(sorted(mapping,key=mapping.get))
    validate_embeddings(reference,[r['sha'] for r in reference],ref_matrix,ref_matrix.shape[1])
    by_id={r['sha']:ref_matrix[i] for i,r in enumerate(reference)}
    norm=fit_train_normalizer(reference,by_id,case='Final',families=families)
    band=fit_train_bandwidth(reference,by_id,normalizer=norm)
    transformed={}
    for stage in ('A','B'):
        validate_embeddings(matched[stage],[r['sha'] for r in matched[stage]],ab_matrices[stage],ref_matrix.shape[1])
        z=norm.transform(ab_matrices[stage])
        transformed[stage]={c:z[[i for i,r in enumerate(matched[stage]) if r['family']==c]] for c in support}
    d=class_conditional_mmd2(transformed['A'],transformed['B'],support=support,bandwidth_h=band['bandwidth_h'])
    return dict(temperature=t,quality=q,drift=asdict(d),d=d.mmd2/2),losses,mask,norm,band


def prepare_view(seed,view,execute=False):
    if not execute:raise PermissionError('Separate owner approval/--execute-posttraining required')
    trainer.validate_seed(seed);require(view in VIEWS,'Unknown view')
    run_binding=trainer.discover(seed,view)
    model,config,checkpoint=trainer.load_frozen_run(run_binding['run_dir'],seed,view,run_binding['complete_sha256'])
    e=Evidence();frozen=e.js(FREEZE,FREEZE_HASH)
    require(frozen['beta']==1 and frozen['gamma']==0 and not frozen['U_used'],'Frozen selection contract mismatch')
    e.read(frozen['beta_source'],frozen['beta_source_sha256'])
    stages,mapping,source,support=selection('Final',e)
    reference,mm,ss=plan('Final','train-reference',e)
    matched={s:plan('Final',s,e)[0] for s in ('A','B')}
    require(mapping==mm==source==ss==config['dataset']['class_to_idx']==frozen['class_to_idx'],'Mapping mismatch')
    require(len(mapping)==51 and len(support)==25 and len(stages['A'])==3799 and len(stages['B'])==4332,'Frozen counts mismatch')
    require(config['dataset']['config_id']==preprocessing(view,e)[1], 'Preprocessing mismatch')
    binding=dict(seed=seed,view=view,checkpoint_sha256=checkpoint,**run_binding)
    output=ROOT/OUTPUT/f'seed_{seed}'/view/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False);sources=code_hashes();before=state_hash(model)
    try:
        for path,h in sources.items():
            target=output/'source'/path;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(trainer.read_bound(ROOT,path,h))
        matrices={};bundles={};contents={}
        for stage,rows in [('train-reference',reference),*stages.items()]:
            ff=[];zz=[];ids=[];content=[]
            for start in range(0,len(rows),32):
                part=rows[start:start+32];decoded=[image(r,view,source,e) for r in part]
                batch=torch.stack([a[0] for a in decoded])
                ff.append(features(model,view,batch));ids.extend(r['sha'] for r in part);content.extend(a[1] for a in decoded)
                if stage!='train-reference':zz.append(infer_logits(model,batch,51))
            matrix=np.concatenate(ff);validate_embeddings(rows,ids,matrix,2048 if view=='sbsmi' else 512)
            matrices[stage]=matrix;contents[stage]=content
            arrays=dict(embeddings=matrix,sample_identifiers=np.array(ids),content_sha256=np.array(content))
            if stage!='train-reference':
                bundle=dict(logits=np.concatenate(zz),sample_identifiers=np.array(ids),labels=np.array([mapping[r['family']] for r in rows],dtype=np.int64),class_names=np.array(sorted(mapping,key=mapping.get)))
                validate_bundle(bundle,rows,mapping);bundles[stage]=bundle;arrays.update(bundle)
            with (output/(stage+'.npz')).open('xb') as f:np.savez(f,**arrays)
        require(state_hash(model)==before and all(p.grad is None for p in model.parameters()),'Frozen encoder state changed')
        capped={}
        for s in ('A','B'):
            index={r['sha']:i for i,r in enumerate(stages[s])}
            require(all(all(r[k]==stages[s][index[r['sha']]][k] for k in ('family','timestamp','group_id')) for r in matched[s]),'Matched source mismatch')
            capped[s]=matrices[s][[index[r['sha']] for r in matched[s]]]
        stats,losses,mask,norm,band=compute_view(stages,mapping,support,bundles,reference,matrices['train-reference'],matched,capped)
        pc=e.js(PLAN+'/checksums.json',PLAN_HASH)
        frozen_pairs=csv_rows(e.read(PLAN+'/Final/train_pairs.csv',pc['Final/train_pairs.csv']))
        require(band['pairs']==frozen_pairs,'Frozen TRAIN pairs changed')
        with (output/'statistics.npz').open('xb') as f:np.savez(f,mean=norm.mean,std=norm.std,keep=norm.keep,B_nll=losses,B_support_mask=mask)
        trainer.write_new_json(output/'statistics.json',dict(binding=binding,results=stats,bandwidth=band,normalizer=dict(rank=norm.rank,condition_number=norm.condition_number,reference_sha256=norm.reference_sha256,embeddings_sha256=norm.embeddings_sha256)))
        trainer.write_new_json(output/'selection.json',dict(stages=stages,reference=reference,matched=matched,class_to_idx=mapping,support=support,content_hashes=contents))
        require(code_hashes()==sources,'Source changed during processing')
        for p,h in e.hashes.items():require(sha256(e.read(p,h))==h,'Input changed')
        trainer.verify_run(run_binding['run_dir'],seed,view)
        checks={p.relative_to(output).as_posix():sha256(p.read_bytes()) for p in output.rglob('*') if p.is_file()}
        receipt=dict(status='PASS',binding=binding,files=checks,input_sha256=e.hashes,source_sha256=sources,
            encoder_state_sha256=before,temperature_fit='A only',Q='B full retained support; no Drift cap',D='primary descriptive MMD2/2',beta=1,gamma=0,future_access=False)
        trainer.write_new_json(output/'receipt.json',receipt)
        return dict(receipt=(output/'receipt.json').relative_to(ROOT).as_posix(),sha256=sha256((output/'receipt.json').read_bytes()))
    except BaseException as exc:
        trainer.write_new_json(output/'FAILED.json',dict(error=type(exc).__name__,message=str(exc)));raise


def finalize(seed,receipts,digests,execute=False):
    if not execute:raise PermissionError('Separate owner approval/--execute-posttraining required')
    trainer.validate_seed(seed);require(len(receipts)==len(digests)==3,'Three pinned view receipts required')
    selections=[];data={};provenance=[]
    for path,digest in zip(receipts,digests):
        p=Path(path);p=p if p.is_absolute() else ROOT/p;p=p.resolve(strict=True)
        require(p.is_relative_to(ROOT/OUTPUT/f'seed_{seed}'),'Wrong seed receipt root')
        r=json.loads(trainer.read_bound(ROOT,p.relative_to(ROOT),digest));binding=r['binding'];view=binding['view']
        require(r['status']=='PASS' and view in VIEWS and view not in data,'Invalid/duplicate view')
        binding_check(binding,dict(seed=seed,view=view,checkpoint_sha256=trainer.verify_run(binding['run_dir'],seed,view)[2]['selected']['sha256'],**trainer.discover(seed,view)))
        require(r['source_sha256']==code_hashes() and not (p.parent/'FAILED.json').exists(),'Stale/failed processing')
        for name,h in r['files'].items():trainer.read_bound(ROOT,(p.parent/name).relative_to(ROOT),h)
        stats=json.loads((p.parent/'statistics.json').read_bytes());binding_check(stats['binding'],binding)
        selections.append(json.loads((p.parent/'selection.json').read_bytes()))
        data[view]=stats['results'];provenance.append(dict(receipt=str(p.relative_to(ROOT)),sha256=digest))
    require(all(s==selections[0] for s in selections),'Three-view sample/content/class alignment mismatch')
    losses={v:data[v]['quality']['class_balanced_nll'] for v in VIEWS};drifts={v:data[v]['d'] for v in VIEWS}
    q=quality_weights(losses);qd=fixed_qd_weights(np.array([losses[v] for v in VIEWS]),np.array([drifts[v] for v in VIEWS]),1)
    out=ROOT/OUTPUT/f'seed_{seed}'/'freeze'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True,exist_ok=False)
    trainer.write_new_json(out/'weights.json',dict(seed=seed,view_order=VIEWS,Q=q.tolist(),QD=qd.tolist(),losses=losses,drifts=drifts,beta=1,gamma=0,inputs=provenance,selection_sha256=sha256(json_bytes(selections[0])),future_access=False))
    return dict(weights=(out/'weights.json').relative_to(ROOT).as_posix(),sha256=sha256((out/'weights.json').read_bytes()))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for command in ('prepare-view','finalize'):
        a=sub.add_parser(command);a.add_argument('--seed',type=int,choices=trainer.SEEDS,required=True);a.add_argument('--execute-posttraining',action='store_true')
        if command=='prepare-view':a.add_argument('--view',choices=VIEWS,required=True)
        else:a.add_argument('--receipt',action='append',required=True);a.add_argument('--receipt-sha256',action='append',required=True)
    a=p.parse_args()
    result=prepare_view(a.seed,a.view,a.execute_posttraining) if a.command=='prepare-view' else finalize(a.seed,a.receipt,a.receipt_sha256,a.execute_posttraining)
    print(json.dumps(result))


if __name__=='__main__':main()
