"""D054 read-only artifact verification; no forward, fitting or kernel evaluation."""
import csv
from collections import Counter
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import torch
from scripts.frozen_encoder_extraction import ROOT,VIEWS,PLAN,PLAN_HASH,Evidence,plan,preprocessing,require,json_bytes,sha256
from scripts.prepare_historical_logits import selection,validate_bundle
from scripts.compute_historical_quality import final_sources
from scripts.train_final_seed import FinalTrainDataset,RUN_SPEC,OUTPUT as TRAIN_ROOT
from scripts.train_prefix_view import EXPECTED_ADAM
from scripts.process_final_seed import FREEZE,FREEZE_HASH,OUTPUT as PROCESS_ROOT
from scripts.freeze_historical_quality_weights import QUALITY,QUALITY_HASH
from scripts.run_frozen_descriptive_drift import PREP,PREP_HASH
from scripts.prepare_frozen_drift_measurements import VERIFIED,VERIFIED_HASH,bound_embeddings
from scripts.mmd_measurement_preparation import train_reference,encoded,weighted_median

OBS='data/derived/observed_descriptive_drift/20261016_122644_350078/report.json'
OBS_HASH='873d2b893d07fb41d74415299d116627c70317b3375abc6fb07d28b61019b393'
SEEDS=(42,43,44,45,46)
ATOL=1e-12


def close(actual,expected,message):
    a,b=np.asarray(actual),np.asarray(expected)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,atol=ATOL,rtol=ATOL),message)


class Audit(Evidence):
    def read(self,path,expected=None):
        path=str(path).replace('\\','/');p=Path(path)
        if p.is_absolute():path=p.relative_to(ROOT).as_posix()
        require(not path.startswith(('data/raw/','data/derived/future','data/manifests/future')),'Forbidden Future/raw input')
        return super().read(path,expected)

    def arrays(self,path,digest=None):
        with np.load(io.BytesIO(self.read(path,digest)),allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def one(paths,description):
    paths=list(paths);require(len(paths)==1,f'AMBIGUOUS/UNKNOWN {description}: candidates={len(paths)}');return paths[0]


def state_digest(state):
    h=hashlib.sha256()
    for name,value in state.items():
        require(isinstance(value,torch.Tensor) and torch.isfinite(value).all().item(),'Nonfinite/invalid checkpoint state')
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def source_snapshot(e,run,hashes,*,legacy=False):
    differences=[]
    for name,h in hashes.items():
        name=name.replace('\\','/')
        if not legacy:e.read(run+'/source/'+name,h)
        current=e.read(name)
        if sha256(current)!=h:differences.append(name)
    # No frozen source is executed. A source difference is a review need, not proof
    # that a historical checkpoint is corrupt. Stop its affected verification.
    require(not differences,'UNKNOWN: current source differs; inspect execution contract: '+str(differences))


def checkpoint(e,seed,view,package,expected_binding):
    if seed==42:
        path=package['checkpoint_paths'][view];run=str(Path(path).parent).replace('\\','/')
        raw=e.read(path,package['checkpoint_sha256'][view]);config=e.js(run+'/config.json',package['source_input_hashes'][run+'/config.json'])
        require(config['seed']==42 and config['epochs']==100 and config['num_classes']==51 and config['train_samples']==18061 and config['class_to_idx']==expected_binding['class_to_idx'] and config['dataset_config_sha256']==expected_binding['config_id'],'Legacy TRAIN/config mismatch')
        source_snapshot(e,run,config['source_sha256'],legacy=True)
    else:
        complete_path=one((ROOT/TRAIN_ROOT/f'seed_{seed}'/view).glob('*/COMPLETE.json'),f'seed{seed}/{view} checkpoint')
        run=complete_path.parent.relative_to(ROOT).as_posix();require(not (complete_path.parent/'FAILED.json').exists(),'Training FAILED marker')
        complete=e.js(run+'/COMPLETE.json');config=e.js(run+'/config.json');selected=e.js(run+'/epoch_100.json')
        require(config['seed']==seed and config['view']==view and config['run_spec']==dict(RUN_SPEC,seed=seed) and config['dataset']==expected_binding,'Seed/view/TRAIN binding mismatch')
        require(json_bytes(config['adam_defaults'])==json_bytes(EXPECTED_ADAM),'Recorded Adam defaults mismatch')
        require(complete['config_sha256']==sha256(json_bytes(config)) and complete['selected']==selected and len(complete['epochs'])==100 and [r['epoch'] for r in complete['epochs']]==list(range(1,101)),'Incomplete/misbound run')
        for epoch in complete['epochs']:
            require(epoch['samples']==18061 and epoch['steps']==565,'Epoch sample coverage metadata mismatch')
            require(e.js(run+f'/epoch_{epoch["epoch"]:03d}.json')==epoch,'Epoch receipt mismatch')
        require(selected['epoch']==100 and selected['checkpoint']=='epoch_100.pt','Wrong selected epoch')
        raw=e.read(run+'/epoch_100.pt',selected['sha256']);path=run+'/epoch_100.pt'
        source_snapshot(e,run,config['code_sha256'])
    payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    require(config['future_test_used'] is False,'Checkpoint Future provenance mismatch')
    require(payload['epoch']==100 and json_bytes(payload['config'])==json_bytes(config),'Embedded config/epoch mismatch')
    if seed!=42:require(payload['config_sha256']==sha256(json_bytes(config)),'Checkpoint config hash mismatch')
    width=2048 if view=='sbsmi' else 512;state=payload['model_state_dict']
    require(tuple(state['classifier.weight'].shape)==(51,width) and tuple(state['classifier.bias'].shape)==(51,),'K51 classifier shape mismatch')
    return dict(seed=seed,view=view,status='ARTIFACT_VERIFIED',checkpoint=path,checkpoint_sha256=sha256(raw),
        config_sha256=sha256(json_bytes(config)),run=run,state_sha256=state_digest(state),K=51,epoch=100,TRAIN_samples=18061,cutoff_exclusive='2020-02',
        complete_sha256=e.hashes.get(run+'/COMPLETE.json'),contract='legacy registered' if seed==42 else 'COMPLETE+snapshot+payload')


def align_bundle(bundle,rows,mapping):
    ids=bundle['sample_identifiers'].tolist();expected=[r['sha'] for r in rows]
    require(len(set(ids))==len(ids) and len(set(expected))==len(expected) and set(ids)==set(expected),'Missing/duplicate/extra canonical IDs')
    index={sid:i for i,sid in enumerate(ids)};order=[index[sid] for sid in expected]
    aligned=dict(logits=bundle['logits'][order],labels=bundle['labels'][order],sample_identifiers=bundle['sample_identifiers'][order],class_names=bundle['class_names'])
    validate_bundle(aligned,rows,mapping)
    return aligned


def independent_q(bundle,rows,mapping,support,temperature):
    bundle=align_bundle(bundle,rows,mapping)
    require(np.isfinite(temperature) and .05<=temperature<=20,'Invalid T')
    z=np.asarray(bundle['logits'],dtype=np.float64)/temperature;maximum=z.max(1)
    nll=maximum+np.log(np.exp(z-maximum[:,None]).sum(1))-z[np.arange(len(z)),bundle['labels']]
    family=np.array([r['family'] for r in rows]);mask=np.isin(family,support)
    means=[]
    for c in support:
        take=family==c;require(take.any(),'Missing support family');means.append(float(nll[take].mean()))
    require(np.isfinite(nll).all() and (nll>=0).all(),'Invalid NLL')
    return dict(class_balanced_nll=float(np.mean(means)),standard_nll_support=float(nll[mask].mean()),standard_nll_all_B=float(nll.mean()),per_family=means),nll,mask


def verify_q(saved,bundle,rows,mapping,support,t,saved_losses,saved_mask):
    close(saved['temperature'],t,'Saved quality temperature mismatch')
    q,loss,mask=independent_q(bundle,rows,mapping,support,t)
    for key in ('class_balanced_nll','standard_nll_support','standard_nll_all_B'):close(saved[key],q[key],'Q aggregation mismatch: '+key)
    require(saved['K']==51 and saved['support_families']==25 and saved['B_support_samples']==int(mask.sum()) and saved['B_retained_samples']==len(rows),'Q counts mismatch')
    require([r['family'] for r in saved['per_family']]==list(support),'Per-family order mismatch')
    close([r['mean_nll'] for r in saved['per_family']],q['per_family'],'Per-family loss mismatch')
    close(saved_losses,loss,'Per-sample loss readback mismatch');require(np.array_equal(mask,saved_mask),'Q support mask mismatch')
    return q['class_balanced_nll']


def verify_normalization(reference,matrix,mapping,arrays,metadata,bandwidth,frozen_pairs):
    require(arrays['mean'].dtype==arrays['std'].dtype==np.float64 and arrays['keep'].dtype==bool and arrays['mean'].ndim==1 and arrays['mean'].shape==arrays['std'].shape==arrays['keep'].shape,'Normalizer dtype/shape mismatch')
    require(matrix.shape==(len(reference),len(arrays['mean'])) and np.isfinite(matrix).all(),'TRAIN embedding shape/finite mismatch')
    canonical=train_reference(reference,case='Final',families=tuple(sorted(mapping,key=mapping.get)))
    index={r['sha']:i for i,r in enumerate(reference)};x=np.asarray(matrix[[index[r['sha']] for r in canonical]],dtype=np.float64)
    require(metadata['reference_sha256']==sha256(encoded([[r['sha'],r['family'],r['timestamp'],r['stage']] for r in canonical])),'TRAIN reference digest mismatch')
    require(metadata['embeddings_sha256']==sha256(x.astype('<f8').tobytes()),'Seed-specific embedding payload mismatch')
    indices=[[i for i,r in enumerate(canonical) if r['family']==c] for c in sorted(mapping,key=mapping.get)]
    mu=np.mean([x[idx].mean(0) for idx in indices],axis=0);std=np.sqrt(np.mean([np.square(x[idx]-mu).mean(0) for idx in indices],axis=0));keep=std>1e-8
    close(arrays['mean'],mu,'TRAIN-only mean mismatch');close(arrays['std'],std,'TRAIN-only std mismatch');require(np.array_equal(arrays['keep'],keep) and keep.any(),'TRAIN coordinate mask mismatch')
    require(bandwidth['pairs']==frozen_pairs and bandwidth['reference_sha256']==metadata['reference_sha256'] and bandwidth['embeddings_sha256']==metadata['embeddings_sha256'],'Bandwidth provenance mismatch')
    z=(x[:,keep]-mu[keep])/std[keep];lookup={r['sha']:i for i,r in enumerate(canonical)}
    distances=[float(np.square(z[lookup[p['left']]]-z[lookup[p['right']]]).sum()/z.shape[1]) for p in frozen_pairs]
    close(bandwidth['distances'],distances,'Saved TRAIN pair distances mismatch')
    counts=Counter(p['family'] for p in frozen_pairs)
    median=weighted_median(distances,[Fraction(1,len(mapping)*counts[p['family']]) for p in frozen_pairs])
    require(median>0,'Degenerate saved bandwidth');close(bandwidth['bandwidth_h'],median,'Saved TRAIN weighted median mismatch')
    return sha256(b''.join(arrays[k].tobytes() for k in ('mean','std','keep')))


def verify_drift(per_family,mmd2,d):
    require(len(per_family)==25,'Drift requires S25')
    values=[]
    for row in per_family:
        require(all(np.isfinite(row[k]) and -ATOL<=row[k]<=1+ATOL for k in ('mean_xx','mean_yy','mean_xy')),'Invalid kernel mean')
        raw=row['mean_xx']+row['mean_yy']-2*row['mean_xy']
        require(-64*np.finfo(float).eps<=raw<=2+ATOL,'Invalid RBF MMD terms')
        close(row['raw_mmd2'],raw,'MMD term arithmetic mismatch');close(row['mmd2'],max(0,raw),'MMD clipping mismatch');values.append(row['mmd2'])
    close(mmd2,float(np.mean(values)),'Equal-family MMD aggregation mismatch');close(d,mmd2/2,'Drift scale mismatch')


def legacy_drift_header(observed):
    require(observed['status']=='PASS_TECHNICAL_COMPUTATION' and observed['null_standardization'] is False and observed['future_access'] is False,'Legacy descriptive D contract mismatch')


def legacy_preparation(e,observed):
    report=e.js(PREP+'/report.json',PREP_HASH)
    require(report['status']=='PASS' and report['observed_mmd'] is False,'Invalid legacy preparation')
    return e.js(PREP+'/checksums.json',observed['source_input_hashes'][PREP+'/checksums.json'])


def frozen_weights(loss,drift,q,qd):
    loss=np.asarray(loss,dtype=np.float64);drift=np.asarray(drift,dtype=np.float64)
    require(loss.shape==drift.shape==(3,) and np.isfinite(loss).all() and np.isfinite(drift).all() and (loss>=0).all() and (drift>=0).all() and (drift<=1).all(),'Invalid signals')
    for score,saved in ((-loss,q),(-loss-drift,qd)):
        expected=np.exp(score-score.max());expected/=expected.sum();close(saved,expected,'Frozen weights mismatch')
        require(np.isfinite(saved).all() and np.all(np.asarray(saved)>=0) and abs(sum(saved)-1)<=ATOL,'Invalid simplex weights')


def summarize(rows):
    require(len(rows)==15 and {(r['seed'],r['view']) for r in rows}=={(s,v) for s in SEEDS for v in VIEWS},'All five seeds required; no selection')
    result=[]
    for view in VIEWS:
        rr=[r for r in rows if r['view']==view]
        for metric in ('temperature','L','mmd2','d','Q_weight','QD_weight'):
            a=np.array([r[metric] for r in rr],dtype=np.float64);require(np.isfinite(a).all(),'Nonfinite summary')
            result.append(dict(view=view,metric=metric,N=5,mean=float(a.mean()),SD=float(a.std(ddof=1)),min=float(a.min()),max=float(a.max())))
    return result


def new_processing(e,seed,checkpoints,stages,mapping,support,reference,matched,pairs):
    weight_path=one((ROOT/PROCESS_ROOT/f'seed_{seed}'/'freeze').glob('*/weights.json'),f'seed{seed} frozen weights')
    w=e.js(weight_path);require(w['seed']==seed and w['view_order']==list(VIEWS) and w['beta']==1 and w['gamma']==0 and w['future_access'] is False,'Frozen weight contract mismatch')
    require(len(w['inputs'])==3,'Three processing receipts required');rows=[];seen=set();selections=[];contents={}
    for item in w['inputs']:
        receipt_path=item['receipt'].replace('\\','/');receipt=e.js(receipt_path,item['sha256']);b=receipt['binding'];view=b['view']
        require(view in VIEWS and view not in seen and b['seed']==seed,'Cross-seed/duplicate processing receipt');seen.add(view)
        cp=checkpoints[seed,view];run=str(Path(receipt_path).parent).replace('\\','/')
        require(Path(run).is_relative_to(Path(PROCESS_ROOT)/f'seed_{seed}'/view) and not (ROOT/run/'FAILED.json').exists(),'Wrong/failed processing root')
        require(b['run_dir']==cp['run'] and b['complete_sha256']==cp['complete_sha256'] and b['checkpoint_sha256']==cp['checkpoint_sha256'] and receipt['encoder_state_sha256']==cp['state_sha256'],'Cross-seed checkpoint/state mismatch')
        require(receipt['status']=='PASS' and receipt['temperature_fit']=='A only' and receipt['beta']==1 and receipt['gamma']==0 and receipt['future_access'] is False,'Processing contract mismatch')
        for name,h in receipt['files'].items():e.read(run+'/'+name,h)
        source_snapshot(e,run,receipt['source_sha256'])
        # Recheck metadata/source dependencies, not the already-verified image corpus.
        for name,h in receipt['input_sha256'].items():
            if name.startswith(tuple('data/derived/'+folder+'/' for folder in ('sbsmi_dataset_v1','raw_byte_dataset_v1','entropy_dataset_v1'))) and not name.endswith('/config.json'):continue
            e.read(name,h)
        saved=e.js(run+'/selection.json');selections.append(saved)
        require(saved['stages']==stages and saved['reference']==reference and saved['matched']==matched and saved['class_to_idx']==mapping and saved['support']==support,'Frozen canonical selection mismatch')
        require(sha256(json_bytes(saved))==w['selection_sha256'],'Selection freeze hash mismatch')
        stats=e.js(run+'/statistics.json');require(stats['binding']==b,'Statistics binding mismatch')
        arrays=e.arrays(run+'/statistics.npz');t=stats['results']['temperature']
        require(t['fit_ids']==[r['sha'] for r in stages['A']] and t['bounds']==[.05,20] and t['objective']=='standard unweighted NLL on complete retained A','Temperature fit provenance mismatch')
        bundles={}
        for stage,rr in [('train-reference',reference),*stages.items()]:
            a=e.arrays(run+'/'+stage+'.npz');ids=a['sample_identifiers'].tolist()
            require(ids==[r['sha'] for r in rr] and len(set(ids))==len(ids),'Embedding/logit ID alignment mismatch')
            require(a['embeddings'].shape==(len(rr),2048 if view=='sbsmi' else 512) and np.isfinite(a['embeddings']).all(),'Invalid embeddings')
            require(a['content_sha256'].tolist()==saved['content_hashes'][stage],'Content identity mismatch')
            contents[view,stage]=saved['content_hashes'][stage]
            if stage!='train-reference':
                validate_bundle(a,rr,mapping);require(a['content_sha256'].tolist()==[r['content_sha256'] for r in rr],'Canonical disarmed-content mismatch');bundles[stage]=a
            else:ref_matrix=a['embeddings']
        q=stats['results']['quality'];L=verify_q(q,bundles['B'],stages['B'],mapping,support,t['temperature'],arrays['B_nll'],arrays['B_support_mask'])
        normhash=verify_normalization(reference,ref_matrix,mapping,arrays,stats['normalizer'],stats['bandwidth'],pairs)
        drift=stats['results']['drift'];require(drift['support']==support,'Drift support mismatch');d=stats['results']['d']
        verify_drift(drift['per_family'],drift['mmd2'],d);close(w['losses'][view],L,'Weight loss provenance mismatch');close(w['drifts'][view],d,'Weight drift provenance mismatch')
        i=VIEWS.index(view);rows.append(dict(seed=seed,view=view,checkpoint_status='ARTIFACT_VERIFIED',processing_status='ARTIFACT_VERIFIED',temperature=t['temperature'],L=L,mmd2=drift['mmd2'],d=d,Q_weight=w['Q'][i],QD_weight=w['QD'][i],normalizer_sha256=normhash,reference_embeddings_sha256=stats['normalizer']['embeddings_sha256'],processing_receipt=receipt_path,weights_file=weight_path.relative_to(ROOT).as_posix()))
    require(all(s==selections[0] for s in selections),'Cross-view canonical/content selection mismatch')
    frozen_weights([w['losses'][v] for v in VIEWS],[w['drifts'][v] for v in VIEWS],w['Q'],w['QD'])
    return rows,contents


def legacy_processing(e,package,checkpoints,stages,mapping,support,reference,matched,pairs):
    completion=e.js(QUALITY+'/completion.json',QUALITY_HASH);require(completion['status']=='PASS','Legacy quality incomplete')
    require(e.js(QUALITY+'/Final/selection.json')==dict(B=stages['B'],class_to_idx=mapping,support=support),'Legacy Q selection mismatch')
    sources=final_sources(stages,mapping,e);observed=e.js(OBS,OBS_HASH)
    legacy_drift_header(observed)
    pc=legacy_preparation(e,observed);registry=e.js(VERIFIED,VERIFIED_HASH);rows=[];contents={}
    for view,bundle,t,provenance in sources:
        cp=checkpoints[42,view];require(provenance['checkpoint']['sha256']==cp['checkpoint_sha256'],'Legacy checkpoint/Q mismatch')
        qpath=QUALITY+'/Final/'+view+'/quality.json'
        q=e.js(qpath,completion['source_input_hashes'].get(qpath))
        # completion.results independently embeds these exact loss values; output
        # files are also protected by the original checksum manifest below.
        checks=e.js(QUALITY+'/checksums.json');e.read(qpath,checks['Final/'+view+'/quality.json'])
        losses=e.arrays(QUALITY+'/Final/'+view+'/sample_losses.npz',checks['Final/'+view+'/sample_losses.npz'])
        require(losses['sample_identifiers'].tolist()==[r['sha'] for r in stages['B']] and losses['labels'].tolist()==[mapping[r['family']] for r in stages['B']],'Legacy loss IDs/labels mismatch')
        L=verify_q(q,bundle,stages['B'],mapping,support,t,losses['nll'],losses['in_support']);close(t,package['temperatures'][view],'Legacy T mismatch')
        prefix='Final/'+view+'/'
        meta=e.js(PREP+'/'+prefix+'parameters.json',pc[prefix+'parameters.json']);band=e.js(PREP+'/'+prefix+'bandwidth.json',pc[prefix+'bandwidth.json']);arrays=e.arrays(PREP+'/'+prefix+'normalizer.npz',pc[prefix+'normalizer.npz'])
        require(meta['normalization_fit_stage']==meta['bandwidth_fit_stage']=='train' and meta['families']==sorted(mapping,key=mapping.get),'Legacy TRAIN normalization contract mismatch')
        result,matrix=bound_embeddings(e,registry,'Final','train-reference',view)
        require(result['samples']==reference and result['class_to_idx']==mapping and result['views'][view]['sha256']==cp['checkpoint_sha256'] and result['views'][view]['state_sha256']==cp['state_sha256'],'Legacy reference encoder mismatch')
        normhash=verify_normalization(reference,matrix,mapping,arrays,meta,band,pairs)
        for stage in ('train-reference','A','B'):
            result,raw=(result,matrix) if stage=='train-reference' else bound_embeddings(e,registry,'Final',stage,view)
            expected=reference if stage=='train-reference' else matched[stage]
            require(result['samples']==expected and result['views'][view]['sha256']==cp['checkpoint_sha256'] and result['views'][view]['state_unchanged'],'Legacy A/B endpoint alignment mismatch')
            contents[view,stage]=result['source_content_sha256']
            normalized=np.load(io.BytesIO(e.read(PREP+'/'+prefix+stage+'.npy',pc[prefix+stage+'.npy'])),allow_pickle=False)
            close(normalized,(raw[:,arrays['keep']]-arrays['mean'][arrays['keep']])/arrays['std'][arrays['keep']],'Legacy saved normalized embeddings mismatch')
        summary=one([r for r in observed['summaries'] if r['case']=='Final' and r['view']==view],'legacy drift summary')
        ff=[r for r in observed['per_family'] if r['case']=='Final' and r['view']==view];require([r['family'] for r in ff]==support,'Legacy drift support mismatch')
        verify_drift(ff,summary['conditional_mmd2'],summary['d_desc']);i=VIEWS.index(view)
        close(L,package['L'][i],'Legacy package Q mismatch');close(summary['d_desc'],package['d_desc'][i],'Legacy package drift mismatch')
        rows.append(dict(seed=42,view=view,checkpoint_status='ARTIFACT_VERIFIED',processing_status='ARTIFACT_VERIFIED',temperature=t,L=L,mmd2=summary['conditional_mmd2'],d=summary['d_desc'],Q_weight=package['Q_weights'][i],QD_weight=package['QD_weights'][i],normalizer_sha256=normhash,reference_embeddings_sha256=meta['embeddings_sha256'],processing_receipt=qpath,weights_file=FREEZE))
    frozen_weights(package['L'],package['d_desc'],package['Q_weights'],package['QD_weights'])
    return rows,contents


def execute(output):
    e=Audit();package=e.js(FREEZE,FREEZE_HASH);require(package['beta']==1 and package['gamma']==0 and not package['U_used'],'Frozen beta/U mismatch')
    stages,mapping,_,support=selection('Final',e);reference,_,_=plan('Final','train-reference',e);matched={s:plan('Final',s,e)[0] for s in ('A','B')}
    pc=e.js(PLAN+'/checksums.json',PLAN_HASH)
    from scripts.episode_training_dataset import csv_rows
    pairs=csv_rows(e.read(PLAN+'/Final/train_pairs.csv',pc['Final/train_pairs.csv']))
    checkpoints={};bindings={v:FinalTrainDataset(v).binding for v in VIEWS}
    for seed in SEEDS:
        for view in VIEWS:checkpoints[seed,view]=checkpoint(e,seed,view,package,bindings[view])
        print(f'Checkpoints seed{seed}: 3/3 VERIFIED',flush=True)
    require(len({r['checkpoint_sha256'] for r in checkpoints.values()})==15,'Duplicate checkpoint payload across seeds/views')
    (output/'checkpoint_phase.json').write_bytes(json_bytes(dict(status='PASS',checkpoints=list(checkpoints.values()),source_input_hashes=e.hashes)))
    rows=[];all_contents={}
    for seed in SEEDS:
        rr,cc=legacy_processing(e,package,checkpoints,stages,mapping,support,reference,matched,pairs) if seed==42 else new_processing(e,seed,checkpoints,stages,mapping,support,reference,matched,pairs)
        rows.extend(rr);all_contents[seed]=cc;print(f'Processing seed{seed}: VERIFIED',flush=True)
    for seed in SEEDS:
        for view in VIEWS:
            require(all_contents[seed][view,'train-reference']==all_contents[42][view,'train-reference'],'Cross-seed TRAIN reference content mismatch')
            # Legacy A/B arrays are capped; new arrays are complete retained sets.
            for stage in ('A','B'):
                values=all_contents[seed][view,stage]
                if seed!=42:
                    lookup={r['sha']:h for r,h in zip(stages[stage],values)};values=[lookup[r['sha']] for r in matched[stage]]
                require(values==all_contents[42][view,stage],'Cross-seed matched source content mismatch')
    rows.sort(key=lambda r:(r['seed'],VIEWS.index(r['view'])));summary=summarize(rows)
    result=dict(status='ARTIFACT_VERIFIED',checkpoints=list(checkpoints.values()),rows=rows,descriptive_statistics=summary,seeds=list(SEEDS),views=list(VIEWS),SD_ddof=1,checkpoint_count=15,processing_seeds=5,beta=1,gamma=0,U_used=False,
        checks='Hash/receipt chains, CPU checkpoint payload readback, source snapshots, canonical IDs/classes/content, independent full-K Q, TRAIN moments/pair distances, stored MMD term algebra and Q/QD weights.',
        limitations=['No CNN forward, temperature fit or kernel/MMD reevaluation.','Source image corpus not re-audited; verify stored embeddings/content identities and recorded input hashes instead.','All five registered seeds included; artifacts cannot prove unrecorded decisions.','Five seeds are not five independent temporal experiments.','E2 negative fusion result and descriptive-MMD limitations retained.'],
        source_input_hashes=e.hashes,source_sha256={p:sha256((ROOT/p).read_bytes()) for p in ('scripts/verify_final_multiseed.py','tests/test_final_multiseed_verification.py')},future_access=False,research_validated=False)
    (output/'summary.json').write_bytes(json_bytes(result))
    for name,table in [('seed_view_results.csv',rows),('descriptive_statistics.csv',summary),('checkpoints.csv',list(checkpoints.values()))]:
        with (output/name).open('x',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    (output/'receipt.json').write_bytes(json_bytes(dict(status='PASS',checkpoints_verified=15,processing_seeds_verified=5,files={p.name:sha256(p.read_bytes()) for p in output.iterdir() if p.is_file()},future_access=False,training=False,inference=False,calibration_fit=False,MMD_evaluation=False)))
    return result


if __name__=='__main__':
    output=ROOT/'data/derived/final_multiseed_verification'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f');output.mkdir(parents=True,exist_ok=False)
    try:
        result=execute(output);print('Receipt:',output/'receipt.json')
    except BaseException as exc:
        (output/'FAILED.json').write_bytes(json_bytes(dict(status='STOP',error=type(exc).__name__,message=str(exc))))
        print('STOP evidence:',output/'FAILED.json');raise
