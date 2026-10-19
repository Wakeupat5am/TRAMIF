"""E1-C metadata readiness and explicit frozen CPU logit export; no C metrics CLI.

Pure comparison helper is synthetic-tested. Outcome scoring awaits the owner's
evaluation/selection decision under V1; no implicit outcome loss inspection.
"""
import argparse
from datetime import datetime,timezone
import numpy as np
import torch
from scripts.prepare_historical_logits import (
    Evidence,ROOT,G2,G2_HASH,VIEWS,MONTHS,selection,static_checkpoint,
    csv_rows,json_bytes,sha256,read_bound,require,image,load_encoder,state_hash,infer_logits,
)
from scripts.uniform_fusion import fuse_uniform,classification_metrics
from scripts.quality_fusion import weighted_fusion

FREEZE='data/derived/new_quality_only_v1/20261017_020915_979476'
FREEZE_HASH='7c7158308955b5e8b54d098af12e221ae06f38685cbb7509fd03c2c778f5d852'


def validate_outcome_rows(rows,mapping):
    require(bool(rows) and len({r['sha'] for r in rows})==len(rows),'Empty/duplicate outcome IDs')
    for row in rows:
        dt=datetime.fromisoformat(row['timestamp'])
        require(dt.utcoffset() is not None and dt.utcoffset().total_seconds()==0,'UTC required')
        require(dt.strftime('%Y-%m')=='2020-01' and row['stage']=='C','E1 outcome January2020 only')
        require(row['family'] in mapping and row['decision']=='KEEP' and row['threshold']=='0','Frozen C retention mismatch')


def frozen_plan(e):
    ab,mapping,source,support=selection('E1',e)
    checks=e.js(G2+'/checksums.json',G2_HASH)
    rows=csv_rows(e.read(G2+'/E1/C_keep.csv',checks['E1/C_keep.csv']))
    validate_outcome_rows(rows,mapping)
    # Canonical historical source is already allowlisted by selection.
    from scripts.episode_training_dataset import HISTORICAL_MANIFEST_HASH
    canonical={r['sha']:r for r in csv_rows(e.read('data/manifests/train_validation_manifest.csv',HISTORICAL_MANIFEST_HASH))}
    for r in rows:
        require(all(canonical[r['sha']][key]==r[key] for key in ('family','timestamp')),'C identity mismatch')
    require(not {r['sha'] for r in rows}&{r['sha'] for stage in ab.values() for r in stage},'C overlaps A/B IDs')
    completion=e.js(FREEZE+'/completion.json',FREEZE_HASH)
    frozen=e.js(FREEZE+'/E1_weights.json')
    require(completion['status']=='PASS' and frozen==next(r for r in completion['results'] if r['case']=='E1'),'Freeze integrity mismatch')
    require(frozen['class_to_idx']==mapping and frozen['K']==32 and frozen['support']==support and
            frozen['views']==list(VIEWS) and frozen['status']=='FROZEN' and frozen['beta']==frozen['gamma']==0,'Wrong frozen baseline')
    weights=np.array(frozen['weights'],dtype=np.float64)
    require(weights.shape==(3,) and np.isfinite(weights).all() and (weights>=0).all()
            and np.isclose(weights.sum(),1,rtol=0,atol=1e-15),'Invalid weights')
    temperatures={}
    for view in VIEWS:
        provenance=frozen['provenance'][view];path=provenance['temperature_file']
        t=e.js(path,completion['source_input_hashes'][path]);cp=static_checkpoint('E1',view,mapping,e)
        require(cp==provenance['checkpoint'] and t['checkpoint']['sha256']==cp['sha256'] and
                t['checkpoint']['K']==32 and t['class_to_idx']==mapping,'Checkpoint/T mismatch')
        require(t['fit_month']==MONTHS['E1'][0] and t['fit_ids']==[r['sha'] for r in ab['A']]
                and .05<=t['temperature']<=20,'A calibration mismatch')
        temperatures[view]=t['temperature']
    return rows,mapping,source,frozen,temperatures


def compare_fixed(bundles,weights):
    """Pure numerical helper; alignment is verified before probability mixing."""
    single,uniform=fuse_uniform(bundles)
    quality=weighted_fusion(np.stack(single),weights)
    names=list(VIEWS)+['uniform','new-quality-only']
    logs=np.stack(single+[uniform,quality])
    return names,logs,{name:classification_metrics(lp,bundles[0]['labels']) for name,lp in zip(names,logs)}


def execute(e,output,extract):
    rows,mapping,source,frozen,temperatures=frozen_plan(e)
    (output/'selection.json').write_bytes(json_bytes(dict(case='E1',stage='C',month='2020-01',
        samples=rows,class_to_idx=mapping,scoring_population='all frozen primary-retained C',
        weights_source=FREEZE,temperatures=temperatures)))
    view_results=[];content_reference=None
    if extract:
        for view in VIEWS:
            model,checkpoint=load_encoder('E1',view,mapping,e);before=state_hash(model)
            require(checkpoint['sha256']==frozen['provenance'][view]['checkpoint']['sha256'],'Wrong encoder')
            parts=[];contents=[]
            for start in range(0,len(rows),32):
                items=[image(row,view,source,e) for row in rows[start:start+32]]
                parts.append(infer_logits(model,torch.stack([x for x,_ in items]),32));contents.extend(h for _,h in items)
            require(state_hash(model)==before and all(p.grad is None for p in model.parameters()),'Frozen state mutation')
            if content_reference is not None:require(contents==content_reference,'Cross-view source content mismatch')
            content_reference=contents
            logits=np.concatenate(parts)
            np.savez(output/(view+'.npz'),logits=logits,labels=np.array([mapping[r['family']] for r in rows],dtype=np.int64),
                sample_identifiers=np.array([r['sha'] for r in rows]),class_names=np.array(sorted(mapping,key=mapping.get)))
            view_results.append(dict(view=view,shape=list(logits.shape),checkpoint=checkpoint,state_unchanged=True))
            print(f'E1/{view}/C: saved {logits.shape}',flush=True)
    for path,h in e.hashes.items():read_bound(ROOT,path,h)
    result=dict(status='LOGITS_COMPLETE' if extract else 'METADATA_READY',case='E1',stage='C',samples=len(rows),K=32,
        present_families=len({r['family'] for r in rows}),temperatures=temperatures,views=view_results,
        source_content_sha256=content_reference,source_input_hashes=e.hashes,
        metrics_computed=False,coefficients_selected=False,future_access=False)
    (output/'receipt.json').write_bytes(json_bytes(result))
    (output/'checksums.json').write_bytes(json_bytes({p.name:sha256(p.read_bytes()) for p in output.iterdir() if p.is_file()}))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['inspect','extract'])
    p.add_argument('--execute-inference',action='store_true');args=p.parse_args()
    if args.command=='extract' and not args.execute_inference:p.error('Explicit --execute-inference required')
    if args.command=='inspect' and args.execute_inference:p.error('inspect never runs inference')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    e=Evidence()
    for path in ('scripts/prepare_e1_outcome.py','tests/test_e1_outcome_preparation.py',
                 'scripts/prepare_historical_logits.py','scripts/frozen_encoder_extraction.py',
                 'scripts/uniform_fusion.py','scripts/quality_fusion.py'):e.read(path)
    output=ROOT/'data/derived/e1_outcome_preparation'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    (output/'STARTED.json').write_bytes(json_bytes(vars(args)))
    try:r=execute(e,output,args.command=='extract')
    except BaseException as exc:
        (output/'FAILED.json').write_bytes(json_bytes(dict(error=type(exc).__name__,message=str(exc))))
        raise
    print(r['status'],r['samples'],'samples,',r['present_families'],'families; no C metrics')
    print('Receipt:',output/'receipt.json')


if __name__=='__main__':main()
