"""D044: fixed-A-temperature historical B Quality; no fitting or fusion."""
import io
import json
from datetime import datetime, timezone
from pathlib import PurePosixPath
import numpy as np
from scripts.prepare_historical_logits import (
    Evidence, ROOT, VIEWS, MONTHS, selection, validate_bundle, static_checkpoint,
    json_bytes, sha256, read_bound, require, T_MIN, T_MAX,
)
from scripts.uniform_fusion import calibrated_log_probabilities

PREFIX_INPUTS={
    'E1':('data/derived/historical_logits_preparation/20261016_170412_203009',
          '7821c6ccbc6bc1b29dccb46c050f250e0d4a32b2ffe9e7eb913f93d560068e19'),
    'E2':('data/derived/historical_logits_preparation/20261016_170643_356292',
          '276aa30fc864ac2967454977b65beda6b0ee98c40a194cfb146cf7598ab62c6a'),
}
FINAL_AUDIT=('data/derived/historical_logits_preparation/20261016_170159_059085/receipt.json',
             '9cd6293aefb50d596043c0555d38a99b76c92aa27757d752876f290446a2807f')
PREFIX_CHECKS={'E1':'850bd7269fc312bbe91b257ceaba884dee0b1b402896aeb27345afd1f8672f7f',
               'E2':'07b8393ca0f1b02371c1ae0f12913e7eb954033b8a3f5127859c923345b6c1f2'}


def canonical_bundle(bundle,rows,mapping):
    """Explicit identity join; never pair predictions by unchecked array index."""
    expected=[r['sha'] for r in rows]
    ids=bundle['sample_identifiers'].tolist()
    require(len(set(expected))==len(expected) and len(set(ids))==len(ids),'Duplicate IDs')
    require(len(ids)==len(expected) and set(ids)==set(expected),'Missing/extra IDs')
    lookup={sid:i for i,sid in enumerate(ids)}
    order=np.array([lookup[sid] for sid in expected],dtype=np.int64)
    aligned=dict(logits=bundle['logits'][order],labels=bundle['labels'][order],
                 sample_identifiers=bundle['sample_identifiers'][order],class_names=bundle['class_names'])
    validate_bundle(aligned,rows,mapping)
    return aligned,order


def quality(bundle,rows,mapping,support,temperature):
    require(type(temperature) in (int,float) and np.isfinite(temperature) and T_MIN<=temperature<=T_MAX,'Invalid fixed T')
    require(isinstance(support,(list,tuple)) and len(support)>0 and len(set(support))==len(support)
            and all(c in mapping for c in support),'Invalid fixed support')
    aligned,order=canonical_bundle(bundle,rows,mapping)
    with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
        logs=calibrated_log_probabilities(aligned['logits'],temperature)
        require(logs.dtype==np.float64 and np.isfinite(logs).all(),'Nonfinite log probabilities')
        require(np.allclose(np.exp(logs).sum(axis=1),1,rtol=0,atol=1e-12),'Invalid K-class normalization')
        losses=-logs[np.arange(len(rows)),aligned['labels']]
    require(np.isfinite(losses).all() and np.all(losses>=0),'Invalid NLL')
    family=np.array([r['family'] for r in rows]);mask=np.isin(family,support)
    per_family=[]
    for c in support:
        ff=family==c
        require(ff.any(),'Missing support family')
        per_family.append(dict(family=c,label=mapping[c],samples=int(ff.sum()),
                               mean_nll=float(losses[ff].mean())))
    means=[r['mean_nll'] for r in per_family]
    balanced=float(np.mean(means))
    result=dict(K=len(mapping),support_families=len(support),B_support_samples=int(mask.sum()),
                B_retained_samples=len(rows),temperature=temperature,class_balanced_nll=balanced,
                Q=-balanced,standard_nll_support=float(losses[mask].mean()),
                standard_nll_all_B=float(losses.mean()),min_family_nll=min(means),
                max_family_nll=max(means),per_family=per_family,status='PASS')
    require(all(np.isfinite(result[k]) for k in ('class_balanced_nll','standard_nll_support','standard_nll_all_B')),'Nonfinite aggregation')
    return result,losses,mask,order


def read_npz(e,path,digest):
    with np.load(io.BytesIO(e.read(path,digest)),allow_pickle=False) as arrays:
        return {k:arrays[k].copy() for k in ('logits','labels','sample_identifiers','class_names')}


def prefix_sources(case,stages,mapping,support,e):
    base,pin=PREFIX_INPUTS[case];receipt=e.js(base+'/receipt.json',pin)
    require(receipt['status']=='PASS' and receipt['command']==dict(command='prepare',case=case,execute_inference=True),'Prefix receipt mismatch')
    checks=e.js(base+'/checksums.json',PREFIX_CHECKS[case])
    def bound(name):return e.read(base+'/'+name,checks[name])
    selected=json.loads(bound('selection.json'))
    require(selected==dict(case=case,stages=stages,class_to_idx=mapping,support=support),'Frozen selection mismatch')
    require({r['view'] for r in receipt['results']}==set(VIEWS) and len(receipt['results'])==3,'Missing/duplicate view')
    output=[]
    for view in VIEWS:
        cp=static_checkpoint(case,view,mapping,e)
        r=next(r for r in receipt['results'] if r['view']==view)
        t=json.loads(bound(view+'/temperature.json'))
        require(r['state_unchanged'] and t['checkpoint']==r['checkpoint'],'Checkpoint provenance mismatch')
        require(t['case']==case and t['view']==view and t['class_to_idx']==mapping and
                t['checkpoint']['sha256']==cp['sha256'] and t['checkpoint']['K']==len(mapping),'T binding mismatch')
        require(t['fit_month']==MONTHS[case][0] and t['fit_ids']==[r['sha'] for r in stages['A']]
                and t['bounds']==[T_MIN,T_MAX],'A-only fit mismatch')
        a=bound(view+'/A.npz');require(sha256(a)==t['A_sha256'],'A source hash mismatch')
        with np.load(io.BytesIO(a),allow_pickle=False) as a_bundle:
            canonical_bundle(a_bundle,stages['A'],mapping)
        path=base+'/'+view+'/B.npz'
        require(t['temperature']==r['temperature'],'T receipt mismatch')
        output.append((view,read_npz(e,path,checks[view+'/B.npz']),t['temperature'],
                       dict(checkpoint=cp,temperature_file=base+'/'+view+'/temperature.json',B_file=path,
                            preparation_receipt=base+'/receipt.json')))
    return output


def final_sources(stages,mapping,e):
    verified=e.js(*FINAL_AUDIT)['input_source_hashes']
    registry=e.js('docs/baseline_artifacts.json',verified['docs/baseline_artifacts.json'])['entries']
    output=[]
    for view in VIEWS:
        entry=next(r for r in registry if r['view']==view);cp=static_checkpoint('Final',view,mapping,e)
        t=e.js(entry['temperature_file'],entry['temperature_file_sha256'])
        require(t['checkpoint_sha256']==cp['sha256']==entry['checkpoint_sha256'] and t['class_to_idx']==mapping
                and t['dataset_config_sha256']==cp['config_id'] and not t['future_test_used'],'Final T provenance mismatch')
        require(t['fit_month']=='2020-02' and t['fit_samples']==len(stages['A']),'Final fit month mismatch')
        for month in MONTHS['Final']:
            ev=next(x for x in entry['evaluations'] if x['month']==month)
            report=e.js(ev['report'],ev['report_sha256'])
            require(report['checkpoint_sha256']==cp['sha256'] and report['class_to_idx']==mapping and
                    report['dataset_config_sha256']==cp['config_id'] and report['selection']['threshold']==0
                    and report['month']==month,'Final prediction provenance mismatch')
            path=str(PurePosixPath(ev['report']).parent/'predictions.npz')
            bundle=read_npz(e,path,ev['predictions_sha256'])
            canonical_bundle(bundle,stages['A' if month=='2020-02' else 'B'],mapping)
            if month=='2020-02':
                require(t['calibration_source']['predictions_sha256']==ev['predictions_sha256'] and
                        t['calibration_source']['report_sha256']==ev['report_sha256'],'Final A source mismatch')
            else: output.append((view,bundle,t['temperature'],dict(checkpoint=cp,temperature_file=entry['temperature_file'],B_file=path)))
    return output


def execute(output):
    e=Evidence()
    for path in ('scripts/compute_historical_quality.py','tests/test_historical_quality.py',
                 'scripts/uniform_fusion.py','scripts/prepare_historical_logits.py'):
        e.read(path)
    config=dict(task='D044',cases=list(MONTHS),loss='float64 stable K-class logsoftmax(logits/fixed_A_T)',
                quality='Q=-class_balanced_NLL',aggregation='equal-family on frozen S',cap=None,
                temperature_fit=False,inference=False,U=False,fusion_weights=False)
    (output/'config.json').write_bytes(json_bytes(config));summaries=[]
    for case in MONTHS:
        stages,mapping,source,support=selection(case,e)
        inputs=final_sources(stages,mapping,e) if case=='Final' else prefix_sources(case,stages,mapping,support,e)
        case_dir=output/case;case_dir.mkdir()
        (case_dir/'selection.json').write_bytes(json_bytes(dict(B=stages['B'],support=support,class_to_idx=mapping)))
        for view,bundle,t,provenance in inputs:
            result,losses,mask,order=quality(bundle,stages['B'],mapping,support,t)
            folder=case_dir/view;folder.mkdir()
            np.savez(folder/'sample_losses.npz',nll=losses,in_support=mask,
                     sample_identifiers=np.array([r['sha'] for r in stages['B']]),
                     labels=np.array([mapping[r['family']] for r in stages['B']],dtype=np.int64),source_row_indices=order)
            (folder/'quality.json').write_bytes(json_bytes(dict(case=case,view=view,**result,provenance=provenance)))
            summaries.append(dict(case=case,view=view,**{k:v for k,v in result.items() if k!='per_family'}))
    for path,h in e.hashes.items():read_bound(ROOT,path,h)
    (output/'completion.json').write_bytes(json_bytes(dict(status='PASS',results=summaries,
        source_input_hashes=e.hashes,numpy=np.__version__,future_access=False,outcome_C_access=False,
        U_computed=False,fusion_weights_created=False)))
    (output/'checksums.json').write_bytes(json_bytes({p.relative_to(output).as_posix():sha256(p.read_bytes()) for p in output.rglob('*') if p.is_file()}))


def main():
    output=ROOT/'data/derived/historical_quality_v1'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    try:execute(output)
    except BaseException as exc:
        (output/'FAILED.json').write_bytes(json_bytes(dict(status='FAIL',error=type(exc).__name__,message=str(exc))))
        raise
    print('Completion:',output/'completion.json')


if __name__=='__main__':main()
