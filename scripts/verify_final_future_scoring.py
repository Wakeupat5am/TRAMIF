"""Independent D062 readback from saved probabilities, not frozen evaluator helpers."""
import json
import math
from pathlib import Path
import statistics
import numpy as np

SEEDS = (42,43,44,45,46)
METHODS = ('sbsmi','raw-byte','entropy','uniform','new-quality-only','quality-plus-drift','learned-static')
MONTHS = tuple(f'2020-{m:02d}' for m in range(4,10))


def require(ok, message):
    if not ok: raise ValueError(message)


def close(actual, expected, path='value'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), 'Readback keys: '+path)
        for key, value in expected.items(): close(actual[key], value, path+'/'+key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'Readback list: '+path)
        for i, value in enumerate(expected): close(actual[i], value, path+'/'+str(i))
    elif isinstance(expected, float):
        require(isinstance(actual, (int,float)) and math.isfinite(actual)
                and abs(actual-expected) <= 1e-12, 'Numerical readback mismatch: '+path)
    else: require(actual == expected, 'Readback mismatch: '+path)


def reference_metrics(logp, labels):
    """Independent bincount confusion, explicit ECE bins, norm-identity Brier."""
    logp = np.asarray(logp, dtype=np.float64); y = np.asarray(labels)
    require(logp.ndim == 2 and logp.shape == (len(y),51) and len(y)>0
            and np.isfinite(logp).all() and y.dtype.kind in 'iu'
            and ((y>=0)&(y<51)).all(), 'Invalid readback metric inputs')
    p = np.exp(logp); pred = logp.argmax(1)
    require(np.allclose(p.sum(1),1,atol=1e-12,rtol=0), 'Readback probability mass')
    require(np.array_equal(pred,p.argmax(1)), 'Argmax numerical contract mismatch')
    cm = np.bincount(y*51+pred, minlength=51*51).reshape(51,51)
    n = len(y); true = cm.sum(1); predicted = cm.sum(0); tp = np.diag(cm)
    f1 = [2*int(tp[k])/int(true[k]+predicted[k]) if true[k]+predicted[k] else 0. for k in range(51)]
    support = np.flatnonzero(true).tolist(); correct = pred==y; conf = p[np.arange(n),pred]
    bins=[]; ece=0.
    for b in range(15):
        # Exactly the frozen floor(15*c) convention, including floating boundaries.
        mask=np.array([min(math.floor(float(c)*15),14)==b for c in conf])
        count=int(mask.sum()); acc=float(correct[mask].mean()) if count else None
        confidence=float(conf[mask].mean()) if count else None
        bins.append(dict(bin=b,count=count,accuracy=acc,confidence=confidence))
        if count: ece+=count/n*abs(acc-confidence)
    return dict(samples=n,supported_classes=len(support),accuracy=float(correct.mean()),
        macro_f1_supported=statistics.mean(f1[k] for k in support), macro_f1_fixed=statistics.mean(f1),
        balanced_accuracy=statistics.mean(int(tp[k])/int(true[k]) for k in support),
        nll=float(np.mean([-logp[i,int(y[i])] for i in range(n)])),
        brier=float(np.mean(np.sum(p*p,axis=1)-2*p[np.arange(n),y]+1)),ece_15=ece,bins=bins,
        confusion_matrix=cm.tolist(),per_class=[dict(class_index=k,support=int(true[k]),
            predicted_count=int(predicted[k]),f1=f1[k],recall=int(tp[k])/int(true[k]) if true[k] else None)
            for k in range(51)])


def spread(values):
    return dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values),min=min(values),max=max(values))


def verify_summary(summary, monthly):
    require(summary['seed_ids']==list(SEEDS) and set(summary['methods'])==set(METHODS), 'Summary identities')
    values={}; checks=0
    for method in METHODS:
        matrix=monthly[method]
        means=[statistics.mean(row) for row in matrix]; worst=[min(row) for row in matrix]
        values[method]=dict(mean_monthly_macro_f1=means,worst_month_macro_f1=worst)
        for name, x in values[method].items():
            close(summary['methods'][method]['outcomes'][name],dict(per_seed=x,**spread(x))); checks+=1
        for i, m in enumerate(MONTHS):
            x=[row[i] for row in matrix]
            close(summary['methods'][method]['monthly'][m],dict(per_seed=x,**spread(x)))
        ties=[[m for m,v in zip(MONTHS,row) if v==min(row)] for row in matrix]
        close(summary['methods'][method]['worst_months_per_seed'],ties)
    pairs=[('primary','quality-plus-drift','new-quality-only'),('secondary','new-quality-only','uniform'),
           ('secondary','quality-plus-drift','uniform'),('secondary','quality-plus-drift','learned-static')]
    expected=[]
    for priority,left,right in pairs:
        for name in ('mean_monthly_macro_f1','worst_month_macro_f1'):
            delta=[a-b for a,b in zip(values[left][name],values[right][name])]
            expected.append(dict(priority=priority,left=left,right=right,outcome=name,per_seed=delta,**spread(delta)))
    close(summary['contrasts'],expected)
    require(summary['p_values'] is None and summary['interval_status']=='NOT_ESTABLISHED_NOT_COMPUTED',
            'Unapproved inference output')
    return checks+len(expected)


def reference_logs(raw, package, seed):
    record=package['seed_records'][str(seed)]; singles=[]
    for view,z in zip(('sbsmi','raw-byte','entropy'),raw):
        z=np.asarray(z,dtype=np.float64)/record['views'][view]['temperature']
        a=z-z.max(axis=1,keepdims=True); singles.append(a-np.log(np.exp(a).sum(axis=1,keepdims=True)))
    result=list(singles)
    for weights in ([1/3]*3,*[record['weights'][k] for k in ('Q','QD','learned_static')]):
        terms=np.stack([s+math.log(w) for s,w in zip(singles,weights) if w>0])
        top=terms.max(axis=0); result.append(top+np.log(np.exp(terms-top).sum(axis=0)))
    return np.stack(result)


def verify(root, output, package):
    config=json.loads((output/'config.json').read_text()); rows=json.loads((output/'scoring_manifest.json').read_text())
    tables=json.loads((output/'retained_decisions.json').read_text()); results=json.loads((output/'results.json').read_text())
    ids=[r['sha'] for r in rows]; mapping=package['class_to_idx']
    labels=np.array([mapping[r['family']] for r in rows]); months=np.array([r['timestamp'][:7] for r in rows])
    logs=np.load(output/'log_probabilities.npy',mmap_mode='r',allow_pickle=False)
    require(logs.shape==(5,7,len(rows),51) and logs.dtype==np.float64 and np.isfinite(logs).all(), 'Saved logp shape/finite')
    with np.load(output/'predictions.npz',allow_pickle=False) as a:
        require(a['sample_identifiers'].tolist()==ids and a['class_names'].tolist()==sorted(mapping,key=mapping.get)
                and a['methods'].tolist()==list(METHODS) and a['seeds'].tolist()==list(SEEDS)
                and np.array_equal(a['labels'],labels) and np.array_equal(a['predictions'],logs.argmax(3)),
                'Saved predictions metadata/value mismatch')
    # Independent formulas from raw logits prove calibration/weights were applied exactly once.
    for si,seed in enumerate(SEEDS):
        raw=[]
        for v in ('sbsmi','raw-byte','entropy'):
            with np.load(root/config['input_directory']/'logits'/f'seed_{seed}_{v}.npz',allow_pickle=False) as a:
                require(a['sample_identifiers'].tolist()==ids, 'Raw logits readback ID mismatch');raw.append(a['logits'])
        expected=reference_logs(raw,package,seed)
        require(np.allclose(logs[si],expected,atol=1e-12,rtol=0), 'Frozen calibration/mixture numerical mismatch')
    checks=0; bucket_checks=0; horizon_checks=0; summary_checks=0
    cutoffs=package['size_stratification']['cutoffs']
    buckets=np.array([sum(r['file_size']>q for q in cutoffs) for r in rows])
    for t in ('0','10','20'):
        keep={r['sha'] for r in tables[t] if r['decision']=='KEEP'}
        take=np.array([sid in keep for sid in ids]); result=results[t]
        if result['status']!='PASS':
            require(t!='0' and (not take.any() or set(months[take])!=set(MONTHS)), 'Invalid NOT_ESTIMABLE');continue
        require(result['samples']==int(take.sum()), 'Retention sample count mismatch')
        report=result['report']; require(report['canonical_ids']==[sid for sid,k in zip(ids,take) if k], 'Subset IDs')
        support={m:sorted(set(labels[take & (months==m)].tolist())) for m in MONTHS}
        close(report['monthly_support'],support)
        monthly={method:[] for method in METHODS}
        for si,seed in enumerate(SEEDS):
            for j,method in enumerate(METHODS):
                d=report['details'][str(seed)][method]; values=[]
                for m in MONTHS:
                    mask=take & (months==m); ref=reference_metrics(logs[si,j,mask],labels[mask])
                    close(d['monthly'][m],ref); values.append(ref['macro_f1_supported']); checks+=1
                    for b in d['file_size'][m]:
                        choose=mask & (buckets==b['bucket']); count=int(choose.sum())
                        require(b['samples']==count and b['support']==sorted(set(labels[choose].tolist())), 'Size stratum support')
                        if count:
                            require(b['status']=='ESTIMABLE','Stratum status')
                            br=reference_metrics(logs[si,j,choose],labels[choose]); close(b['metrics'],br)
                            f1=[br['per_class'][k]['f1'] for k in support[m]]
                            close(b['macro_f1_original_month_support'],statistics.mean(f1))
                        else: require(b['status']=='NOT_ESTIMABLE' and 'metrics' not in b,'Empty stratum')
                        bucket_checks+=1
                monthly[method].append(values)
                for h in (1,3,6):
                    mask=take & np.isin(months,MONTHS[:h])
                    close(d['pooled_horizons'][str(h)],reference_metrics(logs[si,j,mask],labels[mask]));horizon_checks+=1
        summary_checks+=verify_summary(report['summary'],monthly)
    return dict(status='PASS_INDEPENDENT_NUMERICAL_READBACK',primary_samples=len(rows),
        monthly_metric_sets=checks,file_size_cells=bucket_checks,pooled_metric_sets=horizon_checks,
        outcomes_and_contrasts=summary_checks,probability_arrays=35,raw_logits_views=15,
        tolerance_absolute=1e-12,rtol=0,intervals_computed=False,p_values_computed=False,
        independent_formulas='bincount confusion; explicit ECE; norm Brier; statistics aggregation; direct stable mixture',
        inference_scope='descriptive fixed six-month horizon; seed variation is not CI')
