"""V1 proposed statistics consumer: pure arrays, no file readers or real-data CLI.

Not an approved coverage estimator. Synthetic tests establish arithmetic and
pairing only. Empirical resampling and coverage experiments require approval.
"""
import numpy as np


def require(condition,message):
    if not condition:raise ValueError(message)


def fixed_support_f1(truth,prediction,support):
    require(bool(support) and len(set(support))==len(support),'Explicit fixed monthly support required')
    values=[]
    for c in support:
        tp=np.count_nonzero((truth==c)&(prediction==c));fp=np.count_nonzero((truth!=c)&(prediction==c));fn=np.count_nonzero((truth==c)&(prediction!=c))
        denominator=2*tp+fp+fn;values.append(2*tp/denominator if denominator else 0.)
    return float(np.mean(values))


def outcomes(truth,predictions,months,supports,*,indices=None,seeds=None):
    """Return seed x method x [mean monthly, worst month]; never min of seed means."""
    y=np.asarray(truth);p=np.asarray(predictions);m=np.asarray(months)
    require(y.ndim==1 and len(y)>0 and y.dtype.kind in 'iu' and p.ndim==3 and p.shape[0]>0 and p.shape[1]>0 and p.shape[2]==len(y) and p.dtype.kind in 'iu' and m.shape==y.shape,'Invalid aligned arrays')
    require(set(np.unique(m))==set(range(6)) and len(supports)==6,'Exactly six frozen months required')
    require(all(set(y[m==month])==set(supports[month]) for month in range(6)),'Original monthly support differs')
    require((y>=0).all() and (y<51).all() and (p>=0).all() and (p<51).all(),'K51 labels required')
    idx=np.arange(len(y)) if indices is None else np.asarray(indices)
    ss=np.arange(p.shape[0]) if seeds is None else np.asarray(seeds)
    require(idx.ndim==1 and idx.size>0 and idx.dtype.kind in 'iu' and idx.min()>=0 and idx.max()<len(y),'Invalid sample draw')
    require(ss.ndim==1 and ss.size>0 and ss.dtype.kind in 'iu' and ss.min()>=0 and ss.max()<p.shape[0],'Invalid seed draw')
    monthly=np.empty((len(ss),p.shape[1],6));counts=[]
    for month in range(6):
        take=idx[m[idx]==month];counts.append(len(take));require(len(take)>0,'Empty bootstrap month; STOP, no retry/drop')
        for i,seed in enumerate(ss):
            for method in range(p.shape[1]):monthly[i,method,month]=fixed_support_f1(y[take],p[seed,method,take],supports[month])
    result=np.stack([monthly.mean(2),monthly.min(2)],axis=2)
    require(np.isfinite(result).all(),'Nonfinite outcome')
    return result,counts


def paired_draws(groups,seed_count,*,replicates=20000,seed=42):
    """PCG64 group and seed draws; one shared index vector for every method.

    Entire cross-month components repeat intact; sample totals can vary.
    Canonical group ordering is lexicographic and must be frozen in provenance.
    """
    g=np.asarray(groups)
    require(g.ndim==1 and g.size>0 and g.dtype.kind in 'US' and all(bool(x) for x in g),'Nonempty group IDs required')
    require(type(seed_count) is int and seed_count>0 and type(replicates) is int and replicates>0 and type(seed) is int,'Invalid draw configuration')
    names=sorted(set(g.tolist()));members=[np.flatnonzero(g==name) for name in names];rng=np.random.Generator(np.random.PCG64(seed))
    for _ in range(replicates):
        ss=rng.integers(0,seed_count,size=seed_count);gg=rng.integers(0,len(names),size=len(names))
        yield ss,gg,np.concatenate([members[i] for i in gg])


def percentile_intervals(differences):
    d=np.asarray(differences,dtype=np.float64)
    require(d.ndim>=1 and len(d)>=2 and np.isfinite(d).all(),'Finite complete replicates required')
    return dict(pointwise95=np.quantile(d,[.025,.975],axis=0,method='linear').tolist(),
        conservative_family10_99_5=np.quantile(d,[.0025,.9975],axis=0,method='linear').tolist(),
        family_size=10,coverage='NOT VALIDATED; multiplicity family needs owner decision with U arms unavailable')
