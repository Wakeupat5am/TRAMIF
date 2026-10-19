"""Fixed D040 synthetic experiment. No historical input or model dependencies."""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import contextlib
import hashlib
import io
import json
from pathlib import Path
from datetime import datetime, timezone
import time
import numpy as np
from scripts.quality_uncertainty import estimate_uncertainty, draw_group_weights

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'docs/quality_uncertainty_simulation_plan.md'
SCALE = np.array([.4, .7, 1.])
EXPECTED = np.array([.9, 1.3, 1.7])


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def layout(scenario):
    if type(scenario) is not int or scenario not in range(1,8):
        raise ValueError('Unknown fixed scenario')
    ids=[]; families=[]; groups=[]
    for c in range(3):
        sizes = [4]*20
        if scenario==2: sizes=[1]*10+[2]*6+[8]*3+[40]
        if scenario==5 and c==2: sizes=[4]*5
        if scenario==6: sizes=[80]+[4]*19
        for g,n in enumerate(sizes):
            group=f'g{g:02d}' if scenario==7 else f'c{c}_g{g:02d}'
            for member in range(n):
                ids.append(f'c{c}_g{g:02d}_i{member:03d}')
                families.append(f'c{c}'); groups.append(group)
    return dict(sample_ids=ids,families=families,group_ids=groups,support=['c0','c1','c2'])


def generate(scenario, dataset_index):
    spec=layout(scenario)
    if type(dataset_index) is not int or not 0<=dataset_index<300:
        raise ValueError('Dataset stream outside fixed budget')
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261016,scenario,dataset_index])))
    order=sorted(set(spec['group_ids'])); n=len(spec['sample_ids']); g=len(order)
    group=(rng.standard_normal(g)[:,None]+rng.standard_normal((g,3)))*np.sqrt(.5)
    sample=(rng.standard_normal(n)[:,None]+rng.standard_normal((n,3)))*np.sqrt(.5)
    if scenario==4:
        blocks=[f'{name[:2]}_b{int(name[-2:])//4:02d}' for name in order]
        unique=sorted(set(blocks))
        hidden=(rng.standard_normal(len(unique))[:,None]+rng.standard_normal((len(unique),3)))*np.sqrt(.5)
        group=np.sqrt(.75)*hidden[[unique.index(b) for b in blocks]]+np.sqrt(.25)*group
    alpha=.9 if scenario==3 else .5
    z=np.sqrt(alpha)*group[[order.index(gid) for gid in spec['group_ids']]]+np.sqrt(1-alpha)*sample
    c=np.array([int(f[-1]) for f in spec['families']])
    mu=.3+.2*c[:,None]+.1*np.arange(3)[None,:]
    losses=(mu+SCALE*z*z).T
    if losses.shape!=(3,n) or not np.isfinite(losses).all() or (losses<0).any():
        raise FloatingPointError('Invalid generated losses')
    return losses,spec


def reference(losses,spec,draws,order):
    """Independent sample-level weighted means, no group-total aggregation."""
    result=[]
    for draw in draws:
        views=[]
        for v in range(3):
            means=[]
            for family in spec['support']:
                idx=[i for i,c in enumerate(spec['families']) if c==family]
                weights=[float(draw[order.index(spec['group_ids'][i])]) for i in idx]
                means.append(sum(w*float(losses[v,i]) for w,i in zip(weights,idx))/sum(weights))
            views.append(sum(means)/3)
        result.append(views)
    return np.array(result)


def statistic(point,u):
    sd=float(np.std(point,ddof=1))
    if sd<=0 or not np.isfinite(sd):
        raise FloatingPointError('Undefined empirical variability; no epsilon substitute')
    return np.array([sd,float(np.mean(u))/sd,float(np.mean(u*u))/(sd*sd)])


def jackknife_se(point,u):
    n=len(point)
    deletes=np.array([statistic(np.delete(point,i),np.delete(u,i)) for i in range(n)])
    return np.sqrt((n-1)/n*np.square(deletes-deletes.mean(axis=0)).sum(axis=0))


def summarize(points,us,u43):
    if points.shape!=(300,3) or us.shape!=(300,3) or u43.shape!=(30,3):
        raise ValueError('Incomplete simulation budget')
    if not all(np.isfinite(x).all() for x in (points,us,u43)):
        raise FloatingPointError('Nonfinite summary input')
    rows=[]
    for v in range(3):
        p,u=points[:,v],us[:,v]
        sd,ratio,var_ratio=statistic(p,u)
        precision=jackknife_se(p,u)
        diff=np.abs(u[:30]-u43[:,v])
        trigger=diff>np.maximum(.005,.1*np.maximum(u[:30],u43[:,v]))
        rows.append(dict(view=v,expected_loss=float(EXPECTED[v]),mean_loss=float(p.mean()),
            bias=float(p.mean()-EXPECTED[v]),empirical_sd=sd,mean_loss_mcse=sd/np.sqrt(300),
            u_mean=float(u.mean()),u_rms=float(np.sqrt(np.mean(u*u))),ratio_sd=ratio,
            ratio_variance=var_ratio,jackknife_se_sd=float(precision[0]),
            jackknife_se_ratio_sd=float(precision[1]),jackknife_se_ratio_variance=float(precision[2]),
            u_min=float(u.min()),u_q05=float(np.quantile(u,.05)),u_median=float(np.median(u)),
            u_q95=float(np.quantile(u,.95)),u_max=float(u.max()),
            seed43_max_abs_difference=float(diff.max()),seed43_trigger_count=int(trigger.sum()),
            ratio_review_trigger=bool(ratio<.8 or ratio>1.25)))
    return rows


def influence(losses,spec,point):
    rows=[]
    for group in sorted(set(spec['group_ids'])):
        means=[]; missing=[]
        for family in spec['support']:
            idx=[i for i,c in enumerate(spec['families']) if c==family and spec['group_ids'][i]!=group]
            if not idx: missing.append(family)
            else: means.append(losses[:,idx].mean(axis=1))
        rows.append(dict(group=group,removed_samples=spec['group_ids'].count(group),
            missing_families=missing,delta=None if missing else (np.mean(means,axis=0)-point).tolist()))
    return rows


def execute(output):
    start=time.perf_counter()
    sources=[PLAN,'scripts/quality_uncertainty.py','scripts/simulate_quality_uncertainty.py',
             'tests/test_quality_uncertainty_simulation.py']
    hashes={p:digest(ROOT/p) for p in sources}
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf): np.show_config()
    config=dict(scenarios=list(range(1,8)),datasets=300,replicates=2000,seed=42,
        secondary_seed=43,secondary_indices=list(range(30)),
        data_stream='PCG64(SeedSequence([20261016,scenario_id,dataset_index]))',
        numpy_version=np.__version__,blas_config=buf.getvalue(),threads=1,source_hashes=hashes,
        review_ratio_bounds=[.8,1.25],tolerance=dict(rtol=1e-12,atol=1e-12))
    (output/'config.json').write_bytes(json_bytes(config))
    all_rows=[]; counts=dict(primary=0,secondary=0)
    for scenario in range(1,8):
        folder=output/f'scenario_{scenario}';folder.mkdir()
        spec=layout(scenario);order=sorted(set(spec['group_ids']))
        (folder/'layout.json').write_bytes(json_bytes(dict(**spec,group_order=order)))
        # Same RNG matrices for same cardinality stored once, bound by group order.
        draw_dir=output/'draws';draw_dir.mkdir(exist_ok=True)
        matrices=[]
        for seed in (42,43):
            file=draw_dir/f'g{len(order)}_seed{seed}.npy'
            if file.exists(): draws=np.load(file,allow_pickle=False)
            else:
                draws=draw_group_weights(order,seed=seed);np.save(file,draws,allow_pickle=False)
            matrices.append(draws)
        loss_all=[];points=[];us=[];replicates=[];secondary=[];secondary_replicates=[];influences=[]
        for r in range(300):
            losses,actual=generate(scenario,r)
            if actual!=spec: raise ValueError('Layout mutation')
            result=estimate_uncertainty(losses,**spec,draws=matrices[0],draw_group_order=order)
            counts['primary']+=1
            if r==0:
                np.testing.assert_array_equal(losses,generate(scenario,r)[0])
                np.testing.assert_allclose(result['replicate_losses'][:3],reference(losses,spec,matrices[0][:3],order),rtol=1e-12,atol=1e-12)
            loss_all.append(losses);points.append(result['point_loss']);us.append(result['uncertainty'])
            replicates.append(result['replicate_losses'])
            if r<30:
                alt=estimate_uncertainty(losses,**spec,draws=matrices[1],draw_group_order=order)
                counts['secondary']+=1;secondary.append(alt['uncertainty']);secondary_replicates.append(alt['replicate_losses'])
                if scenario in (5,6,7): influences.append(dict(dataset=r,groups=influence(losses,spec,result['point_loss'])))
        points,us,secondary=map(np.asarray,(points,us,secondary))
        np.savez(folder/'outputs.npz',losses=np.asarray(loss_all),point_loss=points,uncertainty=us,
                 replicate_losses=np.asarray(replicates),seed43_uncertainty=secondary,
                 seed43_replicate_losses=np.asarray(secondary_replicates))
        rows=[dict(scenario=scenario,**row) for row in summarize(points,us,secondary)]
        (folder/'summary.json').write_bytes(json_bytes(rows))
        (folder/'influence.json').write_bytes(json_bytes(influences))
        all_rows.extend(rows)
        print(f'Scenario {scenario}/7 complete | ratio_SD: '+', '.join(format(a['ratio_sd'], '.4f') for a in rows),flush=True)
    if counts!={'primary':2100,'secondary':210}: raise ValueError('Budget mismatch')
    for path,h in hashes.items():
        if digest(ROOT/path)!=h: raise ValueError('Source changed during run')
    triggered=any(r['ratio_review_trigger'] or r['seed43_trigger_count'] for r in all_rows)
    report=dict(engineering='PASS',counts=counts,results=all_rows,elapsed_seconds=time.perf_counter()-start,
        statistical_review_triggered=triggered,fusion_status='U_NOT_APPROVED_FOR_FUSION',
        statistical_validity='CONDITIONAL_SYNTHETIC_DIAGNOSTICS_ONLY',source_hashes=hashes,
        real_data_access=False,future_access=False)
    (output/'report.json').write_bytes(json_bytes(report))
    checks={p.relative_to(output).as_posix():digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'checksums.json').write_bytes(json_bytes(checks))


def main():
    output=ROOT/'data/derived/quality_uncertainty_simulation'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    try: execute(output)
    except BaseException as exc:
        (output/'FAILED.json').write_bytes(json_bytes(dict(status='PARTIAL' if isinstance(exc,KeyboardInterrupt) else 'IMPLEMENTATION_FAILURE',error=type(exc).__name__,message=str(exc))))
        raise
    print('Report:',output/'report.json',flush=True)


if __name__=='__main__': main()
