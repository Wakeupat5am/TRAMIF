"""Bounded CPU synthetic model-only benchmark and historical run inventory."""
import argparse
import csv
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import io
import json
import platform
from time import perf_counter
import numpy as np
import torch
from scripts.train_final_seed import ROOT,SEEDS,VIEWS,OUTPUT,make_model,write_new_json,source_hashes,sha256,environment
from scripts.frozen_encoder_extraction import RUNS


def process_memory():
    if platform.system()!='Windows':return dict(status='UNKNOWN: Windows process counters unavailable')
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(x,ctypes.c_size_t) for x in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
    c=Counters();c.cb=ctypes.sizeof(c)
    kernel=ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi=ctypes.WinDLL('psapi');psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb):return dict(status='UNKNOWN: counter query failed')
    return dict(working_set_bytes=c.WorkingSetSize,peak_working_set_bytes=c.PeakWorkingSetSize,private_commit_bytes=c.PrivateUsage,peak_commit_bytes=c.PeakPagefileUsage)


def inventory():
    result=[]
    for view in VIEWS:
        path,digest=RUNS['Final',view];p=ROOT/path
        history=list(csv.DictReader(io.StringIO((p/'history.csv').read_text(encoding='utf-8-sig'))))
        seconds=[float(r['epoch_seconds']) for r in history]
        result.append(dict(seed=42,view=view,checkpoint=path+'/epoch_100.pt',registered_sha256=digest,
            checkpoint_bytes=(p/'epoch_100.pt').stat().st_size,epochs=len(history),epoch_seconds_total=sum(seconds),
            epoch_seconds_median=float(np.median(seconds)),history_sha256=sha256((p/'history.csv').read_bytes()),
            history_path=path+'/history.csv'))
    additional=[]
    for seed in SEEDS:
        for view in VIEWS:
            paths=list((ROOT/OUTPUT/f'seed_{seed}'/view).glob('*/COMPLETE.json'))
            additional.append(dict(seed=seed,view=view,complete_candidates=len(paths),status='MISSING' if not paths else 'REQUIRES_VERIFY',samples=18061,batches=565,epochs=100,optimizer_steps=56500))
    estimated_checkpoint_bytes=sum(r['checkpoint_bytes']*100 for r in result)*4
    return dict(legacy=result,additional=additional,missing_checkpoints=sum(r['status']=='MISSING' for r in additional),
        projected_loop_hours=sum(r['epoch_seconds_total'] for r in result)*4/3600,
        projected_checkpoint_bytes=estimated_checkpoint_bytes,
        projection_limit='Historical throughput only; excludes data/hash checks, source snapshots, serialization, postprocessing and thermal/resource variability. Not a guarantee.')


def execute():
    output=ROOT/'data/derived/pre_future_engineering'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f');output.mkdir(parents=True,exist_ok=False)
    config=dict(device='cpu',random_seed=43,threads=1,warmups=3,iterations=10,batches=[1,32],input=[1,64,64],dtype='float32',random_weights=True,malware_access=False,training=False)
    before=process_memory();previous=torch.get_num_threads();torch.set_num_threads(1);results=[]
    try:
        for view in VIEWS:
            model=make_model(43,view).eval();parameters=sum(p.numel() for p in model.parameters());state_bytes=sum(v.numel()*v.element_size() for v in model.state_dict().values())
            for batch_size in config['batches']:
                generator=torch.Generator().manual_seed(43);x=torch.rand(batch_size,1,64,64,generator=generator);timings=[]
                with torch.inference_mode():
                    for _ in range(config['warmups']):model(x)
                    for _ in range(config['iterations']):
                        start=perf_counter();z=model(x);timings.append((perf_counter()-start)*1000)
                if z.shape!=(batch_size,51) or not torch.isfinite(z).all():raise ValueError('Synthetic model output failed')
                results.append(dict(view=view,batch_size=batch_size,parameters=parameters,state_tensor_bytes=state_bytes,input_tensor_bytes=x.numel()*x.element_size(),
                    latency_mean_ms=float(np.mean(timings)),latency_median_ms=float(np.median(timings)),latency_min_ms=min(timings),latency_p95_ms=float(np.quantile(timings,.95)),
                    timings_ms=timings,model_only=True,end_to_end_latency='NOT MEASURED'))
    finally:torch.set_num_threads(previous)
    report=dict(config=config,environment=environment(),processor=platform.processor(),cpu_count=__import__('os').cpu_count(),before=before,after=process_memory(),results=results,inventory=inventory(),
        source_sha256={**source_hashes(),'scripts/benchmark_pre_future.py':sha256((ROOT/'scripts/benchmark_pre_future.py').read_bytes())},
        limitation='Process peaks include interpreter/imports and preceding models; not isolated activation memory. No GPU/preprocessing/I/O/full pipeline or capacity-matched superiority claim.')
    write_new_json(output/'report.json',report);write_new_json(output/'receipt.json',dict(status='PASS',report_sha256=sha256((output/'report.json').read_bytes()),malware_access=False))
    print(json.dumps(dict(run=str(output.relative_to(ROOT)),report_sha256=sha256((output/'report.json').read_bytes()),missing_checkpoints=report['inventory']['missing_checkpoints'],projected_loop_hours=report['inventory']['projected_loop_hours'],projected_checkpoint_GiB=report['inventory']['projected_checkpoint_bytes']/1024**3)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute-cpu-synthetic',action='store_true');args=p.parse_args()
    if not args.execute_cpu_synthetic:raise PermissionError('Explicit bounded synthetic benchmark flag required')
    execute()
