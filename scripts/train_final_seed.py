"""Additive Final51 seed43..46 trainer. No training on import/preflight/verify.

An execution flag is an operator acknowledgement, not external owner approval.
Legacy seed42 artifacts and prefix trainers are never modified or resumed.
"""
import argparse
import io
import json
import random
import shutil
from datetime import datetime,timezone
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from models.malsbslcnet import MalSBSLCNet
from models.compact_view_net import CompactViewNet
from scripts.historical_policy_dataset import HistoricalPolicyDataset
from scripts.episode_training_dataset import ROOT,VIEWS,HISTORICAL_MANIFEST_HASH,SOURCE_MAPPING_HASH,read_bound,json_bytes,sha256,validate_mapping
from scripts.train_prefix_view import (
    RUN_SPEC,MODEL_HASHES,PROTOCOL_HASHES,make_optimizer,save_checkpoint,write_new_json,environment,git_snapshot,check_coverage,
)

SEEDS=(43,44,45,46)
POLICY='data/derived/historical_near_duplicates_v1/20261009_143301_921802'
OUTPUT='data/derived/final_seed_training'
SOURCES=('scripts/train_final_seed.py','scripts/historical_policy_dataset.py','scripts/sbsmi_dataset.py',
         'scripts/float_image_dataset.py','scripts/train_prefix_view.py','models/malsbslcnet.py',
         'models/compact_view_net.py','scripts/episode_training_dataset.py',
         'docs/RESEARCH_PROTOCOL_V1.md','tests/test_final_seed.py')


def require(ok,msg):
    if not ok:raise ValueError(msg)


def validate_seed(seed):
    require(type(seed) is int and seed in SEEDS,'Only additional Final seeds43..46 allowed')


def validate_train_rows(rows,mapping,expected=18061):
    validate_mapping(mapping,51)
    require(len(rows)==expected and len({r['sha'] for r in rows})==expected,'TRAIN count/duplicate mismatch')
    for r in rows:
        dt=datetime.fromisoformat(r['timestamp'])
        require(dt.utcoffset() is not None and dt.utcoffset().total_seconds()==0,'UTC required')
        require('2019-08'<=dt.strftime('%Y-%m')<'2020-02' and r['split']=='train' and r['family'] in mapping,'Non-TRAIN row')


def source_hashes():
    anchors={**MODEL_HASHES,**PROTOCOL_HASHES}
    return {p:sha256(read_bound(ROOT,p,anchors.get(p))) for p in SOURCES}


class FinalTrainDataset(Dataset):
    def __init__(self,view):
        require(view in VIEWS,'Unknown view');self.view=view
        read_bound(ROOT,'data/manifests/train_validation_manifest.csv',HISTORICAL_MANIFEST_HASH)
        read_bound(ROOT,'docs/class_mapping.json',SOURCE_MAPPING_HASH)
        self.base=HistoricalPolicyDataset(ROOT,POLICY,view,'train',threshold=0)
        require(self.base.samples==self.base.base.samples,'Primary TRAIN changed')
        self.samples=self.base.samples;self.mapping=self.base.class_to_idx
        validate_train_rows(self.samples,self.mapping)
        self.binding=dict(N=len(self.samples),K=51,class_to_idx=self.mapping,cutoff_exclusive='2020-02',
            train_rows_sha256=sha256(json_bytes(self.samples)),manifest_sha256=HISTORICAL_MANIFEST_HASH,
            mapping_sha256=SOURCE_MAPPING_HASH,config_id=self.base.preprocessing_config_id,
            selection=self.base.selection,selection_id=self.base.selection_id)

    def __len__(self):return len(self.samples)

    def __getitem__(self,i):
        x,y=self.base[i];require(x.shape==(1,64,64) and x.dtype==torch.float32 and torch.isfinite(x).all().item()
            and x.min()>=0 and x.max()<=1 and y==self.mapping[self.samples[i]['family']],'Training image/label mismatch')
        return x,y,self.samples[i]['sha']


def make_model(seed,view):
    validate_seed(seed);require(view in VIEWS,'Unknown view')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=MalSBSLCNet(51) if view=='sbsmi' else CompactViewNet(51)
    require(sum(p.numel() for p in model.parameters())==(140467 if view=='sbsmi' else 59059),'Architecture changed')
    return model


def make_loader(dataset,seed,pin_memory=True):
    validate_seed(seed);require(len(dataset)%32!=1,'Singleton BatchNorm batch')
    g=torch.Generator().manual_seed(seed)
    return DataLoader(dataset,batch_size=32,shuffle=True,drop_last=False,num_workers=0,pin_memory=pin_memory,generator=g),g


def preflight(view):
    dataset=FinalTrainDataset(view);missing=[]
    folder=dataset.base.base.dataset_dir/'train';ext='.png' if view=='sbsmi' else '.npy'
    # Existence only; legacy decoder verifies image/report hashes on actual reads.
    for r in dataset.samples:
        for suffix in (ext,'.json'):
            if not (folder/(r['sha']+suffix)).is_file():missing.append(r['sha']+suffix)
    require(not missing,'Missing TRAIN image/report: '+str(missing[:3]))
    run=ROOT/'data/derived/final_seed_preflight'/view/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True,exist_ok=False)
    receipt=dict(status='METADATA_AND_EXISTENCE_PASS',view=view,binding=dataset.binding,code_sha256=source_hashes(),
                 environment=environment(),image_content_decode='performed on training reads, not repeated here')
    write_new_json(run/'receipt.json',receipt)
    return dict(receipt=(run/'receipt.json').relative_to(ROOT).as_posix(),sha256=sha256((run/'receipt.json').read_bytes()))


def verify_preflight(path,digest,view):
    p=Path(path);p=p if p.is_absolute() else ROOT/p
    require(p.resolve().is_relative_to(ROOT/'data/derived/final_seed_preflight'),'Wrong receipt root')
    r=json.loads(read_bound(ROOT,p.relative_to(ROOT),digest))
    require(r['status']=='METADATA_AND_EXISTENCE_PASS' and r['view']==view and r['code_sha256']==source_hashes(),'Stale/wrong preflight')
    require(r['environment']==environment() and environment()['version_parity'],'Environment parity unresolved')
    dataset=FinalTrainDataset(view);require(dataset.binding==r['binding'],'TRAIN binding changed');return dataset


def verify_run(run,seed,view):
    validate_seed(seed);require(view in VIEWS,'Unknown view')
    p=Path(run);p=p if p.is_absolute() else ROOT/p;p=p.resolve(strict=True)
    require(p.parent==(ROOT/OUTPUT/f'seed_{seed}'/view).resolve(),'Wrong seed/view run root')
    require(not (p/'FAILED.json').exists(),'Run marked FAILED')
    complete=json.loads((p/'COMPLETE.json').read_bytes());config=json.loads((p/'config.json').read_bytes())
    require(config['seed']==seed and config['view']==view and config['run_spec']==dict(RUN_SPEC,seed=seed),'Seed/config mismatch')
    require(config['dataset']==FinalTrainDataset(view).binding,'Dataset changed')
    require(config['code_sha256']==source_hashes(),'Training source changed')
    require(complete['config_sha256']==sha256(json_bytes(config)) and len(complete['epochs'])==100 and
            [x['epoch'] for x in complete['epochs']]==list(range(1,101)) and complete['selected']==complete['epochs'][-1], 'Incomplete epoch100')
    selected=json.loads((p/'epoch_100.json').read_bytes());require(selected==complete['selected'],'Selected metadata mismatch')
    raw=read_bound(ROOT,(p/'epoch_100.pt').relative_to(ROOT),selected['sha256'])
    return p,config,complete,raw


def load_frozen_run(run,seed,view,expected_complete_sha):
    p,config,complete,raw=verify_run(run,seed,view)
    require(sha256((p/'COMPLETE.json').read_bytes())==expected_complete_sha,'Completion hash mismatch')
    payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    require(payload['epoch']==100 and payload['config_sha256']==complete['config_sha256'] and json_bytes(payload['config'])==json_bytes(config),'Checkpoint binding mismatch')
    model=make_model(seed,view);model.load_state_dict(payload['model_state_dict'],strict=True)
    require(all(torch.isfinite(t).all().item() for t in model.state_dict().values()),'Nonfinite model state')
    model.eval().requires_grad_(False);return model,config,sha256(raw)


def train(seed,view,receipt,digest,execute=False):
    if not execute:raise PermissionError('Explicit owner training approval/--execute-training required')
    validate_seed(seed);dataset=verify_preflight(receipt,digest,view)
    require(torch.cuda.is_available(),'CUDA required; no CPU fallback')
    require(shutil.disk_usage(ROOT).free>=2*1024**3,'At least2GiB free disk required')
    torch.cuda.manual_seed_all(seed);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    # CPU initialization consumes the same fresh seeded RNG as the baseline.
    model=make_model(seed,view).to('cuda');optimizer=make_optimizer(model);criterion=nn.CrossEntropyLoss()
    loader,generator=make_loader(dataset,seed)
    output=ROOT/OUTPUT/f'seed_{seed}'/view/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False);code=source_hashes()
    config=dict(seed=seed,view=view,run_spec=dict(RUN_SPEC,seed=seed),dataset=dataset.binding,
        code_sha256=code,preflight_sha256=digest,environment=environment(),adam_defaults=optimizer.defaults,
        gpu=torch.cuda.get_device_name(0),git=git_snapshot(ROOT),future_test_used=False)
    config_hash=sha256(json_bytes(config));write_new_json(output/'config.json',config)
    for name,h in code.items():
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(read_bound(ROOT,name,h))
    epochs=[]
    try:
        save_checkpoint(output/'initial_state.pt',dict(seed=seed,config_sha256=config_hash,
            model_state_dict=model.state_dict(),torch_rng_state=torch.get_rng_state(),
            cuda_rng_states=torch.cuda.get_rng_state_all(),loader_generator_state=generator.get_state()))
        for epoch in range(1,101):
            model.train();seen=[];loss_sum=0.;correct=0;torch.cuda.synchronize();start=perf_counter();torch.cuda.reset_peak_memory_stats()
            for x,y,ids in loader:
                x,y=x.to('cuda',non_blocking=True),y.to('cuda',non_blocking=True)
                optimizer.zero_grad(set_to_none=True);z=model(x);loss=criterion(z,y)
                require(torch.isfinite(loss).item(),'Nonfinite loss');loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
                optimizer.step();seen.extend(ids);loss_sum+=loss.detach().item()*len(y);correct+=(z.detach().argmax(1)==y).sum().item()
            torch.cuda.synchronize();elapsed=perf_counter()-start;order_hash=check_coverage(seen,[r['sha'] for r in dataset.samples])
            require(source_hashes()==code and FinalTrainDataset(view).binding==dataset.binding,'Source/TRAIN changed')
            state=np.random.get_state()
            payload=dict(epoch=epoch,config=config,config_sha256=config_hash,model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),torch_rng_state=torch.get_rng_state(),
                cuda_rng_states=torch.cuda.get_rng_state_all(),python_rng_state=random.getstate(),
                numpy_rng_state=[state[0],state[1].tolist(),*state[2:]],loader_generator_state=generator.get_state(),order_sha256=order_hash)
            path=output/f'epoch_{epoch:03d}.pt';h=save_checkpoint(path,payload)
            info=dict(epoch=epoch,samples=len(seen),steps=len(loader),loss=loss_sum/len(seen),online_accuracy=correct/len(seen),
                loop_seconds=elapsed,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,checkpoint=path.name,sha256=h)
            write_new_json(output/f'epoch_{epoch:03d}.json',info);epochs.append(info)
            print(f'seed{seed}/{view} epoch{epoch}/100 saved',flush=True)
        write_new_json(output/'COMPLETE.json',dict(epochs=epochs,selected=epochs[-1],config_sha256=config_hash))
    except BaseException as exc:
        write_new_json(output/'FAILED.json',dict(error=type(exc).__name__,message=str(exc),resume=False,retry=False));raise
    return output


def discover(seed,view):
    validate_seed(seed);require(view in VIEWS,'Unknown view')
    paths=list((ROOT/OUTPUT/f'seed_{seed}'/view).glob('*/COMPLETE.json'))
    require(len(paths)==1,'Expected exactly one COMPLETE run; missing/ambiguous seed/view')
    run,_,_,_=verify_run(paths[0].parent,seed,view)
    return dict(run_dir=run.relative_to(ROOT).as_posix(),complete_sha256=sha256((run/'COMPLETE.json').read_bytes()))


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('preflight');a.add_argument('--view',choices=VIEWS,required=True)
    a=s.add_parser('verify');a.add_argument('--view',choices=VIEWS,required=True);a.add_argument('--seed',type=int,choices=SEEDS,required=True)
    a=s.add_parser('train');a.add_argument('--view',choices=VIEWS,required=True);a.add_argument('--seed',type=int,choices=SEEDS,required=True)
    a.add_argument('--preflight',required=True);a.add_argument('--preflight-sha256',required=True);a.add_argument('--execute-training',action='store_true')
    args=p.parse_args()
    if args.command=='preflight':print(json.dumps(preflight(args.view)))
    elif args.command=='verify':print(json.dumps(discover(args.seed,args.view)))
    else:print('Complete:',train(args.seed,args.view,args.preflight,args.preflight_sha256,args.execute_training))


if __name__=='__main__':main()
