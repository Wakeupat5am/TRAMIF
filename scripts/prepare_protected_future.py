"""D061 authorized input preparation ONLY. No prediction metrics or fusion calls.

Reads disarmed binaries as bytes. Never executes samples. Unique immutable runs;
failure stops the pipeline, with no automatic retry/resume or silent exclusion.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
from importlib.metadata import version
from time import perf_counter
import zlib
import numpy as np
import PIL
from PIL import Image
import torch
import tlsh
from models.malsbslcnet import MalSBSLCNet
from models.compact_view_net import CompactViewNet
from scripts.check_tlsh import hash_stream
from scripts.raw_byte_stream import stream_to_raw_byte
from scripts.entropy_image_stream import stream_to_entropy_image
from scripts.sbsmi_stream import file_to_sbsmi
from scripts.episode_training_dataset import decode_image
from scripts.build_raw_byte_dataset import DigestReader
from scripts.build_combined_freeze import digest, write

ROOT=Path(__file__).resolve().parents[1]
BUNDLE='data/derived/d060_final_bundle/20261018_081222_184175/bundle.json'
BUNDLE_SHA='f52f70c1c23dc815258c71f7f19c578ef7592b76c8e84b9df1361dab8ae5c1b3'
METADATA='data/raw/bodmas_metadata.csv'
METADATA_SHA='5f1dc793cac4a540ef9e0c40fd7897142095621790d3c6f2d6637971806f4c50'
MONTHS=tuple(f'2020-{i:02d}' for i in range(4,10))
SEEDS=(42,43,44,45,46)
VIEWS=('sbsmi','raw-byte','entropy')


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical_time(row):
    dt=datetime.fromisoformat(row['timestamp'])
    require(dt.utcoffset() is not None and dt.utcoffset().total_seconds()==0,'UTC required')
    return dt,row['sha']


def config_id(config):return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()


def contract(root=ROOT):
    require(digest(root/BUNDLE)==BUNDLE_SHA,'D060 bundle checksum mismatch')
    bundle=json.loads((root/BUNDLE).read_text());parts=bundle['components']
    for part in parts.values():
        require(digest(root/part['directory']/'receipt.json')==part['receipt_sha256'],'D060 component receipt mismatch')
    base=root/parts['combined_freeze']['directory']
    require(digest(base/'package.json')==bundle['package_sha256'],'D060 package mismatch')
    package=json.loads((base/'package.json').read_text())
    require(digest(root/'docs/FINAL_STATISTICAL_PROTOCOL.md')==package['specification_sha256']
            and digest(root/'docs/D060_O3_SCOPE_AMENDMENT.md')==package['amendment_sha256'],'Protocol/amendment mismatch')
    for p,h in package['source_hashes'].items():require(digest(root/p)==h,'Frozen source dependency mismatch: '+p)
    assets=json.loads((base/'assets_manifest.json').read_text())
    require(digest(base/'assets_manifest.json')==package['asset_manifest_sha256'],'Asset manifest mismatch')
    dependencies={'scripts/check_tlsh.py','scripts/sbsmi_stream.py','scripts/raw_byte_stream.py',
                  'scripts/entropy_stream.py','scripts/entropy_core.py','scripts/entropy_image_stream.py',
                  'scripts/episode_training_dataset.py','models/malsbslcnet.py','models/compact_view_net.py'}
    recipe_pins={'scripts/sbsmi_stream.py':package['preprocessing']['sbsmi']['sbsmi_stream_source_sha256'],
                 'scripts/raw_byte_stream.py':package['preprocessing']['raw-byte']['raw_byte_stream_source_sha256'],
                 **package['preprocessing']['entropy']['source_sha256']}
    for seed in SEEDS:
        for view in VIEWS:
            cp=package['seed_records'][str(seed)]['views'][view]
            require(cp['seed']==seed and cp['view']==view and cp['K']==51 and cp['epoch']==100
                    and cp['cutoff_exclusive']=='2020-02','Checkpoint identity mismatch')
            require(digest(root/cp['checkpoint'])==cp['checkpoint_sha256'],'Registered checkpoint checksum mismatch')
    refbase=root/parts['historical_reference_addendum']['directory']
    rr=json.loads((refbase/'receipt.json').read_text())
    train_config_path='data/derived/train_tlsh_v1/config.json'
    require(digest(root/train_config_path)==rr['source_assets'][train_config_path],'Historical TLSH helper config mismatch')
    recipe_pins['scripts/check_tlsh.py']=json.loads((root/train_config_path).read_text())['hash_helper_sha256']
    dependency_pins={}
    for p in dependencies:
        expected=assets.get(p,recipe_pins.get(p))
        require(expected is not None,'Unbound preparation dependency: '+p)
        require(digest(root/p)==expected,'Preparation dependency source mismatch: '+p)
        dependency_pins[p]=expected
    refs=refbase/'historical_references.json'
    require(digest(refs)==rr['files_sha256']['historical_references.json'],'Historical retention reference mismatch')
    references=json.loads(refs.read_text())
    require(len(references)==len({r['sha'] for r in references})==26324,'Historical reference coverage mismatch')
    require(all(canonical_time(r)[0].strftime('%Y-%m')<'2020-04' for r in references),'Historical reference cutoff mismatch')
    return package,references,dict(bundle_sha256=BUNDLE_SHA,package_sha256=bundle['package_sha256'],
                                  reference_sha256=digest(refs),dependencies=dependency_pins)


def discover(metadata,mapping,binary_dir):
    require(set(mapping.values())==set(range(51)),'Canonical K51 required')
    seen=set();future=[]
    for row in metadata:
        sha=row['sha'].strip().lower();family=row['family'].strip()
        item=dict(sha=sha,timestamp=row['timestamp'],family=family)
        # Original CSV also contains pre-study benign records (e.g.2016) without
        # timezones. They are outside the authorized Future selection, not inputs
        # to retention/inference. Do not coerce them or relax UTC for selected rows.
        observed=datetime.fromisoformat(row['timestamp']);month=observed.strftime('%Y-%m')
        if month not in MONTHS:continue
        require(re.fullmatch('[0-9a-f]{64}',sha) is not None,'Invalid Future metadata identifier')
        require(sha not in seen,'Duplicate original Future metadata ID');seen.add(sha)
        canonical_time(item)
        kind='known_family' if family in mapping else ('benign' if not family else 'unknown_family')
        item.update(month=month,kind=kind,label=mapping.get(family),
                    eligibility_reason='eligible_K51' if kind=='known_family' else
                        ('excluded_benign_per_BODMAS_README' if kind=='benign' else 'excluded_family_not_in_frozen_K51'))
        if kind!='benign':
            path=binary_dir/(sha+'.exe')
            require(path.is_file(),'Missing malware binary: '+sha)
            stat=path.stat();require(stat.st_size>0,'Empty malware binary: '+sha)
            item.update(file_size=stat.st_size,input_mtime_ns=stat.st_mtime_ns)
        else:item.update(file_size=None,input_mtime_ns=None)
        future.append(item)
    future.sort(key=canonical_time)
    require({r['month'] for r in future}==set(MONTHS),'Missing Future metadata month')
    require(all(any(r['month']==m and r['kind']=='known_family' for r in future) for m in MONTHS),
            'Empty eligible Future month')
    return future


def inventory_one(task):
    root,row=task;path=Path(root)/'data/raw/altered'/(row['sha']+'.exe')
    before=path.stat();require(before.st_size==row['file_size'] and before.st_mtime_ns==row['input_mtime_ns'],
                               'Binary changed since discovery')
    with path.open('rb') as f:
        reader=DigestReader(f);value=hash_stream(reader,65536)
        # hash_stream may stop early only for a file shorter than five bytes.
        require(reader.byte_count==row['file_size'],'Incomplete source read')
    after=path.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Binary changed while hashing')
    return dict(row,disarmed_content_sha256=reader.sha256.hexdigest(),crc32=f'{reader.crc32&0xffffffff:08x}',
                tlsh=value,tlsh_status='TNULL' if value=='TNULL' else 'VALID',split='future')


def retain(references,targets,progress=None):
    """Exact direct-earlier policy, with cached TLSH objects and early stopping.

    Group references are witness links, not an independently estimated population
    partition. No new connected-component exclusion rule or statistical resampling.
    """
    refs=sorted(references,key=canonical_time);targets=sorted(targets,key=canonical_time)
    require(len({r['sha'] for r in refs+targets})==len(refs)+len(targets),'Historical/Future duplicate IDs')
    require(canonical_time(refs[-1])<canonical_time(targets[0]),'Historical/Future chronology overlap')
    previous=[];objects=[];first_content={};tables={t:[] for t in (0,10,20)}
    def append(row):
        obj=None
        if row['tlsh_status']=='VALID':
            obj=tlsh.Tlsh();obj.fromTlshStr(row['tlsh'])
        require(row['tlsh_status'] in ('VALID','TNULL'),'Invalid TLSH status')
        require((obj is None)==(row['tlsh']=='TNULL'),'Inconsistent TLSH digest')
        first_content.setdefault(row['disarmed_content_sha256'],len(previous))
        previous.append(row);objects.append(obj)
    for r in refs:append(r)
    for n,row in enumerate(targets,1):
        exact=first_content.get(row['disarmed_content_sha256']);witness={t:None for t in tables}
        if exact is not None:
            witness={t:(exact,'exact_sha256',None) for t in tables}
        elif row['tlsh_status']=='VALID':
            query=tlsh.Tlsh();query.fromTlshStr(row['tlsh'])
            for j,obj in enumerate(objects):
                if obj is None:continue
                distance=query.diff(obj)
                if distance<=20:
                    for t in tables:
                        if distance<=t and witness[t] is None:witness[t]=(j,'tlsh',distance)
                    if all(x is not None for x in witness.values()):break
        for threshold in tables:
            w=witness[threshold]
            base=dict(sha=row['sha'],timestamp=row['timestamp'],family=row['family'],month=row['month'],
                      threshold=threshold,decision='EXCLUDE' if w else 'KEEP',
                      disarmed_content_sha256=row['disarmed_content_sha256'])
            if w:
                j,reason,distance=w;old=previous[j]
                require(canonical_time(old)<canonical_time(row),'Witness chronology violation')
                base.update(reason=reason,reference_sha=old['sha'],reference_timestamp=old['timestamp'],
                            reference_split=old['split'],tlsh_distance=distance,
                            group_reference_kind='earlier_witness_link',group_reference_sha=old['sha'])
            else:base.update(reason='no_earlier_match',reference_sha='',reference_timestamp='',
                             reference_split='',tlsh_distance=None,group_reference_kind='self',group_reference_sha=row['sha'])
            tables[threshold].append(base)
        append(row)
        if progress and (n%500==0 or n==len(targets)):progress(n,len(targets))
    keep=[{r['sha'] for r in tables[t] if r['decision']=='KEEP'} for t in (0,10,20)]
    require(keep[2]<=keep[1]<=keep[0],'Non-nested retention decisions')
    require(all(any(r['month']==m and r['decision']=='KEEP' for r in tables[0]) for m in MONTHS),'Empty primary retained month')
    return tables


def make_images(task):
    root,out,row,configs,mapping=task;root=Path(root);out=Path(out)
    path=root/'data/raw/altered'/(row['sha']+'.exe');results={}
    for view in VIEWS:
        stat=path.stat();require((stat.st_size,stat.st_mtime_ns)==(row['file_size'],row['input_mtime_ns']),'Changed image source')
        folder=out/'images'/view;folder.mkdir(parents=True,exist_ok=True)
        p=folder/(row['sha']+('.png' if view=='sbsmi' else '.npy'))
        require(not p.exists(),'Do not overwrite existing Future image')
        report=dict(sample_identifier=row['sha'],timestamp=row['timestamp'],family=row['family'],
                    class_index=mapping[row['family']],file_size=row['file_size'],
                    disarmed_content_sha256=row['disarmed_content_sha256'],status='PASS',
                    preprocessing_config_sha256=config_id(configs[view]),
                    selection='D061 primary retention0; sensitivity subsets nested')
        if view=='sbsmi':
            matrix=np.asarray(file_to_sbsmi(path,l=6,chunk_size=65536),dtype=np.uint8)
            require(matrix.shape==(64,64),'Invalid SBSMI dimensions')
            Image.fromarray(matrix).save(p,format='PNG')
            report.update(png_sha256=digest(p),pixel_sha256=hashlib.sha256(matrix.tobytes()).hexdigest())
            # SBSMI helper accepts a path rather than a digesting stream.
            require(digest(path)==row['disarmed_content_sha256'],'SBSMI source changed')
        else:
            with path.open('rb') as f:
                reader=DigestReader(f)
                transform=stream_to_raw_byte if view=='raw-byte' else stream_to_entropy_image
                matrix=transform(reader,row['file_size'],chunk_size=65536)
                require(reader.byte_count==row['file_size'] and reader.sha256.hexdigest()==row['disarmed_content_sha256'],
                        'Cross-view source content mismatch')
            matrix=np.asarray(matrix,dtype='<f4')
            require(matrix.shape==(64,64) and np.isfinite(matrix).all() and matrix.min()>=0 and matrix.max()<=1,
                    'Invalid float image values')
            with p.open('xb') as f:np.save(f,matrix,allow_pickle=False)
            report.update(npy_sha256=digest(p))
        tensor=decode_image(p.read_bytes(),report,view)
        require(tuple(tensor.shape)==(1,64,64) and tensor.dtype==torch.float32,'Decoder contract mismatch')
        rp=p.with_suffix('.json');write(rp,report)
        results[view]=dict(image=p.relative_to(out).as_posix(),report=rp.relative_to(out).as_posix(),
                           image_sha256=digest(p),report_sha256=digest(rp))
    return dict(sha=row['sha'],views=results)


def validate_image(out,row,view,item,config,mapping):
    p=out/item['image'];rp=out/item['report']
    require(digest(p)==item['image_sha256'] and digest(rp)==item['report_sha256'],'Image/report checksum mismatch')
    report=json.loads(rp.read_text())
    for k,value in [('sample_identifier',row['sha']),('family',row['family']),('timestamp',row['timestamp']),
                    ('class_index',mapping[row['family']]),('disarmed_content_sha256',row['disarmed_content_sha256']),
                    ('file_size',row['file_size']),('preprocessing_config_sha256',config_id(config)),('status','PASS')]:
        require(report[k]==value,'Image provenance mismatch: '+k)
    return decode_image(p.read_bytes(),report,view)


def model_state_hash(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def infer_frozen(model,batch):
    require(all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),'Frozen eval required')
    require(batch.dtype==torch.float32 and batch.ndim==4 and tuple(batch.shape[1:])==(1,64,64)
            and torch.isfinite(batch).all().item() and batch.min().item()>=0 and batch.max().item()<=1,'Invalid input tensor')
    with torch.inference_mode():output=model(batch)
    require(output.shape==(len(batch),51) and torch.isfinite(output).all().item() and not output.requires_grad,'Invalid K51 logits')
    return output.detach().cpu().numpy().copy()


def generate_logits(root,out,package,rows,images):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    mapping=package['class_to_idx'];classes=np.array(sorted(mapping,key=mapping.get));ids=np.array([r['sha'] for r in rows])
    labels=np.array([mapping[r['family']] for r in rows],dtype=np.int64)
    records=[];lookup={r['sha']:r['views'] for r in images}
    for seed in SEEDS:
        for view in VIEWS:
            cp=package['seed_records'][str(seed)]['views'][view]
            require(digest(root/cp['checkpoint'])==cp['checkpoint_sha256'],'Checkpoint changed before inference')
            payload=torch.load(root/cp['checkpoint'],map_location='cpu',weights_only=True)
            saved=json.loads((root/cp['run']/'config.json').read_text())
            require(payload['epoch']==100 and json.dumps(payload['config'],sort_keys=True)==json.dumps(saved,sort_keys=True),
                    'Checkpoint embedded config/epoch mismatch')
            cm=saved.get('class_to_idx',saved.get('dataset',{}).get('class_to_idx'))
            require(cm==mapping and saved['seed']==seed,'Checkpoint mapping/seed mismatch')
            model=MalSBSLCNet(51) if view=='sbsmi' else CompactViewNet(51)
            model.load_state_dict(payload['model_state_dict'],strict=True);model.eval().requires_grad_(False)
            before=model_state_hash(model);require(before==cp['state_sha256'],'Checkpoint state differs from D054')
            model.to(device);chunks=[]
            for start in range(0,len(rows),32):
                part=rows[start:start+32]
                tensors=[validate_image(out,r,view,lookup[r['sha']][view],package['preprocessing'][view],mapping) for r in part]
                batch=torch.stack(tensors).to(device)
                output=infer_frozen(model,batch)
                if start==0:require(np.array_equal(output,infer_frozen(model,batch)),'Non-deterministic frozen inference')
                chunks.append(output)
            require(model_state_hash(model)==before and all(p.grad is None for p in model.parameters()),'Weights/BN/grad mutation')
            logits=np.concatenate(chunks);folder=out/'logits';folder.mkdir(exist_ok=True)
            path=folder/f'seed_{seed}_{view}.npz'
            with path.open('xb') as f:np.savez_compressed(f,logits=logits,sample_identifiers=ids,class_names=classes,labels=labels)
            record=dict(seed=seed,view=view,path=path.relative_to(out).as_posix(),sha256=digest(path),shape=list(logits.shape),
                        checkpoint=cp['checkpoint'],checkpoint_sha256=cp['checkpoint_sha256'],temperature=cp['temperature'],
                        calibration='frozen February; NOT applied/refit for raw logits',state_unchanged=True,
                        state_sha256=before,device=str(device),batch_size=32,gradient_updates=False)
            records.append(record);write(path.with_suffix('.json'),record)
            print(f'Logits {seed}/{view}: saved {tuple(logits.shape)}',flush=True)
            del model,payload
            if device.type=='cuda':torch.cuda.empty_cache()
    return records


def execute(out,workers=4):
    start=perf_counter();package,references,binding=contract()
    require(digest(ROOT/METADATA)==METADATA_SHA,'Original metadata hash mismatch')
    sources={p:digest(ROOT/p) for p in ['scripts/prepare_protected_future.py','scripts/verify_protected_future.py',
                                      'tests/test_protected_future.py']}
    write(out/'config.json',dict(decision='D061',authorization='user pasted D061 preparation only',
        owner_request_sha256=digest(Path(r'C:\Users\hoang\.codex\attachments\a101ac38-7d75-4384-bccd-655273fa8ae8\Pasted text.txt')),
        metadata_sha256=METADATA_SHA,binding=binding,source_sha256=sources,workers=workers,
        environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
                         pillow=PIL.__version__,py_tlsh=version('py-tlsh'),cuda_runtime=torch.version.cuda),
        scoring_authorized=False,Future_preparation_authorized=True,archive_CRC_reference_available=False))
    for name in {*sources,*binding['dependencies']}:
        destination=out/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('xb') as f:f.write((ROOT/name).read_bytes())
    with (ROOT/METADATA).open(encoding='utf-8-sig',newline='') as f:metadata=list(csv.DictReader(f))
    future=discover(metadata,package['class_to_idx'],ROOT/'data/raw/altered')
    write(out/'all_future_metadata.json',future)
    known=[r for r in future if r['kind']=='known_family']
    require(not {r['sha'] for r in known}&{r['sha'] for r in references},'Future/historical ID overlap')
    inventory=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n,row in enumerate(pool.map(inventory_one,[(str(ROOT),r) for r in known]),1):
            inventory.append(row)
            if n%500==0 or n==len(known):print(f'Source inventory: {n}/{len(known)}',flush=True)
    write(out/'known_inventory.json',inventory)
    tables=retain(references,inventory,lambda n,total:print(f'Retention: {n}/{total}',flush=True))
    for t,rows in tables.items():write(out/f'retention_{t}.json',rows)
    keep={r['sha'] for r in tables[0] if r['decision']=='KEEP'};eligible=[r for r in inventory if r['sha'] in keep]
    write(out/'primary_manifest.json',eligible)
    counts={m:{str(t):dict(kept=sum(r['month']==m and r['decision']=='KEEP' for r in tables[t]),
                             excluded=sum(r['month']==m and r['decision']=='EXCLUDE' for r in tables[t])) for t in tables} for m in MONTHS}
    write(out/'retention_counts.json',counts)
    print('Retention counts:',json.dumps(counts),flush=True)
    images=[]
    tasks=((str(ROOT),str(out),r,package['preprocessing'],package['class_to_idx']) for r in eligible)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n,item in enumerate(pool.map(make_images,tasks,chunksize=2),1):
            images.append(item)
            if n%100==0 or n==len(eligible):print(f'Three-view images: {n}/{len(eligible)}',flush=True)
    write(out/'images_manifest.json',images)
    require([x['sha'] for x in images]==[r['sha'] for r in eligible],'Three-view identity/order mismatch')
    for row,image in zip(eligible,images):
        for view in VIEWS:validate_image(out,row,view,image['views'][view],package['preprocessing'][view],package['class_to_idx'])
    write(out/'IMAGES_VERIFIED.json',dict(status='PASS',samples=len(eligible),images=3*len(eligible),manifest_sha256=digest(out/'images_manifest.json')))
    logits=generate_logits(ROOT,out,package,eligible,images);write(out/'logits_manifest.json',logits)
    for p,h in sources.items():require(digest(ROOT/p)==h,'D061 source changed during execution')
    contract() # immutable contract must still match before completion/readback.
    from scripts.verify_protected_future import verify
    result=verify(ROOT,out,package,references)
    result.update(elapsed_seconds=perf_counter()-start,source_sha256=sources,metadata_sha256=METADATA_SHA,
                  binding=binding,scoring_performed=False,scoring_authorized=False)
    write(out/'receipt.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute-authorized-preparation',action='store_true')
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if not a.execute_authorized_preparation:p.error('Explicit D061 preparation authorization flag required')
    require(a.workers in (1,2,4),'Bounded CPU workers1/2/4 only')
    output=ROOT/'data/derived/protected_future_inputs_v1'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    try:result=execute(output,a.workers)
    except BaseException as exc:
        write(output/'FAILED.json',dict(status='BLOCKED',error=type(exc).__name__,message=str(exc),
                                      scoring_performed=False,automatic_retry=False));raise
    print(json.dumps(dict(status=result['status'],output=str(output),scoring_performed=False)))


if __name__=='__main__':main()
