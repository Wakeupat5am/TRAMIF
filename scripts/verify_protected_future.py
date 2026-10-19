"""D061 independent saved-input/logit readback. Never computes prediction scores."""
from collections import Counter
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image
import tlsh


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def stamp(row):
    value=datetime.fromisoformat(row['timestamp'])
    require(value.utcoffset() is not None and value.utcoffset().total_seconds()==0,'Readback UTC mismatch')
    return value,row['sha']


def check_logit_array(arrays,rows,mapping):
    ids=[r['sha'] for r in rows];labels=[mapping[r['family']] for r in rows]
    names=sorted(mapping,key=mapping.get);z=arrays['logits']
    require(len(ids)==len(set(ids)) and arrays['sample_identifiers'].tolist()==ids,'Readback ID alignment mismatch')
    require(arrays['labels'].dtype.kind in 'iu' and arrays['labels'].tolist()==labels,'Readback label mismatch')
    require(arrays['class_names'].tolist()==names and z.shape==(len(ids),51)
            and z.dtype==np.float32 and np.isfinite(z).all(),'Readback logits shape/dtype/finite/class mismatch')


def check_witnesses(references,inventory,tables):
    lookup={r['sha']:r for r in references+inventory};expected={r['sha'] for r in inventory}
    keep={};witness_count=0
    for threshold in (0,10,20):
        rows=tables[threshold]
        require(len(rows)==len(expected) and {r['sha'] for r in rows}==expected,'Retention readback coverage mismatch')
        require([r['sha'] for r in rows]==[r['sha'] for r in inventory],'Retention canonical order mismatch')
        keep[threshold]=set()
        for r in rows:
            require(r['threshold']==threshold and r['decision'] in ('KEEP','EXCLUDE'),'Retention value mismatch')
            current=lookup[r['sha']]
            require(all(r[k]==current[k] for k in ('timestamp','family','disarmed_content_sha256')),'Retention identity mismatch')
            if r['decision']=='KEEP':
                require(not r['reference_sha'] and r['reason']=='no_earlier_match','Unexpected KEEP witness')
                keep[threshold].add(r['sha']);continue
            require(r['reference_sha'] in lookup,'Unknown exclusion witness')
            old=lookup[r['reference_sha']]
            require(stamp(old)<stamp(current) and r['reference_timestamp']==old['timestamp'],'Non-earlier witness')
            if r['reason']=='exact_sha256':
                require(old['disarmed_content_sha256']==current['disarmed_content_sha256'],'Invalid content-equality witness')
            else:
                require(r['reason']=='tlsh' and old['tlsh_status']==current['tlsh_status']=='VALID','Invalid TLSH witness status')
                actual=tlsh.diff(current['tlsh'],old['tlsh'])
                require(actual==r['tlsh_distance'] and actual<=threshold,'Invalid TLSH exclusion distance')
            witness_count+=1
    require(keep[20]<=keep[10]<=keep[0],'Readback non-nested retention')
    return keep,witness_count


def verify(root,out,package,references):
    months=tuple(f'2020-{i:02d}' for i in range(4,10));views=('sbsmi','raw-byte','entropy');seeds=(42,43,44,45,46)
    load=lambda name:json.loads((out/name).read_text(encoding='utf-8'))
    all_rows=load('all_future_metadata.json');inventory=load('known_inventory.json');primary=load('primary_manifest.json')
    mapping=package['class_to_idx']
    # Independent canonical reconstruction, without calling discovery/selection code.
    with (root/'data/raw/bodmas_metadata.csv').open(encoding='utf-8-sig',newline='') as f:
        original=[r for r in csv.DictReader(f) if r['timestamp'][:7] in months]
    require(len(all_rows)==len({r['sha'] for r in all_rows})==len(original),'Original Future metadata coverage mismatch')
    canonical=sorted([r for r in original if r['family'].strip() in mapping],key=stamp)
    require([r['sha'] for r in inventory]==[r['sha'] for r in canonical],'Known-family inventory coverage/order mismatch')
    require(all(all(a[k]==b[k] for k in ('sha','timestamp','family')) for a,b in zip(inventory,canonical)),
            'Known-family canonical metadata mismatch')
    for row in inventory:
        stat=(root/'data/raw/altered'/(row['sha']+'.exe')).stat()
        require((stat.st_size,stat.st_mtime_ns)==(row['file_size'],row['input_mtime_ns']),'Source changed since fingerprint')
    tables={t:load(f'retention_{t}.json') for t in (0,10,20)}
    keep,witnesses=check_witnesses(references,inventory,tables)
    require([r['sha'] for r in primary]==[r['sha'] for r in inventory if r['sha'] in keep[0]],'Primary selection mismatch')
    require(all(any(r['month']==m for r in primary) for m in months),'Empty primary month')
    # Bounded independent negative-decision check against ALL earlier rows.
    history=sorted(references+inventory,key=stamp);indices={r['sha']:i for i,r in enumerate(history)}
    rng=random.Random(42);checks=0
    for t in (0,10,20):
        candidates=sorted(keep[t]);sample=rng.sample(candidates,min(12,len(candidates)))
        for sid in sample:
            current=history[indices[sid]]
            for old in history[:indices[sid]]:
                require(old['disarmed_content_sha256']!=current['disarmed_content_sha256'],'KEEP has earlier exact match')
                if old['tlsh_status']==current['tlsh_status']=='VALID':
                    require(tlsh.diff(current['tlsh'],old['tlsh'])>t,'KEEP has earlier forbidden TLSH match')
            checks+=1
    images=load('images_manifest.json')
    require([r['sha'] for r in images]==[r['sha'] for r in primary],'Image identities mismatch')
    image_checks=0
    for row,saved in zip(primary,images):
        for view in views:
            item=saved['views'][view];p=out/item['image'];rp=out/item['report']
            require(digest(p)==item['image_sha256'] and digest(rp)==item['report_sha256'],'Readback image hash mismatch')
            r=json.loads(rp.read_text())
            require(r['sample_identifier']==row['sha'] and r['class_index']==mapping[row['family']]
                    and r['disarmed_content_sha256']==row['disarmed_content_sha256']
                    and r['timestamp']==row['timestamp'] and r['family']==row['family']
                    and r['file_size']==row['file_size'],'Readback image provenance mismatch')
            config_hash=hashlib.sha256(json.dumps(package['preprocessing'][view],sort_keys=True).encode()).hexdigest()
            require(r['preprocessing_config_sha256']==config_hash,'Readback preprocessing hash mismatch')
            if view=='sbsmi':
                with Image.open(p) as im:
                    require(im.format=='PNG' and im.mode=='L' and im.size==(64,64),'PNG decode mismatch')
                    pixels=np.array(im,dtype=np.uint8)
                require(hashlib.sha256(pixels.tobytes()).hexdigest()==r['pixel_sha256'],'Pixel digest mismatch')
            else:
                a=np.load(p,allow_pickle=False)
                require(a.shape==(64,64) and a.dtype==np.dtype('<f4') and np.isfinite(a).all()
                        and a.min()>=0 and a.max()<=1,'Float image readback mismatch')
            image_checks+=1
    logits=load('logits_manifest.json')
    require(len(logits)==15 and {(r['seed'],r['view']) for r in logits}=={(s,v) for s in seeds for v in views},
            'Fifteen output bindings required')
    for r in logits:
        cp=package['seed_records'][str(r['seed'])]['views'][r['view']]
        require(r['checkpoint_sha256']==cp['checkpoint_sha256'] and r['temperature']==cp['temperature']
                and r['state_sha256']==cp['state_sha256'] and r['state_unchanged'] and not r['gradient_updates'],
                'Readback frozen checkpoint/T/state mismatch')
        path=out/r['path'];require(digest(path)==r['sha256'],'Logits file checksum mismatch')
        with np.load(path,allow_pickle=False) as arrays:check_logit_array(arrays,primary,mapping)
        require(json.loads(path.with_suffix('.json').read_text())==r,'Logits manifest/receipt mismatch')
    coverage={m:{kind:sum(r['month']==m and r['kind']==kind for r in all_rows)
                  for kind in ('known_family','unknown_family','benign')} for m in months}
    counts={m:{str(t):dict(kept=sum(r['month']==m and r['decision']=='KEEP' for r in tables[t]),
                              excluded=sum(r['month']==m and r['decision']=='EXCLUDE' for r in tables[t]))
                for t in tables} for m in months}
    require(counts==load('retention_counts.json'),'Retention count report mismatch')
    report=dict(status='PASS_INDEPENDENT_INPUT_READBACK',coverage=coverage,retention=counts,
                complete_months=list(months),known_inventory=len(inventory),primary_samples=len(primary),
                image_checks=image_checks,logit_arrays=15,K=51,all_exclusion_witnesses_checked=witnesses,
                bounded_KEEP_queries=checks,seed=42,quartiles=package['size_stratification'],
                scoring_performed=False,metrics_computed=[],
                limitation='Current source fingerprints pinned; original ZIP/authoritative Future CRC unavailable. Benign binaries not distributed; out-of-K malware not scored or hashed for similarity.',
                original_archive_CRC_verified=False)
    from scripts.build_combined_freeze import write
    write(out/'independent_verification.json',report)
    files={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()}
    return dict(status='READY_FOR_FINAL_SCORING',readback=report,files=files,
                input_package_only=True,final_scoring_authorized=False)
