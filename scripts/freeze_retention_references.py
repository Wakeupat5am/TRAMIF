"""D060 immutable historical-reference addendum; no pairs/graph/Future reads."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile
from scripts.build_combined_freeze import ROOT, BoundInputs, digest, write
from scripts.backup_combined_freeze import restore_archive
from scripts.final_multiseed_evaluator import chronology, require

CSV_PINS = {'train':'3a4446fc91d5d87a23a4ca683c65f21cf51768bc6daa183caf5e8db8abebae42',
            'validation':'cec2c947554fc737de0a84a6a2d71631b7b54fceba16285771cb1dad0bb11506'}


def validate_inventory(rows, config, summary, canonical, expected_count, split):
    config_id=hashlib.sha256(json.dumps(config,sort_keys=True).encode('utf-8')).hexdigest()
    require(summary['config_sha256']==config_id and summary['errors']==0
            and summary['processed_samples']==expected_count
            and summary[f'complete_{split}_inventory'] is True, 'Historical inventory incomplete/config mismatch')
    require(len(rows)==len({r['sha'] for r in rows})==expected_count,'Inventory count/unique ID mismatch')
    expected={r['sha']:r for r in canonical if r['split']==split}
    require(set(expected)=={r['sha'] for r in rows},'Inventory/canonical ID mismatch')
    for r in rows:
        dt,_=chronology(r)
        start,stop=('2019-08','2020-02') if split=='train' else ('2020-02','2020-04')
        require(start<=dt.strftime('%Y-%m')<stop,'Historical reference outside cutoff')
        require(r['config_sha256']==config_id and r['split']==split
                and all(r[k]==expected[r['sha']][k] for k in ('timestamp','family')),
                'Reference identity/config mismatch')
        require(int(r['file_size'])>0,'Invalid reference size')
    return sorted(rows,key=chronology)


def execute(package_dir,output):
    e=BoundInputs(ROOT);package=e.js((package_dir/'package.json').relative_to(ROOT).as_posix())
    canonical=list(csv.DictReader(io.StringIO(e.read('data/manifests/train_validation_manifest.csv',
                      package['preprocessing']['sbsmi']['manifest_sha256']).decode('utf-8-sig'))))
    references=[]
    for split,count in [('train',18061),('validation',8263)]:
        base=f'data/derived/{split}_tlsh_v1';summary=e.js(base+'/summary.json');config=e.js(base+'/config.json')
        require(summary['csv_sha256']==CSV_PINS[split],'Registered historical CSV pin mismatch')
        rows=list(csv.DictReader(io.StringIO(e.read(f'{base}/{split}_tlsh.csv',CSV_PINS[split]).decode('utf-8-sig'))))
        rows=validate_inventory(rows,config,summary,canonical,count,split)
        require(config['zip_index_sha256']==package['preprocessing']['sbsmi']['zip_index_sha256'],
                'Reference inventory/index provenance mismatch')
        references.extend({k:r[k] for k in ('sha','split','timestamp','family','file_size',
                           'disarmed_content_sha256','tlsh','tlsh_status')} for r in rows)
    require(len({r['sha'] for r in references})==26324,'Cross-split reference overlap')
    write(output/'historical_references.json',references)
    write(output/'reference_binding.json',dict(status='PASS_HISTORICAL_REFERENCE_BINDING',
        base_package=(package_dir/'package.json').relative_to(ROOT).as_posix(),
        base_package_sha256=digest(package_dir/'package.json'),records=26324,
        train=18061,validation=8263,references_include_excluded=True,
        cutoff_exclusive='2020-04',labels_used_for_similarity=False,pairs_recomputed=0,
        graph_rebuilt=False,Future_access=False,source_assets=e.assets))
    for p in ['scripts/freeze_retention_references.py','tests/test_retention_reference_binding.py']:
        e.read(p)
    for p in output.iterdir():
        if p.is_file():e.add(p.relative_to(ROOT).as_posix(),digest(p))
    archive=output/'reference_addendum.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p,h in e.assets.items():
            require(digest(ROOT/p)==h,'Addendum source changed');z.write(ROOT/p,p)
    restored=output/'restored';restore_archive(archive,restored,e.assets)
    write(output/'receipt.json',dict(status='PASS_REFERENCE_ADDENDUM_LOCAL_RESTORE',
        files=len(e.assets),archive_sha256=digest(archive),off_device=False,Future_access=False,
        source_assets=e.assets,files_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file()}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package-dir',required=True);a=p.parse_args()
    package=Path(a.package_dir).resolve(strict=True)
    require(package.is_relative_to(ROOT/'data/derived/combined_freeze_v1'),'Wrong base package')
    output=ROOT/'data/derived/d060_retention_reference'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    try:execute(package,output)
    except BaseException as exc:
        write(output/'FAILED.json',dict(status='FAIL',message=str(exc)));raise
    print('Historical reference addendum:',output)


if __name__=='__main__':main()
