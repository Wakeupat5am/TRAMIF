"""Independent stdlib readback of D054 outputs; no model/numpy dependency."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(42,43,44,45,46)
VIEWS=('sbsmi','raw-byte','entropy')


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(blob):return hashlib.sha256(blob).hexdigest()


def statistics_readback(rows,summaries):
    require(len(rows)==15 and {(r['seed'],r['view']) for r in rows}=={(s,v) for s in SEEDS for v in VIEWS},'All five seeds required')
    require(len(summaries)==18,'Expected18 per-view metric summaries')
    for view in VIEWS:
        rr=[r for r in rows if r['view']==view]
        for metric in ('temperature','L','mmd2','d','Q_weight','QD_weight'):
            values=[r[metric] for r in rr];matches=[r for r in summaries if r['view']==view and r['metric']==metric]
            require(len(matches)==1,'Missing/duplicate descriptive summary');saved=matches[0]
            expected=dict(mean=math.fsum(values)/5,SD=statistics.stdev(values),min=min(values),max=max(values))
            require(saved['N']==5,'Five-seed denominator changed')
            for k,value in expected.items():require(math.isfinite(saved[k]) and math.isclose(saved[k],value,rel_tol=1e-12,abs_tol=1e-12),'Independent '+k+' mismatch')
    for seed in SEEDS:
        rr=sorted([r for r in rows if r['seed']==seed],key=lambda r:VIEWS.index(r['view']))
        for field,score in [('Q_weight',[-r['L'] for r in rr]),('QD_weight',[-r['L']-r['d'] for r in rr])]:
            weights=[math.exp(x-max(score)) for x in score];total=math.fsum(weights)
            for row,w in zip(rr,weights):require(math.isclose(row[field],w/total,rel_tol=1e-12,abs_tol=1e-12),'Independent math.softmax mismatch')


def execute(run,receipt_hash):
    run=Path(run);run=run if run.is_absolute() else ROOT/run;run=run.resolve(strict=True)
    require(run.is_relative_to(ROOT/'data/derived/final_multiseed_verification'),'Wrong verifier output root')
    raw=(run/'receipt.json').read_bytes();require(digest(raw)==receipt_hash,'Verification receipt checksum mismatch');receipt=json.loads(raw)
    require(receipt['status']=='PASS' and receipt['checkpoints_verified']==15 and receipt['processing_seeds_verified']==5,'Verification incomplete')
    for name,h in receipt['files'].items():
        p=run/name;require(p.resolve().parent==run,'Artifact path escape');require(digest(p.read_bytes())==h,'Output checksum mismatch:'+name)
    summary=json.loads((run/'summary.json').read_bytes());statistics_readback(summary['rows'],summary['descriptive_statistics'])
    require(summary['seeds']==list(SEEDS) and summary['SD_ddof']==1 and summary['beta']==1 and summary['gamma']==0 and not summary['U_used'] and not summary['future_access'],'Summary protocol mismatch')
    for file,key in [('seed_view_results.csv','rows'),('descriptive_statistics.csv','descriptive_statistics'),('checkpoints.csv','checkpoints')]:
        table=list(csv.DictReader(io.StringIO((run/file).read_text(encoding='utf-8'))));expected=summary[key]
        require(len(table)==len(expected),'CSV row count mismatch')
        for saved,row in zip(table,expected):
            require(set(saved)==set(row),'CSV field mismatch')
            for name,value in row.items():require(saved[name]==('' if value is None else str(value)),'CSV/JSON discrepancy')
    output=ROOT/'data/derived/final_multiseed_readback'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f');output.mkdir(parents=True,exist_ok=False)
    result=dict(status='PASS',verification_receipt=run.relative_to(ROOT).as_posix()+'/receipt.json',verification_receipt_sha256=receipt_hash,
        outputs_verified=len(receipt['files']),rows=15,descriptive_rows=18,SD_ddof=1,independent_numerics='Python math.fsum/statistics.stdev/softmax; no NumPy',
        source_sha256=digest(Path(__file__).read_bytes()),future_access=False)
    blob=(json.dumps(result,sort_keys=True,indent=2)+'\n').encode();(output/'receipt.json').write_bytes(blob)
    print(json.dumps(dict(status='PASS',receipt=(output/'receipt.json').relative_to(ROOT).as_posix(),sha256=digest(blob))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--receipt-sha256',required=True);a=p.parse_args();execute(a.run,a.receipt_sha256)
