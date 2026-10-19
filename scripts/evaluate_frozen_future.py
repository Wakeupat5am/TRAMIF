"""Dormant seed42 six-month POINT evaluator. Never opens data without explicit flag.

Input preparation/retention receipts and owner Future approval are prerequisites.
This is not the V1 five-seed confirmatory experiment or a Future input builder.
"""
import argparse
import csv
import hashlib
import io
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scripts.uniform_fusion import fuse_uniform,classification_metrics
from scripts.quality_fusion import weighted_fusion

MONTHS=tuple(f'2020-{m:02d}' for m in range(4,10))
VIEWS=('sbsmi','raw-byte','entropy')


def require(ok,message):
    if not ok:raise ValueError(message)


def monthly_point_metrics(bundles,timestamps,q_weights,qd_weights):
    single,uniform=fuse_uniform(bundles)
    logs=np.stack(single+[uniform,weighted_fusion(np.stack(single),q_weights),weighted_fusion(np.stack(single),qd_weights)])
    names=list(VIEWS)+['uniform','new-quality-only','quality-plus-drift']
    require(len(timestamps)==len(bundles[0]['ids']),'Timestamp alignment mismatch')
    months=[]
    for value in timestamps:
        dt=datetime.fromisoformat(str(value))
        require(dt.utcoffset() is not None and dt.utcoffset().total_seconds()==0,'UTC required')
        month=dt.strftime('%Y-%m');require(month in MONTHS,'Outside approved six-month window');months.append(month)
    require(set(months)==set(MONTHS),'All six months required; no denominator substitution')
    months=np.array(months);labels=bundles[0]['labels'];monthly={}
    for month in MONTHS:
        mask=months==month
        monthly[month]={name:classification_metrics(lp[mask],labels[mask]) for name,lp in zip(names,logs)}
    summaries={name:dict(mean_monthly_macro_f1=float(np.mean([monthly[m][name]['macro_f1_supported'] for m in MONTHS])),
                         worst_month_macro_f1=float(min(monthly[m][name]['macro_f1_supported'] for m in MONTHS))) for name in names}
    return names,logs,monthly,summaries


def read_hashed(path,expected):
    blob=Path(path).read_bytes();require(hashlib.sha256(blob).hexdigest()==expected,'Checksum mismatch: '+str(path));return blob


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',required=True);p.add_argument('--package-sha256',required=True)
    p.add_argument('--inputs',required=True,help='Directory with input_receipt.json, manifest.csv and logits.npz')
    p.add_argument('--input-receipt-sha256',required=True)
    p.add_argument('--owner-approved-future',action='store_true')
    args=p.parse_args()
    if not args.owner_approved_future:p.error('Future remains closed: explicit owner approval required')
    package=json.loads(read_hashed(args.package,args.package_sha256))
    require(package['schema']=='TRAMIF_SEED42_PRE_FUTURE_V1','Unknown frozen package')
    root=Path(args.inputs);receipt=json.loads(read_hashed(root/'input_receipt.json',args.input_receipt_sha256))
    require(receipt['status']=='FUTURE_INPUTS_READY' and receipt['owner_approval_reference'] and receipt['retention_audit_sha256'],'Unverified input preparation')
    require(receipt['views']==list(VIEWS),'Future receipt view order mismatch')
    require(receipt['class_to_idx']==package['class_to_idx'] and receipt['checkpoint_sha256']==package['checkpoint_sha256']
            and receipt['near_duplicate_protocol_sha256']==package['near_duplicate_protocol_sha256'],'Input provenance mismatch')
    # Future receipt generation and completeness audit are still BLOCKED, not inferred by this consumer.
    rows=list(csv.DictReader(io.StringIO(read_hashed(root/'manifest.csv',receipt['manifest_sha256']).decode('utf-8-sig'))))
    require(rows and len({r['sha'] for r in rows})==len(rows),'Empty/duplicate future IDs')
    mapping=package['class_to_idx'];require(len(mapping)==51,'Final51 required')
    require(all(r['retained']=='True' and r['threshold']=='0' and r['family'] in mapping for r in rows),'Wrong retained cohort')
    with np.load(io.BytesIO(read_hashed(root/'logits.npz',receipt['logits_sha256'])),allow_pickle=False) as arrays:
        require(arrays['view_names'].tolist()==list(VIEWS),'Future logits view order mismatch')
        ids=arrays['sample_identifiers'];classes=arrays['class_names'];labels=arrays['labels'];scores=arrays['logits']
        require(ids.tolist()==[r['sha'] for r in rows] and classes.tolist()==sorted(mapping,key=mapping.get),'Future ID/class mismatch')
        require(labels.tolist()==[mapping[r['family']] for r in rows] and scores.shape==(3,len(rows),51),'Future labels/shape mismatch')
        bundles=[dict(ids=ids,classes=classes,labels=labels,logits=scores[i],temperature=package['temperatures'][v]) for i,v in enumerate(VIEWS)]
    names,logs,monthly,summary=monthly_point_metrics(bundles,[r['timestamp'] for r in rows],package['Q_weights'],package['QD_weights'])
    output=Path(__file__).resolve().parents[1]/'data/derived/future_point_evaluation'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'predictions.npz',method_names=np.array(names),log_probabilities=logs,labels=labels,sample_identifiers=ids,class_names=classes)
    result=dict(status='SEED42_POINT_DIAGNOSTIC_ONLY',monthly=monthly,summary=summary,
        package_sha256=args.package_sha256,input_receipt_sha256=args.input_receipt_sha256,
        multi_seed_complete=False,statistical_intervals=False,weights_retuned=False)
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    checks={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in output.iterdir() if f.is_file()}
    (output/'checksums.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    print('Report:',output/'report.json')


if __name__=='__main__':main()
