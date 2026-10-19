"""D060 deterministic SYNTHETIC replay of a combined package; never real inputs."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.build_combined_freeze import digest, write
from scripts.final_multiseed_evaluator import VIEWS, SEEDS, MONTHS, evaluate_retained, require


def synthetic_inputs(package):
    classes = np.array(sorted(package['class_to_idx'], key=package['class_to_idx'].get))
    q = package['size_stratification']['cutoffs']
    sizes = [1, max(1,int(q[0])+1), max(1,int(q[1])+1), max(1,int(q[2])+1)]
    rows = []
    for month in MONTHS:
        for i in range(8):
            rows.append(dict(sha=hashlib.sha256(f'synthetic/{month}/{i}'.encode()).hexdigest(),
                             family=str(classes[i%3]), timestamp=month+'-15T00:00:00+00:00',
                             file_size=sizes[i%4]))
    labels = np.array([package['class_to_idx'][r['family']] for r in rows], dtype=np.int64)
    ids = np.array([r['sha'] for r in rows]); bundles = {}
    for seed in SEEDS:
        rng = np.random.Generator(np.random.PCG64(seed+6060)); bundles[seed] = []
        for v in VIEWS:
            logits = rng.normal(size=(len(rows),51)); logits[np.arange(len(rows)),labels] += 2
            cp = package['seed_records'][str(seed)]['views'][v]
            bundles[seed].append(dict(ids=ids.copy(), classes=classes.copy(), labels=labels.copy(),
                logits=logits, temperature=cp['temperature'],
                binding=dict(seed=seed,view=v,checkpoint_sha256=cp['checkpoint_sha256'])))
    decisions={t:[dict(sha=r['sha'],threshold=t,decision='EXCLUDE' if
                    ((t==10 and i%8==0) or (t==20 and i%8 in (0,1))) else 'KEEP')
                  for i,r in enumerate(rows)] for t in (0,10,20)}
    return rows,bundles,decisions


def replay(package):
    rows,bundles,decisions=synthetic_inputs(package)
    report=evaluate_retained(package,rows,bundles,decisions)
    return dict(status='PASS_SYNTHETIC_INTEGRATION', data='synthetic logits only; registered models not run',
                seeds=5,arms=7,thresholds=[0,10,20],Future_access=False,results=report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',required=True);p.add_argument('--package-sha256',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    require(digest(a.package)==a.package_sha256,'Combined package checksum mismatch')
    result=replay(json.loads(Path(a.package).read_text(encoding='utf-8')))
    write(a.output,result)
    print(json.dumps(dict(status=result['status'],output_sha256=digest(a.output),Future_access=False)))


if __name__=='__main__': main()
