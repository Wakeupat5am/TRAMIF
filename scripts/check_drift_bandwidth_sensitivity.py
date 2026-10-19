"""Fixed h/2 and 2h diagnostic; no model, sampler or coefficient selection."""
import io
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
import numpy as np
from scripts.run_frozen_descriptive_drift import (
    Evidence, ROOT, PLAN, PLAN_HASH, PREP, PREP_HASH, VIEWS,
    family_matrices, output_checksums, class_conditional_mmd2,
    json_bytes, sha256, csv_rows, require, read_bound,
)

PRIMARY = 'data/derived/observed_descriptive_drift/20261016_122644_350078/report.json'
PRIMARY_HASH = '873d2b893d07fb41d74415299d116627c70317b3375abc6fb07d28b61019b393'
FACTORS = (0.5, 2.0)


def contrasts(primary, alternative):
    require(math.isfinite(primary) and math.isfinite(alternative), 'Nonfinite score')
    return dict(delta=alternative-primary,
                relative_change=(alternative-primary)/primary if primary != 0 else None)


def execute(output):
    evidence = Evidence()
    primary = evidence.js(PRIMARY, PRIMARY_HASH)
    prep = evidence.js(PREP+'/report.json', PREP_HASH)
    require(prep['status'] == 'PASS', 'Preparation not PASS')
    for path in ('scripts/descriptive_mmd.py', 'scripts/mmd_measurement_preparation.py'):
        evidence.read(path, primary['source_input_hashes'][path])
    for path in ('scripts/check_drift_bandwidth_sensitivity.py', 'tests/test_drift_bandwidth_sensitivity.py',
                 'scripts/run_frozen_descriptive_drift.py'):
        evidence.read(path)
    plans = evidence.js(PLAN+'/checksums.json', PLAN_HASH)
    prep_checks = evidence.js(PREP+'/checksums.json')
    config = dict(factors=FACTORS, sample_selection='unchanged primary cap128 manifests',
                  normalization='unchanged TRAIN fit', support='unchanged S17/25/25',
                  primary_replacement=False, selection_by_score=False, block_size=32)
    (output/'config.json').write_bytes(json_bytes(config))
    summaries, per_family = [], []
    for case in ('E1', 'E2', 'Final'):
        def frozen(name):
            return evidence.read(PLAN+'/'+case+'/'+name, plans[case+'/'+name])
        support = json.loads(frozen('support.json'))
        samples = csv_rows(frozen('matched_samples.csv'))
        for view in VIEWS:
            prefix = case+'/'+view+'/'
            def prepared(name):
                blob = evidence.read(PREP+'/'+prefix+name, prep_checks[prefix+name])
                require(sha256(blob) == primary['source_input_hashes'][PREP+'/'+prefix+name], 'Primary input mismatch')
                return blob
            meta = json.loads(prepared('parameters.json'))
            bandwidth = json.loads(prepared('bandwidth.json'))
            original = next(r for r in primary['summaries'] if (r['case'],r['view']) == (case,view))
            require(original['bandwidth_h'] == meta['bandwidth_h'] == bandwidth['bandwidth_h'], 'Bandwidth mismatch')
            matrices = {}
            for stage in ('A','B'):
                rows = [r for r in samples if r['stage'] == stage]
                require(rows == meta['stages'][stage]['samples'], 'Selection/order mismatch')
                array = np.load(io.BytesIO(prepared(stage+'.npy')), allow_pickle=False)
                require(sha256(array.tobytes()) == meta['stages'][stage]['payload_sha256'], 'Payload mismatch')
                matrices[stage] = family_matrices(array, rows, support)
            for factor in FACTORS:
                h = original['bandwidth_h']*factor
                result = class_conditional_mmd2(matrices['A'], matrices['B'], support=support,
                                               bandwidth_h=h, block_size=32)
                for family, item in zip(result.support, result.per_family):
                    base = next(r for r in primary['per_family'] if
                        (r['case'],r['view'],r['family']) == (case,view,family))
                    per_family.append(dict(case=case,view=view,family=family,factor=factor,
                        primary_mmd2=base['mmd2'], **asdict(item), **contrasts(base['mmd2'],item.mmd2)))
                summaries.append(dict(case=case,view=view,factor=factor,bandwidth_h=h,support=len(support),
                    primary_mmd2=original['conditional_mmd2'], mmd2=result.mmd2,
                    **contrasts(original['conditional_mmd2'],result.mmd2)))
                print(f'{case}/{view} h*{factor}: MMD2={result.mmd2:.8f}', flush=True)
    orders = {}
    for case in ('E1','E2','Final'):
        orders[case] = {'primary': [r['view'] for r in sorted(
            [r for r in primary['summaries'] if r['case']==case], key=lambda r:r['conditional_mmd2'])]}
        for factor in FACTORS:
            orders[case][str(factor)] = [r['view'] for r in sorted(
                [r for r in summaries if r['case']==case and r['factor']==factor], key=lambda r:r['mmd2'])]
    for path,digest in evidence.hashes.items():
        read_bound(ROOT,path,digest)
    report = dict(status='PASS_TECHNICAL_DIAGNOSTIC',config=config,summaries=summaries,
        per_family=per_family,ascending_descriptive_orders=orders,source_input_hashes=evidence.hashes,
        cap64_run=False,group_balanced_run=False,weights_created=False,future_access=False,
        limitation='Bandwidth-only sensitivity; no calibrated cross-view ranking, significance or predictive reliability claim.')
    (output/'report.json').write_bytes(json_bytes(report))
    (output/'checksums.json').write_bytes(json_bytes(output_checksums(output)))


def main():
    output=ROOT/'data/derived/drift_bandwidth_sensitivity'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    try:
        execute(output)
        print('Report:',output/'report.json')
    except BaseException as exc:
        (output/'FAILED.json').write_bytes(json_bytes(dict(error=type(exc).__name__,message=str(exc))))
        raise


if __name__=='__main__': main()
