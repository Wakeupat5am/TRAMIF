"""Observed descriptive MMD only: frozen A/B, support, TRAIN parameters."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import io
import csv
import math
import json
from dataclasses import asdict
from datetime import datetime, timezone
from time import perf_counter

import numpy as np

from scripts.descriptive_mmd import class_conditional_mmd2, DEFAULT_BLOCK_SIZE
from scripts.frozen_encoder_extraction import Evidence, PLAN, PLAN_HASH, VIEWS
from scripts.episode_training_dataset import ROOT, json_bytes, sha256, csv_rows, read_bound
from scripts.mmd_measurement_preparation import require, checked_rows
from scripts.verify_frozen_embedding_artifacts import BASE, RUN_IDS

PREP = 'data/derived/frozen_drift_preparation/20261016_122145_477857'
PREP_HASH = '78b31028bd48a9dcc4fc18d4191810197b295e007e1c7a4b2eb91c8de757b276'


def output_checksums(directory):
    return {p.name: sha256(p.read_bytes()) for p in directory.iterdir()
            if p.is_file() and p.name != 'checksums.json'}


def family_matrices(matrix, samples, support):
    require(matrix.dtype == np.float64 and matrix.ndim == 2 and len(matrix) == len(samples)
            and np.isfinite(matrix).all(), 'Invalid normalized matrix')
    require(len({r['sha'] for r in samples}) == len(samples), 'Duplicate sample IDs')
    require(set(r['family'] for r in samples) == set(support), 'Support mismatch')
    result = {family: matrix[[i for i, r in enumerate(samples) if r['family'] == family]]
              for family in support}
    require(all(len(x) for x in result.values()), 'Empty family')
    return result


def execute(output):
    e = Evidence()
    preparation = e.js(PREP+'/report.json', PREP_HASH)
    require(preparation['status'] == 'PASS' and preparation['observed_mmd'] is False, 'Invalid preparation')
    saved = e.js(PREP+'/checksums.json')
    for path, digest in preparation['source_input_hashes'].items():
        e.read(path, digest)
    for path in ('scripts/descriptive_mmd.py', 'scripts/run_frozen_descriptive_drift.py',
                 'tests/test_frozen_descriptive_drift.py'):
        e.read(path)
    plans = e.js(PLAN+'/checksums.json', PLAN_HASH)
    summaries, families_output = [], []
    for case in ('E1', 'E2', 'Final'):
        def frozen(name):
            return e.read(PLAN+'/'+case+'/'+name, plans[case+'/'+name])
        support = json.loads(frozen('support.json'))
        require(len(support) == {'E1': 17, 'E2': 25, 'Final': 25}[case], 'Frozen support changed')
        samples = csv_rows(frozen('matched_samples.csv'))
        mapping = json.loads(frozen('class_mapping.json'))
        checked_rows(samples, case, tuple(sorted(mapping, key=mapping.get)), {'A', 'B'})
        counts = csv_rows(frozen('cap_counts.csv'))
        rows_by_stage = {s: [r for r in samples if r['stage'] == s] for s in ('A', 'B')}
        # Reuse D023 post-cap review; do not reinterpret lost>=5 as exclusion.
        for c in counts:
            require(int(c['groups_after']) >= 5 and c['post_cap_below5'] == 'False', 'Post-cap support failure')
        group_info = {(r['family'], r['stage']): r for r in counts}
        sizes = {}
        for stage, rows in rows_by_stage.items():
            extraction = e.js(f'{BASE}/{RUN_IDS[case, stage]}/receipt.json')
            for row in rows:
                split = 'train' if row['timestamp'][:7] < '2020-02' else 'validation'
                report_path = f'data/derived/sbsmi_dataset_v1/{split}/{row["sha"]}.json'
                source = e.js(report_path, extraction['input_source_hashes'][report_path])
                require(source['sample_identifier'] == row['sha'] and source['family'] == row['family'], 'Size report identity mismatch')
                require(type(source['file_size']) is int and source['file_size'] > 0, 'Invalid file size')
                sizes[row['sha']] = source['file_size']
        for view in VIEWS:
            started = perf_counter()
            directory = case+'/'+view+'/'
            def prepared(name):
                return e.read(PREP+'/'+directory+name, saved[directory+name])
            meta = json.loads(prepared('parameters.json'))
            bandwidth = json.loads(prepared('bandwidth.json'))
            prepared('normalizer.npz')
            require(meta['status'] == 'PASS' and meta['normalization_fit_stage'] == meta['bandwidth_fit_stage'] == 'train', 'Non-TRAIN fit')
            h = bandwidth['bandwidth_h']
            require(h == meta['bandwidth_h'] and math.isfinite(h) and h > 0, 'Invalid frozen h')
            matrices = {}
            for stage in ('A', 'B'):
                arr = np.load(io.BytesIO(prepared(stage+'.npy')), allow_pickle=False)
                require(meta['stages'][stage]['samples'] == rows_by_stage[stage], 'Frozen sample order mismatch')
                require(sha256(arr.tobytes()) == meta['stages'][stage]['payload_sha256'], 'Normalized payload mismatch')
                matrices[stage] = family_matrices(arr, rows_by_stage[stage], support)
            result = class_conditional_mmd2(matrices['A'], matrices['B'], support=support,
                bandwidth_h=h, block_size=DEFAULT_BLOCK_SIZE)
            for family, value in zip(result.support, result.per_family):
                row = dict(case=case, view=view, family=family, **asdict(value))
                for stage in ('A', 'B'):
                    selected = [r for r in rows_by_stage[stage] if r['family'] == family]
                    g = group_info[family, stage]
                    require(len(selected) == int(g['samples_after']), 'Saved cap count mismatch')
                    require(len({r['group_id'] for r in selected}) == int(g['groups_after']), 'Saved group count mismatch')
                    row.update({stage+'_samples': len(selected), stage+'_groups': int(g['groups_after']),
                                stage+'_groups_lost': int(g['groups_lost']),
                                stage+'_largest_group_fraction': float(g['largest_after_fraction'])})
                    values = np.array([sizes[r['sha']] for r in selected], dtype=np.float64)
                    for label, value_q in zip(('min', 'q25', 'median', 'q75', 'max'), np.quantile(values, [0, .25, .5, .75, 1])):
                        row[stage+'_file_size_'+label] = float(value_q)
                families_output.append(row)
            summary = dict(case=case, view=view, support=len(support), bandwidth_h=h,
                conditional_mmd2=result.mmd2, d_desc=result.mmd2/2,
                clipped_families=sum(x.negative_roundoff_clipped for x in result.per_family),
                seconds=perf_counter()-started)
            summaries.append(summary)
            print(f'{case}/{view}: MMD2={result.mmd2:.8f} | S={len(support)}', flush=True)
    for path, digest in e.hashes.items():
        read_bound(ROOT, path, digest)
    report = dict(status='PASS_TECHNICAL_COMPUTATION', summaries=summaries,
        per_family=families_output, source_input_hashes=e.hashes,
        kernel='exp(-||z-zprime||^2/(2*p*h)); biased MMD2 with diagonals',
        aggregation='equal-family frozen support', block_size=DEFAULT_BLOCK_SIZE,
        null_standardization=False, sensitivity_run=False, weights_created=False,
        future_access=False, limitations='Descriptive exploratory measurement; not a significance score, calibrated cross-view scale, or demonstrated predictor of future error.')
    with (output/'report.json').open('xb') as file:
        file.write(json_bytes(report))
    with (output/'per_family.csv').open('x', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(families_output[0]))
        writer.writeheader()
        writer.writerows(families_output)
    manifest = output_checksums(output)
    with (output/'checksums.json').open('xb') as file:
        file.write(json_bytes(manifest))


def main():
    output = ROOT/'data/derived/observed_descriptive_drift'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    try:
        execute(output)
        print('Report:', output/'report.json')
    except BaseException as exc:
        with (output/'FAILED.json').open('xb') as file:
            file.write(json_bytes(dict(error=type(exc).__name__, message=str(exc))))
        raise


if __name__ == '__main__': main()
