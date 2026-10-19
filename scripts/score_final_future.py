"""D062 owner-authorized CPU scoring of immutable D061 logits; no fitting/inference."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
from scripts.build_combined_freeze import digest, write
from scripts.final_multiseed_evaluator import evaluate_retained, require, VIEWS
from scripts.final_point_statistics import SEEDS, METHODS, MONTHS
from scripts.uniform_fusion import fuse_uniform
from scripts.quality_fusion import weighted_fusion

ROOT = Path(__file__).resolve().parents[1]
INPUT = 'data/derived/protected_future_inputs_v1/20261018_082841_011123'
INPUT_SHA = '116f994f5b3bf4372de248fe34b855e01c87f037955795a8156fa34d433ac771'


def bound_json(folder, receipt, name):
    path = (folder/name).resolve(strict=True)
    require(path.is_relative_to(folder.resolve()), 'Input path escape')
    require(digest(path) == receipt['files'][name], 'D061 file checksum mismatch: '+name)
    return json.loads(path.read_text(encoding='utf-8'))


def adapt_inputs(package, rows, inventory, tables, records, arrays):
    """Join by explicit seed/view and canonical IDs, never inferred array positions."""
    mapping = package['class_to_idx']; ids = [r['sha'] for r in rows]
    all_ids = [r['sha'] for r in inventory]
    require(len(set(ids)) == len(ids) and len(set(all_ids)) == len(all_ids), 'Duplicate IDs')
    retained = {}
    for t in (0, 10, 20):
        table = tables[t]
        require([r['sha'] for r in table] == all_ids, 'Full retention coverage/order mismatch')
        require(all(r['threshold'] == t and r['decision'] in ('KEEP', 'EXCLUDE') for r in table),
                'Invalid retention values')
        retained[t] = {r['sha'] for r in table if r['decision'] == 'KEEP'}
    require(retained[20] <= retained[10] <= retained[0] == set(ids), 'Primary/nested selection mismatch')
    require(rows == [r for r in inventory if r['sha'] in retained[0]], 'Primary row provenance/order mismatch')
    require(len(records) == 15 and len({(r['seed'], r['view']) for r in records}) == 15,
            'Duplicate/missing logit records')
    lookup = {(r['seed'], r['view']): r for r in records}
    require(set(lookup) == {(s, v) for s in SEEDS for v in VIEWS}, 'Seed/view coverage mismatch')
    bundles = {}
    for seed in SEEDS:
        bundles[seed] = []
        for view in VIEWS:
            item = lookup[seed, view]; cp = package['seed_records'][str(seed)]['views'][view]
            require(item['checkpoint_sha256'] == cp['checkpoint_sha256']
                    and item['temperature'] == cp['temperature'] and item['state_unchanged']
                    and not item['gradient_updates'], 'Checkpoint/temperature/state mismatch')
            a = arrays[seed, view]
            require(a['sample_identifiers'].tolist() == ids and a['labels'].tolist() ==
                    [mapping[r['family']] for r in rows] and a['class_names'].tolist() ==
                    sorted(mapping, key=mapping.get), 'Canonical ID/class/label mismatch')
            z = a['logits']
            require(z.shape == (len(rows), 51) and z.dtype == np.float32 and np.isfinite(z).all(),
                    'Invalid full-K51 raw logits')
            bundles[seed].append(dict(ids=a['sample_identifiers'], classes=a['class_names'],
                labels=a['labels'], logits=z, temperature=cp['temperature'],
                binding=dict(seed=seed, view=view, checkpoint_sha256=cp['checkpoint_sha256'])))
    decisions = {t: [r for r in tables[t] if r['sha'] in retained[0]] for t in (0, 10, 20)}
    return bundles, decisions


def prediction_logs(package, bundles):
    result = []
    for seed in SEEDS:
        singles, uniform = fuse_uniform(bundles[seed])
        w = package['seed_records'][str(seed)]['weights']
        result.append(np.stack(singles+[uniform]+[
            weighted_fusion(np.stack(singles), w[k]) for k in ('Q', 'QD', 'learned_static')]))
    return np.stack(result)


def export_tables(output, results, package):
    monthly = []; strata = []; horizons = []; primary = []; contrasts = []; descriptive = []; support = []
    keys = ('samples', 'supported_classes', 'accuracy', 'macro_f1_supported', 'macro_f1_fixed',
            'balanced_accuracy', 'nll', 'brier', 'ece_15')
    for threshold, item in results.items():
        if item['status'] != 'PASS': continue
        report = item['report']; summary = report['summary']
        for method in METHODS:
            for outcome, value in summary['methods'][method]['outcomes'].items():
                primary.append(dict(threshold=threshold, method=method, outcome=outcome, **value))
        contrasts.extend(dict(threshold=threshold, **c) for c in summary['contrasts'])
        for v in VIEWS:
            for outcome in ('mean_monthly_macro_f1', 'worst_month_macro_f1'):
                qd = summary['methods']['quality-plus-drift']['outcomes'][outcome]['per_seed']
                base = summary['methods'][v]['outcomes'][outcome]['per_seed']
                d = np.subtract(qd, base)
                descriptive.append(dict(threshold=threshold, left='quality-plus-drift', right=v,
                    outcome=outcome, per_seed=d.tolist(), mean=float(d.mean()),
                    sample_sd=float(d.std(ddof=1)), min=float(d.min()), max=float(d.max()),
                    interpretation='descriptive only; outside fixed contrast family'))
        for month, indices in report['monthly_support'].items():
            support.append(dict(threshold=threshold, month=month, class_indices=indices,
                families=[sorted(package['class_to_idx'], key=package['class_to_idx'].get)[i] for i in indices]))
        for seed in SEEDS:
            for method in METHODS:
                details = report['details'][str(seed)][method]
                for month, m in details['monthly'].items():
                    monthly.append(dict(threshold=threshold, seed=seed, method=method, month=month,
                                        **{k: m[k] for k in keys}))
                for month, buckets in details['file_size'].items():
                    for b in buckets:
                        strata.append(dict(threshold=threshold, seed=seed, method=method, month=month,
                            bucket=b['bucket'], status=b['status'], support=b['support'],
                            **({k: b['metrics'][k] for k in keys} if 'metrics' in b else {'samples': b['samples']}),
                            macro_f1_original_month_support=b.get('macro_f1_original_month_support')))
                for horizon, m in details['pooled_horizons'].items():
                    horizons.append(dict(threshold=threshold, seed=seed, method=method, months=horizon,
                                         **{k: m[k] for k in keys}))
    for name, records in [('monthly_metrics', monthly), ('file_size_metrics', strata),
                          ('pooled_metrics', horizons), ('primary_outcomes', primary), ('contrasts', contrasts),
                          ('descriptive_single_view_differences', descriptive)]:
        fields = list(dict.fromkeys(k for r in records for k in r))
        with (output/(name+'.csv')).open('x', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()}
                            for r in records)
    write(output/'monthly_support.json', support)
    write(output/'descriptive_single_view_differences.json', descriptive)


def execute(output):
    start = perf_counter()
    # Existing verified contract: byte bindings only, no registered model execution.
    from scripts.prepare_protected_future import contract
    package, _, binding = contract(ROOT)
    folder = ROOT/INPUT
    require(digest(folder/'receipt.json') == INPUT_SHA, 'D061 completion receipt mismatch')
    receipt = json.loads((folder/'receipt.json').read_text())
    require(receipt['status'] == 'READY_FOR_FINAL_SCORING' and not receipt['scoring_performed']
            and receipt['binding'] == binding, 'D061 completion/bundle binding mismatch')
    for p, h in receipt['source_sha256'].items():
        require(digest(ROOT/p) == h and digest(folder/'source'/p) == h, 'D061 source/snapshot mismatch')
    sources = dict(package['source_hashes'])
    for p in ('scripts/score_final_future.py', 'scripts/verify_final_future_scoring.py', 'tests/test_final_future_scoring.py'):
        sources[p] = digest(ROOT/p)
    cfg = dict(decision='D062', authorization='Explicit owner final scoring approval in current conversation',
        input_directory=INPUT, input_receipt_sha256=INPUT_SHA, binding=binding,
        specification_sha256=package['specification_sha256'], source_hashes=sources,
        seeds=list(SEEDS), views=list(VIEWS), methods=list(METHODS), months=list(MONTHS),
        thresholds=[0,10,20], beta=1, gamma=0, U_used=False, refitting=False,
        statistical_intervals=False, p_values=False, environment=dict(python=sys.version, numpy=np.__version__))
    write(output/'config.json', cfg)
    for p, h in sources.items():
        require(digest(ROOT/p) == h, 'Source hash mismatch')
        dest = output/'source'/p; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes((ROOT/p).read_bytes())
    rows = bound_json(folder, receipt, 'primary_manifest.json')
    inventory = bound_json(folder, receipt, 'known_inventory.json')
    tables = {t: bound_json(folder, receipt, f'retention_{t}.json') for t in (0,10,20)}
    verification = bound_json(folder, receipt, 'independent_verification.json')
    require(verification['status'] == 'PASS_INDEPENDENT_INPUT_READBACK'
            and verification['quartiles'] == package['size_stratification'], 'D061 verification/size binding mismatch')
    records = bound_json(folder, receipt, 'logits_manifest.json'); arrays = {}
    for item in records:
        path = folder/item['path']
        require(digest(path) == item['sha256'] == receipt['files'][item['path']], 'Logits checksum mismatch')
        require(bound_json(folder, receipt, str(Path(item['path']).with_suffix('.json')).replace('\\','/')) == item,
                'Logit sidecar mismatch')
        with np.load(path, allow_pickle=False) as a: arrays[item['seed'], item['view']] = dict(a)
    bundles, decisions = adapt_inputs(package, rows, inventory, tables, records, arrays)
    write(output/'scoring_manifest.json', rows)
    write(output/'retained_decisions.json', {str(t): x for t,x in decisions.items()})
    print('D062 bindings/alignment PASS; CPU scoring all five seeds/seven arms/three thresholds', flush=True)
    results = evaluate_retained(package, rows, bundles, decisions)
    logs = prediction_logs(package, bundles)
    with (output/'log_probabilities.npy').open('xb') as f: np.save(f, logs, allow_pickle=False)
    with (output/'predictions.npz').open('xb') as f:
        np.savez_compressed(f, predictions=logs.argmax(3).astype(np.int16),
            sample_identifiers=np.array([r['sha'] for r in rows]),
            labels=np.array([package['class_to_idx'][r['family']] for r in rows]),
            class_names=np.array(sorted(package['class_to_idx'], key=package['class_to_idx'].get)),
            seeds=np.array(SEEDS), methods=np.array(METHODS))
    write(output/'results.json', results); export_tables(output, results, package)
    from scripts.verify_final_future_scoring import verify
    readback = verify(ROOT, output, package)
    write(output/'independent_verification.json', readback)
    for p, h in sources.items(): require(digest(ROOT/p) == h, 'Source changed during scoring')
    require(digest(folder/'receipt.json') == INPUT_SHA, 'D061 receipt changed')
    contract(ROOT)
    write(output/'receipt.json', dict(status='FINAL_FUTURE_SCORING_VERIFIED', decision='D062',
        input_receipt_sha256=INPUT_SHA, binding=binding, source_hashes=sources,
        readback=readback, elapsed_seconds=perf_counter()-start, fitting_performed=False,
        confidence_intervals=False, p_values=False, research_claim='descriptive fixed-horizon evaluation',
        files={p.relative_to(output).as_posix():digest(p) for p in output.rglob('*') if p.is_file()}))
    print('FINAL_FUTURE_SCORING_VERIFIED:', output, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--owner-approved-final-scoring', action='store_true'); a = p.parse_args()
    if not a.owner_approved_final_scoring: p.error('Explicit D062 owner authorization required')
    output = ROOT/'data/derived/final_future_scoring_v1'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    try: execute(output)
    except BaseException as e:
        write(output/'FAILED.json', dict(status='BLOCKED', error=type(e).__name__, message=str(e),
                                        automatic_retry=False)); raise


if __name__ == '__main__': main()
