"""D060 seven-arm point evaluator and diagnostics, with no input-file readers.

Real Future input ingestion/execution needs separate authorization. All D060 runs
use synthetic rows/logits. No training, calibration, weighting fit or resampling.
"""
from datetime import datetime, timezone
import re
import numpy as np
from scripts.final_point_statistics import SEEDS, MONTHS, METHODS, aggregate_monthly
from scripts.uniform_fusion import fuse_uniform, classification_metrics
from scripts.quality_fusion import weighted_fusion
from scripts.multiseed_statistics import fixed_support_f1

VIEWS = ('sbsmi', 'raw-byte', 'entropy')


def require(ok, message):
    if not ok: raise ValueError(message)


def simplex(weights):
    w = np.asarray(weights, dtype=np.float64)
    require(w.shape == (3,) and np.isfinite(w).all() and (w >= 0).all()
            and abs(w.sum()-1) <= 1e-12, 'Invalid frozen simplex')
    return w


def freeze_size_bins(train_rows):
    require(bool(train_rows), 'Empty TRAIN sizes')
    ids = [r['sha'] for r in train_rows]
    require(len(set(ids)) == len(ids), 'Duplicate TRAIN IDs')
    for r in train_rows:
        dt = datetime.fromisoformat(r['timestamp'])
        require(dt.utcoffset() is not None and dt.utcoffset().total_seconds() == 0
                and '2019-08' <= dt.strftime('%Y-%m') <= '2020-01', 'TRAIN-only size reference')
        require(type(r['file_size']) is int and r['file_size'] > 0, 'Invalid TRAIN byte size')
    q = np.quantile([r['file_size'] for r in train_rows], [.25, .5, .75], method='linear')
    return dict(cutoffs=q.tolist(), method='linear', reference='frozen TRAIN only',
                samples=len(ids), boundaries='right closed; repeated cutoffs allowed')


def size_bin_indices(sizes, cutoffs):
    sizes = np.asarray(sizes); q = np.asarray(cutoffs, dtype=np.float64)
    require(sizes.ndim == 1 and sizes.dtype.kind in 'iu' and (sizes > 0).all(), 'Invalid sizes')
    require(q.shape == (3,) and np.isfinite(q).all() and (q > 0).all()
            and np.all(q[:-1] <= q[1:]), 'Invalid frozen cutoffs')
    return np.searchsorted(q, sizes, side='left')


def chronology(row):
    dt = datetime.fromisoformat(row['timestamp'])
    require(dt.utcoffset() is not None and dt.utcoffset().total_seconds() == 0, 'UTC timestamp required')
    require(re.fullmatch('[0-9a-f]{64}', row['sha']) is not None, 'Invalid sample ID')
    require(re.fullmatch('[0-9a-f]{64}', row['disarmed_content_sha256']) is not None, 'Invalid content SHA')
    require(row['tlsh_status'] in ('VALID', 'TNULL'), 'Unknown TLSH status')
    require((row['tlsh_status'] == 'TNULL' and row['tlsh'] == 'TNULL') or
            (row['tlsh_status'] == 'VALID' and re.fullmatch('T1[0-9A-F]{70}', row['tlsh']) is not None),
            'TLSH format mismatch')
    return dt, row['sha']


def retention_decisions(reference_rows, target_rows, distance_function):
    """Chronological extension of existing policy; excluded earlier rows remain references.

    Only direct earlier matches decide exclusion, not retrospective components.
    Caller must bind complete reference/target manifests; family is never consulted.
    Default production distance will be standard TLSH including length (not diffxlen).
    This pure routine is tested synthetically, not executed on Future in D060.
    """
    require(bool(reference_rows) and bool(target_rows), 'Reference and target required')
    reference = sorted(reference_rows, key=chronology); targets = sorted(target_rows, key=chronology)
    ids = [r['sha'] for r in reference + targets]
    require(len(set(ids)) == len(ids), 'Duplicate/overlapping retention IDs')
    require(chronology(reference[-1]) < chronology(targets[0]), 'Reference cutoff/order violation')
    previous = list(reference); output = {t: [] for t in (0, 10, 20)}
    for row in targets:
        witnesses = {t: None for t in output}
        for older in previous:
            exact = row['disarmed_content_sha256'] == older['disarmed_content_sha256']
            d = None
            if not exact and row['tlsh_status'] == older['tlsh_status'] == 'VALID':
                d = distance_function(row['tlsh'], older['tlsh'])
                require(type(d) is int and d >= 0, 'Invalid TLSH distance')
            for t in output:
                if witnesses[t] is None and (exact or (d is not None and d <= t)):
                    witnesses[t] = dict(reference_sha=older['sha'], reference_timestamp=older['timestamp'],
                                        reason='exact_sha256' if exact else 'tlsh', tlsh_distance=d)
        for t in output:
            witness = witnesses[t]
            output[t].append(dict(sha=row['sha'], threshold=t, decision='EXCLUDE' if witness else 'KEEP',
                                 **(witness or dict(reason='no_earlier_match', reference_sha='',
                                                    reference_timestamp='', tlsh_distance=None))))
        previous.append(row)
    sets = [{r['sha'] for r in output[t] if r['decision'] == 'KEEP'} for t in (0, 10, 20)]
    require(sets[2] <= sets[1] <= sets[0], 'Retention must be nested')
    return output


def standard_tlsh_distance(first, second):
    import tlsh
    return tlsh.diff(first, second)


def validate_package(package):
    require(package['schema'] == 'TRAMIF_COMBINED_D060_V1' and package['K'] == 51,
            'Wrong combined package schema/K')
    require(package['seeds'] == list(SEEDS) and package['methods'] == list(METHODS)
            and package['views'] == list(VIEWS), 'Seed/view/method binding mismatch')
    mapping = package['class_to_idx']
    require(len(mapping) == 51 and set(mapping.values()) == set(range(51)), 'K51 mapping mismatch')
    require(package['beta'] == 1 and package['gamma'] == 0 and not package['U_used'], 'Frozen methodology mismatch')
    hashes = []
    for seed in SEEDS:
        record = package['seed_records'][str(seed)]
        require(record['seed'] == seed and record['views_order'] == list(VIEWS), 'Seed identity mismatch')
        for v in VIEWS:
            cp = record['views'][v]
            require(cp['seed'] == seed and cp['view'] == v and cp['K'] == 51 and cp['epoch'] == 100
                    and cp['cutoff_exclusive'] == '2020-02', 'Checkpoint identity/cutoff mismatch')
            require(np.isfinite(cp['temperature']) and .05 <= cp['temperature'] <= 20,
                    'Invalid seed-specific temperature')
            require(re.fullmatch('[0-9a-f]{64}', cp['checkpoint_sha256']) is not None, 'Checkpoint hash required')
            hashes.append(cp['checkpoint_sha256'])
        for arm in ('Q', 'QD', 'learned_static'): simplex(record['weights'][arm])
        loss = np.asarray(record['L']); drift = np.asarray(record['d'])
        require(loss.shape == drift.shape == (3,) and np.isfinite(loss).all() and np.isfinite(drift).all()
                and (loss >= 0).all() and (drift >= 0).all() and (drift <= 1).all(), 'Signal binding mismatch')
        for key, score in [('Q', -loss), ('QD', -loss-drift)]:
            w = np.exp(score-score.max()); w /= w.sum()
            require(np.allclose(w, record['weights'][key], atol=1e-12, rtol=0), 'Frozen weight mismatch')
    require(len(set(hashes)) == 15, 'Cross-seed/view checkpoint collision')
    return mapping


def evaluate_arrays(package, rows, bundles_by_seed):
    """Evaluate already verified in-memory arrays. No file I/O or fitting."""
    mapping = validate_package(package)
    require(bool(rows) and len({r['sha'] for r in rows}) == len(rows), 'Missing/duplicate canonical IDs')
    require(set(bundles_by_seed) == set(SEEDS), 'All five seeds required')
    ids = np.array([r['sha'] for r in rows]); classes = np.array(sorted(mapping, key=mapping.get))
    require(all(r['family'] in mapping for r in rows), 'Unknown family in known-family evaluation')
    labels = np.array([mapping[r['family']] for r in rows], dtype=np.int64)
    months = []
    for r in rows:
        dt = datetime.fromisoformat(r['timestamp'])
        require(dt.utcoffset() is not None and dt.utcoffset().total_seconds() == 0, 'UTC required')
        month = dt.strftime('%Y-%m'); require(month in MONTHS, 'Outside six-month evaluation window')
        months.append(month)
    months = np.array(months); require(set(months) == set(MONTHS), 'Empty/missing original month; STOP')
    supports = {m: sorted(set(labels[months == m].tolist())) for m in MONTHS}
    sizes = np.array([r['file_size'] for r in rows]); bins = size_bin_indices(sizes, package['size_stratification']['cutoffs'])
    monthly = np.empty((5, 7, 6)); detail = {}; saved_logs = []
    for si, seed in enumerate(SEEDS):
        bundles = bundles_by_seed[seed]; require(len(bundles) == 3, 'Three views required')
        record = package['seed_records'][str(seed)]
        for v, b in zip(VIEWS, bundles):
            cp = record['views'][v]
            require(b['binding'] == dict(seed=seed, view=v, checkpoint_sha256=cp['checkpoint_sha256']),
                    'Logits checkpoint/seed/view mismatch')
            require(b['temperature'] == cp['temperature'], 'Cross-seed temperature mismatch')
            require(np.array_equal(b['ids'], ids) and np.array_equal(b['classes'], classes)
                    and np.array_equal(b['labels'], labels), 'Canonical sample/class/label alignment mismatch')
        singles, uniform = fuse_uniform(bundles)
        logp = np.stack(singles + [uniform] + [weighted_fusion(np.stack(singles), record['weights'][key])
                                               for key in ('Q', 'QD', 'learned_static')])
        require(logp.shape == (7, len(rows), 51) and np.isfinite(logp).all(), 'Seven-arm log probabilities invalid')
        saved_logs.append(logp); methods = {}
        for j, method in enumerate(METHODS):
            by_month = {}; by_size = {}
            for mi, month in enumerate(MONTHS):
                mask = months == month
                metrics = classification_metrics(logp[j, mask], labels[mask])
                monthly[si, j, mi] = metrics['macro_f1_supported']; by_month[month] = metrics
                by_size[month] = []
                for bucket in range(4):
                    take = mask & (bins == bucket); count = int(take.sum())
                    item = dict(bucket=bucket, samples=count, support=sorted(set(labels[take].tolist())),
                                status='ESTIMABLE' if count else 'NOT_ESTIMABLE')
                    if count:
                        item['metrics'] = classification_metrics(logp[j, take], labels[take])
                        item['macro_f1_original_month_support'] = fixed_support_f1(
                            labels[take], logp[j, take].argmax(1), supports[month])
                    by_size[month].append(item)
            horizons = {}
            for horizon in (1, 3, 6):
                take = np.isin(months, MONTHS[:horizon])
                horizons[str(horizon)] = classification_metrics(logp[j, take], labels[take])
            methods[method] = dict(monthly=by_month, file_size=by_size, pooled_horizons=horizons)
        detail[str(seed)] = methods
    summary = aggregate_monthly(monthly, seed_ids=SEEDS, method_names=METHODS, month_names=MONTHS)
    return dict(summary=summary, details=detail, monthly_support=supports,
                canonical_ids=ids.tolist(), class_names=classes.tolist(),
                uncertainty='descriptive seed variation only; no CI/p-values'), np.stack(saved_logs)


def evaluate_retained(package, rows, bundles_by_seed, decisions):
    """Subset common predictions by ID; never refit or alter class mapping/weights."""
    require(set(decisions) == {0, 10, 20}, 'Frozen thresholds 0/10/20 required')
    expected = {r['sha'] for r in rows}; results = {}
    require(len(expected) == len(rows), 'Duplicate scoring IDs')
    retained = []
    for threshold in (0, 10, 20):
        table = decisions[threshold]
        require(len(table) == len(expected) and {r['sha'] for r in table} == expected,
                'Sensitivity decision ID mismatch')
        require(all(r['threshold'] == threshold and r['decision'] in ('KEEP', 'EXCLUDE') for r in table),
                'Sensitivity threshold/decision mismatch')
        keep = {r['sha'] for r in table if r['decision'] == 'KEEP'}; retained.append(keep)
        indices = [i for i, r in enumerate(rows) if r['sha'] in keep]
        subset = {s: [{**b, **{k: np.asarray(b[k])[indices] for k in ('ids', 'labels', 'logits')}}
                      for b in bundles_by_seed[s]] for s in SEEDS}
        if not indices:
            results[str(threshold)] = dict(status='NOT_ESTIMABLE', samples=0)
            continue
        present = {datetime.fromisoformat(rows[i]['timestamp']).strftime('%Y-%m') for i in indices}
        if present != set(MONTHS):
            results[str(threshold)] = dict(status='NOT_ESTIMABLE', samples=len(indices),
                                           missing_months=sorted(set(MONTHS)-present))
            continue
        report, _ = evaluate_arrays(package, [rows[i] for i in indices], subset)
        results[str(threshold)] = dict(status='PASS', samples=len(indices), report=report)
    require(retained[2] <= retained[1] <= retained[0], 'Non-nested sensitivity sets')
    require(results['0']['status'] == 'PASS', 'Primary evaluation not estimable; STOP')
    return results
