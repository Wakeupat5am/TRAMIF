"""Fit verified TRAIN-only G3-B parameters and transform A/B; no MMD."""
import io
import os
from datetime import datetime, timezone
from time import perf_counter

# Bound BLAS work before importing numpy; not a model/methodology parameter.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np

from scripts.verify_frozen_embedding_artifacts import BASE, RUN_IDS, validate_matrix
from scripts.frozen_encoder_extraction import Evidence, PLAN, PLAN_HASH, VIEWS
from scripts.episode_training_dataset import ROOT, json_bytes, sha256, csv_rows, read_bound
from scripts.mmd_measurement_preparation import (
    require, fit_train_normalizer, fit_train_bandwidth, TrainNormalizer,
)

VERIFIED = 'data/derived/frozen_embedding_verification/20261016_121558_412739/report.json'
VERIFIED_HASH = '16ffe48545637c2720fbd7586794cf86bad65aa5d00613b8cef4c73c1441dd76'


def bound_embeddings(evidence, registry, case, stage, view):
    binding = next(r for r in registry['runs'] if r['case'] == case and r['stage'] == stage)
    directory = f'{BASE}/{RUN_IDS[case, stage]}'
    receipt = evidence.js(binding['receipt'], binding['receipt_sha256'])
    checks = evidence.js(directory+'/checksums.json', binding['checksums_sha256'])
    require(receipt['status'] == 'PASS' and receipt['command'] ==
            dict(command='extract', case=case, stage=stage), 'Wrong extraction binding')
    result = receipt['results'][0]
    blob = evidence.read(directory+'/'+view+'.npy', checks[view+'.npy'])
    validate_matrix(blob, result['views'][view], binding['samples'], 2048 if view == 'sbsmi' else 512)
    return result, np.load(io.BytesIO(blob), allow_pickle=False)


def parameter_arrays(normalizer):
    return dict(mean=normalizer.mean, std=normalizer.std, keep=normalizer.keep)


def parameter_hash(normalizer):
    return sha256(b''.join(value.tobytes() for value in parameter_arrays(normalizer).values()))


def restore_normalizer(path, metadata):
    with np.load(path, allow_pickle=False) as saved:
        mean, std, keep = (saved[k].copy() for k in ('mean', 'std', 'keep'))
    require(mean.dtype == np.float64 and std.dtype == np.float64 and keep.dtype == bool,
            'Parameter storage dtype changed')
    for a in (mean, std, keep):
        a.setflags(write=False)
    restored = TrainNormalizer(metadata['case'], tuple(metadata['families']),
        metadata['reference_sha256'], metadata['embeddings_sha256'], mean, std, keep,
        metadata['rank'], metadata['condition_number'])
    require(parameter_hash(restored) == metadata['parameters_payload_sha256'], 'Parameter round-trip mismatch')
    return restored


def execute(output):
    evidence = Evidence()
    registry = evidence.js(VERIFIED, VERIFIED_HASH)
    require(registry['status'] == 'PASS' and registry['arrays_verified'] == 27, 'Verification not PASS')
    # Reuse saved verification; only recheck relevant model/extractor/preparation
    # sources, not all original images or graph audits.
    for path, digest in registry['source_input_hashes'].items():
        if path.startswith(('scripts/', 'models/')):
            evidence.read(path, digest)
    evidence.read('scripts/prepare_frozen_drift_measurements.py')
    evidence.read('tests/test_frozen_drift_measurements.py')
    manifest_hashes = evidence.js(PLAN+'/checksums.json', PLAN_HASH)
    summaries = []
    for case in ('E1', 'E2', 'Final'):
        pairs_path = case+'/train_pairs.csv'
        pairs = csv_rows(evidence.read(PLAN+'/'+pairs_path, manifest_hashes[pairs_path]))
        for view in VIEWS:
            started = perf_counter()
            reference, matrix = bound_embeddings(evidence, registry, case, 'train-reference', view)
            rows, mapping = reference['samples'], reference['class_to_idx']
            families = tuple(sorted(mapping, key=mapping.get))
            embeddings = {r['sha']: x for r, x in zip(rows, matrix)}
            normalizer = fit_train_normalizer(rows, embeddings, case=case, families=families)
            bandwidth = fit_train_bandwidth(rows, embeddings, normalizer=normalizer)
            require(bandwidth['pairs'] == pairs, 'Frozen TRAIN-pair selection mismatch')
            directory = output/case/view
            directory.mkdir(parents=True, exist_ok=False)
            def save_json(name, data):
                with (directory/name).open('xb') as file:
                    file.write(json_bytes(data))
            metadata = dict(case=case, view=view, families=list(families),
                reference_sha256=normalizer.reference_sha256,
                embeddings_sha256=normalizer.embeddings_sha256,
                rank=normalizer.rank, condition_number=normalizer.condition_number,
                parameters_payload_sha256=parameter_hash(normalizer),
                input_dimensions=len(normalizer.mean), kept_dimensions=int(normalizer.keep.sum()),
                masked_dimensions=int((~normalizer.keep).sum()), train_samples=len(rows),
                mask_std_le=1e-8, ddof=0, weighting='equal-family TRAIN mixture',
                bandwidth_h=bandwidth['bandwidth_h'], pairs=len(pairs),
                zero_distance_pairs=bandwidth['zero_distance_pairs'],
                normalization_fit_stage='train', bandwidth_fit_stage='train', observed_mmd=False)
            with (directory/'normalizer.npz').open('xb') as file:
                np.savez(file, **parameter_arrays(normalizer))
            restored = restore_normalizer(directory/'normalizer.npz', metadata)
            require(np.array_equal(normalizer.transform(matrix), restored.transform(matrix)), 'Transform round-trip mismatch')
            save_json('bandwidth.json', bandwidth)
            stages = {}
            before = parameter_hash(restored)
            for stage in ('train-reference', 'A', 'B'):
                result, x = ((reference, matrix) if stage == 'train-reference'
                             else bound_embeddings(evidence, registry, case, stage, view))
                require(result['class_to_idx'] == mapping, 'A/B mapping mismatch')
                require(result['views'][view]['sha256'] == reference['views'][view]['sha256'] and
                        result['views'][view]['state_sha256'] == reference['views'][view]['state_sha256'],
                        'Different encoder across stages')
                z = restored.transform(x)
                require(z.dtype == np.float64 and np.isfinite(z).all(), 'Normalized array invalid')
                name = stage+'.npy'
                with (directory/name).open('xb') as file:
                    np.save(file, z, allow_pickle=False)
                stages[stage] = dict(shape=list(z.shape), samples=result['samples'],
                    payload_sha256=sha256(z.tobytes()), max_abs=float(np.abs(z).max()),
                    input_receipt=f'{BASE}/{RUN_IDS[case, stage]}/receipt.json')
            require(before == parameter_hash(restored), 'A/B changed TRAIN parameters')
            metadata.update(status='PASS', stages=stages, seconds=perf_counter()-started,
                warning=('Rank deficient; condition_number unavailable; no dimensions removed by rank'
                         if normalizer.condition_number is None else None))
            save_json('parameters.json', metadata)
            save_json('checksums.json', {p.name: sha256(p.read_bytes()) for p in directory.iterdir() if p.is_file()})
            summaries.append({k: v for k, v in metadata.items() if k != 'stages'})
            print(f'{case}/{view}: PASS | kept={metadata["kept_dimensions"]}/{metadata["input_dimensions"]}'
                  f' | rank={normalizer.rank} | h={bandwidth["bandwidth_h"]:.8g}', flush=True)
    for path, digest in evidence.hashes.items():
        read_bound(ROOT, path, digest)
    return dict(status='PASS', summaries=summaries, source_input_hashes=evidence.hashes,
        observed_mmd=False, encoder_training=False, inference=False, future_access=False,
        limitations='Technical measurement preparation only; no null standardization or cross-encoder statistical scale equivalence.')


def main():
    output = ROOT/'data/derived/frozen_drift_preparation'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        with (output/name).open('xb') as file:
            file.write(json_bytes(value))
    try:
        report = execute(output)
        save('report.json', report)
        save('checksums.json', {p.relative_to(output).as_posix(): sha256(p.read_bytes())
                               for p in output.rglob('*') if p.is_file()})
        print('Report:', output/'report.json')
    except BaseException as exc:
        save('FAILED.json', dict(error=type(exc).__name__, message=str(exc)))
        raise


if __name__ == '__main__': main()
