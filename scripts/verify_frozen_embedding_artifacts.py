"""Verify nine explicit historical extraction runs; no forward or fitting."""
import argparse
import io
from datetime import datetime, timezone

import numpy as np

from scripts.frozen_encoder_extraction import (
    ROOT, VIEWS, RUNS, PREFIX, PLAN, PLAN_HASH, Evidence, plan, MODEL_HASHES,
)
from scripts.episode_training_dataset import read_bound, parse_json, sha256, json_bytes, require_hash
from scripts.mmd_measurement_preparation import require

RUN_IDS = {
    ('E1', 'train-reference'): '20261016_111226_165007',
    ('E1', 'A'): '20261016_111417_726553',
    ('E1', 'B'): '20261016_111728_096745',
    ('E2', 'train-reference'): '20261016_111823_484048',
    ('E2', 'A'): '20261016_112824_373308',
    ('E2', 'B'): '20261016_113058_554779',
    ('Final', 'train-reference'): '20261016_113123_612120',
    ('Final', 'A'): '20261016_113258_877739',
    ('Final', 'B'): '20261016_113324_822237',
}
COUNTS = {'E1': (3266, 1137), 'E2': (4197, 1734), 'Final': (5354, 1768)}
BASE = 'data/derived/frozen_encoder_v1'


def validate_matrix(blob, metadata, rows, width):
    matrix = np.load(io.BytesIO(blob), allow_pickle=False)
    require(isinstance(matrix, np.ndarray), 'Not an NPY array')
    require(matrix.dtype == np.dtype('<f4') and matrix.shape == (rows, width), 'Shape/dtype mismatch')
    require(np.isfinite(matrix).all(), 'Non-finite embeddings')
    require(metadata['shape'] == list(matrix.shape), 'Receipt shape mismatch')
    require(sha256(matrix.tobytes()) == metadata['output_sha256'], 'Embedding payload hash mismatch')


def validate_identity(result, rows, mapping):
    require(result['samples'] == rows, 'Sample IDs/order/metadata mismatch')
    require(len({r['sha'] for r in rows}) == len(rows), 'Duplicate sample IDs')
    require(result['class_to_idx'] == mapping and result['alignment'] == 'PASS', 'Mapping/alignment mismatch')
    require(set(result['views']) == set(VIEWS), 'Wrong view set')
    content = result['source_content_sha256']
    require(len(content) == len(rows), 'Content-hash count mismatch')
    for value in content:
        require_hash(value)


def verify_all():
    evidence = Evidence()
    results, bindings, contents = [], {}, {}
    for (case, stage), run_id in RUN_IDS.items():
        directory = f'{BASE}/{run_id}'
        require(not (ROOT/directory/'FAILED.json').exists(), 'FAILED extraction run')
        checks = evidence.js(directory+'/checksums.json')
        require(set(checks) == {'STARTED.json', 'receipt.json', *(v+'.npy' for v in VIEWS)}, 'Incomplete output checksum set')
        blobs = {name: evidence.read(directory+'/'+name, digest) for name, digest in checks.items()}
        receipt = parse_json(blobs['receipt.json'])
        command = dict(command='extract', case=case, stage=stage)
        require(parse_json(blobs['STARTED.json']) == command == receipt['command'], 'Run command mismatch')
        require(receipt['status'] == 'PASS' and receipt['training'] is False and
                receipt['future_access'] is False and receipt['device'] == 'cpu', 'Invalid execution scope')
        require(len(receipt['results']) == 1, 'Unexpected result count')
        result = receipt['results'][0]
        require(result['case'] == case and result['stage'] == stage, 'Case/stage mismatch')
        rows, mapping, source = plan(case, stage, evidence)
        require(len(rows) == COUNTS[case][0 if stage == 'train-reference' else 1], 'Unexpected frozen count')
        validate_identity(result, rows, mapping)
        require(result['bandwidth'] == 'NOT FITTED' and result['observed_mmd'] is False
                and result['normalization'].startswith('NOT FITTED;'), 'Unexpected preprocessing fit')
        prior = receipt['input_source_hashes']
        # Recheck source/config/selection/checkpoint evidence, not full old image
        # preflight. Original image/report hashes remain bound by the receipt.
        reused_images = 0
        for path, digest in prior.items():
            require(not path.startswith(('/', '\\')) and '..' not in path.split('/'), 'Invalid provenance path')
            if any(path.startswith(f'data/derived/{folder}/'+split+'/')
                   for folder in ('sbsmi_dataset_v1', 'raw_byte_dataset_v1', 'entropy_dataset_v1')
                   for split in ('train', 'validation')):
                reused_images += 1
                continue
            evidence.read(path, digest)
        require(prior.get(PLAN+'/checksums.json') == PLAN_HASH, 'Unbound G3-B plan')
        for path, digest in MODEL_HASHES.items():
            require(prior.get(path) == digest, 'Model source binding mismatch')
        for view in VIEWS:
            meta = result['views'][view]
            run, digest = RUNS[case, view]
            checkpoint = (run if case == 'Final' else f'{PREFIX}/{case}/{view}/{run}')+'/epoch_100.pt'
            require(meta['checkpoint'] == checkpoint and meta['sha256'] == digest
                    and prior.get(checkpoint) == digest, 'Checkpoint provenance mismatch')
            require(meta['K'] == len(mapping) and meta['status'] == 'PASS'
                    and meta['state_unchanged'] is True, 'Encoder receipt mismatch')
            require(meta['endpoint'] == ('head_norm' if view == 'sbsmi' else 'forward_features'), 'Endpoint mismatch')
            binding = (checkpoint, digest, meta['state_sha256'], meta['K'], meta['endpoint'])
            key = case, view
            require(key not in bindings or bindings[key] == binding, 'Different frozen encoder across TRAIN/A/B')
            bindings[key] = binding
            validate_matrix(blobs[view+'.npy'], meta, len(rows), 2048 if view == 'sbsmi' else 512)
            folder = {'sbsmi': 'sbsmi', 'raw-byte': 'raw_byte', 'entropy': 'entropy'}[view]
            for row in rows:
                split = 'train' if row['timestamp'][:7] < '2020-02' else 'validation'
                base = f'data/derived/{folder}_dataset_v1/{split}/{row["sha"]}'
                for suffix in ('.json', '.png' if view == 'sbsmi' else '.npy'):
                    require(base+suffix in prior, 'Missing per-sample image/report provenance')
        require(reused_images == len(rows)*6, 'Unexpected image/report provenance count')
        for row, digest in zip(rows, result['source_content_sha256']):
            require(row['sha'] not in contents or contents[row['sha']] == digest, 'Cross-run content hash disagreement')
            contents[row['sha']] = digest
        results.append(dict(case=case, stage=stage, samples=len(rows), K=len(mapping), status='PASS',
            receipt=directory+'/receipt.json', receipt_sha256=checks['receipt.json'],
            checksums_sha256=evidence.hashes[directory+'/checksums.json'],
            reused_image_report_receipts=reused_images,
            ordered_ids_sha256=sha256(json_bytes([r['sha'] for r in rows]))))
        print(f'{case}/{stage}: PASS | {len(rows)} samples | three views', flush=True)
    # Explicit separation for each episode; chronology was also checked by plan().
    for case in COUNTS:
        sets = [set(r['sha'] for r in plan(case, s, evidence)[0]) for s in ('train-reference', 'A', 'B')]
        require(not any(sets[i] & sets[j] for i in range(3) for j in range(i+1, 3)), 'Cross-stage ID overlap')
    for path, digest in evidence.hashes.items():
        read_bound(ROOT, path, digest)
    return dict(status='PASS', runs=results, arrays_verified=27, source_input_hashes=evidence.hashes,
        unique_content_id_bindings=len(contents), inference=False, fitting=False, future_access=False,
        limitations='Checksums and saved execution receipts verify integrity/provenance, not re-inference or scientific drift validity. Original images/reports not rescanned.')


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    output = ROOT/'data/derived/frozen_embedding_verification'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        with (output/name).open('xb') as file:
            file.write(json_bytes(value))
    try:
        report = verify_all()
        report['verifier_sha256'] = sha256((ROOT/'scripts/verify_frozen_embedding_artifacts.py').read_bytes())
        save('report.json', report)
        save('checksums.json', {'report.json': sha256((output/'report.json').read_bytes())})
        print('Report:', output/'report.json')
    except BaseException as exc:
        save('FAILED.json', dict(error=type(exc).__name__, message=str(exc)))
        raise


if __name__ == '__main__':
    main()
