"""G3-C2: pinned frozen encoders and G3-B manifests; CPU-only, no fitting.

verify reads exactly three TRAIN images per case/view. extract is explicit and
never selects samples again. Raw endpoint embeddings are saved, not normalized.
"""
import argparse
import io
import json
from datetime import datetime, timezone

import numpy as np
import torch

from models.malsbslcnet import MalSBSLCNet
from models.compact_view_net import CompactViewNet
from scripts.episode_training_dataset import (
    ROOT, VIEWS, STORAGE, EPISODES, G2, SOURCE_MAPPING_HASH,
    HISTORICAL_MANIFEST_HASH, read_bound, parse_json, csv_rows, sha256,
    json_bytes, validate_mapping, require_hash, decode_image,
)
from scripts.mmd_measurement_preparation import checked_rows, CASES, require
from scripts.train_prefix_view import MODEL_HASHES

PLAN = 'data/derived/g3b_measurement_validation/20261015_073946_804374'
PLAN_HASH = '1b581899db556e2c2076eae1f281323922a80c54e99bf247e16a3566ea1985b1'
PREFIX = 'data/derived/prefix_training_v1'
RUNS = {
    ('E1', 'sbsmi'): ('20261016_012854_728594', '2aec068ba4c55dbd25e56b9a415fc103bce1288ec3be8b6ccfaf6bff60baef4a'),
    ('E1', 'raw-byte'): ('20261016_074709_581324', 'a6627f17cb28f1bb0340a8df7dff4f7f4ab44659a7f5c5f298ee5cf30216a426'),
    ('E1', 'entropy'): ('20261016_080934_875031', '895b4639dd4fe4092e12afb4ed1f151df95eb95089ecb7c582df4ead48ee2eab'),
    ('E2', 'sbsmi'): ('20261016_083046_068241', 'aeacfb7b77592fe67982274193a74d15ec21bda08f5e4d9c62770e224aaebac4'),
    ('E2', 'raw-byte'): ('20261016_091116_862715', 'a7c4e7b07f771a40e23ef6cb7256011a60a522b4a6b83595488b4d4e3bea6463'),
    ('E2', 'entropy'): ('20261016_095719_094103', '3547a848504c1971e62a4e767923c63a7bbe8fa41246bb7d5279f088c084a36d'),
    ('Final', 'sbsmi'): ('data/derived/sbsmi_training/20261005_105325_352508', '0c7d76ec3e077e5f88dd6d455b14a041ee2d701a2dc170d5718928695bcf2c4d'),
    ('Final', 'raw-byte'): ('data/derived/float_view_training/raw-byte/20261009_145102_408184', '893eb01789858abab523a3909d40a403328f27d17901dc76a9c94bfbd551341f'),
    ('Final', 'entropy'): ('data/derived/float_view_training/entropy/20261011_001259_203865', '1fc07d49f8dbc0f83cb8e88e1e987cbb070de707713f1f8d941f9e1626771cfa'),
}


class Evidence:
    def __init__(self):
        self.hashes = {}

    def read(self, path, expected=None):
        data = read_bound(ROOT, path, expected)
        digest = sha256(data)
        require(path not in self.hashes or self.hashes[path] == digest, 'Input changed')
        self.hashes[path] = digest
        return data

    def js(self, path, expected=None):
        return parse_json(self.read(path, expected))


def plan(case, stage, evidence):
    require(case in CASES and stage in ('train-reference', 'A', 'B'), 'Forbidden case/stage')
    hashes = evidence.js(PLAN + '/checksums.json', PLAN_HASH)
    def bound(name):
        key = case + '/' + name
        return evidence.read(PLAN + '/' + key, hashes[key])
    mapping = parse_json(bound('class_mapping.json'))
    validate_mapping(mapping, {'E1': 32, 'E2': 42, 'Final': 51}[case])
    source = evidence.js('docs/class_mapping.json', SOURCE_MAPPING_HASH)['class_to_idx']
    if case != 'Final':
        require(mapping == evidence.js(f'{G2}/{case}/class_mapping.json', EPISODES[case][3][1]), 'Episode mapping mismatch')
    else:
        require(mapping == source, 'Final mapping mismatch')
    name = 'train_reference.csv' if stage == 'train-reference' else 'matched_samples.csv'
    rows = csv_rows(bound(name))
    checked_rows(rows, case, tuple(sorted(mapping, key=mapping.get)),
                 {'train'} if stage == 'train-reference' else {'A', 'B'})
    if stage != 'train-reference':
        support = parse_json(bound('support.json'))
        require(all(r['family'] in support for r in rows), 'Frozen support mismatch')
        rows = [r for r in rows if r['stage'] == stage]
    canonical = {r['sha']: r for r in csv_rows(evidence.read(
        'data/manifests/train_validation_manifest.csv', HISTORICAL_MANIFEST_HASH))}
    for row in rows:
        require(all(row[k] == canonical[row['sha']][k] for k in ('family', 'timestamp')), 'Source identity mismatch')
    require(bool(rows), 'Empty plan')
    return rows, mapping, source


def preprocessing(view, evidence):
    folder, digest = STORAGE[view]
    config = evidence.js(f'data/derived/{folder}/config.json', digest)
    require(config['manifest_sha256'] == HISTORICAL_MANIFEST_HASH and
            config['class_mapping_sha256'] == SOURCE_MAPPING_HASH, 'Preprocessing binding mismatch')
    return folder, sha256(json.dumps(config, sort_keys=True).encode())


def image(row, view, source_mapping, evidence):
    folder, config_id = preprocessing(view, evidence)
    split = 'train' if row['timestamp'][:7] < '2020-02' else 'validation'
    # Paths derive only from hash-pinned historical manifests, never directory scans.
    require(row['timestamp'][:7] < '2020-04', 'Future prohibited')
    require_hash(row['sha'])
    base = f'data/derived/{folder}/{split}/{row["sha"]}'
    report = evidence.js(base + '.json')
    expected = dict(sample_identifier=row['sha'], family=row['family'], timestamp=row['timestamp'],
                    split=split, class_index=source_mapping[row['family']],
                    config_sha256=config_id, status='PASS')
    if view == 'sbsmi':
        expected['pixel_round_trip'] = 'PASS'
    else:
        expected.update(representation='raw-byte' if view == 'raw-byte' else 'local-entropy',
                        image_shape=[64, 64], storage_dtype='<f4', array_round_trip='PASS')
    require(all(report.get(k) == v for k, v in expected.items()), 'Image report provenance mismatch')
    require_hash(report['disarmed_content_sha256'])
    tensor = decode_image(evidence.read(base + ('.png' if view == 'sbsmi' else '.npy')), report, view)
    return tensor, report['disarmed_content_sha256']


def verify_config_equivalence(embedded, saved):
    """Compare the trainer's JSON contract, including tuple-to-array encoding.

    Do not coerce numbers/strings or ignore fields. Hash validation for prefix
    checkpoints remains separate and unchanged below.
    """
    require(json_bytes(embedded) == json_bytes(saved), 'Checkpoint/config mismatch')


def load_encoder(case, view, mapping, evidence):
    for path, digest in MODEL_HASHES.items():
        evidence.read(path, digest)
    run, digest = RUNS[case, view]
    directory = run if case == 'Final' else f'{PREFIX}/{case}/{view}/{run}'
    blob = evidence.read(directory + '/epoch_100.pt', digest)
    checkpoint = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
    config = checkpoint['config']
    require(checkpoint['epoch'] == 100, 'Wrong checkpoint epoch')
    verify_config_equivalence(config, evidence.js(directory + '/config.json'))
    _, config_id = preprocessing(view, evidence)
    if case == 'Final':
        require(config['num_classes'] == 51 and config['class_to_idx'] == mapping and
                config['dataset_config_sha256'] == config_id, 'Final checkpoint binding mismatch')
        require(config['model'] == ('MalSBSLCNet' if view == 'sbsmi' else 'CompactViewNet'), 'Architecture mismatch')
        require(config['future_test_used'] is False, 'Invalid Final provenance')
    else:
        complete = evidence.js(directory + '/COMPLETE.json')
        selected = evidence.js(directory + '/epoch_100.json')
        require(complete['selected'] == selected and selected['sha256'] == digest, 'Completion receipt mismatch')
        require(complete['config_sha256'] == checkpoint['config_sha256'] == sha256(json_bytes(config)), 'Config hash mismatch')
        binding = config['dataset']
        require(config['case'] == case and config['view'] == view and
                binding['class_to_idx'] == mapping and binding['num_classes'] == len(mapping) and
                binding['cutoff_exclusive'] == CASES[case][0] and binding['config_id'] == config_id,
                'Prefix checkpoint binding mismatch')
    model = MalSBSLCNet(len(mapping)) if view == 'sbsmi' else CompactViewNet(len(mapping))
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    require(all(torch.isfinite(t).all().item() for t in model.state_dict().values()), 'Non-finite state')
    model.eval().requires_grad_(False)
    return model, dict(checkpoint=directory + '/epoch_100.pt', sha256=digest,
                       K=len(mapping), endpoint='head_norm' if view == 'sbsmi' else 'forward_features')


def state_hash(model):
    import hashlib
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def features(model, view, batch):
    require(view in VIEWS, 'Unknown view')
    require(all(not m.training for m in model.modules()), 'Encoder must be eval')
    require(all(not p.requires_grad for p in model.parameters()), 'Encoder must be frozen')
    require(batch.dtype == torch.float32 and tuple(batch.shape[1:]) == (1, 64, 64)
            and torch.isfinite(batch).all().item(), 'Invalid input')
    with torch.inference_mode():
        if view == 'sbsmi':
            captured = []
            hook = model.head_norm.register_forward_hook(lambda m, args, output: captured.append(output.clone()))
            try:
                logits = model(batch)
            finally:
                hook.remove()
            require(len(captured) == 1 and logits.shape == (len(batch), model.num_classes), 'Endpoint mismatch')
            result = captured[0]
        else:
            result = model.forward_features(batch)
        require(result.shape == (len(batch), 2048 if view == 'sbsmi' else 512)
                and torch.isfinite(result).all().item() and not result.requires_grad, 'Invalid embedding')
        return result.cpu().numpy().copy()


def aligned(expected_ids, observed):
    require(len(set(expected_ids)) == len(expected_ids), 'Duplicate IDs')
    require(set(observed) == set(VIEWS), 'Missing view')
    require(all(value == expected_ids for value in observed.values()), 'Cross-view ID/order mismatch')


def process(case, stage, evidence, verify=False, output=None):
    rows, mapping, source = plan(case, stage, evidence)
    if verify:
        require(stage == 'train-reference', 'Verification TRAIN only')
        rows = rows[:3]
    ids = [r['sha'] for r in rows]
    results, observed, contents = {}, {}, {}
    for view in VIEWS:
        model, metadata = load_encoder(case, view, mapping, evidence)
        before = state_hash(model)
        arrays, seen, hashes = [], [], []
        for start in range(0, len(rows), 3 if verify else 32):
            part = rows[start:start + (3 if verify else 32)]
            inputs = [image(r, view, source, evidence) for r in part]
            batch = torch.stack([x[0] for x in inputs])
            array = features(model, view, batch)
            if verify:
                require(np.array_equal(array, features(model, view, batch)), 'Repeated output changed')
                single = np.concatenate([features(model, view, x[None]) for x in batch])
                # CPU batch kernels may change float32 accumulation order. Fixed
                # engineering tolerance, not fitted on accuracy or drift.
                require(np.allclose(array, single, rtol=1e-5, atol=1e-5), 'Single/batch mismatch')
                metadata['single_batch_max_abs_error'] = float(np.max(np.abs(array-single)))
            arrays.append(array)
            seen.extend(r['sha'] for r in part)
            hashes.extend(x[1] for x in inputs)
        require(state_hash(model) == before, 'Weights/BatchNorm state changed')
        require(all(p.grad is None for p in model.parameters()), 'Unexpected gradients')
        matrix = np.concatenate(arrays)
        metadata.update(status='PASS', shape=list(matrix.shape), state_sha256=before,
                        state_unchanged=True, output_sha256=sha256(matrix.tobytes()))
        observed[view], contents[view] = seen, hashes
        if output is not None and not verify:
            with (output / (view + '.npy')).open('xb') as file:
                np.save(file, matrix, allow_pickle=False)
        results[view] = metadata
        print(f'{case}/{view}: PASS {tuple(matrix.shape)}', flush=True)
    aligned(ids, observed)
    require(all(h == contents[VIEWS[0]] for h in contents.values()), 'Cross-view source-content mismatch')
    return dict(case=case, stage=stage, samples=rows, class_to_idx=mapping,
                source_content_sha256=contents[VIEWS[0]], views=results, alignment='PASS',
                normalization='NOT FITTED; raw endpoints; TRAIN-only G3-B contract retained',
                bandwidth='NOT FITTED', observed_mmd=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('verify')
    extract = sub.add_parser('extract')
    extract.add_argument('--case', required=True, choices=tuple(CASES))
    extract.add_argument('--stage', required=True, choices=('train-reference', 'A', 'B'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    evidence = Evidence()
    for path in ('scripts/frozen_encoder_extraction.py', 'scripts/episode_training_dataset.py',
                 'scripts/mmd_measurement_preparation.py', 'scripts/train_prefix_view.py'):
        evidence.read(path)
    run = ROOT / 'data/derived/frozen_encoder_v1' / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        with (run/name).open('xb') as f:
            f.write(json_bytes(value))
    save('STARTED.json', vars(args))
    try:
        results = ([process(c, 'train-reference', evidence, verify=True) for c in CASES]
                   if args.command == 'verify' else [process(args.case, args.stage, evidence, output=run)])
        for path, digest in evidence.hashes.items():
            read_bound(ROOT, path, digest)
        save('receipt.json', dict(status='PASS', command=vars(args), results=results,
             input_source_hashes=evidence.hashes, torch=str(torch.__version__), numpy=np.__version__,
             device='cpu', batch_size=3 if args.command == 'verify' else 32,
             rtol=1e-5, atol=1e-5, training=False, future_access=False))
        save('checksums.json', {p.name: sha256(p.read_bytes()) for p in run.iterdir() if p.is_file()})
        print('Receipt:', run/'receipt.json')
    except BaseException as exc:
        save('FAILED.json', dict(error=type(exc).__name__, message=str(exc)))
        raise


if __name__ == '__main__':
    main()
