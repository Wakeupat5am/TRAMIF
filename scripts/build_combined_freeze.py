"""D060 combined freeze: reuse D054/D058 evidence, no model/fit/Future execution."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import numpy as np
from scripts.final_multiseed_evaluator import VIEWS, validate_package, freeze_size_bins, require
from scripts.final_point_statistics import SEEDS, METHODS

ROOT = Path(__file__).resolve().parents[1]
D054 = 'data/derived/final_multiseed_verification/20261017_164739_728045'
D054_SHA = 'fcd99219417e19f61520d85c868e7a3a074ce0b32ac1727bfd48d6681aad9f86'
D058 = 'data/derived/final_learned_static_v2/20261018_003544_337822'
D058_SHA = '0dc546341da26602d83c237209fd0df5319a12ef6967af6a06ea186634b97bd8'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def safe_relative(name):
    p = PurePosixPath(name)
    require(isinstance(name, str) and not p.is_absolute() and '..' not in p.parts
            and ':' not in name and '\\' not in name, 'Unsafe asset path')
    require(not name.startswith(('data/raw/', 'data/derived/future', 'data/manifests/future')),
            'Forbidden raw/Future asset')
    return p


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')


class BoundInputs:
    def __init__(self, root): self.root = Path(root); self.assets = {}

    def add(self, name, expected):
        safe_relative(name)
        require(len(expected) == 64 and all(c in '0123456789abcdef' for c in expected), 'Invalid asset hash')
        require(name not in self.assets or self.assets[name] == expected, 'Conflicting source bindings')
        self.assets[name] = expected

    def read(self, name, expected=None):
        safe_relative(name)
        path = (self.root/name).resolve(strict=True)
        require(path.is_relative_to(self.root.resolve()), 'Asset symlink escape')
        actual = digest(path)
        if expected is None: expected = self.assets.get(name, actual)
        require(actual == expected, 'Hash mismatch: '+name)
        self.add(name, expected)
        return path.read_bytes()

    def js(self, name, expected=None): return json.loads(self.read(name, expected))

    def receipt(self, directory, expected):
        receipt = self.js(directory+'/receipt.json', expected)
        require(receipt['status'] == 'PASS', 'Prior verification receipt not PASS')
        for name, h in receipt['files'].items(): self.add(directory+'/'+name, h)
        return receipt


def compose_package(summary, learned, mapping, size_config):
    """No file reads; binds existing records and seed-specific calibrated mixtures."""
    require(summary['status'] == 'ARTIFACT_VERIFIED' and summary['seeds'] == list(SEEDS)
            and len(summary['checkpoints']) == 15 and len(summary['rows']) == 15,
            'D054 completeness mismatch')
    records = {}
    for seed in SEEDS:
        rows = [r for r in summary['rows'] if r['seed'] == seed]
        require(len(rows) == 3 and {r['view'] for r in rows} == set(VIEWS), 'Seed/view missing')
        views = {}
        ordered = [next(r for r in rows if r['view'] == v) for v in VIEWS]
        fit = learned[seed]
        require(fit['seed'] == seed and fit['K'] == 51 and fit['view_order'] == list(VIEWS)
                and fit['fit']['status'] == 'CONVERGED_VERIFIED' and fit['weights_registered']
                and fit['fit_samples'] == 4064, 'Learned-static frozen identity mismatch')
        for v, row in zip(VIEWS, ordered):
            cps = [r for r in summary['checkpoints'] if r['seed'] == seed and r['view'] == v]
            require(len(cps) == 1, 'Ambiguous checkpoint')
            cp = cps[0]; pv = fit['provenance'][v]
            require(cp['status'] == row['checkpoint_status'] == 'ARTIFACT_VERIFIED'
                    and row['processing_status'] == 'ARTIFACT_VERIFIED', 'Prior verification missing')
            require(pv['checkpoint_sha256'] == cp['checkpoint_sha256']
                    and pv['temperature'] == row['temperature']
                    and pv['processing_receipt'] == row['processing_receipt'], 'Cross-seed provenance mismatch')
            views[v] = {**cp, 'temperature': row['temperature'], 'processing_receipt': row['processing_receipt'],
                        'normalizer_sha256': row['normalizer_sha256'],
                        'reference_embeddings_sha256': row['reference_embeddings_sha256']}
        records[str(seed)] = dict(seed=seed, views_order=list(VIEWS), views=views,
            L=[r['L'] for r in ordered], d=[r['d'] for r in ordered],
            weights=dict(Q=[r['Q_weight'] for r in ordered], QD=[r['QD_weight'] for r in ordered],
                         learned_static=fit['fit']['weights']),
            learned_static_file=f'{D058}/seed_{seed}/weights.json')
    package = dict(schema='TRAMIF_COMBINED_D060_V1', K=51, seeds=list(SEEDS), views=list(VIEWS),
                   methods=list(METHODS), class_to_idx=mapping, seed_records=records,
                   beta=1, gamma=0, U_used=False, size_stratification=size_config,
                   Future_access=False, Future_execution_authorized=False,
                   retention=dict(primary=0, sensitivity=[10,20], references='all earlier cohort rows including excluded',
                                  label_dependent=False, tlsh_distance='standard including length'),
                   uncertainty='point estimates + sample SD and range; no CIs or p-values')
    validate_package(package)
    return package


def build(root, output):
    e = BoundInputs(root)
    e.receipt(D054, D054_SHA); e.receipt(D058, D058_SHA)
    summary = e.js(D054+'/summary.json')
    for name, h in summary['source_input_hashes'].items(): e.add(name, h)
    for name, h in summary['source_sha256'].items(): e.add(name, h)
    selection = e.js(D058+'/selection.json')
    mapping = selection['class_to_idx']; learned = {}
    for seed in SEEDS:
        complete = e.js(f'{D058}/seed_{seed}/COMPLETE.json')
        require(complete['seed'] == seed and complete['status'] == 'PASS', 'Frozen fit incomplete')
        for name,h in complete['files'].items(): e.add(f'{D058}/seed_{seed}/{name}', h)
        learned[seed] = e.js(f'{D058}/seed_{seed}/weights.json')
        require(learned[seed]['support'] == selection['support'], 'Learned support mismatch')
    configs = {v:e.js(f'data/derived/{folder}/config.json') for v,folder in
               zip(VIEWS, ('sbsmi_dataset_v1','raw_byte_dataset_v1','entropy_dataset_v1'))}
    require(len({c['manifest_sha256'] for c in configs.values()}) == 1
            and len({c['zip_index_sha256'] for c in configs.values()}) == 1, 'Preprocessing cohort mismatch')
    manifest = list(csv.DictReader(io.StringIO(e.read('data/manifests/train_validation_manifest.csv',
                              configs['sbsmi']['manifest_sha256']).decode('utf-8-sig'))))
    index = list(csv.DictReader(io.StringIO(e.read('data/manifests/train_validation_zip_index.csv',
                           configs['sbsmi']['zip_index_sha256']).decode('utf-8-sig'))))
    require(all('2019-08' <= r['timestamp'][:7] < '2020-04' for r in manifest+index),
            'Nonhistorical input in registered historical metadata')
    lookup = {r['sha']:r for r in index}
    require(len(lookup) == len(index) and len({r['sha'] for r in manifest}) == len(manifest), 'Duplicate metadata ID')
    train = []
    for row in manifest:
        if row['split'] != 'train': continue
        ir = lookup[row['sha']]
        require(all(row[k] == ir[k] for k in ('sha','split','timestamp','family'))
                and row['family'] in mapping, 'TRAIN metadata join mismatch')
        train.append({**row, 'file_size':int(ir['file_size']), 'label':mapping[row['family']]})
    require(len(train) == 18061 and len({r['family'] for r in train}) == 51, 'Frozen TRAIN coverage mismatch')
    size_config = freeze_size_bins(train)
    package = compose_package(summary, learned, mapping, size_config)
    package['preprocessing'] = configs
    package['approval'] = dict(decision='D060', statistical_scope='owner O1/O2/O4 and modified O3 approved',
                               amendment='docs/D060_O3_SCOPE_AMENDMENT.md')
    for p in ['docs/FINAL_STATISTICAL_PROTOCOL.md','docs/D060_O3_SCOPE_AMENDMENT.md',
              'docs/RESEARCH_PROTOCOL_V1.md','docs/RESEARCH_PROTOCOL_V1_AMENDMENT_01_DESCRIPTIVE_DRIFT.md',
              'docs/near_duplicate_protocol.md','requirements.txt','AGENTS.md',
              'scripts/final_point_statistics.py','scripts/final_multiseed_evaluator.py',
              'scripts/build_combined_freeze.py','scripts/backup_combined_freeze.py',
              'scripts/replay_combined_freeze.py','scripts/uniform_fusion.py','scripts/quality_fusion.py',
              'scripts/multiseed_statistics.py','scripts/calibrate_sbsmi.py','scripts/sbsmi_dataset.py',
              'tests/test_final_point_statistics.py','tests/test_d060_integration.py',
              'tests/test_combined_freeze.py']:
        e.read(p)
    for p in ('models/__init__.py','models/malsbslcnet.py','models/compact_view_net.py'): e.read(p)
    package['specification_sha256'] = e.assets['docs/FINAL_STATISTICAL_PROTOCOL.md']
    package['amendment_sha256'] = e.assets['docs/D060_O3_SCOPE_AMENDMENT.md']
    package['source_hashes'] = {p:h for p,h in e.assets.items() if p.startswith(('scripts/','models/'))}
    package['git_head'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    # Verify bytes for packaging only; no checkpoint deserialization or numerical recomputation.
    for name,h in e.assets.items():
        require(digest(root/name) == h, 'Packaging source/hash mismatch: '+name)
    write(output/'assets_manifest.json',e.assets)
    write(output/'TRAIN_size_reference.json',train)
    package['asset_manifest_sha256'] = digest(output/'assets_manifest.json')
    package['TRAIN_size_reference_sha256'] = digest(output/'TRAIN_size_reference.json')
    write(output/'package.json',package)
    write(output/'receipt.json',dict(status='PASS_COMBINED_FREEZE', checkpoint_bindings=15,
        seeds=5, arms=7, training=False, inference=False, fitting=False, Future_access=False,
        verification='D054/D058 receipts reused; bytes/provenance joins checked for package, no re-audit',
        files={p.name:digest(p) for p in output.iterdir() if p.is_file()}, assets=len(e.assets)))
    return package


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    output=ROOT/'data/derived/combined_freeze_v1'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    try: build(ROOT,output)
    except BaseException as exc:
        write(output/'FAILED.json',dict(status='FAIL',error=type(exc).__name__,message=str(exc)));raise
    print('Combined freeze:', output)


if __name__ == '__main__': main()
