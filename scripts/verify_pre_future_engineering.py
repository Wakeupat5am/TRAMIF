"""Reproduce only the new CPU/synthetic tests and optional local source snapshot.

No CNN training, real inference, empirical solver fit, real resampling or Future I/O.
This is a curated addendum to the previous historical snapshot, not a full backup.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import tempfile
import unittest
import zipfile
from scripts.train_final_seed import ROOT,write_new_json,sha256

PATTERNS=('test_final_seed.py','test_final_seed_processing.py','test_learned_static_solver.py','test_multiseed_statistics.py')
FILES=('scripts/train_final_seed.py','scripts/process_final_seed.py','scripts/learned_static_solver.py',
 'scripts/multiseed_statistics.py','scripts/benchmark_pre_future.py','scripts/verify_pre_future_engineering.py',
 'automation/final_multiseed.ps1','automation/reproduce_pre_future_engineering.ps1','automation/continuation_state.json',
 'docs/FINAL_MULTISEED_EXECUTION.md','docs/PRE_FUTURE_OWNER_DECISIONS.md','docs/PRE_FUTURE_ENGINEERING_RESULTS.md',
 'docs/PRE_FUTURE_READINESS_REPORT.md','docs/REPRODUCIBILITY.md','docs/PAPER_DRAFT.md',
 'PROJECT_PROGRESS.md','DECISION_LOG.md','TECHNICAL_WALKTHROUGH.md')+tuple('tests/'+p for p in PATTERNS)
BENCH='data/derived/pre_future_engineering/20261017_040408_325260'
PREFLIGHTS=(
 'data/derived/final_seed_preflight/sbsmi/20261017_040529_714452/receipt.json',
 'data/derived/final_seed_preflight/raw-byte/20261017_040535_383626/receipt.json',
 'data/derived/final_seed_preflight/entropy/20261017_040541_122490/receipt.json')


def test_ids(suite):
    result=[]
    for child in suite:
        result.extend(test_ids(child) if isinstance(child,unittest.TestSuite) else [child.id()])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--local-snapshot',action='store_true');args=parser.parse_args()
    suite=unittest.TestSuite()
    for pattern in PATTERNS:suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=pattern))
    ids=test_ids(suite);result=unittest.TextTestRunner(verbosity=1).run(suite)
    output=ROOT/'data/derived/pre_future_engineering/verification'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f');output.mkdir(parents=True,exist_ok=False)
    files=FILES+PREFLIGHTS+(BENCH+'/report.json',BENCH+'/receipt.json')
    manifest={name:sha256((ROOT/name).read_bytes()) for name in files}
    receipt=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),test_ids=ids,python=platform.python_version(),source_sha256=manifest,future_access=False,CNN_training=False,real_inference=False)
    if args.local_snapshot and result.wasSuccessful():
        archive=output/'engineering_addendum.zip'
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for name,h in manifest.items():
                blob=(ROOT/name).read_bytes()
                if sha256(blob)!=h:raise ValueError('Snapshot input changed')
                z.writestr(name,blob)
        with zipfile.ZipFile(archive) as z,tempfile.TemporaryDirectory() as tmp:
            if z.testzip() is not None or set(z.namelist())!=set(manifest):raise ValueError('Archive integrity failed')
            for name,h in manifest.items():
                destination=Path(tmp)/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(z.read(name))
                if sha256(destination.read_bytes())!=h:raise ValueError('Restore checksum mismatch')
        receipt['snapshot']=dict(path=archive.relative_to(ROOT).as_posix(),sha256=sha256(archive.read_bytes()),files=len(manifest),restore='PASS',off_device=False,
            excludes='Raw binaries, image datasets and new CNN checkpoints (not trained). Use prior historical snapshot separately.')
    write_new_json(output/'manifest.json',manifest);write_new_json(output/'receipt.json',receipt)
    print(json.dumps(dict(status=receipt['status'],tests=result.testsRun,receipt=(output/'receipt.json').relative_to(ROOT).as_posix(),sha256=sha256((output/'receipt.json').read_bytes()))))
    raise SystemExit(not result.wasSuccessful())


if __name__=='__main__':main()
