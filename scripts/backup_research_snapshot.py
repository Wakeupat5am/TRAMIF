"""Curated local historical/frozen-package snapshot and verified restore.

Excludes malware bytes, generated image datasets and all Future inputs. This is
not an off-device backup or full image-regeneration archive.
"""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def allowed(relative):
    p=PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts or ':' in relative or '\\' in relative:return False
    if relative.startswith(('data/raw/','data/derived/future')):return False
    if p.suffix.lower() not in ('.py','.md','.json','.csv','.npz','.pt','.txt','.ps1'):return False
    if any('/'+name+'/' in '/'+relative for name in ('sbsmi_dataset_v1','raw_byte_dataset_v1','entropy_dataset_v1')):
        return p.name=='config.json'
    return True


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(package_dir,output):
    files=set()
    def add(relative):
        if allowed(relative) and (ROOT/relative).is_file():files.add(relative)
    package=json.loads((package_dir/'package.json').read_text())
    reports=[package_dir/'package.json']+[ROOT/p/'report.json' for p in package['historical_evidence'].values()]
    for report in reports:
        record=json.loads(report.read_text());add(report.relative_to(ROOT).as_posix())
        for name in record.get('source_input_hashes',{}):add(name)
    directories=[package_dir]+[ROOT/p for p in package['historical_evidence'].values()]
    directories.extend([ROOT/'data/derived/historical_quality_v1/20261017_005343_157189',
                        ROOT/'data/derived/new_quality_only_v1/20261017_020915_979476',
                        ROOT/'data/derived/qd_selection_protocol_v1/20261017_033146_321989',
                        ROOT/'data/derived/historical_research_summary/20261017_034235_417680'])
    for directory in directories:
        for file in directory.rglob('*'):
            if file.is_file():add(file.relative_to(ROOT).as_posix())
    for directory in ('scripts','models','tests'):
        for file in (ROOT/directory).rglob('*.py'):add(file.relative_to(ROOT).as_posix())
    for name in ('README.md','AGENTS.md','PROJECT_PROGRESS.md','DECISION_LOG.md','TECHNICAL_WALKTHROUGH.md',
                 'requirements.txt','automation/continuation_state.json'):
        add(name)
    for name in ('historical_research_results.md','PRE_FUTURE_READINESS_REPORT.md','PAPER_DRAFT.md','REPRODUCIBILITY.md'):
        add('docs/'+name)
    manifest={name:digest(ROOT/name) for name in sorted(files)}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=output/'research_snapshot.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted(files):z.write(ROOT/name,name)
    restored=output/'restore_check';restored.mkdir()
    if not restored.resolve().is_relative_to(ROOT):raise ValueError('Restore escapes workspace')
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Archive CRC failed')
        for name in z.namelist():
            if not allowed(name):raise ValueError('Forbidden archive member')
            dest=restored/Path(name)
            if not dest.resolve().is_relative_to(restored.resolve()):raise ValueError('Archive path escape')
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
    for name,h in manifest.items():
        if digest(restored/name)!=h or digest(ROOT/name)!=h:raise ValueError('Restore/source integrity mismatch')
    result=dict(status='PASS_LOCAL_CURATED_RESTORE',files=len(files),archive_sha256=digest(archive),
        archive_bytes=archive.stat().st_size,source_bytes=sum((ROOT/name).stat().st_size for name in files),
        off_device_backup=False,images_included=False,malware_bytes_included=False,future_access=False,
        limitation='Replay saved historical metrics and frozen package; full input images/binaries are intentionally excluded.')
    (output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));print('Snapshot:',output)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--package-dir',required=True);args=parser.parse_args()
    package=(ROOT/args.package_dir).resolve()
    if not package.is_relative_to(ROOT/'data/derived/pre_future_package'):raise ValueError('Wrong package directory')
    output=ROOT/'data/derived/research_snapshot'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False);execute(package,output)


if __name__=='__main__':main()
