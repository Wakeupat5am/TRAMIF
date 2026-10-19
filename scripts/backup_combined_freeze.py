"""D060 explicit-asset local backup, safe restore and synthetic numerical replay."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
from scripts.build_combined_freeze import ROOT, digest, write, safe_relative, require


def restore_archive(archive,destination,manifest):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        require(len(names)==len(set(names)) and set(names)==set(manifest),'Archive membership mismatch')
        for name in names:
            safe_relative(name)
            target=(destination/name).resolve()
            require(target.is_relative_to(destination.resolve()),'Archive path escape')
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(name) as src,target.open('xb') as dst: shutil.copyfileobj(src,dst,1024*1024)
            require(digest(target)==manifest[name],'Restored hash mismatch: '+name)


def backup(root,package_dir,output):
    root=Path(root);package_dir=Path(package_dir);output=Path(output)
    receipt=json.loads((package_dir/'receipt.json').read_text(encoding='utf-8'))
    require(receipt['status']=='PASS_COMBINED_FREEZE','Combined freeze incomplete')
    for name,h in receipt['files'].items():
        safe_relative(name);require(digest(package_dir/name)==h,'Package receipt mismatch')
    assets=json.loads((package_dir/'assets_manifest.json').read_text())
    for file in package_dir.iterdir():
        if file.is_file():assets[file.relative_to(root).as_posix()]=digest(file)
    for name,h in assets.items():safe_relative(name);require((root/name).is_file(),'Missing asset')
    size=sum((root/name).stat().st_size for name in assets)
    require(shutil.disk_usage(output).free>size*3+2*2**30,'Insufficient backup/restore headroom')
    write(output/'manifest.json',assets)
    archive=output/'combined_research_backup.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for name,h in sorted(assets.items()):
            require(digest(root/name)==h,'Source changed before archive: '+name)
            z.write(root/name,name)
    restored=output/'restored'
    restore_archive(archive,restored,assets)
    package_rel=(package_dir/'package.json').relative_to(root)
    command=[sys.executable,'-B','-m','scripts.replay_combined_freeze','--package',str(restored/package_rel),
             '--package-sha256',digest(restored/package_rel),'--output',str(output/'restored_synthetic_replay.json')]
    run=subprocess.run(command,cwd=restored,capture_output=True,text=True)
    (output/'reproduction_stdout.txt').write_text(run.stdout,encoding='utf-8')
    (output/'reproduction_stderr.txt').write_text(run.stderr,encoding='utf-8')
    require(run.returncode==0,'Restored synthetic replay failed: '+run.stderr[-2000:])
    # Compare archived/original report numerically, not just the replay process exit.
    from scripts.replay_combined_freeze import replay
    original=replay(json.loads((package_dir/'package.json').read_text()))
    restored_report=json.loads((output/'restored_synthetic_replay.json').read_text())
    require(original==restored_report,'Restored numerical replay mismatch')
    checkpoint_paths={v['checkpoint'] for s in original_package(package_dir)['seed_records'].values() for v in s['views'].values()}
    require(len(checkpoint_paths)==15 and checkpoint_paths<=set(assets),'Backup lacks registered checkpoints')
    result=dict(status='PASS_LOCAL_BACKUP_RESTORE_AND_REPLAY',files=len(assets),
        source_bytes=size,archive_bytes=archive.stat().st_size,archive_sha256=digest(archive),
        checkpoint_files=15,hashes_verified=True,synthetic_replay_equal=True,
        replay_interpreter=sys.executable,original_environment_reused=True,
        off_device_backup=False,full_raw_data_backup=False,full_image_data_backup=False,
        environment_install_archive=False,Future_access=False,
        limitation='Local same-machine archive/restore only; original dependencies reused. Not a fresh-machine install or full training reproduction.',
        outputs={p.name:digest(p) for p in output.iterdir() if p.is_file()})
    write(output/'receipt.json',result)
    return result


def original_package(directory):return json.loads((Path(directory)/'package.json').read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package-dir',required=True);a=p.parse_args()
    package=Path(a.package_dir).resolve(strict=True)
    require(package.is_relative_to(ROOT/'data/derived/combined_freeze_v1'),'Wrong combined freeze directory')
    output=ROOT/'data/derived/combined_freeze_backup'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    try:result=backup(ROOT,package,output)
    except BaseException as exc:
        write(output/'FAILED.json',dict(status='FAIL',error=type(exc).__name__,message=str(exc)));raise
    print(json.dumps(dict(status=result['status'],files=result['files'],archive_bytes=result['archive_bytes'],output=str(output))))


if __name__=='__main__':main()
