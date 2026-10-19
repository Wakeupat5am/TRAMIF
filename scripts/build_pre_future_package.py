"""Freeze seed42 Final components; never reads Future metadata or predictions."""
import subprocess
from datetime import datetime,timezone
import numpy as np
from scripts.evaluate_historical_episode import (
    Evidence,ROOT,VIEWS,FREEZE,FREEZE_HASH,LOCK,LOCK_HASH,LOCK_CHECK,
    selection,static_checkpoint,fixed_qd_weights,json_bytes,sha256,read_bound,require,
)

E1='data/derived/historical_episode_evaluation/E1/20261017_033742_471099'
E1_REPORT_HASH='06c988e1ea17ae2214dc711732ea2c9ed008f91a53a93f3dec7c2af4c71e1a98'
E1_SELECTION_HASH='a36ec42417923921739f455814d8a41e41a9937fb8789ce014794236802820c2'
E2='data/derived/historical_episode_evaluation/E2/20261017_033806_789235'
E2_REPORT_HASH='e9bb1a06628f95b207505eb7859d57c40c44accaabf703422f80d5ec5c10e9c7'


def execute(output):
    e=Evidence();r1=e.js(E1+'/report.json',E1_REPORT_HASH);selected=e.js(E1+'/beta_selection.json',E1_SELECTION_HASH)
    r2=e.js(E2+'/report.json',E2_REPORT_HASH)
    require(r1['status']==r2['status']=='PASS' and r2['E1_selection_sha256']==E1_SELECTION_HASH,'Historical completion mismatch')
    require(r1['beta']==r2['beta']==selected['beta'] and not r2['E2_candidate_selection_used'],'Beta not frozen from E1')
    e.js(LOCK+'/completion.json',LOCK_HASH);checks=e.js(LOCK+'/checksums.json',LOCK_CHECK)
    forecast=e.js(LOCK+'/Final_forecasts.json',checks['Final_forecasts.json'])
    completion=e.js(FREEZE+'/completion.json',FREEZE_HASH);q=e.js(FREEZE+'/Final_weights.json')
    require(q==next(r for r in completion['results'] if r['case']=='Final'),'Final Q freeze mismatch')
    stages,mapping,source,support=selection('Final',e)
    require(q['class_to_idx']==forecast['class_to_idx']==mapping and q['support']==forecast['support']==support,'Final support/mapping mismatch')
    checkpoints={};temperatures={};temperature_files={}
    for view in VIEWS:
        cp=static_checkpoint('Final',view,mapping,e);provenance=q['provenance'][view]
        require(cp==provenance['checkpoint'],'Final checkpoint mismatch')
        t=e.js(provenance['temperature_file'],completion['source_input_hashes'][provenance['temperature_file']])
        require(t['fit_month']=='2020-02' and t['fit_samples']==3799 and t['class_to_idx']==mapping and
                t['checkpoint_sha256']==cp['sha256'] and not t['future_test_used'],'Final calibration mismatch')
        checkpoints[view]=cp['sha256'];temperatures[view]=t['temperature'];temperature_files[view]=provenance['temperature_file']
    policy=e.read('docs/near_duplicate_protocol.md')
    for path in ('scripts/build_pre_future_package.py','scripts/evaluate_frozen_future.py',
                 'tests/test_frozen_future_point.py','docs/RESEARCH_PROTOCOL_V1.md',
                 'docs/RESEARCH_PROTOCOL_V1_AMENDMENT_01_DESCRIPTIVE_DRIFT.md'):
        e.read(path)
    qd=fixed_qd_weights(q['L'],forecast['d_desc'],selected['beta'])
    blockers=[
        'Future processing/evaluation requires separate owner approval; no future input manifest/retention receipt opened or built.',
        'Learned static fusion objective proposed, but numerical solver/tie-break not approved; baseline BLOCKED.',
        'Only seed42 exists; V1 five-seed primary experiment needs four additional seeds or owner-approved scope departure.',
        'Final inferential intervals/multiplicity procedure not validated/approved; only point evaluator prepared.',
        'Artifact backup/restore verification not established; Git does not back up data.',
        'Capacity/cost controls and historical file-size-stratified RQ2 diagnostics incomplete; no efficiency/SOTA or adjusted causal claim.',
        'Future input generation/provenance completeness must be implemented after authorization; saved-logit consumer alone is not ingestion.'
    ]
    git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty=subprocess.check_output(['git','diff','--binary'],cwd=ROOT)
    package=dict(schema='TRAMIF_SEED42_PRE_FUTURE_V1',status='AWAITING_OWNER_FUTURE_APPROVAL',
        scientific_protocol_ready=False,seed=42,views=list(VIEWS),class_to_idx=mapping,checkpoint_sha256=checkpoints,
        checkpoint_paths={v:q['provenance'][v]['checkpoint']['path'] for v in VIEWS},
        temperatures=temperatures,temperature_files=temperature_files,Q_weights=q['weights'],QD_weights=qd.tolist(),
        L=q['L'],d_desc=forecast['d_desc'],beta=selected['beta'],gamma=0,support=support,
        beta_source=E1+'/beta_selection.json',beta_source_sha256=E1_SELECTION_HASH,
        historical_evidence=dict(E1=E1,E2=E2),near_duplicate_protocol_sha256=sha256(policy),
        primary_near_duplicate_distance=0,sensitivity_distances=[10,20],
        feature_statistics='Frozen TRAIN-only; primary descriptive MMD, not null-standardized',
        scope='Six-method seed42 point diagnostic package only; not full V1 primary result',
        mean_worst_rule='mean and min of six monthly supported-class Macro-F1 per fixed seed',
        git_head=git_head,tracked_dirty_diff_sha256=sha256(dirty),source_input_hashes=e.hashes,blockers=blockers,
        future_access=False,U_used=False)
    (output/'package.json').write_bytes(json_bytes(package));package_sha=sha256((output/'package.json').read_bytes())
    # Dormant consumer requires independently approved Future input preparation.
    commands='''param([switch]$OwnerApprovedFuture)
if (-not $OwnerApprovedFuture) { throw 'AWAITING_OWNER_FUTURE_APPROVAL: do not run before explicit approval.' }
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\\..\\..\\..'))
$futureInputs = Join-Path $repoRoot 'data\\derived\\future_inputs_v1\\approved'
$receipt = Join-Path $futureInputs 'input_receipt.json'
if (-not (Test-Path -LiteralPath $receipt)) { throw 'BLOCKED: approved Future input receipt has not been prepared.' }
$receiptSHA = (Get-FileHash -LiteralPath $receipt -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Location -LiteralPath $repoRoot
& (Join-Path $repoRoot '.venv\\Scripts\\python.exe') -B -m scripts.evaluate_frozen_future --package (Join-Path $PSScriptRoot 'package.json') --package-sha256 PACKAGE_SHA --inputs $futureInputs --input-receipt-sha256 $receiptSHA --owner-approved-future
if ($LASTEXITCODE -ne 0) { throw 'Future point evaluator failed; do not retry or retune.' }
'''.replace('PACKAGE_SHA',package_sha)
    (output/'future_commands.ps1').write_text(commands,encoding='utf-8')
    for path,h in e.hashes.items():read_bound(ROOT,path,h)
    (output/'completion.json').write_bytes(json_bytes(dict(status='AWAITING_OWNER_FUTURE_APPROVAL',package_sha256=package_sha,
        engineering='SEED42_COMPONENT_FREEZE_PASS',framework_primary='BLOCKED',blockers=blockers,future_access=False)))
    (output/'checksums.json').write_bytes(json_bytes({p.name:sha256(p.read_bytes()) for p in output.iterdir() if p.is_file()}))


def main():
    output=ROOT/'data/derived/pre_future_package'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    try:execute(output)
    except BaseException as exc:
        (output/'FAILED.json').write_bytes(json_bytes(dict(status='FAIL',error=type(exc).__name__,message=str(exc))))
        raise
    print('Package:',output/'package.json')


if __name__=='__main__':main()
