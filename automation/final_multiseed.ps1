param(
    [ValidateSet('preflight','train','verify','post')][string]$Stage='preflight',
    [ValidateSet(43,44,45,46)][int]$Seed=43,
    [ValidateSet('sbsmi','raw-byte','entropy','all')][string]$View='all',
    [switch]$OwnerApprovedTraining,
    [switch]$OwnerApprovedPostTraining
)
$ErrorActionPreference='Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
function Invoke-JSON([string[]]$CommandArgs) {
    $raw = & $python -B @CommandArgs
    if ($LASTEXITCODE -ne 0) { throw "Python failed: $($CommandArgs -join ' ')" }
    return ($raw | ConvertFrom-Json)
}
if ($Stage -eq 'train' -and -not $OwnerApprovedTraining) { throw 'Separate owner GPU training approval required.' }
if ($Stage -eq 'post' -and -not $OwnerApprovedPostTraining) { throw 'Separate owner real-data post-training approval required.' }
if ($Stage -eq 'preflight') {
    foreach ($v in @('sbsmi','raw-byte','entropy')) {
        $result = Invoke-JSON @('-m','scripts.train_final_seed','preflight','--view',$v)
        Write-Host "$v $($result.receipt) SHA256=$($result.sha256)"
    }
    return
}
if ($Stage -eq 'train') {
    if ($View -eq 'all') { throw 'Choose exactly one view/run; no automatic twelve-run batch.' }
    # Check AC, temperatures, available RAM/VRAM and >=2 GiB disk before each run.
    # Metadata/existence preflight is refreshed once for this selected view.
    $ready = Invoke-JSON @('-m','scripts.train_final_seed','preflight','--view',$View)
    & $python -B -m scripts.train_final_seed train --seed $Seed --view $View --preflight $ready.receipt --preflight-sha256 $ready.sha256 --execute-training
    if ($LASTEXITCODE -ne 0) { throw 'Training stopped. No automatic retry/resume.' }
    Invoke-JSON @('-m','scripts.train_final_seed','verify','--seed',"$Seed",'--view',$View) | ConvertTo-Json
    return
}
if ($Stage -eq 'verify') {
    foreach ($v in @('sbsmi','raw-byte','entropy')) {
        Invoke-JSON @('-m','scripts.train_final_seed','verify','--seed',"$Seed",'--view',$v) | ConvertTo-Json
    }
    return
}
# No real processing unless all three selected-seed checkpoints verify first.
foreach ($v in @('sbsmi','raw-byte','entropy')) {
    $null = Invoke-JSON @('-m','scripts.train_final_seed','verify','--seed',"$Seed",'--view',$v)
}
$results = foreach ($v in @('sbsmi','raw-byte','entropy')) {
    Invoke-JSON @('-m','scripts.process_final_seed','prepare-view','--seed',"$Seed",'--view',$v,'--execute-posttraining')
}
$arguments = @('-m','scripts.process_final_seed','finalize','--seed',"$Seed",'--execute-posttraining')
foreach ($r in $results) { $arguments += @('--receipt',$r.receipt,'--receipt-sha256',$r.sha256) }
Invoke-JSON $arguments | ConvertTo-Json
