param([switch]$LocalSnapshot)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$arguments=@('-B','-m','scripts.verify_pre_future_engineering')
if ($LocalSnapshot) { $arguments+='--local-snapshot' }
& '.\.venv\Scripts\python.exe' @arguments
if ($LASTEXITCODE -ne 0) { throw 'Targeted engineering verification failed.' }
