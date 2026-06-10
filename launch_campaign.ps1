$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo

$env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:TF_CPP_MIN_LOG_LEVEL = "3"

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$logDir = Join-Path $repo "runs\campaign_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outLog = Join-Path $logDir "campaign_${stamp}_stdout.log"
$errLog = Join-Path $logDir "campaign_${stamp}_stderr.log"

$args = @(
    "campaign_runner.py",
    "--campaign-id", "readability_2026_06_08",
    "--sleep-hours", "0",
    "--max-runs", "15"
)

$proc = Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $repo -PassThru -RedirectStandardOutput $outLog -RedirectStandardError $errLog

Write-Output "campaign started"
Write-Output "pid: $($proc.Id)"
Write-Output "stdout: $outLog"
Write-Output "stderr: $errLog"
