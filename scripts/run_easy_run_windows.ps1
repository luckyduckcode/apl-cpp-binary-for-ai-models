<#
Windows convenience wrapper to setup venv (optionally), install requirements, build the native backend, and run `easy_run.py`.
Usage: run in PowerShell as `.
un_easy_run_windows.ps1 -Model tinyllama`.
#>

param(
    [string]$Model = 'tinyllama',
    [switch]$SkipInstall,
    [switch]$SkipBuild
)

function Log($m) { Write-Host "[run_easy_run_windows] $m" }

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Push-Location $root

if (-not $SkipInstall) {
    if (-not (Test-Path -Path .venv)) {
        Log "Creating virtual environment..."
        python -m venv .venv
    }
    Log "Activating virtual environment"
    . .\.venv\Scripts\Activate.ps1
    Log "Installing requirements..."
    pip install -r requirements.txt
}

if (-not $SkipBuild) {
    Log "Building backend (cross-platform)..."
    python scripts/build_backend.py
}

Log "Running easy_run.py for model $Model"
python easy_run.py --model $Model

Pop-Location
