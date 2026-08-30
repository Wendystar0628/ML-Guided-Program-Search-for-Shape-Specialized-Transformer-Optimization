[CmdletBinding()]
param(
    [string]$Device = 'cuda:0',
    [int]$Seed = 1234
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $projectRoot 'environments\activate_windows_rtx4080.ps1')

$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$cli = Join-Path $projectRoot 'cli.py'

# One streamed sweep with at most one new Screen trial: roughly ten minutes.
& $python $cli optimize `
    --group shape14 `
    --device $Device `
    --budget-seconds 600 `
    --max-trials 1 `
    --no-progress-patience 1 `
    --max-iterations 1 `
    --seed $Seed
exit $LASTEXITCODE
