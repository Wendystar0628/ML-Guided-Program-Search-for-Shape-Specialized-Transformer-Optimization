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

# One 45-second search budget per resident shape: roughly ten minutes total.
& $python $cli optimize `
    --group resident `
    --device $Device `
    --budget-seconds 45 `
    --max-trials 24 `
    --no-progress-patience 1 `
    --max-iterations 1 `
    --seed $Seed
exit $LASTEXITCODE
