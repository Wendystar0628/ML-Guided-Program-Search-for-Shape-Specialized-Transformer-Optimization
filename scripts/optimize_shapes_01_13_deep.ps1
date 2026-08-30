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

# Up to three 180-second sweeps per resident shape: roughly two hours total.
& $python $cli optimize `
    --group resident `
    --device $Device `
    --budget-seconds 180 `
    --max-trials 128 `
    --no-progress-patience 2 `
    --max-iterations 3 `
    --seed $Seed
exit $LASTEXITCODE
