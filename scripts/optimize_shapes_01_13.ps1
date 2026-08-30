[CmdletBinding()]
param(
    [string]$Device = 'cuda:0',
    [double]$BudgetSeconds = 900.0,
    [int]$NoProgressPatience = 3,
    [int]$MaxIterations = 12
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $projectRoot 'environments\activate_windows_rtx4080.ps1')

$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$cli = Join-Path $projectRoot 'cli.py'
& $python $cli optimize `
    --group resident `
    --device $Device `
    --budget-seconds $BudgetSeconds `
    --no-progress-patience $NoProgressPatience `
    --max-iterations $MaxIterations
exit $LASTEXITCODE
