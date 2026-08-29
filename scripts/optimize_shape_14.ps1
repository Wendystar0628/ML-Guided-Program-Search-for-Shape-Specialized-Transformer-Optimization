[CmdletBinding()]
param(
    [string]$Device = 'cuda:0',
    [double]$BudgetSeconds = 900.0,
    [int]$NoDeploymentPatience = 6,
    [int]$MaxIterations = 20
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'activate_windows_rtx4080.ps1')

$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$cli = Join-Path $projectRoot 'cli.py'
& $python $cli optimize `
    --group shape14 `
    --device $Device `
    --budget-seconds $BudgetSeconds `
    --no-deployment-patience $NoDeploymentPatience `
    --max-iterations $MaxIterations
exit $LASTEXITCODE
