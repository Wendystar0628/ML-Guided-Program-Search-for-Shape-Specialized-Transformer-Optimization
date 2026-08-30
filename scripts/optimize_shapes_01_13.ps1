[CmdletBinding()]
param(
    [string]$Device = 'cuda:0',
    [int]$Seed = 1234,
    [double]$BudgetSecondsPerShape = 180,
    [int]$MaxNewTrialsPerShape = 96,
    [int]$MaxIterations = 4,
    [int]$NoDeploymentPatience = 3
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $projectRoot 'environments\activate_windows_rtx4080.ps1')

$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$cli = Join-Path $projectRoot 'cli.py'

# One foreground, serial optimization run for Shapes 01-13. The 96-Trial cap is
# above the initial structure-coverage requirement and leaves room for eta=2
# racing plus branch-local TPE. Wall time remains the primary budget.
& $python $cli optimize `
    --group resident `
    --device $Device `
    --budget-seconds $BudgetSecondsPerShape `
    --max-trials $MaxNewTrialsPerShape `
    --max-iterations $MaxIterations `
    --no-deployment-patience $NoDeploymentPatience `
    --seed $Seed `
    --dtype float32 `
    --padding-ratio 0 `
    --input-scale 1
exit $LASTEXITCODE
