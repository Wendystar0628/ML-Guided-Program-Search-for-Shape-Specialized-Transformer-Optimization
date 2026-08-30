[CmdletBinding()]
param(
    [string]$Device = 'cuda:0',
    [int]$Seed = 1234,
    [double]$BudgetSecondsPerIteration = 1200,
    [int]$MaxNewTrialsPerIteration = 12,
    [int]$MaxIterations = 4,
    [int]$NoDeploymentPatience = 3
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $projectRoot 'environments\activate_windows_rtx4080.ps1')

$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$cli = Join-Path $projectRoot 'cli.py'

# Shape 14 has 36 Screen configurations. Batches of 12 cover the 10-point
# mandatory structure set, leave two racing points, and provide up to three
# Formal challenger opportunities before the finite space is exhausted.
& $python $cli optimize `
    --group shape14 `
    --device $Device `
    --budget-seconds $BudgetSecondsPerIteration `
    --max-trials $MaxNewTrialsPerIteration `
    --max-iterations $MaxIterations `
    --no-deployment-patience $NoDeploymentPatience `
    --seed $Seed `
    --dtype float32 `
    --padding-ratio 0 `
    --input-scale 1
exit $LASTEXITCODE
