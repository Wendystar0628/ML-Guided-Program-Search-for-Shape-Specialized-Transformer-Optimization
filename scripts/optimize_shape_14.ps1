[CmdletBinding()]
param(
    [string]$Device = 'cuda:0',
    [int]$Seed = 1234,
    [double]$BudgetSeconds = 900,
    [int]$MaxNewTrials = 12
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $projectRoot 'environments\activate_windows_rtx4080.ps1')

$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$cli = Join-Path $projectRoot 'cli.py'

# One invocation is one bounded search-to-deployment pass. Shape 14 has 36
# Screen configurations; persistent studies let later invocations continue
# from prior evidence without placing several expensive Formal comparisons in
# one run. The 15-minute soft search budget plus one sequential Formal test is
# designed to finish in roughly 20-27 minutes on the validated RTX 4080.
& $python $cli search `
    --case-id official_14 `
    --device $Device `
    --budget-seconds $BudgetSeconds `
    --max-trials $MaxNewTrials `
    --seed $Seed `
    --dtype float32 `
    --padding-ratio 0 `
    --input-scale 1
exit $LASTEXITCODE
