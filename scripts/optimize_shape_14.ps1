[CmdletBinding()]
param(
    [string]$Device = 'cuda:0',
    [int]$Seed = 1234,
    [int]$StructureSeed = 1234,
    [double]$BudgetSeconds = 900,
    [int]$MaxNewTrials = 12
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $projectRoot 'environments\activate_windows_rtx4080.ps1')

$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$cli = Join-Path $projectRoot 'cli.py'
$effectiveStructureSeed = if ($PSBoundParameters.ContainsKey('StructureSeed')) {
    $StructureSeed
} else {
    $Seed
}

# One invocation is one bounded search-to-deployment pass. Shape 14 persistently
# enumerates 34 high-value Triton/native points without replacement; the very
# slow Reference implementation is retained only as a fallback. One invocation
# runs at most one sequential Formal comparison.
& $python $cli search `
    --case-id official_14 `
    --device $Device `
    --budget-seconds $BudgetSeconds `
    --max-trials $MaxNewTrials `
    --seed $Seed `
    --structure-seed $effectiveStructureSeed `
    --dtype float32 `
    --padding-ratio 0 `
    --input-scale 1
exit $LASTEXITCODE
