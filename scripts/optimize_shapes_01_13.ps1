[CmdletBinding()]
param(
    [string]$Device = 'cuda:0',
    [int]$Seed = 1234,
    [int]$StructureSeed = 1234,
    [double]$BudgetSecondsPerShape = 180,
    [int]$MaxNewTrialsPerShape = 96,
    [int]$MaxIterations = 4,
    [int]$NoDeploymentPatience = 3,
    [switch]$IncludeShape06
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

# One foreground, serial optimization run for resident Shapes. Shape 06 is
# excluded from routine structure-seed rotation because its deployed fused
# family is much faster than repeatedly sampled general branches; opt in when
# a new Shape 06 mechanism needs a full sweep. Each generated
# structure gets one witness; most remaining Trials activate branch-local TPE,
# while a small exploration floor revisits least-sampled alternatives.
$arguments = @(
    $cli,
    'optimize',
    '--group', 'resident',
    '--device', $Device,
    '--budget-seconds', $BudgetSecondsPerShape,
    '--max-trials', $MaxNewTrialsPerShape,
    '--max-iterations', $MaxIterations,
    '--no-deployment-patience', $NoDeploymentPatience,
    '--seed', $Seed,
    '--structure-seed', $effectiveStructureSeed,
    '--dtype', 'float32',
    '--padding-ratio', 0,
    '--input-scale', 1
)
foreach ($index in 1..13) {
    if ($index -eq 6 -and -not $IncludeShape06) {
        continue
    }
    $arguments += '--case-id'
    $arguments += ('official_{0:D2}' -f $index)
}

& $python @arguments
exit $LASTEXITCODE
