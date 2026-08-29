[CmdletBinding()]
param(
    [string]$Device = 'cuda:0',
    [double]$DurationHours = 6.0,
    [double]$ResidentBudgetSeconds = 300.0,
    [double]$Shape14BudgetSeconds = 1200.0,
    [int]$Seed = 1234
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

foreach ($value in @($DurationHours, $ResidentBudgetSeconds, $Shape14BudgetSeconds)) {
    if ($value -le 0) {
        throw 'Duration and search budgets must be positive.'
    }
}

$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $projectRoot 'environments\activate_windows_rtx4080.ps1')

$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$cli = Join-Path $projectRoot 'cli.py'
$runStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$runRoot = Join-Path $projectRoot "search_state\overnight\$runStamp"
$statusPath = Join-Path $runRoot 'status.log'
$latestRunPath = Join-Path $projectRoot 'search_state\overnight\latest_run.txt'
$deadline = (Get-Date).AddHours($DurationHours)
$env:PYTHONUNBUFFERED = '1'

New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
Set-Content -LiteralPath $latestRunPath -Value $runRoot -Encoding UTF8
Set-Content -LiteralPath (Join-Path $runRoot 'coordinator.pid') -Value $PID -Encoding ASCII

function Write-Status {
    param([string]$Message)

    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Host $line
    Add-Content -LiteralPath $statusPath -Value $line -Encoding UTF8
}

function Invoke-OptimizationPass {
    param(
        [string]$Group,
        [double]$BudgetSeconds,
        [int]$PassSeed,
        [int]$PassIndex
    )

    $passLog = Join-Path $runRoot ("{0:D2}_{1}.log" -f $PassIndex, $Group)
    Write-Status (
        "Starting group=$Group budget_seconds=$BudgetSeconds seed=$PassSeed " +
        "remaining_seconds=$([Math]::Floor(($deadline - (Get-Date)).TotalSeconds))"
    )
    $arguments = @(
        $cli,
        'optimize',
        '--group', $Group,
        '--device', $Device,
        '--budget-seconds', $BudgetSeconds,
        '--no-deployment-patience', 1,
        '--max-iterations', 1,
        '--seed', $PassSeed
    )
    $outputLines = @(& $python @arguments 2>&1 | Tee-Object -FilePath $passLog)
    $exitCode = $LASTEXITCODE
    $stopLine = $outputLines |
        Where-Object { "$_" -like 'stopped: *' } |
        Select-Object -Last 1
    $stopReason = if ($null -eq $stopLine) {
        'unknown'
    } else {
        ("$stopLine" -replace '^stopped:\s*', '')
    }
    Write-Status "Finished group=$Group exit_code=$exitCode stop_reason=$stopReason"
    return [PSCustomObject]@{
        ExitCode = $exitCode
        Exhausted = ($stopReason -eq 'search_space_exhausted')
    }
}

$minimumResidentBudget = 120.0
$minimumShape14Budget = 300.0
$shutdownReserveSeconds = 60.0
$residentExhausted = $false
$shape14Exhausted = $false
$passIndex = 0

Write-Status (
    "Overnight optimization started device=$Device duration_hours=$DurationHours " +
    "deadline=$($deadline.ToString('o'))"
)

while ((Get-Date) -lt $deadline) {
    $ranPass = $false

    if (-not $residentExhausted) {
        $remainingSeconds = ($deadline - (Get-Date)).TotalSeconds
        $residentBudget = [Math]::Min(
            $ResidentBudgetSeconds,
            [Math]::Floor(
                ($remainingSeconds - $minimumShape14Budget - $shutdownReserveSeconds) / 13.0
            )
        )
        if ($residentBudget -ge $minimumResidentBudget) {
            $passIndex += 1
            $result = Invoke-OptimizationPass `
                -Group 'resident' `
                -BudgetSeconds $residentBudget `
                -PassSeed ($Seed + $passIndex - 1) `
                -PassIndex $passIndex
            $ranPass = $true
            $residentExhausted = $result.Exhausted
            if ($result.ExitCode -eq 130) {
                Write-Status 'Interrupted during resident optimization.'
                exit 130
            }
        }
    }

    if (-not $shape14Exhausted) {
        $remainingSeconds = ($deadline - (Get-Date)).TotalSeconds
        $shape14Budget = [Math]::Min(
            $Shape14BudgetSeconds,
            [Math]::Floor($remainingSeconds - $shutdownReserveSeconds)
        )
        if ($shape14Budget -ge $minimumShape14Budget) {
            $passIndex += 1
            $result = Invoke-OptimizationPass `
                -Group 'shape14' `
                -BudgetSeconds $shape14Budget `
                -PassSeed ($Seed + $passIndex - 1) `
                -PassIndex $passIndex
            $ranPass = $true
            $shape14Exhausted = $result.Exhausted
            if ($result.ExitCode -eq 130) {
                Write-Status 'Interrupted during Shape 14 optimization.'
                exit 130
            }
        }
    }

    if (($residentExhausted -and $shape14Exhausted) -or -not $ranPass) {
        break
    }
}

Write-Status (
    "Overnight optimization finished passes=$passIndex " +
    "resident_exhausted=$residentExhausted shape14_exhausted=$shape14Exhausted"
)
exit 0
