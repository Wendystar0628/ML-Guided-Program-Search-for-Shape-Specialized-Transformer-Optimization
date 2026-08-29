[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptRoot
$venvActivate = Join-Path $projectRoot '.venv\Scripts\Activate.ps1'
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$cudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'

if (-not (Test-Path -LiteralPath $venvActivate)) {
    throw "Project virtual environment not found: $venvActivate"
}
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw "Visual Studio Installer discovery tool not found: $vswhere"
}
if (-not (Test-Path -LiteralPath $cudaRoot)) {
    throw "CUDA Toolkit 13.2 not found: $cudaRoot"
}

$vsInstall = & $vswhere `
    -latest `
    -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $vsInstall) {
    throw 'Visual Studio C++ Build Tools were not found.'
}

$devShell = Join-Path $vsInstall 'Common7\Tools\Launch-VsDevShell.ps1'
if (-not (Test-Path -LiteralPath $devShell)) {
    throw "Visual Studio developer shell was not found: $devShell"
}

# Populate cl.exe, link.exe, the Windows SDK and related build variables in the
# current PowerShell process without permanently modifying the system PATH.
& $devShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation

# Activate the project-local Python environment.
. $venvActivate

$cacheRoot = Join-Path $projectRoot '.cache'
$torchExtensions = Join-Path $cacheRoot 'torch_extensions'
$tritonCache = Join-Path $cacheRoot 'triton'
$inductorCache = Join-Path $cacheRoot 'torch_inductor'
New-Item -ItemType Directory -Force -Path @(
    $torchExtensions,
    $tritonCache,
    $inductorCache
) | Out-Null

$env:CUDA_PATH = $cudaRoot
$env:CUDA_HOME = $cudaRoot
$env:PATH = "$(Join-Path $cudaRoot 'bin');$env:PATH"

$nvidiaToolsRoot = 'C:\Program Files\NVIDIA Corporation'
$nsightCompute = Get-ChildItem -LiteralPath $nvidiaToolsRoot -Directory `
    -Filter 'Nsight Compute *' -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending |
    Select-Object -First 1
$nsightSystems = Get-ChildItem -LiteralPath $nvidiaToolsRoot -Directory `
    -Filter 'Nsight Systems *' -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending |
    Select-Object -First 1
if ($nsightCompute) {
    $env:PATH = "$($nsightCompute.FullName);$env:PATH"
}
if ($nsightSystems) {
    $nsysCommandPath = Join-Path $nsightSystems.FullName 'target-windows-x64'
    if (Test-Path -LiteralPath $nsysCommandPath) {
        $env:PATH = "$nsysCommandPath;$env:PATH"
    }
}

$env:CC = 'cl'
$env:CXX = 'cl'
$env:CL = '/Zc:preprocessor'
$env:CMAKE_GENERATOR = 'Ninja'
$env:TORCH_CUDA_ARCH_LIST = '8.9'
$env:TORCH_EXTENSIONS_DIR = $torchExtensions
$env:TRITON_CACHE_DIR = $tritonCache
$env:TORCHINDUCTOR_CACHE_DIR = $inductorCache
$env:PYTHONUTF8 = '1'

Write-Host ''
Write-Host 'TikTok TechJam local Windows RTX 4080 environment activated.' -ForegroundColor Green
Write-Host "Project          : $projectRoot"
Write-Host "Python           : $((Get-Command python).Source)"
Write-Host "CUDA_PATH        : $env:CUDA_PATH"
Write-Host "CUDA architecture: $env:TORCH_CUDA_ARCH_LIST (RTX 4080 / sm_89)"
Write-Host "C++ compiler     : $((Get-Command cl).Source)"
Write-Host "Nsight Compute   : $((Get-Command ncu -ErrorAction SilentlyContinue).Source)"
Write-Host "Nsight Systems   : $((Get-Command nsys -ErrorAction SilentlyContinue).Source)"
Write-Host "Build caches     : $cacheRoot"
