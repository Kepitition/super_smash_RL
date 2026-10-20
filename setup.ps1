# Windows counterpart of setup.sh: creates .venv (Python 3.10-3.12), installs the
# dependencies and downloads the portable Slippi Dolphin 3.6.4 into dolphin\.
#
#   powershell -ExecutionPolicy Bypass -File setup.ps1          # CUDA PyTorch if an NVIDIA GPU is found
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -Cpu     # CPU-only PyTorch (~200 MB instead of ~3 GB)
param([switch]$Cpu)
# Native commands are checked through $LASTEXITCODE: with "Stop", PowerShell 5.1 would
# treat any stderr output of python / pip as a fatal error.
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"  # Invoke-WebRequest is very slow with the progress bar
Set-Location $PSScriptRoot

$python = ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    # numpy<2 has no wheels for Python 3.13+, so use the newest of 3.12 / 3.11 / 3.10.
    $version = $null
    foreach ($v in "3.12", "3.11", "3.10") {
        & py "-$v" -c "pass" 2>$null
        if ($LASTEXITCODE -eq 0) { $version = $v; break }
    }
    if (-not $version) { throw "Python 3.10, 3.11 or 3.12 not found (py launcher). Install one from python.org." }
    Write-Host "Creating .venv with Python $version..."
    & py "-$version" -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}
& $python -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

& $python -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    $gpu = (-not $Cpu) -and [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
    $index = if ($gpu) { "https://download.pytorch.org/whl/cu128" } else { "https://download.pytorch.org/whl/cpu" }
    Write-Host "Installing PyTorch from $index..."
    & $python -m pip install torch --index-url $index
    if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed" }
}
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt failed" }

$dolphinDir = "dolphin\Slippi Dolphin"
if (-not (Test-Path "$dolphinDir\Slippi Dolphin.exe")) {
    Write-Host "Downloading Slippi Dolphin 3.6.4 for Windows..."
    New-Item -ItemType Directory -Force dolphin -ErrorAction Stop | Out-Null
    $zip = "dolphin\FM-Slippi-3.6.4-Win.zip"
    Invoke-WebRequest -Uri "https://github.com/project-slippi/Ishiiruka/releases/download/v3.6.4/FM-Slippi-3.6.4-Win.zip" -OutFile $zip -ErrorAction Stop
    Expand-Archive -Path $zip -DestinationPath $dolphinDir -Force -ErrorAction Stop
    Remove-Item $zip
}
if (-not (Test-Path "$env:SystemRoot\System32\vcruntime140_1.dll")) {
    Write-Warning "Dolphin needs the Visual C++ runtime: https://aka.ms/vs/17/release/vc_redist.x64.exe"
}

if (-not (Test-Path "Super Smash Bros Melee v1.02.iso")) {
    Write-Warning "put your NTSC v1.02 Melee ISO here as 'Super Smash Bros Melee v1.02.iso'"
}

Write-Host ""
Write-Host "Done. Activate with:  .venv\Scripts\Activate.ps1"
Write-Host "Then check the game:  python test_env.py"
