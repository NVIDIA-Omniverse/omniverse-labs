# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
<# Install the public Newton and Package exhibits in isolated environments. #>
param([string]$Python='python', [string]$RuntimeRoot)
$ErrorActionPreference='Stop'
$repoRoot=Split-Path $PSScriptRoot -Parent
if (-not $RuntimeRoot) { $RuntimeRoot=Join-Path $repoRoot '.ov_library_lab_runtime' }
& $Python -c 'import sys,struct; assert sys.version_info[:2]==(3,12) and struct.calcsize(chr(80))==8, "CPython 3.12 x64 is required"'
if ($LASTEXITCODE) { throw 'Choose CPython 3.12 x64 using -Python.' }
foreach ($library in @('newton','package')) {
    $environmentRoot=Join-Path $RuntimeRoot ($library+'-env')
    $workerPython=Join-Path $environmentRoot 'Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $workerPython)) {
        & $Python -m venv $environmentRoot
        if ($LASTEXITCODE) { throw "Could not create $library environment." }
    }
    & $workerPython -m pip install --require-hashes -r (Join-Path $repoRoot ('config/requirements-'+$library+'-win.lock'))
    if ($LASTEXITCODE) { throw "$library installation failed." }
    & $workerPython -m pip check
    if ($LASTEXITCODE) { throw "$library dependencies are incompatible." }
    $roomLock=Join-Path $repoRoot ('config/requirements-'+$library+'-win.lock')
    $roomHash=(Get-FileHash -LiteralPath $roomLock -Algorithm SHA256).Hash
    Set-Content -LiteralPath (Join-Path $RuntimeRoot ($library+'-installation-lock.sha256')) -Value $roomHash -Encoding Ascii
}
Write-Host 'Newton and Package room libraries installed. Run scripts/launch_ov_library_lab.bat.'
