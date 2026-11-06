# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
<# Install the main OV Libraries environment. #>
param([string]$Python='python', [string]$RuntimeRoot, [ValidateSet('public')][string]$Profile='public')
$ErrorActionPreference='Stop'
$repoRoot=Split-Path $PSScriptRoot -Parent
if (-not $RuntimeRoot) { $RuntimeRoot=Join-Path $repoRoot '.ov_library_lab_runtime' }
& $Python -c 'import sys,struct; assert sys.version_info[:2]==(3,12) and struct.calcsize(chr(80))==8, "CPython 3.12 x64 is required"'
if ($LASTEXITCODE) { throw 'Choose CPython 3.12 x64 using -Python.' }
$environmentRoot=Join-Path $RuntimeRoot 'libraries-env'
$nativePython=Join-Path $environmentRoot 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $nativePython)) {
    & $Python -m venv $environmentRoot
    if ($LASTEXITCODE) { throw 'Could not create the library environment.' }
}
& $nativePython -m pip install --require-hashes -r (Join-Path $repoRoot 'config/requirements-win.lock')
if ($LASTEXITCODE) { throw 'Library installation failed; see pip output above.' }
& $nativePython -m pip check
if ($LASTEXITCODE) { throw 'The library dependencies are incompatible.' }
& (Join-Path $PSScriptRoot 'setup_extended_libraries.ps1') -Python $Python -RuntimeRoot $RuntimeRoot
& $nativePython (Join-Path $PSScriptRoot 'setup_scene.py') --runtime-root $RuntimeRoot --profile $Profile
if ($LASTEXITCODE) { throw 'Scene preparation failed. See docs/CUSTOMER_GUIDE.md and the setup log.' }
Write-Host 'LAB_PROGRESS: Preparing NVIDIA SimReady exhibit assets'
& $nativePython (Join-Path $PSScriptRoot 'setup_simready.py') --runtime-root $RuntimeRoot
if ($LASTEXITCODE) { throw 'SimReady asset preparation failed. Retry to reuse verified downloads.' }
Write-Host 'Standalone libraries installed. Run scripts/launch_ov_library_lab.bat.'
