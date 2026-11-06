# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
param([string]$RuntimeRoot, [switch]$Verify, [ValidateSet('public')][string]$Profile='public', [string]$Quality)
$ErrorActionPreference='Stop'
$repoRoot=Split-Path $PSScriptRoot -Parent
if (-not $RuntimeRoot) { $RuntimeRoot=Join-Path $repoRoot '.ov_library_lab_runtime' }
$nativePython=Join-Path $RuntimeRoot 'libraries-env/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $nativePython)) { throw 'Run tools/setup_libraries.ps1 first.' }
$env:PYTHONNOUSERSITE='1'
foreach ($variable in @('PYTHONPATH','PYTHONHOME','CARB_APP_PATH')) {
    [Environment]::SetEnvironmentVariable($variable,$null,'Process')
}
& $nativePython (Join-Path $PSScriptRoot 'setup_scene.py') --runtime-root $RuntimeRoot --profile public
if ($LASTEXITCODE) { throw 'Current scene setup is incomplete. See docs/SCENE_ASSETS.md.' }
$arguments=@('-u',(Join-Path $PSScriptRoot 'run_lab.py'),'--runtime-root',$RuntimeRoot,'--profile',$Profile)
if ($Quality) { $arguments+=@('--quality',$Quality) }
if ($Verify) { $arguments+='--verify' }
& $nativePython @arguments
exit $LASTEXITCODE
