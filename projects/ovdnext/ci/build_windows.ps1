# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string] $SourceRoot,
    [Parameter(Mandatory=$true)][string] $BuildRoot,
    [Parameter(Mandatory=$true)][string] $PhysicsRoot,
    [Parameter(Mandatory=$true)][string] $SourceCommit,
    [string] $PhysicsRef = 'a12f79ce585ffb79e5ffa4848cb45efc3de2f632',
    [string] $Configuration = 'RelWithDebInfo',
    [string] $ArtifactRoot,
    [string] $Generator = 'Ninja',
    [string] $CMakeExecutable = 'cmake',
    [string] $CTestExecutable = 'ctest',
    [string] $PythonExecutable = 'python'
)
$ErrorActionPreference = 'Stop'
$arguments = @((Join-Path $PSScriptRoot 'build_native.py'),
    '--source-root', $SourceRoot, '--build-root', $BuildRoot, '--physics-root', $PhysicsRoot,
    '--source-commit', $SourceCommit, '--physics-ref', $PhysicsRef,
    '--configuration', $Configuration, '--generator', $Generator,
    '--cmake', $CMakeExecutable, '--ctest', $CTestExecutable)
if ($ArtifactRoot) { $arguments += @('--artifact-root', $ArtifactRoot) }
& $PythonExecutable @arguments
exit $LASTEXITCODE
