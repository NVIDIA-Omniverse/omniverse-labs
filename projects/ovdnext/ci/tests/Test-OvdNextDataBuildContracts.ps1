# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$standalonePath = Join-Path $repositoryRoot 'ci/build_ovdnext_data_windows.ps1'

function Assert-Contains {
    param([string] $Text, [string] $Expected, [string] $Message)
    if (-not $Text.Contains($Expected)) { throw "FAIL: $Message" }
}

function Assert-True {
    param([bool] $Condition, [string] $Message)
    if (-not $Condition) { throw "FAIL: $Message" }
}

function Get-NativeLogFunctionDefinition {
    param([string] $Path)
    $tokens = $null
    $parseErrors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile(
        $Path,
        [ref] $tokens,
        [ref] $parseErrors)
    Assert-True ($parseErrors.Count -eq 0) "$Path must parse before its log helper is tested"
    $functionAst = $ast.Find({
        param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
            $node.Name -ceq 'Invoke-NativeWithLog'
    }, $true)
    Assert-True ($null -ne $functionAst) "$Path must define Invoke-NativeWithLog"
    return [scriptblock]::Create($functionAst.Extent.Text)
}

function Assert-NativeLogBehavior {
    param([string] $Path, [string] $Label)
    . (Get-NativeLogFunctionDefinition -Path $Path)
    $logRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("ovdnext-log-{0}" -f [guid]::NewGuid())
    New-Item -ItemType Directory -Path $logRoot | Out-Null
    try {
        $successLog = Join-Path $logRoot 'silent-success.log'
        Invoke-NativeWithLog -Command $env:ComSpec -Arguments @('/d', '/c', 'exit /b 0') -LogPath $successLog
        Assert-True (Test-Path -LiteralPath $successLog -PathType Leaf) `
            "$Label must create a log for a silent successful command"
        Assert-True ((Get-Item -LiteralPath $successLog).Length -eq 0) `
            "$Label silent successful command log must be empty"

        $failureLog = Join-Path $logRoot 'silent-failure.log'
        $failed = $false
        try {
            Invoke-NativeWithLog -Command $env:ComSpec -Arguments @('/d', '/c', 'exit /b 23') -LogPath $failureLog
        }
        catch {
            $failed = $_.Exception.Message -match 'exit code 23'
        }
        Assert-True $failed "$Label must preserve a silent native failure exit code"
        Assert-True (Test-Path -LiteralPath $failureLog -PathType Leaf) `
            "$Label must create a log for a silent failed command"
    }
    finally {
        Remove-Item -LiteralPath $logRoot -Recurse -Force
    }
}

if (-not (Test-Path -LiteralPath $standalonePath -PathType Leaf)) {
    throw "FAIL: Windows standalone OvdNext Data build script is missing: $standalonePath"
}
$standalone = Get-Content -Raw -LiteralPath $standalonePath
foreach ($expected in @(
    'src/OvdDataOmniPvd',
    '-DCMAKE_BUILD_TYPE=',
    '-DCMAKE_MAKE_PROGRAM=',
    '-DPVDRUNTIME_ROOT=',
    '-DOVDDATA_BUILD_TESTS=ON',
    '-DOVDDATAOMNIPVD_BUILD_TESTS=ON',
    'ovddata_tests',
    'ovddata_ingest_tests',
    'ovddata_omnipvd_tests',
    'ovdnext_data_probe',
    'ovdnext_data_probe_tests',
    'ctest-discovery.json',
    'ctest.junit.xml',
    '--graphviz',
    'targets.dot',
    'verify_test_inventory.py',
    'inventory-verifier.log',
    '--scope',
    'ovdnext-data',
    'verify_ovdnext_data_dependencies.py',
    'test/data/test_scene.ovd',
    'dumpbin',
    'imports.txt'
)) {
    Assert-Contains $standalone $expected "Windows standalone build contract is missing $expected"
}
Assert-Contains $standalone '14.29.30133' `
    'Windows standalone build must validate exact MSVC 14.29.30133 activation'
Assert-Contains $standalone '10.0.17763.0' `
    'Windows standalone build must validate exact Windows SDK 10.0.17763.0 activation'
Assert-Contains $standalone "'--platform', 'windows-x86_64'" `
    'Windows standalone build must select the native inventory platform explicitly'

Assert-NativeLogBehavior $standalonePath 'Windows standalone build driver'

Write-Host 'PASS: Windows OvdNext Data build contracts'
