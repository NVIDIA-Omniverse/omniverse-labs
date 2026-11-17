# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $SourceRoot,
    [Parameter(Mandatory = $true)]
    [string] $BuildRoot,
    [Parameter(Mandatory = $true)]
    [string] $PhysicsRoot,
    [string] $Configuration = 'RelWithDebInfo',
    [Parameter(Mandatory = $true)]
    [string] $SourceCommit,
    [string] $PhysicsRef = 'a12f79ce585ffb79e5ffa4848cb45efc3de2f632',
    [string] $Generator = 'Ninja',
    [string] $CMakeExecutable = 'cmake',
    [string] $CTestExecutable = 'ctest',
    [string] $NinjaExecutable = 'ninja',
    [string] $PythonExecutable = 'python',
    [string] $DumpbinExecutable = 'dumpbin'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Assert-Commit {
    param([string] $Name, [string] $Value)
    if ($Value -notmatch '^[0-9a-fA-F]{40}$') {
        throw "$Name must be an exact 40-character hexadecimal commit ID."
    }
}

function Assert-Directory {
    param([string] $Name, [string] $Path)
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Name is missing: $Path"
    }
}

function Assert-File {
    param([string] $Name, [string] $Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Name is missing: $Path"
    }
}

function Invoke-Native {
    param([string] $Command, [string[]] $Arguments)
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Invoke-NativeWithLog {
    param([string] $Command, [string[]] $Arguments, [string] $LogPath)
    [System.IO.File]::WriteAllText($LogPath, '', [System.Text.UTF8Encoding]::new($false))
    $savedErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Command @Arguments 2>&1 | Tee-Object -FilePath $LogPath
        $nativeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $savedErrorActionPreference
    }
    if ($nativeExitCode -ne 0) {
        throw "$Command $($Arguments -join ' ') failed with exit code $nativeExitCode"
    }
}

function Get-GitHead {
    param([string] $Name, [string] $Root)
    $head = (& git -C $Root rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head -notmatch '^[0-9a-fA-F]{40}$') {
        throw "Unable to resolve $Name HEAD as an exact commit."
    }
    return $head.ToLowerInvariant()
}

function Assert-ActivatedToolchain {
    $vcToolsVersion = ([string]$env:VCToolsVersion).Trim().TrimEnd('\', '/')
    if ($vcToolsVersion -cne '14.29.30133') {
        throw "Standalone OvdNext Data verification requires VCToolsVersion 14.29.30133; got '$vcToolsVersion'."
    }
    $windowsSDKVersion = ([string]$env:WindowsSDKVersion).Trim().TrimEnd('\', '/')
    if ($windowsSDKVersion -cne '10.0.17763.0') {
        throw "Standalone OvdNext Data verification requires WindowsSDKVersion 10.0.17763.0; got '$windowsSDKVersion'."
    }
}

try {
    Assert-Commit 'SourceCommit' $SourceCommit
    Assert-Commit 'PhysicsRef' $PhysicsRef
    Assert-Directory 'SourceRoot' $SourceRoot
    Assert-Directory 'PhysicsRoot' $PhysicsRoot
    if ((Get-GitHead 'SourceRoot' $SourceRoot) -ine $SourceCommit) {
        throw 'SourceRoot HEAD does not match SourceCommit.'
    }
    if ((Get-GitHead 'PhysicsRoot' $PhysicsRoot) -ine $PhysicsRef) {
        throw 'PhysicsRoot HEAD does not match PhysicsRef.'
    }
    if ($Configuration -cne 'RelWithDebInfo') {
        throw 'Standalone OvdNext Data verification requires RelWithDebInfo.'
    }
    Assert-ActivatedToolchain

    $adapterSource = Join-Path $SourceRoot 'src/OvdDataOmniPvd'
    $runtimeRoot = Join-Path $PhysicsRoot 'physx/pvdruntime'
    $fixture = Join-Path $SourceRoot 'test/data/test_scene.ovd'
    $baseline = Join-Path $SourceRoot 'ci/baselines/qt-free/ovdnext-ci-parity.json'
    $inventoryVerifier = Join-Path $SourceRoot 'ci/verify_test_inventory.py'
    $dependencyVerifier = Join-Path $SourceRoot 'ci/verify_ovdnext_data_dependencies.py'
    Assert-Directory 'OvdDataOmniPvd source' $adapterSource
    Assert-Directory 'PVDRuntime root' $runtimeRoot
    Assert-File 'Probe fixture' $fixture
    Assert-File 'Test inventory baseline' $baseline
    Assert-File 'Test inventory verifier' $inventoryVerifier
    Assert-File 'Dependency verifier' $dependencyVerifier

    New-Item -ItemType Directory -Path $BuildRoot -Force | Out-Null
    $configureArguments = @(
        '-S', $adapterSource,
        '-B', $BuildRoot,
        '-G', $Generator,
        "-DCMAKE_BUILD_TYPE=$Configuration",
        "-DPVDRUNTIME_ROOT=$runtimeRoot",
        '-DOVDDATA_BUILD_TESTS=ON',
        '-DOVDDATAOMNIPVD_BUILD_TESTS=ON'
    )
    if ($Generator -ceq 'Ninja') {
        $configureArguments += "-DCMAKE_MAKE_PROGRAM=$NinjaExecutable"
    }
    elseif ($Generator -like 'Visual Studio*') {
        $configureArguments += @('-A', 'x64')
    }
    Invoke-Native -Command $CMakeExecutable -Arguments $configureArguments

    $graphPath = Join-Path $BuildRoot 'targets.dot'
    Invoke-Native -Command $CMakeExecutable -Arguments @("--graphviz=$graphPath", $BuildRoot)
    Invoke-Native -Command $CMakeExecutable -Arguments @(
        '--build', $BuildRoot, '--config', $Configuration,
        '--target',
        'ovddata_tests',
        'ovddata_ingest_tests',
        'ovddata_omnipvd_tests',
        'ovdnext_data_probe',
        'ovdnext_data_probe_tests',
        '--parallel'
    )

    $discoveryPath = Join-Path $BuildRoot 'ctest-discovery.json'
    $junitPath = Join-Path $BuildRoot 'ctest.junit.xml'
    $discovery = (& $CTestExecutable --test-dir $BuildRoot -C $Configuration --show-only=json-v1 | Out-String)
    if ($LASTEXITCODE -ne 0) {
        throw "$CTestExecutable discovery failed with exit code $LASTEXITCODE"
    }
    [System.IO.File]::WriteAllText(
        $discoveryPath,
        $discovery,
        [System.Text.UTF8Encoding]::new($false))
    Invoke-Native -Command $CTestExecutable -Arguments @(
        '--test-dir', $BuildRoot, '-C', $Configuration,
        '--no-tests=error', '--output-on-failure',
        '--output-junit', $junitPath
    )
    Invoke-NativeWithLog -Command $PythonExecutable -Arguments @(
        $inventoryVerifier, 'verify',
        '--baseline', $baseline,
        '--discovery', $discoveryPath,
        '--junit', $junitPath,
        '--scope', 'ovdnext-data',
        '--platform', 'windows-x86_64'
    ) -LogPath (Join-Path $BuildRoot 'inventory-verifier.log')

    $probeCandidates = @(
        (Join-Path $BuildRoot 'ovdnext_data_probe.exe'),
        (Join-Path $BuildRoot "$Configuration/ovdnext_data_probe.exe")
    )
    $probe = $probeCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    if (-not $probe) {
        throw "Built ovdnext_data_probe.exe was not found under $BuildRoot."
    }
    $probeOutput = Join-Path $BuildRoot 'probe-output.txt'
    $probeLines = (& $probe $fixture | Out-String)
    if ($LASTEXITCODE -ne 0) {
        throw "ovdnext_data_probe failed with exit code $LASTEXITCODE"
    }
    [System.IO.File]::WriteAllText(
        $probeOutput,
        $probeLines,
        [System.Text.UTF8Encoding]::new($false))

    $importsPath = Join-Path $BuildRoot 'imports.txt'
    $imports = (& $DumpbinExecutable /dependents $probe | Out-String)
    if ($LASTEXITCODE -ne 0) {
        throw "$DumpbinExecutable /dependents failed with exit code $LASTEXITCODE"
    }
    [System.IO.File]::WriteAllText(
        $importsPath,
        $imports,
        [System.Text.UTF8Encoding]::new($false))
    Invoke-Native -Command $PythonExecutable -Arguments @(
        $dependencyVerifier,
        '--graph', $graphPath,
        '--root-target', 'ovdnext_data_probe',
        '--imports', $importsPath
    )
}
catch {
    Write-Error $_
    exit 1
}
