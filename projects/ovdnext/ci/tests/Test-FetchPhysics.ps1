# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$fetcher = Join-Path $repositoryRoot 'ci\fetch_physics.ps1'
$temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("fetch-physics-tests-" + [Guid]::NewGuid().ToString('N'))

function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]] $Arguments)

    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Assert-True {
    param([bool] $Condition, [string] $Message)

    if (-not $Condition) {
        throw $Message
    }
}

function New-FixtureRepository {
    param([string] $Name, [bool] $IncludePackman)

    $repository = Join-Path $temporaryRoot $Name
    New-Item -ItemType Directory -Path $repository -Force | Out-Null
    Invoke-Git init -q $repository
    Invoke-Git -C $repository config user.email 'ci@example.invalid'
    Invoke-Git -C $repository config user.name 'CI Test'

    $paths = @(
        'physx/pvddom/CMakeLists.txt',
        'physx/pvdruntime/compiler/cmake/PVDRuntime.cmake',
        'unrelated/sentinel.txt'
    )
    if ($IncludePackman) {
        $paths += 'physx/buildtools/packman/packman.cmd'
    }

    foreach ($path in $paths) {
        $fullPath = Join-Path $repository $path
        New-Item -ItemType Directory -Path (Split-Path -Parent $fullPath) -Force | Out-Null
        Set-Content -Path $fullPath -Value $path -NoNewline
    }

    Invoke-Git -C $repository add .
    Invoke-Git -C $repository commit -qm 'fixture'
    return [string](& git -C $repository rev-parse HEAD)
}

function Invoke-Fetcher {
    param(
        [string] $RepositoryUrl,
        [string] $Ref,
        [string] $Destination,
        [bool] $IncludePackman = $false
    )

    $arguments = @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', $fetcher,
        '-RepositoryUrl', $RepositoryUrl,
        '-Ref', $Ref,
        '-Destination', $Destination
    )
    if ($IncludePackman) {
        $arguments += '-IncludePackman'
    }

    $null = & powershell @arguments
    return $LASTEXITCODE
}

function Assert-PathPresent {
    param([string] $Path)
    Assert-True (Test-Path -LiteralPath $Path -PathType Leaf) "Expected path to exist: $Path"
}

function Assert-Failure {
    param([scriptblock] $Action, [string] $Message)
    $exitCode = & $Action
    Assert-True ($exitCode -ne 0) $Message
}

try {
    New-Item -ItemType Directory -Path $temporaryRoot -Force | Out-Null
    $shaWithPackman = New-FixtureRepository -Name 'with-packman' -IncludePackman $true
    $shaWithoutPackman = New-FixtureRepository -Name 'without-packman' -IncludePackman $false

    Assert-True ($shaWithPackman -cmatch '^[0-9a-f]{40}$') 'Fixture commit must be a 40-character SHA.'
    $successDestination = Join-Path $temporaryRoot 'success'
    $exitCode = Invoke-Fetcher -RepositoryUrl (Join-Path $temporaryRoot 'with-packman') -Ref $shaWithPackman -Destination $successDestination -IncludePackman $true
    Assert-True ($exitCode -eq 0) 'Fetcher should succeed for an exact SHA and empty destination.'
    $head = [string](& git -C $successDestination rev-parse HEAD)
    Assert-True ($head -ceq $shaWithPackman) 'Fetched HEAD must match the requested SHA byte-for-byte.'
    Assert-PathPresent (Join-Path $successDestination 'physx/pvddom/CMakeLists.txt')
    Assert-PathPresent (Join-Path $successDestination 'physx/pvdruntime/compiler/cmake/PVDRuntime.cmake')
    Assert-PathPresent (Join-Path $successDestination 'physx/buildtools/packman/packman.cmd')
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $successDestination 'unrelated/sentinel.txt'))) 'Sparse checkout must exclude the sentinel.'

    Assert-Failure { Invoke-Fetcher -RepositoryUrl (Join-Path $temporaryRoot 'with-packman') -Ref 'not-a-commit' -Destination (Join-Path $temporaryRoot 'invalid-ref') } 'Non-hex refs must fail.'

    $nonemptyDestination = Join-Path $temporaryRoot 'nonempty'
    New-Item -ItemType Directory -Path $nonemptyDestination | Out-Null
    $preservedFile = Join-Path $nonemptyDestination 'preserve.txt'
    Set-Content -Path $preservedFile -Value 'preserve' -NoNewline
    Assert-Failure { Invoke-Fetcher -RepositoryUrl (Join-Path $temporaryRoot 'with-packman') -Ref $shaWithPackman -Destination $nonemptyDestination } 'Nonempty destinations must fail.'
    Assert-PathPresent $preservedFile

    Assert-Failure { Invoke-Fetcher -RepositoryUrl (Join-Path $temporaryRoot 'without-packman') -Ref $shaWithoutPackman -Destination (Join-Path $temporaryRoot 'missing-packman') -IncludePackman $true } 'Missing requested Packman paths must fail.'

    $preexistingEmptyDestination = Join-Path $temporaryRoot 'preexisting-empty'
    New-Item -ItemType Directory -Path $preexistingEmptyDestination | Out-Null
    Assert-Failure { Invoke-Fetcher -RepositoryUrl (Join-Path $temporaryRoot 'without-packman') -Ref $shaWithoutPackman -Destination $preexistingEmptyDestination -IncludePackman $true } 'A later failure must not remove a pre-existing empty destination.'
    Assert-True (Test-Path -LiteralPath $preexistingEmptyDestination -PathType Container) 'A pre-existing empty destination must be preserved after a later failure.'

    Write-Host 'PASS: fetch_physics.ps1 local fixture tests'
}
finally {
    if (Test-Path -LiteralPath $temporaryRoot) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
}
