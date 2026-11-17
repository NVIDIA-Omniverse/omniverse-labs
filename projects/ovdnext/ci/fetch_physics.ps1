# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $RepositoryUrl,

    [Parameter(Mandatory = $true)]
    [string] $Ref,

    [Parameter(Mandatory = $true)]
    [string] $Destination,

    [switch] $IncludePackman
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]] $Arguments)

    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Test-RequiredFile {
    param([string] $Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required fetched file is missing: $Path"
    }
}

if ($Ref -notmatch '^[0-9a-fA-F]{40}$') {
    Write-Error 'Ref must be an exact 40-character hexadecimal commit ID.'
    exit 1
}

$destinationParent = [System.IO.Path]::GetDirectoryName([System.IO.Path]::GetFullPath($Destination))
if (-not (Test-Path -LiteralPath $destinationParent -PathType Container)) {
    Write-Error "Destination parent does not exist: $destinationParent"
    exit 1
}

$destinationCreatedByThisInvocation = $false
if (Test-Path -LiteralPath $Destination) {
    if (-not (Test-Path -LiteralPath $Destination -PathType Container)) {
        Write-Error "Destination must be an empty directory: $Destination"
        exit 1
    }
    if (@(Get-ChildItem -LiteralPath $Destination -Force).Count -ne 0) {
        Write-Error "Destination must be empty: $Destination"
        exit 1
    }
}
else {
    try {
        New-Item -ItemType Directory -Path $Destination -ErrorAction Stop | Out-Null
        $destinationCreatedByThisInvocation = $true
    }
    catch {
        Write-Error "Destination could not be created (it may have appeared concurrently): $Destination"
        exit 1
    }
}

try {
    Invoke-Git init $Destination
    Invoke-Git -C $Destination remote add origin $RepositoryUrl
    Invoke-Git -C $Destination sparse-checkout init --cone

    $sparsePaths = @('physx/pvddom', 'physx/pvdruntime', 'physx/include/foundation')
    if ($IncludePackman) {
        $sparsePaths += 'physx/buildtools/packman'
    }
    Invoke-Git -C $Destination sparse-checkout set @sparsePaths
    Invoke-Git -C $Destination fetch --depth=1 origin $Ref
    Invoke-Git -C $Destination checkout --detach FETCH_HEAD

    $normalizedRef = $Ref.ToLowerInvariant()
    $head = [string](Invoke-Git -C $Destination rev-parse HEAD)
    if ($head -cne $normalizedRef) {
        throw "Fetched HEAD '$head' does not match requested commit '$normalizedRef'."
    }

    Test-RequiredFile (Join-Path $Destination 'physx/pvddom/CMakeLists.txt')
    Test-RequiredFile (Join-Path $Destination 'physx/pvdruntime/compiler/cmake/PVDRuntime.cmake')
    if ($IncludePackman) {
        Test-RequiredFile (Join-Path $Destination 'physx/buildtools/packman/packman.cmd')
    }
}
catch {
    if ($destinationCreatedByThisInvocation -and (Test-Path -LiteralPath $Destination)) {
        Remove-Item -LiteralPath $Destination -Recurse -Force -ErrorAction SilentlyContinue
    }
    Write-Error $_
    exit 1
}

Write-Host "Fetched Physics commit $($Ref.ToLowerInvariant()) into $Destination"
