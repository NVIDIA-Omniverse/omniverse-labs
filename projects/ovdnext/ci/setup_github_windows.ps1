# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [string] $SourceRoot,
    [string] $GitHubEnvPath = $env:GITHUB_ENV,
    [string] $PythonExecutable = 'python',
    [string] $VswherePath = (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'),
    [string] $WindowsKitsRoot = (Join-Path ${env:ProgramFiles(x86)} 'Windows Kits/10')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    $SourceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}

$script:RequiredVCToolsVersion = '14.29.30133'
$script:RequiredWindowsSDKVersion = '10.0.17763.0'
$script:CMakeVersion = '3.25.1'
$script:CMakeArchiveName = 'cmake-3.25.1-windows-x86_64.zip'
$script:CMakeArchiveUrl = 'https://github.com/Kitware/CMake/releases/download/v3.25.1/cmake-3.25.1-windows-x86_64.zip'
$script:CMakeArchiveSha256 = 'd93958d87cc9b91983489f0b37a268b03a3c891894d11f5437fa2a5ce94aab24'
$script:PythonLockName = 'windows-cp311-x64.lock'
$script:RequiredRuntimeDlls = @('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll')

function Assert-LeafFile {
    param([string] $Label, [string] $Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label is missing: $Path"
    }
}

function Assert-Directory {
    param([string] $Label, [string] $Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Label is missing: $Path"
    }
}

function Assert-SafeEnvironmentValue {
    param([string] $Name, [string] $Value)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "GitHub environment value '$Name' is empty."
    }
    if ($Value -match '[\r\n]') {
        throw "GitHub environment value '$Name' contains a newline."
    }
}

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory)] [string] $Command,
        [string[]] $Arguments = @(),
        [string] $Label = $Command
    )
    & $Command @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode."
    }
}

function Get-CheckedNativeVersion {
    param(
        [Parameter(Mandatory)] [string] $Command,
        [string[]] $Arguments = @(),
        [Parameter(Mandatory)] [string] $Label,
        [Parameter(Mandatory)] [string] $ExpectedPattern,
        [Parameter(Mandatory)] [string] $ExpectedDescription
    )
    $output = @(& $Command @Arguments)
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Label version query failed with exit code $exitCode."
    }
    $lines = @($output | ForEach-Object { ([string]$_).Trim() } | Where-Object { $_ -ne '' })
    if ($lines.Count -eq 0) { throw "$Label version query produced no output." }
    $version = $lines[0]
    if ($version -notmatch $ExpectedPattern) {
        throw "$Label must report $ExpectedDescription; got '$version'."
    }
    return $version
}

function Assert-WindowsPython311X64 {
    param([Parameter(Mandatory)] [string] $Command)
    $query = "import platform, struct, sys; print('%d.%d|%s|%d' % (sys.version_info[0], sys.version_info[1], platform.machine(), struct.calcsize('P') * 8))"
    $output = @(& $Command -c $query)
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Python compatibility query failed with exit code $exitCode."
    }
    $lines = @($output | ForEach-Object { ([string]$_).Trim() } | Where-Object { $_ -ne '' })
    if ($lines.Count -ne 1 -or $lines[0] -notmatch '^3\.11\|(?:AMD64|x86_64)\|64$') {
        throw "Python must be CPython-compatible 3.11 x64 on Windows; got '$($lines -join ' ')'."
    }
}

function Save-PublicFile {
    param(
        [Parameter(Mandatory)] [string] $Uri,
        [Parameter(Mandatory)] [string] $Destination,
        [scriptblock] $DownloadAction = {
            param($RequestedUri, $TemporaryDestination)
            Invoke-WebRequest -UseBasicParsing -Uri $RequestedUri -OutFile $TemporaryDestination
        }
    )
    if (Test-Path -LiteralPath $Destination) {
        throw "Download destination already exists: $Destination"
    }
    $parent = [System.IO.Path]::GetDirectoryName($Destination)
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $temporaryDestination = "$Destination.partial-$([guid]::NewGuid().ToString('N'))"
    try {
        & $DownloadAction $Uri $temporaryDestination
        if (-not (Test-Path -LiteralPath $temporaryDestination -PathType Leaf)) {
            throw "Public download did not create a file for $Uri."
        }
        if ((Get-Item -LiteralPath $temporaryDestination).Length -eq 0) {
            throw "Public download produced an empty file for $Uri."
        }
        Move-Item -LiteralPath $temporaryDestination -Destination $Destination
    }
    catch {
        throw "Public download failed for $Uri`: $($_.Exception.Message)"
    }
    finally {
        Remove-Item -LiteralPath $temporaryDestination -Force -ErrorAction SilentlyContinue
    }
}

function Install-CMakeArchive {
    param(
        [Parameter(Mandatory)] [string] $ArchivePath,
        [Parameter(Mandatory)] [string] $InstallRoot,
        [Parameter(Mandatory)] [string] $ExpectedSha256
    )
    Assert-LeafFile 'CMake archive' $ArchivePath
    $actualHash = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -cne $ExpectedSha256.ToLowerInvariant()) {
        throw "CMake archive SHA-256 mismatch: expected $ExpectedSha256, got $actualHash."
    }
    if (Test-Path -LiteralPath $InstallRoot) {
        throw "CMake install destination already exists: $InstallRoot"
    }

    $parent = [System.IO.Path]::GetDirectoryName($InstallRoot)
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $extractRoot = Join-Path $parent ("cmake-extract-{0}" -f [guid]::NewGuid().ToString('N'))
    try {
        Expand-Archive -LiteralPath $ArchivePath -DestinationPath $extractRoot
        $archiveRoot = Join-Path $extractRoot 'cmake-3.25.1-windows-x86_64'
        Assert-LeafFile 'CMake executable in archive' (Join-Path $archiveRoot 'bin/cmake.exe')
        Assert-LeafFile 'CTest executable in archive' (Join-Path $archiveRoot 'bin/ctest.exe')
        Move-Item -LiteralPath $archiveRoot -Destination $InstallRoot
    }
    finally {
        Remove-Item -LiteralPath $extractRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Get-NumericVersionDirectory {
    param([string] $Root)
    Assert-Directory 'Visual Studio redistributable root' $Root
    return @(Get-ChildItem -LiteralPath $Root -Directory | ForEach-Object {
        $parsed = $null
        if ([System.Version]::TryParse($_.Name, [ref] $parsed)) {
            [pscustomobject]@{ Directory = $_; Version = $parsed }
        }
    } | Sort-Object Version -Descending)
}

function Resolve-VCRuntimeDirectory {
    param(
        [Parameter(Mandatory)] [string] $RedistRoot,
        [version] $MinimumVersion = [version]$script:RequiredVCToolsVersion
    )
    foreach ($versionDirectory in Get-NumericVersionDirectory $RedistRoot) {
        if ($versionDirectory.Version -lt $MinimumVersion) { continue }
        $x64Root = Join-Path $versionDirectory.Directory.FullName 'x64'
        if (-not (Test-Path -LiteralPath $x64Root -PathType Container)) { continue }
        foreach ($candidate in @(Get-ChildItem -LiteralPath $x64Root -Directory -Filter 'Microsoft.VC*.CRT' |
            Sort-Object Name)) {
            $complete = $true
            foreach ($dll in $script:RequiredRuntimeDlls) {
                if (-not (Test-Path -LiteralPath (Join-Path $candidate.FullName $dll) -PathType Leaf)) {
                    $complete = $false
                    break
                }
            }
            if ($complete) { return $candidate.FullName }
        }
    }
    throw "No complete x64 Visual Studio app-local runtime at or newer than $MinimumVersion contains $($script:RequiredRuntimeDlls -join ', ')."
}

function Resolve-PublicWindowsToolchain {
    param(
        [Parameter(Mandatory)] [string] $VsRoot,
        [Parameter(Mandatory)] [string] $WindowsKitsRoot
    )
    Assert-Directory 'Visual Studio 2022 installation' $VsRoot
    $vcvarsall = Join-Path $VsRoot 'VC/Auxiliary/Build/vcvarsall.bat'
    $vcToolsRoot = Join-Path $VsRoot "VC/Tools/MSVC/$script:RequiredVCToolsVersion"
    $compiler = Join-Path $vcToolsRoot 'bin/Hostx64/x64/cl.exe'
    $dumpbin = Join-Path $vcToolsRoot 'bin/Hostx64/x64/dumpbin.exe'
    Assert-LeafFile 'Visual Studio vcvarsall.bat' $vcvarsall
    Assert-LeafFile "MSVC $script:RequiredVCToolsVersion compiler" $compiler
    Assert-LeafFile "MSVC $script:RequiredVCToolsVersion dumpbin" $dumpbin

    foreach ($requiredSdkPath in @(
        "bin/$script:RequiredWindowsSDKVersion/x64/rc.exe",
        "Include/$script:RequiredWindowsSDKVersion/um/Windows.h",
        "Lib/$script:RequiredWindowsSDKVersion/um/x64/kernel32.lib",
        "Lib/$script:RequiredWindowsSDKVersion/ucrt/x64/ucrt.lib"
    )) {
        Assert-LeafFile "Windows SDK $script:RequiredWindowsSDKVersion component" (Join-Path $WindowsKitsRoot $requiredSdkPath)
    }
    $runtimeDirectory = Resolve-VCRuntimeDirectory -RedistRoot (Join-Path $VsRoot 'VC/Redist/MSVC')

    return [pscustomobject]@{
        VsRoot = (Resolve-Path -LiteralPath $VsRoot).Path
        Vcvarsall = $vcvarsall
        VCToolsVersion = $script:RequiredVCToolsVersion
        WindowsSDKVersion = $script:RequiredWindowsSDKVersion
        Compiler = $compiler
        Dumpbin = $dumpbin
        RuntimeDirectory = $runtimeDirectory
    }
}

function Get-VisualStudioRoot {
    param([Parameter(Mandatory)] [string] $Path)
    Assert-LeafFile 'vswhere.exe' $Path
    $output = @(& $Path -latest -products * -version '[17.0,18.0)' `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) { throw "vswhere.exe failed with exit code $exitCode." }
    $paths = @($output | ForEach-Object { ([string]$_).Trim() } | Where-Object { $_ -ne '' })
    if ($paths.Count -ne 1) { throw "vswhere.exe must resolve exactly one Visual Studio 2022 installation; found $($paths.Count)." }
    Assert-SafeEnvironmentValue 'OVDNEXT_PUBLIC_VS_ROOT' $paths[0]
    return $paths[0]
}

function Write-GitHubEnvironment {
    param(
        [Parameter(Mandatory)] [string] $Path,
        [Parameter(Mandatory)] [System.Collections.IDictionary] $Values
    )
    if ([string]::IsNullOrWhiteSpace($Path)) { throw 'GITHUB_ENV path is missing.' }
    $lines = @()
    foreach ($name in $Values.Keys) {
        $value = [string]$Values[$name]
        Assert-SafeEnvironmentValue ([string]$name) $value
        $lines += "$name=$value"
    }
    Add-Content -LiteralPath $Path -Value $lines -Encoding UTF8
}

function Remove-WorkspaceDependencyPath {
    param([string] $DependencyRoot, [string] $Path)
    $rootFull = [System.IO.Path]::GetFullPath($DependencyRoot).TrimEnd('\', '/')
    $pathFull = [System.IO.Path]::GetFullPath($Path)
    if (-not $pathFull.StartsWith("$rootFull$([System.IO.Path]::DirectorySeparatorChar)", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove dependency path outside workspace dependency root: $pathFull"
    }
    Remove-Item -LiteralPath $pathFull -Recurse -Force -ErrorAction Stop
}

function Invoke-PublicWindowsSetup {
    param(
        [Parameter(Mandatory)] [string] $Root,
        [Parameter(Mandatory)] [AllowNull()] [AllowEmptyString()] [string] $EnvironmentPath,
        [Parameter(Mandatory)] [string] $Python,
        [Parameter(Mandatory)] [string] $Vswhere,
        [Parameter(Mandatory)] [string] $KitsRoot
    )
    if ([string]::IsNullOrWhiteSpace($EnvironmentPath)) { throw 'GITHUB_ENV path is missing.' }
    Assert-SafeEnvironmentValue 'GITHUB_ENV' $EnvironmentPath
    Assert-Directory 'SourceRoot' $Root
    $pythonLock = Join-Path $Root "ci/$script:PythonLockName"
    Assert-LeafFile 'Windows CPython 3.11 x64 dependency lock' $pythonLock
    Assert-WindowsPython311X64 -Command $Python
    $dependencyRoot = Join-Path $Root '_ci/deps/public-windows'
    $downloadRoot = Join-Path $dependencyRoot 'downloads'
    $cmakeArchive = Join-Path $downloadRoot $script:CMakeArchiveName
    $cmakeRoot = Join-Path $dependencyRoot 'cmake-3.25.1-windows-x86_64'
    $venvRoot = Join-Path $dependencyRoot 'venv'
    $venvPython = Join-Path $venvRoot 'Scripts/python.exe'
    $ninjaRoot = Join-Path $venvRoot 'Scripts'
    $ninja = Join-Path $ninjaRoot 'ninja.exe'

    New-Item -ItemType Directory -Force -Path $dependencyRoot, $downloadRoot | Out-Null

    if (Test-Path -LiteralPath $cmakeRoot) {
        $cachedCMake = Join-Path $cmakeRoot 'bin/cmake.exe'
        if (-not (Test-Path -LiteralPath $cachedCMake -PathType Leaf)) {
            Remove-WorkspaceDependencyPath $dependencyRoot $cmakeRoot
        }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $cmakeRoot 'bin/cmake.exe') -PathType Leaf)) {
        if (Test-Path -LiteralPath $cmakeArchive -PathType Leaf) {
            $cachedHash = (Get-FileHash -LiteralPath $cmakeArchive -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($cachedHash -cne $script:CMakeArchiveSha256) {
                Remove-Item -LiteralPath $cmakeArchive -Force
            }
        }
        if (-not (Test-Path -LiteralPath $cmakeArchive -PathType Leaf)) {
            Save-PublicFile -Uri $script:CMakeArchiveUrl -Destination $cmakeArchive
        }
        Install-CMakeArchive -ArchivePath $cmakeArchive -InstallRoot $cmakeRoot -ExpectedSha256 $script:CMakeArchiveSha256
    }
    Get-CheckedNativeVersion -Command (Join-Path $cmakeRoot 'bin/cmake.exe') -Arguments @('--version') `
        -Label 'CMake' -ExpectedPattern '^cmake version 3\.25\.1$' `
        -ExpectedDescription "exactly version $script:CMakeVersion" | Out-Null
    Assert-LeafFile 'CTest 3.25.1 executable' (Join-Path $cmakeRoot 'bin/ctest.exe')

    if ((Test-Path -LiteralPath $venvRoot) -and -not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        Remove-WorkspaceDependencyPath $dependencyRoot $venvRoot
    }
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        Invoke-NativeChecked -Command $Python -Arguments @('-m', 'venv', $venvRoot) -Label 'workspace Python venv creation'
    }
    Assert-LeafFile 'workspace venv Python' $venvPython
    Assert-WindowsPython311X64 -Command $venvPython
    Invoke-NativeChecked -Command $venvPython -Arguments @(
        '-m', 'pip', '--isolated', 'install', '--disable-pip-version-check', '--no-input',
        '--index-url', 'https://pypi.org/simple', '--only-binary=:all:', '--require-hashes',
        '-r', $pythonLock
    ) -Label 'public Python dependency installation'
    Invoke-NativeChecked -Command $venvPython -Arguments @(
        '-m', 'pip', '--isolated', 'check', '--disable-pip-version-check'
    ) -Label 'public Python dependency validation'
    Assert-LeafFile 'Ninja executable' $ninja
    Get-CheckedNativeVersion -Command $ninja -Arguments @('--version') -Label 'Ninja' `
        -ExpectedPattern '^1\.10\.2(?:\..+)?$' -ExpectedDescription 'the pinned 1.10.2 family' | Out-Null

    $vsRoot = Get-VisualStudioRoot -Path $Vswhere
    $toolchain = Resolve-PublicWindowsToolchain -VsRoot $vsRoot -WindowsKitsRoot $KitsRoot
    $values = [ordered]@{
        OVDNEXT_PUBLIC_WINDOWS_TOOLCHAIN_READY = '1'
        OVDNEXT_PUBLIC_VS_ROOT = $toolchain.VsRoot
        OVDNEXT_PUBLIC_VCVARSALL = $toolchain.Vcvarsall
        OVDNEXT_PUBLIC_CL = $toolchain.Compiler
        OVDNEXT_PUBLIC_DUMPBIN = $toolchain.Dumpbin
        OVDNEXT_PUBLIC_CMAKE_ROOT = $cmakeRoot
        OVDNEXT_PUBLIC_NINJA_ROOT = $ninjaRoot
        OVDNEXT_PUBLIC_NINJA = $ninja
        OVDNEXT_PUBLIC_PYTHON = $venvPython
        OVDNEXT_PUBLIC_VC_RUNTIME_DIR = $toolchain.RuntimeDirectory
    }
    Write-GitHubEnvironment -Path $EnvironmentPath -Values $values
    Write-Host "Provisioned public GitHub Windows dependencies under: $dependencyRoot"
}

if ($MyInvocation.InvocationName -ne '.') {
    try {
        Invoke-PublicWindowsSetup -Root $SourceRoot -EnvironmentPath $GitHubEnvPath `
            -Python $PythonExecutable -Vswhere $VswherePath -KitsRoot $WindowsKitsRoot
    }
    catch {
        Write-Error $_
        exit 1
    }
}
