# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$setupScript = Join-Path $repoRoot 'ci/setup_github_windows.ps1'
$wrapperScript = Join-Path $repoRoot 'ci/run_windows_ci.bat'
$temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("ovdnext-public-windows-{0}" -f [guid]::NewGuid())
$physicsRef = 'a12f79ce585ffb79e5ffa4848cb45efc3de2f632'

function Assert-True {
    param([bool] $Condition, [string] $Message)
    if (-not $Condition) { throw "FAIL: $Message" }
}

function Assert-Throws {
    param([scriptblock] $Action, [string] $Pattern, [string] $Message)
    try {
        & $Action
    }
    catch {
        Assert-True ($_.Exception.Message -match $Pattern) "$Message; got: $($_.Exception.Message)"
        return
    }
    throw "FAIL: $Message; action unexpectedly succeeded"
}

function New-File {
    param([string] $Path, [string] $Contents = 'fixture')
    New-Item -ItemType Directory -Force -Path ([System.IO.Path]::GetDirectoryName($Path)) | Out-Null
    Set-Content -LiteralPath $Path -Value $Contents
}

function New-CompleteRuntime {
    param([string] $Path)
    foreach ($name in @('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll')) {
        New-File (Join-Path $Path $name)
    }
}

function New-FakeGit {
    param([string] $Path)
    $typeName = 'FakeGit' + [guid]::NewGuid().ToString('N')
    $source = @"
using System;
public static class $typeName {
    public static int Main(string[] args) {
        if (args.Length > 2 && args[2] == "show") Console.WriteLine("1700000000");
        else Console.WriteLine("$physicsRef");
        return 0;
    }
}
"@
    Add-Type -TypeDefinition $source -OutputAssembly $Path -OutputType ConsoleApplication
}

function New-FakePowerShell {
    param([string] $Path)
    $typeName = 'FakePowerShell' + [guid]::NewGuid().ToString('N')
    $source = @"
using System;
using System.IO;
public static class $typeName {
    public static int Main(string[] args) {
        File.AppendAllText(Environment.GetEnvironmentVariable("OVDNEXT_TEST_LOG"),
            String.Join(" ", args) + Environment.NewLine);
        return 0;
    }
}
"@
    Add-Type -TypeDefinition $source -OutputAssembly $Path -OutputType ConsoleApplication
}

function New-FakeVersionTool {
    param([string] $Path, [string] $Version, [int] $ExitCode = 0)
    $typeName = 'FakeVersionTool' + [guid]::NewGuid().ToString('N')
    $source = @"
using System;
using System.Threading;
public static class $typeName {
    public static int Main(string[] args) {
        Console.WriteLine("$Version");
        Console.Out.Flush();
        Thread.Sleep(100);
        Console.WriteLine("details after version");
        return $ExitCode;
    }
}
"@
    Add-Type -TypeDefinition $source -OutputAssembly $Path -OutputType ConsoleApplication
}

function New-FakeOutputTool {
    param([string] $Path, [string] $Output, [int] $ExitCode = 0)
    $typeName = 'FakeOutputTool' + [guid]::NewGuid().ToString('N')
    $source = @"
using System;
public static class $typeName {
    public static int Main(string[] args) {
        Console.WriteLine(@"$Output");
        return $ExitCode;
    }
}
"@
    Add-Type -TypeDefinition $source -OutputAssembly $Path -OutputType ConsoleApplication
}

function New-FakeDependencyPython {
    param([string] $Path)
    $typeName = 'FakeDependencyPython' + [guid]::NewGuid().ToString('N')
    $source = @"
using System;
using System.IO;
public static class $typeName {
    public static int Main(string[] args) {
        string log = Environment.GetEnvironmentVariable("OVDNEXT_TEST_LOG");
        if (!String.IsNullOrEmpty(log)) {
            File.AppendAllText(log, String.Join(" ", args) + Environment.NewLine);
        }
        if (args.Length > 0 && args[0] == "--version") {
            Console.WriteLine("Python 3.11.9");
        } else if (args.Length > 0 && args[0] == "-c") {
            Console.WriteLine("3.11|AMD64|64");
        }
        return 0;
    }
}
"@
    Add-Type -TypeDefinition $source -OutputAssembly $Path -OutputType ConsoleApplication
}

function New-FakeCMake {
    param([string] $Path, [int] $ExitCode = 1, [string] $Output = 'PVD3_REQUIRE_OVD_SUPPORT: missing roots')
    $typeName = 'FakeCMake' + [guid]::NewGuid().ToString('N')
    $source = @"
using System;
using System.IO;
public static class $typeName {
    public static int Main(string[] args) {
        File.WriteAllLines(typeof($typeName).Assembly.Location + ".args", args);
        Console.Error.WriteLine("$Output");
        return $ExitCode;
    }
}
"@
    Add-Type -TypeDefinition $source -OutputAssembly $Path -OutputType ConsoleApplication
}

function New-ToolchainFixture {
    param([string] $Root)
    $vsRoot = Join-Path $Root 'VisualStudio'
    $kitsRoot = Join-Path $Root 'WindowsKits10'
    New-File (Join-Path $vsRoot 'VC/Auxiliary/Build/vcvarsall.bat') '@exit /b 0'
    New-File (Join-Path $vsRoot 'VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64/cl.exe')
    New-File (Join-Path $vsRoot 'VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64/dumpbin.exe')
    New-File (Join-Path $kitsRoot 'bin/10.0.17763.0/x64/rc.exe')
    New-File (Join-Path $kitsRoot 'Include/10.0.17763.0/um/Windows.h')
    New-File (Join-Path $kitsRoot 'Lib/10.0.17763.0/um/x64/kernel32.lib')
    New-File (Join-Path $kitsRoot 'Lib/10.0.17763.0/ucrt/x64/ucrt.lib')

    # Lexical order would put 14.40 after 14.100. The newest folder is also
    # deliberately incomplete; selection must choose the newest complete set.
    New-CompleteRuntime (Join-Path $vsRoot 'VC/Redist/MSVC/14.40/x64/Microsoft.VC143.CRT')
    New-CompleteRuntime (Join-Path $vsRoot 'VC/Redist/MSVC/14.100/x64/Microsoft.VC143.CRT')
    New-File (Join-Path $vsRoot 'VC/Redist/MSVC/14.999/x64/Microsoft.VC143.CRT/msvcp140.dll')
    return [pscustomobject]@{ VsRoot = $vsRoot; KitsRoot = $kitsRoot }
}

function Invoke-DefaultSetupEntrypoint {
    param([string] $Root)
    $entrypoint = Join-Path $Root 'source/ci/setup_github_windows.ps1'
    New-Item -ItemType Directory -Force -Path ([System.IO.Path]::GetDirectoryName($entrypoint)) | Out-Null
    Copy-Item -LiteralPath $setupScript -Destination $entrypoint

    $start = [System.Diagnostics.ProcessStartInfo]::new()
    $start.FileName = Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0/powershell.exe'
    $start.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$entrypoint`""
    $start.UseShellExecute = $false
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $start.CreateNoWindow = $true
    $start.EnvironmentVariables.Remove('GITHUB_ENV')
    $process = [System.Diagnostics.Process]::Start($start)
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()
    return [pscustomobject]@{
        ExitCode = $process.ExitCode
        Output = $stdout + $stderr
        DependencyRoot = Join-Path $Root 'source/_ci/deps/public-windows'
    }
}

function Invoke-WrapperFixture {
    param(
        [string] $Root,
        [ValidateSet('verify', 'package')] [string] $Mode,
        [hashtable] $Overrides = @{}
    )
    $sourceRoot = Join-Path $Root 'source'
    $ciRoot = Join-Path $sourceRoot 'ci'
    $toolsRoot = Join-Path $Root 'tools'
    $logPath = Join-Path $Root 'powershell.log'
    New-Item -ItemType Directory -Force -Path $ciRoot, $toolsRoot | Out-Null
    Copy-Item -LiteralPath $wrapperScript -Destination (Join-Path $ciRoot 'run_windows_ci.bat')

    New-FakePowerShell (Join-Path $toolsRoot 'powershell.exe')
    New-FakeGit (Join-Path $toolsRoot 'git.exe')

    $physicsRoot = Join-Path $sourceRoot '_ci/deps/physics'
    New-Item -ItemType Directory -Force -Path (Join-Path $physicsRoot '.git') | Out-Null
    New-File (Join-Path $physicsRoot 'physx/pvdruntime/compiler/cmake/PVDRuntime.cmake')
    New-File (Join-Path $physicsRoot 'physx/pvddom/CMakeLists.txt')
    New-File (Join-Path $physicsRoot 'physx/buildtools/packman/packman.cmd') @'
@echo PACKMAN>>"%OVDNEXT_TEST_LOG%"
exit /b 91
'@

    $vsRoot = Join-Path $Root 'vs'
    New-File (Join-Path $vsRoot 'VC/Auxiliary/Build/vcvarsall.bat') @'
@echo off
set "VCToolsVersion=14.29.30133"
set "WindowsSDKVersion=10.0.17763.0\"
set "VCToolsInstallDir=%~dp0..\..\Tools\MSVC\14.29.30133\"
exit /b 0
'@
    $cmakeRoot = Join-Path $Root 'cmake'
    $ninjaRoot = Join-Path $Root 'ninja'
    $runtimeRoot = Join-Path $Root 'runtime'
    New-File (Join-Path $cmakeRoot 'bin/cmake.exe')
    New-File (Join-Path $cmakeRoot 'bin/ctest.exe')
    New-File (Join-Path $ninjaRoot 'ninja.exe')
    New-File (Join-Path $ninjaRoot 'python.exe')
    New-CompleteRuntime $runtimeRoot

    $environment = @{
        GITHUB_ACTIONS = 'true'
        GITHUB_SHA = ('a' * 40)
        SystemRoot = $env:SystemRoot
        ProgramFiles = $env:ProgramFiles
        'ProgramFiles(x86)' = ${env:ProgramFiles(x86)}
        TEMP = $Root
        TMP = $Root
        PATH = "$toolsRoot;$env:SystemRoot\System32"
        OVDNEXT_TEST_LOG = $logPath
        OVDNEXT_PUBLIC_WINDOWS_TOOLCHAIN_READY = '1'
        OVDNEXT_PUBLIC_VS_ROOT = $vsRoot
        OVDNEXT_PUBLIC_CMAKE_ROOT = $cmakeRoot
        OVDNEXT_PUBLIC_NINJA_ROOT = $ninjaRoot
        OVDNEXT_PUBLIC_PYTHON = (Join-Path $ninjaRoot 'python.exe')
        OVDNEXT_PUBLIC_VC_RUNTIME_DIR = $runtimeRoot
    }
    foreach ($name in $Overrides.Keys) {
        if ($null -eq $Overrides[$name]) { $environment.Remove($name) }
        else { $environment[$name] = [string]$Overrides[$name] }
    }

    $start = [System.Diagnostics.ProcessStartInfo]::new()
    $start.FileName = (Join-Path $env:SystemRoot 'System32/cmd.exe')
    $start.Arguments = "/d /c `"$ciRoot\run_windows_ci.bat`" $Mode"
    $start.UseShellExecute = $false
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $start.CreateNoWindow = $true
    $start.EnvironmentVariables.Clear()
    foreach ($entry in $environment.GetEnumerator()) {
        $start.EnvironmentVariables[$entry.Key] = $entry.Value
    }
    $process = [System.Diagnostics.Process]::Start($start)
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()
    $log = if (Test-Path -LiteralPath $logPath) { Get-Content -Raw -LiteralPath $logPath } else { '' }
    return [pscustomobject]@{
        ExitCode = $process.ExitCode
        Output = $stdout + $stderr
        Log = $log
    }
}

try {
    New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
    Assert-True (Test-Path -LiteralPath $setupScript -PathType Leaf) 'public Windows setup script must exist'
    Assert-True (Test-Path -LiteralPath $wrapperScript -PathType Leaf) 'Windows wrapper must exist'

    $cmakeContract = Join-Path $repoRoot 'ci/tests/Test-CMakeContracts.ps1'
    $fakeCMake = Join-Path $temporaryRoot 'cmake.exe'
    $successfulCMake = Join-Path $temporaryRoot 'successful-cmake.exe'
    $unrelatedFailureCMake = Join-Path $temporaryRoot 'unrelated-failure-cmake.exe'
    New-FakeCMake $fakeCMake
    New-FakeCMake $successfulCMake -ExitCode 0
    New-FakeCMake $unrelatedFailureCMake -ExitCode 23 -Output 'unrelated configure failure'
    foreach ($case in @(
        @{ Name = 'explicit Ninja'; Parameters = @{ Generator = 'Ninja' }; Expected = @('-G', 'Ninja') },
        @{ Name = 'default Visual Studio'; Parameters = @{}; Expected = @('-G', 'Visual Studio 16 2019', '-A', 'x64') }
    )) {
        $parameters = $case.Parameters
        $sourceFixture = Join-Path $temporaryRoot 'source with spaces'
        & $cmakeContract -SourceRoot $sourceFixture -CMakeExecutable $fakeCMake @parameters
        $nativeArguments = @(Get-Content -LiteralPath "$fakeCMake.args")
        Assert-True ($nativeArguments.Count -ge 4) "$($case.Name) must invoke the native CMake process"
        Assert-True ($nativeArguments[3] -match 'ovdnext-cmake-[0-9a-f-]+$') "$($case.Name) must use a fresh probe build directory"
        $expectedArguments = @('-S', $sourceFixture, '-B', $nativeArguments[3]) + $case.Expected + @(
            '-DCMAKE_DISABLE_FIND_PACKAGE_Qt5=ON', '-DPVD3_BUILD_TESTS=OFF', '-DPVD3_REQUIRE_OVD_SUPPORT=ON'
        )
        Assert-True (($nativeArguments -join "`n") -ceq ($expectedArguments -join "`n")) `
            "$($case.Name) must preserve configure arguments and omit Physics/OVD roots; got: $($nativeArguments -join ' ')"
        Assert-Throws {
            & $cmakeContract -SourceRoot $sourceFixture -CMakeExecutable $successfulCMake @parameters
        } 'unexpectedly succeeded' "$($case.Name) must reject native success even with the fail-closed marker"
        Assert-Throws {
            & $cmakeContract -SourceRoot $sourceFixture -CMakeExecutable $unrelatedFailureCMake @parameters
        } 'did not report the fail-closed option' "$($case.Name) must reject native failure without the expected marker"
    }

    $defaultEntrypoint = Invoke-DefaultSetupEntrypoint (Join-Path $temporaryRoot 'default-entrypoint')
    Assert-True ($defaultEntrypoint.ExitCode -ne 0) 'default setup entrypoint must reject missing GITHUB_ENV'
    Assert-True ($defaultEntrypoint.Output -match 'GITHUB_ENV.*(?:missing|empty)') `
        "default setup entrypoint must resolve SourceRoot before rejecting missing GITHUB_ENV; got: $($defaultEntrypoint.Output)"
    Assert-True ($defaultEntrypoint.Output -notmatch 'PSScriptRoot|Join-Path') `
        "default setup entrypoint must not fail during parameter default evaluation; got: $($defaultEntrypoint.Output)"
    Assert-True (-not (Test-Path -LiteralPath $defaultEntrypoint.DependencyRoot)) `
        'missing GITHUB_ENV must fail before dependency provisioning starts'

    . $setupScript

    $realPython = $env:OVDNEXT_PYTHON
    if ([string]::IsNullOrWhiteSpace($realPython)) {
        $realPython = (Get-Command python -CommandType Application -ErrorAction Stop |
            Select-Object -First 1).Source
    }
    Assert-True (Test-Path -LiteralPath $realPython -PathType Leaf) `
        "real Python executable for native argument validation is missing: $realPython"
    try {
        Assert-WindowsPython311X64 -Command $realPython
    }
    catch {
        throw "FAIL: real CPython 3.11 x64 validation must survive native argument marshalling; got: $($_.Exception.Message)"
    }

    $dependencyLock = Join-Path $repoRoot 'ci/windows-cp311-x64.lock'
    Assert-True (Test-Path -LiteralPath $dependencyLock -PathType Leaf) `
        'the Windows CPython 3.11 x64 dependency lock must exist'
    $lockedRequirements = @(Get-Content -LiteralPath $dependencyLock | Where-Object { $_ -ne '' })
    Assert-True ($lockedRequirements.Count -eq 1) 'the Windows dependency lock must contain only the pinned Ninja wheel'
    foreach ($requirement in $lockedRequirements) {
        Assert-True ($requirement -cmatch '^[A-Za-z0-9_.-]+==[^ ]+ --hash=sha256:[0-9a-f]{64}$') `
            "locked requirement must contain one exact version and SHA-256: $requirement"
    }
    Assert-True (@($lockedRequirements | Where-Object { $_ -like 'ninja==1.10.2.4 *' }).Count -eq 1) `
        'the dependency lock must retain Ninja package 1.10.2.4'

    $noQtSource = Join-Path $temporaryRoot 'no-qt/source'
    $noQtCi = Join-Path $noQtSource 'ci'
    $noQtDependencyRoot = Join-Path $noQtSource '_ci/deps/public-windows'
    $noQtCMakeRoot = Join-Path $noQtDependencyRoot 'cmake-3.25.1-windows-x86_64'
    $noQtVenvRoot = Join-Path $noQtDependencyRoot 'venv'
    New-Item -ItemType Directory -Force -Path `
        $noQtCi, `
        (Join-Path $noQtCMakeRoot 'bin'), `
        (Join-Path $noQtVenvRoot 'Scripts') | Out-Null
    Copy-Item -LiteralPath $dependencyLock -Destination (Join-Path $noQtCi 'windows-cp311-x64.lock')
    New-FakeVersionTool (Join-Path $noQtCMakeRoot 'bin/cmake.exe') 'cmake version 3.25.1'
    New-File (Join-Path $noQtCMakeRoot 'bin/ctest.exe')
    New-FakeDependencyPython (Join-Path $noQtVenvRoot 'Scripts/python.exe')
    New-FakeVersionTool (Join-Path $noQtVenvRoot 'Scripts/ninja.exe') '1.10.2.git.kitware.jobserver-1'
    $noQtToolchain = New-ToolchainFixture (Join-Path $temporaryRoot 'no-qt/toolchain')
    $noQtVswhere = Join-Path $temporaryRoot 'no-qt/vswhere.exe'
    New-Item -ItemType Directory -Force -Path ([System.IO.Path]::GetDirectoryName($noQtVswhere)) | Out-Null
    New-FakeOutputTool $noQtVswhere $noQtToolchain.VsRoot
    $noQtEnv = Join-Path $temporaryRoot 'no-qt/github-env.txt'
    $noQtLog = Join-Path $temporaryRoot 'no-qt/python.log'
    $savedTestLog = $env:OVDNEXT_TEST_LOG
    try {
        $env:OVDNEXT_TEST_LOG = $noQtLog
        Invoke-PublicWindowsSetup `
            -Root $noQtSource `
            -EnvironmentPath $noQtEnv `
            -Python (Join-Path $noQtVenvRoot 'Scripts/python.exe') `
            -Vswhere $noQtVswhere `
            -KitsRoot $noQtToolchain.KitsRoot
    }
    finally {
        $env:OVDNEXT_TEST_LOG = $savedTestLog
    }
    $noQtEnvironment = Get-Content -Raw -LiteralPath $noQtEnv
    Assert-True ($noQtEnvironment -match '(?m)^OVDNEXT_PUBLIC_WINDOWS_TOOLCHAIN_READY=1\r?$') `
        'no-Qt setup must emit a toolchain-specific readiness signal'
    Assert-True ($noQtEnvironment -notmatch '(?m)^OVDNEXT_PUBLIC_WINDOWS_READY=') `
        'no-Qt setup must not emit the root/package readiness signal'
    Assert-True ($noQtEnvironment -notmatch '(?m)^OVDNEXT_PUBLIC_QT_ROOT=') `
        'no-Qt setup must not emit a Qt root'
    Assert-True ($noQtEnvironment -match '(?m)^OVDNEXT_PUBLIC_PYTHON=.+python\.exe\r?$') `
        'no-Qt setup must emit the locked venv Python path'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $noQtDependencyRoot 'qt'))) `
        'no-Qt setup must not create a Qt install root'
    $noQtPythonLog = Get-Content -Raw -LiteralPath $noQtLog
    Assert-True ($noQtPythonLog -match '(?s)-m pip.*--only-binary=:all:.*--require-hashes.*windows-cp311-x64\.lock') `
        'no-Qt setup must install the exact binary/hash-required dependency lock'

    $validVersionTool = Join-Path $temporaryRoot 'valid-version.exe'
    New-FakeVersionTool $validVersionTool '1.10.2.git.kitware.jobserver-1'
    $capturedVersion = Get-CheckedNativeVersion -Command $validVersionTool -Arguments @('--version') `
        -Label 'Ninja' -ExpectedPattern '^1\.10\.2(?:\..+)?$' -ExpectedDescription 'the pinned 1.10.2 family'
    Assert-True ($capturedVersion -ceq '1.10.2.git.kitware.jobserver-1') `
        'checked native version capture must consume the complete successful process before selecting its first line'

    $failedVersionTool = Join-Path $temporaryRoot 'failed-version.exe'
    New-FakeVersionTool $failedVersionTool '1.10.2.git.kitware.jobserver-1' 23
    Assert-Throws {
        Get-CheckedNativeVersion -Command $failedVersionTool -Arguments @('--version') `
            -Label 'Ninja' -ExpectedPattern '^1\.10\.2(?:\..+)?$' -ExpectedDescription 'the pinned 1.10.2 family'
    } 'Ninja.*23|23.*Ninja' 'genuinely nonzero version command must fail even when its version text is valid'

    $wrongVersionTool = Join-Path $temporaryRoot 'wrong-version.exe'
    New-FakeVersionTool $wrongVersionTool '9.9.9'
    Assert-Throws {
        Get-CheckedNativeVersion -Command $wrongVersionTool -Arguments @('--version') `
            -Label 'Ninja' -ExpectedPattern '^1\.10\.2(?:\..+)?$' -ExpectedDescription 'the pinned 1.10.2 family'
    } 'Ninja.*1\.10\.2.*9\.9\.9|9\.9\.9.*1\.10\.2' 'wrong successful version output must fail'

    $toolchainFixture = New-ToolchainFixture (Join-Path $temporaryRoot 'toolchain')
    $toolchain = Resolve-PublicWindowsToolchain -VsRoot $toolchainFixture.VsRoot -WindowsKitsRoot $toolchainFixture.KitsRoot
    Assert-True ($toolchain.VCToolsVersion -ceq '14.29.30133') 'toolchain resolver must preserve exact MSVC 14.29.30133'
    Assert-True ($toolchain.WindowsSDKVersion -ceq '10.0.17763.0') 'toolchain resolver must preserve exact SDK 10.0.17763.0'
    Assert-True ($toolchain.RuntimeDirectory -ceq (Join-Path $toolchainFixture.VsRoot 'VC/Redist/MSVC/14.100/x64/Microsoft.VC143.CRT')) `
        'runtime resolver must use numeric version order and skip incomplete folders'

    $tooOldFixture = New-ToolchainFixture (Join-Path $temporaryRoot 'too-old-runtime')
    Remove-Item -LiteralPath (Join-Path $tooOldFixture.VsRoot 'VC/Redist/MSVC') -Recurse -Force
    New-CompleteRuntime (Join-Path $tooOldFixture.VsRoot 'VC/Redist/MSVC/14.20/x64/Microsoft.VC142.CRT')
    Assert-Throws {
        Resolve-PublicWindowsToolchain -VsRoot $tooOldFixture.VsRoot -WindowsKitsRoot $tooOldFixture.KitsRoot
    } 'runtime|14\.29\.30133|newer' 'runtime older than the selected compiler must fail'

    $compilerPath = Join-Path $toolchainFixture.VsRoot 'VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64/cl.exe'
    Remove-Item -LiteralPath $compilerPath
    Assert-Throws {
        Resolve-PublicWindowsToolchain -VsRoot $toolchainFixture.VsRoot -WindowsKitsRoot $toolchainFixture.KitsRoot
    } 'compiler|cl\.exe|14\.29\.30133' 'missing exact compiler must fail'
    New-File $compilerPath

    $sdkLibrary = Join-Path $toolchainFixture.KitsRoot 'Lib/10.0.17763.0/um/x64/kernel32.lib'
    Remove-Item -LiteralPath $sdkLibrary
    Assert-Throws {
        Resolve-PublicWindowsToolchain -VsRoot $toolchainFixture.VsRoot -WindowsKitsRoot $toolchainFixture.KitsRoot
    } 'SDK|10\.0\.17763\.0|kernel32' 'missing exact Windows SDK must fail'

    Assert-Throws {
        Invoke-NativeChecked -Command (Join-Path $env:SystemRoot 'System32/cmd.exe') -Arguments @('/d', '/c', 'exit 23') -Label 'failing child'
    } '23|failing child' 'native dependency command failure must propagate'

    $downloadDestination = Join-Path $temporaryRoot 'download/missing.zip'
    Assert-Throws {
        Save-PublicFile -Uri 'https://example.invalid/archive.zip' -Destination $downloadDestination -DownloadAction {
            param($Uri, $Destination)
            throw 'fixture download failure'
        }
    } 'download|fixture' 'public download failure must propagate'
    Assert-True (-not (Test-Path -LiteralPath $downloadDestination)) 'failed download must not leave a destination file'

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archiveSource = Join-Path $temporaryRoot 'archive-source'
    $archivePath = Join-Path $temporaryRoot 'bad-cmake.zip'
    New-File (Join-Path $archiveSource 'cmake-3.25.1-windows-x86_64/bin/cmake.exe')
    [System.IO.Compression.ZipFile]::CreateFromDirectory($archiveSource, $archivePath)
    $installRoot = Join-Path $temporaryRoot 'bad-cmake-install'
    Assert-Throws {
        Install-CMakeArchive -ArchivePath $archivePath -InstallRoot $installRoot -ExpectedSha256 ('0' * 64)
    } 'SHA-256|hash' 'bad CMake archive hash must fail before extraction'
    Assert-True (-not (Test-Path -LiteralPath $installRoot)) 'bad CMake archive must not be extracted'

    $envFile = Join-Path $temporaryRoot 'github-env.txt'
    Assert-Throws {
        Write-GitHubEnvironment -Path $envFile -Values ([ordered]@{ SAFE = "bad`nvalue" })
    } 'newline|SAFE' 'GitHub environment values containing newlines must fail'

    $missingSetup = Invoke-WrapperFixture (Join-Path $temporaryRoot 'missing-setup') verify @{
        OVDNEXT_PUBLIC_WINDOWS_TOOLCHAIN_READY = $null
    }
    Assert-True ($missingSetup.ExitCode -ne 0) 'GitHub wrapper must fail when setup readiness is missing'
    Assert-True ($missingSetup.Output -match 'OVDNEXT_PUBLIC_WINDOWS_TOOLCHAIN_READY|public Windows setup') `
        "missing setup failure must be explicit; got: $($missingSetup.Output)"
    Assert-True ($missingSetup.Log -notmatch 'PACKMAN') 'missing GitHub setup must not fall back to Packman'

    $verify = Invoke-WrapperFixture (Join-Path $temporaryRoot 'verify-dispatch') verify
    Assert-True ($verify.ExitCode -eq 0) "fixture verify dispatch must pass; got: $($verify.Output)"
    Assert-True ($verify.Log -match 'Test-CMakeContracts\.ps1') 'verify mode must dispatch CMake contracts'
    Assert-True ($verify.Log -match 'Test-CMakeContracts\.ps1[^\r\n]* -Generator Ninja(?:\s|$)') `
        'verify mode must explicitly pass the selected Ninja generator to CMake contracts'
    Assert-True ($verify.Log -match 'build_windows\.ps1') 'verify mode must dispatch the build driver'
    Assert-True ($verify.Log -notmatch 'package_windows\.ps1|PACKMAN') 'verify mode must not package or invoke Packman'

    Write-Host 'PASS: public GitHub Windows dependency contracts'
}
finally {
    if (Test-Path -LiteralPath $temporaryRoot) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
}
