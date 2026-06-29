param(
    [string]$OVRTX_DIR = "",
    [switch]$SkipOvphysx,
    [switch]$SkipFmuBuild,
    [switch]$InstallCudaPython
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$AppVenv = Join-Path $ScriptDir ".venv"
$UsdVenv = Join-Path $ScriptDir ".usd_venv"
$OvrtxPython = Join-Path $RepoRoot "third-party\ovrtx\python"
$OvrtxWheelVersion = "0.3.0.312915"
$OvphysxVersion = "0.4.9"

function Get-InstalledOvrtxBin {
    param([string]$PythonExe)

    $code = @'
from pathlib import Path
import ovrtx

bin_dir = Path(ovrtx.__file__).resolve().parent.joinpath('bin')
print(bin_dir)
'@
    $bin = & $PythonExe -c $code
    if ($LASTEXITCODE -ne 0) {
        throw "Could not locate installed ovrtx package."
    }
    $dll = Join-Path $bin "ovrtx-dynamic.dll"
    if (-not (Test-Path -LiteralPath $dll)) {
        throw "Installed ovrtx package does not contain ovrtx-dynamic.dll at $dll"
    }
    return (Resolve-Path $bin).Path
}

function Find-VcVars64 {
    $vswhere = $null
    if (${env:ProgramFiles(x86)}) {
        $candidate = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
        if (Test-Path -LiteralPath $candidate) {
            $vswhere = $candidate
        }
    }

    if ($vswhere) {
        $installRoots = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        foreach ($root in $installRoots) {
            if (-not $root) {
                continue
            }
            $candidate = Join-Path $root "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path -LiteralPath $candidate) {
                return (Resolve-Path $candidate).Path
            }
        }
    }

    $roots = @()
    if ($env:ProgramFiles) {
        $roots += Join-Path $env:ProgramFiles "Microsoft Visual Studio\2022"
    }
    if (${env:ProgramFiles(x86)}) {
        $roots += Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\2022"
    }

    $editions = @("BuildTools", "Community", "Professional", "Enterprise")
    foreach ($root in $roots) {
        foreach ($edition in $editions) {
            $candidate = Join-Path $root "$edition\VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path -LiteralPath $candidate) {
                return (Resolve-Path $candidate).Path
            }
        }
    }

    foreach ($root in $roots) {
        if (Test-Path -LiteralPath $root) {
            $candidate = Get-ChildItem -LiteralPath $root -Recurse -Filter vcvars64.bat -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if ($candidate) {
                return $candidate.FullName
            }
        }
    }

    return $null
}

function Invoke-FmuBuild {
    param([string]$PythonExe)

    if ($SkipFmuBuild) {
        Write-Host "Skipping FMU/SSP build because -SkipFmuBuild was specified."
        return
    }

    $BuildScript = Join-Path $ScriptDir "build_fmu.py"
    Write-Host "Building demo FMUs and SSPs..."

    if ($env:CXX) {
        Write-Host "Using compiler from CXX: $env:CXX"
        & $PythonExe $BuildScript
    } else {
        $VcVars = Find-VcVars64
        if (-not $VcVars) {
            throw @"
Cannot build FMUs because setup.ps1 could not find vcvars64.bat.

Install Visual Studio 2022 Build Tools with:
  - Desktop development with C++
  - MSVC v143 x64/x86 build tools

Then rerun from a normal PowerShell:
  powershell -ExecutionPolicy Bypass -File apps\ov-fmi\setup.ps1

Do not rely on the generic Visual Studio "Developer Command Prompt" button for
this setup path; it may not target x64. If you intentionally want to use a
custom compiler, set CXX to an x64 compiler command. To skip FMU compilation,
rerun setup.ps1 with -SkipFmuBuild.
"@
        }

        Write-Host "Using Visual Studio environment: $VcVars"
        $cmd = "call `"$VcVars`" && `"$PythonExe`" `"$BuildScript`""
        & cmd.exe /d /s /c $cmd
    }

    if ($LASTEXITCODE -ne 0) {
        throw "FMU/SSP build failed."
    }
}

if (-not (Test-Path -LiteralPath $AppVenv)) {
    python -m venv $AppVenv
}
if (-not (Test-Path -LiteralPath $UsdVenv)) {
    python -m venv $UsdVenv
}

$AppPython = Join-Path $AppVenv "Scripts\python.exe"
$UsdPython = Join-Path $UsdVenv "Scripts\python.exe"

& $AppPython -m pip install --upgrade pip
if ($OVRTX_DIR) {
    $OvrtxBin = Join-Path $OVRTX_DIR "bin"
    $OvrtxDll = Join-Path $OvrtxBin "ovrtx-dynamic.dll"
    if (-not (Test-Path -LiteralPath $OvrtxDll)) {
        throw "-OVRTX_DIR must point to an extracted ovrtx package root containing bin\ovrtx-dynamic.dll"
    }
    $OvrtxBin = (Resolve-Path $OvrtxBin).Path
    & $AppPython -m pip install -e $OvrtxPython
} else {
    & $AppPython -m pip install "ovrtx==$OvrtxWheelVersion" --extra-index-url https://pypi.nvidia.com
    $OvrtxBin = Get-InstalledOvrtxBin $AppPython
}
& $AppPython -m pip install -e $ScriptDir
Write-Host "Using ovrtx native library from: $OvrtxBin"

if ($SkipOvphysx) {
    Write-Host "Skipping ovphysx because -SkipOvphysx was specified."
} else {
    & $AppPython -m pip install "ovphysx==$OvphysxVersion" --extra-index-url https://pypi.nvidia.com
}

if ($InstallCudaPython) {
    & $AppPython -m pip install cuda-python
} else {
    Write-Host "Skipping cuda-python. Use -InstallCudaPython to enable CUDA/OpenGL zero-copy display support."
}

& $UsdPython -m pip install --upgrade pip
& $UsdPython -m pip install usd-core

$EnvFile = Join-Path $ScriptDir ".env"
@"
OVRTX_LIBRARY_PATH_HINT="$OvrtxBin"
USD_PYTHON="$UsdPython"
PYTHONPATH="$ScriptDir;$env:PYTHONPATH"
"@ | Set-Content -LiteralPath $EnvFile -Encoding ASCII

Invoke-FmuBuild $AppPython

Write-Host "Wrote $EnvFile"
Write-Host "Run:"
Write-Host "  $AppPython apps\ov-fmi\main.py usd\ov-fmi\fmi_parser_test.usda"
