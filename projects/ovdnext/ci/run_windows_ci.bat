REM SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
REM SPDX-License-Identifier: Apache-2.0
@echo off
setlocal EnableExtensions
set "PSModulePath=%ProgramFiles%\WindowsPowerShell\Modules;%SystemRoot%\system32\WindowsPowerShell\v1.0\Modules"
if /I not "%~1"=="verify" goto :usage
set "SOURCE_ROOT=%~dp0.."
for %%I in ("%SOURCE_ROOT%") do set "SOURCE_ROOT=%%~fI"
set "SOURCE_COMMIT=%GITHUB_SHA%"
if not defined SOURCE_COMMIT set "SOURCE_COMMIT=%CI_COMMIT_SHA%"
if not defined SOURCE_COMMIT for /f %%S in ('git -C "%SOURCE_ROOT%" rev-parse HEAD') do set "SOURCE_COMMIT=%%S"
set "PHYSICS_ROOT=%SOURCE_ROOT%\_ci\deps\physics"
if defined OVDNEXT_PHYSICS_ROOT set "PHYSICS_ROOT=%OVDNEXT_PHYSICS_ROOT%"
if not "%OVDNEXT_PUBLIC_WINDOWS_TOOLCHAIN_READY%"=="1" goto :missing_public_setup
if not exist "%OVDNEXT_PUBLIC_VS_ROOT%\VC\Auxiliary\Build\vcvarsall.bat" goto :missing_public_setup
if not exist "%OVDNEXT_PUBLIC_CMAKE_ROOT%\bin\cmake.exe" goto :missing_public_setup
if not exist "%OVDNEXT_PUBLIC_CMAKE_ROOT%\bin\ctest.exe" goto :missing_public_setup
if not exist "%OVDNEXT_PUBLIC_NINJA_ROOT%\ninja.exe" goto :missing_public_setup
if not exist "%OVDNEXT_PUBLIC_PYTHON%" goto :missing_public_setup
if not exist "%OVDNEXT_PUBLIC_VC_RUNTIME_DIR%\msvcp140.dll" goto :missing_public_setup
if not exist "%OVDNEXT_PUBLIC_VC_RUNTIME_DIR%\vcruntime140.dll" goto :missing_public_setup
if not exist "%OVDNEXT_PUBLIC_VC_RUNTIME_DIR%\vcruntime140_1.dll" goto :missing_public_setup
set "CMAKE_EXE=%OVDNEXT_PUBLIC_CMAKE_ROOT%\bin\cmake.exe"
set "CTEST_EXE=%OVDNEXT_PUBLIC_CMAKE_ROOT%\bin\ctest.exe"
set "PATH=%OVDNEXT_PUBLIC_CMAKE_ROOT%\bin;%OVDNEXT_PUBLIC_NINJA_ROOT%;%PATH%"
call "%OVDNEXT_PUBLIC_VS_ROOT%\VC\Auxiliary\Build\vcvarsall.bat" amd64 10.0.17763.0 -vcvars_ver=14.29.30133
if errorlevel 1 exit /b %errorlevel%
if /I not "%VCToolsVersion%"=="14.29.30133" goto :wrong_toolchain
if /I "%WindowsSDKVersion%"=="10.0.17763.0" goto :toolchain_ready
if /I "%WindowsSDKVersion%"=="10.0.17763.0\" goto :toolchain_ready
goto :wrong_toolchain
:toolchain_ready
powershell -NoProfile -ExecutionPolicy Bypass -File "%SOURCE_ROOT%\ci\tests\Test-CMakeContracts.ps1" -SourceRoot "%SOURCE_ROOT%" -CMakeExecutable "%CMAKE_EXE%" -Generator Ninja
if errorlevel 1 exit /b %errorlevel%
powershell -NoProfile -ExecutionPolicy Bypass -File "%SOURCE_ROOT%\ci\build_windows.ps1" -SourceRoot "%SOURCE_ROOT%" -BuildRoot "%SOURCE_ROOT%\_ci\build\windows-x86_64" -PhysicsRoot "%PHYSICS_ROOT%" -SourceCommit "%SOURCE_COMMIT%" -ArtifactRoot "%SOURCE_ROOT%\artifacts\windows-x86_64" -CMakeExecutable "%CMAKE_EXE%" -CTestExecutable "%CTEST_EXE%" -PythonExecutable "%OVDNEXT_PUBLIC_PYTHON%"
exit /b %errorlevel%
:missing_public_setup
echo ERROR: Run public Windows setup first; OVDNEXT_PUBLIC_WINDOWS_TOOLCHAIN_READY and its dependency paths must be valid. 1>&2
exit /b 1
:wrong_toolchain
echo ERROR: Native CI requires MSVC 14.29.30133 and Windows SDK 10.0.17763.0. 1>&2
exit /b 1
:usage
echo Usage: %~nx0 verify 1>&2
exit /b 1
