@echo off
setlocal enabledelayedexpansion
REM Build and run the C doc tests.
REM Usage: run_tests.bat [--prefix C:\path\to\ovrtx]
REM If --prefix is not given, CMAKE_PREFIX_PATH must be set in the environment.

set "SCRIPT_DIR=%~dp0"
set "BUILD_DIR=%SCRIPT_DIR%build"
set "PREFIX_PATH="

:parse_args
if "%~1"=="" goto done_args
if /i "%~1"=="--prefix" (
    set "PREFIX_PATH=%~2"
    shift
    shift
    goto parse_args
)
echo Unknown option: %~1 >&2
exit /b 1

:done_args

set "CMAKE_ARGS="
if defined PREFIX_PATH (
    set "CMAKE_ARGS=-DCMAKE_PREFIX_PATH=!PREFIX_PATH!"
)

echo === Configuring ===
cmake -B "%BUILD_DIR%" -S "%SCRIPT_DIR%." !CMAKE_ARGS!
if !errorlevel! neq 0 exit /b !errorlevel!

echo === Building ===
cmake --build "%BUILD_DIR%" --config Release
if !errorlevel! neq 0 exit /b !errorlevel!

echo === Running tests ===
pushd "%BUILD_DIR%"
ctest --verbose -C Release
set "TEST_RESULT=!errorlevel!"
popd
exit /b !TEST_RESULT!
