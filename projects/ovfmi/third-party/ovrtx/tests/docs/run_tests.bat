@echo off
setlocal enabledelayedexpansion
REM Run all doc test suites (usd, python, c) and report results.
REM Usage: run_tests.bat [pytest args...]
REM Example: run_tests.bat -v --tb=short

set "SCRIPT_DIR=%~dp0"
set "FAILED="
set "PASSED="
set "FAIL_COUNT=0"

echo ========================================
echo Running: usd
echo ========================================
pushd "%SCRIPT_DIR%usd"
uv run pytest %*
if !errorlevel! equ 0 (
    set "PASSED=!PASSED! usd"
) else (
    set "FAILED=!FAILED! usd"
    set /a FAIL_COUNT+=1
)
popd
echo.

echo ========================================
echo Running: c (CMake + GoogleTest)
echo ========================================
call "%SCRIPT_DIR%c\run_tests.bat" %*
if !errorlevel! equ 0 (
    set "PASSED=!PASSED! c"
) else (
    set "FAILED=!FAILED! c"
    set /a FAIL_COUNT+=1
)
echo.

echo ========================================
echo Running: python
echo ========================================
pushd "%SCRIPT_DIR%python"
uv run pytest %*
if !errorlevel! equ 0 (
    set "PASSED=!PASSED! python"
) else (
    set "FAILED=!FAILED! python"
    set /a FAIL_COUNT+=1
)
popd
echo.

echo ========================================
echo Results
echo ========================================

for %%s in (!PASSED!) do (
    echo   PASSED: %%s
)

for %%s in (!FAILED!) do (
    echo   FAILED: %%s
)

if !FAIL_COUNT! gtr 0 (
    echo.
    echo !FAIL_COUNT! suite(s^) failed.
    exit /b 1
) else (
    echo.
    echo All suites passed.
    exit /b 0
)
