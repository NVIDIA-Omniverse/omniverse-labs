REM SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
REM SPDX-License-Identifier: Apache-2.0
@echo off
setlocal

set QTDIR=D:\pforce\physx\externals\Qt\5.10.0\msvc2017_64

:: Accept optional build dir and config arguments
:: Usage: deploy.bat [build_dir] [config]
::   e.g. deploy.bat build_ovd Release
::        deploy.bat build_ovd Debug
if "%~1"=="" (set BUILDDIR=build) else (set BUILDDIR=%~1)
if "%~2"=="" (set CONFIG=Release) else (set CONFIG=%~2)
set DEST=%BUILDDIR%\src\PVDNext\%CONFIG%

if not exist "%DEST%\PVDNext.exe" (
    echo ERROR: PVDNext.exe not found at %DEST%\PVDNext.exe
    echo Usage: deploy.bat [build_dir] [config]
    echo   e.g. deploy.bat build_ovd Release
    echo        deploy.bat build_ovd Debug
    pause
    exit /b 1
)

if /I "%CONFIG%"=="Debug" (
    set SUFFIX=d
    echo Copying Qt DEBUG DLLs to %DEST% ...
) else (
    set SUFFIX=
    echo Copying Qt DLLs to %DEST% ...
)

copy /Y "%QTDIR%\bin\Qt5Core%SUFFIX%.dll"         "%DEST%\"
copy /Y "%QTDIR%\bin\Qt5Gui%SUFFIX%.dll"          "%DEST%\"
copy /Y "%QTDIR%\bin\Qt5Widgets%SUFFIX%.dll"      "%DEST%\"
copy /Y "%QTDIR%\bin\Qt5OpenGL%SUFFIX%.dll"       "%DEST%\"
copy /Y "%QTDIR%\bin\Qt5Network%SUFFIX%.dll"      "%DEST%\"
copy /Y "%QTDIR%\bin\Qt5Concurrent%SUFFIX%.dll"   "%DEST%\"
copy /Y "%QTDIR%\bin\Qt5PrintSupport%SUFFIX%.dll" "%DEST%\"
copy /Y "%QTDIR%\bin\Qt5Svg%SUFFIX%.dll"          "%DEST%\"

:: Qt platform plugin
if not exist "%DEST%\platforms" mkdir "%DEST%\platforms"
copy /Y "%QTDIR%\plugins\platforms\qwindows%SUFFIX%.dll" "%DEST%\platforms\"

:: Qwt — try debug version first, fall back to release
if /I "%CONFIG%"=="Debug" (
    if exist "3rdparty\Qwt\lib\qwtd.dll" (
        copy /Y "3rdparty\Qwt\lib\qwtd.dll" "%DEST%\"
    ) else (
        copy /Y "3rdparty\Qwt\lib\qwt.dll" "%DEST%\"
    )
) else (
    copy /Y "3rdparty\Qwt\lib\qwt.dll" "%DEST%\"
)

echo.
echo Done. You can now run %DEST%\PVDNext.exe directly.
pause
