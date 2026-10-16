@echo off
rem SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rem SPDX-License-Identifier: Apache-2.0

setlocal
set "ROOT=%~dp0"
if not defined OVUI_PYTHON set "OVUI_PYTHON=%ROOT%.venv-ovui\Scripts\python.exe"
if not exist "%OVUI_PYTHON%" (
    echo Create .venv-ovui with Python 3.12 and install requirements-ovui.txt, or set OVUI_PYTHON.
    exit /b 1
)
set "PYTHONPATH=%ROOT%source;%PYTHONPATH%"
"%OVUI_PYTHON%" -m clients.ovui_client %*
exit /b %ERRORLEVEL%
