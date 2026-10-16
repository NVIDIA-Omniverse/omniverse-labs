@echo off
rem SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rem SPDX-License-Identifier: Apache-2.0

setlocal
set "ROOT=%~dp0"
if not defined CB_REFERENCE_CLIENT_CONFIG set "CB_REFERENCE_CLIENT_CONFIG=%ROOT%.runtime\default\client.json"
if not exist "%CB_REFERENCE_CLIENT_CONFIG%" (
    echo Start python reference.py demo --no-console first.
    exit /b 1
)
if not defined KIT_ROOT set "KIT_ROOT=%ROOT%..\kit"
if not defined KIT_EXECUTABLE set "KIT_EXECUTABLE=%KIT_ROOT%\kit.exe"
if not exist "%KIT_EXECUTABLE%" (
    echo Set KIT_ROOT to your extracted Kit SDK directory.
    exit /b 1
)
for %%I in ("%KIT_EXECUTABLE%") do set "KIT_DIR=%%~dpI"
"%KIT_EXECUTABLE%" "%ROOT%apps\reference.kit" --ext-folder "%KIT_DIR%exts" --ext-folder "%KIT_DIR%extscache" --ext-folder "%KIT_DIR%extsPhysics" --ext-path "%ROOT%source\clients\extensions\utils\omni.example.reference_client" "--/exts/omni.example.reference_client/clientConfig=%CB_REFERENCE_CLIENT_CONFIG%" %*
exit /b %ERRORLEVEL%
