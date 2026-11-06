# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
param([string]$RuntimeRoot, [ValidateSet('public')][string]$Profile='public', [switch]$Repair)
$ErrorActionPreference='Stop'
try {
    & (Join-Path $PSScriptRoot 'prepare_customer.ps1') -RuntimeRoot $RuntimeRoot -Profile $Profile -Repair:$Repair
    do {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'launch.ps1') -RuntimeRoot $RuntimeRoot -Profile $Profile
        $appExit=$LASTEXITCODE
    } while ($appExit -eq 75)
    Write-Host ('LAB_EXIT: '+$appExit)
    exit $appExit
} catch {
    Write-Host ('LAB_ERROR: '+$_.Exception.Message)
    exit 1
}
