# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

[CmdletBinding()]
param(
    [string]$VulkanSdk = $env:VULKAN_SDK,
    [string]$NanousdDir = (Join-Path $PSScriptRoot "..\..\nanousd"),
    [string]$BuildDir = (Join-Path $PSScriptRoot "..\build-win"),
    [string]$NanousdBuildDir = (Join-Path $PSScriptRoot "..\..\nanousd\build-win"),
    [string]$Generator = "",
    [string]$BuildConfig = "Release",
    [string]$MaterialXSourceDir = "",
    [switch]$SkipRun
)

$ErrorActionPreference = "Stop"

function Resolve-AbsolutePath([string]$Path) {
    $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
}

$RepoDir = Resolve-AbsolutePath (Join-Path $PSScriptRoot "..")
$NanousdDir = Resolve-AbsolutePath $NanousdDir
$BuildDir = Resolve-AbsolutePath $BuildDir
$NanousdBuildDir = Resolve-AbsolutePath $NanousdBuildDir

if (-not $VulkanSdk) {
    throw "VULKAN_SDK is not set. Install the Vulkan SDK or pass -VulkanSdk C:\VulkanSDK\<version>."
}
$VulkanSdk = Resolve-AbsolutePath $VulkanSdk
$env:VULKAN_SDK = $VulkanSdk
$Glslc = Join-Path $VulkanSdk "Bin\glslc.exe"
if (-not (Test-Path $Glslc)) {
    throw "glslc.exe was not found at $Glslc."
}

if (-not (Test-Path (Join-Path $NanousdDir "CMakeLists.txt"))) {
    throw "nanousd was not found at $NanousdDir."
}

Write-Host "==> Building nanousd"
$NanousdArgs = @(
    "-S", $NanousdDir,
    "-B", $NanousdBuildDir,
    "-DCMAKE_BUILD_TYPE=$BuildConfig",
    "-DNANOUSD_BUILD_TESTS=OFF"
)
if ($Generator) {
    $NanousdArgs += @("-G", $Generator)
}
cmake @NanousdArgs
cmake --build $NanousdBuildDir --config $BuildConfig --target nanousd nanousdapi

Write-Host "==> Configuring nanousd-vulkan-renderer"
$RendererArgs = @(
    "-S", $RepoDir,
    "-B", $BuildDir,
    "-DCMAKE_BUILD_TYPE=$BuildConfig",
    "-DNANOUSD_DIR=$NanousdDir",
    "-DGLSLC=$Glslc",
    "-DNUSD_ENABLE_OPENMP=OFF"
)
if ($Generator) {
    $RendererArgs += @("-G", $Generator)
}
if ($MaterialXSourceDir) {
    $RendererArgs += "-DFETCHCONTENT_SOURCE_DIR_MATERIALX=$(Resolve-AbsolutePath $MaterialXSourceDir)"
}
cmake @RendererArgs

Write-Host "==> Building test_headless_render"
cmake --build $BuildDir --config $BuildConfig --target test_headless_render

if (-not $SkipRun) {
    $Exe = Join-Path $BuildDir "test_headless_render.exe"
    if (-not (Test-Path $Exe)) {
        $Exe = Join-Path $BuildDir "Release\test_headless_render.exe"
    }
    if (-not (Test-Path $Exe)) {
        throw "test_headless_render.exe was not found under $BuildDir."
    }
    Write-Host "==> Running $Exe"
    & $Exe
}
