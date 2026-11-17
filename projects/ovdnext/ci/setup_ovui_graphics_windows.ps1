# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Hosted CI graphics only; these files are never added to the application ZIP.
$ErrorActionPreference = 'Stop'
$archive = "$env:RUNNER_TEMP/mesa-24.0.5.7z"
Invoke-WebRequest -Uri 'https://github.com/pal1000/mesa-dist-win/releases/download/24.0.5/mesa3d-24.0.5-release-msvc.7z' -OutFile $archive
if ((Get-Item -LiteralPath $archive).Length -ne 57414126 -or (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ef71c6907d424439f158e5ec102c7915bbf446fb00e04f62daa575b4780c596a') { throw 'Mesa archive mismatch' }
# This archive uses BCJ2, which py7zr does not support.
& "$env:ProgramFiles/7-Zip/7z.exe" x -y $archive "-o$env:RUNNER_TEMP/mesa-ci"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
"GLCONTEXT_WIN_LIBGL=$env:RUNNER_TEMP/mesa-ci/x64/opengl32.dll" >> $env:GITHUB_ENV
'GALLIUM_DRIVER=llvmpipe' >> $env:GITHUB_ENV
'LIBGL_ALWAYS_SOFTWARE=1' >> $env:GITHUB_ENV
$vulkan = "$env:WINDIR/System32/vulkan-1.dll"
if (-not (Test-Path -LiteralPath $vulkan)) {
    $archive = "$env:RUNNER_TEMP/vulkan-runtime.zip"
    Invoke-WebRequest -Uri 'https://sdk.lunarg.com/sdk/download/1.4.357.0/windows/VulkanRT-X64-1.4.357.0-Components.zip' -OutFile $archive
    if ((Get-Item -LiteralPath $archive).Length -ne 18134567 -or (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'a14672efed15aafc7f5a16572d35cd3a3416eadf670aeee3cdf50ee32d5fbf83') { throw 'Vulkan archive mismatch' }
    Expand-Archive -LiteralPath $archive -DestinationPath "$env:RUNNER_TEMP/vulkan-runtime"
    Copy-Item -LiteralPath "$env:RUNNER_TEMP/vulkan-runtime/VulkanRT-X64-1.4.357.0-Components/x64/vulkan-1.dll" -Destination $vulkan
}
& $env:OVUI_PYTHON -B -c "import ctypes; ctypes.WinDLL('vulkan-1.dll'); print('Vulkan loader available')"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
