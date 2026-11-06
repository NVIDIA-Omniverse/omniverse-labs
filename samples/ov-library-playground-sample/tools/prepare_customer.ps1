# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
<# Private Python runtime and pinned public dependencies. No PATH or registry changes. #>
param([string]$RuntimeRoot, [ValidateSet('public')][string]$Profile='public', [switch]$Repair)
$ErrorActionPreference='Stop'
# Windows PowerShell may have loaded only the binary Utility commands in a
# nested GUI launch. Import its manifest so script commands such as Get-FileHash
# are available as well; do not depend on the caller's module auto-loading.
Import-Module (Join-Path $PSHOME 'Modules/Microsoft.PowerShell.Utility/Microsoft.PowerShell.Utility.psd1') -Force
$repoRoot=Split-Path $PSScriptRoot -Parent
if (-not $RuntimeRoot) { $RuntimeRoot=Join-Path $repoRoot '.ov_library_lab_runtime' }
$RuntimeRoot=[IO.Path]::GetFullPath($RuntimeRoot)
$bootstrap=Get-Content -LiteralPath (Join-Path $repoRoot 'config/bootstrap.json') -Raw | ConvertFrom-Json
$nativePython=Join-Path $RuntimeRoot 'libraries-env/Scripts/python.exe'
$lockFile=Join-Path $repoRoot 'config/requirements-win.lock'
$stamp=Join-Path $RuntimeRoot 'installation-lock.sha256'
$expectedHash=(Get-FileHash -LiteralPath $lockFile -Algorithm SHA256).Hash
New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null
$env:PYTHONNOUSERSITE='1'
foreach ($variable in @('PYTHONPATH','PYTHONHOME','CARB_APP_PATH')) { [Environment]::SetEnvironmentVariable($variable,$null,'Process') }

if (-not (Test-Path -LiteralPath $nativePython)) {
    Write-Host 'LAB_PROGRESS: Downloading the verified Python bootstrap'
    $uvFolder=Join-Path $RuntimeRoot ('bootstrap/uv-'+$bootstrap.uv_version)
    $uvExe=Join-Path $uvFolder 'uv.exe'
    $archive=Join-Path $RuntimeRoot 'bootstrap/uv.zip'
    New-Item -ItemType Directory -Force -Path (Split-Path $archive -Parent) | Out-Null
    if (-not (Test-Path -LiteralPath $archive) -or (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $bootstrap.uv_sha256) {
        Invoke-WebRequest -UseBasicParsing -Uri $bootstrap.uv_url -OutFile $archive
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $bootstrap.uv_sha256) { throw 'The bootstrap checksum does not match. Retry the download; nothing was executed.' }
    Import-Module Microsoft.PowerShell.Archive
    Expand-Archive -LiteralPath $archive -DestinationPath $uvFolder -Force
    $env:UV_PYTHON_INSTALL_DIR=Join-Path $RuntimeRoot 'python'
    $env:UV_CACHE_DIR=Join-Path $RuntimeRoot 'cache/uv'
    $env:UV_NO_MODIFY_PATH='1'
    Write-Host ('LAB_PROGRESS: Installing private Python '+$bootstrap.python)
    & $uvExe python install $bootstrap.python --no-bin
    if ($LASTEXITCODE) { throw 'Python download failed. Check your network/proxy and retry; cached downloads are reused.' }
    & $uvExe venv --python $bootstrap.python --seed (Join-Path $RuntimeRoot 'libraries-env')
    if ($LASTEXITCODE) { throw 'Could not create the private Python environment.' }
}
Write-Host 'LAB_PROGRESS: Checking your GPU and available disk space'
& $nativePython (Join-Path $PSScriptRoot 'preflight.py') --runtime-root $RuntimeRoot --hardware-only --output (Join-Path $RuntimeRoot 'diagnostics/hardware.json')
if ($LASTEXITCODE) { throw 'Hardware checks failed. See the checks above for the required action.' }
$installedHash=if (Test-Path -LiteralPath $stamp) { (Get-Content -LiteralPath $stamp -Raw).Trim() } else { '' }
if ($Repair -or $installedHash -ne $expectedHash) {
    Write-Host 'LAB_PROGRESS: Installing public libraries; the first download can take several minutes'
    & $nativePython -m pip install --require-hashes -r $lockFile --disable-pip-version-check
    if ($LASTEXITCODE) { throw 'Library installation failed. Retry to resume cached downloads. See the log for proxy or package errors.' }
    & $nativePython -m pip check
    if ($LASTEXITCODE) { throw 'Dependencies conflict. Use a separate runtime folder or repair the environment.' }
    Set-Content -LiteralPath $stamp -Value $expectedHash -Encoding Ascii
}
Write-Host 'LAB_PROGRESS: Verifying installed library versions'
foreach ($library in @('newton','package')) {
    $roomLock=Join-Path $repoRoot ('config/requirements-'+$library+'-win.lock')
    $roomStamp=Join-Path $RuntimeRoot ($library+'-installation-lock.sha256')
    $roomHash=(Get-FileHash -LiteralPath $roomLock -Algorithm SHA256).Hash
    $roomInstalled=if (Test-Path -LiteralPath $roomStamp) { (Get-Content -LiteralPath $roomStamp -Raw).Trim() } else { '' }
    $roomPython=Join-Path $RuntimeRoot ($library+'-env/Scripts/python.exe')
    if ($Repair -or $roomInstalled -ne $roomHash -or -not (Test-Path -LiteralPath $roomPython)) {
        Write-Host ('LAB_PROGRESS: Installing the '+$library+' room library')
        & (Join-Path $PSScriptRoot 'setup_extended_libraries.ps1') -Python $nativePython -RuntimeRoot $RuntimeRoot
        break
    }
}
& $nativePython (Join-Path $PSScriptRoot 'preflight.py') --runtime-root $RuntimeRoot --output (Join-Path $RuntimeRoot 'diagnostics/preflight.json')
if ($LASTEXITCODE) { throw 'Dependency verification failed. Choose Repair and retry.' }
$sceneArgs=@((Join-Path $PSScriptRoot 'setup_scene.py'),'--runtime-root',$RuntimeRoot,'--profile',$Profile)
if ($Repair) { $sceneArgs+='--repair' }
& $nativePython @sceneArgs
if ($LASTEXITCODE) { throw 'Scene preparation failed. See the log, then Retry.' }
Write-Host 'LAB_PROGRESS: Installing prepared NVIDIA SimReady exhibit assets'
$assetArgs=@((Join-Path $PSScriptRoot 'setup_simready.py'),'--runtime-root',$RuntimeRoot)
if ($Repair) { $assetArgs+='--repair' }
& $nativePython @assetArgs
if ($LASTEXITCODE) { throw 'SimReady asset preparation failed. Retry to reuse verified downloads.' }
Write-Host 'LAB_PROGRESS: Installation ready'
