# SPDX-License-Identifier: Apache-2.0
param(
    [string]$Config = (Join-Path $PSScriptRoot '..\..\out\public-sdk-config.json'),
    [string]$Scene,
    [switch]$Demo
)
$ErrorActionPreference = 'Stop'
$settings = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
if (-not (Test-Path -LiteralPath $settings.python -PathType Leaf)) {
    throw 'Run setup_public_sdk.py first; configured SDK Python was not found.'
}
$launchArgs = @((Join-Path $PSScriptRoot 'launch_public_sdk_blender.py'), '--config', $Config)
if ($Scene) { $launchArgs += @('--scene', $Scene) }
if ($Demo) { $launchArgs += '--demo' }
& $settings.python @launchArgs
exit $LASTEXITCODE
