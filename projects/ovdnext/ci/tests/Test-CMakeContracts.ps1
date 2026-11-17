# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
param([Parameter(Mandatory=$true)][string]$SourceRoot,
      [string]$CMakeExecutable = 'cmake',
      [string]$Generator = 'Visual Studio 16 2019')
$ErrorActionPreference = 'Stop'
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ("ovdnext-cmake-" + [guid]::NewGuid())
$generatorArguments = @('-G', $Generator)
if ($Generator -like 'Visual Studio*') { $generatorArguments += @('-A', 'x64') }
try {
    $output = & {
        $ErrorActionPreference = 'Continue'
        & $CMakeExecutable -S $SourceRoot -B $testRoot @generatorArguments `
            -DCMAKE_DISABLE_FIND_PACKAGE_Qt5=ON -DPVD3_BUILD_TESTS=OFF `
            -DPVD3_REQUIRE_OVD_SUPPORT=ON 2>&1
    }
    $cmakeExitCode = $LASTEXITCODE
    if ($cmakeExitCode -eq 0) { throw 'Required OVD configure unexpectedly succeeded without roots.' }
    if (($output -join "`n") -notmatch 'PVD3_REQUIRE_OVD_SUPPORT') {
        throw 'Required OVD configure did not report the fail-closed option.'
    }
} finally {
    if (Test-Path $testRoot) { Remove-Item -LiteralPath $testRoot -Recurse -Force }
}
