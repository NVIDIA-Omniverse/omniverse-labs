# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
& python (Join-Path $root 'ci/tests/test_pipeline_structure.py')
if ($LASTEXITCODE -ne 0) { throw 'Native pipeline contract failed.' }
& python -m unittest ci.tests.test_native_build
if ($LASTEXITCODE -ne 0) { throw 'Native build-driver contracts failed.' }
