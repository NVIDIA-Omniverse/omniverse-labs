#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

repo_root=$(cd "$(dirname "$0")/../.." && pwd)
standalone=$repo_root/ci/build_ovdnext_data_linux.sh

contains() {
    grep -Fq -- "$2" "$1" || fail "$3"
}

cd "$repo_root"
python3 -m unittest ci.tests.test_native_build

[[ -f "$standalone" ]] || fail "Linux standalone OvdNext Data build script is missing: $standalone"
for expected in \
    'src/OvdDataOmniPvd' \
    '-DCMAKE_BUILD_TYPE=' \
    '-DPVDRUNTIME_ROOT=' \
    '-DOVDDATA_BUILD_TESTS=ON' \
    '-DOVDDATAOMNIPVD_BUILD_TESTS=ON' \
    ovddata_tests \
    ovddata_ingest_tests \
    ovddata_omnipvd_tests \
    ovdnext_data_probe \
    ovdnext_data_probe_tests \
    ctest-discovery.json \
    ctest.junit.xml \
    --graphviz \
    targets.dot \
    verify_test_inventory.py \
    inventory-verifier.log \
    --scope \
    ovdnext-data \
    verify_ovdnext_data_dependencies.py \
    test/data/test_scene.ovd \
    readelf \
    imports.txt; do
    contains "$standalone" "$expected" "Linux standalone build contract is missing $expected"
done
contains "$standalone" '--platform linux-x86_64' \
    'Linux standalone build must select the native inventory platform explicitly'

printf 'PASS: Linux OvdNext Data build contracts\n'
