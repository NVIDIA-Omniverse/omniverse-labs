#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_directory() {
    [[ -n "$2" && -d "$2" ]] || fail "$1 is missing: $2"
}

require_file() {
    [[ -f "$2" ]] || fail "$1 is missing: $2"
}

usage() {
    printf 'Usage: %s --source-root <path> --build-root <path> --physics-root <path> --source-commit <sha> [options]\n' "$0" >&2
    exit 1
}

source_root=
build_root=
physics_root=
configuration=RelWithDebInfo
source_commit=
physics_ref=a12f79ce585ffb79e5ffa4848cb45efc3de2f632

while [[ $# -gt 0 ]]; do
    case $1 in
        --source-root|--build-root|--physics-root|--configuration|--source-commit|--physics-ref)
            [[ $# -ge 2 ]] || usage
            option=${1#--}
            option=${option//-/_}
            printf -v "$option" '%s' "$2"
            shift 2
            ;;
        *) usage ;;
    esac
done

[[ "$source_commit" =~ ^[0-9a-fA-F]{40}$ ]] || fail 'source commit must be an exact 40-character hexadecimal commit ID.'
[[ "$physics_ref" =~ ^[0-9a-fA-F]{40}$ ]] || fail 'Physics ref must be an exact 40-character hexadecimal commit ID.'
[[ "$configuration" == RelWithDebInfo ]] || fail 'Standalone OvdNext Data verification requires RelWithDebInfo.'
require_directory 'Source root' "$source_root"
require_directory 'Physics root' "$physics_root"
[[ -n "$build_root" ]] || fail 'build root is required.'

for command_name in git cmake ninja ctest python3 readelf; do
    command -v "$command_name" >/dev/null || fail "$command_name was not found."
done
source_head=$(git -C "$source_root" rev-parse HEAD) || fail 'Unable to resolve Source root HEAD.'
physics_head=$(git -C "$physics_root" rev-parse HEAD) || fail 'Unable to resolve Physics root HEAD.'
[[ "${source_head,,}" == "${source_commit,,}" ]] || fail "Source root HEAD does not match source commit."
[[ "${physics_head,,}" == "${physics_ref,,}" ]] || fail "Physics root HEAD does not match Physics ref."

adapter_source=$source_root/src/OvdDataOmniPvd
runtime_root=$physics_root/physx/pvdruntime
fixture=$source_root/test/data/test_scene.ovd
baseline=$source_root/ci/baselines/qt-free/ovdnext-ci-parity.json
inventory_verifier=$source_root/ci/verify_test_inventory.py
dependency_verifier=$source_root/ci/verify_ovdnext_data_dependencies.py
require_directory 'OvdDataOmniPvd source' "$adapter_source"
require_directory 'PVDRuntime root' "$runtime_root"
require_file 'Probe fixture' "$fixture"
require_file 'Test inventory baseline' "$baseline"
require_file 'Test inventory verifier' "$inventory_verifier"
require_file 'Dependency verifier' "$dependency_verifier"

mkdir -p -- "$build_root"
cmake -S "$adapter_source" -B "$build_root" -G Ninja \
    "-DCMAKE_BUILD_TYPE=$configuration" \
    "-DPVDRUNTIME_ROOT=$runtime_root" \
    -DOVDDATA_BUILD_TESTS=ON \
    -DOVDDATAOMNIPVD_BUILD_TESTS=ON

graph_path=$build_root/targets.dot
cmake --graphviz="$graph_path" "$build_root"
cmake --build "$build_root" --target \
    ovddata_tests \
    ovddata_ingest_tests \
    ovddata_omnipvd_tests \
    ovdnext_data_probe \
    ovdnext_data_probe_tests \
    --parallel

discovery_path=$build_root/ctest-discovery.json
junit_path=$build_root/ctest.junit.xml
ctest --test-dir "$build_root" --show-only=json-v1 > "$discovery_path"
ctest --test-dir "$build_root" --no-tests=error --output-on-failure \
    --output-junit "$junit_path"
python3 "$inventory_verifier" verify \
    --baseline "$baseline" \
    --discovery "$discovery_path" \
    --junit "$junit_path" \
    --scope ovdnext-data \
    --platform linux-x86_64 2>&1 | tee "$build_root/inventory-verifier.log"

probe=$build_root/ovdnext_data_probe
require_file 'Built ovdnext_data_probe' "$probe"
"$probe" "$fixture" > "$build_root/probe-output.txt"
readelf -d "$probe" > "$build_root/imports.txt"
python3 "$dependency_verifier" \
    --graph "$graph_path" \
    --root-target ovdnext_data_probe \
    --imports "$build_root/imports.txt"
