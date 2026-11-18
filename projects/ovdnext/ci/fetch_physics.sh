#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

usage() {
    printf 'Usage: %s --repository-url <url> --ref <40-character commit> --destination <path> [--include-packman]\n' "$0" >&2
    exit 1
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

repository_url=
ref=
destination=
include_packman=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repository-url)
            [[ $# -ge 2 ]] || usage
            repository_url=$2
            shift 2
            ;;
        --ref)
            [[ $# -ge 2 ]] || usage
            ref=$2
            shift 2
            ;;
        --destination)
            [[ $# -ge 2 ]] || usage
            destination=$2
            shift 2
            ;;
        --include-packman)
            include_packman=true
            shift
            ;;
        *)
            usage
            ;;
    esac
done

[[ -n "$repository_url" && -n "$ref" && -n "$destination" ]] || usage
[[ "$ref" =~ ^[0-9a-fA-F]{40}$ ]] || fail 'Ref must be an exact 40-character hexadecimal commit ID.'

case "$destination" in
    */*)
        destination_parent=${destination%/*}
        [[ -n "$destination_parent" ]] || destination_parent=/
        ;;
    *)
        destination_parent=.
        ;;
esac
[[ -d "$destination_parent" ]] || fail "Destination parent does not exist: $destination_parent"

destination_created_by_this_invocation=false
if [[ -e "$destination" || -L "$destination" ]]; then
    [[ -d "$destination" && ! -L "$destination" ]] || fail "Destination must be an empty directory: $destination"
    shopt -s dotglob nullglob
    destination_entries=("$destination"/*)
    shopt -u dotglob nullglob
    [[ ${#destination_entries[@]} -eq 0 ]] || fail "Destination must be empty: $destination"
else
    if mkdir -- "$destination"; then
        destination_created_by_this_invocation=true
    else
        fail "Destination could not be created (it may have appeared concurrently): $destination"
    fi
fi

cleanup_on_failure() {
    local status=$?
    if [[ $status -ne 0 && "$destination_created_by_this_invocation" == true && -e "$destination" ]]; then
        if ! rm -rf -- "$destination"; then
            printf 'ERROR: Failed to remove destination created by this invocation: %s\n' "$destination" >&2
        fi
    fi
    exit "$status"
}
trap cleanup_on_failure EXIT

git init "$destination"
git -C "$destination" remote add origin "$repository_url"
git -C "$destination" sparse-checkout init --cone

sparse_paths=(physx/pvddom physx/pvdruntime physx/include/foundation)
if [[ "$include_packman" == true ]]; then
    sparse_paths+=(physx/buildtools/packman)
fi
git -C "$destination" sparse-checkout set "${sparse_paths[@]}"
git -C "$destination" fetch --depth=1 origin "$ref"
git -C "$destination" checkout --detach FETCH_HEAD

normalized_ref=${ref,,}
head=$(git -C "$destination" rev-parse HEAD)
[[ "$head" == "$normalized_ref" ]] || fail "Fetched HEAD '$head' does not match requested commit '$normalized_ref'."

[[ -f "$destination/physx/pvddom/CMakeLists.txt" ]] || fail 'Required fetched file is missing: physx/pvddom/CMakeLists.txt'
[[ -f "$destination/physx/pvdruntime/compiler/cmake/PVDRuntime.cmake" ]] || fail 'Required fetched file is missing: physx/pvdruntime/compiler/cmake/PVDRuntime.cmake'
if [[ "$include_packman" == true ]]; then
    [[ -f "$destination/physx/buildtools/packman/packman.cmd" ]] || fail 'Required fetched file is missing: physx/buildtools/packman/packman.cmd'
fi

trap - EXIT
printf 'Fetched Physics commit %s into %s\n' "$normalized_ref" "$destination"
