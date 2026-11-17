#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fetcher="$repository_root/ci/fetch_physics.sh"
temporary_root=$(mktemp -d)
trap 'rm -rf "$temporary_root"' EXIT

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

assert_file() {
    [[ -f "$1" ]] || fail "Expected file to exist: $1"
}

assert_failure() {
    if "$@"; then
        fail "$1 unexpectedly succeeded"
    fi
}

assert_executable_in_git() {
    local path=$1
    local mode
    mode=$(git -C "$repository_root" ls-files -s -- "$path")
    [[ "$mode" == 100755* ]] || fail "$path must be committed as executable"
}

new_fixture_repository() {
    local name=$1
    local include_packman=$2
    local repository="$temporary_root/$name"

    git init -q "$repository"
    git -C "$repository" config user.email ci@example.invalid
    git -C "$repository" config user.name 'CI Test'
    mkdir -p "$repository/physx/pvddom" "$repository/physx/pvdruntime/compiler/cmake" "$repository/unrelated"
    printf 'pvddom\n' > "$repository/physx/pvddom/CMakeLists.txt"
    printf 'pvdruntime\n' > "$repository/physx/pvdruntime/compiler/cmake/PVDRuntime.cmake"
    printf 'sentinel\n' > "$repository/unrelated/sentinel.txt"
    if [[ "$include_packman" == true ]]; then
        mkdir -p "$repository/physx/buildtools/packman"
        printf 'packman\n' > "$repository/physx/buildtools/packman/packman.cmd"
    fi
    git -C "$repository" add .
    git -C "$repository" commit -qm fixture
    git -C "$repository" rev-parse HEAD
}

sha_with_packman=$(new_fixture_repository with-packman true)
sha_without_packman=$(new_fixture_repository without-packman false)
[[ "$sha_with_packman" =~ ^[0-9a-f]{40}$ ]] || fail 'Fixture commit must be a 40-character SHA'
assert_executable_in_git ci/fetch_physics.sh
assert_executable_in_git ci/tests/test_fetch_physics.sh

success_destination="$temporary_root/success"
"$fetcher" --repository-url "$temporary_root/with-packman" --ref "$sha_with_packman" --destination "$success_destination" --include-packman
actual_sha=$(git -C "$success_destination" rev-parse HEAD)
[[ "$actual_sha" == "$sha_with_packman" ]] || fail 'Fetched HEAD must match the requested SHA byte-for-byte'
assert_file "$success_destination/physx/pvddom/CMakeLists.txt"
assert_file "$success_destination/physx/pvdruntime/compiler/cmake/PVDRuntime.cmake"
assert_file "$success_destination/physx/buildtools/packman/packman.cmd"
[[ ! -e "$success_destination/unrelated/sentinel.txt" ]] || fail 'Sparse checkout must exclude the sentinel'

assert_failure "$fetcher" --repository-url "$temporary_root/with-packman" --ref not-a-commit --destination "$temporary_root/invalid-ref"

nonempty_destination="$temporary_root/nonempty"
mkdir "$nonempty_destination"
printf 'preserve\n' > "$nonempty_destination/preserve.txt"
assert_failure "$fetcher" --repository-url "$temporary_root/with-packman" --ref "$sha_with_packman" --destination "$nonempty_destination"
assert_file "$nonempty_destination/preserve.txt"

assert_failure "$fetcher" --repository-url "$temporary_root/without-packman" --ref "$sha_without_packman" --destination "$temporary_root/missing-packman" --include-packman

preexisting_empty_destination="$temporary_root/preexisting-empty"
mkdir "$preexisting_empty_destination"
assert_failure "$fetcher" --repository-url "$temporary_root/without-packman" --ref "$sha_without_packman" --destination "$preexisting_empty_destination" --include-packman
[[ -d "$preexisting_empty_destination" ]] || fail 'A pre-existing empty destination must be preserved after a later failure'

printf 'PASS: fetch_physics.sh local fixture tests\n'
