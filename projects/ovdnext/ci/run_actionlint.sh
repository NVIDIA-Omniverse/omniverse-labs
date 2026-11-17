#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

version=1.7.12
archive="actionlint_1.7.12_linux_amd64.tar.gz"
expected_sha256="8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8"
download_url="https://github.com/rhysd/actionlint/releases/download/v${version}/${archive}"
temporary_root=$(mktemp -d "${TMPDIR:-/tmp}/ovdnext-actionlint.XXXXXXXX")

cleanup() {
  local status=$?
  if [[ -n "$temporary_root" && -d "$temporary_root" && "$temporary_root" == *ovdnext-actionlint.* ]]; then
    rm -rf -- "$temporary_root"
  fi
  exit "$status"
}
trap cleanup EXIT

curl --fail --location --silent --show-error \
  --output "$temporary_root/$archive" \
  "$download_url"

printf '%s  %s\n' "$expected_sha256" "$temporary_root/$archive" | sha256sum --check --status
tar -C "$temporary_root" -xzf "$temporary_root/$archive" actionlint

"$temporary_root/actionlint" \
  -no-color \
  .github/workflows/ci.yml

# Prove the unsuppressed actionlint invocation still rejects invalid action
# metadata. This unknown checkout input must remain rejected.
negative_fixture="$temporary_root/invalid-action-input.yml"
negative_output="$temporary_root/invalid-action-input.log"
printf '%s\n' \
  'name: actionlint negative contract' \
  'on: push' \
  'jobs:' \
  '  negative-contract:' \
  '    runs-on: ubuntu-22.04' \
  '    steps:' \
  '      - uses: actions/checkout@v4' \
  '        with:' \
  '          ovdnext-unknown-contract-input: true' \
  > "$negative_fixture"

if "$temporary_root/actionlint" \
  -no-color \
  "$negative_fixture" > "$negative_output" 2>&1; then
  printf 'actionlint negative contract unexpectedly passed\n' >&2
  exit 1
fi
grep -Fq 'input "ovdnext-unknown-contract-input" is not defined in action "actions/checkout@v4"' "$negative_output"
