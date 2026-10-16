#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$ROOT/settings.local.sh" ]]; then source "$ROOT/settings.local.sh"; fi
CLIENT_CONFIG="${CB_REFERENCE_CLIENT_CONFIG:-$ROOT/.runtime/default/client.json}"
if [[ ! -r "$CLIENT_CONFIG" ]]; then
    echo "Start ./run-services.sh first. Missing $CLIENT_CONFIG" >&2
    exit 1
fi
if [[ -z "${KIT_EXECUTABLE:-}" ]]; then
    for candidate in "${KIT_ROOT:-$HOME/src/kit}/kit" \
        "${KIT_ROOT:-$HOME/src/kit}/_build/linux-x86_64/release/kit/kit" \
        "${KIT_ROOT:-$HOME/src/kit}/_build/linux-x86_64/release/kit"; do
        if [[ -x "$candidate" && -f "$candidate" ]]; then KIT_EXECUTABLE="$candidate"; break; fi
    done
fi
if [[ ! -x "${KIT_EXECUTABLE:-}" ]]; then echo "Set KIT_ROOT to your Kit SDK" >&2; exit 1; fi
KIT_DIR="$(cd -- "$(dirname -- "$KIT_EXECUTABLE")" && pwd)"
ARGS=()
for folder in exts extscache extsPhysics; do
    if [[ -d "$KIT_DIR/$folder" ]]; then ARGS+=(--ext-folder "$KIT_DIR/$folder"); fi
done
exec "$KIT_EXECUTABLE" "$ROOT/apps/reference.kit" ${ARGS[@]+"${ARGS[@]}"} \
    --ext-path "$ROOT/source/clients/extensions/utils/omni.example.reference_client" \
    "--/exts/omni.example.reference_client/clientConfig=$CLIENT_CONFIG" "$@"
