#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python3}"
commit="v1.0.1"
source_repo=""
worktree_dir=""
source_pdf=""
out_dir=""
prefer_pdf=0
passthrough=()

usage() {
    cat <<'EOF'
Usage:
  ./materialize-spec.sh [options]

Options:
  --source-repo PATH     Local aousd/specifications-public checkout.
  --commit REF           Spec tag or commit to pin.
  --worktree-dir PATH    Detached worktree directory.
  --source-pdf PATH      USD Core Spec PDF fallback.
  --out-dir PATH         Output directory. Defaults to spec/pinned.
  --prefer-pdf           Use the PDF fallback even with a source repo.
  -h, --help             Show this help.

The PYTHON environment variable overrides the Python executable; otherwise
python3 is used.
EOF
    return 0
}

need_value() {
    local opt_name="$1"
    if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "materialize-spec.sh: $opt_name requires a value" >&2
        exit 2
    fi
    return 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-repo)
            need_value "$@"
            source_repo="$2"
            shift 2
            ;;
        --commit)
            need_value "$@"
            commit="$2"
            shift 2
            ;;
        --worktree-dir)
            need_value "$@"
            worktree_dir="$2"
            shift 2
            ;;
        --source-pdf)
            need_value "$@"
            source_pdf="$2"
            shift 2
            ;;
        --out-dir)
            need_value "$@"
            out_dir="$2"
            shift 2
            ;;
        --prefer-pdf)
            prefer_pdf=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            passthrough+=("$@")
            break
            ;;
        *)
            passthrough+=("$1")
            shift
            ;;
    esac
done

python_args=("$script_dir/materialize-spec.py" "--commit" "$commit")

if [[ -n "$source_repo" ]]; then
    python_args+=("--source-repo" "$source_repo")
fi

if [[ -n "$worktree_dir" ]]; then
    python_args+=("--worktree-dir" "$worktree_dir")
fi

if [[ -n "$source_pdf" ]]; then
    python_args+=("--source-pdf" "$source_pdf")
fi

if [[ -n "$out_dir" ]]; then
    python_args+=("--out-dir" "$out_dir")
fi

if [[ "$prefer_pdf" -eq 1 ]]; then
    python_args+=("--prefer-pdf")
fi

python_args+=("${passthrough[@]}")

exec "$python_bin" "${python_args[@]}"
