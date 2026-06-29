#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Wire .githooks/ as the active hooks dir for this repo.
# Run once after clone:  ./scripts/setup_hooks.sh
set -euo pipefail
cd "$(dirname "$0")/.."
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
echo "Pre-commit hook wired. Commits that touch rendering source will run"
echo "test/verify_rendering_correctness.py before completing."
echo "Bypass with NU_SKIP_RENDER_CHECK=1."
