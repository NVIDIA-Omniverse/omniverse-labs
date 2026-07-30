# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]


def _project_version(pyproject: Path) -> str:
    match = re.search(
        r'(?m)^version\s*=\s*"([^"]+)"\s*$',
        pyproject.read_text(encoding="utf-8"),
    )
    assert match is not None, f"missing static project version in {pyproject}"
    return match.group(1)


def test_release_versions_match():
    version = (_ROOT / "VERSION.md").read_text(encoding="utf-8").strip()

    assert re.fullmatch(r"\d+\.\d+\.\d+", version)
    assert _project_version(_ROOT / "pyproject.toml") == version
    assert _project_version(_ROOT / "demoapp" / "pyproject.toml") == version
