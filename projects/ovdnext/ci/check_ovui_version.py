# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Require a matching next release version before a PR can merge into main."""
import argparse
import re

from package_ovui_portable import ROOT, run, validate_version
from release_ovui import release_notes


def check_version(root, base):
    if not re.fullmatch(r"[0-9a-f]{40}", base):
        raise ValueError("Base must be a full commit SHA")
    previous = run(["git", "show", f"{base}:CHANGELOG.md"], cwd=root)
    version, _ = release_notes((root / "CHANGELOG.md").read_text(encoding="utf-8"), previous)
    if validate_version(root) != version:
        raise ValueError("Application version must match the newest release section")
    return version


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="PR base commit SHA")
    args = parser.parse_args(argv)
    print(f"Release version verified: {check_version(ROOT, args.base)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
