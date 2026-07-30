# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]


def test_windows_package_probe_survives_powershell_51_quoting():
    setup = (_ROOT / "demoapp" / "setup.ps1").read_text(encoding="utf-8")
    match = re.search(
        r"(?s)function Assert-OvPackages.*?\$code = @'\n(?P<code>.*?)\n'@",
        setup,
    )

    assert match is not None
    code = match.group("code")
    assert '"' not in code, (
        "Windows PowerShell 5.1 strips embedded double quotes when passing "
        "the here-string to Python's native -c argument"
    )
    compile(code, "Assert-OvPackages", "exec")
