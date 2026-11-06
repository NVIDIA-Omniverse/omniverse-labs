# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pillow is test-only (build 29 review, line-and-render-fixes n-07-04).

The locked SDK Python ships without Pillow, so nothing that runs there may import it:
the add-on, the runtime and the probes that run in the locked SDK Python. Developer
tools that need Pillow take it from the consumer/test environment instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[1]
# Probes run by the locked SDK Python (see their docstrings / --usd-site-packages split).
SDK_PYTHON_PROBES = ('scripts/run_scene_transaction_probe.py', 'tests/public_sdk_physics_projection_probe.py')


def pil_imports(path):
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'), str(path))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            names = [node.module or '']
        elif (isinstance(node, ast.Call) and getattr(node.func, 'attr', getattr(node.func, 'id', '')) in
              ('import_module', '__import__') and node.args and isinstance(node.args[0], ast.Constant)):
            names = [str(node.args[0].value)]
        else:
            continue
        found += [f'{path.relative_to(PUBLIC).as_posix()}:{node.lineno} {name}'
                  for name in names if name.split('.')[0] == 'PIL']
    return found


def test_sdk_python_probes_import_no_pil():
    sources = [PUBLIC / name for name in SDK_PYTHON_PROBES]
    sources += sorted((PUBLIC / 'runtime').rglob('*.py')) + sorted((PUBLIC / 'addon').rglob('*.py'))
    assert all(path.is_file() for path in sources[:len(SDK_PYTHON_PROBES)])
    assert [hit for path in sources for hit in pil_imports(path)] == []

