# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only syntax and asset-notice check for the self-contained source edition."""
import ast
from pathlib import Path
root=Path(__file__).resolve().parents[1]
count=0
for folder in ('source','tools','scripts'):
    for path in (root/folder).rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path));count+=1
for path in (root/'source/nvidia/ov_library_lab').glob('*.py'):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    for node in ast.walk(tree):
        names=([item.name for item in node.names] if isinstance(node,ast.Import)
               else [node.module or ''] if isinstance(node,ast.ImportFrom) else [])
        assert not any(name=='omni.kit' or name.startswith('omni.kit.') or
                       name in ('omni.usd','omni.physx','omni.ext','omni.appwindow')
                       for name in names), 'Kit runtime import in '+path.name
for path in ('data/models/car/CC-BY-4.0.txt','data/models/irb120/LICENSE.txt',
             'data/models/irb4600/LICENSE.txt','data/surfaces/CC0-1.0.txt'):
    assert (root/path).is_file(),path
print('PASS:',count,'Python files parse; no Kit runtime imports; asset notices present')
