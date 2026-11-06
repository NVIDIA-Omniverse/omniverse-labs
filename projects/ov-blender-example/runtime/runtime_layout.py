# SPDX-License-Identifier: Apache-2.0
"""Locate stdlib helpers in source and installed extension layouts."""
from pathlib import Path


def addon_helper_path(runtime_file, filename):
    """Return a sibling addon helper without importing its Blender package."""
    if Path(filename).name != filename or not filename.endswith('.py'):
        raise ValueError('Helper must be a Python filename')
    root = Path(runtime_file).resolve().parents[1]
    # ZIP: extension/{runtime,ovrtx_blender_example}; source:
    # public/{runtime,addon/ovrtx_blender_example}.
    for package in (root/'ovrtx_blender_example', root/'addon'/'ovrtx_blender_example'):
        candidate = package/filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'Missing packaged runtime helper {filename} under {root}')
