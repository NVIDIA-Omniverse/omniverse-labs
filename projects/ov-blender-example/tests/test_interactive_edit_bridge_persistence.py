# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Headless regression: the live-edit depsgraph bridge survives file loads.

Blender clears non-persistent ``depsgraph_update_post`` handlers on every
file load. The interactive edit bridge is how object transform, light,
camera, and material deltas reach the OVRTX runtime — without the
``@persistent`` tag, opening a .blend silently killed every live edit for
the rest of the session while the viewport kept rendering. This boots real
headless Blender, registers the add-on, loads a file, and proves the
handler is still registered and still fires.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from blender_test_support import blender_executable


ROOT = Path(__file__).resolve().parents[1]


_DRIVER = """
import json
import sys
import traceback

result = {"errors": []}
output_path = sys.argv[sys.argv.index("--") + 1]

try:
    import bpy

    sys.path.insert(0, __ADDON_PATH__)
    import ovrtx_blender_example as addon
    from ovrtx_blender_example import engine

    addon.register()
    handler = engine._live_interactive_edit_depsgraph_handler
    handlers = bpy.app.handlers.depsgraph_update_post
    result["registered_after_register"] = handler in handlers
    # Blender marks persistent handlers by attribute presence (the value
    # is None), so hasattr is the correct probe.
    result["persistent_tagged"] = hasattr(handler, "_bpy_persistent")

    # The user flow that killed the bridge: opening a file after the
    # add-on registered (also covers launching Blender on a .blend).
    bpy.ops.wm.read_homefile(use_empty=True)
    result["registered_after_file_load"] = handler in bpy.app.handlers.depsgraph_update_post

    # The surviving handler still fires on a real depsgraph event.
    before_ns = int(engine.interactive_edit_bridge_diagnostics()["updated_at_ns"])
    bpy.ops.mesh.primitive_cube_add()
    bpy.ops.transform.translate(value=(1.0, 0.0, 0.0))
    diagnostics = engine.interactive_edit_bridge_diagnostics()
    result["handler_fired_after_file_load"] = (
        int(diagnostics["updated_at_ns"]) > before_ns
    )
    result["bridge_last_error"] = str(diagnostics.get("last_error", ""))

    addon.unregister()
    result["registered_after_unregister"] = (
        handler in bpy.app.handlers.depsgraph_update_post
    )
except Exception:
    result["errors"].append(traceback.format_exc())

with open(output_path, "w", encoding="utf-8") as stream:
    json.dump(result, stream)
"""


def test_live_edit_bridge_survives_file_load(tmp_path: Path) -> None:
    blender = blender_executable()
    if blender is None:
        pytest.skip("no Blender executable available for headless bridge test")

    driver_source = _DRIVER.replace("__ADDON_PATH__", repr(str(ROOT / "addon")))
    driver = tmp_path / "bridge_persistence_driver.py"
    driver.write_text(driver_source, encoding="utf-8")
    output = tmp_path / "result.json"

    completed = subprocess.run(
        (
            str(blender),
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(driver),
            "--",
            str(output),
        ),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert output.is_file(), completed.stdout + completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))

    assert result["errors"] == []
    assert result["registered_after_register"] is True
    assert result["persistent_tagged"] is True
    # The load-bearing assertion: a file load must not strip the bridge.
    assert result["registered_after_file_load"] is True
    assert result["handler_fired_after_file_load"] is True
    # Unregister still removes it cleanly.
    assert result["registered_after_unregister"] is False
