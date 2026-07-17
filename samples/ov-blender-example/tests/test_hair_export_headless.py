# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Headless regression: particle hair exports as OVRTX-correct BasisCurves.

Spec hair-rendering, Phase 1 (tasks 01-02/01-03): with a real Blender document,
a particle HAIR system with Simple children converts to a Curves object and the
add-on's ``_attach_particle_hair_export_modifier`` drives Blender's native USD
export to emit a ``BasisCurves`` prim that meets the OVRTX contract:

- ``type = cubic``, ``basis = catmullRom`` (the normalized target; the Set Spline
  Type node keeps catmullRom after resampling, which otherwise drops to linear),
- one width per point with ``vertex`` interpolation (a sibling
  ``widths:interpolation`` attribute is silently ignored by OVRTX),
- child-strand density (more curves than parent particles),
- per-strand resolution derived from ``render_step`` (``2**render_step + 1``),
- physical root->tip tapered widths from the shared ``curve_widths`` module.

Rendered-hair confirmation on a real OVRTX runtime is not possible on this
machine (no GPU/worker bundle); the exported-geometry assertions here are the
recorded evidence, with on-GPU render validation tracked as Phase 1 task01-05.

Runs headless Blender when an executable is available (``BLENDER_COMMAND`` env
var or a known install location); skips otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from blender_test_support import blender_executable

ROOT = Path(__file__).resolve().parents[1]


_DRIVER = r'''
import json, os, sys, tempfile, traceback
result = {"errors": [], "steps": []}
out_path = sys.argv[sys.argv.index("--") + 1]
try:
    import bpy
    sys.path.insert(0, __ADDON_PATH__)
    from ovrtx_blender_example import scene_generation as sg

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=1.0)
    emitter = bpy.context.object
    emitter.modifiers.new("Hair", type="PARTICLE_SYSTEM")
    st = emitter.particle_systems[0].settings
    st.type = "HAIR"; st.count = 8; st.hair_length = 0.5
    st.root_radius = 1.0; st.tip_radius = 0.1; st.radius_scale = 0.02
    st.child_type = "SIMPLE"; st.child_percent = 4; st.rendered_child_count = 4
    st.render_step = 4  # expect 2**4 + 1 = 17 points per strand
    result["steps"].append("scene_built")

    result["parent_particles"] = len(emitter.evaluated_get(
        bpy.context.evaluated_depsgraph_get()).particle_systems[0].particles)

    out = os.path.join(tempfile.gettempdir(), "hair_export_headless.usda")
    import pathlib
    sg._stock_export(bpy.context.scene, pathlib.Path(out))
    result["export_created"] = os.path.isfile(out)
    result["steps"].append("stock_exported")

    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(out)
    for prim in stage.Traverse():
        if prim.GetTypeName() == "BasisCurves":
            bc = UsdGeom.BasisCurves(prim)
            vc = list(bc.GetCurveVertexCountsAttr().Get() or [])
            w = bc.GetWidthsAttr().Get() or []
            pts = bc.GetPointsAttr().Get() or []
            result["type"] = str(bc.GetTypeAttr().Get())
            result["basis"] = str(bc.GetBasisAttr().Get())
            result["wrap"] = str(bc.GetWrapAttr().Get())
            result["num_curves"] = len(vc)
            result["points_per_curve"] = int(vc[0]) if vc else 0
            result["num_points"] = len(pts)
            result["widths_len"] = len(w)
            result["widths_interp"] = str(bc.GetWidthsInterpolation())
            result["width_root"] = round(float(w[0]), 6) if w else 0.0
            result["width_tip"] = round(float(w[int(vc[0]) - 1]), 6) if (w and vc) else 0.0
            result["points_match_counts"] = (len(pts) == sum(vc))
            pw = prim.GetAttribute("primvars:widths")
            has_pw = bool(pw and pw.HasAuthoredValue())
            result["has_primvar_widths"] = has_pw
            result["primvar_widths_len"] = len(pw.Get() or []) if has_pw else 0
            result["primvar_widths_interp"] = (
                str(UsdGeom.Primvar(pw).GetInterpolation()) if has_pw else "")
            break

    result["steps"].append("inspected")
except Exception as exc:
    result["errors"].append("%s: %s" % (type(exc).__name__, exc))
    result["traceback"] = traceback.format_exc()
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(result, fh)
'''


def _run_headless(tmp_path: Path) -> dict:
    blender = blender_executable()
    if blender is None:
        pytest.skip("no Blender executable available")
    driver = tmp_path / "driver.py"
    driver.write_text(
        _DRIVER.replace("__ADDON_PATH__", repr(str(ROOT / "addon"))),
        encoding="utf-8",
    )
    out = tmp_path / "result.json"
    proc = subprocess.run(
        [
            str(blender),
            "--background",
            "--factory-startup",
            "--python",
            str(driver),
            "--",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if not out.is_file():
        raise AssertionError(
            f"driver produced no result.json\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return json.loads(out.read_text(encoding="utf-8"))


def test_particle_hair_exports_ovrtx_correct_basis_curves(tmp_path: Path) -> None:
    result = _run_headless(tmp_path)
    assert not result["errors"], result.get("traceback", result["errors"])
    assert result["export_created"] is True
    # OVRTX renders hair only as cubic catmullRom nonperiodic with per-point
    # vertex widths (the known-good Junk Shop fixture config; pinned wrap and
    # linear type render invisible).
    assert result["type"] == "cubic"
    assert result["basis"] == "catmullRom"
    assert result["wrap"] == "nonperiodic"
    assert result["widths_interp"] == "vertex"
    # primvars:widths is mirrored alongside the builtin widths attribute (vertex,
    # same length) for RTX/Hydra paths that read the primvar form.
    assert result["has_primvar_widths"] is True
    assert result["primvar_widths_len"] == result["num_points"]
    assert result["primvar_widths_interp"] == "vertex"
    assert result["widths_len"] == result["num_points"]
    assert result["points_match_counts"] is True
    # Child density: more exported curves than parent particles.
    assert result["num_curves"] > result["parent_particles"]
    # render_step resolution: 2**4 + 1 = 17 points per strand.
    assert result["points_per_curve"] == 17
    # Physical tapered widths (root wider than tip, both sub-millimetre-ish).
    assert result["width_root"] > result["width_tip"] > 0.0
    assert result["width_root"] == pytest.approx(0.02, rel=1e-3)
