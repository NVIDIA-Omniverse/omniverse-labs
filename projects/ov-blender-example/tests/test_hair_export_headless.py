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

This CPU-only regression also exercises scene reconciliation and custom render
opinion invalidation. Native rendered qualification runs separately; exported
geometry and authored opinions alone do not establish native visual effects.

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
    owner = sg.SceneGenerationOwner(pathlib.Path(out_path).parent / 'generations')
    scene = bpy.context.scene
    emitter_id = sg.blender_id(emitter, 'OBJECT')
    mesh_id = sg.blender_id(emitter.data, 'MESH')
    accepted = owner.replace(scene)
    result['transactions'] = []
    def inspect_generation(label, generation, expected):
        stage = Usd.Stage.Open(generation.usd_path)
        curves = [UsdGeom.BasisCurves(p) for p in stage.Traverse() if p.GetTypeName() == 'BasisCurves']
        assert len(curves) == (0 if expected is None else 1), (label, [str(c.GetPath()) for c in curves])
        widths = [] if not curves else list(curves[0].GetWidthsAttr().Get() or [])
        if expected is not None:
            assert widths and abs(widths[0] - expected) < 1e-6, (label, widths[:1])
            assert UsdGeom.Imageable(curves[0]).ComputeVisibility() == (
                'invisible' if emitter.hide_render else 'inherited'), label
            parent = curves[0].GetPrim().GetParent()
            assert parent.GetAttribute(sg.PARTICLE_EMITTER_NAME_ATTRIBUTE).Get() == emitter.name_full
        result['transactions'].append(dict(label=label, curves=len(curves), root_width=widths[:1],
                                           mode=generation.diagnostics['mode']))
    inspect_generation('initial', accepted, .02)
    def reconcile(label, identities, expected):
        bpy.context.view_layer.update()
        candidate = owner.reconcile(scene, identities)
        assert candidate is not None, label
        inspect_generation(label, candidate, expected)
        owner.accept(candidate)
    st.radius_scale = .1
    reconcile('width_changed', [emitter_id], .1)
    emitter.hide_render = True
    reconcile('hair_hidden', [emitter_id], .1)
    emitter.name = 'Renamed Hair Emitter'
    reconcile('hidden_emitter_renamed', [emitter_id], .1)
    emitter.hide_render = False
    reconcile('hair_restored', [emitter_id], .1)
    # An unrelated sparse replacement must retain accepted particle ownership.
    bpy.ops.mesh.primitive_cube_add(location=(4, 0, 0))
    reconcile('unrelated_added', [sg.blender_id(bpy.context.object, 'OBJECT')], .1)
    st.radius_scale = .07
    reconcile('shared_mesh_notification', [mesh_id], .07)
    emitter.modifiers.remove(next(m for m in emitter.modifiers if m.type == 'PARTICLE_SYSTEM'))
    reconcile('particle_modifier_removed', [emitter_id], None)
    emitter.modifiers.new('HairAgain', type='PARTICLE_SYSTEM')
    st = emitter.particle_systems[0].settings
    st.type = 'HAIR'; st.count = 8; st.hair_length = .5
    st.root_radius = 1.; st.tip_radius = .1; st.radius_scale = .03
    reconcile('particle_modifier_added', [emitter_id], .03)
    bpy.data.objects.remove(emitter, do_unlink=True)
    reconcile('emitter_deleted', [emitter_id], None)
    # Ordinary geometry no-op admission must still author custom render controls.
    from ovrtx_blender_example import authoring_properties
    authoring_properties.register()
    cube = bpy.context.object
    cube_id = sg.blender_id(cube, 'OBJECT')
    result['matte_transactions'] = []
    result['candidate_admission_cache'] = []
    for field, attr in [('matte', 'primvars:isMatteObject'), ('holdout', 'primvars:holdoutObject')]:
        for value in ('ON', 'OFF', 'INHERIT'):
            setattr(cube.ov.rendering, field, value)
            bpy.context.view_layer.update()
            candidate = owner.reconcile(scene, [cube_id])
            assert candidate is not None, (field, value, 'dropped opinion edit')
            stage = Usd.Stage.Open(candidate.usd_path)
            prim = stage.GetPrimAtPath(candidate.blender_prim_paths[cube_id].schema_path)
            authored = prim.GetAttribute(attr)
            expected = None if value == 'INHERIT' else value == 'ON'
            actual = authored.Get() if authored else None
            assert actual == expected, (field, value, actual)
            assert bool(candidate.unadmitted_rendering_overrides) == (value != 'INHERIT')
            owner.accept(candidate)
            result['matte_transactions'].append(dict(field=field, value=value, actual=actual))
            if value != 'INHERIT':
                bpy.ops.mesh.primitive_cube_add(location=(5, 0, 0))
                unrelated = bpy.context.object
                unrelated_id = sg.blender_id(unrelated, 'OBJECT')
                for edit in ('add', 'visibility', 'delete'):
                    if edit == 'visibility':
                        unrelated.hide_render = True
                    if edit == 'delete':
                        bpy.data.objects.remove(unrelated, do_unlink=True)
                    bpy.context.view_layer.update()
                    candidate = owner.reconcile(scene, [unrelated_id])
                    assert candidate is not None
                    assert candidate.unadmitted_rendering_overrides == (cube.name,)
                    owner.accept(candidate)
                    result['candidate_admission_cache'].append(dict(field=field, value=value, unrelated_edit=edit))
    owner.close()
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
    assert proc.returncode == 0, f"Blender exited {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    if not out.is_file():
        raise AssertionError(
            f"driver produced no result.json\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return json.loads(out.read_text(encoding="utf-8"))


def test_particle_hair_exports_ovrtx_correct_basis_curves(tmp_path: Path) -> None:
    result = _run_headless(tmp_path)
    assert not result["errors"], result.get("traceback", result["errors"])
    assert result["export_created"] is True
    assert [case['label'] for case in result['transactions']] == [
        'initial', 'width_changed', 'hair_hidden', 'hidden_emitter_renamed', 'hair_restored',
        'unrelated_added', 'shared_mesh_notification',
        'particle_modifier_removed', 'particle_modifier_added', 'emitter_deleted']
    assert len(result['matte_transactions']) == 6
    assert len(result['candidate_admission_cache']) == 12
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
