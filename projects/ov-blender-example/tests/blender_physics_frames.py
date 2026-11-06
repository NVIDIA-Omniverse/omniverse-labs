# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline --background --factory-startup --python test; no runtime required."""
import json
import math
from pathlib import Path
import sys
import threading

import bpy
from mathutils import Matrix, Quaternion, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example.physics_frames import (
    FrameError, PhysicsUnits, RigidFrame, evaluated_owner_frame,
    evaluated_owner_transform, local_frame,
    normalize_stock_export_units,
)


def close(a, b, tolerance=2e-5):
    assert max(abs(x-y) for x, y in zip(a, b)) < tolerance, (a, b)


def reject(fn, text):
    try:
        fn()
    except FrameError as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError(f"expected {text}")


scene = bpy.context.scene
obj = bpy.data.objects.new("FrameTest", None)
scene.collection.objects.link(obj)
obj.location = (2, 3, 4)
obj.rotation_euler = (.3, .2, .7)
obj.scale = (2, 2, 2)
offset = RigidFrame((.5, .1, -.2), (1, .3, .2, .1))
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
snapshot = evaluated_owner_transform(obj, depsgraph, offset=offset)
expected = obj.matrix_world @ Matrix(offset.column_matrix())
close(snapshot.frame.position, expected.translation)
assert abs(Quaternion(snapshot.frame.rotation_wxyz).dot(expected.to_quaternion())) > 1-1e-5
assert abs(snapshot.uniform_scale-2) < 1e-5
obj.location.x += 1
bpy.context.view_layer.update()
assert snapshot.frame.position != evaluated_owner_frame(obj, depsgraph, offset=offset).position

# Physical equivalence when all Blender distances are in centimeters.
meters = evaluated_owner_frame(obj, depsgraph, offset=offset)
obj.location *= 100
cm_offset = RigidFrame(tuple(x*100 for x in offset.position), offset.rotation_wxyz)
bpy.context.view_layer.update()
centimeters = evaluated_owner_frame(obj, depsgraph, offset=cm_offset, units=PhysicsUnits(.01))
close(meters.position, centimeters.position)
close(meters.rotation_wxyz, centimeters.rotation_wxyz)

rig_data = bpy.data.armatures.new("FramesRig")
rig = bpy.data.objects.new("FramesRig", rig_data)
scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
parent = rig_data.edit_bones.new("Parent")
parent.head, parent.tail = (0, 0, 0), (0, 1, 0)
child = rig_data.edit_bones.new("Child")
child.head, child.tail = (0, 1, 0), (0, 2, 0)
child.parent = parent
bpy.ops.object.mode_set(mode="OBJECT")
rig.location = (3, 4, 5)
rig.rotation_euler = (.2, -.3, .8)
rig.scale = (1.5, 1.5, 1.5)
rig.pose.bones["Parent"].scale = (2, 2, 2)
rig.pose.bones["Parent"].rotation_mode = "XYZ"
rig.pose.bones["Parent"].rotation_euler.z = .4
rig.pose.bones["Child"].rotation_mode = "XYZ"
rig.pose.bones["Child"].rotation_euler.x = .3
tested = []
for mode in ("FULL", "NONE"):
    rig_data.bones["Child"].inherit_scale = mode
    bpy.context.view_layer.update()
    graph = bpy.context.evaluated_depsgraph_get()
    evaluated = rig.evaluated_get(graph)
    expected = evaluated.matrix_world @ evaluated.pose.bones["Child"].matrix @ Matrix(offset.column_matrix())
    result = evaluated_owner_transform(rig, graph, bone_name="Child", offset=offset)
    close(result.frame.position, expected.translation)
    assert abs(Quaternion(result.frame.rotation_wxyz).dot(expected.to_quaternion())) > 1-1e-5
    anchor = RigidFrame((8, 7, 6), (1, .2, .3, .4))
    close(result.frame.compose(local_frame(result.frame, anchor)).position, anchor.position)
    tested.append(mode)
rig_data.bones["Child"].inherit_scale = "FIX_SHEAR"
bpy.context.view_layer.update()
reject(lambda: evaluated_owner_frame(rig, bpy.context.evaluated_depsgraph_get(), bone_name="Child"), "inherit-scale")
rig_data.bones["Child"].inherit_scale = "FULL"
rig.scale = (1, 2, 1)
bpy.context.view_layer.update()
reject(lambda: evaluated_owner_frame(rig, bpy.context.evaluated_depsgraph_get(), bone_name="Child"), "sheared")
obj.scale = (-1, -1, 1)
bpy.context.view_layer.update()
reject(lambda: evaluated_owner_frame(obj, depsgraph), "negative")
errors = []
def worker():
    try:
        evaluated_owner_frame(obj, depsgraph)
    except FrameError as error:
        errors.append(str(error))
thread = threading.Thread(target=worker)
thread.start()
thread.join()
assert errors and "main thread" in errors[0]
assert scene.rigidbody_world is None
assert all(o.rigid_body is None and o.rigid_body_constraint is None for o in scene.objects)
assert not rig.constraints and not any(b.constraints for b in rig.pose.bones)

# Verify the actual stock exporter convention, including Gf row-vector layout.
from pxr import Gf, Usd, UsdGeom
artifact_dir = (Path(sys.argv[sys.argv.index('--') + 1]).resolve() if '--' in sys.argv
                else Path(__file__).resolve().parents[4] / "_build" / "ovphysx-frames")
artifact_dir.mkdir(parents=True, exist_ok=True)
mesh = bpy.data.meshes.new("UnitsMesh")
mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
export_obj = bpy.data.objects.new("UnitsMesh", mesh)
scene.collection.objects.link(export_obj)
for selected in list(bpy.context.selected_objects):
    selected.select_set(False)
export_obj.select_set(True)
bpy.context.view_layer.objects.active = export_obj
positions = []
for scale in (1.0, .01):
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = scale
    export_obj.location = (2/scale, 3/scale, 4/scale)
    bpy.context.view_layer.update()
    path = artifact_dir / f"units-{scale}.usda"
    bpy.ops.wm.usd_export(filepath=str(path), selected_objects_only=True,
                          convert_scene_units="METERS", export_animation=False)
    stage = Usd.Stage.Open(str(path))
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1
    prim = next(p for p in stage.Traverse() if p.GetName() == "UnitsMesh" and p.IsA(UsdGeom.Xform))
    # Evidence of stock behavior before add-on correction: raw Blender units.
    raw = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    close(raw.ExtractTranslation(), export_obj.location)
    source_bytes = path.read_bytes()
    normalize_stock_export_units(stage, PhysicsUnits.from_scene(scene))
    normalized_once = stage.GetRootLayer().ExportToString()
    normalize_stock_export_units(stage, PhysicsUnits.from_scene(scene))
    assert stage.GetRootLayer().ExportToString() == normalized_once
    assert path.read_bytes() == source_bytes
    reject(lambda: normalize_stock_export_units(stage, PhysicsUnits(scale*2)), "different scene scale")
    stage.GetRootLayer().Export(str(artifact_dir / f"normalized-{scale}.usda"))
    world = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    frame = evaluated_owner_frame(export_obj, bpy.context.evaluated_depsgraph_get(), units=PhysicsUnits.from_scene(scene))
    close(world.ExtractTranslation(), frame.position)
    gf = Gf.Matrix4d(*[v for row in frame.gf_matrix() for v in row])
    close(gf.Transform(Gf.Vec3d(0)), frame.position)
    positions.append(tuple(world.ExtractTranslation()))
close(*positions)
print(json.dumps({"ok": True, "blender": bpy.app.version_string, "inherit_scale": tested,
                  "meter_centimeter_equivalent": True, "body_offsets": True,
                  "copied_evaluated_frames": True, "no_blender_physics": True,
                  "stock_usd_meters": True, "usd_artifacts": str(artifact_dir)}))
