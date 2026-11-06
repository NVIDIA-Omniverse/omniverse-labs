"""Fresh stock Blender, no addon import/registration and no physics worker."""
import argparse
import json
import math
from pathlib import Path
import sys
import bpy
from mathutils import Matrix, Vector
p = argparse.ArgumentParser()
p.add_argument("--directory", type=Path, required=True)
a = p.parse_args(sys.argv[sys.argv.index("--") + 1:])
oracle = json.loads((a.directory / "offline-oracle.json").read_text())
bpy.ops.wm.open_mainfile(filepath=str(a.directory / "baked.blend"))
assert not hasattr(bpy.types.Scene, "ov_physics_bake")
assert "ovrtx_blender_example" not in sys.modules
scene = bpy.context.scene
checks = 0
for sample in oracle["frames"]:
    scene.frame_set(sample["frame"])
    bpy.context.view_layer.update()
    for row in oracle["owners"]:
        obj = bpy.data.objects[row["object"]]
        world = obj.matrix_world @ obj.pose.bones[row["bone"]].matrix if row["bone"] else obj.matrix_world
        desired = Matrix(sample["worlds"][row["path"]])
        assert (world.translation - desired.translation).length * row["units"] <= 1e-4
        qa, qb = world.to_quaternion(), desired.to_quaternion()
        dot = abs(sum(float(x) * float(y) for x, y in zip(qa, qb)))
        norm = math.sqrt(sum(float(x)**2 for x in qa) * sum(float(x)**2 for x in qb))
        assert 2 * math.acos(min(1., dot / norm)) <= 1e-4
        assert obj.animation_data.action
        checks += 1
    skin = bpy.data.objects[oracle["skin"]].evaluated_get(bpy.context.evaluated_depsgraph_get())
    actual = skin.matrix_world @ skin.data.vertices[0].co
    assert (actual - Vector(sample["vertex"])).length <= 1e-4
    checks += 1
# Durable keys remain ordinary editable FCurve points in installed-version slots.
obj = bpy.data.objects[oracle["owners"][0]["object"]]
from bpy_extras.anim_utils import action_get_channelbag_for_slot
bag = action_get_channelbag_for_slot(obj.animation_data.action, obj.animation_data.action_slot)
point = bag.fcurves[0].keyframe_points[0]
old = point.co.y
point.co.y += .25
assert point.co.y != old and point.interpolation == "LINEAR"
print("PASS addon-disabled offline Actions/skin", checks, flush=True)
