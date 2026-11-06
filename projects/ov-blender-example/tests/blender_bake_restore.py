"""Previous shared Action and exact slot survive save/reload and deactivation."""
from pathlib import Path
import sys
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props, physics_bake_ui as ui
from ovrtx_blender_example.physics_bake_actions import deactivate
props.register()
ui.register()
bpy.ops.wm.open_mainfile(filepath=sys.argv[sys.argv.index("--") + 1])
scene = bpy.context.scene
assert scene.ov_physics_bake.active
record = scene.ov_physics_bake.records[0]
previous, slot = record.previous_action, record.previous_slot
assert previous and previous == bpy.data.objects["SharedActionUser"].animation_data.action
record.owner.pose.bones["Bone"].name = "RenamedAfterLoad"
deactivate(scene)
assert not scene.ov_physics_bake.active
assert record.owner.animation_data.action == previous
assert record.owner.animation_data.action_slot.identifier == slot
assert bpy.data.objects["SharedActionUser"].animation_data.action == previous
scene.frame_set(2)
assert abs(record.owner.location.x - .2) < 1e-5
assert abs(record.owner.pose.bones["Unaffected"].location.z - .4) < 1e-5
print("PASS saved original Action/slot/shared user and renamed-bone deactivation", flush=True)
