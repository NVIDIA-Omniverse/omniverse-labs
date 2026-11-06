"""Actual modern Action and NLA slot ownership conflict regression."""
import json
from pathlib import Path
import sys
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example.physics_pose_presentation import validate_output

out = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
out.mkdir(parents=True, exist_ok=True)
checks = []
def check(label, condition):
    assert condition, label
    checks.append(label)
def rejected(label):
    try:
        validate_output(owned, owned)
    except ValueError as error:
        check(label, "animation or drivers conflict" in str(error))
    else:
        raise AssertionError(label)

owned = bpy.data.objects.new("PhysicsOutput", None)
other = bpy.data.objects.new("OtherAnimated", None)
for obj in (owned, other):
    bpy.context.scene.collection.objects.link(obj)
action = bpy.data.actions.new("SharedMultiSlot")
owned_slot = action.slots.new(id_type="OBJECT", name=owned.name)
other_slot = action.slots.new(id_type="OBJECT", name=other.name)
strip = action.layers.new("Layer").strips.new(type="KEYFRAME")
own_bag, other_bag = strip.channelbags.new(owned_slot), strip.channelbags.new(other_slot)
other_curve = other_bag.fcurves.new(data_path="location", index=0)
other_curve.keyframe_points.insert(1, 7)
for obj, slot in ((owned, owned_slot), (other, other_slot)):
    obj.animation_data_create()
    obj.animation_data.action = action
    obj.animation_data.action_slot = slot
bpy.context.scene.frame_set(1)
check("shared slots evaluate independently", owned.location.x == 0 and other.location.x == 7)
validate_output(owned, owned)
checks.append("unrelated active Action slot admitted")
own_curve = own_bag.fcurves.new(data_path="location", index=0)
own_curve.keyframe_points.insert(1, 2)
rejected("own active Action slot rejected")
own_bag.fcurves.remove(own_curve)
owned.animation_data.action = None
track = owned.animation_data.nla_tracks.new()
nla = track.strips.new("Shared NLA", 1, action)
nla.action_slot = owned_slot
validate_output(owned, owned)
checks.append("NLA bound empty slot admitted despite unrelated animation")
nla.action_slot = other_slot
bpy.context.scene.frame_set(1)
check("NLA mapped slot evaluates on output", owned.location.x == 7)
rejected("NLA mapped animated slot rejected")
nla.action_slot = owned_slot
own_curve = own_bag.fcurves.new(data_path="location", index=1)
own_curve.keyframe_points.insert(1, 3)
rejected("own NLA slot curve rejected")
track.mute = True
validate_output(owned, owned)
checks.append("muted NLA remains inactive")
driver = owned.driver_add("location", 2)
driver.driver.expression = "1"
rejected("owner drivers remain independently rejected")
check("unrelated user's binding and keys preserved", other.animation_data.action is action
      and other.animation_data.action_slot == other_slot and other_curve.keyframe_points[0].co.y == 7)
(out / "results.json").write_text(json.dumps({"checks": checks}, indent=2))
print("PRESENTATION_ACTION_SLOTS_OK", len(checks), flush=True)
