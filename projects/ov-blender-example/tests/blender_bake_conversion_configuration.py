"""Post-capture edits reject; animated delta/ancestor evaluation stays valid."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import bpy
from mathutils import Matrix
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example.physics_endpoints import initialize_owner
from ovrtx_blender_example.physics_frames import RigidFrame
from ovrtx_blender_example.physics_presentation_targets import PhysicsPresentationTarget
from ovrtx_blender_example.physics_bake_capture import _conversion_context, _SampleWriter, BakeCaptureError
from ovrtx_blender_example.physics_bake_conversion import convert_capture
props.register()
scene = bpy.context.scene
root = Path(tempfile.mkdtemp(prefix="ov-bake-configuration-"))
checks = 0

def make(case, animated=False):
    objects = [bpy.data.objects.new(case + suffix, None) for suffix in ("Root", "Middle", "Output")]
    for obj in objects:
        scene.collection.objects.link(obj)
        initialize_owner(obj)
    parent, middle, obj = objects
    middle.parent, obj.parent = parent, middle
    obj.rotation_mode = "QUATERNION" if case == "quaternion" else "XYZ"
    if animated:
        for frame in (1, 3):
            obj.delta_location.x = frame * .4
            obj.delta_scale = (1 + frame * .1,) * 3
            obj.delta_rotation_euler.z = frame * .1
            for name in ("delta_location", "delta_scale", "delta_rotation_euler"):
                obj.keyframe_insert(data_path=name, frame=frame)
            middle.location.y = frame * .2
            middle.keyframe_insert(data_path="location", frame=frame)
    scene.frame_set(1)
    bpy.context.view_layer.update()
    target = PhysicsPresentationTarget("/" + case, obj.ov.physics.persistent_id, obj.session_uid,
        obj.ov.physics.persistent_id, "", "", "DYNAMIC", True, 1., 1., RigidFrame(), RigidFrame())
    directory = root / (case + "-capture")
    directory.mkdir()
    writer = _SampleWriter(directory, (target.body_path,))
    for frame in (1, 2, 3):
        scene.frame_set(frame)
        writer.append(frame, frame - 1, [SimpleNamespace(prim_path=target.body_path,
            translate=(2., 0., 0.), orient=(0., 0., 0., 1.))], _conversion_context(scene, (target,)), 1)
    artifact = writer.finish(dict(targets=[asdict(target)], output_paths=[target.body_path], timing={}))
    return parent, middle, obj, target, artifact

for case in ("delta-location", "delta-scale", "delta-euler", "quaternion", "ancestor-inverse",
             "ancestor-location", "ancestor-delta", "ancestor-identity", "animated-definition"):
    parent, middle, obj, target, capture = make(case, animated=case == "animated-definition")
    if case == "delta-location": obj.delta_location.x += 3
    elif case == "delta-scale": obj.delta_scale.x += 1
    elif case == "delta-euler": obj.delta_rotation_euler.z += .3
    elif case == "quaternion": obj.delta_rotation_quaternion.z = .3
    elif case == "ancestor-inverse": middle.matrix_parent_inverse = Matrix.Translation((3, 0, 0))
    elif case == "ancestor-location": middle.location.x += 3
    elif case == "ancestor-delta": middle.delta_location.x += 3
    elif case == "ancestor-identity": middle.ov.physics.persistent_id = "changed"
    else:
        from ovrtx_blender_example.physics_pose_presentation import _curves
        next(c for c in _curves(obj.animation_data) if c.data_path == "delta_location").keyframe_points[0].co.y += 3
    bpy.context.view_layer.update()
    output = root / (case + "-converted")
    try:
        convert_capture(scene, capture, output)
        raise AssertionError(case + " edit accepted")
    except BakeCaptureError:
        assert not output.exists()
    checks += 1

parent, middle, obj, target, capture = make("animated", animated=True)
for current in (1, 2, 3):
    scene.frame_set(current)
    converted = convert_capture(scene, capture, root / ("animated-converted-" + str(current)))
    assert scene.frame_current == current
    for row in converted.samples():
        scene.frame_set(row["frame"])
        for name, value in row["channels"][target.body_path].items():
            setattr(obj, name, value)
        bpy.context.view_layer.update()
        assert (obj.matrix_world.translation - Matrix.Translation((2, 0, 0)).translation).length <= 1e-4
        assert obj.matrix_world.to_quaternion().angle <= 1e-4
        checks += 1
print(f"PASS configuration {checks} checks; artifacts {root}", flush=True)
