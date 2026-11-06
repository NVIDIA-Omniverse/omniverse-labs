# SPDX-License-Identifier: Apache-2.0
"""Admitted shapes: copied authoring/runtime geometry in actual Blender units."""
import json
import math
from pathlib import Path
import sys
import uuid
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example.physics_authoring_snapshot import capture, geometry, PresentationPoses
from ovrtx_blender_example.physics_endpoints import initialize_owner
from ovrtx_blender_example.physics_frames import RigidFrame

props.register()
scene = bpy.context.scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
checks = []

def check(label, value):
    assert value, label
    checks.append(label)

def close(a, b):
    return max(abs(x-y) for x, y in zip(a, b)) < 2e-5

for unit in (1., .01):
    scene.unit_settings.scale_length = unit
    for bone in (False, True):
        if bone:
            bpy.ops.object.armature_add()
            obj = bpy.context.object
            bpy.ops.object.mode_set(mode="EDIT")
            obj.data.edit_bones[0].tail = (0, 1, 0)
            bpy.ops.object.mode_set(mode="OBJECT")
            target = initialize_owner(obj, obj.data.bones[0])
        else:
            obj = bpy.data.objects.new("Body", None)
            scene.collection.objects.link(obj)
            target = initialize_owner(obj)
        obj.location = (10, 20, 30)
        obj.scale = (2, 2, 2)
        physics = target.ov.physics
        physics.rigid_body.schema_opinion = "APPLY"
        physics.rigid_body.offset.translation = (1, 0, 0)
        identity = obj.ov.physics.persistent_id + ("_" + target.bone.ov.persistent_id if bone else "")
        for shape in ("BOX", "CAPSULE"):
            record = physics.colliders.add()
            record.name = record.persistent_id = uuid.uuid4().hex
            record.shape, record.schema_opinion = shape, "APPLY"
            record.dimensions = (1, 2, 3)
            record.radius_m, record.height_m = .2, .6
            record.offset.translation = (0, 1, 0)
            record.offset.rotation = (math.sqrt(.5), 0, math.sqrt(.5), 0)
        def snap(presentation=None):
            bpy.context.view_layer.update()
            return capture(scene, bpy.context.view_layer, bpy.context.evaluated_depsgraph_get(), generation=8, presentation=presentation)
        def world(source, anchor, frame):
            # Snapshot frames are local to their anchor; Source.anchors holds the world frames.
            frames = {a.identity: a.frame for a in source.anchors}
            return frame if anchor is None else frames[anchor].compose(frame)
        source = snap()
        copied = source.colliders
        check("admitted shapes", len(copied) == 2 and all(not c.error for c in copied))
        check("owner and collider scaled offsets", all(close(world(source, c.anchor, c.frame).position, (12, 22, 30)) for c in copied))
        check("box dimensions use owner BU scale", close(copied[0].dimensions, (2, 4, 6)))
        check("capsule absolute meters to BU", abs(copied[1].radius-.2/unit)<2e-5 and abs(copied[1].height-.6/unit)<2e-5)
        geo = geometry(source)
        wires = [s for s in geo.strokes if s.role == "collider-wire"]
        check("both visible wires", len(wires) == 2 and not geo.labels)
        center = world(source, copied[0].anchor, copied[0].frame).position
        for wire, half in zip(wires, ((3, 2, 1), (.5/unit, .2/unit, .2/unit))):
            frame = world(source, wire.anchor, wire.local_frame)
            points = [frame.compose(RigidFrame(p)).position for p in wire.vertices]
            check("rotated numeric bounds", all(close((min(p[i] for p in points), max(p[i] for p in points)), (center[i]-half[i], center[i]+half[i])) for i in range(3)))
        runtime = snap(PresentationPoses(8, ((identity, RigidFrame((1, 2, 3))),)))
        check("runtime pose in BU keeps scaled local offset", all(close(world(runtime, c.anchor, c.frame).position, (1/unit, 2/unit+2, 3/unit)) for c in runtime.colliders))
        check("missing runtime diagnosed", all("missing" in c.error for c in snap(PresentationPoses(8)).colliders))
        physics.rigid_body.role = "STATIC"
        static = snap(PresentationPoses(8))
        check("static needs no runtime pose", all(not c.error and world(static, c.anchor, c.frame) == world(source, old.anchor, old.frame) for c, old in zip(static.colliders, copied)))
        physics.colliders[0].dimensions = (0, 2, 3)
        physics.colliders[1].radius_m = 0
        invalid = snap()
        check("invalid dimension diagnostics", all(c.error for c in invalid.colliders) and len([s for s in geometry(invalid).strokes if s.role == "collider-error"]) == 2)
        check("prior snapshot immutable", geometry(source) == geo)
        physics.colliders[1].radius_m = .2
        physics.colliders[1].height_m = 0
        check("zero cylinder is valid sphere capsule", not snap().colliders[1].error)
        obj.hide_set(True)
        check("hidden shapes removed", not snap().colliders)
        bpy.data.objects.remove(obj, do_unlink=True)

out = Path(sys.argv[sys.argv.index("--")+1])
out.mkdir(parents=True, exist_ok=True)
(out / "results.json").write_text(json.dumps({"checks": checks}, indent=2))
print("EXPANDED_INDICATORS_PASS", len(checks))
