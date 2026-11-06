# SPDX-License-Identifier: Apache-2.0
"""Real evaluated, immutable authoring/presentation indicator snapshot tests."""
import json
from pathlib import Path
import sys
import threading
import bpy
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example.physics_authoring import apply_anchor
from ovrtx_blender_example.physics_endpoints import initialize_owner, initialize_joint
from ovrtx_blender_example.physics_frames import RigidFrame
from ovrtx_blender_example.physics_authoring_snapshot import capture, geometry, PresentationPoses, SnapshotError

props.register()
scene = bpy.context.scene
scene.unit_settings.scale_length = .01
bpy.ops.mesh.primitive_cube_add(location=(2, 3, 4))
body = bpy.context.object
body.name = "IndicatorBody"
body.rotation_euler.z = .6
body.scale = (2, 2, 2)
initialize_owner(body)
physics = body.ov.physics
physics.rigid_body.schema_opinion = "APPLY"
physics.rigid_body.offset.translation = (.3, .2, .1)
sphere = physics.colliders.add()
sphere.name = sphere.persistent_id = "1"*32
sphere.schema_opinion, sphere.shape, sphere.radius_m = "APPLY", "ANALYTIC_SPHERE", .4
hull = physics.colliders.add()
hull.name = hull.persistent_id = "2"*32
hull.schema_opinion = "APPLY"
handle = bpy.data.objects.new("IndicatorJoint", None)
scene.collection.objects.link(handle)
handle.location = (3, 4, 5)
joint = initialize_joint(handle)
joint.enabled = True
joint.endpoint_a.object = body
joint.endpoint_b.kind = "WORLD"
bpy.context.view_layer.update()
apply_anchor(handle, scene, bpy.context.evaluated_depsgraph_get())
checks = []
def check(name, value):
    assert value, name
    checks.append(name)
def snapshot(**kwargs):
    bpy.context.view_layer.update()
    return capture(scene, bpy.context.view_layer, bpy.context.evaluated_depsgraph_get(), generation=7, **kwargs)
def close(a, b):
    return max(abs(x-y) for x, y in zip(a, b)) < 1e-5
def world(source, anchor, frame):
    # Snapshot frames are local to their anchor; Source.anchors holds the world frames.
    frames = {a.identity: a.frame for a in source.anchors}
    return frame if anchor is None else frames[anchor].compose(frame)
def frame_a(source):
    return world(source, source.joints[0].anchor_a, source.joints[0].frame_a)

source = snapshot()
check("committed frames reconstruct anchor", close(frame_a(source).position, (3, 4, 5)) and close(source.joints[0].frame_b.position, (3, 4, 5)))
check("meter sphere in centimeter viewport", abs(source.colliders[0].radius-40) < 1e-4)
check("evaluated convex hull copied", len(source.colliders[1].wire) >= 12)
check("capture does not materialize mesh source", not hull.is_property_set("mesh_source"))
prior_vertices = geometry(source).strokes
handle.location.x += 2
pending = snapshot()
check("pending marker copied", close(pending.joints[0].pending.position, (5, 4, 5)))
check("pending leaves committed frames", close(frame_a(pending).position, (3, 4, 5)))
check("old snapshot unaffected by Blender mutation", source.joints[0].pending is None and geometry(source).strokes == prior_vertices)
before_handle = tuple(tuple(row) for row in handle.matrix_world)
presentation = PresentationPoses(7, ((body.ov.physics.persistent_id, RigidFrame((10, 0, 0))),))
runtime = snapshot(presentation=presentation)
check("runtime body frame uses copied meters", close(world(runtime, runtime.joints[0].anchor_a, runtime.joints[0].body_a).position, (10/scene.unit_settings.scale_length, 0, 0)))
check("runtime indicator follows body", not close(frame_a(runtime).position, frame_a(source).position))
check("runtime never moves authoring Empty", tuple(tuple(row) for row in handle.matrix_world) == before_handle)
missing_runtime = snapshot(presentation=PresentationPoses(7))
check("missing runtime pose visibly diagnosed", any("Runtime body pose missing" in text for text in missing_runtime.joints[0].errors))
try:
    snapshot(presentation=PresentationPoses(6))
except SnapshotError:
    checks.append("stale presentation rejected")
else:
    raise AssertionError("stale presentation accepted")
check("overlay off removes geometry", not geometry(snapshot(overlays=False)).strokes)
check("collider display toggle does not disable physics", not snapshot(show_colliders=False).colliders and sphere.enabled)
handle.hide_set(True)
check("hidden joint omitted", not snapshot().joints)
handle.hide_set(False)
for obj in scene.objects:
    obj.select_set(False)
handle.select_set(True)
selected = snapshot(selected_only=True)
check("selected-only scope", len(selected.joints) == 1 and not selected.colliders)
joint.endpoint_a.object = None
broken = snapshot()
check("missing endpoint diagnostic geometry", any(s.role == "broken-link" for s in geometry(broken).strokes))
check("missing endpoint not world fallback", any("missing" in text.lower() for text in broken.joints[0].errors))
joint.endpoint_a.object = body

# Bone bodies are evaluated through transformed armatures and body offsets.
bpy.ops.object.armature_add(location=(8, 2, 1))
rig = bpy.context.object
rig.rotation_euler.z = .3
bone = rig.pose.bones[0]
initialize_owner(rig, bone.bone)
bone.ov.physics.rigid_body.schema_opinion = "APPLY"
record = bone.ov.physics.colliders.add()
record.name = record.persistent_id = "3"*32
record.schema_opinion, record.shape = "APPLY", "ANALYTIC_SPHERE"
record.offset.translation = (.1, .2, .3)
bone.ov.physics.rigid_body.offset.translation = (.2, 0, 0)
bone_source = snapshot()
check("bone collider captured", any(c.identity.startswith(rig.ov.physics.persistent_id + "_") for c in bone_source.colliders))
bone_collider = next(c for c in bone_source.colliders if c.identity.startswith(rig.ov.physics.persistent_id + "_"))
expected_bone = rig.matrix_world @ bone.matrix @ Matrix.Translation((.2, 0, 0)) @ Matrix.Translation((.1, .2, .3))
check("bone collider follows transformed rig and offsets", close(world(bone_source, bone_collider.anchor, bone_collider.frame).position, expected_bone.translation))
record.shape = "CAPSULE"
record.radius_m = 0
check("invalid collider visibly diagnosed", any(s.role == "collider-error" for s in geometry(snapshot()).strokes))
record.radius_m = .5
record.shape = "ANALYTIC_SPHERE"
second_layer = scene.view_layers.new("IndicatorOtherLayer")
other = capture(scene, second_layer, bpy.context.evaluated_depsgraph_get(), generation=7, viewport_id=22)
check("view layer and viewport scopes differ", other.scope != bone_source.scope and other.scope.view_layer == "IndicatorOtherLayer" and other.scope.viewport == 22)
errors = []
def worker():
    try:
        capture(scene, bpy.context.view_layer, None, generation=7)
    except SnapshotError as error:
        errors.append(str(error))
thread = threading.Thread(target=worker)
thread.start()
thread.join()
check("off-main-thread capture rejected", bool(errors) and "main thread" in errors[0])
check("no Blender physics created", scene.rigidbody_world is None and all(o.rigid_body is None and o.rigid_body_constraint is None for o in scene.objects))
print(json.dumps({"ok": True, "blender": bpy.app.version_string, "checks": checks}))
