# SPDX-License-Identifier: Apache-2.0
"""Static collision-only owners remain visible without dynamic output poses."""
import sys
import uuid
from pathlib import Path
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example.physics_authoring_snapshot import capture, geometry, PresentationPoses

props.register()
scene = bpy.context.scene
scene.unit_settings.scale_length = .01
objects = []
for schema in ("INHERIT", "REMOVE", "APPLY"):
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.object
    obj.name = "StaticCollider_" + schema
    obj.location.x = len(objects)*4
    obj.scale = (2, 2, 2)
    obj.ov.physics.rigid_body.schema_opinion = schema
    if schema == "APPLY":
        obj.ov.physics.rigid_body.role = "STATIC"
    collider = obj.ov.physics.colliders.add()
    collider.name = collider.persistent_id = uuid.uuid4().hex
    collider.schema_opinion = "APPLY"
    collider.shape = "ANALYTIC_SPHERE"
    collider.radius_m = .5
    objects.append(obj)
bpy.context.view_layer.update()

def source():
    return capture(scene, bpy.context.view_layer, bpy.context.evaluated_depsgraph_get(), generation=8,
                   presentation=PresentationPoses(8))

snapshot = source()
assert len(snapshot.colliders) == 3
assert len({c.identity for c in snapshot.colliders}) == 3
assert all(not c.error for c in snapshot.colliders)
# Collection radii are absolute meters, independent of Object scale: .5 m is 50 BU at centimeter units.
assert all(abs(c.radius-50) < 1e-4 for c in snapshot.colliders)
assert len([s for s in geometry(snapshot).strokes if s.role == "collider-wire"]) == 3
objects[1].hide_set(True)
assert len(source().colliders) == 2
assert not objects[0].ov.physics.persistent_id  # Display never initializes identity.
assert scene.rigidbody_world is None
print("STATIC INDICATORS PASS: inherit/removed-rigid/static roles, unique copied IDs, no runtime demand, units and hidden state")
