# SPDX-License-Identifier: Apache-2.0
"""Regression cases from independent n-01-04 verification."""
import json
from pathlib import Path
import sys
import uuid
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example.physics_colliders import edit_colliders
from ovrtx_blender_example.physics_authoring import AuthoringError

props.register()
bpy.ops.mesh.primitive_cube_add()
body = bpy.context.object
first = edit_colliders(body, bpy.context.scene)
second = edit_colliders(body, bpy.context.scene)
first.radius_m, second.radius_m = .2, .8
first_id = first.persistent_id
material = bpy.data.materials.new("DriverMaterial")
material.use_nodes = True
body.data.materials.append(material)
body.shape_key_add(name="Basis")
shape = body.shape_key_add(name="DrivenShape")
scene = bpy.context.scene
scene["ov_test_radius"] = 0.0
node_tree = material.node_tree
node_tree["ov_test_radius"] = 0.0
cases = [
    ("Material", material, "diffuse_color", 0),
    ("EmbeddedNodeTree", node_tree, '["ov_test_radius"]', None),
    ("ShapeKey", shape, "value", None),
    ("Scene", scene, '["ov_test_radius"]', None),
]
checks = []
for label, owner, path, component in cases:
    curve = owner.driver_add(path, component) if component is not None else owner.driver_add(path)
    variable = curve.driver.variables.new()
    variable.name = "radius"
    target = variable.targets[0]
    target.id = body
    index = next(i for i, r in enumerate(body.ov.physics.colliders) if r.persistent_id == first_id)
    target.data_path = f"ov.physics.colliders[{index}].radius_m"
    curve.driver.expression = "radius"
    original = body.path_resolve(target.data_path)
    edit_colliders(body, scene, move_id=first_id, direction=1 if index == 0 else -1)
    assert first_id in target.data_path, (label, target.data_path)
    assert abs(body.path_resolve(target.data_path) - original) < 1e-7
    checks.append(label + " survivor keeps original radius")
    before = [r.persistent_id for r in body.ov.physics.colliders]
    try:
        edit_colliders(body, scene, remove_id=first_id)
    except AuthoringError as error:
        assert "driver input" in str(error)
    else:
        raise AssertionError(label + " referenced collider removal was accepted")
    assert [r.persistent_id for r in body.ov.physics.colliders] == before
    assert abs(body.path_resolve(target.data_path) - original) < 1e-7
    checks.append(label + " removal rejected before mutation")
    owner.driver_remove(path, component) if component is not None else owner.driver_remove(path)

bpy.ops.object.armature_add()
rig = bpy.context.object
bpy.ops.object.mode_set(mode="POSE")
physics = rig.pose.bones[0].ov.physics
assert bpy.ops.ov.physics_add_body() == {"FINISHED"}
assert len(physics.colliders) == 0
checks.append("assign bone body adds no default collider")
sphere = physics.colliders.add()
sphere.name = sphere.persistent_id = uuid.uuid4().hex
sphere.schema_opinion = "APPLY"
sphere.shape = "ANALYTIC_SPHERE"
sphere.radius_m = 1.23
assert bpy.ops.ov.physics_add_body() == {"FINISHED"}
assert len(physics.colliders) == 1 and physics.colliders[0].persistent_id == sphere.persistent_id
assert physics.colliders[0].schema_opinion == "APPLY"
assert physics.colliders[0].shape == "ANALYTIC_SPHERE"
assert abs(physics.colliders[0].radius_m - 1.23) < 1e-6
checks.append("assign bone body preserves authored sphere collider")
print(json.dumps({"ok": True, "blender": bpy.app.version_string, "checks": checks}))
