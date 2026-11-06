"""Reconstruct joint anchors through real scaled mesh-body USD transforms (convex-hull collider records)."""
import json
from pathlib import Path
import sys
import uuid
import bpy
from pxr import Gf, Usd, UsdGeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example import scene_generation as generation
from ovrtx_blender_example.physics_endpoints import initialize_owner, initialize_joint
from ovrtx_blender_example.physics_authoring import apply_anchor

props.register()
out = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
out.mkdir(parents=True, exist_ok=True)
checks = []
anchors = {}
for units in (1.0, .01):
    for parent_scale in (1.0, 3.0):
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        scene = bpy.context.scene
        scene.unit_settings.scale_length = units
        parent = bpy.data.objects.new("ScaledParent", None)
        scene.collection.objects.link(parent)
        parent.location = tuple(v/units for v in (5, 7, 9))
        parent.rotation_euler = (.3, -.2, .4)
        parent.scale = (parent_scale,) * 3
        bpy.ops.mesh.primitive_cube_add()
        body = bpy.context.object
        body.name = "MeshBody"
        body.parent = parent
        body.location = tuple(v/units for v in (1, 2, 3))
        body.rotation_euler = (.2, .3, .4)
        body.scale = (2, 2, 2)
        initialize_owner(body)
        body.ov.physics.rigid_body.schema_opinion = "APPLY"
        collider = body.ov.physics.colliders.add()
        collider.name = collider.persistent_id = uuid.uuid4().hex
        collider.schema_opinion = "APPLY"
        collider.shape = "CONVEX_HULL"
        handle = bpy.data.objects.new("Handle", None)
        scene.collection.objects.link(handle)
        handle.location = tuple(v/units for v in (2, -1, 4))
        handle.rotation_euler = (.1, .2, .3)
        joint = initialize_joint(handle)
        joint.enabled = True
        joint.endpoint_a.object = body
        joint.endpoint_b.kind = "WORLD"
        bpy.context.view_layer.update()
        apply_anchor(handle, scene, bpy.context.evaluated_depsgraph_get())
        label = f"mesh-{units}-{parent_scale}"
        saved = tuple(joint.frame_a.translation), tuple(joint.frame_a.rotation)
        owner = generation.SceneGenerationOwner(out / label)
        result = owner.replace(scene)
        stage = Usd.Stage.Open(result.usd_path)
        path = scene.ov.physics_namespace + "/Joints/j_" + joint.persistent_id
        prim = stage.GetPrimAtPath(path)
        endpoint = stage.GetPrimAtPath(prim.GetRelationship("physics:body0").GetTargets()[0])
        hull = stage.GetPrimAtPath(str(endpoint.GetPath()) + "/Colliders/c_" + collider.persistent_id)
        assert hull.IsA(UsdGeom.Mesh) and hull.GetAttribute("physics:approximation").Get() == "convexHull", label
        checks.append(label + " generated hull collider")
        world_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(endpoint)
        actual = world_matrix.Transform(Gf.Vec3d(prim.GetAttribute("physics:localPos0").Get()))
        expected = Gf.Vec3d(2, -1, 4)
        assert (actual-expected).GetLength() < 2e-5, (label, actual, expected)
        checks.append(label + " anchor translation")
        rotation = Gf.Transform(world_matrix).GetRotation().GetQuat() * Gf.Quatd(prim.GetAttribute("physics:localRot0").Get())
        expected_rotation = handle.matrix_world.to_quaternion()
        assert abs(sum(x*y for x, y in zip((rotation.GetReal(), *rotation.GetImaginary()), expected_rotation))) > 1-2e-5
        checks.append(label + " anchor orientation")
        assert saved == (tuple(joint.frame_a.translation), tuple(joint.frame_a.rotation))
        checks.append(label + " durable frames unchanged")
        assert (Gf.Vec3d(prim.GetAttribute("physics:localPos1").Get())-expected).GetLength() < 2e-5
        checks.append(label + " world endpoint stays meters")
        anchors[(units, parent_scale)] = actual
for parent_scale in (1., 3.):
    assert (anchors[(1., parent_scale)] - anchors[(.01, parent_scale)]).GetLength() < 2e-5
    checks.append(f"meter centimeter equivalence {parent_scale}")
(out / "results.json").write_text(json.dumps({"blender": bpy.app.version_string, "checks": checks}, indent=2))
print("SCALED JOINT FRAMES PASS", len(checks))
