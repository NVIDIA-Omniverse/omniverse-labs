# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise stock-exported scene generations in real Blender."""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

import bpy  # noqa: E402

from ovrtx_blender_example.scene_generation import (  # noqa: E402
    SceneGenerationOwner,
    blender_id,
)
from ovrtx_blender_example import authoring_properties  # noqa: E402
from ovrtx_blender_example.blender_callback_adapters import (  # noqa: E402
    BlenderRenderCallbackAdapter,
)
from ovrtx_blender_example.blender_signal_translation import (  # noqa: E402
    RenderRequestTranslator,
)
from ovrtx_blender_example import ovrtx_session  # noqa: E402
from ovrtx_blender_example import scene_generation_sessions  # noqa: E402
from ovrtx_blender_example import color_presentation  # noqa: E402
from ovrtx_blender_example import engine as engine_module  # noqa: E402
from ovrtx_blender_example.ovrtx_session_controller import (  # noqa: E402
    OvrtxSessionController,
)
from ovrtx_blender_example.properties import DEFAULT_RENDER_PRODUCT_PATH  # noqa: E402
from ovrtx_blender_example.render_requests import RenderRequest  # noqa: E402
from ovrtx_blender_example.ovrtx_value_updates import (  # noqa: E402
    OvrtxAttributeValue,
    OvrtxTransformValue,
)
from ovrtx_blender_example.ovrtx_runtime_client import RenderResult  # noqa: E402


def _output_path() -> Path:
    separator = sys.argv.index("--")
    return Path(sys.argv[separator + 1]).expanduser().resolve()


def _create_scene() -> tuple[object, object]:
    bpy.ops.mesh.primitive_cube_add()
    cube = bpy.context.object
    cube.name = "Cube"
    material = bpy.data.materials.new("Cube")
    cube.data.materials.append(material)
    cube.ov.physics.rigid_body.schema_opinion = authoring_properties.APPLY
    cube.ov.physics.rigid_body.mass_kg = 12.5
    cube.ov.physics.collision.schema_opinion = authoring_properties.APPLY
    cube.ov.physics.collision.hide_from_render = True
    physics_material = bpy.data.materials.new("Physics Rubber")
    physics_material.ov.physics.schema_opinion = authoring_properties.APPLY
    physics_material.ov.physics.static_friction = 0.8
    physics_material.ov.physics.dynamic_friction = 0.7
    physics_material.ov.physics.restitution = 0.2
    cube.ov.physics.collision.physics_material = physics_material
    bpy.ops.object.light_add(type="SUN")
    bpy.context.object.name = "Sun"
    bpy.ops.object.camera_add(location=(4.0, -4.0, 3.0))
    camera = bpy.context.object
    camera.name = "Camera"
    bpy.context.scene.camera = camera
    return cube, camera


def _paths(generation: object) -> dict[str, dict[str, str]]:
    return {
        f"{mapping.blender_id_type}:{mapping.blender_id_name}": {
            "blender_id_type": mapping.blender_id_type,
            "object_path": mapping.object_path,
            "schema_path": mapping.schema_path,
        }
        for mapping in generation.blender_prim_paths.values()
    }


def _replace_mesh_topology(obj: object, height: float) -> None:
    obj.data.clear_geometry()
    obj.data.from_pydata(
        [
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (0.0, 0.0, height),
        ],
        [],
        [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4), (0, 3, 2, 1)],
    )
    obj.data.update()


def _preserved_selection(camera: object, operation: object) -> object:
    bpy.ops.object.select_all(action="DESELECT")
    camera.select_set(True)
    bpy.context.view_layer.objects.active = camera
    result = operation()
    if tuple(bpy.context.selected_objects) != (camera,):
        raise RuntimeError("sparse reconciliation changed Blender selection")
    if bpy.context.view_layer.objects.active is not camera:
        raise RuntimeError("sparse reconciliation changed the active Blender object")
    return result


def _preserved_edit_mode(obj: object, operation: object) -> object:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    result = operation()
    if tuple(bpy.context.selected_objects) != (obj,):
        raise RuntimeError("sparse reconciliation changed Blender edit selection")
    if bpy.context.view_layer.objects.active is not obj or obj.mode != "EDIT":
        raise RuntimeError("sparse reconciliation changed Blender edit mode")
    bpy.ops.object.mode_set(mode="OBJECT")
    return result


def _select_only(obj: object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _validate_live_lights(generation: object) -> dict[str, object]:
    from pxr import Usd

    live = tuple(obj for obj in bpy.context.scene.objects if obj.type == "LIGHT")
    mappings = {
        identity: mapping
        for identity, mapping in generation.blender_prim_paths.items()
        if mapping.blender_id_type == "LIGHT"
    }
    live_ids = {blender_id(obj, "OBJECT") for obj in live}
    if set(mappings) != live_ids:
        raise RuntimeError("generation light mappings do not match live Blender lights")
    object_paths = {mapping.object_path for mapping in mappings.values()}
    schema_paths = {mapping.schema_path for mapping in mappings.values()}
    if len(object_paths) != len(live) or len(schema_paths) != len(live):
        raise RuntimeError("distinct Blender lights share a generated USD path")
    stage = Usd.Stage.Open(generation.materialize_usd())
    if stage is None:
        raise RuntimeError("light generation could not be opened")
    for mapping in mappings.values():
        if not stage.GetPrimAtPath(mapping.object_path).IsActive():
            raise RuntimeError(f"mapped light object is inactive: {mapping.object_path}")
        if not stage.GetPrimAtPath(mapping.schema_path).IsActive():
            raise RuntimeError(f"mapped light schema is inactive: {mapping.schema_path}")
    return {
        "count": len(live),
        "object_paths": sorted(object_paths),
        "schema_paths": sorted(schema_paths),
    }


def _point_light_runtime_contract(work_root: Path) -> dict[str, object]:
    from pxr import Usd

    worker_command = os.environ.get("OV_BLENDER_EXAMPLE_WORKER_COMMAND", "").strip()
    native_path = os.environ.get("OV_BLENDER_EXAMPLE_NATIVE_CLIENT_PATH", "").strip()
    if not worker_command or not native_path:
        return {"status": "not_requested"}
    if native_path not in sys.path:
        sys.path.insert(0, native_path)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add()
    cube = bpy.context.object
    cube.name = "RuntimeCube"
    bpy.ops.object.camera_add(location=(4.0, -4.0, 3.0))
    camera = bpy.context.object
    camera.name = "RuntimeCamera"
    bpy.context.scene.camera = camera
    bpy.ops.object.light_add(type="POINT", location=(0.0, 1.0, 2.0))
    point = bpy.context.object
    point.name = "RuntimePoint"
    point.data.energy = 1000.0

    scene = bpy.context.scene
    first = scene_generation_sessions.generation_for_scene(
        scene, work_root=work_root
    )
    controller = OvrtxSessionController()

    def request_for(generation: object) -> RenderRequest:
        camera_path = generation.blender_prim_paths[
            blender_id(camera, "OBJECT")
        ].schema_path
        return RenderRequest(
            input_usd_path=generation.materialize_usd(),
            sensor_paths=(DEFAULT_RENDER_PRODUCT_PATH,),
            selected_sensor_paths=(DEFAULT_RENDER_PRODUCT_PATH,),
            width=320,
            height=180,
            min_samples=1,
            max_samples=1,
            camera_prim_path=camera_path,
            worker_command=worker_command,
            native_client_module="ovrtx_bridge_client",
            color_presentation=color_presentation.presentation_from_scene(
                None,
                requested_mode=color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH,
            ),
        )

    first_request = request_for(first)
    runtime = scene_generation_sessions.activate_for_viewport(
        scene,
        first_request,
        viewport_id="point-light-contract",
        controller=controller,
    )
    initial = runtime.ovrtx.controller.render(
        first_request, additional_samples=1
    )
    expected_frame_size = first_request.width * first_request.height * 4
    if len(initial.rgba8) != expected_frame_size:
        raise RuntimeError("initial Point light viewport render is incomplete")

    accepted = first
    activations = []

    def activate_change(
        operation: str,
        affected_id: object,
        expected_count: int,
    ) -> tuple[object, object]:
        nonlocal accepted, runtime
        predecessor = accepted
        scene_generation_sessions.mark_scene_dirty(scene, {affected_id})
        candidate = scene_generation_sessions.generation_for_scene(
            scene, work_root=work_root
        )
        if candidate.number != predecessor.number + 1:
            raise RuntimeError(f"runtime {operation} did not create a new generation")
        candidate_request = request_for(candidate)
        runtime = scene_generation_sessions.activate_for_viewport(
            scene,
            candidate_request,
            viewport_id="point-light-contract",
            controller=controller,
        )
        rendered = runtime.ovrtx.controller.render(
            candidate_request, additional_samples=1
        )
        accepted = scene_generation_sessions.generation_for_scene(
            scene, work_root=work_root
        )
        lights = _validate_live_lights(accepted)
        if accepted.number != candidate.number:
            raise RuntimeError(f"runtime {operation} candidate was not accepted")
        if lights["count"] != expected_count:
            raise RuntimeError(f"runtime {operation} light mappings are incomplete")
        if len(rendered.rgba8) != expected_frame_size:
            raise RuntimeError(f"runtime {operation} viewport render is incomplete")
        activations.append(
            {
                "operation": operation,
                "predecessor_generation": predecessor.number,
                "activated_generation": accepted.number,
                "candidate_accepted": accepted.number == candidate.number,
                "light_count": lights["count"],
                "frame_sha256": hashlib.sha256(rendered.rgba8).hexdigest(),
            }
        )
        return accepted, rendered

    bpy.ops.object.light_add(type="POINT", location=(1.0, 1.0, 2.0))
    created = bpy.context.object
    created.name = "RuntimePointCreated"
    created_id = blender_id(created, "OBJECT")
    activate_change("create", created_id, 2)

    _select_only(point)
    if "FINISHED" not in bpy.ops.view3d.copybuffer():
        raise RuntimeError("runtime Point light copybuffer operation failed")
    bpy.ops.object.select_all(action="DESELECT")
    if "FINISHED" not in bpy.ops.view3d.pastebuffer():
        raise RuntimeError("runtime Point light pastebuffer operation failed")
    pasted = bpy.context.object
    pasted.name = "RuntimePointPasted"
    pasted_id = blender_id(pasted, "OBJECT")
    activate_change("copy_paste", pasted_id, 3)
    selected = tuple(obj.name for obj in bpy.context.selected_objects)
    if selected != (pasted.name,):
        raise RuntimeError("runtime Point light copy/paste selection was not retained")

    _select_only(point)
    if "FINISHED" not in bpy.ops.object.duplicate(linked=False):
        raise RuntimeError("runtime independent Point light duplication failed")
    independent = bpy.context.object
    independent.name = "RuntimePointIndependent"
    independent_id = blender_id(independent, "OBJECT")
    activate_change("unlinked_duplicate", independent_id, 4)

    _select_only(point)
    if "FINISHED" not in bpy.ops.object.duplicate(linked=True):
        raise RuntimeError("runtime linked Point light duplication failed")
    linked = bpy.context.object
    linked.name = "RuntimePointLinked"
    linked_id = blender_id(linked, "OBJECT")
    accepted, rendered = activate_change("linked_duplicate", linked_id, 5)

    deleted_mapping = accepted.blender_prim_paths[pasted_id]
    bpy.data.objects.remove(pasted, do_unlink=True)
    accepted, rendered = activate_change("delete", pasted_id, 4)
    deleted_stage = Usd.Stage.Open(accepted.materialize_usd())
    if deleted_stage.GetPrimAtPath(deleted_mapping.object_path).IsActive():
        raise RuntimeError("runtime deleted Point light predecessor remained active")

    scene_generation_sessions.close()
    return {
        "status": "pass-real",
        "initial_generation": first.number,
        "activated_generation": accepted.number,
        "activations": activations,
        "pasted_identity": {
            "kind": pasted_id.kind,
            "session_uid": pasted_id.session_uid,
        },
        "light_count": activations[-1]["light_count"],
        "deleted_predecessor_inactive": True,
        "selected_objects": list(selected),
        "initial_frame_sha256": hashlib.sha256(initial.rgba8).hexdigest(),
        "activated_frame_sha256": hashlib.sha256(rendered.rgba8).hexdigest(),
        "render_var": rendered.render_var,
    }


def main() -> None:
    output = _output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    authoring_properties.register()
    cube, camera = _create_scene()
    cube_id = blender_id(cube, "OBJECT")
    cube_mesh_data = cube.data
    cube_mesh_id = blender_id(cube_mesh_data, "MESH")
    cube_material = cube.data.materials[0]
    cube_material_id = blender_id(cube_material, "MATERIAL")
    work = output.parent / "scene-generations"
    owner = SceneGenerationOwner(work)

    first = owner.replace(bpy.context.scene)
    if first is None:
        raise RuntimeError("first scene generation was not committed")
    reused = owner.reuse()
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    first_stage = Usd.Stage.Open(first.materialize_usd())
    if first_stage is None:
        raise RuntimeError("first scene generation could not be opened")
    cube_mapping = first.blender_prim_paths[blender_id(cube, "OBJECT")]
    cube_root = first_stage.GetPrimAtPath(cube_mapping.object_path)
    cube_mesh = first_stage.GetPrimAtPath(cube_mapping.schema_path)
    if "PhysicsRigidBodyAPI" not in cube_root.GetAppliedSchemas():
        raise RuntimeError("mapped cube has no sparse rigid-body opinion")
    if "PhysicsCollisionAPI" not in cube_mesh.GetAppliedSchemas():
        raise RuntimeError("mapped cube has no sparse collision opinion")
    cube_mass = UsdPhysics.MassAPI(cube_root).GetMassAttr().Get()
    cube_visibility = UsdGeom.Imageable(cube_mesh).GetVisibilityAttr().Get()
    object_record = next(
        record
        for record in first.opinion_records
        if record.diagnostics.get("blender_owner") == "Cube"
    )
    material_record = next(
        record
        for record in first.opinion_records
        if record.diagnostics.get("blender_owner") == "Physics Rubber"
    )
    record_layer = Sdf.Layer.CreateAnonymous(".usda")
    if not record_layer.ImportFromString(object_record.layer_text):
        raise RuntimeError("could not inspect add-on USD opinion record")
    record_root = record_layer.GetPrimAtPath(cube_mapping.object_path)
    record_mesh = record_layer.GetPrimAtPath(cube_mapping.schema_path)
    sparse_record = (
        record_root.specifier == Sdf.SpecifierOver
        and record_mesh.specifier == Sdf.SpecifierOver
        and record_layer.GetPropertyAtPath(
            record_mesh.path.AppendProperty("points")
        ) is None
    )
    physics_material_prim = first_stage.GetPrimAtPath(material_record.usd_prim_path)
    binding_targets = cube_mesh.GetRelationship("material:binding:physics").GetTargets()
    add_on_physics_material = (
        physics_material_prim.GetTypeName() == "Material"
        and "PhysicsMaterialAPI" in physics_material_prim.GetAppliedSchemas()
        and [str(path) for path in binding_targets] == [material_record.usd_prim_path]
    )
    callback = BlenderRenderCallbackAdapter(
        generation_for_scene=lambda _scene: first,
        translator=RenderRequestTranslator(
            blender_module_provider=lambda: SimpleNamespace(
                data=SimpleNamespace(materials=())
            )
        ),
    )
    final_request = callback.final_render_from_scene(bpy.context.scene)
    final_spec = ovrtx_session.build_spec(final_request)
    presentation_path = next(
        Path(str(record["path"]))
        for record in final_spec.ovrtx_scene_composition.presentation_layers
        if record["source"] == "viewport_camera_projection"
    )
    presentation_text = presentation_path.read_text(encoding="utf-8")
    callback_contract = (
        final_request.input_usd_path == first.materialize_usd()
        and final_request.camera_prim_path
        == first.blender_prim_paths[blender_id(camera, "OBJECT")].schema_path
        and "def RenderProduct" in presentation_text
        and f"rel camera = <{final_request.camera_prim_path}>" in presentation_text
    )
    if owner.reconcile(bpy.context.scene, {cube_id}) is not None:
        raise RuntimeError("unchanged scene created another generation")

    _replace_mesh_topology(cube, 1.5)
    changed_once = _preserved_selection(
        camera,
        lambda: owner.reconcile(
            bpy.context.scene,
            {blender_id(cube.data, "MESH")},
        ),
    )
    if changed_once is None:
        raise RuntimeError("first mesh topology change did not create a generation")
    owner.accept(changed_once)
    _replace_mesh_topology(cube, 2.0)
    changed_twice = _preserved_edit_mode(
        cube,
        lambda: owner.reconcile(
            bpy.context.scene,
            {blender_id(cube.data, "MESH")},
        ),
    )
    if changed_twice is None:
        raise RuntimeError("second mesh topology change did not create a generation")
    owner.accept(changed_twice)

    bpy.ops.mesh.primitive_uv_sphere_add(location=(2.0, 0.0, 0.0))
    sphere = bpy.context.object
    sphere.name = "Sphere"
    sphere.ov.physics.rigid_body.schema_opinion = authoring_properties.APPLY
    sphere.ov.physics.rigid_body.mass_kg = 80.0
    sphere.ov.physics.collision.schema_opinion = authoring_properties.APPLY
    sphere.ov.physics.collision.shape = "ANALYTIC_SPHERE"
    sphere.ov.physics.collision.radius_m = 0.65
    second = _preserved_selection(
        camera,
        lambda: owner.reconcile(
            bpy.context.scene,
            {blender_id(sphere, "OBJECT"), blender_id(sphere.data, "MESH")},
        ),
    )
    if second is None:
        raise RuntimeError("added sphere did not replace the scene generation")
    owner.accept(second)
    first_retained = Path(first.materialize_usd()).is_file()
    second_stage = Usd.Stage.Open(second.materialize_usd())
    sphere_mapping = second.blender_prim_paths[blender_id(sphere, "OBJECT")]
    sphere_root = second_stage.GetPrimAtPath(sphere_mapping.object_path)
    sphere_collider = second_stage.GetPrimAtPath(
        sphere_mapping.object_path + "/Collider"
    )

    bpy.data.objects.remove(cube, do_unlink=True)
    bpy.data.meshes.remove(cube_mesh_data)
    bpy.data.materials.remove(cube_material)
    third = _preserved_selection(
        camera,
        lambda: owner.reconcile(
            bpy.context.scene,
            {cube_id, cube_mesh_id, cube_material_id},
        ),
    )
    if third is None:
        raise RuntimeError("deleted cube did not replace the scene generation")
    owner.accept(third)
    third_path = third.materialize_usd()

    third_stage = Usd.Stage.Open(third_path)
    if third_stage is None or third_stage.GetPrimAtPath(cube_mapping.object_path).IsActive():
        raise RuntimeError("deleted Cube remained in the replacement generation")

    bpy.ops.mesh.primitive_cube_add()
    recreated_cube = bpy.context.object
    recreated_cube.name = "Cube"
    recreated_material = bpy.data.materials.new("Cube")
    recreated_cube.data.materials.append(recreated_material)
    recreated_id = blender_id(recreated_cube, "OBJECT")
    recreated_mesh_id = blender_id(recreated_cube.data, "MESH")
    recreated_material_id = blender_id(recreated_material, "MATERIAL")
    if (
        recreated_id == cube_id
        or recreated_mesh_id == cube_mesh_id
        or recreated_material_id == cube_material_id
    ):
        raise RuntimeError("recreated Blender data reused a deleted session_uid")
    fourth = _preserved_selection(
        camera,
        lambda: owner.reconcile(
            bpy.context.scene,
            {recreated_id, recreated_mesh_id, recreated_material_id},
        ),
    )
    if fourth is None:
        raise RuntimeError("equivalent recreated content did not create a generation")
    owner.accept(fourth)
    fourth_stage = Usd.Stage.Open(fourth.materialize_usd())
    recreated_mapping = fourth.blender_prim_paths[recreated_id]
    if not fourth_stage.GetPrimAtPath(recreated_mapping.object_path).IsActive():
        raise RuntimeError("recreated Cube mapping is inactive")

    bpy.ops.object.light_add(type="POINT", location=(0.0, 2.0, 2.0))
    point = bpy.context.object
    point.name = "Point"
    point_id = blender_id(point, "OBJECT")
    point_data_id = blender_id(point.data, "LIGHT")
    point_generation = _preserved_selection(
        camera,
        lambda: owner.reconcile(bpy.context.scene, {point_id}),
    )
    if point_generation is None:
        raise RuntimeError("new Point light did not create a generation")
    owner.accept(point_generation)
    point_lights = _validate_live_lights(point_generation)

    _select_only(point)
    if "FINISHED" not in bpy.ops.view3d.copybuffer():
        raise RuntimeError("Point light copybuffer operation failed")
    bpy.ops.object.select_all(action="DESELECT")
    if "FINISHED" not in bpy.ops.view3d.pastebuffer():
        raise RuntimeError("Point light pastebuffer operation failed")
    pasted = bpy.context.object
    pasted.name = "PointPasted"
    pasted_id = blender_id(pasted, "OBJECT")
    pasted_data_id = blender_id(pasted.data, "LIGHT")
    pasted_generation = _preserved_selection(
        camera,
        lambda: owner.reconcile(bpy.context.scene, {pasted_id}),
    )
    if pasted_generation is None:
        raise RuntimeError("pasted Point light did not create a generation")
    owner.accept(pasted_generation)
    pasted_lights = _validate_live_lights(pasted_generation)

    _select_only(point)
    if "FINISHED" not in bpy.ops.object.duplicate(linked=False):
        raise RuntimeError("independent Point light duplication failed")
    independent = bpy.context.object
    independent.name = "PointIndependent"
    independent_id = blender_id(independent, "OBJECT")
    independent_data_id = blender_id(independent.data, "LIGHT")
    independent_generation = _preserved_selection(
        camera,
        lambda: owner.reconcile(bpy.context.scene, {independent_id}),
    )
    if independent_generation is None:
        raise RuntimeError("independent Point light duplicate did not create a generation")
    owner.accept(independent_generation)
    independent_lights = _validate_live_lights(independent_generation)

    _select_only(point)
    if "FINISHED" not in bpy.ops.object.duplicate(linked=True):
        raise RuntimeError("linked Point light duplication failed")
    linked = bpy.context.object
    linked.name = "PointLinked"
    linked_id = blender_id(linked, "OBJECT")
    linked_data_id = blender_id(linked.data, "LIGHT")
    linked_generation = _preserved_selection(
        camera,
        lambda: owner.reconcile(bpy.context.scene, {linked_id}),
    )
    if linked_generation is None:
        raise RuntimeError("linked Point light duplicate did not create a generation")
    owner.accept(linked_generation)
    linked_lights = _validate_live_lights(linked_generation)

    _select_only(point)
    if "FINISHED" not in bpy.ops.object.duplicate(linked=False):
        raise RuntimeError("parented Point light duplication failed")
    parented = bpy.context.object
    parented.name = "PointParented"
    parented.parent = recreated_cube
    parented_id = blender_id(parented, "OBJECT")
    parented_generation = _preserved_selection(
        camera,
        lambda: owner.reconcile(bpy.context.scene, {parented_id}),
    )
    if parented_generation is None:
        raise RuntimeError("parented Point light duplicate did not create a generation")
    owner.accept(parented_generation)
    parented_lights = _validate_live_lights(parented_generation)
    parented_mapping = parented_generation.blender_prim_paths[parented_id]
    parent_mapping = parented_generation.blender_prim_paths[recreated_id]
    if not parented_mapping.object_path.startswith(parent_mapping.object_path + "/"):
        raise RuntimeError("parented Point light did not retain mapped hierarchy")

    _replace_mesh_topology(recreated_cube, 1.25)
    parent_updated_generation = _preserved_selection(
        camera,
        lambda: owner.reconcile(bpy.context.scene, {recreated_mesh_id}),
    )
    if parent_updated_generation is None:
        raise RuntimeError("parent update did not create a generation")
    owner.accept(parent_updated_generation)
    parent_updated_lights = _validate_live_lights(parent_updated_generation)
    updated_parented_mapping = parent_updated_generation.blender_prim_paths[parented_id]
    updated_parent_mapping = parent_updated_generation.blender_prim_paths[recreated_id]
    if not updated_parented_mapping.object_path.startswith(
        updated_parent_mapping.object_path + "/"
    ):
        raise RuntimeError("parented Point light did not follow its mapped parent")
    parent_updated_stage = Usd.Stage.Open(parent_updated_generation.materialize_usd())
    if parent_updated_stage.GetPrimAtPath(parent_mapping.object_path).IsActive():
        raise RuntimeError("updated parent predecessor root remained active")

    deleted_id = pasted_id
    deleted_mapping = parent_updated_generation.blender_prim_paths[deleted_id]
    bpy.data.objects.remove(pasted, do_unlink=True)
    deleted_generation = _preserved_selection(
        camera,
        lambda: owner.reconcile(bpy.context.scene, {deleted_id}),
    )
    if deleted_generation is None:
        raise RuntimeError("deleted Point light did not create a generation")
    owner.accept(deleted_generation)
    deleted_lights = _validate_live_lights(deleted_generation)
    deleted_stage = Usd.Stage.Open(deleted_generation.materialize_usd())
    if deleted_stage.GetPrimAtPath(deleted_mapping.object_path).IsActive():
        raise RuntimeError("deleted Point light predecessor root remained active")

    final_work = output.parent / "final-scene-generation"
    final_owner = SceneGenerationOwner(final_work)
    final_generation = final_owner.replace(bpy.context.scene)
    if final_generation is None:
        raise RuntimeError("final complete scene export was not created")
    final_lights = _validate_live_lights(final_generation)
    final_owner.close()

    session_root = output.parent / "session-recovery"
    scene_generation_sessions.close()
    accepted = scene_generation_sessions.generation_for_scene(
        bpy.context.scene, work_root=session_root
    )
    bpy.ops.object.camera_add()
    unsupported = bpy.context.object
    unsupported.name = "UnsupportedCamera"
    unsupported_id = blender_id(unsupported, "OBJECT")
    scene_generation_sessions.mark_scene_dirty(bpy.context.scene, {unsupported_id})
    blocked_count = 0
    for _attempt in range(2):
        try:
            scene_generation_sessions.generation_for_scene(
                bpy.context.scene, work_root=session_root
            )
        except Exception:
            blocked_count += 1
    blocked_before_undo = scene_generation_sessions.diagnostics()
    bpy.data.objects.remove(unsupported, do_unlink=True)
    recovered = scene_generation_sessions.generation_for_scene(
        bpy.context.scene, work_root=session_root
    )
    blocked_after_undo = scene_generation_sessions.diagnostics()
    scene_generation_sessions.close()

    handoff_predecessor = owner.current_generation
    sun = bpy.data.objects["Sun"]
    sun_id = blender_id(sun, "OBJECT")
    handoff_mapping = handoff_predecessor.blender_prim_paths[recreated_id]
    retained_transform = OvrtxTransformValue(
        handoff_mapping.object_path,
        tuple(tuple(row) for row in recreated_cube.matrix_world),
    )
    retained_material = OvrtxAttributeValue(
        handoff_predecessor.blender_prim_paths[recreated_material_id].schema_path,
        "inputs:diffuseColor",
        tuple(recreated_material.diffuse_color[:3]),
        "Color3f",
    )
    retained_light = OvrtxAttributeValue(
        handoff_predecessor.blender_prim_paths[sun_id].schema_path,
        "intensity",
        float(sun.data.energy),
        "Float",
    )
    owner.retain_transform_values((retained_transform,))
    owner.retain_attribute_values(
        (retained_material, retained_light)
    )

    class _FinalAdapter:
        fail = False
        activations = []
        last_error = ""

        def __init__(self, _controller: object) -> None:
            self.last_ensure_result = SimpleNamespace(composition=None)

        def update_request(self, _request: object) -> None:
            pass

        def activate(self, generation: object, **values: object) -> bool:
            self.activations.append((generation, values))
            if self.fail:
                self.last_error = "contract handoff failure"
                return False
            return True

        def deactivate(self) -> str:
            return "stopped"

    class _FinalController:
        def render(self, _request: object) -> object:
            return SimpleNamespace(
                result=RenderResult(
                    width=1,
                    height=1,
                    rgba8=bytes((0, 0, 0, 255)),
                    completed_samples=1,
                    session_completed_samples=1,
                    simulation_time_ns=0,
                )
            )

        def shutdown(self) -> None:
            pass

    scene_uid = int(bpy.context.scene.session_uid)
    scene_generation_sessions._owners[scene_uid] = owner
    scene_generation_sessions._dirty.pop(scene_uid, None)
    original_adapter = scene_generation_sessions.OvrtxGenerationAdapter
    original_controller = engine_module.OvrtxSessionController
    scene_generation_sessions.OvrtxGenerationAdapter = _FinalAdapter
    engine_module.OvrtxSessionController = _FinalController
    engine_module._RENDER_CALLBACK_ADAPTERS.clear()
    engine_registered = hasattr(bpy.types, "OvrtxExampleRenderEngine")
    if not engine_registered:
        bpy.utils.register_class(engine_module.OvrtxExampleRenderEngine)
    bpy.context.scene.render.engine = engine_module.ENGINE_ID
    bpy.context.scene.render.resolution_x = 1
    bpy.context.scene.render.resolution_y = 1
    bpy.context.scene.render.resolution_percentage = 100
    try:
        _replace_mesh_topology(recreated_cube, 1.5)
        scene_generation_sessions.mark_scene_dirty(
            bpy.context.scene,
            {blender_id(recreated_cube.data, "MESH"), recreated_material_id},
        )
        _FinalAdapter.fail = True
        try:
            bpy.ops.render.render()
        except RuntimeError as exc:
            if "contract handoff failure" not in str(exc):
                raise
        else:
            raise RuntimeError("failed final handoff unexpectedly rendered")
        failed_candidate = _FinalAdapter.activations[-1][0]
        predecessor_after_failure = owner.current_generation
        _FinalAdapter.fail = False
        bpy.ops.render.render()
        predecessor_reuse, predecessor_reuse_values = _FinalAdapter.activations[-1]
        predecessor_retained_value_reuse = (
            predecessor_reuse is handoff_predecessor
            and predecessor_reuse_values["transform_values"]
            == (retained_transform,)
            and set(predecessor_reuse_values["attribute_values"])
            == {retained_material, retained_light}
        )
        if not predecessor_retained_value_reuse:
            raise RuntimeError("predecessor reuse omitted retained scene values")

        _replace_mesh_topology(recreated_cube, 2.0)
        scene_generation_sessions.mark_scene_dirty(
            bpy.context.scene,
            {blender_id(recreated_cube.data, "MESH"), recreated_material_id},
        )
        bpy.ops.render.render()
        final_candidate = owner.current_generation
    finally:
        scene_generation_sessions.OvrtxGenerationAdapter = original_adapter
        engine_module.OvrtxSessionController = original_controller
        engine_module._RENDER_CALLBACK_ADAPTERS.clear()
        scene_generation_sessions._owners.pop(scene_uid, None)
        scene_generation_sessions._dirty.pop(scene_uid, None)
        if not engine_registered:
            bpy.context.scene.render.engine = "BLENDER_WORKBENCH"
            bpy.utils.unregister_class(engine_module.OvrtxExampleRenderEngine)

    replay_generation, replay_values = _FinalAdapter.activations[-1]
    replay_paths = {
        value.prim_path
        for values in (
            replay_values["transform_values"],
            replay_values["attribute_values"],
        )
        for value in values
    }
    retained_value_handoff = (
        predecessor_after_failure is handoff_predecessor
        and predecessor_reuse is handoff_predecessor
        and failed_candidate is not handoff_predecessor
        and owner.current_generation is final_candidate
        and replay_generation is final_candidate
        and final_candidate.blender_prim_paths[recreated_id].object_path in replay_paths
        and final_candidate.blender_prim_paths[recreated_material_id].schema_path in replay_paths
        and final_candidate.blender_prim_paths[sun_id].schema_path in replay_paths
    )

    result = {
        "status": "passed",
        "generation_numbers": [
            first.number,
            changed_once.number,
            changed_twice.number,
            second.number,
            third.number,
            fourth.number,
            point_generation.number,
            pasted_generation.number,
            independent_generation.number,
            linked_generation.number,
            parented_generation.number,
            parent_updated_generation.number,
            deleted_generation.number,
        ],
        "reused_generation": reused.number,
        "first_generation_retained": first_retained,
        "first_generation_opinion_records": len(first.opinion_records),
        "first_generation_sparse_change_replacements": len(
            first.sparse_change.replacement_records
        ),
        "first_generation_sparse_record": sparse_record,
        "first_generation_add_on_physics_material": add_on_physics_material,
        "first_generation_mass_kg": float(cube_mass),
        "first_generation_collision_invisible": cube_visibility == UsdGeom.Tokens.invisible,
        "second_generation_analytic_sphere": (
            sphere_collider.GetTypeName() == "Sphere"
            and "PhysicsCollisionAPI" in sphere_collider.GetAppliedSchemas()
            and math.isclose(
                float(UsdGeom.Sphere(sphere_collider).GetRadiusAttr().Get()),
                0.65,
                abs_tol=1.0e-6,
            )
            and float(UsdPhysics.MassAPI(sphere_root).GetMassAttr().Get()) == 80.0
        ),
        "complete_export_modes": [
            first.diagnostics["complete_export"],
            changed_once.diagnostics["complete_export"],
            changed_twice.diagnostics["complete_export"],
            second.diagnostics["complete_export"],
            third.diagnostics["complete_export"],
            fourth.diagnostics["complete_export"],
            point_generation.diagnostics["complete_export"],
            pasted_generation.diagnostics["complete_export"],
            independent_generation.diagnostics["complete_export"],
            linked_generation.diagnostics["complete_export"],
            parented_generation.diagnostics["complete_export"],
            parent_updated_generation.diagnostics["complete_export"],
            deleted_generation.diagnostics["complete_export"],
        ],
        "topology_delta_counts": [
            len(generation.topology_deltas)
            for generation in (
                first,
                changed_once,
                changed_twice,
                second,
                third,
                fourth,
                point_generation,
                pasted_generation,
                independent_generation,
                linked_generation,
                parented_generation,
                parent_updated_generation,
                deleted_generation,
            )
        ],
        "recreated_object_has_new_session_uid": recreated_id != cube_id,
        "recreated_mesh_has_new_session_uid": recreated_mesh_id != cube_mesh_id,
        "recreated_material_has_new_session_uid": (
            recreated_material_id != cube_material_id
        ),
        "edit_mode_preserved": True,
        "current_scene_callback_contract": callback_contract,
        "predecessor_retained_value_reuse": predecessor_retained_value_reuse,
        "retained_value_final_handoff": retained_value_handoff,
        "first_generation_mesh_count": sum(
            1 for prim in first_stage.Traverse() if prim.GetTypeName() == "Mesh"
        ),
        "first_generation_paths": _paths(first),
        "second_generation_paths": _paths(second),
        "third_generation_paths": _paths(third),
        "fourth_generation_paths": _paths(fourth),
        "point_light_generations": [
            point_lights,
            pasted_lights,
            independent_lights,
            linked_lights,
            parented_lights,
            parent_updated_lights,
            deleted_lights,
        ],
        "point_light_identities": {
            "objects_distinct": len({point_id, pasted_id, independent_id, linked_id}) == 4,
            "copy_data_distinct": pasted_data_id != point_data_id,
            "independent_data_distinct": independent_data_id != point_data_id,
            "linked_data_shared": linked_data_id == point_data_id,
        },
        "parented_light_hierarchy": True,
        "parented_light_followed_parent": True,
        "deleted_light_inactive": True,
        "final_complete_export": {
            "complete_export": final_generation.diagnostics["complete_export"],
            "light_count": final_lights["count"],
            "work_directory_exists_after_close": final_work.exists(),
        },
        "blocked_callbacks_before_undo": blocked_count,
        "blocked_affected_before_undo": blocked_before_undo[
            "blocked_reconciliations"
        ],
        "undo_reused_accepted_generation": recovered.number == accepted.number,
        "blocked_after_undo": blocked_after_undo["blocked_reconciliations"],
    }
    scene_generation_sessions.close()
    owner.close()
    result["closed_work_directory_exists"] = work.exists()
    result["point_light_runtime"] = _point_light_runtime_contract(
        output.parent / "point-light-runtime"
    )
    authoring_properties.unregister()
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


main()
