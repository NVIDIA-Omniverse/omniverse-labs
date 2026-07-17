# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blender-side setup for scripted interactive OVRTX sessions."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import usd_paths as usd_paths


RENDERABLE_SELECTION_TYPES = {"MESH", "CURVE", "SURFACE", "META", "FONT"}


def run(config: Mapping[str, Any]) -> dict[str, Any]:
    """Configure Blender for an interactive OVRTX session."""

    import bpy  # type: ignore
    import ovrtx_blender_example

    usd_path = Path(str(config["usd_path"]))
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.wm.usd_import(
        filepath=str(usd_path),
        property_import_mode="ALL",
        merge_parent_xform=False,
    )

    os.environ["OV_BLENDER_EXAMPLE_WORKER_COMMAND"] = str(config["worker_command"])
    os.environ["OV_BLENDER_EXAMPLE_NATIVE_CLIENT_MODULE"] = str(config["native_client_module"])

    ovrtx_blender_example.register()
    from . import engine

    engine.configure_exact_stage(
        input_usd_path=str(usd_path),
        camera_prim_path=str(config["camera_prim_path"]),
        render_product_path=str(config["render_product_path"]),
    )

    scene = bpy.context.scene
    configure_scene(scene, config)
    path_index = usd_paths.load_usd_path_index(usd_path)
    camera_binding = bind_scene_camera(scene, str(config.get("camera_prim_path", "")), path_index)
    if scene.camera is not None and hasattr(scene.camera.data, "show_passepartout"):
        scene.camera.data.show_passepartout = False

    selectable_imported_objects = bool(config["selectable_imported_objects"])
    tag_dynamic_body_transforms = bool(config["tag_dynamic_body_transforms"])
    _deselect_scene_objects(bpy)
    bpy.context.view_layer.objects.active = None
    visual_source_ids = _prepare_imported_object_interaction_state(
        scene.objects,
        selectable_imported_objects=selectable_imported_objects,
        tag_dynamic_body_transforms=tag_dynamic_body_transforms,
    )

    tag_result = {
        "dynamic_body_edit_owners": [],
        "dynamic_body_selection_sources": [],
        "dynamic_body_root": str(config["body_root"]),
    }
    if tag_dynamic_body_transforms:
        tag_result = _tag_dynamic_body_transforms(
            scene,
            path_index,
            dynamic_body_root=str(config["body_root"]),
            visual_source_ids=visual_source_ids,
            selectable_imported_objects=selectable_imported_objects,
        )

    configure_viewports(bpy, scene, config)
    _print_setup_summary(scene, usd_path, config, path_index, camera_binding, tag_result)
    _register_bounded_shutdown(bpy, config, ovrtx_blender_example)
    _register_shared_stage_redraw(bpy, config)

    return {
        "camera_binding": camera_binding,
        "usd_path_index": {
            "available": path_index["available"],
            "reason": path_index["reason"],
        },
        "tagged_dynamic_body_count": len(tag_result["dynamic_body_edit_owners"]),
        "tagged_dynamic_body_selection_source_count": len(tag_result["dynamic_body_selection_sources"]),
    }


def configure_scene(scene: Any, config: Mapping[str, Any]) -> None:
    scene.render.engine = "OVRTX_EXAMPLE"
    scene.render.resolution_x = int(config["width"])
    scene.render.resolution_y = int(config["height"])
    scene.render.resolution_percentage = 100
    if config.get("shared_stage_composition"):
        scene.frame_start = 0
        scene.frame_end = int(config["timeline_end_frame"])
        scene.frame_set(scene.frame_start)
        scene.render.fps = max(1, int(round(float(config["composition_update_fps"]))))
    scene.ovrtx_example.min_samples = int(config["min_samples"])
    scene.ovrtx_example.max_samples = int(config["max_samples"])
    scene.ovrtx_example.color_presentation_mode = str(config["color_presentation"])
    scene.ovrtx_example.sync_viewport_camera = bool(config["sync_viewport_camera"])


def bind_scene_camera(
    scene: Any,
    camera_prim_path: str,
    path_index: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind scene.camera to the imported Blender camera for camera_prim_path."""

    camera_prim_path = usd_paths.clean_usd_path(camera_prim_path)
    scene.camera = None
    cameras = [obj for obj in getattr(scene, "objects", ()) if getattr(obj, "type", "") == "CAMERA"]
    result: dict[str, Any] = {
        "status": "unresolved",
        "camera_prim_path": camera_prim_path,
        "scene_camera": "",
        "match_source": "",
        "matched_usd_path": "",
        "candidate_count": 0,
    }
    if not camera_prim_path:
        result["status"] = "missing_camera_prim_path"
        return result

    for source in usd_paths.camera_match_sources(camera_prim_path):
        matches = [
            (obj, path)
            for obj, path in (
                (obj, usd_paths.camera_usd_path_for_source(obj, source)) for obj in cameras
            )
            if path == camera_prim_path
        ]
        if len(matches) == 1:
            obj, matched_path = matches[0]
            scene.camera = obj
            result.update(
                {
                    "status": "bound",
                    "scene_camera": str(getattr(obj, "name", "")),
                    "match_source": source,
                    "matched_usd_path": matched_path,
                    "candidate_count": 1,
                }
            )
            return result
        if len(matches) > 1:
            result.update(
                {
                    "status": "ambiguous",
                    "match_source": source,
                    "candidate_count": len(matches),
                    "matched_usd_path": camera_prim_path,
                }
            )
            return result

    return result


def configure_viewports(
    bpy: Any,
    scene: Any,
    config: Mapping[str, Any],
    *,
    screen: Any | None = None,
) -> None:
    for screen in (screen,) if screen is not None else bpy.data.screens:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type != "VIEW_3D":
                    continue
                space.shading.type = "RENDERED"
                space.overlay.show_overlays = True
                space.overlay.show_performance = True
                space.overlay.show_text = False
                space.overlay.show_stats = False
                space.overlay.show_floor = False
                space.overlay.show_ortho_grid = False
                space.overlay.show_axis_x = False
                space.overlay.show_axis_y = False
                space.overlay.show_axis_z = False
                space.overlay.show_cursor = False
                space.overlay.show_annotation = False
                space.overlay.show_extras = False
                space.overlay.show_relationship_lines = False
                space.overlay.show_bones = False
                space.overlay.show_motion_paths = False
                space.overlay.show_camera_guides = False
                space.overlay.show_camera_passepartout = False
                space.overlay.show_outline_selected = bool(config["selectable_imported_objects"])
                space.show_gizmo = bool(config["selectable_imported_objects"])
                _seed_perspective_view_from_scene_camera(space, scene, config)


def _scene_renderable_bounds_center(scene: Any) -> Any | None:
    try:
        from mathutils import Vector
    except Exception:
        return None
    points = []
    for obj in scene.objects:
        if getattr(obj, "type", "") not in RENDERABLE_SELECTION_TYPES:
            continue
        bound_box = getattr(obj, "bound_box", None)
        matrix_world = getattr(obj, "matrix_world", None)
        if not bound_box or matrix_world is None:
            continue
        try:
            points.extend(matrix_world @ Vector(corner) for corner in bound_box)
        except Exception:
            continue
    if not points:
        return None
    minimum = points[0].copy()
    maximum = points[0].copy()
    for point in points[1:]:
        minimum.x = min(minimum.x, point.x)
        minimum.y = min(minimum.y, point.y)
        minimum.z = min(minimum.z, point.z)
        maximum.x = max(maximum.x, point.x)
        maximum.y = max(maximum.y, point.y)
        maximum.z = max(maximum.z, point.z)
    return (minimum + maximum) * 0.5


def _view3d_lens_from_scene_camera(camera: Any) -> float | None:
    data = getattr(camera, "data", None)
    lens = getattr(data, "lens", None)
    if lens is None:
        return None
    try:
        lens = float(lens)
    except Exception:
        return None
    if not math.isfinite(lens) or lens <= 0.0:
        return None

    source_sensor = getattr(data, "sensor_height", None)
    reference_sensor = 32.0
    try:
        source_sensor = float(source_sensor)
    except Exception:
        source_sensor = 0.0
    if not math.isfinite(source_sensor) or source_sensor <= 0.0:
        return None
    converted = lens * reference_sensor / source_sensor
    if not math.isfinite(converted) or converted <= 0.0:
        return None
    return max(1.0, min(250.0, converted))


def _positive_config_float(config: Mapping[str, Any], key: str) -> float | None:
    value = config.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def _resolved_view_distance(config: Mapping[str, Any], auto_distance: float) -> float:
    configured = _positive_config_float(config, "viewport_orbit_distance")
    if configured is not None:
        return configured
    if math.isfinite(auto_distance) and auto_distance > 0.0:
        return auto_distance
    return 10.0


def _view_rotation_from_scene_camera(camera_rotation: Any, config: Mapping[str, Any]) -> Any:
    pitch_degrees = _positive_config_float(config, "viewport_pitch_offset_degrees")
    if pitch_degrees is None:
        return camera_rotation
    from mathutils import Matrix

    pitch = Matrix.Rotation(math.radians(pitch_degrees), 4, "X").to_quaternion()
    return camera_rotation @ pitch


def _seed_perspective_view_from_scene_camera(space: Any, scene: Any, config: Mapping[str, Any]) -> None:
    region_data = getattr(space, "region_3d", None)
    camera = getattr(scene, "camera", None)
    if region_data is None or camera is None:
        return
    try:
        from mathutils import Vector

        camera_matrix = camera.matrix_world.copy()
        camera_location = camera_matrix.translation
        view_rotation = _view_rotation_from_scene_camera(camera_matrix.to_quaternion(), config)
        forward = view_rotation @ Vector((0.0, 0.0, -1.0))
        center = _scene_renderable_bounds_center(scene)
        view_distance = 10.0
        if center is not None:
            to_center = center - camera_location
            direct_distance = float(to_center.length)
            projected_distance = float(to_center.dot(forward))
            if math.isfinite(projected_distance) and projected_distance > max(0.001, direct_distance * 0.25):
                view_distance = projected_distance
            elif math.isfinite(direct_distance) and direct_distance > 0.001:
                view_distance = direct_distance
        view_distance = _resolved_view_distance(config, view_distance)
        region_data.view_perspective = "PERSP"
        region_data.view_rotation = view_rotation
        region_data.view_location = camera_location + forward * view_distance
        region_data.view_distance = view_distance
        lens = _view3d_lens_from_scene_camera(camera)
        if lens is not None:
            space.lens = lens
        camera_data = getattr(camera, "data", None)
        clip_start = _positive_config_float(config, "viewport_clip_start")
        if clip_start is None:
            clip_start = getattr(camera_data, "clip_start", None)
        clip_end = _positive_config_float(config, "viewport_clip_end")
        if clip_end is None:
            clip_end = getattr(camera_data, "clip_end", None)
        if clip_start is not None:
            space.clip_start = float(clip_start)
        if clip_end is not None:
            space.clip_end = float(clip_end)
        region_data.update()
    except Exception:
        region_data.view_perspective = "PERSP"
        region_data.update()


def _visual_selection_source_ids(objects: Sequence[Any]) -> set[int]:
    renderable_descendant_ids: set[int] = set()
    for obj in objects:
        if getattr(obj, "type", "") in RENDERABLE_SELECTION_TYPES:
            current = getattr(obj, "parent", None)
            while current is not None:
                renderable_descendant_ids.add(id(current))
                current = getattr(current, "parent", None)
    return {
        id(obj)
        for obj in objects
        if getattr(obj, "type", "") in RENDERABLE_SELECTION_TYPES or id(obj) in renderable_descendant_ids
    }


def _deselect_scene_objects(bpy: Any) -> None:
    try:
        bpy.ops.object.select_all(action="DESELECT")
    except RuntimeError:
        pass


def _prepare_imported_object_interaction_state(
    objects: Sequence[Any],
    *,
    selectable_imported_objects: bool,
    tag_dynamic_body_transforms: bool,
) -> set[int]:
    if not selectable_imported_objects and not tag_dynamic_body_transforms:
        return set()
    for obj in objects:
        obj.hide_select = True
    visual_source_ids = _visual_selection_source_ids(objects)
    if selectable_imported_objects:
        _set_imported_object_selectability(objects, visual_source_ids)
    return visual_source_ids


def _set_imported_object_selectability(objects: Sequence[Any], visual_source_ids: set[int]) -> None:
    for obj in objects:
        if getattr(obj, "type", "") in {"CAMERA", "LIGHT"} or id(obj) in visual_source_ids:
            obj.hide_select = False


def _tag_dynamic_body_transforms(
    scene: Any,
    path_index: Mapping[str, Any],
    *,
    dynamic_body_root: str,
    visual_source_ids: set[int],
    selectable_imported_objects: bool,
) -> dict[str, Any]:
    dynamic_body_edit_owners = []
    dynamic_body_selection_sources = []
    object_usd_paths = {}
    objects_by_usd_path = {}
    for obj in scene.objects:
        usd_path = usd_paths.resolved_usd_path(obj, path_index)
        if not usd_path:
            continue
        object_usd_paths[obj.name] = usd_path
        objects_by_usd_path.setdefault(usd_path, obj)

    tagged_body_paths = set()
    for obj in scene.objects:
        usd_path = object_usd_paths.get(obj.name, "")
        body_path = usd_paths.nearest_dynamic_body_path(usd_path, path_index, dynamic_body_root)
        if not body_path:
            continue
        owner_obj = objects_by_usd_path.get(body_path)
        if owner_obj is None:
            continue
        if usd_path == body_path and body_path not in tagged_body_paths:
            dynamic_body_edit_owners.append(obj)
            tagged_body_paths.add(body_path)
            usd_paths.tag_body_edit_owner(obj, body_path)
            continue
        if id(obj) in visual_source_ids:
            dynamic_body_selection_sources.append(obj)
            usd_paths.tag_body_selection_source(obj, body_path, owner_obj)
    if selectable_imported_objects:
        for owner in dynamic_body_edit_owners:
            owner.hide_select = False
        for source in dynamic_body_selection_sources:
            source.hide_select = False
    return {
        "dynamic_body_edit_owners": dynamic_body_edit_owners,
        "dynamic_body_selection_sources": dynamic_body_selection_sources,
        "dynamic_body_root": dynamic_body_root,
    }


def _print_setup_summary(
    scene: Any,
    usd_path: Path,
    config: Mapping[str, Any],
    path_index: Mapping[str, Any],
    camera_binding: Mapping[str, Any],
    tag_result: Mapping[str, Any],
) -> None:
    print("OVRTX interactive setup complete")
    print(f"usd_path={usd_path}")
    print(f"render_engine={scene.render.engine}")
    print(f"render_product_path={config['render_product_path']}")
    print(f"sample_range={scene.ovrtx_example.min_samples}-{scene.ovrtx_example.max_samples}")
    print(f"preview_camera={config['camera_prim_path']}")
    print(f"sync_viewport_camera={scene.ovrtx_example.sync_viewport_camera}")
    print(f"viewport_orbit_distance={config.get('viewport_orbit_distance')}")
    print(f"viewport_pitch_offset_degrees={config.get('viewport_pitch_offset_degrees')}")
    print(f"viewport_clip_start={config.get('viewport_clip_start')}")
    print(f"viewport_clip_end={config.get('viewport_clip_end')}")
    print(f"timeline_range={scene.frame_start}-{scene.frame_end}")
    print(f"camera_binding_status={camera_binding['status']}")
    print(f"camera_binding_source={camera_binding['match_source']}")
    print(f"camera_binding_candidate_count={camera_binding['candidate_count']}")
    print(f"scene_camera={camera_binding['scene_camera']}")
    if config["tag_dynamic_body_transforms"]:
        owners = tag_result["dynamic_body_edit_owners"]
        sources = tag_result["dynamic_body_selection_sources"]
        print(f"tagged_dynamic_body_count={len(owners)}")
        print("tagged_dynamic_body_paths=" + ",".join(str(owner.get("ovrtx.usd_prim_path", "")) for owner in owners[:24]))
        print(f"tagged_dynamic_body_selection_source_count={len(sources)}")
        print(f"dynamic_body_root={tag_result['dynamic_body_root']}")
        print(f"usd_path_index_available={path_index['available']}")
        if not path_index["available"]:
            print(f"usd_path_index_reason={path_index['reason']}")
    print("live_viewport=ovrtx render-product preview with Blender viewport camera controls")


def _register_bounded_shutdown(bpy: Any, config: Mapping[str, Any], addon: Any) -> None:
    if float(config.get("interactive_duration_s") or 0.0) <= 0.0:
        return

    def _ovrtx_bounded_session_shutdown() -> None:
        print(f"bounded_interactive_shutdown=after_{config['interactive_duration_s']}_seconds")
        try:
            written = addon.write_viewport_session_outputs()
            print(f"viewport_session_output_write_count={written}")
            bpy.ops.wm.quit_blender()
        except Exception as exc:
            print(f"bounded_interactive_shutdown=failed: {type(exc).__name__}: {exc}")
        return None

    bpy.app.timers.register(
        _ovrtx_bounded_session_shutdown,
        first_interval=max(0.1, float(config["interactive_duration_s"])),
    )


def _register_shared_stage_redraw(bpy: Any, config: Mapping[str, Any]) -> None:
    if not config.get("shared_stage_composition"):
        return

    def _ovrtx_shared_stage_redraw_interval() -> float:
        mode = config["viewport_redraw_pressure_mode"]
        if mode in {"uncapped-timer", "forced-redraw"}:
            return 0.0
        return max(0.01, 1.0 / float(config["composition_update_fps"]))

    def _ovrtx_shared_stage_redraw() -> float:
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
        if config["viewport_redraw_pressure_mode"] == "forced-redraw":
            try:
                bpy.ops.wm.redraw_timer(type=config["forced_redraw_timer_type"], iterations=1)
            except RuntimeError:
                pass
        return _ovrtx_shared_stage_redraw_interval()

    bpy.app.timers.register(
        _ovrtx_shared_stage_redraw,
        first_interval=_ovrtx_shared_stage_redraw_interval(),
    )
    print("timeline_playback=paused")
    print("shared_stage_composition=enabled")
    print("composition_mode=async_latest_pose")
    print(f"viewport_redraw_pressure_mode={config['viewport_redraw_pressure_mode']}")
