# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.interactive_setup import (  # noqa: E402
    configure_scene,
    configure_viewports,
    bind_scene_camera,
    _prepare_imported_object_interaction_state,
    _resolved_view_distance,
    _visual_selection_source_ids,
)


class _FakeObject(dict):
    def __init__(
        self,
        name: str,
        type: str = "EMPTY",
        *,
        data: object | None = None,
        parent: "_FakeObject | None" = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.type = type
        self.data = data
        self.parent = parent
        self.children: list[_FakeObject] = []
        self.hide_select = False
        if parent is not None:
            parent.children.append(self)


def _scene(*objects: _FakeObject) -> SimpleNamespace:
    return SimpleNamespace(objects=list(objects), camera=_FakeObject("PreviousCamera", "CAMERA"))


def test_exact_stage_setup_keeps_viewport_sync_enabled_without_path_properties() -> None:
    scene = SimpleNamespace(render=SimpleNamespace(), ovrtx_example=SimpleNamespace())

    configure_scene(
        scene,
        {
            "width": 1280,
            "height": 720,
            "render_product_path": "/Render/Product",
            "min_samples": 1,
            "max_samples": 1,
            "color_presentation": "scene_linear_hdr",
            "camera_prim_path": "",
            "sync_viewport_camera": True,
        },
    )

    assert scene.ovrtx_example.sync_viewport_camera is True
    assert scene.ovrtx_example.color_presentation_mode == "scene_linear_hdr"


def test_configure_viewports_can_limit_rendered_shading_to_active_screen() -> None:
    def screen() -> SimpleNamespace:
        overlay = SimpleNamespace(
            show_overlays=False,
            show_performance=False,
            show_text=True,
            show_stats=True,
        )
        space = SimpleNamespace(
            type="VIEW_3D",
            shading=SimpleNamespace(type="SOLID"),
            overlay=overlay,
        )
        return SimpleNamespace(
            areas=[SimpleNamespace(type="VIEW_3D", spaces=[space])],
            space=space,
        )

    active = screen()
    inactive = screen()
    bpy = SimpleNamespace(data=SimpleNamespace(screens=[active, inactive]))

    configure_viewports(
        bpy, object(), {"selectable_imported_objects": False}, screen=active
    )

    assert active.space.shading.type == "RENDERED"
    assert inactive.space.shading.type == "SOLID"


def test_bind_scene_camera_uses_imported_source_path() -> None:
    debug = _FakeObject("DebugCamera", "CAMERA")
    shot = _FakeObject("ShotCamera", "CAMERA")
    shot["ovrtx:sourceUsdPath"] = "/World/ShotCamera"
    scene = _scene(debug, shot)

    result = bind_scene_camera(scene, "/World/ShotCamera")

    assert scene.camera is shot
    assert result["status"] == "bound"
    assert result["match_source"] == "sourceUsdPath"
    assert result["scene_camera"] == "ShotCamera"


def test_bind_scene_camera_checks_camera_data_source_path() -> None:
    camera_data = {"ovrtx:sourceUsdPath": "/World/Camera"}
    camera = _FakeObject("ImportedCamera", "CAMERA", data=camera_data)
    scene = _scene(camera)

    result = bind_scene_camera(scene, "/World/Camera")

    assert scene.camera is camera
    assert result["match_source"] == "sourceUsdPath"


def test_bind_scene_camera_uses_hierarchy_path() -> None:
    world = _FakeObject("World")
    camera = _FakeObject("Camera", "CAMERA", parent=world)
    scene = _scene(world, camera)

    result = bind_scene_camera(scene, "/World/Camera")

    assert scene.camera is camera
    assert result["match_source"] == "hierarchy_path"


def test_bind_scene_camera_allows_unique_root_object_fallback() -> None:
    camera = _FakeObject("OvrtxCamera.001", "CAMERA")
    scene = _scene(camera)

    result = bind_scene_camera(scene, "/OvrtxCamera")

    assert scene.camera is camera
    assert result["match_source"] == "root_object_path"


def test_bind_scene_camera_does_not_guess_nested_path_from_root_name() -> None:
    camera = _FakeObject("Camera", "CAMERA")
    scene = _scene(camera)

    result = bind_scene_camera(scene, "/World/Camera")

    assert scene.camera is None
    assert result["status"] == "unresolved"


def test_bind_scene_camera_does_not_match_parented_camera_as_root_path() -> None:
    world = _FakeObject("World")
    camera = _FakeObject("Camera", "CAMERA", parent=world)
    scene = _scene(world, camera)

    result = bind_scene_camera(scene, "/Camera")

    assert scene.camera is None
    assert result["status"] == "unresolved"


def test_bind_scene_camera_leaves_scene_unset_when_ambiguous() -> None:
    first = _FakeObject("ShotCamera", "CAMERA")
    second = _FakeObject("ShotCamera.001", "CAMERA")
    first["ovrtx:sourceUsdPath"] = "/World/ShotCamera"
    second["ovrtx:sourceUsdPath"] = "/World/ShotCamera"
    scene = _scene(first, second)

    result = bind_scene_camera(scene, "/World/ShotCamera")

    assert scene.camera is None
    assert result["status"] == "ambiguous"
    assert result["candidate_count"] == 2


def test_visual_selection_source_ids_precomputes_renderable_descendants() -> None:
    body = _FakeObject("Orange_00")
    mesh = _FakeObject("Orange_00_mesh", "MESH", parent=body)
    light = _FakeObject("KeyLight", "LIGHT")

    visual_ids = _visual_selection_source_ids([body, mesh, light])

    assert id(body) in visual_ids
    assert id(mesh) in visual_ids
    assert id(light) not in visual_ids


def test_visual_interaction_state_skips_per_object_selectability() -> None:
    mesh = _FakeObject("FactoryMesh", "MESH")
    camera = _FakeObject("OvrtxCamera", "CAMERA")

    visual_ids = _prepare_imported_object_interaction_state(
        [mesh, camera],
        selectable_imported_objects=False,
        tag_dynamic_body_transforms=False,
    )

    assert visual_ids == set()
    assert mesh.hide_select is False
    assert camera.hide_select is False


def test_selectable_interaction_state_keeps_visual_objects_selectable() -> None:
    root = _FakeObject("Root")
    mesh = _FakeObject("FactoryMesh", "MESH", parent=root)
    empty = _FakeObject("Helper")
    camera = _FakeObject("OvrtxCamera", "CAMERA")

    visual_ids = _prepare_imported_object_interaction_state(
        [root, mesh, empty, camera],
        selectable_imported_objects=True,
        tag_dynamic_body_transforms=False,
    )

    assert visual_ids == {id(root), id(mesh)}
    assert root.hide_select is False
    assert mesh.hide_select is False
    assert camera.hide_select is False
    assert empty.hide_select is True


def test_resolved_view_distance_prefers_fixture_navigation_default() -> None:
    assert _resolved_view_distance({"viewport_orbit_distance": 100.0}, 5590.0) == 100.0


def test_resolved_view_distance_uses_auto_distance_without_fixture_default() -> None:
    assert _resolved_view_distance({}, 12.5) == 12.5
    assert _resolved_view_distance({"viewport_orbit_distance": -1.0}, 12.5) == 12.5
