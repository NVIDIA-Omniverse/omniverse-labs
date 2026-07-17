# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import viewport_presentation  # noqa: E402


def setup_function() -> None:
    viewport_presentation.reset_viewport_presentation_state()


def _space(*, shading_type: str = "RENDERED", view_perspective: str = "ORTHO") -> SimpleNamespace:
    return SimpleNamespace(
        type="VIEW_3D",
        shading=SimpleNamespace(type=shading_type),
        overlay=SimpleNamespace(
            show_overlays=False,
            show_floor=False,
            show_ortho_grid=False,
            show_axis_x=False,
            show_axis_y=False,
            show_axis_z=False,
        ),
        region_3d=SimpleNamespace(view_perspective=view_perspective),
    )


def _scene(*, camera_type: str = "PERSP") -> SimpleNamespace:
    return SimpleNamespace(camera=SimpleNamespace(data=SimpleNamespace(type=camera_type)))


def test_rendered_orthographic_user_view_stays_ovrtx_rendered() -> None:
    space = _space(shading_type="RENDERED", view_perspective="ORTHO")

    state = viewport_presentation.reconcile_space_presentation(space, _scene())

    assert state["presentation_mode"] == viewport_presentation.OVRTX_RENDERED_PRESENTATION
    assert state["fallback_reason"] == ""
    assert state["fallback_owned_by_addon"] is False
    assert state["changed"] is False
    assert space.shading.type == "RENDERED"
    assert space.overlay.show_ortho_grid is False
    assert space.overlay.show_axis_x is False
    assert space.overlay.show_axis_y is False
    assert space.overlay.show_axis_z is False


def test_orthographic_active_camera_view_stays_ovrtx_rendered() -> None:
    space = _space(shading_type="RENDERED", view_perspective="CAMERA")

    state = viewport_presentation.reconcile_space_presentation(space, _scene(camera_type="ORTHO"))

    assert state["presentation_mode"] == viewport_presentation.OVRTX_RENDERED_PRESENTATION
    assert state["fallback_reason"] == ""
    assert state["fallback_owned_by_addon"] is False
    assert space.shading.type == "RENDERED"


def test_perspective_active_camera_view_stays_ovrtx_rendered() -> None:
    space = _space(shading_type="RENDERED", view_perspective="CAMERA")

    state = viewport_presentation.reconcile_space_presentation(space, _scene(camera_type="PERSP"))

    assert state["presentation_mode"] == viewport_presentation.OVRTX_RENDERED_PRESENTATION
    assert state["fallback_reason"] == ""
    assert state["fallback_owned_by_addon"] is False
    assert space.shading.type == "RENDERED"


def test_existing_solid_orthographic_view_is_not_addon_owned() -> None:
    space = _space(shading_type="SOLID", view_perspective="ORTHO")

    state = viewport_presentation.reconcile_space_presentation(space, _scene())

    assert state["presentation_mode"] == viewport_presentation.OVRTX_RENDERED_PRESENTATION
    assert state["fallback_reason"] == ""
    assert state["fallback_owned_by_addon"] is False
    assert state["changed"] is False
    assert space.shading.type == "SOLID"


def test_monitor_does_not_create_first_fallback_without_render_engine_artifact() -> None:
    space = _space(shading_type="RENDERED", view_perspective="ORTHO")
    area = SimpleNamespace(type="VIEW_3D", spaces=[space], tag_redraw=lambda: None)
    screen = SimpleNamespace(areas=[area])
    bpy = SimpleNamespace(
        context=SimpleNamespace(
            window_manager=SimpleNamespace(windows=[SimpleNamespace(screen=screen, scene=_scene())])
        )
    )

    states = viewport_presentation.reconcile_all_viewports(bpy)

    assert states[0]["presentation_mode"] == viewport_presentation.OVRTX_RENDERED_PRESENTATION
    assert states[0]["fallback_reason"] == ""
    assert states[0]["fallback_owned_by_addon"] is False
    assert states[0]["changed"] is False
    assert space.shading.type == "RENDERED"
