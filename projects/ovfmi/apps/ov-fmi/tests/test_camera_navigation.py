import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_OV_FMI = Path(__file__).resolve().parents[1]
if str(_OV_FMI) not in sys.path:
    sys.path.insert(0, str(_OV_FMI))

from main import (  # noqa: E402
    FirstPersonCameraController,
    _ancestor_prim_paths,
    _camera_matrix_from_pose,
    _camera_pose_from_matrix,
    _local_matrix_from_world,
    _make_preview_overlay_usda,
    _preview_light_from_bounds,
    _preview_camera_pose_from_bounds,
    _scale_matrix_from_local_xform,
    _select_preview_camera_prim,
    _select_tracked_body_prims,
    _world_up_from_axis,
    _world_up_from_schema,
)


class _FakeBinding:
    def __init__(self):
        self.writes = []

    def write(self, matrix):
        self.writes.append(np.array(matrix, copy=True))


class _FakeDisplay:
    def __init__(self, nav):
        self._nav = nav

    def get_navigation_input(self):
        return self._nav


def _nav(**kwargs):
    defaults = {
        "forward": 0.0,
        "strafe": 0.0,
        "lift": 0.0,
        "look_dx": 0.0,
        "look_dy": 0.0,
        "fast": False,
        "slow": False,
        "look_active": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _translation_matrix(x, y, z):
    matrix = np.eye(4, dtype=np.float64)
    matrix[3, 0:3] = [x, y, z]
    return matrix


def _z_rotation_matrix(angle):
    matrix = np.eye(4, dtype=np.float64)
    c = math.cos(angle)
    s = math.sin(angle)
    matrix[0, 0] = c
    matrix[0, 1] = s
    matrix[1, 0] = -s
    matrix[1, 1] = c
    return matrix


def test_camera_pose_round_trips_through_ovrtx_matrix():
    position = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    yaw = math.radians(35.0)
    pitch = math.radians(-12.0)

    matrix = _camera_matrix_from_pose(position, yaw, pitch)
    out_position, out_yaw, out_pitch = _camera_pose_from_matrix(matrix)

    np.testing.assert_allclose(out_position, position)
    assert math.isclose(out_yaw, yaw)
    assert math.isclose(out_pitch, pitch)


def test_camera_pose_round_trips_with_z_up():
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    position = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    yaw = math.radians(-20.0)
    pitch = math.radians(15.0)

    matrix = _camera_matrix_from_pose(position, yaw, pitch, world_up)
    out_position, out_yaw, out_pitch = _camera_pose_from_matrix(matrix, world_up)

    np.testing.assert_allclose(out_position, position)
    assert math.isclose(out_yaw, yaw)
    assert math.isclose(out_pitch, pitch)


def test_first_person_controller_moves_forward_in_camera_space():
    binding = _FakeBinding()
    display = _FakeDisplay(_nav(forward=1.0))
    initial = _camera_matrix_from_pose(np.zeros(3, dtype=np.float64), 0.0, 0.0)

    controller = FirstPersonCameraController(
        display,
        binding,
        initial,
        move_speed=2.0,
        mouse_sensitivity_degrees=0.1,
    )
    controller.update(0.5)

    assert len(binding.writes) == 1
    np.testing.assert_allclose(binding.writes[0][0, 3, 0:3], [0.0, 0.0, -1.0])


def test_first_person_controller_lifts_along_z_up_axis():
    binding = _FakeBinding()
    display = _FakeDisplay(_nav(lift=1.0))
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    initial = _camera_matrix_from_pose(np.zeros(3, dtype=np.float64), 0.0, 0.0, world_up)

    controller = FirstPersonCameraController(
        display,
        binding,
        initial,
        move_speed=2.0,
        mouse_sensitivity_degrees=0.1,
        world_up=world_up,
    )
    controller.update(0.5)

    assert len(binding.writes) == 1
    np.testing.assert_allclose(binding.writes[0][0, 3, 0:3], [0.0, 0.0, 1.0])


def test_z_up_horizontal_mouselook_yaws_around_world_up():
    binding = _FakeBinding()
    display = _FakeDisplay(_nav(look_active=True, look_dx=100.0))
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    initial = _camera_matrix_from_pose(
        np.zeros(3, dtype=np.float64),
        math.radians(-45.0),
        math.radians(-25.0),
        world_up,
    )
    initial_forward = -initial[2, 0:3]

    controller = FirstPersonCameraController(
        display,
        binding,
        initial,
        move_speed=2.0,
        mouse_sensitivity_degrees=0.1,
        world_up=world_up,
    )
    controller.update(0.0)

    assert len(binding.writes) == 1
    updated = binding.writes[0][0]
    updated_forward = -updated[2, 0:3]
    updated_up = updated[1, 0:3]

    # Horizontal mouse movement should change heading while preserving pitch.
    assert not np.allclose(updated_forward, initial_forward)
    assert math.isclose(
        float(np.dot(updated_forward, world_up)),
        float(np.dot(initial_forward, world_up)),
    )

    # With no roll, camera-up is the stage-up vector projected into the image plane.
    expected_up = world_up - updated_forward * float(np.dot(world_up, updated_forward))
    expected_up = expected_up / np.linalg.norm(expected_up)
    np.testing.assert_allclose(updated_up, expected_up)


def test_preview_camera_pose_uses_stage_bounds():
    bounds = {
        "valid": True,
        "center": [10.0, 2.0, -3.0],
        "radius": 4.0,
    }

    position, yaw, pitch, near, far = _preview_camera_pose_from_bounds(bounds)
    matrix = _camera_matrix_from_pose(position, yaw, pitch)
    _out_position, out_yaw, out_pitch = _camera_pose_from_matrix(matrix)

    assert np.linalg.norm(position - np.array(bounds["center"])) > bounds["radius"]
    assert near > 0.0
    assert far > near
    assert math.isclose(out_yaw, yaw)
    assert math.isclose(out_pitch, pitch)


def test_preview_camera_pose_respects_z_up():
    bounds = {
        "valid": True,
        "center": [10.0, 2.0, -3.0],
        "radius": 4.0,
    }
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    position, yaw, pitch, _near, _far = _preview_camera_pose_from_bounds(bounds, world_up)
    matrix = _camera_matrix_from_pose(position, yaw, pitch, world_up)
    _out_position, out_yaw, out_pitch = _camera_pose_from_matrix(matrix, world_up)

    assert position[2] > bounds["center"][2]
    assert math.isclose(out_yaw, yaw)
    assert math.isclose(out_pitch, pitch)


def test_preview_overlay_authors_render_product_without_touching_source(tmp_path):
    source = tmp_path / "plain.usda"
    source.write_text("#usda 1.0\n")
    usda = _make_preview_overlay_usda(
        str(source),
        "/Render/Camera",
        "/OvFmiPreviewCamera",
        {"valid": False},
    )

    assert f"@{source.resolve().as_posix()}@" in usda
    assert 'def Camera "OvFmiPreviewCamera"' in usda
    assert 'def SphereLight "OvFmiPreviewLight"' in usda
    assert "float inputs:intensity" in usda
    assert 'def RenderProduct "Camera"' in usda
    assert "rel camera = </OvFmiPreviewCamera>" in usda
    assert "rel orderedVars = </Render/Camera/LdrColor>" in usda
    assert source.read_text() == "#usda 1.0\n"


def test_preview_overlay_can_preserve_existing_camera(tmp_path):
    source = tmp_path / "plain.usda"
    source.write_text("#usda 1.0\n")
    usda = _make_preview_overlay_usda(
        str(source),
        "/Render/Camera",
        "/World/Render/Camera",
        {"valid": False},
        author_camera=False,
    )

    assert 'def Camera "Camera"' not in usda
    assert 'def SphereLight "OvFmiPreviewLight"' in usda
    assert 'def RenderProduct "Camera"' in usda
    assert "rel camera = </World/Render/Camera>" in usda
    assert source.read_text() == "#usda 1.0\n"


def test_preview_camera_selection_prefers_existing_default_camera():
    assert _select_preview_camera_prim(
        {"cameras": ["/World/Camera"]}, "/World/Camera"
    ) == ("/World/Camera", False)
    assert _select_preview_camera_prim(
        {"cameras": []}, "/World/Camera"
    ) == ("/OvFmiPreviewCamera", True)
    assert _select_preview_camera_prim(
        {"cameras": []}, "/CustomCamera"
    ) == ("/CustomCamera", True)


def test_preview_camera_selection_prefers_stage_render_camera():
    assert _select_preview_camera_prim(
        {"cameras": ["/World/Render/Camera"]},
        "/World/Camera",
    ) == ("/World/Render/Camera", False)


def test_preview_camera_selection_avoids_render_product_path():
    assert _select_preview_camera_prim(
        {"cameras": ["/Render/Camera", "/World/Render/Camera"]},
        "/World/Camera",
        "/Render/Camera",
    ) == ("/World/Render/Camera", False)


def test_preview_light_scales_from_stage_bounds():
    position = np.array([5.0, 6.0, 7.0], dtype=np.float64)
    light_position, light_radius, intensity = _preview_light_from_bounds(
        {"valid": True, "radius": 10.0},
        position,
    )

    np.testing.assert_allclose(light_position, position)
    assert light_radius > 0.0
    assert intensity >= 50000.0


def test_world_up_from_schema_normalises_vector():
    up, source = _world_up_from_schema({
        "world_up": {
            "vector": [0.0, 0.0, 10.0],
            "source": "stage upAxis=Z",
        }
    })

    np.testing.assert_allclose(up, [0.0, 0.0, 1.0])
    assert source == "stage upAxis=Z"


def test_world_up_from_axis_overrides_to_z_up():
    up, source = _world_up_from_axis("z")

    np.testing.assert_allclose(up, [0.0, 0.0, 1.0])
    assert source == "command line --up-axis=Z"


def test_world_up_from_axis_overrides_to_y_up():
    up, source = _world_up_from_axis("Y")

    np.testing.assert_allclose(up, [0.0, 1.0, 0.0])
    assert source == "command line --up-axis=Y"


def test_select_tracked_body_prims_uses_tensor_paths():
    requested = ["/World/A", "/World/Container", "/World/B"]
    tensor_paths = ["/World/B", "/World/A"]

    prims, indices, skipped = _select_tracked_body_prims(requested, tensor_paths, 2)

    assert prims == ["/World/A", "/World/B"]
    assert indices == [1, 0]
    assert skipped == ["/World/Container"]


def test_select_tracked_body_prims_falls_back_to_tensor_count():
    prims, indices, skipped = _select_tracked_body_prims(
        ["/World/A", "/World/B", "/World/C"],
        None,
        2,
    )

    assert prims == ["/World/A", "/World/B"]
    assert indices == [0, 1]
    assert skipped == ["/World/C"]


def test_ancestor_prim_paths_excludes_body_prim():
    assert _ancestor_prim_paths("/World/Asset/Geometry/Body") == [
        "/World",
        "/World/Asset",
        "/World/Asset/Geometry",
    ]


def test_local_matrix_from_world_removes_translated_parent():
    parent = _translation_matrix(2.0, -3.0, 4.0)
    expected_local = _translation_matrix(5.0, 6.0, 7.0)
    world = expected_local @ parent

    local = _local_matrix_from_world(world, np.linalg.inv(parent))

    np.testing.assert_allclose(local, expected_local, atol=1e-12)


def test_local_matrix_from_world_removes_rotated_parent():
    parent = _z_rotation_matrix(math.radians(90.0)) @ _translation_matrix(2.0, 0.0, 0.0)
    expected_local = _translation_matrix(1.0, 2.0, 3.0)
    world = expected_local @ parent

    local = _local_matrix_from_world(world, np.linalg.inv(parent))

    np.testing.assert_allclose(local, expected_local, atol=1e-12)


def test_scale_matrix_from_local_xform_preserves_body_prim_scale():
    scale = np.diag([0.5, 0.25, 0.75, 1.0])
    local = (
        scale
        @ _z_rotation_matrix(math.radians(30.0))
        @ _translation_matrix(4.0, 5.0, 6.0)
    )

    scale_matrix = _scale_matrix_from_local_xform(local)

    np.testing.assert_allclose(scale_matrix, scale)


def test_physics_pose_render_matrix_keeps_scale_without_scaling_translation():
    rigid = (
        _z_rotation_matrix(math.radians(20.0))
        @ _translation_matrix(4.0, 5.0, 6.0)
    )
    scale_matrix = np.diag([0.5, 0.25, 0.75, 1.0])

    rendered = scale_matrix @ rigid

    np.testing.assert_allclose(rendered[3, 0:3], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(
        np.linalg.norm(rendered[0:3, 0:3], axis=1),
        [0.5, 0.25, 0.75],
    )
