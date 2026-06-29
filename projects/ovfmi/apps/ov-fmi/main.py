#!/usr/bin/env python3
"""
ov-fmi: FMI co-simulation with ovrtx rendering and ovphysx physics.

Parses FmuInstance/FmuConnection/FmuMapping prims from a USD stage, loads the
referenced FMU files via fmpy, steps them each frame, writes their outputs back
as USD attribute updates via the ovrtx Python API, and renders the evolving
stage in a real-time OpenGL window.

The default mode opens a native OS window and presents rendered frames using
OpenGL.  On NVIDIA GPUs with cuda-python installed, frames stay GPU-resident
(CUDA → GL texture via cudaGraphicsGLRegisterImage, zero CPU readback).  Without
CUDA interop the frames are mapped to CPU and uploaded to the GL texture each
frame.

When ovphysx is available, the physics loop runs:
  1. Read rigid-body poses and velocities from ovphysx tensor bindings
  2. Feed physics state into FMU inputs (physx:position, physx:velocity)
  3. Step FMUs (they also drive visual attributes via the FMI USD schema)
  4. Write FMU force outputs (physx:force) to the force tensor binding
  5. Step physics simulation
  6. Convert updated poses to 4x4 column-major float64 matrices
  7. Push transforms to ovrtx via bind_attribute

Optional flags:
  --headless   Run without a display window (for CI/testing/batch rendering)
  --png        Save rendered frames as PNG images to _output/ (implies --headless)

NOTE on init order
------------------
ovrtx MUST be initialised BEFORE ovphysx.  Shared Carbonite plugin conflicts
cause failures if ovphysx loads first.  See AGENTS.md.

NOTE on process isolation (legacy, kept for fmi_usd_helper path)
----------------------------------------------------------------
Although ovrtx 0.3 vendors its own namespaced OpenUSD, the fmi_usd_helper
binary and parse_fmi_schema.py subprocess remain the primary schema parsers
(avoiding a pxr pip dependency in the main process).
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Auto-load .env (written by setup.sh) so the app works without sourcing it.
# ---------------------------------------------------------------------------
_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#"):
            continue
        if _line.startswith("export "):
            _line = _line[len("export "):]
        if "=" in _line:
            _k, _v = _line.split("=", 1)
            # Expand ${VAR:-} style references already in the environment
            _v = _v.strip('"').replace("${PYTHONPATH:-}", os.environ.get("PYTHONPATH", ""))
            os.environ.setdefault(_k, _v)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def quaternion_to_4x4_column_major(
    px: float, py: float, pz: float,
    qx: float, qy: float, qz: float, qw: float,
    out,  # length-16 writeable buffer (float64)
) -> None:
    """Convert position + unit quaternion to a column-major 4x4 float64 matrix.

    Fills *out* in-place.  Used to pack ovphysx RIGID_BODY_POSE tensor rows into
    the numpy array format accepted by ovrtx bind_attribute (transform_4x4 semantic).
    """
    length = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if length > 1e-6:
        qx /= length; qy /= length; qz /= length; qw /= length
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    out[ 0] = 1 - 2*(yy+zz);   out[ 1] = 2*(xy+wz);    out[ 2] = 2*(xz-wy);     out[ 3] = 0.0
    out[ 4] = 2*(xy-wz);       out[ 5] = 1 - 2*(xx+zz); out[ 6] = 2*(yz+wx);     out[ 7] = 0.0
    out[ 8] = 2*(xz+wy);       out[ 9] = 2*(yz-wx);     out[10] = 1 - 2*(xx+yy); out[11] = 0.0
    out[12] = px;               out[13] = py;             out[14] = pz;             out[15] = 1.0


_DEFAULT_WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)


def _normalised(vec, fallback):
    length = float(np.linalg.norm(vec))
    if length < 1e-8:
        return np.array(fallback, dtype=np.float64)
    return np.asarray(vec, dtype=np.float64) / length


def _horizontal_camera_frame(world_up):
    up_axis = _normalised(world_up, _DEFAULT_WORLD_UP)
    preferred_forward = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    if abs(float(np.dot(preferred_forward, up_axis))) > 0.95:
        preferred_forward = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    forward0 = preferred_forward - up_axis * float(np.dot(preferred_forward, up_axis))
    forward0 = _normalised(forward0, [0.0, 0.0, -1.0])
    right0 = _normalised(np.cross(forward0, up_axis), [1.0, 0.0, 0.0])
    forward0 = _normalised(np.cross(up_axis, right0), forward0)
    return up_axis, forward0, right0


def _world_up_from_schema(schema: dict) -> tuple[np.ndarray, str]:
    info = schema.get("world_up") or {}
    up = _normalised(info.get("vector", _DEFAULT_WORLD_UP), _DEFAULT_WORLD_UP)
    return up, str(info.get("source", "default Y-up"))


def _world_up_from_axis(up_axis: str) -> tuple[np.ndarray, str]:
    axis = up_axis.strip().upper()
    if axis == "Y":
        return np.array([0.0, 1.0, 0.0], dtype=np.float64), "command line --up-axis=Y"
    if axis == "Z":
        return np.array([0.0, 0.0, 1.0], dtype=np.float64), "command line --up-axis=Z"
    raise ValueError(f"unsupported up axis: {up_axis!r}")


def _select_tracked_body_prims(
    requested_body_prims: list[str],
    tensor_prim_paths,
    num_bodies: int,
) -> tuple[list[str], list[int], list[str]]:
    """Match requested rigid-body prims to ovphysx tensor row indices."""
    if tensor_prim_paths:
        tensor_index = {str(path): i for i, path in enumerate(tensor_prim_paths)}
        tracked = [
            (prim, tensor_index[prim])
            for prim in requested_body_prims
            if prim in tensor_index
        ]
        skipped = [prim for prim in requested_body_prims if prim not in tensor_index]
        return [prim for prim, _idx in tracked], [idx for _prim, idx in tracked], skipped

    n_tracked = min(len(requested_body_prims), int(num_bodies))
    return (
        list(requested_body_prims[:n_tracked]),
        list(range(n_tracked)),
        list(requested_body_prims[n_tracked:]),
    )


def _ancestor_prim_paths(prim_path: str) -> list[str]:
    parts = [part for part in prim_path.split("/") if part]
    ancestors = []
    current = ""
    for part in parts[:-1]:
        current = f"{current}/{part}"
        ancestors.append(current)
    return ancestors


def _local_matrix_from_world(world_matrix: np.ndarray, parent_world_inverse: np.ndarray) -> np.ndarray:
    return np.asarray(world_matrix, dtype=np.float64).reshape(4, 4) @ np.asarray(
        parent_world_inverse,
        dtype=np.float64,
    ).reshape(4, 4)


def _scale_matrix_from_local_xform(local_matrix: np.ndarray) -> np.ndarray:
    """Extract local row-vector scale from an ovrtx transform matrix."""
    matrix = np.asarray(local_matrix, dtype=np.float64).reshape(4, 4)
    scale = np.linalg.norm(matrix[0:3, 0:3], axis=1)
    scale = np.where(scale > 1e-8, scale, 1.0)
    scale_matrix = np.eye(4, dtype=np.float64)
    scale_matrix[0, 0] = scale[0]
    scale_matrix[1, 1] = scale[1]
    scale_matrix[2, 2] = scale[2]
    return scale_matrix


def _read_local_xform_matrix(renderer, prim_path: str) -> np.ndarray:
    try:
        tensor = renderer.read_attribute("omni:xform", [prim_path])
        values = np.from_dlpack(tensor).astype(np.float64, copy=True)
        if values.size != 16:
            raise RuntimeError(f"expected 16 values, got {values.size}")
        return values.reshape(4, 4)
    except Exception as exc:
        print(
            f"WARNING: Could not read parent transform for {prim_path}: {exc}; "
            "assuming identity.",
            file=sys.stderr,
        )
        return np.eye(4, dtype=np.float64)


def _body_parent_world_inverse_matrices(renderer, body_prims: list[str]) -> np.ndarray:
    local_cache: dict[str, np.ndarray] = {}
    inverse_matrices = np.zeros((len(body_prims), 4, 4), dtype=np.float64)

    for i, prim_path in enumerate(body_prims):
        parent_world = np.eye(4, dtype=np.float64)
        for ancestor_path in _ancestor_prim_paths(prim_path):
            local = local_cache.get(ancestor_path)
            if local is None:
                local = _read_local_xform_matrix(renderer, ancestor_path)
                local_cache[ancestor_path] = local
            parent_world = local @ parent_world

        try:
            inverse_matrices[i, :, :] = np.linalg.inv(parent_world)
        except np.linalg.LinAlgError:
            print(
                f"WARNING: Parent transform for {prim_path} is singular; "
                "using identity parent transform for physics rendering.",
                file=sys.stderr,
            )
            inverse_matrices[i, :, :] = np.eye(4, dtype=np.float64)

    return inverse_matrices


def _body_local_scale_matrices(renderer, body_prims: list[str]) -> np.ndarray:
    scale_matrices = np.zeros((len(body_prims), 4, 4), dtype=np.float64)

    for i, prim_path in enumerate(body_prims):
        local = _read_local_xform_matrix(renderer, prim_path)
        scale_matrices[i, :, :] = _scale_matrix_from_local_xform(local)

    return scale_matrices


def _camera_pose_from_matrix(matrix, world_up=_DEFAULT_WORLD_UP) -> tuple[np.ndarray, float, float]:
    """Extract a simple yaw/pitch camera pose from an ovrtx transform matrix."""
    mat = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    position = np.array(mat[3, 0:3], dtype=np.float64)
    up_axis, forward0, right0 = _horizontal_camera_frame(world_up)

    # USD cameras look along local -Z. ovrtx transform matrices are row-vector
    # matrices, so row 2 is local +Z in world space.
    forward = _normalised(-mat[2, 0:3], forward0)
    vertical = max(-1.0, min(1.0, float(np.dot(forward, up_axis))))
    pitch = math.asin(vertical)
    horizontal = forward - up_axis * vertical
    if float(np.linalg.norm(horizontal)) < 1e-8:
        yaw = 0.0
    else:
        horizontal = _normalised(horizontal, forward0)
        yaw = math.atan2(float(np.dot(horizontal, right0)), float(np.dot(horizontal, forward0)))
    return position, yaw, pitch


def _camera_basis_from_yaw_pitch(yaw: float, pitch: float, world_up=_DEFAULT_WORLD_UP):
    up_axis, forward0, right0 = _horizontal_camera_frame(world_up)
    cos_pitch = math.cos(pitch)
    horizontal = forward0 * math.cos(yaw) + right0 * math.sin(yaw)
    forward = _normalised(horizontal * cos_pitch + up_axis * math.sin(pitch), forward0)
    right = _normalised(np.cross(forward, up_axis), right0)
    up = _normalised(np.cross(right, forward), up_axis)
    return right, up, forward


def _camera_matrix_from_pose(position, yaw: float, pitch: float, world_up=_DEFAULT_WORLD_UP) -> np.ndarray:
    right, up, forward = _camera_basis_from_yaw_pitch(yaw, pitch, world_up)
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0:3] = right
    matrix[1, 0:3] = up
    matrix[2, 0:3] = -forward
    matrix[3, 0:3] = position
    return matrix


def _read_camera_matrix(renderer, camera_prim: str):
    try:
        tensor = renderer.read_attribute("omni:xform", [camera_prim])
        values = np.from_dlpack(tensor).astype(np.float64, copy=True)
        if values.size != 16:
            raise RuntimeError(f"expected 16 values, got {values.size}")
        return values.reshape(4, 4)
    except Exception as exc:
        print(
            f"WARNING: Could not read camera transform for {camera_prim}: {exc}",
            file=sys.stderr,
        )
        return None


def _preview_camera_pose_from_bounds(
    bounds: dict,
    world_up=_DEFAULT_WORLD_UP,
) -> tuple[np.ndarray, float, float, float, float]:
    """Return a conservative camera pose and clipping range for a stage bound."""
    if bounds.get("valid"):
        center = np.array(bounds.get("center", [0.0, 0.0, 0.0]), dtype=np.float64)
        radius = max(float(bounds.get("radius", 0.0)), 1.0)
    else:
        center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        radius = 2.0

    up_axis, forward0, right0 = _horizontal_camera_frame(world_up)
    view_dir = _normalised(right0 + up_axis * 0.65 - forward0, [0.0, 0.0, 1.0])
    vertical_fov = math.radians(46.0)
    distance = max(radius / math.sin(vertical_fov * 0.5), radius * 2.5, 1.0)
    position = center + view_dir * distance * 1.15

    forward = _normalised(center - position, forward0)
    vertical = max(-1.0, min(1.0, float(np.dot(forward, up_axis))))
    pitch = math.asin(vertical)
    horizontal = _normalised(forward - up_axis * vertical, forward0)
    yaw = math.atan2(float(np.dot(horizontal, right0)), float(np.dot(horizontal, forward0)))

    near = max(radius * 0.0005, 0.001)
    far = max(distance + radius * 6.0, 1000.0)
    return position, yaw, pitch, near, far


def _preview_light_from_bounds(bounds: dict, camera_position: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return a point-like preview light position, radius, and intensity."""
    radius = max(float(bounds.get("radius", 0.0)), 1.0) if bounds.get("valid") else 2.0
    light_radius = max(radius * 0.04, 0.05)
    intensity = max(radius * radius * 2500.0, 500000.0)
    return np.asarray(camera_position, dtype=np.float64), light_radius, intensity


def _usd_quote(value: str) -> str:
    return json.dumps(value)


def _usd_asset_path(path: str) -> str:
    return Path(path).resolve().as_posix().replace("@", "%40")


def _split_usd_prim_path(path: str) -> list[str]:
    parts = [p for p in path.split("/") if p]
    if not path.startswith("/") or not parts:
        raise ValueError(f"USD prim path must be absolute: {path}")
    return parts


def _indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def _usda_prim_block(path: str, type_name: str, body: str) -> str:
    """Build nested USDA `def` blocks for an absolute prim path."""
    parts = _split_usd_prim_path(path)
    block = f"def {type_name} {_usd_quote(parts[-1])}\n{{\n{_indent(body, 4)}\n}}"
    for part in reversed(parts[:-1]):
        block = f"def {_usd_quote(part)}\n{{\n{_indent(block, 4)}\n}}"
    return block


def _format_usda_matrix(matrix: np.ndarray) -> str:
    rows = []
    for r in range(4):
        rows.append("(" + ", ".join(f"{float(matrix[r, c]):.17g}" for c in range(4)) + ")")
    return "(" + ", ".join(rows) + ")"


def _make_preview_overlay_usda(
    usd_file: str,
    render_product_path: str,
    camera_prim: str,
    bounds: dict,
    world_up=_DEFAULT_WORLD_UP,
    author_camera: bool = True,
) -> str:
    position, yaw, pitch, near, far = _preview_camera_pose_from_bounds(bounds, world_up)
    light_position, light_radius, light_intensity = _preview_light_from_bounds(bounds, position)
    render_var_path = f"{render_product_path.rstrip('/')}/LdrColor"

    camera_block = ""
    if author_camera:
        camera_matrix = _camera_matrix_from_pose(position, yaw, pitch, world_up)
        camera_body = f"""float2 clippingRange = ({near:.9g}, {far:.9g})
float focalLength = 24
float horizontalAperture = 20.955
float verticalAperture = 15.2908
token projection = "perspective"
matrix4d xformOp:transform = {_format_usda_matrix(camera_matrix)}
uniform token[] xformOpOrder = ["xformOp:transform"]"""
        camera_block = f"\n{_usda_prim_block(camera_prim, 'Camera', camera_body)}\n"

    render_product_body = f"""rel camera = <{camera_prim}>
uniform int2 resolution = (1280, 720)
rel orderedVars = <{render_var_path}>

def RenderVar "LdrColor"
{{
    uniform string sourceName = "LdrColor"
}}"""

    light_body = f"""color3f inputs:color = (1, 1, 1)
float inputs:intensity = {light_intensity:.9g}
float inputs:radius = {light_radius:.9g}
double3 xformOp:translate = ({light_position[0]:.17g}, {light_position[1]:.17g}, {light_position[2]:.17g})
uniform token[] xformOpOrder = ["xformOp:translate"]"""

    return f"""#usda 1.0
(
    subLayers = [
        @{_usd_asset_path(usd_file)}@
    ]
)
{camera_block}
{_usda_prim_block("/OvFmiPreviewLight", "SphereLight", light_body)}

{_usda_prim_block(render_product_path, "RenderProduct", render_product_body)}
"""


def _select_preview_camera_prim(
    schema: dict,
    requested_camera_prim: str,
    render_product_path: str = "/Render/Camera",
) -> tuple[str, bool]:
    cameras = list(schema.get("cameras", []))
    usable_cameras = [camera for camera in cameras if camera != render_product_path]

    if requested_camera_prim in usable_cameras:
        return requested_camera_prim, False
    if requested_camera_prim != "/World/Camera":
        return requested_camera_prim, True

    for preferred_camera in ("/World/Render/Camera", "/World/Camera"):
        if preferred_camera in usable_cameras:
            return preferred_camera, False
    if usable_cameras:
        return usable_cameras[0], False
    return "/OvFmiPreviewCamera", True


class FirstPersonCameraController:
    """WASD + mouse-look controller that writes a camera omni:xform binding."""

    _MAX_PITCH = math.radians(89.0)

    def __init__(
        self,
        display,
        binding,
        initial_matrix,
        move_speed: float,
        mouse_sensitivity_degrees: float,
        world_up=_DEFAULT_WORLD_UP,
    ):
        self._display = display
        self._binding = binding
        self._world_up = _normalised(world_up, _DEFAULT_WORLD_UP)
        self._move_speed = max(0.0, float(move_speed))
        self._mouse_radians_per_pixel = math.radians(float(mouse_sensitivity_degrees))
        self._matrix = np.zeros((1, 4, 4), dtype=np.float64)

        if initial_matrix is None:
            _up, forward0, _right = _horizontal_camera_frame(self._world_up)
            initial_matrix = _camera_matrix_from_pose(-forward0 * 5.0, 0.0, 0.0, self._world_up)

        self._position, self._yaw, self._pitch = _camera_pose_from_matrix(initial_matrix, self._world_up)

    def update(self, dt: float):
        nav = self._display.get_navigation_input()
        dirty = False

        if nav.look_active and (nav.look_dx != 0.0 or nav.look_dy != 0.0):
            self._yaw += nav.look_dx * self._mouse_radians_per_pixel
            self._pitch -= nav.look_dy * self._mouse_radians_per_pixel
            self._pitch = max(-self._MAX_PITCH, min(self._MAX_PITCH, self._pitch))
            dirty = True

        right, _up, forward = _camera_basis_from_yaw_pitch(self._yaw, self._pitch, self._world_up)
        movement = (
            forward * nav.forward
            + right * nav.strafe
            + self._world_up * nav.lift
        )
        move_length = float(np.linalg.norm(movement))
        if move_length > 1e-8 and dt > 0.0:
            if move_length > 1.0:
                movement = movement / move_length
            speed = self._move_speed
            if nav.fast:
                speed *= 4.0
            if nav.slow:
                speed *= 0.25
            self._position += movement * speed * dt
            dirty = True

        if dirty:
            self.write()

    def write(self):
        self._matrix[0, :, :] = _camera_matrix_from_pose(
            self._position, self._yaw, self._pitch, self._world_up
        )
        self._binding.write(self._matrix)


# ---------------------------------------------------------------------------
# Phase 1: parse FMI schema (subprocess / compiled helper)
# ---------------------------------------------------------------------------

def _parse_fmi_schema(usd_path: str) -> dict:
    """Run the FMI schema parser and return the parsed JSON.

    Preferred: compiled fmi_usd_helper binary (no usd-core pip dependency).
    Fallback: parse_fmi_schema.py subprocess (requires usd-core installed).
    """
    script_dir = Path(__file__).parent
    usd_python = os.environ.get("USD_PYTHON", sys.executable)
    script = script_dir / "parse_fmi_schema.py"

    def run_python_parser() -> dict:
        result = subprocess.run(
            [usd_python, str(script), usd_path],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"FMI schema parsing failed (exit {result.returncode}):\n{result.stderr}"
            )
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        return json.loads(result.stdout)

    # --- Try compiled C++ helper first ---
    for candidate in [
        script_dir / "fmi_usd_helper" / "build" / "fmi_usd_helper",
        script_dir / "fmi_usd_helper" / "fmi_usd_helper",
    ]:
        if candidate.exists():
            result = subprocess.run(
                [str(candidate), usd_path],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"fmi_usd_helper failed (exit {result.returncode}):\n{result.stderr}"
                )
            if result.stderr.strip():
                print(result.stderr.strip(), file=sys.stderr)
            parsed = json.loads(result.stdout)
            if (
                "render_products" in parsed
                and "stage_bounds" in parsed
                and "world_up" in parsed
                and "overlap_sensors" in parsed
            ):
                return parsed
            print(
                "WARNING: fmi_usd_helper does not provide all preview/physics "
                "metadata; "
                "falling back to parse_fmi_schema.py.",
                file=sys.stderr,
            )
            try:
                return run_python_parser()
            except Exception as exc:
                print(
                    "WARNING: parse_fmi_schema.py fallback failed; preview "
                    f"metadata will use conservative defaults.\n         {exc}",
                    file=sys.stderr,
                )
                parsed.setdefault("body_prims", [])
                parsed.setdefault("render_products", [])
                parsed.setdefault("cameras", [])
                parsed.setdefault("overlap_sensors", {})
                parsed.setdefault("sensor_positions", {})
                parsed.setdefault("stage_bounds", {"valid": False})
                parsed.setdefault(
                    "world_up",
                    {"vector": _DEFAULT_WORLD_UP.tolist(), "source": "default Y-up"},
                )
                return parsed

    # --- Fall back to pxr/usd-core subprocess ---
    # usd-core lives in an isolated venv (USD_PYTHON) because ovrtx refuses
    # to load when usd-core is present in the same environment.
    return run_python_parser()


def _deserialise_instances(raw: dict):
    """Reconstruct FmuParserInstance dataclasses from the subprocess JSON.

    SSP instances (those with "ssp" key instead of "fmu") are flagged so that
    the runtime can load them via SspRuntimeExtracted instead of FmuRuntimeExtractedFMU.
    """
    from fmi_parser import FmuParserInstance, FmuParserConnection, FmuParserMapping, FmuDirection

    instances = {}
    for path, inst in raw.items():
        connections = []
        for conn in inst["connections"]:
            mappings = [
                FmuParserMapping(
                    fmiAttributeName=m["fmiAttributeName"],
                    usdAttributeName=m["usdAttributeName"],
                    direction=FmuDirection(m["direction"]),
                    usdMapping=tuple(m["usdMapping"]),
                )
                for m in conn["mappings"]
            ]
            connections.append(FmuParserConnection(
                enabled=conn["enabled"],
                targets=conn["targets"],
                mappings=mappings,
            ))
        # Use SSP path if available, otherwise FMU path
        fmu_or_ssp = inst.get("ssp") or inst.get("fmu")
        pi = FmuParserInstance(
            enabled=inst["enabled"],
            fmu=fmu_or_ssp,
            path=inst["path"],
            connections=connections,
        )
        # Tag SSP instances for the runtime
        pi._is_ssp = bool(inst.get("ssp"))
        instances[path] = pi
    return instances


# ---------------------------------------------------------------------------
# DLTensor helper — kept for reference but no longer needed in ovrtx 0.3.
# bind_attribute().write() now accepts numpy arrays directly.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Physics attribute conventions
# ---------------------------------------------------------------------------

# Attributes routed to/from ovphysx tensor bindings or scene queries rather
# than ovrtx.  These are recognised in FmuMapping prims and handled by
# PhysxInputHead / OvrtxOutputTail / the simulation loop.
_PHYSX_POSITION_ATTR = "physx:position"
_PHYSX_VELOCITY_ATTR = "physx:velocity"
_PHYSX_FORCE_ATTR = "physx:force"
_PHYSX_OVERLAP_ATTR = "physx:overlap"
_PHYSX_DRIVE_TARGET_VELOCITY_ATTR = "drive:angular:physics:targetVelocity"
_PHYSX_ATTRS = frozenset({
    _PHYSX_POSITION_ATTR,
    _PHYSX_VELOCITY_ATTR,
    _PHYSX_FORCE_ATTR,
    _PHYSX_OVERLAP_ATTR,
    _PHYSX_DRIVE_TARGET_VELOCITY_ATTR,
})


# ---------------------------------------------------------------------------
# FMI / ovrtx adapters
# ---------------------------------------------------------------------------

# Attributes routed through bind_attribute + numpy write (transform_4x4 semantic).
# Maps attr_name → {schema_offset → element_index in 16-float column-major 4x4 matrix}.
#
#   omni:xform:        schema offset IS the matrix element index (0-15) directly.
#   xformOp:translate: schema offset is Vec3 component; maps to the translation
#                      column of the matrix (X=12, Y=13, Z=14).
#
# Adding an entry here is enough to route a new attribute through bind_attribute.
_TRANSFORM_ATTRS: dict = {
    "omni:xform":        {i: i for i in range(16)},
    "xformOp:translate": {0: 12, 1: 13, 2: 14},
}
_TRANSFORM_GROUP_KEY = "_transform_"  # single key for the shared transform group


class OvrtxInputHead:
    """
    Feeds USD attribute values into FMU inputs.

    Initial values are captured from the USD stage by the parser subprocess.
    When FMU outputs are written back to USD attributes that also appear as
    inputs (feedback loops), _attr_cache is updated by OvrtxOutputTail so
    the next step sees the latest value.
    """

    def __init__(self, initial_values: dict):
        # {prim_path: {attr_name: [float, ...]}}
        self._attr_cache = {
            prim: {attr: list(vals) for attr, vals in attrs.items()}
            for prim, attrs in initial_values.items()
        }
        # {instance_path: {fmi_var: (prim_path, attr_name, offset, count)}}
        self._input_map: dict = {}

    def cache_connections(self, instance):
        from fmi_parser import FmuDirection
        imap = {}
        for conn in instance.get_parser_instance().connections:
            if not conn.enabled:
                continue
            for target in conn.targets:
                for m in conn.mappings:
                    if m.direction == FmuDirection.INPUT:
                        # Skip physx: attributes — handled by PhysxInputHead
                        if m.usdAttributeName in _PHYSX_ATTRS:
                            continue
                        offset, count = m.usdMapping
                        imap[m.fmiAttributeName] = (target, m.usdAttributeName, offset, count)
        self._input_map[instance.get_parser_instance().path] = imap

    def empty_cache_for(self, instance):
        self._input_map.pop(instance.get_parser_instance().path, None)

    def write_start_values(self, instance):
        """Set FMU start values from the initial USD attribute state."""
        from fmi_parser import FmuDirection
        start_values = {}
        for conn in instance.get_parser_instance().connections:
            if not conn.enabled:
                continue
            for target in conn.targets:
                for m in conn.mappings:
                    if m.direction != FmuDirection.INPUT:
                        continue
                    # Skip physx: inputs — they have no initial USD values
                    if m.usdAttributeName in _PHYSX_ATTRS:
                        continue
                    vals = self._attr_cache.get(target, {}).get(m.usdAttributeName)
                    if vals is None:
                        continue
                    offset, count = m.usdMapping
                    if count == 1:
                        start_values[m.fmiAttributeName] = vals[offset]
                    else:
                        start_values[m.fmiAttributeName] = (
                            vals[offset: offset + count] if count else vals
                        )
        instance.set_start_values(start_values)

    def get_inputs_for(self, instance) -> dict:
        imap = self._input_map.get(instance.get_parser_instance().path, {})
        inputs = {}
        for fmi_var, (prim, attr, offset, count) in imap.items():
            vals = self._attr_cache.get(prim, {}).get(attr)
            if vals is None:
                continue
            if count == 1:
                inputs[fmi_var] = vals[offset]
            else:
                inputs[fmi_var] = vals[offset: offset + count] if count else vals
        return inputs

    def update_attr(self, prim_path: str, attr_name: str, component: int, value: float):
        """Called by OvrtxOutputTail to keep the input cache in sync with FMU outputs."""
        if prim_path in self._attr_cache and attr_name in self._attr_cache[prim_path]:
            self._attr_cache[prim_path][attr_name][component] = value


class PhysxInputHead:
    """Feeds physics tensor/query data into FMU inputs.

    This is a companion to OvrtxInputHead that handles the `physx:position`
    and `physx:velocity` attribute conventions. It also routes temporary
    conveyor sensor mappings to ovphysx overlap queries. It reads from ovphysx
    tensor bindings (pose and velocity) and scene queries, then provides those
    values as FMU inputs.

    body_prim_index maps prim paths to indices in the pose/velocity tensors.
    """

    def __init__(self, body_prim_index: dict, overlap_sensors: dict | None = None):
        # {prim_path: index_in_tensor}
        self._body_prim_index = body_prim_index
        self._overlap_sensors = {}
        self._overlap_proxy_to_sensor = {}
        for sensor_path, config in (overlap_sensors or {}).items():
            position = config.get("position")
            if position is None:
                continue
            radius = float(config.get("radius", 0.1))
            if radius <= 0.0:
                continue
            proxies = list(config.get("proxies") or [])
            if sensor_path not in proxies:
                proxies.append(sensor_path)
            self._overlap_sensors[sensor_path] = {
                "position": [float(v) for v in position],
                "radius": radius,
                "proxies": proxies,
            }
            for proxy in proxies:
                self._overlap_proxy_to_sensor[proxy] = sensor_path
        # {instance_path: [(fmi_var, prim_path, attr_name, offset, count)]}
        self._input_map: dict = {}
        # Latest physics state, updated each frame before FMU step
        self._physx_cache: dict = {
            p: {
                _PHYSX_POSITION_ATTR: [0.0, 0.0, 0.0],
                _PHYSX_VELOCITY_ATTR: [0.0, 0.0, 0.0],
            }
            for p in body_prim_index
        }
        for sensor_path in self._overlap_sensors:
            self._physx_cache.setdefault(sensor_path, {})[_PHYSX_OVERLAP_ATTR] = [0.0]
        self._overlap_warning_printed = False

    def _overlap_sensor_for_mapping(self, target: str, attr_name: str) -> str | None:
        if attr_name == _PHYSX_OVERLAP_ATTR:
            return self._overlap_proxy_to_sensor.get(target) or (
                target if target in self._overlap_sensors else None
            )
        # Isaac Sim authoring can easily target the standard UsdGeomSphere
        # radius attribute.  When that target belongs to a detected Sensor prim,
        # treat it as the overlap presence signal instead of the static radius.
        if attr_name == "radius":
            return self._overlap_proxy_to_sensor.get(target)
        return None

    def cache_connections(self, instance):
        from fmi_parser import FmuDirection
        entries = []
        for conn in instance.get_parser_instance().connections:
            if not conn.enabled:
                continue
            for target in conn.targets:
                for m in conn.mappings:
                    if m.direction != FmuDirection.INPUT:
                        continue
                    offset, count = m.usdMapping
                    sensor_path = self._overlap_sensor_for_mapping(
                        target, m.usdAttributeName
                    )
                    if sensor_path is not None:
                        entries.append((
                            m.fmiAttributeName,
                            sensor_path,
                            _PHYSX_OVERLAP_ATTR,
                            offset,
                            count,
                        ))
                    elif m.usdAttributeName in (
                        _PHYSX_POSITION_ATTR,
                        _PHYSX_VELOCITY_ATTR,
                    ):
                        entries.append((
                            m.fmiAttributeName,
                            target,
                            m.usdAttributeName,
                            offset,
                            count,
                        ))
        self._input_map[instance.get_parser_instance().path] = entries

    def empty_cache_for(self, instance):
        self._input_map.pop(instance.get_parser_instance().path, None)

    def update_from_tensors(self, pose_data, vel_data):
        """Update internal cache from ovphysx tensor binding data.

        pose_data: np.ndarray [N, 7] — px py pz qx qy qz qw
        vel_data:  np.ndarray [N, 6] — vx vy vz wx wy wz (linear + angular)
        """
        for prim, idx in self._body_prim_index.items():
            if idx < pose_data.shape[0]:
                self._physx_cache[prim][_PHYSX_POSITION_ATTR] = [
                    float(pose_data[idx, 0]),
                    float(pose_data[idx, 1]),
                    float(pose_data[idx, 2]),
                ]
            if vel_data is not None and idx < vel_data.shape[0]:
                self._physx_cache[prim][_PHYSX_VELOCITY_ATTR] = [
                    float(vel_data[idx, 0]),
                    float(vel_data[idx, 1]),
                    float(vel_data[idx, 2]),
                ]

    def update_overlap_sensors(self, physx, geometry_type, query_mode):
        """Update overlap sensor cache from ovphysx scene queries."""
        if not self._overlap_sensors:
            return
        if geometry_type is None or query_mode is None:
            if not self._overlap_warning_printed:
                print(
                    "WARNING: overlap sensor FMI inputs disabled; "
                    "ovphysx scene query enums are unavailable.",
                    file=sys.stderr,
                )
                self._overlap_warning_printed = True
            return
        for sensor_path, config in self._overlap_sensors.items():
            try:
                hits = physx.overlap(
                    geometry_type,
                    mode=query_mode,
                    radius=config["radius"],
                    position=config["position"],
                )
            except Exception as exc:
                if not self._overlap_warning_printed:
                    print(
                        "WARNING: overlap sensor FMI inputs disabled; "
                        f"scene query failed for {sensor_path}.\n"
                        f"         {exc}",
                        file=sys.stderr,
                    )
                    self._overlap_warning_printed = True
                return
            values = [1.0 if hits else 0.0]
            self._physx_cache.setdefault(sensor_path, {})[_PHYSX_OVERLAP_ATTR] = values

    def get_inputs_for(self, instance) -> dict:
        """Return physics-sourced inputs for this FMU instance."""
        entries = self._input_map.get(instance.get_parser_instance().path, [])
        inputs = {}
        for fmi_var, prim, attr_name, offset, count in entries:
            vals = self._physx_cache.get(prim, {}).get(attr_name)
            if vals is None:
                continue
            if count == 1:
                inputs[fmi_var] = vals[offset]
            else:
                inputs[fmi_var] = vals[offset: offset + count] if count else vals
        return inputs


class OvrtxOutputTail:
    """
    Consumes FMU output results and writes them to the ovrtx renderer.

    For regular attributes: renderer.write_attribute() with float32 data.
    For transform attributes (omni:xform, xformOp:translate): bind_attribute()
    with numpy float64 [N, 4, 4] matrices written via binding.write().
    Recognised attrs are in _TRANSFORM_ATTRS.

    Call finalize_transform_bindings() once after all cache_connections() calls.
    """

    def __init__(self, renderer, input_head: OvrtxInputHead):
        self._renderer = renderer
        self._input_head = input_head
        # {instance_path: [(fmi_var, [(prim, attr, offset, count)])]}
        self._output_map: dict = {}
        # Non-transform partial-write cache: {prim: {attr: [float, ...]}}
        self._attr_cache: dict = {}
        # Transform group (one per unique transform attr, covering all prims):
        # {attr_name: {"binding": AttributeBinding, "data": ndarray, "prim_index": dict}}
        self._transform_groups: dict = {}
        # Physics force outputs: {instance_path: [(fmi_var, prim, offset, count)]}
        self._force_output_map: dict = {}
        # Accumulated force outputs: {prim_path: [fx, fy, fz]}
        self._pending_forces: dict = {}
        # Articulation drive velocity outputs:
        # {instance_path: [(fmi_var, joint_prim, offset, count)]}
        self._drive_target_output_map: dict = {}
        # Latest target velocity outputs: {joint_prim_path: target_velocity}
        self._pending_drive_targets: dict = {}

    def cache_connections(self, instance):
        from fmi_parser import FmuDirection
        omap = []
        force_entries = []
        drive_target_entries = []
        for conn in instance.get_parser_instance().connections:
            if not conn.enabled:
                continue
            for m in conn.mappings:
                if m.direction == FmuDirection.OUTPUT:
                    offset, count = m.usdMapping
                    if m.usdAttributeName == _PHYSX_FORCE_ATTR:
                        # Track force outputs separately for physics injection
                        for t in conn.targets:
                            force_entries.append((m.fmiAttributeName, t, offset, count))
                    elif m.usdAttributeName == _PHYSX_DRIVE_TARGET_VELOCITY_ATTR:
                        # This is an authored USD drive attribute, but ovphysx
                        # 0.4 requires runtime writes through the articulation
                        # DOF tensor target instead of live USD mutation.
                        for t in conn.targets:
                            drive_target_entries.append((
                                m.fmiAttributeName, t, offset, count
                            ))
                    else:
                        targets = [(t, m.usdAttributeName, offset, count)
                                   for t in conn.targets]
                        omap.append((m.fmiAttributeName, targets))
        self._output_map[instance.get_parser_instance().path] = omap
        if force_entries:
            self._force_output_map[instance.get_parser_instance().path] = force_entries
        if drive_target_entries:
            self._drive_target_output_map[instance.get_parser_instance().path] = (
                drive_target_entries
            )

    def empty_cache_for(self, instance):
        self._output_map.pop(instance.get_parser_instance().path, None)
        self._force_output_map.pop(instance.get_parser_instance().path, None)
        self._drive_target_output_map.pop(instance.get_parser_instance().path, None)

    def get_outputs_for(self, instance) -> list:
        ovrtx_outputs = [fmi_var for fmi_var, _ in
                         self._output_map.get(instance.get_parser_instance().path, [])]
        force_outputs = [fmi_var for fmi_var, _, _, _ in
                         self._force_output_map.get(instance.get_parser_instance().path, [])]
        drive_target_outputs = [fmi_var for fmi_var, _, _, _ in
                                self._drive_target_output_map.get(
                                    instance.get_parser_instance().path, []
                                )]
        return ovrtx_outputs + force_outputs + drive_target_outputs

    def has_drive_target_outputs(self) -> bool:
        return any(self._drive_target_output_map.values())

    def finalize_transform_bindings(self):
        """Create a bind_attribute binding for all transform output targets.

        Covers every attribute listed in _TRANSFORM_ATTRS.
        Must be called once after all cache_connections() calls and before the
        first call to write_outputs().
        """
        # Collect unique prims that have any transform output mapping
        prims_ordered: list = []
        prim_set: set = set()
        for omap in self._output_map.values():
            for _fmi_var, targets in omap:
                for prim, attr_name, _offset, _count in targets:
                    if attr_name in _TRANSFORM_ATTRS and prim not in prim_set:
                        prim_set.add(prim)
                        prims_ordered.append(prim)

        if not prims_ordered:
            return

        from ovrtx import Semantic, PrimMode  # noqa: PLC0415 — deferred: ovrtx unavailable at module level

        n = len(prims_ordered)
        transform_data = np.zeros((n, 4, 4), dtype=np.float64)
        # Identity matrices as baseline (rotation = I, translation = 0)
        for i in range(n):
            transform_data[i, 0, 0] = 1.0
            transform_data[i, 1, 1] = 1.0
            transform_data[i, 2, 2] = 1.0
            transform_data[i, 3, 3] = 1.0

        binding = self._renderer.bind_attribute(
            prims_ordered,
            "omni:xform",
            semantic=Semantic.XFORM_MAT4x4,
            prim_mode=PrimMode.EXISTING_ONLY,
        )
        prim_index = {p: i for i, p in enumerate(prims_ordered)}
        self._transform_groups[_TRANSFORM_GROUP_KEY] = {
            "binding": binding,
            "data": transform_data,
            "prim_index": prim_index,
        }

    def write_outputs(self, instance, outputs: list, result):
        """
        result is a single numpy structured-array row from fmpy.simulate_fmu().
        Fields: ('time', 'output_var1', 'output_var2', ...).
        """
        omap = self._output_map.get(instance.get_parser_instance().path, [])

        # Track whether any transform group was updated this call
        dirty_transforms: set = set()

        for fmi_var, targets in omap:
            if fmi_var not in result.dtype.names:
                continue
            scalar = float(result[fmi_var])

            by_attr: dict = {}
            for prim, attr_name, offset, count in targets:
                by_attr.setdefault((attr_name, offset, count), []).append(prim)

            for (attr_name, offset, count), prims in by_attr.items():

                if attr_name in _TRANSFORM_ATTRS:
                    # Transform write — update the float64 matrix cache and
                    # push via bind_attribute.write() (transform_4x4 semantic).
                    # _TRANSFORM_ATTRS maps schema offset → matrix element index.
                    group = self._transform_groups.get(_TRANSFORM_GROUP_KEY)
                    if group is None:
                        continue
                    comp = offset if count != 0 else 0
                    mat_idx = _TRANSFORM_ATTRS[attr_name].get(comp, comp)
                    row, col = mat_idx // 4, mat_idx % 4
                    for prim in prims:
                        idx = group["prim_index"].get(prim)
                        if idx is None:
                            continue
                        group["data"][idx, row, col] = scalar
                        # Keep input cache in sync using the original schema offset
                        # so feedback-loop reads see the right Vec3 component.
                        self._input_head.update_attr(prim, attr_name, comp, scalar)
                    dirty_transforms.add(_TRANSFORM_GROUP_KEY)

                elif count == 0:
                    # Whole attribute is a scalar
                    arr = np.array([[scalar]] * len(prims), dtype=np.float32)
                    self._renderer.write_attribute(prims, attr_name, arr)
                    for prim in prims:
                        self._input_head.update_attr(prim, attr_name, 0, scalar)

                else:
                    # Partial component write into a cached float32 vector
                    rows = []
                    for prim in prims:
                        cached = self._attr_cache.get(prim, {}).get(attr_name)
                        if cached is None:
                            cached = [0.0] * max(offset + count, count)
                            self._attr_cache.setdefault(prim, {})[attr_name] = cached
                        cached[offset] = scalar
                        self._input_head.update_attr(prim, attr_name, offset, scalar)
                        rows.append(list(cached))
                    arr = np.array(rows, dtype=np.float32)
                    self._renderer.write_attribute(prims, attr_name, arr)

        # Flush all dirty transform groups
        for key in dirty_transforms:
            g = self._transform_groups[key]
            g["binding"].write(g["data"])

        # Process physx:force outputs — accumulate into _pending_forces
        force_entries = self._force_output_map.get(
            instance.get_parser_instance().path, []
        )
        for fmi_var, prim, offset, count in force_entries:
            if fmi_var not in result.dtype.names:
                continue
            scalar = float(result[fmi_var])
            if prim not in self._pending_forces:
                self._pending_forces[prim] = [0.0, 0.0, 0.0]
            comp = offset if count != 0 else 0
            if 0 <= comp < 3:
                self._pending_forces[prim][comp] = scalar

        # Process articulation drive target velocity outputs.
        drive_target_entries = self._drive_target_output_map.get(
            instance.get_parser_instance().path, []
        )
        for fmi_var, prim, offset, count in drive_target_entries:
            if fmi_var not in result.dtype.names:
                continue
            comp = offset if count != 0 else 0
            if comp == 0:
                self._pending_drive_targets[prim] = float(result[fmi_var])

    def get_pending_forces(self) -> dict:
        """Return accumulated force outputs: {prim_path: [fx, fy, fz]}."""
        return self._pending_forces

    def clear_pending_forces(self):
        """Clear accumulated forces after they've been written to physics."""
        self._pending_forces.clear()

    def get_pending_drive_targets(self) -> dict:
        """Return accumulated drive targets: {joint_path: target_velocity}."""
        return self._pending_drive_targets

    def clear_pending_drive_targets(self):
        """Clear drive targets after they've been written to physics."""
        self._pending_drive_targets.clear()


class ArticulationDriveTargetRouter:
    """Routes authored drive target velocity mappings to ovphysx tensors."""

    def __init__(self, physx, tensor_type, root_paths: list[str]):
        if tensor_type is None:
            raise RuntimeError("ovphysx does not expose ARTICULATION_DOF_VELOCITY_TARGET")

        self._binding = None
        self._data = None
        self._dof_lookup: dict[str, list[int]] = {}
        self._missing_targets: set[str] = set()

        if root_paths:
            self._binding = physx.create_tensor_binding(
                prim_paths=root_paths,
                tensor_type=tensor_type,
                raise_if_empty=True,
            )
        else:
            self._binding = physx.create_tensor_binding(
                pattern="/World/**",
                tensor_type=tensor_type,
                raise_if_empty=True,
            )

        if len(self._binding.shape) < 2 or self._binding.shape[1] == 0:
            raise RuntimeError(
                "articulation velocity target binding has no DOFs "
                f"(shape={self._binding.shape})"
            )

        self._data = np.zeros(self._binding.shape, dtype=np.float32)
        try:
            self._binding.read(self._data)
        except Exception:
            # Some tensor bindings are write-only.  Zero is a conservative base.
            pass

        self._root_paths = list(getattr(self._binding, "prim_paths", []) or [])
        self._dof_names = [str(v) for v in (getattr(self._binding, "dof_names", []) or [])]
        for dof_index, name in enumerate(self._dof_names):
            self._dof_lookup.setdefault(name, []).append(dof_index)
            basename = name.rstrip("/").rsplit("/", 1)[-1]
            if basename != name:
                self._dof_lookup.setdefault(basename, []).append(dof_index)

    @property
    def shape(self):
        return self._binding.shape if self._binding is not None else None

    @property
    def root_paths(self):
        return self._root_paths

    @property
    def dof_names(self):
        return self._dof_names

    def _root_indices_for_target(self, target_path: str) -> list[int]:
        if self._data is None:
            return []
        if self._data.shape[0] == 1:
            return [0]
        matched = [
            i for i, root in enumerate(self._root_paths)
            if target_path == root or target_path.startswith(root.rstrip("/") + "/")
        ]
        return matched if matched else list(range(self._data.shape[0]))

    def _dof_indices_for_target(self, target_path: str) -> list[int]:
        target_path = target_path.rstrip("/")
        basename = target_path.rsplit("/", 1)[-1]
        direct = self._dof_lookup.get(target_path)
        if direct:
            return direct
        by_name = self._dof_lookup.get(basename)
        if by_name:
            return by_name
        suffix = "/" + basename
        suffix_matches = [
            i for i, name in enumerate(self._dof_names)
            if name == basename or name.endswith(suffix)
        ]
        return suffix_matches

    def write_targets(self, targets: dict[str, float]) -> bool:
        if self._binding is None or self._data is None:
            return False

        dirty = False
        for target_path, value in targets.items():
            dof_indices = self._dof_indices_for_target(target_path)
            if not dof_indices:
                if target_path not in self._missing_targets:
                    print(
                        "WARNING: no articulation DOF matched drive target "
                        f"{target_path}; available DOFs={self._dof_names}",
                        file=sys.stderr,
                    )
                    self._missing_targets.add(target_path)
                continue
            root_indices = self._root_indices_for_target(target_path)
            for root_index in root_indices:
                for dof_index in dof_indices:
                    if root_index < self._data.shape[0] and dof_index < self._data.shape[1]:
                        self._data[root_index, dof_index] = float(value)
                        dirty = True

        if dirty:
            self._binding.write(self._data)
        return dirty

    def destroy(self):
        if self._binding is not None:
            self._binding.destroy()
            self._binding = None
            self._data = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FMI co-simulation with ovrtx rendering"
    )
    parser.add_argument("usd_file", help="USD stage containing FmuInstance prims")
    parser.add_argument(
        "--render-product", default="/Render/Camera",
        help="Render product prim path (default: /Render/Camera)",
    )
    parser.add_argument(
        "--camera-prim", default="/World/Camera",
        help=(
            "Camera prim to control in the live display window "
            "(default: /World/Camera)."
        ),
    )
    parser.add_argument(
        "--up-axis", type=str.upper, choices=("Y", "Z"),
        help=(
            "Override inferred viewer up axis for preview camera and navigation "
            "(Y or Z)."
        ),
    )
    parser.add_argument(
        "--nav-speed", type=float, default=2.0,
        help="WASD navigation speed in scene units per second (default: 2.0).",
    )
    parser.add_argument(
        "--mouse-sensitivity", type=float, default=0.12,
        help="Mouse-look sensitivity in degrees per pixel (default: 0.12).",
    )
    parser.add_argument(
        "--dt", type=float, default=1.0 / 60.0,
        help="Simulation time step in seconds (default: 1/60)",
    )
    parser.add_argument(
        "--duration", type=float, default=math.inf,
        help="Total simulation duration in seconds (default: infinity)",
    )
    parser.add_argument(
        "--body-prims", default="",
        help=(
            "(Deprecated: physics bodies are now auto-detected from the USD scene.) "
            "Explicit comma-separated list of rigid-body prim paths to track via ovphysx. "
            "Overrides auto-detection when set."
        ),
    )
    parser.add_argument(
        "--no-physics", action="store_true",
        help=(
            "Disable ovphysx simulation even when rigid bodies are present; "
            "render the authored USD transforms only."
        ),
    )
    parser.add_argument(
        "--articulation-drive-sine", action="store_true",
        help=(
            "TEMP: drive all ovphysx articulation DOF velocity targets with a "
            "sine wave before each physics step."
        ),
    )
    parser.add_argument(
        "--articulation-drive-sine-amplitude", type=float, default=20.0,
        help=(
            "TEMP: articulation velocity target sine amplitude "
            "(default: 20.0)."
        ),
    )
    parser.add_argument(
        "--articulation-drive-sine-period", type=float, default=4.0,
        help=(
            "TEMP: articulation velocity target sine period in seconds "
            "(default: 4.0)."
        ),
    )
    parser.add_argument(
        "--articulation-drive-roots", default="",
        help=(
            "TEMP: comma-separated articulation root prim paths to drive. "
            "Defaults to auto-detected PhysicsArticulationRootAPI prims."
        ),
    )
    parser.add_argument(
        "--overlap-sensor", action="store_true",
        help=(
            "TEMP: run a sphere overlap query around --overlap-sensor-prim "
            "after each physics step and log hit changes."
        ),
    )
    parser.add_argument(
        "--overlap-sensor-prim", default="/World/Sensor",
        help="TEMP: USD prim whose world position is used as the overlap center.",
    )
    parser.add_argument(
        "--overlap-sensor-radius", type=float, default=0.1,
        help="TEMP: sphere overlap radius in scene units (default: 0.1).",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without a display window (for CI/testing/batch rendering).",
    )
    parser.add_argument(
        "--png", action="store_true",
        help=(
            "Save rendered frames as PNG images to _output/ directory. "
            "Implies --headless.  Requires Pillow (pip install Pillow)."
        ),
    )
    args = parser.parse_args()

    # --png implies headless (no window needed for batch frame export)
    if args.png:
        args.headless = True

    usd_path = Path(args.usd_file).expanduser().resolve()
    if not usd_path.is_file():
        parser.error(
            f"USD stage not found: {args.usd_file}\n"
            "Pass the path to an existing .usd/.usda/.usdc stage; "
            "the bundled sample is usd/ov-fmi/fmi_parser_test.usda."
        )

    usd_file = str(usd_path)
    explicit_body_prims = [p.strip() for p in args.body_prims.split(",") if p.strip()]

    # ------------------------------------------------------------------
    # Phase 1: parse FMI schema (fmi_usd_helper or pxr subprocess)
    # ------------------------------------------------------------------
    print("Parsing FMI schema...")
    schema = _parse_fmi_schema(usd_file)

    raw_instances = schema.get("instances", {})
    initial_values = schema.get("initial_values", {})
    auto_body_prims = schema.get("body_prims", [])
    auto_articulation_roots = schema.get("articulation_roots", [])
    sensor_positions = schema.get("sensor_positions", {})
    overlap_sensors = schema.get("overlap_sensors", {})
    render_products = set(schema.get("render_products", []))
    stage_bounds = schema.get("stage_bounds", {"valid": False})
    if args.up_axis is not None:
        world_up, world_up_source = _world_up_from_axis(args.up_axis)
    else:
        world_up, world_up_source = _world_up_from_schema(schema)
    world_up_display = [0.0 if abs(float(v)) < 1e-12 else float(v) for v in world_up]
    print(
        "World up: "
        f"({world_up_display[0]:.3g}, {world_up_display[1]:.3g}, {world_up_display[2]:.3g}) "
        f"from {world_up_source}"
    )

    # Use explicit --body-prims if provided, otherwise auto-detected
    if explicit_body_prims:
        body_prims = explicit_body_prims
    else:
        body_prims = auto_body_prims
        if body_prims:
            print(f"Auto-detected {len(body_prims)} rigid body prim(s): {', '.join(body_prims)}")
    explicit_articulation_roots = [
        p.strip() for p in args.articulation_drive_roots.split(",") if p.strip()
    ]
    articulation_drive_roots = explicit_articulation_roots or auto_articulation_roots
    if args.articulation_drive_sine and articulation_drive_roots:
        print(
            "Temporary articulation sine drive roots: "
            f"{', '.join(articulation_drive_roots)}"
        )
    overlap_sensor_position = None
    if args.overlap_sensor:
        overlap_sensor_position = sensor_positions.get(args.overlap_sensor_prim)
        if overlap_sensor_position is None:
            print(
                "WARNING: temporary overlap sensor disabled; "
                f"could not find world position for {args.overlap_sensor_prim}.",
                file=sys.stderr,
            )
            args.overlap_sensor = False
        elif args.overlap_sensor_radius <= 0.0:
            print(
                "WARNING: temporary overlap sensor disabled; "
                "--overlap-sensor-radius must be > 0.",
                file=sys.stderr,
            )
            args.overlap_sensor = False
        else:
            print(
                "Temporary overlap sensor enabled: "
                f"prim={args.overlap_sensor_prim} "
                f"center=({overlap_sensor_position[0]:.4g}, "
                f"{overlap_sensor_position[1]:.4g}, "
                f"{overlap_sensor_position[2]:.4g}) "
                f"radius={args.overlap_sensor_radius:g}"
            )
    if args.no_physics and body_prims:
        print(
            "Physics disabled by --no-physics; rendering authored rigid-body "
            "transforms without ovphysx simulation."
        )

    if raw_instances:
        print(f"Found {len(raw_instances)} FMI/SSP instance prim(s):")
        for path in raw_instances:
            fmu = raw_instances[path].get("fmu")
            ssp = raw_instances[path].get("ssp")
            source = ssp or fmu
            kind = "ssp" if ssp else "fmu"
            n_conn = len(raw_instances[path]["connections"])
            print(f"  {path}  {kind}={source}  connections={n_conn}")
    elif body_prims:
        print(
            "WARNING: No FmuInstance or SspInstance prims found; "
            "running without FMI/SSP simulation."
        )
    else:
        print(
            "WARNING: No FmuInstance, SspInstance, or PhysX rigid bodies found; "
            "displaying USD stage only."
        )

    preview_overlay_usda = None
    effective_camera_prim = args.camera_prim
    if args.render_product not in render_products:
        effective_camera_prim, author_preview_camera = _select_preview_camera_prim(
            schema,
            args.camera_prim,
            args.render_product,
        )
        preview_overlay_usda = _make_preview_overlay_usda(
            usd_file,
            args.render_product,
            effective_camera_prim,
            stage_bounds,
            world_up,
            author_camera=author_preview_camera,
        )
        if author_preview_camera:
            print(
                f"WARNING: Render product {args.render_product} not found; "
                f"creating in-memory preview camera {effective_camera_prim}."
            )
        else:
            print(
                f"WARNING: Render product {args.render_product} not found; "
                f"creating in-memory render product using existing camera "
                f"{effective_camera_prim}."
            )

    # ------------------------------------------------------------------
    # Phase 1b: set up GPU display window (unless --headless)
    # ------------------------------------------------------------------
    _display = None
    if not args.headless:
        try:
            from gpu_display import GPUDisplay, is_cuda_interop_available  # noqa: PLC0415
            _display = GPUDisplay(width=1280, height=720, title="ov-fmi")
            if is_cuda_interop_available():
                print("GPU display: CUDA→OpenGL interop (zero-copy, GPU-resident)")
            else:
                print("GPU display: CPU upload path (install cuda-python for zero-copy)")
        except Exception as e:
            print(
                f"WARNING: Could not open display window: {e}\n"
                "         Falling back to headless mode.",
                file=sys.stderr,
            )
            _display = None

    # ------------------------------------------------------------------
    # Phase 2: import ovrtx
    # ------------------------------------------------------------------
    ovrtx_hint = os.environ.get("OVRTX_LIBRARY_PATH_HINT")
    if ovrtx_hint:
        # ovrtx 0.3 auto-registers USD schema paths during import. Skip that
        # import-time side effect when setup provided an explicit native binary
        # path, avoiding duplicate USD/plugin registration when ovphysx is used.
        os.environ.setdefault("OVRTX_SKIP_SCHEMA_AUTO_REGISTER", "1")

    import ovrtx  # noqa: PLC0415 — intentionally deferred

    if ovrtx_hint:
        ovrtx._src.bindings.OVRTX_LIBRARY_PATH_HINT = ovrtx_hint
    from ovrtx import Renderer  # noqa: PLC0415

    # ovphysx is optional.  It MUST be imported AFTER ovrtx — shared Carbonite
    # plugin conflicts cause failures if ovphysx loads first (see AGENTS.md).
    _physics_enabled = bool(body_prims) and not args.no_physics
    PhysX = None
    _TENSOR_POSE = None
    _TENSOR_FORCE = None
    _TENSOR_VELOCITY = None
    _TENSOR_ARTICULATION_DOF_VELOCITY_TARGET = None
    _SCENE_QUERY_MODE_ALL = None
    _SCENE_QUERY_GEOMETRY_SPHERE = None
    if _physics_enabled:
        try:
            from ovphysx import PhysX as _PhysX  # noqa: PLC0415
            PhysX = _PhysX
            try:
                from ovphysx import (  # noqa: PLC0415
                    SceneQueryGeometryType as _SceneQueryGeometryType,
                    SceneQueryMode as _SceneQueryMode,
                )
                _SCENE_QUERY_MODE_ALL = _SceneQueryMode.ALL
                _SCENE_QUERY_GEOMETRY_SPHERE = _SceneQueryGeometryType.SPHERE
            except ImportError:
                _SCENE_QUERY_MODE_ALL = None
                _SCENE_QUERY_GEOMETRY_SPHERE = None
            try:
                from ovphysx import TensorType as _TensorType  # noqa: PLC0415
                _TENSOR_ARTICULATION_DOF_VELOCITY_TARGET = (
                    _TensorType.ARTICULATION_DOF_VELOCITY_TARGET
                )
            except ImportError:
                try:
                    from ovphysx import (  # noqa: PLC0415
                        OVPHYSX_TENSOR_ARTICULATION_DOF_VELOCITY_TARGET_F32,
                    )
                    _TENSOR_ARTICULATION_DOF_VELOCITY_TARGET = (
                        OVPHYSX_TENSOR_ARTICULATION_DOF_VELOCITY_TARGET_F32
                    )
                except ImportError:
                    _TENSOR_ARTICULATION_DOF_VELOCITY_TARGET = None
            try:
                from ovphysx import (  # noqa: PLC0415
                    OVPHYSX_TENSOR_RIGID_BODY_POSE_F32,
                    OVPHYSX_TENSOR_RIGID_BODY_FORCE_F32,
                    OVPHYSX_TENSOR_RIGID_BODY_VELOCITY_F32,
                )
                _TENSOR_POSE = OVPHYSX_TENSOR_RIGID_BODY_POSE_F32
                _TENSOR_FORCE = OVPHYSX_TENSOR_RIGID_BODY_FORCE_F32
                _TENSOR_VELOCITY = OVPHYSX_TENSOR_RIGID_BODY_VELOCITY_F32
            except ImportError:
                from ovphysx import TensorType  # noqa: PLC0415
                _TENSOR_POSE = TensorType.RIGID_BODY_POSE
                _TENSOR_FORCE = TensorType.RIGID_BODY_FORCE
                _TENSOR_VELOCITY = TensorType.RIGID_BODY_VELOCITY
        except ImportError:
            print(
                "WARNING: ovphysx not found; physics disabled.\n"
                "         Install with: pip install ovphysx==0.4.9 "
                "--extra-index-url https://pypi.nvidia.com",
                file=sys.stderr,
            )
            _physics_enabled = False

    instances = _deserialise_instances(raw_instances) if raw_instances else {}

    # ------------------------------------------------------------------
    # Phase 3: initialise ovrtx renderer  *** MUST come before ovphysx ***
    # ------------------------------------------------------------------
    print("Initialising renderer...")
    renderer = Renderer()
    if preview_overlay_usda is not None:
        renderer.open_usd_from_string(preview_overlay_usda)
    else:
        renderer.open_usd(usd_file)
    if _display:
        print("Renderer ready (live display window active).")
    else:
        print("Renderer ready (headless mode — use --png to save frames).")

    camera_binding = None
    camera_controller = None
    if _display is not None:
        try:
            from ovrtx import Semantic as _Sem, PrimMode as _PM  # noqa: PLC0415
            initial_camera_matrix = _read_camera_matrix(renderer, effective_camera_prim)
            camera_binding = renderer.bind_attribute(
                [effective_camera_prim],
                "omni:xform",
                semantic=_Sem.XFORM_MAT4x4,
                prim_mode=_PM.EXISTING_ONLY,
            )
            camera_controller = FirstPersonCameraController(
                _display,
                camera_binding,
                initial_camera_matrix,
                move_speed=args.nav_speed,
                mouse_sensitivity_degrees=args.mouse_sensitivity,
                world_up=world_up,
            )
            print(
                "Navigation: WASD move, Q/E up/down, Shift fast, Ctrl slow, "
                f"hold right mouse for mouselook ({effective_camera_prim})."
            )
        except Exception as e:
            print(
                "WARNING: Could not enable live camera navigation.\n"
                f"         {e}",
                file=sys.stderr,
            )
            if camera_binding is not None:
                try:
                    camera_binding.unbind()
                except Exception:
                    pass
            camera_binding = None
            camera_controller = None

    # ------------------------------------------------------------------
    # Phase 3b: optionally initialise ovphysx  *** AFTER renderer ***
    # ------------------------------------------------------------------
    physx = None
    physx_usd_handle = None
    pose_binding = None
    vel_binding = None
    force_binding = None
    pose_data = None
    vel_data = None
    force_data = None
    num_bodies = 0
    body_transform_data = None
    body_parent_world_inverse_data = None
    body_local_scale_data = None
    ovrtx_body_binding = None
    physx_input_head = None
    body_prim_index = None
    body_tensor_indices = None
    physx_init_failed = False
    articulation_drive_router = None
    articulation_drive_binding = None
    articulation_drive_data = None
    overlap_sensor_last_signature = None
    drive_target_router_warning_printed = False

    if _physics_enabled:
        try:
            print("Initialising ovphysx physics...")
            physx = PhysX(device="cpu")

            physx_usd_handle, _ = physx.add_usd(usd_file)
            physx.wait_all()

            # Rigid-body pose binding [N, 7]: px py pz qx qy qz qw
            pose_binding = physx.create_tensor_binding(
                prim_paths=body_prims,
                tensor_type=_TENSOR_POSE,
            )
            num_bodies = pose_binding.shape[0]
            pose_data = np.zeros(pose_binding.shape, dtype=np.float32)

            # Rigid-body force binding [N, 3]: fx fy fz
            force_binding = physx.create_tensor_binding(
                prim_paths=body_prims,
                tensor_type=_TENSOR_FORCE,
            )
            force_data = np.zeros(force_binding.shape, dtype=np.float32)

            # Rigid-body velocity binding [N, 6]: vx vy vz wx wy wz
            vel_binding = physx.create_tensor_binding(
                prim_paths=body_prims,
                tensor_type=_TENSOR_VELOCITY,
            )
            vel_data = np.zeros(vel_binding.shape, dtype=np.float32)

            tensor_prim_paths = getattr(pose_binding, "prim_paths", None)
            body_prims, body_tensor_indices, skipped_body_prims = _select_tracked_body_prims(
                body_prims,
                tensor_prim_paths,
                num_bodies,
            )
            if skipped_body_prims:
                preview = ", ".join(skipped_body_prims[:5])
                suffix = "" if len(skipped_body_prims) <= 5 else ", ..."
                print(
                    "WARNING: Skipping "
                    f"{len(skipped_body_prims)} detected rigid body prim(s) not present "
                    f"in ovphysx tensors: {preview}{suffix}",
                    file=sys.stderr,
                )
            if not body_prims:
                raise RuntimeError("no detected rigid body prims matched ovphysx tensors")

            # Build prim -> tensor-index map for PhysxInputHead and force routing.
            body_prim_index = dict(zip(body_prims, body_tensor_indices))

            # ovrtx transform binding for the tracked body prims
            n_tracked = len(body_prims)
            body_transform_data = np.zeros((n_tracked, 16), dtype=np.float64)
            for i in range(n_tracked):
                body_transform_data[i, 0] = body_transform_data[i, 5] = 1.0
                body_transform_data[i, 10] = body_transform_data[i, 15] = 1.0
            body_parent_world_inverse_data = _body_parent_world_inverse_matrices(
                renderer,
                body_prims,
            )
            body_local_scale_data = _body_local_scale_matrices(renderer, body_prims)

            from ovrtx import Semantic as _Sem, PrimMode as _PM  # noqa: PLC0415
            ovrtx_body_binding = renderer.bind_attribute(
                body_prims,
                "omni:xform",
                semantic=_Sem.XFORM_MAT4x4,
                prim_mode=_PM.EXISTING_ONLY,
            )
            print(f"Physics: {num_bodies} rigid bodies, {n_tracked} tracked in ovrtx")

            if args.articulation_drive_sine:
                try:
                    if _TENSOR_ARTICULATION_DOF_VELOCITY_TARGET is None:
                        raise RuntimeError(
                            "ovphysx does not expose ARTICULATION_DOF_VELOCITY_TARGET"
                        )
                    if args.articulation_drive_sine_period <= 0.0:
                        raise RuntimeError("--articulation-drive-sine-period must be > 0")
                    if articulation_drive_roots:
                        articulation_drive_binding = physx.create_tensor_binding(
                            prim_paths=articulation_drive_roots,
                            tensor_type=_TENSOR_ARTICULATION_DOF_VELOCITY_TARGET,
                            raise_if_empty=True,
                        )
                    else:
                        articulation_drive_binding = physx.create_tensor_binding(
                            pattern="/World/**",
                            tensor_type=_TENSOR_ARTICULATION_DOF_VELOCITY_TARGET,
                            raise_if_empty=True,
                        )
                    if (
                        len(articulation_drive_binding.shape) < 2
                        or articulation_drive_binding.shape[1] == 0
                    ):
                        raise RuntimeError(
                            "articulation velocity target binding has no DOFs "
                            f"(shape={articulation_drive_binding.shape})"
                        )
                    articulation_drive_data = np.zeros(
                        articulation_drive_binding.shape,
                        dtype=np.float32,
                    )
                    dof_names = getattr(articulation_drive_binding, "dof_names", [])
                    root_paths = getattr(articulation_drive_binding, "prim_paths", [])
                    print(
                        "Temporary articulation sine drive enabled: "
                        f"shape={articulation_drive_binding.shape} "
                        f"amplitude={args.articulation_drive_sine_amplitude:g} "
                        f"period={args.articulation_drive_sine_period:g}s "
                        f"roots={root_paths} dofs={dof_names}"
                    )
                except Exception as drive_exc:
                    if articulation_drive_binding is not None:
                        try:
                            articulation_drive_binding.destroy()
                        except Exception:
                            pass
                    articulation_drive_binding = None
                    articulation_drive_data = None
                    print(
                        "WARNING: temporary articulation sine drive disabled.\n"
                        f"         {drive_exc}",
                        file=sys.stderr,
                    )

            if args.overlap_sensor and (
                _SCENE_QUERY_MODE_ALL is None
                or _SCENE_QUERY_GEOMETRY_SPHERE is None
            ):
                print(
                    "WARNING: temporary overlap sensor disabled; "
                    "ovphysx scene query enums are unavailable.",
                    file=sys.stderr,
                )
                args.overlap_sensor = False
        except Exception as e:
            print(
                "WARNING: ovphysx physics initialisation failed; continuing without physics.\n"
                f"         {e}",
                file=sys.stderr,
            )
            for binding in (force_binding, vel_binding, pose_binding):
                if binding is not None:
                    try:
                        binding.destroy()
                    except Exception:
                        pass
            if articulation_drive_binding is not None:
                try:
                    articulation_drive_binding.destroy()
                except Exception:
                    pass
            if articulation_drive_router is not None:
                try:
                    articulation_drive_router.destroy()
                except Exception:
                    pass
            physx_init_failed = True
            physx = None
            physx_usd_handle = None
            pose_binding = vel_binding = force_binding = None
            articulation_drive_router = None
            articulation_drive_binding = None
            pose_data = vel_data = force_data = None
            articulation_drive_data = None
            body_transform_data = None
            body_parent_world_inverse_data = None
            body_local_scale_data = None
            ovrtx_body_binding = None
            physx_input_head = None
            body_prim_index = None
            body_tensor_indices = None
            _physics_enabled = False

    # ------------------------------------------------------------------
    # Phase 4: wire up FMI adapters and runtime, if the stage has FMI/SSP
    # ------------------------------------------------------------------
    input_head = OvrtxInputHead(initial_values)
    output_tail = OvrtxOutputTail(renderer, input_head)
    fmi_rt = None
    if instances:
        from fmi_runtime import FMIRuntime  # noqa: PLC0415
        fmi_rt = FMIRuntime(input_head, output_tail)
        fmi_rt.init(instances)
        fmi_rt.resume()
        output_tail.finalize_transform_bindings()

    # Set up physics input head if physics is enabled
    if fmi_rt is not None and _physics_enabled and body_prim_index:
        physx_input_head = PhysxInputHead(body_prim_index, overlap_sensors)
        # Cache physx input connections for all FMU instances
        for runtime_fmu in fmi_rt._runtime_fmus.values():
            for inst in runtime_fmu.get_runtime_instances():
                physx_input_head.cache_connections(inst)

    if fmi_rt is not None and physx is not None and output_tail.has_drive_target_outputs():
        try:
            articulation_drive_router = ArticulationDriveTargetRouter(
                physx,
                _TENSOR_ARTICULATION_DOF_VELOCITY_TARGET,
                articulation_drive_roots,
            )
            print(
                "Articulation drive target routing enabled: "
                f"shape={articulation_drive_router.shape} "
                f"roots={articulation_drive_router.root_paths} "
                f"dofs={articulation_drive_router.dof_names}"
            )
        except Exception as drive_route_exc:
            articulation_drive_router = None
            print(
                "WARNING: articulation drive target routing disabled.\n"
                f"         {drive_route_exc}",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # Phase 4b: set up PNG output if requested
    # ------------------------------------------------------------------
    _save_png = False
    _png_output_dir = None
    if args.png:
        try:
            from PIL import Image as _PILImage  # noqa: PLC0415
            _save_png = True
            _png_output_dir = Path("_output")
            _png_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"PNG output enabled: saving frames to {_png_output_dir}/")
        except ImportError:
            print(
                "WARNING: --png requires Pillow. Install with: pip install Pillow",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # Phase 5: simulation loop
    # ------------------------------------------------------------------
    sim_time = 0.0
    frame = 0
    wall_start = time.monotonic()
    nav_last_wall = wall_start
    run_mode = "simulation" if fmi_rt is not None or physx is not None else "USD preview"
    duration_label = "infinity" if math.isinf(args.duration) else f"{args.duration}s"
    print(
        f"Running {run_mode}: dt={args.dt:.4f}s  duration={duration_label}  "
        f"fmi={'on' if fmi_rt is not None else 'off'}  "
        f"physics={'on' if physx else 'off'}"
    )

    try:
        while sim_time < args.duration:
            if _display is not None:
                _display.poll_events()
                if _display.should_close():
                    print("\nDisplay window closed by user.")
                    break

            if camera_controller is not None:
                now = time.monotonic()
                camera_controller.update(min(now - nav_last_wall, 0.1))
                nav_last_wall = now

            # --- (Optional) read physics state before FMU step ---
            if physx is not None:
                pose_binding.read(pose_data)
                vel_binding.read(vel_data)

                # Feed physics state into PhysxInputHead cache
                if physx_input_head is not None:
                    physx_input_head.update_from_tensors(pose_data, vel_data)
                    physx_input_head.update_overlap_sensors(
                        physx,
                        _SCENE_QUERY_GEOMETRY_SPHERE,
                        _SCENE_QUERY_MODE_ALL,
                    )

            # --- Step all FMUs ---
            # If we have physics inputs, inject them into each FMU step
            if physx_input_head is not None and fmi_rt is not None:
                for runtime_fmu in fmi_rt._runtime_fmus.values():
                    for inst in runtime_fmu.get_runtime_instances():
                        ovrtx_inputs = input_head.get_inputs_for(inst)
                        physx_inputs = physx_input_head.get_inputs_for(inst)
                        merged_inputs = {**ovrtx_inputs, **physx_inputs}
                        outputs = output_tail.get_outputs_for(inst)
                        result = inst.step(
                            filename=runtime_fmu.get_unzip_dir(),
                            inputs=merged_inputs,
                            outputs=outputs,
                            time=sim_time,
                        )
                        if result is not None:
                            output_tail.write_outputs(inst, outputs, result)
            elif fmi_rt is not None:
                fmi_rt.step(sim_time)

            # --- Write FMU force outputs to physics, then step ---
            if physx is not None:
                if articulation_drive_binding is not None:
                    target_velocity = (
                        args.articulation_drive_sine_amplitude
                        * math.sin(
                            2.0 * math.pi * sim_time
                            / args.articulation_drive_sine_period
                        )
                    )
                    articulation_drive_data.fill(target_velocity)
                    articulation_drive_binding.write(articulation_drive_data)

                pending_drive_targets = output_tail.get_pending_drive_targets()
                if pending_drive_targets:
                    if articulation_drive_router is not None:
                        articulation_drive_router.write_targets(pending_drive_targets)
                    elif not drive_target_router_warning_printed:
                        print(
                            "WARNING: FMU drive target outputs are present, but "
                            "articulation drive target routing is unavailable.",
                            file=sys.stderr,
                        )
                        drive_target_router_warning_printed = True
                    output_tail.clear_pending_drive_targets()

                # Collect force outputs and write to force tensor
                pending = output_tail.get_pending_forces()
                if pending and body_prim_index:
                    force_data[:] = 0.0  # reset forces each frame
                    for prim, forces in pending.items():
                        idx = body_prim_index.get(prim)
                        if idx is not None and idx < force_data.shape[0]:
                            force_data[idx, 0] = forces[0]
                            force_data[idx, 1] = forces[1]
                            force_data[idx, 2] = forces[2]
                    force_binding.write(force_data)
                    output_tail.clear_pending_forces()

                step_op = physx.step(args.dt, sim_time)
                physx.wait_op(step_op)

                if args.overlap_sensor:
                    try:
                        hits = physx.overlap(
                            _SCENE_QUERY_GEOMETRY_SPHERE,
                            mode=_SCENE_QUERY_MODE_ALL,
                            radius=args.overlap_sensor_radius,
                            position=overlap_sensor_position,
                        )
                        signature = tuple(
                            sorted(
                                (
                                    int(hit.get("collision", 0)),
                                    int(hit.get("rigid_body", 0)),
                                )
                                for hit in hits
                            )
                        )
                        if signature != overlap_sensor_last_signature:
                            if hits:
                                preview = ", ".join(
                                    f"collision=0x{int(hit.get('collision', 0)):x} "
                                    f"body=0x{int(hit.get('rigid_body', 0)):x}"
                                    for hit in hits[:5]
                                )
                                suffix = "" if len(hits) <= 5 else ", ..."
                                print(
                                    "Overlap sensor hit: "
                                    f"t={sim_time:.3f}s count={len(hits)} "
                                    f"{preview}{suffix}"
                                )
                            elif overlap_sensor_last_signature:
                                print(
                                    "Overlap sensor clear: "
                                    f"t={sim_time:.3f}s"
                                )
                            overlap_sensor_last_signature = signature
                    except Exception as overlap_exc:
                        print(
                            "WARNING: temporary overlap sensor query failed; "
                            "disabling it.\n"
                            f"         {overlap_exc}",
                            file=sys.stderr,
                        )
                        args.overlap_sensor = False

                # Read updated poses
                pose_binding.read(pose_data)

                # Pack matched pose tensor rows into column-major 4x4 float64
                # matrices and push them to ovrtx in binding order.
                n_tracked = len(body_prims)
                for out_i, tensor_i in enumerate(body_tensor_indices):
                    quaternion_to_4x4_column_major(
                        float(pose_data[tensor_i, 0]), float(pose_data[tensor_i, 1]),
                        float(pose_data[tensor_i, 2]),
                        float(pose_data[tensor_i, 3]), float(pose_data[tensor_i, 4]),
                        float(pose_data[tensor_i, 5]), float(pose_data[tensor_i, 6]),
                        body_transform_data[out_i],
                    )
                    local_matrix = _local_matrix_from_world(
                        body_transform_data[out_i],
                        body_parent_world_inverse_data[out_i],
                    )
                    local_matrix = body_local_scale_data[out_i] @ local_matrix
                    body_transform_data[out_i, :] = local_matrix.reshape(16)
                ovrtx_body_binding.write(
                    body_transform_data.reshape(n_tracked, 4, 4)
                )

            products = renderer.step({args.render_product}, args.dt)

            # --- Present frame to display window or save PNG ---
            if products is not None and hasattr(products, 'items'):
                for _product_name, product in products.items():
                    for prod_frame in product.frames:
                        if "LdrColor" not in prod_frame.render_vars:
                            continue
                        with prod_frame.render_vars["LdrColor"].map() as mapping:
                            np_array = np.from_dlpack(mapping)

                            # Live display window
                            if _display is not None:
                                _display.present_frame_cpu(np_array)

                            # PNG save (headless batch mode)
                            if _save_png:
                                from PIL import Image as _PILImage  # noqa: PLC0415
                                _PILImage.fromarray(np_array).save(
                                    str(_png_output_dir / f"frame_{frame:04d}.png")
                                )

            sim_time += args.dt
            frame += 1
    finally:
        if fmi_rt is not None:
            fmi_rt.destroy()
        if _display is not None:
            _display.destroy()
            _display = None

    elapsed = time.monotonic() - wall_start
    print(f"Done: {frame} frames in {elapsed:.1f}s  ({frame / elapsed:.1f} fps)")

    if physx is not None or physx_init_failed:
        # ovrtx + ovphysx can report failure during native DLL/SO unload
        # after the simulation has completed (Windows DLL teardown crash,
        # Linux SIGSEGV in omni_physx_sdk cleanup).  Exit directly so CLI
        # scripts see the successful run instead of the native teardown status.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    # ------------------------------------------------------------------
    # Phase 6: cleanup
    # ------------------------------------------------------------------
    # Physics bindings must be released before physx itself
    if physx is not None:
        if force_binding is not None:
            force_binding.destroy()
        if vel_binding is not None:
            vel_binding.destroy()
        if pose_binding is not None:
            pose_binding.destroy()
        if articulation_drive_router is not None:
            articulation_drive_router.destroy()
        if articulation_drive_binding is not None:
            articulation_drive_binding.destroy()
        if physx_usd_handle is not None:
            remove_op = physx.remove_usd(physx_usd_handle)
            physx.wait_op(remove_op)
        physx.release()

    if ovrtx_body_binding is not None:
        ovrtx_body_binding.unbind()

    if camera_binding is not None:
        camera_binding.unbind()

    # ovrtx may SIGSEGV here when co-loaded with ovphysx — known non-fatal issue.
    # Simulation has already completed; ignore cleanup errors.
    try:
        del renderer
    except Exception:
        pass


if __name__ == "__main__":
    main()
