"""DRAWER-inspired category mesh fitting, UV/PBR bundles, and mesh-bound splats."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .core import AXES, Bounds, RealToSimError, Workspace, content_digest, sha256_file
from .gaussian import load_gaussians, write_mesh_bound_gaussians
from .sim import sweep_joint


MESH_COMPLETION_SCHEMA = 1
MESH_COMPLETION_GENERATOR = "nanousd-rts mesh-bound PBR completion v1"
LOCAL_MATERIAL_PROVIDER = "measured-front-palette-pbr-v1"
EXTERNAL_MATERIAL_PROVIDER = "external-pbr-atlas-v1"
PBR_MAPS = ("baseColor.png", "roughness.png", "metallic.png", "normal.png", "ao.png")
MESH_GAUSSIAN_RENDER_SAFE_LIMIT = 4095

LEARNED_MATERIAL_PROMPTS = {
    "oven-door": (
        "charcoal gray heat-resistant porcelain enamel for an oven interior, "
        "fine orange-peel texture, subtle baked-on wear, seamless photorealistic PBR material"
    ),
    "refrigerator-door": (
        "clean warm-white molded refrigerator interior plastic, very fine satin texture, "
        "subtle manufacturing variation, seamless photorealistic PBR material"
    ),
    "cabinet-door": (
        "warm off-white painted maple cabinet interior, fine wood grain beneath satin paint, "
        "subtle realistic wear, seamless photorealistic PBR material"
    ),
    "drawer": (
        "light maple wood drawer interior, fine straight grain, matte clear finish, "
        "subtle realistic variation, seamless photorealistic PBR material"
    ),
}


@dataclass(slots=True)
class MeshSurface:
    vertices: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    face_uvs: np.ndarray
    face_colors: np.ndarray
    face_weights: np.ndarray
    face_opacities: np.ndarray
    face_patch_indices: np.ndarray
    atlas_tiles: list[dict[str, Any]]


def _replace_directory(destination: Path, staged: Path) -> None:
    backup = destination.with_name(f".{destination.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.rename(backup)
    staged.rename(destination)
    if backup.exists():
        shutil.rmtree(backup)


def _accepted_completion(workspace: Workspace, node_id: str) -> dict[str, Any]:
    matches = [
        item
        for item in workspace.completions
        if item.get("node") == node_id and item.get("status") == "accepted"
    ]
    if len(matches) != 1:
        raise RealToSimError(
            f"mesh fitting requires exactly one accepted completion for {node_id}, got {len(matches)}"
        )
    completion = dict(matches[0])
    template = completion.get("template")
    if not isinstance(template, dict) or not template.get("static_patches"):
        raise RealToSimError(
            "accepted completion predates the mesh template contract; re-author the "
            "visual completion before running fit-mesh-pbr"
        )
    return completion


def _typed_patches(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list) or not values:
        raise RealToSimError("mesh template requires at least one surface patch")
    patches: list[dict[str, Any]] = []
    for value in values:
        try:
            patch = {
                "bounds": Bounds.from_json(value["bounds"]),
                "axis": int(value["axis"]),
                "side": int(value["side"]),
                "color": tuple(float(item) for item in value["color"]),
                "weight": float(value.get("weight", 1.0)),
                "opacity": float(value.get("opacity", 0.97)),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise RealToSimError("mesh template contains an invalid surface patch") from exc
        if patch["axis"] not in (0, 1, 2) or patch["side"] not in (-1, 1):
            raise RealToSimError("mesh template patch axis/side is invalid")
        patches.append(patch)
    return patches


def _apply_fit_transform(
    patches: list[dict[str, Any]],
    fit_diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    transform = fit_diagnostics["transform"]
    source_center = np.asarray(transform["source_center"], dtype=np.float64)
    target_center = np.asarray(transform["target_center"], dtype=np.float64)
    scale = np.asarray(transform["scale"], dtype=np.float64)
    fitted: list[dict[str, Any]] = []
    for patch in patches:
        bounds = patch["bounds"]
        minimum = target_center + (np.asarray(bounds.minimum) - source_center) * scale
        maximum = target_center + (np.asarray(bounds.maximum) - source_center) * scale
        fitted.append(
            {
                **patch,
                "bounds": Bounds(tuple(minimum), tuple(maximum)),
            }
        )
    return fitted


def _mesh_from_patches(
    patches: list[dict[str, Any]],
    *,
    texture_size: int,
) -> MeshSurface:
    if texture_size < 128 or texture_size > 4096:
        raise RealToSimError("PBR texture size must be within [128, 4096]")
    grid = int(math.ceil(math.sqrt(len(patches))))
    vertices: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    face_colors: list[tuple[float, float, float]] = []
    face_weights: list[float] = []
    face_opacities: list[float] = []
    face_patch_indices: list[int] = []
    atlas_tiles: list[dict[str, Any]] = []
    padding_uv = 2.0 / texture_size

    for patch_index, patch in enumerate(patches):
        bounds = patch["bounds"]
        axis = patch["axis"]
        side = patch["side"]
        tangent = [component for component in range(3) if component != axis]
        coordinate = bounds.minimum[axis] if side < 0 else bounds.maximum[axis]
        corners = []
        for first, second in ((0, 0), (1, 0), (1, 1), (0, 1)):
            point = np.zeros(3, dtype=np.float64)
            point[axis] = coordinate
            point[tangent[0]] = (
                bounds.minimum[tangent[0]] if first == 0 else bounds.maximum[tangent[0]]
            )
            point[tangent[1]] = (
                bounds.minimum[tangent[1]] if second == 0 else bounds.maximum[tangent[1]]
            )
            corners.append(point)

        row, column = divmod(patch_index, grid)
        u0 = column / grid + padding_uv
        u1 = (column + 1) / grid - padding_uv
        v0 = 1.0 - (row + 1) / grid + padding_uv
        v1 = 1.0 - row / grid - padding_uv
        uv_corners = [
            np.asarray((u0, v0), dtype=np.float64),
            np.asarray((u1, v0), dtype=np.float64),
            np.asarray((u1, v1), dtype=np.float64),
            np.asarray((u0, v1), dtype=np.float64),
        ]
        expected_normal = np.zeros(3, dtype=np.float64)
        expected_normal[axis] = side
        if np.dot(
            np.cross(corners[1] - corners[0], corners[2] - corners[0]),
            expected_normal,
        ) < 0.0:
            order = (0, 3, 2, 1)
            corners = [corners[index] for index in order]
            uv_corners = [uv_corners[index] for index in order]

        offset = len(vertices)
        vertices.extend(corners)
        uvs.extend(uv_corners)
        faces.extend(((offset, offset + 1, offset + 2), (offset, offset + 2, offset + 3)))
        face_colors.extend((patch["color"], patch["color"]))
        face_weights.extend((patch["weight"], patch["weight"]))
        face_opacities.extend((patch["opacity"], patch["opacity"]))
        face_patch_indices.extend((patch_index, patch_index))
        x0 = int(math.floor(column * texture_size / grid))
        x1 = int(math.ceil((column + 1) * texture_size / grid))
        y0 = int(math.floor(row * texture_size / grid))
        y1 = int(math.ceil((row + 1) * texture_size / grid))
        atlas_tiles.append(
            {
                "patch": patch_index,
                "pixel_bounds": [x0, y0, x1, y1],
                "uv_bounds": [u0, v0, u1, v1],
                "source_color": [float(item) for item in patch["color"]],
            }
        )

    vertex_values = np.asarray(vertices, dtype=np.float64)
    face_values = np.asarray(faces, dtype=np.uint32)
    uv_values = np.asarray(uvs, dtype=np.float64)
    return MeshSurface(
        vertices=vertex_values,
        faces=face_values,
        uvs=uv_values,
        face_uvs=uv_values[face_values],
        face_colors=np.asarray(face_colors, dtype=np.float64),
        face_weights=np.asarray(face_weights, dtype=np.float64),
        face_opacities=np.asarray(face_opacities, dtype=np.float64),
        face_patch_indices=np.asarray(face_patch_indices, dtype=np.uint32),
        atlas_tiles=atlas_tiles,
    )


def _measured_palette(workspace: Workspace, node_id: str) -> dict[str, Any]:
    scene = load_gaussians(workspace.source_path)
    indices = workspace.load_selection(node_id).astype(np.uint32)
    if not len(indices) or int(indices.max()) >= scene.count:
        raise RealToSimError(f"measured selection is empty or invalid for {node_id}")
    c0 = 0.28209479177387814
    rgb = np.clip(scene.sh_coefficients[indices, 0, :] * c0 + 0.5, 0.0, 1.0)
    return {
        "median": np.quantile(rgb, 0.50, axis=0).tolist(),
        "dark": np.quantile(rgb, 0.20, axis=0).tolist(),
        "light": np.quantile(rgb, 0.80, axis=0).tolist(),
        "sample_count": int(len(indices)),
        "selection_sha256": sha256_file(workspace.root / workspace.node(node_id).selection_file),
    }


def _material_parameters(kind: str, attachment: str) -> tuple[float, float]:
    if kind == "oven-door":
        return (0.34 if attachment == "world" else 0.28, 0.72)
    if kind == "refrigerator-door":
        return (0.30, 0.48)
    if kind == "drawer":
        return (0.46, 0.04)
    return (0.50, 0.02)


def _local_material_maps(
    destination: Path,
    mesh: MeshSurface,
    *,
    kind: str,
    attachment: str,
    measured_palette: dict[str, Any],
    texture_size: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    base_color = np.zeros((texture_size, texture_size, 3), dtype=np.float64)
    roughness = np.full((texture_size, texture_size), 0.5, dtype=np.float64)
    metallic = np.zeros((texture_size, texture_size), dtype=np.float64)
    normal = np.zeros((texture_size, texture_size, 3), dtype=np.float64)
    normal[..., 2] = 1.0
    ambient_occlusion = np.ones((texture_size, texture_size), dtype=np.float64)
    observed = np.asarray(measured_palette["median"], dtype=np.float64)
    blend = {
        "oven-door": 0.16,
        "refrigerator-door": 0.30,
        "drawer": 0.42,
        "cabinet-door": 0.46,
    }[kind]
    base_roughness, base_metallic = _material_parameters(kind, attachment)

    for tile in mesh.atlas_tiles:
        x0, y0, x1, y1 = tile["pixel_bounds"]
        height = max(y1 - y0, 1)
        width = max(x1 - x0, 1)
        source = np.asarray(tile["source_color"], dtype=np.float64)
        color = np.clip(source * (1.0 - blend) + observed * blend, 0.015, 0.985)
        yy, xx = np.mgrid[0:height, 0:width]
        unit_x = xx / max(width - 1, 1)
        unit_y = yy / max(height - 1, 1)
        phase = rng.uniform(0.0, math.tau)
        if kind in {"cabinet-door", "drawer"}:
            pattern = (
                0.028 * np.sin(math.tau * (unit_y * 5.0 + 0.18 * np.sin(unit_x * 7.0) + phase))
                + 0.010 * np.sin(math.tau * (unit_y * 17.0 + phase * 0.5))
            )
        else:
            pattern = 0.018 * np.sin(math.tau * (unit_y * 34.0 + phase))
        pattern += rng.normal(0.0, 0.004, size=(height, width))
        tile_color = np.clip(color[None, None, :] * (1.0 + pattern[..., None]), 0.0, 1.0)
        base_color[y0:y1, x0:x1] = tile_color
        roughness[y0:y1, x0:x1] = np.clip(
            base_roughness - pattern * 0.45,
            0.08,
            0.92,
        )
        metallic[y0:y1, x0:x1] = base_metallic

        gradient_y, gradient_x = np.gradient(pattern)
        tile_normal = np.stack((-gradient_x * 2.2, -gradient_y * 2.2, np.ones_like(pattern)), axis=2)
        tile_normal /= np.linalg.norm(tile_normal, axis=2, keepdims=True)
        normal[y0:y1, x0:x1] = tile_normal
        edge = np.minimum.reduce((xx + 1, yy + 1, width - xx, height - yy))
        ambient_occlusion[y0:y1, x0:x1] = np.clip(0.80 + edge / 18.0, 0.80, 1.0)

    destination.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(base_color * 255.0).astype(np.uint8), mode="RGB").save(
        destination / "baseColor.png"
    )
    Image.fromarray(np.rint(roughness * 255.0).astype(np.uint8), mode="L").save(
        destination / "roughness.png"
    )
    Image.fromarray(np.rint(metallic * 255.0).astype(np.uint8), mode="L").save(
        destination / "metallic.png"
    )
    encoded_normal = np.clip(normal * 0.5 + 0.5, 0.0, 1.0)
    Image.fromarray(np.rint(encoded_normal * 255.0).astype(np.uint8), mode="RGB").save(
        destination / "normal.png"
    )
    Image.fromarray(np.rint(ambient_occlusion * 255.0).astype(np.uint8), mode="L").save(
        destination / "ao.png"
    )
    return {
        "provider": LOCAL_MATERIAL_PROVIDER,
        "texture_size": texture_size,
        "base_color": base_color,
        "provenance": {
            "measured_input": "robust color palette from immutable selected front Gaussians",
            "generated_output": "category-conditioned deterministic PBR atlas",
            "learned": False,
        },
    }


def _external_material_maps(
    bundle: Path,
    destination: Path,
    *,
    role_slug: str,
) -> dict[str, Any]:
    root = Path(bundle).resolve()
    source = root / role_slug if (root / role_slug).is_dir() else root
    required = ("baseColor.png", "roughness.png", "normal.png")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise RealToSimError(
            f"external material bundle for {role_slug} is missing maps: {missing}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    dimensions: set[tuple[int, int]] = set()
    for name in required:
        with Image.open(source / name) as image:
            dimensions.add(image.size)
        shutil.copy2(source / name, destination / name)
    if len(dimensions) != 1:
        raise RealToSimError("external PBR maps must use one shared resolution")
    width, height = next(iter(dimensions))
    if width != height or not 128 <= width <= 4096:
        raise RealToSimError("external PBR maps must be square within [128, 4096]")
    for name, value in (("metallic.png", 0), ("ao.png", 255)):
        source_map = source / name
        if source_map.is_file():
            with Image.open(source_map) as image:
                if image.size != (width, height):
                    raise RealToSimError(
                        "all external PBR maps must share one resolution"
                    )
            shutil.copy2(source_map, destination / name)
        else:
            Image.new("L", (width, height), color=value).save(destination / name)
    with Image.open(destination / "baseColor.png") as image:
        base_color = np.asarray(image.convert("RGB")).copy()
    learned_provenance: dict[str, Any] | None = None
    learned_manifest_path = root / "manifest.json"
    if learned_manifest_path.is_file():
        try:
            learned_manifest = json.loads(learned_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RealToSimError("external learned-material manifest is invalid JSON") from exc
        if learned_manifest.get("contract") == "nanousd-rts-learned-pbr-bundle-v1":
            role_manifest = next(
                (
                    item
                    for item in learned_manifest.get("roles", [])
                    if item.get("role", "").replace("_", "-") == role_slug
                ),
                None,
            )
            if role_manifest is None:
                raise RealToSimError(
                    f"learned-material manifest has no role matching {role_slug}"
                )
            for name in PBR_MAPS:
                expected = role_manifest.get("maps", {}).get(name, {}).get("sha256")
                if not expected or sha256_file(source / name) != expected:
                    raise RealToSimError(
                        f"learned-material map is missing or changed: {role_slug}/{name}"
                    )
            learned_provenance = {
                "backend": learned_manifest.get("backend"),
                "model": learned_manifest.get("model"),
                "generation": learned_manifest.get("generation"),
                "runtime": learned_manifest.get("runtime"),
                "role": role_slug,
                "manifest_sha256": sha256_file(learned_manifest_path),
            }
    return {
        "provider": EXTERNAL_MATERIAL_PROVIDER,
        "texture_size": width,
        "base_color": base_color,
        "provenance": {
            "measured_input": "provider-defined",
            "generated_output": "externally supplied UV-aligned PBR atlas",
            "learned": True if learned_provenance else None,
            "provider_claim_unverified": learned_provenance is None,
            "bundle_label": root.name,
            "learned_bundle": learned_provenance,
        },
    }


def _write_obj(
    path: Path,
    mesh: MeshSurface,
    *,
    material_name: str,
    mtl_name: str,
) -> None:
    triangles = mesh.vertices[mesh.faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = cross / np.linalg.norm(cross, axis=1)[:, None]
    lines = [f"mtllib {mtl_name}", "o generated_completion"]
    lines.extend(
        f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}"
        for vertex in mesh.vertices
    )
    lines.extend(f"vt {uv[0]:.9g} {uv[1]:.9g}" for uv in mesh.uvs)
    lines.extend(
        f"vn {normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g}"
        for normal in normals
    )
    lines.append(f"usemtl {material_name}")
    for face_index, face in enumerate(mesh.faces):
        references = " ".join(
            f"{int(vertex) + 1}/{int(vertex) + 1}/{face_index + 1}" for vertex in face
        )
        lines.append(f"f {references}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_mtl(path: Path, *, material_name: str) -> None:
    path.write_text(
        "\n".join(
            (
                f"newmtl {material_name}",
                "Kd 1 1 1",
                "Pr 0.5",
                "Pm 0",
                "map_Kd baseColor.png",
                "map_Pr roughness.png",
                "map_Pm metallic.png",
                "map_Bump -bm 1 normal.png",
                "map_Ao ao.png",
                "",
            )
        ),
        encoding="utf-8",
    )


def _fit_diagnostics(
    workspace: Workspace,
    completion: dict[str, Any],
) -> dict[str, Any]:
    node = workspace.node(completion["node"])
    scene = load_gaussians(workspace.source_path)
    points = scene.positions[workspace.load_selection(node).astype(np.uint32)]
    profile = completion["visual_profile"]
    panel = Bounds.from_json(completion["template"]["panel_bounds"])
    axis = AXES[profile["front_axis"]]
    tangent = [component for component in range(3) if component != axis]
    quantile_min = np.quantile(points, 0.02, axis=0)
    quantile_max = np.quantile(points, 0.98, axis=0)
    quantile_max = np.maximum(quantile_max, quantile_min + 1e-6)
    observed_center = (quantile_min + quantile_max) * 0.5
    observed_size = quantile_max - quantile_min
    source_center = np.asarray(panel.center)
    panel_size = np.asarray(panel.size)
    target_center = source_center.copy()
    scale = np.ones(3, dtype=np.float64)
    if len(points) >= 32:
        for component in tangent:
            scale[component] = float(
                np.clip(observed_size[component] / panel_size[component], 0.85, 1.15)
            )
            center_delta = observed_center[component] - source_center[component]
            target_center[component] += float(
                np.clip(
                    center_delta,
                    -panel_size[component] * 0.10,
                    panel_size[component] * 0.10,
                )
            )
    initial_extent_error = {
        "XYZ"[component]: float(
            abs(panel_size[component] - observed_size[component])
            / max(panel_size[component], observed_size[component], 1e-6)
        )
        for component in tangent
    }
    outward_sign = int(profile["outward_sign"])
    source_front = panel.maximum[axis] if outward_sign > 0 else panel.minimum[axis]
    observed_front = quantile_max[axis] if outward_sign > 0 else quantile_min[axis]
    observed_front_delta = float(
        np.clip(
            observed_front - source_front,
            -panel_size[axis] * 0.5,
            panel_size[axis] * 0.5,
        )
    )
    # Gaussian selections provide reliable panel footprint but ambiguous depth:
    # splat extent, back-facing leakage, and partial occlusion bias front-axis
    # quantiles. Keep the authored joint/aperture plane locked and fit only the
    # two tangent dimensions locally.
    front_shift = 0.0
    fitted_size = panel_size * scale
    fitted_panel = Bounds.from_center_size(target_center, fitted_size)
    fitted_extent_error = {
        "XYZ"[component]: float(
            abs(fitted_size[component] - observed_size[component])
            / max(fitted_size[component], observed_size[component], 1e-6)
        )
        for component in tangent
    }
    fitted_front = (
        fitted_panel.maximum[axis]
        if outward_sign > 0
        else fitted_panel.minimum[axis]
    )
    front_residual = np.abs(points[:, axis] - fitted_front)
    threshold = float(max(panel_size[axis] * 2.5, 0.12))
    p95 = float(np.quantile(front_residual, 0.95))
    return {
        "fit_method": "robust-selected-point-quantile-affine-v1",
        "template_fit_method": completion["template"]["fit_method"],
        "measured_points": int(len(points)),
        "measured_quantile_bounds": Bounds(tuple(quantile_min), tuple(quantile_max)).to_json(),
        "template_panel_bounds": panel.to_json(),
        "fitted_panel_bounds": fitted_panel.to_json(),
        "initial_tangent_extent_relative_error": initial_extent_error,
        "tangent_extent_relative_error": fitted_extent_error,
        "front_plane_residual_p95": p95,
        "front_plane_threshold": threshold,
        "passed": bool(
            max(fitted_extent_error.values(), default=0.0) <= 0.15
            and p95 <= threshold
        ),
        "diagnostic_only": False,
        "transform": {
            "source_center": source_center.tolist(),
            "target_center": target_center.tolist(),
            "scale": scale.tolist(),
            "front_shift": front_shift if len(points) >= 32 else 0.0,
            "observed_front_delta": observed_front_delta,
            "front_axis_locked": True,
            "translation_clamped": True,
            "scale_range": [0.85, 1.15],
        },
    }


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _material_conditioning(
    workspace: Workspace,
    *,
    node_id: str,
) -> dict[str, Any]:
    node = workspace.node(node_id)
    renders = []
    for rgb_path in sorted((workspace.root / "evidence" / "render").glob("*/rgb.png")):
        report_path = rgb_path.with_name("render.json")
        renders.append(
            {
                "rgb": _artifact(rgb_path, workspace.root),
                "render_report": (
                    _artifact(report_path, workspace.root)
                    if report_path.is_file()
                    else None
                ),
            }
        )
    return {
        "immutable_source": {
            "path": workspace.state["source"]["path"],
            "sha256": workspace.state["source"]["sha256"],
        },
        "measured_selection": {
            "path": node.selection_file,
            "sha256": sha256_file(workspace.root / node.selection_file),
            "selected_gaussians": node.selected_gaussians,
        },
        "available_evidence_renders": renders,
    }


def fit_mesh_pbr_completion(
    workspace: Workspace,
    *,
    node_id: str,
    material_provider: str = LOCAL_MATERIAL_PROVIDER,
    external_material_bundle: Path | None = None,
    texture_size: int = 512,
    gaussian_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Upgrade one accepted completion to a UV/PBR mesh with face-bound Gaussians."""

    if material_provider not in {LOCAL_MATERIAL_PROVIDER, EXTERNAL_MATERIAL_PROVIDER}:
        raise RealToSimError(f"unsupported material provider: {material_provider}")
    if material_provider == EXTERNAL_MATERIAL_PROVIDER and external_material_bundle is None:
        raise RealToSimError("external-pbr-atlas-v1 requires --material-bundle")
    if not math.isfinite(gaussian_multiplier) or not 0.5 <= gaussian_multiplier <= 8.0:
        raise RealToSimError("Gaussian multiplier must be within [0.5, 8.0]")
    completion = _accepted_completion(workspace, node_id)
    node = workspace.node(node_id)
    if node.joint is None:
        raise RealToSimError("mesh/PBR completion requires an articulated target")
    template = completion["template"]
    kind = completion.get("template_kind")
    if kind not in {"cabinet-door", "drawer", "oven-door", "refrigerator-door"}:
        raise RealToSimError("accepted completion has no supported template_kind")
    measured_palette = _measured_palette(workspace, node_id)
    material_conditioning = _material_conditioning(workspace, node_id=node_id)
    fit_diagnostics = _fit_diagnostics(workspace, completion)
    if not fit_diagnostics["passed"]:
        raise RealToSimError(
            f"robust selected-point template fit failed for {node_id}: "
            f"{fit_diagnostics}"
        )
    source_assets = completion.get("surface_assets", completion["assets"])
    asset_by_role = {asset["role"]: asset for asset in source_assets}
    roles = (
        ("static-cavity", "world", "static_patches"),
        ("moving-interior", "joint", "moving_patches"),
    )
    output_root = workspace.root / "generated" / "mesh-pbr-completions" / node_id
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent)
    )
    staged_results: list[dict[str, Any]] = []
    try:
        for role, attachment, patch_key in roles:
            source_asset = asset_by_role.get(role)
            if source_asset is None:
                raise RealToSimError(f"accepted completion is missing its {role} asset")
            role_slug = role.replace("_", "-")
            role_directory = staged / role_slug
            role_directory.mkdir(parents=True, exist_ok=True)
            mesh = _mesh_from_patches(
                _apply_fit_transform(
                    _typed_patches(template[patch_key]),
                    fit_diagnostics,
                ),
                texture_size=texture_size,
            )
            material_name = "generated_completion_pbr"
            obj_path = role_directory / "mesh.obj"
            mtl_path = role_directory / "material.mtl"
            _write_obj(
                obj_path,
                mesh,
                material_name=material_name,
                mtl_name=mtl_path.name,
            )
            _write_mtl(mtl_path, material_name=material_name)
            request = {
                "schema_version": MESH_COMPLETION_SCHEMA,
                "contract": "nanousd-rts-pbr-atlas-v1",
                "node": node_id,
                "completion": completion["id"],
                "role": role,
                "attachment": attachment,
                "template_kind": kind,
                "mesh": obj_path.name,
                "uv_origin": "bottom-left",
                "atlas_tiles": mesh.atlas_tiles,
                "expected_maps": list(PBR_MAPS),
                "measured_palette": measured_palette,
                "conditioning": material_conditioning,
                "fit": fit_diagnostics,
                "model_prompt": LEARNED_MATERIAL_PROMPTS[kind],
                "prompt": (
                    f"Reconstruct a coherent high-quality {kind} hidden {role} material. "
                    "Match the observed front palette, preserve UV layout exactly, and "
                    "return base color, roughness, metallic, tangent-space normal, and AO."
                ),
                "provenance": {
                    "measured": False,
                    "reason": "hidden material is generated from a measured-front prior",
                },
            }
            request_path = role_directory / "material-request.json"
            request_path.write_text(
                json.dumps(request, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            if material_provider == LOCAL_MATERIAL_PROVIDER:
                material = _local_material_maps(
                    role_directory,
                    mesh,
                    kind=kind,
                    attachment=attachment,
                    measured_palette=measured_palette,
                    texture_size=texture_size,
                    seed=int(
                        content_digest(
                            {"completion": completion["id"], "role": role, "provider": material_provider}
                        )[-8:],
                        16,
                    ),
                )
            else:
                material = _external_material_maps(
                    Path(external_material_bundle),
                    role_directory,
                    role_slug=role_slug,
                )
            gaussian_path = role_directory / "mesh-bound.ply"
            association_path = role_directory / "mesh-bindings.npz"
            gaussian_count = int(
                math.ceil(int(source_asset["generated_gaussians"]) * gaussian_multiplier)
            )
            if gaussian_count > MESH_GAUSSIAN_RENDER_SAFE_LIMIT:
                raise RealToSimError(
                    f"{role} requests {gaussian_count} generated Gaussians; the current "
                    f"Metal RT path requires at most {MESH_GAUSSIAN_RENDER_SAFE_LIMIT} "
                    "per generated mesh asset to avoid its large-scene sigma clamp"
                )
            binding = write_mesh_bound_gaussians(
                gaussian_path,
                mesh.vertices,
                mesh.faces,
                face_colors=mesh.face_colors,
                count=gaussian_count,
                association_path=association_path,
                face_uvs=mesh.face_uvs,
                base_color_texture=material["base_color"],
                face_weights=mesh.face_weights,
                face_opacities=mesh.face_opacities,
                face_groups=mesh.face_patch_indices,
                seed=int(
                    content_digest(
                        {"completion": completion["id"], "role": role, "binding": 1}
                    )[-8:],
                    16,
                ),
            )
            role_manifest = {
                "schema_version": MESH_COMPLETION_SCHEMA,
                "role": role,
                "attachment": attachment,
                "template_kind": kind,
                "material_provider": material["provider"],
                "material_provenance": material["provenance"],
                "mesh": {
                    "vertices": binding["vertex_count"],
                    "faces": binding["face_count"],
                    "surface_area": binding["face_area"],
                    "bounds": Bounds(
                        tuple(mesh.vertices.min(axis=0)),
                        tuple(mesh.vertices.max(axis=0)),
                    ).to_json(),
                    "intended_open_shell": True,
                },
                "gaussian_binding": {
                    "count": binding["gaussian_count"],
                    "face_indices": True,
                    "barycentric_coordinates": True,
                    "uv_coordinates": True,
                    "face_aligned_frames": True,
                },
                "artifacts": {
                    "obj": obj_path.name,
                    "mtl": mtl_path.name,
                    "ply": gaussian_path.name,
                    "associations": association_path.name,
                    "material_request": request_path.name,
                    "pbr_maps": list(PBR_MAPS),
                },
            }
            role_manifest_path = role_directory / "manifest.json"
            role_manifest_path.write_text(
                json.dumps(role_manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            staged_results.append(
                {
                    "role": role,
                    "role_slug": role_slug,
                    "attachment": attachment,
                    "bounds": source_asset["bounds"],
                    "fitted_bounds": Bounds(
                        tuple(mesh.vertices.min(axis=0)),
                        tuple(mesh.vertices.max(axis=0)),
                    ).to_json(),
                    "generated_gaussians": gaussian_count,
                    "manifest": role_manifest,
                }
            )

        bundle_manifest = {
            "schema_version": MESH_COMPLETION_SCHEMA,
            "node": node_id,
            "completion": completion["id"],
            "template_digest": content_digest(template),
            "material_provider": material_provider,
            "fit_diagnostics": fit_diagnostics,
            "measured_palette": measured_palette,
            "conditioning": material_conditioning,
            "roles": [item["manifest"] for item in staged_results],
            "provenance": {
                "measured": False,
                "generator": MESH_COMPLETION_GENERATOR,
                "representation_separation": "measured source front remains immutable and separate",
            },
        }
        (staged / "manifest.json").write_text(
            json.dumps(bundle_manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _replace_directory(output_root, staged)
    except Exception:
        if staged.exists():
            shutil.rmtree(staged)
        raise

    upgraded_assets: list[dict[str, Any]] = []
    mesh_assets: list[dict[str, Any]] = []
    for result in staged_results:
        role_directory = output_root / result["role_slug"]
        gaussian_path = role_directory / "mesh-bound.ply"
        association_path = role_directory / "mesh-bindings.npz"
        upgraded_assets.append(
            {
                "role": result["role"],
                "attachment": result["attachment"],
                "asset": gaussian_path.relative_to(workspace.root).as_posix(),
                "asset_sha256": sha256_file(gaussian_path),
                "generated_gaussians": result["generated_gaussians"],
                "template_bounds": result["bounds"],
                "bounds": result["fitted_bounds"],
                "measured": False,
                "representation": "mesh-bound-gaussian",
            }
        )
        mesh_assets.append(
            {
                "role": result["role"],
                "attachment": result["attachment"],
                "manifest": _artifact(role_directory / "manifest.json", workspace.root),
                "mesh": _artifact(role_directory / "mesh.obj", workspace.root),
                "material": _artifact(role_directory / "material.mtl", workspace.root),
                "associations": _artifact(association_path, workspace.root),
                "material_request": _artifact(
                    role_directory / "material-request.json",
                    workspace.root,
                ),
                "pbr_maps": {
                    name: _artifact(role_directory / name, workspace.root)
                    for name in PBR_MAPS
                },
            }
        )

    sweep = sweep_joint(workspace, node_id=node_id)
    if not sweep["passed"]:
        raise RealToSimError(f"mesh/PBR upgrade failed articulation sweep for {node_id}")
    upgraded = dict(completion)
    upgraded["surface_assets"] = source_assets
    upgraded["assets"] = upgraded_assets
    moving_asset = next(item for item in upgraded_assets if item["attachment"] == "joint")
    upgraded["asset"] = moving_asset["asset"]
    upgraded["asset_sha256"] = moving_asset["asset_sha256"]
    upgraded["generated_gaussians"] = sum(
        int(item["generated_gaussians"]) for item in upgraded_assets
    )
    fitted_minimum = np.min(
        np.asarray([item["bounds"]["min"] for item in upgraded_assets]),
        axis=0,
    )
    fitted_maximum = np.max(
        np.asarray([item["bounds"]["max"] for item in upgraded_assets]),
        axis=0,
    )
    upgraded["bounds"] = Bounds(
        tuple(fitted_minimum),
        tuple(fitted_maximum),
    ).to_json()
    upgraded["mesh_assets"] = mesh_assets
    upgraded["mesh_bundle_manifest"] = _artifact(
        output_root / "manifest.json",
        workspace.root,
    )
    upgraded["representation"] = {
        "type": "mesh-bound-gaussian-pbr",
        "schema_version": MESH_COMPLETION_SCHEMA,
        "mesh_face_association": True,
        "barycentric_coordinates": True,
        "face_aligned_gaussians": True,
        "pbr_material_bundle": True,
        "material_provider": material_provider,
    }
    upgraded["fit_diagnostics"] = fit_diagnostics
    evaluation = dict(upgraded.get("evaluation", {}))
    evaluation.update(
        {
            "mesh_bundle_complete": True,
            "mesh_face_associations_complete": True,
            "pbr_maps_complete": True,
            "post_upgrade_articulation_sweep_passed": True,
        }
    )
    upgraded["evaluation"] = evaluation
    previous_provenance = dict(upgraded.get("provenance", {}))
    upgraded["provenance"] = {
        **previous_provenance,
        "source": "measured-panel-fitted category mesh with generated hidden PBR material",
        "measured": False,
        "generator": MESH_COMPLETION_GENERATOR,
        "material_provider": material_provider,
        "lineage": {
            "surface_generator": previous_provenance.get("generator"),
            "template_digest": content_digest(template),
        },
    }
    workspace.put_completion(upgraded)
    workspace.trace(
        "fit-mesh-pbr",
        {
            "node": node_id,
            "completion": completion["id"],
            "material_provider": material_provider,
            "external_material_bundle": (
                str(Path(external_material_bundle).resolve())
                if external_material_bundle is not None
                else None
            ),
            "texture_size": texture_size,
            "gaussian_multiplier": gaussian_multiplier,
        },
        {
            "representation": upgraded["representation"],
            "mesh_bundle_manifest": upgraded["mesh_bundle_manifest"],
            "fit_diagnostics": fit_diagnostics,
        },
    )
    return upgraded
