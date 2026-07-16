"""Procedural amodal articulation interiors with explicit generated provenance."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .core import AXES, Bounds, Collider, RealToSimError, Workspace, content_digest, sha256_file
from .gaussian import write_surface_patch_gaussians
from .sim import sweep_joint
from .visual_articulation import SPLAT_TRANSFORM_PACKAGE


VISUAL_COMPLETION_SCHEMA = 1
VISUAL_COMPLETION_GENERATOR = "nanousd-rts amodal articulation surfaces v2"


def _union_bounds(bounds: Iterable[Bounds]) -> Bounds:
    values = tuple(bounds)
    if not values:
        raise RealToSimError("cannot union an empty bounds collection")
    return Bounds(
        tuple(np.min(np.asarray([item.minimum for item in values]), axis=0)),
        tuple(np.max(np.asarray([item.maximum for item in values]), axis=0)),
    )


def _patch(
    bounds: Bounds,
    axis: int,
    side: int,
    color: tuple[float, float, float],
    *,
    weight: float = 1.0,
    opacity: float = 0.97,
) -> dict[str, Any]:
    return {
        "bounds": bounds,
        "axis": axis,
        "side": side,
        "color": color,
        "weight": weight,
        "opacity": opacity,
    }


def _box_patches(
    bounds: Bounds,
    color: tuple[float, float, float],
    *,
    omit: set[tuple[int, int]] | None = None,
    weight: float = 1.0,
    opacity: float = 0.97,
) -> list[dict[str, Any]]:
    omitted = omit or set()
    return [
        _patch(bounds, axis, side, color, weight=weight, opacity=opacity)
        for axis in range(3)
        for side in (-1, 1)
        if (axis, side) not in omitted
    ]


def _inset_bounds(
    bounds: Bounds,
    *,
    axes: Iterable[int],
    inset: float,
) -> Bounds:
    minimum = np.asarray(bounds.minimum, dtype=np.float64)
    maximum = np.asarray(bounds.maximum, dtype=np.float64)
    for axis in axes:
        amount = min(inset, (maximum[axis] - minimum[axis]) * 0.18)
        minimum[axis] += amount
        maximum[axis] -= amount
    return Bounds(tuple(minimum), tuple(maximum))


def _axis_interval(first: float, second: float) -> tuple[float, float]:
    return (min(first, second), max(first, second))


def _slab(
    base: Bounds,
    *,
    axis: int,
    minimum: float,
    maximum: float,
) -> Bounds:
    lo = np.asarray(base.minimum, dtype=np.float64)
    hi = np.asarray(base.maximum, dtype=np.float64)
    lo[axis], hi[axis] = _axis_interval(minimum, maximum)
    return Bounds(tuple(lo), tuple(hi))


def _shelf_bounds(
    cavity: Bounds,
    *,
    front_axis: int,
    up_axis: int,
    center: float,
    thickness: float,
) -> Bounds:
    axes = [axis for axis in range(3) if axis not in {front_axis, up_axis}]
    shelf = _inset_bounds(cavity, axes=axes, inset=0.018)
    shelf = _inset_bounds(shelf, axes=(front_axis,), inset=0.055)
    return _slab(
        shelf,
        axis=up_axis,
        minimum=center - thickness * 0.5,
        maximum=center + thickness * 0.5,
    )


def _completion_geometry(
    workspace: Workspace,
    *,
    node_id: str,
    kind: str,
    front_axis: str,
    outward_sign: int,
    depth: float,
    up_sign: int,
    shelf_count: int,
) -> dict[str, Any]:
    node = workspace.node(node_id)
    if node.collider is None or node.joint is None:
        raise RealToSimError("visual completion requires an articulated node and collider")
    if kind not in {"cabinet-door", "drawer", "oven-door", "refrigerator-door"}:
        raise RealToSimError(f"unsupported visual completion kind: {kind}")
    if front_axis not in AXES or outward_sign not in {-1, 1} or up_sign not in {-1, 1}:
        raise RealToSimError("completion axes/signs are invalid")
    if depth <= 0.08:
        raise RealToSimError("completion depth must be greater than 0.08")

    axis = AXES[front_axis]
    up_axis = AXES[workspace.up_axis]
    tangent_axes = [index for index in range(3) if index != axis]
    panel = node.collider.bounds
    panel_inner = panel.minimum[axis] if outward_sign > 0 else panel.maximum[axis]
    open_coordinate = panel_inner - outward_sign * 0.045
    back_coordinate = open_coordinate - outward_sign * depth
    cavity = _inset_bounds(panel, axes=tangent_axes, inset=0.035)
    cavity = _slab(
        cavity,
        axis=axis,
        minimum=open_coordinate,
        maximum=back_coordinate,
    )
    open_side = 1 if outward_sign > 0 else -1

    palette = {
        "cabinet-door": {
            "cavity": (0.78, 0.72, 0.61),
            "moving": (0.91, 0.90, 0.84),
            "accent": (0.58, 0.52, 0.44),
        },
        "drawer": {
            "cavity": (0.69, 0.63, 0.54),
            "moving": (0.82, 0.77, 0.67),
            "accent": (0.52, 0.47, 0.40),
        },
        "oven-door": {
            "cavity": (0.33, 0.35, 0.38),
            "moving": (0.24, 0.26, 0.29),
            "accent": (0.68, 0.70, 0.72),
        },
        "refrigerator-door": {
            "cavity": (0.62, 0.68, 0.68),
            "moving": (0.48, 0.55, 0.56),
            "accent": (0.30, 0.38, 0.40),
        },
    }[kind]

    static_patches = _box_patches(
        cavity,
        palette["cavity"],
        omit={(axis, open_side)},
        opacity=0.985,
    )
    if kind == "oven-door":
        static_patches.append(
            _patch(
                cavity,
                axis,
                -open_side,
                palette["cavity"],
                weight=2.6,
                opacity=0.99,
            )
        )
        readable_back_coordinate = (
            open_coordinate - outward_sign * depth * 0.56
        )
        readable_back = _slab(
            cavity,
            axis=axis,
            minimum=readable_back_coordinate,
            maximum=readable_back_coordinate - outward_sign * 0.012,
        )
        static_patches.append(
            _patch(
                readable_back,
                axis,
                open_side,
                (0.46, 0.48, 0.51),
                weight=3.0,
                opacity=0.995,
            )
        )
    vertical_min = cavity.minimum[up_axis]
    vertical_max = cavity.maximum[up_axis]
    for index in range(shelf_count):
        fraction = (index + 1) / (shelf_count + 1)
        center = vertical_min + (vertical_max - vertical_min) * fraction
        shelf = _shelf_bounds(
            cavity,
            front_axis=axis,
            up_axis=up_axis,
            center=center,
            thickness=0.018 if kind != "oven-door" else 0.012,
        )
        static_patches.extend(
            _box_patches(
                shelf,
                palette["accent"],
                weight=0.65,
                opacity=0.975,
            )
        )

    liner = _inset_bounds(panel, axes=tangent_axes, inset=0.022)
    liner = _slab(
        liner,
        axis=axis,
        minimum=panel_inner,
        maximum=panel_inner - outward_sign * 0.024,
    )
    moving_patches = _box_patches(liner, palette["moving"], opacity=0.985)
    moving_bounds = [liner]
    collision_bounds = liner

    if kind == "oven-door":
        window = _inset_bounds(liner, axes=tangent_axes, inset=0.12)
        inner_coordinate = (
            window.minimum[axis] if outward_sign > 0 else window.maximum[axis]
        )
        moving_patches.append(
            _patch(
                window,
                axis,
                -open_side,
                (0.035, 0.045, 0.055),
                weight=1.8,
                opacity=0.99,
            )
        )
        rim = _slab(
            window,
            axis=axis,
            minimum=inner_coordinate,
            maximum=inner_coordinate - outward_sign * 0.008,
        )
        moving_bounds.append(rim)
    elif kind == "refrigerator-door":
        width_axis = next(
            index for index in range(3) if index not in {axis, up_axis}
        )
        for fraction in (0.30, 0.68):
            center = vertical_min + (vertical_max - vertical_min) * fraction
            bin_bounds = _inset_bounds(liner, axes=(width_axis,), inset=0.055)
            bin_bounds = _slab(
                bin_bounds,
                axis=up_axis,
                minimum=center - 0.075,
                maximum=center + 0.075,
            )
            bin_bounds = _slab(
                bin_bounds,
                axis=axis,
                minimum=panel_inner - outward_sign * 0.02,
                maximum=panel_inner - outward_sign * 0.13,
            )
            moving_patches.extend(
                _box_patches(
                    bin_bounds,
                    palette["accent"],
                    omit={(up_axis, up_sign)},
                    weight=0.8,
                    opacity=0.98,
                )
            )
            moving_bounds.append(bin_bounds)
    elif kind == "drawer":
        drawer = _inset_bounds(cavity, axes=tangent_axes, inset=0.025)
        drawer = _slab(
            drawer,
            axis=axis,
            minimum=panel_inner - outward_sign * 0.015,
            maximum=panel_inner - outward_sign * depth * 0.82,
        )
        top_side = up_sign
        moving_patches = _box_patches(
            drawer,
            palette["moving"],
            omit={(axis, open_side), (up_axis, top_side)},
            opacity=0.985,
        )
        moving_bounds = [drawer]
        collision_bounds = drawer

    return {
        "static_patches": static_patches,
        "moving_patches": moving_patches,
        "static_bounds": cavity,
        "moving_bounds": _union_bounds(moving_bounds),
        "collision_bounds": collision_bounds,
        "front_axis": front_axis,
        "outward_sign": outward_sign,
        "up_sign": up_sign,
        "depth": depth,
        "kind": kind,
    }


def author_visual_completion(
    workspace: Workspace,
    *,
    node_id: str,
    kind: str,
    front_axis: str,
    outward_sign: int,
    depth: float,
    up_sign: int = 1,
    shelf_count: int = 1,
    static_gaussians: int = 2200,
    moving_gaussians: int = 900,
    confidence: float = 0.72,
    background_occlusion_bounds: Bounds | None = None,
) -> dict[str, Any]:
    """Author and immediately accept one provenance-labeled amodal completion."""

    if not 0.0 <= confidence <= 1.0:
        raise RealToSimError("completion confidence must be in [0, 1]")
    geometry = _completion_geometry(
        workspace,
        node_id=node_id,
        kind=kind,
        front_axis=front_axis,
        outward_sign=outward_sign,
        depth=depth,
        up_sign=up_sign,
        shelf_count=shelf_count,
    )
    completion_id = f"{node_id}.amodal.01"
    output_root = workspace.root / "generated" / "visual-completions" / node_id
    static_asset = write_surface_patch_gaussians(
        output_root / "static-cavity.ply",
        geometry["static_patches"],
        count=static_gaussians,
        seed=int(content_digest({"id": completion_id, "role": "static"})[-8:], 16),
    )
    moving_asset = write_surface_patch_gaussians(
        output_root / "moving-interior.ply",
        geometry["moving_patches"],
        count=moving_gaussians,
        seed=int(content_digest({"id": completion_id, "role": "moving"})[-8:], 16),
    )

    node = workspace.node(node_id)
    if node.collider is None:
        raise RealToSimError("visual completion target has no collider")
    original = node
    moving_union = _union_bounds((node.collider.bounds, geometry["collision_bounds"]))
    updated = replace(
        node,
        collider=Collider(
            kind="box",
            center=moving_union.center,
            size=moving_union.size,
            rotation_wxyz=node.collider.rotation_wxyz,
            provenance=f"accepted-completion:{completion_id}",
            confidence=max(node.collider.confidence, confidence),
            collision_mode=node.collider.collision_mode,
        ),
    )
    workspace.put_node(updated)
    sweep = sweep_joint(workspace, node_id=node_id)
    if not sweep["passed"]:
        workspace.put_node(original)
        raise RealToSimError(
            f"generated moving interior failed the articulation sweep: {node_id}"
        )

    for existing in workspace.completions:
        if existing.get("node") != node_id:
            continue
        changed = dict(existing)
        changed["status"] = "rejected"
        changed["acceptance"] = {
            "articulation_sweep_passed": True,
            "selected_by": "superseded by amodal articulation completion",
        }
        workspace.put_completion(changed)

    static_relative = static_asset.relative_to(workspace.root).as_posix()
    moving_relative = moving_asset.relative_to(workspace.root).as_posix()
    static_sha = sha256_file(static_asset)
    moving_sha = sha256_file(moving_asset)
    all_bounds = _union_bounds(
        (geometry["static_bounds"], geometry["moving_bounds"])
    )
    record = {
        "id": completion_id,
        "node": node_id,
        "status": "accepted",
        "kind": f"generated-amodal-{kind}-surfaces",
        "asset": moving_relative,
        "asset_sha256": moving_sha,
        "assets": [
            {
                "role": "static-cavity",
                "attachment": "world",
                "asset": static_relative,
                "asset_sha256": static_sha,
                "generated_gaussians": static_gaussians,
                "bounds": geometry["static_bounds"].to_json(),
                "measured": False,
            },
            {
                "role": "moving-interior",
                "attachment": "joint",
                "asset": moving_relative,
                "asset_sha256": moving_sha,
                "generated_gaussians": moving_gaussians,
                "bounds": geometry["moving_bounds"].to_json(),
                "measured": False,
            },
        ],
        "bounds": all_bounds.to_json(),
        "collider_bounds": geometry["collision_bounds"].to_json(),
        "generated_gaussians": static_gaussians + moving_gaussians,
        "confidence": confidence,
        "visual_profile": {
            "front_axis": front_axis,
            "outward_sign": outward_sign,
            "up_sign": up_sign,
            "depth": depth,
            "shelf_count": shelf_count,
            "background_occlusion_bounds": (
                background_occlusion_bounds.to_json()
                if background_occlusion_bounds
                else None
            ),
        },
        "evaluation": {
            "articulation_sweep_passed": True,
            "static_cavity_is_not_attached_to_joint": True,
            "moving_interior_is_attached_to_joint": True,
        },
        "provenance": {
            "source": "procedural category-level amodal completion prior",
            "measured": False,
            "generator": VISUAL_COMPLETION_GENERATOR,
            "requires_review": True,
            "representation_separation": "measured fronts remain separate assets",
        },
        "acceptance": {
            "articulation_sweep_passed": True,
            "selected_by": "explicit deterministic scene profile",
        },
    }
    workspace.put_completion(record)
    workspace.trace(
        "author-visual-completion",
        {
            "node": node_id,
            "kind": kind,
            "front_axis": front_axis,
            "outward_sign": outward_sign,
            "depth": depth,
            "up_sign": up_sign,
            "shelf_count": shelf_count,
            "background_occlusion_bounds": (
                background_occlusion_bounds.to_json()
                if background_occlusion_bounds
                else None
            ),
        },
        {"completion": completion_id, "assets": record["assets"]},
    )
    return record


def _replace_directory(destination: Path, staged: Path) -> None:
    backup = destination.with_name(f".{destination.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.rename(backup)
    staged.rename(destination)
    if backup.exists():
        shutil.rmtree(backup)


def materialize_visual_completions(
    workspace: Workspace,
    destination: Path,
) -> dict[str, Any]:
    """Compress accepted generated completion parts for the browser viewer."""

    accepted = [
        item
        for item in workspace.completions
        if item.get("status") == "accepted" and item.get("assets")
    ]
    signature = content_digest(
        {
            "schema_version": VISUAL_COMPLETION_SCHEMA,
            "converter": SPLAT_TRANSFORM_PACKAGE,
            "assets": [
                {
                    "id": item["id"],
                    "node": item["node"],
                    "assets": [
                        {
                            "role": asset["role"],
                            "asset_sha256": asset["asset_sha256"],
                            "attachment": asset["attachment"],
                        }
                        for asset in item["assets"]
                    ],
                }
                for item in accepted
            ],
        }
    )
    destination = Path(destination).resolve()
    marker = destination / "manifest.json"
    if marker.is_file():
        try:
            current = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = None
        if current and current.get("signature") == signature:
            if all(
                (destination / item["output"]).is_file()
                for item in current.get("assets", [])
            ):
                return current

    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    reports: list[dict[str, Any]] = []
    try:
        for completion in accepted:
            node = workspace.node(completion["node"])
            for asset in completion["assets"]:
                source = workspace.root / asset["asset"]
                if not source.is_file() or sha256_file(source) != asset["asset_sha256"]:
                    raise RealToSimError(
                        f"generated completion asset is missing or changed: {source}"
                    )
                role_slug = asset["role"].replace("_", "-")
                output_directory = staged / completion["node"] / role_slug
                output_directory.mkdir(parents=True, exist_ok=True)
                output_meta = output_directory / "meta.json"
                completed = subprocess.run(
                    [
                        "npx",
                        "--yes",
                        SPLAT_TRANSFORM_PACKAGE,
                        "-w",
                        str(source),
                        str(output_meta),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if completed.returncode != 0 or not output_meta.is_file():
                    raise RealToSimError(
                        f"failed to compress generated completion {completion['id']} "
                        f"{asset['role']}:\n"
                        + (
                            completed.stderr
                            or completed.stdout
                            or "splat-transform produced no meta.json"
                        )
                    )
                relative_output = (
                    Path(completion["node"]) / role_slug / "meta.json"
                ).as_posix()
                reports.append(
                    {
                        "id": f"{completion['node']}:{asset['role']}",
                        "node": completion["node"],
                        "completion": completion["id"],
                        "role": asset["role"],
                        "attachment": asset["attachment"],
                        "url": f"./generated/{relative_output}",
                        "output": relative_output,
                        "gaussian_count": int(asset["generated_gaussians"]),
                        "bounds": asset["bounds"],
                        "joint": (
                            node.to_json().get("joint")
                            if asset["attachment"] == "joint"
                            else None
                        ),
                        "measured": False,
                        "provenance": completion["provenance"],
                    }
                )

        report = {
            "schema_version": VISUAL_COMPLETION_SCHEMA,
            "signature": signature,
            "converter": SPLAT_TRANSFORM_PACKAGE,
            "assets": reports,
            "static_assets": [
                item for item in reports if item["attachment"] == "world"
            ],
            "moving_assets": [
                item for item in reports if item["attachment"] == "joint"
            ],
            "generated_gaussians": sum(
                int(item["gaussian_count"]) for item in reports
            ),
        }
        (staged / "manifest.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _replace_directory(destination, staged)
        return report
    except Exception:
        if staged.exists():
            shutil.rmtree(staged)
        raise
