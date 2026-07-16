"""Lossless SOG background masking and high-resolution articulated splat extraction."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .core import Bounds, RealToSimError, SceneNode, content_digest, sha256_file
from .gaussian import GaussianScene, load_gaussians, write_gaussians


SPLAT_TRANSFORM_PACKAGE = "@playcanvas/splat-transform@2.7.1"
ARTICULATION_ASSET_SCHEMA = 4
REFERENCE_BOUNDS_EPSILON = 1e-4
REFERENCE_LABEL_BATCH_SIZE = 1024


def _decode_sog_means(
    meta: dict[str, Any],
    means_low: np.ndarray,
    means_high: np.ndarray,
    count: int,
) -> np.ndarray:
    low = np.asarray(means_low, dtype=np.uint16).reshape(-1, 4)[:count, :3]
    high = np.asarray(means_high, dtype=np.uint16).reshape(-1, 4)[:count, :3]
    quantized = low + (high << 8)
    minimum = np.asarray(meta["means"]["mins"], dtype=np.float64)
    maximum = np.asarray(meta["means"]["maxs"], dtype=np.float64)
    transformed = minimum + quantized.astype(np.float64) / 65535.0 * (maximum - minimum)
    positions = np.sign(transformed) * np.expm1(np.abs(transformed))
    return np.ascontiguousarray(positions, dtype=np.float32)


def _decode_sog_quaternions(encoded: np.ndarray) -> np.ndarray:
    pixels = np.asarray(encoded, dtype=np.uint8).reshape(-1, 4)
    compressed = ((pixels[:, :3].astype(np.float32) / 255.0) * 2.0 - 1.0) / math.sqrt(2.0)
    maximum_component = pixels[:, 3].astype(np.int16) - 252
    if np.any((maximum_component < 0) | (maximum_component > 3)):
        raise RealToSimError("SOG quaternion texture contains an invalid maximum-component tag")
    result = np.zeros((len(pixels), 4), dtype=np.float32)
    component_tables = (
        (1, 2, 3),
        (0, 2, 3),
        (0, 1, 3),
        (0, 1, 2),
    )
    for component, others in enumerate(component_tables):
        selected = maximum_component == component
        if not np.any(selected):
            continue
        result[np.ix_(selected, others)] = compressed[selected]
        remainder = 1.0 - np.sum(result[selected] * result[selected], axis=1)
        result[selected, component] = np.sqrt(np.maximum(remainder, 0.0))
    norms = np.linalg.norm(result, axis=1)
    if np.any(norms < 1e-8):
        raise RealToSimError("decoded SOG quaternion has zero length")
    result /= norms[:, None]
    return result


def _nearest_reference_labels(
    query_positions: np.ndarray,
    reference_positions: np.ndarray,
    reference_labels: np.ndarray,
    *,
    batch_size: int = REFERENCE_LABEL_BATCH_SIZE,
) -> np.ndarray:
    """Transfer boolean labels to query points from their nearest spatial reference."""

    query = np.asarray(query_positions, dtype=np.float32)
    reference = np.asarray(reference_positions, dtype=np.float32)
    labels = np.asarray(reference_labels, dtype=bool)
    if query.ndim != 2 or query.shape[1] != 3:
        raise RealToSimError("query positions must have shape (N, 3)")
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise RealToSimError("reference positions must have shape (N, 3)")
    if len(reference) == 0 or labels.shape != (len(reference),):
        raise RealToSimError("reference positions require one boolean label per point")
    if batch_size <= 0:
        raise RealToSimError("reference-label batch size must be positive")

    transferred = np.empty(len(query), dtype=bool)
    for start in range(0, len(query), batch_size):
        stop = min(start + batch_size, len(query))
        delta = query[start:stop, None, :] - reference[None, :, :]
        squared_distance = np.sum(delta * delta, axis=2)
        transferred[start:stop] = labels[np.argmin(squared_distance, axis=1)]
    return transferred


def _load_articulation_references(
    working_ply: Path,
    selection_root: Path,
    nodes: list[SceneNode],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, dict[str, int]]]:
    """Build local positive/negative labels from stable working-PLY selections."""

    working_scene = load_gaussians(working_ply)
    selection_root = Path(selection_root).resolve()
    references: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    reports: dict[str, dict[str, int]] = {}
    for node in nodes:
        if node.selection_mode == "bounds":
            if node.selection_bounds is None:
                raise RealToSimError(
                    f"bounds-selected articulation has no selection bounds: {node.node_id}"
                )
            reports[node.node_id] = {
                "working_ply_gaussians": working_scene.count,
                "local_references": 0,
                "positive_references": node.selected_gaussians,
                "negative_references": 0,
            }
            continue
        selection_path = (selection_root / node.selection_file).resolve()
        if not selection_path.is_relative_to(selection_root) or not selection_path.is_file():
            raise RealToSimError(f"visual articulation selection is missing: {selection_path}")
        selected_indices = np.unique(np.load(selection_path, allow_pickle=False).astype(np.int64))
        if len(selected_indices) != node.selected_gaussians:
            raise RealToSimError(
                f"visual articulation selection count changed for {node.node_id}: "
                f"expected {node.selected_gaussians}, got {len(selected_indices)}"
            )
        if len(selected_indices) == 0 or selected_indices[0] < 0 or selected_indices[-1] >= working_scene.count:
            raise RealToSimError(f"visual articulation selection is out of range: {node.node_id}")

        minimum = np.asarray(node.visual_bounds.minimum, dtype=np.float32) - REFERENCE_BOUNDS_EPSILON
        maximum = np.asarray(node.visual_bounds.maximum, dtype=np.float32) + REFERENCE_BOUNDS_EPSILON
        inside = np.all(
            (working_scene.positions >= minimum) & (working_scene.positions <= maximum),
            axis=1,
        )
        local_indices = np.flatnonzero(inside)
        labels = np.zeros(len(local_indices), dtype=bool)
        labels[np.isin(local_indices, selected_indices, assume_unique=True)] = True
        positive = int(labels.sum())
        negative = int(len(labels) - positive)
        if positive != len(selected_indices):
            raise RealToSimError(
                f"visual bounds exclude selected references for {node.node_id}: "
                f"expected {len(selected_indices)}, found {positive}"
            )
        if negative == 0:
            raise RealToSimError(
                f"visual articulation needs negative neighborhood references: {node.node_id}"
            )
        references[node.node_id] = (
            np.ascontiguousarray(working_scene.positions[local_indices], dtype=np.float32),
            labels,
        )
        reports[node.node_id] = {
            "working_ply_gaussians": working_scene.count,
            "local_references": int(len(local_indices)),
            "positive_references": positive,
            "negative_references": negative,
        }
    return references, reports


def _node_signature(node: SceneNode) -> dict[str, Any]:
    if node.joint is None:
        raise RealToSimError(f"visual articulation node has no joint: {node.node_id}")
    return {
        "id": node.node_id,
        "bounds": node.visual_bounds.to_json(),
        "selection_mode": node.selection_mode,
        "selection_bounds": (
            node.selection_bounds.to_json() if node.selection_bounds else None
        ),
        "joint": asdict(node.joint),
    }


def _normalize_background_occlusions(
    values: list[dict[str, Any]] | None,
) -> list[tuple[str, Bounds]]:
    normalized: list[tuple[str, Bounds]] = []
    identifiers: set[str] = set()
    for value in values or []:
        identifier = str(value.get("id", "")).strip()
        if not identifier:
            raise RealToSimError("background occlusion requires a non-empty id")
        if identifier in identifiers:
            raise RealToSimError(f"duplicate background occlusion id: {identifier}")
        bounds_value = value.get("bounds")
        if not isinstance(bounds_value, dict):
            raise RealToSimError(
                f"background occlusion requires JSON bounds: {identifier}"
            )
        normalized.append((identifier, Bounds.from_json(bounds_value)))
        identifiers.add(identifier)
    return normalized


def _replace_directory(destination: Path, staged: Path) -> None:
    backup = destination.with_name(f".{destination.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.rename(backup)
    staged.rename(destination)
    if backup.exists():
        shutil.rmtree(backup)


def _link_file(destination: Path, source: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve())


def _save_masked_opacity(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA").save(
        path,
        format="WEBP",
        lossless=True,
        quality=100,
        method=4,
    )


def _marker_is_current(destination: Path, signature: str, node_ids: tuple[str, ...]) -> dict[str, Any] | None:
    marker = destination / "manifest.json"
    if not marker.is_file():
        return None
    try:
        report = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if report.get("signature") != signature:
        return None
    if not (destination / "background" / "lod-meta.json").is_file():
        return None
    if any(not (destination / "objects" / node_id / "meta.json").is_file() for node_id in node_ids):
        return None
    return report


def materialize_articulated_sog(
    source: Path,
    destination: Path,
    nodes: list[SceneNode],
    *,
    working_ply: Path,
    selection_root: Path,
    background_occlusions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Mask articulated objects and generated-replacement apertures from the background."""

    source = Path(source).resolve()
    destination = Path(destination).resolve()
    working_ply = Path(working_ply).resolve()
    selection_root = Path(selection_root).resolve()
    articulated = [node for node in nodes if node.joint is not None]
    occlusions = _normalize_background_occlusions(background_occlusions)
    if not articulated:
        return {
            "schema_version": ARTICULATION_ASSET_SCHEMA,
            "background_url": None,
            "objects": [],
            "masked_gaussians_by_lod": {},
            "background_occluded_gaussians_by_lod": {},
        }
    lod_meta_path = source / "lod-meta.json"
    if not lod_meta_path.is_file():
        raise RealToSimError("visual Gaussian articulation currently requires a SOG/LOD source")
    lod_meta = json.loads(lod_meta_path.read_text(encoding="utf-8"))
    selection_digests = {
        node.node_id: sha256_file(selection_root / node.selection_file)
        for node in articulated
    }
    signature = content_digest(
        {
            "schema_version": ARTICULATION_ASSET_SCHEMA,
            "source_lod_meta_sha256": sha256_file(lod_meta_path),
            "working_ply_sha256": sha256_file(working_ply),
            "selection_sha256": selection_digests,
            "nodes": [_node_signature(node) for node in articulated],
            "background_occlusions": [
                {"id": identifier, "bounds": bounds.to_json()}
                for identifier, bounds in occlusions
            ],
            "extract_lod": 0,
            "converter": SPLAT_TRANSFORM_PACKAGE,
        }
    )
    node_ids = tuple(node.node_id for node in articulated)
    current = _marker_is_current(destination, signature, node_ids)
    if current is not None:
        return current

    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    background = staged / "background"
    objects_root = staged / "objects"
    background.mkdir(parents=True)
    objects_root.mkdir(parents=True)

    references, reference_reports = _load_articulation_references(
        working_ply,
        selection_root,
        articulated,
    )
    boxes = {
        node.node_id: (
            np.asarray(
                (node.selection_bounds or node.visual_bounds).minimum,
                dtype=np.float32,
            ),
            np.asarray(
                (node.selection_bounds or node.visual_bounds).maximum,
                dtype=np.float32,
            ),
        )
        for node in articulated
    }
    selection_modes = {node.node_id: node.selection_mode for node in articulated}
    occlusion_boxes = {
        identifier: (
            np.asarray(bounds.minimum, dtype=np.float32),
            np.asarray(bounds.maximum, dtype=np.float32),
        )
        for identifier, bounds in occlusions
    }
    extracted: dict[str, dict[str, list[np.ndarray]]] = {
        node.node_id: {
            "positions": [],
            "scales": [],
            "orientations": [],
            "opacities": [],
            "colors": [],
        }
        for node in articulated
    }
    masked_by_lod: dict[str, dict[str, int]] = {
        str(level): {node.node_id: 0 for node in articulated}
        for level in range(int(lod_meta["lodLevels"]))
    }
    background_occluded_by_lod: dict[str, dict[str, int]] = {
        str(level): {identifier: 0 for identifier, _ in occlusions}
        for level in range(int(lod_meta["lodLevels"]))
    }

    try:
        _link_file(background / "lod-meta.json", lod_meta_path)
        chunk_meta_files = [Path(name) for name in lod_meta.get("filenames", [])]
        for relative_meta in chunk_meta_files:
            source_chunk = source / relative_meta.parent
            destination_chunk = background / relative_meta.parent
            destination_chunk.mkdir(parents=True, exist_ok=True)
            meta = json.loads((source_chunk / "meta.json").read_text(encoding="utf-8"))
            if int(meta.get("version", 0)) != 2:
                raise RealToSimError(f"visual articulation requires SOG v2 chunks: {source_chunk}")
            if meta.get("shN"):
                raise RealToSimError("visual articulation currently supports degree-zero SOG sources")
            count = int(meta["count"])
            level = int(relative_meta.parts[0].split("_", 1)[0])

            means_low = np.asarray(Image.open(source_chunk / meta["means"]["files"][0]).convert("RGBA"))
            means_high = np.asarray(Image.open(source_chunk / meta["means"]["files"][1]).convert("RGBA"))
            positions = _decode_sog_means(meta, means_low, means_high, count)
            masks: dict[str, np.ndarray] = {}
            moving_union = np.zeros(count, dtype=bool)
            for node_id, (minimum, maximum) in boxes.items():
                candidates = np.all((positions >= minimum) & (positions <= maximum), axis=1)
                selected = np.zeros(count, dtype=bool)
                candidate_indices = np.flatnonzero(candidates)
                if candidate_indices.size:
                    if selection_modes[node_id] == "bounds":
                        selected[candidate_indices] = True
                    else:
                        reference_positions, reference_labels = references[node_id]
                        selected[candidate_indices] = _nearest_reference_labels(
                            positions[candidate_indices],
                            reference_positions,
                            reference_labels,
                        )
                masks[node_id] = selected
                moving_union |= selected
                masked_by_lod[str(level)][node_id] += int(selected.sum())

            background_union = moving_union.copy()
            for identifier, (minimum, maximum) in occlusion_boxes.items():
                occluded = np.all(
                    (positions >= minimum) & (positions <= maximum),
                    axis=1,
                )
                background_union |= occluded
                background_occluded_by_lod[str(level)][identifier] += int(
                    occluded.sum()
                )

            sh0_path = source_chunk / meta["sh0"]["files"][0]
            sh0_rgba = np.asarray(Image.open(sh0_path).convert("RGBA")).copy()
            sh0_original = sh0_rgba.copy()
            if np.any(background_union):
                sh0_flat = sh0_rgba.reshape(-1, 4)
                sh0_flat[:count, 3][background_union] = 0
                _save_masked_opacity(destination_chunk / sh0_path.name, sh0_rgba)
            else:
                _link_file(destination_chunk / sh0_path.name, sh0_path)

            for source_file in source_chunk.iterdir():
                if source_file.name == sh0_path.name:
                    continue
                _link_file(destination_chunk / source_file.name, source_file)

            if level != 0 or not np.any(moving_union):
                continue

            selected_indices = np.flatnonzero(moving_union)
            selected_lookup = {int(source_index): output_index for output_index, source_index in enumerate(selected_indices)}
            scales_rgba = np.asarray(
                Image.open(source_chunk / meta["scales"]["files"][0]).convert("RGBA")
            ).reshape(-1, 4)
            quats_rgba = np.asarray(
                Image.open(source_chunk / meta["quats"]["files"][0]).convert("RGBA")
            ).reshape(-1, 4)
            sh0_flat_original = sh0_original.reshape(-1, 4)
            scale_codebook = np.asarray(meta["scales"]["codebook"], dtype=np.float32)
            color_codebook = np.asarray(meta["sh0"]["codebook"], dtype=np.float32)
            selected_scales = np.exp(scale_codebook[scales_rgba[selected_indices, :3]])
            selected_quaternions = _decode_sog_quaternions(quats_rgba[selected_indices])
            selected_colors = color_codebook[sh0_flat_original[selected_indices, :3]]
            selected_opacities = sh0_flat_original[selected_indices, 3].astype(np.float32) / 255.0

            for node_id, mask in masks.items():
                source_indices = np.flatnonzero(mask)
                if source_indices.size == 0:
                    continue
                local = np.fromiter(
                    (selected_lookup[int(index)] for index in source_indices),
                    dtype=np.int64,
                    count=source_indices.size,
                )
                target = extracted[node_id]
                target["positions"].append(positions[source_indices])
                target["scales"].append(selected_scales[local])
                target["orientations"].append(selected_quaternions[local])
                target["opacities"].append(selected_opacities[local])
                target["colors"].append(selected_colors[local])

        object_reports: list[dict[str, Any]] = []
        for node in articulated:
            values = extracted[node.node_id]
            if not values["positions"]:
                raise RealToSimError(f"LOD0 extraction selected no Gaussians for {node.node_id}")
            positions = np.concatenate(values["positions"]).astype(np.float32, copy=False)
            scales = np.concatenate(values["scales"]).astype(np.float32, copy=False)
            orientations = np.concatenate(values["orientations"]).astype(np.float32, copy=False)
            opacities = np.concatenate(values["opacities"]).astype(np.float32, copy=False)
            colors = np.concatenate(values["colors"]).astype(np.float32, copy=False)
            object_root = objects_root / node.node_id
            object_root.mkdir(parents=True)
            measured_ply = object_root / "measured-lod0.ply"
            scene = GaussianScene(
                source_path=lod_meta_path,
                source_sha256=sha256_file(lod_meta_path),
                positions=positions,
                scales=scales,
                orientations=orientations,
                opacities=opacities,
                sh_coefficients=colors[:, None, :],
                sh_degree=0,
            )
            write_gaussians(scene, measured_ply)
            output_meta = object_root / "meta.json"
            command = [
                "npx",
                "--yes",
                SPLAT_TRANSFORM_PACKAGE,
                "-w",
                str(measured_ply),
                str(output_meta),
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            if completed.returncode != 0 or not output_meta.is_file():
                raise RealToSimError(
                    f"failed to compress articulated splats for {node.node_id}:\n"
                    + (completed.stderr or completed.stdout or "splat-transform produced no meta.json")
                )
            object_reports.append(
                {
                    "id": node.node_id,
                    "url": f"./articulated/objects/{node.node_id}/meta.json",
                    "source_lod": 0,
                    "gaussian_count": int(scene.count),
                    "bounds": scene.bounds.to_json(),
                    "joint": asdict(node.joint),
                    "measured": True,
                    "ply": f"./articulated/objects/{node.node_id}/measured-lod0.ply",
                    "selection_method": (
                        "direct high-resolution source bounds"
                        if node.selection_mode == "bounds"
                        else "nearest labeled working-PLY reference within measured visual bounds"
                    ),
                    "selection_references": reference_reports[node.node_id],
                }
            )

        report = {
            "schema_version": ARTICULATION_ASSET_SCHEMA,
            "signature": signature,
            "source_lod_meta_sha256": sha256_file(lod_meta_path),
            "working_ply_sha256": sha256_file(working_ply),
            "selection_sha256": selection_digests,
            "background_url": "./articulated/background/lod-meta.json",
            "background_mask": (
                "lossless SOG sh0 opacity alpha=0 for measured articulated-object "
                "selections plus explicitly accepted generated-replacement apertures"
            ),
            "background_occlusions": [
                {"id": identifier, "bounds": bounds.to_json()}
                for identifier, bounds in occlusions
            ],
            "selection_method": (
                "per-node direct bounds or nearest labeled working-PLY references"
            ),
            "objects": object_reports,
            "masked_gaussians_by_lod": masked_by_lod,
            "background_occluded_gaussians_by_lod": background_occluded_by_lod,
            "converter": SPLAT_TRANSFORM_PACKAGE,
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
