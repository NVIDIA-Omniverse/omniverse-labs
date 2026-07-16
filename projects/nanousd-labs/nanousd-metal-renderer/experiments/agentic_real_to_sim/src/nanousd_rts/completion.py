"""Hidden-geometry completion candidates with explicit generated provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .core import AXES, Bounds, Collider, RealToSimError, Workspace, sha256_file
from .gaussian import write_box_gaussians
from .sim import sweep_joint


def _candidate_bounds(workspace: Workspace, node_id: str, factor: float) -> Bounds:
    node = workspace.node(node_id)
    if node.collider is None or node.joint is None:
        raise RealToSimError("hidden completion requires an articulated node and fitted joint")
    parent = workspace.node(node.joint.parent)
    if parent.collider is None:
        raise RealToSimError("hidden completion requires a physical parent")
    child = node.collider.bounds
    parent_bounds = parent.collider.bounds
    minimum = np.asarray(child.minimum, dtype=np.float64)
    maximum = np.asarray(child.maximum, dtype=np.float64)
    if node.joint.kind == "prismatic":
        axis = AXES[node.joint.axis]
        depth = min(
            parent_bounds.size[axis] * factor,
            parent_bounds.size[axis],
        )
        if node.joint.axis_sign < 0:
            minimum[axis] = child.minimum[axis]
            maximum[axis] = min(parent_bounds.maximum[axis], minimum[axis] + depth)
        else:
            maximum[axis] = child.maximum[axis]
            minimum[axis] = max(parent_bounds.minimum[axis], maximum[axis] - depth)
    else:
        # A scanned door usually exposes only its front. Complete thickness
        # along the smallest horizontal dimension while keeping the hinge edge.
        horizontal = [index for index in range(3) if index != AXES[workspace.up_axis]]
        axis = min(horizontal, key=lambda index: child.size[index])
        thickness = min(parent_bounds.size[axis] * 0.12 * factor, parent_bounds.size[axis])
        center = (minimum[axis] + maximum[axis]) * 0.5
        minimum[axis] = center - thickness * 0.5
        maximum[axis] = center + thickness * 0.5
    return Bounds(tuple(minimum), tuple(maximum))


def _inside_fraction(inner: Bounds, outer: Bounds) -> float:
    intersection = inner.overlap_volume(outer)
    return min(1.0, intersection / max(float(np.prod(inner.size)), 1e-12))


def _union_bounds(first: Bounds, second: Bounds) -> Bounds:
    return Bounds(
        tuple(np.minimum(first.minimum, second.minimum)),
        tuple(np.maximum(first.maximum, second.maximum)),
    )


def propose_hidden_interiors(
    workspace: Workspace,
    *,
    node_id: str,
    factors: Iterable[float] = (0.75, 0.9, 1.0),
    gaussian_count: int = 600,
) -> list[dict[str, Any]]:
    factors = tuple(float(item) for item in factors)
    node = workspace.node(node_id)
    if node.joint is None:
        raise RealToSimError("fit a joint before proposing hidden interiors")
    parent = workspace.node(node.joint.parent)
    if parent.collider is None:
        raise RealToSimError("completion parent has no collider")
    output_dir = workspace.root / "generated" / "completions" / node_id
    records = []
    for index, factor in enumerate(factors):
        if not 0.1 <= factor <= 1.5:
            raise RealToSimError("completion factors must be in [0.1, 1.5]")
        bounds = _candidate_bounds(workspace, node_id, factor)
        completion_id = f"{node_id}.interior.{index:02d}"
        asset = write_box_gaussians(
            output_dir / f"{completion_id}.ply",
            bounds,
            count=gaussian_count,
            seed=int(hashlib.sha256(completion_id.encode()).hexdigest()[:8], 16),
        )
        inside = _inside_fraction(bounds, parent.collider.bounds)
        size_ratio = float(
            np.prod(bounds.size) / max(np.prod(parent.collider.bounds.size), 1e-12)
        )
        score = inside * 0.75 + (1.0 - min(abs(factor - 0.9), 0.9) / 0.9) * 0.2
        score += max(0.0, 0.05 - size_ratio * 0.02)
        record = {
            "id": completion_id,
            "node": node_id,
            "status": "candidate",
            "kind": "generated-box-surface-gaussians",
            "bounds": bounds.to_json(),
            "asset": str(asset.relative_to(workspace.root)),
            "asset_sha256": sha256_file(asset),
            "generated_gaussians": gaussian_count,
            "factor": factor,
            "confidence": min(score, 0.95),
            "evaluation": {
                "inside_parent_fraction": inside,
                "parent_volume_ratio": size_ratio,
                "score": score,
            },
            "provenance": {
                "source": "geometric completion prior",
                "measured": False,
                "generator": "nanousd-rts box-surface v1",
                "requires_review": True,
            },
        }
        workspace.put_completion(record)
        records.append(record)
    records.sort(key=lambda item: (-item["evaluation"]["score"], item["id"]))
    workspace.trace(
        "propose-hidden-interiors",
        {"node": node_id, "factors": list(factors), "gaussian_count": gaussian_count},
        {"candidates": records},
    )
    return records


def accept_completion(workspace: Workspace, *, completion_id: str) -> dict[str, Any]:
    try:
        candidate = next(item for item in workspace.completions if item["id"] == completion_id)
    except StopIteration as exc:
        raise RealToSimError(f"unknown completion candidate: {completion_id}") from exc
    node = workspace.node(candidate["node"])
    if node.collider is None:
        raise RealToSimError("completion target has no collider")
    asset = workspace.root / candidate["asset"]
    if not asset.is_file() or sha256_file(asset) != candidate["asset_sha256"]:
        raise RealToSimError("completion asset is missing or its checksum changed")
    generated_bounds = Bounds.from_json(candidate["bounds"])
    # Completion augments the measured object; it must never replace or shrink the
    # already registered physical proxy. Use a conservative union so accepted
    # generated surfaces cannot break measured visual-to-collider coverage.
    bounds = _union_bounds(node.collider.bounds, generated_bounds)
    original = node
    updated = replace(
        node,
        collider=Collider(
            kind="box",
            center=bounds.center,
            size=bounds.size,
            rotation_wxyz=node.collider.rotation_wxyz,
            provenance=f"accepted-completion:{completion_id}",
            confidence=float(candidate["confidence"]),
            collision_mode=node.collider.collision_mode,
        ),
    )
    workspace.put_node(updated)
    sweep = sweep_joint(workspace, node_id=node.node_id) if node.joint else {"passed": True}
    if not sweep["passed"]:
        workspace.put_node(original)
        raise RealToSimError("completion candidate failed the articulation sweep and was not accepted")
    for record in workspace.completions:
        if record["node"] != node.node_id:
            continue
        changed = dict(record)
        changed["status"] = "accepted" if record["id"] == completion_id else "rejected"
        changed["acceptance"] = {
            "articulation_sweep_passed": True,
            "selected_by": "explicit agent action",
        }
        workspace.put_completion(changed)
    accepted = next(item for item in workspace.completions if item["id"] == completion_id)
    workspace.trace(
        "accept-completion",
        {"completion": completion_id},
        {"candidate": accepted, "collider": updated.to_json()["collider"]},
    )
    return accepted


def completion_report(workspace: Workspace) -> dict[str, Any]:
    def artifact_validity(
        descriptor: Any,
        *,
        role: str,
        artifact_kind: str,
    ) -> dict[str, Any]:
        if not isinstance(descriptor, dict):
            return {
                "role": role,
                "artifact_kind": artifact_kind,
                "asset": None,
                "valid": False,
            }
        relative = descriptor.get("path")
        expected = descriptor.get("sha256")
        path = workspace.root / relative if isinstance(relative, str) else None
        return {
            "role": role,
            "artifact_kind": artifact_kind,
            "asset": relative,
            "valid": bool(
                path is not None
                and path.is_file()
                and isinstance(expected, str)
                and sha256_file(path) == expected
            ),
        }

    records = []
    for candidate in workspace.completions:
        assets = candidate.get("assets") or [
            {
                "role": "completion",
                "asset": candidate["asset"],
                "asset_sha256": candidate["asset_sha256"],
            }
        ]
        asset_validity = []
        for record in assets:
            asset = workspace.root / record["asset"]
            asset_validity.append(
                {
                    "role": record.get("role", "completion"),
                    "asset": record["asset"],
                    "valid": (
                        asset.is_file()
                        and sha256_file(asset) == record["asset_sha256"]
                    ),
                }
            )
        mesh_asset_validity = []
        for mesh_asset in candidate.get("mesh_assets", []):
            role = mesh_asset.get("role", "mesh-completion")
            for key in (
                "manifest",
                "mesh",
                "material",
                "associations",
                "material_request",
            ):
                mesh_asset_validity.append(
                    artifact_validity(
                        mesh_asset.get(key),
                        role=role,
                        artifact_kind=key,
                    )
                )
            for name, descriptor in mesh_asset.get("pbr_maps", {}).items():
                mesh_asset_validity.append(
                    artifact_validity(
                        descriptor,
                        role=role,
                        artifact_kind=f"pbr:{name}",
                    )
                )
        if candidate.get("mesh_bundle_manifest") is not None:
            mesh_asset_validity.append(
                artifact_validity(
                    candidate["mesh_bundle_manifest"],
                    role="bundle",
                    artifact_kind="mesh_bundle_manifest",
                )
            )
        all_valid = all(item["valid"] for item in asset_validity + mesh_asset_validity)
        records.append(
            {
                **candidate,
                "asset_valid": all_valid,
                "asset_validity": asset_validity,
                "mesh_asset_validity": mesh_asset_validity,
            }
        )
    return {
        "schema_version": 1,
        "candidates": records,
        "accepted": [item["id"] for item in records if item["status"] == "accepted"],
        "all_assets_valid": all(item["asset_valid"] for item in records),
    }
