"""Closed/half/open visual evidence gates for articulated segmentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .core import RealToSimError, Workspace, content_digest, sha256_file


SEGMENTATION_REVIEW_SCHEMA = 1
MINIMUM_POSE_DELTA = 1.0
SEGMENTATION_REVIEW_POSES: dict[str, float] = {
    "closed": 0.0,
    "half": 0.5,
    "open": 1.0,
}


def _review_root(workspace: Workspace) -> Path:
    return workspace.root / "evidence" / "segmentation"


def _required_nodes(workspace: Workspace) -> list[str]:
    return sorted(
        node.node_id
        for node in workspace.nodes
        if node.joint is not None and "visual-refined" in node.tags
    )


def _selected_nodes(workspace: Workspace, node_ids: Iterable[str] | None) -> list[str]:
    required = _required_nodes(workspace)
    if node_ids is None:
        return required
    selected = sorted(set(node_ids))
    unknown = sorted(set(selected) - set(required))
    if unknown:
        raise RealToSimError(
            f"segmentation review nodes are not visual-refined articulations: {unknown}"
        )
    return selected


def _evidence_path(workspace: Workspace, node_id: str, pose: str) -> Path:
    return _review_root(workspace) / "review" / node_id / f"{pose}.png"


def _part_digest(workspace: Workspace, node_id: str) -> str:
    node = workspace.node(node_id)
    return content_digest(
        {
            "node": node.to_json(),
            "selection_sha256": sha256_file(workspace.root / node.selection_file),
        }
    )


def create_segmentation_review_plan(
    workspace: Workspace,
    *,
    node_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Write the deterministic browser-review contract for current scene state."""

    nodes = _selected_nodes(workspace, node_ids)
    parts = []
    for node_id in nodes:
        node = workspace.node(node_id)
        if node.selection_mode != "stable-reference":
            raise RealToSimError(
                f"visual review requires stable-reference segmentation: {node_id}"
            )
        parts.append(
            {
                "id": node_id,
                "selection_mode": node.selection_mode,
                "part_digest": _part_digest(workspace, node_id),
                "working_positive_references": node.selected_gaussians,
                "proposal_bounds": (
                    node.selection_bounds.to_json() if node.selection_bounds else None
                ),
                "poses": [
                    {
                        "name": pose,
                        "joint_fraction": fraction,
                        "evidence": str(
                            _evidence_path(workspace, node_id, pose).relative_to(
                                workspace.root
                            )
                        ),
                    }
                    for pose, fraction in SEGMENTATION_REVIEW_POSES.items()
                ],
            }
        )
    report = {
        "schema_version": SEGMENTATION_REVIEW_SCHEMA,
        "scene_revision": workspace.state["scene_revision"],
        "scene_digest": workspace.state["logical_digest"],
        "parts": parts,
        "capture_url": "preview/index.html?segmentation-review=1",
        "capture_mode": "measured-articulation-only",
        "required_observations": {
            "closed": "measured source remains intact and generated completion is hidden",
            "half": "the selected measured part moves coherently without adjacent-scene leakage",
            "open": "the full measured part remains coherent and hidden completion is revealed",
        },
        "acceptance": {
            "machine": "PNG integrity, usable luminance, and visual-region pose deltas",
            "minimum_pose_delta": MINIMUM_POSE_DELTA,
            "semantic": "Codex or a human reviewer must inspect every closed/half/open triplet",
        },
    }
    root = _review_root(workspace)
    root.mkdir(parents=True, exist_ok=True)
    (root / "review-plan.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    workspace.trace(
        "create-segmentation-review-plan",
        {"nodes": nodes},
        {"plan": str(root / "review-plan.json"), "part_count": len(parts)},
    )
    return report


def _image_report(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    if not path.is_file():
        raise RealToSimError(f"segmentation review evidence is missing: {path}")
    try:
        with Image.open(path) as opened:
            opened.load()
            image = opened.convert("RGB")
    except OSError as exc:
        raise RealToSimError(f"invalid segmentation review PNG: {path}: {exc}") from exc
    width, height = image.size
    if width < 640 or height < 360:
        raise RealToSimError(
            f"segmentation review image is too small ({width}x{height}): {path}"
        )

    # The current experience reserves the right-most panel for controls. The
    # left 72% below the header is the rendered scene and excludes slider text,
    # so pose-change metrics cannot pass merely because the UI value changed.
    visual = image.crop((0, int(height * 0.09), int(width * 0.72), height))
    grayscale = np.asarray(
        visual.convert("L").resize((320, 180), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    luma_mean = float(np.mean(grayscale))
    luma_std = float(np.std(grayscale))
    black_fraction = float(np.mean(grayscale <= 3.0))
    white_fraction = float(np.mean(grayscale >= 252.0))
    usable = (
        4.0 <= luma_mean <= 251.0
        and luma_std >= 8.0
        and black_fraction < 0.85
        and white_fraction < 0.85
    )
    return (
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "width": width,
            "height": height,
            "visual_luma_mean": luma_mean,
            "visual_luma_std": luma_std,
            "visual_black_fraction": black_fraction,
            "visual_white_fraction": white_fraction,
            "usable": usable,
        },
        grayscale,
    )


def _load_review(workspace: Workspace) -> dict[str, Any]:
    path = _review_root(workspace) / "review.json"
    if not path.is_file():
        return {
            "schema_version": SEGMENTATION_REVIEW_SCHEMA,
            "scene_revision": workspace.state["scene_revision"],
            "scene_digest": workspace.state["logical_digest"],
            "parts": {},
        }
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != SEGMENTATION_REVIEW_SCHEMA:
        raise RealToSimError("unsupported segmentation review schema")
    if report.get("scene_digest") != workspace.state["logical_digest"]:
        return {
            "schema_version": SEGMENTATION_REVIEW_SCHEMA,
            "scene_revision": workspace.state["scene_revision"],
            "scene_digest": workspace.state["logical_digest"],
            "parts": report.get("parts", {}),
            "supersedes_scene_digest": report.get("scene_digest"),
        }
    return report


def _evaluate_node_evidence(workspace: Workspace, node_id: str) -> dict[str, Any]:
    images: dict[str, dict[str, Any]] = {}
    arrays: dict[str, np.ndarray] = {}
    for pose in SEGMENTATION_REVIEW_POSES:
        image_report, grayscale = _image_report(_evidence_path(workspace, node_id, pose))
        images[pose] = image_report
        arrays[pose] = grayscale
    deltas = {
        "closed_to_half": float(np.mean(np.abs(arrays["closed"] - arrays["half"]))),
        "half_to_open": float(np.mean(np.abs(arrays["half"] - arrays["open"]))),
    }
    machine_checks = {
        "all_images_usable": all(item["usable"] for item in images.values()),
        "closed_to_half_changes_scene": deltas["closed_to_half"] >= MINIMUM_POSE_DELTA,
        "half_to_open_changes_scene": deltas["half_to_open"] >= MINIMUM_POSE_DELTA,
    }
    return {
        "passed": all(machine_checks.values()),
        "machine_checks": machine_checks,
        "pose_deltas": deltas,
        "evidence": images,
    }


def check_segmentation_review_evidence(
    workspace: Workspace,
    *,
    node_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Evaluate all pose triplets without recording a semantic acceptance."""

    nodes = _selected_nodes(workspace, node_ids)
    parts = {
        node_id: _evaluate_node_evidence(workspace, node_id) for node_id in nodes
    }
    failed = [node_id for node_id, report in parts.items() if not report["passed"]]
    return {
        "schema_version": SEGMENTATION_REVIEW_SCHEMA,
        "scene_revision": workspace.state["scene_revision"],
        "scene_digest": workspace.state["logical_digest"],
        "capture_mode": "measured-articulation-only",
        "minimum_pose_delta": MINIMUM_POSE_DELTA,
        "parts": parts,
        "failed_nodes": failed,
        "passed": not failed,
    }


def accept_segmentation_review(
    workspace: Workspace,
    *,
    reviewer: str,
    note: str,
    node_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Accept captured pose triplets after an explicit semantic inspection."""

    reviewer = reviewer.strip()
    note = note.strip()
    if not reviewer:
        raise RealToSimError("segmentation review requires a reviewer identity")
    if len(note) < 12:
        raise RealToSimError("segmentation review note must describe the visual verdict")
    nodes = _selected_nodes(workspace, node_ids)
    evaluation = check_segmentation_review_evidence(workspace, node_ids=nodes)
    if not evaluation["passed"]:
        failed = {
            node_id: evaluation["parts"][node_id]["machine_checks"]
            for node_id in evaluation["failed_nodes"]
        }
        raise RealToSimError(
            f"segmentation pose evidence failed machine checks: {failed}"
        )
    report = _load_review(workspace)
    for node_id in nodes:
        node_evaluation = evaluation["parts"][node_id]
        report["parts"][node_id] = {
            "status": "accepted",
            "part_digest": _part_digest(workspace, node_id),
            "reviewer": reviewer,
            "note": note,
            "semantic_checks": {
                "closed_source_intact": True,
                "half_selection_coherent": True,
                "open_selection_coherent": True,
                "adjacent_scene_leakage_absent": True,
            },
            "machine_checks": node_evaluation["machine_checks"],
            "pose_deltas": node_evaluation["pose_deltas"],
            "evidence": node_evaluation["evidence"],
        }
    root = _review_root(workspace)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "review.json"
    path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    workspace.trace(
        "accept-segmentation-review",
        {"nodes": nodes, "reviewer": reviewer},
        {"review": str(path), "accepted_count": len(nodes)},
    )
    return segmentation_review_status(workspace)


def segmentation_review_status(workspace: Workspace) -> dict[str, Any]:
    """Return a fail-closed review gate tied to the current logical scene."""

    required = _required_nodes(workspace)
    report = _load_review(workspace)
    valid: list[str] = []
    stale_or_invalid: list[str] = []
    for node_id in required:
        record = report.get("parts", {}).get(node_id)
        if not record or record.get("status") != "accepted":
            continue
        if record.get("part_digest") != _part_digest(workspace, node_id):
            stale_or_invalid.append(node_id)
            continue
        evidence_valid = True
        for pose in SEGMENTATION_REVIEW_POSES:
            saved = record.get("evidence", {}).get(pose, {})
            path = _evidence_path(workspace, node_id, pose)
            if not path.is_file() or saved.get("sha256") != sha256_file(path):
                evidence_valid = False
                break
        if evidence_valid and all(record.get("machine_checks", {}).values()):
            valid.append(node_id)
        else:
            stale_or_invalid.append(node_id)
    pending = sorted(set(required) - set(valid))
    return {
        "schema_version": SEGMENTATION_REVIEW_SCHEMA,
        "scene_revision": workspace.state["scene_revision"],
        "scene_digest": workspace.state["logical_digest"],
        "required_nodes": required,
        "required_count": len(required),
        "accepted_nodes": valid,
        "accepted_count": len(valid),
        "pending_nodes": pending,
        "stale_or_invalid_nodes": stale_or_invalid,
        "passed": not pending,
        "review": str(_review_root(workspace) / "review.json"),
    }
