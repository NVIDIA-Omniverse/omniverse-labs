"""Scene graph construction and deterministic rigid/articulation verification."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from typing import Any

import numpy as np

from .core import AXES, Bounds, Collider, Joint, RealToSimError, SceneNode, Workspace
from .gaussian import GaussianScene, load_gaussians, select_bounds


def add_node(
    workspace: Workspace,
    *,
    node_id: str,
    label: str,
    role: str,
    source_indices: np.ndarray,
    scene: GaussianScene | None = None,
    collider_bounds: Bounds | None = None,
    collider_padding: float = 0.0,
    collider_confidence: float = 0.6,
    collider_provenance: str = "selection-aabb",
    collision_mode: str = "solid",
    tags: tuple[str, ...] = (),
    selection_mode: str = "stable-reference",
    selection_bounds: Bounds | None = None,
) -> SceneNode:
    scene = scene or load_gaussians(workspace.source_path)
    indices = np.unique(np.asarray(source_indices, dtype=np.uint32))
    if indices.size == 0:
        raise RealToSimError(f"node {node_id} selection is empty")
    if int(indices.max()) >= scene.count:
        raise RealToSimError(f"node {node_id} selection references an unknown source row")
    positions = scene.positions[indices]
    visual = Bounds(tuple(positions.min(axis=0)), tuple(positions.max(axis=0)))
    physical = collider_bounds or visual
    if collider_padding:
        if collider_padding < 0:
            raise RealToSimError("collider_padding cannot be negative")
        half_padding = np.full(3, collider_padding, dtype=np.float64)
        physical = Bounds(
            tuple(np.asarray(physical.minimum) - half_padding),
            tuple(np.asarray(physical.maximum) + half_padding),
        )
    selection_file = workspace.save_selection(node_id, indices)
    collider = None
    if role != "background":
        collider = Collider(
            kind="box",
            center=physical.center,
            size=physical.size,
            provenance=collider_provenance,
            confidence=collider_confidence,
            collision_mode=collision_mode,
        )
    node = SceneNode(
        node_id=node_id,
        label=label,
        role=role,
        visual_bounds=visual,
        selection_file=selection_file,
        selected_gaussians=int(indices.size),
        collider=collider,
        tags=tags,
        selection_mode=selection_mode,
        selection_bounds=selection_bounds,
    )
    workspace.put_node(node)
    workspace.trace(
        "add-node",
        {
            "id": node_id,
            "label": label,
            "role": role,
            "collider_padding": collider_padding,
            "collision_mode": collision_mode,
            "selection_mode": selection_mode,
            "selection_bounds": selection_bounds.to_json() if selection_bounds else None,
        },
        node.to_json(),
    )
    return node


def add_node_from_bounds(
    workspace: Workspace,
    *,
    node_id: str,
    label: str,
    role: str,
    bounds: Bounds,
    **kwargs: Any,
) -> SceneNode:
    scene = load_gaussians(workspace.source_path)
    return add_node(
        workspace,
        node_id=node_id,
        label=label,
        role=role,
        source_indices=select_bounds(scene, bounds),
        scene=scene,
        **kwargs,
    )


def add_node_from_render_mask(
    workspace: Workspace,
    *,
    node_id: str,
    label: str,
    role: str,
    source_indices: np.ndarray,
    **kwargs: Any,
) -> SceneNode:
    return add_node(
        workspace,
        node_id=node_id,
        label=label,
        role=role,
        source_indices=source_indices,
        **kwargs,
    )


def _contains(container: Bounds, child: Bounds, tolerance: float) -> bool:
    return bool(
        np.all(np.asarray(child.minimum) >= np.asarray(container.minimum) - tolerance)
        and np.all(np.asarray(child.maximum) <= np.asarray(container.maximum) + tolerance)
    )


def infer_support(workspace: Workspace, *, max_gap: float | None = None) -> list[dict[str, Any]]:
    nodes = workspace.nodes
    up_index = AXES[workspace.up_axis]
    scale = float(workspace.state["source"]["report"]["bounds"]["diagonal"])
    gap_limit = max_gap if max_gap is not None else max(scale * 0.025, 0.02)
    results: list[dict[str, Any]] = []
    for child in nodes:
        if child.role in {"background", "static"} or child.collider is None or child.joint is not None:
            continue
        candidates: list[tuple[float, SceneNode, str, dict[str, float]]] = []
        child_bounds = child.collider.bounds
        for parent in nodes:
            if parent.node_id == child.node_id or parent.collider is None:
                continue
            parent_bounds = parent.collider.bounds
            if _contains(parent_bounds, child_bounds, gap_limit):
                volume_ratio = float(
                    np.prod(child_bounds.size) / max(np.prod(parent_bounds.size), 1e-12)
                )
                score = 0.72 - min(volume_ratio, 0.5) * 0.2
                candidates.append((score, parent, "contained", {"volume_ratio": volume_ratio}))
                continue
            gap = child_bounds.minimum[up_index] - parent_bounds.maximum[up_index]
            overlap = child_bounds.horizontal_overlap_fraction(parent_bounds, workspace.up_axis)
            if -gap_limit <= gap <= gap_limit and overlap > 0.05:
                score = overlap * 0.8 + (1.0 - abs(gap) / gap_limit) * 0.2
                candidates.append((score, parent, "surface", {"gap": gap, "overlap": overlap}))
        if not candidates:
            results.append({"child": child.node_id, "status": "unresolved", "candidates": []})
            continue
        candidates.sort(key=lambda item: (-item[0], item[1].node_id))
        best_score, best_parent, relation, metrics = candidates[0]
        updated = replace(child, support_parent=best_parent.node_id)
        workspace.put_node(updated)
        results.append(
            {
                "child": child.node_id,
                "parent": best_parent.node_id,
                "relation": relation,
                "score": best_score,
                "metrics": metrics,
                "candidate_count": len(candidates),
            }
        )
    workspace.trace("infer-support", {"max_gap": gap_limit}, {"relations": results})
    return results


def set_support(workspace: Workspace, *, child_id: str, parent_id: str) -> SceneNode:
    child = workspace.node(child_id)
    _ = workspace.node(parent_id)
    if child_id == parent_id:
        raise RealToSimError("a node cannot support itself")
    updated = replace(child, support_parent=parent_id)
    workspace.put_node(updated)
    _assert_acyclic(workspace)
    workspace.trace("set-support", {"child": child_id, "parent": parent_id}, updated.to_json())
    return updated


def _nearest_parent_face(child: Bounds, parent: Bounds, up_axis: str) -> tuple[int, int, float]:
    child_center = np.asarray(child.center)
    parent_min = np.asarray(parent.minimum)
    parent_max = np.asarray(parent.maximum)
    parent_size = np.asarray(parent.size)
    options = []
    for axis in range(3):
        if axis == AXES[up_axis]:
            continue
        negative = abs(child_center[axis] - parent_min[axis]) / max(parent_size[axis], 1e-9)
        positive = abs(parent_max[axis] - child_center[axis]) / max(parent_size[axis], 1e-9)
        options.append((negative, axis, -1))
        options.append((positive, axis, 1))
    distance, axis, sign = min(options)
    return axis, sign, float(distance)


def fit_joint(
    workspace: Workspace,
    *,
    node_id: str,
    parent_id: str | None = None,
    kind: str = "auto",
    axis: str | None = None,
    axis_sign: int | None = None,
    origin: tuple[float, float, float] | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> Joint:
    node = workspace.node(node_id)
    if node.collider is None:
        raise RealToSimError("articulation fitting requires a collider")
    selected_parent = parent_id or node.support_parent
    if not selected_parent:
        raise RealToSimError("articulation fitting requires an explicit or inferred parent")
    parent = workspace.node(selected_parent)
    if parent.collider is None:
        raise RealToSimError("joint parent requires a collider")
    text = " ".join((node.node_id, node.label, *node.tags)).lower()
    selected_kind = kind
    diagnostics: list[str] = []
    if kind == "auto":
        if any(token in text for token in ("door", "hinge", "lid")):
            selected_kind = "revolute"
            diagnostics.append("kind=revolute from door/hinge/lid semantic cue")
        else:
            selected_kind = "prismatic"
            diagnostics.append("kind=prismatic from drawer/default articulated cue")
    if selected_kind not in {"prismatic", "revolute"}:
        raise RealToSimError("joint kind must be auto, prismatic, or revolute")
    child_bounds = node.collider.bounds
    parent_bounds = parent.collider.bounds
    face_axis, face_sign, face_distance = _nearest_parent_face(
        child_bounds, parent_bounds, workspace.up_axis
    )
    confidence = 0.55
    if selected_kind == "prismatic":
        selected_axis = axis or "XYZ"[face_axis]
        selected_sign = axis_sign or face_sign
        selected_origin = origin or child_bounds.center
        extent = child_bounds.size[AXES[selected_axis]]
        selected_lower = 0.0 if lower is None else float(lower)
        selected_upper = max(extent * 0.9, parent_bounds.size[AXES[selected_axis]] * 0.45) if upper is None else float(upper)
        diagnostics.append(
            f"opening axis inferred from nearest parent face; normalized face distance={face_distance:.4f}"
        )
        confidence += max(0.0, 0.25 - face_distance * 0.2)
    else:
        selected_axis = axis or workspace.up_axis
        selected_sign = axis_sign or 1
        selected_origin_array = np.asarray(child_bounds.center)
        horizontal = [index for index in range(3) if index != AXES[workspace.up_axis]]
        hinge_axis = max(horizontal, key=lambda index: child_bounds.size[index])
        # Place the hinge on the child edge nearest the matching parent edge.
        distance_min = abs(child_bounds.minimum[hinge_axis] - parent_bounds.minimum[hinge_axis])
        distance_max = abs(parent_bounds.maximum[hinge_axis] - child_bounds.maximum[hinge_axis])
        selected_origin_array[hinge_axis] = (
            child_bounds.minimum[hinge_axis]
            if distance_min <= distance_max
            else child_bounds.maximum[hinge_axis]
        )
        selected_origin = origin or tuple(float(item) for item in selected_origin_array)
        selected_lower = 0.0 if lower is None else float(lower)
        selected_upper = 95.0 if upper is None else float(upper)
        diagnostics.append("hinge axis follows scene up; origin placed on nearest long child edge")
        confidence += 0.12 if "door" in text else 0.02
    joint = Joint(
        kind=selected_kind,
        parent=selected_parent,
        axis=selected_axis,
        axis_sign=selected_sign,
        origin=tuple(selected_origin),
        lower=float(selected_lower),
        upper=float(selected_upper),
        confidence=min(confidence, 0.95),
        provenance="local-heuristic-candidate",
        diagnostics=tuple(diagnostics),
    )
    updated = replace(node, role="articulated", support_parent=selected_parent, joint=joint)
    workspace.put_node(updated)
    workspace.trace(
        "fit-joint",
        {
            "node": node_id,
            "parent": selected_parent,
            "kind": kind,
            "axis_override": axis,
            "origin_override": origin,
            "limits_override": [lower, upper],
        },
        asdict(joint),
    )
    return joint


def _corners(bounds: Bounds) -> np.ndarray:
    return np.asarray(
        [
            (x, y, z)
            for x in (bounds.minimum[0], bounds.maximum[0])
            for y in (bounds.minimum[1], bounds.maximum[1])
            for z in (bounds.minimum[2], bounds.maximum[2])
        ],
        dtype=np.float64,
    )


def joint_bounds(node: SceneNode, value: float) -> Bounds:
    if node.collider is None or node.joint is None:
        raise RealToSimError("joint_bounds requires an articulated node")
    joint = node.joint
    axis_index = AXES[joint.axis]
    axis_vector = np.zeros(3, dtype=np.float64)
    axis_vector[axis_index] = 1.0
    if joint.kind == "prismatic":
        return node.collider.bounds.translated(axis_vector * value * joint.axis_sign)
    angle = math.radians(value * joint.axis_sign)
    origin = np.asarray(joint.origin, dtype=np.float64)
    points = _corners(node.collider.bounds) - origin
    cross = np.cross(np.broadcast_to(axis_vector, points.shape), points)
    dot = points @ axis_vector
    rotated = (
        points * math.cos(angle)
        + cross * math.sin(angle)
        + np.outer(dot, axis_vector) * (1.0 - math.cos(angle))
        + origin
    )
    return Bounds(tuple(rotated.min(axis=0)), tuple(rotated.max(axis=0)))


def sweep_joint(
    workspace: Workspace,
    *,
    node_id: str,
    samples: int = 17,
    overlap_tolerance: float = 1e-7,
) -> dict[str, Any]:
    if samples < 3:
        raise RealToSimError("joint sweep requires at least three samples")
    node = workspace.node(node_id)
    if node.joint is None or node.collider is None:
        raise RealToSimError(f"node is not articulated: {node_id}")
    values = np.linspace(node.joint.lower, node.joint.upper, samples)
    frames = []
    forbidden_peak = 0.0
    for value in values:
        bounds = joint_bounds(node, float(value))
        collisions = []
        for other in workspace.nodes:
            if (
                other.node_id in {node.node_id, node.joint.parent}
                or other.collider is None
                or other.collider.collision_mode == "support"
            ):
                continue
            overlap = bounds.overlap_volume(other.collider.bounds)
            if overlap > overlap_tolerance:
                collisions.append({"node": other.node_id, "overlap_volume": overlap})
                forbidden_peak = max(forbidden_peak, overlap)
        frames.append(
            {
                "value": float(value),
                "bounds": bounds.to_json(),
                "forbidden_collisions": collisions,
            }
        )
    gates = {
        "limits_ordered": node.joint.lower < node.joint.upper,
        "all_samples_finite": all(
            all(math.isfinite(item) for item in (*frame["bounds"]["min"], *frame["bounds"]["max"]))
            for frame in frames
        ),
        "no_forbidden_overlap": forbidden_peak <= overlap_tolerance,
        "parent_exists": any(item.node_id == node.joint.parent for item in workspace.nodes),
    }
    report = {
        "schema_version": 1,
        "node": node.node_id,
        "joint": asdict(node.joint),
        "samples": frames,
        "gates": gates,
        "passed": all(gates.values()),
        "peak_forbidden_overlap_volume": forbidden_peak,
        "physics_scope": "deterministic conservative AABB sweep; not PhysX",
    }
    output_dir = workspace.root / "evidence" / "sweeps" / node.node_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sweep.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    workspace.trace(
        "sweep-joint",
        {"node": node_id, "samples": samples, "overlap_tolerance": overlap_tolerance},
        {"report": str(output_dir / "sweep.json"), "gates": gates},
    )
    return report


def settle_scene(workspace: Workspace) -> dict[str, Any]:
    up_index = AXES[workspace.up_axis]
    results = []
    for node in workspace.nodes:
        if node.role != "movable" or node.collider is None or node.joint is not None:
            continue
        body = node.collider.bounds
        candidates = []
        for support in workspace.nodes:
            if support.node_id == node.node_id or support.collider is None:
                continue
            overlap = body.horizontal_overlap_fraction(support.collider.bounds, workspace.up_axis)
            support_top = support.collider.bounds.maximum[up_index]
            if overlap > 0.05 and support_top <= body.minimum[up_index] + body.size[up_index]:
                candidates.append((support_top, overlap, support))
        if not candidates:
            results.append({"node": node.node_id, "status": "unresolved"})
            continue
        support_top, overlap, support = max(candidates, key=lambda item: (item[0], item[1]))
        target_bottom = support_top
        displacement = target_bottom - body.minimum[up_index]
        results.append(
            {
                "node": node.node_id,
                "support": support.node_id,
                "translation": [
                    displacement if index == up_index else 0.0 for index in range(3)
                ],
                "horizontal_overlap_fraction": overlap,
                "status": "settled",
            }
        )
    report = {
        "schema_version": 1,
        "results": results,
        "passed": all(item["status"] == "settled" for item in results),
        "physics_scope": "one-step gravity projection onto conservative AABB supports; not PhysX",
    }
    output = workspace.root / "evidence" / "settle.json"
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    workspace.trace("settle", {}, {"report": str(output), "passed": report["passed"]})
    return report


def push_test(
    workspace: Workspace,
    *,
    node_id: str,
    delta: tuple[float, float, float],
    overlap_tolerance: float = 1e-7,
) -> dict[str, Any]:
    node = workspace.node(node_id)
    if node.collider is None:
        raise RealToSimError("push test requires a collider")
    moved = node.collider.bounds.translated(delta)
    collisions = []
    for other in workspace.nodes:
        if other.node_id in {node.node_id, node.support_parent} or other.collider is None:
            continue
        overlap = moved.overlap_volume(other.collider.bounds)
        if overlap > overlap_tolerance:
            collisions.append({"node": other.node_id, "overlap_volume": overlap})
    report = {
        "node": node_id,
        "delta": list(delta),
        "result_bounds": moved.to_json(),
        "collisions": collisions,
        "passed": not collisions,
        "physics_scope": "kinematic conservative AABB push; not PhysX",
    }
    output = workspace.root / "evidence" / f"push-{node_id}.json"
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    workspace.trace("push-test", {"node": node_id, "delta": list(delta)}, report)
    return report


def _assert_acyclic(workspace: Workspace) -> None:
    parent = {node.node_id: node.support_parent for node in workspace.nodes}
    for start in parent:
        seen: set[str] = set()
        current: str | None = start
        while current is not None:
            if current in seen:
                raise RealToSimError(f"support graph contains a cycle through {current}")
            seen.add(current)
            current = parent.get(current)


def _collider_visual_score(node: SceneNode) -> float:
    if node.collider is None:
        return 1.0 if node.role == "background" else 0.0
    visual = node.visual_bounds
    physical = node.collider.bounds
    if node.selection_mode == "bounds":
        overlap = np.asarray(visual.overlap(physical), dtype=np.float64)
        visual_size = np.asarray(visual.size, dtype=np.float64)
        per_axis = overlap / np.maximum(visual_size, 1e-12)
        # A panel selected from reconstructed splats can have substantial noisy
        # depth. Registration is meaningful in its two tangential dimensions;
        # requiring the physics proxy to reproduce that depth would turn a thin
        # door into an implausibly thick collision block.
        tangential = np.sort(per_axis)[-2:]
        return float(np.prod(tangential))
    intersection = visual.overlap_volume(physical)
    visual_volume = max(float(np.prod(visual.size)), 1e-12)
    return min(1.0, intersection / visual_volume)


def verify(workspace: Workspace, *, run_sweeps: bool = True) -> dict[str, Any]:
    workspace.verify_source()
    scene = load_gaussians(workspace.source_path)
    errors = []
    node_reports = []
    # A sparse or partially occluded measured front can have a poor AABB overlap
    # score even when its deterministic closed/half/open evidence is accepted.
    # Keep the numeric score as a diagnostic, but let that explicit review attest
    # to the profile-to-proxy registration instead of rejecting a reviewed front.
    from .segmentation_review import segmentation_review_status

    segmentation_reviews = segmentation_review_status(workspace)
    reviewed_nodes = set(segmentation_reviews["accepted_nodes"])
    ids = {node.node_id for node in workspace.nodes}
    try:
        _assert_acyclic(workspace)
        support_acyclic = True
    except RealToSimError as exc:
        support_acyclic = False
        errors.append(str(exc))
    for node in workspace.nodes:
        selection = workspace.load_selection(node)
        selection_ok = (
            selection.ndim == 1
            and selection.size == node.selected_gaussians
            and (selection.size == 0 or int(selection.max()) < scene.count)
        )
        support_ok = node.support_parent is None or node.support_parent in ids
        joint_ok = (
            node.joint is None
            or (
                node.joint.parent in ids
                and node.role == "articulated"
                and node.support_parent == node.joint.parent
            )
        )
        link_score = _collider_visual_score(node)
        link_ok = link_score >= 0.5 or node.node_id in reviewed_nodes
        collider_ok = node.role == "background" or node.collider is not None
        node_report = {
            "id": node.node_id,
            "selection_ok": selection_ok,
            "support_ok": support_ok,
            "joint_ok": joint_ok,
            "collider_ok": collider_ok,
            "visual_collider_coverage": link_score,
            "visual_collider_link_ok": link_ok,
            "visual_collider_link_mode": (
                "aabb-coverage" if link_score >= 0.5 else "accepted-visual-review"
            ),
        }
        node_reports.append(node_report)
        if not all(
            node_report[key]
            for key in ("selection_ok", "support_ok", "joint_ok", "collider_ok", "visual_collider_link_ok")
        ):
            errors.append(f"node gate failed: {node.node_id}")
    sweep_reports = []
    if run_sweeps:
        for node in workspace.nodes:
            if node.joint is not None:
                sweep_reports.append(sweep_joint(workspace, node_id=node.node_id))
    from .completion import completion_report

    completions = completion_report(workspace)
    accepted_by_node: dict[str, list[dict[str, Any]]] = {}
    for candidate in completions["candidates"]:
        if candidate["status"] == "accepted":
            accepted_by_node.setdefault(candidate["node"], []).append(candidate)
    completion_links_ok = True
    for node_id, candidates in accepted_by_node.items():
        if len(candidates) != 1:
            completion_links_ok = False
            errors.append(f"node has multiple accepted completions: {node_id}")
            continue
        node = workspace.node(node_id)
        expected = f"accepted-completion:{candidates[0]['id']}"
        if node.collider is None or node.collider.provenance != expected:
            completion_links_ok = False
            errors.append(f"accepted completion is not linked to collider: {candidates[0]['id']}")
    gates = {
        "immutable_source": True,
        "scene_has_nodes": bool(workspace.nodes),
        "support_graph_acyclic": support_acyclic,
        "all_node_contracts": all(
            all(
                report[key]
                for key in (
                    "selection_ok",
                    "support_ok",
                    "joint_ok",
                    "collider_ok",
                    "visual_collider_link_ok",
                )
            )
            for report in node_reports
        ),
        "all_joint_sweeps": all(report["passed"] for report in sweep_reports),
        "completion_assets_valid": completions["all_assets_valid"],
        "accepted_completions_linked": completion_links_ok,
        "visual_segmentation_review": segmentation_reviews["passed"],
    }
    if not segmentation_reviews["passed"]:
        errors.append(
            "visual segmentation review is pending for: "
            + ", ".join(segmentation_reviews["pending_nodes"])
        )
    report = {
        "schema_version": 1,
        "scene_revision": workspace.state["scene_revision"],
        "scene_digest": workspace.state["logical_digest"],
        "gates": gates,
        "passed": all(gates.values()),
        "errors": errors,
        "nodes": node_reports,
        "joint_sweeps": [
            {
                "node": item["node"],
                "passed": item["passed"],
                "gates": item["gates"],
                "peak_forbidden_overlap_volume": item["peak_forbidden_overlap_volume"],
            }
            for item in sweep_reports
        ],
        "completions": completions,
        "segmentation_reviews": segmentation_reviews,
        "continuous_scores": {
            "mean_visual_collider_coverage": float(
                np.mean([item["visual_collider_coverage"] for item in node_reports])
            )
            if node_reports
            else 0.0,
        },
        "verification_scope": {
            "local": [
                "source integrity",
                "selection provenance",
                "support graph",
                "visual-collider registration",
                "conservative AABB settle/push/sweep",
                "closed/half/open visual segmentation evidence",
            ],
            "requires_external_simulator": [
                "PhysX contact fidelity",
                "robot task success",
                "high-speed dynamics",
                "deformables and fluids",
            ],
        },
    }
    output_dir = workspace.root / "evidence" / "verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "report.json"
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    workspace.trace("verify", {"run_sweeps": run_sweeps}, {"report": str(output), "gates": gates})
    return report
