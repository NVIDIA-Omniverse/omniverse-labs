"""Deterministic agent-plan executor and end-to-end DRAWER-style demo."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .collision import voxelize
from .completion import accept_completion, propose_hidden_interiors
from .core import Bounds, RealToSimError, Workspace
from .gaussian import Camera, ingest, make_drawer_fixture, render, select_render_mask
from .preview import write_preview
from .sim import (
    add_node,
    add_node_from_bounds,
    fit_joint,
    infer_support,
    push_test,
    set_support,
    settle_scene,
    sweep_joint,
    verify,
)
from .usda import compile_usda, render_usda


TOOL_CATALOG = {
    "add-node-bounds": "Select immutable Gaussian rows in an AABB and bind them to a physical box.",
    "add-node-render-mask": "Resolve a pixel mask through the stable Gaussian ID AOV.",
    "infer-support": "Build containment/surface support candidates from registered colliders.",
    "set-support": "Author one explicit support edge.",
    "fit-joint": "Create a reviewable prismatic/revolute candidate with confidence and diagnostics.",
    "propose-hidden-interiors": "Generate multiple clearly labeled, non-measured Gaussian completion candidates.",
    "accept-completion": "Promote one completion only after its collider passes the articulation sweep.",
    "sweep-joint": "Sweep a joint through limits with fail-closed forbidden-overlap gates.",
    "settle": "Project movable AABBs onto supports under gravity.",
    "push-test": "Apply a deterministic kinematic push and report collisions.",
    "render": "Render Gaussian RGB/depth/normal/stable-ID evidence with Metal RT.",
    "voxelize": "Generate voxel occupancy and a collision GLB from the full scene or one stable-ID selection.",
    "compile-usda": "Compile physical proxies, support links, joints, and Gaussian provenance.",
    "render-usda": "Load and render the compiled proxy scene through NanoUSD Metal.",
    "verify": "Run immutable-source, graph, registration, and articulation hard gates.",
    "preview": "Write the dependency-free interactive joint/scene inspection page.",
}


def _bounds(action: dict[str, Any]) -> Bounds:
    if "bounds" in action:
        return Bounds.from_json(action["bounds"])
    return Bounds(tuple(action["min"]), tuple(action["max"]))


def execute_action(workspace: Workspace, action: dict[str, Any]) -> dict[str, Any]:
    tool = action.get("tool")
    if tool not in TOOL_CATALOG:
        raise RealToSimError(f"unknown or unauthorized plan tool: {tool}")
    if tool == "add-node-bounds":
        node = add_node_from_bounds(
            workspace,
            node_id=action["id"],
            label=action.get("label", action["id"]),
            role=action["role"],
            bounds=_bounds(action),
            collider_bounds=Bounds.from_json(action["collider_bounds"]) if action.get("collider_bounds") else None,
            collider_padding=float(action.get("collider_padding", 0.0)),
            collider_confidence=float(action.get("collider_confidence", 0.6)),
            collider_provenance=action.get("collider_provenance", "selection-aabb"),
            collision_mode=action.get("collision_mode", "solid"),
            tags=tuple(action.get("tags", ())),
        )
        return node.to_json()
    if tool == "add-node-render-mask":
        indices = select_render_mask(Path(action["id_aov"]), Path(action["mask"]))
        node = add_node(
            workspace,
            node_id=action["id"],
            label=action.get("label", action["id"]),
            role=action["role"],
            source_indices=indices,
            collider_bounds=Bounds.from_json(action["collider_bounds"]) if action.get("collider_bounds") else None,
            collider_padding=float(action.get("collider_padding", 0.0)),
            collider_confidence=float(action.get("collider_confidence", 0.6)),
            collider_provenance=action.get("collider_provenance", "stable-id-render-mask"),
            collision_mode=action.get("collision_mode", "solid"),
            tags=tuple(action.get("tags", ())),
        )
        return node.to_json()
    if tool == "infer-support":
        return {"relations": infer_support(workspace, max_gap=action.get("max_gap"))}
    if tool == "set-support":
        return set_support(workspace, child_id=action["child"], parent_id=action["parent"]).to_json()
    if tool == "fit-joint":
        joint = fit_joint(
            workspace,
            node_id=action["node"],
            parent_id=action.get("parent"),
            kind=action.get("kind", "auto"),
            axis=action.get("axis"),
            axis_sign=action.get("axis_sign"),
            origin=tuple(action["origin"]) if action.get("origin") else None,
            lower=action.get("lower"),
            upper=action.get("upper"),
        )
        return asdict(joint)
    if tool == "propose-hidden-interiors":
        return {
            "candidates": propose_hidden_interiors(
                workspace,
                node_id=action["node"],
                factors=tuple(action.get("factors", (0.75, 0.9, 1.0))),
                gaussian_count=int(action.get("gaussian_count", 600)),
            )
        }
    if tool == "accept-completion":
        return accept_completion(workspace, completion_id=action["completion"])
    if tool == "sweep-joint":
        return sweep_joint(workspace, node_id=action["node"], samples=int(action.get("samples", 17)))
    if tool == "settle":
        return settle_scene(workspace)
    if tool == "push-test":
        return push_test(
            workspace,
            node_id=action["node"],
            delta=tuple(float(item) for item in action["delta"]),
        )
    if tool == "render":
        camera = Camera.from_json(action["camera"], up_axis=workspace.up_axis) if action.get("camera") else None
        return render(
            workspace,
            name=action.get("name", "agent"),
            camera=camera,
            width=int(action.get("width", 960)),
            height=int(action.get("height", 540)),
            k=int(action.get("k", 16)),
        )
    if tool == "voxelize":
        return voxelize(
            workspace,
            node_id=action.get("node"),
            voxel_size=float(action.get("voxel_size", 0.05)),
            opacity_threshold=float(action.get("opacity_threshold", 0.1)),
            mesh_shape=action.get("mesh_shape", "faces"),
            external_fill=action.get("external_fill"),
            floor_fill=action.get("floor_fill"),
            carve=tuple(action["carve"]) if action.get("carve") else None,
            seed=tuple(action["seed"]) if action.get("seed") else None,
        )
    if tool == "compile-usda":
        return compile_usda(workspace, output=Path(action["output"]) if action.get("output") else None)
    if tool == "render-usda":
        return render_usda(
            workspace,
            usda_path=Path(action["usda"]) if action.get("usda") else None,
            width=int(action.get("width", 640)),
            height=int(action.get("height", 360)),
        )
    if tool == "verify":
        return verify(workspace, run_sweeps=bool(action.get("run_sweeps", True)))
    if tool == "preview":
        return {"html": str(write_preview(workspace, output=Path(action["output"]) if action.get("output") else None))}
    raise AssertionError(tool)


def run_plan(workspace: Workspace, plan_path: Path) -> dict[str, Any]:
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1 or not isinstance(plan.get("actions"), list):
        raise RealToSimError("agent plan must contain schema_version=1 and an actions array")
    results = []
    for index, action in enumerate(plan["actions"]):
        if not isinstance(action, dict):
            raise RealToSimError(f"actions[{index}] must be an object")
        try:
            result = execute_action(workspace, action)
            results.append({"index": index, "tool": action.get("tool"), "status": "ok", "result": result})
        except Exception as exc:
            failure = {
                "index": index,
                "tool": action.get("tool"),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            results.append(failure)
            report = {
                "schema_version": 1,
                "plan": str(plan_path),
                "status": "failed",
                "failed_action": index,
                "steps": results,
            }
            output = workspace.root / "trace" / "plan-result.json"
            output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
            workspace.trace("run-plan", {"plan": str(plan_path)}, failure)
            raise RealToSimError(f"agent plan failed closed at action {index}: {failure['error']}") from exc
    report = {
        "schema_version": 1,
        "plan": str(plan_path),
        "status": "passed",
        "steps": results,
    }
    output = workspace.root / "trace" / "plan-result.json"
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    workspace.trace("run-plan", {"plan": str(plan_path)}, {"status": "passed", "steps": len(results)})
    return report


def build_demo(workspace_path: Path, *, render_gaussians: bool = True) -> dict[str, Any]:
    workspace_path = Path(workspace_path).resolve()
    fixture = workspace_path.parent / f"{workspace_path.name}-source" / "drawer-room.ply"
    make_drawer_fixture(fixture)
    workspace = ingest(fixture, workspace_path, replace=True, up_axis="Y")
    ranges = {
        "floor": np.arange(0, 900, dtype=np.uint32),
        "cabinet": np.arange(900, 2100, dtype=np.uint32),
        "drawer": np.arange(2100, 2750, dtype=np.uint32),
        "door": np.arange(2750, 3400, dtype=np.uint32),
        "obstacle": np.arange(3400, 3750, dtype=np.uint32),
    }
    scene = None
    add_node(
        workspace,
        node_id="floor",
        label="floor",
        role="static",
        source_indices=ranges["floor"],
        scene=scene,
        collision_mode="support",
        collider_confidence=0.98,
        collider_provenance="synthetic-ground-truth",
    )
    add_node(
        workspace,
        node_id="cabinet",
        label="cabinet shell",
        role="static",
        source_indices=ranges["cabinet"],
        collision_mode="shell",
        collider_confidence=0.96,
        collider_provenance="synthetic-ground-truth",
    )
    add_node(
        workspace,
        node_id="drawer",
        label="lower drawer",
        role="movable",
        source_indices=ranges["drawer"],
        collider_confidence=0.95,
        collider_provenance="synthetic-ground-truth",
        tags=("drawer", "handle-bearing"),
    )
    add_node(
        workspace,
        node_id="door",
        label="left cabinet door",
        role="movable",
        source_indices=ranges["door"],
        collider_confidence=0.95,
        collider_provenance="synthetic-ground-truth",
        tags=("door", "hinged"),
    )
    add_node(
        workspace,
        node_id="obstacle",
        label="nearby obstacle",
        role="static",
        source_indices=ranges["obstacle"],
        collider_confidence=0.95,
        collider_provenance="synthetic-ground-truth",
    )
    set_support(workspace, child_id="cabinet", parent_id="floor")
    set_support(workspace, child_id="drawer", parent_id="cabinet")
    set_support(workspace, child_id="door", parent_id="cabinet")
    set_support(workspace, child_id="obstacle", parent_id="floor")
    fit_joint(workspace, node_id="drawer", parent_id="cabinet", kind="auto")
    fit_joint(workspace, node_id="door", parent_id="cabinet", kind="auto", upper=105.0)
    drawer_candidates = propose_hidden_interiors(workspace, node_id="drawer")
    door_candidates = propose_hidden_interiors(workspace, node_id="door")
    accept_completion(workspace, completion_id=drawer_candidates[0]["id"])
    accept_completion(workspace, completion_id=door_candidates[0]["id"])
    if render_gaussians:
        render(workspace, name="demo", width=800, height=500)
    compile_manifest = compile_usda(workspace)
    usda_render = render_usda(workspace, width=800, height=500)
    verification = verify(workspace)
    preview = write_preview(workspace)
    return {
        "workspace": str(workspace.root),
        "source": str(fixture),
        "compile": compile_manifest,
        "usda_render": usda_render,
        "verification": verification,
        "preview": str(preview),
    }
