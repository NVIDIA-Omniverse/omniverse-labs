"""Compile the registered Gaussian/physical scene contract into USDA."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .core import AXES, RealToSimError, SceneNode, Workspace, sha256_file
from .gaussian import _load_renderer_binding


BOX_POINTS = (
    (-0.5, -0.5, -0.5),
    (0.5, -0.5, -0.5),
    (-0.5, 0.5, -0.5),
    (0.5, 0.5, -0.5),
    (-0.5, -0.5, 0.5),
    (0.5, -0.5, 0.5),
    (-0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5),
)
BOX_FACE_COUNTS = (4, 4, 4, 4, 4, 4)
BOX_FACE_INDICES = (0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3)
COLORS = {
    "background": (0.18, 0.18, 0.2),
    "static": (0.34, 0.38, 0.44),
    "movable": (0.12, 0.48, 0.86),
    "articulated": (0.92, 0.38, 0.08),
}


def _tuple(values: tuple[float, ...] | list[float]) -> str:
    return "(" + ", ".join(f"{float(value):.9g}" for value in values) + ")"


def _array(values: tuple[Any, ...] | list[Any]) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"


def _usd_string(value: str) -> str:
    return json.dumps(value)


def _asset_reference(path: Path, *, relative_to: Path) -> str:
    return Path(
        os.path.relpath(Path(path).resolve(), start=Path(relative_to).resolve())
    ).as_posix()


def _body_schemas(node: SceneNode, joint_parents: set[str]) -> tuple[list[str], bool]:
    if node.role in {"movable", "articulated"} or node.node_id in joint_parents:
        return ["PhysicsRigidBodyAPI", "PhysicsMassAPI"], node.role == "static"
    return [], False


def _node_block(
    node: SceneNode,
    *,
    joint_parents: set[str],
    source_sha256: str,
    accepted_completion: dict[str, Any] | None,
    source_asset: str,
    selection_asset: str,
    completion_asset: str | None,
) -> list[str]:
    if node.collider is None:
        center = node.visual_bounds.center
        size = node.visual_bounds.size
    else:
        center = node.collider.center
        size = node.collider.size
    schemas, kinematic = _body_schemas(node, joint_parents)
    schema_text = (
        " (\n            prepend apiSchemas = "
        + _array([_usd_string(item) for item in schemas])
        + "\n        )"
        if schemas
        else ""
    )
    color = COLORS[node.role]
    lines = [
        f'        def Xform "{node.node_id}"{schema_text}',
        "        {",
        f"            custom string nanousdRts:label = {_usd_string(node.label)}",
        f"            custom token nanousdRts:role = {_usd_string(node.role)}",
        f"            custom asset nanousdRts:gaussianSource = @{source_asset}@",
        f"            custom string nanousdRts:sourceSha256 = {_usd_string(source_sha256)}",
        f"            custom asset nanousdRts:selection = @{selection_asset}@",
        f"            custom int nanousdRts:selectedGaussians = {node.selected_gaussians}",
        f"            double3 xformOp:translate = {_tuple(center)}",
        "            quatd xformOp:orient = (1, 0, 0, 0)",
        "            double3 xformOp:scale = (1, 1, 1)",
        '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]',
    ]
    if schemas:
        lines.extend(
            [
                f"            bool physics:kinematicEnabled = {str(kinematic).lower()}",
                "            bool physics:rigidBodyEnabled = true",
                "            float physics:mass = 1",
            ]
        )
    if node.support_parent:
        lines.append(
            f"            custom rel nanousdRts:supportParent = </World/Nodes/{node.support_parent}>"
        )
    if accepted_completion:
        if completion_asset is None:
            raise RealToSimError(
                f"accepted completion for {node.node_id} has no portable asset path"
            )
        lines.extend(
            [
                f"            custom asset nanousdRts:hiddenCompletion = @{completion_asset}@",
                f"            custom string nanousdRts:hiddenCompletionSha256 = {_usd_string(accepted_completion['asset_sha256'])}",
                f"            custom float nanousdRts:hiddenCompletionConfidence = {float(accepted_completion['confidence']):.9g}",
                "            custom bool nanousdRts:hiddenCompletionMeasured = false",
            ]
        )
    if node.collider is not None:
        lines.extend(
            [
                "",
                '            def Mesh "Collider" (',
                '                prepend apiSchemas = ["PhysicsCollisionAPI"]',
                "            )",
                "            {",
                f"                custom string nanousdRts:colliderProvenance = {_usd_string(node.collider.provenance)}",
                f"                custom float nanousdRts:colliderConfidence = {node.collider.confidence:.9g}",
                f"                custom token nanousdRts:collisionMode = {_usd_string(node.collider.collision_mode)}",
                "                bool physics:collisionEnabled = true",
                f"                float3[] extent = [{_tuple((-0.5, -0.5, -0.5))}, {_tuple((0.5, 0.5, 0.5))}]",
                f"                point3f[] points = [{', '.join(_tuple(point) for point in BOX_POINTS)}]",
                f"                int[] faceVertexCounts = {_array(list(BOX_FACE_COUNTS))}",
                f"                int[] faceVertexIndices = {_array(list(BOX_FACE_INDICES))}",
                f"                color3f[] primvars:displayColor = [{_tuple(color)}] (",
                '                    interpolation = "constant"',
                "                )",
                '                uniform token subdivisionScheme = "none"',
                f"                double3 xformOp:scale = {_tuple(size)}",
                '                uniform token[] xformOpOrder = ["xformOp:scale"]',
                "            }",
            ]
        )
    lines.extend(["        }", ""])
    return lines


def _joint_block(workspace: Workspace, node: SceneNode) -> list[str]:
    if node.joint is None or node.collider is None:
        return []
    joint = node.joint
    parent = workspace.node(joint.parent)
    if parent.collider is None:
        raise RealToSimError(f"joint parent {parent.node_id} has no collider")
    local0 = np.asarray(joint.origin) - np.asarray(parent.collider.center)
    local1 = np.asarray(joint.origin) - np.asarray(node.collider.center)
    type_name = "PhysicsPrismaticJoint" if joint.kind == "prismatic" else "PhysicsRevoluteJoint"
    signed_limits = sorted((joint.lower * joint.axis_sign, joint.upper * joint.axis_sign))
    return [
        f'        def {type_name} "{node.node_id}_joint"',
        "        {",
        f"            custom float nanousdRts:fitConfidence = {joint.confidence:.9g}",
        f"            custom string nanousdRts:fitProvenance = {_usd_string(joint.provenance)}",
        f"            uniform token physics:axis = {_usd_string(joint.axis)}",
        f"            rel physics:body0 = </World/Nodes/{joint.parent}>",
        f"            rel physics:body1 = </World/Nodes/{node.node_id}>",
        f"            point3f physics:localPos0 = {_tuple(tuple(local0))}",
        f"            point3f physics:localPos1 = {_tuple(tuple(local1))}",
        "            quatf physics:localRot0 = (1, 0, 0, 0)",
        "            quatf physics:localRot1 = (1, 0, 0, 0)",
        f"            float physics:lowerLimit = {signed_limits[0]:.9g}",
        f"            float physics:upperLimit = {signed_limits[1]:.9g}",
        "        }",
        "",
    ]


def compile_usda(workspace: Workspace, *, output: Path | None = None) -> dict[str, Any]:
    workspace.verify_source()
    output = Path(output).resolve() if output else workspace.root / "exports" / "scene.usda"
    nodes = workspace.nodes
    joint_nodes = [node for node in nodes if node.joint is not None]
    joint_parents = {node.joint.parent for node in joint_nodes if node.joint is not None}
    accepted_completions = {
        item["node"]: item
        for item in workspace.completions
        if item.get("status") == "accepted"
    }
    gravity = [0.0, 0.0, 0.0]
    gravity[AXES[workspace.up_axis]] = -1.0
    root_schemas = ["PhysicsArticulationRootAPI"] if joint_nodes else []
    root_schema_text = (
        " (\n    prepend apiSchemas = "
        + _array([_usd_string(item) for item in root_schemas])
        + "\n)"
        if root_schemas
        else ""
    )
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        f"    metersPerUnit = {float(workspace.state['meters_per_unit']):.9g}",
        f'    upAxis = "{workspace.up_axis}"',
        "    timeCodesPerSecond = 60",
        ")",
        "",
        f'def Xform "World"{root_schema_text}',
        "{",
        '    custom string nanousdRts:schema = "nanousd-agentic-real-to-sim/1"',
        f"    custom string nanousdRts:sceneDigest = {_usd_string(workspace.state['logical_digest'])}",
        f"    custom string nanousdRts:sourceSha256 = {_usd_string(workspace.state['source']['sha256'])}",
        "",
        '    def PhysicsScene "PhysicsScene"',
        "    {",
        f"        vector3f physics:gravityDirection = {_tuple(tuple(gravity))}",
        "        float physics:gravityMagnitude = 9.81",
        "    }",
        "",
        '    def Scope "Nodes"',
        "    {",
    ]
    for node in nodes:
        accepted_completion = accepted_completions.get(node.node_id)
        lines.extend(
            _node_block(
                node,
                joint_parents=joint_parents,
                source_sha256=workspace.state["source"]["sha256"],
                accepted_completion=accepted_completion,
                source_asset=_asset_reference(
                    workspace.source_path,
                    relative_to=output.parent,
                ),
                selection_asset=_asset_reference(
                    workspace.root / node.selection_file,
                    relative_to=output.parent,
                ),
                completion_asset=(
                    _asset_reference(
                        workspace.root / accepted_completion["asset"],
                        relative_to=output.parent,
                    )
                    if accepted_completion
                    else None
                ),
            )
        )
    lines.extend(["    }", "", '    def Scope "Joints"', "    {"])
    for node in joint_nodes:
        lines.extend(_joint_block(workspace, node))
    lines.extend(["    }", "}"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    validation = validate_usda_text(output, expected_nodes={node.node_id for node in nodes})
    manifest = {
        "schema_version": 1,
        "usda": str(output),
        "sha256": sha256_file(output),
        "source_sha256": workspace.state["source"]["sha256"],
        "scene_digest": workspace.state["logical_digest"],
        "nodes": [node.node_id for node in nodes],
        "joints": [node.node_id for node in joint_nodes],
        "accepted_completions": sorted(item["id"] for item in accepted_completions.values()),
        "validation": validation,
        "representation": {
            "visual_truth": "immutable Gaussian PLY plus per-node source-row selections",
            "physical_truth": "registered box colliders, support graph, and USD Physics joints",
        },
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    workspace.trace(
        "compile-usda",
        {"output": str(output)},
        {"manifest": str(manifest_path), "sha256": manifest["sha256"], "validation": validation},
    )
    return manifest


def validate_usda_text(path: Path, *, expected_nodes: set[str] | None = None) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    gates = {
        "magic": text.startswith("#usda 1.0"),
        "balanced_braces": text.count("{") == text.count("}"),
        "physics_scene": 'def PhysicsScene "PhysicsScene"' in text,
        "source_provenance": "nanousdRts:sourceSha256" in text,
        "node_presence": all(f'def Xform "{node}"' in text for node in (expected_nodes or set())),
        "joint_relationship_pairs": text.count("physics:body0") == text.count("physics:body1"),
    }
    return {"gates": gates, "passed": all(gates.values())}


def render_usda(
    workspace: Workspace,
    *,
    usda_path: Path | None = None,
    width: int = 640,
    height: int = 360,
) -> dict[str, Any]:
    path = Path(usda_path).resolve() if usda_path else workspace.root / "exports" / "scene.usda"
    if not path.is_file():
        raise RealToSimError(f"compiled USDA is missing: {path}")
    renderer_type, dependency = _load_renderer_binding()
    renderer = renderer_type(
        width=width,
        height=height,
        enable_rt=True,
        enable_materials=False,
        visible=False,
    )
    scene_bounds = workspace.state["source"]["report"]["bounds"]
    minimum = np.asarray(scene_bounds["min"], dtype=np.float64)
    maximum = np.asarray(scene_bounds["max"], dtype=np.float64)
    center = (minimum + maximum) * 0.5
    diagonal = float(np.linalg.norm(maximum - minimum))
    up_index = AXES[workspace.up_axis]
    horizontal = [index for index in range(3) if index != up_index]
    eye = center.copy()
    eye[horizontal[1]] -= max(diagonal * 1.1, 3.0)
    eye[up_index] += max(diagonal * 0.25, 1.0)
    up = np.zeros(3)
    up[up_index] = 1.0
    try:
        mesh_count = renderer.load_usd(str(path))
        if mesh_count <= 0:
            raise RealToSimError("NanoUSD loaded no meshes from the compiled scene")
        renderer.set_camera_explicit(
            tuple(eye),
            tuple(center),
            tuple(up),
            55.0,
            max(0.01, diagonal / 100_000.0),
            max(100.0, diagonal * 20.0),
        )
        renderer.render()
        pixels = renderer.fetch_pixels()[..., :3]
        backend = renderer.get_backend_info()
    finally:
        renderer.close()
    if int(pixels.sum(dtype=np.uint64)) == 0:
        raise RealToSimError("compiled USDA render is blank")
    output_dir = workspace.root / "evidence" / "usda"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "scene.png"
    Image.fromarray(pixels, mode="RGB").save(image_path)
    report = {
        "schema_version": 1,
        "usda": str(path),
        "usda_sha256": sha256_file(path),
        "mesh_count": mesh_count,
        "rgb_sum": int(pixels.sum(dtype=np.uint64)),
        "image": str(image_path),
        "backend": backend,
        "dependency": dependency,
        "passed": True,
    }
    report_path = output_dir / "render.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    workspace.trace("render-usda", {"usda": str(path)}, {"report": str(report_path), "mesh_count": mesh_count})
    return report
