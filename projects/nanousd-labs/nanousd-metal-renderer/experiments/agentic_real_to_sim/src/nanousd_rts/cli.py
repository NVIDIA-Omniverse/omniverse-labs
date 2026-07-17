"""Command-line surface for Codex and local human iteration."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .collision import voxelize
from .completion import accept_completion, completion_report, propose_hidden_interiors
from .core import Bounds, RealToSimError, Workspace
from .experience import serve_preview, write_experience
from .gaussian import Camera, _renderer_root, ingest, load_gaussians, render, select_render_mask
from .home_kitchen import author_home_scan_kitchen
from .learned_materials import generate_material_bundle, learned_material_status
from .material_preview import write_material_comparison
from .mesh_completion import (
    ART_DIRECTED_OVEN_PROVIDER,
    EXTERNAL_MATERIAL_PROVIDER,
    LOCAL_MATERIAL_PROVIDER,
    fit_mesh_pbr_completion,
)
from .preview import write_preview
from .segmentation_review import (
    accept_segmentation_review,
    check_segmentation_review_evidence,
    create_segmentation_review_plan,
    segmentation_review_status,
)
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
from .workflow import TOOL_CATALOG, build_demo, run_plan


def _json(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, default=str))


def _workspace(value: str) -> Workspace:
    return Workspace.open(Path(value))


def _bounds(values: list[float]) -> Bounds:
    return Bounds(tuple(values[:3]), tuple(values[3:]))


def _add_common_workspace(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("workspace", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nanousd-rts",
        description="Local M5 Gaussian-to-interactive-USD development oracle.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("tools", help="Print the deterministic agent tool catalog.")
    sub.add_parser("doctor", help="Check local NanoUSD, Metal, Python, and splat-transform prerequisites.")

    ingest_parser = sub.add_parser("ingest", help="Create an immutable workspace from PLY or SOG/LOD input.")
    ingest_parser.add_argument("source", type=Path)
    ingest_parser.add_argument("workspace", type=Path)
    ingest_parser.add_argument("--lod", type=int)
    ingest_parser.add_argument("--up-axis", choices=("X", "Y", "Z"), default="Y")
    ingest_parser.add_argument("--meters-per-unit", type=float, default=1.0)
    ingest_parser.add_argument("--replace", action="store_true")

    probe_parser = sub.add_parser("probe", help="Inspect source and scene graph state.")
    _add_common_workspace(probe_parser)

    render_parser = sub.add_parser("render", help="Render Gaussian RGB/depth/normal/stable-ID evidence.")
    _add_common_workspace(render_parser)
    render_parser.add_argument("--name", default="baseline")
    render_parser.add_argument("--width", type=int, default=960)
    render_parser.add_argument("--height", type=int, default=540)
    render_parser.add_argument("--k", type=int, choices=(8, 16, 32), default=16)
    render_parser.add_argument("--eye", nargs=3, type=float)
    render_parser.add_argument("--target", nargs=3, type=float)
    render_parser.add_argument("--up", nargs=3, type=float)
    render_parser.add_argument("--fov", type=float, default=60.0)

    voxel_parser = sub.add_parser("voxelize", help="Generate voxel occupancy and a collision GLB.")
    _add_common_workspace(voxel_parser)
    voxel_parser.add_argument("--node")
    voxel_parser.add_argument("--voxel-size", type=float, default=0.05)
    voxel_parser.add_argument("--opacity-threshold", type=float, default=0.1)
    voxel_parser.add_argument("--mesh-shape", choices=("faces", "smooth"), default="faces")
    voxel_parser.add_argument("--external-fill", type=float)
    voxel_parser.add_argument("--floor-fill", type=float)
    voxel_parser.add_argument("--carve", nargs=2, type=float, metavar=("HEIGHT", "RADIUS"))
    voxel_parser.add_argument("--seed", nargs=3, type=float)

    add_parser = sub.add_parser("add-node", help="Create a visual selection and atomically linked collider.")
    _add_common_workspace(add_parser)
    add_parser.add_argument("--id", required=True)
    add_parser.add_argument("--label")
    add_parser.add_argument("--role", choices=("background", "static", "movable", "articulated"), required=True)
    source_group = add_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--bounds", nargs=6, type=float, metavar=("MIN_X", "MIN_Y", "MIN_Z", "MAX_X", "MAX_Y", "MAX_Z"))
    source_group.add_argument("--id-aov", type=Path)
    add_parser.add_argument("--mask", type=Path, help="Required with --id-aov; PNG or NumPy boolean mask.")
    add_parser.add_argument("--collider-bounds", nargs=6, type=float)
    add_parser.add_argument("--collider-padding", type=float, default=0.0)
    add_parser.add_argument("--collider-confidence", type=float, default=0.6)
    add_parser.add_argument("--collider-provenance", default="agent-authored")
    add_parser.add_argument("--collision-mode", choices=("solid", "support", "shell"), default="solid")
    add_parser.add_argument("--tag", action="append", default=[])

    support_parser = sub.add_parser("infer-support", help="Infer containment and surface support edges.")
    _add_common_workspace(support_parser)
    support_parser.add_argument("--max-gap", type=float)

    set_support_parser = sub.add_parser("set-support", help="Set one explicit support edge.")
    _add_common_workspace(set_support_parser)
    set_support_parser.add_argument("--child", required=True)
    set_support_parser.add_argument("--parent", required=True)

    joint_parser = sub.add_parser("fit-joint", help="Fit a DRAWER-style prismatic or revolute candidate.")
    _add_common_workspace(joint_parser)
    joint_parser.add_argument("--node", required=True)
    joint_parser.add_argument("--parent")
    joint_parser.add_argument("--kind", choices=("auto", "prismatic", "revolute"), default="auto")
    joint_parser.add_argument("--axis", choices=("X", "Y", "Z"))
    joint_parser.add_argument("--axis-sign", type=int, choices=(-1, 1))
    joint_parser.add_argument("--origin", nargs=3, type=float)
    joint_parser.add_argument("--lower", type=float)
    joint_parser.add_argument("--upper", type=float)

    completion_parser = sub.add_parser(
        "propose-completions",
        help="Generate non-measured hidden-interior Gaussian candidates.",
    )
    _add_common_workspace(completion_parser)
    completion_parser.add_argument("--node", required=True)
    completion_parser.add_argument("--factor", type=float, action="append")
    completion_parser.add_argument("--gaussian-count", type=int, default=600)

    accept_parser = sub.add_parser(
        "accept-completion",
        help="Accept one hidden completion after an articulation sweep.",
    )
    _add_common_workspace(accept_parser)
    accept_parser.add_argument("--completion", required=True)

    completion_report_parser = sub.add_parser("completions", help="Inspect hidden-completion candidates.")
    _add_common_workspace(completion_report_parser)

    mesh_completion_parser = sub.add_parser(
        "fit-mesh-pbr",
        help="Upgrade an accepted completion to a UV/PBR mesh with face-bound Gaussians.",
    )
    _add_common_workspace(mesh_completion_parser)
    mesh_completion_parser.add_argument("--node", required=True)
    mesh_completion_parser.add_argument(
        "--material-provider",
        choices=(
            LOCAL_MATERIAL_PROVIDER,
            EXTERNAL_MATERIAL_PROVIDER,
            ART_DIRECTED_OVEN_PROVIDER,
        ),
        default=LOCAL_MATERIAL_PROVIDER,
    )
    mesh_completion_parser.add_argument(
        "--material-bundle",
        type=Path,
        help="UV-aligned external PBR bundle; required by external-pbr-atlas-v1.",
    )
    mesh_completion_parser.add_argument("--texture-size", type=int, default=512)
    mesh_completion_parser.add_argument("--gaussian-multiplier", type=float, default=1.0)

    sweep_parser = sub.add_parser("sweep", help="Sweep an articulation and fail on forbidden overlap.")
    _add_common_workspace(sweep_parser)
    sweep_parser.add_argument("--node", required=True)
    sweep_parser.add_argument("--samples", type=int, default=17)

    settle_parser = sub.add_parser("settle", help="Run the deterministic local gravity/support check.")
    _add_common_workspace(settle_parser)

    push_parser = sub.add_parser("push", help="Run a deterministic collider push test.")
    _add_common_workspace(push_parser)
    push_parser.add_argument("--node", required=True)
    push_parser.add_argument("--delta", nargs=3, type=float, required=True)

    compile_parser = sub.add_parser("compile", help="Compile the dual-representation scene to USDA.")
    _add_common_workspace(compile_parser)
    compile_parser.add_argument("--output", type=Path)

    render_usda_parser = sub.add_parser("render-usda", help="Load and render the compiled proxy scene.")
    _add_common_workspace(render_usda_parser)
    render_usda_parser.add_argument("--usda", type=Path)
    render_usda_parser.add_argument("--width", type=int, default=640)
    render_usda_parser.add_argument("--height", type=int, default=360)

    verify_parser = sub.add_parser("verify", help="Run all local fail-closed gates.")
    _add_common_workspace(verify_parser)
    verify_parser.add_argument("--skip-sweeps", action="store_true")

    preview_parser = sub.add_parser("preview", help="Write the interactive local scene/joint preview.")
    _add_common_workspace(preview_parser)
    preview_parser.add_argument("--output", type=Path)
    preview_parser.add_argument("--open", action="store_true", dest="open_preview")

    experience_parser = sub.add_parser(
        "experience-preview",
        help="Write the original streamed SOG viewer plus articulation oracle.",
    )
    _add_common_workspace(experience_parser)
    experience_parser.add_argument("--output", type=Path)
    experience_parser.add_argument("--budget", type=float, default=32.0)

    home_kitchen_parser = sub.add_parser(
        "author-home-kitchen",
        help="Author the full visible Home Scan kitchen with generated amodal interiors.",
    )
    _add_common_workspace(home_kitchen_parser)

    segmentation_plan_parser = sub.add_parser(
        "segmentation-review-plan",
        help="Write closed/half/open visual-review requirements for refined articulations.",
    )
    _add_common_workspace(segmentation_plan_parser)
    segmentation_plan_parser.add_argument("--node", action="append")

    segmentation_accept_parser = sub.add_parser(
        "accept-segmentation-review",
        help="Accept captured closed/half/open evidence after semantic inspection.",
    )
    _add_common_workspace(segmentation_accept_parser)
    segmentation_accept_parser.add_argument("--node", action="append")
    segmentation_accept_parser.add_argument("--reviewer", required=True)
    segmentation_accept_parser.add_argument("--note", required=True)

    segmentation_status_parser = sub.add_parser(
        "segmentation-reviews",
        help="Report the fail-closed visual segmentation review gate.",
    )
    _add_common_workspace(segmentation_status_parser)

    segmentation_check_parser = sub.add_parser(
        "check-segmentation-review",
        help="Check every captured pose triplet before semantic acceptance.",
    )
    _add_common_workspace(segmentation_check_parser)
    segmentation_check_parser.add_argument("--node", action="append")

    serve_parser = sub.add_parser(
        "serve-preview",
        help="Serve the high-fidelity local preview over HTTP.",
    )
    _add_common_workspace(serve_parser)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--budget", type=float, default=32.0)
    serve_parser.add_argument("--open", action="store_true", dest="open_preview")

    sub.add_parser(
        "material-models",
        help="Report pinned learned-material models, runtime dependencies, and MPS availability.",
    )

    generate_material_parser = sub.add_parser(
        "generate-materials",
        help="Run official MatFuse or StableMaterials weights into an external PBR bundle.",
    )
    generate_material_parser.add_argument("request_bundle", type=Path)
    generate_material_parser.add_argument("output_bundle", type=Path)
    generate_material_parser.add_argument(
        "--backend",
        choices=("matfuse", "stablematerials"),
        required=True,
    )
    generate_material_parser.add_argument(
        "--device", choices=("auto", "mps", "cuda", "cpu"), default="auto"
    )
    generate_material_parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16"), default="auto"
    )
    generate_material_parser.add_argument("--seed", type=int, default=42)
    generate_material_parser.add_argument("--steps", type=int)
    generate_material_parser.add_argument("--guidance-scale", type=float)
    generate_material_parser.add_argument(
        "--stable-variant", choices=("base", "lcm"), default="lcm"
    )
    generate_material_parser.add_argument("--prompt")
    generate_material_parser.add_argument(
        "--no-matfuse-palette",
        action="store_true",
        help="Use text-only MatFuse generation instead of the measured-front palette.",
    )
    generate_material_parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Fail instead of downloading a missing pinned model snapshot.",
    )

    compare_material_parser = sub.add_parser(
        "compare-materials",
        help="Write a side-by-side PBR inspector for MatFuse and StableMaterials bundles.",
    )
    compare_material_parser.add_argument("matfuse_bundle", type=Path)
    compare_material_parser.add_argument("stablematerials_bundle", type=Path)
    compare_material_parser.add_argument("output", type=Path)
    compare_material_parser.add_argument("--matfuse-scene")
    compare_material_parser.add_argument("--stablematerials-scene")

    plan_parser = sub.add_parser("run-plan", help="Execute a bounded JSON agent action plan.")
    _add_common_workspace(plan_parser)
    plan_parser.add_argument("plan", type=Path)

    demo_parser = sub.add_parser("demo", help="Build, render, verify, and preview a synthetic drawer/door scene.")
    demo_parser.add_argument("workspace", type=Path)
    demo_parser.add_argument("--no-gaussian-render", action="store_true")
    demo_parser.add_argument("--open", action="store_true", dest="open_preview")
    return parser


def doctor() -> dict[str, Any]:
    root = _renderer_root()
    library = root / "build" / "libnusd_renderer.dylib"
    gaussian_test = root / "build" / "test_gaussian_render"
    npx = subprocess.run(["npx", "--version"], capture_output=True, text=True, check=False)
    hardware = subprocess.run(
        ["system_profiler", "SPHardwareDataType", "-json"],
        capture_output=True,
        text=True,
        check=False,
    )
    chip = None
    memory = None
    if hardware.returncode == 0:
        try:
            item = json.loads(hardware.stdout)["SPHardwareDataType"][0]
            chip = item.get("chip_type")
            memory = item.get("physical_memory")
        except (KeyError, IndexError, json.JSONDecodeError):
            pass
    gates = {
        "darwin": platform.system() == "Darwin",
        "apple_silicon": platform.machine() == "arm64",
        "renderer_library": library.is_file(),
        "gaussian_smoke_binary": gaussian_test.is_file(),
        "splat_transform_runner": npx.returncode == 0,
    }
    return {
        "gates": gates,
        "passed": all(gates.values()),
        "machine": {"chip": chip, "memory": memory, "platform": platform.platform()},
        "renderer_root": str(root),
        "renderer_library": str(library),
        "npx_version": npx.stdout.strip(),
        "remediation": "Build nanousd and nanousd-metal-renderer, then rerun doctor."
        if not all(gates.values())
        else None,
    }


def dispatch(args: argparse.Namespace) -> tuple[Any, int]:
    if args.command == "tools":
        return {"schema_version": 1, "tools": TOOL_CATALOG}, 0
    if args.command == "doctor":
        report = doctor()
        return report, 0 if report["passed"] else 1
    if args.command == "material-models":
        report = learned_material_status()
        return report, 0 if report["runtime"]["dependencies"] else 1
    if args.command == "generate-materials":
        return generate_material_bundle(
            args.request_bundle,
            args.output_bundle,
            backend=args.backend,
            device=args.device,
            dtype=args.dtype,
            seed=args.seed,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            stable_variant=args.stable_variant,
            prompt_override=args.prompt,
            use_matfuse_palette=not args.no_matfuse_palette,
            local_files_only=args.local_files_only,
        ), 0
    if args.command == "compare-materials":
        output = write_material_comparison(
            args.matfuse_bundle,
            args.stablematerials_bundle,
            output=args.output,
            matfuse_scene=args.matfuse_scene,
            stablematerials_scene=args.stablematerials_scene,
        )
        return {"html": str(output)}, 0
    if args.command == "ingest":
        workspace = ingest(
            args.source,
            args.workspace,
            lod=args.lod,
            up_axis=args.up_axis,
            meters_per_unit=args.meters_per_unit,
            replace=args.replace,
        )
        return workspace.state, 0
    if args.command == "demo":
        report = build_demo(args.workspace, render_gaussians=not args.no_gaussian_render)
        if args.open_preview:
            subprocess.run(["open", report["preview"]], check=False)
        return report, 0 if report["verification"]["passed"] else 1
    workspace = _workspace(str(args.workspace))
    if args.command == "probe":
        scene = load_gaussians(workspace.source_path)
        return {
            "workspace": str(workspace.root),
            "source": scene.report(),
            "source_sha256": scene.source_sha256,
            "scene_revision": workspace.state["scene_revision"],
            "scene_digest": workspace.state["logical_digest"],
            "nodes": [node.to_json() for node in workspace.nodes],
            "support_edges": workspace.state["support_edges"],
        }, 0
    if args.command == "render":
        camera = None
        if args.eye or args.target or args.up:
            if not args.eye or not args.target:
                raise RealToSimError("--eye and --target must be provided together")
            default_up = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[workspace.up_axis]
            camera = Camera(
                name=args.name,
                eye=tuple(args.eye),
                target=tuple(args.target),
                up=tuple(args.up or default_up),
                fov_degrees=args.fov,
            )
        return render(
            workspace,
            name=args.name,
            camera=camera,
            width=args.width,
            height=args.height,
            k=args.k,
        ), 0
    if args.command == "voxelize":
        return voxelize(
            workspace,
            node_id=args.node,
            voxel_size=args.voxel_size,
            opacity_threshold=args.opacity_threshold,
            mesh_shape=args.mesh_shape,
            external_fill=args.external_fill,
            floor_fill=args.floor_fill,
            carve=tuple(args.carve) if args.carve else None,
            seed=tuple(args.seed) if args.seed else None,
        ), 0
    if args.command == "add-node":
        collider_bounds = _bounds(args.collider_bounds) if args.collider_bounds else None
        common = {
            "node_id": args.id,
            "label": args.label or args.id,
            "role": args.role,
            "collider_bounds": collider_bounds,
            "collider_padding": args.collider_padding,
            "collider_confidence": args.collider_confidence,
            "collider_provenance": args.collider_provenance,
            "collision_mode": args.collision_mode,
            "tags": tuple(args.tag),
        }
        if args.bounds:
            node = add_node_from_bounds(workspace, bounds=_bounds(args.bounds), **common)
        else:
            if args.mask is None:
                raise RealToSimError("--mask is required with --id-aov")
            indices = select_render_mask(args.id_aov, args.mask)
            node = add_node(workspace, source_indices=indices, **common)
        return node.to_json(), 0
    if args.command == "infer-support":
        return {"relations": infer_support(workspace, max_gap=args.max_gap)}, 0
    if args.command == "set-support":
        return set_support(workspace, child_id=args.child, parent_id=args.parent).to_json(), 0
    if args.command == "fit-joint":
        joint = fit_joint(
            workspace,
            node_id=args.node,
            parent_id=args.parent,
            kind=args.kind,
            axis=args.axis,
            axis_sign=args.axis_sign,
            origin=tuple(args.origin) if args.origin else None,
            lower=args.lower,
            upper=args.upper,
        )
        return asdict(joint), 0
    if args.command == "propose-completions":
        candidates = propose_hidden_interiors(
            workspace,
            node_id=args.node,
            factors=tuple(args.factor) if args.factor else (0.75, 0.9, 1.0),
            gaussian_count=args.gaussian_count,
        )
        return {"candidates": candidates}, 0
    if args.command == "accept-completion":
        return accept_completion(workspace, completion_id=args.completion), 0
    if args.command == "completions":
        report = completion_report(workspace)
        return report, 0 if report["all_assets_valid"] else 1
    if args.command == "fit-mesh-pbr":
        return fit_mesh_pbr_completion(
            workspace,
            node_id=args.node,
            material_provider=args.material_provider,
            external_material_bundle=args.material_bundle,
            texture_size=args.texture_size,
            gaussian_multiplier=args.gaussian_multiplier,
        ), 0
    if args.command == "sweep":
        report = sweep_joint(workspace, node_id=args.node, samples=args.samples)
        return report, 0 if report["passed"] else 1
    if args.command == "settle":
        report = settle_scene(workspace)
        return report, 0 if report["passed"] else 1
    if args.command == "push":
        report = push_test(workspace, node_id=args.node, delta=tuple(args.delta))
        return report, 0 if report["passed"] else 1
    if args.command == "compile":
        report = compile_usda(workspace, output=args.output)
        return report, 0 if report["validation"]["passed"] else 1
    if args.command == "render-usda":
        return render_usda(
            workspace,
            usda_path=args.usda,
            width=args.width,
            height=args.height,
        ), 0
    if args.command == "verify":
        report = verify(workspace, run_sweeps=not args.skip_sweeps)
        return report, 0 if report["passed"] else 1
    if args.command == "preview":
        output = write_preview(workspace, output=args.output)
        if args.open_preview:
            subprocess.run(["open", str(output)], check=False)
        return {"html": str(output)}, 0
    if args.command == "experience-preview":
        return write_experience(
            workspace,
            output=args.output,
            budget=args.budget,
        ), 0
    if args.command == "author-home-kitchen":
        return author_home_scan_kitchen(workspace), 0
    if args.command == "segmentation-review-plan":
        return create_segmentation_review_plan(
            workspace,
            node_ids=args.node,
        ), 0
    if args.command == "accept-segmentation-review":
        report = accept_segmentation_review(
            workspace,
            reviewer=args.reviewer,
            note=args.note,
            node_ids=args.node,
        )
        return report, 0 if report["passed"] else 1
    if args.command == "segmentation-reviews":
        report = segmentation_review_status(workspace)
        return report, 0 if report["passed"] else 1
    if args.command == "check-segmentation-review":
        report = check_segmentation_review_evidence(
            workspace,
            node_ids=args.node,
        )
        return report, 0 if report["passed"] else 1
    if args.command == "serve-preview":
        return serve_preview(
            workspace,
            host=args.host,
            port=args.port,
            budget=args.budget,
            open_browser=args.open_preview,
        ), 0
    if args.command == "run-plan":
        return run_plan(workspace, args.plan), 0
    raise AssertionError(args.command)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        value, exit_code = dispatch(args)
        _json(value)
        return exit_code
    except RealToSimError as exc:
        print(
            json.dumps(
                {"status": "error", "error": str(exc), "error_type": type(exc).__name__},
                sort_keys=True,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
