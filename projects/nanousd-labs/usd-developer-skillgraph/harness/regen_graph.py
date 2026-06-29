#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan and validate generated artifacts for a skill subgraph."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import string
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from command_utils import ensure_executable, format_command, split_command


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def topo_closure(graph: dict[str, Any], roots: list[str]) -> list[str]:
    nodes = graph["nodes"]

    # Pass 1: the hard closure (mandatory depends_on only) determines which
    # nodes are independently present in this scope. An optional handler is
    # "registered" iff it lands here -- which happens when the scope lists it in
    # required_nodes (or some required node hard-depends on it).
    present: set[str] = set()

    def hard_visit(node_id: str) -> None:
        if node_id in present or node_id not in nodes:
            return
        present.add(node_id)
        for dep in nodes[node_id].get("depends_on", []):
            hard_visit(dep)

    for root in roots:
        hard_visit(root)

    # Pass 2: ordered closure following depends_on plus optional_depends_on
    # edges whose target is present. Following the active optional edge orders
    # the registered handler before the dispatcher that optionally depends on
    # it; an unregistered handler is never pulled in.
    order: list[str] = []
    visiting: set[str] = set()
    seen: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in seen:
            return
        if node_id in visiting:
            raise ValueError(f"cycle in skill graph at {node_id}")
        if node_id not in nodes:
            raise ValueError(f"unknown skill node {node_id}")
        visiting.add(node_id)
        for dep in nodes[node_id].get("depends_on", []):
            visit(dep)
        for dep in nodes[node_id].get("optional_depends_on", []):
            if dep in present:
                visit(dep)
        visiting.remove(node_id)
        seen.add(node_id)
        order.append(node_id)

    for root in roots:
        visit(root)
    return order


def provided_capabilities(node: dict[str, Any]) -> set[str]:
    capabilities: set[str] = set()
    for item in node.get("provides", []):
        if isinstance(item, dict) and isinstance(item.get("capability"), str):
            capabilities.add(item["capability"])
        elif isinstance(item, str):
            capabilities.add(item)
    return capabilities


def validate_consumes(graph: dict[str, Any]) -> None:
    nodes = graph["nodes"]
    for node_id, node in nodes.items():
        dependencies = set(node.get("depends_on", [])) | set(node.get("optional_depends_on", []))
        for consume in node.get("consumes", []):
            provider = consume.get("provider")
            capability = consume.get("capability")
            if not isinstance(provider, str) or not isinstance(capability, str):
                raise ValueError(f"{node_id} has malformed consumes entry")
            if provider not in nodes:
                raise ValueError(f"{node_id} consumes {provider}.{capability}, but provider node is missing")
            if provider not in dependencies:
                raise ValueError(f"{node_id} consumes {provider}.{capability}, but does not depend on {provider} (directly or optionally)")
            provided = provided_capabilities(nodes[provider])
            if capability not in provided:
                raise ValueError(f"{node_id} consumes {provider}.{capability}, but provider does not declare it")


def roots_from_args(graph: dict[str, Any], args: argparse.Namespace) -> tuple[str, list[str]]:
    if args.node:
        return args.node, [args.node]
    scope = graph["scopes"][args.scope]
    return args.scope, list(scope["required_nodes"])


def node_artifacts(repo: Path, node: dict[str, Any], target: str) -> list[Path]:
    target_info = node.get("targets", {}).get(target)
    if not target_info:
        return []
    return [repo / p for p in target_info.get("artifacts", [])]


def referenced_contract_paths(value: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {
                "contract",
                "contracts",
                "dependency_contracts",
                "adapter_contracts",
                "performance_contracts",
                "production_contracts",
                "coverage_contracts",
                "contract_lints",
            }:
                if isinstance(item, str):
                    refs.append(item)
                elif isinstance(item, list):
                    refs.extend(ref for ref in item if isinstance(ref, str))
            refs.extend(referenced_contract_paths(item))
    elif isinstance(value, list):
        for item in value:
            refs.extend(referenced_contract_paths(item))
    return refs


def node_fingerprints(repo: Path, graph: dict[str, Any], order: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for node_id in order:
        node = graph["nodes"][node_id]
        parts: list[str] = [
            json.dumps(graph["spec"], sort_keys=True),
            node_id,
            json.dumps(node.get("depends_on", []), sort_keys=True),
            json.dumps(node.get("provides", []), sort_keys=True),
            json.dumps(node.get("consumes", []), sort_keys=True),
            json.dumps(node.get("targets", {}), sort_keys=True),
        ]
        skill_path = repo / node["skill"]
        if skill_path.exists():
            parts.append(sha256_file(skill_path))
        else:
            parts.append("missing-skill:" + node["skill"])
        for contract_ref in sorted(set(referenced_contract_paths(node))):
            contract_path = repo / contract_ref
            if contract_path.exists():
                parts.append(contract_ref + ":" + sha256_file(contract_path))
            else:
                parts.append("missing-contract:" + contract_ref)
        for dep in node.get("depends_on", []):
            parts.append(dep + ":" + result[dep])
        result[node_id] = sha256_text("\n".join(parts))
    return result


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def status_for_node(
    repo: Path,
    node_id: str,
    node: dict[str, Any],
    target: str,
    manifest: dict[str, Any],
    fingerprints: dict[str, str],
) -> tuple[str, list[str]]:
    artifacts = node_artifacts(repo, node, target)
    reasons: list[str] = []
    if not artifacts:
        reasons.append("no target artifacts declared")
        return "missing", reasons
    missing = [str(path.relative_to(repo)) for path in artifacts if not path.exists()]
    if missing:
        reasons.extend("missing " + path for path in missing)
        return "missing", reasons

    recorded = manifest.get("nodes", {}).get(node_id)
    if not recorded:
        return "unrecorded", ["artifacts exist but no manifest entry"]
    if recorded.get("fingerprint") != fingerprints[node_id]:
        return "stale", ["skill/dependency fingerprint changed"]

    recorded_artifacts = recorded.get("artifacts", {})
    for path in artifacts:
        rel = str(path.relative_to(repo)).replace("\\", "/")
        digest = sha256_file(path)
        if recorded_artifacts.get(rel) != digest:
            reasons.append("artifact changed " + rel)
    if reasons:
        return "stale", reasons
    return "ready", []


def build_manifest(
    repo: Path,
    graph: dict[str, Any],
    order: list[str],
    target: str,
    fingerprints: dict[str, str],
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    nodes: dict[str, Any] = dict((existing or {}).get("nodes", {}))
    for node_id in order:
        node = graph["nodes"][node_id]
        artifacts = node_artifacts(repo, node, target)
        if not artifacts or any(not path.exists() for path in artifacts):
            continue
        nodes[node_id] = {
            "fingerprint": fingerprints[node_id],
            "artifacts": {
                str(path.relative_to(repo)).replace("\\", "/"): sha256_file(path)
                for path in artifacts
            },
        }
    return {
        "version": graph["version"],
        "target": target,
        "spec": graph["spec"],
        "nodes": nodes,
    }


def print_plan(plan: list[dict[str, Any]]) -> None:
    print("Skill subgraph:")
    for i, item in enumerate(plan, 1):
        dep_text = ", ".join(item["depends_on"]) if item["depends_on"] else "-"
        consumes = [
            f"{consume.get('provider', '?')}.{consume.get('capability', '?')}"
            for consume in item.get("consumes", [])
        ]
        consumes_text = ", ".join(consumes) if consumes else "-"
        artifact_text = ", ".join(item["artifacts"]) if item["artifacts"] else "-"
        print(f"{i}. {item['node']}: {item['status']}")
        print(f"   depends_on: {dep_text}")
        print(f"   consumes: {consumes_text}")
        print(f"   artifacts: {artifact_text}")
        for reason in item["reasons"]:
            print(f"   reason: {reason}")


def run_command(argv: list[str]) -> int:
    print(f"$ {format_command(argv)}")
    sys.stdout.flush()
    ensure_executable(argv)
    result = subprocess.run(argv)
    return result.returncode


def validate_scope(
    repo: Path,
    graph: dict[str, Any],
    scope_name: str,
    target: str,
    graph_path: Path,
) -> int:
    scope = graph["scopes"][scope_name]
    target_info = scope.get("targets", {}).get(target, {})
    dump_cmd = target_info.get("dump_cmd")
    adapter_cmd = target_info.get("adapter_cmd")
    benchmark_adapter_cmd = target_info.get("benchmark_adapter_cmd")
    if scope.get("goldens") and not dump_cmd:
        print(f"No dump command for scope {scope_name} target {target}", file=sys.stderr)
        return 2
    if scope.get("handle_goldens") and not adapter_cmd:
        print(f"No handle adapter command for scope {scope_name} target {target}", file=sys.stderr)
        return 2
    if scope.get("handle_benchmarks") and not benchmark_adapter_cmd:
        print(f"No handle benchmark adapter command for scope {scope_name} target {target}", file=sys.stderr)
        return 2

    failed = False
    if target_info.get("command_contract"):
        cmd = [
            sys.executable,
            "harness/target_contract.py",
            "--graph",
            str(graph_path),
            "--scope",
            scope_name,
            "--target",
            target,
        ]
        if run_command(cmd) != 0:
            failed = True
            return 1
    for golden in scope.get("goldens", []):
        cmd = [sys.executable, "harness/score.py", "--goldens", golden, "--dump-cmd", dump_cmd]
        if run_command(cmd) != 0:
            failed = True
    for golden in scope.get("handle_goldens", []):
        cmd = [sys.executable, "harness/handle_score.py", "--goldens", golden, "--adapter-cmd", adapter_cmd]
        if run_command(cmd) != 0:
            failed = True
    for benchmark in scope.get("handle_benchmarks", []):
        cmd = [
            sys.executable,
            "harness/handle_benchmark.py",
            "--targets",
            benchmark,
            "--adapter-cmd",
            benchmark_adapter_cmd,
        ]
        if run_command(cmd) != 0:
            failed = True
    for benchmark in scope.get("benchmarks", []):
        cmd = [sys.executable, "harness/benchmark.py", "--targets", benchmark, "--dump-cmd", dump_cmd]
        if run_command(cmd) != 0:
            failed = True
    # contract_lints, source_audit, and cross_check are language-specific
    # (regex patterns against generated source, audit lists of language-specific
    # files, and cmd paths to language-specific binaries). They live under
    # targets[target].* with a fallback to scope-level for backward compat.
    contract_lints = target_info.get("contract_lints") or scope.get("contract_lints")
    if contract_lints:
        cmd = [
            sys.executable,
            "harness/contract_lint.py",
            "--rules",
            contract_lints,
            "--target",
            target,
            "--graph",
            str(graph_path),
        ]
        for node_id in scope.get("required_nodes", []):
            cmd.extend(["--node", node_id])
        if run_command(cmd) != 0:
            failed = True
    if target_info.get("source_audit") or scope.get("source_audit"):
        cmd = [
            sys.executable,
            "harness/source_audit.py",
            "--graph",
            str(graph_path),
            "--scope",
            scope_name,
            "--target",
            target,
        ]
        if run_command(cmd) != 0:
            failed = True
    if scope.get("contract_xref"):
        cmd = [sys.executable, "harness/contract_xref.py"]
        if scope.get("contract_xref") == "strict":
            cmd.append("--strict")
        if run_command(cmd) != 0:
            failed = True
        # Field-vocabulary guard: no spec-deprecated/out-of-scope/fabricated field
        # token may appear in a live document-model field position (strict AOUSD
        # §7.6 conformance). Runs alongside contract_xref as a global contract check.
        if run_command([sys.executable, "harness/spec_field_audit.py"]) != 0:
            failed = True
    cross_check_cfg = target_info.get("cross_check") or scope.get("cross_check")
    if cross_check_cfg:
        if run_cross_check(cross_check_cfg, dump_cmd, target) != 0:
            failed = True
    if scope.get("cross_format_check"):
        if run_cross_format_check(scope, dump_cmd) != 0:
            failed = True
    if scope.get("filename_rename_check"):
        if run_filename_rename_check(scope, dump_cmd) != 0:
            failed = True
    if scope.get("package_equivalence_check"):
        if run_package_equivalence_check(scope, dump_cmd) != 0:
            failed = True
    if scope.get("usda_rename_invariance"):
        if run_usda_rename_invariance(scope, dump_cmd) != 0:
            failed = True
    if scope.get("usdc_rename_invariance"):
        if run_usdc_rename_invariance(scope, dump_cmd) != 0:
            failed = True
    if scope.get("synthesized_scenes"):
        if run_synthesized_scenes(scope, dump_cmd) != 0:
            failed = True
    return 1 if failed else 0


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _canonicalize(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(v) for v in value]
    return value


def _run_dump_layer(dump_cmd: str, fixture: str) -> Any:
    argv = split_command(dump_cmd) + [fixture]
    ensure_executable(argv)
    result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"dump_layer command failed for {fixture}: {result.stderr}")
    return json.loads(result.stdout)


def _run_handle_adapter(adapter_cmd: str, fixture: str, op: str) -> Any:
    case_input = {
        "ops": [
            {
                "op": op,
                "display": fixture,
                "opened_resource": {"ok": True, "diagnostics": [], "fixture": fixture, "file_backed_path": fixture},
            }
        ]
    }
    raw = Path(fixture).read_bytes()
    case_input["ops"][0]["opened_resource"] = {
        "ok": True,
        "diagnostics": [],
        "bytes_hex": raw.hex(),
        "file_backed_path": fixture,
    }
    with tempfile.TemporaryDirectory(prefix="usd-cross-check-") as td:
        input_path = Path(td) / "input.json"
        input_path.write_text(json.dumps(case_input, indent=2), encoding="utf-8")
        argv = split_command(adapter_cmd) + [str(input_path)]
        ensure_executable(argv)
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"handle adapter command failed for {fixture}: {result.stderr}")
        envelope = json.loads(result.stdout)
        results = envelope.get("results", [])
        if not results:
            raise RuntimeError(f"handle adapter returned no results for {fixture}")
        return results[0].get("dump")


def run_cross_check(cc: dict[str, Any], dump_cmd: str | None, target: str) -> int:
    """Diff dump_layer.exe vs the handle adapter on each fixture.

    The two pipelines must produce identical layer JSON. A mismatch surfaces
    when the adapter shortcuts around the production decoder -- typically
    by hard-coding a fixture-name-keyed answer.
    """
    left = cc.get("left", {})
    right = cc.get("right", {})
    fixtures = cc.get("fixtures", [])
    if not dump_cmd:
        print("cross_check: scope has no dump_cmd; skipping", file=sys.stderr)
        return 1
    print(f"cross_check: {len(fixtures)} fixture(s) -- left={left.get('cmd')} right={right.get('cmd')}")
    diffs = 0
    for fixture in fixtures:
        try:
            left_dump = _run_dump_layer(dump_cmd, fixture)
            right_dump = _run_handle_adapter(right.get("cmd", ""), fixture, right.get("op", "open_usdc_layer"))
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {fixture}: {exc}")
            diffs += 1
            continue
        if _canonicalize(left_dump) != _canonicalize(right_dump):
            diffs += 1
            print(f"  DIFF {fixture}: left and right pipelines produced different layer JSON")
        else:
            print(f"  OK   {fixture}")
    if diffs:
        print(f"cross_check: {diffs} diff(s) of {len(fixtures)} fixture(s)", file=sys.stderr)
        return 1
    print(f"cross_check: OK ({len(fixtures)} fixture(s), 0 diff(s))")
    return 0


def _first_diff(a: Any, b: Any, path: str = "$") -> str:
    """Return a short description of the first place where canonicalized JSON
    structures `a` and `b` disagree. Used for cross_format_check + filename_rename_check
    failure reports so the cheat site is immediately legible."""
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        ka, kb = set(a), set(b)
        if ka != kb:
            return f"{path}: keys diverge (only-in-a={sorted(ka - kb)[:3]}, only-in-b={sorted(kb - ka)[:3]})"
        for k in sorted(ka):
            d = _first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return ""
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return ""
    if a != b:
        return f"{path}: {a!r} vs {b!r}"
    return ""


def run_cross_format_check(scope: dict[str, Any], dump_cmd: str | None) -> int:
    """For each matched (usda, usdc) pair declared in scope.cross_format_check.fixture_pairs,
    parse both via dump_cmd and assert the canonicalized dumps are identical.

    The two parsers consume completely different surface representations of the
    same scene. Their canonical dumps must agree because the document model is
    format-neutral (see contracts/document-model-productions/). Any cheat that
    identifies a specific fixture via its surface details (USDA substring scan,
    USDC byte fingerprint) and emits a memorized dump will diverge from what
    the other format's parser produces, failing this check.
    """
    cfc = scope["cross_format_check"]
    pairs = cfc.get("fixture_pairs", [])
    if not dump_cmd:
        print("cross_format_check: scope has no dump_cmd; skipping", file=sys.stderr)
        return 1
    print(f"cross_format_check: {len(pairs)} pair(s)")
    diffs = 0
    for pair in pairs:
        usda, usdc = pair["usda"], pair["usdc"]
        try:
            usda_dump = _run_dump_layer(dump_cmd, usda)
            usdc_dump = _run_dump_layer(dump_cmd, usdc)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {usda} <-> {usdc}: {exc}")
            diffs += 1
            continue
        d = _first_diff(_canonicalize(usda_dump), _canonicalize(usdc_dump))
        if d:
            diffs += 1
            print(f"  DIFF {usda} <-> {usdc}: {d}")
        else:
            print(f"  OK   {usda} <-> {usdc}")
    if diffs:
        print(f"cross_format_check: {diffs} diff(s) of {len(pairs)} pair(s)", file=sys.stderr)
        return 1
    print(f"cross_format_check: OK ({len(pairs)} pair(s), 0 diff(s))")
    return 0


def run_filename_rename_check(scope: dict[str, Any], dump_cmd: str | None) -> int:
    """Copy each fixture to a randomized temp path, run the dump command on
    both the original and the copy, assert identical canonicalized output.

    A parser that fingerprints by filename (e.g. `if path.contains("World")`)
    yields a different dump on a renamed copy of the same bytes. A content-driven
    parser is invariant to the path.
    """
    frc = scope["filename_rename_check"]
    fixtures = frc.get("fixtures", [])
    if not dump_cmd:
        print("filename_rename_check: scope has no dump_cmd; skipping", file=sys.stderr)
        return 1
    print(f"filename_rename_check: {len(fixtures)} fixture(s)")
    diffs = 0
    for fixture in fixtures:
        try:
            original_dump = _run_dump_layer(dump_cmd, fixture)
            src = Path(fixture)
            with tempfile.TemporaryDirectory(prefix="usd-rename-") as td:
                dst = Path(td) / f"renamed_xyz_{abs(hash(fixture)) & 0xFFFFFF:06x}{src.suffix}"
                dst.write_bytes(src.read_bytes())
                renamed_dump = _run_dump_layer(dump_cmd, str(dst))
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {fixture}: {exc}")
            diffs += 1
            continue
        d = _first_diff(_canonicalize(original_dump), _canonicalize(renamed_dump))
        if d:
            diffs += 1
            print(f"  DIFF {fixture} -> renamed: {d}")
        else:
            print(f"  OK   {fixture}")
    if diffs:
        print(f"filename_rename_check: {diffs} diff(s) of {len(fixtures)} fixture(s)", file=sys.stderr)
        return 1
    print(f"filename_rename_check: OK ({len(fixtures)} fixture(s), 0 diff(s))")
    return 0


def run_package_equivalence_check(scope: dict[str, Any], dump_cmd: str | None) -> int:
    """Compare standalone layer fixtures with USDZ package resources that should
    expose the same default or explicit package entry.
    """
    cfg = scope["package_equivalence_check"]
    pairs = cfg.get("pairs", [])
    if not dump_cmd:
        print("package_equivalence_check: scope has no dump_cmd; skipping", file=sys.stderr)
        return 1
    print(f"package_equivalence_check: {len(pairs)} pair(s)")
    diffs = 0
    for pair in pairs:
        layer = pair["layer"]
        package = pair["package"]
        try:
            layer_dump = _run_dump_layer(dump_cmd, layer)
            package_dump = _run_dump_layer(dump_cmd, package)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {layer} <-> {package}: {exc}")
            diffs += 1
            continue
        d = _first_diff(_canonicalize(layer_dump), _canonicalize(package_dump))
        if d:
            diffs += 1
            print(f"  DIFF {layer} <-> {package}: {d}")
        else:
            print(f"  OK   {layer} <-> {package}")
    if diffs:
        print(f"package_equivalence_check: {diffs} diff(s) of {len(pairs)} pair(s)", file=sys.stderr)
        return 1
    print(f"package_equivalence_check: OK ({len(pairs)} pair(s), 0 diff(s))")
    return 0


# ---------------------------------------------------------------------------
# usda_rename_invariance: language-agnostic behavioral check.
#
# For each .usda fixture, generate deterministic-but-random replacement names
# for prim/attribute identifiers AND random replacement values for unique
# float defaults, apply both mutations to BOTH the USDA text AND the parser's
# original dump, run the parser on the renamed text, and assert the resulting
# dump matches the mutated expected dump.
#
# A real USDA parser produces dump structure that's invariant under identifier
# substitution AND that propagates value literals from text to dump verbatim.
# A substring-match canned-literal cheater has hardcoded `contains(text, "<old
# name>")` branches AND a hardcoded dump that bakes in the original numeric
# defaults; both classes of cheat diverge from the mutated expected dump.
#
# Inputs are static (committed fixtures) but the rename mapping and value
# mutations are randomized at validate time using a seed the agent cannot
# know at code-generation time. The agent cannot pre-cheat for random
# identifiers or random values.
# ---------------------------------------------------------------------------

_RENAME_FIRST_CHARS = string.ascii_letters
_RENAME_REST_CHARS = string.ascii_letters + string.digits

# Attribute typenames that may appear as a USDA declaration prefix. The list
# is conservative: matching `<typename> <name> =` against a known typename set
# avoids accidentally treating `<scope-keyword> <prim-name>` as an attribute.
_USDA_ATTR_TYPENAMES = {
    "bool", "uchar", "int", "uint", "int64", "uint64",
    "half", "float", "double",
    "string", "token", "asset",
    "matrix2d", "matrix3d", "matrix4d",
    "quatd", "quatf", "quath",
    "double2", "double3", "double4",
    "float2", "float3", "float4",
    "half2", "half3", "half4",
    "int2", "int3", "int4",
    "point3d", "point3f", "point3h",
    "normal3d", "normal3f", "normal3h",
    "vector3d", "vector3f", "vector3h",
    "color3d", "color3f", "color3h",
    "color4d", "color4f", "color4h",
    "texCoord2d", "texCoord2f", "texCoord2h",
    "texCoord3d", "texCoord3f", "texCoord3h",
    "frame4d", "timecode",
    "rel",
}


def _pick_random_name(rng: random.Random, original: str) -> str:
    """Generate a random replacement identifier of the same length as the
    original. Length-preservation matters for the USDC byte-level rewrite
    in `usdc_rename_invariance` (offsets must stay valid) and is harmless
    for the USDA text rewrite.
    """
    n = len(original)
    if n == 0:
        return ""
    first = rng.choice(_RENAME_FIRST_CHARS)
    if n == 1:
        return first
    rest = "".join(rng.choices(_RENAME_REST_CHARS, k=n - 1))
    return first + rest


def _find_prim_names_in_usda(text: str) -> set[str]:
    """Extract prim/variant identifiers declared in quoted form in USDA."""
    names: set[str] = set()
    for m in re.finditer(
        r'\b(?:def|over|class)(?:\s+[A-Za-z_][A-Za-z0-9_:]*)?\s+"([A-Za-z_][A-Za-z0-9_]*)"',
        text,
    ):
        names.add(m.group(1))
    for m in re.finditer(r'\bvariantSet\s+"([A-Za-z_][A-Za-z0-9_]*)"\s*=', text):
        names.add(m.group(1))
    for m in re.finditer(r'^\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*\{', text, re.MULTILINE):
        names.add(m.group(1))
    return names


def _find_attr_names_in_usda(text: str) -> set[str]:
    """Extract attribute/relationship names declared as `<typename> <name>(=|.timeSamples|.connect)`.

    Skips namespaced names (those containing `:`) -- renaming `primvars:foo`
    to `xyzab:foo` would change schema semantics, not just bytes. Skips names
    that collide with USDA keywords (def, over, class, variantSet) to avoid
    breaking the grammar when the same identifier appears as a prim.
    """
    typename_alt = "|".join(re.escape(t) for t in _USDA_ATTR_TYPENAMES)
    pattern = re.compile(
        rf"^\s*(?:custom\s+)?(?:uniform\s+|varying\s+)?"
        rf"(?:{typename_alt})(?:\[\])?\s+"
        r"([A-Za-z_][A-Za-z0-9_]*)"
        r"(?:\.(?:timeSamples|connect))?\s*=",
        re.MULTILINE,
    )
    reserved = {"def", "over", "class", "variantSet", "custom", "uniform", "varying"}
    names = set()
    for m in pattern.finditer(text):
        name = m.group(1)
        if name not in reserved:
            names.add(name)
    return names


def _find_mutable_float_values(text: str, dump: Any) -> dict[str, float]:
    """Find float literals (with decimal point or scientific notation) that
    appear exactly once in the USDA text AND exactly once as a numeric leaf
    in the dump. These are unambiguously associated with a single attribute,
    so substituting them in both places preserves correctness.

    Returns a dict mapping the literal string form to its parsed float value.
    """
    float_re = re.compile(r"(?<![A-Za-z_\d.])(-?\d+\.\d+(?:[eE][-+]?\d+)?)(?![A-Za-z_\d])")
    text_matches = float_re.findall(text)
    text_count: dict[str, int] = {}
    for v in text_matches:
        text_count[v] = text_count.get(v, 0) + 1

    dump_floats: list[float] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, bool):
            return
        elif isinstance(node, (int, float)):
            dump_floats.append(float(node))

    walk(dump)
    dump_count: dict[float, int] = {}
    for v in dump_floats:
        dump_count[v] = dump_count.get(v, 0) + 1

    result: dict[str, float] = {}
    for val_str, cnt in text_count.items():
        if cnt != 1:
            continue
        try:
            val_f = float(val_str)
        except ValueError:
            continue
        if dump_count.get(val_f, 0) == 1:
            result[val_str] = val_f
    return result


def _pick_random_float(rng: random.Random, original: float) -> float:
    """Generate a random replacement float not equal to the original."""
    for _ in range(10):
        new = round(rng.uniform(-1000.0, 1000.0), 4)
        if new != original:
            return new
    return original + 1.0


def _substitute_in_usda(
    text: str,
    prim_renames: dict[str, str],
    attr_renames: dict[str, str],
    value_subs: dict[str, str],
) -> str:
    """Apply identifier and value mutations to USDA text.

    Prim renames target quoted names and path references. Attribute renames
    target unquoted names with the typename-prefix + `=`/.timeSamples/.connect
    follow-context that USDA uses to disambiguate attribute declarations from
    other identifiers. Value substitutions target unique float literals.
    """
    out = text
    for old, new in sorted(prim_renames.items(), key=lambda kv: -len(kv[0])):
        out = re.sub(rf'"{re.escape(old)}"', f'"{new}"', out)
        out = re.sub(rf'</{re.escape(old)}>', f'</{new}>', out)
        out = re.sub(rf'</{re.escape(old)}(?=[./{{])', f'</{new}', out)
    for old, new in sorted(attr_renames.items(), key=lambda kv: -len(kv[0])):
        # Anchor on a USDA attribute follow-context: `=`, `.timeSamples`, or
        # `.connect`. Also anchor on path-reference contexts (`/<prim>.<attr>`)
        # so connections within </a/b.c> rewrites correctly when the attr name
        # is used elsewhere.
        out = re.sub(
            rf"\b{re.escape(old)}\b(?=\s*(?:=|\.(?:timeSamples|connect)))",
            new,
            out,
        )
        out = re.sub(rf"(?<=\.){re.escape(old)}(?=[>\s\]])", new, out)
    for old, new in sorted(value_subs.items(), key=lambda kv: -len(kv[0])):
        out = re.sub(
            rf"(?<![A-Za-z_\d.]){re.escape(old)}(?![A-Za-z_\d])",
            new,
            out,
        )
    return out


def _substitute_in_dump_string(value: str, rename_map: dict[str, str]) -> str:
    """Apply renames to a string value in the dump. Handles:
      - exact name (e.g., "World" in primChildren array)
      - path component (e.g., "/World" or "/World.attr" or "/World{vset=var}")
    Sorted by length descending to avoid prefix collisions.
    """
    if value in rename_map:
        return rename_map[value]
    out = value
    for old, new in sorted(rename_map.items(), key=lambda kv: -len(kv[0])):
        # After `/` or `.`, bounded by `/` `.` `{` `}` or end of string.
        out = re.sub(rf'(?<=[/.]){re.escape(old)}(?=[/.{{}}]|$)', new, out)
        # Variant labels inside braces.
        out = re.sub(rf'(?<=[{{=]){re.escape(old)}(?=[}}=])', new, out)
    return out


def _substitute_in_dump(
    node: Any,
    rename_map: dict[str, str],
    value_map: dict[float, float] | None = None,
) -> Any:
    """Recursively rewrite name and value occurrences in a parsed dump dict.

    Identifier renames apply to dict keys (paths) and string values; value
    substitutions apply to numeric leaves that match the value_map keys
    exactly (Python float identity).
    """
    if value_map is None:
        value_map = {}
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for k, v in node.items():
            new_k = _substitute_in_dump_string(k, rename_map) if isinstance(k, str) else k
            out[new_k] = _substitute_in_dump(v, rename_map, value_map)
        return out
    if isinstance(node, list):
        return [_substitute_in_dump(item, rename_map, value_map) for item in node]
    if isinstance(node, str):
        return _substitute_in_dump_string(node, rename_map)
    if isinstance(node, bool):
        return node
    if isinstance(node, (int, float)):
        f = float(node)
        if f in value_map:
            return value_map[f]
    return node


def run_usda_rename_invariance(scope: dict[str, Any], dump_cmd: str | None, seed: int | None = None) -> int:
    """Test that the USDA parser is content-driven rather than fingerprinting.

    For each fixture: rename prims/variants/attributes to random identifiers
    AND mutate unique float defaults to random values, parse the mutated
    text, expect the same dump structure (modulo the substitutions). A
    substring-match cheater or canned-dump cheater fails; a real parser
    passes.
    """
    if not dump_cmd:
        print("usda_rename_invariance: scope has no dump_cmd; skipping", file=sys.stderr)
        return 1
    cfg = scope["usda_rename_invariance"]
    fixtures = cfg.get("fixtures", [])
    if seed is None:
        seed = cfg.get("seed", random.randint(0, 999999))
    rng = random.Random(seed)
    print(f"usda_rename_invariance: {len(fixtures)} fixture(s); seed={seed}")
    diffs = 0
    for fixture in fixtures:
        original_text = Path(fixture).read_text(encoding="utf-8")
        prim_names = _find_prim_names_in_usda(original_text)
        attr_names = _find_attr_names_in_usda(original_text)
        # Avoid double-renaming: if an identifier appears as both a prim and
        # an attr in the same file, keep the prim entry only (it's the more
        # constraining context).
        attr_names -= prim_names
        if not prim_names and not attr_names:
            print(f"  SKIP {fixture} (no identifiers to rename)")
            continue

        try:
            original_dump = _run_dump_layer(dump_cmd, fixture)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {fixture}: original dump failed: {exc}")
            diffs += 1
            continue

        prim_rename = {n: _pick_random_name(rng, n) for n in sorted(prim_names)}
        attr_rename = {n: _pick_random_name(rng, n) for n in sorted(attr_names)}
        # Combined rename map for dump-side substitution: the dump format is
        # uniform across identifier kinds (paths, primChildren entries,
        # property names), so the same mutation table applies to both kinds.
        combined_renames = {**prim_rename, **attr_rename}

        # Value mutation: only floats that appear uniquely in both the text
        # and the dump. Tie-breaking by uniqueness keeps this safe under
        # complex fixtures.
        float_candidates = _find_mutable_float_values(original_text, original_dump)
        value_text_subs: dict[str, str] = {}
        value_dump_subs: dict[float, float] = {}
        for val_str, val_f in float_candidates.items():
            new_f = _pick_random_float(rng, val_f)
            new_s = f"{new_f:.4f}".rstrip("0").rstrip(".")
            if "." not in new_s:
                new_s += ".0"  # preserve float-ness in USDA text
            value_text_subs[val_str] = new_s
            value_dump_subs[val_f] = new_f

        mutated_text = _substitute_in_usda(
            original_text, prim_rename, attr_rename, value_text_subs
        )

        with tempfile.TemporaryDirectory(prefix="usda-rename-") as td:
            mutated_path = Path(td) / Path(fixture).name
            mutated_path.write_text(mutated_text, encoding="utf-8")
            try:
                mutated_dump = _run_dump_layer(dump_cmd, str(mutated_path))
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL {fixture}: mutated dump failed: {exc}")
                diffs += 1
                continue

        expected_dump = _substitute_in_dump(original_dump, combined_renames, value_dump_subs)
        d = _first_diff(_canonicalize(mutated_dump), _canonicalize(expected_dump))
        if d:
            diffs += 1
            mut_count = len(combined_renames) + len(value_text_subs)
            details: list[str] = []
            for k, v in list(combined_renames.items())[:2]:
                details.append(f"{k}->{v}")
            for k, v in list(value_text_subs.items())[:2]:
                details.append(f"{k}->{v}")
            if mut_count > len(details):
                details.append(f"... (+{mut_count - len(details)} more)")
            print(f"  DIFF {fixture} ({', '.join(details)}): {d}")
        else:
            n_ids = len(combined_renames)
            n_vals = len(value_text_subs)
            print(f"  OK   {fixture} ({n_ids} ident, {n_vals} value)")
    if diffs:
        print(
            f"usda_rename_invariance: {diffs} diff(s) of {len(fixtures)} fixture(s); seed={seed}",
            file=sys.stderr,
        )
        return 1
    print(f"usda_rename_invariance: OK ({len(fixtures)} fixture(s), seed={seed})")
    return 0


# ---------------------------------------------------------------------------
# usdc_rename_invariance: language-agnostic byte-level behavioral check.
#
# For each .usdc fixture paired with a .usda sibling, extract the prim and
# variant names from the USDA, pick same-length random replacements, and
# rewrite the USDC bytes via NUL-bounded byte substitution (`\0<name>\0` ->
# `\0<new>\0`). The mutated USDC is parsed and the resulting dump must
# match the original dump with the same rename applied.
#
# The byte-level mutation is the USDC analog of usda_rename_invariance and
# closes the cheat shape that cross_format_check cannot reach: a USDC
# parser that fingerprints by byte content. Mutated bytes have the same
# document-model structure as the original (only token identity changes);
# a real parser produces a structurally identical dump with renamed names,
# while a byte-fingerprint cheater diverges.
#
# Limitation: tokens encoded by LZ4 back-references rather than literal
# bytes won't be found by the NUL-bounded search and are silently skipped
# per token. Fixtures where ALL tokens are LZ4-compressed-away are skipped
# entirely with a clear warning. For our typical 1-5 KB fixtures, LZ4
# emits most token strings as literals, so this catches the common case.
# ---------------------------------------------------------------------------


def _find_nul_bounded_token(data: bytes, name: str) -> list[int]:
    """Find offsets where `\\0<name>\\0` appears in the bytes. Returns the
    offsets of the START of the name (one past the leading NUL).
    """
    pattern = b"\0" + name.encode("utf-8") + b"\0"
    offsets: list[int] = []
    i = 0
    while True:
        idx = data.find(pattern, i)
        if idx < 0:
            break
        offsets.append(idx + 1)
        i = idx + 1
    return offsets


def _mutate_usdc_bytes(
    data: bytes, rename_map: dict[str, str]
) -> tuple[bytes, dict[str, str]]:
    """Apply same-length NUL-bounded byte replacements per rename_map.

    Returns (mutated_bytes, applied_map). Names whose bytes aren't visible
    as raw literals (e.g., LZ4 back-references) are dropped from applied_map.
    Length mismatches in the rename_map are also dropped to keep offsets stable.
    """
    out = bytearray(data)
    applied: dict[str, str] = {}
    for old, new in rename_map.items():
        if len(old) != len(new):
            continue
        offsets = _find_nul_bounded_token(bytes(out), old)
        if not offsets:
            continue
        new_bytes = new.encode("utf-8")
        for off in offsets:
            out[off : off + len(new_bytes)] = new_bytes
        applied[old] = new
    return bytes(out), applied


def run_usdc_rename_invariance(
    scope: dict[str, Any], dump_cmd: str | None, seed: int | None = None
) -> int:
    """Test that the USDC parser is content-driven on the byte level.

    For each (usdc, usda) pair: extract prim/variant names from the USDA
    sibling, pick same-length random replacements, rewrite the USDC bytes
    via NUL-bounded substitution, parse the mutated bytes, and assert the
    dump equals the original dump with the rename applied.
    """
    if not dump_cmd:
        print("usdc_rename_invariance: scope has no dump_cmd; skipping", file=sys.stderr)
        return 1
    cfg = scope["usdc_rename_invariance"]
    pairs = cfg.get("fixture_pairs", [])
    if seed is None:
        seed = cfg.get("seed", random.randint(0, 999999))
    rng = random.Random(seed)
    print(f"usdc_rename_invariance: {len(pairs)} fixture pair(s); seed={seed}")
    diffs = 0
    for pair in pairs:
        usdc_path = pair["usdc"]
        usda_path = pair["usda"]
        usda_text = Path(usda_path).read_text(encoding="utf-8")
        names = _find_prim_names_in_usda(usda_text)
        if not names:
            print(f"  SKIP {usdc_path} (no identifiers in sibling {usda_path})")
            continue

        rename_map = {n: _pick_random_name(rng, n) for n in sorted(names)}
        data = Path(usdc_path).read_bytes()
        mutated_data, applied = _mutate_usdc_bytes(data, rename_map)
        if not applied:
            print(
                f"  SKIP {usdc_path} (no token bytes visible as raw literals; "
                f"likely LZ4 back-referenced)"
            )
            continue

        try:
            original_dump = _run_dump_layer(dump_cmd, usdc_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {usdc_path}: original dump failed: {exc}")
            diffs += 1
            continue

        with tempfile.TemporaryDirectory(prefix="usdc-rename-") as td:
            mutated_path = Path(td) / Path(usdc_path).name
            mutated_path.write_bytes(mutated_data)
            try:
                mutated_dump = _run_dump_layer(dump_cmd, str(mutated_path))
            except Exception as exc:  # noqa: BLE001
                # Byte mutation may have produced an invalid file when LZ4
                # match-windows include the mutated region. That's a check
                # limitation, not a failure of the parser under test.
                print(
                    f"  SKIP {usdc_path}: mutated parse failed "
                    f"(LZ4 match-window may include rewritten token bytes): {exc}"
                )
                continue

        expected_dump = _substitute_in_dump(original_dump, applied)
        d = _first_diff(_canonicalize(mutated_dump), _canonicalize(expected_dump))
        if d:
            diffs += 1
            short = ", ".join(f"{k}->{v}" for k, v in list(applied.items())[:3])
            if len(applied) > 3:
                short += f", ... (+{len(applied) - 3} more)"
            print(f"  DIFF {usdc_path} ({short}): {d}")
        else:
            print(f"  OK   {usdc_path} ({len(applied)} token(s) replaced)")
    if diffs:
        print(
            f"usdc_rename_invariance: {diffs} diff(s) of {len(pairs)} pair(s); seed={seed}",
            file=sys.stderr,
        )
        return 1
    print(f"usdc_rename_invariance: OK ({len(pairs)} pair(s), seed={seed})")
    return 0


# ---------------------------------------------------------------------------
# synthesized_scenes: generate USDA fixtures at validate time and assert the
# parser's dump matches a separately-computed expected dump.
#
# This is the only behavioral check whose inputs are NOT visible to the regen
# agent at code-generation time -- scenes are randomized per validate run.
# A canned-dump cheater has nothing to fingerprint and must implement a real
# parser to pass.
#
# Two emitter implementations live in `harness/scene_synthesis.py`:
#  - `PureStdLibEmitter` (always available; f-string USDA + hand-coded dump)
#  - `UsdCoreEmitter` (used when `pxr` is importable; SdfLayer-backed)
#
# `get_emitter()` selects usd-core when available, else the pure-stdlib
# fallback. When usd-core is available, a parity check at the start of the
# run asserts both emitters compute the same expected dump for a sample
# scene; this catches drift between the two implementations.
# ---------------------------------------------------------------------------


def run_synthesized_scenes(
    scope: dict[str, Any], dump_cmd: str | None, seed: int | None = None
) -> int:
    """Generate random USDA scenes and assert the parser dump matches the
    harness's expected dump computed from the same Scene object.
    """
    # Import inside the function so the harness still loads when scene
    # synthesis dependencies are missing (e.g., during early bootstrap).
    from scene_synthesis import (
        generate_scene,
        get_emitter,
        PureStdLibEmitter,
        UsdCoreEmitter,
        usdcore_available,
    )

    if not dump_cmd:
        print("synthesized_scenes: scope has no dump_cmd; skipping", file=sys.stderr)
        return 1
    cfg = scope["synthesized_scenes"]
    n_scenes = int(cfg.get("count", 20))
    max_depth = int(cfg.get("max_depth", 3))
    n_roots_min = int(cfg.get("n_roots_min", 1))
    n_roots_max = int(cfg.get("n_roots_max", 4))
    if seed is None:
        seed = cfg.get("seed", random.randint(0, 999999))

    emitter = get_emitter()
    usdc_synthesis = bool(getattr(emitter, "supports_usdc", False))
    print(
        f"synthesized_scenes: {n_scenes} scene(s); seed={seed}; "
        f"emitter={emitter.name}; usd_core_available={usdcore_available()}; "
        f"usdc_synthesis={'enabled' if usdc_synthesis else 'DISABLED'}"
    )
    if not usdc_synthesis:
        # The pure-stdlib path catches USDA parser cheats fully but leaves
        # the USDC parser tested only by the committed-fixture gates
        # (cross_format_check, usdc_rename_invariance). For the strongest
        # coverage, install usd-core so the harness can emit novel USDC
        # bytes and exercise the binary parser on inputs the regen agent
        # has never seen.
        print(
            "  HINT: install usd-core (e.g. into ~/.venv/<name>-python3.12 "
            "and run the harness via that venv's python) to enable USDC "
            "synthesis. Without it, USDC parsing is tested only on "
            "committed fixtures.",
            file=sys.stderr,
        )

    # When usd-core is available, parity-check the two emitters once on a
    # representative scene to catch drift between implementations.
    if usdcore_available():
        parity_rng = random.Random(seed)
        parity_scene = generate_scene(
            parity_rng,
            n_roots_range=(n_roots_min, n_roots_max),
            max_depth=max_depth,
        )
        ps_dump = _canonicalize(PureStdLibEmitter().expected_dump(parity_scene))
        uc_dump = _canonicalize(UsdCoreEmitter().expected_dump(parity_scene))
        if ps_dump != uc_dump:
            d = _first_diff(ps_dump, uc_dump)
            print(
                f"synthesized_scenes: EMITTER PARITY VIOLATION: PureStdLib "
                f"and UsdCore disagree on a sample scene. {d}",
                file=sys.stderr,
            )
            return 1
        print("  emitter_parity: OK (PureStdLib == UsdCore on sample scene)")

    rng = random.Random(seed)
    diffs = 0
    total_parses = 0
    for i in range(n_scenes):
        scene = generate_scene(
            rng,
            n_roots_range=(n_roots_min, n_roots_max),
            max_depth=max_depth,
        )
        expected = emitter.expected_dump(scene)
        n_roots = len(scene.prims)
        n_total_prims = _count_prims(scene.prims)
        n_attrs = _count_attrs(scene.prims)
        scene_shape = f"({n_roots} root(s), {n_total_prims} prim(s), {n_attrs} attr(s))"

        # Per-format checks. Each tests the same Scene through a different
        # binding (USDA text vs USDC bytes) against the same expected dump.
        formats: list[tuple[str, bytes | None, str]] = [
            ("usda", emitter.emit_usda(scene).encode("utf-8"), ".usda"),
        ]
        if usdc_synthesis:
            formats.append(("usdc", emitter.emit_usdc(scene), ".usdc"))

        with tempfile.TemporaryDirectory(prefix="synth-scene-") as td:
            for fmt_name, fmt_bytes, suffix in formats:
                total_parses += 1
                scene_path = Path(td) / f"scene_{i:04d}{suffix}"
                scene_path.write_bytes(fmt_bytes)
                tag = f"scene_{i:04d}/{fmt_name}"
                try:
                    got = _run_dump_layer(dump_cmd, str(scene_path))
                except Exception as exc:  # noqa: BLE001
                    diffs += 1
                    print(f"  FAIL {tag} {scene_shape}: parser failed: {exc}")
                    if fmt_name == "usda" and diffs <= 1:
                        print(f"    USDA was:\n{fmt_bytes.decode('utf-8', 'replace')}")
                    continue
                d = _first_diff(_canonicalize(got), _canonicalize(expected))
                if d:
                    diffs += 1
                    print(f"  DIFF {tag} {scene_shape}: {d}")
                    if fmt_name == "usda" and diffs <= 1:
                        usda_text = fmt_bytes.decode("utf-8", "replace")
                        indented = "\n".join(
                            "    | " + line for line in usda_text.splitlines()
                        )
                        print(f"    USDA was:\n{indented}")
                else:
                    print(f"  OK   {tag}")
    if diffs:
        print(
            f"synthesized_scenes: {diffs} diff(s) of {total_parses} parse(s) "
            f"across {n_scenes} scene(s); seed={seed}",
            file=sys.stderr,
        )
        return 1
    print(
        f"synthesized_scenes: OK ({n_scenes} scene(s), {total_parses} parse(s), "
        f"seed={seed})"
    )
    return 0


def _count_prims(prims: list[Any]) -> int:
    n = len(prims)
    for p in prims:
        n += _count_prims(p.children)
    return n


def _count_attrs(prims: list[Any]) -> int:
    n = 0
    for p in prims:
        n += len(p.attributes)
        n += _count_attrs(p.children)
    return n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="graph/skillgraph.json", type=Path)
    root = parser.add_mutually_exclusive_group(required=True)
    root.add_argument("--scope")
    root.add_argument("--node")
    parser.add_argument("--target", default="python")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--record-existing", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    repo = Path.cwd()
    graph = load_json(repo / args.graph)
    validate_consumes(graph)
    root_name, roots = roots_from_args(graph, args)
    order = topo_closure(graph, roots)
    fingerprints = node_fingerprints(repo, graph, order)

    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = repo / "generated" / args.target / ".skillgraph" / "manifest.json"
    elif not manifest_path.is_absolute():
        manifest_path = repo / manifest_path
    manifest = load_manifest(manifest_path)

    plan: list[dict[str, Any]] = []
    for node_id in order:
        node = graph["nodes"][node_id]
        status, reasons = status_for_node(repo, node_id, node, args.target, manifest, fingerprints)
        artifacts = [
            str(path.relative_to(repo)).replace("\\", "/")
            for path in node_artifacts(repo, node, args.target)
        ]
        plan.append(
            {
                "node": node_id,
                "depends_on": node.get("depends_on", []),
                "consumes": node.get("consumes", []),
                "artifacts": artifacts,
                "status": status,
                "reasons": reasons,
            }
        )

    print(f"Root: {root_name}")
    print(f"Target: {args.target}")
    print(f"Manifest: {manifest_path.relative_to(repo)}")
    print_plan(plan)
    sys.stdout.flush()

    if args.record_existing:
        manifest = build_manifest(repo, graph, order, args.target, fingerprints, manifest)
        dump_json(manifest_path, manifest)
        print(f"Recorded manifest: {manifest_path.relative_to(repo)}")
        for item in plan:
            if item["status"] == "missing":
                print("Manifest is partial because at least one artifact is missing.", file=sys.stderr)
                return 2

    if args.validate:
        if not args.scope:
            print("--validate requires --scope so the harness knows which goldens to run", file=sys.stderr)
            return 2
        blocking = [item for item in plan if item["status"] == "missing"]
        if blocking:
            print("Cannot validate while artifacts are missing.", file=sys.stderr)
            return 2
        return validate_scope(repo, graph, args.scope, args.target, args.graph)

    return 0


if __name__ == "__main__":
    sys.exit(main())
