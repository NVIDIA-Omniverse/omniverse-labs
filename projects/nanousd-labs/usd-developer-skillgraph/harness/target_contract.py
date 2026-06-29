#!/usr/bin/env python3
"""Validate generated target command contracts before scoring.

The contract gate is intentionally language-profile based. It prevents a
generated target from satisfying a command interface by delegating to an oracle
runtime or a prebuilt executable that the harness did not build.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from command_utils import format_command, split_command


CPP_MAIN_SOURCE_RE = re.compile(r"(?:^dump_layer|_adapter|_perf_adapter)\.cpp$")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def topo_closure(graph: dict[str, Any], roots: list[str]) -> list[str]:
    nodes = graph["nodes"]
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
        visiting.remove(node_id)
        seen.add(node_id)
        order.append(node_id)

    for root in roots:
        visit(root)
    return order


def roots_for_scope(graph: dict[str, Any], scope_name: str) -> list[str]:
    return list(graph["scopes"][scope_name]["required_nodes"])


def relpath(repo: Path, path: Path) -> str:
    return str(path.relative_to(repo)).replace(os.sep, "/")


def expand_globs(repo: Path, patterns: list[str]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(repo.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                result.append(path)
    return result


def command_executable(command: str | None) -> str | None:
    if not command:
        return None
    argv = split_command(command)
    return argv[0] if argv else None


def collect_scope_entrypoints(
    graph: dict[str, Any], scope_name: str, target: str
) -> list[str]:
    scope = graph["scopes"][scope_name]
    target_info = scope.get("targets", {}).get(target, {})
    commands: list[str] = []
    for key in ("dump_cmd", "adapter_cmd", "benchmark_adapter_cmd", "entrypoint"):
        exe = command_executable(target_info.get(key))
        if exe:
            commands.append(exe)

    cross_check = target_info.get("cross_check") or scope.get("cross_check")
    if isinstance(cross_check, dict):
        for side in ("left", "right"):
            exe = command_executable(cross_check.get(side, {}).get("cmd"))
            if exe:
                commands.append(exe)

    # Keep any explicitly declared command outputs source-built as well.
    for node_id in topo_closure(graph, roots_for_scope(graph, scope_name)):
        node = graph["nodes"][node_id]
        for artifact in node.get("targets", {}).get(target, {}).get("build_outputs", []):
            if artifact.endswith(".exe"):
                commands.append(artifact)

    out: list[str] = []
    seen: set[str] = set()
    for item in commands:
        norm = item.replace("\\", "/")
        if norm not in seen and norm.startswith("generated/"):
            seen.add(norm)
            out.append(norm)
    return out


def find_forbidden_files(repo: Path, contract: dict[str, Any]) -> list[str]:
    patterns = contract.get("forbidden_file_globs", [])
    hits = [relpath(repo, p) for p in expand_globs(repo, patterns)]
    generated_dir = repo / "generated" / "cpp"
    if generated_dir.exists():
        for path in generated_dir.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".pyc", ".sh"}:
                rel = relpath(repo, path)
                if rel not in hits:
                    hits.append(rel)
    return sorted(hits)


def token_pattern(token: str) -> re.Pattern[str]:
    escaped = re.escape(token)
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", token):
        return re.compile(r"\b" + escaped + r"\b")
    return re.compile(escaped)


def scan_sources(
    repo: Path,
    sources: list[Path],
    forbidden_runtime_apis: list[str],
    oracle_denylist: list[str],
) -> list[str]:
    patterns = [(f"runtime:{s}", token_pattern(s)) for s in forbidden_runtime_apis]
    patterns.extend((f"oracle:{s}", token_pattern(s)) for s in oracle_denylist)
    patterns.extend(
        [
            ("repo-read:goldens", re.compile(r"goldens/|goldens\\\\")),
            ("repo-read:actual-reports", re.compile(r"reports/actual|reports\\\\actual")),
        ]
    )

    findings: list[str] = []
    for path in sources:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"{relpath(repo, path)}: not valid UTF-8 source")
            continue
        for label, pattern in patterns:
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{relpath(repo, path)}:{line}: forbidden {label}")
    return findings


def cpp_main_source_for(repo: Path, output_rel: str) -> Path:
    output = repo / output_rel
    name = output.name
    if name.endswith(".exe"):
        name = name[:-4]
    return output.with_name(name + ".cpp")


def cpp_production_sources(sources: list[Path]) -> list[Path]:
    return [
        p
        for p in sources
        if p.suffix == ".cpp" and not CPP_MAIN_SOURCE_RE.search(p.name)
    ]


def run_checked(argv: list[str], cwd: Path) -> tuple[int, str, str]:
    result = subprocess.run(
        argv,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    return result.returncode, result.stdout, result.stderr


def inspect_binary(
    repo: Path,
    output: Path,
    oracle_denylist: list[str],
    forbidden_runtime_apis: list[str],
) -> list[str]:
    findings: list[str] = []
    rel = relpath(repo, output)
    deny_patterns = [(f"oracle:{s}", token_pattern(s)) for s in oracle_denylist]
    deny_patterns.extend((f"runtime:{s}", token_pattern(s)) for s in forbidden_runtime_apis)

    # GNU tools (ldd/nm/objdump) cover Linux and MinGW; dumpbin covers MSVC on
    # Windows. Each is skipped when not on PATH, so whichever toolchain built the
    # binary, the available introspector runs.
    ran_any = False
    for tool in ("ldd", "nm", "objdump", "dumpbin"):
        if shutil.which(tool) is None:
            continue
        ran_any = True
        if tool == "ldd":
            argv = ["ldd", str(output)]
        elif tool == "nm":
            argv = ["nm", "-D", str(output)]
        elif tool == "objdump":
            argv = ["objdump", "-T", str(output)]
        else:
            # MSVC: /imports lists imported DLLs and the symbols pulled from each.
            argv = ["dumpbin", "/nologo", "/imports", str(output)]
        code, stdout, stderr = run_checked(argv, repo)
        text = stdout + "\n" + stderr
        # `nm -D` can return non-zero for fully static/no-symbol binaries; the
        # useful part for us is still the text it produced.
        if tool in {"ldd", "objdump", "dumpbin"} and code not in {0, 1}:
            findings.append(f"{rel}: {tool} failed with exit {code}")
        for label, pattern in deny_patterns:
            if pattern.search(text):
                findings.append(f"{rel}: forbidden binary dependency/symbol {label} found by {tool}")
    if not ran_any:
        print(
            f"target_contract: WARNING binary audit skipped for {rel}: no binary "
            "introspector on PATH (looked for ldd, nm, objdump, dumpbin); "
            "forbidden-dependency / forbidden-runtime-API enforcement was not performed",
            file=sys.stderr,
        )
    return findings


DEFAULT_GCC_FLAGS = ["-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-Igenerated/cpp"]


def _resolve_compiler(name: str) -> str | None:
    """Resolve a compiler name to a path: absolute paths literally, bare names via PATH."""
    if Path(name).is_absolute():
        return name if Path(name).exists() else None
    return shutil.which(name)


def select_toolchain(contract: dict[str, Any]) -> tuple[str | None, str, list[str], list[str]]:
    """Pick the first configured toolchain whose compiler is on PATH.

    Returns (compiler_path, style, compile_flags, tried_names). ``style`` is
    ``"gcc"`` (GCC/Clang flag dialect, ``-o`` output) or ``"msvc"`` (MSVC dialect,
    ``/Fe:`` exe + ``/Fo:`` object dir). Each toolchain lists candidate compiler
    names tried in order, so the same contract builds on Linux (g++/clang++) and
    Windows (cl, or a MinGW g++). Falls back to the legacy single
    ``compiler``/``compile_flags`` keys when ``toolchains`` is absent.
    """
    toolchains = contract.get("toolchains")
    if not toolchains:
        toolchains = [{
            "style": "gcc",
            "compilers": [contract.get("compiler", "g++")],
            "compile_flags": contract.get("compile_flags", DEFAULT_GCC_FLAGS),
        }]
    tried: list[str] = []
    for tc in toolchains:
        style = tc.get("style", "gcc")
        flags = tc.get("compile_flags", DEFAULT_GCC_FLAGS)
        for name in tc.get("compilers", []):
            tried.append(name)
            path = _resolve_compiler(name)
            if path:
                return path, style, flags, tried
    return None, "gcc", [], tried


def build_cpp_native(
    repo: Path,
    graph: dict[str, Any],
    scope_name: str,
    target: str,
    contract: dict[str, Any],
    sources: list[Path],
) -> list[str]:
    compiler_path, style, flags, tried = select_toolchain(contract)
    if not compiler_path:
        return [f"no configured compiler found on PATH (tried: {', '.join(tried) or 'none'})"]

    entrypoints = collect_scope_entrypoints(graph, scope_name, target)
    if not entrypoints:
        return [f"{scope_name}/{target}: command contract found no entrypoints to build"]

    production = cpp_production_sources(sources)
    findings: list[str] = []
    produced: list[Path] = []

    print("target_contract: cpp-native provenance")
    print(f"  compiler: {compiler_path} (toolchain: {style})")
    print("  sources:")
    for src in sources:
        print(f"    {relpath(repo, src)}")

    for output_rel in entrypoints:
        output = repo / output_rel
        main_src = cpp_main_source_for(repo, output_rel)
        if not main_src.exists():
            findings.append(f"{output_rel}: missing main source {relpath(repo, main_src)}")
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            if output.exists():
                output.unlink()
        except OSError as exc:
            findings.append(f"{output_rel}: failed to remove stale output: {exc}")
            continue

        compile_sources = [main_src] + [p for p in production if p != main_src]
        src_args = [relpath(repo, p) for p in compile_sources]
        if style == "msvc":
            # MSVC: /Fe: names the exe, /Fo: redirects .obj files into a temp dir
            # (cl scatters them in cwd otherwise); the objs are not needed past link.
            with tempfile.TemporaryDirectory(prefix="target-contract-obj-") as objdir:
                argv = [str(compiler_path), *flags, f"/Fo:{objdir}{os.sep}", *src_args, f"/Fe:{output_rel}"]
                print(f"  build: {format_command(argv)}")
                code, stdout, stderr = run_checked(argv, repo)
        else:
            argv = [str(compiler_path), *flags, *src_args, "-o", output_rel]
            print(f"  build: {format_command(argv)}")
            code, stdout, stderr = run_checked(argv, repo)
        if stdout:
            print(stdout, end="")
        if stderr:
            print(stderr, end="", file=sys.stderr)
        if code != 0:
            findings.append(f"{output_rel}: compile failed with exit {code}")
            continue
        produced.append(output)
        mode = output.stat().st_mode
        output.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        findings.extend(
            inspect_binary(
                repo,
                output,
                contract.get("oracle_dependency_denylist", []),
                contract.get("forbidden_runtime_apis", []),
            )
        )

    if contract.get("reject_extra_executables", False):
        allowed = {p.resolve() for p in produced}
        for path in (repo / "generated" / target).glob("*.exe"):
            if path.resolve() not in allowed:
                findings.append(f"{relpath(repo, path)}: extra executable not produced by this contract")

    return findings


def validate_contract(repo: Path, graph_path: Path, scope_name: str, target: str) -> int:
    graph = load_json(graph_path)
    scope = graph["scopes"][scope_name]
    target_info = scope.get("targets", {}).get(target, {})
    raw_contract = target_info.get("command_contract")
    if not raw_contract:
        print(f"target_contract: SKIP {scope_name}/{target} (no command_contract)")
        return 0
    if isinstance(raw_contract, str):
        contracts = graph.get("command_contracts", {})
        contract = contracts.get(raw_contract)
        if not isinstance(contract, dict):
            print(
                f"target_contract: FAIL unknown command_contract {raw_contract!r}",
                file=sys.stderr,
            )
            return 1
    elif isinstance(raw_contract, dict):
        contract = raw_contract
    else:
        print(f"target_contract: FAIL malformed command_contract for {scope_name}/{target}", file=sys.stderr)
        return 1

    profile = contract.get("profile")
    source_globs = contract.get("source_globs", [])
    sources = expand_globs(repo, source_globs)
    findings: list[str] = []

    print(f"target_contract: scope={scope_name} target={target} profile={profile}")

    if not sources:
        findings.append(f"{scope_name}/{target}: no generated sources matched {source_globs}")
    forbidden_files = find_forbidden_files(repo, contract)
    findings.extend(f"{item}: forbidden generated runtime/helper file" for item in forbidden_files)
    if not findings:
        findings.extend(
            scan_sources(
                repo,
                sources,
                contract.get("forbidden_runtime_apis", []),
                contract.get("oracle_dependency_denylist", []),
            )
        )

    if not findings:
        if profile == "cpp-native":
            findings.extend(build_cpp_native(repo, graph, scope_name, target, contract, sources))
        else:
            findings.append(f"{scope_name}/{target}: unsupported command contract profile {profile!r}")

    if findings:
        print(f"target_contract: FAIL ({len(findings)} finding(s))", file=sys.stderr)
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        return 1
    print("target_contract: OK")
    return 0


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="target-contract-selftest-") as td:
        repo = Path(td)
        generated = repo / "generated" / "cpp"
        generated.mkdir(parents=True)
        (generated / "runtime.py").write_text("print('oracle')\n", encoding="utf-8")
        contract = {
            "forbidden_file_globs": ["generated/cpp/*.py"],
            "forbidden_runtime_apis": ["exec", "system", "popen", "posix_spawn", "fork", "dlopen", "dlsym", "Py_", "python"],
            "oracle_dependency_denylist": ["pxr", "Sdf", "Usd", "Tf", "libusd", "libpxr", "usd_core"],
        }
        assert find_forbidden_files(repo, contract), "expected generated runtime file rejection"

        bad = generated / "bad.cpp"
        bad.write_text('#include <cstdlib>\nint main(){ return system("python x.py"); }\n', encoding="utf-8")
        findings = scan_sources(repo, [bad], contract["forbidden_runtime_apis"], contract["oracle_dependency_denylist"])
        assert findings, "expected subprocess/Python source rejection"

        oracle = generated / "oracle.cpp"
        oracle.write_text("namespace pxr { struct Sdf {}; }\n", encoding="utf-8")
        findings = scan_sources(repo, [oracle], contract["forbidden_runtime_apis"], contract["oracle_dependency_denylist"])
        assert findings, "expected oracle source rejection"

        (generated / "runtime.py").unlink()
        clean = generated / "dump_layer.cpp"
        clean.write_text("int main(){return 0;}\n", encoding="utf-8")
        cc = {
            "profile": "cpp-native",
            "source_globs": ["generated/cpp/dump_layer.cpp"],
            "forbidden_file_globs": [],
            "toolchains": [
                {"style": "gcc", "compilers": ["g++", "clang++"], "compile_flags": ["-std=c++20", "-O2", "-Igenerated/cpp"]},
                {"style": "msvc", "compilers": ["cl"], "compile_flags": ["/std:c++20", "/O2", "/EHsc", "/nologo", "/Igenerated/cpp"]},
            ],
            "forbidden_runtime_apis": contract["forbidden_runtime_apis"],
            "oracle_dependency_denylist": contract["oracle_dependency_denylist"],
        }
        graph = {
            "nodes": {},
            "scopes": {
                "clean": {
                    "required_nodes": [],
                    "targets": {
                        "cpp": {
                            "dump_cmd": "generated/cpp/dump_layer.exe",
                            "command_contract": cc,
                        }
                    },
                }
            },
        }
        graph_path = repo / "graph.json"
        graph_path.write_text(json.dumps(graph), encoding="utf-8")
        # Exercise the build with whatever supported compiler is on PATH
        # (g++/clang++ on Linux or MinGW, cl on MSVC) rather than gating on g++.
        if select_toolchain(cc)[0] is not None:
            assert validate_contract(repo, graph_path, "clean", "cpp") == 0
    print("target_contract: self-test OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=Path("graph/skillgraph.json"))
    parser.add_argument("--scope")
    parser.add_argument("--target", default="cpp")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not args.scope:
        parser.error("--scope is required unless --self-test is set")

    repo = args.repo.resolve()
    graph_path = args.graph if args.graph.is_absolute() else repo / args.graph
    return validate_contract(repo, graph_path, args.scope, args.target)


if __name__ == "__main__":
    sys.exit(main())
