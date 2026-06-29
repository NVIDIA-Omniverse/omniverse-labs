#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden-image harness for nanousdview backends.

The harness drives nanousdview's public screenshot path:

    build/nanousdview --backend <backend> --screenshot <out.ppm> <scene.usda>

It stores backend-specific goldens by default so Vulkan, OpenGL, and Metal can
each have stable regression baselines while still sharing the same viewer code
path. Cross-backend parity checks are available with --reference-backend.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse


BACKENDS = ("vulkan", "opengl", "metal", "ovrtx")
BACKEND_ALIASES = {
    "gles": "opengl",
    "opengles": "opengl",
    "opengl-es": "opengl",
    "opengl_es": "opengl",
}
AOVS = ("color", "depth", "normals", "segmentation")
AOV_ALIASES = {
    "normal": "normals",
    "prim": "segmentation",
    "prim_id": "segmentation",
    "primid": "segmentation",
    "mesh_id": "segmentation",
    "meshid": "segmentation",
    "id": "segmentation",
}
_USD_ASSET_RE = re.compile(r"@([^@]+)@")


@dataclass(frozen=True)
class Case:
    name: str
    scene: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _workspace_root() -> Path:
    return _repo_root().parent


def _split_csv(
    values: list[str] | None,
    default: tuple[str, ...],
    aliases: dict[str, str] | None = None,
) -> list[str]:
    if not values:
        return list(default)
    aliases = aliases or {}
    out: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                item = item.lower().replace("-", "_").replace(" ", "_")
                out.append(aliases.get(item, item))
    return out


def _safe_name(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    safe = "".join(keep).strip("_")
    return safe or "case"


def _parse_case(spec: str, repo: Path) -> Case:
    if "=" in spec:
        name, raw_path = spec.split("=", 1)
    else:
        raw_path = spec
        name = Path(raw_path).stem
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repo / path
    return Case(_safe_name(name), path.resolve())


def _local_asset_reference_path(ref: str, anchor: Path) -> Path | None:
    parsed = urlparse(ref)
    if parsed.scheme and parsed.scheme != "file":
        return None
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))

    ref_path = Path(ref)
    if ref_path.is_absolute():
        return ref_path
    return anchor / ref_path


def _missing_asset_references(path: Path) -> list[Path]:
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return []
    missing: list[Path] = []
    for match in _USD_ASSET_RE.finditer(text):
        ref = match.group(1)
        ref_path = _local_asset_reference_path(ref, path.parent)
        if ref_path is not None and not ref_path.exists():
            missing.append(ref_path)
    return missing


def _default_cases(repo: Path, include_renderer_assets: bool) -> list[Case]:
    cases = [Case("test_cube", (repo / "test_cube.usda").resolve())]
    if include_renderer_assets:
        asset_dir = (
            _workspace_root()
            / "nanousd-vulkan-renderer"
            / "tests"
            / "correctness"
            / "assets"
        )
        if asset_dir.is_dir():
            for path in sorted(asset_dir.glob("*.usda")):
                missing_refs = _missing_asset_references(path)
                if missing_refs:
                    print(
                        f"SKIP asset {path.name}: missing external reference "
                        f"{missing_refs[0]}"
                    )
                    continue
                cases.append(Case(_safe_name(path.stem), path.resolve()))
    return cases


def _opengl_renderer_cases() -> list[Case]:
    root = _workspace_root() / "nanousd-opengl-renderer"
    specs = [
        ("opengl_cube", root / "test_cube.usda"),
        ("opengl_materials", root / "test_materials.usda"),
        ("opengl_pbr", root / "test_pbr_materials.usda"),
        (
            "opengl_textured_cube",
            root / "tests" / "textures_debug" / "textured_cube.usda",
        ),
        ("opengl_curves", root / "tests" / "curves" / "basicCurves.usda"),
    ]
    cases: list[Case] = []
    for name, path in specs:
        if path.is_file():
            cases.append(Case(name, path.resolve()))
        else:
            print(f"SKIP OpenGL renderer asset {name}: missing {path}")
    return cases


def _ensure_launcher(repo: Path, no_build: bool, launcher_arg: str | None) -> Path:
    if launcher_arg:
        launcher = Path(launcher_arg).expanduser()
        if not launcher.is_absolute():
            launcher = repo / launcher
        if launcher.exists() and os.access(launcher, os.X_OK):
            return launcher.resolve()
        raise RuntimeError(f"{launcher} does not exist or is not executable")
    launcher = repo / "build" / "nanousdview"
    if launcher.exists() and os.access(launcher, os.X_OK):
        return launcher
    if no_build:
        raise RuntimeError(f"{launcher} does not exist; run ./build.sh first")
    subprocess.run(
        [str(repo / "build.sh"), "--no-clean"],
        cwd=repo,
        check=True,
    )
    if not launcher.exists() or not os.access(launcher, os.X_OK):
        raise RuntimeError(f"{launcher} was not produced by build.sh")
    return launcher


def _read_ppm(path: Path) -> tuple[int, int, bytes]:
    raw = path.read_bytes()
    i = 0

    def token() -> bytes:
        nonlocal i
        n = len(raw)
        while i < n:
            c = raw[i]
            if c == ord("#"):
                while i < n and raw[i] not in b"\r\n":
                    i += 1
            elif c in b" \t\r\n":
                i += 1
            else:
                break
        start = i
        while i < n and raw[i] not in b" \t\r\n":
            i += 1
        if start == i:
            raise ValueError(f"truncated PPM header: {path}")
        return raw[start:i]

    magic = token()
    if magic != b"P6":
        raise ValueError(f"{path} is not a binary P6 PPM")
    width = int(token())
    height = int(token())
    max_value = int(token())
    if max_value != 255:
        raise ValueError(f"{path} has unsupported max value {max_value}")
    if i >= len(raw) or raw[i] not in b" \t\r\n":
        raise ValueError(f"{path} has malformed PPM raster separator")
    i += 1
    pixels = raw[i:]
    expected = width * height * 3
    if len(pixels) != expected:
        raise ValueError(
            f"{path} has {len(pixels)} pixel bytes, expected {expected}"
        )
    return width, height, pixels


def _write_ppm(path: Path, width: int, height: int, pixels: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + pixels)


def _rgba_from_ppm(path: Path) -> tuple[int, int, bytes]:
    width, height, rgb = _read_ppm(path)
    rgba = bytearray(width * height * 4)
    for i in range(width * height):
        rgba[i * 4 + 0] = rgb[i * 3 + 0]
        rgba[i * 4 + 1] = rgb[i * 3 + 1]
        rgba[i * 4 + 2] = rgb[i * 3 + 2]
        rgba[i * 4 + 3] = 255
    return width, height, bytes(rgba)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def _write_png_rgba(path: Path, width: int, height: int, rgba: bytes) -> None:
    if len(rgba) != width * height * 4:
        raise ValueError(f"RGBA buffer size does not match {width}x{height}")
    rows = []
    stride = width * 4
    for y in range(height):
        rows.append(b"\x00" + rgba[y * stride:(y + 1) * stride])
    data = zlib.compress(b"".join(rows), 6)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + _png_chunk(b"IDAT", data)
        + _png_chunk(b"IEND", b"")
    )


def _write_report_png(src: Path, dst: Path) -> Path | None:
    try:
        width, height, rgba = _rgba_from_ppm(src)
        _write_png_rgba(dst, width, height, rgba)
        return dst
    except Exception as exc:  # noqa: BLE001
        print(f"WARN could not convert {src} to PNG: {exc}", file=sys.stderr)
        return None


def _blank_rgba(width: int, height: int, color: tuple[int, int, int, int]) -> bytes:
    return bytes(color) * (width * height)


def _paste_rgba(
    dst: bytearray,
    dst_w: int,
    src: bytes,
    src_w: int,
    src_h: int,
    x: int,
    y: int,
) -> None:
    for row in range(src_h):
        src_off = row * src_w * 4
        dst_off = ((y + row) * dst_w + x) * 4
        dst[dst_off:dst_off + src_w * 4] = src[src_off:src_off + src_w * 4]


def _path_for_image(base: Path, backend: str, aov: str, case_name: str) -> Path:
    if aov == "color":
        return base / backend / f"{case_name}.ppm"
    return base / backend / aov / f"{case_name}.ppm"


def _write_contact_sheet(results: list[dict[str, object]], output_dir: Path) -> Path | None:
    rows: list[tuple[dict[str, object], int, int, bytes, bytes, bytes]] = []
    base_w = base_h = None
    for result in results:
        current = Path(str(result.get("output", "")))
        if not current.is_file():
            continue
        try:
            w, h, cur = _rgba_from_ppm(current)
            if base_w is None:
                base_w, base_h = w, h
            if w != base_w or h != base_h:
                continue
            golden_path = Path(str(result.get("golden", "")))
            diff_path = Path(str(result.get("diff", "")))
            if golden_path.is_file():
                _, _, golden = _rgba_from_ppm(golden_path)
            else:
                golden = _blank_rgba(w, h, (38, 38, 38, 255))
            if diff_path.is_file():
                _, _, diff = _rgba_from_ppm(diff_path)
            else:
                diff = _blank_rgba(w, h, (0, 0, 0, 255))
            rows.append((result, w, h, cur, golden, diff))
        except Exception as exc:  # noqa: BLE001
            print(f"WARN contact sheet skipped {current}: {exc}", file=sys.stderr)
    if not rows or base_w is None or base_h is None:
        return None

    gutter = 8
    stripe = 6
    row_w = stripe + base_w * 3 + gutter * 2
    row_h = base_h
    sheet_h = row_h * len(rows) + gutter * (len(rows) - 1)
    sheet = bytearray(_blank_rgba(row_w, sheet_h, (24, 24, 26, 255)))
    status_colors = {
        "PASS": (48, 164, 88, 255),
        "FAIL": (216, 68, 68, 255),
        "MISSING": (206, 156, 48, 255),
        "UPDATE": (66, 133, 244, 255),
        "SKIP": (120, 120, 128, 255),
    }
    for row_idx, (result, w, h, current, golden, diff) in enumerate(rows):
        y = row_idx * (row_h + gutter)
        color = status_colors.get(str(result.get("status", "")), (120, 120, 128, 255))
        for yy in range(y, y + h):
            for xx in range(stripe):
                off = (yy * row_w + xx) * 4
                sheet[off:off + 4] = bytes(color)
        x = stripe
        _paste_rgba(sheet, row_w, current, w, h, x, y)
        x += w + gutter
        _paste_rgba(sheet, row_w, golden, w, h, x, y)
        x += w + gutter
        _paste_rgba(sheet, row_w, diff, w, h, x, y)

    path = output_dir / "contact_sheet.png"
    _write_png_rgba(path, row_w, sheet_h, bytes(sheet))
    return path


def _rel(path: Path, start: Path) -> str:
    return os.path.relpath(path, start).replace(os.sep, "/")


def _write_html_report(
    output_dir: Path,
    summary: dict[str, object],
    results: list[dict[str, object]],
    contact_sheet: Path | None,
) -> Path:
    assets = output_dir / "report_assets"
    assets.mkdir(parents=True, exist_ok=True)
    for result in results:
        key = _safe_name(
            f"{result.get('backend', '')}_{result.get('aov', '')}_{result.get('case', '')}"
        )
        for field, suffix in (
            ("output", "current"),
            ("golden", "golden"),
            ("diff", "diff"),
        ):
            src_raw = result.get(field)
            if not src_raw:
                continue
            src = Path(str(src_raw))
            if src.is_file():
                dst = assets / f"{key}.{suffix}.png"
                if _write_report_png(src, dst) is not None:
                    result[f"{field}_png"] = str(dst)

    def img_cell(field: str, result: dict[str, object]) -> str:
        path_raw = result.get(f"{field}_png")
        if not path_raw:
            return ""
        path = Path(str(path_raw))
        return (
            f'<a href="{html.escape(_rel(path, output_dir))}">'
            f'<img src="{html.escape(_rel(path, output_dir))}" /></a>'
        )

    rows = []
    for result in results:
        status = html.escape(str(result.get("status", "")))
        status_class = status.lower()
        metrics = ""
        if "rms" in result:
            metrics = (
                f"rms={float(result['rms']):.3f}<br>"
                f"mae={float(result['mae']):.3f}<br>"
                f"max={int(result['max_delta'])}"
            )
        reason = html.escape(str(result.get("reason", "")))
        log = result.get("log")
        log_link = ""
        if log:
            log_link = (
                f'<a href="{html.escape(_rel(Path(str(log)), output_dir))}">log</a>'
            )
        rows.append(
            "<tr>"
            f'<td class="{status_class}">{status}</td>'
            f"<td>{html.escape(str(result.get('backend', '')))}</td>"
            f"<td>{html.escape(str(result.get('aov', '')))}</td>"
            f"<td>{html.escape(str(result.get('case', '')))}</td>"
            f"<td>{metrics}{('<br>' + reason) if reason else ''}<br>{log_link}</td>"
            f"<td>{img_cell('output', result)}</td>"
            f"<td>{img_cell('golden', result)}</td>"
            f"<td>{img_cell('diff', result)}</td>"
            "</tr>"
        )

    contact = ""
    if contact_sheet is not None:
        rel_contact = html.escape(_rel(contact_sheet, output_dir))
        contact = f'<p><a href="{rel_contact}">contact_sheet.png</a></p>'

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>nanousdview golden report</title>
<style>
body {{ background: #17181b; color: #e9e9ee; font: 14px sans-serif; }}
a {{ color: #8ab4ff; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #383a40; padding: 6px; vertical-align: top; }}
th {{ background: #24262b; }}
td.pass {{ color: #5fd17b; font-weight: 700; }}
td.fail {{ color: #ff7777; font-weight: 700; }}
td.missing, td.update {{ color: #ffd166; font-weight: 700; }}
td.skip {{ color: #c5c7d0; font-weight: 700; }}
img {{ max-width: 220px; image-rendering: pixelated; }}
</style>
</head>
<body>
<h1>nanousdview golden report</h1>
<p>{int(summary['updated'])} updated, {int(summary['skipped'])} skipped,
{int(summary['missing'])} missing, {int(summary['failures'])} failed.</p>
{contact}
<table>
<thead><tr><th>Status</th><th>Backend</th><th>AOV</th><th>Case</th>
<th>Metrics</th><th>Current</th><th>Golden</th><th>Diff</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</body>
</html>
"""
    path = output_dir / "report.html"
    path.write_text(html_text)
    return path


def _compare_ppm(
    current: Path,
    golden: Path,
    diff_path: Path | None,
) -> dict[str, float | int]:
    cw, ch, cpx = _read_ppm(current)
    gw, gh, gpx = _read_ppm(golden)
    if (cw, ch) != (gw, gh):
        raise ValueError(f"size mismatch: current {cw}x{ch}, golden {gw}x{gh}")
    sq = 0
    abs_sum = 0
    max_delta = 0
    diff = bytearray(len(cpx)) if diff_path is not None else None
    for index, (c, g) in enumerate(zip(cpx, gpx)):
        d = int(c) - int(g)
        ad = abs(d)
        sq += d * d
        abs_sum += ad
        if ad > max_delta:
            max_delta = ad
        if diff is not None:
            diff[index] = min(255, ad * 4)
    n = len(cpx)
    if diff_path is not None and diff is not None:
        _write_ppm(diff_path, cw, ch, bytes(diff))
    return {
        "width": cw,
        "height": ch,
        "rms": math.sqrt(sq / n),
        "mae": abs_sum / n,
        "max_delta": max_delta,
    }


def _is_unavailable(returncode: int, text: str) -> bool:
    if returncode != 2:
        return False
    needles = (
        "unavailable",
        "cannot import",
        "macOS-only",
        "Unknown NU_BACKEND",
    )
    return any(needle in text for needle in needles)


def _is_unsupported_aov(returncode: int, text: str) -> bool:
    if returncode != 2:
        return False
    return "AOV" in text and "unsupported" in text


def _is_unsupported_feature(returncode: int, text: str) -> bool:
    if returncode != 2:
        return False
    return "unsupported" in text


def _render_case(
    launcher: Path,
    backend: str,
    aov: str,
    case: Case,
    output_path: Path,
    log_path: Path,
    width: int,
    height: int,
    timeout: float,
    qt_platform: str | None,
    display: str | None,
    xauthority: str | None,
    render_mode: str | None,
    frame: float | None,
    camera: str | None,
    envmap: str | None,
    envmap_intensity: float | None,
) -> tuple[int, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(launcher),
        "--backend",
        backend,
        "--screenshot",
        str(output_path),
        "--width",
        str(width),
        "--height",
        str(height),
        "--defaultsettings",
    ]
    if aov != "color":
        cmd += ["--aov", aov]
    if render_mode:
        cmd += ["--render-mode", render_mode]
    if frame is not None:
        cmd += ["--frame", str(frame)]
    if camera:
        cmd += ["--camera", camera]
    if envmap:
        cmd += ["--envmap", envmap]
    if envmap_intensity is not None:
        cmd += ["--envmap-intensity", str(envmap_intensity)]
    cmd.append(str(case.scene))
    env = os.environ.copy()
    if qt_platform:
        env["QT_QPA_PLATFORM"] = qt_platform
    if display:
        env["DISPLAY"] = display
    if xauthority:
        env["XAUTHORITY"] = xauthority

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=_repo_root(),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        def as_text(value: str | bytes | None) -> str:
            if value is None:
                return ""
            if isinstance(value, bytes):
                return value.decode("utf-8", "replace")
            return value

        log_text = "\n".join(
            [
                "$ " + " ".join(cmd),
                f"returncode=124 elapsed_ms={elapsed_ms:.1f}",
                f"timeout={timeout:.1f}s",
                "",
                "[stdout]",
                as_text(exc.stdout),
                "",
                "[stderr]",
                as_text(exc.stderr),
            ]
        )
        log_path.write_text(log_text)
        return 124, log_text
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    log = [
        "$ " + " ".join(cmd),
        f"returncode={proc.returncode} elapsed_ms={elapsed_ms:.1f}",
        "",
        "[stdout]",
        proc.stdout,
        "",
        "[stderr]",
        proc.stderr,
    ]
    log_text = "\n".join(log)
    log_path.write_text(log_text)
    return proc.returncode, log_text


def _run(args: argparse.Namespace) -> int:
    repo = _repo_root()
    launcher = _ensure_launcher(repo, args.no_build, args.launcher)
    backends = _split_csv(args.backend, BACKENDS, BACKEND_ALIASES)
    unknown = [b for b in backends if b not in BACKENDS]
    if unknown:
        raise RuntimeError(f"unknown backend(s): {', '.join(unknown)}")
    aovs = _split_csv(args.aov, ("color",), AOV_ALIASES)
    unknown_aovs = [aov for aov in aovs if aov not in AOVS]
    if unknown_aovs:
        raise RuntimeError(f"unknown AOV(s): {', '.join(unknown_aovs)}")

    if args.case:
        cases = [_parse_case(spec, repo) for spec in args.case]
    else:
        cases = _default_cases(repo, args.include_renderer_correctness_assets)
        if args.include_opengl_renderer_assets:
            seen = {case.name for case in cases}
            for case in _opengl_renderer_cases():
                if case.name not in seen:
                    cases.append(case)
                    seen.add(case.name)
    if args.only:
        cases = [case for case in cases if args.only in case.name]
    if not cases:
        raise RuntimeError("no cases selected")
    for case in cases:
        if not case.scene.is_file():
            raise RuntimeError(f"case {case.name!r} scene not found: {case.scene}")

    golden_dir = Path(args.golden_dir)
    output_dir = Path(args.output_dir)
    if not golden_dir.is_absolute():
        golden_dir = repo / golden_dir
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    failures = 0
    missing = 0
    skipped = 0
    updated = 0

    for backend in backends:
        for aov in aovs:
            for case in cases:
                current = _path_for_image(output_dir, backend, aov, case.name)
                log_path = current.with_suffix(".log")
                rc, log_text = _render_case(
                    launcher=launcher,
                    backend=backend,
                    aov=aov,
                    case=case,
                    output_path=current,
                    log_path=log_path,
                    width=args.width,
                    height=args.height,
                    timeout=args.timeout,
                    qt_platform=args.qt_platform,
                    display=args.display,
                    xauthority=args.xauthority,
                    render_mode=args.render_mode,
                    frame=args.frame,
                    camera=args.camera,
                    envmap=args.envmap,
                    envmap_intensity=args.envmap_intensity,
                )
                result: dict[str, object] = {
                    "backend": backend,
                    "aov": aov,
                    "case": case.name,
                    "scene": str(case.scene),
                    "output": str(current),
                    "log": str(log_path),
                }

                if rc != 0:
                    if _is_unavailable(rc, log_text) and not args.strict_backends:
                        result["status"] = "SKIP"
                        result["reason"] = "backend unavailable"
                        skipped += 1
                        print(
                            f"SKIP {backend:7s} {aov:12s} {case.name}: "
                            "backend unavailable"
                        )
                    elif _is_unsupported_aov(rc, log_text) and not args.strict_aovs:
                        result["status"] = "SKIP"
                        result["reason"] = "AOV unsupported"
                        skipped += 1
                        print(
                            f"SKIP {backend:7s} {aov:12s} {case.name}: "
                            "AOV unsupported"
                        )
                    elif _is_unsupported_feature(rc, log_text) and not (
                        args.strict_aovs and _is_unsupported_aov(rc, log_text)
                    ):
                        result["status"] = "SKIP"
                        result["reason"] = "feature unsupported"
                        skipped += 1
                        print(
                            f"SKIP {backend:7s} {aov:12s} {case.name}: "
                            "feature unsupported"
                        )
                    else:
                        result["status"] = "FAIL"
                        result["reason"] = f"render command failed with rc={rc}"
                        failures += 1
                        print(
                            f"FAIL {backend:7s} {aov:12s} {case.name}: "
                            f"render rc={rc}"
                        )
                    results.append(result)
                    continue

                if not current.is_file():
                    result["status"] = "FAIL"
                    result["reason"] = "render command succeeded but output is missing"
                    failures += 1
                    print(
                        f"FAIL {backend:7s} {aov:12s} {case.name}: "
                        "output missing"
                    )
                    results.append(result)
                    continue

                golden_backend = (
                    backend if args.update else (args.reference_backend or backend)
                )
                golden = _path_for_image(golden_dir, golden_backend, aov, case.name)
                result["golden"] = str(golden)

                if args.update:
                    golden.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(current, golden)
                    result["status"] = "UPDATE"
                    updated += 1
                    print(
                        f"UPDATE {backend:7s} {aov:12s} {case.name}: {golden}"
                    )
                    results.append(result)
                    continue

                if not golden.is_file():
                    result["status"] = "MISSING"
                    result["reason"] = "golden missing; rerun with --update"
                    missing += 1
                    print(f"MISS {backend:7s} {aov:12s} {case.name}: {golden}")
                    results.append(result)
                    continue

                try:
                    diff = current.with_name(f"{current.stem}.diff.ppm")
                    result["diff"] = str(diff)
                    metrics = _compare_ppm(current, golden, diff)
                except Exception as exc:  # noqa: BLE001
                    result["status"] = "FAIL"
                    result["reason"] = str(exc)
                    failures += 1
                    print(f"FAIL {backend:7s} {aov:12s} {case.name}: {exc}")
                    results.append(result)
                    continue

                result.update(metrics)
                if float(metrics["rms"]) <= args.threshold_rms:
                    result["status"] = "PASS"
                    print(
                        f"PASS {backend:7s} {aov:12s} {case.name}: "
                        f"rms={metrics['rms']:.3f} max={metrics['max_delta']}"
                    )
                else:
                    result["status"] = "FAIL"
                    result["reason"] = (
                        f"rms {metrics['rms']:.3f} > {args.threshold_rms}"
                    )
                    failures += 1
                    print(
                        f"FAIL {backend:7s} {aov:12s} {case.name}: "
                        f"rms={metrics['rms']:.3f} > {args.threshold_rms}"
                    )
                results.append(result)

    summary = {
        "updated": updated,
        "skipped": skipped,
        "missing": missing,
        "failures": failures,
        "results": results,
    }
    if not args.no_report:
        contact_sheet = _write_contact_sheet(results, output_dir)
        if contact_sheet is not None:
            summary["contact_sheet"] = str(contact_sheet)
        report = _write_html_report(output_dir, summary, results, contact_sheet)
        summary["report"] = str(report)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary: {summary_path}")
    if "report" in summary:
        print(f"report: {summary['report']}")
    if "contact_sheet" in summary:
        print(f"contact sheet: {summary['contact_sheet']}")
    print(
        f"{updated} updated, {skipped} skipped, "
        f"{missing} missing, {failures} failed"
    )

    if failures:
        return 1
    if missing and not args.allow_missing:
        return 1
    if results and skipped == len(results):
        return 77
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render nanousdview backends and compare against PPM goldens."
    )
    parser.add_argument(
        "--backend",
        action="append",
        help="Backend(s) to run: vulkan,opengl/opengles,metal,ovrtx. Comma-separated allowed.",
    )
    parser.add_argument(
        "--aov",
        action="append",
        help="AOV(s) to run: color,depth,normals,segmentation. Comma-separated allowed.",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="Render case as name=path.usda. May be passed multiple times.",
    )
    parser.add_argument("--only", help="Run cases whose names contain this text.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--threshold-rms", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--golden-dir", default="tests/golden")
    parser.add_argument("--output-dir", default="tests/output")
    parser.add_argument(
        "--launcher",
        help="Use this nanousdview launcher instead of build/nanousdview.",
    )
    parser.add_argument(
        "--render-mode",
        choices=("raster", "rt", "raytrace", "shadow"),
        help="Pass a deterministic render mode through to nanousdview.",
    )
    parser.add_argument("--frame", type=float, help="Capture this time code.")
    parser.add_argument("--camera", help="Capture through this scene camera.")
    parser.add_argument("--envmap", help="Capture with this HDR environment map.")
    parser.add_argument(
        "--envmap-intensity",
        type=float,
        help="Optional environment intensity multiplier.",
    )
    parser.add_argument(
        "--reference-backend",
        choices=BACKENDS,
        help="Compare all backends against this backend's goldens.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Replace selected backend goldens with current renders.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Return success when renders pass but selected goldens are absent.",
    )
    parser.add_argument(
        "--strict-backends",
        action="store_true",
        help="Treat unavailable requested backends as failures instead of skips.",
    )
    parser.add_argument(
        "--strict-aovs",
        action="store_true",
        help="Treat unsupported requested AOVs as failures instead of skips.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write report.html or contact_sheet.png.",
    )
    parser.add_argument(
        "--qt-platform",
        help="Set QT_QPA_PLATFORM for the render subprocess.",
    )
    parser.add_argument("--display", help="Set DISPLAY for the render subprocess.")
    parser.add_argument(
        "--xauthority",
        help="Set XAUTHORITY for the render subprocess.",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Do not run ./build.sh --no-clean if build/nanousdview is missing.",
    )
    parser.add_argument(
        "--include-renderer-correctness-assets",
        action="store_true",
        help="Also run sibling nanousd-vulkan-renderer correctness assets.",
    )
    parser.add_argument(
        "--include-opengl-renderer-assets",
        action="store_true",
        help="Also run portable fixtures from sibling nanousd-opengl-renderer.",
    )
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"render_goldens.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
