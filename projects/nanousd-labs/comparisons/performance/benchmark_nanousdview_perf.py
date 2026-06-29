#!/usr/bin/env python3
"""Benchmark nanousdview load and camera-orbit performance.

The parent process fans out one worker subprocess per scene/profile so the
official OVRTX wheel and the nanousd OVRTX facade never share one interpreter.
Workers drive only nanousdview._backend.OvrtxViewportRenderer: load_stage,
set_camera, and render_ldr.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "comparisons" / "performance"
CAPTURE_DIR = OUT_DIR / "captures"
LOG_DIR = OUT_DIR / "logs"
WORKER_DIR = OUT_DIR / "worker"
RESULTS_JSON = OUT_DIR / "results.json"
README = OUT_DIR / "README.md"

DEFAULT_WAREHOUSE = Path(
    "$HOME/assets/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
)
DEFAULT_DSX = Path(
    "$HOME/dsx-assets/dsx_dataset_2.1/DSX_BP_/DSX_BP/Assembly/DSX_Main_BP.usda"
)
DEFAULT_OVRTX_PYTHON = ROOT / ".ovrtx03-venv" / "bin" / "python"


@dataclass(frozen=True)
class CameraSpec:
    name: str
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]
    fov_degrees: float = 50.0
    near_clip: float = 0.05
    far_clip: float = 10000.0


@dataclass(frozen=True)
class OrbitSpec:
    center: tuple[float, float, float]
    radius: float
    height: float
    fov_degrees: float
    near_clip: float
    far_clip: float


@dataclass(frozen=True)
class SceneSpec:
    key: str
    title: str
    usd_path: Path
    note: str
    captures: tuple[CameraSpec, ...]
    orbit: OrbitSpec


@dataclass(frozen=True)
class ProfileSpec:
    key: str
    title: str
    backend: str
    render_mode: str


PROFILES = (
    ProfileSpec("ovrtx", "OVRTX", "ovrtx", "rt"),
    ProfileSpec("vulkan_rt", "Vulkan RT", "vulkan", "rt"),
    ProfileSpec("vulkan_raster", "Vulkan Raster", "vulkan", "raster"),
    ProfileSpec("opengl", "OpenGL GLES", "opengl", "raster"),
)


def _scenes(warehouse: Path, dsx: Path) -> tuple[SceneSpec, ...]:
    return (
        SceneSpec(
            key="isaac_warehouse_full",
            title="Isaac Sim Simple Warehouse Full",
            usd_path=warehouse,
            note="Full Isaac Sim Simple_Warehouse/full_warehouse.usd scene.",
            captures=(
                CameraSpec(
                    "rack_aisle_a",
                    (-2.0, -16.0, 1.9),
                    (-6.0, 22.0, 1.4),
                    (0.0, 0.0, 1.0),
                    50.0,
                    0.05,
                    400.0,
                ),
                CameraSpec(
                    "rack_aisle_b",
                    (3.5, -6.0, 2.6),
                    (-14.0, 20.0, 1.0),
                    (0.0, 0.0, 1.0),
                    50.0,
                    0.05,
                    400.0,
                ),
            ),
            orbit=OrbitSpec(
                center=(-7.0, 18.0, 1.5),
                radius=9.0,
                height=2.2,
                fov_degrees=55.0,
                near_clip=0.05,
                far_clip=400.0,
            ),
        ),
        SceneSpec(
            key="dsx_datacenter_full",
            title="NVIDIA DSX Full Datacenter",
            usd_path=dsx,
            note="Full DSX_Main_BP.usda assembly from the local DSX dataset.",
            captures=(
                CameraSpec(
                    "datahall_01",
                    (38.903, 104.531, 20.384),
                    (444.497, -809.420, 6.751),
                    (0.029, -0.002, 1.0),
                    45.69,
                    1.0,
                    10000000.0,
                ),
                CameraSpec(
                    "datahall_02",
                    (28.655, 33.378, 20.439),
                    (28.655, 1033.378, 20.439),
                    (0.0, 0.0, 1.0),
                    45.69,
                    1.0,
                    10000000.0,
                ),
            ),
            orbit=OrbitSpec(
                center=(30.0, 82.0, 20.5),
                radius=42.0,
                height=20.8,
                fov_degrees=45.69,
                near_clip=1.0,
                far_clip=10000000.0,
            ),
        ),
    )


def _font(size: int, bold: bool = False) -> Any:
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _ppm_write(path: Path, pixels: np.ndarray) -> None:
    arr = np.asarray(pixels)
    if arr.dtype != np.uint8:
        rgb = np.clip(arr[..., :3].astype(np.float32), 0.0, None)
        rgb = rgb / (1.0 + rgb)
        arr = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    else:
        arr = arr[..., :3]
    arr = np.ascontiguousarray(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = arr.shape[:2]
    with path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(arr.tobytes())


def _ppm_to_png(ppm: Path, png: Path) -> None:
    from PIL import Image

    png.parent.mkdir(parents=True, exist_ok=True)
    Image.open(ppm).convert("RGB").save(png)


def _mean_rgb(path: Path) -> list[float]:
    from PIL import Image

    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return [float(arr[..., c].mean()) for c in range(3)]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    idx = (len(ordered) - 1) * (pct / 100.0)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _orbit_camera(scene: SceneSpec, frame: int, frames: int) -> CameraSpec:
    angle = 2.0 * math.pi * (frame / max(frames, 1))
    center = scene.orbit.center
    eye = (
        center[0] + math.cos(angle) * scene.orbit.radius,
        center[1] + math.sin(angle) * scene.orbit.radius,
        scene.orbit.height,
    )
    return CameraSpec(
        "orbit",
        eye,
        center,
        (0.0, 0.0, 1.0),
        scene.orbit.fov_degrees,
        scene.orbit.near_clip,
        scene.orbit.far_clip,
    )


def _set_camera(renderer: Any, camera: CameraSpec) -> None:
    renderer.set_camera(
        camera.eye,
        camera.target,
        camera.up,
        camera.fov_degrees,
        camera.near_clip,
        camera.far_clip,
    )


def _worker_paths(profile: ProfileSpec) -> None:
    sys.path.insert(0, str(ROOT / "nanousdview" / "python"))
    if profile.backend != "ovrtx":
        sys.path.insert(0, str(ROOT / "nanousd-vulkan-renderer" / "python"))
        if profile.backend == "opengl":
            sys.path.insert(0, str(ROOT / "nanousd-opengl-renderer" / "python"))


def _configure_worker_env(profile: ProfileSpec) -> None:
    os.environ["NANOUSD_VIEW_BACKEND"] = profile.backend
    os.environ["NANOUSD_VIEW_RENDER_MODE"] = profile.render_mode
    os.environ.setdefault("NUSD_ENABLE_MATERIALS", "1")
    os.environ.setdefault("NUSD_ENABLE_PTEX_MATERIALS", "1")
    os.environ.setdefault("DISPLAY", ":1")
    os.environ.setdefault("XAUTHORITY", "/run/user/1000/gdm/Xauthority")
    os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    os.environ.setdefault("NANOUSD_VIEW_STEP_TIMEOUT_NS", "120000000000")
    os.environ.setdefault("NANOUSD_VIEW_NOFILE", "8192")

    if profile.backend == "ovrtx":
        os.environ.pop("NANOUSD_OVRTX_BACKEND", None)
        os.environ.setdefault("OVRTX_SKIP_USD_CHECK", "1")
        os.environ.setdefault("NUVIEW_OVRTX_RENDER_MODE", "rt2")
        os.environ.setdefault("NUVIEW_OVRTX_SPP", "1")
        os.environ.setdefault("NUVIEW_OVRTX_DENOISE", "0")
        os.environ.setdefault("NUVIEW_OVRTX_DEFAULT_LIGHTING", "1")
        return

    os.environ["NANOUSD_OVRTX_BACKEND"] = profile.backend
    if profile.backend == "vulkan":
        os.environ.setdefault(
            "NUSD_RENDERER_LIB",
            str(ROOT / "nanousd-vulkan-renderer" / "build" / "libnusd_renderer.so"),
        )
    elif profile.backend == "opengl":
        os.environ.setdefault(
            "NUSD_RENDERER_LIB",
            str(ROOT / "nanousd-opengl-renderer" / "build" / "libnusd_renderer_opengl.so"),
        )
    # Prefer an OPTIMIZED (Release) backend: composition/parse/traversal/geometry
    # decode all run in the backend, and a -O3 build is ~6-8x faster than the
    # -O0 Debug build (measured: DSX nanousd_open 161s->26s, proto DFS 110s->13s).
    # Debug is a last-resort fallback only. Override with NANOUSD_BACKEND.
    for cand in (
        ROOT / "nanousd" / "build-release" / "Release" / "libnanousd.so",
        ROOT / "nanousd" / "build" / "Release" / "libnanousd.so",
        ROOT / "nanousd" / "build" / "libnanousd.so",
        ROOT / ".local" / "lib" / "libnanousd.so",
        ROOT / "nanousd" / "build" / "Debug" / "libnanousd.so",
    ):
        if cand.exists():
            os.environ.setdefault("NANOUSD_BACKEND", str(cand))
            break
    lib_parts = [
        ROOT / "nanousd" / "build" / "Debug",
        ROOT / "nanousd" / "build",
        ROOT / "nanousd-vulkan-renderer" / "build",
        ROOT / "nanousd-opengl-renderer" / "build",
        ROOT / "nanousd-opengl-renderer" / "build" / "Release",
    ]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    prefix = [str(p) for p in lib_parts if p.is_dir()]
    os.environ["LD_LIBRARY_PATH"] = ":".join(prefix + ([current] if current else []))


def _render_mode_const(profile: ProfileSpec) -> int:
    from nanousdview._backend import VIEW_RENDER_RASTER, VIEW_RENDER_RT

    if profile.render_mode == "raster":
        return VIEW_RENDER_RASTER
    return VIEW_RENDER_RT


def run_worker(args: argparse.Namespace) -> int:
    profile = next(p for p in PROFILES if p.key == args.profile)
    scene = next(s for s in _scenes(Path(args.warehouse), Path(args.dsx)) if s.key == args.scene)
    out_json = Path(args.worker_out)
    capture_base = Path(args.capture_base)
    _worker_paths(profile)
    _configure_worker_env(profile)

    from nanousdview._backend import OvrtxViewportRenderer, configure_backend

    configure_backend(profile.backend, ROOT)
    mode_const = _render_mode_const(profile)

    result: dict[str, Any] = {
        "scene": scene.key,
        "scene_title": scene.title,
        "profile": profile.key,
        "profile_title": profile.title,
        "backend": profile.backend,
        "render_mode": profile.render_mode,
        "run_index": int(args.run_index),
        "usd_path": str(scene.usd_path),
        "resolution": [int(args.width), int(args.height)],
        "frames": int(args.frames),
        "warmup_frames": int(args.warmup),
        "captures": [],
        "status": "ok",
    }
    renderer = None
    try:
        t0 = time.perf_counter()
        renderer = OvrtxViewportRenderer(
            width=args.width,
            height=args.height,
            enable_rt=None if profile.backend == "ovrtx" else profile.render_mode == "rt",
            enable_materials=True,
        )
        result["renderer_create_ms"] = (time.perf_counter() - t0) * 1000.0
        if profile.backend == "ovrtx":
            renderer.set_default_lighting(camera_light=True, dome_light=True)
        renderer.set_render_mode(mode_const)

        t0 = time.perf_counter()
        renderer.load_stage(str(scene.usd_path))
        result["load_stage_ms"] = (time.perf_counter() - t0) * 1000.0

        first_frame_ms: float | None = None
        for cap in scene.captures:
            _set_camera(renderer, cap)
            t0 = time.perf_counter()
            pixels = renderer.render_ldr(render_mode=mode_const)
            frame_ms = (time.perf_counter() - t0) * 1000.0
            if first_frame_ms is None:
                first_frame_ms = frame_ms
            ppm = capture_base / f"{scene.key}_{profile.key}_run{int(args.run_index):02d}_{cap.name}.ppm"
            _ppm_write(ppm, pixels)
            result["captures"].append({
                "name": cap.name,
                "ppm": str(ppm),
                "frame_ms": frame_ms,
                "camera": asdict(cap),
            })

        result["first_frame_ms"] = first_frame_ms or 0.0
        result["load_to_first_frame_ms"] = result["load_stage_ms"] + result["first_frame_ms"]

        for i in range(max(args.warmup, 0)):
            _set_camera(renderer, _orbit_camera(scene, i, max(args.warmup, 1)))
            renderer.render_ldr(render_mode=mode_const)

        frame_ms: list[float] = []
        t_orbit = time.perf_counter()
        for i in range(max(args.frames, 1)):
            cam = _orbit_camera(scene, i, max(args.frames, 1))
            t0 = time.perf_counter()
            _set_camera(renderer, cam)
            renderer.render_ldr(render_mode=mode_const)
            frame_ms.append((time.perf_counter() - t0) * 1000.0)
        orbit_wall_ms = (time.perf_counter() - t_orbit) * 1000.0
        fps = (len(frame_ms) * 1000.0 / orbit_wall_ms) if orbit_wall_ms > 0 else 0.0
        result["orbit"] = {
            "wall_ms": orbit_wall_ms,
            "fps": fps,
            "frame_ms_min": min(frame_ms),
            "frame_ms_p50": _percentile(frame_ms, 50),
            "frame_ms_p90": _percentile(frame_ms, 90),
            "frame_ms_p99": _percentile(frame_ms, 99),
            "frame_ms_max": max(frame_ms),
            "frame_ms_mean": statistics.fmean(frame_ms),
        }
    except Exception as exc:  # noqa: BLE001
        result["status"] = "failed"
        result["error"] = repr(exc)
    finally:
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"worker_out": str(out_json), "status": result["status"]}), flush=True)
    return 0 if result["status"] == "ok" or args.allow_failures else 1


def _base_env(profile: ProfileSpec, ovrtx_python: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "nanousdview" / "python")
    if profile.backend != "ovrtx":
        env["PYTHONPATH"] = ":".join(
            [
                str(ROOT / "nanousdview" / "python"),
                str(ROOT / "nanousd-vulkan-renderer" / "python"),
                str(ROOT / "nanousd-opengl-renderer" / "python"),
            ]
        )
    if profile.backend == "ovrtx":
        venv = ovrtx_python.parent.parent
        ovrtx_bin = venv / "lib" / "python3.12" / "site-packages" / "ovrtx" / "bin"
        plugins = ovrtx_bin / "plugins"
        gpu_foundation = plugins / "gpu.foundation"
        glib = gpu_foundation / "libglib-2.0.so.0"
        gobject = gpu_foundation / "libgobject-2.0.so.0"
        if glib.exists() and gobject.exists():
            preload = f"{glib}:{gobject}"
            if env.get("LD_PRELOAD"):
                preload += ":" + env["LD_PRELOAD"]
            env["LD_PRELOAD"] = preload
        ld = [str(plugins), str(ovrtx_bin)]
        inherited = [
            p for p in env.get("LD_LIBRARY_PATH", "").split(":")
            if p and "openusd" not in p.lower()
        ]
        env["LD_LIBRARY_PATH"] = ":".join(ld + inherited)
    return env


def _worker_python(profile: ProfileSpec, ovrtx_python: Path) -> Path:
    if profile.backend == "ovrtx":
        return ovrtx_python
    return Path(sys.executable)


def _run_one(
    scene: SceneSpec,
    profile: ProfileSpec,
    args: argparse.Namespace,
    run_index: int,
) -> dict[str, Any]:
    worker_json = WORKER_DIR / f"{scene.key}_{profile.key}_run{run_index:02d}.json"
    log_path = LOG_DIR / f"{scene.key}_{profile.key}_run{run_index:02d}.log"
    if args.resume and worker_json.exists():
        result = json.loads(worker_json.read_text(encoding="utf-8"))
        result.setdefault("worker_returncode", 0 if result.get("status") == "ok" else 1)
        result.setdefault("worker_elapsed_s", None)
        result.setdefault("timeout_s", float(args.timeout))
        result["log"] = str(log_path.relative_to(OUT_DIR))
        result["resumed"] = True
        print(f"   resume {scene.key} / {profile.key} run {run_index}", flush=True)
        return result
    cmd = [
        str(_worker_python(profile, Path(args.ovrtx_python))),
        str(Path(__file__).resolve()),
        "--worker",
        "--scene",
        scene.key,
        "--profile",
        profile.key,
        "--run-index",
        str(run_index),
        "--warehouse",
        str(args.warehouse),
        "--dsx",
        str(args.dsx),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--frames",
        str(args.frames),
        "--warmup",
        str(args.warmup),
        "--worker-out",
        str(worker_json),
        "--capture-base",
        str(WORKER_DIR),
        "--allow-failures",
    ]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    timed_out = False
    def stream_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=_base_env(profile, Path(args.ovrtx_python)),
            text=True,
            capture_output=True,
            timeout=float(args.timeout),
        )
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = stream_text(exc.stdout)
        stderr = stream_text(exc.stderr)
        returncode = 124
    else:
        stdout = stream_text(stdout)
        stderr = stream_text(stderr)
    elapsed = time.perf_counter() - t0
    log_path.write_text(
        "$ " + " ".join(cmd) + "\n\nSTDOUT:\n" + stdout + "\nSTDERR:\n" + stderr,
        encoding="utf-8",
    )
    if worker_json.exists():
        result = json.loads(worker_json.read_text(encoding="utf-8"))
    else:
        result = {
            "scene": scene.key,
            "scene_title": scene.title,
            "profile": profile.key,
            "profile_title": profile.title,
            "backend": profile.backend,
            "render_mode": profile.render_mode,
            "run_index": run_index,
            "status": "failed",
            "error": f"worker did not write result json; rc={returncode}",
        }
    if timed_out:
        result["status"] = "failed"
        result["error"] = f"worker timed out after {float(args.timeout):.1f}s"
    result["worker_returncode"] = returncode
    result["worker_elapsed_s"] = elapsed
    result["timeout_s"] = float(args.timeout)
    result["log"] = str(log_path.relative_to(OUT_DIR))
    return result


def _convert_captures(results: list[dict[str, Any]]) -> None:
    for result in results:
        for cap in result.get("captures", []):
            ppm = Path(cap["ppm"])
            png = CAPTURE_DIR / f"{ppm.stem}.png"
            if ppm.exists():
                _ppm_to_png(ppm, png)
                cap["png"] = str(png.relative_to(OUT_DIR))
                cap["mean_rgb"] = _mean_rgb(png)


def _thumb(path: Path, size: tuple[int, int]) -> Any:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (20, 23, 26))
    canvas.paste(img, ((size[0] - img.size[0]) // 2, (size[1] - img.size[1]) // 2))
    return canvas


def _make_contact_sheets(results: list[dict[str, Any]], scenes: tuple[SceneSpec, ...]) -> list[str]:
    from PIL import Image, ImageDraw

    outputs: list[str] = []
    font_title = _font(24, True)
    font_head = _font(14, True)
    font_small = _font(11)
    for scene in scenes:
        scene_results = [r for r in results if r.get("scene") == scene.key and r.get("status") == "ok"]
        if not scene_results:
            continue
        capture_names = [cap.name for cap in scene.captures]
        cell_w, cell_h = 290, 170
        label_w = 190
        gap = 14
        title_h = 82
        header_h = 34
        width = label_w + gap * (len(PROFILES) + 2) + cell_w * len(PROFILES)
        height = title_h + len(capture_names) * (header_h + cell_h + 48 + gap) + gap
        sheet = Image.new("RGB", (width, height), (13, 16, 19))
        draw = ImageDraw.Draw(sheet)
        draw.rectangle((0, 0, width, title_h), fill=(17, 22, 27))
        draw.rectangle((0, title_h - 4, width, title_h), fill=(54, 117, 136))
        draw.text((gap, 16), f"{scene.title} Rack-Close Captures", fill=(247, 250, 252), font=font_title)
        draw.text((gap, 50), "Rendered through nanousdview. No subtract/diff images.", fill=(181, 192, 201), font=font_small)
        y = title_h + gap
        by_profile: dict[str, dict[str, Any]] = {}
        for result in sorted(scene_results, key=lambda r: int(r.get("run_index", 9999))):
            by_profile.setdefault(result["profile"], result)
        for capture_name in capture_names:
            draw.text((gap, y + 8), capture_name, fill=(232, 237, 241), font=font_head)
            x = label_w + gap * 2
            for profile in PROFILES:
                draw.text((x, y + 8), profile.title, fill=(232, 237, 241), font=font_head)
                x += cell_w + gap
            y += header_h
            x = label_w + gap * 2
            for profile in PROFILES:
                result = by_profile.get(profile.key)
                cap = None
                if result:
                    cap = next((c for c in result.get("captures", []) if c.get("name") == capture_name), None)
                png_rel = cap.get("png") if cap else None
                if png_rel and (OUT_DIR / png_rel).exists():
                    tile = _thumb(OUT_DIR / png_rel, (cell_w, cell_h))
                else:
                    tile = Image.new("RGB", (cell_w, cell_h), (42, 27, 30))
                    ImageDraw.Draw(tile).text((16, cell_h // 2 - 8), "missing", fill=(245, 190, 190), font=font_head)
                sheet.paste(tile, (x, y))
                if cap:
                    draw.text((x, y + cell_h + 8), f"capture {cap['frame_ms']:.1f} ms", fill=(181, 192, 201), font=font_small)
                x += cell_w + gap
            y += cell_h + 48 + gap
        out = CAPTURE_DIR / f"{scene.key}_rack_close_sheet.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out)
        outputs.append(str(out.relative_to(OUT_DIR)))
    return outputs


def _fmt_ms(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):,.1f}"
    except Exception:
        return "-"


def _fmt_fps(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return "-"


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)


def _cv_percent(values: list[float]) -> float | None:
    mean = _mean(values)
    stdev = _stdev(values)
    if mean is None or stdev is None or abs(mean) <= 1.0e-12:
        return None
    return 100.0 * stdev / mean


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "-"


def _fmt_range(values: list[float], unit: str) -> str:
    if not values:
        return "-"
    if unit == "fps":
        return f"{min(values):.2f}-{max(values):.2f}"
    return f"{min(values):.1f}-{max(values):.1f}"


def _successful_values(group: list[dict[str, Any]], getter: Any) -> list[float]:
    values: list[float] = []
    for result in group:
        if result.get("status") != "ok":
            continue
        value = getter(result)
        if value is None:
            continue
        try:
            values.append(float(value))
        except Exception:
            pass
    return values


def _group_results(results: list[dict[str, Any]], scene_key: str, profile_key: str) -> list[dict[str, Any]]:
    return sorted(
        [r for r in results if r.get("scene") == scene_key and r.get("profile") == profile_key],
        key=lambda r: int(r.get("run_index", 0)),
    )


def _markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    out = []
    for idx, row in enumerate(rows):
        out.append("| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |")
        if idx == 0:
            out.append("| " + " | ".join("-" * widths[i] for i in range(len(row))) + " |")
    return "\n".join(out)


def _write_readme(payload: dict[str, Any]) -> None:
    results = payload["results"]
    scenes = _scenes(Path(payload["assets"]["warehouse"]), Path(payload["assets"]["dsx"]))
    lines = [
        "# Renderer Performance Comparison",
        "",
        "This is a separate performance document from the visual parity comparison.",
        "The benchmark uses `nanousdview._backend.OvrtxViewportRenderer` for every",
        "row, so each renderer is loaded and driven through the same viewer-facing",
        "`load_stage`, `set_camera`, and `render_ldr` path.",
        "",
        "Scenes:",
        "",
        f"- Isaac Sim warehouse full: `{payload['assets']['warehouse']}`",
        f"- NVIDIA DSX full datacenter: `{payload['assets']['dsx']}`",
        "",
        "Renderer profiles: OVRTX, Vulkan RT, Vulkan Raster, and OpenGL GLES.",
        "",
        "## Summary",
        "",
    ]
    for scene in scenes:
        lines.extend([f"### {scene.title}", ""])
        rows = [[
            "Renderer",
            "OK runs",
            "load_to_first mean ms",
            "load CV",
            "load range ms",
            "orbit fps mean",
            "fps CV",
            "fps range",
            "orbit p50 mean ms",
        ]]
        for profile in PROFILES:
            group = _group_results(results, scene.key, profile.key)
            ok = [r for r in group if r.get("status") == "ok"]
            load_to_first = _successful_values(group, lambda r: r.get("load_to_first_frame_ms"))
            orbit_fps = _successful_values(group, lambda r: r.get("orbit", {}).get("fps"))
            orbit_p50 = _successful_values(group, lambda r: r.get("orbit", {}).get("frame_ms_p50"))
            rows.append([
                profile.title,
                f"{len(ok)}/{payload['settings']['runs']}",
                _fmt_ms(_mean(load_to_first)),
                _fmt_pct(_cv_percent(load_to_first)),
                _fmt_range(load_to_first, "ms"),
                _fmt_fps(_mean(orbit_fps)),
                _fmt_pct(_cv_percent(orbit_fps)),
                _fmt_range(orbit_fps, "fps"),
                _fmt_ms(_mean(orbit_p50)),
            ])
        lines.extend([_markdown_table(rows), ""])
        sheet = f"captures/{scene.key}_rack_close_sheet.png"
        if (OUT_DIR / sheet).exists():
            lines.extend([f"![{scene.title} rack-close captures]({sheet})", ""])
        lines.extend(["Per-run detail:", ""])
        detail_rows = [[
            "Run",
            "Renderer",
            "Status",
            "load_stage ms",
            "first_frame ms",
            "load_to_first ms",
            "orbit fps",
            "orbit p50 ms",
        ]]
        for profile in PROFILES:
            for result in _group_results(results, scene.key, profile.key):
                orbit = result.get("orbit", {})
                detail_rows.append([
                    str(result.get("run_index", "-")),
                    profile.title,
                    result.get("status", "unknown"),
                    _fmt_ms(result.get("load_stage_ms")),
                    _fmt_ms(result.get("first_frame_ms")),
                    _fmt_ms(result.get("load_to_first_frame_ms")),
                    _fmt_fps(orbit.get("fps")),
                    _fmt_ms(orbit.get("frame_ms_p50")),
                ])
        lines.extend([_markdown_table(detail_rows), ""])
        failures = [r for r in results if r.get("scene") == scene.key and r.get("status") != "ok"]
        if failures:
            lines.extend(["Failed rows:", ""])
            for failure in failures:
                lines.append(f"- {failure.get('profile_title', failure.get('profile'))}: `{failure.get('error', 'unknown')}`")
            lines.append("")

    lines.extend([
        "## Method",
        "",
        "- `load_stage ms` times the `nanousdview` stage load call.",
        "- `first_frame ms` is the first rack-close render after load. For nanousd",
        "  renderer profiles this is where native scene materialization, upload,",
        "  and acceleration setup can appear.",
        "- `load_to_first ms` is `load_stage + first_frame`.",
        "- `orbit fps` measures repeated camera updates plus `render_ldr` calls over",
        f"  `{payload['settings']['frames']}` timed frames after `{payload['settings']['warmup']}` warmup frames.",
        f"- Each scene/profile row is repeated `{payload['settings']['runs']}` times in",
        "  independent worker subprocesses. CV is sample standard deviation divided",
        "  by mean across successful runs.",
        f"- Worker timeout for this run: `{payload['settings']['timeout_s']:.0f}` seconds.",
        "- Rack-close captures use authored/hand-picked camera views near warehouse",
        "  shelves and DSX data-hall racks. No subtract/diff images are generated.",
        "",
        "## Reproduce",
        "",
        "From the repo root:",
        "",
        "```bash",
        "$HOME/nanousd-labs/.venv/bin/python \\",
        "  comparisons/performance/benchmark_nanousdview_perf.py \\",
        f"  --width {payload['settings']['width']} --height {payload['settings']['height']} \\",
        f"  --frames {payload['settings']['frames']} --warmup {payload['settings']['warmup']} \\",
        f"  --runs {payload['settings']['runs']} --timeout {payload['settings']['timeout_s']:.0f} \\",
        "  --allow-failures",
        "```",
        "",
        "Use `--profiles` or `--scenes` to run a subset while iterating. The official",
        "OVRTX row uses `--ovrtx-python` and defaults to",
        f"`{payload['settings']['ovrtx_python']}`.",
        "",
        f"Raw results: [`results.json`](results.json).",
    ])
    README.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_parent(args: argparse.Namespace) -> int:
    scenes = _scenes(Path(args.warehouse), Path(args.dsx))
    requested_scenes = set(args.scenes.split(",")) if args.scenes else {s.key for s in scenes}
    requested_profiles = set(args.profiles.split(",")) if args.profiles else {p.key for p in PROFILES}
    scene_list = [s for s in scenes if s.key in requested_scenes]
    profile_list = [p for p in PROFILES if p.key in requested_profiles]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    WORKER_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for run_index in range(1, int(args.runs) + 1):
        for scene in scene_list:
            if not scene.usd_path.exists():
                results.append({
                    "scene": scene.key,
                    "scene_title": scene.title,
                    "run_index": run_index,
                    "status": "failed",
                    "error": f"USD path not found: {scene.usd_path}",
                })
                continue
            for profile in profile_list:
                print(f"== run {run_index}/{args.runs} {scene.key} / {profile.key}", flush=True)
                results.append(_run_one(scene, profile, args, run_index))

    _convert_captures(results)
    sheets = _make_contact_sheets(results, scenes)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "platform": platform.platform(),
        "assets": {
            "warehouse": str(Path(args.warehouse)),
            "dsx": str(Path(args.dsx)),
        },
        "settings": {
            "width": int(args.width),
            "height": int(args.height),
            "frames": int(args.frames),
            "warmup": int(args.warmup),
            "runs": int(args.runs),
            "timeout_s": float(args.timeout),
            "ovrtx_python": str(Path(args.ovrtx_python)),
        },
        "profiles": [asdict(p) for p in PROFILES],
        "capture_sheets": sheets,
        "results": results,
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_readme(payload)
    return 0 if args.allow_failures or all(r.get("status") == "ok" for r in results) else 1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warehouse", default=str(DEFAULT_WAREHOUSE))
    ap.add_argument("--dsx", default=str(DEFAULT_DSX))
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=2400.0)
    ap.add_argument("--ovrtx-python", default=str(DEFAULT_OVRTX_PYTHON))
    ap.add_argument("--profiles", default="", help="Comma-separated subset of profile keys.")
    ap.add_argument("--scenes", default="", help="Comma-separated subset of scene keys.")
    ap.add_argument("--allow-failures", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Reuse existing worker JSON artifacts.")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--scene", default="")
    ap.add_argument("--profile", default="")
    ap.add_argument("--run-index", type=int, default=1)
    ap.add_argument("--worker-out", default="")
    ap.add_argument("--capture-base", default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
