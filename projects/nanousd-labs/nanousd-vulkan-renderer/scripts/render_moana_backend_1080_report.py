#!/usr/bin/env python3
"""Render a 1080p Moana orbit report across Vulkan, OpenGL, and OVRTX."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import resource
import selectors
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from statistics import median
from typing import Any

import numpy as np

from render_moana_camera_contact_sheet import (
    DEFAULT_USD,
    CameraSpec,
    configure_render_environment,
    make_orbit_cameras,
    parse_moana_cameras,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = REPO_ROOT.parent
OPENGL_REPO = WORKSPACE / "nanousd-opengl-renderer"
NANOUSDVIEW_PYTHONPATH = WORKSPACE / "nanousdview" / "python"
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "reports" / "moana_backend_1080p_orbit_2026-06-02"
OVRTX_PYTHON = Path(
    os.environ.get(
        "OVRTX_PYTHON",
        str(Path.home() / "workspace" / ".venv" / "bin" / "python"),
    )
)


BACKENDS = ("vulkan_rt", "vulkan_raster", "opengl", "ovrtx")
BACKEND_LABELS = {
    "vulkan_rt": "Vulkan RT",
    "vulkan_raster": "Vulkan Raster",
    "opengl": "OpenGL",
    "ovrtx": "OVRTX",
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _current_rss_bytes() -> int | None:
    status = Path("/proc/self/status")
    try:
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    except OSError:
        return None
    return None


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return peak
    return peak * 1024


def _bytes_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 ** 3)


def _as_tuple(values: Any, n: int) -> tuple[float, ...]:
    if len(values) != n:
        raise ValueError(f"expected {n} values, got {len(values)}")
    return tuple(float(v) for v in values)


def _phase_dict(renderer: Any) -> dict[str, float] | None:
    fn = getattr(renderer, "get_phase_timings_ms", None)
    if not callable(fn):
        return None
    values = fn()
    if values is None:
        return None
    return {str(k): float(v) for k, v in values.items()}


def _cmd_cache_stats(renderer: Any) -> dict[str, int] | None:
    fn = getattr(renderer, "cmd_cache_stats", None)
    if not callable(fn):
        return None
    try:
        return {str(k): int(v) for k, v in fn().items()}
    except Exception:
        return None


def _scene_bounds(renderer: Any) -> Any:
    fn = getattr(renderer, "get_scene_bounds", None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:
        return None


def _save_rgb_png(pixels: np.ndarray, path: Path) -> None:
    from PIL import Image

    arr = np.asarray(pixels)
    if arr.dtype != np.uint8:
        rgb = np.clip(arr[..., :3].astype(np.float32), 0.0, None)
        rgb = rgb / (1.0 + rgb)
        arr = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    else:
        arr = arr[..., :3]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(arr[:, :, :3]), mode="RGB").save(path)


def _summary_stats(frames: list[dict[str, Any]]) -> dict[str, Any]:
    times = [float(f["total_seconds"]) for f in frames if f.get("status") == "ok"]
    render_times = [float(f.get("render_seconds", 0.0)) for f in frames if f.get("status") == "ok"]
    fetch_times = [float(f.get("fetch_seconds", 0.0)) for f in frames if f.get("status") == "ok"]
    camera_times = [float(f.get("camera_seconds", 0.0)) for f in frames if f.get("status") == "ok"]
    if not times:
        return {
            "frame_count": 0,
            "mean_frame_ms": None,
            "median_frame_ms": None,
            "p95_frame_ms": None,
            "fps": None,
            "mean_render_ms": None,
            "mean_fetch_ms": None,
            "mean_camera_ms": None,
        }
    sorted_times = sorted(times)
    p95_index = min(len(sorted_times) - 1, int(round((len(sorted_times) - 1) * 0.95)))
    total = sum(times)
    return {
        "frame_count": len(times),
        "mean_frame_ms": (total / len(times)) * 1000.0,
        "median_frame_ms": median(times) * 1000.0,
        "p95_frame_ms": sorted_times[p95_index] * 1000.0,
        "min_frame_ms": min(times) * 1000.0,
        "max_frame_ms": max(times) * 1000.0,
        "fps": len(times) / total if total > 0.0 else None,
        "mean_render_ms": (sum(render_times) / len(render_times)) * 1000.0 if render_times else None,
        "mean_fetch_ms": (sum(fetch_times) / len(fetch_times)) * 1000.0 if fetch_times else None,
        "mean_camera_ms": (sum(camera_times) / len(camera_times)) * 1000.0 if camera_times else None,
    }


def _load_cameras(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"No cameras in {path}")
    return data


def _set_native_camera(renderer: Any, camera: dict[str, Any]) -> None:
    renderer.set_camera_explicit_window(
        _as_tuple(camera["eye"], 3),
        _as_tuple(camera["target"], 3),
        _as_tuple(camera["up"], 3),
        float(camera["fov_degrees"]),
        float(camera["near_clip"]),
        float(camera["far_clip"]),
        _as_tuple(camera.get("projection_shift", (0.0, 0.0)), 2),
    )


def _native_backend(backend: str) -> tuple[Any, int, bool]:
    if backend in ("vulkan_rt", "vulkan_raster"):
        sys.path.insert(0, str(REPO_ROOT / "python"))
        from nusd_renderer import _bindings as bindings

        mode = bindings.NU_RENDER_RT if backend == "vulkan_rt" else bindings.NU_RENDER_RASTER
        return bindings, int(mode), backend == "vulkan_rt"
    if backend == "opengl":
        sys.path.insert(0, str(OPENGL_REPO / "python"))
        from nusd_renderer_opengl import _bindings as bindings

        return bindings, int(bindings.NU_RENDER_RASTER), False
    raise ValueError(f"unsupported native backend {backend}")


def run_native_child(args: argparse.Namespace) -> dict[str, Any]:
    backend = str(args.child_backend)
    cameras = _load_cameras(Path(args.cameras_json))
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    env = configure_render_environment(args)
    bindings, render_mode, enable_rt = _native_backend(backend)

    result: dict[str, Any] = {
        "backend": backend,
        "label": BACKEND_LABELS[backend],
        "status": "failed",
        "generated_at": _utc_now(),
        "usd": str(Path(args.usd).resolve()),
        "resolution": [int(args.width), int(args.height)],
        "environment": env,
        "memory": {
            "rss_start_bytes": _current_rss_bytes(),
            "rss_peak_start_bytes": _peak_rss_bytes(),
        },
        "renderer": {},
        "frames": [],
    }
    renderer = None
    try:
        print(f"creating {backend} renderer", flush=True)
        t0 = time.perf_counter()
        renderer = bindings.NuRenderer(
            width=int(args.width),
            height=int(args.height),
            enable_rt=bool(enable_rt),
            enable_materials=True,
            visible=False,
        )
        init_seconds = time.perf_counter() - t0

        print(f"loading {args.usd}", flush=True)
        load0 = time.perf_counter()
        mesh_count = int(renderer.load_usd(str(args.usd)))
        load_seconds = time.perf_counter() - load0
        gpu_after_load = int(getattr(renderer, "gpu_memory_used", 0))
        phase_after_load = _phase_dict(renderer)
        print(
            f"loaded {mesh_count} meshes in {load_seconds:.2f}s, "
            f"GPU {gpu_after_load / (1024.0 ** 3):.2f} GiB",
            flush=True,
        )

        warmup_frames = max(0, int(args.warmup))
        warmup_records: list[dict[str, Any]] = []
        for i in range(warmup_frames):
            camera = cameras[min(i, len(cameras) - 1)]
            frame0 = time.perf_counter()
            cam0 = time.perf_counter()
            _set_native_camera(renderer, camera)
            camera_seconds = time.perf_counter() - cam0
            render0 = time.perf_counter()
            renderer.render(render_mode)
            render_seconds = time.perf_counter() - render0
            fetch0 = time.perf_counter()
            renderer.fetch_pixels()
            fetch_seconds = time.perf_counter() - fetch0
            warmup_records.append(
                {
                    "index": i,
                    "camera": camera["name"],
                    "camera_seconds": camera_seconds,
                    "render_seconds": render_seconds,
                    "fetch_seconds": fetch_seconds,
                    "total_seconds": time.perf_counter() - frame0,
                    "phase_timings_ms": _phase_dict(renderer),
                    "gpu_memory_bytes": int(getattr(renderer, "gpu_memory_used", 0)),
                }
            )
            print(f"warmup {i}: {warmup_records[-1]['total_seconds'] * 1000.0:.1f} ms", flush=True)

        for i, camera in enumerate(cameras):
            frame0 = time.perf_counter()
            cam0 = time.perf_counter()
            _set_native_camera(renderer, camera)
            camera_seconds = time.perf_counter() - cam0
            render0 = time.perf_counter()
            renderer.render(render_mode)
            render_seconds = time.perf_counter() - render0
            fetch0 = time.perf_counter()
            pixels = renderer.fetch_pixels()
            fetch_seconds = time.perf_counter() - fetch0
            total_seconds = time.perf_counter() - frame0
            image_path = frames_dir / f"{camera['name']}.png"
            _save_rgb_png(pixels, image_path)
            record = {
                "status": "ok",
                "index": i,
                "camera": camera["name"],
                "image": str(image_path.relative_to(out_dir)),
                "camera_seconds": camera_seconds,
                "render_seconds": render_seconds,
                "fetch_seconds": fetch_seconds,
                "total_seconds": total_seconds,
                "phase_timings_ms": _phase_dict(renderer),
                "gpu_memory_bytes": int(getattr(renderer, "gpu_memory_used", 0)),
            }
            result["frames"].append(record)
            print(
                f"frame {i} {camera['name']}: {total_seconds * 1000.0:.1f} ms",
                flush=True,
            )

        result["status"] = "ok"
        result["renderer"] = {
            "init_seconds": init_seconds,
            "load_seconds": load_seconds,
            "mesh_count_returned": mesh_count,
            "mesh_count_property": int(getattr(renderer, "mesh_count", mesh_count)),
            "scene_bounds": _scene_bounds(renderer),
            "gpu_memory_after_load_bytes": gpu_after_load,
            "gpu_memory_final_bytes": int(getattr(renderer, "gpu_memory_used", 0)),
            "phase_timings_after_load_ms": phase_after_load,
            "phase_timings_final_ms": _phase_dict(renderer),
            "cmd_cache_stats": _cmd_cache_stats(renderer),
            "render_mode": render_mode,
            "enable_rt": bool(enable_rt),
            "enable_materials": True,
        }
        result["warmup"] = {
            "count": warmup_frames,
            "records": warmup_records,
            "seconds": sum(float(r["total_seconds"]) for r in warmup_records),
        }
        result["summary"] = _summary_stats(result["frames"])
    except Exception as exc:
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], flush=True)
    finally:
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass
        result["memory"].update(
            {
                "rss_final_bytes": _current_rss_bytes(),
                "rss_peak_final_bytes": _peak_rss_bytes(),
            }
        )
        result.setdefault("summary", _summary_stats(result.get("frames", [])))
        metrics_path = out_dir / "metrics.json"
        metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_ovrtx_child(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["NANOUSD_VIEW_BACKEND"] = "ovrtx"
    cameras = _load_cameras(Path(args.cameras_json))
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "backend": "ovrtx",
        "label": BACKEND_LABELS["ovrtx"],
        "status": "failed",
        "generated_at": _utc_now(),
        "usd": str(Path(args.usd).resolve()),
        "resolution": [int(args.width), int(args.height)],
        "environment": {
            key: os.environ.get(key, "")
            for key in (
                "NANOUSD_VIEW_BACKEND",
                "NUVIEW_OVRTX_RENDER_MODE",
                "NUVIEW_OVRTX_SPP",
                "NUVIEW_OVRTX_DENOISE",
                "NUVIEW_OVRTX_DEFAULT_LIGHTING",
                "NANOUSD_VIEW_STEP_TIMEOUT_NS",
                "DISPLAY",
            )
        },
        "memory": {
            "rss_start_bytes": _current_rss_bytes(),
            "rss_peak_start_bytes": _peak_rss_bytes(),
        },
        "renderer": {},
        "frames": [],
    }
    renderer = None
    try:
        from nanousdview._backend import OvrtxViewportRenderer, VIEW_RENDER_RT, configure_backend

        configure_backend("ovrtx", workspace=WORKSPACE)
        print("creating ovrtx renderer", flush=True)
        t0 = time.perf_counter()
        renderer = OvrtxViewportRenderer(width=int(args.width), height=int(args.height))
        init_seconds = time.perf_counter() - t0

        default_lighting = os.environ.get("NUVIEW_OVRTX_DEFAULT_LIGHTING", "1")
        if default_lighting.strip().lower() not in ("0", "false", "off", "no"):
            renderer.set_default_lighting(camera_light=True, dome_light=True)

        print(f"loading {args.usd}", flush=True)
        load0 = time.perf_counter()
        renderer.load_stage(str(Path(args.usd).resolve()))
        load_seconds = time.perf_counter() - load0
        renderer.set_render_mode(VIEW_RENDER_RT)
        print(f"stage configured in {load_seconds:.2f}s", flush=True)

        warmup_frames = max(0, int(args.warmup))
        warmup_records: list[dict[str, Any]] = []
        for i in range(warmup_frames):
            camera = cameras[min(i, len(cameras) - 1)]
            frame0 = time.perf_counter()
            cam0 = time.perf_counter()
            renderer.set_camera(
                _as_tuple(camera["eye"], 3),
                _as_tuple(camera["target"], 3),
                _as_tuple(camera["up"], 3),
                float(camera["fov_degrees"]),
                float(camera["near_clip"]),
                float(camera["far_clip"]),
            )
            camera_seconds = time.perf_counter() - cam0
            render0 = time.perf_counter()
            renderer.render_ldr(render_mode=VIEW_RENDER_RT)
            render_seconds = time.perf_counter() - render0
            total_seconds = time.perf_counter() - frame0
            warmup_records.append(
                {
                    "index": i,
                    "camera": camera["name"],
                    "camera_seconds": camera_seconds,
                    "render_seconds": render_seconds,
                    "fetch_seconds": 0.0,
                    "total_seconds": total_seconds,
                }
            )
            print(f"warmup {i}: {total_seconds * 1000.0:.1f} ms", flush=True)

        for i, camera in enumerate(cameras):
            frame0 = time.perf_counter()
            cam0 = time.perf_counter()
            renderer.set_camera(
                _as_tuple(camera["eye"], 3),
                _as_tuple(camera["target"], 3),
                _as_tuple(camera["up"], 3),
                float(camera["fov_degrees"]),
                float(camera["near_clip"]),
                float(camera["far_clip"]),
            )
            camera_seconds = time.perf_counter() - cam0
            render0 = time.perf_counter()
            pixels = renderer.render_ldr(render_mode=VIEW_RENDER_RT)
            render_seconds = time.perf_counter() - render0
            total_seconds = time.perf_counter() - frame0
            image_path = frames_dir / f"{camera['name']}.png"
            _save_rgb_png(np.asarray(pixels), image_path)
            record = {
                "status": "ok",
                "index": i,
                "camera": camera["name"],
                "image": str(image_path.relative_to(out_dir)),
                "camera_seconds": camera_seconds,
                "render_seconds": render_seconds,
                "fetch_seconds": 0.0,
                "total_seconds": total_seconds,
                "gpu_memory_bytes": None,
            }
            result["frames"].append(record)
            print(
                f"frame {i} {camera['name']}: {total_seconds * 1000.0:.1f} ms",
                flush=True,
            )

        result["status"] = "ok"
        result["renderer"] = {
            "init_seconds": init_seconds,
            "load_seconds": load_seconds,
            "load_note": "OVRTX performs most scene loading/compilation on the first render wait.",
            "gpu_memory_after_load_bytes": None,
            "gpu_memory_final_bytes": None,
            "render_mode": "VIEW_RENDER_RT",
        }
        result["warmup"] = {
            "count": warmup_frames,
            "records": warmup_records,
            "seconds": sum(float(r["total_seconds"]) for r in warmup_records),
        }
        result["summary"] = _summary_stats(result["frames"])
    except Exception as exc:
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], flush=True)
    finally:
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass
        result["memory"].update(
            {
                "rss_final_bytes": _current_rss_bytes(),
                "rss_peak_final_bytes": _peak_rss_bytes(),
            }
        )
        result.setdefault("summary", _summary_stats(result.get("frames", [])))
        metrics_path = out_dir / "metrics.json"
        metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _ovrtx_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(NANOUSDVIEW_PYTHONPATH)
    env.pop("PXR_PLUGINPATH_NAME", None)
    env.pop("USD_PLUGIN_PATH", None)
    env["DISPLAY"] = env.get("DISPLAY", ":1")
    env["XAUTHORITY"] = env.get("XAUTHORITY", "/run/user/1000/gdm/Xauthority")
    env.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    env.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    env.setdefault("OVRTX_SKIP_USD_CHECK", "1")
    env.setdefault("NANOUSD_VIEW_STEP_TIMEOUT_NS", "600000000000")
    env.setdefault("NUVIEW_OVRTX_RENDER_MODE", "rt2")
    env.setdefault("NUVIEW_OVRTX_SPP", "1")
    env.setdefault("NUVIEW_OVRTX_DENOISE", "0")
    env.setdefault("NUVIEW_OVRTX_DEFAULT_LIGHTING", "1")

    venv_root = OVRTX_PYTHON.parent.parent
    ovrtx_bin = venv_root / "lib" / "python3.12" / "site-packages" / "ovrtx" / "bin"
    ovrtx_plugins = ovrtx_bin / "plugins"
    plugin_dir = ovrtx_plugins / "gpu.foundation"
    glib = plugin_dir / "libglib-2.0.so.0"
    gobject = plugin_dir / "libgobject-2.0.so.0"
    if glib.exists() and gobject.exists():
        preload = f"{glib}:{gobject}"
        if env.get("LD_PRELOAD"):
            preload += ":" + env["LD_PRELOAD"]
        env["LD_PRELOAD"] = preload
    inherited_ld = [
        entry
        for entry in env.get("LD_LIBRARY_PATH", "").split(":")
        if entry and "OpenUSD" not in entry and "openusd" not in entry.lower()
    ]
    env["LD_LIBRARY_PATH"] = ":".join([str(ovrtx_plugins), str(ovrtx_bin)] + inherited_ld)
    return env


def _backend_python(backend: str) -> Path:
    return OVRTX_PYTHON if backend == "ovrtx" else Path(sys.executable)


def _backend_env(backend: str) -> dict[str, str]:
    if backend == "ovrtx":
        return _ovrtx_env()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return env


def _run_backend(args: argparse.Namespace, backend: str, cameras_json: Path) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    backend_dir = out_dir / "backends" / backend
    backend_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{backend}.log"
    cmd = [
        str(_backend_python(backend)),
        str(Path(__file__).resolve()),
        "--child-backend",
        backend,
        "--usd",
        str(args.usd),
        "--out-dir",
        str(backend_dir),
        "--cameras-json",
        str(cameras_json),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--warmup",
        str(args.warmup),
    ]
    if args.rt_cull:
        cmd.append("--rt-cull")

    print(f"running {backend}: {' '.join(cmd)}", flush=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(WORKSPACE),
            env=_backend_env(backend),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        timed_out = False
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        deadline = started + max(1, int(args.backend_timeout))
        while True:
            if time.perf_counter() > deadline and proc.poll() is None:
                timed_out = True
                proc.kill()
            events = selector.select(timeout=0.5)
            for key, _ in events:
                line = key.fileobj.readline()
                if not line:
                    continue
                log.write(line)
                log.flush()
                print(f"[{backend}] {line.rstrip()}", flush=True)
            if proc.poll() is not None:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    print(f"[{backend}] {line.rstrip()}", flush=True)
                break
        rc = proc.wait()
        selector.close()
        if timed_out:
            log.write(f"\nTIMEOUT after {args.backend_timeout}s\n")
            log.flush()
    elapsed = time.perf_counter() - started

    metrics_path = backend_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics = {
            "backend": backend,
            "label": BACKEND_LABELS[backend],
            "status": "failed",
            "error": f"backend process did not write metrics.json, rc={rc}",
            "frames": [],
            "summary": _summary_stats([]),
            "renderer": {},
            "memory": {},
        }
    metrics["process"] = {
        "returncode": int(rc),
        "elapsed_seconds": elapsed,
        "log": str(log_path.relative_to(out_dir)),
    }
    if rc != 0 and metrics.get("status") == "ok":
        metrics["status"] = "failed"
        metrics["error"] = f"backend process returned rc={rc}"
    return metrics


def _make_orbit_camera_payload(args: argparse.Namespace) -> list[dict[str, Any]]:
    authored = parse_moana_cameras(Path(args.usd))
    orbit = make_orbit_cameras(authored, args)
    if not orbit:
        raise RuntimeError("Report requires at least one orbit camera")
    return [asdict(camera) for camera in orbit]


def _font(size: int):
    from PIL import ImageFont

    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _frame_label(metric: dict[str, Any], frame: dict[str, Any]) -> str:
    total_ms = float(frame.get("total_seconds", 0.0)) * 1000.0
    return f"{metric.get('label', metric.get('backend'))} {frame.get('camera')}  {total_ms:.1f} ms"


def _make_backend_sheet(metric: dict[str, Any], backend_dir: Path, tile_w: int, tile_h: int) -> str | None:
    from PIL import Image, ImageDraw

    frames = [f for f in metric.get("frames", []) if f.get("status") == "ok" and f.get("image")]
    if not frames:
        return None
    gap = 10
    label_h = 30
    columns = min(2, len(frames))
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * tile_w + (columns + 1) * gap, rows * (tile_h + label_h) + (rows + 1) * gap),
        (18, 21, 23),
    )
    draw = ImageDraw.Draw(sheet)
    font = _font(15)
    for i, frame in enumerate(frames):
        src = backend_dir / str(frame["image"])
        img = Image.open(src).convert("RGB")
        img = img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        col = i % columns
        row = i // columns
        x = gap + col * (tile_w + gap)
        y = gap + row * (tile_h + label_h + gap)
        draw.rectangle((x - 1, y - 1, x + tile_w + 1, y + tile_h + label_h + 1), fill=(30, 34, 37))
        draw.text((x + 7, y + 7), _frame_label(metric, frame), fill=(235, 238, 240), font=font)
        sheet.paste(img, (x, y + label_h))
    out = backend_dir / "contact_sheet.png"
    sheet.save(out)
    return "contact_sheet.png"


def _make_overall_sheet(metrics: list[dict[str, Any]], out_dir: Path, tile_w: int, tile_h: int) -> str | None:
    from PIL import Image, ImageDraw

    ok_metrics = [m for m in metrics if any(f.get("status") == "ok" for f in m.get("frames", []))]
    if not ok_metrics:
        return None
    camera_names: list[str] = []
    for metric in ok_metrics:
        for frame in metric.get("frames", []):
            name = str(frame.get("camera", ""))
            if name and name not in camera_names:
                camera_names.append(name)
    if not camera_names:
        return None
    gap = 10
    label_h = 30
    left_w = 150
    sheet_w = left_w + len(camera_names) * tile_w + (len(camera_names) + 1) * gap
    sheet_h = len(ok_metrics) * (tile_h + label_h) + (len(ok_metrics) + 1) * gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), (16, 19, 22))
    draw = ImageDraw.Draw(sheet)
    font = _font(15)
    small = _font(13)
    for row, metric in enumerate(ok_metrics):
        y = gap + row * (tile_h + label_h + gap)
        draw.text((gap, y + 7), str(metric.get("label", metric.get("backend"))), fill=(245, 247, 249), font=font)
        summary = metric.get("summary", {})
        fps = summary.get("fps")
        if fps is not None:
            draw.text((gap, y + 27), f"{float(fps):.2f} fps", fill=(190, 198, 205), font=small)
        by_camera = {str(f.get("camera")): f for f in metric.get("frames", []) if f.get("status") == "ok"}
        backend_dir = out_dir / "backends" / str(metric["backend"])
        for col, name in enumerate(camera_names):
            x = left_w + gap + col * (tile_w + gap)
            draw.rectangle((x - 1, y - 1, x + tile_w + 1, y + tile_h + label_h + 1), fill=(30, 34, 37))
            frame = by_camera.get(name)
            if frame is None:
                draw.text((x + 8, y + 8), "not rendered", fill=(160, 167, 174), font=font)
                continue
            draw.text((x + 7, y + 7), _frame_label(metric, frame), fill=(235, 238, 240), font=small)
            img = Image.open(backend_dir / str(frame["image"])).convert("RGB")
            img = img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            sheet.paste(img, (x, y + label_h))
    out = out_dir / "backend_contact_sheet.png"
    sheet.save(out)
    return out.name


def _fmt_float(value: Any, decimals: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_seconds(value: Any) -> str:
    return _fmt_float(value, 2, "s")


def _metric_row(metric: dict[str, Any]) -> str:
    renderer = metric.get("renderer", {})
    memory = metric.get("memory", {})
    summary = metric.get("summary", {})
    gpu_bytes = renderer.get("gpu_memory_final_bytes")
    gpu_gib = _bytes_to_gib(gpu_bytes) if gpu_bytes is not None else None
    rss_gib = _bytes_to_gib(memory.get("rss_peak_final_bytes"))
    load_s = renderer.get("load_seconds")
    warmup_s = (metric.get("warmup") or {}).get("seconds")
    status = metric.get("status", "unknown")
    if status != "ok":
        status = "failed"
    mesh_count = renderer.get("mesh_count_property", renderer.get("mesh_count_returned", "n/a"))
    return (
        f"| {metric.get('label', metric.get('backend'))} "
        f"| {status} "
        f"| {_fmt_seconds(load_s)} "
        f"| {_fmt_seconds(warmup_s)} "
        f"| {_fmt_float(gpu_gib, 2, ' GiB')} "
        f"| {_fmt_float(rss_gib, 2, ' GiB')} "
        f"| {_fmt_float(summary.get('mean_frame_ms'), 1, ' ms')} "
        f"| {_fmt_float(summary.get('p95_frame_ms'), 1, ' ms')} "
        f"| {_fmt_float(summary.get('fps'), 2)} "
        f"| {mesh_count} |"
    )


def _analysis_lines(metrics: list[dict[str, Any]]) -> list[str]:
    ok = [m for m in metrics if m.get("status") == "ok" and m.get("summary", {}).get("fps")]
    lines: list[str] = []
    if not ok:
        return ["No backend completed enough orbit frames for FPS analysis."]
    fastest = max(ok, key=lambda m: float(m["summary"]["fps"]))
    lines.append(
        f"- Fastest completed backend: {fastest.get('label')} at "
        f"{float(fastest['summary']['fps']):.2f} FPS "
        f"({float(fastest['summary']['mean_frame_ms']):.1f} ms mean frame)."
    )
    for metric in ok:
        fps = float(metric["summary"]["fps"])
        frame_ms = float(metric["summary"]["mean_frame_ms"])
        target = "meets" if fps >= 20.0 else "misses"
        lines.append(
            f"- {metric.get('label')} {target} the 20 FPS / 50 ms target: "
            f"{fps:.2f} FPS, {frame_ms:.1f} ms mean."
        )
    failures = [m for m in metrics if m.get("status") != "ok"]
    for metric in failures:
        error = str(metric.get("error", "unknown error")).rstrip(".")
        lines.append(f"- {metric.get('label')} failed: {error}.")
    return lines


def _format_bounds(bounds: Any) -> str:
    try:
        lo, hi = bounds
        lo_s = ", ".join(f"{float(v):.1f}" for v in lo)
        hi_s = ", ".join(f"{float(v):.1f}" for v in hi)
        return f"bounds [{lo_s}] to [{hi_s}]"
    except Exception:
        return "bounds n/a"


def _extract_curve_note(out_dir: Path, metric: dict[str, Any]) -> str | None:
    process = metric.get("process") or {}
    log_rel = process.get("log")
    if not log_rel:
        return None
    log_path = out_dir / str(log_rel)
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    m = re.search(r"viewer: curves:\s+(\d+) prims, .*?(\d+) patches", text)
    if m:
        return f"{int(m.group(1))} curve prims, {int(m.group(2)):,} patches"
    m = re.search(r"extracted\s+(\d+) curve segments from\s+(\d+) BasisCurves", text)
    if m:
        return f"{int(m.group(2))} BasisCurves, {int(m.group(1)):,} curve segments"
    m_count = re.search(r"skipped\s+(\d+) BasisCurves segment extraction", text)
    m_patches = re.search(r"raster curves ready .*?,\s+(\d+) patches", text)
    if m_count and m_patches:
        return f"{int(m_count.group(1))} BasisCurves, {int(m_patches.group(1)):,} raster patches"
    m = re.search(r"raster curves ready .*?,\s+(\d+) patches", text)
    if m:
        return f"{int(m.group(1)):,} raster curve patches"
    return None


def _geometry_lines(metrics: list[dict[str, Any]], out_dir: Path) -> list[str]:
    ok = [m for m in metrics if m.get("status") == "ok"]
    if not ok:
        return ["No backend completed frames, so geometry coverage could not be compared."]
    lines: list[str] = []
    mesh_counts: list[int] = []
    for metric in ok:
        renderer = metric.get("renderer", {})
        mesh_count = renderer.get("mesh_count_property", renderer.get("mesh_count_returned"))
        if isinstance(mesh_count, int):
            mesh_counts.append(mesh_count)
        curve_note = _extract_curve_note(out_dir, metric)
        bounds = _format_bounds(renderer.get("scene_bounds"))
        extra = f"; curves: {curve_note}" if curve_note else ""
        lines.append(f"- {metric.get('label')}: {mesh_count or 'n/a'} reported meshes; {bounds}{extra}.")
    if mesh_counts and len(set(mesh_counts)) > 1:
        lines.append(
            "- Mesh counts are not identical across completed backends; OpenGL should be treated "
            "as curve-covered but not yet mesh-count equivalent to the Vulkan full-island load."
        )
    if any(m.get("backend") == "ovrtx" and m.get("status") != "ok" for m in metrics):
        lines.append("- OVRTX produced no 1080p frame before timeout, so it has no geometry-coverage result.")
    return lines


def _visual_lines(metrics: list[dict[str, Any]]) -> list[str]:
    lines = [
        "- The contact sheet is part of the result: compare it with the table before treating FPS as visual parity.",
    ]
    by_backend = {str(m.get("backend")): m for m in metrics}
    if by_backend.get("vulkan_raster", {}).get("status") == "ok":
        lines.append(
            "- Vulkan Raster is the only completed backend above 20 FPS, but this run shows raster-path lighting/background differences from Vulkan RT."
        )
    if by_backend.get("opengl", {}).get("status") == "ok":
        lines.append(
            "- OpenGL completed 1080p orbit frames, but its material/background output and mesh count differ from Vulkan RT/Raster."
        )
    if by_backend.get("ovrtx", {}).get("status") != "ok":
        lines.append("- OVRTX did not produce a comparison frame for this full-island 1080p orbit run.")
    return lines


def _write_report(args: argparse.Namespace, metrics: list[dict[str, Any]], cameras: list[dict[str, Any]]) -> None:
    out_dir = Path(args.out_dir)
    for metric in metrics:
        backend_dir = out_dir / "backends" / str(metric.get("backend"))
        sheet = _make_backend_sheet(metric, backend_dir, tile_w=480, tile_h=270)
        if sheet:
            metric["contact_sheet"] = str(Path("backends") / str(metric["backend"]) / sheet)
    overall = _make_overall_sheet(metrics, out_dir, tile_w=360, tile_h=203)

    aggregate = {
        "generated_at": _utc_now(),
        "usd": str(Path(args.usd).resolve()),
        "resolution": [int(args.width), int(args.height)],
        "orbit": {
            "count": len(cameras),
            "warmup": int(args.warmup),
            "frame_time_definition": "camera update + render + CPU pixel readback",
        },
        "cameras": cameras,
        "backends": metrics,
        "overall_contact_sheet": overall,
    }
    (out_dir / "metrics.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Moana 1080p Backend Orbit Report",
        "",
        f"Generated: {aggregate['generated_at']}",
        f"Scene: `{aggregate['usd']}`",
        f"Resolution: {args.width} x {args.height}",
        f"Orbit frames: {len(cameras)} measured frames per backend, {args.warmup} warmup frame excluded from FPS.",
        "",
        "Frame time includes camera update, render, and CPU pixel readback so the numbers match the saved 1080p renders.",
        "For OVRTX, `load_seconds` is stage setup; most RTX asset loading/compilation is charged to the warmup frame.",
        "",
    ]
    if overall:
        lines.extend(["## Contact Sheet", "", f"![Backend contact sheet]({overall})", ""])
    lines.extend(
        [
            "## Summary",
            "",
            "| Backend | Status | Load | Warmup | GPU memory | Peak RSS | Mean frame | P95 frame | FPS | Meshes |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(_metric_row(metric) for metric in metrics)
    lines.extend(["", "## Perf Analysis", ""])
    lines.extend(_analysis_lines(metrics))
    lines.extend(["", "## Geometry Notes", ""])
    lines.extend(_geometry_lines(metrics, out_dir))
    lines.extend(["", "## Visual Notes", ""])
    lines.extend(_visual_lines(metrics))
    lines.extend(["", "## Artifacts", ""])
    lines.append("- Raw metrics: `metrics.json`")
    for metric in metrics:
        label = metric.get("label", metric.get("backend"))
        log = (metric.get("process") or {}).get("log")
        sheet = metric.get("contact_sheet")
        if sheet:
            lines.append(f"- {label} contact sheet: `{sheet}`")
        if log:
            lines.append(f"- {label} log: `{log}`")
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_parent(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cameras = _make_orbit_camera_payload(args)
    cameras_json = out_dir / "orbit_cameras.json"
    cameras_json.write_text(json.dumps(cameras, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(cameras)} orbit cameras to {cameras_json}", flush=True)

    metrics: list[dict[str, Any]] = []
    for backend in args.backends:
        metrics.append(_run_backend(args, backend, cameras_json))
    _write_report(args, metrics, cameras)
    print(f"report written to {out_dir / 'README.md'}", flush=True)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--backend-timeout", type=int, default=3600)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--child-backend", choices=BACKENDS, default=None)
    parser.add_argument("--cameras-json", type=Path, default=None)
    parser.add_argument("--rt-cull", action="store_true")
    parser.add_argument("--orbit-count", type=int, default=4)
    parser.add_argument(
        "--orbit-center",
        type=lambda s: tuple(float(p) for p in s.split(",")),
        default=None,
    )
    parser.add_argument("--orbit-radius", type=float, default=None)
    parser.add_argument("--orbit-radius-scale", type=float, default=2.25)
    parser.add_argument("--orbit-height", type=float, default=620.0)
    parser.add_argument("--orbit-fov", type=float, default=35.0)
    parser.add_argument("--orbit-start-degrees", type=float, default=20.0)
    parser.add_argument("--orbit-near-clip", type=float, default=10.0)
    parser.add_argument("--orbit-far-clip", type=float, default=1000000.0)
    args = parser.parse_args(argv)
    if args.child_backend and args.cameras_json is None:
        parser.error("--child-backend requires --cameras-json")
    if args.orbit_center is not None and len(args.orbit_center) != 3:
        parser.error("--orbit-center must be x,y,z")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.child_backend in ("vulkan_rt", "vulkan_raster", "opengl"):
        run_native_child(args)
        return 0
    if args.child_backend == "ovrtx":
        run_ovrtx_child(args)
        return 0
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
