# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warm the installed OVRTX shader cache before the user's first render."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Callable

from . import bundled_runtime
from .preflight import ensure_native_client_path


_CAMERA_PATH = "/World/Camera"
_PROGRESS_INTERVAL_SECONDS = 1.0
_SCENE = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
    def Cube "Cube" { double size = 1 }
    def Camera "Camera"
    {
        double3 xformOp:translate = (0, -4, 0)
        quatd xformOp:orient = (0.7071068, 0.7071068, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    }
    def DistantLight "Light" { float inputs:intensity = 50000 }
}

def "Render"
{
    def "OmniverseKit"
    {
        def "HydraTextures"
        {
            def RenderProduct "ViewportTexture0"
            {
                rel camera = </World/Camera>
                rel orderedVars = </Render/Vars/LdrColor>
                uniform int2 resolution = (1, 1)
                token omni:rtx:rendermode = "RealTimePathTracing"
            }
        }
    }
    def Scope "Vars"
    {
        def RenderVar "LdrColor"
        {
            token dataType = "color4f"
            string sourceName = "LdrColor"
            token sourceType = "raw"
        }
    }
}
"""


def warm_shader_cache(
    runtime_root: Path,
    storage_root: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> None:
    """Render and discard one tiny frame using the newly installed runtime."""

    started = time.monotonic()
    done = threading.Event()
    ticker: threading.Thread | None = None
    if progress is not None:
        def report_elapsed() -> None:
            progress(_progress_message(0))
            while not done.wait(_PROGRESS_INTERVAL_SECONDS):
                progress(_progress_message(int(time.monotonic() - started)))

        ticker = threading.Thread(target=report_elapsed, daemon=True)
        ticker.start()
    try:
        from .ovrtx_session_controller import OvrtxSessionController
        from .render_requests import RenderRequest

        defaults = bundled_runtime.defaults(root=runtime_root)
        ensure_native_client_path(defaults.native_client_path)
        with tempfile.TemporaryDirectory(prefix="ovrtx-warmup-", dir=storage_root) as temporary:
            scene = Path(temporary) / "warmup.usda"
            scene.write_text(_SCENE, encoding="utf-8")
            request = RenderRequest(
                input_usd_path=str(scene),
                width=1,
                height=1,
                min_samples=1,
                max_samples=1,
                camera_prim_path=_CAMERA_PATH,
                worker_command=defaults.worker_command,
            )
            controller = OvrtxSessionController(
                simulation_id=f"ovrtx-blender-warmup-{os.getpid()}"
            )
            primary_error: BaseException | None = None
            try:
                controller.ensure(request)
                controller.render(
                    request,
                    additional_samples=1,
                )  # One discarded frame warms the cache.
            except BaseException as exc:
                primary_error = exc
                raise
            finally:
                try:
                    stop_status = controller.shutdown()
                except Exception:
                    if primary_error is None:
                        raise
                else:
                    if (
                        stop_status not in {"stopped", "not_found"}
                        and primary_error is None
                    ):
                        raise RuntimeError(
                            "OVRTX warmup cleanup was not confirmed "
                            f"(shutdown status: {stop_status!r})"
                        )
    finally:
        done.set()
        if ticker is not None:
            ticker.join()


def _progress_message(elapsed_seconds: int) -> str:
    minutes, seconds = divmod(max(0, elapsed_seconds), 60)
    return f"Warming shader cache (can take several minutes) — {minutes}:{seconds:02d} elapsed"


__all__ = ["warm_shader_cache"]
