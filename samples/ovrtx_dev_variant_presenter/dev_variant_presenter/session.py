# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Crash-resilient session checkpoint.

All live state — per-camera looks and framing overrides, the variant selection,
the active camera, the free-navigation viewer pose, the stream resolution — lives
in render-server process memory, and the ovrtx/ovstream native layer can hard-die
at any moment (run_server.ps1 relaunches it). A background thread snapshots that
state to a JSON file every couple of seconds when it changes; when the SAME stage
is reopened (the web client auto-reopens its last stage after a server restart),
/api/open merges the checkpoint back in. A crash then costs at most the last
couple of seconds of dialing instead of the whole session.

Explicit data always wins: a project load passes its own looks/xforms in the open
body, and those are never overridden by the checkpoint. The checkpoint is a
RECOVERY mechanism, not a second project store.

Reads of runtime._looks/_xforms/_camera are racy by design (same pattern as
/api/stage badge reads) — a torn read produces one slightly stale checkpoint that
the next tick overwrites.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

DEFAULT_PATH = "data/session.json"


def snapshot(state, runtime) -> dict | None:
    """One checkpoint dict of the current session, or None if no stage is open."""
    usd = getattr(state, "user_usd", "") or ""
    if not usd or getattr(state, "stage_info", None) is None:
        return None
    cam_ctl = getattr(runtime, "_camera", None)
    pose = None
    if cam_ctl is not None:
        try:
            pose = {"m": [float(x) for x in cam_ctl.to_xform().reshape(-1)],
                    "dist": float(cam_ctl.distance)}
        except Exception:  # noqa: BLE001 - torn read mid-navigation; next tick wins
            pose = None
    settings = getattr(runtime, "_settings", None)
    res = list(settings.stream_resolution) if settings is not None else None
    return {
        "usd": usd,
        "source_url": getattr(state, "source_url", "") or "",
        "camera": getattr(state, "active_camera", "") or "",
        "selection": list(getattr(state, "live_selection", []) or []),
        "looks": {k: dict(v) for k, v in dict(getattr(runtime, "_looks", {})).items() if v},
        "xforms": {k: dict(v) for k, v in dict(getattr(runtime, "_xforms", {})).items() if v},
        "pose": pose,
        "resolution": res,
    }


def load(path: str = DEFAULT_PATH) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    for _ in range(5):   # Windows: a concurrent reader briefly blocks the replace
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)


def start_checkpointer(state, runtime, path: str = DEFAULT_PATH,
                       interval: float = 2.0) -> threading.Thread:
    """Daemon thread: write the checkpoint whenever the session state changes."""
    p = Path(path)

    def loop():
        last = ""
        while True:
            time.sleep(interval)
            try:
                snap = snapshot(state, runtime)
                if snap is None:
                    continue
                text = json.dumps(snap, indent=1)
                if text == last:
                    continue
                p.parent.mkdir(parents=True, exist_ok=True)
                _write_atomic(p, text)
                last = text
            except Exception:  # noqa: BLE001 - checkpointing must never hurt the app
                pass

    t = threading.Thread(target=loop, name="session-checkpoint", daemon=True)
    t.start()
    return t
