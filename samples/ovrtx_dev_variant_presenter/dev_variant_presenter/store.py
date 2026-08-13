# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Save/load projects and their named timeline "track views".

Pure file IO — no ovrtx/pxr. A project bundles a stage + base selection + display +
per-camera looks/framing + the workspace timeline + its named track views. Track views
live INSIDE the project: a view's clips reference the stage's variant sets and cameras
(and lean on the project's per-camera overrides), so a view is only valid for the project
it was authored in — never a global pool.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_PROJ = Path(__file__).resolve().parents[1] / "projects"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip() or "view"


# ----- projects: a named bundle of USD + base look + display + its track views -----
def _proj_dir() -> Path:
    _PROJ.mkdir(parents=True, exist_ok=True)
    return _PROJ


def _proj_path(name: str) -> Path:
    return _proj_dir() / f"{_safe(name)}.json"


def save_project(name: str, usd_path: str, base_selection: list, display: dict,
                 timelines: dict | None = None,
                 looks: dict | None = None, xforms: dict | None = None,
                 timeline: dict | None = None, camera: str = "") -> str:
    """Write the project. `timeline` + `camera` are the WORKSPACE (the editor strip's
    working timeline + selected camera). `timelines` is this project's named track-view
    library; pass None to PRESERVE whatever views are already saved in the project on disk
    (a workspace re-save must never wipe the library — views are managed via
    save_project_view, not here)."""
    p = _proj_path(name)
    if timelines is None:
        existing = load_project(name) or {}
        timelines = existing.get("timelines", {})
    p.write_text(json.dumps({
        "name": name, "usd_path": usd_path, "base_selection": base_selection,
        "display": display, "looks": looks or {}, "xforms": xforms or {},
        "timeline": timeline, "camera": camera,
        "timelines": timelines}, indent=2), encoding="utf-8")
    return name


# ----- project track views (a view lives inside its project) -----
def project_view_names(project: str) -> list[str]:
    rec = load_project(project)
    return sorted((rec or {}).get("timelines", {}).keys()) if rec else []


def save_project_view(project: str, view_name: str, timeline: dict) -> bool:
    """Add/replace a named track view inside the project. False if the project is missing
    (a view has nowhere valid to live without its project)."""
    rec = load_project(project)
    if rec is None:
        return False
    rec.setdefault("timelines", {})[view_name] = timeline
    _proj_path(project).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return True


def load_project_view(project: str, view_name: str) -> dict | None:
    rec = load_project(project)
    tl = (rec or {}).get("timelines", {}).get(view_name) if rec else None
    if tl is None:
        return None
    return {"name": view_name, "timeline": tl, "usd_path": rec.get("usd_path", "")}


def delete_project_view(project: str, view_name: str) -> bool:
    rec = load_project(project)
    if rec is None or view_name not in rec.get("timelines", {}):
        return False
    del rec["timelines"][view_name]
    _proj_path(project).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return True


def list_projects() -> list[dict]:
    out = []
    for f in sorted(_proj_dir().glob("*.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            out.append({"name": r.get("name", f.stem), "usd_path": r.get("usd_path", "")})
        except (OSError, ValueError):
            pass
    return out


def load_project(name: str) -> dict | None:
    p = _proj_dir() / f"{_safe(name)}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def delete_project(name: str) -> bool:
    p = _proj_dir() / f"{_safe(name)}.json"
    if p.is_file():
        p.unlink()
        return True
    return False
