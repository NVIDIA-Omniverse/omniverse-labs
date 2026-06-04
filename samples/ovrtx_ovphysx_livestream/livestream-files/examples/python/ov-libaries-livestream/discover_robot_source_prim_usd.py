# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover the robot root prim path using pxr ``Usd.Stage`` and prim traversal.

Runs in a process **without** ovrtx (same family as ``clone_offset_layer_usd.py``)::

    uv run --with usd-core python discover_robot_source_prim_usd.py <usd_url_or_path>

Prints **one line** to stdout: the absolute prim path (e.g. ``/World/GR1T2_fourier_hand_6dof``).

Opens a composed ``Usd.Stage`` (same URL/path handling as ``clone_offset_layer_usd.py``), then walks
the stage with ``Usd.PrimRange.AllPrims(root_prim)`` (includes prims that default ``Traverse()`` would
skip). We consider **direct children of ``/World``** only.

Robot-OVRTX’s GR-1 root often has an **empty** ``typeName`` until its payload resolves, so
``IsA(UsdGeom.Xformable)`` is false offline; we still pick it via **authored payloads/references**, and
we exclude cameras, meshes, lights, and ``Scope`` prims like ``Looks`` that are not composition roots.
"""

from __future__ import annotations

import sys
import urllib.request

from pxr import Sdf, Usd, UsdGeom


def open_stage(usd_url_or_path: str) -> Usd.Stage:
    """Open a ``Usd.Stage`` from a filesystem path or from ``http(s)`` USDA text."""
    if usd_url_or_path.startswith("http://") or usd_url_or_path.startswith("https://"):
        with urllib.request.urlopen(usd_url_or_path, timeout=120) as resp:
            text = resp.read().decode("utf-8")
        layer = Sdf.Layer.CreateAnonymous(".usda")
        layer.ImportFromString(text)
        stage = Usd.Stage.Open(layer)
    else:
        stage = Usd.Stage.Open(usd_url_or_path)
    if not stage:
        raise SystemExit(f"Failed to open Usd.Stage for {usd_url_or_path!r}")
    return stage


_SKIP_NAME_SUBSTR = (
    "ground",
    "plane",
    "floor",
    "grid",
    "light",
    "camera",
    "dome",
    "environment",
    "physics",
    "collision",
    "render",
)

_NAME_HINTS = (
    "gr1",
    "robot",
    "fourier",
    "humanoid",
    "manipulator",
    "hand",
    "biped",
)

_LIGHT_TYPENAMES = frozenset(
    {
        "DomeLight",
        "RectLight",
        "SphereLight",
        "DiskLight",
        "CylinderLight",
        "DistantLight",
        "PluginLight",
    }
)

# Robot-OVRTX sample: prefer this hand / robot root over other high-scoring /World children
# (e.g. alternate roots or scopes that win on composition score alone).
_PREFERRED_ROBOT_PATHS: tuple[str, ...] = ("/World/GR1T2_fourier_hand_6dof",)


def _composition_score(prim: Usd.Prim) -> int:
    score = 0
    if prim.HasAuthoredPayloads():
        score += 100
    if prim.HasAuthoredReferences():
        score += 50
    return score


def _skip_prim_by_schema_or_role(prim: Usd.Prim) -> bool:
    """Drop obvious set-dressing / render prims; keep composition roots (payload/ref) for robot."""
    if prim.IsA(UsdGeom.Camera):
        return True
    if prim.IsA(UsdGeom.Mesh):
        return True
    tname = prim.GetTypeName()
    if tname in _LIGHT_TYPENAMES:
        return True
    # e.g. /World/Looks — not a clone target
    if tname == "Scope" and not (prim.HasAuthoredPayloads() or prim.HasAuthoredReferences()):
        return True
    return False


def _is_robot_candidate_prim(prim: Usd.Prim) -> bool:
    """Robot root is usually Xform, or an unloaded payload arc with empty typeName (GR-1 sample)."""
    if prim.HasAuthoredPayloads() or prim.HasAuthoredReferences():
        return True
    if prim.GetTypeName() == "Xform" and prim.IsA(UsdGeom.Xformable):
        return True
    return False


def discover_robot_root_prim_path(stage: Usd.Stage) -> str:
    """Pick the best robot-like prim among **direct children of** ``/World``."""
    world_path = Sdf.Path("/World")
    world = stage.GetPrimAtPath(world_path)
    if not world.IsValid():
        raise SystemExit("No valid prim at /World")

    for preferred in _PREFERRED_ROBOT_PATHS:
        ppath = Sdf.Path(preferred)
        if ppath.GetParentPath() != world_path:
            continue
        prim = stage.GetPrimAtPath(ppath)
        if not prim.IsValid():
            continue
        if _skip_prim_by_schema_or_role(prim):
            continue
        if not _is_robot_candidate_prim(prim):
            continue
        return preferred

    scored: list[tuple[int, str]] = []

    # AllPrims: visit full stage from pseudo-root; filter to direct children of /World.
    for prim in Usd.PrimRange.AllPrims(stage.GetPseudoRoot()):
        path = prim.GetPath()
        if path.GetParentPath() != world_path:
            continue

        name_lower = path.name.lower()
        if any(s in name_lower for s in _SKIP_NAME_SUBSTR):
            continue
        if _skip_prim_by_schema_or_role(prim):
            continue
        if not _is_robot_candidate_prim(prim):
            continue

        score = _composition_score(prim)
        for hint in _NAME_HINTS:
            if hint in name_lower:
                score += 15
        # Strong tie-break toward the GR-1 / Fourier hand root on Robot-OVRTX when multiple candidates score high.
        if "gr1t2" in name_lower or "fourier_hand" in name_lower:
            score += 80

        scored.append((score, str(path)))

    if not scored:
        raise SystemExit(
            "No robot-like direct child of /World (payload/ref/Xform) passed filters; "
            "inspect the stage in usdview or adjust heuristics."
        )

    scored.sort(key=lambda t: (-t[0], t[1]))
    return scored[0][1]


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: discover_robot_source_prim_usd.py <usd_url_or_path>", file=sys.stderr)
        raise SystemExit(2)
    stage = open_stage(sys.argv[1])
    path = discover_robot_root_prim_path(stage)
    sys.stdout.write(path)


if __name__ == "__main__":
    main()
