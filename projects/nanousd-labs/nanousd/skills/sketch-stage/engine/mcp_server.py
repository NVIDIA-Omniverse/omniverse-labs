# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP server exposing the placement Session as tools.

Configured via env vars:
  SKETCH_ANCHOR  — path to the anchor sketch.json
  SKETCH_PACK    — path to asset pack directory (with pack.json)
  SKETCH_OUT_DIR — output directory (events.jsonl + final root.usd)

Defaults match the in-repo demo paths. Run with stdio transport:
  python mcp_server.py
"""
from __future__ import annotations

import os
import os
import sys
from pathlib import Path

# Ensure local imports resolve (session.py / spatial_index.py).
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from session import Session  # noqa: E402

mcp = FastMCP("sketch-incremental-placement")

_DEFAULT_ANCHOR = ""    # no built-in anchor; caller must pass one
_DEFAULT_PACK = os.environ.get("SKETCH_STAGE_DEFAULT_PACK", "")
_DEFAULT_OUT = str(Path.home() / ".cache" / "sketch-stage" / "demo")

_session: Session | None = None


def _sess() -> Session:
    """Return the active Session, or auto-open one from env-var defaults
    if none has been opened yet (legacy convenience). Raises with a
    friendly message if neither is available.
    """
    global _session
    if _session is not None:
        return _session
    anchor = os.environ.get("SKETCH_ANCHOR", _DEFAULT_ANCHOR)
    pack = os.environ.get("SKETCH_PACK", _DEFAULT_PACK)
    out = os.environ.get("SKETCH_OUT_DIR", _DEFAULT_OUT)
    if not (anchor and Path(anchor).exists()) and not pack:
        raise RuntimeError(
            "no sketch session is open; call open_session(anchor=..., pack=..., "
            "out=...) first. (You may also set SKETCH_ANCHOR / SKETCH_PACK / "
            "SKETCH_OUT_DIR env vars in mcpServers config for an auto-open "
            "default, but they are not required.)")
    _session = Session(Path(anchor) if anchor else None,
                        Path(pack) if pack else None,
                        Path(out))
    return _session


@mcp.tool()
def open_session(anchor: str | None = None,
                 pack: str | None = None,
                 out: str | None = None) -> dict:
    """Open (or replace) the active sketch session with explicit per-call
    anchor / pack / out. This is the recommended way to use the MCP server
    in a multi-project setup — register the server once in
    `~/.claude/settings.json` with NO env vars, and the LLM calls this at
    the start of each task to point the session at the right paths.

    Args (all optional; falls back to SKETCH_ANCHOR / SKETCH_PACK /
    SKETCH_OUT_DIR env vars if unset):
      anchor — path to the anchor sketch.json (use `null` for empty canvas)
      pack   — path to the asset pack root (flat or manifest)
      out    — output directory (events.jsonl + realized USD land here)

    Replaces any prior session in-place. The previous Session's
    `events.jsonl` is left on disk; in-memory state is discarded.
    """
    global _session
    if _session is not None:
        try:
            _session.close()
        except Exception:
            pass
        _session = None
    a = anchor if anchor is not None else os.environ.get("SKETCH_ANCHOR", _DEFAULT_ANCHOR)
    p = pack if pack is not None else os.environ.get("SKETCH_PACK", _DEFAULT_PACK)
    o = out if out is not None else os.environ.get("SKETCH_OUT_DIR", _DEFAULT_OUT)
    _session = Session(Path(a) if a else None,
                        Path(p) if p else None,
                        Path(o))
    return {
        "ok": True,
        "anchor": a, "pack": p, "out": o,
        "placementCount": len(_session.index),
        "zoneCount": len(_session.zones),
        "archetypeCount": len(_session._archetype_meta),
        "surfaceCount": (len(_session.surface_registry)
                         if _session.surface_registry else 0),
    }


@mcp.tool()
def close_session() -> dict:
    """Close the active session. Subsequent stateful tool calls will need a
    fresh `open_session(...)` (or env-var defaults) to proceed."""
    global _session
    if _session is not None:
        try:
            _session.close()
        except Exception:
            pass
    _session = None
    return {"ok": True}


@mcp.tool()
def shutdown(graceSeconds: float = 0.2) -> dict:
    """Tell the MCP server process to exit cleanly. Use at the end of a task
    so the skill doesn't leave a stale server lingering on its port (HTTP
    path) or holding a child process open (stdio MCP path). Exit happens
    in a background thread so this tool's response is delivered first."""
    global _session
    if _session is not None:
        try:
            _session.close()
        except Exception:
            pass
    _session = None
    import os
    import signal
    import threading
    pid = os.getpid()

    def _exit_soon():
        import time
        time.sleep(max(0.05, graceSeconds))
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            os._exit(0)

    threading.Thread(target=_exit_soon, daemon=True).start()
    return {"ok": True, "pid": pid, "graceSeconds": graceSeconds}


@mcp.tool()
def query_stage_graph(archetype: str | None = None) -> list[dict]:
    """List every current placement: id, archetype, posM (x,y,z meters),
    slotM (widthM/depthM/heightM as a 3-list), yawDeg.

    Pass archetype to filter. Use this first to see what's in the stage."""
    return _sess().query_stage_graph(archetype=archetype)


@mcp.tool()
def query_nearby(posM: list[float], radiusM: float,
                 archetype: str | None = None, limit: int = 50) -> list[dict]:
    """List placements within radiusM of point posM (XY-distance, ignores Z),
    sorted nearest-first.

    Use this before placing something so you understand the local context."""
    return _sess().query_nearby(posM, radiusM, archetype, limit)


@mcp.tool()
def query_collision(posM: list[float], slotM: list[float]) -> dict:
    """Check whether a slotM (widthM, depthM, heightM) placed at posM would
    collide with an existing placement.

    Returns {ok, colliders[]}. ok=True means it's safe to place."""
    return _sess().query_collision(posM, slotM)


@mcp.tool()
def query_zones() -> list[dict]:
    """List zones in the current sketch: id, boundsM (widthM/depthM/heightM),
    originM (world position of the zone's local origin), type."""
    return _sess().query_zones()


@mcp.tool()
def query_source_hierarchy(prefix: str | None = None,
                            maxDepth: int | None = None) -> dict:
    """Return the tree implied by every absorbed placement's `sourcePath`.

    For each node: children, placementIds (all under this subtree),
    archetypeCounts, bboxM (world-space), depth. Use this to inspect the
    source's structure and decide which path segments should become zones,
    then call create_zone(attachFromSourcePathPrefix=...) to materialize.

    `prefix` (optional): restrict the result to a subtree rooted at that path.
    `maxDepth` (optional): cap traversal depth.
    """
    return _sess().query_source_hierarchy(prefix, maxDepth)


@mcp.tool()
def create_zone(id: str,
                boundsM: dict | None = None,
                originWorldM: list[float] | None = None,
                purpose: str | None = None,
                allowedArchetypes: list[str] | None = None,
                attachFromSourcePathPrefix: str | None = None,
                attachByArchetypePrefix: str | None = None,
                attachPlacementIds: list[str] | None = None) -> dict:
    """Author a zone in the current sketch and attach placements to it.

    Pick whichever attach mode matches how the source's groupings are
    expressed (multiple modes can be combined; the union is attached):
      - `attachFromSourcePathPrefix` — path-prefix match against each
        placement's `sourcePath` (use for hierarchical sources).
      - `attachByArchetypePrefix` — name-prefix match against each
        placement's `archetype` (use for flat sources that encode groups
        in archetype names, e.g. `Kitchen_*`).
      - `attachPlacementIds` — explicit id list (use when the LLM has
        already clustered placements by spatial proximity, semantic
        affordance, or any other criterion).

    `boundsM` / `originWorldM` default to the bbox computed from the
    attached placements. Returns the zone entry + `attachedCount`."""
    return _sess().create_zone(id, boundsM, originWorldM, purpose,
                                allowedArchetypes,
                                attachFromSourcePathPrefix,
                                attachByArchetypePrefix,
                                attachPlacementIds)


@mcp.tool()
def query_pack_archetypes() -> list[dict]:
    """List archetypes available in the asset pack with their candidate count.

    Use to know what archetypes you can `place(...)` with."""
    return _sess().query_pack_archetypes()


@mcp.tool()
def place(archetype: str, posM: list[float], slotM: list[float],
          yawDeg: float = 0.0, id: str | None = None,
          assetPath: str | None = None,
          scaleM: list[float] | None = None,
          parentZoneId: str | None = None) -> dict:
    """Place an asset of `archetype` at world (x, y, z) posM with bounding
    slot of (widthM, depthM, heightM), rotated yawDeg about the Z axis.

    Optional:
      assetPath — absolute path to a specific asset USD; bypasses best-fit and
        uses this exact asset.
      scaleM — per-axis [sx, sy, sz] scale applied at the reference. Use to
        fit an asset whose bbox doesn't match the slot. Default (no scale) is
        equivalent to [1, 1, 1].
      parentZoneId — semantic zone id this placement belongs to. Use the id
        from query_zones() when editing an imported DBT layout.

    Rejected if the slot collides with an existing placement (collision gate).
    On success returns {ok:true, id}. On rejection returns {ok:false,
    rejected:{reason, colliders}}."""
    return _sess().place(archetype, posM, slotM, yawDeg, id, "reject", assetPath, scaleM, parentZoneId)


@mcp.tool()
def query_pack_assets(archetype: str | None = None) -> dict:
    """Return the full per-archetype candidate list. Each entry has
    {archetype, path, relPath}. Use this to pick a specific asset for
    `place(..., assetPath=...)` instead of relying on best-fit. Combine
    with `query_archetype_semantics` to reason about which asset belongs
    where."""
    return _sess().query_pack_assets(archetype)


@mcp.tool()
def query_template_placements(archetype: str | None = None,
                              filledOnly: bool = False,
                              unfilledOnly: bool = False) -> list[dict]:
    """Return ONLY the placements from the anchor template (i.e. the
    layout's intended positions), not free additions. Each entry includes
    `isFilled` (whether the LLM has bound an explicit asset to it) so you
    can iterate the unfilled ones. Use as the FIRST step of the placement
    workflow: enumerate template positions, decide which asset to use for
    each, then call `update_placement` to bind."""
    return _sess().query_template_placements(archetype, filledOnly, unfilledOnly)


@mcp.tool()
def update_placement(id: str, assetPath: str | None = None,
                     scaleM: list[float] | None = None,
                     yawDeg: float | None = None) -> dict:
    """Modify an existing placement's chosen asset / scale / yaw without
    removing+re-placing it. Use to bind specific assets to template
    placements after picking them via `query_pack_assets`."""
    return _sess().update_placement(id, assetPath, scaleM, yawDeg)


@mcp.tool()
def bind_archetype(archetype: str, assetPath: str,
                   scaleM: list[float] | None = None,
                   fitMode: str = "explicit") -> dict:
    """Bind every placement of `archetype` to one specific asset in a
    single call. This is the recommended Mode A entry point — pick one
    asset per archetype after holistic reasoning, then call this once
    per archetype instead of 153 individual update_placement calls.

    scaleM is in WORLD axes (slot widthM/depthM/heightM factors). The
    realize step permutes for Y-up assets automatically.

    fitMode:
      - "explicit"    : use scaleM verbatim (None = no scale)
      - "fit-to-slot" : engine measures the asset's world bbox once and
                        sets per-placement scaleM = slotM / asset_world_bbox
                        so each placement matches its slot exactly. Use
                        this when slot dims vary across placements of the
                        same archetype."""
    return _sess().bind_archetype(archetype, assetPath, scaleM, fitMode)


@mcp.tool()
def remove(id: str) -> dict:
    """Remove a placement by id."""
    return _sess().remove(id)


@mcp.tool()
def place_many(placements: list[dict], onCollision: str = "reject") -> dict:
    """Batch place. `placements` is a list of dicts each with at least
    {archetype, posM, slotM, yawDeg?, id?, parentZoneId?}. Each goes through
    the collision gate. Use parentZoneId from query_zones() to preserve DBT
    semantic zones."""
    return _sess().place_many(placements, onCollision)


@mcp.tool()
def list_recent_events(n: int = 20) -> list[dict]:
    """Last N session events (placed / rejected / removed) for review."""
    return _sess().list_recent_events(n)


@mcp.tool()
def query_archetype_semantics(archetype: str | None = None) -> dict:
    """Return per-archetype semantics: description (what the object IS),
    typicalContext (where it usually lives), preferredNear (archetypes it
    co-locates well with), avoidNear (archetypes to keep distance from),
    footprintCueM (typical bbox).

    Pass `archetype` to get one entry; omit to get the full map. Use this
    BEFORE planning placements so the layout you propose is contextually
    reasonable."""
    return _sess().query_archetype_semantics(archetype)


@mcp.tool()
def realize_passthrough(targetComposedPrimCount: int = 1000,
                        absorbedTemplate: str | None = None) -> dict:
    """Realize via passthrough_rootcell mode: tile the pack's rootStage USD
    to hit the target prim count. Use when the pack has a pack.json with
    `rootStage` set. The session's placements are bypassed entirely; output
    is N references to rootStage in a grid. `absorbedTemplate` (path)
    optionally enables per-placement viz events for the cell's interior
    layout."""
    return _sess().realize_passthrough(targetComposedPrimCount, absorbedTemplate)


@mcp.tool()
def query_surfaces(regionId: str | None = None,
                   minFreeAreaM2: float = 0.0,
                   label: str | None = None) -> list[dict]:
    """List surfaces available for `place_on`. A surface is a flat top of an
    existing placement (e.g. a counter, table top, rack top) that smaller
    placements can rest on.

    Filters:
      - regionId — limit to surfaces whose owner is inside a named region
        (id from `query_zones`)
      - minFreeAreaM2 — only surfaces with at least this much free area
      - label — exact match against the surface's vocabulary label
        (`top`, `counter`, `rack_top`, etc.)

    Each result: {owner, ownerArchetype, label, centerWorldXY, topWorldZ,
    footprintM, freeAreaM2, ownerYawDeg}. Use `owner` + `label` as the
    target of `place_on`."""
    return _sess().query_surfaces(regionId, minFreeAreaM2, label)


@mcp.tool()
def place_on(parentPlacementId: str, surfaceLabel: str,
             localXY: list[float], yawDeg: float,
             archetype: str, archetypeSizeM: list[float],
             regionId: str | None = None,
             placementId: str | None = None) -> dict:
    """Place an `archetype` ON an existing placement's surface (cup on table,
    monitor on desk, server on rack).

    Args:
      parentPlacementId — owner placement id (from `query_stage_graph` or
        `query_surfaces`)
      surfaceLabel — which surface on that owner (e.g. "top", "counter")
      localXY — [x, y] in metres, in the SURFACE'S LOCAL FRAME (origin at
        surface centre, x→right of the owner's facing direction). The
        engine rotates by the owner's yaw to compose to world.
      yawDeg — yaw of the new placement (world frame)
      archetype — the type being placed (e.g. "cup", "monitor")
      archetypeSizeM — [w, d, h] bbox of the new placement
      regionId — optional; if set, `archetype` must be in the region's
        `allowedArchetypes` (catches "food_table in bathroom"-style errors)
      placementId — optional explicit id; default auto-generated

    Returns {ok, placement|rejected}. Rejection reasons:
      lookup (no such surface), region_filter (archetype not allowed in
      region), edge_check (would fall off surface), sibling_collision
      (overlap with another item on same surface)."""
    return _sess().place_on(parentPlacementId, surfaceLabel, localXY,
                              yawDeg, archetype, archetypeSizeM,
                              regionId, placementId)


@mcp.tool()
def session_status() -> dict:
    """Current session counts and footprint."""
    s = _sess()
    items = s.index.all()
    if items:
        xs = [p.posM[0] for p in items]
        ys = [p.posM[1] for p in items]
        footprint = {"minX": min(xs), "maxX": max(xs),
                     "minY": min(ys), "maxY": max(ys)}
    else:
        footprint = None
    return {
        "placementCount": len(items),
        "zoneCount": len(s.zones),
        "archetypes": list(s._archetype_meta.keys()),
        "footprintM": footprint,
        "outDir": str(s.out_dir),
    }


if __name__ == "__main__":
    mcp.run()
