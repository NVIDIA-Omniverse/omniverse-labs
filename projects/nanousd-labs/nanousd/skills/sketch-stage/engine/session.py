# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Incremental placement session — the stateful core.

Owns: a flat sketch (loaded from anchor), a PlacementIndex, an asset pack
manifest, and an events.jsonl writer. Each placement attempt is gated by
collision and recorded as an event.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from spatial_index import IndexedPlacement, PlacementIndex


# Per-process default for the live event stream. Uses the OS temp dir so
# the path resolves correctly on Linux (/tmp), macOS (/var/folders/...),
# and Windows. Override per-session with the SKETCH_LIVE_EVENTS env var.
_DEFAULT_LIVE_EVENTS = str(Path(tempfile.gettempdir()) / "sketch_live" / "events.jsonl")


# Archetype semantics are pack-driven: every archetype's preferredNear /
# avoidNear / anchors / surfaces / affordances live in pack.json
# (populated by harvest_pack_semantics + LLM gap-fill, see SKILL.md §1.7).
# The skill is pack-agnostic; domain knowledge belongs to the pack, not
# the engine.
#
# Sketches can also carry per-archetype overrides at the top level
# (sketch["semantics"]), which take precedence over pack.semantics.


def _semantics_for(archetype: str,
                    pack_sem: dict | None = None,
                    template_overrides: dict | None = None) -> dict:
    """Look up semantics for an archetype. Order of precedence:
      1. template_overrides[archetype]  (from sketch["semantics"])
      2. pack_sem[archetype]            (from pack.json's
         archetypes[...].semantics, flattened by the caller)

    Returns a dict with at minimum {description, preferredNear,
    avoidNear} so legacy callers don't crash on missing keys. Returns a
    "no semantics declared" placeholder when neither source has data —
    the LLM uses that as a cue to populate pack.semantics.
    """
    template_overrides = template_overrides or {}
    pack_sem = pack_sem or {}
    base = dict(pack_sem.get(archetype, {})) if pack_sem else {}
    base.update(template_overrides.get(archetype, {}))
    if not base:
        return {
            "description": f"No semantics declared for {archetype!r}. Populate "
                            "pack.json via harvest_pack_semantics + LLM gap-fill.",
            "preferredNear": [],
            "avoidNear": [],
        }
    # Normalize the common shape so callers don't have to defensively
    # check every key. anchors/surfaces/affordances pass through verbatim
    # when present; the others get safe defaults.
    base.setdefault("description", "")
    base.setdefault("preferredNear", [])
    base.setdefault("avoidNear", [])
    return base


def _walk_placements(node: dict, parent_zone: str = "",
                     parent_world: tuple[float, float, float] = (0.0, 0.0, 0.0),
                     out: list | None = None) -> list[dict]:
    """Tree-flatten placements with parentZoneId and world-space position."""
    if out is None:
        out = []
    kind = node.get("type")
    if kind == "site":
        for c in node.get("children", []):
            _walk_placements(c, parent_zone, parent_world, out)
    elif kind == "zone":
        zid = node.get("id", "")
        new_parent = f"{parent_zone}/{zid}" if parent_zone else zid
        t = node.get("transform", {}).get("translateM", [0.0, 0.0, 0.0])
        world = (parent_world[0] + float(t[0]),
                 parent_world[1] + float(t[1]),
                 parent_world[2] + float(t[2]))
        for c in node.get("children", []):
            _walk_placements(c, new_parent, world, out)
    elif kind == "placement":
        t = node["transform"]["translateM"]
        out.append({
            "id": node["id"],
            "archetype": node["archetype"],
            "posM": (
                parent_world[0] + float(t[0]),
                parent_world[1] + float(t[1]),
                parent_world[2] + float(t[2]),
            ),
            "slotM": (
                float(node["slotM"]["widthM"]),
                float(node["slotM"]["depthM"]),
                float(node["slotM"]["heightM"]),
            ),
            "yawDeg": float(node["transform"].get("yawDeg", 0.0)),
            "parentZoneId": node.get("parentZoneId") or parent_zone,
            "parentPlacementId": node.get("parentPlacementId"),
            "assetPath": node.get("assetPath"),
            "scaleM": node.get("scaleM"),
            "sourcePath": node.get("sourcePath"),
        })
    return out


def _walk_zones(node: dict, parent_zone: str = "",
                parent_world: tuple[float, float, float] = (0.0, 0.0, 0.0),
                out: list | None = None) -> list[dict]:
    if out is None:
        out = []
    kind = node.get("type")
    if kind in ("zone", "site"):
        zid = node.get("id", "")
        new_parent = f"{parent_zone}/{zid}" if parent_zone and kind == "zone" else (zid if kind == "zone" else parent_zone)
        if kind == "zone":
            t = node.get("transform", {}).get("translateM", [0.0, 0.0, 0.0])
            world = (parent_world[0] + float(t[0]),
                     parent_world[1] + float(t[1]),
                     parent_world[2] + float(t[2]))
        else:
            world = parent_world
        if kind == "zone" and "boundsM" in node:
            out.append({
                "id": new_parent,
                "boundsM": node["boundsM"],
                "originWorldM": list(world),
                "type": node.get("zoneType", "default"),
            })
        for c in node.get("children", []):
            _walk_zones(c, new_parent, world, out)
    return out


def _walk_shells(node: dict, parent_zone: str = "",
                 parent_world: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 out: list | None = None) -> list[dict]:
    if out is None:
        out = []
    kind = node.get("type")
    if kind in ("zone", "site"):
        zid = node.get("id", "")
        new_parent = f"{parent_zone}/{zid}" if parent_zone and kind == "zone" else (zid if kind == "zone" else parent_zone)
        if kind == "zone":
            t = node.get("transform", {}).get("translateM", [0.0, 0.0, 0.0])
            world = (parent_world[0] + float(t[0]),
                     parent_world[1] + float(t[1]),
                     parent_world[2] + float(t[2]))
        else:
            world = parent_world
        shell = node.get("shell")
        if shell and "boundsM" in shell:
            out.append({
                "ownerId": new_parent or zid,
                "boundsM": shell["boundsM"],
                "originWorldM": list(world),
            })
        for c in node.get("children", []):
            _walk_shells(c, new_parent, world, out)
    return out


class Session:
    def __init__(self, anchor_sketch_path: Path | None, asset_pack_dir: Path | None,
                 out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.out_dir / "events.jsonl"
        self._t0 = time.monotonic()
        self.events_path.write_text("")  # start fresh
        # Truncate the shared live-viz log too so the viz clears between runs.
        live_path = os.environ.get("SKETCH_LIVE_EVENTS", _DEFAULT_LIVE_EVENTS)
        try:
            os.makedirs(os.path.dirname(live_path), exist_ok=True)
            open(live_path, "w").close()
        except OSError:
            pass
        self.index = PlacementIndex()
        self.anchor_path = Path(anchor_sketch_path) if anchor_sketch_path else None
        self.asset_pack_dir = Path(asset_pack_dir) if asset_pack_dir else None
        self.zones: list[dict] = []
        self.shells: list[dict] = []
        self._archetype_meta: dict[str, dict] = {}
        # pack.json-driven per-archetype semantics (preferredNear/avoidNear/
        # anchors/surfaces/affordances/placementBias). Populated by
        # _load_pack; replaces the legacy hardcoded ARCHETYPE_SEMANTICS
        # dict that used to live in this module.
        self._pack_sem_by_arch: dict[str, dict] = {}
        self.template_semantics: dict[str, dict] = {}
        self.anchor_placement_count: int = 0  # template placements at session_open
        if self.anchor_path and self.anchor_path.exists():
            self._load_anchor(self.anchor_path)
        self.anchor_placement_count = len(self.index)
        self._anchor_ids: set[str] = {p.id for p in self.index.all()}
        if self.asset_pack_dir and (self.asset_pack_dir / "pack.json").exists():
            self._load_pack(self.asset_pack_dir)
        # Runtime surface registry — built from pack archetype semantics
        # (anchors/surfaces/affordances, see Stage 2). Empty if pack lacks
        # `semantics` entries; query_surfaces will just return nothing.
        self.surface_registry = None
        try:
            from surface_registry import SurfaceRegistry
            if self.asset_pack_dir:
                self.surface_registry = (
                    SurfaceRegistry.from_pack_and_placements(
                        self.asset_pack_dir, self.index.all()))
            else:
                self.surface_registry = SurfaceRegistry()
        except Exception as e:
            print(f"[session] surface_registry init failed: {e}", flush=True)
        self._emit("session_open", {
            "anchorSketch": str(self.anchor_path) if self.anchor_path else None,
            "assetPack": str(self.asset_pack_dir) if self.asset_pack_dir else None,
            "anchorPlacementCount": len(self.index),
            "zoneCount": len(self.zones),
            "archetypes": list(self._archetype_meta.keys()),
            "surfaceCount": (len(self.surface_registry)
                             if self.surface_registry else 0),
        })

    def _load_anchor(self, path: Path) -> None:
        sketch = json.loads(path.read_text())
        # Strict no-overlap dedup at anchor load. Any placement whose
        # OBB intersects an already-loaded placement by more than the
        # threshold below is dropped. Compound assemblies (wheel-on-
        # axle, box-on-shelf) in absorbed sources were preserved by
        # earlier permissive thresholds but produced visible "stacked
        # cube" artifacts when bound to fresh assets. Strict drop here
        # is the safer default for absorbed scenes; runtime `place_on`
        # is the right tool for genuine on-surface stacking afterward.
        # The 5% floor only spares paper-thin edge-touch cases (walls
        # at corners, decorative co-location).
        SAME_OVERLAP_DROP = 0.05
        CROSS_OVERLAP_DROP = 0.05

        def _xy_overlap_frac(p_pos, p_slot, other) -> float:
            # Axis-aligned proxy is fine for the dedup decision; OBB
            # SAT-confirmed candidates from query_collision are the
            # only ones we score against.
            ax0, ay0 = p_pos[0] - p_slot[0]/2, p_pos[1] - p_slot[1]/2
            ax1, ay1 = p_pos[0] + p_slot[0]/2, p_pos[1] + p_slot[1]/2
            bx0, by0 = other.posM[0] - other.slotM[0]/2, other.posM[1] - other.slotM[1]/2
            bx1, by1 = other.posM[0] + other.slotM[0]/2, other.posM[1] + other.slotM[1]/2
            ox = max(0.0, min(ax1, bx1) - max(ax0, bx0))
            oy = max(0.0, min(ay1, by1) - max(ay0, by0))
            overlap = ox * oy
            a_area = p_slot[0] * p_slot[1]
            b_area = other.slotM[0] * other.slotM[1]
            smaller = min(a_area, b_area)
            return (overlap / smaller) if smaller > 0 else 0.0

        # Two-pass load:
        #   1. Dedup non-parented placements first; record which ids were dropped.
        #   2. Insert children only if their parentPlacementId survived.
        # That way a child-on-parent (a box on a shelf, a stacked rack)
        # never appears without its support: if the support is dropped,
        # the child goes with it.
        all_placements = list(_walk_placements(sketch.get("tree", {})))
        non_parented = [ph for ph in all_placements if not ph.get("parentPlacementId")]
        parented = [ph for ph in all_placements if ph.get("parentPlacementId")]

        loaded = 0
        skipped = 0
        kept_ids: set[str] = set()
        for ph in non_parented:
            cols = self.index.query_collision(ph["posM"], ph["slotM"], ph["yawDeg"])
            drop = False
            for c in cols or []:
                other = c[0]
                frac = _xy_overlap_frac(ph["posM"], ph["slotM"], other)
                threshold = SAME_OVERLAP_DROP if other.archetype == ph["archetype"] else CROSS_OVERLAP_DROP
                if frac > threshold:
                    drop = True
                    break
            if drop:
                skipped += 1
                continue
            kept_ids.add(ph["id"])
            scale_in = ph.get("scaleM")
            scale_t = tuple(float(v) for v in scale_in) if scale_in else None
            self.index.insert(IndexedPlacement(
                id=ph["id"], archetype=ph["archetype"], posM=ph["posM"],
                slotM=ph["slotM"], yawDeg=ph["yawDeg"],
                # Resume support: a snapshot from a prior realize() carries
                # full bindings (assetPath/scaleM); without these, resume
                # would drop everything and re-anchor as plain template.
                assetPath=ph.get("assetPath"),
                scaleM=scale_t,
                parentZoneId=ph.get("parentZoneId"),
                sourcePath=ph.get("sourcePath"),
            ))
            loaded += 1

        # Pass 2: parented children — insert iff the parent survived.
        orphaned = 0
        for ph in parented:
            if ph["parentPlacementId"] not in kept_ids:
                orphaned += 1
                continue
            scale_in = ph.get("scaleM")
            scale_t = tuple(float(v) for v in scale_in) if scale_in else None
            self.index.insert(IndexedPlacement(
                id=ph["id"], archetype=ph["archetype"], posM=ph["posM"],
                slotM=ph["slotM"], yawDeg=ph["yawDeg"],
                assetPath=ph.get("assetPath"),
                scaleM=scale_t,
                parentZoneId=ph.get("parentZoneId"),
                sourcePath=ph.get("sourcePath"),
            ))
            kept_ids.add(ph["id"])  # grandchildren can chain
            loaded += 1
        if skipped or orphaned:
            print(f"[session] anchor dedup: kept {loaded}, skipped {skipped} overlapping, "
                  f"dropped {orphaned} children whose parent was culled",
                  flush=True)
        self.zones = _walk_zones(sketch.get("tree", {}))
        self.shells = _walk_shells(sketch.get("tree", {}))
        # Template-level semantics overrides — absorbed templates may bake
        # in pack-specific descriptions / context rules at the top of the
        # sketch.json. These layer on top of the hardcoded defaults.
        self.template_semantics = dict(sketch.get("semantics", {}))
        # Template-level zones with purpose + allowedArchetypes (from absorb)
        self.template_zones = list(sketch.get("templateZones", []))
        self.anchor_sketch_data = sketch  # keep for realize

    def _load_pack(self, pack_dir: Path) -> None:
        # Use pack_loader so manifest packs (multi-theme) are flattened
        # the same way the surface registry sees them.
        try:
            from pack_loader import load_pack  # type: ignore
            pack = load_pack(pack_dir)
            archetypes = pack["archetypes"]
        except Exception:
            pack = json.loads((pack_dir / "pack.json").read_text())
            archetypes = pack.get("archetypes", {})
        for arche, entries in archetypes.items():
            self._archetype_meta[arche] = {
                "candidateCount": len(entries),
                "samplePath": entries[0]["path"] if entries and isinstance(entries[0], dict) else None,
            }
            # Cache per-archetype semantics for query_archetype_semantics.
            # First entry wins (entries usually share the same semantics
            # block per archetype, since the LLM gap-fill writes one block
            # per archetype name; measure_surfaces refreshes per-entry
            # surfaces but keeps preferredNear/avoidNear/anchors uniform).
            sem = entries[0].get("semantics") if entries and isinstance(entries[0], dict) else None
            if sem:
                # Store under both the bare name (<theme>.<arch> → <arch>)
                # and the namespaced form, so callers using either resolve.
                self._pack_sem_by_arch[arche] = sem
                bare = arche.split(".", 1)[1] if "." in arche else arche
                self._pack_sem_by_arch.setdefault(bare, sem)

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        rec = {"t": round(time.monotonic() - self._t0, 4), "type": event_type, **payload}
        line = json.dumps(rec) + "\n"
        with self.events_path.open("a") as f:
            f.write(line)
            f.flush()
        # Also mirror to the shared live-viz log so the browser viz updates
        # in real time (session events were previously only written to the
        # per-run events.jsonl, which the viz doesn't tail).
        live_path = os.environ.get("SKETCH_LIVE_EVENTS", _DEFAULT_LIVE_EVENTS)
        try:
            os.makedirs(os.path.dirname(live_path), exist_ok=True)
            with open(live_path, "a") as f:
                f.write(line)
                f.flush()
        except OSError:
            pass  # don't fail placements over a viz log glitch

    # ---- query tools ----

    def query_stage_graph(self, archetype: str | None = None,
                          parent_zone: str | None = None) -> list[dict]:
        """Returns every placement with the fields needed to author tile
        copies: id, archetype, posM, slotM, yawDeg, AND the binding
        (assetPath, scaleM) when present, plus parentZoneId. Without
        assetPath/scaleM here the LLM can't forward bindings to place_many,
        so tile copies would all fall back to best-fit and fail when the
        pack's archetype names don't match the source layout's."""
        out = []
        for p in self.index.all():
            if archetype and p.archetype != archetype:
                continue
            entry = {
                "id": p.id, "archetype": p.archetype,
                "posM": list(p.posM), "slotM": list(p.slotM), "yawDeg": p.yawDeg,
                "semantics": _semantics_for(p.archetype, self.template_semantics),
            }
            if p.assetPath:
                entry["assetPath"] = p.assetPath
            if p.scaleM:
                entry["scaleM"] = list(p.scaleM)
            if p.parentZoneId:
                entry["parentZoneId"] = p.parentZoneId
            if p.sourcePath:
                entry["sourcePath"] = p.sourcePath
            out.append(entry)
        return out

    def query_nearby(self, posM: list[float], radiusM: float,
                     archetype: str | None = None, limit: int = 50) -> list[dict]:
        items = self.index.query_nearby((float(posM[0]), float(posM[1])),
                                        float(radiusM), archetype, int(limit))
        return [{
            "id": p.id, "archetype": p.archetype,
            "posM": list(p.posM), "slotM": list(p.slotM), "yawDeg": p.yawDeg,
            "distanceM": round(math.hypot(p.posM[0] - float(posM[0]),
                                          p.posM[1] - float(posM[1])), 4),
            "semantics": _semantics_for(p.archetype),
        } for p in items]

    def query_archetype_semantics(self, archetype: str | None = None) -> dict:
        """Return the full semantics map (or just one archetype's).

        Sources, in precedence order:
          1. Sketch top-level `semantics` (template overrides — usually
             absorbed-template authors describing pack-specific rules
             baked into the sketch at absorb time).
          2. `pack.json` per-archetype `semantics` block (populated by
             `harvest_pack_semantics.py` + LLM gap-fill, refined by
             `measure_surfaces.py`).

        Returns a placeholder (`description: 'No semantics declared'`)
        when neither source has data, signalling that the LLM should
        run the gap-fill workflow.
        """
        pack_sem = getattr(self, "_pack_sem_by_arch", {}) or {}
        if archetype:
            return {archetype: _semantics_for(archetype, pack_sem,
                                                self.template_semantics)}
        keys = set(pack_sem) | set(self.template_semantics or {}) | set(self._archetype_meta)
        return {a: _semantics_for(a, pack_sem, self.template_semantics)
                for a in keys}

    def query_collision(self, posM: list[float], slotM: list[float],
                        yawDeg: float = 0.0) -> dict:
        cols = self.index.query_collision(
            (float(posM[0]), float(posM[1]), float(posM[2]) if len(posM) > 2 else 0.0),
            (float(slotM[0]), float(slotM[1]), float(slotM[2])),
            float(yawDeg),
        )
        return {
            "ok": len(cols) == 0,
            "colliders": [{"id": p.id, "archetype": p.archetype,
                           "overlapM2": round(area, 4)} for p, area in cols],
        }

    def query_zones(self) -> list[dict]:
        # Merge spatial zones (geometry) with template zones (purpose +
        # allowedArchetypes captured at absorb time). Template zones are
        # named (e.g., "Loading_Zone"), spatial zones come from the tree.
        merged = []
        for z in self.zones:
            entry = dict(z)
            for tz in self.template_zones:
                if tz.get("id") and (tz["id"] == entry["id"] or entry["id"].endswith(f"/{tz['id']}")):
                    entry.update({k: v for k, v in tz.items() if k != "id"})
                    break
            merged.append(entry)
        # Also surface any template zones that don't have spatial counterparts
        for tz in self.template_zones:
            if not any(tz.get("id") and (tz["id"] == m["id"] or m["id"].endswith(f"/{tz['id']}")) for m in merged):
                merged.append({"id": tz["id"], **{k: v for k, v in tz.items() if k != "id"}})
        return merged

    def query_source_hierarchy(self, prefix: str | None = None,
                               maxDepth: int | None = None) -> dict:
        """Return the tree implied by every placement's `sourcePath`.

        For each node in the implied hierarchy, includes:
          - `children`: list of immediate child path segments
          - `placementIds`: ids of placements at or under this node
          - `archetypeCounts`: how many of each archetype live under here
          - `bboxM`: world-space {minX, maxX, minY, maxY, minZ, maxZ} of
                    all placements under this node
          - `depth`: depth from root (0 = top-level prim under default)

        The LLM uses this to decide which path segments make sense as
        zones. Filter with `prefix` (only show subtree rooted there) or
        `maxDepth` (cap tree depth).
        """
        # Collect every placement's source path; skip ones without it.
        nodes: dict[str, dict] = {}
        for p in self.index.all():
            if not p.sourcePath:
                continue
            segs = [s for s in p.sourcePath.split("/") if s]
            # Walk every ancestor path and accumulate placement membership.
            acc = ""
            for depth, seg in enumerate(segs):
                acc = (acc + "/" + seg) if acc else "/" + seg
                if prefix and not acc.startswith(prefix):
                    continue
                if maxDepth is not None and depth >= maxDepth:
                    break
                node = nodes.setdefault(acc, {
                    "path": acc, "depth": depth,
                    "children": set(), "placementIds": [],
                    "archetypeCounts": {},
                    "bboxM": None,
                })
                node["placementIds"].append(p.id)
                node["archetypeCounts"][p.archetype] = (
                    node["archetypeCounts"].get(p.archetype, 0) + 1)
                # Accumulate bbox of all placements under this node
                x0, y0, z0 = p.posM
                w, d, h = p.slotM
                mn = (x0 - w / 2, y0 - d / 2, z0)
                mx = (x0 + w / 2, y0 + d / 2, z0 + h)
                bb = node["bboxM"]
                if bb is None:
                    node["bboxM"] = {
                        "minX": mn[0], "maxX": mx[0],
                        "minY": mn[1], "maxY": mx[1],
                        "minZ": mn[2], "maxZ": mx[2],
                    }
                else:
                    bb["minX"] = min(bb["minX"], mn[0])
                    bb["minY"] = min(bb["minY"], mn[1])
                    bb["minZ"] = min(bb["minZ"], mn[2])
                    bb["maxX"] = max(bb["maxX"], mx[0])
                    bb["maxY"] = max(bb["maxY"], mx[1])
                    bb["maxZ"] = max(bb["maxZ"], mx[2])
                # Link this node as a child of its parent (if there is one).
                if depth > 0:
                    parent = "/" + "/".join(segs[:depth])
                    if parent in nodes:
                        nodes[parent]["children"].add(seg)
        # Materialize: convert child sets to sorted lists, sort the tree.
        out = {}
        for path, node in nodes.items():
            out[path] = {
                **node,
                "children": sorted(node["children"]),
                # placementIds may grow large; sort but don't truncate
                "placementIds": sorted(node["placementIds"]),
            }
        return out

    def create_zone(self, id: str,
                    boundsM: dict | None = None,
                    originWorldM: list[float] | None = None,
                    purpose: str | None = None,
                    allowedArchetypes: list[str] | None = None,
                    attachFromSourcePathPrefix: str | None = None,
                    attachByArchetypePrefix: str | None = None,
                    attachPlacementIds: list[str] | None = None) -> dict:
        """Add a new zone to the session and (optionally) attach placements
        to it. Use whichever attach mode matches how the source's grouping
        is expressed:

          - `attachFromSourcePathPrefix` — path-prefix match against each
            placement's `sourcePath`. Use when the source uses hierarchy
            (e.g. `/World/Kitchen`).
          - `attachByArchetypePrefix` — name-prefix match against each
            placement's `archetype`. Use when the source is flat and groups
            are encoded in names (e.g. `Kitchen_*` → all kitchen placements).
          - `attachPlacementIds` — explicit id list. Use when the LLM has
            already clustered placements by some other criterion (e.g.
            spatial proximity, semantic affordance) and wants to materialize
            that grouping as a zone.

        Multiple modes can be combined; the union of matches is attached.

        `boundsM` / `originWorldM` default to the bbox computed from the
        attached placements when omitted. Returns the zone entry plus
        `attachedCount`.
        """
        attached: list = []
        seen_ids: set[str] = set()

        def _claim(p):
            if p.id in seen_ids:
                return
            seen_ids.add(p.id)
            p.parentZoneId = id
            attached.append(p)

        prefix = (attachFromSourcePathPrefix or "").rstrip("/")
        arch_prefix = attachByArchetypePrefix or ""
        target_ids = set(attachPlacementIds or [])
        for p in self.index.all():
            if prefix and p.sourcePath and (
                    p.sourcePath == prefix
                    or p.sourcePath.startswith(prefix + "/")):
                _claim(p); continue
            if arch_prefix and p.archetype.startswith(arch_prefix):
                _claim(p); continue
            if p.id in target_ids:
                _claim(p); continue
        # Compute bbox from attached placements if caller didn't supply one.
        if boundsM is None or originWorldM is None:
            ref = attached if attached else list(self.index.all())
            if ref:
                xs = [p.posM[0] for p in ref]
                ys = [p.posM[1] for p in ref]
                zs = [p.posM[2] for p in ref]
                ws = [p.slotM[0] for p in ref]
                ds = [p.slotM[1] for p in ref]
                hs = [p.slotM[2] for p in ref]
                min_x = min(xs[i] - ws[i] / 2 for i in range(len(ref)))
                max_x = max(xs[i] + ws[i] / 2 for i in range(len(ref)))
                min_y = min(ys[i] - ds[i] / 2 for i in range(len(ref)))
                max_y = max(ys[i] + ds[i] / 2 for i in range(len(ref)))
                min_z = min(zs)
                max_z = max(zs[i] + hs[i] for i in range(len(ref)))
                if originWorldM is None:
                    originWorldM = [min_x, min_y, min_z]
                if boundsM is None:
                    boundsM = {
                        "widthM": max(max_x - min_x, 0.1),
                        "depthM": max(max_y - min_y, 0.1),
                        "heightM": max(max_z - min_z, 0.1),
                    }
        zone = {
            "id": id,
            "boundsM": boundsM or {"widthM": 1.0, "depthM": 1.0, "heightM": 1.0},
            "originWorldM": originWorldM or [0.0, 0.0, 0.0],
            "purpose": purpose or id,
            "allowedArchetypes": list(allowedArchetypes) if allowedArchetypes
                                  else sorted({p.archetype for p in attached}),
            "sourcePathPrefix": attachFromSourcePathPrefix,
            "attachedCount": len(attached),
        }
        # Drop any prior entry with the same id (LLM can iteratively refine).
        self.zones = [z for z in self.zones if z.get("id") != id]
        self.zones.append(zone)
        self.template_zones = [z for z in self.template_zones if z.get("id") != id]
        self.template_zones.append(zone)
        self._emit("zone_created", {"id": id, "zone": zone,
                                    "attachedPlacements": len(attached)})
        return zone

    # --- Stage 4: on-surface placement ----------------------------------

    def _region_by_id(self, region_id: str) -> Optional[dict]:
        for z in (self.template_zones or []):
            if z.get("id") == region_id:
                return z
        for z in self.zones or []:
            if z.get("id") == region_id:
                return z
        return None

    def query_surfaces(self, regionId: str | None = None,
                       minFreeAreaM2: float = 0.0,
                       label: str | None = None) -> list[dict]:
        """Return surfaces matching the filters.

        - `regionId` (optional): restrict to surfaces whose owner's centre
          lies inside the named region's `footprintM`. Region id comes from
          `query_zones()` (template zones merged with spatial zones).
        - `minFreeAreaM2` (default 0): only surfaces with at least this
          much unoccupied area.
        - `label` (optional): exact surface label match (e.g., "top",
          "counter", "rack_top").
        """
        if not self.surface_registry:
            return []
        region_fp = None
        if regionId:
            region = self._region_by_id(regionId)
            if region and region.get("footprintM"):
                fp = region["footprintM"]
                if (isinstance(fp, list) and len(fp) == 2
                        and isinstance(fp[0], list) and isinstance(fp[1], list)):
                    region_fp = ((float(fp[0][0]), float(fp[0][1])),
                                  (float(fp[1][0]), float(fp[1][1])))
        return self.surface_registry.query(region_fp=region_fp,
                                            min_free_area_m2=minFreeAreaM2,
                                            label=label)

    def place_on(self, parentPlacementId: str, surfaceLabel: str,
                 localXY: list[float], yawDeg: float,
                 archetype: str, archetypeSizeM: list[float],
                 regionId: str | None = None,
                 placementId: str | None = None,
                 surfaceIndex: int | None = None,
                 topWorldZ: float | None = None) -> dict:
        """Place an `archetype` ON an existing placement's surface.

        Parent-child semantics: the new placement is recorded in the
        surface's occupant list AND inserted into the spatial index at
        the resolved world XYZ. Sibling collisions (other things on the
        same surface) are checked here; the regular collision gate is
        bypassed for on-surface placements (otherwise the parent's slot
        would always reject — that was the original cup-on-table bug).

        Multi-surface owners: if `parentPlacementId` has more than one
        surface with `surfaceLabel` (e.g. a rack with three shelves at
        different z), pass `surfaceIndex` (0-based, matches `labelIndex`
        in `query_surfaces`) or `topWorldZ` to pick one. Without either,
        a multi-match is rejected with stage="ambiguous_surface".

        Returns `{ok: bool, placement|rejected}`.
        """
        if not self.surface_registry:
            return {"ok": False, "rejected": {
                "reason": "surface_registry not initialised "
                          "(no pack semantics?)",
                "stage": "setup"}}
        region = self._region_by_id(regionId) if regionId else None
        allowed = (region or {}).get("allowedArchetypes")
        region_name = (region or {}).get("name") or regionId
        result = self.surface_registry.place_on(
            parent_id=parentPlacementId,
            surface_label=surfaceLabel,
            local_xy=(float(localXY[0]), float(localXY[1])),
            yaw_deg=float(yawDeg),
            archetype=archetype,
            archetype_sizeM=(float(archetypeSizeM[0]),
                              float(archetypeSizeM[1]),
                              float(archetypeSizeM[2])),
            region_allowed_archetypes=allowed,
            region_name=region_name,
            placement_id=placementId,
            surface_index=surfaceIndex,
            top_world_z=topWorldZ,
        )
        if not result.get("ok"):
            self._emit("place_on_rejected", {
                "parent": parentPlacementId, "surface": surfaceLabel,
                "archetype": archetype, **result.get("rejected", {})})
            return result
        pl = result["placement"]
        # Insert into the spatial index using the resolved world XYZ; sibling
        # collisions on the surface were already checked above. Other
        # placements in the index that happen to overlap (e.g. the parent
        # slot itself) are NOT checked — that's the deliberate behaviour.
        try:
            self.index.insert(IndexedPlacement(
                id=pl["id"], archetype=pl["archetype"],
                posM=tuple(pl["posM"]),
                slotM=(pl["slotM"]["widthM"], pl["slotM"]["depthM"],
                       pl["slotM"]["heightM"]),
                yawDeg=float(pl["yawDeg"]),
                assetPath=None, scaleM=None,
                parentZoneId=regionId,
            ))
        except Exception as e:
            print(f"[session] place_on: index insert failed: {e}", flush=True)
        # Normalize slotM to list form [w, d, h] so the event schema matches
        # what `place()` emits. surface_registry returns slotM as a dict with
        # widthM/depthM/heightM keys; the viz destructures slotM as a list.
        sm = pl["slotM"]
        if isinstance(sm, dict):
            sm = [sm["widthM"], sm["depthM"], sm["heightM"]]
        self._emit("placed", {
            "id": pl["id"], "archetype": pl["archetype"],
            "posM": list(pl["posM"]), "slotM": list(sm), "yawDeg": pl["yawDeg"],
            "parentPlacement": pl["parentPlacement"],
            "parentSurface": pl["parentSurface"],
        })
        return result

    def query_pack_archetypes(self) -> list[dict]:
        return [{"archetype": a, **m,
                 "semantics": _semantics_for(a, self.template_semantics)}
                for a, m in self._archetype_meta.items()]

    def query_template_placements(self, archetype: str | None = None,
                                   filledOnly: bool = False,
                                   unfilledOnly: bool = False) -> list[dict]:
        """Return ONLY the placements that came from the anchor template
        (not free additions made via place/place_many). Each entry includes
        whether it has an explicit assetPath assigned (=`filled` in the
        LLM-assigns-asset workflow). Use to enumerate what still needs an
        asset chosen for it.

        `filledOnly`/`unfilledOnly` filter by whether the placement has a
        `assetPath` set (i.e. the LLM has bound it to a specific asset)."""
        out = []
        for p in self.index.all():
            if p.id not in self._anchor_ids:
                continue
            if archetype and p.archetype != archetype:
                continue
            has_asset = bool(p.assetPath)
            if filledOnly and not has_asset:
                continue
            if unfilledOnly and has_asset:
                continue
            out.append({
                "id": p.id, "archetype": p.archetype,
                "posM": list(p.posM), "slotM": list(p.slotM), "yawDeg": p.yawDeg,
                "assetPath": p.assetPath,
                "scaleM": list(p.scaleM) if p.scaleM else None,
                "parentZoneId": p.parentZoneId,
                "sourcePath": p.sourcePath,
                "isFilled": has_asset,
                "semantics": _semantics_for(p.archetype, self.template_semantics),
            })
        return out

    def update_placement(self, id: str, assetPath: str | None = None,
                         scaleM: list[float] | None = None,
                         yawDeg: float | None = None) -> dict:
        """Modify an existing placement's chosen asset and/or scale.
        Use this to bind a specific asset to a template placement
        without removing+re-placing it."""
        fields = {}
        if assetPath is not None:
            fields["assetPath"] = assetPath
        if scaleM is not None:
            fields["scaleM"] = tuple(float(v) for v in scaleM)
        if yawDeg is not None:
            fields["yawDeg"] = float(yawDeg)
        ok = self.index.update(id, **fields)
        if ok:
            self._emit("updated", {"id": id, **fields})
        return {"ok": ok, "id": id, "applied": fields}

    def bind_archetype(self, archetype: str, assetPath: str,
                       scaleM: list[float] | None = None,
                       fitMode: str = "explicit") -> dict:
        """Bind every placement of `archetype` to `assetPath` in one call.

        This is the canonical Mode A entry point: the LLM picks one asset
        per archetype (after holistic reasoning over semantics + bboxes),
        then binds the entire group with one tool call instead of 153
        update_placement calls.

        scaleM is in WORLD axes (slot.w, slot.d, slot.h factors). The realize
        step permutes it into asset-local axes if the asset stage is Y-up.

        fitMode:
          - "explicit"    : use the given scaleM verbatim (None ⇒ no scale)
          - "fit-to-slot" : compute scaleM per-placement so the asset's world
                            bbox matches that placement's slotM. assetPath
                            is opened once to read its bbox + upAxis.

        Returns counts of updated placements + the resolved asset bbox.
        """
        resolved_scale = (tuple(float(v) for v in scaleM)
                          if scaleM and fitMode == "explicit" else None)

        asset_bbox_world = None
        if fitMode == "fit-to-slot":
            try:
                from pxr import Usd, UsdGeom
                s = Usd.Stage.Open(assetPath)
                if s is None:
                    return {"ok": False, "error": f"cannot open {assetPath}"}
                up = str(UsdGeom.GetStageUpAxis(s) or "Z").upper()
                mpu = float(UsdGeom.GetStageMetersPerUnit(s) or 1.0)
                cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                          ["default", "render"])
                dp = s.GetDefaultPrim()
                rng = cache.ComputeWorldBound(dp).ComputeAlignedRange()
                sz = rng.GetSize()
                # Convert raw asset bbox to "world after upAxis correction":
                #   world_X = asset_X * mpu
                #   world_Y = (asset_Z if Y-up else asset_Y) * mpu
                #   world_Z = (asset_Y if Y-up else asset_Z) * mpu
                if up == "Y":
                    asset_bbox_world = (sz[0] * mpu, sz[2] * mpu, sz[1] * mpu)
                else:
                    asset_bbox_world = (sz[0] * mpu, sz[1] * mpu, sz[2] * mpu)
            except Exception as e:
                return {"ok": False, "error": f"bbox measurement failed: {e}"}

        updated = 0
        for p in self.index.all():
            if p.archetype != archetype:
                continue
            fields = {"assetPath": assetPath}
            if fitMode == "fit-to-slot" and asset_bbox_world is not None:
                slot = p.slotM
                fields["scaleM"] = (slot[0] / asset_bbox_world[0],
                                    slot[1] / asset_bbox_world[1],
                                    slot[2] / asset_bbox_world[2])
            elif resolved_scale is not None:
                fields["scaleM"] = resolved_scale
            if self.index.update(p.id, **fields):
                updated += 1
                self._emit("updated", {"id": p.id, **fields})
        # Refresh the surface registry so subsequent query_surfaces /
        # surface_fill see the surfaces of the *bound* asset (and the
        # right scaleM-stretched Z values), not whatever best-fit guess
        # add_placement used at insert time. This is the second half of
        # the per-entry surface resolution: pick at insert time, lock at
        # bind time. Without it, boxes land at the entry[0] shelf Z's
        # (the original bug — boxes clipping through racks because the
        # bound asset's shelves were elsewhere).
        rebind_report = None
        if self.surface_registry is not None:
            try:
                rebind_report = self.surface_registry.rebind_archetype(
                    archetype, assetPath,
                    [p for p in self.index.all() if p.archetype == archetype])
            except Exception as e:
                print(f"[session] surface_registry.rebind_archetype failed: {e}",
                      flush=True)
        return {
            "ok": True,
            "archetype": archetype,
            "assetPath": assetPath,
            "fitMode": fitMode,
            "assetBboxWorldM": list(asset_bbox_world) if asset_bbox_world else None,
            "placementsUpdated": updated,
            "surfaceRebind": rebind_report,
        }

    def query_pack_assets(self, archetype: str | None = None) -> dict:
        """Return the full per-archetype candidate list with each asset's
        relative path + measured bbox. Use this to pick an explicit asset
        for `place(..., assetPath=...)` and decide whether scaling is
        needed."""
        if not self.asset_pack_dir:
            return {}
        try:
            pack = json.loads((self.asset_pack_dir / "pack.json").read_text())
        except Exception:
            return {}
        out: dict[str, list[dict]] = {}
        archetypes = pack.get("archetypes", {})
        targets = [archetype] if archetype and archetype in archetypes else list(archetypes.keys())
        for a in targets:
            entries = archetypes.get(a, [])
            cands = []
            for e in entries:
                rel = e["path"] if isinstance(e, dict) else e
                absp = (self.asset_pack_dir / rel).resolve()
                if not absp.exists():
                    continue
                cands.append({
                    "archetype": a,
                    "path": str(absp),
                    "relPath": rel,
                })
            out[a] = cands
        return out

    # ---- mutation tools ----

    def place(self, archetype: str, posM: list[float], slotM: list[float],
              yawDeg: float = 0.0, id: str | None = None,
              onCollision: str = "reject",
              assetPath: str | None = None,
              scaleM: list[float] | None = None,
              parentZoneId: str | None = None) -> dict:
        pid = id or f"{archetype}_{uuid.uuid4().hex[:8]}"
        # Bounds check — when shells or zones are declared, the placement's
        # XY center must lie inside at least one of them. Without this an
        # LLM that assumes the wrong coord convention (e.g. 0-origin vs
        # centered) silently lands placements far outside the canvas; the
        # user then sees floating geometry next to the shell wireframe in
        # the live viz with no diagnostic.
        regions: list[dict] = []
        for sh in self.shells:
            ox, oy = (sh.get("originWorldM") or [0.0, 0.0, 0.0])[:2]
            regions.append({"kind": "shell", "id": sh.get("ownerId") or "site",
                            "originXY": [float(ox), float(oy)],
                            "extentXY": [float(ox + sh["boundsM"]["widthM"]),
                                         float(oy + sh["boundsM"]["depthM"])]})
        for z in self.zones:
            ox, oy = (z.get("originWorldM") or [0.0, 0.0, 0.0])[:2]
            regions.append({"kind": "zone", "id": z.get("id") or "",
                            "originXY": [float(ox), float(oy)],
                            "extentXY": [float(ox + z["boundsM"]["widthM"]),
                                         float(oy + z["boundsM"]["depthM"])]})
        if regions and onCollision in ("reject", "skip"):
            px, py = float(posM[0]), float(posM[1])
            in_any = any(r["originXY"][0] <= px <= r["extentXY"][0]
                         and r["originXY"][1] <= py <= r["extentXY"][1]
                         for r in regions)
            if not in_any:
                self._emit("rejected", {
                    "archetype": archetype, "posM": list(posM),
                    "slotM": list(slotM), "yawDeg": yawDeg,
                    "reason": "out_of_bounds", "regions": regions,
                })
                if onCollision == "skip":
                    return {"ok": False, "skipped": True,
                            "reason": "out_of_bounds", "regions": regions}
                return {"ok": False, "rejected": {"reason": "out_of_bounds",
                                                  "regions": regions}}

        # Stage A: region.allowedArchetypes gate. When the caller declares a
        # parentZoneId and that zone (templateZones[] from absorb's region
        # pipeline, or LLM-labelled) carries an `allowedArchetypes` list,
        # reject placements whose archetype isn't on the list. Catches
        # category mismatches (food_table in bathroom, server_rack in
        # cooling_alley) for floor placements; place_on already does this
        # for on-surface placements.
        if parentZoneId and onCollision in ("reject", "skip"):
            zone = self._region_by_id(parentZoneId)
            allowed = (zone or {}).get("allowedArchetypes")
            if allowed:  # empty list => everything rejected => probably bug; skip the check
                allowed_set = set(allowed)
                if archetype not in allowed_set:
                    payload = {
                        "archetype": archetype, "posM": list(posM),
                        "slotM": list(slotM), "yawDeg": yawDeg,
                        "reason": "region_filter",
                        "parentZoneId": parentZoneId,
                        "regionName": zone.get("name"),
                        "allowed": sorted(allowed_set),
                    }
                    self._emit("rejected", payload)
                    if onCollision == "skip":
                        return {"ok": False, "skipped": True, **{
                                k: v for k, v in payload.items()
                                if k not in {"posM", "slotM", "yawDeg"}}}
                    return {"ok": False, "rejected": {
                        "reason": "region_filter",
                        "parentZoneId": parentZoneId,
                        "regionName": zone.get("name"),
                        "allowed": sorted(allowed_set),
                    }}

        col = self.query_collision(posM, slotM, yawDeg)
        # Both "reject" and "skip" mean "do NOT insert this placement when it
        # collides" — they only differ in caller semantics: "reject" reports
        # the collision via ok=False (so a single place() can surface the
        # error), "skip" silently drops it (so place_many can keep going).
        # Only "force" (or "ignore") inserts despite collision; that's an
        # explicit override the caller must opt into.
        if not col["ok"] and onCollision in ("reject", "skip"):
            self._emit("rejected", {
                "archetype": archetype, "posM": list(posM), "slotM": list(slotM),
                "yawDeg": yawDeg, "reason": "collision", "colliders": col["colliders"],
            })
            if onCollision == "skip":
                return {"ok": False, "skipped": True,
                        "reason": "collision",
                        "colliders": col["colliders"]}
            return {"ok": False, "rejected": {"reason": "collision",
                                              "colliders": col["colliders"]}}
        scale_t = (tuple(float(v) for v in scaleM) if scaleM else None)
        new_placement = IndexedPlacement(
            id=pid, archetype=archetype,
            posM=(float(posM[0]), float(posM[1]),
                  float(posM[2]) if len(posM) > 2 else 0.0),
            slotM=(float(slotM[0]), float(slotM[1]), float(slotM[2])),
            yawDeg=float(yawDeg),
            assetPath=assetPath,
            scaleM=scale_t,
            parentZoneId=parentZoneId,
        )
        self.index.insert(new_placement)
        # Keep the surface registry in sync — if the inserted archetype
        # declares surfaces in pack.semantics, they need to become
        # queryable for subsequent query_surfaces / place_on calls.
        # Without this hook, a session that started empty + then placed a
        # rack via random_fill would report zero surfaces, and you couldn't
        # `place_on` a box on the rack's shelf.
        if self.surface_registry is not None:
            try:
                self.surface_registry.add_placement(new_placement)
            except Exception as e:
                print(f"[session] surface_registry.add_placement failed: {e}",
                      flush=True)
        self._emit("placed", {
            "id": pid, "archetype": archetype, "posM": list(posM),
            "slotM": list(slotM), "yawDeg": yawDeg,
            "assetPath": assetPath, "scaleM": scale_t,
            "parentZoneId": parentZoneId,
        })
        return {"ok": True, "id": pid}

    def remove(self, id: str) -> dict:
        ok = self.index.remove(id)
        if ok:
            if self.surface_registry is not None:
                try:
                    # 1. Drop surfaces THIS placement OWNED (e.g. removing a
                    #    table also drops the cup-on-table parent surface).
                    self.surface_registry.remove_placement(id)
                    # 2. Drop the id from any surface.occupied list it
                    #    appears in (e.g. removing a box from a rack shelf
                    #    must free that slot so future place_on calls don't
                    #    keep colliding with the orphan).
                    self.surface_registry.remove_occupant(id)
                except Exception:
                    pass
            self._emit("removed", {"id": id, "reason": "user-edit"})
        return {"ok": ok}

    def place_many(self, placements: list[dict],
                   onCollision: str = "reject") -> dict:
        """Batch place — runs `place(...)` for each placement in order with
        collision check. Returns counts + per-item results."""
        results = []
        placed = 0
        rejected = 0
        for p in placements:
            r = self.place(
                p["archetype"], p["posM"], p["slotM"],
                p.get("yawDeg", 0.0), p.get("id"), onCollision,
                p.get("assetPath"), p.get("scaleM"), p.get("parentZoneId"),
            )
            results.append(r)
            if r.get("ok"):
                placed += 1
            else:
                rejected += 1
        return {"placed": placed, "rejected": rejected, "total": len(placements),
                "results": results}

    def list_recent_events(self, n: int = 20) -> list[dict]:
        if not self.events_path.exists():
            return []
        lines = self.events_path.read_text().splitlines()[-n:]
        return [json.loads(l) for l in lines if l.strip()]

    def close(self) -> dict:
        self._emit("session_close", {"finalPlacementCount": len(self.index)})
        return {"ok": True, "finalPlacementCount": len(self.index)}

    # ---- snapshot + realize ----

    @staticmethod
    def _sanitize_usd_name(s: str) -> str:
        import re
        out = re.sub(r"[^A-Za-z0-9_]", "_", s)
        if not out or not (out[0].isalpha() or out[0] == "_"):
            out = "p_" + out
        return out

    def snapshot_to_sketch(self, out_path: Path,
                           on_miss_override: str | None = None) -> Path:
        """Write the current in-memory state as a sketch.json compatible with
        sketch_realize.py. `on_miss_override` (skip|synth|abort) overrides the
        anchor's defaults.defaultOnMiss in the snapshot."""
        # Build a hierarchical sketch by reusing the anchor's zone hierarchy.
        # All placements live under a special zone "_session" if no anchor was
        # loaded; otherwise we attach to the deepest zone the anchor exposes.
        if hasattr(self, "anchor_sketch_data"):
            sketch = json.loads(json.dumps(self.anchor_sketch_data))  # deep copy
            # Replace placements under each zone with the current index — keep
            # zones + shells intact, drop original placement children, append
            # current placements grouped by parentZoneId where possible.
            def strip_placements(node):
                if "children" in node:
                    node["children"] = [c for c in node["children"]
                                        if c.get("type") != "placement"]
                    for c in node["children"]:
                        strip_placements(c)
            strip_placements(sketch.get("tree", {}))

            zone_nodes: dict[str, tuple[dict, tuple[float, float, float]]] = {}
            first_zone: tuple[dict, tuple[float, float, float]] | None = None

            def collect_zones(node, parent_zone: str = "",
                              parent_world: tuple[float, float, float] = (0.0, 0.0, 0.0)):
                nonlocal first_zone
                kind = node.get("type")
                world = parent_world
                zone_path = parent_zone
                if kind == "zone":
                    zid = node.get("id", "")
                    zone_path = f"{parent_zone}/{zid}" if parent_zone else zid
                    t = node.get("transform", {}).get("translateM", [0.0, 0.0, 0.0])
                    world = (parent_world[0] + float(t[0]),
                             parent_world[1] + float(t[1]),
                             parent_world[2] + float(t[2]))
                    zone_nodes[zone_path] = (node, world)
                    zone_nodes.setdefault(zid, (node, world))
                    if first_zone is None:
                        first_zone = (node, world)
                for child in node.get("children", []):
                    collect_zones(child, zone_path, world)

            collect_zones(sketch.get("tree", {}))
            fallback_host = first_zone or (sketch.get("tree", {}), (0.0, 0.0, 0.0))

            for p in self.index.all():
                host, origin = zone_nodes.get(p.parentZoneId or "", fallback_host)
                local_pos = [
                    p.posM[0] - origin[0],
                    p.posM[1] - origin[1],
                    p.posM[2] - origin[2],
                ]
                pl = {
                    "type": "placement", "id": self._sanitize_usd_name(p.id), "archetype": p.archetype,
                    "slotM": {"widthM": p.slotM[0], "depthM": p.slotM[1], "heightM": p.slotM[2]},
                    "transform": {"translateM": local_pos, "yawDeg": p.yawDeg},
                }
                if p.parentZoneId:
                    pl["parentZoneId"] = p.parentZoneId
                if p.assetPath:
                    pl["assetPath"] = p.assetPath
                if p.scaleM:
                    pl["scaleM"] = list(p.scaleM)
                if p.sourcePath:
                    pl["sourcePath"] = p.sourcePath
                host.setdefault("children", []).append(pl)
        else:
            sketch = {
                "schemaVersion": 1, "seed": 0,
                "assetPack": str(self.asset_pack_dir or ""),
                "tree": {
                    "type": "site", "id": "session",
                    "children": [{
                        "type": "zone", "id": "scene",
                        "transform": {"translateM": [0,0,0], "yawDeg": 0.0},
                        "boundsM": {"widthM": 100, "depthM": 100, "heightM": 10},
                        "children": [{
                            "type": "placement", "id": self._sanitize_usd_name(p.id), "archetype": p.archetype,
                            "slotM": {"widthM": p.slotM[0], "depthM": p.slotM[1], "heightM": p.slotM[2]},
                            "transform": {"translateM": list(p.posM), "yawDeg": p.yawDeg},
                            **({"parentZoneId": p.parentZoneId} if p.parentZoneId else {}),
                            **({"assetPath": p.assetPath} if p.assetPath else {}),
                            **({"scaleM": list(p.scaleM)} if p.scaleM else {}),
                            **({"sourcePath": p.sourcePath} if p.sourcePath else {}),
                        } for p in self.index.all()],
                    }],
                },
            }
        if on_miss_override:
            sketch.setdefault("defaults", {})["defaultOnMiss"] = on_miss_override
        # Always override assetPack with the session's configured pack.
        # The anchor may have its own embedded assetPack; the session is the
        # source of truth for which pack the realize step should consume.
        if self.asset_pack_dir:
            sketch["assetPack"] = str(self.asset_pack_dir)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(sketch, indent=2))
        return out_path

    def realize(self, on_miss: str | None = None,
                asset_pack_override: str | None = None,
                composition_strategy: dict | None = None) -> dict:
        """Snapshot and run the existing sketch realizer to author root.usd.
        `on_miss` overrides defaultOnMiss in the snapshot (skip|synth|abort).
        `asset_pack_override` switches the pack used by the realizer.
        `composition_strategy` is a dict matching sketch_realize.py's
            compositionStrategy block — keys: composition (reference|payload|mixed),
            payloadSizeThresholdBytes, instancing (none|protoPerAsset),
            hierarchy (as-sketch|flat). Stored on the snapshot.
        Returns paths and counts."""
        import os, subprocess
        snapshot = self.out_dir / "_snapshot.sketch.json"
        on_miss = on_miss or os.environ.get("SKETCH_DEFAULT_ON_MISS")
        self.snapshot_to_sketch(snapshot, on_miss_override=on_miss)
        if asset_pack_override or composition_strategy:
            data = json.loads(snapshot.read_text())
            if asset_pack_override:
                data["assetPack"] = asset_pack_override
            if composition_strategy:
                data["compositionStrategy"] = composition_strategy
            snapshot.write_text(json.dumps(data, indent=2))
        out_root = self.out_dir / "realized"
        realizer = Path(__file__).parent / "sketch_realize.py"
        venv_py = os.environ.get("SKETCH_STAGE_PY", sys.executable)
        timeout_s = int(os.environ.get("SKETCH_REALIZE_TIMEOUT_S", "1800"))
        result = subprocess.run(
            [venv_py, str(realizer), str(snapshot), str(out_root)],
            capture_output=True, text=True, timeout=timeout_s,
        )
        manifest_path = out_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        # Surface anchor + added counts so the LLM can report a tile-level breakdown
        total = len(self.index)
        anchor = self.anchor_placement_count
        added = total - anchor
        manifest["templateBreakdown"] = {
            "anchorPlacementCount": anchor,
            "addedPlacementCount": added,
            "totalPlacementCount": total,
            "tileCountIfAddedAreFullCopies": (added // anchor + 1) if anchor > 0 else None,
            "perTilePlacementCount": anchor,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        ok = (result.returncode == 0)
        event = {"outRoot": str(out_root),
                 "returncode": result.returncode,
                 "stdout_tail": result.stdout.splitlines()[-5:],
                 "summary": manifest.get("summary", {}),
                 "templateBreakdown": manifest["templateBreakdown"]}
        if not ok:
            # Surface the error so callers can see why realize failed instead
            # of swallowing the subprocess stderr.
            event["stderr_tail"] = result.stderr.splitlines()[-20:]
        self._emit("realized", event)
        response = {
            "ok": ok,
            "rootUsd": str(out_root / "root.usd"),
            "snapshot": str(snapshot),
            "manifest": str(manifest_path),
            "summary": manifest.get("summary", {}),
            "templateBreakdown": manifest["templateBreakdown"],
        }
        if not ok:
            response["returncode"] = result.returncode
            response["stderr_tail"] = result.stderr.splitlines()[-20:]
        return response

    def realize_passthrough(self, targetComposedPrimCount: int = 1000,
                            absorbedTemplate: str | None = None) -> dict:
        """Realize via passthrough_rootcell mode: tile the asset pack's
        rootStage USD enough times to hit the target prim count. Bypasses
        the per-placement session entirely. Requires the pack to have a
        pack.json with `rootStage` set."""
        import os, subprocess
        sketch_path = self.out_dir / "_passthrough_sketch.json"
        sk = {
            "schemaVersion": 1,
            "mode": "passthrough_rootcell",
            "assetPack": str(self.asset_pack_dir) if self.asset_pack_dir else "",
            "targetComposedPrimCount": int(targetComposedPrimCount),
            "seed": 20260506,
        }
        if absorbedTemplate:
            sk["absorbedTemplate"] = absorbedTemplate
        elif self.anchor_path:
            sk["absorbedTemplate"] = str(self.anchor_path)
        sketch_path.write_text(json.dumps(sk, indent=2))
        out_root = self.out_dir / "realized"
        env = dict(os.environ)
        env["SKETCH_LIVE_EVENTS"] = env.get(
            "SKETCH_LIVE_EVENTS", _DEFAULT_LIVE_EVENTS)
        venv_py = os.environ.get("SKETCH_STAGE_PY", sys.executable)
        realizer = str(Path(__file__).parent / "sketch_realize.py")
        result = subprocess.run(
            [venv_py, realizer, str(sketch_path), str(out_root)],
            capture_output=True, text=True, env=env, timeout=600,
        )
        manifest_path = out_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        return {
            "ok": result.returncode == 0,
            "rootUsd": str(out_root / "root.usd"),
            "manifest": str(manifest_path),
            "tilesAuthored": manifest.get("tilesAuthored"),
            "estimatedComposedPrimCount": manifest.get("estimatedComposedPrimCount"),
            "composedPrimCount_usdcore": manifest.get("composedPrimCount_usdcore"),
        }

    def template_view(self) -> dict:
        """Expose zones + shells for the viz to draw as a template backdrop."""
        return {
            "zones": [dict(z) for z in self.zones],
            "shells": [dict(s) for s in self.shells],
            "footprintM": ({"minX": min(p.posM[0] for p in self.index.all()),
                            "maxX": max(p.posM[0] for p in self.index.all()),
                            "minY": min(p.posM[1] for p in self.index.all()),
                            "maxY": max(p.posM[1] for p in self.index.all())}
                           if self.index.all() else None),
        }
