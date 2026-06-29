# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime registry of available 'surfaces' on every placement, plus the
`query_surfaces` and `place_on` operations consumed by MCP tools.

A "surface" is a flat top-face of an existing placement that other (smaller)
placements can rest on. Sources of surface info, merged at session open:

  1. **Pack-declared** — each archetype's `semantics.surfaces: [{label,
     localTopZ, footprintM: [w, d]}]` (from `harvest_pack_semantics.py` +
     LLM gap-fill). These are *local* to the asset's frame.
  2. **Sketch overrides** (optional) — an absorbed sketch may specify
     per-placement overrides under `placement.surfaces: [...]` to replace
     the archetype defaults for one specific instance (e.g. a partly-
     draped table).

At session open the per-placement surfaces are transformed to world space
using the placement's centre + yaw, so `query_surfaces` returns world-frame
poses and `place_on` only needs surface-local XY from the caller.

Collision is **sibling-only on the same surface**: two cups on the same
table check each other but not the table itself, and not anything in the
spatial index outside the surface. This is the change that makes
cup-on-table viable; before, the cup's XY footprint would overlap the
table's slot and the gate would reject.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from pack_loader import load_pack  # noqa: E402


# ---------------------------------------------------------------------------
# Surface records
# ---------------------------------------------------------------------------

@dataclass
class _Surface:
    label: str
    owner_id: str                  # the placement this surface lives on
    owner_archetype: str
    owner_yaw_deg: float
    center_world_xy: tuple[float, float]   # surface centre projected to world XY
    top_world_z: float                      # world Z of the surface's top
    footprint_m: tuple[float, float]        # surface dimensions (w, d) in metres
    label_index: int = 0                    # 0-based ordinal among siblings with the same label on this owner
    occupied: list[dict] = field(default_factory=list)
    # Each occupant: {"id", "archetype", "localXY", "sizeM"}

    @property
    def surface_id(self) -> str:
        """Stable id unique within the registry. Format: '<owner_id>:<label>#<label_index>'.
        Use this to address a specific surface in `place_on` when an owner has
        multiple surfaces with the same label (e.g. a rack with three shelves)."""
        return f"{self.owner_id}:{self.label}#{self.label_index}"

    @property
    def free_area_m2(self) -> float:
        w, d = self.footprint_m
        used = sum(o["sizeM"][0] * o["sizeM"][1] for o in self.occupied)
        return max(0.0, w * d - used)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class SurfaceRegistry:
    def __init__(self) -> None:
        self._by_owner: dict[str, list[_Surface]] = {}

    def __len__(self) -> int:
        return sum(len(s) for s in self._by_owner.values())

    @classmethod
    def from_pack_and_placements(cls, pack_dir: Path,
                                  placements: Iterable[Any],
                                  sketch_overrides: dict | None = None) -> "SurfaceRegistry":
        """Build a fresh registry from the pack semantics + the session's
        current placements.

        `placements` is iterable of objects with attributes
        `id, archetype, posM, slotM, yawDeg, scaleM` (matches IndexedPlacement).
        Pass `sketch_overrides` (a dict keyed by placement id → list of
        surface decls) to override pack defaults for specific instances.
        """
        sketch_overrides = sketch_overrides or {}
        reg = cls()
        reg._pack_dir = pack_dir
        reg._sketch_overrides = sketch_overrides
        # Per-archetype: a list of full pack entries (each with `path`,
        # `size`, `semantics`). This replaces the old "flatten to first
        # entry" behaviour — placements now resolve to a specific entry
        # via _pick_entry(slotM), and bind_archetype can refresh to a
        # different entry's surfaces later.
        reg._entries_by_arch: dict[str, list[dict]] = {}
        # Track each owner's currently bound asset path (set when
        # bind_archetype refreshes the registry). None == "matcher
        # hasn't picked yet; we used best-fit-by-slotM as initial guess".
        reg._bound_asset_by_owner: dict[str, str | None] = {}
        try:
            pack = load_pack(pack_dir)
        except Exception:
            return reg
        for ns_name, entries in pack["archetypes"].items():
            bare = ns_name.split(".", 1)[1] if "." in ns_name else ns_name
            reg._entries_by_arch.setdefault(bare, entries)
            reg._entries_by_arch[ns_name] = entries
        for p in placements:
            reg.add_placement(p)
        return reg

    @staticmethod
    def _entry_size(entry: dict) -> tuple[float, float, float]:
        s = entry.get("size") or {}
        return (float(s.get("widthM", 0)),
                float(s.get("depthM", 0)),
                float(s.get("heightM", 0)))

    def _pick_entry(self, archetype: str,
                    slotM: tuple[float, float, float]) -> dict | None:
        """Best-fit entry for a placement's slotM. Mirrors the realize-step
        matcher: prefer entries whose bbox fits inside the slot (so the
        asset doesn't poke out), then prefer the largest among those (so
        the asset fills the slot rather than rattling around). Falls back
        to the first entry when no slot is provided.
        """
        entries = self._entries_by_arch.get(archetype) or []
        if not entries:
            return None
        if not slotM or len(slotM) < 3:
            return entries[0]
        sw, sd, sh = float(slotM[0]), float(slotM[1]), float(slotM[2])

        def cost(e: dict) -> tuple[int, float, float]:
            ew, ed, eh = self._entry_size(e)
            # Two-pass cost:
            #   1) penalize entries bigger than slot in any axis (asset
            #      will poke out / fail collision)
            #   2) among the rest, prefer the entry whose volume is
            #      closest to the slot's volume.
            over = max(0.0, ew - sw) + max(0.0, ed - sd) + max(0.0, eh - sh)
            slot_vol = max(sw * sd * sh, 1e-9)
            ent_vol = ew * ed * eh
            vol_gap = abs(slot_vol - ent_vol) / slot_vol
            tier = 1 if over > 0 else 0
            return (tier, over, vol_gap)

        return min(entries, key=cost)

    def _pick_entry_by_path(self, archetype: str, asset_path: str) -> dict | None:
        """Find the entry whose `path` matches `asset_path`. Used by
        session.bind_archetype after the user (or the matcher) picks a
        specific USD.
        """
        entries = self._entries_by_arch.get(archetype) or []
        if not entries:
            return None
        # Accept both prefix- and suffix-style matches so callers can pass
        # either a relative or absolute path.
        ap = asset_path or ""
        for e in entries:
            p = e.get("path", "")
            if p == ap or ap.endswith("/" + p) or p.endswith(ap):
                return e
        return None

    @staticmethod
    def _placement_field(p: Any, name: str, default=None):
        if hasattr(p, name):
            v = getattr(p, name)
            return v if v is not None else default
        if isinstance(p, dict):
            return p.get(name, default)
        return default

    def add_placement(self, p: Any, *, force_entry: dict | None = None) -> int:
        """Add surfaces for a newly inserted placement.

        `force_entry`, when set, bypasses the best-fit lookup and uses
        that entry's surfaces directly. Used by bind_archetype to refresh
        all placements of an archetype to a specific asset's surfaces.

        Safe to call before from_pack_and_placements has been invoked
        (returns 0 in that case — no pack semantics loaded yet).
        """
        entries_by_arch = getattr(self, "_entries_by_arch", None)
        if not entries_by_arch:
            return 0
        sketch_overrides = getattr(self, "_sketch_overrides", {}) or {}

        arch = self._placement_field(p, "archetype")
        pid = self._placement_field(p, "id")
        pos = self._placement_field(p, "posM")
        slot = self._placement_field(p, "slotM", (0.0, 0.0, 0.0))
        yaw = self._placement_field(p, "yawDeg", 0.0) or 0.0
        scale = self._placement_field(p, "scaleM")  # None when not bound

        if arch is None or pid is None or pos is None:
            return 0

        # Pick the entry whose `surfaces` block we'll project into world
        # space for this placement.
        entry = force_entry if force_entry is not None else self._pick_entry(arch, slot)
        if entry is None:
            return 0
        sem = entry.get("semantics") or {}
        decls = sketch_overrides.get(pid) or sem.get("surfaces") or []

        # When the placement has been scaled (e.g. fit-to-slot via
        # bind_archetype), the asset's local Z stretches with it. The
        # pack's `localTopZ` is declared in the asset's NATIVE frame;
        # we multiply by scaleM[2] to recover the world Z extent. Same
        # logic for footprint (scaleM[0], scaleM[1]).
        sx = float(scale[0]) if scale and len(scale) > 0 else 1.0
        sy = float(scale[1]) if scale and len(scale) > 1 else 1.0
        sz = float(scale[2]) if scale and len(scale) > 2 else 1.0

        # Runtime safety net for pack-measurement errors: when an entry
        # declares exactly one surface whose localTopZ is far from the
        # asset's native bbox top, snap it to native height. Catches three
        # failure modes:
        #
        #  1. Drum/bucket/pail shapes whose only detected flat face is the
        #     BASE (localTopZ ≈ 0.02 against a 0.7 m drum) — boxes would
        #     land near the floor instead of on the lid (sunken).
        #  2. Placeholder localTopZ values shared across an archetype's
        #     entries (e.g. every `box` declares 0.4 regardless of the
        #     bound asset's real height) — boxes either sink (tall asset)
        #     or float (short asset) by the declared/actual delta.
        #  3. Slight measurement undershoots from picking the underside of
        #     a thin top mesh (e.g. crate lid at 1.012 vs bbox top 1.018).
        #
        # Skips multi-surface entries — those legitimately have intermediate
        # surfaces below the bbox top. Tolerance is 1 cm.
        size = entry.get("size") or {}
        native_h = float(size.get("heightM") or 0.0)
        if (len(decls) == 1 and native_h > 0.05
                and isinstance(decls[0].get("localTopZ"), (int, float))
                and abs(float(decls[0]["localTopZ"]) - native_h) > 0.01
                and not decls[0].get("_source") == "bbox_top_fallback"):
            d0 = dict(decls[0])
            d0["localTopZ"] = native_h
            d0["_runtimeBboxFallback"] = True
            decls = [d0]

        added = 0
        label_seen: dict[str, int] = {}
        for d in decls:
            label = d.get("label")
            top_z_local = d.get("localTopZ")
            fp = d.get("footprintM")
            if not (label and isinstance(top_z_local, (int, float))
                    and isinstance(fp, list) and len(fp) == 2):
                continue
            cx, cy = float(pos[0]), float(pos[1])
            cz = float(pos[2]) if len(pos) > 2 else 0.0
            li = label_seen.get(str(label), 0)
            label_seen[str(label)] = li + 1
            self._by_owner.setdefault(pid, []).append(_Surface(
                label=str(label),
                owner_id=pid,
                owner_archetype=arch,
                owner_yaw_deg=float(yaw),
                center_world_xy=(cx, cy),
                top_world_z=cz + float(top_z_local) * sz,
                footprint_m=(float(fp[0]) * sx, float(fp[1]) * sy),
                label_index=li,
            ))
            added += 1
        return added

    def replace_placement(self, p: Any, *, force_entry: dict | None = None) -> int:
        """Drop all surfaces currently registered for this placement, then
        rebuild from `force_entry` (or best-fit again). Called by
        session.bind_archetype when the bound asset changes — old surfaces
        from the previous entry need to go before new ones are added.
        """
        pid = self._placement_field(p, "id")
        if pid is not None:
            self._by_owner.pop(pid, None)
        return self.add_placement(p, force_entry=force_entry)

    def rebind_archetype(self, archetype: str, asset_path: str,
                          placements: Iterable[Any]) -> dict:
        """Refresh the surfaces of every placement of `archetype` to use
        the surface block from `asset_path`'s pack entry. Called from
        session.bind_archetype after the binding (and possibly scaleM)
        is updated on the spatial index.

        Returns {entryFound, refreshed, surfacesAdded}.
        """
        entry = self._pick_entry_by_path(archetype, asset_path)
        if entry is None:
            return {"entryFound": False, "refreshed": 0, "surfacesAdded": 0}
        refreshed = 0
        added = 0
        for p in placements:
            if self._placement_field(p, "archetype") != archetype:
                continue
            n = self.replace_placement(p, force_entry=entry)
            refreshed += 1
            added += n
            pid = self._placement_field(p, "id")
            if pid:
                self._bound_asset_by_owner[pid] = asset_path
        return {"entryFound": True, "refreshed": refreshed,
                "surfacesAdded": added, "entryPath": entry.get("path")}

    def remove_placement(self, owner_id: str) -> int:
        """Drop all surfaces owned by a placement (used by session.remove)."""
        removed = len(self._by_owner.pop(owner_id, []))
        self._bound_asset_by_owner.pop(owner_id, None)
        return removed

    # ----- queries -----

    def list_all(self) -> list[_Surface]:
        return [s for owner in self._by_owner.values() for s in owner]

    def surfaces_of(self, owner_id: str) -> list[_Surface]:
        return list(self._by_owner.get(owner_id, []))

    def query(self, region_fp: Optional[tuple[tuple[float, float], tuple[float, float]]] = None,
              min_free_area_m2: float = 0.0,
              label: Optional[str] = None) -> list[dict]:
        """Return surfaces that satisfy the filters. `region_fp` is an
        AABB ((x0, y0), (x1, y1)) in world XY; only surfaces whose centre
        lies inside the AABB are included.
        """
        out: list[dict] = []
        for s in self.list_all():
            if label and s.label != label:
                continue
            if s.free_area_m2 < min_free_area_m2:
                continue
            if region_fp is not None:
                (x0, y0), (x1, y1) = region_fp
                cx, cy = s.center_world_xy
                if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                    continue
            out.append({
                "owner": s.owner_id,
                "ownerArchetype": s.owner_archetype,
                "label": s.label,
                "labelIndex": s.label_index,
                "surfaceId": s.surface_id,
                "centerWorldXY": list(s.center_world_xy),
                "topWorldZ": s.top_world_z,
                "footprintM": list(s.footprint_m),
                "freeAreaM2": round(s.free_area_m2, 4),
                "ownerYawDeg": s.owner_yaw_deg,
            })
        return out

    def remove_occupant(self, occupant_id: str) -> int:
        """Strip an occupant id from any surface.occupied lists.

        Used by session.remove(id) so that removing a place_on child also
        clears its slot on the parent's surface — otherwise siblings keep
        seeing the orphan and reject new placements at the freed XY.

        Returns the number of surfaces from which the occupant was removed
        (typically 0 or 1)."""
        n = 0
        for surfaces in self._by_owner.values():
            for s in surfaces:
                before = len(s.occupied)
                s.occupied = [o for o in s.occupied if o.get("id") != occupant_id]
                if len(s.occupied) < before:
                    n += 1
        return n

    # ----- placement -----

    def place_on(self, *, parent_id: str, surface_label: str,
                 local_xy: tuple[float, float], yaw_deg: float,
                 archetype: str, archetype_sizeM: tuple[float, float, float],
                 region_allowed_archetypes: Optional[list[str]] = None,
                 region_name: Optional[str] = None,
                 placement_id: Optional[str] = None,
                 surface_index: Optional[int] = None,
                 top_world_z: Optional[float] = None,
                 z_tolerance_m: float = 0.05) -> dict:
        """Validate + record a new child placement on an existing surface.

        When `parent_id` has multiple surfaces sharing `surface_label` (e.g.
        a rack with three shelves), pass exactly one disambiguator:

          - `surface_index`: 0-based ordinal among surfaces with this label
            (matches `labelIndex` in `query_surfaces` output), OR
          - `top_world_z`: select the surface whose top z is within
            `z_tolerance_m` of this value.

        With no disambiguator the first matching surface is used (legacy
        behaviour). With both, `surface_index` wins.

        Failure modes (returned as `{"ok": False, "rejected": {reason, stage}}`):

          - `lookup` — no such surface on the parent
          - `ambiguous_surface` — multiple matches and no disambiguator
          - `region_filter` — archetype not in the region's allowed list
          - `edge_check` — localXY would put the bbox off the surface
          - `sibling_collision` — overlaps another occupant on the same surface

        On success returns the new child placement payload that callers
        can persist into the sketch's tree (parent-child relationship is
        explicit via `parentPlacement` / `parentSurface`).
        """
        surfaces = self._by_owner.get(parent_id, [])
        label_matches = [s for s in surfaces if s.label == surface_label]
        if not label_matches:
            return {"ok": False, "rejected": {
                "reason": f"no surface {surface_label!r} on placement {parent_id!r}",
                "stage": "lookup",
            }}
        if surface_index is not None:
            if not 0 <= surface_index < len(label_matches):
                return {"ok": False, "rejected": {
                    "reason": (f"surfaceIndex {surface_index} out of range; "
                               f"{parent_id!r} has {len(label_matches)} "
                               f"surface(s) labeled {surface_label!r}"),
                    "stage": "lookup",
                }}
            surface = label_matches[surface_index]
        elif top_world_z is not None:
            cand = [(abs(s.top_world_z - float(top_world_z)), s)
                    for s in label_matches]
            cand.sort(key=lambda t: t[0])
            if cand[0][0] > float(z_tolerance_m):
                return {"ok": False, "rejected": {
                    "reason": (f"no surface {surface_label!r} on {parent_id!r} "
                               f"with topWorldZ ≈ {top_world_z} "
                               f"(closest off by {cand[0][0]:.3f} m)"),
                    "stage": "lookup",
                }}
            surface = cand[0][1]
        elif len(label_matches) > 1:
            zs = [round(s.top_world_z, 3) for s in label_matches]
            return {"ok": False, "rejected": {
                "reason": (f"{parent_id!r} has {len(label_matches)} surfaces "
                           f"labeled {surface_label!r} at topWorldZ={zs}; "
                           f"pass surfaceIndex or topWorldZ to disambiguate"),
                "stage": "ambiguous_surface",
            }}
        else:
            surface = label_matches[0]

        if region_allowed_archetypes:
            allowed = set(region_allowed_archetypes)
            if archetype not in allowed:
                return {"ok": False, "rejected": {
                    "reason": (f"archetype {archetype!r} not allowed in region "
                               f"{region_name!r} (allowed: {sorted(allowed)})"),
                    "stage": "region_filter",
                }}

        fw, fd = surface.footprint_m
        lx, ly = float(local_xy[0]), float(local_xy[1])
        cw, cd, ch = (float(archetype_sizeM[0]), float(archetype_sizeM[1]),
                       float(archetype_sizeM[2]))
        half_w, half_d = fw / 2, fd / 2
        if not (-half_w + cw / 2 <= lx <= half_w - cw / 2
                and -half_d + cd / 2 <= ly <= half_d - cd / 2):
            return {"ok": False, "rejected": {
                "reason": (f"localXY {list(local_xy)} would put the "
                            f"{archetype} bbox off the {surface_label} "
                            f"(footprint {list(surface.footprint_m)})"),
                "stage": "edge_check",
            }}

        for occ in surface.occupied:
            ox, oy = occ["localXY"]
            ow, od = occ["sizeM"][:2]
            if (abs(lx - ox) < (cw + ow) / 2
                    and abs(ly - oy) < (cd + od) / 2):
                return {"ok": False, "rejected": {
                    "reason": (f"would overlap sibling {occ['id']!r} "
                                f"on same surface"),
                    "stage": "sibling_collision",
                }}

        # Compose to world frame: rotate localXY by owner's yaw, translate.
        # posM[2] follows the realize-step convention: the asset's bbox-bottom
        # lands at posM[2]. surface.top_world_z is the parent surface's top in
        # world coords, so the child's posM[2] is exactly that — child rests on
        # the surface.
        a = math.radians(surface.owner_yaw_deg)
        rx = lx * math.cos(a) - ly * math.sin(a)
        ry = lx * math.sin(a) + ly * math.cos(a)
        cx, cy = surface.center_world_xy
        world = [cx + rx, cy + ry, surface.top_world_z]

        # Include the surface's label + label_index in the auto-id so that
        # two different shelves of the same rack don't collide on
        # `child_{rack}_{archetype}_1`. The spatial index silently drops
        # duplicates, so collisions would manifest as missing placements
        # (the 450→150 drop the user reported when boxes-on-shelves first
        # ran without this fix).
        new_id = placement_id or (
            f"child_{parent_id}_{surface.label}{surface.label_index}_"
            f"{archetype}_{len(surface.occupied) + 1}")
        occupant = {"id": new_id, "archetype": archetype,
                    "localXY": [lx, ly], "sizeM": [cw, cd, ch]}
        surface.occupied.append(occupant)

        return {"ok": True, "placement": {
            "id": new_id,
            "archetype": archetype,
            "parentPlacement": parent_id,
            "parentSurface": surface_label,
            "parentSurfaceIndex": surface.label_index,
            "parentSurfaceId": surface.surface_id,
            "localXY": [lx, ly],
            "yawDeg": yaw_deg,
            "posM": [round(world[0], 4), round(world[1], 4),
                     round(world[2], 4)],
            "slotM": {"widthM": cw, "depthM": cd, "heightM": ch},
        }}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_archetype_semantics(archetypes: dict) -> dict:
    """Return {plain_arch_name: semantics_block} from a (possibly
    namespaced-with-theme) archetype dict. Plain names take precedence over
    later collisions across themes — the caller should restrict to one
    sub-pack first if collisions matter.
    """
    out: dict = {}
    for name, entries in archetypes.items():
        if not entries:
            continue
        sem = entries[0].get("semantics") if isinstance(entries[0], dict) else None
        if not sem:
            continue
        bare = name.split(".", 1)[1] if "." in name else name
        out.setdefault(bare, sem)
        out[name] = sem
    return out
