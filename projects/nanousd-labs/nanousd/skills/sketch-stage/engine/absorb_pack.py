# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic absorb-source-as-layout tool.

Takes any source USD URL (local path or http://) and emits a sketch.json
that the sketch-incremental-placement skill can consume as an anchor
template.

Design principle: the absorbed sketch is a **layout + placement semantics**,
NOT a snapshot of the source's assets. Each placement records what archetype
was there (derived from the source itself) and where, but does NOT carry the
original asset reference forward. Later, the LLM decides which asset from a
target pack to bind to each placement, using the placement's archetype +
position semantics and the pack's per-archetype semantics — driven by tools,
not by a fixed mapping.

Archetype labels come from the source stage itself (the referenced asset's
filename stem, falling back to the prim name) — no built-in vocabulary, no
keyword filter, no target-pack dependency at absorb time. This makes the
absorber scenario-agnostic: it captures whatever's in the source, whether
warehouse, apartment, factory, hospital, restaurant, or anything else.

Usage:
  absorb_pack.py --source /path/or/url.usd --out /path/to/template.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path

from pxr import Usd, UsdGeom, Gf
try:
    from pxr import UsdSemantics
    _HAS_USD_SEMANTICS = True
except ImportError:
    _HAS_USD_SEMANTICS = False

# OBB-SAT collision (yaw-aware, 3D) — shared with the runtime spatial index
sys.path.insert(0, str(Path(__file__).parent))
from spatial_index import obb_overlap as _obb_overlap  # noqa: E402


# Archetype labels come from the source stage itself: the referenced
# asset's filename stem (e.g. ref to "./SubUSDs/Kitchen_Cabinet.usd" →
# archetype "Kitchen_Cabinet"), falling back to the prim name (with the
# "_instances" suffix stripped) when no ref/payload is present.
_INSTANCE_SUFFIX_RE = ("_instances", "_instance")


def _archetype_from_prim(prim: Usd.Prim) -> str:
    # Prefer the referenced asset's filename — instances of the same asset
    # all share that, so 6 Kitchen_Oven references collapse to one archetype
    # automatically.
    for spec in prim.GetPrimStack():
        for r in spec.referenceList.GetAddedOrExplicitItems():
            ap = r.assetPath
            if ap:
                stem = Path(str(ap).split("?", 1)[0]).stem
                if stem:
                    return stem
        for r in spec.payloadList.GetAddedOrExplicitItems():
            ap = r.assetPath
            if ap:
                stem = Path(str(ap).split("?", 1)[0]).stem
                if stem:
                    return stem
    # Fall back to the prim name itself, stripping the common
    # "_instances" / "_instance{N}" suffix so instance variants merge.
    name = prim.GetName()
    for suf in _INSTANCE_SUFFIX_RE:
        idx = name.rfind(suf)
        if idx >= 0:
            return name[:idx]
    return name


def _resolve_source(source: str) -> Path:
    """Local path or http(s):// URL → local file path."""
    if source.startswith(("http://", "https://")):
        td = Path(tempfile.mkdtemp(prefix="absorb_pack_"))
        local = td / Path(source.split("?")[0]).name
        print(f"[absorb] downloading {source} → {local}", flush=True)
        urllib.request.urlretrieve(source, local)
        return local
    p = Path(source).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(source)
    return p


def _has_authored_ref_or_payload(prim: Usd.Prim) -> bool:
    return bool(prim.HasAuthoredReferences() or prim.HasAuthoredPayloads())


_SR_KINDS = {"component", "assembly", "group", "subcomponent"}


def _prim_has_simready_metadata(prim: Usd.Prim) -> bool:
    """Per-prim SimReady check: does THIS prim carry semantic metadata
    that makes it an "imaginable" object (one the LLM can reason about
    and bind to a target asset)? Anything without such metadata is
    internal SimReady scaffolding (material refs, physics rigs,
    geometry payloads) and not a placement.
    """
    cd = prim.GetCustomData() or {}
    if isinstance(cd, dict) and cd.get("simReady"):
        return True
    if _HAS_USD_SEMANTICS:
        try:
            if UsdSemantics.LabelsAPI.GetDirectTaxonomies(prim):
                return True
        except Exception:
            pass
    k = Usd.ModelAPI(prim).GetKind()
    if k and str(k).lower() in _SR_KINDS:
        return True
    return False


def _yaw_from_matrix(M: Gf.Matrix4d) -> float:
    r = M.ExtractRotation()
    yaw = r.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))[2]
    return 0.0 if math.isnan(yaw) else float(yaw)


def _pos_slot_from_bbox(prim: Usd.Prim, bbox_cache: UsdGeom.BBoxCache,
                        src_up: str, src_mpu: float):
    """Measure the prim's world-space bbox and return (posM_world, slotM)
    in Z-up meters, where `posM` is the bbox's **center-floor** (the
    skill's slot-anchor convention — slot center in X/Y, slot floor in Z)
    and `slotM` is {widthM, depthM, heightM}. None if the bbox is
    degenerate (no geometry under the prim).

    Using bbox-center-floor here — not `M.ExtractTranslation()` — keeps
    the placement's collision OBB aligned with the rendered geometry
    regardless of where the source asset's xform pivot happens to sit.
    """
    try:
        bb = bbox_cache.ComputeWorldBound(prim)
    except Exception:
        return None
    r = bb.ComputeAlignedRange()
    if r.IsEmpty():
        return None
    mn, mx = r.GetMin(), r.GetMax()
    # bbox centers and span along each source-frame axis (still in source units).
    cx = (float(mn[0]) + float(mx[0])) * 0.5
    cy = (float(mn[1]) + float(mx[1])) * 0.5
    cz = (float(mn[2]) + float(mx[2])) * 0.5
    sx = float(mx[0]) - float(mn[0])
    sy = float(mx[1]) - float(mn[1])
    sz = float(mx[2]) - float(mn[2])
    # Convert to world Z-up meters. "Floor" is the bbox-min along the
    # source's up-axis (Y for Y-up, Z for Z-up).
    if src_up == "Y":
        # source X (horizontal) → world X
        # source Z (horizontal) → world Y
        # source Y (vertical)   → world Z;  floor = source Y min
        posM = [cx * src_mpu, cz * src_mpu, float(mn[1]) * src_mpu]
        slot = {"widthM": sx * src_mpu, "depthM": sz * src_mpu, "heightM": sy * src_mpu}
    else:
        # Z-up source: identity.  floor = source Z min
        posM = [cx * src_mpu, cy * src_mpu, float(mn[2]) * src_mpu]
        slot = {"widthM": sx * src_mpu, "depthM": sy * src_mpu, "heightM": sz * src_mpu}
    if slot["widthM"] <= 0 or slot["depthM"] <= 0 or slot["heightM"] <= 0:
        return None
    return posM, slot


def _semantic_info_from_prim(prim: Usd.Prim) -> dict:
    out: dict = {}
    kind_api = Usd.ModelAPI(prim)
    k = kind_api.GetKind() if kind_api else None
    if k:
        out["kind"] = str(k)
    if _HAS_USD_SEMANTICS:
        try:
            for tax in UsdSemantics.LabelsAPI.GetDirectTaxonomies(prim):
                api = UsdSemantics.LabelsAPI(prim, tax)
                attr = api.GetLabelsAttr()
                labels = attr.Get() if attr else None
                if labels:
                    out.setdefault("usdSemanticsLabels", {})[str(tax)] = list(labels)
        except Exception:
            pass
    cd = prim.GetCustomData() or {}
    if "simReady" in cd:
        v = cd["simReady"]
        out["simReady"] = dict(v) if hasattr(v, "keys") else v
    ai = prim.GetAssetInfo() or {}
    if ai:
        out["assetInfo"] = {k: str(v) for k, v in ai.items()}
    return out


def _scan_simready_coverage(stage: Usd.Stage) -> tuple[int, int]:
    """Return (ref_or_payload_prim_count, prims_with_simready_metadata).

    "SimReady metadata" = any of: customData["simReady"] is set, OR a
    UsdSemantics direct taxonomy is attached, OR Kind is "component"/
    "assembly" (the SimReady taxonomy markers). Walk every ref/payload-
    bearing Xformable prim and count how many carry at least one such
    annotation.
    """
    total = 0; tagged = 0
    SR_KINDS = {"component", "assembly", "group", "subcomponent"}
    for prim in stage.Traverse():
        if not UsdGeom.Xformable(prim):
            continue
        if not (prim.HasAuthoredReferences() or prim.HasAuthoredPayloads()):
            continue
        total += 1
        has_tag = False
        cd = prim.GetCustomData() or {}
        if isinstance(cd, dict) and cd.get("simReady"):
            has_tag = True
        if not has_tag and _HAS_USD_SEMANTICS:
            try:
                if UsdSemantics.LabelsAPI.GetDirectTaxonomies(prim):
                    has_tag = True
            except Exception:
                pass
        if not has_tag:
            kind_api = Usd.ModelAPI(prim)
            k = kind_api.GetKind() if kind_api else None
            if k and str(k).lower() in SR_KINDS:
                has_tag = True
        if has_tag:
            tagged += 1
    return total, tagged


def absorb(source: str, out: str, target_pack: str | None = None) -> dict:
    """Capture a SimReady source USD as a layout sketch.

    The absorbed sketch is a **layout + placement semantics** description,
    not a snapshot of the source's asset references. Each placement records
    its archetype label (derived from the source itself), world position/
    yaw, slot dimensions from the prim's own bbox, and any UsdSemantics /
    Kind / customData["simReady"] semantic info — but NOT the original
    asset URL. Binding to actual pack assets happens later, in a separate
    LLM-driven step that uses the placement's archetype + pack semantics
    + tools.

    **Source must be SimReady-tagged.** The absorber refuses non-SimReady
    sources: without `customData["simReady"]` / `UsdSemantics` labels /
    SimReady `Kind` markers, the absorber has no reliable way to drive
    downstream binding decisions.

    Captures **every** ref/payload-bearing Xformable prim — including
    walls, floors, ceilings, rack assemblies, and other so-called
    "structural" geometry. Nothing is discarded by size or vocabulary;
    SimReady metadata carries each prim's semantic role, so the LLM
    (not the absorber) decides at bind time which placements to skip,
    passthrough, or bind to pack assets.

    `target_pack` is recorded in the sketch metadata so realize knows
    which pack to consult later, but it doesn't influence absorption.
    """
    src = _resolve_source(source)
    print(f"[absorb] opening {src}", flush=True)
    s = Usd.Stage.Open(str(src))
    # SimReady gate: refuse to absorb sources without SimReady annotations.
    total, tagged = _scan_simready_coverage(s)
    SR_COVERAGE_MIN = 0.5  # require ≥50% of ref/payload prims to carry SR metadata
    coverage = (tagged / total) if total else 0.0
    if total == 0 or coverage < SR_COVERAGE_MIN:
        raise SystemExit(
            f"[absorb] source is not SimReady-tagged "
            f"({tagged}/{total} ref/payload prims carry SimReady metadata, "
            f"coverage {coverage*100:.0f}% < {SR_COVERAGE_MIN*100:.0f}%). "
            f"The absorber only accepts SimReady sources — bring a USD whose "
            f"prims carry customData[\"simReady\"] or UsdSemantics labels or "
            f"SimReady Kind markers (component/assembly). Without that "
            f"semantic info the LLM has no reliable way to bind absorbed "
            f"placements to a target pack downstream."
        )
    print(f"[absorb] SimReady coverage: {tagged}/{total} "
          f"({coverage*100:.0f}%)", flush=True)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                   ["default"], useExtentsHint=True)
    # Source-axis/mpu normalization: the absorbed sketch is always Z-up
    # meters, but the source may be Y-up cm (typical for asset-library
    # USDs). Transform every world translate accordingly:
    #   - multiply by metersPerUnit to convert source units to meters
    #   - if upAxis is Y, swap Y and Z (Y-up "up" maps to Z-up "up")
    _src_up = str(UsdGeom.GetStageUpAxis(s) or "Z").upper()
    _src_mpu = float(UsdGeom.GetStageMetersPerUnit(s) or 1.0)

    def _to_world_m(t):
        x, y, z = float(t[0]) * _src_mpu, float(t[1]) * _src_mpu, float(t[2]) * _src_mpu
        if _src_up == "Y":
            return [x, z, y]  # source Y (up) -> world Z; source Z -> world Y
        return [x, y, z]

    # Capture the topmost RENDERABLE Xform instance prims. Rules:
    #   1. Must be an Xform (the wrapper). Mesh/Cube/etc. are internal
    #      asset geometry, part of an outer placement, not separate
    #      placements themselves.
    #   2. Must instantiate an external asset (reference or payload) —
    #      a bare Xform with no asset is a transform grouping, not a
    #      placement.
    #   3. Must render something — non-empty computed world bbox.
    #      Material refs, physics-only refs, deactivated/invisible
    #      prims have empty bboxes and get filtered out here.
    #   4. Must be the topmost such prim along its path. Internal
    #      asset Xforms reached via composition (sub-component groups,
    #      geometry payloads, LOD wrappers) are part of an outer
    #      placement, not new ones.
    placement_prims: list[Usd.Prim] = []
    seen_ancestor_paths: list[str] = []
    for prim in s.Traverse():
        if prim.GetTypeName() != "Xform":
            continue
        if not _has_authored_ref_or_payload(prim):
            continue
        ppath = prim.GetPath().pathString
        if any(ppath.startswith(a + "/") for a in seen_ancestor_paths):
            continue
        try:
            bb = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        except Exception:
            continue
        if bb.IsEmpty():
            continue
        mn, mx = bb.GetMin(), bb.GetMax()
        if (float(mx[0]) - float(mn[0])) <= 0 or \
           (float(mx[1]) - float(mn[1])) <= 0 or \
           (float(mx[2]) - float(mn[2])) <= 0:
            continue
        placement_prims.append(prim)
        seen_ancestor_paths.append(ppath)

    placements = []
    archetype_source_info: dict[str, dict] = {}
    skipped_no_bbox = 0
    skipped_decal = 0
    DECAL_HEIGHT_THRESHOLD_M = 0.05  # anything thinner is floor paint / sticker
    for prim in placement_prims:
        archetype = _archetype_from_prim(prim)
        pos_slot = _pos_slot_from_bbox(prim, bbox_cache, _src_up, _src_mpu)
        if pos_slot is None:
            skipped_no_bbox += 1
            continue
        posM, slot = pos_slot
        # Skip thin-slab placements — floor stripes, signage decals, etc.
        # They're 2D paint in the source, not 3D placements. Binding them
        # to 3D pack assets produces giant tape rolls overlapping pallets.
        if slot["heightM"] < DECAL_HEIGHT_THRESHOLD_M:
            skipped_decal += 1
            continue
        sem = _semantic_info_from_prim(prim)
        if archetype not in archetype_source_info and sem:
            archetype_source_info[archetype] = sem
        # Yaw comes from the xform rotation about the world up-axis, but
        # the translate comes from the bbox (above). Decoupling these
        # keeps collision/render aligned regardless of pivot offset.
        M = cache.GetLocalToWorldTransform(prim)
        yaw = _yaw_from_matrix(M)
        source_path = prim.GetPath().pathString
        path_id = source_path.replace("/", "_").lstrip("_")
        placements.append({
            "type": "placement", "id": path_id, "archetype": archetype,
            "slotM": slot,
            "transform": {"translateM": posM, "yawDeg": yaw},
            "sourcePath": source_path,
        })
    if skipped_decal:
        print(f"[absorb] skipped {skipped_decal} thin-slab decals "
              f"(heightM < {DECAL_HEIGHT_THRESHOLD_M} m)", flush=True)
    # Spatial dedup (same-archetype, near-coincident). Z-aware so that
    # two stacked instances of the same archetype (e.g. a 3 m rack on
    # the floor with a 2nd-story rack at z=3) aren't collapsed into one.
    DEDUP = 0.6
    by_arche: dict[str, list[tuple[float, float, float, float]]] = {}
    deduped = []
    for p in placements:
        a = p["archetype"]
        x, y, fz = p["transform"]["translateM"]
        h = p["slotM"]["heightM"]
        zc = fz + h / 2.0
        kept = by_arche.setdefault(a, [])
        dup = False
        for kx, ky, kzc, kh in kept:
            if (kx - x) ** 2 + (ky - y) ** 2 >= DEDUP ** 2:
                continue
            # XY-coincident — also require Z intervals to overlap to count as a dup.
            if abs(kzc - zc) <= (kh + h) / 2.0:
                dup = True
                break
        if dup:
            continue
        kept.append((x, y, zc, h))
        deduped.append(p)
    placements = deduped

    # OBB-SAT (yaw-aware, 3D) same-archetype collision dedup: two placements
    # of the same archetype whose OBBs overlap are treated as duplicates and
    # the second one is dropped. OBB SAT (vs axis-aligned bbox) catches the
    # skewed cases (e.g. yaw=180 against a wall, yaw=45 props) the old
    # AABB-only check missed or over-reported.

    def _obb(p):
        # absorb_pack's posM stores bbox-FLOOR Z (slot-anchor convention),
        # but spatial_index._obb_overlap's Z interval test assumes pos[2]
        # is bbox-CENTER. Convert here so two stacked-but-non-overlapping
        # placements (a rack on the floor + a rack at z=3 m on top of it)
        # don't get falsely flagged as duplicates of each other. Without
        # this, the second-story rack gets dropped before on-surface
        # parent detection can link it to the floor rack, leaving its
        # source-side companion (also stacked, different archetype)
        # orphaned and visibly floating.
        tx, ty, fz = p["transform"]["translateM"]
        h = p["slotM"]["heightM"]
        pos = (tx, ty, fz + h / 2.0)
        slot = (p["slotM"]["widthM"], p["slotM"]["depthM"], h)
        yaw = float(p["transform"].get("yawDeg", 0.0))
        return pos, slot, yaw

    same_drop = 0
    kept: list[tuple[str, tuple, tuple, float]] = []
    final2 = []
    for p in placements:
        a = p["archetype"]
        pos, slot, yaw = _obb(p)
        skip = False
        for ka, kpos, kslot, kyaw in kept:
            if ka != a:
                continue
            if _obb_overlap(pos, slot, yaw, kpos, kslot, kyaw):
                skip = True
                break
        if skip:
            same_drop += 1
            continue
        kept.append((a, pos, slot, yaw))
        final2.append(p)
    placements = final2
    print(f"[absorb] dropped {same_drop} same-archetype OBB overlappers", flush=True)
    if not placements:
        raise SystemExit("no asset-bearing placements found in source")

    # On-surface child detection. An absorbed source typically has many
    # placements sitting at z > 0 (a box on a rack shelf, a tape on a
    # crate, a sign on a counter). Their z value reflects the source's
    # actual geometry but does NOT survive a fresh asset binding — the
    # bound asset may have its top at a different height, leaving the
    # child floating or clipping.
    #
    # For each elevated placement, find its parent (a sibling placement
    # whose XY bbox contains this one's XY centre AND whose top surface
    # is near this one's floor z). Snap the child's z to (parent.posZ +
    # parent.heightM) and record parentPlacementId for downstream tools.
    # If no parent is found, the placement is left alone (treated as
    # floor-level or genuinely free-floating like a ceiling light).
    SURFACE_Z_TOLERANCE_M = 0.60   # parent.top vs child.z gap tolerance
    XY_MARGIN_M = 0.15             # accept child just outside parent's XY bbox
    ELEVATED_Z_THRESHOLD_M = 0.05

    def _xy_in_bbox(px: float, py: float, p: dict) -> bool:
        cx, cy, _ = p["transform"]["translateM"]
        w = p["slotM"]["widthM"]; d = p["slotM"]["depthM"]
        return ((cx - w/2 - XY_MARGIN_M) <= px <= (cx + w/2 + XY_MARGIN_M)
                and (cy - d/2 - XY_MARGIN_M) <= py <= (cy + d/2 + XY_MARGIN_M))

    on_surface_count = 0
    for child in placements:
        cx, cy, cz = child["transform"]["translateM"]
        if cz < ELEVATED_Z_THRESHOLD_M:
            continue
        # Find candidate parents
        best = None
        best_top_diff = SURFACE_Z_TOLERANCE_M
        for parent in placements:
            if parent is child:
                continue
            top_z = parent["transform"]["translateM"][2] + parent["slotM"]["heightM"]
            top_diff = abs(top_z - cz)
            if top_diff > SURFACE_Z_TOLERANCE_M:
                continue
            if not _xy_in_bbox(cx, cy, parent):
                continue
            # Prefer the parent whose top is closest to the child's z.
            if top_diff < best_top_diff:
                best = parent
                best_top_diff = top_diff
        if best is not None:
            new_z = best["transform"]["translateM"][2] + best["slotM"]["heightM"]
            child["transform"]["translateM"][2] = new_z
            child["parentPlacementId"] = best["id"]
            on_surface_count += 1
    if on_surface_count:
        print(f"[absorb] snapped {on_surface_count} elevated placements to "
              f"their detected parent surface (avoids floating after bind)",
              flush=True)

    # Drop orphaned floaters. An elevated placement (z > 0.2 m) with no
    # detected parent surface is "floating" — its actual support in the
    # source isn't an absorbed prim, so binding it will leave it hovering.
    # Keep items above CEILING_Z_THRESHOLD_M (ceiling lights, hanging
    # signage, overhead structure) — those are intentionally suspended.
    #
    # Note: this CEILING threshold is intentionally generous (4 m) because
    # source warehouse floors sit on top of a substrate slab; a "true"
    # ceiling object is typically >4 m above the detected floor, while a
    # 2nd-story rack-on-rack top can reach 3.5 m and is correctly parented
    # via the on-surface logic above (so it has parentPlacementId set and
    # bypasses this filter).
    CEILING_Z_THRESHOLD_M = 4.0
    FLOATER_Z_MIN_M = 0.20
    # Iteratively scrub stale parentPlacementId refs and drop orphans
    # until no more changes. A single pass isn't enough because an
    # intermediate parent (itself parent-less and elevated) gets dropped
    # AFTER its child was scrubbed, leaving the grandchild's reference
    # newly stale. Repeat until the placement set is stable.
    total_orphans = 0
    while True:
        surviving_ids = {p["id"] for p in placements}
        for p in placements:
            pid = p.get("parentPlacementId")
            if pid is not None and pid not in surviving_ids:
                p["parentPlacementId"] = None
        orphans = []
        keep = []
        for p in placements:
            z = p["transform"]["translateM"][2]
            if FLOATER_Z_MIN_M < z < CEILING_Z_THRESHOLD_M and not p.get("parentPlacementId"):
                orphans.append(p)
            else:
                keep.append(p)
        if not orphans:
            break
        total_orphans += len(orphans)
        placements = keep
    if total_orphans:
        print(f"[absorb] dropped {total_orphans} orphaned floating placements "
              f"({FLOATER_Z_MIN_M} m < z < {CEILING_Z_THRESHOLD_M} m, no parent surface; "
              f"transitive)", flush=True)
    if not placements:
        raise SystemExit("no placements left after orphan-floater drop")

    # No structural-vs-object filtering: capture every ref/payload-
    # bearing Xformable prim. Size heuristics misclassify large content
    # (e.g. a multi-rack assembly group can exceed 1000 m² and still be
    # content, not shell), and vocabulary filters were already rejected
    # by design. The LLM decides what's shell vs content at binding
    # time by inspecting placements and choosing whether to skip /
    # passthrough / bind each archetype — typically via
    # `query_template_placements` then `bind_archetype(...)` or
    # `remove(id)` per the placement's role.

    # Re-center to origin using the placements' own extents — with one
    # important wrinkle for Z. Real-world sources often have a few
    # structural-substrate prims at very low Z (building foundation,
    # facade panels extending below grade) and a big CLUSTER of placement
    # prims at some higher Z that represents the actual interior floor
    # (e.g. a warehouse with concrete floor at z=1.3 m above the absolute
    # origin). Naively subtracting min_z leaves the cluster floating
    # above the auto-shell's z=0 floor.
    #
    # Floor detection: bin Z values into 0.1 m buckets; the lowest bucket
    # holding at least 5% of the placements is the "real floor" — that's
    # the Z we subtract. Placements below the floor get shifted to z=0
    # (clamped); they're structural substrate the shell already handles.
    def _ext(p):
        x, y, z = p["transform"]["translateM"]
        w, d, h = p["slotM"]["widthM"], p["slotM"]["depthM"], p["slotM"]["heightM"]
        return x - w/2, x + w/2, y - d/2, y + d/2, z, z + h
    all_exts = [_ext(p) for p in placements]
    min_x = min(e[0] for e in all_exts)
    max_x = max(e[1] for e in all_exts)
    min_y = min(e[2] for e in all_exts)
    max_y = max(e[3] for e in all_exts)
    min_z_raw = min(e[4] for e in all_exts)
    max_z_raw = max(e[5] for e in all_exts)

    # Histogram Z floors at 0.1 m resolution. Use 25% of total
    # placements as the threshold so a few structural-substrate items
    # at very low Z don't get picked as the floor — the dominant low-Z
    # cluster (the actual interior floor where most content sits) does.
    BUCKET = 0.1
    DENSITY_THRESHOLD = max(2, int(len(placements) * 0.25))
    counts: dict[int, int] = {}
    for e in all_exts:
        b = int(round(e[4] / BUCKET))
        counts[b] = counts.get(b, 0) + 1
    sorted_buckets = sorted(counts.keys())
    floor_bucket = sorted_buckets[0]
    for b in sorted_buckets:
        if counts[b] >= DENSITY_THRESHOLD:
            floor_bucket = b
            break
    detected_floor = floor_bucket * BUCKET
    if detected_floor > min_z_raw + 0.2:
        print(f"[absorb] detected floor z={detected_floor:.2f} m "
              f"(min={min_z_raw:.2f} m); substrate placements below the "
              f"floor will be clamped to z=0", flush=True)
    min_z = detected_floor
    for p in placements:
        p["transform"]["translateM"][0] -= min_x
        p["transform"]["translateM"][1] -= min_y
        z = p["transform"]["translateM"][2] - min_z
        p["transform"]["translateM"][2] = max(z, 0.0)  # clamp substrate
    footprint = {
        "widthM": float(max(max_x - min_x + 2.0, 4.0)),
        "depthM": float(max(max_y - min_y + 2.0, 4.0)),
        "heightM": float(max(max_z_raw - min_z + 2.0, 4.0)),
    }

    # Build template_semantics — neutral placeholder per observed
    # archetype + any sourceDerived semantic info (UsdSemantics labels,
    # Kind, customData["simReady"]) the absorber harvested directly from
    # the source prim. Real preferredNear / avoidNear / surfaces come
    # from the *target pack's* pack.json (populated separately by
    # harvest_pack_semantics + LLM gap-fill, see SKILL.md "Enrich
    # pack.json with per-archetype semantics"), not from the absorber —
    # the absorber's job is layout capture, not opinions about how the
    # archetypes relate to each other.
    template_semantics = {}
    seen = Counter(p["archetype"] for p in placements)
    for arche in seen:
        sem = {"description": arche, "preferredNear": [], "avoidNear": []}
        if arche in archetype_source_info:
            sem["sourceDerived"] = archetype_source_info[arche]
        template_semantics[arche] = sem

    sketch = {
        "schemaVersion": 1,
        "seed": 42,
        "assetPack": target_pack or "",
        "absorbedFrom": str(src),
        "semantics": template_semantics,
        "templateZones": [],
        "defaults": {"marginFactor": 1.05, "defaultOnMiss": "skip", "pickStrategy": "best-fit"},
        "tree": {
            "type": "site", "id": "absorbed_site",
            "shell": {"boundsM": footprint},
            "children": [{
                "type": "zone", "id": "scene",
                "transform": {"translateM": [0.0, 0.0, 0.0], "yawDeg": 0.0},
                "boundsM": footprint,
                "children": placements,
            }],
        },
        "absorbStats": {
            "sourcePrimsScanned": sum(1 for _ in s.Traverse()),
            "placementsKept": len(placements),
            "skippedNoBbox": skipped_no_bbox,
            "footprintM": footprint,
            "byArchetype": dict(seen),
            "archetypesWithSourceSemantics": list(archetype_source_info),
        },
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(sketch, indent=2))
    print(f"[absorb] wrote {out}")
    print(f"  placements: {len(placements)}  unique archetypes: {len(seen)}")
    print(f"  shell: {footprint['widthM']:.2f} x {footprint['depthM']:.2f} x {footprint['heightM']:.2f} m")
    return sketch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="local path or http(s):// to a USD")
    ap.add_argument("--out", required=True, help="path to write absorbed sketch.json")
    ap.add_argument("--target-pack", help=(
        "optional asset pack root; recorded in the sketch metadata so "
        "downstream tools know which pack to consult for binding. Does "
        "NOT influence absorption — the absorbed sketch is the same "
        "regardless of which (or whether any) pack is supplied here."))
    args = ap.parse_args()
    absorb(args.source, args.out, args.target_pack)


if __name__ == "__main__":
    main()
