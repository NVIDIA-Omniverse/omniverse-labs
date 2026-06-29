#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure actual shelf/top/counter mesh positions in each asset USD and
write them as per-entry ``semantics.surfaces`` in pack.json. Replaces the
analytical ``h/3, 2h/3, h`` placeholders that harvest_pack_semantics emits
by default.

Why this matters: when boxes-on-shelves placements are computed from
declared (h/3 …) shelf Z's, the actual mesh face is often a few centimetres
off — the box ends up floating above the shelf or clipping into a shelf
below. The structural fix (per-entry resolution + bind-time refresh +
scaleM-aware Z, commit 05148ab) makes sure the right entry's declarations
are used, but if those declarations are approximations, the placements
still don't snap to the visible mesh. Measuring solves that.

How it works, per asset:

  1. Open the USD via ``UsdGeom.BBoxCache``.
  2. Find candidate prims:
       a. By name — anything matching the regex ``shelf|tier|level|deck|
          panel|plate|countertop|deskTop`` (case-insensitive).
       b. By geometric heuristic if (a) returns nothing — flat horizontal
          mesh prims (thin in the up-axis, large in the other two) at
          distinct levels along the up-axis.
  3. Group candidates by their top-of-bbox in the asset's up-axis
     (clustering with 5 cm tolerance).
  4. For each cluster: top-Z = cluster max; footprint = cluster max W × D.
  5. Emit ``[{label, localTopZ, footprintM}, ...]`` and write into the
     entry's ``semantics.surfaces``.

The script preserves anchors/affordances/preferredNear/avoidNear/
placementBias verbatim — only ``surfaces`` is rewritten.

Usage:

    measure_surfaces.py --pack /path/to/pack [--archetype NAME]+ \\
        [--label shelf] [--min-flatness 5.0] [--write] [--verbose]

Without ``--write`` the script reports what it would change but doesn't
modify pack.json. With ``--write`` it persists the new surfaces. Pass
``--archetype Racks`` (repeatable) to restrict to specific archetypes;
default is every archetype in the pack.

For manifest packs, the script recurses into each sub-pack.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from pxr import Usd, UsdGeom, Gf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack_loader import is_manifest


_NAME_PATTERN = re.compile(
    r"shelf|tier|level|deck|panel|plate|countertop|desktop|"
    r"tabletop|worksurface|workTop",
    re.IGNORECASE,
)


def _up_axis_index(stage: Usd.Stage) -> int:
    up = str(UsdGeom.GetStageUpAxis(stage) or "Z").upper()
    return 1 if up == "Y" else 2  # 0=x, 1=y, 2=z


def _mesh_world_range(bbox_cache: UsdGeom.BBoxCache, prim: Usd.Prim):
    try:
        bbox = bbox_cache.ComputeWorldBound(prim)
        return bbox.ComputeAlignedRange()
    except Exception:
        return None


def _by_name_candidates(stage: Usd.Stage) -> list[Usd.Prim]:
    out: list[Usd.Prim] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Imageable):
            continue
        name = prim.GetName()
        if _NAME_PATTERN.search(name):
            out.append(prim)
    return out


def _geometric_candidates(stage: Usd.Stage, bbox_cache: UsdGeom.BBoxCache,
                          up_idx: int, min_flatness_ratio: float,
                          mpu: float) -> list[Usd.Prim]:
    """Flat horizontal meshes whose top-axis extent is ≥ min_flatness_ratio
    smaller than each of the other two extents."""
    out = []
    other = [a for a in (0, 1, 2) if a != up_idx]
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        rng = _mesh_world_range(bbox_cache, prim)
        if rng is None or rng.IsEmpty():
            continue
        sz = rng.GetSize()
        # Convert to meters for stable thresholds
        sz_m = (float(sz[0]) * mpu, float(sz[1]) * mpu, float(sz[2]) * mpu)
        up_extent = sz_m[up_idx]
        if up_extent <= 0:
            continue
        plane_ws = [sz_m[a] for a in other]
        if min(plane_ws) <= 0:
            continue
        if min(plane_ws) / up_extent < min_flatness_ratio:
            continue
        if max(plane_ws) < 0.3:  # too small to be a useful shelf
            continue
        out.append(prim)
    return out


def _cluster_surfaces(prims: list[Usd.Prim],
                      bbox_cache: UsdGeom.BBoxCache,
                      up_idx: int, mpu: float,
                      cluster_tolerance_m: float = 0.05
                      ) -> list[dict]:
    """Group prims by their top-of-bbox in the up axis; emit one shelf per
    cluster with top-Z and footprint = max W × max D across the cluster."""
    levels: list[dict] = []
    other = [a for a in (0, 1, 2) if a != up_idx]
    for prim in prims:
        rng = _mesh_world_range(bbox_cache, prim)
        if rng is None or rng.IsEmpty():
            continue
        sz = rng.GetSize()
        mx = rng.GetMax()
        sz_m = (float(sz[0]) * mpu, float(sz[1]) * mpu, float(sz[2]) * mpu)
        top_m = float(mx[up_idx]) * mpu
        fp = (sz_m[other[0]], sz_m[other[1]])
        # Find an existing cluster within tolerance
        merged = False
        for lvl in levels:
            if abs(lvl["top_m"] - top_m) <= cluster_tolerance_m:
                lvl["top_m"] = max(lvl["top_m"], top_m)
                lvl["fp_w"] = max(lvl["fp_w"], fp[0])
                lvl["fp_d"] = max(lvl["fp_d"], fp[1])
                lvl["count"] += 1
                merged = True
                break
        if not merged:
            levels.append({"top_m": top_m, "fp_w": fp[0], "fp_d": fp[1], "count": 1})
    levels.sort(key=lambda l: l["top_m"])
    return levels


def _measure_faces_in_stage(stage: Usd.Stage, up_idx: int, mpu: float,
                             min_face_area_m2: float = 0.0005,
                             cluster_tolerance_m: float = 0.05,
                             up_normal_threshold: float = 0.7) -> list[dict]:
    """Mesh-face level analysis. For each horizontal triangle/quad in any
    mesh prim, take its up-axis position and area. Cluster by up-Z.

    Required for assets where the surfaces are NOT exposed as separate
    prims — e.g. an asset that ships as a single monolithic mesh whose
    internal horizontal faces ARE the shelves/platforms/tabletops.
    Prim-level bbox-clustering finds nothing here; this method walks
    vertices and classifies faces.
    """
    horizontal: list[tuple[float, float, float, float]] = []  # (z, area, x_extent, y_extent)
    other = [a for a in (0, 1, 2) if a != up_idx]
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        fvc = mesh.GetFaceVertexCountsAttr().Get()
        fvi = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not fvc or not fvi:
            continue
        local_to_world = xform_cache.GetLocalToWorldTransform(prim)
        # Transform every point to world once (cheaper than per-face).
        pts_w = [local_to_world.Transform(Gf.Vec3d(p)) for p in pts]
        cursor = 0
        for cnt in fvc:
            if cnt < 3:
                cursor += cnt
                continue
            try:
                v0 = pts_w[fvi[cursor]]
                v1 = pts_w[fvi[cursor + 1]]
                v2 = pts_w[fvi[cursor + 2]]
            except IndexError:
                cursor += cnt
                continue
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = Gf.Cross(edge1, edge2)
            n_len = normal.GetLength()
            if n_len < 1e-9:
                cursor += cnt
                continue
            normal = normal / n_len
            # Horizontal = normal points along +up (within 25°). Faces with
            # normal pointing DOWN (-up) we skip — those are the underside
            # of shelves and would double-count.
            if normal[up_idx] < up_normal_threshold:
                cursor += cnt
                continue
            # Triangle area; for quads we approximate 2x triangle area.
            tri_area = 0.5 * n_len * mpu * mpu
            face_area = tri_area * 2 if cnt == 4 else tri_area
            if face_area < min_face_area_m2:
                cursor += cnt
                continue
            z_m = float(v0[up_idx]) * mpu
            # Use the face's own footprint axis-aligned bounds for fp_w/fp_d
            xs = [float(pts_w[fvi[cursor + i]][other[0]]) * mpu
                  for i in range(cnt)]
            ys = [float(pts_w[fvi[cursor + i]][other[1]]) * mpu
                  for i in range(cnt)]
            horizontal.append((z_m, face_area, max(xs) - min(xs),
                                max(ys) - min(ys)))
            cursor += cnt

    if not horizontal:
        return []
    horizontal.sort(key=lambda t: t[0])
    clusters: list[dict] = []
    for z, area, ex, dy in horizontal:
        if clusters and abs(z - clusters[-1]["top_m"]) <= cluster_tolerance_m:
            clusters[-1]["top_m"] = max(clusters[-1]["top_m"], z)
            clusters[-1]["fp_w"] = max(clusters[-1]["fp_w"], ex)
            clusters[-1]["fp_d"] = max(clusters[-1]["fp_d"], dy)
            clusters[-1]["count"] += 1
            clusters[-1]["total_area"] += area
        else:
            clusters.append({"top_m": z, "fp_w": ex, "fp_d": dy,
                             "count": 1, "total_area": area})
    # Drop clusters with trivial total area — these are tiny details, not
    # shelves we'd want to drop boxes on.
    clusters = [c for c in clusters if c["total_area"] >= 0.5]
    # Drop the very-bottom cluster if it looks like a floor plate (z < 10 cm
    # AND we have other higher clusters). The floor of a rack frame isn't
    # a useful "shelf" for box placement.
    if len(clusters) > 1 and clusters[0]["top_m"] < 0.1:
        clusters = clusters[1:]
    # Cap to the 5 largest-area clusters (per real rack norms — even a
    # 5-shelf bulk rack rarely has more than 5 distinct levels). This
    # discards support-beam clusters that survived the 0.5 m² area
    # filter on long but thin cross-bracing.
    if len(clusters) > 5:
        clusters = sorted(clusters, key=lambda c: -c["total_area"])[:5]
        clusters.sort(key=lambda c: c["top_m"])
    return clusters


def _open_with_payload_fallback(asset_path: Path) -> Usd.Stage | None:
    """Open the asset; if it yields no meshes (e.g. its payload reference
    is broken — Windows-backslash refs in cross-platform packs are a
    common cause), fall back to a sibling `payloads/geometries.usd` next
    to the asset.
    Returns None if neither path produced a usable stage.
    """
    primary = Usd.Stage.Open(str(asset_path))
    if primary:
        has_mesh = any(p.IsA(UsdGeom.Mesh) for p in primary.Traverse())
        if has_mesh:
            return primary
    fallback = asset_path.parent / "payloads" / "geometries.usd"
    if fallback.exists():
        stage = Usd.Stage.Open(str(fallback))
        if stage:
            return stage
    return primary  # may have no meshes, but caller will report 0


def measure_one_asset(asset_path: Path, label: str = "shelf",
                      min_flatness: float = 5.0, verbose: bool = False
                      ) -> tuple[list[dict], dict]:
    """Returns (surfaces, debug). Tries three methods in order:
      1. Prim-name match (`shelf|tier|level|deck|...`)
      2. Prim-level geometric heuristic (flat horizontal mesh prims)
      3. Mesh-face analysis (walks vertices; required for monolithic
         meshes whose internal faces ARE the shelves/platforms).
    First method that yields ≥ 1 cluster wins.
    """
    stage = _open_with_payload_fallback(asset_path)
    if not stage:
        return [], {"error": "could not open stage"}
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    up_idx = _up_axis_index(stage)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])

    src = "by-name"
    levels = _cluster_surfaces(_by_name_candidates(stage), bbox_cache, up_idx, mpu)
    if not levels:
        src = "geometric-prim"
        levels = _cluster_surfaces(
            _geometric_candidates(stage, bbox_cache, up_idx, min_flatness, mpu),
            bbox_cache, up_idx, mpu,
        )
    if not levels:
        src = "mesh-faces"
        levels = _measure_faces_in_stage(stage, up_idx, mpu)

    # Compute the asset's overall bbox so we can detect the "all clusters
    # near the bottom" pathology — common for drums, buckets, and similar
    # cylindrical assets where the top is a curved cap (no flat horizontal
    # face) but the base is flat. Without this guard the only detected
    # cluster is the BASE, and any place_on lands the child near z=0
    # instead of on top of the asset.
    asset_bbox = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot())
    arng = asset_bbox.ComputeAlignedRange() if not asset_bbox.GetRange().IsEmpty() else None
    asset_top_m = float(arng.GetMax()[up_idx]) * mpu if arng else 0.0
    asset_bot_m = float(arng.GetMin()[up_idx]) * mpu if arng else 0.0
    asset_h_m = max(asset_top_m - asset_bot_m, 0.0)

    bbox_fallback_used = False
    if asset_h_m > 0.05 and levels:
        highest = max(l["top_m"] for l in levels)
        # If every detected cluster sits in the lower 50% of the asset's
        # vertical extent, the actual top mesh has no flat horizontal face
        # we could detect — synthesize one at the asset's bbox top with
        # the asset's full XY footprint, so children land on the real top.
        if (highest - asset_bot_m) < 0.5 * asset_h_m:
            xy_axes = [a for a in (0, 1, 2) if a != up_idx]
            fp_w = float(arng.GetMax()[xy_axes[0]] - arng.GetMin()[xy_axes[0]]) * mpu
            fp_d = float(arng.GetMax()[xy_axes[1]] - arng.GetMin()[xy_axes[1]]) * mpu
            levels.append({"top_m": asset_top_m, "fp_w": fp_w, "fp_d": fp_d,
                           "count": 0, "total_area": fp_w * fp_d,
                           "_synthetic": True})
            bbox_fallback_used = True

    if verbose:
        print(f"    {asset_path.name}: src={src} clusters={len(levels)}"
              f"{' +bbox-fallback' if bbox_fallback_used else ''}",
              file=sys.stderr)

    surfaces = []
    for lvl in levels:
        s = {
            "label": label,
            "localTopZ": round(lvl["top_m"], 4),
            "footprintM": [round(lvl["fp_w"], 4), round(lvl["fp_d"], 4)],
            "_measuredFromMeshes": lvl.get("count", 0),
        }
        if lvl.get("_synthetic"):
            s["_source"] = "bbox_top_fallback"
        surfaces.append(s)
    return surfaces, {"upAxisIndex": up_idx, "metersPerUnit": mpu,
                       "sourcedFrom": src,
                       "clustersFound": len(levels),
                       "bboxFallbackUsed": bbox_fallback_used}


def _discover_label(entries: list, fallback: str = "shelf") -> str:
    """If the archetype already has surfaces declared in pack.json, use
    the first one's label (so subsequent measurements preserve the
    existing label even when the user doesn't pass --label). Falls back
    to `fallback` (default 'shelf') for archetypes with no prior
    surfaces.
    """
    if not entries:
        return fallback
    first = entries[0] if isinstance(entries[0], dict) else None
    sem = (first or {}).get("semantics") or {}
    surfs = sem.get("surfaces") or []
    if surfs and isinstance(surfs[0], dict) and surfs[0].get("label"):
        return str(surfs[0]["label"])
    return fallback


def measure_pack(pack_root: Path, archetypes: list[str] | None,
                 label: str | None, min_flatness: float, verbose: bool) -> dict:
    pack_path = pack_root / "pack.json"
    raw = json.loads(pack_path.read_text())

    if is_manifest(raw):
        # Recurse into each sub-pack
        totals = {"subpacks": [], "totalArchetypesMeasured": 0,
                  "totalAssetsScanned": 0, "totalSurfacesEmitted": 0}
        for entry in raw.get("subpacks", []):
            sub_root = pack_root / entry["path"]
            print(f"measure sub-pack {entry['theme']!r} @ {sub_root}",
                  file=sys.stderr)
            sub = measure_pack(sub_root, archetypes, label, min_flatness, verbose)
            sub["theme"] = entry["theme"]
            totals["subpacks"].append(sub)
            totals["totalArchetypesMeasured"] += sub.get("archetypesMeasured", 0)
            totals["totalAssetsScanned"] += sub.get("assetsScanned", 0)
            totals["totalSurfacesEmitted"] += sub.get("surfacesEmitted", 0)
        return totals

    out_archetypes = {}
    asset_count = 0
    surf_count = 0
    arch_count = 0
    for arch_name, entries in (raw.get("archetypes") or {}).items():
        if archetypes and arch_name not in archetypes:
            continue
        arch_count += 1
        # If the caller didn't pin a label, use the archetype's existing
        # one from pack.json (e.g. one archetype declares "platform",
        # another "table_top", another "conveyor_belt"). Keeps
        # `measure_surfaces --pack X` working across the whole pack
        # without per-archetype --label invocations.
        resolved_label = label if label is not None else _discover_label(entries)
        for entry in entries:
            asset_count += 1
            asset_path = (pack_root / entry["path"]).resolve()
            if not asset_path.exists():
                continue
            try:
                surfaces, debug = measure_one_asset(asset_path, resolved_label,
                                                     min_flatness, verbose)
            except Exception as e:
                print(f"  warning: {entry['path']}: {e}", file=sys.stderr)
                continue
            sem = entry.setdefault("semantics", {})
            if surfaces:
                sem["surfaces"] = surfaces
                surf_count += len(surfaces)
                out_archetypes.setdefault(arch_name, []).append({
                    "path": entry["path"], **debug,
                    "surfaceTopZs": [s["localTopZ"] for s in surfaces],
                })
    return {
        "pack": str(pack_path),
        "archetypesMeasured": arch_count,
        "assetsScanned": asset_count,
        "surfacesEmitted": surf_count,
        "raw": raw,
        "report": out_archetypes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, help="Pack root (flat or manifest)")
    ap.add_argument("--archetype", action="append", default=[],
                    help="Restrict to these archetypes (repeatable); default all")
    ap.add_argument("--label", default=None,
                    help="Surface label to write. Default: auto-discover from "
                         "each archetype's existing semantics.surfaces[0].label "
                         "in pack.json (each archetype keeps whatever label "
                         "it already declared). Pass an explicit value to "
                         "override or to seed a label for archetypes that "
                         "have no prior surfaces (default fallback in that "
                         "case is 'shelf').")
    ap.add_argument("--min-flatness", type=float, default=5.0,
                    help="Geometric heuristic threshold: min(plane W/D) / "
                         "up-axis-extent. Higher = stricter 'flat' definition "
                         "(default 5).")
    ap.add_argument("--write", action="store_true",
                    help="Persist results into pack.json. Without this, the "
                         "script reports what it WOULD change.")
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-asset measurement debug to stderr")
    ap.add_argument("--recursive", action="store_true",
                    help="Walk the pack root tree for every `pack.json` and "
                         "measure each as a separate pack. Use when an asset "
                         "library nests independent packs at multiple levels "
                         "(e.g. <root>/pack.json AND <root>/<theme>/pack.json). "
                         "Skips manifest pack.jsons (they delegate to their "
                         "sub-pack entries, which the walk picks up separately).")
    args = ap.parse_args()

    pack_root = Path(args.pack).expanduser().resolve()

    if args.recursive:
        pack_paths = sorted(pack_root.rglob("pack.json"))
        if not pack_paths:
            sys.exit(f"--recursive: no pack.json files found under {pack_root}")
        total_packs = 0
        total_assets = 0
        total_surfaces = 0
        for pack_path in pack_paths:
            try:
                raw_check = json.loads(pack_path.read_text())
            except Exception as e:
                print(f"skip {pack_path}: unreadable ({e})", file=sys.stderr)
                continue
            if raw_check.get("_isManifest"):
                # Manifest just lists sub-packs by path; the walk already
                # picked them up as their own entries.
                if args.verbose:
                    print(f"skip manifest {pack_path}", file=sys.stderr)
                continue
            sub_root = pack_path.parent
            print(f"\n=== {sub_root} ===", file=sys.stderr)
            sub_result = measure_pack(sub_root, args.archetype or None,
                                       args.label, args.min_flatness,
                                       args.verbose)
            if "subpacks" in sub_result:
                # Shouldn't happen because manifests are skipped above,
                # but be defensive.
                continue
            total_packs += 1
            total_assets += sub_result.get("assetsScanned", 0)
            total_surfaces += sub_result.get("surfacesEmitted", 0)
            print(f"  archetypesMeasured={sub_result.get('archetypesMeasured', 0)}"
                  f"  assetsScanned={sub_result.get('assetsScanned', 0)}"
                  f"  surfacesEmitted={sub_result.get('surfacesEmitted', 0)}",
                  file=sys.stderr)
            if args.write and sub_result.get("raw"):
                pack_path.write_text(json.dumps(sub_result["raw"], indent=2))
                print(f"  wrote {pack_path}", file=sys.stderr)
        print(f"\n[recursive summary] packs={total_packs}  "
              f"assets={total_assets}  surfaces={total_surfaces}",
              file=sys.stderr)
        if not args.write:
            print("(dry-run; pass --write to persist)", file=sys.stderr)
        return 0

    result = measure_pack(pack_root, args.archetype or None,
                           args.label, args.min_flatness, args.verbose)

    if "subpacks" in result:
        for sub in result["subpacks"]:
            print(f"\n[{sub.get('theme', '?')}]  "
                  f"archetypes={sub.get('archetypesMeasured', 0)}  "
                  f"assets={sub.get('assetsScanned', 0)}  "
                  f"surfaces={sub.get('surfacesEmitted', 0)}",
                  file=sys.stderr)
            for arch, items in (sub.get("report") or {}).items():
                print(f"  {arch}:", file=sys.stderr)
                for it in items:
                    print(f"    {it['path']}  topZs={it['surfaceTopZs']}  "
                          f"(src={it['sourcedFrom']})", file=sys.stderr)
            if args.write:
                sub_pack = pack_root / [s for s in
                    json.loads((pack_root / "pack.json").read_text())
                    .get("subpacks", []) if s.get("theme") == sub.get("theme")
                ][0]["path"] / "pack.json"
                sub_pack.write_text(json.dumps(sub.get("raw", {}), indent=2))
                print(f"  wrote {sub_pack}", file=sys.stderr)
    else:
        print(f"\narchetypesMeasured={result['archetypesMeasured']}  "
              f"assetsScanned={result['assetsScanned']}  "
              f"surfacesEmitted={result['surfacesEmitted']}", file=sys.stderr)
        for arch, items in (result.get("report") or {}).items():
            print(f"\n{arch}:", file=sys.stderr)
            for it in items:
                print(f"  {it['path']}", file=sys.stderr)
                print(f"    topZs={it['surfaceTopZs']}  "
                      f"src={it['sourcedFrom']}  "
                      f"upAxisIndex={it['upAxisIndex']}", file=sys.stderr)
        if args.write:
            pack_path = pack_root / "pack.json"
            pack_path.write_text(json.dumps(result.get("raw", {}), indent=2))
            print(f"\nwrote {pack_path}", file=sys.stderr)
        else:
            print("\n(dry-run; pass --write to persist)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
