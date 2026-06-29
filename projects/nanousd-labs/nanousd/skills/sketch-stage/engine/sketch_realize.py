# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sketch realizer — prototype.

Reads a sketch.json (scene tree of zones + placements with explicit slot
bboxes), a pack.json with per-archetype asset candidate lists, computes each
candidate's actual bbox once, walks the sketch tree composing transforms,
picks a best-fit candidate per slot under a margin factor, authors a USD
stage referencing the picked assets, and writes a manifest with unfilled
slots reported.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom, Gf, Vt


def _semantic_label(archetype: str) -> str:
    return archetype or "equipment"


def _create_crate_layer() -> Sdf.Layer:
    return Sdf.Layer.CreateAnonymous(".usdc")


@dataclass
class Candidate:
    archetype: str
    rel_path: str
    abs_path: str
    bbox: tuple[float, float, float]  # (W, D, H) in meters
    bbox_min: tuple[float, float, float]  # asset-local min corner
    bbox_max: tuple[float, float, float]  # asset-local max corner


def _load_pack(asset_pack_dir: Path) -> tuple[dict[str, list[Candidate]], dict]:
    pack = json.loads((asset_pack_dir / "pack.json").read_text())
    mpu = float(pack.get("metersPerUnit", 1.0))
    up_axis = str(pack.get("upAxis", "Z")).upper()
    aliases: dict[str, str] = pack.get("archetypeAliases", {}) or {}
    # pack.json's "path" entries are joined with asset_pack_dir below; reject any
    # that resolve outside the pack root to close the path-traversal surface
    # (pythonsecurity:S2083 — e.g. a malicious pack.json with "../../etc/passwd").
    pack_root = asset_pack_dir.resolve()
    out: dict[str, list[Candidate]] = {}
    for arche, entries in pack.get("archetypes", {}).items():
        cands: list[Candidate] = []
        for e in entries:
            rel = e["path"] if isinstance(e, dict) else e
            absp = (asset_pack_dir / rel).resolve()
            if not absp.is_relative_to(pack_root):
                continue
            if not absp.exists():
                continue
            measured = _measure_bbox(str(absp))
            if measured is None:
                continue
            sz, mn, mx = measured
            # Per-asset upAxis/mpu is authoritative (per _asset_orientation).
            # Pack-level metadata is only a fallback when the asset's stage
            # doesn't declare its own — real asset packs often mix Y-up and
            # Z-up assets even when pack.json majority-votes one upAxis.
            asset_up, asset_mpu = _asset_orientation(str(absp))
            asset_up = asset_up if asset_up in ("Y", "Z") else up_axis
            asset_mpu = asset_mpu if asset_mpu > 0 else mpu
            if asset_up == "Y":
                sz = (sz[0], sz[2], sz[1])
                mn = (mn[0], mn[2], mn[1])
                mx = (mx[0], mx[2], mx[1])
            sz = (sz[0] * asset_mpu, sz[1] * asset_mpu, sz[2] * asset_mpu)
            mn = (mn[0] * asset_mpu, mn[1] * asset_mpu, mn[2] * asset_mpu)
            mx = (mx[0] * asset_mpu, mx[1] * asset_mpu, mx[2] * asset_mpu)
            cands.append(Candidate(arche, rel, str(absp), sz, mn, mx))
        out[arche] = cands
    # Apply archetype aliases: when the sketch asks for X but the pack only
    # has Y, alias `X -> Y` lets the matcher use Y's candidates for X-slots.
    for src, dst in aliases.items():
        if src not in out and dst in out:
            out[src] = out[dst]
    return out, {
        "metersPerUnit": mpu,
        "upAxis": up_axis,
        "rootStage": pack.get("rootStage"),
        "rootStageBboxM": pack.get("rootStageBboxM"),
        "name": pack.get("name") or asset_pack_dir.name,
        "archetypeAliases": aliases,
    }


_bbox_cache: dict[str, tuple | None] = {}

def _measure_bbox(usd_path: str):
    """Open the asset stage once per unique path and cache the result.
    Without this, a 17k-placement realize calls Stage.Open ~17k times for
    just 8 distinct assets — that's the dominant cost.
    """
    cached = _bbox_cache.get(usd_path)
    if cached is not None or usd_path in _bbox_cache:
        return cached
    s = Usd.Stage.Open(usd_path)
    if s is None:
        _bbox_cache[usd_path] = None
        return None
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                              [UsdGeom.Tokens.default_],
                              useExtentsHint=True)
    root = s.GetDefaultPrim() or next(iter(s.Traverse()), None)
    if root is None:
        _bbox_cache[usd_path] = None
        return None
    bbox = cache.ComputeWorldBound(root).ComputeAlignedBox()
    sz = bbox.GetSize()
    if any(math.isinf(v) or math.isnan(v) for v in sz):
        _bbox_cache[usd_path] = None
        return None
    mn = bbox.GetMin()
    mx = bbox.GetMax()
    result = (
        (float(sz[0]), float(sz[1]), float(sz[2])),
        (float(mn[0]), float(mn[1]), float(mn[2])),
        (float(mx[0]), float(mx[1]), float(mx[2])),
    )
    _bbox_cache[usd_path] = result
    return result


_orientation_cache: dict[str, tuple[str, float]] = {}

def _asset_orientation(usd_path: str) -> tuple[str, float]:
    """Read the asset stage's own upAxis + metersPerUnit. Authoritative —
    pack-level metadata is a hint and can disagree with reality (a pack
    may declare Y-up while its assets are actually authored Z-up cm)."""
    cached = _orientation_cache.get(usd_path)
    if cached is not None:
        return cached
    s = Usd.Stage.Open(usd_path)
    if s is None:
        result = ("Z", 1.0)
    else:
        up = str(UsdGeom.GetStageUpAxis(s) or "Z").upper()
        mpu = float(UsdGeom.GetStageMetersPerUnit(s) or 1.0)
        result = (up, mpu)
    _orientation_cache[usd_path] = result
    return result


def _fits(bbox: tuple[float, float, float],
          slot: tuple[float, float, float],
          margin: float) -> bool:
    return all(b <= s * margin for b, s in zip(bbox, slot))


def _waste(bbox: tuple[float, float, float],
           slot: tuple[float, float, float]) -> float:
    return sum(max(0.0, s - b) for b, s in zip(bbox, slot))


def _pick(arche: str,
          slot: tuple[float, float, float],
          margin: float,
          candidates: dict[str, list[Candidate]],
          strategy: str,
          rng) -> Candidate | None:
    fits = [c for c in candidates.get(arche, []) if _fits(c.bbox, slot, margin)]
    if not fits:
        return None
    if strategy == "random":
        return rng.choice(fits)
    return min(fits, key=lambda c: _waste(c.bbox, slot))


def _compose(parent_M: Gf.Matrix4d, transform: dict) -> Gf.Matrix4d:
    t = transform.get("translateM", [0.0, 0.0, 0.0])
    yaw = float(transform.get("yawDeg", 0.0))
    M = Gf.Matrix4d().SetIdentity()
    M = M * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw))
    M.SetTranslateOnly(Gf.Vec3d(*t))
    return parent_M * M


def _auto_fit_slots(sketch: dict, candidates: dict, margin: float) -> int:
    """For each archetype, find the max bbox across candidates; resize every
    placement of that archetype so the slot fits the largest candidate × margin.
    Returns number of slots resized."""
    max_by_arche: dict[str, tuple[float, float, float]] = {}
    for arche, cands in candidates.items():
        if not cands:
            continue
        max_by_arche[arche] = (
            max(c.bbox[0] for c in cands),
            max(c.bbox[1] for c in cands),
            max(c.bbox[2] for c in cands),
        )
    resized = 0
    def walk_resize(node):
        nonlocal resized
        if node.get("type") == "placement":
            arche = node.get("archetype")
            if arche in max_by_arche:
                w, d, h = max_by_arche[arche]
                node["slotM"] = {
                    "widthM": float(w * margin),
                    "depthM": float(d * margin),
                    "heightM": float(h * margin),
                }
                resized += 1
        for child in node.get("children", []):
            walk_resize(child)
    walk_resize(sketch.get("tree", {}))
    return resized


def _emit_event(events_paths: list[Path], rec: dict) -> None:
    line = json.dumps(rec) + "\n"
    for p in events_paths:
        try:
            with p.open("a") as f:
                f.write(line)
        except Exception:
            pass


def _author_arc_layer(path: Path, arc_kind: str, arc_path: str,
                      *, up_axis: str = "Z", meters_per_unit: float = 1.0) -> None:
    layer = _create_crate_layer()
    layer.defaultPrim = "World"
    try:
        layer.SetField(Sdf.Path.absoluteRootPath, "metersPerUnit", meters_per_unit)
        layer.SetField(Sdf.Path.absoluteRootPath, "upAxis", up_axis)
    except Exception:
        pass
    world = Sdf.CreatePrimInLayer(layer, Sdf.Path("/World"))
    world.specifier = Sdf.SpecifierDef
    world.typeName = "Xform"
    if arc_kind == "payload":
        world.payloadList.prependedItems.append(Sdf.Payload(arc_path))
    elif arc_kind == "reference":
        world.referenceList.prependedItems.append(Sdf.Reference(arc_path))
    else:
        raise ValueError(f"unsupported wrapper arc kind: {arc_kind}")
    path.parent.mkdir(parents=True, exist_ok=True)
    layer.Export(str(path))


def _export_with_structure(layer: Sdf.Layer, out_dir: Path, structure: str,
                           *, up_axis: str, meters_per_unit: float,
                           reference_chain_depth: int) -> dict:
    root_path = out_dir / "root.usd"
    if structure == "direct":
        layer.Export(str(root_path))
        return {
            "rootPath": str(root_path),
            "contentLayer": str(root_path),
            "wrapperLayers": [],
        }
    if structure == "single_scene_payload":
        content_path = out_dir / "payloads" / "scene.usd"
        content_path.parent.mkdir(parents=True, exist_ok=True)
        layer.Export(str(content_path))
        # Prefix the relative arc with `./` so spec-strict resolvers
        # (AOUSD §9.4.1) treat it as anchored to the authoring layer
        # rather than as a search-path identifier. nanoUSD's default
        # resolver follows the spec strictly; OpenUSD's also accepts
        # the `./` form.
        _author_arc_layer(
            root_path,
            "payload",
            "./payloads/scene.usd",
            up_axis=up_axis,
            meters_per_unit=meters_per_unit,
        )
        return {
            "rootPath": str(root_path),
            "contentLayer": str(content_path),
            "wrapperLayers": [str(root_path)],
        }
    if structure == "deep_reference_chain":
        depth = max(1, int(reference_chain_depth))
        references_dir = out_dir / "references"
        content_path = references_dir / "scene.usd"
        references_dir.mkdir(parents=True, exist_ok=True)
        layer.Export(str(content_path))
        wrapper_layers: list[str] = []
        # next_arc tracks the arc target authored INSIDE the current
        # wrapper layer; siblings live in the same `references/` dir so
        # the relative path is a bare filename. Prefix with `./` for
        # spec-strict anchored-relative resolution.
        next_arc = "./scene.usd"
        for idx in range(depth, 0, -1):
            chain_path = references_dir / f"chain_{idx:03d}.usd"
            _author_arc_layer(
                chain_path,
                "reference",
                next_arc,
                up_axis=up_axis,
                meters_per_unit=meters_per_unit,
            )
            wrapper_layers.append(str(chain_path))
            next_arc = f"./chain_{idx:03d}.usd"
        _author_arc_layer(
            root_path,
            "reference",
            "./references/chain_001.usd",
            up_axis=up_axis,
            meters_per_unit=meters_per_unit,
        )
        wrapper_layers.reverse()
        return {
            "rootPath": str(root_path),
            "contentLayer": str(content_path),
            "wrapperLayers": [str(root_path), *wrapper_layers],
        }
    raise ValueError(f"compositionStrategy.structure: {structure!r}")


def _realize_rootcell_passthrough(sketch: dict, asset_pack: Path, pack_meta: dict,
                                   out_dir: Path) -> dict:
    """Author N tiled references to the pack's rootStage USD."""
    import os
    import time
    target = int(sketch.get("targetComposedPrimCount", 1000))
    cell_prims = int(pack_meta.get("rootStageBboxM", {}).get("estimatedPrimCount", 17000))
    n_tiles = max(1, (target + cell_prims - 1) // cell_prims)
    bbox = pack_meta.get("rootStageBboxM", {}) or {}
    cell_w = float(bbox.get("widthM", 80.0))
    cell_d = float(bbox.get("depthM", 60.0))
    gap = 5.0
    cols = max(1, int(round(n_tiles ** 0.5)))
    rows = (n_tiles + cols - 1) // cols
    root_stage_path = (asset_pack / pack_meta["rootStage"]).resolve()

    layer = _create_crate_layer()
    layer.defaultPrim = "World"
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # Determine events paths so the live viz sees passthrough placements too.
    events_paths: list[Path] = []
    out_events = out_dir / "events.jsonl"
    out_events.parent.mkdir(parents=True, exist_ok=True)
    out_events.write_text("")  # truncate per run
    events_paths.append(out_events)
    shared = os.environ.get("SKETCH_LIVE_EVENTS")
    if shared:
        sp = Path(shared)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text("")  # also truncate the shared live log
        events_paths.append(sp)
    t0 = time.monotonic()
    _emit_event(events_paths, {
        "t": 0.0, "type": "session_open",
        "anchorSketch": None, "assetPack": str(asset_pack),
        "anchorPlacementCount": 0, "zoneCount": 0,
        "archetypes": ["<rootcell>"], "passthrough": True,
    })

    # Optional: read an absorbed template so the viz can show the
    # per-archetype placements WITHIN each cell. The realized USD stays
    # cell-level (one reference to rootStage); this is purely for the
    # viz layer.
    template_placements = []
    abs_template = sketch.get("absorbedTemplate")
    if abs_template and Path(abs_template).exists():
        try:
            tpl = json.loads(Path(abs_template).read_text())
            def collect(node, out):
                if node.get("type") == "placement":
                    out.append(node)
                for ch in node.get("children", []):
                    collect(ch, out)
            collect(tpl.get("tree", {}), template_placements)
        except Exception:
            template_placements = []

    cells = 0
    for r in range(rows):
        for c in range(cols):
            if cells >= n_tiles:
                break
            tile = UsdGeom.Xform.Define(stage, f"/World/cell_r{r}_c{c}")
            tx, ty = c * (cell_w + gap), r * (cell_d + gap)
            UsdGeom.Xformable(tile).AddTranslateOp().Set(Gf.Vec3d(tx, ty, 0.0))
            tile.GetPrim().GetReferences().AddReference(str(root_stage_path))
            _emit_event(events_paths, {
                "t": round(time.monotonic() - t0, 4),
                "type": "placed",
                "id": f"cell_r{r}_c{c}",
                "archetype": "<rootcell>",
                "posM": [tx + cell_w / 2, ty + cell_d / 2, 0.0],
                "slotM": [cell_w, cell_d, max(1.0, float(bbox.get("heightM", 6.0)))],
                "yawDeg": 0.0,
            })
            # If absorbed template is available, also emit one event per
            # archetype placement inside this cell (translated by tile offset).
            for ph in template_placements:
                px = ph["transform"]["translateM"][0] + tx
                py = ph["transform"]["translateM"][1] + ty
                _emit_event(events_paths, {
                    "t": round(time.monotonic() - t0, 4),
                    "type": "placed",
                    "id": f"cell_r{r}_c{c}/{ph['id'][:60]}",
                    "archetype": ph["archetype"],
                    "posM": [px, py, ph["transform"]["translateM"][2]],
                    "slotM": [ph["slotM"]["widthM"], ph["slotM"]["depthM"], ph["slotM"]["heightM"]],
                    "yawDeg": float(ph["transform"].get("yawDeg", 0.0)),
                })
            cells += 1
        if cells >= n_tiles:
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    root_path = out_dir / "root.usd"
    layer.Export(str(root_path))
    counted = None
    try:
        s = Usd.Stage.Open(str(root_path))
        if s is not None:
            counted = sum(1 for _ in s.Traverse())
    except Exception:
        counted = None

    manifest = {
        "schemaVersion": 1,
        "mode": "passthrough_rootcell",
        "assetPack": str(asset_pack),
        "rootStage": str(root_stage_path),
        "tilesAuthored": cells,
        "tileGridM": [cell_w, cell_d],
        "estimatedComposedPrimCount": cells * cell_prims,
        "composedPrimCount_usdcore": counted,
        "summary": {"placements": cells, "filled": cells, "unfilled": [],
                    "byArchetype": {"<rootcell>": {"filled": cells,
                                                    "picks": {pack_meta["rootStage"]: cells}}}},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def realize(sketch_path: Path, out_dir: Path, *,
            asset_pack_override: str | None = None,
            auto_fit: bool = False) -> dict:
    import random
    sketch = json.loads(sketch_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    if asset_pack_override:
        sketch["assetPack"] = asset_pack_override
    asset_pack = Path(sketch["assetPack"])
    defaults = sketch.get("defaults", {})
    default_margin = float(defaults.get("marginFactor", 1.05))
    default_onmiss = defaults.get("defaultOnMiss", "skip")
    pick_strategy = defaults.get("pickStrategy", "best-fit")
    rng = random.Random(int(sketch.get("seed", 0)))
    candidates, pack_meta = _load_pack(asset_pack)
    auto_fit_resized = 0
    if auto_fit:
        auto_fit_resized = _auto_fit_slots(sketch, candidates, default_margin)

    # ---- Composition strategy ------------------------------------------------
    # Sketch is the blueprint; the realizer decides USD organization. Default
    # everything to current behavior so existing callers don't change.
    cs = dict(sketch.get("compositionStrategy") or {})
    cs.setdefault("composition", "reference")
    cs.setdefault("payloadSizeThresholdBytes", 1_000_000)
    cs.setdefault("instancing", "none")
    cs.setdefault("hierarchy", "as-sketch")
    cs.setdefault("layerSplit", "single")
    cs.setdefault("upAxis", "Z")
    cs.setdefault("metersPerUnit", 1.0)
    cs.setdefault("structure", "direct")
    cs_composition = cs.get("composition", "reference")  # reference | payload | mixed
    cs_payload_threshold = int(cs.get("payloadSizeThresholdBytes", 1_000_000))
    cs_instancing = cs.get("instancing", "none")          # none | protoPerAsset
    cs_hierarchy = cs.get("hierarchy", "as-sketch")       # as-sketch | flat
    cs_layer_split = cs.get("layerSplit", "single")       # single (per-zone TODO)
    cs_structure = cs.get("structure", "direct")          # direct | single_scene_payload | deep_reference_chain
    cs_reference_chain_depth = int(cs.get("referenceChainDepth", 3))
    cs_primitive_mode = cs.get("primitiveMode")
    # Stage-level metadata. Defaults match prior behavior.
    cs_up_axis = str(cs.get("upAxis", "Z")).upper()       # Z | Y
    cs_mpu = float(cs.get("metersPerUnit", 1.0))          # any positive float
    if cs_composition not in ("reference", "payload", "mixed"):
        raise ValueError(f"compositionStrategy.composition: {cs_composition!r}")
    if cs_instancing not in ("none", "protoPerAsset"):
        raise ValueError(f"compositionStrategy.instancing: {cs_instancing!r}")
    if cs_hierarchy not in ("as-sketch", "flat"):
        raise ValueError(f"compositionStrategy.hierarchy: {cs_hierarchy!r}")
    if cs_structure not in ("direct", "single_scene_payload", "deep_reference_chain"):
        raise ValueError(f"compositionStrategy.structure: {cs_structure!r}")
    if cs_up_axis not in ("Z", "Y"):
        raise ValueError(f"compositionStrategy.upAxis: {cs_up_axis!r} (must be Z or Y)")
    if cs_mpu <= 0:
        raise ValueError(f"compositionStrategy.metersPerUnit: {cs_mpu!r} (must be > 0)")
    if cs_reference_chain_depth <= 0:
        raise ValueError("compositionStrategy.referenceChainDepth must be > 0")

    # rootStage passthrough: pack.json declares rootStage → the realizer
    # ignores the sketch's per-placement structure and authors tiled
    # references to the rootStage USD instead.
    if pack_meta.get("rootStage") and sketch.get("mode") == "passthrough_rootcell":
        return _realize_rootcell_passthrough(sketch, asset_pack, pack_meta, out_dir)

    layer = _create_crate_layer()
    layer.defaultPrim = "World"

    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(stage, cs_mpu)
    UsdGeom.SetStageUpAxis(stage,
                           UsdGeom.Tokens.y if cs_up_axis == "Y"
                           else UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # Root xform converts from the realizer's native frame ("Z-up meters" —
    # what the per-placement loop authors) to the requested stage's frame.
    # Per-placement code stays simple; /World adjusts the whole subtree.
    #
    # For Z-up → Y-up: rotateX(-90) maps (x,y,z) → (x, z, -y), so a Z-up
    # height of z=h becomes a Y-up height of y=h. (rotateX(+90) would map
    # z=h to y=-h — floor would float above the placements.)
    #
    # Then scale by 1/mpu so 1 metre of authoring becomes the correct number
    # of stage units. Order: rotate first, then scale (op-order [rotateX,
    # scale] applies as M = rotateX × scale, i.e. scale runs in local frame
    # then rotate; the sign of the scale value is unaffected).
    world_to_stage = 1.0 / cs_mpu  # meters → stage-unit factor
    world_xform = UsdGeom.Xformable(world)
    if cs_up_axis == "Y":
        world_xform.AddRotateXOp().Set(-90.0)
    if abs(world_to_stage - 1.0) > 1e-9:
        world_xform.AddScaleOp().Set(Gf.Vec3f(world_to_stage,
                                              world_to_stage,
                                              world_to_stage))

    summary = {"placements": 0, "filled": 0, "unfilled": [], "byArchetype": {}}

    # `protoPerAsset` instancing relies on USD's native composition-based
    # instancing: two prims with IDENTICAL composition stacks (same reference
    # + same xform ops) share a prototype automatically. The ref-bearing prim
    # (`_pack_normalize`) is the natural place to mark instanceable, since
    # multiple placements of the same asset+scale produce identical
    # composition there even though the parent placement xforms differ.
    inst_count = {"protos": 0}

    def _resolve_composition(asset_path: str) -> str:
        if cs_composition == "mixed":
            try:
                return ("payload" if Path(asset_path).stat().st_size
                        >= cs_payload_threshold else "reference")
            except OSError:
                return "reference"
        return cs_composition

    # ---- Sdf-level authoring helpers (batched under one ChangeBlock) -----
    def _sdf_xform(layer: Sdf.Layer, path: str,
                   ops: list[tuple[str, object]] | None = None,
                   reference: str | None = None,
                   payload: str | None = None,
                   instanceable: bool = False,
                   type_name: str = "Xform",
                   semantic_label: str | None = None) -> Sdf.PrimSpec:
        """Author an Xform (or any typed) prim and its xformOp:* attrs
        directly through Sdf. ~5-10× faster than UsdGeom.Xform.Define +
        AddTranslateOp/etc., and avoids per-call stage notifications.

        ops: list of (op_type, value) — op_type ∈ {translate, rotateX, rotateZ, scale}.
        """
        spec = Sdf.CreatePrimInLayer(layer, Sdf.Path(path))
        spec.specifier = Sdf.SpecifierDef
        spec.typeName = type_name
        op_names: list[str] = []
        for op_type, value in (ops or []):
            attr_name = f"xformOp:{op_type}"
            op_names.append(attr_name)
            if op_type == "translate":
                a = Sdf.AttributeSpec(spec, attr_name, Sdf.ValueTypeNames.Double3)
                a.default = Gf.Vec3d(*value)
            elif op_type == "rotateZ":
                a = Sdf.AttributeSpec(spec, attr_name, Sdf.ValueTypeNames.Float)
                a.default = float(value)
            elif op_type == "rotateX":
                a = Sdf.AttributeSpec(spec, attr_name, Sdf.ValueTypeNames.Float)
                a.default = float(value)
            elif op_type == "scale":
                a = Sdf.AttributeSpec(spec, attr_name, Sdf.ValueTypeNames.Float3)
                a.default = Gf.Vec3f(*value)
            else:
                raise ValueError(f"unsupported xform op type: {op_type}")
        if op_names:
            order = Sdf.AttributeSpec(spec, "xformOpOrder",
                                      Sdf.ValueTypeNames.TokenArray)
            order.default = Vt.TokenArray(op_names)
        if reference:
            spec.referenceList.prependedItems.append(Sdf.Reference(reference))
        if payload:
            spec.payloadList.prependedItems.append(Sdf.Payload(payload))
        if instanceable:
            spec.instanceable = True
        if semantic_label:
            spec.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit(["SemanticsAPI"]))
            semantic_type = Sdf.AttributeSpec(
                spec,
                "semantics:Semantics:params:semanticType",
                Sdf.ValueTypeNames.Token,
            )
            semantic_type.default = "class"
            semantic_data = Sdf.AttributeSpec(
                spec,
                "semantics:Semantics:params:semanticData",
                Sdf.ValueTypeNames.Token,
            )
            semantic_data.default = semantic_label
        return spec

    def _sdf_cube(parent_path: str, name: str, size_xyz, center_xyz):
        """Author a UsdGeomCube prim (size attr + translate/scale ops) via
        pure Sdf. Safe inside Sdf.ChangeBlock."""
        cube_path = f"{parent_path}/{name}"
        cspec = Sdf.CreatePrimInLayer(layer, Sdf.Path(cube_path))
        cspec.specifier = Sdf.SpecifierDef
        cspec.typeName = "Cube"
        size_attr = Sdf.AttributeSpec(cspec, "size", Sdf.ValueTypeNames.Double)
        size_attr.default = 1.0
        t_attr = Sdf.AttributeSpec(cspec, "xformOp:translate", Sdf.ValueTypeNames.Double3)
        t_attr.default = Gf.Vec3d(*center_xyz)
        s_attr = Sdf.AttributeSpec(cspec, "xformOp:scale", Sdf.ValueTypeNames.Float3)
        s_attr.default = Gf.Vec3f(*size_xyz)
        order_attr = Sdf.AttributeSpec(cspec, "xformOpOrder",
                                       Sdf.ValueTypeNames.TokenArray)
        order_attr.default = Vt.TokenArray(["xformOp:translate", "xformOp:scale"])

    def _author_preview_material(parent_path: str, name: str,
                                  diffuse: tuple[float, float, float],
                                  roughness: float = 0.85,
                                  metallic: float = 0.0) -> str:
        """Author a tiny UsdPreviewSurface material at <parent>/<name>.
        Returns the material's Sdf path so the caller can bind to it.

        Default material is opaque, mostly-rough, non-metallic — fits the
        "indoor surface" aesthetic (concrete, drywall, paint). For roof
        metal we crank metallic + lower roughness on a per-call basis.
        """
        mat_path = f"{parent_path}/{name}"
        mat_spec = Sdf.CreatePrimInLayer(layer, Sdf.Path(mat_path))
        mat_spec.specifier = Sdf.SpecifierDef
        mat_spec.typeName = "Material"

        shader_path = f"{mat_path}/PreviewSurface"
        sh_spec = Sdf.CreatePrimInLayer(layer, Sdf.Path(shader_path))
        sh_spec.specifier = Sdf.SpecifierDef
        sh_spec.typeName = "Shader"

        id_attr = Sdf.AttributeSpec(sh_spec, "info:id", Sdf.ValueTypeNames.Token)
        id_attr.default = "UsdPreviewSurface"
        dc = Sdf.AttributeSpec(sh_spec, "inputs:diffuseColor",
                                Sdf.ValueTypeNames.Color3f)
        dc.default = Gf.Vec3f(*diffuse)
        r = Sdf.AttributeSpec(sh_spec, "inputs:roughness",
                               Sdf.ValueTypeNames.Float)
        r.default = float(roughness)
        m = Sdf.AttributeSpec(sh_spec, "inputs:metallic",
                               Sdf.ValueTypeNames.Float)
        m.default = float(metallic)
        sh_out = Sdf.AttributeSpec(sh_spec, "outputs:surface",
                                    Sdf.ValueTypeNames.Token)

        # Wire Material.outputs:surface -> Shader.outputs:surface so the
        # renderer can find the shader from the binding.
        mat_out = Sdf.AttributeSpec(mat_spec, "outputs:surface",
                                     Sdf.ValueTypeNames.Token)
        mat_out.connectionPathList.explicitItems = [
            Sdf.Path(f"{shader_path}.outputs:surface")
        ]
        return mat_path

    def _bind_material(prim_path: str, material_path: str) -> None:
        prim_spec = layer.GetPrimAtPath(Sdf.Path(prim_path))
        if prim_spec is None:
            return
        rel = Sdf.RelationshipSpec(prim_spec, "material:binding",
                                    custom=False)
        rel.targetPathList.explicitItems = [Sdf.Path(material_path)]

    def _author_shell(parent_path: str, bounds: dict):
        w = float(bounds["widthM"]); d = float(bounds["depthM"]); h = float(bounds["heightM"])
        # Defensive cap: when the declared wall height is much taller than
        # the footprint (e.g. legacy bounds with H=100 m for an 8x6 m room),
        # cap to half the longer horizontal side, with a 3 m minimum. Lets
        # reasonable values pass unchanged.
        cap_h = max(3.0, max(w, d) * 0.5)
        if h > cap_h:
            h = cap_h
        thick = 0.1
        # floor + 4 walls + roof, Sdf-authored. The roof is authored with
        # `visibility=invisible` by default so the realized viewport still
        # sees inside — toggle the prim's visibility to `inherited` in Kit
        # to render the closed box (or set SKETCH_STAGE_ROOF_VISIBLE=1 in
        # the environment before realize to make it visible by default).
        shell_path = f"{parent_path}/shell"
        shell_spec = Sdf.CreatePrimInLayer(layer, Sdf.Path(shell_path))
        shell_spec.specifier = Sdf.SpecifierDef
        shell_spec.typeName = "Xform"
        roof_visible = bool(int(os.environ.get(
            "SKETCH_STAGE_ROOF_VISIBLE", "0") or "0"))

        # Default indoor-space palette — warmer than pure gray, with
        # enough roughness/colour contrast that the shell reads as a
        # built space rather than a featureless void. Floor is darker
        # concrete, walls off-white, roof a cooler corrugated-steel
        # grey. Material binding is set on each Cube below.
        mats_path = f"{shell_path}/Materials"
        mats_scope = Sdf.CreatePrimInLayer(layer, Sdf.Path(mats_path))
        mats_scope.specifier = Sdf.SpecifierDef
        mats_scope.typeName = "Scope"

        # Neutral default shell materials. The sketch may override per
        # part by setting `sketch["shellMaterials"] = {floor|wall|roof:
        # {diffuseColor: [r,g,b], roughness: float, metallic: float}}`.
        # The skill ships no per-domain material palettes — bring your own.
        prof = {
            "floor": {"diffuseColor": (0.55, 0.55, 0.55), "roughness": 0.85, "metallic": 0.0},
            "wall":  {"diffuseColor": (0.90, 0.90, 0.90), "roughness": 0.75, "metallic": 0.0},
            "roof":  {"diffuseColor": (0.60, 0.60, 0.60), "roughness": 0.50, "metallic": 0.0},
        }
        overrides = (sketch.get("shellMaterials") or {}) if isinstance(sketch, dict) else {}
        for k in ("floor", "wall", "roof"):
            if isinstance(overrides.get(k), dict):
                prof[k] = overrides[k]

        def _mat_of(part: str) -> dict:
            return prof.get(part) or {
                "diffuseColor": (0.6, 0.6, 0.6), "roughness": 0.8, "metallic": 0.0}

        m_floor = _author_preview_material(
            mats_path, "shell_floor",
            tuple(_mat_of("floor")["diffuseColor"]),
            roughness=float(_mat_of("floor").get("roughness", 0.85)),
            metallic=float(_mat_of("floor").get("metallic", 0.0)))
        m_wall = _author_preview_material(
            mats_path, "shell_wall",
            tuple(_mat_of("wall")["diffuseColor"]),
            roughness=float(_mat_of("wall").get("roughness", 0.75)),
            metallic=float(_mat_of("wall").get("metallic", 0.0)))
        m_roof = _author_preview_material(
            mats_path, "shell_roof",
            tuple(_mat_of("roof")["diffuseColor"]),
            roughness=float(_mat_of("roof").get("roughness", 0.5)),
            metallic=float(_mat_of("roof").get("metallic", 0.4)))

        for name, size, center, hidden, mat in [
            ("floor",  (w, d, thick), (w / 2, d / 2, -thick / 2),
                False, m_floor),
            ("wall_s", (w, thick, h), (w / 2, thick / 2, h / 2),
                False, m_wall),
            ("wall_n", (w, thick, h), (w / 2, d - thick / 2, h / 2),
                False, m_wall),
            ("wall_w", (thick, d, h), (thick / 2, d / 2, h / 2),
                False, m_wall),
            ("wall_e", (thick, d, h), (w - thick / 2, d / 2, h / 2),
                False, m_wall),
            ("roof",   (w, d, thick), (w / 2, d / 2, h + thick / 2),
                not roof_visible, m_roof),
        ]:
            _sdf_cube(shell_path, name, size, center)
            cube_path = f"{shell_path}/{name}"
            cube_spec = layer.GetPrimAtPath(Sdf.Path(cube_path))
            if cube_spec is not None:
                if hidden:
                    vis_attr = Sdf.AttributeSpec(cube_spec, "visibility",
                                                  Sdf.ValueTypeNames.Token)
                    vis_attr.default = "invisible"
            _bind_material(cube_path, mat)

    def walk(node: dict, parent_M: Gf.Matrix4d, parent_path: str):
        kind = node["type"]
        if kind == "site":
            shell = node.get("shell")
            if shell:
                _author_shell(parent_path, shell["boundsM"])
            for child in node.get("children", []):
                walk(child, parent_M, parent_path)
            return

        node_id = node.get("id") or "node"
        if kind == "zone":
            M = _compose(parent_M, node.get("transform", {}))
            if cs_hierarchy == "flat":
                # No zone xform; bake the zone transform into M and walk
                # children directly under `parent_path` (i.e., /World).
                shell = node.get("shell")
                if shell:
                    _author_shell(parent_path, shell["boundsM"])
                for child in node.get("children", []):
                    walk(child, M, parent_path)
                return
            zone_path = f"{parent_path}/{node_id}"
            t = M.ExtractTranslation()
            r = M.ExtractRotation()
            yaw = r.Decompose(Gf.Vec3d(1, 0, 0),
                              Gf.Vec3d(0, 1, 0),
                              Gf.Vec3d(0, 0, 1))[2]
            _sdf_xform(layer, zone_path,
                       ops=[("translate", (t[0], t[1], t[2])),
                            ("rotateZ", yaw)])
            shell = node.get("shell")
            if shell:
                _author_shell(zone_path, shell["boundsM"])
            for child in node.get("children", []):
                walk(child, Gf.Matrix4d().SetIdentity(), zone_path)
            return

        if kind == "placement":
            summary["placements"] += 1
            arche = node["archetype"]
            slot = (
                float(node["slotM"]["widthM"]),
                float(node["slotM"]["depthM"]),
                float(node["slotM"]["heightM"]),
            )
            margin = float(node.get("marginFactor", default_margin))
            strategy = node.get("pickStrategy", pick_strategy)
            # Explicit assetPath bypasses best-fit; build a synthetic Candidate.
            explicit_path = node.get("assetPath")
            if cs_primitive_mode == "synth":
                picked = None
            elif explicit_path:
                # Accept either an absolute path or a path relative to the
                # asset pack (matching `query_pack_assets` / `samplePath`).
                explicit_p = Path(explicit_path)
                if not explicit_p.is_absolute():
                    explicit_p = (asset_pack / explicit_path).resolve()
                explicit_abs = str(explicit_p)
                try:
                    explicit_rel = str(explicit_p.relative_to(asset_pack))
                except ValueError:
                    explicit_rel = explicit_path
                measured = _measure_bbox(explicit_abs)
                if measured is not None:
                    sz, mn, mx = measured
                    # Mirror `_load_pack`'s per-asset normalization so the
                    # synthetic Candidate's bbox is in the same world-Z-up
                    # meters frame as regular candidates.
                    asset_up, asset_mpu = _asset_orientation(explicit_abs)
                    pack_up = str(pack_meta.get("upAxis", "Z")).upper()
                    pack_mpu = float(pack_meta.get("metersPerUnit", 1.0))
                    asset_up = asset_up if asset_up in ("Y", "Z") else pack_up
                    asset_mpu = asset_mpu if asset_mpu > 0 else pack_mpu
                    if asset_up == "Y":
                        sz = (sz[0], sz[2], sz[1])
                        mn = (mn[0], mn[2], mn[1])
                        mx = (mx[0], mx[2], mx[1])
                    sz = (sz[0] * asset_mpu, sz[1] * asset_mpu, sz[2] * asset_mpu)
                    mn = (mn[0] * asset_mpu, mn[1] * asset_mpu, mn[2] * asset_mpu)
                    mx = (mx[0] * asset_mpu, mx[1] * asset_mpu, mx[2] * asset_mpu)
                    picked = Candidate(arche, explicit_rel, explicit_abs, sz, mn, mx)
                else:
                    picked = None
            else:
                picked = _pick(arche, slot, margin, candidates, strategy, rng)
            on_miss = "synth" if cs_primitive_mode == "synth" else node.get("onMiss", default_onmiss)
            if picked is None:
                if on_miss == "synth":
                    # Cube primitive sized to slot at the placement (Sdf path).
                    M = _compose(parent_M, node.get("transform", {}))
                    placement_path = f"{parent_path}/{node_id}"
                    t = M.ExtractTranslation()
                    r = M.ExtractRotation()
                    yaw = r.Decompose(Gf.Vec3d(1, 0, 0),
                                      Gf.Vec3d(0, 1, 0),
                                      Gf.Vec3d(0, 0, 1))[2]
                    _sdf_xform(layer, placement_path,
                               ops=[("translate", (t[0], t[1], t[2] + slot[2] / 2)),
                                    ("rotateZ", yaw)],
                               semantic_label=_semantic_label(arche))
                    _sdf_cube(placement_path, "synth",
                              size_xyz=(slot[0], slot[1], slot[2]),
                              center_xyz=(0.0, 0.0, 0.0))
                    summary["filled"] += 1
                    summary.setdefault("synthesized", 0)
                    summary["synthesized"] += 1
                    summary["byArchetype"].setdefault(arche, {"filled": 0, "picks": {}})
                    summary["byArchetype"][arche]["filled"] += 1
                    key = "<synth>"
                    summary["byArchetype"][arche]["picks"].setdefault(key, 0)
                    summary["byArchetype"][arche]["picks"][key] += 1
                    return
                if on_miss == "abort":
                    raise RuntimeError(f"abort: no candidate fits placement {node_id} (archetype {arche})")
                summary["unfilled"].append({
                    "id": node_id, "archetype": arche, "slotM": slot,
                    "reason": "no candidate within margin",
                    "onMiss": on_miss,
                })
                return

            # Read the asset stage's actual upAxis + mpu (authoritative).
            asset_up, asset_mpu = _asset_orientation(picked.abs_path)
            is_y_up = asset_up == "Y"
            mpu = asset_mpu

            # Pivot policy: translateM names the SLOT center-floor in the
            # parent (zone) frame in METERS. Apply a corrective so the asset's
            # bbox center-floor lands on that point. Bbox values are in raw
            # asset stage units; convert to world meters via mpu, and pick the
            # 'floor' axis based on the asset's own upAxis.
            mn, mx = picked.bbox_min, picked.bbox_max
            if is_y_up:
                # World after rotateX(+90):  world.X = asset.X * mpu,
                #                            world.Y = -asset.Z * mpu,
                #                            world.Z =  asset.Y * mpu
                cx_world_m = ((mn[0] + mx[0]) / 2.0) * mpu
                cy_world_m = -((mn[2] + mx[2]) / 2.0) * mpu
                floor_z_world_m = mn[1] * mpu
            else:
                cx_world_m = ((mn[0] + mx[0]) / 2.0) * mpu
                cy_world_m = ((mn[1] + mx[1]) / 2.0) * mpu
                floor_z_world_m = mn[2] * mpu

            corrective = {
                "translateM": [-cx_world_m, -cy_world_m, -floor_z_world_m],
                "yawDeg": 0.0,
            }
            M = _compose(parent_M, node.get("transform", {}))
            # Apply corrective in asset-local frame (BEFORE rotation), not world frame.
            # Gf uses row-vector convention: p_world = p_local * M, so left-multiply.
            M = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*corrective["translateM"])) * M
            placement_path = f"{parent_path}/{node_id}"
            t = M.ExtractTranslation()
            r = M.ExtractRotation()
            yaw = r.Decompose(Gf.Vec3d(1, 0, 0),
                              Gf.Vec3d(0, 1, 0),
                              Gf.Vec3d(0, 0, 1))[2]
            scaleM = node.get("scaleM")
            need_normalize = (mpu != 1.0) or is_y_up or bool(scaleM)
            composition = _resolve_composition(picked.abs_path)
            ref_kw = {"reference": picked.abs_path} if composition == "reference" \
                else {"payload": picked.abs_path}
            mark_instanceable = (cs_instancing == "protoPerAsset")

            placement_ops = [
                ("translate", (t[0], t[1], t[2])),
                ("rotateZ", yaw),
            ]
            if need_normalize:
                # placement xform: just translate + yaw, no asset arc
                _sdf_xform(layer, placement_path, ops=placement_ops,
                           semantic_label=_semantic_label(arche))
                # _pack_normalize child carries the rotateX (Y-up) + scale +
                # the asset arc (and is the prim USD instances on).
                norm_ops: list = []
                if is_y_up:
                    norm_ops.append(("rotateX", 90.0))
                if scaleM:
                    sw_ = float(scaleM[0]); sd_ = float(scaleM[1]); sh_ = float(scaleM[2])
                    if is_y_up:
                        ax, ay, az = sw_, sh_, sd_
                    else:
                        ax, ay, az = sw_, sd_, sh_
                    norm_ops.append(("scale", (mpu * ax, mpu * ay, mpu * az)))
                else:
                    norm_ops.append(("scale", (mpu, mpu, mpu)))
                _sdf_xform(layer, f"{placement_path}/_pack_normalize",
                           ops=norm_ops, instanceable=mark_instanceable, **ref_kw)
                if mark_instanceable:
                    inst_count["protos"] += 1
            else:
                # Trivial case: ref + translate/yaw on the same prim.
                _sdf_xform(layer, placement_path, ops=placement_ops,
                           instanceable=mark_instanceable,
                           semantic_label=_semantic_label(arche), **ref_kw)
                if mark_instanceable:
                    inst_count["protos"] += 1

            summary["filled"] += 1
            summary["byArchetype"].setdefault(arche, {"filled": 0, "picks": {}})
            summary["byArchetype"][arche]["filled"] += 1
            summary["byArchetype"][arche]["picks"].setdefault(picked.rel_path, 0)
            summary["byArchetype"][arche]["picks"][picked.rel_path] += 1
            return

    # Batch all stage edits under one Sdf.ChangeBlock so notifications
    # and recomposition fire only once at exit.
    import time as _time
    _t_phase = {}
    _t0 = _time.perf_counter()
    with Sdf.ChangeBlock():
        walk(sketch["tree"], Gf.Matrix4d().SetIdentity(), "/World")
    _t_phase["walk_s"] = round(_time.perf_counter() - _t0, 3)

    _t0 = _time.perf_counter()
    export_info = _export_with_structure(
        layer,
        out_dir,
        cs_structure,
        up_axis=cs_up_axis,
        meters_per_unit=cs_mpu,
        reference_chain_depth=cs_reference_chain_depth,
    )
    root_path = Path(export_info["rootPath"])
    _t_phase["export_s"] = round(_time.perf_counter() - _t0, 3)

    # Open the realized stage and count composed prims. On Linux, assets
    # with Windows-style payload paths (`payloads\base.usda`) do not resolve
    # via usd-core, so this count reflects the leaf-scaffolding only. Kit
    # on Linux resolves them at runtime, so its in-app count is higher.
    counted = None
    if os.environ.get("SKETCH_STAGE_SKIP_USDCORE_COUNT") in {"1", "true", "TRUE", "yes"}:
        _t_phase["count_s"] = 0.0
    else:
        _t0 = _time.perf_counter()
        try:
            s = Usd.Stage.Open(str(root_path))
            if s is not None:
                counted = sum(1 for _ in s.Traverse())
        except Exception:
            counted = None
        _t_phase["count_s"] = round(_time.perf_counter() - _t0, 3)

    manifest = {
        "schemaVersion": 1,
        "sketch": str(sketch_path),
        "assetPack": str(asset_pack),
        "autoFit": {"enabled": auto_fit, "slotsResized": auto_fit_resized},
        "summary": summary,
        "composedPrimCount_usdcore": counted,
        "candidatesByArchetype": {a: len(cs) for a, cs in candidates.items()},
        "compositionStrategy": {
            "composition": cs_composition,
            "payloadSizeThresholdBytes": cs_payload_threshold,
            "instancing": cs_instancing,
            "hierarchy": cs_hierarchy,
            "layerSplit": cs_layer_split,
            "structure": cs_structure,
            "referenceChainDepth": cs_reference_chain_depth,
            "primitiveMode": cs_primitive_mode,
            "upAxis": cs_up_axis,
            "metersPerUnit": cs_mpu,
            "instanceableMarkedCount": inst_count["protos"],
        },
        "layers": export_info,
        "timings": _t_phase,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str]) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("sketch")
    parser.add_argument("out_dir")
    parser.add_argument("--asset-pack", help="override the sketch's assetPack with a different pack root")
    parser.add_argument("--auto-fit-template", action="store_true",
                        help="resize every placement's slotM to fit the largest candidate per archetype × margin")
    args = parser.parse_args(argv[1:])
    m = realize(Path(args.sketch), Path(args.out_dir),
                asset_pack_override=args.asset_pack,
                auto_fit=args.auto_fit_template)
    if m.get("mode") == "passthrough_rootcell":
        print(f"mode=passthrough_rootcell  pack={m['assetPack']}  tiles={m.get('tilesAuthored')}")
        print(f"estimatedComposedPrimCount={m.get('estimatedComposedPrimCount')}  "
              f"composedPrimCount_usdcore={m.get('composedPrimCount_usdcore')}")
        return 0
    s = m["summary"]
    print(f"pack={m['assetPack']}  autoFit={m.get('autoFit')}")
    print(f"placements={s['placements']} filled={s['filled']} unfilled={len(s['unfilled'])}")
    for a, info in s["byArchetype"].items():
        picks = ", ".join(f"{p}={n}" for p, n in info["picks"].items())
        print(f"  {a}: filled={info['filled']}  picks: {picks}")
    if s["unfilled"]:
        print("unfilled:")
        for u in s["unfilled"][:5]:
            print(f"  {u['id']} ({u['archetype']}) slot={u['slotM']} -> {u['reason']}")
        if len(s["unfilled"]) > 5:
            print(f"  ... +{len(s['unfilled']) - 5} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
