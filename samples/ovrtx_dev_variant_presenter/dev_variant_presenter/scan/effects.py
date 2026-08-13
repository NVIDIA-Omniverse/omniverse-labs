# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Variant-effect classifier (pxr): diff authored attribute values per variant and
record replayable writes. Pure pxr — off the render thread. On the ConceptCar sample
stage this classifies 0 structural sets (8 shader-input, 2 transform, 3 visibility).

The runtime consults this to apply the cheapest correct action; reload is the
universal fallback.
"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass

from pxr import Usd

Write = namedtuple("Write", "prim attr usd_type value")


@dataclass
class VariantAction:
    kind: str                                   # shader-input | transform | visibility | binding | structural
    per_variant: dict[str, list]                # variant -> [Write]
    swatches: dict | None = None                # variant -> "#rrggbb" (sRGB), or None if no usable color


# Attribute-name fragments that mark a color3f input worth swatching.
_COLOR_HINTS = ("color", "emissi", "tint", "diffuse", "coat", "albedo", "base")


def _color_vec(value):
    """value -> [r,g,b] floats if it's a 3-component vector, else None."""
    try:
        xs = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    return xs if len(xs) == 3 else None


def _linear_to_srgb_hex(rgb) -> str:
    """Linear render color -> '#rrggbb'. USD shader color inputs are linear; without
    the sRGB transfer, dark paints (e.g. Noir 0.036) display as near-black mud."""
    def ch(c: float) -> int:
        c = 0.0 if c < 0.0 else (1.0 if c > 1.0 else c)
        s = 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055
        return max(0, min(255, round(s * 255)))
    r, g, b = (ch(x) for x in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _swatches(per_variant: dict) -> dict | None:
    """Pick the color3f input that varies MOST across the set's variants and return
    {variant: '#rrggbb'}. Data-driven (no per-material hardcoding): for stitching this
    auto-selects diffuse_tint over a constant base; for the screen, emissive_color.
    Returns None when no color input is present for every variant (transform/visibility
    sets)."""
    variants = list(per_variant)
    if not variants:
        return None
    by_attr: dict[str, dict[str, list]] = {}
    for v in variants:
        for w in per_variant[v]:
            if not any(h in w.attr.lower() for h in _COLOR_HINTS):
                continue
            vec = _color_vec(w.value)
            if vec is not None:
                by_attr.setdefault(w.attr, {})[v] = vec
    full = {a: m for a, m in by_attr.items() if len(m) == len(variants)}  # defined for all variants
    if not full:
        return None

    def variance(m: dict) -> float:
        cols = list(m.values())
        mean = [sum(c[i] for c in cols) / len(cols) for i in range(3)]
        return sum(sum((c[i] - mean[i]) ** 2 for i in range(3)) for c in cols)

    best = max(full, key=lambda a: variance(full[a]))
    return {v: _linear_to_srgb_hex(full[best][v]) for v in variants}


def _subtree_snapshot(stage: Usd.Stage, root_path: str):
    """Authored attribute values + the prim-path set under root_path."""
    attrs: dict[tuple[str, str], object] = {}
    prims: set[str] = set()
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return attrs, prims
    for prim in Usd.PrimRange(root):
        p = str(prim.GetPath())
        prims.add(p)
        for a in prim.GetAttributes():
            if a.HasAuthoredValue():
                try:
                    attrs[(p, a.GetName())] = a.Get()
                except Exception:
                    pass
    return attrs, prims


def _kind_of(changed_attrs: set[str], prim_set_changed: bool) -> str:
    if prim_set_changed:
        return "structural"
    if any("xformOp" in a for a in changed_attrs):
        return "transform"
    if any(a == "material:binding" for a in changed_attrs):
        return "binding"
    if any(a.startswith("inputs:") for a in changed_attrs):
        return "shader-input"
    if any(a == "visibility" for a in changed_attrs):
        return "visibility"
    return "structural"  # unknown change -> safe fallback (reload)


def classify_variants(usd_path: str, stage_info=None) -> dict[str, VariantAction]:
    """Return {set_name: VariantAction}. `stage_info` (from scan_stage) is optional;
    if omitted, the sets are discovered under the default prim.

    Holds USD_LOCK around every stage composition/authoring region (open, discovery, the
    per-set SetVariantSelection sweep) so it never races the render thread's
    build_composite/open_usd — concurrent pxr authoring crashes the process. The lock is
    released between sets and for the pure-Python diff/swatch work, so a render-thread
    stage op waits at most one set."""
    from dev_variant_presenter.usd_guard import USD_LOCK

    with USD_LOCK:
        stage = Usd.Stage.Open(usd_path)
        root = f"/{stage.GetRootLayer().defaultPrim}"
        targets: list[tuple[str, str]] = []
        if stage_info is not None:
            targets = [(vs.prim_path, vs.set_name) for vs in stage_info.variant_sets]
        else:
            for prim in stage.Traverse():
                if not str(prim.GetPath()).startswith(root):
                    continue
                for sn in prim.GetVariantSets().GetNames():
                    targets.append((str(prim.GetPath()), sn))

    actions: dict[str, VariantAction] = {}
    for prim_path, set_name in targets:
        with USD_LOCK:   # all stage reads/writes for this set under the lock
            prim = stage.GetPrimAtPath(prim_path)
            vs = prim.GetVariantSets().GetVariantSet(set_name)
            base_sel = vs.GetVariantSelection()
            base_attrs, base_prims = _subtree_snapshot(stage, prim_path)

            # pass 1: union of changed (prim, attr) across variants + per-variant diff values
            union: set[tuple[str, str]] = set()
            all_changed: set[str] = set()
            prim_set_changed = False
            diff_values: dict[str, dict] = {}
            for variant in vs.GetVariantNames():
                vs.SetVariantSelection(variant)
                attrs, prims = _subtree_snapshot(stage, prim_path)
                if prims != base_prims:
                    prim_set_changed = True
                dv = {}
                for key in set(attrs) | set(base_attrs):
                    if attrs.get(key) != base_attrs.get(key) and key in attrs:
                        union.add(key)
                        all_changed.add(key[1])
                        dv[key] = attrs[key]
                diff_values[variant] = dv
            vs.SetVariantSelection(base_sel)

            # attribute types are variant-invariant; capture once at base
            types = {}
            for (p, a) in union:
                ao = stage.GetPrimAtPath(p).GetAttribute(a)
                types[(p, a)] = ao.GetTypeName().type.typeName if ao else ""
            variant_names = list(vs.GetVariantNames())

        # pass 2 is pure Python (no stage access) -> outside the lock
        per_variant: dict[str, list] = {}
        for variant in variant_names:
            dv = diff_values[variant]
            writes = []
            for key in union:
                val = dv[key] if key in dv else base_attrs.get(key)
                if val is None:
                    continue  # authored only in some variants; can't safely revert -> skip
                writes.append(Write(key[0], key[1], types[key], val))
            per_variant[variant] = writes
        actions[set_name] = VariantAction(kind=_kind_of(all_changed, prim_set_changed),
                                          per_variant=per_variant,
                                          swatches=_swatches(per_variant))
    return actions
