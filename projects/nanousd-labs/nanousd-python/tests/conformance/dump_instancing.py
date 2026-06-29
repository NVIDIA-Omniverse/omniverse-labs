#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 nanousd
# SPDX-License-Identifier: Apache-2.0
"""Backend-agnostic INSTANCING dump for the OpenUSD <-> nanousd differential test.

Where ``dump_stage.py`` walks the broad read surface, this targets the instancing
surface specifically — the area that produced real renderer bugs (e.g. a
PointInstancer prototype with multiple nested Mesh children whose sub-meshes were
dropped). It records, for one asset, everything a renderer/consumer needs to
expand instances correctly, so a diff vs OpenUSD flags subtle instancing bugs:

* native / scene-graph instancing: ``IsInstance`` / ``IsInstanceable`` /
  ``IsInstanceProxy`` / ``IsInPrototype`` / ``GetPrototype`` per prim,
  ``Stage.GetPrototypes()``, and the prim tree (path+type) under each prototype.
* instance-proxy traversal: ``Usd.PrimRange(.., TraverseInstanceProxies())``.
* every ``UsdGeom.PointInstancer``: protoIndices / positions / orientations /
  scales / prototypes targets, ``GetInstanceCount``,
  ``ComputeInstanceTransformsAtTime``, ``ComputeMaskAtTime`` (invisibleIds), and
  the **per-prototype Mesh count** (the "pawn" class — a proto Xform with
  Geom_Body + Geom_Top must enumerate BOTH meshes).

Any API the shim does not implement is recorded as ``<unsupported:...>`` (and so
flagged against OpenUSD as a coverage gap) — never allowed to crash the walk.

Emitted as JSON; see ``compare.compare_generic``. Output keys are prim paths plus
the synthetic keys ``__prototypes__``, ``__instance_proxies__``.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings

from normalize import normalize

warnings.filterwarnings("ignore")

# OpenUSD mints prototype names ``/__Prototype_<n>`` in a non-deterministic order
# (and nanousd uses different indices), so the raw index is both flaky across
# runs and a meaningless cosmetic difference. Canonicalize every occurrence to
# ``/__Prototype_*`` so the dump is deterministic and the diff flags only real
# structural divergences (prototype COUNT/SHAPE, captured separately).
_PROTO_RE = re.compile(r"/__Prototype_\d+")


def _canon(obj):
    if isinstance(obj, str):
        return _PROTO_RE.sub("/__Prototype_*", obj)
    if isinstance(obj, dict):
        return {_canon(k): _canon(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_canon(x) for x in obj]
    return obj


def _get_mods(backend: str):
    if backend == "shim":
        from nanousd.pxr_compat import Sdf, Usd, UsdGeom  # noqa: F401
    elif backend == "openusd":
        from pxr import Sdf, Usd, UsdGeom  # noqa: F401
    else:  # pragma: no cover
        raise SystemExit(f"unknown backend {backend!r}")
    return Usd, Sdf, UsdGeom


def _safe(getter):
    try:
        return getter()
    except Exception as exc:  # noqa: BLE001
        return f"<unsupported:{type(exc).__name__}>"


def _proto_path(prim):
    # GetPrototype() exists on instances; the returned prototype's path is the
    # only backend-stable thing to compare (the /__Prototype_N number itself
    # differs between backends, so we compare structure under it, not the index).
    proto = prim.GetPrototype()
    return str(proto.GetPath()) if proto and proto.IsValid() else None


def _prim_instance_entry(prim) -> dict:
    g = {
        "type": lambda: str(prim.GetTypeName()),
        "active": lambda: bool(prim.IsActive()),
        "is_instance": lambda: bool(prim.IsInstance()),
        "is_instanceable": lambda: bool(prim.IsInstanceable()),
        "is_instance_proxy": lambda: bool(prim.IsInstanceProxy()),
        "is_in_prototype": lambda: bool(prim.IsInPrototype()),
        "is_prototype": lambda: bool(prim.IsPrototype()),
        "prototype": lambda: _proto_path(prim) if prim.IsInstance() else None,
    }
    return {k: _safe(v) for k, v in g.items()}


def _subtree(prim) -> dict:
    """path-suffix -> typeName for every descendant, so two prototypes (or two
    PI-prototype subtrees) are compared by SHAPE, independent of the prototype's
    synthetic /__Prototype_N index. This is what catches "lost a sub-mesh"."""
    root = str(prim.GetPath())
    out = {}
    for d in prim.GetChildren():
        _subtree_rec(d, root, out)
    return out


def _subtree_rec(prim, root, out):
    p = str(prim.GetPath())
    rel = p[len(root):] if p.startswith(root) else p
    out[rel] = _safe(lambda: str(prim.GetTypeName()))
    for c in prim.GetChildren():
        _subtree_rec(c, root, out)


def _count_meshes(prim) -> int:
    n = 0
    stack = [prim]
    while stack:
        cur = stack.pop()
        try:
            if str(cur.GetTypeName()) == "Mesh":
                n += 1
            stack.extend(cur.GetChildren())
        except Exception:  # noqa: BLE001
            pass
    return n


def _proxy_detail(p, Usd, UsdGeom) -> dict:
    """Per-instance-proxy facts a renderer relies on: composed world transform
    (the "pawn at the wrong place" surface), and composed displayColor /
    visibility / purpose (catches a local descendant `over` leaking into the
    SHARED prototype, and broken visibility/purpose resolution through proxies)."""
    d = {"type": _safe(lambda: str(p.GetTypeName()))}

    def _wt():
        m = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        # translation row works for Gf.Matrix4d and a 4x4 numpy array alike
        return normalize([m[3][0], m[3][1], m[3][2]])
    d["world_translate"] = _safe(_wt)

    def _attr(name):
        a = p.GetAttribute(name)
        if not a or not a.HasAuthoredValue():
            # composed (inherited from prototype) value still resolves via Get()
            a = p.GetAttribute(name)
        v = a.Get() if a else None
        return normalize(v) if v is not None else None
    d["displayColor"] = _safe(lambda: _attr("primvars:displayColor"))
    d["visibility"] = _safe(lambda: str(UsdGeom.Imageable(p).ComputeVisibility()))
    d["purpose"] = _safe(lambda: str(UsdGeom.Imageable(p).ComputePurpose()))
    return d


def _pointinstancer_entry(prim, Usd, UsdGeom) -> dict:
    pi = UsdGeom.PointInstancer(prim)
    out = {}
    # GetTypeName()=="PointInstancer" but IsA(PointInstancer)==False is exactly
    # the flat-traversal type-detection bug that dropped the chess pawns.
    out["is_a_pointinstancer"] = _safe(lambda: bool(prim.IsA(UsdGeom.PointInstancer)))

    def arr(attr_getter):
        a = attr_getter()
        if a is None:
            return "<no-attr>"
        v = a.Get()
        return normalize(list(v)) if v is not None else None

    out["protoIndices"] = _safe(lambda: arr(pi.GetProtoIndicesAttr))
    out["positions"] = _safe(lambda: arr(pi.GetPositionsAttr))
    out["orientations"] = _safe(lambda: arr(pi.GetOrientationsAttr))
    out["scales"] = _safe(lambda: arr(pi.GetScalesAttr))
    out["prototypes_targets"] = _safe(
        lambda: sorted(str(t) for t in pi.GetPrototypesRel().GetTargets()))
    out["instance_count"] = _safe(lambda: int(pi.GetInstanceCount()))

    def _xforms():
        t = Usd.TimeCode.Default()
        xf = pi.ComputeInstanceTransformsAtTime(t, t)
        return {"n": len(xf), "values": normalize([list(m) for m in xf])}
    out["computed_transforms"] = _safe(_xforms)

    def _mask():
        t = Usd.TimeCode.Default()
        return normalize(list(pi.ComputeMaskAtTime(t)))
    out["mask"] = _safe(_mask)

    # Per-prototype mesh count: THE pawn-class check. Resolve each prototypes-rel
    # target prim and count Mesh descendants. A proto Xform with two sub-meshes
    # must report 2; dropping one is exactly the pawns regression.
    def _proto_mesh_counts():
        stage = prim.GetStage()
        counts = {}
        for t in pi.GetPrototypesRel().GetTargets():
            tp = stage.GetPrimAtPath(t)
            counts[str(t)] = _count_meshes(tp) if tp and tp.IsValid() else "<missing>"
        return counts
    out["proto_mesh_counts"] = _safe(_proto_mesh_counts)
    return out


def dump(backend: str, asset: str) -> dict:
    Usd, Sdf, UsdGeom = _get_mods(backend)
    stage = Usd.Stage.Open(asset)
    if stage is None:
        raise SystemExit(f"{backend}: failed to open {asset}")

    result: dict[str, dict] = {}

    # 1) Every prim's instancing flags (default traversal — no proxies).
    for prim in stage.Traverse():
        entry = _prim_instance_entry(prim)
        if str(prim.GetTypeName()) == "PointInstancer":
            entry["pointinstancer"] = _pointinstancer_entry(prim, Usd, UsdGeom)
        result[str(prim.GetPath())] = entry

    # 2) Prototypes: list + the SHAPE (path-suffix -> type) under each, so
    #    "did the prototype keep all its geometry" is comparable across backends
    #    despite /__Prototype_N index differences.
    def _prototypes():
        protos = list(stage.GetPrototypes())
        return {
            "count": len(protos),
            "shapes": sorted(
                json.dumps(_subtree(p), sort_keys=True) for p in protos),
        }
    result["__prototypes__"] = _safe(_prototypes)

    # 3) Instance-proxy traversal: count + per-proxy detail (type, composed world
    #    transform, displayColor, visibility, purpose) keyed by path. Catches
    #    dropped nested geometry, mis-placed instances, and over-leak/visibility.
    def _proxies():
        try:
            rng = Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
        except Exception:
            rng = Usd.PrimRange(stage.GetPseudoRoot())
        prims = list(rng)
        detail = {str(p.GetPath()): _proxy_detail(p, Usd, UsdGeom) for p in prims}
        return {"count": len(prims), "detail": detail}
    result["__instance_proxies__"] = _safe(_proxies)

    return _canon(result)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True, choices=("shim", "openusd"))
    ap.add_argument("--asset", required=True)
    args = ap.parse_args()
    print(json.dumps(dump(args.backend, args.asset), sort_keys=True))


if __name__ == "__main__":
    main()
