# SPDX-License-Identifier: Apache-2.0
"""Conservative semantic closure for retaining an unchanged native simulation."""
from __future__ import annotations

import hashlib
import json


def physics_generation_fingerprint(usd_path: str) -> str | None:
    """Ignore unrelated visual prims, but include all composed physics inputs.

    A fingerprint is optional: uncertainty disables reuse. Instances, external
    asset-valued physics inputs, and time-varying physics stages are deliberately
    excluded until their dependency and evaluation contracts are measured.
    """
    try:
        return _fingerprint(usd_path)
    except Exception:
        # This is only a reuse optimization. Failed inspection always keeps the
        # ordinary validated generation replacement path available.
        return None


def _fingerprint(usd_path, *, ignore_mass=False, ignore_geometry=False):
    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        return None
    closure = set()
    pending = []
    for prim in stage.Traverse():
        schemas = tuple(str(name) for name in prim.GetAppliedSchemas())
        relevant = (prim.GetTypeName().startswith(("Physics", "Physx"))
                    or any(name.startswith(("Physics", "Physx")) for name in schemas)
                    or any(attr.GetName().startswith(("physics:", "physx")) for attr in prim.GetAttributes())
                    or prim.HasRelationship("material:binding:physics"))
        if not relevant:
            continue
        pending.append(prim)
    while pending:
        prim = pending.pop()
        path = str(prim.GetPath())
        if path in closure:
            continue
        closure.add(path)
        # A collision schema on an ancestor may consume untagged child meshes.
        pending.extend(prim.GetChildren())
        for rel in prim.GetRelationships():
            for target in rel.GetForwardedTargets():
                dependency = stage.GetPrimAtPath(target.GetPrimPath())
                if not dependency:
                    return None
                pending.append(dependency)
    physics = []
    ancestors = {}
    for path in sorted(closure):
        prim = stage.GetPrimAtPath(path)
        schemas = tuple(str(name) for name in prim.GetAppliedSchemas())
        if prim.IsInstance() or prim.IsInstanceProxy():
            return None
        attributes = []
        for attr in sorted(prim.GetAttributes(), key=lambda value: value.GetName()):
            if attr.ValueMightBeTimeVarying():
                return None
            value = attr.Get()
            if attr.GetTypeName() in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray):
                return None
            attributes.append((attr.GetName(), str(attr.GetTypeName()), "<mass>" if ignore_mass and attr.GetName() == "physics:mass" else "<geometry>" if ignore_geometry and attr.GetName() in _GEOMETRY_TYPES else repr(value)))
        relationships = [(rel.GetName(), tuple(str(path) for path in rel.GetTargets()))
                         for rel in sorted(prim.GetRelationships(), key=lambda value: value.GetName())]
        physics.append((str(prim.GetPath()), prim.GetTypeName(), schemas, attributes, relationships))
        parent = prim.GetParent()
        while parent and not parent.IsPseudoRoot():
            if parent.IsInstance() or parent.IsInstanceProxy():
                return None
            values = []
            for attr in parent.GetAttributes():
                if attr.ValueMightBeTimeVarying() or attr.GetTypeName() in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray):
                    return None
                values.append((attr.GetName(), str(attr.GetTypeName()), "<mass>" if ignore_mass and attr.GetName() == "physics:mass" else "<geometry>" if ignore_geometry and attr.GetName() in _GEOMETRY_TYPES else repr(attr.Get())))
            # Inherited bindings and collection membership on ancestors need a
            # broader dependency closure; decline reuse until measured.
            if any(rel.GetTargets() for rel in parent.GetRelationships()):
                return None
            ancestors[str(parent.GetPath())] = (parent.GetTypeName(), sorted(values))
            parent = parent.GetParent()
    # Instance contents may contain inherited collisions not reached by Traverse.
    # Refuse the shortcut whenever any instance exists, even a visual one.
    if any(prim.IsInstance() for prim in stage.Traverse()):
        return None
    payload = {"physics": physics, "ancestors": sorted(ancestors.items()),
               "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
               "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
               "time_codes_per_second": stage.GetTimeCodesPerSecond()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def physics_mass_delta(old_path, new_path, body_paths):
    """Return positive explicit mass edits only, or None for any other change."""
    try:
        from pxr import Usd
        import math
        old_key = _fingerprint(old_path, ignore_mass=True)
        if old_key is None or old_key != _fingerprint(new_path, ignore_mass=True):
            return None
        old, new = Usd.Stage.Open(old_path), Usd.Stage.Open(new_path)
        changes = []
        for prim in new.Traverse():
            attribute = prim.GetAttribute("physics:mass")
            if not attribute:
                continue
            previous = old.GetPrimAtPath(prim.GetPath()).GetAttribute("physics:mass")
            before, after = previous.Get(), attribute.Get()
            if before == after:
                continue
            if (str(prim.GetPath()) not in body_paths or not previous.HasAuthoredValueOpinion()
                    or not attribute.HasAuthoredValueOpinion() or before is None or after is None
                    or not math.isfinite(float(before)) or not math.isfinite(float(after))
                    or not 0 < float(before) <= 3.4028234e38 or not 0 < float(after) <= 3.4028234e38):
                return None
            changes.append({"prim_path": str(prim.GetPath()), "mass": float(after)})
        return changes or None
    except Exception:
        return None


_GEOMETRY_TYPES = {'points': 'Point3fArray', 'faceVertexCounts': 'IntArray',
                   'faceVertexIndices': 'IntArray', 'extent': 'Float3Array',
                   'size': 'Double', 'radius': 'Double', 'height': 'Double'}


def physics_geometry_delta(old_path, new_path, *, allow_dynamic=False):
    """Existing static collider geometry only; every other physics input identical."""
    try:
        from pxr import Usd
        old_key = _fingerprint(old_path, ignore_geometry=True)
        if old_key is None or old_key != _fingerprint(new_path, ignore_geometry=True):
            return None
        old, new = Usd.Stage.Open(str(old_path)), Usd.Stage.Open(str(new_path))
        changes = []
        for prim in new.Traverse():
            for name, kind in _GEOMETRY_TYPES.items():
                attribute = prim.GetAttribute(name)
                if not attribute:
                    continue
                previous = old.GetPrimAtPath(prim.GetPath()).GetAttribute(name)
                before, after = previous.Get(), attribute.Get()
                if before == after:
                    continue
                collision = False
                dynamic_body = None
                parent = prim
                while parent and not parent.IsPseudoRoot():
                    schemas = tuple(str(v) for v in parent.GetAppliedSchemas())
                    if 'PhysicsRigidBodyAPI' in schemas:
                        if not allow_dynamic or bool(parent.GetAttribute('physics:kinematicEnabled').Get()):
                            return None
                        dynamic_body = str(parent.GetPath())
                    collision |= 'PhysicsCollisionAPI' in schemas
                    parent = parent.GetParent()
                if not collision:
                    continue
                if dynamic_body:
                    for candidate in new.Traverse():
                        if candidate.GetTypeName().startswith(('Physics', 'Physx')) and 'Joint' in candidate.GetTypeName():
                            if any(str(target.GetPrimPath()) == dynamic_body for rel in candidate.GetRelationships() for target in rel.GetTargets()):
                                return None
                if not previous or before is None or after is None or not previous.HasAuthoredValueOpinion() or not attribute.HasAuthoredValueOpinion():
                    return None
                value = [list(v) for v in after] if kind in {'Point3fArray', 'Float3Array'} else list(after) if kind == 'IntArray' else float(after)
                changes.append(dict(prim_path=str(prim.GetPath()), attribute_name=name, value_type=kind, value=value))
        return changes or None
    except Exception:
        return None


def physics_static_geometry_delta(old_path, new_path):
    return physics_geometry_delta(old_path, new_path)
