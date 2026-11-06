# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export Blender mesh ray visibility as RTX constant boolean primvars.

Object.visible_shadow (``doNotCastShadows``) and Object.visible_camera
(``hideForCamera``, authored only when off) are the durable authorities;
materials are never modified. The temporary stock-export object identity
distinguishes linked mesh instances. Objects inside instanced collections are
not scene objects and carry no export identity; they are matched by the
exporter's ``userProperties:blender:object_name`` instead.
"""
from pathlib import Path

OBJECT_UID_ATTRIBUTE = "ovrtx:object_session_uid"
OBJECT_NAME_ATTRIBUTE = "userProperties:blender:object_name"
SHADOW_PRIMVAR = "doNotCastShadows"
CAMERA_PRIMVAR = "hideForCamera"


def instanced_mesh_objects(scene) -> dict:
    """Mesh objects reached only through collection instances, by object name.

    Names that more than one instanced object shares are dropped: the stock
    export identifies these objects by name alone, so a shared name cannot be
    attributed safely.
    """
    direct = {int(getattr(obj, "session_uid", 0) or 0) for obj in scene.objects}
    found: dict = {}
    ambiguous = set()
    seen = set()

    def visit(collection):
        if collection is None or id(collection) in seen:
            return
        seen.add(id(collection))
        for obj in getattr(collection, "objects", ()):
            if getattr(obj, "type", "") == "MESH" and int(getattr(obj, "session_uid", 0) or 0) not in direct:
                name = str(getattr(obj, "name", ""))
                if name in found and found[name] is not obj:
                    ambiguous.add(name)
                found[name] = obj
            visit(getattr(obj, "instance_collection", None))
        for child in getattr(collection, "children", ()):
            visit(child)

    for obj in scene.objects:
        visit(getattr(obj, "instance_collection", None))
    return {name: obj for name, obj in found.items() if name not in ambiguous}


def author_mesh_shadow_visibility(path: Path, scene) -> int:
    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Cannot open stock USD for shadow visibility: {path}")
    objects = {
        int(obj.session_uid): obj
        for obj in scene.objects if getattr(obj, "type", "") == "MESH"
    }
    instanced = instanced_mesh_objects(scene)

    def author(prim, obj):
        pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
            SHADOW_PRIMVAR, Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.constant,
        )
        if not pv.Set(not bool(getattr(obj, "visible_shadow", True))):
            raise RuntimeError(f"Cannot author shadow visibility on {prim.GetPath()}")
        if getattr(obj, "visible_camera", True) is False:
            camera = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
                CAMERA_PRIMVAR, Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.constant,
            )
            if not camera.Set(True):
                raise RuntimeError(f"Cannot author camera visibility on {prim.GetPath()}")

    count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        owner = prim
        while owner and not owner.IsPseudoRoot():
            identity = owner.GetAttribute(OBJECT_UID_ATTRIBUTE)
            if identity and identity.HasAuthoredValueOpinion():
                uid = int(identity.Get())
                if uid in objects:
                    author(prim, objects[uid])
                    count += 1
                # Never inherit another object's setting through nested objects.
                break
            name = owner.GetAttribute(OBJECT_NAME_ATTRIBUTE)
            if instanced and name and name.HasAuthoredValueOpinion() and str(name.Get()) in instanced:
                author(prim, instanced[str(name.Get())])
                count += 1
                break
            owner = owner.GetParent()
    stage.GetRootLayer().Save()
    return count
