#!/usr/bin/env python3
"""Print a read-only JSON probe of common official SimReady Blender surfaces."""

import json
import addon_utils
import bpy


OPERATOR_CANDIDATES = {
    "create_collections": "sr_core.create_simready_collections",
    "assign_physx": "simready.assign_physx_properties",
    "simready_export": "export_scene.simready_usd",
}


def operator_record(identifier):
    category, name = identifier.split(".", 1)
    try:
        operator = getattr(getattr(bpy.ops, category), name)
        rna = operator.get_rna_type()
        return {"registered": True, "rna": rna.identifier,
                "properties": sorted(item.identifier for item in rna.properties)}
    except Exception as exc:
        return {"registered": False, "error": f"{type(exc).__name__}: {exc}"}


modules = []
for module in addon_utils.modules():
    identity = " ".join((module.__name__, str(getattr(module, "bl_info", {}).get("name", ""))))
    if "simready" in identity.casefold() or "artisttools" in identity.casefold():
        modules.append({"module": module.__name__, "name": getattr(module, "bl_info", {}).get("name"),
                        "version": list(getattr(module, "bl_info", {}).get("version", ())),
                        "enabled": bool(addon_utils.check(module.__name__)[1])})
operators = {label: operator_record(identifier) for label, identifier in OPERATOR_CANDIDATES.items()}
collections = {name: bpy.data.collections.get(name) is not None for name in
               ("Export", "Geometry", "ReferencePrims", "Colliders")}
result = {"schema": "simready_blender_surface_probe.v1",
          "ok": bool(modules) and operators["create_collections"]["registered"] and operators["assign_physx"]["registered"],
          "blender_version": bpy.app.version_string, "modules": modules,
          "operators": operators, "collections": collections,
          "boundary": "Registration only; authoring, validation, export, and OVRTX conversion are separate gates."}
print(json.dumps(result, sort_keys=True))
