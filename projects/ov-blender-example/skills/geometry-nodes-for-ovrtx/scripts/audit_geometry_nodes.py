#!/usr/bin/env python3
"""Read-only JSON audit for named Geometry Nodes objects in Blender."""

import argparse
import json
import math
import sys

import bpy
from mathutils import Vector


def _args():
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", nargs="+", required=True)
    return parser.parse_args(raw)


def _geometry_sockets(group):
    records = []
    for item in group.interface.items_tree:
        if getattr(item, "item_type", None) != "SOCKET":
            continue
        records.append({
            "name": item.name,
            "identifier": item.identifier,
            "in_out": item.in_out,
            "socket_type": item.socket_type,
        })
    return records


def _record(obj, depsgraph):
    issues = []
    modifiers = []
    nodes_modifiers = [modifier for modifier in obj.modifiers if modifier.type == "NODES"]
    if not nodes_modifiers:
        issues.append("no_geometry_nodes_modifier")
    for modifier in nodes_modifiers:
        group = modifier.node_group
        item = {"name": modifier.name, "node_group": group.name if group else None}
        if group is None:
            item["issues"] = ["missing_node_group"]
            issues.append(f"modifier:{modifier.name}:missing_node_group")
        else:
            sockets = _geometry_sockets(group)
            geometry_outputs = [socket for socket in sockets if socket["in_out"] == "OUTPUT" and socket["socket_type"] == "NodeSocketGeometry"]
            group_issues = []
            if not geometry_outputs:
                group_issues.append("missing_geometry_output")
            if not any(node.type == "GROUP_OUTPUT" and any(socket.is_linked and socket.bl_idname == "NodeSocketGeometry" for socket in node.inputs) for node in group.nodes):
                group_issues.append("geometry_output_unlinked")
            item.update({"nodes": len(group.nodes), "links": len(group.links), "interface": sockets, "issues": group_issues})
            issues.extend(f"modifier:{modifier.name}:{issue}" for issue in group_issues)
        modifiers.append(item)

    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        points = [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices]
        finite = all(all(math.isfinite(float(axis)) for axis in point) for point in points)
        if not mesh.vertices:
            issues.append("evaluated_mesh_has_no_vertices")
        if not mesh.polygons:
            issues.append("evaluated_mesh_has_no_polygons")
        if not finite:
            issues.append("evaluated_mesh_has_nonfinite_coordinates")
        bounds = None
        if points and finite:
            minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
            maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
            bounds = {"minimum": list(minimum), "maximum": list(maximum), "dimensions": list(maximum - minimum)}
        evaluated_record = {"vertices": len(mesh.vertices), "edges": len(mesh.edges), "polygons": len(mesh.polygons), "finite": finite, "world_bounds": bounds}
    finally:
        evaluated.to_mesh_clear()
    return {"name": obj.name, "ok": not issues, "issues": issues, "modifiers": modifiers, "evaluated": evaluated_record}


def audit(object_names):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    records = []
    missing = []
    for name in object_names:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            missing.append(name)
        else:
            records.append(_record(obj, depsgraph))
    checks = {"all_objects_resolved": not missing, "objects_audited": bool(records), "all_objects_pass": bool(records) and all(record["ok"] for record in records)}
    return {"schema": "blender_geometry_nodes_audit.v1", "ok": all(checks.values()), "blender_version": bpy.app.version_string, "checks": checks, "missing": missing, "objects": records}


if "GN_AUDIT_REQUEST" in globals():
    print(json.dumps(audit(GN_AUDIT_REQUEST["objects"]), sort_keys=True))
elif __name__ == "__main__":
    result = audit(_args().objects)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["ok"] else 2)
