#!/usr/bin/env python3
"""Read-only JSON audit of named Blender mesh objects and evaluated output."""

import argparse
import json
import math
import sys

import bpy


def _finite(values):
    return all(math.isfinite(float(value)) for value in values)


def _record(obj, depsgraph):
    issues = []
    source = obj.data
    invalid_source_faces = [p.index for p in source.polygons if any(i >= len(source.vertices) for i in p.vertices)]
    if invalid_source_faces:
        issues.append("source_face_index_invalid")
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        world = [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices]
        nonfinite = [vertex.index for vertex, point in zip(mesh.vertices, world) if not _finite(point)]
        degenerate = [p.index for p in mesh.polygons if not math.isfinite(float(p.area)) or p.area <= 1.0e-12]
        edge_faces = [0] * len(mesh.edges)
        edge_by_vertices = {tuple(sorted(edge.vertices)): edge.index for edge in mesh.edges}
        for polygon in mesh.polygons:
            verts = list(polygon.vertices)
            for index, start in enumerate(verts):
                edge = edge_by_vertices.get(tuple(sorted((start, verts[(index + 1) % len(verts)]))))
                if edge is not None:
                    edge_faces[edge] += 1
        boundary = [index for index, count in enumerate(edge_faces) if count == 1]
        nonmanifold = [index for index, count in enumerate(edge_faces) if count == 0 or count > 2]
        if not mesh.vertices:
            issues.append("evaluated_vertices_empty")
        if not mesh.polygons:
            issues.append("evaluated_polygons_empty")
        if nonfinite:
            issues.append("evaluated_coordinates_nonfinite")
        if degenerate:
            issues.append("evaluated_faces_degenerate")
        bounds = None
        if world and not nonfinite:
            minimum = [min(point[axis] for point in world) for axis in range(3)]
            maximum = [max(point[axis] for point in world) for axis in range(3)]
            bounds = {"minimum": minimum, "maximum": maximum,
                      "dimensions": [maximum[i] - minimum[i] for i in range(3)]}
        return {"name": obj.name, "ok": not issues, "issues": issues,
                "source": {"vertices": len(source.vertices), "edges": len(source.edges), "polygons": len(source.polygons)},
                "evaluated": {"vertices": len(mesh.vertices), "edges": len(mesh.edges), "polygons": len(mesh.polygons),
                              "nonfinite_vertices": nonfinite[:25], "degenerate_faces": degenerate[:25],
                              "boundary_edge_count": len(boundary), "nonmanifold_edge_count": len(nonmanifold),
                              "world_bounds": bounds},
                "modifiers": [{"name": item.name, "type": item.type, "show_render": item.show_render} for item in obj.modifiers]}
    finally:
        evaluated.to_mesh_clear()


def audit(names):
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    missing = []
    records = []
    for name in dict.fromkeys(names):
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            missing.append(name)
        else:
            records.append(_record(obj, depsgraph))
    return {"schema": "blender_mesh_audit.v1", "ok": not missing and bool(records) and all(item["ok"] for item in records),
            "blender_version": bpy.app.version_string, "missing": missing, "objects": records}


def _args():
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", nargs="+", required=True)
    return parser.parse_args(raw)


if "MESH_AUDIT_REQUEST" in globals():
    print(json.dumps(audit(MESH_AUDIT_REQUEST["objects"]), sort_keys=True))
elif __name__ == "__main__":
    result = audit(_args().objects)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["ok"] else 2)
