"""Print a compact, read-only JSON description of the current Blender scene."""

from __future__ import annotations

import json

import bpy


def _object_type_counts(scene: bpy.types.Scene) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obj in scene.objects:
        counts[obj.type] = counts.get(obj.type, 0) + 1
    return dict(sorted(counts.items()))


def build_probe() -> dict[str, object]:
    scene = bpy.context.scene
    camera = scene.camera
    active = bpy.context.view_layer.objects.active
    filepath = bpy.data.filepath or None
    return {
        "schema": "blender_scene_probe.v1",
        "ok": True,
        "blender": {
            "version": bpy.app.version_string,
            "version_tuple": list(bpy.app.version),
        },
        "file": {
            "path": filepath,
            "saved": filepath is not None,
            "dirty": bool(bpy.data.is_dirty),
        },
        "scene": {
            "name": scene.name,
            "frame": int(scene.frame_current),
            "frame_start": int(scene.frame_start),
            "frame_end": int(scene.frame_end),
            "render_engine": scene.render.engine,
            "resolution": [
                int(scene.render.resolution_x),
                int(scene.render.resolution_y),
                int(scene.render.resolution_percentage),
            ],
            "camera": camera.name if camera else None,
            "world": scene.world.name if scene.world else None,
            "object_count": len(scene.objects),
            "object_types": _object_type_counts(scene),
        },
        "context": {
            "mode": bpy.context.mode,
            "active_object": active.name if active else None,
            "selected_objects": sorted(obj.name for obj in bpy.context.selected_objects),
        },
    }


def main() -> None:
    print(json.dumps(build_probe(), sort_keys=True))


if __name__ == "__main__":
    main()
