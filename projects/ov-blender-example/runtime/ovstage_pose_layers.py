# SPDX-License-Identifier: Apache-2.0
"""Owned native USD opinions for parent transforms in an attached render stage."""
from pathlib import Path
import math
import re


def pose_layer_text(poses):
    tree = {}
    for path, matrix in sorted(poses.items()):
        parts = path.strip("/").split("/")
        if not path.startswith("/") or not all(re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", part) for part in parts):
            raise ValueError("Pose layer requires valid absolute USD prim paths")
        if len(matrix) != 4 or any(len(row) != 4 for row in matrix) or not all(math.isfinite(float(v)) for row in matrix for v in row):
            raise ValueError("Pose layer requires finite 4x4 matrices")
        node = tree
        for part in parts: node = node.setdefault(part, {})
        node[None] = matrix
    lines = ["#usda 1.0"]
    def emit(nodes):
        for name, node in nodes.items():
            if name is None: continue
            lines.append('over "' + name + '" {')
            matrix = node.get(None)
            if matrix is not None:
                literal = "(" + ", ".join("(" + ", ".join(repr(float(v)) for v in row) + ")" for row in matrix) + ")"
                lines.append("matrix4d xformOp:transform = " + literal)
                lines.append('uniform token[] xformOpOrder = ["!resetXformStack!", "xformOp:transform"]')
            emit(node)
            lines.append("}")
    emit(tree)
    return "\n".join(lines) + "\n"


class PoseLayerPopulation:
    """Retain accepted world-pose opinions while the accepted base remains active.

    Population replacement owns native rollback. An unsuccessful candidate never
    enters this cache; source generation replacement implicitly discards old poses.
    """
    def __init__(self, owner, output_root):
        self.owner = owner
        self.output_root = Path(output_root)
        self.base = owner.path
        self.wrapper = None
        self.poses = {}
        self.files = ()
        self.retained_files = ()

    def accepted_generation_replaced(self, *, retain_sources=False):
        self.base, self.wrapper, self.poses = self.owner.path, None, {}
        previous, self.files = self.retained_files + self.files, ()
        self.retained_files = previous if retain_sources else ()
        if not retain_sources:
            for old in previous:
                old.unlink(missing_ok=True)

    def apply(self, updates, *, ordinal):
        retained = self.poses if self.owner.path == self.wrapper else {}
        base = self.base if self.owner.path == self.wrapper else self.owner.path
        poses = {**retained, **updates}
        text = pose_layer_text(poses)
        pose = self.output_root / ("native-poses-" + str(ordinal) + ".usda")
        wrapper = self.output_root / ("native-poses-composed-" + str(ordinal) + ".usda")
        paths = (pose.as_posix(), Path(base).as_posix())
        if any("@" in path or "\n" in path or "\r" in path for path in paths):
            raise ValueError("Pose composition asset path cannot be safely represented")
        pose.write_text(text, encoding="utf-8")
        wrapper.write_text("#usda 1.0\n(\n subLayers = [@" + paths[0] + "@, @" + paths[1] + "@]\n)\n", encoding="utf-8")
        try:
            result = self.owner.replace(wrapper, ordinal=ordinal)
        except Exception as error:
            if getattr(error, "rollback_succeeded", False):
                pose.unlink(missing_ok=True)
                wrapper.unlink(missing_ok=True)
            raise
        previous = self.retained_files + self.files
        self.retained_files = ()
        self.base, self.wrapper, self.poses = base, wrapper, poses
        self.files = (pose, wrapper)
        # replace() has released the old SDK source and completed projection;
        # only exact files created by this owner may now be reclaimed.
        for old in previous:
            old.unlink(missing_ok=True)
        return result
