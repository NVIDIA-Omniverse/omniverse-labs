# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 2b: draw a geometry thumbnail of each unlabeled group for the VLM.

For each group representative, draws the unit in orange among its gray neighbors
from two viewpoints (oblique and top) and saves <crops>/<unit id>.png. This is a
shape-and-context view, not a photoreal render; it needs only numpy and matplotlib
and runs without a GPU. An ovrtx render saved under the same file name can replace it.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _usd import SEMANTICS_DIR, Usd, UsdGeom, group_units, is_rendered_gprim, open_stage, unit_id

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

MAX_FACES = 6000          # per draw call; denser meshes are subsampled


def world_triangles(mesh_prim, xform_cache):
    mesh = UsdGeom.Mesh(mesh_prim)
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if not points or not counts or indices is None:
        return np.zeros((0, 3, 3))
    matrix = np.array(xform_cache.GetLocalToWorldTransform(mesh_prim), dtype=float)
    pts = np.asarray(points, dtype=float)
    pts = np.c_[pts, np.ones(len(pts))] @ matrix   # USD matrices are row-major, row vectors
    pts = pts[:, :3]
    indices = np.asarray(indices)
    tris, offset = [], 0
    for count in counts:                            # fan-triangulate polygons
        face = indices[offset:offset+count]
        for k in range(1, count-1):
            tris.append((face[0], face[k], face[k+1]))
        offset += count
    return pts[np.asarray(tris)] if tris else np.zeros((0, 3, 3))


def subsample(tris, limit, rng):
    return tris[rng.choice(len(tris), limit, replace=False)] if len(tris) > limit else tris


def draw(ax, unit_tris, context_tris, lo, hi, elev, azim, y_up):
    def view(t):                                    # matplotlib draws z up
        return t[..., [0, 2, 1]] if y_up else t
    if len(context_tris):
        ax.add_collection3d(Poly3DCollection(view(context_tris), facecolor=(.6, .6, .6, .25), edgecolor='none'))
    ax.add_collection3d(Poly3DCollection(view(unit_tris), facecolor=(1., .45, .05, 1.), edgecolor=(.5, .2, 0, .4), linewidths=.2))
    lo, hi = view(np.array([lo, hi]))
    center, half = (lo+hi)/2, max(hi-lo)/2
    ax.set_xlim(center[0]-half, center[0]+half)
    ax.set_ylim(center[1]-half, center[1]+half)
    ax.set_zlim(center[2]-half, center[2]+half)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('labels', type=Path, help='*.labels.json from enumerate_prims.py')
    ap.add_argument('--stage', type=Path, help='Defaults to the stage recorded in the labels file')
    ap.add_argument('--crops', type=Path, default=SEMANTICS_DIR/'crops')
    ap.add_argument('--context', type=float, default=1.5, help='Neighborhood radius as a multiple of the unit size')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--force', action='store_true', help='Redraw existing thumbnails')
    args = ap.parse_args()
    doc = json.loads(args.labels.read_text(encoding='utf-8'))
    stage = open_stage(args.stage or doc['stage'])
    y_up = UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
    groups = group_units(doc['units'])
    keys = sorted(groups)[:args.limit] if args.limit else sorted(groups)
    args.crops.mkdir(parents=True, exist_ok=True)

    xform_cache = UsdGeom.XformCache()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    meshes = []                                     # (prim, world min, world max) for neighbor lookup
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and is_rendered_gprim(prim):
            box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if not box.IsEmpty():
                meshes.append((prim, np.array(box.GetMin()), np.array(box.GetMax())))
    mins = np.array([m[1] for m in meshes])
    maxs = np.array([m[2] for m in meshes])
    rng = np.random.default_rng(0)
    tri_cache = {}

    def tris_of(prim):
        key = prim.GetPath()
        if key not in tri_cache:
            tri_cache[key] = world_triangles(prim, xform_cache)
        return tri_cache[key]

    drawn = 0
    for key in keys:
        rep = groups[key][0]
        out = args.crops/(unit_id(rep['path'])+'.png')
        if out.is_file() and not args.force:
            continue
        root = stage.GetPrimAtPath(rep['path'])
        own = [p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh) and is_rendered_gprim(p)]
        unit_tris = np.concatenate([tris_of(p) for p in own]) if own else np.zeros((0, 3, 3))
        if not len(unit_tris):
            continue
        lo, hi = unit_tris.reshape(-1, 3).min(0), unit_tris.reshape(-1, 3).max(0)
        pad = max(float(max(hi-lo))*args.context, 0.5)
        near_lo, near_hi = lo-pad, hi+pad
        own_paths = {p.GetPath() for p in own}
        near = np.where((maxs >= near_lo).all(1) & (mins <= near_hi).all(1))[0]
        context = [tris_of(meshes[i][0]) for i in near if meshes[i][0].GetPath() not in own_paths
                   and float(max(meshes[i][2]-meshes[i][1])) < 40*max(float(max(hi-lo)), 1.0)]   # skip sky/terrain shells
        context_tris = np.concatenate(context) if context else np.zeros((0, 3, 3))
        unit_tris = subsample(unit_tris, MAX_FACES, rng)
        context_tris = subsample(context_tris, MAX_FACES, rng)
        fig = plt.figure(figsize=(6.4, 3.2), dpi=100)
        for i, (elev, azim, lo_v, hi_v) in enumerate(((25, -60, near_lo, near_hi), (85, -90, lo-pad/3, hi+pad/3))):
            ax = fig.add_subplot(1, 2, i+1, projection='3d')
            draw(ax, unit_tris, context_tris, lo_v, hi_v, elev, azim, y_up)
        fig.subplots_adjust(0, 0, 1, 1, 0, 0)
        fig.savefig(out, facecolor='white')
        plt.close(fig)
        drawn += 1
    print('%d thumbnails drawn for %d groups -> %s' % (drawn, len(keys), args.crops))


if __name__ == '__main__':
    main()
