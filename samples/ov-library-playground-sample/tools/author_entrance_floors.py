# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Close the five R48 corridor-to-gallery floor gaps with matching stone infills."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))
from nvidia.ov_library_lab.bootstrap import configure
configure()
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

BASE_REVISION = 'r48-wayfinding-landscape'
# Existing artist assembly names are stable identities in the pinned campus.
CORRIDORS = {
    'interface': ('Pav_R34_world_R42Hallway_G40_interface_Floor2_floor_001', 0, -1),
    'storage': ('Pav_R34_world_R42Hallway_G40_storage_Floor2_floor_001', 0, 1),
    'streaming': ('Pav_R46_world_R46_streaming_Floor2_floor', 2, 1),
    'newton': ('Pav_R46_world_R46_newton_Floor2_floor', 2, 1),
    'reserved_east': ('Pav_R34_world_R42Hallway_G40_clash_Floor2_floor_001', 0, 1),
}


def top_mapping(mesh, points):
    """Recover the corridor's planar UV projection without stretching its finish."""
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    counts = mesh.GetFaceVertexCountsAttr().Get()
    uv = np.asarray(UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st').ComputeFlattened())
    assert len(uv) == len(indices), 'Expected face-varying corridor UVs'
    positions, values = [], []
    offset = 0
    for count in counts:
        face = points[indices[offset:offset+count]]
        if face[:, 1].min() >= -1e-5 and face[:, 1].max() > .02:
            positions.extend(face[:, [0, 2]])
            values.extend(uv[offset:offset+count])
        offset += count
    positions = np.asarray(positions)
    system = np.column_stack([positions, np.ones(len(positions))])
    mapping, _, rank, _ = np.linalg.lstsq(system, values, rcond=None)
    assert rank == 3 and np.max(np.abs(system@mapping-values)) < 1e-5
    return mapping


def main():
    registry = json.loads((ROOT/'config/campus-revisions.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campus', type=Path,
                        default=ROOT/'data/campus'/registry['revisions'][BASE_REVISION]['directory'])
    parser.add_argument('--output', type=Path,
                        default=ROOT/'data/scene-overrides/r51-entrance-floor-infills.usda')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    assert hashlib.sha256((args.campus/'manifest.json').read_bytes()).hexdigest() == \
        registry['revisions'][BASE_REVISION]['manifest_sha256'], 'Expected the pinned R48 campus'
    source = Usd.Stage.Open(str(args.campus/'runtime/environment.usda'))
    environment = source.GetPrimAtPath('/Environment')
    cache = UsdGeom.BBoxCache(0, ['default', 'render'])
    floors = []
    for prim in environment.GetChildren():
        if 'floor' not in prim.GetName().lower():
            continue
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if not bounds.IsEmpty() and abs(bounds.GetMax()[1]) < 1e-5:
            floors.append((prim, np.array(bounds.GetMin()), np.array(bounds.GetMax())))

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, 'Y')
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    stage.OverridePrim('/World')
    UsdGeom.Xform.Define(stage, '/World/EntranceFloorRepairs')
    report = []
    for key, (name, axis, sign) in CORRIDORS.items():
        assembly = source.GetPrimAtPath('/Environment/'+name)
        mesh = next(UsdGeom.Mesh(p) for p in Usd.PrimRange(assembly) if p.IsA(UsdGeom.Mesh))
        matrix = UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(0)
        points = np.array([matrix.Transform(Gf.Vec3d(*p)) for p in mesh.GetPointsAttr().Get()])
        end = points[np.isclose(points[:, 1], 0., atol=1e-5)]
        assert len(end) == 2 and np.ptp(end[:, axis]) < 1e-5
        lateral = 2 if axis == 0 else 0
        candidates = [(p, lo, hi) for p, lo, hi in floors
                      if (lo[axis] if sign > 0 else hi[axis])*sign >= 17.99
                      and (lo[axis] if sign > 0 else hi[axis])*sign < 18.01
                      and lo[lateral] < end[:, lateral].mean() < hi[lateral]]
        assert candidates, 'Missing adjacent finished floor: '+key
        adjacent, lo, hi = min(candidates, key=lambda row: (row[1][axis] if sign > 0 else -row[2][axis]))
        edge = float(lo[axis] if sign > 0 else hi[axis])
        gap = (edge-end[0, axis])*sign
        assert .29 < gap < .31, (key, gap)

        # End faces meet exactly; the original ramp, threshold trim and tiles
        # stay intact, with no coplanar overlay or raised step on the walking lane.
        far = end.copy()
        far[:, axis] = edge
        top = np.array([end[0], far[0], far[1], end[1]])
        top[:, 1] = 0.
        if np.cross(top[1]-top[0], top[2]-top[0])[1] < 0:
            top = top[::-1]
        bottom = top.copy()
        bottom[:, 1] = -.03
        vertices = np.concatenate([top, bottom])
        faces = [[0, 1, 2, 3], [7, 6, 5, 4]]
        faces.extend([[i, i+4, (i+1)%4+4, (i+1)%4] for i in range(4)])
        mapping = top_mapping(mesh, points)
        uvs = np.column_stack([vertices[:, [0, 2]], np.ones(8)])@mapping
        path = '/World/EntranceFloorRepairs/'+key
        infill = UsdGeom.Mesh.Define(stage, path)
        infill.CreateSubdivisionSchemeAttr('none')
        infill.CreatePointsAttr([Gf.Vec3f(*p) for p in vertices])
        infill.CreateFaceVertexCountsAttr([4]*6)
        infill.CreateFaceVertexIndicesAttr([v for face in faces for v in face])
        infill.CreateExtentAttr([Gf.Vec3f(*vertices.min(axis=0)), Gf.Vec3f(*vertices.max(axis=0))])
        normals = []
        for face in faces:
            a, b, c = vertices[face[:3]]
            normal = np.cross(b-a, c-a)
            normal /= np.linalg.norm(normal)
            normals.extend([Gf.Vec3f(*normal)]*4)
        infill.CreateNormalsAttr(normals)
        infill.SetNormalsInterpolation('faceVarying')
        UsdGeom.PrimvarsAPI(infill).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
                                               'vertex').Set([Gf.Vec2f(*uv) for uv in uvs])
        UsdShade.MaterialBindingAPI.Apply(infill.GetPrim())
        infill.GetPrim().CreateRelationship('material:binding').SetTargets(
            ['/World/Environment/_materials/pavilion_floor'])
        report.append(dict(path=path, corridor=str(mesh.GetPath()), adjacent_floor=str(adjacent.GetPath()),
                           gap_m=gap, width_m=float(np.ptp(end[:, lateral])),
                           minimum=vertices.min(axis=0).tolist(), maximum=vertices.max(axis=0).tolist(),
                           triangles=12, uv_projection= mapping.tolist()))

    stage.GetRootLayer().customLayerData = {
        'baseRevision': BASE_REVISION,
        'purpose': 'Continuous finished floors across five corridor-to-gallery thresholds',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(args.output))
    text = args.output.read_text(encoding='utf-8')
    header = ('# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n'
              '# SPDX-License-Identifier: Apache-2.0\n')
    args.output.write_text(text.replace('#usda 1.0\n', '#usda 1.0\n'+header, 1).rstrip()+'\n', encoding='utf-8')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    print('Authored five matched floor infills (60 triangles): '+str(args.output))


if __name__ == '__main__':
    main()
