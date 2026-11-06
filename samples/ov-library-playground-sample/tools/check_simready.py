# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check prepared props, material closure, bounds and the operator's movement."""
import json
from pathlib import Path
import sys
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from nvidia.ov_library_lab.scene import Gallery


def check():
    directory=ROOT/'data/models/simready'
    index=json.loads((directory/'index.json').read_text())
    report={'assets':{},'checks':[]}
    for key,info in index['assets'].items():
        stage=Usd.Stage.Open(str(directory/info['file']))
        assert UsdGeom.GetStageUpAxis(stage)=='Y' and UsdGeom.GetStageMetersPerUnit(stage)==1.
        meshes=vertices=textures=0
        for prim in stage.Traverse():
            assert not prim.HasAPI(UsdPhysics.RigidBodyAPI) and not prim.HasAPI(UsdPhysics.CollisionAPI),prim.GetPath()
            if prim.IsA(UsdGeom.Mesh):
                mesh=UsdGeom.Mesh(prim);meshes+=1;vertices+=len(mesh.GetPointsAttr().Get())
                valid,why=UsdGeom.Mesh.ValidateTopology(mesh.GetFaceVertexIndicesAttr().Get(),mesh.GetFaceVertexCountsAttr().Get(),len(mesh.GetPointsAttr().Get()))
                assert valid,why
                uv=UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st')
                if uv:
                    expected=sum(mesh.GetFaceVertexCountsAttr().Get()) if uv.GetInterpolation()=='faceVarying' else len(mesh.GetPointsAttr().Get())
                    assert len(uv.ComputeFlattened())==expected,(key,prim.GetPath(),'UV topology')
            for attr in prim.GetAttributes():
                if attr.GetTypeName()!=Sdf.ValueTypeNames.Asset or not attr.Get():continue
                value=attr.Get()
                path=Path(value.resolvedPath or value.path)
                if not path.is_absolute():path=(Path(attr.GetPropertyStack()[0].layer.realPath).parent/path).resolve()
                files=list(path.parent.glob(path.name.replace('<UDIM>','*'))) if '<UDIM>' in path.name else [path]
                assert files and all(p.is_file() for p in files),(key,value.path)
                textures+=len(files)
        assert meshes>0
        report['assets'][key]=dict(meshes=meshes,vertices=vertices,texture_references=textures,bytes=(directory/info['file']).stat().st_size)
    assert report['assets']['amr']['vertices']+report['assets']['amr_load']['vertices']<160000
    assert report['assets']['conveyor']['vertices']<20000
    report['checks'].append('Prepared props: valid geometry and UVs, closed textures, metre/Y-up units, no imported physics')
    g=Gallery(Usd.Stage.CreateInMemory(),ROOT/'data')
    e=g.extra
    positions=[]
    for value in (0.,.5,1.):
        e.set_lift(value)
        positions.append(g.position('/World/Interface/Lift')[1])
        assert abs(e.lift_table.rise-value*1.51)<1e-6
    assert abs(positions[-1]-positions[0]-1.51)<1e-6
    report['checks'].append('Shared operator model moves fork carriage and pallet through the source mast travel')
    for n in range(3):
        e.set_asset_variant(n)
        assert e.working.GetPrim().GetAttribute('lab:variant').Get()==n
        block=e.working_blocks[n]
        cartons=[p for p in block.GetPrim().GetChildren() if p.GetName().startswith('Stock')]
        assert len(cartons)==n*3
    report['checks'].append('Empty, pick-ready and replenished rack configurations contain 0/3/6 cartons')
    # The larger cargo module must fit the route without overlapping another
    # robot or the protective bollards at every end of its commanded travel.
    for count in (1, 3, 6):
        g.workshops.set_fleet(count)
        for offset in (-2.4, 0., 2.4):
            g.shared_offset = offset
            g.workshops.update_fleet()
            bounds = UsdGeom.BBoxCache(0, ['default'])
            route = bounds.ComputeWorldBound(g.stage.GetPrimAtPath('/World/Shared/Route')).ComputeAlignedRange()
            robots = (g.shared, *g.workshops.fleet)[:count]
            occupied = [bounds.ComputeWorldBound(robot.GetPrim()).ComputeAlignedRange() for robot in robots]
            bollards = [bounds.ComputeWorldBound(g.stage.GetPrimAtPath(f'/World/Shared/Bollard{i}')).ComputeAlignedRange() for i in range(2)]
            def overlap(a, b):
                return all(min(a.GetMax()[j], b.GetMax()[j]) - max(a.GetMin()[j], b.GetMin()[j]) > .001 for j in (0, 2))
            for i, a in enumerate(occupied):
                assert all(route.GetMin()[j] <= a.GetMin()[j] and a.GetMax()[j] <= route.GetMax()[j] for j in (0, 2)), ('Route containment', count, offset, i)
                assert not any(overlap(a, b) for b in occupied[i+1:] + bollards), ('Fleet clearance', count, offset, i)
    report['checks'].append('One, three and six loaded trolleys remain separated and on the route throughout travel')
    return report


if __name__=='__main__':
    result=check()
    if len(sys.argv)>1:
        path=Path(sys.argv[1]);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    for item in result['checks']:print('PASS',item)
