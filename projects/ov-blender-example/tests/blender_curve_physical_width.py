# SPDX-License-Identifier: Apache-2.0
"""CPU-only actual Blender round-bevel diameter and scene-unit export checks."""
import json
from pathlib import Path
import sys
import bpy
from pxr import Gf, Usd, UsdGeom

sys.path.insert(0, str(Path(__file__).parents[1] / 'addon'))
from ovrtx_blender_example.scene_generation import _stock_export
out = Path(sys.argv[sys.argv.index('--') + 1])
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
data = bpy.data.curves.new('RoundBevel', 'CURVE')
data.dimensions, data.bevel_depth = '3D', .25
spline = data.splines.new('POLY')
spline.points.add(2)
for point, x, radius in zip(spline.points, (-2, 0, 2), (1, .5, 2)):
    point.co, point.radius = (x, 0, 0, 1), radius
obj = bpy.data.objects.new('RoundBevel', data)
scene.collection.objects.link(obj)
report = dict(blender=bpy.app.version_string, cases=[])
for scale in (1.0, .01):
    scene.unit_settings.system, scene.unit_settings.scale_length = 'METRIC', scale
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    diameters = []
    for x in (-2, 0, 2):
        ys = [v.co.y for v in mesh.vertices if abs(v.co.x - x) < 1e-4]
        diameters.append((max(ys) - min(ys)) * scale)
    evaluated.to_mesh_clear()
    path = out / f'curve-{scale}.usda'
    _stock_export(scene, path)
    stage = Usd.Stage.Open(str(path))
    checked = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != 'BasisCurves':
            continue
        curve = UsdGeom.BasisCurves(prim)
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        widths = list(curve.GetWidthsAttr().Get())
        physical = [value * Gf.Transform(matrix).GetScale()[1] for value in widths]
        assert len(physical) == len(diameters) == 3
        assert all(abs(left-right) < 1e-6 for left,right in zip(physical, diameters)), (physical, diameters)
        assert list(prim.GetAttribute('primvars:widths').Get()) == widths
        checked.append(dict(path=str(prim.GetPath()), widths=widths, physical_widths=physical))
    assert checked
    report['cases'].append(dict(scale_length=scale, evaluated_diameters=diameters, curves=checked))
report['status'] = 'passed'
(out / 'result.json').write_text(json.dumps(report, indent=2))
print('CURVE_PHYSICAL_WIDTH_PASS', json.dumps(report))
