"""Final presentation shares one current composition without retained scene caches."""
from pathlib import Path
from types import SimpleNamespace as N
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'addon'))
from ovrtx_blender_example import usd_preview_emission_layer as preview
from ovrtx_blender_example import light_scene_layer as lights
from ovrtx_blender_example import materialx_openpbr_conversion as materials


def test_one_open_and_traversal_preserves_bindings_assets_and_presentation(tmp_path, monkeypatch):
    pxr = pytest.importorskip('pxr')
    from pxr import Usd, Sdf, UsdShade
    stage = Usd.Stage.CreateNew(str(tmp_path / 'source.usda'))
    material = stage.DefinePrim('/World/Materials/Paint', 'Material')
    shader = stage.DefinePrim('/World/Materials/Paint/Surface', 'Shader')
    shader.CreateAttribute('info:id', Sdf.ValueTypeNames.Token).Set('UsdPreviewSurface')
    shader.CreateAttribute('inputs:diffuseColor', Sdf.ValueTypeNames.Color3f).SetConnections([
        Sdf.Path('/World/Materials/Paint/Texture.outputs:rgb')])
    shader.CreateAttribute('inputs:opacity', Sdf.ValueTypeNames.Float).SetConnections([
        Sdf.Path('/World/Materials/Paint/Opacity.outputs:r')])
    for name, file in [('Texture', 'textures/Paint_SSS.png'),
                       ('Base', 'textures/Paint_BaseColor.png'),
                       ('Opacity', 'textures/Paint_opacity.png')]:
        texture = stage.DefinePrim('/World/Materials/Paint/' + name, 'Shader')
        texture.CreateAttribute('info:id', Sdf.ValueTypeNames.Token).Set('UsdUVTexture')
        texture.CreateAttribute('inputs:file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(file))
        if name == 'Opacity':
            texture.CreateAttribute('inputs:bias', Sdf.ValueTypeNames.Float4).Set((1, 1, 1, 1))
    uv = stage.DefinePrim('/World/Materials/Paint/UV', 'Shader')
    uv.CreateAttribute('info:id', Sdf.ValueTypeNames.Token).Set('UsdPrimvarReader_float2')
    for path, kind in [('/World/Geometry', 'Mesh'), ('/World/Geometry/Subset', 'GeomSubset')]:
        prim = stage.DefinePrim(path, kind)
        prim.CreateRelationship('material:binding').SetTargets([material.GetPath()])
    lamp = stage.DefinePrim('/World/Lamp', 'Xform')
    lamp.CreateAttribute('userProperties:blender:object_name', Sdf.ValueTypeNames.String).Set('Lamp')
    stage.DefinePrim('/World/Lamp/Light', 'SphereLight')
    stage.GetRootLayer().Save()
    path = str(tmp_path / 'source.usda')
    calls = {'open': 0, 'traverse': 0}
    def opened(path):
        calls['open'] += 1
        actual = Usd.Stage.Open(path)
        def traverse():
            calls['traverse'] += 1
            return actual.Traverse()
        return N(Traverse=traverse)
    monkeypatch.setattr(pxr, 'Usd', N(Stage=N(Open=opened)))
    materials._MATERIALX_BINDING_IDENTITY_CACHE.clear()
    old_identity = materials._load_materialx_binding_identity(path)
    old_preview = preview._usd_preview_material_index(path)
    old_lights = lights._stock_light_index(path)
    assert calls == {'open': 3, 'traverse': 3}
    calls.update(open=0, traverse=0)
    shared = preview.presentation_indexes(path)
    assert calls == {'open': 1, 'traverse': 1}
    assert shared.preview_materials == old_preview
    assert shared.lights == old_lights
    assert shared.binding_identity['bindings'] == old_identity['bindings']
    assert shared.binding_identity['material_paths'] == old_identity['material_paths']
    assert shared.preview_materials[str(material.GetPath())]['textures_by_path'][
        '/World/Materials/Paint/Base']['asset_path'] == 'textures/Paint_BaseColor.png'
    source_material = N(name='Paint')
    assert materials._resolve_binding(source_material, shared.binding_identity) == materials._resolve_binding(source_material, old_identity)
    monkeypatch.setattr(materials, '_classify_material', lambda *args: {
        'openpbr_values': {'textures': {'emission_color': {'asset_path': 'textures/emission.png'}},
                          'emission_luminance': materials._EMISSION_LUMINANCE_SCALE}})
    materials._MATERIALX_BINDING_IDENTITY_CACHE[path] = old_identity
    old_layer = preview.scene_layer_from_materials([source_material], path)
    calls.update(open=0, traverse=0)
    new_layer = preview.scene_layer_from_materials([source_material], path, presentation=shared)
    assert new_layer == old_layer
    assert calls == {'open': 0, 'traverse': 0}
    source_light = N(name='Lamp', type='LIGHT', data=N(type='POINT', energy=75.0,
                     color=(1.0, 0.8, 0.6), shadow_soft_size=0.15))
    new_light_layer = lights.scene_layer_from_lights([source_light], path, light_index=shared.lights)
    assert new_light_layer is not None
    assert new_light_layer == lights.scene_layer_from_lights([source_light], path)


def test_indexed_material_resolution_preserves_ambiguity_and_missing_targets():
    legacy = {'available': True, 'material_paths': ('/A/Paint', '/B/Paint'),
              'bindings': ({'material_path': '/A/Paint', 'binding_target': '/Mesh'},)}
    indexed = {**legacy, 'materials_by_name': {'Paint': ['/A/Paint', '/B/Paint']},
               'bindings_by_material': {'/A/Paint': legacy['bindings']}}
    for name in ('Paint', 'Missing'):
        material = N(name=name)
        assert materials._resolve_binding(material, indexed) == materials._resolve_binding(material, legacy)
    assert preview.presentation_indexes('') is None
