# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Final-render preview route repairs constant emission stock export drops."""
from pathlib import Path
from types import SimpleNamespace as N
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'addon'))
from ovrtx_blender_example import usd_preview_emission_layer as preview  # noqa: E402
from ovrtx_blender_example import materialx_openpbr_conversion as materials  # noqa: E402

SCALE = materials._EMISSION_LUMINANCE_SCALE


def _export(tmp_path):
    """Stock-export shapes: Add Shader loses emission, Principled keeps it, Emission has no surface."""
    from pxr import Sdf, Usd
    stage = Usd.Stage.CreateNew(str(tmp_path / 'export.usda'))
    for name, emissive in (('add', None), ('principled', (5.0, 5.0, 5.0))):
        material = stage.DefinePrim(f'/World/_materials/{name}', 'Material')
        shader = stage.DefinePrim(f'/World/_materials/{name}/Principled_BSDF', 'Shader')
        shader.CreateAttribute('info:id', Sdf.ValueTypeNames.Token).Set('UsdPreviewSurface')
        shader.CreateAttribute('inputs:diffuseColor', Sdf.ValueTypeNames.Color3f).Set((0, 0, 0))
        if emissive is not None:
            shader.CreateAttribute('inputs:emissiveColor', Sdf.ValueTypeNames.Color3f).Set(emissive)
        shader.CreateAttribute('outputs:surface', Sdf.ValueTypeNames.Token)
        material.CreateAttribute('outputs:surface', Sdf.ValueTypeNames.Token).SetConnections(
            [shader.GetPath().AppendProperty('outputs:surface')])
    for name in ('emission', 'screen', 'dark'):
        stage.DefinePrim(f'/World/_materials/{name}', 'Material')
    for name in ('add', 'principled', 'emission', 'screen', 'dark'):
        mesh = stage.DefinePrim(f'/World/{name}_mesh', 'Mesh')
        mesh.CreateRelationship('material:binding').SetTargets([Sdf.Path(f'/World/_materials/{name}')])
    stage.GetRootLayer().Save()
    return tmp_path / 'export.usda'


VALUES = {
    'add': {'base_color': (0.0, 0.0, 0.0), 'emission_color': (1.0, 0.5, 0.25),
            'emission_luminance': 5 * SCALE, 'textures': {}},
    'principled': {'base_color': (0.0, 0.0, 0.0), 'emission_color': (1.0, 1.0, 1.0),
                   'emission_luminance': 5 * SCALE, 'textures': {}},
    'emission': {'base_color': (0.0, 0.0, 0.0), 'emission_color': (1.0, 1.0, 1.0),
                 'emission_luminance': 5 * SCALE, 'textures': {}},
    'screen': {'base_color': (0.0, 0.0, 0.0), 'emission_color': (1.0, 1.0, 1.0),
               'emission_luminance': 2 * SCALE,
               'textures': {'emission_color': {'asset_path': 'textures/screen.png', 'color_space': 'sRGB'}}},
    'dark': {'base_color': (0.0, 0.0, 0.0), 'emission_color': (1.0, 1.0, 1.0),
             'emission_luminance': 0.0, 'textures': {}},
}


@pytest.fixture
def layer(tmp_path, monkeypatch):
    pytest.importorskip('pxr')
    path = _export(tmp_path)
    monkeypatch.setattr(materials, '_classify_material',
                        lambda material, *_: {'openpbr_values': VALUES[material.name]})
    materials._MATERIALX_BINDING_IDENTITY_CACHE.clear()
    result = preview.scene_layer_from_materials([N(name=name) for name in VALUES], str(path))
    return path, result


def _composed(tmp_path, path, layer):
    from pxr import Usd
    overlay = tmp_path / 'overlay.usda'
    overlay.write_text('#usda 1.0\n' + layer.layer_body + '\n')
    root = tmp_path / 'root.usda'
    root.write_text(f'#usda 1.0\n(\n    subLayers = [@./{overlay.name}@, @./{path.name}@]\n)\n')
    return Usd.Stage.Open(str(root))


def test_add_shader_emission_gets_constant_emissive_color(tmp_path, layer):
    path, result = layer
    stage = _composed(tmp_path, path, result)
    shader = stage.GetPrimAtPath('/World/_materials/add/Principled_BSDF')
    assert tuple(shader.GetAttribute('inputs:emissiveColor').Get()) == pytest.approx((5.0, 2.5, 1.25))
    assert ('/World/_materials/add/Principled_BSDF', 'inputs:emissiveColor') in result.authored_properties


def test_exporter_authored_emission_is_left_alone(tmp_path, layer):
    path, result = layer
    stage = _composed(tmp_path, path, result)
    shader = stage.GetPrimAtPath('/World/_materials/principled/Principled_BSDF')
    assert tuple(shader.GetAttribute('inputs:emissiveColor').Get()) == pytest.approx((5.0, 5.0, 5.0))
    assert not any(prim.startswith('/World/_materials/principled') for prim, _ in result.authored_properties)


def test_surfaceless_emission_shader_gets_an_emissive_preview_surface(tmp_path, layer):
    from pxr import UsdShade
    path, result = layer
    stage = _composed(tmp_path, path, result)
    material = UsdShade.Material(stage.GetPrimAtPath('/World/_materials/emission'))
    surface = material.ComputeSurfaceSource()[0]
    assert surface.GetPath() == '/World/_materials/emission/OVRTX_Emission_Surface'
    assert surface.GetIdAttr().Get() == 'UsdPreviewSurface'
    assert tuple(surface.GetInput('emissiveColor').Get()) == pytest.approx((5.0, 5.0, 5.0))
    assert tuple(surface.GetInput('diffuseColor').Get()) == pytest.approx((0.0, 0.0, 0.0))


def test_surfaceless_textured_emission_reads_st_and_scales(tmp_path, layer):
    from pxr import UsdShade
    path, result = layer
    stage = _composed(tmp_path, path, result)
    surface = UsdShade.Material(stage.GetPrimAtPath('/World/_materials/screen')).ComputeSurfaceSource()[0]
    texture = UsdShade.Shader(surface.GetInput('emissiveColor').GetConnectedSources()[0][0].source.GetPrim())
    assert texture.GetPath() == '/World/_materials/screen/OVRTX_Emission_Texture'
    assert texture.GetInput('file').Get().path == 'textures/screen.png'
    assert tuple(texture.GetInput('scale').Get()) == pytest.approx((2.0, 2.0, 2.0, 1.0))
    reader = UsdShade.Shader(texture.GetInput('st').GetConnectedSources()[0][0].source.GetPrim())
    assert reader.GetIdAttr().Get() == 'UsdPrimvarReader_float2'
    assert reader.GetInput('varname').Get() == 'st'


def test_zero_strength_surfaceless_material_stays_skipped(layer):
    _path, result = layer
    assert {'material_name': 'dark', 'reason': 'missing_usd_preview_surface'} in result.diagnostics['skipped_materials']
    assert '_materials/dark' not in result.layer_body and '"dark"' not in result.layer_body


def test_emission_values_change_the_digest(monkeypatch, layer):
    path, first = layer
    brighter = {**VALUES, 'add': {**VALUES['add'], 'emission_luminance': 6 * SCALE}}
    monkeypatch.setattr(materials, '_classify_material',
                        lambda material, *_: {'openpbr_values': brighter[material.name]})
    second = preview.scene_layer_from_materials([N(name=name) for name in brighter], str(path))
    assert second.diagnostics['digest'] != first.diagnostics['digest']
    assert second.digest_content != first.digest_content


def test_scene_with_only_surfaceless_emitters_still_gets_a_layer(tmp_path, monkeypatch):
    pytest.importorskip('pxr')
    from pxr import Sdf, Usd
    stage = Usd.Stage.CreateNew(str(tmp_path / 'export.usda'))
    stage.DefinePrim('/World/_materials/emission', 'Material')
    stage.DefinePrim('/World/mesh', 'Mesh').CreateRelationship('material:binding').SetTargets(
        [Sdf.Path('/World/_materials/emission')])
    stage.GetRootLayer().Save()
    monkeypatch.setattr(materials, '_classify_material',
                        lambda material, *_: {'openpbr_values': VALUES['emission']})
    materials._MATERIALX_BINDING_IDENTITY_CACHE.clear()
    result = preview.scene_layer_from_materials([N(name='emission')], str(tmp_path / 'export.usda'))
    assert result is not None
    assert result.target_path == '/World/_materials/emission'


def test_surfaceless_emitter_keeps_classified_opacity(tmp_path, monkeypatch, layer):
    path, _ = layer
    faded = {**VALUES, 'emission': {**VALUES['emission'], 'geometry_opacity': 0.25}}
    monkeypatch.setattr(materials, '_classify_material',
                        lambda material, *_: {'openpbr_values': faded[material.name]})
    result = preview.scene_layer_from_materials([N(name=name) for name in faded], str(path))
    stage = _composed(tmp_path, path, result)
    surface = stage.GetPrimAtPath('/World/_materials/emission/OVRTX_Emission_Surface')
    assert surface.GetAttribute('inputs:opacity').Get() == pytest.approx(0.25)


def test_authored_black_emission_is_repaired(tmp_path, monkeypatch):
    pytest.importorskip('pxr')
    path = _export(tmp_path)
    from pxr import Sdf, Usd
    stage = Usd.Stage.Open(str(path))
    stage.GetPrimAtPath('/World/_materials/add/Principled_BSDF').CreateAttribute(
        'inputs:emissiveColor', Sdf.ValueTypeNames.Color3f).Set((0, 0, 0))
    stage.GetRootLayer().Save()
    monkeypatch.setattr(materials, '_classify_material',
                        lambda material, *_: {'openpbr_values': VALUES[material.name]})
    materials._MATERIALX_BINDING_IDENTITY_CACHE.clear()
    result = preview.scene_layer_from_materials([N(name='add')], str(path))
    composed = _composed(tmp_path, path, result)
    shader = composed.GetPrimAtPath('/World/_materials/add/Principled_BSDF')
    assert tuple(shader.GetAttribute('inputs:emissiveColor').Get()) == pytest.approx((5.0, 2.5, 1.25))


def test_real_classification_of_emission_and_add_shader_graphs(tmp_path):
    """Blender-shaped node graphs through the unstubbed classifier."""
    pytest.importorskip('pxr')
    from test_materialx_openpbr_conversion import _Material, _Node, _Socket, _link
    path = _export(tmp_path)
    emission = _Node('EMISSION', 'Emission', [_Socket('Color', (1.0, 1.0, 1.0, 1.0)), _Socket('Strength', 5.0)])
    pure = _Material('emission', [emission, _Node('OUTPUT_MATERIAL', 'Material Output',
                                                  [_link(_Socket('Surface'), emission)])])
    principled = _Node('BSDF_PRINCIPLED', 'Principled BSDF', [
        _Socket('Base Color', (0.0, 0.0, 0.0, 1.0)), _Socket('Metallic', 0.0), _Socket('Roughness', 0.5),
        _Socket('Alpha', 1.0), _Socket('Emission Color', (0.0, 0.0, 0.0, 1.0)), _Socket('Emission Strength', 0.0)])
    branch = _Node('EMISSION', 'Emission', [_Socket('Color', (1.0, 0.5, 0.25, 1.0)), _Socket('Strength', 5.0)])
    add = _Node('ADD_SHADER', 'Add Shader', [_link(_Socket('Shader', socket_type='SHADER'), principled),
                                             _link(_Socket('Shader', socket_type='SHADER'), branch)])
    mixed = _Material('add', [principled, branch, add, _Node('OUTPUT_MATERIAL', 'Material Output',
                                                               [_link(_Socket('Surface'), add)])])
    materials._MATERIALX_BINDING_IDENTITY_CACHE.clear()
    result = preview.scene_layer_from_materials([pure, mixed], str(path))
    composed = _composed(tmp_path, path, result)
    surface = composed.GetPrimAtPath('/World/_materials/emission/OVRTX_Emission_Surface')
    assert tuple(surface.GetAttribute('inputs:emissiveColor').Get()) == pytest.approx((5.0, 5.0, 5.0))
    shader = composed.GetPrimAtPath('/World/_materials/add/Principled_BSDF')
    assert tuple(shader.GetAttribute('inputs:emissiveColor').Get()) == pytest.approx((5.0, 2.5, 1.25))


def test_mapped_emission_texture_is_not_repaired_flat_or_unmapped(tmp_path, monkeypatch):
    pytest.importorskip('pxr')
    path = _export(tmp_path)
    mapped = {**VALUES['screen'], 'textures': {'emission_color': {
        'asset_path': 'textures/screen.png', 'color_space': 'sRGB',
        'uv_transform': {'scale': (2.0, 2.0), 'offset': (0.0, 0.0)}}}}
    monkeypatch.setattr(materials, '_classify_material', lambda material, *_: {'openpbr_values': mapped})
    materials._MATERIALX_BINDING_IDENTITY_CACHE.clear()
    result = preview.scene_layer_from_materials([N(name='screen')], str(path))
    assert result is None or 'screen' not in result.layer_body
