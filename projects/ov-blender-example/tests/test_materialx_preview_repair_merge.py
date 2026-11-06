# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MaterialX presentation keeps USD Preview repairs for materials it did not convert."""
from pathlib import Path
from types import SimpleNamespace as N
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'addon'))
from ovrtx_blender_example import usd_preview_emission_layer as preview  # noqa: E402
from ovrtx_blender_example import materialx_openpbr_conversion as materials  # noqa: E402
from ovrtx_blender_example.render_requests import MaterialPresentationLayer  # noqa: E402
from ovrtx_blender_example import material_direct_updates  # noqa: E402

MATERIALX_BODY = '''def Scope "OVRTX_Materials"
{
    def Material "Paint"
    {
    }
}

over "World"
{
    over "paint_mesh"
    {
        rel material:binding = </OVRTX_Materials/Paint>
    }
}'''


def _stock_export(tmp_path):
    """Ads: Blender writes Invert(Opacity)->Transmission as opacity = 1 - texture."""
    from pxr import Sdf, Usd
    stage = Usd.Stage.CreateNew(str(tmp_path / 'export.usda'))
    for name in ('Ads', 'Paint'):
        material = stage.DefinePrim(f'/World/_materials/{name}', 'Material')
        surface = stage.DefinePrim(f'/World/_materials/{name}/Principled_BSDF', 'Shader')
        surface.CreateAttribute('info:id', Sdf.ValueTypeNames.Token).Set('UsdPreviewSurface')
        texture = stage.DefinePrim(f'/World/_materials/{name}/Opacity_Texture', 'Shader')
        texture.CreateAttribute('info:id', Sdf.ValueTypeNames.Token).Set('UsdUVTexture')
        texture.CreateAttribute('inputs:file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f'textures/{name}_Opacity.png'))
        texture.CreateAttribute('inputs:bias', Sdf.ValueTypeNames.Float4).Set((1, 0, 0, 0))
        texture.CreateAttribute('inputs:scale', Sdf.ValueTypeNames.Float4).Set((-1, 1, 1, 1))
        texture.CreateAttribute('outputs:r', Sdf.ValueTypeNames.Float)
        surface.CreateAttribute('inputs:opacity', Sdf.ValueTypeNames.Float).SetConnections(
            [texture.GetPath().AppendProperty('outputs:r')])
        surface.CreateAttribute('outputs:surface', Sdf.ValueTypeNames.Token)
        material.CreateAttribute('outputs:surface', Sdf.ValueTypeNames.Token).SetConnections(
            [surface.GetPath().AppendProperty('outputs:surface')])
        stage.DefinePrim(f'/World/{name.lower()}_mesh', 'Mesh').CreateRelationship('material:binding').SetTargets(
            [material.GetPath()])
    stage.GetRootLayer().Save()
    return tmp_path / 'export.usda'


def _conversion(paint_status='generated', body=MATERIALX_BODY):
    layer = MaterialPresentationLayer(
        target_path='/World/paint_mesh', layer_body=body,
        authored_properties=(('/World/paint_mesh', 'material:binding'),),
        digest_content={'source': 'materialx_openpbr', 'digest': 'mx', 'layer_body': body},
        diagnostics={'source': 'materialx_openpbr',
                     'materials': [{'material_name': 'Paint', 'status': paint_status,
                                    'identity': {'material_path': '/World/_materials/Paint'},
                                    'generated_material_path': '/OVRTX_Materials/Paint'}]})
    return N(value=layer, diagnostics={'materials': [
        {'material_name': 'Ads', 'status': 'unsupported_no_entry'},
        {'material_name': 'Paint', 'status': paint_status}]})


@pytest.fixture
def merged(tmp_path, monkeypatch):
    pytest.importorskip('pxr')
    path = _stock_export(tmp_path)
    monkeypatch.setattr(materials, '_classify_material', lambda material, *_: {'openpbr_values': {}})
    materials._MATERIALX_BINDING_IDENTITY_CACHE.clear()
    layer = preview.with_repairs_for_unconverted(_conversion(), [N(name='Ads'), N(name='Paint')], str(path))
    return path, layer


def _compose(tmp_path, path, layer):
    from pxr import Usd
    overlay = tmp_path / 'overlay.usda'
    overlay.write_text('#usda 1.0\n' + layer.layer_body + '\n')
    root = tmp_path / 'root.usda'
    root.write_text(f'#usda 1.0\n(\n    subLayers = [@./{overlay.name}@, @./{path.name}@]\n)\n')
    return Usd.Stage.Open(str(root))


def test_rejected_material_gets_the_opacity_repair_and_converted_one_does_not(tmp_path, merged):
    path, layer = merged
    stage = _compose(tmp_path, path, layer)
    ads = stage.GetPrimAtPath('/World/_materials/Ads/Opacity_Texture')
    assert tuple(ads.GetAttribute('inputs:bias').Get()) == (0, 0, 0, 0)
    assert tuple(ads.GetAttribute('inputs:scale').Get()) == (1, 1, 1, 1)
    paint = stage.GetPrimAtPath('/World/_materials/Paint/Opacity_Texture')
    assert tuple(paint.GetAttribute('inputs:scale').Get()) == (-1, 1, 1, 1)
    assert not any(prim.startswith('/World/_materials/Paint') for prim, _ in layer.authored_properties)


def test_merged_layer_keeps_the_materialx_overlay_and_records(tmp_path, merged):
    path, layer = merged
    stage = _compose(tmp_path, path, layer)
    assert stage.GetPrimAtPath('/OVRTX_Materials/Paint').IsA(__import__('pxr').UsdShade.Material)
    binding = stage.GetPrimAtPath('/World/paint_mesh').GetRelationship('material:binding').GetTargets()
    assert [str(target) for target in binding] == ['/OVRTX_Materials/Paint']
    assert layer.diagnostics['materials'][0]['generated_material_path'] == '/OVRTX_Materials/Paint'
    assert layer.diagnostics['preview_repair']['materials'] == ['Ads']
    assert ('/World/paint_mesh', 'material:binding') in layer.authored_properties


def test_digest_changes_with_either_part(tmp_path, monkeypatch, merged):
    path, layer = merged
    other_mx = preview.with_repairs_for_unconverted(
        _conversion(body=MATERIALX_BODY.replace('"Paint"\n    {\n    }', '"Paint"\n    {\n        custom int v = 2\n    }')),
        [N(name='Ads'), N(name='Paint')], str(path))
    assert other_mx.digest_content != layer.digest_content
    no_repair = preview.with_repairs_for_unconverted(_conversion(), [N(name='Paint')], str(path))
    assert no_repair.digest_content != layer.digest_content
    assert 'preview_repair' not in no_repair.digest_content


def test_without_a_materialx_layer_the_repairs_stand_alone(tmp_path, monkeypatch):
    pytest.importorskip('pxr')
    path = _stock_export(tmp_path)
    monkeypatch.setattr(materials, '_classify_material', lambda material, *_: {'openpbr_values': {}})
    materials._MATERIALX_BINDING_IDENTITY_CACHE.clear()
    conversion = N(value=None, diagnostics={'materials': [{'material_name': 'Ads', 'status': 'unsupported_no_entry'}]})
    layer = preview.with_repairs_for_unconverted(conversion, [N(name='Ads')], str(path))
    assert layer is not None and layer.diagnostics['source'] == 'usd_preview_material_repair'


def test_direct_material_contract_reads_the_merged_layer(tmp_path, merged):
    path, layer = merged
    from pxr import Usd
    stage = Usd.Stage.Open(str(path))
    captured = {}

    def capture(stage_, scene, generation, material_path_overrides=None):
        captured.update(material_path_overrides or {})
        return 'contract'

    original = material_direct_updates.capture_contract
    material_direct_updates.capture_contract = capture
    try:
        assert material_direct_updates.capture_composed_contract(stage, None, None, layer) == 'contract'
    finally:
        material_direct_updates.capture_contract = original
    assert captured == {'/World/_materials/Paint': '/OVRTX_Materials/Paint'}
