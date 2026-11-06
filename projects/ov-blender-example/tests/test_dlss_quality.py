# SPDX-License-Identifier: Apache-2.0
"""One canonical quality policy reaches global composition and native startup."""
from dataclasses import replace
import json
from types import SimpleNamespace as NS
import pytest
from ovrtx_blender_example import ovrtx_session, rtpt_worker_config
from ovrtx_blender_example.properties import DLSS_EXECMODE_ATTRIBUTE, resolve_dlss_quality
from ovrtx_blender_example.direct_sdk_client import PublicSdkRenderClient
from test_dlss_toggle import (_live_request, _presentation_text, _render_scene,
    BlenderRenderSignal, BlenderRenderSignalSource, BlenderRenderIntent,
    RenderRequestTranslator)

@pytest.mark.parametrize('quality,carb', [('performance',0), ('balanced',1), ('quality',2), ('auto',3)])
def test_global_quality_authors_native_token_and_legacy_carb_int(tmp_path, monkeypatch, quality, carb):
    monkeypatch.setenv('OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR', str(tmp_path/'work'))
    request = _live_request(tmp_path, dlss_quality=quality)
    spec = ovrtx_session.build_spec(request)
    assert f'token {DLSS_EXECMODE_ATTRIBUTE} = "{quality}"' in _presentation_text(spec.ovrtx_scene_composition)
    client = PublicSdkRenderClient.__new__(PublicSdkRenderClient)
    client.request = request
    native_spec = NS(ovrtx_scene_composition=spec.ovrtx_scene_composition, stage_metadata={},
        sensor_paths=spec.sensor_paths, width=320, height=180, render_var='LdrColor')
    # Retained authoring primitives are not runtime admission for this candidate.
    from ovrtx_blender_example.ovrtx_runtime_client import RenderClientError
    with pytest.raises(RenderClientError, match='not admitted'):
        client._load_params(native_spec)
    tree = json.loads(rtpt_worker_config.compose_worker_config('{}', None, True, quality))
    assert tree['rtx']['post']['dlss']['execMode'] == carb


def test_quality_and_reset_recreate_products_and_preserve_inherit(tmp_path, monkeypatch):
    monkeypatch.setenv('OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR', str(tmp_path/'work'))
    specs = [ovrtx_session.build_spec(_live_request(tmp_path, dlss_quality=value))
        for value in ('INHERIT','quality','performance','INHERIT')]
    assert len({spec.ovrtx_scene_composition.digest for spec in specs[:3]}) == 3
    assert specs[0].ovrtx_scene_composition.digest == specs[-1].ovrtx_scene_composition.digest
    assert DLSS_EXECMODE_ATTRIBUTE not in _presentation_text(specs[-1].ovrtx_scene_composition)
    assert ovrtx_session.reuse_decision(specs[1], specs[2]).reason == 'scene_composition_changed'


def test_canonical_scene_property_reaches_render_request():
    scene = _render_scene(dlss_quality='performance')
    request = RenderRequestTranslator().translate(BlenderRenderSignal(
        BlenderRenderSignalSource.VIEW_UPDATE, BlenderRenderIntent.VIEWPORT,
        scene, '', camera_prim_path='/Generated/Camera', render_product_path='/Generated/Product',
        context=NS(), current_scene_generation=True))
    assert request.dlss_quality == 'performance'


def test_explicit_quality_overrides_legacy_boolean_and_rejects_unknown():
    assert resolve_dlss_quality('quality', False) == 'quality'
    assert resolve_dlss_quality('INHERIT', False) == 'performance'
    assert resolve_dlss_quality('INHERIT', True) is None
    with pytest.raises(ValueError, match='Invalid DLSS'):
        resolve_dlss_quality('ultra-performance')
