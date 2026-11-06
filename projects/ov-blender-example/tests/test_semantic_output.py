# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
import base64
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).parents[1] / 'addon'))
from ovrtx_blender_example.semantic_output import SemanticCapture, capture_from_reply, decode_id_map, save_capture
from ovrtx_blender_example.semantic_aov import finalize_sidecar
from ovrtx_blender_example.semantic_identity import scene_labels, opinion

def id_map(identity=4294967295):
    label = b'class: class_a;'
    return struct.pack('<6I', identity, 0, 0, 0, len(label), 24) + label + struct.pack('<I', 1)

def capture():
    return SemanticCapture(2, 1, struct.pack('<2I', 0, 4294967295), id_map(), '/Render/Camera', 'epoch', 2, 17, 0.25, 0.25)

def reply():
    result = dict(width=2, height=1, sensor_path='/Render/Camera', epoch='epoch', ordinal=2, camera_prim_path='/World/Camera', simulation_time_ns=17, frame_start_time=0.25, frame_end_time=0.25)
    result['semantic'] = {**result, 'origin': 'top_left', 'id_dtype': 'uint32', 'id_map_dtype': 'uint8', 'id_map_base64': base64.b64encode(id_map()).decode()}
    return result

def test_lossless_uint32_palette_is_bound_to_exact_rgb(tmp_path):
    value = capture()
    result = SimpleNamespace(semantic_capture=value, width=2, height=1, simulation_time_ns=17)
    stem = tmp_path / 'frame.0010'
    labels = [dict(id='class_a', name='First class', source_ids=['A'])]
    receipt = save_capture(stem, result, labels, camera='/World/Camera', frame=10)
    assert receipt['verified']
    assert np.load(receipt['ids_path']).tolist() == [[0, 4294967295]]
    assert receipt['metadata']['camera'] == '/World/Camera'
    rgb = tmp_path / 'frame.0010.jpg'
    rgb.write_bytes(b'exact final display image')
    metadata = finalize_sidecar(rgb, stem)
    assert metadata['labels'][0]['native_ids'] == [4294967295]
    assert metadata['rgb_filename'] == rgb.name
    assert (tmp_path / 'frame.0010.semantic.png').read_bytes().startswith(b'\x89PNG')
    assert metadata == json.loads((tmp_path / 'frame.0010.semantic.json').read_text())


def test_reserved_unlabeled_id_preserves_native_pixels_and_map(tmp_path):
    value = replace(capture(), ids=struct.pack('<2I', 1, 4294967295))
    result = SimpleNamespace(semantic_capture=value, width=2, height=1, simulation_time_ns=17)
    stem = tmp_path / 'mixed'
    receipt = save_capture(stem, result, [dict(id='class_a')], camera='/World/Camera', frame=10)
    assert np.load(receipt['ids_path']).tolist() == [[1, 4294967295]]
    assert receipt['metadata']['native_id_map'] == {4294967295: 'class: class_a;'}
    assert receipt['metadata']['reserved_native_ids'] == {'0': 'BACKGROUND', '1': 'UNLABELLED'}
    rgb = tmp_path / 'mixed.jpg'
    rgb.write_bytes(b'final RGB')
    metadata = finalize_sidecar(rgb, stem)
    assert metadata['labels'][0]['native_ids'] == [4294967295]
    assert metadata['reserved_native_ids']['1'] == 'UNLABELLED'


@pytest.mark.parametrize('unmapped', [2, 4, 4294967294])
def test_nonreserved_unmapped_native_ids_still_fail(tmp_path, unmapped):
    value = replace(capture(), ids=struct.pack('<2I', unmapped, 4294967295))
    result = SimpleNamespace(semantic_capture=value, width=2, height=1, simulation_time_ns=17)
    stem = tmp_path / 'invalid'
    with pytest.raises(ValueError, match='unmapped native ID'):
        save_capture(stem, result, [dict(id='class_a')], camera='/World/Camera', frame=10)
    assert not Path(str(stem) + '.ids.npy').exists()


def test_unlabeled_only_is_not_a_verified_labeled_capture(tmp_path):
    value = replace(capture(), ids=struct.pack('<2I', 0, 1))
    result = SimpleNamespace(semantic_capture=value, width=2, height=1, simulation_time_ns=17)
    with pytest.raises(ValueError, match='no recognized labeled regions'):
        save_capture(tmp_path / 'unlabeled', result, [dict(id='class_a')], camera='/World/Camera', frame=10)

@pytest.mark.parametrize('key,value', [('epoch', 'old'), ('sensor_path', '/Other'), ('ordinal', 1), ('simulation_time_ns', 16), ('frame_end_time', 0.24), ('width', True), ('origin', 'bottom_left'), ('id_dtype', 'float32')])
def test_reject_stale_or_mistyped_semantic_frame(key, value):
    data = reply()
    data['semantic'][key] = value
    with pytest.raises(ValueError, match='identity'):
        capture_from_reply(data, capture().ids, width=2, height=1, sensor_path='/Render/Camera')

def test_mutable_payload_is_owned_and_truncated_idmap_rejected():
    pixels = bytearray(capture().ids)
    value = replace(capture(), ids=pixels)
    pixels[:] = bytes(8)
    assert value.ids == capture().ids
    with pytest.raises(ValueError):
        replace(value, id_map=b'x')
    with pytest.raises(ValueError):
        decode_id_map(struct.pack('<I', 999))

def test_unknown_class_never_writes_completed_sidecars(tmp_path):
    result = SimpleNamespace(semantic_capture=capture(), width=2, height=1, simulation_time_ns=17)
    with pytest.raises(ValueError, match='unknown semantic class'):
        save_capture(tmp_path / 'frame', result, [], camera='/World/Camera', frame=1)
    assert not (tmp_path / 'frame.ids.npy').exists()

def test_identity_deduplicates_shared_classes_and_authors_parent_schema():

    class Obj(dict):

        def __init__(self, name):
            super().__init__(ovrtx_semantic_class='class_a', ovrtx_semantic_name='First class')
            self.name_full = name
    left, right = (Obj('Left'), Obj('Right'))
    labels = scene_labels(SimpleNamespace(objects=[right, left]))
    assert labels == (dict(id='class_a', name='First class', source_ids=['Left', 'Right']),)
    pytest.importorskip('pxr')
    record = opinion(left, '/World/Left')
    assert 'SemanticsAPI:class' in record.layer_text
    assert 'semantic:class:params:semanticData' in record.layer_text


def test_generation_initial_and_value_candidate_compile_semantic_object_opinions(tmp_path):
    from ovrtx_blender_example import scene_generation
    from pxr import Sdf, Usd
    class Obj(dict):
        def __init__(self):
            super().__init__(ovrtx_semantic_class='class_a')
            self.name_full='Mesh'; self.session_uid=11; self.type='MESH'
            self.data=SimpleNamespace(name_full='Geometry',session_uid=12,materials=[])
            self.parent=None; self.material_slots=[]
    obj=Obj()
    scene=SimpleNamespace(objects=[obj])
    mapping=scene_generation.BlenderPrimPath('Mesh','MESH','/World/Mesh','/World/Mesh/Geometry')
    mappings={scene_generation.blender_id(obj,'OBJECT'):mapping}
    records=scene_generation._compile_add_on_opinions(scene,mappings,tmp_path/'unused.usda')
    semantic=next(record for record in records if record.usd_prim_path=='/World/Mesh')
    layer=Sdf.Layer.CreateAnonymous('.usda'); layer.ImportFromString(semantic.layer_text)
    stage=Usd.Stage.Open(layer)
    assert stage.GetPrimAtPath('/World/Mesh').GetAttribute('semantic:class:params:semanticData').Get()=='class_a'
    before=scene_generation._mesh_render_inputs(obj,obj.data,include_elements=False)
    obj['ovrtx_semantic_class']='class_b'
    assert scene_generation._mesh_render_inputs(obj,obj.data,include_elements=False)!=before
    changed=scene_generation._compile_add_on_opinions(scene,mappings,tmp_path/'unused.usda')
    semantic=next(record for record in changed if record.usd_prim_path=='/World/Mesh')
    assert 'class_b' in semantic.layer_text


@pytest.mark.parametrize("generated", [False, True])
def test_build_spec_defines_every_semantic_target_on_existing_blender_product(tmp_path, monkeypatch, generated):
    from dataclasses import replace
    from pxr import Usd
    from ovrtx_blender_example import ovrtx_session, ovrtx_scene_composition
    from ovrtx_blender_example.render_requests import RenderRequest
    # A stock-style product begins with only a color variable. Testing compose
    # directly would miss the build_spec request-to-product boundary.
    sensor='/Render/OmniverseKit/HydraTextures/ViewportTexture0'
    source=tmp_path/'existing.usda'
    source.write_text("""#usda 1.0
 def Xform "World" {
     def Camera "Camera" { }
 }
 def Scope "Render" {
     def Scope "OmniverseKit" {
         def Scope "HydraTextures" {
             def RenderProduct "ViewportTexture0" {
                 rel camera = </World/Camera>
                 rel orderedVars = [<HdrColor>]
                 def RenderVar "HdrColor" {
                     uniform string sourceName = "HdrColor"
                 }
             }
         }
     }
 }
 """)
    monkeypatch.setenv('OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR',str(tmp_path/'work'))
    request=RenderRequest(input_usd_path=str(source),current_scene_generation=generated,
        camera_prim_path='/World/Camera',sensor_paths=(sensor,),selected_sensor_paths=(sensor,),
        width=32,height=24,semantic_output_path=str(tmp_path/'capture'),
        color_presentation={'render_var':'HdrColor'})
    spec=ovrtx_session.build_spec(request)
    composition=spec.ovrtx_scene_composition
    stage=Usd.Stage.Open(composition.composed_scene_path)
    assert stage is not None
    targets=stage.GetPrimAtPath(sensor).GetRelationship('orderedVars').GetTargets()
    assert [path.name for path in targets]==['HdrColor','SemanticSegmentation','SemanticIdMap']
    for path in targets:
        prim=stage.GetPrimAtPath(path)
        assert prim and prim.IsDefined() and prim.GetTypeName()=='RenderVar'
        assert prim.GetAttribute('sourceName').Get()==path.name
    # Destination changes do not change products. Toggling requested products
    # does, and diagnostics must match the actual authored composition digest.
    renamed=ovrtx_session.build_spec(replace(request,semantic_output_path=str(tmp_path/'other')))
    color_only=ovrtx_session.build_spec(replace(request,semantic_output_path=''))
    assert composition.digest==renamed.ovrtx_scene_composition.digest
    assert composition.digest!=color_only.ovrtx_scene_composition.digest
    assert ovrtx_scene_composition.diagnostics(composition, request=request)['composition_digest']==composition.digest
