# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))
from ovrtx_blender_example import material_direct_updates as direct

@dataclass(frozen=True)
class Id:
    kind: str
    session_uid: int

def identity(obj, kind):
    return Id(kind, obj.session_uid)

@pytest.fixture
def setup(monkeypatch):
    a = NS(session_uid=1, graph='unchanged', values=(('inputs:base_color', 'Color3f', (1., 0., 0.)),))
    b = NS(session_uid=2, graph='unchanged', values=a.values)
    mesh = NS(session_uid=10, polygons=[NS(material_index=0)])
    obj = NS(session_uid=20, type='MESH', data=mesh, control=('fixed',), material_slots=[NS(material=a)])
    scene = NS(objects=[obj], materials=[a, b])
    monkeypatch.setattr(direct, '_helpers', lambda: (identity, lambda o: tuple(s.material for s in o.material_slots),
                                                   lambda o: tuple(s.material for s in o.material_slots), lambda o: o.control))
    monkeypatch.setattr(direct, '_control', lambda o: o.control)
    monkeypatch.setattr(direct, '_values', lambda m: m.values)
    monkeypatch.setattr(direct, 'material_graph_signature', lambda m: m.graph)
    paths = {Id('MATERIAL', 1): '/M/A', Id('MATERIAL', 2): '/M/B'}
    state = direct.MaterialState('unchanged', a.values, {'inputs:base_color': '/M/A/OpenPBR'})
    binding = direct.BindingState('/Objects/Mesh', '/M/A', obj.control, (0,))
    contract = direct.MaterialContract((1, 'digest'), {Id('MATERIAL', 1): state}, {Id('OBJECT', 20): binding}, paths)
    contract = replace(contract, material_handles={Id('MATERIAL', 1): a, Id('MATERIAL', 2): b},
                       object_handles={Id('OBJECT', 20): obj},
                       mesh_users={Id('MESH', 10): (Id('OBJECT', 20),)},
                       path_users={'/Objects/Mesh': (Id('OBJECT', 20),)})
    generation = NS(number=1, digest='digest')
    return scene, generation, contract, a, b, obj

def shading(key):
    return {key: NS(geometry=False, transform=False, shading=True)}

def test_constant_update_targets_exact_accepted_shader_and_acknowledges_only_on_success(setup):
    scene, generation, contract, a, _, _ = setup
    a.values = (('inputs:base_color', 'Color3f', (0., 0., 1.)),)
    result = direct.plan(scene, generation, {Id('MATERIAL', 1)}, contract)
    assert not result.fallback
    assert result.values == ({'prim_path': '/M/A/OpenPBR', 'attribute_name': 'inputs:base_color',
                              'value_type': 'Color3f', 'value': (0., 0., 1.)},)
    assert contract.materials[Id('MATERIAL', 1)].values != a.values
    assert direct.plan(scene, generation, {Id('MATERIAL', 1)}, result.next_contract).values == ()

def test_graph_change_rejects_whole_batch_even_with_other_valid_constant(setup):
    scene, generation, contract, a, b, _ = setup
    contract = replace(contract, materials={**contract.materials, Id('MATERIAL', 2):
                                           direct.MaterialState('unchanged', b.values, {'inputs:base_color': '/M/B/S'})})
    a.values = (('inputs:base_color', 'Color3f', (0., 0., 1.)),)
    b.graph = 'texture connected'
    result = direct.plan(scene, generation, {Id('MATERIAL', 1), Id('MATERIAL', 2)}, contract)
    assert result.values == () and result.handled == frozenset()
    assert 'material graph changed' in result.fallback

def test_existing_object_override_binding_and_unexported_target_fallback(setup):
    scene, generation, contract, _, b, obj = setup
    obj.material_slots = [NS(material=b, link='OBJECT')]
    key = Id('OBJECT', 20)
    result = direct.plan(scene, generation, {key}, contract, update_provenance=shading(key))
    assert result.values == ({'prim_path': '/Objects/Mesh', 'attribute_name': 'material:binding',
                              'value_type': 'Relationship', 'value': ['/M/B']},)
    obj.material_slots[0].material = NS(session_uid=999)
    result = direct.plan(scene, generation, {key}, contract, update_provenance=shading(key))
    assert result.values == () and result.fallback

def test_face_reassignment_or_geometry_notification_cannot_be_swallowed(setup):
    scene, generation, contract, _, b, obj = setup
    key = Id('OBJECT', 20); obj.material_slots[0].material = b
    assert direct.plan(scene, generation, {key}, contract).fallback
    obj.data.polygons[0].material_index = 1
    assert direct.plan(scene, generation, {key}, contract, update_provenance=shading(key)).fallback

def test_shared_mesh_override_conflict_fails_closed(setup):
    scene, generation, contract, a, b, obj = setup
    alias = NS(session_uid=21, type='MESH', data=obj.data, control=obj.control, material_slots=[NS(material=a)])
    scene.objects.append(alias)
    contract = replace(contract, bindings={**contract.bindings, Id('OBJECT', 21): contract.bindings[Id('OBJECT', 20)]},
                       object_handles={**contract.object_handles, Id('OBJECT', 21): alias},
                       path_users={'/Objects/Mesh': (Id('OBJECT', 20), Id('OBJECT', 21))})
    obj.material_slots[0].material = b
    key = Id('OBJECT', 20)
    result = direct.plan(scene, generation, {key}, contract, update_provenance=shading(key))
    assert 'shared USD mesh has conflicting object bindings' in result.fallback
    assert not result.values

def test_stale_contract_after_sparse_population_rejected(setup):
    scene, generation, contract, *_ = setup
    generation.number = 2
    assert direct.plan(scene, generation, {Id('MATERIAL', 1)}, contract).fallback == ('stale material contract',)

def test_composed_overlay_uses_runtime_openpbr_paths_and_generation_graph_guard(setup):
    from pxr import Usd, UsdShade, Sdf
    scene, generation, _, a, _, obj = setup
    obj.modifiers = ()
    stage = Usd.Stage.CreateInMemory()
    UsdShade.Material.Define(stage, '/M/A')
    mesh = stage.DefinePrim('/Objects/Mesh', 'Mesh')
    mesh.CreateRelationship('material:binding').SetTargets(['/M/A'])
    shader = UsdShade.Shader.Define(stage, '/M/A/Preview')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set((1., 0., 0.))
    generation.blender_prim_paths = {Id('MATERIAL', 1): NS(schema_path='/M/A'),
                                   Id('OBJECT', 20): NS(schema_path='/Objects/Mesh')}
    generation.material_graph_fingerprints = {Id('MATERIAL', 1): 'accepted graph'}
    overlay = NS(layer_body='''def Scope "RuntimeMaterials" {
        def Material "A" {
            def Shader "OpenPBR" {
                uniform token info:id = "ND_open_pbr_surface_surfaceshader"
                color3f inputs:base_color = (1, 0, 0)
            }
        }
    }
    over "Objects" {
        over "Mesh" {
            rel material:binding = </RuntimeMaterials/A>
        }
    }
    ''', diagnostics={'materials': [{'identity': {'material_path': '/M/A'},
                                    'generated_material_path': '/RuntimeMaterials/A'}]})
    captured = direct.capture_composed_contract(stage, scene, generation, overlay)
    state = captured.materials[Id('MATERIAL', 1)]
    assert state.targets['inputs:base_color'] == '/RuntimeMaterials/A/OpenPBR'
    assert captured.material_paths[Id('MATERIAL', 1)] == '/RuntimeMaterials/A'
    assert state.graph == 'accepted graph'
    assert direct.plan(scene, generation, {Id('MATERIAL', 1)}, captured).fallback == ('material graph changed',)


def test_preview_repair_overlay_lists_names_and_keeps_source_paths(setup):
    from pxr import Usd, UsdShade, Sdf
    scene, generation, _, a, _, obj = setup
    obj.modifiers = ()
    stage = Usd.Stage.CreateInMemory()
    UsdShade.Material.Define(stage, '/M/A')
    stage.DefinePrim('/Objects/Mesh', 'Mesh').CreateRelationship('material:binding').SetTargets(['/M/A'])
    shader = UsdShade.Shader.Define(stage, '/M/A/Preview')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set((1., 0., 0.))
    generation.blender_prim_paths = {Id('MATERIAL', 1): NS(schema_path='/M/A'),
                                   Id('OBJECT', 20): NS(schema_path='/Objects/Mesh')}
    generation.material_graph_fingerprints = {Id('MATERIAL', 1): 'accepted graph'}
    # usd_preview_emission_layer diagnostics: material names, no generated paths.
    overlay = NS(layer_body='''over "M" {
        over "A" {
            over "Preview" {
                color3f inputs:emissiveColor = (5, 5, 5)
            }
        }
    }
    ''', diagnostics={'source': 'usd_preview_material_repair', 'materials': ['A']})
    captured = direct.capture_composed_contract(stage, scene, generation, overlay)
    assert captured.material_paths[Id('MATERIAL', 1)] == '/M/A'


def test_object_override_on_modifier_owner_cannot_disappear_behind_mesh_data(setup):
    scene, generation, contract, a, b, obj = setup
    key = Id('OBJECT', 20)
    obj.data.materials = [a]
    obj.modifiers = [NS(type='BEVEL')]
    obj.material_slots[0].link = 'DATA'
    contract = replace(contract, bindings={}, effective_slots={key: direct.effective_slot_signature(obj)})
    obj.material_slots = [NS(material=b, link='OBJECT')]
    assert obj.data.materials == [a]
    assert direct.changed_effective_slots(contract, {key}) == frozenset({key})
    assert direct.changed_effective_slots(contract, {Id('MESH', 10)}) == frozenset({key})
    result = direct.plan(scene, generation, {key}, contract, update_provenance=shading(key))
    assert result.fallback and not result.values
    assert result.next_contract.effective_slots == contract.effective_slots


def test_binding_success_acknowledges_effective_slots_only_in_proposed_contract(setup):
    scene, generation, contract, _, b, obj = setup
    key = Id('OBJECT', 20)
    contract = replace(contract, effective_slots={key: direct.effective_slot_signature(obj)})
    obj.material_slots = [NS(material=b, link='OBJECT')]
    result = direct.plan(scene, generation, {key}, contract, update_provenance=shading(key))
    assert not result.fallback and result.values
    assert direct.changed_effective_slots(contract, {key}) == frozenset({key})
    assert direct.changed_effective_slots(result.next_contract, {key}) == frozenset()


def test_real_mesh_shading_controls_allow_only_visual_material_identity_change(monkeypatch):
    from ovrtx_blender_example import scene_generation as generation
    # Exercise the real nested signature, including SimReady's material UID.
    a = NS(session_uid=1, name_full='Blue')
    b = NS(session_uid=2, name_full='Red')
    data = NS(session_uid=10, name_full='Mesh', materials=[a], vertices=[], polygons=[])
    obj = NS(session_uid=20, name_full='Object', type='MESH', data=data,
             material_slots=[NS(material=a, link='DATA')], modifiers=[], constraints=[],
             users_collection=[], vertex_groups=[])
    original = direct._control(obj)
    obj.material_slots = [NS(material=b, link='OBJECT')]
    assert direct._control(obj) == original
    obj.hide_render = True
    assert direct._control(obj) != original
    obj.hide_render = False
    # An actual authored physics property keeps the binding change conservative.
    class PhysicsMaterial:
        session_uid = 2
        def __contains__(self, key):
            return key == 'pxr:usd:physics_density'
        def get(self, key, default=None):
            return 500 if key == 'pxr:usd:physics_density' else default
    obj.material_slots[0].material = PhysicsMaterial()
    assert direct._control(obj) != original


def test_bootstrap_targets_material_subtree_not_whole_stage_and_reuses_graph(setup, monkeypatch):
    from pxr import Usd, UsdShade, Sdf
    scene,generation,_,a,_,obj=setup
    obj.modifiers=()
    class NoFaces:
        def __iter__(self):raise AssertionError('bootstrap must not copy polygon slots')
    obj.data.polygons=NoFaces()
    stage=Usd.Stage.CreateInMemory()
    UsdShade.Material.Define(stage,'/M/A')
    shader=UsdShade.Shader.Define(stage,'/M/A/OpenPBR')
    shader.CreateIdAttr('ND_open_pbr_surface_surfaceshader')
    shader.CreateInput('base_color',Sdf.ValueTypeNames.Color3f).Set((1.,0.,0.))
    mesh=stage.DefinePrim('/Objects/Mesh','Mesh');mesh.CreateRelationship('material:binding').SetTargets(['/M/A'])
    stage.DefinePrim('/Unrelated/HeavyMesh','Mesh')
    generation.blender_prim_paths={Id('MATERIAL',1):NS(schema_path='/M/A'),Id('OBJECT',20):NS(schema_path='/Objects/Mesh')}
    generation.material_graph_fingerprints={Id('MATERIAL',1):'cached graph'}
    monkeypatch.setattr(direct,'material_graph_signature',lambda m:(_ for _ in ()).throw(AssertionError('cache hit rehashed graph')))
    class BoundedStage:
        def GetPrimAtPath(self,path):return stage.GetPrimAtPath(path)
        def Traverse(self):raise AssertionError('material bootstrap walked whole stage')
    captured=direct.capture_contract(BoundedStage(),scene,generation)
    assert captured.materials[Id('MATERIAL',1)].targets['inputs:base_color']=='/M/A/OpenPBR'
    assert captured.materials[Id('MATERIAL',1)].graph=='cached graph'
    assert captured.bindings[Id('OBJECT',20)].face_slots is None


def test_lazy_uniform_face_guard_reads_only_on_binding_event(setup):
    scene,generation,contract,_,b,obj=setup
    binding=replace(contract.bindings[Id('OBJECT',20)],face_slots=None)
    contract=replace(contract,bindings={Id('OBJECT',20):binding})
    obj.material_slots[0].material=b
    obj.data.polygons=[NS(material_index=0),NS(material_index=0)]
    result=direct.plan(scene,generation,{Id('OBJECT',20)},contract,update_provenance=shading(Id('OBJECT',20)))
    assert not result.fallback and result.values
    obj.data.polygons=[NS(material_index=1)]
    result=direct.plan(scene,generation,{Id('OBJECT',20)},contract,update_provenance=shading(Id('OBJECT',20)))
    assert result.fallback and not result.values
