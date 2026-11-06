# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'addon'))
from ovrtx_blender_example.scene_population_delta import compile_population_delta
Sdf=pytest.importorskip('pxr.Sdf')
Usd=pytest.importorskip('pxr.Usd')


def pair(tmp_path,a,b):
    old,new=tmp_path/'old.usda',tmp_path/'new.usda'
    old.write_text('#usda 1.0\n'+a.replace(' }','\n}'));new.write_text('#usda 1.0\n'+b.replace(' }','\n}'))
    return old,new


def test_scoped_geometry_binding_visibility_and_return(tmp_path):
    a='def Xform "Root" { def Mesh "Mesh" { point3f[] points=[(0,0,0)]\n rel material:binding=</Red>\n token visibility="inherited"\n } }\n def Material "Red" {}\n def Material "Blue" {}\n'
    b=a.replace('(0,0,0)','(1,2,3)').replace('</Red>','</Blue>').replace('"inherited"','"invisible"')
    old,new=pair(tmp_path,a,b)
    delta=compile_population_delta(old,new)
    assert not delta.fallback_reason and delta.changed_prims==1 and delta.changed_properties==3
    assert delta.changed_prim_paths == ('/Root/Mesh',)
    patch=Sdf.Layer.FindOrOpen(delta.path)
    assert patch.GetPrimAtPath('/Root/Mesh')
    assert not patch.GetPrimAtPath('/Red')
    root=Sdf.Layer.CreateAnonymous();root.subLayerPaths=[delta.path,str(old)]
    composed=Usd.Stage.Open(root);expected=Usd.Stage.Open(str(new))
    assert composed.GetPrimAtPath('/Root/Mesh').GetAttribute('points').Get()==expected.GetPrimAtPath('/Root/Mesh').GetAttribute('points').Get()
    assert composed.GetPrimAtPath('/Root/Mesh').GetRelationship('material:binding').GetTargets()==[Sdf.Path('/Blue')]
    reverse=compile_population_delta(new,old)
    assert not reverse.fallback_reason and reverse.changed_properties==3


def test_create_and_delete_prims(tmp_path):
    old,new=pair(tmp_path,'def Xform "Old" {}\n','def Sphere "New" { double radius=2 }\n')
    delta=compile_population_delta(old,new)
    assert not delta.fallback_reason
    assert delta.created_prims==delta.deleted_prims==1
    assert delta.changed_prim_paths == ('/New', '/Old')
    assert not Sdf.Layer.FindOrOpen(delta.path).GetPrimAtPath('/Old').active


@pytest.mark.parametrize('replacement,reason',[
    ('def Sphere "Mesh" {}\n','existing_prim_schema_changed'),
    ('def Mesh "Mesh" {}\n','existing_property_removed')])
def test_unsupported_changes_fall_back_before_mutation(tmp_path,replacement,reason):
    old,new=pair(tmp_path,'def Mesh "Mesh" { float custom=1 }\n',replacement)
    delta=compile_population_delta(old,new)
    assert delta.path is None and delta.fallback_reason==reason


def test_noop_overlay_has_no_changed_prims(tmp_path):
    old,new=pair(tmp_path,'def Mesh "Mesh" { float custom=1 }\n','def Mesh "Mesh" { float custom=1 }\n')
    delta=compile_population_delta(old,new)
    assert not delta.fallback_reason and delta.changed_prims==0


@pytest.mark.parametrize('field,first,second', [
    ('double3 xformOp:translate', '(0,0,0)', '(4,0,0)'),
    ('uniform token[] xformOpOrder', '["xformOp:translate"]', '["!resetXformStack!", "xformOp:translate"]'),
])
def test_existing_transform_stack_changes_require_complete_projection(tmp_path, field, first, second):
    old, new = pair(tmp_path,
                    f'def Xform "Parent" {{ {field} = {first}\n }}\n',
                    f'def Xform "Parent" {{ {field} = {second}\n }}\n')
    delta = compile_population_delta(old, new)
    assert delta.path is None
    assert delta.fallback_reason == 'existing_transform_requires_replacement'
    assert delta.candidate_snapshot is not None


def test_deleted_geometry_is_explicitly_reactivated_when_returning(tmp_path):
    old,new=pair(tmp_path,'def Xform "Root" { def Mesh "Mesh" { point3f[] points=[(0,0,0)]\n } }\n',
                 'def Xform "Root" { def Mesh "Mesh" (active=false) {}\n }\n')
    forward=compile_population_delta(old,new)
    # Inactivation can omit geometry properties; that route remains conservative.
    reverse=compile_population_delta(new,old)
    assert not reverse.fallback_reason
    layer=Sdf.Layer.FindOrOpen(reverse.path)
    assert layer.GetPrimAtPath('/Root/Mesh').HasInfo('active')
    assert layer.GetPrimAtPath('/Root/Mesh').active
    composed=Sdf.Layer.CreateAnonymous();composed.subLayerPaths=[reverse.path,str(new)]
    assert Usd.Stage.Open(composed).GetPrimAtPath('/Root/Mesh').IsActive()


def test_new_geometry_sets_explicit_active_even_after_an_older_overlay(tmp_path):
    old,new=pair(tmp_path,'def Xform "Root" {}\n','def Xform "Root" { def Sphere "Sphere" { double radius=1\n } }\n')
    delta=compile_population_delta(old,new)
    assert not delta.fallback_reason
    layer=Sdf.Layer.FindOrOpen(delta.path)
    assert layer.GetPrimAtPath('/Root/Sphere').HasInfo('active')
    assert layer.GetPrimAtPath('/Root/Sphere').active


def test_delta_files_are_immutable_across_repeated_compilation(tmp_path):
    old,new=pair(tmp_path,'def Sphere "Ball" { double radius=1\n }\n','def Sphere "Ball" { double radius=2\n }\n')
    a=compile_population_delta(old,new);before=Path(a.path).read_bytes()
    b=compile_population_delta(old,new)
    assert a.path==b.path and Path(a.path).read_bytes()==before


def test_equal_standalone_image_bytes_do_not_rebind_unrelated_assets(tmp_path):
    from ovrtx_blender_example.scene_population_delta import capture_population_snapshot
    a,b=tmp_path/'a.png',tmp_path/'b.png';a.write_bytes(b'identical-image');b.write_bytes(a.read_bytes())
    old,new=pair(tmp_path,'def Shader "Shader" { asset texture=@a.png@\n }\n',
                 'def Shader "Shader" { asset texture=@b.png@\n }\n')
    snapshot=capture_population_snapshot(old)
    delta=compile_population_delta(old,new,accepted_snapshot=snapshot)
    assert not delta.fallback_reason and delta.changed_prims==0


def test_mutated_asset_at_same_filename_retains_conservative_replacement(tmp_path):
    from ovrtx_blender_example.scene_population_delta import capture_population_snapshot
    image=tmp_path/'a.png';image.write_bytes(b'first-image')
    old,new=pair(tmp_path,'def Shader "Shader" { asset texture=@a.png@\n }\n',
                 'def Shader "Shader" { asset texture=@a.png@\n }\n')
    snapshot=capture_population_snapshot(old);image.write_bytes(b'changed-image')
    delta=compile_population_delta(old,new,accepted_snapshot=snapshot)
    assert delta.path is None and delta.fallback_reason=='mutable_asset_dependency_changed'


def test_snapshot_camera_rebase_reuses_only_proven_unchanged_sources(tmp_path):
    from ovrtx_blender_example.scene_population_delta import capture_population_snapshot
    shared=tmp_path/'shared.usda';shared.write_text('#usda 1.0\ndef Mesh "Mesh" { float scalar=1\n }\n')
    a,b=tmp_path/'camera-a.usda',tmp_path/'camera-b.usda'
    a.write_text('#usda 1.0\n( subLayers=[@shared.usda@] )\ndef Camera "A" {}\n')
    b.write_text('#usda 1.0\n( subLayers=[@shared.usda@] )\ndef Camera "B" {}\n')
    previous=capture_population_snapshot(a)
    current=capture_population_snapshot(b,unchanged_source_snapshot=previous)
    assert current.sources[shared.as_posix()] is previous.sources[shared.as_posix()]
    assert current.source_path==str(b)
    assert any(key[0]=='/B' for key in current.fields)


def test_shared_snapshot_invalidates_when_layer_or_texture_changes(tmp_path, monkeypatch):
    from ovrtx_blender_example.scene_population_delta import capture_population_snapshot
    shared=tmp_path/'shared.usda'
    image=tmp_path/'a.png'; image.write_bytes(b'original')
    shared.write_text('#usda 1.0\ndef Shader "Shader" { asset texture=@a.png@\n float value=1\n}\n')
    root=tmp_path/'root.usda'; root.write_text('#usda 1.0\n( subLayers=[@shared.usda@] )\n')
    first=capture_population_snapshot(root)
    reused=capture_population_snapshot(root,unchanged_source_snapshot=first)
    assert reused.sources[shared.as_posix()] is first.sources[shared.as_posix()]
    image.write_bytes(b'changed')
    textured=capture_population_snapshot(root,unchanged_source_snapshot=reused)
    assert textured.sources[shared.as_posix()] is not reused.sources[shared.as_posix()]
    assert textured.assets[image.as_posix()] != first.assets[image.as_posix()]
    # Keep an open cached layer and edit disk without an explicit USD reload.
    # The snapshot must not certify stale parsed values with a fresh signature.
    cached_layer = Sdf.Layer.FindOrOpen(str(shared))
    before = shared.stat()
    real_stat = Path.stat
    monkeypatch.setattr(Path, 'stat', lambda path, *a, **k:
                        before if path == shared else real_stat(path, *a, **k))
    shared.write_text(shared.read_text().replace('value=1','value=2'))
    edited=capture_population_snapshot(root,unchanged_source_snapshot=textured)
    assert edited.sources[shared.as_posix()] is not textured.sources[shared.as_posix()]
    assert edited.fields[('/Shader','/Shader.value','default')]==[2.0]


def test_binary_overlay_preserves_large_arrays_and_reports_phases(tmp_path):
    old,new=pair(tmp_path,'def Mesh "Mesh" { int[] faceVertexIndices=[0,1,2]\n}\n',
        'def Mesh "Mesh" { int[] faceVertexIndices=['+','.join(str(i) for i in range(10000))+']\n}\n')
    delta=compile_population_delta(old,new)
    assert not delta.fallback_reason and Path(delta.path).suffix=='.usdc'
    assert len(Sdf.Layer.FindOrOpen(delta.path).GetAttributeAtPath('/Mesh.faceVertexIndices').default)==10000
    assert set(delta.phase_timings)=={'source_snapshot_ms','source_difference_ms','masked_composition_ms',
        'spec_comparison_ms','delta_serialization_ms'}


def test_incremental_contributors_handle_removed_layers_and_shared_reordering(tmp_path):
    from ovrtx_blender_example.scene_population_delta import capture_population_snapshot
    a,b=tmp_path/'a.usda',tmp_path/'b.usda'
    a.write_text('#usda 1.0\ndef Sphere "Ball" { double radius=1\n}\n')
    b.write_text('#usda 1.0\nover "Ball" { double radius=2\n}\n')
    def root(name,children):
        path=tmp_path/name
        path.write_text('#usda 1.0\n( subLayers=['+','.join('@'+p.name+'@' for p in children)+'] )\n')
        return path
    first=capture_population_snapshot(root('first.usda',[a,b]))
    reordered=capture_population_snapshot(root('reordered.usda',[b,a]),unchanged_source_snapshot=first)
    assert reordered.changed_field_keys is None
    assert reordered.fields[('/Ball','/Ball.radius','default')]==[2.0,1.0]
    removed=capture_population_snapshot(root('removed.usda',[b]),unchanged_source_snapshot=reordered)
    assert removed.fields[('/Ball','/Ball.radius','default')]==[2.0]
    assert ('/Ball','/Ball.radius','default') in removed.changed_field_keys


def test_snapshot_repeated_sublayer_contributors_keep_order(tmp_path):
    from ovrtx_blender_example.scene_population_delta import capture_population_snapshot
    a,b=tmp_path/'a.usda',tmp_path/'b.usda'
    a.write_text('#usda 1.0\ndef Sphere "Ball" { double radius=1\n}\n')
    b.write_text('#usda 1.0\nover "Ball" { double radius=2\n}\n')
    def root(name,children):
        path=tmp_path/name
        path.write_text('#usda 1.0\n( subLayers=['+','.join('@'+p.name+'@' for p in children)+'] )\n')
        return path
    nested=root('nested.usda',[b,a])
    old=capture_population_snapshot(root('old.usda',[a,nested]))
    assert old.fields[('/Ball','/Ball.radius','default')]==[1.0,2.0,1.0]
    new=capture_population_snapshot(root('new.usda',[nested,a]),unchanged_source_snapshot=old)
    assert new.changed_field_keys is None
    assert new.fields[('/Ball','/Ball.radius','default')]==[2.0,1.0]


def test_unsaved_shared_layer_edits_cannot_reuse_snapshot_or_sparse_predecessor(tmp_path):
    from ovrtx_blender_example.scene_population_delta import capture_population_snapshot
    shared=tmp_path/'shared.usda'; shared.write_text('#usda 1.0\ndef Sphere "Ball" { double radius=1\n}\n')
    old,new=pair(tmp_path,'( subLayers=[@shared.usda@] )\n','( subLayers=[@shared.usda@] )\n')
    snapshot=capture_population_snapshot(old)
    layer=Sdf.Layer.FindOrOpen(str(shared))
    layer.GetAttributeAtPath('/Ball.radius').default=2.0
    current=capture_population_snapshot(new,unchanged_source_snapshot=snapshot)
    assert current.fields[('/Ball','/Ball.radius','default')]==[2.0]
    delta=compile_population_delta(old,new,accepted_snapshot=snapshot)
    assert delta.fallback_reason=='mutable_source_layer_changed' and delta.path is None


def test_retiring_owned_camera_projection_resets_defaults_without_population_replacement(tmp_path):
    source = tmp_path/'geometry.usda'
    source.write_text('#usda 1.0\ndef Xform "World" {\n def Camera "Old" {}\n def Camera "New" {}\n def Mesh "Untouched" { point3f[] points=[(1,2,3)]\n }\n}\n')
    projection = tmp_path/'projection.usda'
    projection.write_text('#usda 1.0\nover "World" {\n over "Old" {\n float fStop=8\n float horizontalApertureOffset=2\n float verticalApertureOffset=3\n bool omni:rtx:autoExposure:enabled=true\n float omni:rtx:autoExposure:whitePointScale=2\n }\n}\n')
    old, new = tmp_path/'old.usda', tmp_path/'new.usda'
    old.write_text(f'#usda 1.0\n( subLayers=[@{projection}@,@{source}@] )\n')
    new.write_text(f'#usda 1.0\n( subLayers=[@{source}@] )\n')
    delta = compile_population_delta(old,new,retired_camera_projection={
        'camera_path':'/World/Old','layer_path':str(projection)})
    assert not delta.fallback_reason
    assert delta.changed_prims == 1 and delta.changed_properties == 5
    patch = Sdf.Layer.FindOrOpen(delta.path)
    assert not patch.GetPrimAtPath('/World/Untouched')
    combined = Sdf.Layer.CreateAnonymous();combined.subLayerPaths=[delta.path,str(old)]
    actual, expected = Usd.Stage.Open(combined), Usd.Stage.Open(str(new))
    for name in ('fStop','horizontalApertureOffset','verticalApertureOffset',
                 'omni:rtx:autoExposure:enabled','omni:rtx:autoExposure:whitePointScale'):
        wanted = expected.GetPrimAtPath('/World/Old').GetAttribute(name).Get()
        if name=='omni:rtx:autoExposure:enabled': wanted=False
        if name=='omni:rtx:autoExposure:whitePointScale': wanted=10.0
        assert actual.GetPrimAtPath('/World/Old').GetAttribute(name).Get() == wanted


@pytest.mark.parametrize('source_owned,approved_path',[(True,'/Cam'),(False,'/Other')])
def test_camera_removal_without_exclusive_presentation_ownership_remains_guarded(tmp_path,source_owned,approved_path):
    source=tmp_path/'source.usda';projection=tmp_path/'projection.usda'
    source.write_text('#usda 1.0\ndef Camera "Cam" {\n'+('float fStop=4\n' if source_owned else '')+'}\n')
    projection.write_text('#usda 1.0\nover "Cam" {\n float fStop=8\n}\n')
    old,new=tmp_path/'old.usda',tmp_path/'new.usda'
    old.write_text(f'#usda 1.0\n( subLayers=[@{projection}@,@{source}@] )\n')
    new.write_text('#usda 1.0\ndef Camera "Cam" {}\n')
    delta=compile_population_delta(old,new,retired_camera_projection={
        'camera_path':approved_path,'layer_path':str(projection)})
    assert delta.path is None and delta.fallback_reason=='existing_property_removed'


def test_ancestor_contributor_history_does_not_expand_effective_delta_mask(tmp_path,monkeypatch):
    base=tmp_path/'base.usda';base.write_text('#usda 1.0\ndef Xform "World" {\n def Camera "Cam" {}\n def Mesh "LargeUnchangedMesh" { point3f[] points=[(1,2,3)]\n }\n}\n')
    over=tmp_path/'over.usda';over.write_text('#usda 1.0\nover "World" {\n over "Cam" { float focalLength=40\n }\n}\n')
    extra=tmp_path/'extra.usda';extra.write_text('#usda 1.0\ndef "World" {\n over "Cam" { float focalLength=40\n }\n}\n')
    old,new=tmp_path/'old.usda',tmp_path/'new.usda'
    old.write_text(f'#usda 1.0\n( subLayers=[@{extra}@,@{over}@,@{base}@] )\n')
    new.write_text(f'#usda 1.0\n( subLayers=[@{over}@,@{base}@] )\n')
    flattened=[];original=Usd.Stage.Flatten
    def flatten(stage,*args):
        flattened.append(tuple(str(p.GetPath()) for p in stage.Traverse()))
        return original(stage,*args)
    monkeypatch.setattr(Usd.Stage,'Flatten',flatten)
    delta=compile_population_delta(old,new)
    assert not delta.fallback_reason and delta.changed_prims==0
    assert all('/World/LargeUnchangedMesh' not in paths for paths in flattened)
    assert delta.changed_prim_paths == ()
    from ovrtx_blender_example.scene_population_delta import capture_population_snapshot
    from ovrtx_blender_example.direct_sdk_client import _unaffected_population_values
    before = capture_population_snapshot(str(old)).fields
    after = delta.candidate_snapshot.fields
    values = {('/World/LargeUnchangedMesh', 'omni:xform'): {'value': 1}}
    # This exact authored-history change used to discard the entire World pose
    # ledger despite compiling a patch with no effective changed prims.
    assert _unaffected_population_values(values, before, after) == {}
    assert _unaffected_population_values(values, before, after,
        effective_changed_paths=delta.changed_prim_paths) == values


def test_real_ancestor_specifier_change_is_not_pruned(tmp_path):
    old,new=pair(tmp_path,'def Xform "World" {}\n','over Xform "World" {}\n')
    delta=compile_population_delta(old,new)
    assert delta.fallback_reason=='existing_prim_schema_changed'


def test_property_contributors_use_direct_immutable_specs_and_preserve_mutated_snapshot(tmp_path):
    from ovrtx_blender_example.scene_population_delta import capture_population_snapshot,_property_contributors
    path=tmp_path/'camera.usda'
    path.write_text('#usda 1.0\ndef Camera "Camera" { float fStop=2\n}\n')
    snapshot=capture_population_snapshot(path)
    cache={}
    class NoFieldScan(dict):
        def __iter__(self):
            pytest.fail('Immutable property ownership scanned unrelated fields')
    snapshot.sources[path.as_posix()]=(NoFieldScan(snapshot.sources[path.as_posix()][0]),{})
    assert _property_contributors(snapshot,Sdf.Path('/Camera.fStop'),Sdf,cache)=={path.as_posix()}
    assert _property_contributors(snapshot,Sdf.Path('/Camera.missing'),Sdf,cache)==set()
    # The retired layer can disappear from the next composition. Mutation of
    # such a layer must not replace its accepted ownership snapshot.
    snapshot.sources[path.as_posix()]=(dict(snapshot.sources[path.as_posix()][0]),{})
    layer=Sdf.Layer.FindOrOpen(str(path))
    Sdf.AttributeSpec(layer.GetPrimAtPath('/Camera'),'newField',Sdf.ValueTypeNames.Float).default=5
    assert layer.dirty
    assert _property_contributors(snapshot,Sdf.Path('/Camera.newField'),Sdf,{})==set()
    assert _property_contributors(snapshot,Sdf.Path('/Camera.fStop'),Sdf,{})=={path.as_posix()}
