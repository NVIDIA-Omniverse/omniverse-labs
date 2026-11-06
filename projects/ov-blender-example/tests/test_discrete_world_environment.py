from types import SimpleNamespace as N
import pytest
from pathlib import Path
from ovrtx_blender_example import world_dome_conversion as world

def socket(value=None,source=None):return N(default_value=value,is_linked=source is not None,links=[N(from_node=source)] if source else [])
def graph(factor=0,strength=1):
    images=[N(name='morning',filepath='missing.exr',packed_file=N(data=b'v/1\x01morning')),N(name='afternoon',filepath='missing.exr',packed_file=N(data=b'v/1\x01afternoon'))]
    env=[N(type='TEX_ENVIRONMENT',image=image,inputs={}) for image in images]
    mix=N(type='MIX_RGB',blend_type='MIX',use_clamp=False,inputs={'Fac':socket(factor),'Color1':socket(source=env[0]),'Color2':socket(source=env[1])})
    background=N(type='BACKGROUND',inputs={'Color':socket(source=mix),'Strength':socket(strength)})
    output=N(type='OUTPUT_WORLD',is_active_output=True,inputs={'Surface':socket(source=background)})
    return N(use_nodes=True,node_tree=N(nodes=[*env,mix,background,output])),mix,background,output

def test_packed_selection(tmp_path,monkeypatch):
    monkeypatch.setenv('OV_BLENDER_EXAMPLE_AUTHORING_WORK_DIR',str(tmp_path/'authoring'))
    source,mix,bg,out=graph();morning=world.environment_texture_dome(source);mix.inputs['Fac'].default_value=1
    afternoon=world.environment_texture_dome(source)
    assert Path(morning['texture_file']).read_bytes().endswith(b'morning')
    assert Path(afternoon['texture_file']).read_bytes().endswith(b'afternoon')
    bg.inputs['Strength'].default_value=0;assert world.environment_texture_dome(source)['intensity']==0
    assert morning['intensity']==afternoon['intensity']==world.DOME_LIGHT_SCALE

@pytest.mark.parametrize('factor',[.5,-1,2,float('nan'),float('inf')])
def test_invalid_mix_not_approximated(factor):assert world.environment_texture_dome(graph(factor)[0]) is None

def test_linked_clamped_nonmix_cycle_fail_closed():
    source,mix,bg,out=graph();mix.inputs['Fac'].is_linked=True;assert world.environment_texture_dome(source) is None
    mix.inputs['Fac'].is_linked=False;mix.use_clamp=True;assert world.environment_texture_dome(source) is None
    mix.use_clamp=False;mix.blend_type='ADD';assert world.environment_texture_dome(source) is None
    mix.blend_type='MIX';mix.inputs['Color1']=socket(source=mix);assert world.environment_texture_dome(source) is None

def test_selected_output_stock_dome_deactivation(tmp_path,monkeypatch):
    Usd=pytest.importorskip('pxr.Usd');from pxr import UsdLux
    from ovrtx_blender_example.world_scene_layer import opinion_records
    monkeypatch.setenv('OV_BLENDER_EXAMPLE_AUTHORING_WORK_DIR',str(tmp_path/'authoring'))
    source,mix,bg,out=graph();source.node_tree.nodes.append(N(type='BACKGROUND',inputs={}))
    stock=Usd.Stage.CreateNew(str(tmp_path/'stock.usda'));UsdLux.DomeLight.Define(stock,'/World/env_light');stock.GetRootLayer().Save()
    records=opinion_records(N(world=source),tmp_path/'stock.usda');composed=Usd.Stage.CreateInMemory()
    composed.GetRootLayer().subLayerPaths=[str(tmp_path/'stock.usda')]
    for index,record in enumerate(reversed(records)):
        path=tmp_path/f'layer-{index}.usda';path.write_text(record.layer_text);composed.GetRootLayer().subLayerPaths.insert(0,str(path))
    assert [str(p.GetPath()) for p in composed.Traverse() if p.GetTypeName()=='DomeLight']==['/World/StudioDome']
    assert composed.GetPrimAtPath('/World/StudioDome').GetAttribute('inputs:intensity').Get()==pytest.approx(world.DOME_LIGHT_SCALE)
    assert composed.GetPrimAtPath('/World/StudioDome').GetAttribute('inputs:diffuse').Get()==pytest.approx(world.DOME_DIFFUSE_SCALE)


def _compose(tmp_path,stock_path,records):
    from pxr import Usd
    composed=Usd.Stage.CreateInMemory();composed.GetRootLayer().subLayerPaths=[str(stock_path)]
    for index,record in enumerate(reversed(records)):
        path=tmp_path/f'flat-layer-{index}.usda';path.write_text(record.layer_text);composed.GetRootLayer().subLayerPaths.insert(0,str(path))
    return composed


def _flat_world(color=(0.5,0.25,1.0),strength=1.6):
    background=N(type='BACKGROUND',inputs={'Color':socket((*color,1.0)),'Strength':socket(strength)})
    return N(use_nodes=True,node_tree=N(nodes=[background]))


def test_flat_world_converts_stock_dome_with_live_edit_policy(tmp_path):
    Usd=pytest.importorskip('pxr.Usd');from pxr import Sdf,UsdLux
    from ovrtx_blender_example.world_scene_layer import opinion_records
    stock=Usd.Stage.CreateNew(str(tmp_path/'stock.usda'))
    dome=UsdLux.DomeLight.Define(stock,'/root/env_light')
    dome.CreateIntensityAttr().Set(1.6);dome.CreateTextureFileAttr().Set(Sdf.AssetPath('./textures/color.hdr'))
    stock.GetRootLayer().Save()
    records=opinion_records(N(world=_flat_world()),tmp_path/'stock.usda')
    assert [record.diagnostics['source'] for record in records]==['world_flat_color']
    composed=_compose(tmp_path,tmp_path/'stock.usda',records);prim=composed.GetPrimAtPath('/root/env_light')
    spec=world.world_dome_spec(_flat_world())
    assert prim.GetAttribute('inputs:intensity').Get()==pytest.approx(1.6*world.DOME_LIGHT_SCALE)
    assert prim.GetAttribute('inputs:intensity').Get()==pytest.approx(spec.intensity)
    assert tuple(prim.GetAttribute('inputs:color').Get())==pytest.approx((0.5,0.25,1.0))
    assert prim.GetAttribute('inputs:texture:file').Get().path==''
    assert prim.GetAttribute('inputs:diffuse').Get()==pytest.approx(world.DOME_DIFFUSE_SCALE)


def test_flat_world_leaves_ambiguous_stock_domes_untouched(tmp_path):
    Usd=pytest.importorskip('pxr.Usd');from pxr import UsdLux
    from ovrtx_blender_example.world_scene_layer import opinion_records
    stock=Usd.Stage.CreateNew(str(tmp_path/'stock.usda'))
    UsdLux.DomeLight.Define(stock,'/root/a');UsdLux.DomeLight.Define(stock,'/root/b');stock.GetRootLayer().Save()
    assert opinion_records(N(world=_flat_world()),tmp_path/'stock.usda')==()
    assert opinion_records(N(world=None),tmp_path/'stock.usda')==()
