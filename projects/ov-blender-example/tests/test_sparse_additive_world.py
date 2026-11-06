# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace as N
import pytest
from pxr import Sdf,Usd,UsdLux
from ovrtx_blender_example import scene_generation as g
from ovrtx_blender_example.usd_opinion_records import AddOnUsdOpinionRecord
from ovrtx_blender_example.world_dome_conversion import DEFAULT_DOME_OWNER_PATH


@pytest.mark.parametrize('define_world',[True,False])
def test_sparse_stock_validation_defers_additive_world_but_final_validation_requires_it(tmp_path,monkeypatch,define_world):
    obj=N(session_uid=10,name_full='Owner',type='EMPTY',parent=None)
    world=N(session_uid=50,name_full='Environment')
    scene=N(objects=(obj,),world=world)
    object_id=g.BlenderId('OBJECT',10);world_id=g.BlenderId('WORLD',50)
    mappings={object_id:g.BlenderPrimPath('Owner','EMPTY','/World/Owner','/World/Owner'),
              world_id:g.BlenderPrimPath('Environment','WORLD',DEFAULT_DOME_OWNER_PATH,DEFAULT_DOME_OWNER_PATH)}
    def stock(path):
        stage=Usd.Stage.CreateNew(str(path));prim=stage.DefinePrim('/World/Owner','Xform')
        prim.CreateAttribute(g._OBJECT_UID_ATTRIBUTE,Sdf.ValueTypeNames.Int).Set(10)
        UsdLux.DomeLight.Define(stage,'/World/env_light');stage.GetRootLayer().Save()
    base=tmp_path/'base.usda';stock(base)
    def opinions(*_, stock_stage=None):
        if not define_world:return ()
        layer=Sdf.Layer.CreateAnonymous('.usda');stage=Usd.Stage.Open(layer)
        UsdLux.DomeLight.Define(stage,DEFAULT_DOME_OWNER_PATH).CreateIntensityAttr().Set(17)
        return (AddOnUsdOpinionRecord(DEFAULT_DOME_OWNER_PATH,layer,{'source':'world_environment'}),)
    monkeypatch.setattr(g,'_selected_stock_export',lambda scene,objects,path:stock(path))
    monkeypatch.setattr(g,'_validated_blender_prim_paths',lambda *_:mappings)
    monkeypatch.setattr(g,'_compile_add_on_opinions',opinions)
    monkeypatch.setattr(g,'_blender_module_provider',lambda:None)
    predecessor=g.SceneGeneration(number=0,digest='old',predecessor_number=None,base_digest='base',
        usd_path=str(base),blender_prim_paths=mappings,opinion_records=(),sparse_change=None,
        base_usd_path=str(base),topology_deltas=(),topology_fingerprints={},diagnostics={})
    if not define_world:
        with pytest.raises(g.SceneGenerationError,match="mapping for 'Environment' is inactive"):
            g._reconcile_sparse_generation(tmp_path/'work',scene,predecessor,(object_id,))
        # Failure occurs after opinions, not prematurely in stock validation.
        assert list((tmp_path/'work').glob('failed*/composed.usda'))
        return
    candidate=g._reconcile_sparse_generation(tmp_path/'work',scene,predecessor,(object_id,))
    assert candidate.blender_prim_paths[world_id]==mappings[world_id]
    result=Usd.Stage.Open(candidate.usd_path)
    assert result.GetPrimAtPath(DEFAULT_DOME_OWNER_PATH).IsActive()
    assert result.GetPrimAtPath(DEFAULT_DOME_OWNER_PATH).GetAttribute('inputs:intensity').Get()==17
    stock_stage=Usd.Stage.Open(str(Path(candidate.usd_path).parent/'stock.usda'))
    assert not stock_stage.GetPrimAtPath(DEFAULT_DOME_OWNER_PATH)
