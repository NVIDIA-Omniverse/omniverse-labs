# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS
import pytest
from ovrtx_blender_example.object_matte import opinions, opinion_records
from ovrtx_blender_example.object_visibility import merge_records


def mesh(matte='INHERIT', holdout='INHERIT'):
    return NS(type='MESH', ov=NS(rendering=NS(matte=matte, holdout=holdout)))


def test_inheritance_reset_and_explicit_false():
    assert opinions(mesh()) == {}
    assert opinions(mesh('ON', 'OFF')) == {'primvars:isMatteObject': True, 'primvars:holdoutObject': False}
    with pytest.raises(ValueError):
        opinions(mesh('yes'))


def test_constant_bool_opinions_survive_visibility_merge():
    from pxr import Usd, UsdGeom, Sdf
    from ovrtx_blender_example.usd_opinion_records import AddOnUsdOpinionRecord
    obj = mesh('ON', 'OFF')
    rows = opinion_records(NS(objects=[obj]), {1: NS(schema_path='/Mesh')}, lambda *_: 1)
    layer = Sdf.Layer.CreateAnonymous('.usda')
    stage = Usd.Stage.Open(layer)
    UsdGeom.Imageable(stage.OverridePrim('/Mesh')).CreateVisibilityAttr().Set('invisible')
    rows = merge_records((*rows, AddOnUsdOpinionRecord('/Mesh', layer, {})))
    assert len(rows) == 1
    stage = Usd.Stage.Open(Sdf.Layer.CreateAnonymous('.usda'))
    stage.GetRootLayer().ImportFromString(rows[0].layer_text)
    prim = stage.GetPrimAtPath('/Mesh')
    assert prim.GetAttribute('visibility').Get() == 'invisible'
    for name, value in opinions(obj).items():
        pv = UsdGeom.Primvar(prim.GetAttribute(name))
        assert pv.Get() == value and pv.GetInterpolation() == 'constant'


def test_missing_mapping_rejects_instead_of_mutating_unrelated_prim():
    with pytest.raises(ValueError, match='identity'):
        opinion_records(NS(objects=[mesh('ON')]), {}, lambda *_: 1)
