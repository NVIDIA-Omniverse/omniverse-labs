# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation and restart decoding for display modes and independent sizes."""
from dataclasses import replace
import json

import pytest

from ovd_ui.settings import Preferences
from ovd_ui.viewport_settings import ViewportDisplaySettings, gizmo_value, FLOAT32_MAX, SCALE_FIELDS
from ovd_ui.visual_reset import GIZMO_SCALE


def test_existing_defaults_and_legacy_store_keys_are_not_imported(tmp_path):
    preferences=Preferences(tmp_path/'preferences.json')
    preferences.values.update(boundingBox=1,coordinateSystem=2,velocity=3,joints=1,
        transparent=4,transparencyAmount=.9,gizmoScale=31)
    result=ViewportDisplaySettings.from_preferences(preferences)
    assert result.to_dict()==dict(bounds=0,actor_axes=0,linear_velocity=0,joints=False,
        contacts=0,scene_queries=False,transparency_mode=0,transparency_amount=.5,gizmo_scale=1.,
        center_of_mass=0,master_scale=1.,**{name:1. for name in SCALE_FIELDS.values()})
    assert not preferences.path.exists()


@pytest.mark.parametrize('field,value',[("bounds",True),("bounds",1.0),("bounds",4),
    ("actor_axes",-1),("linear_velocity","1"),("joints",4),("contacts",None),
    ("scene_queries","false"),("transparency_mode",5),("transparency_mode",False),
    ("transparency_amount",True),("transparency_amount",float('nan')),
    ("transparency_amount",1.01),("gizmo_scale",0),("gizmo_scale",-1),
    ("gizmo_scale",float('inf')),("gizmo_scale",10**1000)])
def test_malformed_block_uses_defaults_without_rewriting_bytes(tmp_path,field,value):
    preferences=Preferences(tmp_path/'preferences.json')
    block=dict(bounds=3,joints=True);block[field]=value
    preferences.set('viewport_display',block)
    before=preferences.path.read_bytes()
    assert ViewportDisplaySettings.from_preferences(Preferences(preferences.path))==ViewportDisplaySettings()
    assert preferences.path.read_bytes()==before
    with pytest.raises(ValueError):replace(ViewportDisplaySettings(),**{field:value})


@pytest.mark.parametrize('value',[None,False,3,[],"invalid"])
def test_malformed_preference_container_does_not_crash(value):
    assert ViewportDisplaySettings.from_preferences({'viewport_display':value})==ViewportDisplaySettings()


def test_all_mode_boundaries_and_amount_direction_roundtrip(tmp_path):
    preferences=Preferences(tmp_path/'preferences.json')
    for mode in range(4):
        value=ViewportDisplaySettings(mode,mode,mode,1,0,True,mode,.75,3.25)
        preferences.set('viewport_display',value.to_dict())
        assert ViewportDisplaySettings.from_preferences(Preferences(preferences.path))==value
    tagged=replace(value,transparency_mode=4,transparency_amount=0)
    assert tagged.mask==1|2|4|16|64
    assert replace(tagged,transparency_amount=1).transparency_amount==1
    with pytest.raises(ValueError):replace(tagged,transparency_amount=-.01)
    assert json.loads(json.dumps(tagged.to_dict()))['transparency_amount']==0


def test_reset_defaults_keep_joint_visibility_and_gizmo_keeps_existing_minimum():
    before=ViewportDisplaySettings(3,2,1,1,1,True,4,.8,19)
    reset=before.visual_reset(GIZMO_SCALE)
    assert reset==ViewportDisplaySettings(joints=1,gizmo_scale=GIZMO_SCALE,
        **{name:GIZMO_SCALE for name in SCALE_FIELDS.values()})
    assert gizmo_value(-3)==gizmo_value(0)==gizmo_value(1e-5)==.0001
    assert gizmo_value(FLOAT32_MAX)==FLOAT32_MAX
    with pytest.raises(ValueError):gizmo_value(FLOAT32_MAX*2)


def test_unknown_top_level_preferences_survive_an_accepted_write(tmp_path):
    preferences=Preferences(tmp_path/'preferences.json')
    preferences.set_many({'future_feature':{'choice':[1,2,3]},'joint_scale':.2})
    value=ViewportDisplaySettings.from_preferences(preferences)
    preferences.set('viewport_display',replace(value,joints=1).to_dict())
    fresh=Preferences(preferences.path)
    assert fresh.get('future_feature')=={'choice':[1,2,3]} and fresh.get('joint_scale')==.2
    assert set(fresh.get('viewport_display'))==set(value.to_dict())


@pytest.mark.parametrize('old,mode',[(False,0),(True,1),(2,2),(3,3)])
def test_contact_mode_restores_old_toggle_and_new_modes_without_losing_other_choices(tmp_path,old,mode):
    preferences=Preferences(tmp_path/'preferences.json')
    preferences.set('viewport_display',dict(contacts=old,bounds=2,gizmo_scale=3.25))
    before=preferences.path.read_bytes()
    value=ViewportDisplaySettings.from_preferences(preferences)
    assert value.contacts==mode and value.modes[32]==mode
    assert value.bounds==2 and value.gizmo_scale==3.25
    assert bool(value.mask & 32)==bool(mode)
    assert preferences.path.read_bytes()==before
    preferences.set('viewport_display',value.to_dict())
    assert ViewportDisplaySettings.from_preferences(Preferences(preferences.path))==value
    reset=value.visual_reset(GIZMO_SCALE)
    assert reset.contacts==0 and reset.modes[32]==0 and not reset.mask & 32


def test_family_scales_inherit_old_size_once_then_persist_independently(tmp_path):
    preferences=Preferences(tmp_path/'preferences.json')
    preferences.set('viewport_display',dict(gizmo_scale=3.25,joints=True,contacts=True))
    old=ViewportDisplaySettings.from_preferences(preferences)
    assert old.scales=={2:3.25,4:3.25,8:3.25,32:3.25,256:3.25}
    assert old.modes[16]==1 and old.modes[256]==0
    changed=replace(old,axes_scale=2.,linear_velocity_scale=7.,angular_velocity_scale=4.,
                    com_scale=.5,contact_scale=6.,joints=2,center_of_mass=3)
    preferences.set('viewport_display',changed.to_dict())
    restored=ViewportDisplaySettings.from_preferences(Preferences(preferences.path))
    assert restored==changed and restored.scales=={2:2.,4:7.,8:4.,32:6.,256:.5}
    assert restored.modes[16]==2 and restored.modes[256]==3
    assert replace(restored,gizmo_scale=9.).scales==restored.scales


@pytest.mark.parametrize('field',['axes_scale','linear_velocity_scale','angular_velocity_scale','com_scale','contact_scale'])
@pytest.mark.parametrize('value',[True,0,-1,float('nan'),float('inf')])
def test_family_scales_reject_invalid_native_values(field,value):
    with pytest.raises(ValueError):replace(ViewportDisplaySettings(),**{field:value})


def test_master_defaults_to_neutral_and_persists_without_rewriting_family_factors(tmp_path):
    preferences=Preferences(tmp_path/'preferences.json')
    preferences.set('viewport_display',dict(gizmo_scale=3.25,axes_scale=7.))
    before=preferences.path.read_bytes()
    old=ViewportDisplaySettings.from_preferences(preferences)
    assert old.master_scale==1. and old.axes_scale==7.
    assert preferences.path.read_bytes()==before
    changed=replace(old,master_scale=2.)
    preferences.set('viewport_display',changed.to_dict())
    assert ViewportDisplaySettings.from_preferences(Preferences(preferences.path))==changed
    assert changed.scales==old.scales and changed.gizmo_scale==old.gizmo_scale
    assert changed.visual_reset(GIZMO_SCALE).master_scale==1.


@pytest.mark.parametrize('value',[True,0,-1,float('nan'),float('inf')])
def test_invalid_master_never_becomes_a_native_request(value):
    with pytest.raises(ValueError):replace(ViewportDisplaySettings(),master_scale=value)


@pytest.mark.parametrize('family,master',[(FLOAT32_MAX,20.),(FLOAT32_MAX/1.1,1.1)])
def test_combined_master_and_family_overflow_is_rejected(family,master):
    with pytest.raises(ValueError):
        replace(ViewportDisplaySettings(),axes_scale=family,master_scale=master)
