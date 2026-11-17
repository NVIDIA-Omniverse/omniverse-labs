# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import FrozenInstanceError

import pytest

from ovd_ui.array_settings import ArrayDisplaySettings, VISIBLE_ARRAY_WINDOW


@pytest.mark.parametrize('field,value', [('group_size',9),('group_size',10000),('group_size',True),
    ('group_size',10.0),('group_size','10'),('max_array_to_load',-1),('max_array_to_load',100000),
    ('max_array_to_load',False),('max_array_to_load',1000.0)])
def test_strict_original_domains(field,value):
    with pytest.raises(ValueError):ArrayDisplaySettings(**{field:value})


def test_defaults_limits_and_immutable_policy():
    settings=ArrayDisplaySettings()
    assert settings.to_preferences()=={'array_group_size':10,'max_array_to_load':1000}
    assert settings.allows_expansion(1000) and not settings.allows_expansion(1001)
    assert ArrayDisplaySettings(max_array_to_load=0).allows_expansion(0)
    assert not ArrayDisplaySettings(max_array_to_load=0).allows_expansion(1)
    with pytest.raises(FrozenInstanceError):settings.group_size=20


@pytest.mark.parametrize('old,expected',[(1,10),(9,10),(10,10),(1000,1000),(0,10),(True,10),('20',10),(1001,10)])
def test_legacy_fallback_does_not_write_or_destroy_keys(old,expected):
    prefs={'array_window':old,'unrelated':'keep'}
    before=dict(prefs)
    assert ArrayDisplaySettings.from_preferences(prefs).group_size==expected
    assert prefs==before
    prefs['array_group_size']=20
    assert ArrayDisplaySettings.from_preferences(prefs).group_size==20


def test_invalid_canonical_key_does_not_silently_use_legacy():
    with pytest.raises(ValueError):ArrayDisplaySettings.from_preferences({'array_group_size':9,'array_window':20})


def test_logical_group_size_is_not_a_read_budget():
    policy=ArrayDisplaySettings(9999,99999)
    assert policy.range_span(99999)==9999
    assert policy.window(99999,9000)==(9000,VISIBLE_ARRAY_WINDOW)
    assert policy.window(10000,9999)==(9999,1)
    assert ArrayDisplaySettings().window(17,16)==(10,7)


def test_null_canonical_is_invalid_and_byte_budget_is_separate():
    from ovd_ui.array_settings import MAX_ARRAY_WINDOW_BYTES
    with pytest.raises(ValueError):ArrayDisplaySettings.from_preferences({'array_group_size':None})
    policy=ArrayDisplaySettings(9999,99999)
    assert policy.window(9999,0,element_bytes=1024*1024)==(0,8)
    assert policy.window(9999,0,element_bytes=MAX_ARRAY_WINDOW_BYTES+1)==(0,0)


def test_actual_preferences_copying_get_preserves_absent_key_and_legacy_migration(tmp_path):
    from ovd_ui.settings import Preferences
    preferences=Preferences(tmp_path/'preferences.json')
    assert ArrayDisplaySettings.from_preferences(preferences)==ArrayDisplaySettings()
    assert not preferences.path.exists()
    preferences.set('array_window',100)
    before=preferences.path.read_bytes()
    assert ArrayDisplaySettings.from_preferences(preferences)==ArrayDisplaySettings(100,1000)
    assert preferences.path.read_bytes()==before
    preferences.set('array_group_size',None)
    with pytest.raises(ValueError):ArrayDisplaySettings.from_preferences(preferences)
