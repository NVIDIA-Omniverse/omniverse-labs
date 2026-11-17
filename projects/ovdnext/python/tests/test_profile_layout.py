# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-resize behavior of the floating profiler, without a desktop or GPU."""
from types import SimpleNamespace as NS

import pytest
from ovd_ui.widgets.profile_view import ProfileView,profile_window_rect


def test_default_open_fits_actual_host_and_large_hosts_keep_preferred_size():
    assert profile_window_rect((1280,800))==(16,16,1248,768)
    assert profile_window_rect((1600,1000))==(130,90,1340,820)
    # No assumed physical framebuffer/DPI scaling: dimensions are logical UI pixels.
    assert profile_window_rect((2560,1600))==(610,390,1340,820)


def test_host_growth_preserves_user_position_and_chosen_size():
    manual=(55,44,1000,700)
    assert profile_window_rect((1600,1000),manual)==manual
    assert profile_window_rect((1920,1080),manual)==manual
    assert profile_window_rect((1280,800),manual)==manual


def test_shrink_clamps_only_required_axes_and_does_not_regrow_afterward():
    old=(120,55,1340,820)
    small=profile_window_rect((1280,800),old)
    assert small==(16,16,1248,768)
    assert profile_window_rect((1600,1000),small)==small
    assert profile_window_rect((1280,800),(200,22,1000,700))==(200,22,1000,700)
    assert profile_window_rect((1280,800),(600,22,1000,700))==(264,22,1000,700)


@pytest.mark.parametrize('host',[None,(),(0,800),(-1,800),(float('nan'),800),(1280,float('inf'))])
def test_unavailable_measurements_do_not_move_existing_window(host):
    current=(55,44,1000,700)
    assert profile_window_rect(host,current)==current
    assert profile_window_rect(host)==(0,0,1340,820)


def test_polling_same_host_never_overrides_a_user_drag_and_resize():
    host=[1600,1000]
    moves=[]
    view=ProfileView.__new__(ProfileView)
    view.host_size=lambda:tuple(host)
    view.host_extent=tuple(host)
    view.window=NS(position_x=200,position_y=130,width=1200,height=740,
                   setPosition=lambda x,y:moves.append((x,y)))
    view._sync_host_size()
    assert moves==[] and view.window.width==1200
    host[:]=[1280,800]
    view._sync_host_size()
    assert moves==[(64,44)]
    assert (view.window.width,view.window.height)==(1200,740)
    view._sync_host_size()
    assert moves==[(64,44)]
