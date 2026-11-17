# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Column widths follow available space until a deliberate user resize."""
import math
import pytest
from ovd_ui.widgets.table_columns import ColumnWidths


def columns():return ColumnWidths((146,52,74,76),(100,44,60,60),(1200,300,400,300))


def test_auto_fill_shrinks_with_host_but_preserves_horizontal_overflow():
    model=columns()
    assert model.layout(600)==(398,52,74,76)
    assert model.layout(200)==(146,52,74,76)
    assert model.layout(2000)==(1200,52,74,76)
    assert model.layout(math.nan)==model.defaults


def test_message_column_absorbs_space_without_widening_frame_or_type():
    model=ColumnWidths((64,160,280,200),(48,100,160,120),(240,520,1600,1200),stretch=2)
    assert model.layout(1000)==(64,160,576,200)
    model.resize(1,220,model.layout(1000))
    assert model.layout(1200)==(64,220,576,200)


def test_first_resize_freezes_displayed_widths_across_frames_and_host_changes():
    model=columns();displayed=model.layout(600)
    model.resize(1,90,displayed)
    assert model.layout(800)==(398,90,74,76)
    restored=columns();restored.restore(model.state())
    assert restored.layout(200)==model.layout(800)
    restored.reset()
    assert restored.layout(200)==restored.defaults


@pytest.mark.parametrize('index',range(4))
def test_resize_limits_and_nonfinite_input_preserve_usable_columns(index):
    model=columns()
    model.resize(index,-100,model.layout(600))
    assert model.widths[index]==model.minimums[index]
    model.resize(index,10000,model.widths)
    assert model.widths[index]==model.maximums[index]
    before=model.state()
    for invalid in (math.nan,math.inf,-math.inf):
        model.resize(index,invalid,model.widths)
        assert model.state()==before
