# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recorded message ordering, severity and CSV rows from the pinned native writer."""
import csv
import os
from pathlib import Path
import subprocess
import sys

from gui_runtime import GUI_AVAILABLE
import pytest
import _pvd_native as native
from ovd_ui.analysis import CaptureMessages,MessageSeverity
from native_fixtures import executable

ROOT=Path(__file__).resolve().parents[2]


def make_messages_capture(directory):
    binary=executable('pvd_geometry_fixture','PVD_GEOMETRY_FIXTURE')
    path=Path(directory)/'messages.ovd'
    subprocess.run([str(binary),str(path),'--messages'],check=True)
    return path


@pytest.fixture(scope='module')
def recorded_messages(tmp_path_factory):
    directory=tmp_path_factory.mktemp('native_messages')
    source=native.Session(str(make_messages_capture(directory)),str(directory))
    adapter=CaptureMessages(source);adapter.refresh()
    return adapter


def test_native_types_counts_unknown_and_numeric_frame_order(recorded_messages):
    rows=recorded_messages
    assert rows.total==6
    assert rows.counts=={MessageSeverity.ERROR:2,MessageSeverity.WARNING:1,MessageSeverity.INFO:3}
    assert [r.frame for r in rows.filtered()]==[0,2,2,3,10,11]
    assert [r.index for r in rows.filtered(reverse=True)]==[5,4,3,1,2,0]
    assert rows.records[-1].type_name=='' and rows.records[-1].type==99
    assert rows.records[0].target_frame(13) is None
    assert rows.records[4].target_frame(13)==10


@pytest.mark.parametrize('column,indices',[
    ('frame',[0,1,2,3,4,5]),('type_name',[5,0,4,3,1,2]),
    ('text',[0,4,1,2,5,3]),('location',[1,2,3,4,5,0]),
])
def test_recorded_columns_sort_display_values_and_csv_keeps_visible_order(recorded_messages,tmp_path,column,indices):
    adapter=recorded_messages
    assert [row.index for row in adapter.filtered(sort=column)]==indices
    expected=adapter.filtered(sort=column,reverse=True)
    path=adapter.export_csv(tmp_path/'messages.csv',sort=column,reverse=True)
    with path.open(newline='') as stream:export=list(csv.reader(stream))
    assert export[0]==['frame','severity','type','message','file','line']
    assert [r[3] for r in export[1:]]==[r.text for r in expected]
    assert [r[2] for r in export[1:]]==[r.type_name for r in expected]
    assert [r[4] for r in export[1:]]==[r.file for r in expected]


def test_severity_and_search_intersect_without_changing_total(recorded_messages):
    rows=recorded_messages
    assert [r.index for r in rows.filtered(MessageSeverity.ERROR,'radius')]==[2]
    assert [r.index for r in rows.filtered(MessageSeverity.WARNING)]==[3]
    assert rows.filtered(MessageSeverity.INFO,'radius')==()
    assert [r.index for r in rows.filtered(text='A.cpp:10')]==[1]
    assert rows.total==6


@pytest.mark.skipif(not GUI_AVAILABLE,reason='Requires an OVUI graphics display')
def test_native_message_table_owns_selection_sort_and_frozen_export():
    result=subprocess.run([sys.executable,str(ROOT/'python/tests/message_table_gui.py')],cwd=ROOT,
        env=dict(os.environ,OVUIINSPECT_ENABLED='0'),capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'Native message table checks passed' in result.stdout
