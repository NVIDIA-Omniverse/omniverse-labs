# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Temporary native unavailability recovers from owned observations, not live reads."""
import gc
from pathlib import Path
import struct
import subprocess

import pytest
import _pvd_native
from _pvd_native import Session, SnapshotDataUnavailable

from ovd_ui.adapters import PvdDataAdapter
from ovd_ui.widgets.properties import PropertyInspector
from test_property_context import display
from test_native_presentation import live, wait_until, HANDSHAKE
from native_fixtures import executable


def recording(tmp_path,death=False):
    tmp_path.mkdir(parents=True,exist_ok=True)
    binary=executable('pvd_geometry_fixture','PVD_GEOMETRY_FIXTURE')
    path=tmp_path/'array-lifetime.ovd'
    subprocess.run([str(binary),str(path)],check=True,capture_output=True)
    data=path.read_bytes()
    last_value=struct.pack('<Iff',8,4,5)
    assert data.count(last_value)==1
    offset=data.index(last_value)
    prefix=data[offset-22:offset]
    command,context,instance,arity,attribute=struct.unpack('<BQQBI',prefix)
    assert (command,context,instance,arity)==(6,1,110,1)
    # Keep the native writer's schema/events. Replace two complete value packets
    # with80KB each, retaining all length fields and actual native parsing.
    for original,base in ((prefix+struct.pack('<I',0),0),(prefix+last_value,100000)):
        assert data.count(original)==1
        values=struct.pack('<20000f',*range(base,base+20000))
        data=data.replace(original,prefix+struct.pack('<I',len(values))+values)
    if death:
        # DESTROY_OBJECT before the second STOP_FRAME. Drop the later frame so
        # the fixture never writes properties through a destroyed writer ID.
        stop=struct.pack('<BQQ',12,1,2)
        assert data.count(stop)==1
        index=data.index(stop)
        data=data[:index]+struct.pack('<BQQ',10,1,110)+stop
    path.write_bytes(data)
    source=Session(str(path),str(tmp_path))
    first=source.presentation(frame=1)
    cls=next(c.id for c in first.classes() if c.name=='AnalysisSample')
    identity=next(i.id for i in first.instances() if i.class_id==cls)
    adapter=PvdDataAdapter(source.presentation(frame=1,inspect_ids=[identity]))
    prop=adapter.property(identity,'variable')
    assert adapter.session.property_size(identity,prop.id)==80000
    request=(identity,prop.id,72000,256)
    calls=[]
    inspector=PropertyInspector(lambda *_:None,lambda *_:None,
        {'array_group_size':9999,'max_array_to_load':99999},on_request=lambda:calls.append(True))
    inspector.build(adapter,[identity])
    inspector.array_positions[(identity,prop.id)]=18000
    inspector.array_expanded.add((identity,prop.id))
    return source,identity,prop,request,inspector,calls


def draw(inspector,view,identity,display):
    adapter=PvdDataAdapter(view)
    display.widgets.clear()
    inspector.build(adapter,[identity])
    frames=[w for w in display.widgets if w.kind=='Frame']
    for frame in frames:frame.rebuild()
    return adapter,frames


def test_real_native_death_then_rewind_recovers_same_window_once(tmp_path,display):
    source,identity,prop,request,inspector,calls=recording(tmp_path,death=True)
    first=source.presentation(frame=1,inspect_ids=[identity],read_requests=[request])
    draw(inspector,first,identity,display)
    assert display.find('18000.000000')
    dead=source.presentation(frame=2,inspect_ids=[identity],read_requests=[request],allow_unavailable_requests=True)
    assert dead.unavailable_ids==[identity] and dead.unavailable_reads==[request]
    dead_adapter,_=draw(inspector,dead,identity,display)
    assert display.find('Not alive at this frame')
    assert request in inspector._array_unavailable and request not in inspector._array_denied
    for _ in range(5):
        same=source.presentation(frame=2,inspect_ids=[identity])
        draw(inspector,same,identity,display)
        assert inspector.read_requests(dead_adapter,[identity])==[]
    before=len(calls)
    restored=source.presentation(frame=1,inspect_ids=[identity])
    with pytest.raises(SnapshotDataUnavailable):restored.read(*request)
    adapter,frames=draw(inspector,restored,identity,display)
    assert len(calls)==before+1 and request not in inspector._array_unavailable
    assert inspector.read_requests(adapter,[identity])==[request]
    # Fresh generation at the same scene/frame/size is not new evidence.
    for _ in range(5):
        same=source.presentation(frame=1,inspect_ids=[identity])
        draw(inspector,same,identity,display)
    assert len(calls)==before+1
    completed=source.presentation(frame=1,inspect_ids=[identity],read_requests=[request])
    draw(inspector,completed,identity,display)
    assert display.find('18000.000000') and display.find('18063.000000')
    del source;gc.collect()
    assert struct.unpack('<f',first.read(identity,prop.id,72000,4))==(18000.,)
    assert struct.unpack('<f',completed.read(identity,prop.id,72000,4))==(18000.,)


def test_real_shrink_unavailability_recovers_in_later_frame_with_restored_owned_size(tmp_path,display):
    source,identity,prop,request,inspector,calls=recording(tmp_path)
    shrunk=source.presentation(frame=2,inspect_ids=[identity],read_requests=[request],allow_unavailable_requests=True)
    assert shrunk.property_size(identity,prop.id)==4 and shrunk.unavailable_reads==[request]
    adapter=PvdDataAdapter(shrunk)
    # A pending range can be inspected before the property's rebuilt parent
    # clamps its position. Use the real native marker, without altering bytes.
    inspector._observe_array_frame(shrunk)
    assert inspector._array_window_denial(shrunk,request)
    for _ in range(5):
        again=source.presentation(frame=2,inspect_ids=[identity])
        inspector._observe_array_frame(again)
        assert inspector._array_window_denial(again,request)
    assert not calls
    restored=source.presentation(frame=3,inspect_ids=[identity])
    adapter,frames=draw(inspector,restored,identity,display)
    assert len(calls)==1 and inspector.read_requests(adapter,[identity])==[request]
    final=source.presentation(frame=3,inspect_ids=[identity],read_requests=[request])
    draw(inspector,final,identity,display)
    assert display.find('118000.000000') and len(calls)==1
    # Applying a truly smaller parent still clamps the visible range. Automatic
    # data recovery does not promise resurrection of a discarded scroll offset.
    draw(inspector,shrunk,identity,display)
    assert inspector.array_positions[(identity,prop.id)]==0


def test_budget_rejection_remains_explicit_after_native_death_and_rewind(tmp_path,display):
    source,identity,prop,request,inspector,calls=recording(tmp_path,death=True)
    inspector.reject_read_requests([request],'Optional window paused at the copy budget.')
    for frame in (2,1,2,1):
        owned=source.presentation(frame=frame,inspect_ids=[identity])
        adapter,frames=draw(inspector,owned,identity,display)
        assert inspector.read_requests(adapter,[identity])==[]
        assert request in inspector._array_denied
    assert not calls and display.find('Optional window paused at the copy budget.')
    section=next(w for w in display.widgets if w.kind=='CollapsableFrame')
    section.collapsed(True);section.collapsed(False)
    assert len(calls)==1 and inspector.read_requests(adapter,[identity])==[request]


def test_stale_capture_controls_cannot_reactivate_denied_intent(tmp_path,display):
    source,identity,prop,request,inspector,calls=recording(tmp_path/'old')
    old=source.presentation(frame=1,inspect_ids=[identity])
    _,old_frames=draw(inspector,old,identity,display)
    old_sections=[w for w in display.widgets if w.kind=='CollapsableFrame']
    inspector.reject_read_requests([request],'Budget')
    other,other_id,_,_,_,_=recording(tmp_path/'new')
    new=other.presentation(frame=1,inspect_ids=[other_id])
    draw(inspector,new,other_id,display)
    before=len(calls)
    for section in old_sections:section.collapsed(False)
    for frame in old_frames:frame.rebuild()
    assert len(calls)==before and not inspector._array_denied and not inspector._array_unavailable
    assert inspector.read_requests(PvdDataAdapter(old),[identity])==[]


def test_restored_owned_size_can_retire_native_marker_without_new_frame_number(tmp_path,display):
    source,identity,prop,request,inspector,calls=recording(tmp_path)
    missing=source.presentation(frame=2,inspect_ids=[identity],read_requests=[request],allow_unavailable_requests=True)
    assert inspector._array_window_denial(missing,request)
    restored=source.presentation(frame=3,inspect_ids=[identity])
    # Hold only the frame-number component constant to isolate size invalidation;
    # all availability/byte-size facts still come from real immutable views.
    class SameFrame:
        frame=2
        def __getattr__(self,name):return getattr(restored,name)
    assert inspector._array_window_denial(SameFrame(),request) is None
    assert request not in inspector._array_unavailable
    inspector.reject_read_requests([request],'Budget requires explicit retry')
    assert inspector._array_window_denial(SameFrame(),request)=='Budget requires explicit retry'


def test_actual_live_segment_change_cancels_old_array_callbacks_and_denials(live,display):
    source,connection=live
    payload=(Path(__file__).resolve().parents[2]/'test/data/resample_two_segment.ovd').read_bytes()
    first_end,second_end=42117,98924
    connection.sendall(HANDSHAKE+payload[:first_end])
    wait_until(lambda:source.info().completed_serial==120)
    first=source.presentation(frame=1)
    scene=next(s for s in first.scenes() if s.alive)
    first=source.presentation(frame=1,inspect_ids=[scene.instance_id])
    adapter=PvdDataAdapter(first)
    prop=next(p for p in adapter.properties(scene.instance_id)
              if p.type=='ObjectRef' and len(first.references(scene.instance_id,p.id))>0)
    calls=[]
    inspector=PropertyInspector(lambda *_:None,lambda *_:None,{},on_request=lambda:calls.append(True))
    inspector.build(adapter,[scene.instance_id])
    sections=[w for w in display.widgets if w.kind=='CollapsableFrame']
    assert sections
    key=(scene.instance_id,prop.id)
    inspector.array_positions[key]=0;inspector.array_expanded.add(key)
    request=(*key,0,4)
    inspector.reject_read_requests([request],'Budget')
    connection.sendall(payload[first_end:first_end+1024])
    wait_until(lambda:source.info().unpublished_changes)
    deferred=source.presentation()
    assert not deferred.request_applied
    inspector._observe_array_frame(deferred)
    assert inspector._array_denied=={request:'Budget'}
    connection.sendall(payload[first_end+1024:second_end])
    wait_until(lambda:source.info().completed_serial==241)
    second=source.presentation(frame=1)
    assert second.capture_key==first.capture_key and second.segment_epoch!=first.segment_epoch
    inspector.build(PvdDataAdapter(second),[])
    assert not inspector.array_positions and not inspector.array_expanded
    assert not inspector._array_denied and not inspector._array_unavailable
    for section in sections:section.collapsed(False)
    assert not calls and not inspector.array_expanded
    assert inspector.read_requests(adapter,[scene.instance_id])==[]
