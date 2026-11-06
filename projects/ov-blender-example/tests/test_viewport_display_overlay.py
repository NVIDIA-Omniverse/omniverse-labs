from contextlib import contextmanager
import gc
import sys
import weakref
import threading
from types import SimpleNamespace

import pytest

from ovrtx_blender_example import viewport_display_overlay
from ovrtx_blender_example.viewport_display_overlay import DisplayOverlay, claims_view, labels_drawn, view_key


class Pointer(SimpleNamespace):
    def as_pointer(self):
        return id(self)


def context():
    return SimpleNamespace(window=Pointer(), area=Pointer(type="VIEW_3D"),
        region=Pointer(width=640,height=480), scene=Pointer(), view_layer=Pointer(),
        space_data=Pointer(shading=SimpleNamespace(type="RENDERED")))


class Owner:
    def __init__(self):
        self._viewport_display_epoch=object()
        self.events=[]
    def _finish_viewport_presentation(self, **values):
        self.events.append(("presented",values))
    def _finish_cached_frame_presentation(self, frame, packet):
        self.events.append(("cached",frame,packet))
    def _report_viewport_error(self, error):
        self.events.append(("error",str(error)))


def queue(manager,ctx,owner,publication=1):
    work=manager.queue(ctx,owner,object(),publication,(0,0),(640,480))
    work.completion={"publication":publication,"span_boundaries":{}}
    return work


def test_success_commits_only_after_actual_draw_once(monkeypatch):
    manager=DisplayOverlay();ctx=context();owner=Owner();work=queue(manager,ctx,owner)
    def draw(*args):
        assert owner.events==[]
        owner.events.append(("draw",work.result))
    monkeypatch.setattr(manager,"draw_pixels",draw)
    assert owner.events==[]
    manager.present(ctx)
    assert [event[0] for event in owner.events]==["draw","presented"]
    assert owner.events[1][1]["publication"]==1
    assert owner.events[1][1]["span_boundaries"]["viewport_draw_started_monotonic_ns"]>0
    monkeypatch.setattr(manager,"draw_pixels",lambda *args:None)
    manager.present(ctx)
    assert len(owner.events)==2


def test_failed_draw_never_claims_presentation(monkeypatch):
    manager=DisplayOverlay();ctx=context();owner=Owner();queue(manager,ctx,owner)
    def fail(*args):raise RuntimeError("GPU draw refused")
    monkeypatch.setattr(manager,"draw_pixels",fail)
    manager.present(ctx)
    assert owner.events==[("error","GPU draw refused")]


def test_new_mirror_packet_rejects_queued_old_image_before_gpu(monkeypatch):
    manager=DisplayOverlay();ctx=context();owner=Owner()
    old_packet=object();current_packet=object()
    work=queue(manager,ctx,owner)
    work.completion['prepared_presentation']=old_packet
    owner._presentation_packet_is_current=lambda packet:packet is current_packet
    monkeypatch.setattr(manager,'draw_pixels',lambda *args:pytest.fail('obsolete image must not draw'))
    manager.present(ctx)
    assert owner.events==[] and manager.entries=={}


def test_latest_queued_publication_owns_completion(monkeypatch):
    manager=DisplayOverlay();ctx=context();owner=Owner()
    queue(manager,ctx,owner,1);queue(manager,ctx,owner,2)
    monkeypatch.setattr(manager,"draw_pixels",lambda *args:None)
    manager.present(ctx)
    assert owner.events[0][1]["publication"]==2


@pytest.mark.parametrize("change",["epoch","resize","space","scene","view_layer","editor","shading"])
def test_stale_context_cannot_draw_or_claim_presentation(monkeypatch,change):
    manager=DisplayOverlay();ctx=context();owner=Owner();queue(manager,ctx,owner)
    manager.buffers[view_key(ctx)]=object()
    if change=="epoch":owner._viewport_display_epoch=object()
    elif change=="resize":ctx.region.width+=1
    elif change=="space":ctx.space_data=Pointer(shading=SimpleNamespace(type="RENDERED"))
    elif change in ("scene","view_layer"):setattr(ctx,change,Pointer())
    elif change=="editor":ctx.area.type="IMAGE_EDITOR"
    else:ctx.space_data.shading.type="SOLID"
    monkeypatch.setattr(manager,"draw_pixels",lambda *args:pytest.fail("stale draw"))
    manager.present(ctx)
    assert owner.events==[] and manager.entries=={} and manager.buffers=={}


def test_two_views_preserve_distinct_resources_and_completions(monkeypatch):
    manager=DisplayOverlay();a,b=context(),context();one,two=Owner(),Owner()
    queue(manager,a,one,1);queue(manager,b,two,9)
    manager.buffers[view_key(a)]="a";manager.buffers[view_key(b)]="b"
    monkeypatch.setattr(manager,"draw_pixels",lambda ctx,work,key:None)
    manager.present(a)
    assert two.events==[] and one.events[0][1]["publication"]==1
    manager.discard_owner(one)
    assert manager.buffers=={view_key(b):"b"}
    manager.present(b)
    assert two.events[0][1]["publication"]==9


def test_registry_does_not_keep_destroyed_engine_alive(monkeypatch):
    manager=DisplayOverlay();ctx=context();owner=Owner();queue(manager,ctx,owner)
    ref=weakref.ref(owner);del owner;gc.collect();assert ref() is None
    monkeypatch.setattr(manager,"draw_pixels",lambda *args:pytest.fail("dead owner"))
    manager.present(ctx)
    assert manager.entries=={}


def test_registered_handler_is_blocked_during_final_and_removed_at_teardown(monkeypatch):
    manager=DisplayOverlay();ctx=context();owner=Owner();queue(manager,ctx,owner)
    callbacks=[];removed=[];blocked=[True]
    space=SimpleNamespace(draw_handler_add=lambda f,*a:callbacks.append(f) or 17,
                          draw_handler_remove=lambda *a:removed.append(a))
    bpy=SimpleNamespace(context=ctx,types=SimpleNamespace(SpaceView3D=space))
    monkeypatch.setattr(manager,"draw_pixels",lambda *args:None)
    manager.register(bpy,lambda:blocked[0]);manager.register(bpy,lambda:False)
    assert len(callbacks)==1
    callbacks[0]();assert owner.events==[]
    blocked[0]=False;callbacks[0]();assert owner.events[0][0]=="presented"
    manager.discard_owner(owner)
    assert removed==[(17,"WINDOW")] and manager.handler is None
    manager.unregister(bpy)
    assert removed==[(17,"WINDOW")]
    replacement=Owner()
    queue(manager,ctx,replacement,7)
    manager.register(bpy,lambda:False)
    assert len(callbacks)==2
    callbacks[1]()
    assert replacement.events[0][1]['publication']==7
    assert len(owner.events)==1
    manager.discard_owner(replacement)
    assert removed==[(17,"WINDOW"),(17,"WINDOW")]


def test_job_thread_retirement_releases_gpu_references_on_main_timer_only():
    manager=DisplayOverlay();ctx=context();owner=Owner();queue(manager,ctx,owner)
    callbacks=[];removed=[]
    manager.bpy=SimpleNamespace(app=SimpleNamespace(timers=SimpleNamespace(register=lambda fn,**kw:callbacks.append(fn))),
        types=SimpleNamespace(SpaceView3D=SimpleNamespace(draw_handler_remove=lambda *args:removed.append(threading.current_thread()))))
    manager.handler=17
    thread=threading.Thread(target=lambda:manager.discard_owner(owner))
    thread.start();thread.join(2)
    assert not thread.is_alive() and len(callbacks)==1
    assert manager.entries and manager.handler==17 and removed==[]
    callbacks[0]()
    assert manager.entries=={} and manager.handler is None
    assert removed==[threading.main_thread()]


def test_job_retirement_refuses_draw_before_cleanup_timer(monkeypatch):
    manager=DisplayOverlay();ctx=context();owner=Owner();queue(manager,ctx,owner)
    callbacks=[]
    manager.bpy=SimpleNamespace(app=SimpleNamespace(timers=SimpleNamespace(register=lambda fn,**kw:callbacks.append(fn))))
    thread=threading.Thread(target=lambda:manager.discard_owner(owner))
    thread.start();thread.join(2)
    assert len(callbacks)==1 and manager.entries
    monkeypatch.setattr(manager,"draw_pixels",lambda *args:pytest.fail("retired owner drew before cleanup timer"))
    manager.present(ctx)
    assert owner.events==[] and manager.entries=={}
    callbacks[0]()


def registered_manager(ctx,blocked=lambda:False):
    """A drawer with its POST_VIEW handler registered through a fake bpy."""
    manager=DisplayOverlay();callbacks=[];removed=[];timers=[]
    space=SimpleNamespace(draw_handler_add=lambda f,*a:callbacks.append(f) or 17,
                          draw_handler_remove=lambda *a:removed.append(a))
    bpy=SimpleNamespace(context=ctx,types=SimpleNamespace(SpaceView3D=space),
                        app=SimpleNamespace(timers=SimpleNamespace(register=lambda fn,**kw:timers.append(fn))))
    manager.register(bpy,blocked)
    return manager,callbacks,removed,timers


def test_discard_work_keeps_handler_shader_and_buffers(display_gpu):
    ctx=drawable_context();owner=Owner();other_ctx=drawable_context();other=Owner()
    manager,callbacks,removed,_timers=registered_manager(ctx)
    manager.line_pass=line_pass(display_gpu)
    queue(manager,ctx,owner,1);queue(manager,other_ctx,other,5)
    callbacks[0]()
    key=view_key(ctx)
    shader,buffer=manager.shader,manager.buffers[key]
    assert shader is not None and owner.events[0][0]=="presented"
    manager.claimed.add(key);manager.labelled.add(key)
    manager.discard_work(owner)
    # Only this owner's queued draw and its view markers go; the drawer stays.
    assert key not in manager.entries and view_key(other_ctx) in manager.entries
    assert key not in manager.claimed and key not in manager.labelled
    assert manager.handler==17 and removed==[] and manager.shader is shader and manager.buffers[key] is buffer
    manager.discard_work(other)
    assert manager.entries=={} and manager.handler==17 and removed==[]
    # The next frame reuses all of it: no re-registration, compile or reallocation.
    queue(manager,ctx,owner,2)
    manager.register(manager.bpy,lambda:False)
    callbacks[0]()
    assert len(callbacks)==1 and manager.shader is shader and manager.buffers[key] is buffer
    assert owner.events[-1][1]["publication"]==2


def test_discard_work_prevents_stale_image_draw(display_gpu,monkeypatch):
    ctx=drawable_context();owner=Owner()
    manager,callbacks,removed,_timers=registered_manager(ctx)
    old=queue(manager,ctx,owner,1)
    manager.note_direct(ctx,owner,object())
    manager.discard_work(owner)
    monkeypatch.setattr(manager,"draw_pixels",lambda *args:pytest.fail("discarded image drew"))
    assert not claims_view(ctx)
    callbacks[0]()
    # Never drawn, never claimed as presented, and its direct packet is gone too.
    assert owner.events==[] and old.completion is not None and manager.direct=={}
    assert manager.handler==17 and removed==[]
    new=queue(manager,ctx,owner,2)
    monkeypatch.setattr(manager,"draw_pixels",lambda ctx,work,key:owner.events.append(("draw",work)))
    callbacks[0]()
    assert [event[0] for event in owner.events]==["draw","presented"]
    assert owner.events[0][1] is new and owner.events[1][1]["publication"]==2


def test_discard_owner_still_tears_down(display_gpu):
    ctx=drawable_context();owner=Owner()
    manager,callbacks,removed,_timers=registered_manager(ctx)
    manager.line_pass=line_pass(display_gpu)
    queue(manager,ctx,owner,1)
    callbacks[0]()
    key=view_key(ctx)
    assert manager.shader is not None and key in manager.buffers
    manager.note_direct(ctx,owner,object());manager.claimed.add(key);manager.labelled.add(key)
    manager.discard_owner(owner)
    assert removed==[(17,"WINDOW")] and manager.handler is None and manager.shader is None
    assert manager.entries=={} and manager.buffers=={} and manager.direct=={}
    assert not manager.claimed and not manager.labelled


def test_discard_work_from_job_thread_defers_to_main_timer(monkeypatch):
    ctx=context();owner=Owner()
    manager,callbacks,removed,timers=registered_manager(ctx)
    old=queue(manager,ctx,owner,1)
    manager.buffers[view_key(ctx)]="buffer";manager.shader="shader"
    thread=threading.Thread(target=lambda:manager.discard_work(owner))
    thread.start();thread.join(2)
    assert not thread.is_alive() and len(timers)==1
    # Nothing is dropped off the main thread, but the old image is already refused.
    assert manager.entries[view_key(ctx)] is old
    monkeypatch.setattr(manager,"draw_pixels",lambda *args:pytest.fail("discarded image drew before cleanup timer"))
    assert not claims_view(ctx)
    callbacks[0]()
    assert owner.events==[] and manager.entries[view_key(ctx)] is old
    timers[0]()
    assert manager.entries=={} and manager.discarded_work==[]
    assert manager.handler==17 and removed==[] and manager.shader=="shader"
    assert manager.buffers=={view_key(ctx):"buffer"}
    # Work queued after the discard is drawn normally.
    monkeypatch.setattr(manager,"draw_pixels",lambda *args:None)
    queue(manager,ctx,owner,2)
    callbacks[0]()
    assert owner.events[0][1]["publication"]==2


@pytest.mark.parametrize("invalidate",["retire","epoch","replacement","final_latch"])
def test_invalidation_during_admitted_draw_cannot_commit(monkeypatch,invalidate):
    manager=DisplayOverlay();ctx=context();owner=Owner();work=queue(manager,ctx,owner)
    callbacks=[]
    manager.bpy=SimpleNamespace(app=SimpleNamespace(timers=SimpleNamespace(register=lambda fn,**kw:callbacks.append(fn))))
    def draw(*args):
        if invalidate=="retire":
            thread=threading.Thread(target=lambda:manager.discard_owner(owner))
            thread.start();thread.join(2)
            assert not thread.is_alive() and len(callbacks)==1
        elif invalidate=="epoch":owner._viewport_display_epoch=object()
        elif invalidate=="final_latch":
            thread=threading.Thread(target=lambda:setattr(manager,"blocked",lambda:True))
            thread.start();thread.join(2)
            assert not thread.is_alive()
        else:queue(manager,ctx,owner,2)
    monkeypatch.setattr(manager,"draw_pixels",draw)
    manager.present(ctx)
    assert owner.events==[] and work.completion is None
    if invalidate=="retire":
        assert manager.entries,'resource cleanup must remain queued on main'
        callbacks[0]()
        assert manager.entries=={}


class FakeMatrix:
    def __init__(self, name="view"):
        self.name=name
    @staticmethod
    def Identity(size):
        return FakeMatrix("identity")
    def __matmul__(self, other):
        return FakeMatrix(self.name+"@"+getattr(other,"name","?"))


@pytest.fixture
def display_gpu(monkeypatch):
    """Fake gpu/gpu_extras/mathutils that record draw_pixels' call sequence and matrix depth."""
    record=SimpleNamespace(calls=[],depth=0,blend="NONE",depth_test="NONE")
    @contextmanager
    def push_pop():
        record.depth+=1
        try:yield
        finally:record.depth-=1
    def blend_set(value):record.blend=value
    def depth_test_set(value):record.depth_test=value
    framebuffer=SimpleNamespace(read_color=lambda *a,**k:record.calls.append(("readback",record.depth)),
                                clear=lambda **k:record.calls.append(("clear",record.depth)))
    class Info(SimpleNamespace):
        def __getattr__(self,name):return lambda *a,**k:None
    shader=SimpleNamespace(bind=lambda:None,uniform_float=lambda *a:None,uniform_sampler=lambda *a:None)
    gpu=SimpleNamespace(
        state=SimpleNamespace(active_framebuffer_get=lambda:framebuffer,blend_get=lambda:record.blend,blend_set=blend_set,
                              depth_test_get=lambda:record.depth_test,depth_test_set=depth_test_set),
        types=SimpleNamespace(Buffer=lambda *a:object(),GPUTexture=lambda *a,**k:"overlay-texture",
                              GPUStageInterfaceInfo=lambda *a:Info(),GPUShaderCreateInfo=lambda:Info()),
        shader=SimpleNamespace(create_from_info=lambda info:shader),
        matrix=SimpleNamespace(push_pop=push_pop,push_pop_projection=push_pop,load_matrix=lambda m:None,
                               load_projection_matrix=lambda m:None,get_projection_matrix=FakeMatrix,
                               get_model_view_matrix=FakeMatrix))
    def batch_for_shader(shader,kind,content):
        return SimpleNamespace(draw=lambda shader:record.calls.append(("image",record.depth)))
    def draw_texture_2d(texture,position,width,height):
        record.calls.append(("replay",record.depth,record.blend))
    monkeypatch.setitem(sys.modules,"gpu",gpu)
    monkeypatch.setitem(sys.modules,"gpu_extras",SimpleNamespace())
    monkeypatch.setitem(sys.modules,"gpu_extras.batch",SimpleNamespace(batch_for_shader=batch_for_shader))
    monkeypatch.setitem(sys.modules,"gpu_extras.presets",SimpleNamespace(draw_texture_2d=draw_texture_2d))
    monkeypatch.setitem(sys.modules,"mathutils",SimpleNamespace(Matrix=FakeMatrix))
    # The label pass draws nothing unless a test records it.
    monkeypatch.setattr(viewport_display_overlay,"_indicator_label_pass",lambda ctx,packet:True)
    return record


def drawable_context():
    ctx=context()
    ctx.space_data.overlay=SimpleNamespace(show_overlays=True)
    return ctx


def line_pass(record,owner=None):
    def draw(ctx,packet):
        record.calls.append(("lines",record.depth,packet))
        if owner is not None:
            owner.events.append(("lines",packet))
        return True
    return draw


def test_display_pass_draws_lines_after_image_before_overlay_replay(display_gpu):
    manager=DisplayOverlay();ctx=drawable_context();owner=Owner()
    packet=object();owner._presentation_packet_is_current=lambda value:value is packet
    work=queue(manager,ctx,owner)
    work.completion["prepared_presentation"]=packet
    manager.line_pass=line_pass(display_gpu)
    manager.present(ctx)
    kinds=[call[0] for call in display_gpu.calls]
    assert kinds==["readback","clear","image","lines","replay"]
    calls=dict((call[0],call) for call in display_gpu.calls)
    # The image and the replay use the pixel-space block; the lines use the view's own matrices.
    assert calls["image"][1]>0 and calls["replay"][1]>0 and calls["lines"][1]==0
    assert calls["lines"][2] is packet and calls["replay"][2]=="ALPHA_PREMULT"
    assert (display_gpu.blend,display_gpu.depth_test)==("NONE","NONE")
    assert [event[0] for event in owner.events]==["presented"]


@pytest.mark.parametrize("path",["viewport","cached"])
def test_display_pass_draws_lines_before_presentation_commit(display_gpu,path):
    manager=DisplayOverlay();ctx=drawable_context();owner=Owner()
    packet=object();frame=object();owner._presentation_packet_is_current=lambda value:value is packet
    work=queue(manager,ctx,owner)
    if path=="viewport":
        work.completion["prepared_presentation"]=packet
    else:
        work.completion=None;work.presentation_frame=frame;work.presentation_packet=packet
    manager.line_pass=line_pass(display_gpu,owner)
    manager.present(ctx)
    assert [event[0] for event in owner.events]==["lines","presented" if path=="viewport" else "cached"]
    assert owner.events[0][1] is packet
    if path=="cached":
        assert owner.events[1][1:]==(frame,packet)


def label_pass(record,owner=None):
    def draw(ctx,packet):
        record.calls.append(("labels",record.depth,packet,record.blend))
        if owner is not None:
            owner.events.append(("labels",packet))
        return True
    return draw


@pytest.mark.parametrize("path",["viewport","cached"])
def test_labels_drawn_in_display_pass_after_lines(display_gpu,path):
    manager=DisplayOverlay();ctx=drawable_context();owner=Owner()
    packet=object();frame=object();owner._presentation_packet_is_current=lambda value:value is packet
    work=queue(manager,ctx,owner)
    if path=="viewport":
        work.completion["prepared_presentation"]=packet
    else:
        work.completion=None;work.presentation_frame=frame;work.presentation_packet=packet
    manager.line_pass=line_pass(display_gpu,owner)
    manager.label_pass=label_pass(display_gpu,owner)
    manager.present(ctx)
    # Image, lines, then labels, all before the overlay replay and the commit.
    assert [call[0] for call in display_gpu.calls]==["readback","clear","image","lines","labels","replay"]
    calls=dict((call[0],call) for call in display_gpu.calls)
    # Lines use the view's matrices; labels are in the pixel-space block, before
    # the premultiplied replay changes the blend mode.
    assert calls["lines"][1]==0 and calls["labels"][1]>0 and calls["labels"][3]=="NONE"
    # Both passes get the presented frame's packet.
    assert calls["lines"][2] is packet and calls["labels"][2] is packet
    assert [event[0] for event in owner.events]==["lines","labels","presented" if path=="viewport" else "cached"]
    # The view is marked for the standalone label pass of this redraw, once.
    assert labels_drawn(ctx) and not labels_drawn(ctx)


def test_label_pass_failure_still_commits_and_leaves_view_to_standalone_labels(display_gpu):
    manager=DisplayOverlay();ctx=drawable_context();owner=Owner()
    def fail(ctx,packet):raise RuntimeError("font gone")
    manager.line_pass=line_pass(display_gpu)
    manager.label_pass=fail
    queue(manager,ctx,owner,1)
    manager.present(ctx)
    assert owner.events[0]==("error","Physics indicator labels were not drawn: font gone")
    assert owner.events[1][0]=="presented"
    assert [call[0] for call in display_gpu.calls][-2:]==["lines","replay"]
    # Unmarked: the standalone label pass still draws this view's labels.
    assert not labels_drawn(ctx)


def test_line_pass_failure_still_commits_frame(display_gpu):
    manager=DisplayOverlay();ctx=drawable_context();owner=Owner()
    def fail(ctx,packet):raise RuntimeError("line batch gone")
    manager.line_pass=fail
    queue(manager,ctx,owner,1)
    manager.present(ctx)
    assert owner.events[0]==("error","Physics indicator lines were not drawn: line batch gone")
    assert owner.events[1][0]=="presented" and owner.events[1][1]["publication"]==1
    # The overlay replay still runs after a failed line pass.
    assert [call[0] for call in display_gpu.calls][-1]=="replay"
    # A repeated failure is reported once; every frame still commits.
    queue(manager,ctx,owner,2)
    manager.present(ctx)
    assert [event[0] for event in owner.events]==["error","presented","presented"]


def redraw(manager,ctx,counts):
    """One redraw: the standalone handler (registered first) decides, then present()."""
    if not claims_view(ctx):
        counts.append("standalone")
    manager.present(ctx)


@pytest.mark.parametrize("rejection",["none","stale_packet","draw_failure","failure_after_lines","no_entry"])
def test_rejected_display_draw_falls_back_to_standalone_pass(display_gpu,monkeypatch,rejection):
    manager=DisplayOverlay();ctx=drawable_context();owner=Owner();manager.handler=17
    counts=[]
    packet=object();current=[packet]
    owner._presentation_packet_is_current=lambda value:value is current[0]
    if rejection!="no_entry":
        queue(manager,ctx,owner).completion["prepared_presentation"]=packet
    manager.line_pass=lambda ctx,packet:counts.append("display") or True
    manager.standalone_pass=lambda ctx:counts.append("fallback") or True
    if rejection=="stale_packet":
        # The marker is decided first; the packet goes stale before present() admits it.
        assert claims_view(ctx)
        current[0]=object()
        manager.present(ctx)
        assert counts==["fallback"] and owner.events==[] and manager.entries=={}
        return
    if rejection=="draw_failure":
        def fail(*args):raise RuntimeError("readback refused")
        monkeypatch.setattr(manager,"draw_pixels",fail)
    elif rejection=="failure_after_lines":
        def fail(ctx,work,key):
            manager._draw_lines(ctx,work)
            raise RuntimeError("replay refused")
        monkeypatch.setattr(manager,"draw_pixels",fail)
    redraw(manager,ctx,counts)
    # Exactly one line pass for the view in this redraw.
    expected={"none":["display"],"draw_failure":["fallback"],"failure_after_lines":["display"],
              "no_entry":["standalone"]}[rejection]
    assert counts==expected
    assert not manager.claimed
    # Only a display pass that reached the label pass marks the view for POST_PIXEL.
    assert labels_drawn(ctx)==(rejection in ("none",))
