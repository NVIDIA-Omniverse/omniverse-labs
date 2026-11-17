# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Application state transitions use real native views and lightweight UI sinks.

The engine, TCP producer, adapters, ownership tree, selection bus and playback
are real. The GPU and widgets are sinks here so assertions can inspect atomic
presentation changes without competing with an interactive Inspector session.
"""
from capture_semantics import test_scene_roles, unique_instance_id
from pathlib import Path
import socket
import struct
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest

from _pvd_native import Session
import _pvd_native
from ovd_ui.adapters import PvdDataAdapter
from ovui_widgets.common.selection import SelectionBus
from ovd_ui.application import Application
from ovd_ui.renderer import ViewportRenderer
from ovd_ui.interaction import InteractionState
from ovd_ui.row_actions import RowActionState
from ovd_ui.models import ObjectTreeModel
from ovd_ui.playback import Playback
from ovd_ui.widgets.properties import PropertyInspector
from native_fixtures import executable

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = (ROOT / "test/data/test_scene.ovd").read_bytes()
HANDSHAKE = struct.pack("<IHH", 0x4F56444C, 1, 0)
FRAME2_STOP = struct.pack("<BQQ", 12, 0x138AB174E00, 2)
FRAME2_END = PAYLOAD.index(FRAME2_STOP) + len(FRAME2_STOP)


class Value:
    def __init__(self, value=0): self.value = value
    def set_value(self, value): self.value = value
    @property
    def as_int(self): return int(self.value)
    def get_item_value_model(self): return self
    def add_value_changed_fn(self, callback): self.callback = callback


class Frame:
    def __init__(self): self.rebuilds = 0
    def rebuild(self): self.rebuilds += 1


class ClipsSink:
    def __init__(self):self.refreshes=0
    def refresh(self,*_):self.refreshes+=1


class Renderer:
    _frame_bounds = ViewportRenderer._frame_bounds
    selection_pivot = ViewportRenderer.selection_pivot
    def __init__(self): self.clear()
    def clear(self):
        self.snapshot = self.overlays = None
        self.settings = None
        self.scene_id = 0
        self.selected, self.hidden = set(), set()
        self.instance_alpha = None
    def set_snapshot(self, snapshot, reset_camera=False): self.snapshot = snapshot
    def set_overlays(self, overlays): self.overlays = overlays
    @property
    def instance_count(self):
        return int(np.count_nonzero(self.snapshot.scene_ids == self.scene_id)) if self.snapshot else 0


@pytest.fixture
def app():
    application = Application.__new__(Application)
    application.capture = application.session = application.adapter = application.presentation = None
    application.pending_presentation = application.pending_display = application.pending_playback = None
    application.presentation_dirty = application.refreshing = application.selection_guard = False
    application.playback = Playback()
    application.interaction = InteractionState()
    application.row_actions = RowActionState()
    application.replay_tree_rows = False
    application.interaction_replay = False
    application.selected_record = None
    application.selection = SelectionBus()
    application.selection_history = []
    application.history_index = -1
    application.object_model = ObjectTreeModel()
    application.object_browser = SimpleNamespace(refresh=lambda: None,refresh_tags=lambda signature=None:None,
                                                 close_context_menu=lambda:None,
                                                 ensure_native_visible=application.object_model.ensure_native_visible)
    application.tree = SimpleNamespace(selection=[], clear_selection=lambda: None,
                                       set_expanded=lambda *_: None)
    application.renderer = Renderer()
    application.palette = None
    application.scenes = []
    application.scene_id = application.scene_index = application.frame = 0
    application.overlay_mask = 0
    application.gizmo_scale, application.tolerance_speed = 1., 10.
    application.property_inspector = PropertyInspector(lambda *_: None, lambda *_: None,
        {"array_window": 10}, on_request=application._request_presentation)
    application.workflow = SimpleNamespace(live_session=None, follow_live=False,
        display_session=None, display_path=None, accept_connections=False, poll=lambda: ())
    for name in ("file_label", "summary_label", "viewport_info", "status_label",
                 "play_button", "reverse_button", "parity_button", "parity_slot", "accept_button"):
        setattr(application, name, SimpleNamespace(text="", enabled=True))
    application.frame_model = Value()
    application.mode_combo = SimpleNamespace(model=Value(), enabled=True)
    application.frame_slider = SimpleNamespace(min=0, max=1, set_domain=lambda *_:None)
    for name in ("property_frame", "scene_frame"):
        setattr(application, name, Frame())
    application.clips_view=ClipsSink()
    application.selection_subscription = application.selection.subscribe(application._selection_changed)
    return application


def open_capture(tmp_path, name="cloth_physx34.pxd2"):
    path = ROOT / "test/data" / name
    return path, Session(str(path), str(tmp_path))


def displayed(app):
    return (app.capture, app.session, app.presentation.generation, app.scene_id,
            app.frame, app.playback.frame, tuple(app.selection.get_snapshot().paths()),
            tuple((s.id, s.current_frame) for s in app.scenes),
            app.renderer.snapshot.transforms.tobytes(), app.viewport_info.text)


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if result := predicate(): return result
        time.sleep(.005)
    pytest.fail("Timed out waiting for native producer")


@pytest.fixture
def live(tmp_path):
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        port = candidate.getsockname()[1]
    source = Session.listen(port, str(tmp_path / "capture.part"))
    def connect():
        try: return socket.create_connection(("127.0.0.1", port), timeout=1)
        except (ConnectionRefusedError, TimeoutError): return None
    connection = wait_until(connect)
    try: yield source, connection
    finally:
        connection.close()
        source.stop_live()


def test_file_trim_and_split_survive_native_seeks_and_inspector_refresh(app, tmp_path):
    assert app._display(*open_capture(tmp_path))
    assert app.seek(30, snap=False)
    app._clip_action(app.playback.trim_end)
    assert app.playback.clip.stop == 30 and app.frame == app.playback.frame == 29
    assert app.seek(10, snap=False)
    app._request_presentation()
    assert app._refresh()
    assert app.playback.clip.stop == 30
    app._clip_action(app.playback.split)
    ranges = [(clip.start, clip.stop) for clip in app.playback.clips]
    assert ranges == [(1, 11), (11, 30)]
    assert app.seek(8, snap=False)
    assert [(clip.start, clip.stop) for clip in app.playback.clips] == ranges
    assert app.frame == app.playback.frame == app.presentation.frame == 8


def test_active_overlay_settings_are_included_when_switching_capture(app, tmp_path):
    app.overlay_mask = 3
    assert app._display(*open_capture(tmp_path))
    assert app.renderer.overlays is not None
    assert app.renderer.overlays.mask == 3
    assert app.renderer.overlays.current_frame == app.frame


def test_partial_live_seek_retains_whole_view_then_applies_after_detach(app, live):
    source, connection = live
    app.workflow.live_session = source
    connection.sendall(HANDSHAKE + PAYLOAD[:FRAME2_END])
    wait_until(lambda: source.info().completed_serial == 2)
    assert app._display(None, source)
    assert app.seek(2, snap=False)
    app.selection.publish(["/objects/1"], source="test")
    assert app._refresh()
    before = displayed(app)
    connection.sendall(PAYLOAD[FRAME2_END:])
    wait_until(lambda: source.info().recording_size == len(PAYLOAD)
               and source.info().unpublished_changes)
    assert not app.seek(1, snap=False)
    assert displayed(app) == before
    assert app.presentation_dirty and app.pending_presentation == (app.scene_id, 1)
    source.stop_live()
    assert app._refresh()
    assert app.frame == app.playback.frame == app.presentation.frame == 1
    assert app.selection.get_snapshot().paths() == ["/objects/1"]


def test_background_display_waits_for_safe_view_and_preserves_selection(app, tmp_path, live):
    assert app._display(*open_capture(tmp_path))
    app.selection.publish(["/objects/15"], source="test")
    assert app._refresh()
    before = displayed(app)
    source, connection = live
    app.workflow.live_session = source
    connection.sendall(HANDSHAKE + PAYLOAD[:1024])
    wait_until(lambda: source.info().unpublished_changes)
    assert not app._display(None, source)
    assert displayed(app) == before
    assert app.pending_display == (None, source)
    connection.sendall(PAYLOAD[1024:FRAME2_END])
    wait_until(lambda: source.info().completed_serial == 2)
    assert app._display(*app.pending_display)
    assert app.session is source and app.pending_display is None
    assert app.frame == app.presentation.frame == app.playback.frame == 2
    assert app.selection.get_snapshot().paths() == []


def test_record_shrink_and_object_death_keep_requested_frame_applied(app, tmp_path):
    assert app._display(*open_capture(tmp_path))
    ray = app.object_model.canonical_item(-70)
    app._tree_selection([ray])
    assert app._refresh()
    assert app.presentation.array_record(6, 401, 4).fields
    assert app.seek(2, snap=False)
    assert app.frame == app.presentation.frame == app.playback.frame == 2
    assert app.presentation.unavailable_ids == [-70]
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    actor=unique_instance_id(app.session,"PxRigidDynamic")
    app.selection.publish([f"/objects/{actor}"], source="test")
    assert app._refresh()
    assert app.seek(3, snap=False)
    assert app.frame == app.presentation.frame == app.playback.frame == 3
    assert app.presentation.unavailable_ids == [actor]
    assert len(app.renderer.snapshot.object_ids) == 0


def test_selection_restored_by_historical_seek_requests_its_owned_properties(app, tmp_path):
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    actor=unique_instance_id(app.session,"PxRigidDynamic")
    app.selection.publish([f"/objects/{actor}"], source="test")
    assert app._refresh()
    assert app.seek(3, snap=False)
    assert app.selection.get_snapshot().paths() == []
    assert app.seek(1, snap=False)
    assert app.selection.get_snapshot().paths() == [f"/objects/{actor}"]
    assert app.presentation_dirty
    assert app._refresh()
    assert app.presentation.inspected_ids == [actor]
    pose = app.adapter.property(actor, "globalPose")
    assert app.presentation.has_read(actor, pose.id)


def test_refresh_restores_reentrancy_guard_after_ui_failure(app, tmp_path):
    assert app._display(*open_capture(tmp_path))
    def fail(): raise RuntimeError("widget rebuild failed")
    app.object_browser.refresh = fail
    with pytest.raises(RuntimeError, match="widget rebuild failed"):
        app._refresh(frame=2)
    assert not app.refreshing and not app.selection_guard


def test_resampling_follow_never_snaps_into_an_unpublished_new_phase(app, live):
    source, connection = live
    payload = (ROOT / "test/data/resample_two_segment.ovd").read_bytes()
    first_end = 42117
    # Known first STOP_FRAME in the second recording segment of this fixture.
    first_stop = struct.pack("<BQQ", 12, 2929590687920, 1)
    second_first_end = payload.index(first_stop, first_end) + len(first_stop)
    connection.sendall(HANDSHAKE + payload[:first_end])
    wait_until(lambda: source.info().completed_serial == 120)
    app.workflow.live_session = source
    assert app._display(None, source)
    assert app.frame == 120
    old_scene = app.scene_id
    connection.sendall(payload[first_end:second_first_end])
    wait_until(lambda: source.info().completed_serial == 121)
    update = source.refresh_live()
    assert update.current_frame == 1 and update.max_frame >= 120
    app.workflow.follow_live = True
    event = SimpleNamespace(kind="frames", session=source, scenes=update.scenes,
        segment_changed=True, max_frame=update.max_frame, current_frame=update.current_frame)
    app.workflow.poll = lambda: [event]
    app._poll_workflow()
    assert app.scene_id != old_scene
    assert app.frame == app.presentation.frame == app.playback.frame == 1
    before = displayed(app)
    connection.sendall(payload[second_first_end:])
    wait_until(lambda: source.info().completed_serial == 241 and source.info().unpublished_changes)
    requested_index = next(i for i, scene in enumerate(app.scenes) if scene.id == old_scene)
    app._select_scene(Value(requested_index), None)
    assert displayed(app) == before
    assert app.pending_presentation[0] == old_scene
    source.stop_live()
    assert app._refresh()
    assert app.scene_id == old_scene
    assert app.frame == app.presentation.frame == app.playback.frame == 2


class WidgetSink:
    def __init__(self, *args, **kwargs):
        self.model = Value()
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def set_collapsed_changed_fn(self, _): pass
    def set_build_fn(self, callback): self.callback = callback
    def rebuild(self): self.callback()


@pytest.fixture
def inspector_ui(monkeypatch):
    from ovd_ui.widgets import properties
    frames, labels = [], []
    def frame():
        widget = WidgetSink()
        frames.append(widget)
        return widget
    def label(text, **kwargs):
        labels.append(text)
        return WidgetSink()
    monkeypatch.setattr(properties, "ui", SimpleNamespace(Label=label, Button=label,
        Separator=WidgetSink, Spacer=WidgetSink, HStack=WidgetSink, VStack=WidgetSink,
        ZStack=WidgetSink, Rectangle=WidgetSink, color=SimpleNamespace(pvd_subtle=0),
        CollapsableFrame=WidgetSink, Frame=frame, IntSlider=WidgetSink))
    return frames, labels


def generated_capture(tmp_path, hierarchy=False):
    binary = executable("pvd_geometry_fixture", "PVD_GEOMETRY_FIXTURE")
    path = tmp_path / ("hierarchy.ovd" if hierarchy else "animated.ovd")
    subprocess.run([str(binary), str(path), *(["--hierarchy"] if hierarchy else [])], check=True)
    return Session(str(path), str(tmp_path))


def test_delayed_numeric_inspector_builder_retains_owned_array_frame(tmp_path, inspector_ui):
    source = generated_capture(tmp_path)
    initial = source.presentation(frame=2)
    cls = next(c.id for c in initial.classes() if c.name == "AnalysisSample")
    identity = next(i.id for i in initial.instances() if i.class_id == cls)
    view = source.presentation(frame=2, inspect_ids=[identity])
    adapter = PvdDataAdapter(view)
    inspector = PropertyInspector(lambda *_: None, lambda *_: None, {"array_window": 10})
    inspector.build(adapter, [identity])
    frames, labels = inspector_ui
    old_builders = tuple(frames)
    adapter.session = source.presentation(frame=3, inspect_ids=[identity])
    adapter.reload()
    labels.clear()
    for frame in old_builders: frame.rebuild()
    assert "3" in labels and "4" not in labels and "5" not in labels


def test_delayed_reference_labels_retain_their_owned_name_frame(tmp_path, inspector_ui):
    source = generated_capture(tmp_path, hierarchy=True)
    view = source.presentation(frame=1)
    adapter = PvdDataAdapter(view)
    adapter.hierarchy()  # Populate the name caches before retaining the widget.
    inspector = PropertyInspector(lambda *_: None, lambda *_: None, {"array_window": 10})
    inspector.build(adapter, [view.scene_id])
    frames, labels = inspector_ui
    old_builders = tuple(frames)
    adapter.session = source.presentation(frame=3)
    adapter.reload()
    adapter.hierarchy()  # A new frame populates its current name caches.
    labels.clear()
    for frame in old_builders: frame.rebuild()
    assert any(label.startswith("Alpha [") for label in labels)
    assert not any(label.startswith("Beta [") for label in labels)


def test_shrinking_numeric_array_clamps_its_page_and_requests_owned_recapture(tmp_path, inspector_ui):
    source = generated_capture(tmp_path)
    initial = source.presentation(frame=3)
    cls = next(c.id for c in initial.classes() if c.name == "AnalysisSample")
    identity = next(i.id for i in initial.instances() if i.class_id == cls)
    prop = next(p for p in initial.properties(cls) if p.name == "variable")
    adapter = PvdDataAdapter(source.presentation(frame=3, inspect_ids=[identity]))
    requests = []
    inspector = PropertyInspector(lambda *_: None, lambda *_: None,
        {"array_window": 1}, on_request=lambda: requests.append(True))
    inspector.build(adapter, [identity])
    inspector.array_positions[(identity, prop.id)] = 1
    inspector.array_expanded.add((identity, prop.id))
    windows = inspector.read_requests(adapter, [identity])
    assert (identity, prop.id, 0, 8) in windows  # Legacy size1 migrates to a logical group10.
    adapter.session = source.presentation(frame=2, inspect_ids=[identity],
        read_requests=windows, allow_unavailable_requests=True)
    adapter.reload()
    inspector.build(adapter, [identity])
    assert inspector.array_positions[(identity, prop.id)] == 0
    assert requests
    assert (identity, prop.id, 0, 4) in inspector.read_requests(adapter, [identity])
