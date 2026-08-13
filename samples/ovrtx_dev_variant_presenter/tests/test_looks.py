# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-camera display looks: each authored camera keeps its own optics (ISO/FOV/DOF/focus),
so the look you dial in Configure flows into the timeline render at that camera's cuts.

These exercise the pure look bookkeeping on RenderRuntime — no ovrtx (start() is never called,
_has_stage stays False, so the methods just mutate self._looks and never reopen)."""
import numpy as np

from dev_variant_presenter.config import Settings
from dev_variant_presenter.render.runtime import (
    RenderRuntime, SetCamera, SetCameraState, SnapToCamera,
)

IDENT = tuple(float(v) for v in np.eye(4).reshape(-1))


def _rt():
    return RenderRuntime(Settings())


def test_set_camera_writes_the_active_cameras_look():
    rt = _rt()
    rt._camera_path = "/Cam/A"
    rt._do_set_camera(SetCamera(params={"f_stop": 2.8, "focal_length": 85}))
    assert rt._looks["/Cam/A"] == {"f_stop": 2.8, "focal_length": 85}


def test_each_camera_keeps_its_own_look():
    rt = _rt()
    rt._camera_path = "/Cam/A"
    rt._do_set_camera(SetCamera(params={"focal_length": 85, "f_stop": 2.0}))
    rt._camera_path = "/Cam/B"
    rt._do_set_camera(SetCamera(params={"focal_length": 24, "iso": 400}))
    assert rt._looks["/Cam/A"] == {"focal_length": 85, "f_stop": 2.0}
    assert rt._looks["/Cam/B"] == {"focal_length": 24, "iso": 400}
    assert rt._active_look() == rt._looks["/Cam/B"]   # active follows the selected camera


def test_snap_switches_the_active_look_and_emits_it():
    events = []
    rt = RenderRuntime(Settings(), on_event=events.append)
    rt._camera_path = "/Cam/A"
    rt._do_set_camera(SetCamera(params={"iso": 800}))          # indoor
    rt._camera_path = "/Cam/B"
    rt._do_set_camera(SetCamera(params={"iso": 100}))          # outdoor
    events.clear()
    rt._do_snap(SnapToCamera(camera_xform=IDENT, focus_distance=0.0, camera_path="/Cam/A"))
    assert rt._camera_path == "/Cam/A"
    params = [e for e in events if e.get("type") == "camera_params"][-1]["params"]
    assert params == {"iso": 800}                              # snapping restores that camera's look


def test_f_stop_zero_clears_dof_for_that_camera():
    rt = _rt()
    rt._camera_path = "/Cam/A"
    rt._do_set_camera(SetCamera(params={"f_stop": 4.0, "focus_distance": 500}))
    rt._do_set_camera(SetCamera(params={"f_stop": 0}))         # 0 => DOF off
    assert "f_stop" not in rt._looks["/Cam/A"]
    assert rt._looks["/Cam/A"].get("focus_distance") == 500    # focus value is kept


def test_effective_look_auto_focuses_live_but_not_in_timeline():
    class Ctl:
        distance = 300.0
    look = {"f_stop": 2.8}
    live = RenderRuntime._effective_look(look, Ctl())          # live: orbit target fills focus
    assert live["focus_distance"] == 300.0
    tl = RenderRuntime._effective_look(look, None)             # timeline: no orbit target -> unset
    assert "focus_distance" not in tl
    assert look == {"f_stop": 2.8}                             # never mutates the stored look


def test_snapshot_and_restore_round_trip_drops_empties():
    rt = _rt()
    rt._looks = {"/Cam/A": {"iso": 800}, "/Cam/B": {}, "/Cam/C": {"focal_length": 35}}
    snap = rt._snapshot_looks()
    assert snap == {"/Cam/A": {"iso": 800}, "/Cam/C": {"focal_length": 35}}   # empty /Cam/B dropped
    rt2 = _rt()
    rt2._do_set_camera_state(SetCameraState(looks=snap, xforms={}))
    assert rt2._looks == snap


# ----- per-camera transform overrides (your live framing: orbit / pan / dolly) -----

def test_navigation_is_free_and_framing_commits_explicitly():
    rt = _rt()
    rt._camera_path = "/Cam/A"
    rt._cam_in.update(px=5.0, py=-3.0, dirty=True)   # a pan gesture
    rt._apply_camera_input()
    assert "/Cam/A" not in rt._xforms                # orbiting never rewrites a shot camera
    rt._capture_xform()                              # the explicit Save framing action
    ov = rt._xforms["/Cam/A"]
    assert len(ov["m"]) == 16 and ov["dist"] > 0


def test_snap_restores_a_cameras_framing_override():
    rt = _rt()
    rt._camera_path = "/Cam/A"
    rt._cam_in.update(py=4.0, dirty=True)            # reframe camera A
    rt._apply_camera_input()
    rt._capture_xform()                              # commit (Save framing)
    saved = list(rt._camera.to_xform().reshape(-1))
    # go to B (authored), then back to A — A should restore its saved framing, not the authored xform
    rt._do_snap(SnapToCamera(camera_xform=IDENT, focus_distance=0.0, camera_path="/Cam/B"))
    rt._do_snap(SnapToCamera(camera_xform=IDENT, focus_distance=0.0, camera_path="/Cam/A"))
    assert np.allclose(rt._camera.to_xform().reshape(-1), saved, atol=1e-6)


def test_reset_drops_the_framing_override_and_snaps_authored():
    rt = _rt()
    rt._camera_path = "/Cam/A"
    rt._cam_in.update(px=9.0, dirty=True)
    rt._apply_camera_input()
    rt._capture_xform()
    assert "/Cam/A" in rt._xforms
    rt._do_snap(SnapToCamera(camera_xform=IDENT, focus_distance=0.0, camera_path="/Cam/A", reset=True))
    assert "/Cam/A" not in rt._xforms                # override dropped
    assert np.allclose(rt._camera.to_xform().reshape(-1), IDENT, atol=1e-6)   # back to authored


def test_camera_state_round_trips_both_looks_and_xforms():
    rt = _rt()
    rt._camera_path = "/Cam/A"
    rt._do_set_camera(SetCamera(params={"focal_length": 50}))
    rt._cam_in.update(px=2.0, dirty=True)
    rt._apply_camera_input()
    rt._capture_xform()
    state = {"looks": rt._snapshot_looks(), "xforms": rt._snapshot_xforms()}
    assert state["looks"] and state["xforms"]
    rt2 = _rt()
    rt2._do_set_camera_state(SetCameraState(looks=state["looks"], xforms=state["xforms"]))
    assert rt2._looks == state["looks"]
    assert rt2._xforms == state["xforms"]


def test_same_size_set_resolution_is_a_noop():
    """A project open re-posts its saved display incl. resolution; an unchanged size must NOT
    rebuild the streamer (that drops the WebRTC client with nobody scheduled to reconnect)."""
    from dev_variant_presenter.render.runtime import SetResolution

    rt = _rt()   # Settings() default stream_resolution == (1280, 720)

    class StubStreamer:
        stopped = False
        def stop(self):
            self.stopped = True

    rt._streamer = StubStreamer()
    rt._do_set_resolution(SetResolution(1280, 720))
    assert rt._streamer.stopped is False          # same size -> untouched
    assert rt._settings.stream_resolution == (1280, 720)


def test_open_seeds_camera_state_and_uses_framing_override(monkeypatch):
    """A project's looks/xforms ride in WITH OpenStage so the single open-reopen already
    carries them (no original-view-then-settle); the open camera's saved framing beats the
    authored xform."""
    from dev_variant_presenter.models import StageInfo
    from dev_variant_presenter.render.runtime import OpenStage

    rt = _rt()
    monkeypatch.setattr(rt, "_reopen", lambda: None)
    monkeypatch.setattr(rt, "_write_camera", lambda: None)
    monkeypatch.setattr(rt, "_classify_async", lambda *a, **k: None)
    m = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 120.0, 30.0, 250.0, 1]
    info = StageInfo("x.usd", "World", "Y", 0, 100, 60, (), ())
    rt._do_open(OpenStage(
        user_usd="x.usd", selection=(), camera_path="/Cam/B",
        camera_xform=IDENT, focus_distance=0.0, stage_info=info,
        looks={"/Cam/B": {"iso": 400}, "/Cam/Empty": {}},
        xforms={"/Cam/B": {"m": m, "dist": 275.0}}))
    assert rt._looks == {"/Cam/B": {"iso": 400}}                 # seeded, empties dropped
    assert rt._xforms == {"/Cam/B": {"m": m, "dist": 275.0}}
    assert np.allclose(rt._camera.to_xform().reshape(-1), m, atol=1e-6)   # override wins


# --- render look = WYSIWYG: the active camera renders with the SAME effective look (incl. the
#     live orbit auto-focus) the viewport shows, so DOF matches what you see (bug: turntable
#     render applied f_stop with no focus -> wrong DOF plane) ---

class _Ctl:   # minimal stand-in for the orbit controller (_effective_look reads .distance)
    def __init__(self, distance):
        self.distance = distance


def test_render_look_uses_live_focus_for_the_active_camera():
    rt = _rt()
    rt._camera_path = "/Cam/TT"
    rt._looks = {"/Cam/TT": {"f_stop": 1.2, "focal_length": 40}}
    rt._camera = _Ctl(900.0)
    # the camera you're viewing: DOF focus = the live orbit distance -> render matches viewport
    assert rt._render_look("/Cam/TT") == {"f_stop": 1.2, "focal_length": 40, "focus_distance": 900.0}


def test_render_look_respects_an_explicit_focus_on_the_active_camera():
    rt = _rt()
    rt._camera_path = "/Cam/TT"
    rt._looks = {"/Cam/TT": {"f_stop": 1.2, "focus_distance": 500.0}}
    rt._camera = _Ctl(900.0)
    assert rt._render_look("/Cam/TT")["focus_distance"] == 500.0   # a picked focus is never overwritten


def test_render_look_for_a_non_active_camera_uses_its_stored_look():
    rt = _rt()
    rt._camera_path = "/Cam/Active"
    rt._looks = {"/Cam/Other": {"f_stop": 1.2}}
    rt._camera = _Ctl(900.0)
    # not the camera you're viewing -> no live controller -> stored look as-is (its own picked focus)
    assert "focus_distance" not in rt._render_look("/Cam/Other")
