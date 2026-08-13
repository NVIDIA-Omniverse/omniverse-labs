# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Crash-resilient session checkpoint: snapshot shape, atomic write/load, /api/open merge."""
import numpy as np
from fastapi.testclient import TestClient

from dev_variant_presenter import session
from dev_variant_presenter.api.routes import create_app
from dev_variant_presenter.config import Settings
from tests.test_routes import FakeRuntime, _info


class _State:
    user_usd = "x.usd"
    source_url = ""
    active_camera = "/W/Cam"
    stage_info = object()
    live_selection = [{"prim_path": "/W/Looks", "set_name": "Carpaint", "variant": "Sakura"}]


class _Cam:
    distance = 7.5
    def to_xform(self):
        return np.arange(16, dtype=np.float64).reshape(4, 4)


class _Rt:
    _camera = _Cam()
    _looks = {"/W/Cam": {"iso": 400.0}, "/W/Other": {}}
    _xforms = {"/W/Cam": {"m": [1.0] * 16, "dist": 5.0}}
    _settings = Settings(stream_resolution=(1920, 1080))


def test_snapshot_captures_session_and_skips_empty():
    snap = session.snapshot(_State(), _Rt())
    assert snap["usd"] == "x.usd"
    assert snap["camera"] == "/W/Cam"
    assert snap["selection"][0]["variant"] == "Sakura"
    assert snap["looks"] == {"/W/Cam": {"iso": 400.0}}      # empty look dropped
    assert snap["xforms"]["/W/Cam"]["dist"] == 5.0
    assert snap["pose"]["m"] == [float(i) for i in range(16)]
    assert snap["pose"]["dist"] == 7.5
    assert snap["resolution"] == [1920, 1080]
    s = _State(); s.user_usd = ""
    assert session.snapshot(s, _Rt()) is None               # no stage -> no checkpoint


def test_checkpoint_roundtrip(tmp_path):
    p = tmp_path / "session.json"
    snap = session.snapshot(_State(), _Rt())
    import json
    session._write_atomic(p, json.dumps(snap))
    assert session.load(str(p)) == snap
    assert session.load(str(tmp_path / "missing.json")) is None


def _checkpoint(usd="x.usd"):
    return {"usd": usd, "camera": "/W/Cam",
            "selection": [{"prim_path": "/W/Looks", "set_name": "Carpaint", "variant": "Sakura"}],
            "looks": {"/W/Cam": {"iso": 400.0}},
            "xforms": {"/W/Cam": {"m": [1.0] * 16, "dist": 5.0}},
            "pose": {"m": [float(i) for i in range(16)], "dist": 7.5},
            "resolution": [1920, 1080]}


def _client(monkeypatch, ck):
    import dev_variant_presenter.api.routes as R
    monkeypatch.setattr(R, "scan_stage", lambda p, e=(): _info(p))
    monkeypatch.setattr(R, "read_camera", lambda p, c, e=(): (tuple(float(i) for i in range(16)), 5.0))
    monkeypatch.setattr(R.session, "load", lambda path=session.DEFAULT_PATH: ck)
    rt = FakeRuntime()
    return rt, TestClient(create_app(rt, Settings()))


def test_open_restores_matching_checkpoint(monkeypatch):
    rt, c = _client(monkeypatch, _checkpoint())
    r = c.post("/api/open", json={"usd_path": "x.usd"})
    assert r.status_code == 200
    assert r.json()["camera"] == "/W/Cam"
    assert r.json()["selection"][0]["variant"] == "Sakura"
    op = next(cmd for cmd in rt.posted if type(cmd).__name__ == "OpenStage")
    assert op.looks == {"/W/Cam": {"iso": 400.0}}            # committed optics survive
    assert op.xforms == {"/W/Cam": {"m": [1.0] * 16, "dist": 5.0}}   # committed framings survive
    assert op.selection[0].variant == "Sakura"               # variant selection survives
    mv = next(cmd for cmd in rt.posted if type(cmd).__name__ == "MoveCamera")
    assert mv.matrix == tuple(float(i) for i in range(16))   # exact viewer pose survives
    assert mv.focus_distance == 7.5
    res = next(cmd for cmd in rt.posted if type(cmd).__name__ == "SetResolution")
    assert (res.width, res.height) == (1920, 1080)           # stream size survives


def test_open_ignores_checkpoint_for_other_stage(monkeypatch):
    rt, c = _client(monkeypatch, _checkpoint(usd="other.usd"))
    r = c.post("/api/open", json={"usd_path": "x.usd"})
    assert r.status_code == 200
    op = next(cmd for cmd in rt.posted if type(cmd).__name__ == "OpenStage")
    assert op.looks == {} and op.xforms == {}
    assert op.selection[0].variant == "Noir"                 # stage default, not checkpoint
    assert not any(type(cmd).__name__ in ("MoveCamera", "SetResolution") for cmd in rt.posted)


def test_open_explicit_state_beats_checkpoint(monkeypatch):
    rt, c = _client(monkeypatch, _checkpoint())
    body = {"usd_path": "x.usd", "camera_path": "/W/Cam",
            "selections": [{"prim_path": "/W/Looks", "set_name": "Carpaint", "variant": "Noir"}],
            "looks": {"/W/Cam": {"iso": 100.0}}}
    c.post("/api/open", json=body)
    op = next(cmd for cmd in rt.posted if type(cmd).__name__ == "OpenStage")
    assert op.looks == {"/W/Cam": {"iso": 100.0}}            # the project's looks, not the checkpoint's
    assert op.selection[0].variant == "Noir"                 # the explicit selection
