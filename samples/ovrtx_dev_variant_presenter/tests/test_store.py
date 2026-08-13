# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import dev_variant_presenter.store as store


def _proj(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "_PROJ", tmp_path / "projects")


def test_project_round_trip(monkeypatch, tmp_path):
    _proj(monkeypatch, tmp_path)
    store.save_project("Demo", "x.usd",
                       [{"prim_path": "/W", "set_name": "Carpaint", "variant": "Noir"}],
                       {"exposure": 0.5}, timelines={},
                       looks={"/Cam": {"iso": 300}}, xforms={}, camera="/Cam")
    assert any(p["name"] == "Demo" for p in store.list_projects())
    rec = store.load_project("Demo")
    assert rec["usd_path"] == "x.usd" and rec["display"]["exposure"] == 0.5
    assert rec["base_selection"][0]["variant"] == "Noir"
    assert rec["looks"]["/Cam"]["iso"] == 300
    assert store.delete_project("Demo") is True


def test_track_views_live_in_their_project(monkeypatch, tmp_path):
    """A view is saved/listed/loaded/deleted INSIDE its project — never a global pool."""
    _proj(monkeypatch, tmp_path)
    tl = {"duration_s": 4.0, "fps": 60.0, "tracks": [
        {"kind": "camera", "clips": [{"value": "/TurntableRig/Turntable",
                                      "start_s": 0.0, "duration_s": 4.0}]}]}
    # no project yet -> nowhere valid to live
    assert store.save_project_view("Demo", "Hero", tl) is False
    store.save_project("Demo", "x.usd", [], {}, timelines={})
    assert store.save_project_view("Demo", "Hero", tl) is True
    assert store.project_view_names("Demo") == ["Hero"]
    rec = store.load_project_view("Demo", "Hero")
    assert rec["timeline"] == tl and rec["usd_path"] == "x.usd"
    # a different project never sees it
    store.save_project("Other", "y.usd", [], {}, timelines={})
    assert store.project_view_names("Other") == []
    assert store.delete_project_view("Demo", "Hero") is True
    assert store.project_view_names("Demo") == []
    assert store.delete_project_view("Demo", "Hero") is False


def test_workspace_resave_preserves_the_view_library(monkeypatch, tmp_path):
    """save_project(timelines=None) must NOT wipe views added via save_project_view —
    a workspace Ctrl+S keeps the named-view library intact."""
    _proj(monkeypatch, tmp_path)
    store.save_project("Demo", "x.usd", [], {}, timelines={})
    store.save_project_view("Demo", "Hero", {"duration_s": 1, "fps": 60, "tracks": []})
    # re-save the workspace (new camera/timeline) WITHOUT passing timelines
    store.save_project("Demo", "x.usd", [], {"exposure": 1.0}, timelines=None,
                       timeline={"duration_s": 2, "fps": 60, "tracks": []}, camera="/Cam")
    rec = store.load_project("Demo")
    assert "Hero" in rec["timelines"]               # library survived the workspace save
    assert rec["display"]["exposure"] == 1.0        # workspace fields updated
    assert rec["camera"] == "/Cam"


def test_safe_project_name_sanitizes(monkeypatch, tmp_path):
    _proj(monkeypatch, tmp_path)
    store.save_project("a/b:c*?", "x.usd", [], {}, timelines={})
    files = list((tmp_path / "projects").glob("*.json"))
    assert len(files) == 1 and "/" not in files[0].name


def test_project_saves_the_workspace_timeline_and_camera(monkeypatch, tmp_path):
    """The project carries the WORKSPACE (editor strip's working timeline + selected camera)
    so open restores what you were looking at."""
    _proj(monkeypatch, tmp_path)
    wt = {"duration_s": 4.0, "fps": 60, "tracks": [
        {"kind": "camera", "clips": [{"value": "/Cam/B", "start_s": 0.0, "duration_s": 4.0}]}]}
    store.save_project("Demo", "x.usd", [], {}, timelines={}, timeline=wt, camera="/Cam/B")
    rec = store.load_project("Demo")
    assert rec["timeline"] == wt
    assert rec["camera"] == "/Cam/B"
    assert rec.get("timelines") == {}
