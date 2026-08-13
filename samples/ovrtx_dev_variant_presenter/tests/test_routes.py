# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from fastapi.testclient import TestClient

from dev_variant_presenter.api.routes import create_app
from dev_variant_presenter.config import Settings
from dev_variant_presenter.models import CameraInfo, StageInfo, VariantSetInfo


class FakeRuntime:
    """Records posted commands; never touches ovrtx."""
    def __init__(self):
        self.posted = []
        self.cancel_requests = 0

    def post(self, cmd):
        self.posted.append(cmd)

    def request_cancel(self):
        self.cancel_requests += 1

    def start(self):
        pass


def _info(usd="x.usd"):
    return StageInfo(usd, "World", "Y", 0, 100, 60,
                     (VariantSetInfo("Carpaint", "/W/Looks", ("Noir", "Sakura"), "Noir"),),
                     (CameraInfo("/W/Cam", "Cam"),))


def test_open_scans_and_returns_stage_info(monkeypatch):
    import dev_variant_presenter.api.routes as R
    monkeypatch.setattr(R, "scan_stage", lambda p, e=(): _info(p))
    monkeypatch.setattr(R, "read_camera", lambda p, c, e=(): (tuple(float(i) for i in range(16)), 5.0))
    rt = FakeRuntime()
    c = TestClient(create_app(rt, Settings()))
    r = c.post("/api/open", json={"usd_path": "x.usd"})
    assert r.status_code == 200
    body = r.json()
    assert body["default_prim"] == "World"
    assert body["variant_sets"][0]["set_name"] == "Carpaint"
    assert any(type(cmd).__name__ == "OpenStage" for cmd in rt.posted)


def test_variant_posts_set_selection():
    rt = FakeRuntime()
    c = TestClient(create_app(rt, Settings()))
    r = c.post("/api/variant", json={"selections": [
        {"prim_path": "/W/Looks", "set_name": "Carpaint", "variant": "Sakura"}]})
    assert r.status_code == 200
    sel_cmds = [cmd for cmd in rt.posted if type(cmd).__name__ == "SetSelection"]
    assert sel_cmds and sel_cmds[-1].selection[0].variant == "Sakura"


def test_config_reports_signal_port():
    rt = FakeRuntime()
    c = TestClient(create_app(rt, Settings(signal_port=49123)))
    r = c.get("/api/config")
    assert r.status_code == 200
    assert r.json()["signal_port"] == 49123


def test_batch_computes_count_applies_guard_and_posts(monkeypatch):
    import dev_variant_presenter.api.routes as R
    from dev_variant_presenter.models import StageInfo, VariantSetInfo
    info = StageInfo("x.usd", "World", "Y", 0, 100, 60, (
        VariantSetInfo("Carpaint", "/W/Looks", ("Noir", "Sakura"), "Noir"),
    ), ())
    monkeypatch.setattr(R, "scan_stage", lambda p, e=(): info)
    rt = FakeRuntime()
    c = TestClient(create_app(rt, Settings()))
    c.post("/api/open", json={"usd_path": "x.usd"})
    body = {"job": {
        "mode": "one_at_a_time",
        "base_selection": [{"prim_path": "/W/Looks", "set_name": "Carpaint", "variant": "Noir"}],
        "included": {"Carpaint": ["Noir", "Sakura"]},
        "cameras": ["/Cam"],
        "quality": {"mode": "RealTimePathTracing", "samples_per_pixel": 64,
                    "max_bounces": 4, "resolution": [640, 360]},
        "frame_mode": "single", "out_dir": "C:/out"}}
    r = c.post("/api/batch", json=body)
    assert r.status_code == 200 and r.json()["count"] == 2
    assert any(type(cmd).__name__ == "RunBatch" for cmd in rt.posted)


def test_batch_guard_blocks_without_confirm(monkeypatch):
    import dev_variant_presenter.api.routes as R
    from dev_variant_presenter.models import StageInfo, VariantSetInfo
    big = tuple(f"v{i}" for i in range(40))
    info = StageInfo("x.usd", "World", "Y", 0, 100, 60, (
        VariantSetInfo("A", "/W", big, big[0]),
        VariantSetInfo("B", "/W", big, big[0]),
    ), ())
    monkeypatch.setattr(R, "scan_stage", lambda p, e=(): info)
    c = TestClient(create_app(FakeRuntime(), Settings()))
    c.post("/api/open", json={"usd_path": "x.usd"})
    body = {"job": {"mode": "full_cartesian", "base_selection": [],
                    "included": {"A": list(big), "B": list(big)},
                    "cameras": ["/Cam"], "frame_mode": "single",
                    "out_dir": "C:/out"}}
    assert c.post("/api/batch", json=body).status_code == 409  # 1600 > 500


def _png2(path, size=(8, 8)):
    from PIL import Image
    Image.new("RGB", size, (10, 20, 30)).save(path)
    return path


def test_post_overlay_then_video_then_results(tmp_path):
    perm = tmp_path / "Carpaint-Noir"; perm.mkdir()
    for i in range(2):
        _png2(perm / f"{i:04d}.png", size=(48, 24))
    c = TestClient(create_app(FakeRuntime(), Settings()))
    assert c.post("/api/post/overlay", json={"out_dir": str(tmp_path)}).json()["count"] == 1
    assert c.post("/api/post/video", json={"out_dir": str(tmp_path), "fps": 24}).json()["count"] == 1
    assert (tmp_path / "Carpaint-Noir.mp4").exists()
    perms = c.get("/api/results", params={"dir": str(tmp_path)}).json()["permutations"]
    noir = next(p for p in perms if p["name"] == "Carpaint-Noir")
    assert noir["label"] == "Carpaint: Noir" and noir["video"].endswith("Carpaint-Noir.mp4")
    vid = c.get("/api/video", params={"path": noir["video"]})
    assert vid.status_code == 200 and vid.headers["content-type"] == "video/mp4"


def test_post_compress_no_ffmpeg_returns_null_path(tmp_path, monkeypatch):
    import dev_variant_presenter.post.processing as post
    monkeypatch.setattr(post.shutil, "which", lambda _n: None)
    clip = tmp_path / "Carpaint-Noir.mp4"; clip.write_bytes(b"\x00")
    c = TestClient(create_app(FakeRuntime(), Settings()))
    assert c.post("/api/post/compress", json={"video_path": str(clip)}).json()["path"] is None


def test_batch_cancel_calls_request_cancel():
    rt = FakeRuntime()
    c = TestClient(create_app(rt, Settings()))
    assert c.post("/api/batch/cancel").status_code == 200
    assert rt.cancel_requests == 1


def test_timeline_render_validates_and_posts():
    rt = FakeRuntime()
    c = TestClient(create_app(rt, Settings()))
    body = {"timeline": {"duration_s": 1.0, "fps": 30.0, "tracks": [
                {"kind": "variant_set", "set_name": "Carpaint", "prim_path": "/W/Looks",
                 "clips": [{"value": "Noir", "start_s": 0.0, "duration_s": 1.0}]}]},
            "quality": {"mode": "RealTimePathTracing", "samples_per_pixel": 64,
                        "max_bounces": 4, "resolution": [1280, 720]},
            "out_dir": "out/tl"}
    r = c.post("/api/timeline/render", json=body)
    assert r.status_code == 200 and r.json()["frames"] == 30
    assert any(type(cmd).__name__ == "RunTimeline" for cmd in rt.posted)


def test_timeline_render_rejects_overlapping_clips():
    c = TestClient(create_app(FakeRuntime(), Settings()))
    body = {"timeline": {"duration_s": 5.0, "fps": 30.0, "tracks": [
                {"kind": "variant_set", "set_name": "Carpaint", "prim_path": "/W/Looks",
                 "clips": [{"value": "A", "start_s": 0.0, "duration_s": 4.0},
                           {"value": "B", "start_s": 2.0, "duration_s": 2.0}]}]},
            "quality": {"mode": "RealTimePathTracing", "samples_per_pixel": 64,
                        "max_bounces": 4, "resolution": [1280, 720]},
            "out_dir": "out/tl"}
    assert c.post("/api/timeline/render", json=body).status_code == 422


def test_timeline_cancel_calls_request_cancel():
    rt = FakeRuntime()
    c = TestClient(create_app(rt, Settings()))
    assert c.post("/api/timeline/cancel", json={}).status_code == 200
    assert rt.cancel_requests == 1


def test_timeline_views_are_project_scoped(tmp_path, monkeypatch):
    import dev_variant_presenter.store as store
    monkeypatch.setattr(store, "_PROJ", tmp_path / "projects")
    c = TestClient(create_app(FakeRuntime(), Settings()))
    tl = {"duration_s": 2.0, "fps": 60.0, "tracks": []}
    # no project -> a view has nowhere valid to live
    assert c.post("/api/timelines/save", json={"name": "Shot A", "timeline": tl}).status_code == 400
    store.save_project("Demo", "x.usd", [], {}, timelines={})
    assert c.post("/api/timelines/save",
                  json={"name": "Shot A", "timeline": tl, "project": "Demo"}).json()["ok"]
    assert any(v["name"] == "Shot A"
               for v in c.get("/api/timelines", params={"project": "Demo"}).json()["views"])
    # a different project never sees it; no project -> empty
    assert c.get("/api/timelines", params={"project": "Other"}).json()["views"] == []
    assert c.get("/api/timelines").json()["views"] == []
    assert c.get("/api/timelines/load",
                 params={"name": "Shot A", "project": "Demo"}).json()["timeline"] == tl
    assert c.post("/api/timelines/delete", json={"name": "Shot A", "project": "Demo"}).json()["ok"] is True
    assert c.get("/api/timelines/load",
                 params={"name": "Shot A", "project": "Demo"}).status_code == 404


def test_results_and_frame_serving(tmp_path):
    (tmp_path / "Carpaint-Noir.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # not a valid image, just a file
    rt = FakeRuntime()
    c = TestClient(create_app(rt, Settings()))
    r = c.get("/api/results", params={"dir": str(tmp_path)})
    assert r.status_code == 200
    perms = r.json()["permutations"]
    assert len(perms) == 1 and perms[0]["name"] == "Carpaint-Noir"
    # frame serving: existing png ok, missing 404
    assert c.get("/api/frame", params={"path": str(tmp_path / "Carpaint-Noir.png")}).status_code == 200
    assert c.get("/api/frame", params={"path": str(tmp_path / "nope.png")}).status_code == 404


def test_camera_snap_and_render_mode(monkeypatch):
    import dev_variant_presenter.api.routes as R
    monkeypatch.setattr(R, "scan_stage", lambda p, e=(): _info(p))
    monkeypatch.setattr(R, "read_camera", lambda p, c, e=(): (tuple([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]), 0.0))
    rt = FakeRuntime()
    c = TestClient(create_app(rt, Settings()))
    c.post("/api/open", json={"usd_path": "x.usd"})
    assert c.post("/api/camera/snap", json={"camera_path": "/W/Cam"}).status_code == 200
    assert any(type(cmd).__name__ == "SnapToCamera" for cmd in rt.posted)
    r = c.post("/api/render-mode", json={"quality": {
        "mode": "PathTracing", "samples_per_pixel": 128, "max_bounces": 4, "resolution": [1280, 720]}})
    assert r.status_code == 200
    q_cmds = [cmd for cmd in rt.posted if type(cmd).__name__ == "SetQuality"]
    assert q_cmds and q_cmds[-1].quality.mode == "PathTracing"


def test_stage_endpoint_reports_whats_open(monkeypatch):
    """A reloaded tab attaches to the running session via GET /api/stage."""
    import dev_variant_presenter.api.routes as R
    monkeypatch.setattr(R, "scan_stage", lambda p, e=(): _info(p))
    monkeypatch.setattr(R, "read_camera", lambda p, c, e=(): (tuple(float(i) for i in range(16)), 5.0))
    c = TestClient(create_app(FakeRuntime(), Settings()))
    assert c.get("/api/stage").json() == {"open": False}          # nothing open yet
    c.post("/api/open", json={"usd_path": "x.usd"})
    st = c.get("/api/stage").json()
    assert st["open"] is True and st["usd_path"] == "x.usd"
    assert st["camera"] == "/W/Cam"                               # tracked from the open
    assert st["selection"] == [{"prim_path": "/W/Looks", "set_name": "Carpaint", "variant": "Noir"}]
    # a variant change keeps the mirror current
    c.post("/api/variant", json={"selections": [
        {"prim_path": "/W/Looks", "set_name": "Carpaint", "variant": "Sakura"}]})
    assert c.get("/api/stage").json()["selection"][0]["variant"] == "Sakura"


def test_pick_and_project_routes_guard_stageless_server():
    """A stale tab's gizmo timers hammering a freshly relaunched server must get 400s —
    stepping a stageless renderer poisons the native sensor scheduler (process death)."""
    rt = FakeRuntime()
    c = TestClient(create_app(rt, Settings()))
    assert c.post("/api/pick-focus", json={"nx": 0.5, "ny": 0.5}).status_code == 400
    assert c.post("/api/pick-point", json={"nx": 0.5, "ny": 0.5}).status_code == 400
    assert c.post("/api/project", json={"points": [[0, 0, 0]]}).status_code == 400
    assert c.post("/api/probe-occlusion",
                  json={"point": [0, 0, 0], "nx": 0.5, "ny": 0.5}).status_code == 400
    assert rt.posted == []        # nothing reached the render thread
