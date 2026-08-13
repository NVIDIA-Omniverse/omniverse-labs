# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turntable camera authoring: a sidecar edits layer holds /Turntable/Camera orbiting a
pivot 360°, composed ABOVE the user stage so it scans like authored content."""
import math

import numpy as np

from dev_variant_presenter import turntable
from dev_variant_presenter.scan.variants import read_camera, scan_stage

USER_USDA = """#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
)
def Xform "World" {
    def Mesh "Asset" {}
}
"""


def _user(tmp_path):
    p = tmp_path / "asset.usda"
    p.write_text(USER_USDA, encoding="utf-8")
    return str(p)


def test_orbit_matrix_looks_at_pivot_and_closes():
    pivot = [10.0, 5.0, -2.0]
    m0 = turntable.orbit_matrix(pivot, radius=100.0, height=20.0, theta=0.0)
    eye = m0[3, :3]
    fwd = -m0[2, :3]
    to_pivot = np.array(pivot) - eye
    to_pivot /= np.linalg.norm(to_pivot)
    assert np.allclose(fwd, to_pivot, atol=1e-9)               # looks AT the pivot
    assert np.isclose(np.linalg.norm(eye - np.array(pivot) - np.array([0, 20, 0])), 100.0)
    m_full = turntable.orbit_matrix(pivot, 100.0, 20.0, 2 * math.pi)
    assert np.allclose(m0, m_full, atol=1e-9)                  # full revolution closes


def test_author_and_scan_composed(tmp_path):
    user = _user(tmp_path)
    layer = turntable.author_turntable(
        user, pivot=[0, 50, 0], radius=300.0, height=25.0, frames=48, fps=24.0,
        data_root=str(tmp_path / "data"))
    info = scan_stage(user, (layer,))
    assert any(c.path == turntable.CAMERA_PATH for c in info.cameras)   # camera visible
    assert info.start_time == 0 and info.end_time == 47                 # range from edits layer
    assert info.fps == 24.0
    assert info.default_prim == "World"                                 # user metadata preserved
    xf, _ = read_camera(user, turntable.CAMERA_PATH, (layer,))
    assert len(xf) == 16                                                # pose resolves


def test_rig_spins_camera_around_pivot_always_facing_it(tmp_path):
    """The PIVOT xform's animated up-axis rotation carries the child camera around;
    the camera's static local offset keeps it facing the pivot the whole revolution."""
    from pxr import Usd, UsdGeom
    user = _user(tmp_path)
    pivot = [100.0, 50.0, -20.0]
    layer = turntable.author_turntable(
        user, pivot=pivot, radius=200.0, height=30.0, frames=120,
        data_root=str(tmp_path / "data"))
    stage = Usd.Stage.Open(layer)
    cam = UsdGeom.Xformable(stage.GetPrimAtPath(turntable.CAMERA_PATH))
    p = np.array(pivot)
    eyes = []
    for t in (0.0, 30.0, 60.0, 90.0):                       # quarter revolutions
        m = np.array(cam.ComputeLocalToWorldTransform(t))
        eye = m[3, :3]
        eyes.append(eye)
        offset = eye - p
        # constant orbit: radius in the spin plane + height along up stay fixed
        assert np.isclose(math.hypot(offset[0], offset[2]), 200.0, atol=1e-4)
        assert np.isclose(offset[1], 30.0, atol=1e-4)
        # always looking at the pivot (forward = -row2 points from eye to pivot)
        fwd = -m[2, :3]
        to_pivot = (p - eye) / np.linalg.norm(p - eye)
        assert np.allclose(fwd / np.linalg.norm(fwd), to_pivot, atol=1e-4)
    # quarter turns land on distinct positions; t=60 is opposite t=0 through the pivot
    assert np.allclose(eyes[0] + eyes[2], 2 * np.array([p[0], eyes[0][1], p[2]]), atol=1e-3)
    # the rig is the editable structure: an animated pivot + a static child offset
    spin = stage.GetPrimAtPath(turntable.PIVOT_PATH).GetAttribute("xformOp:rotateY:spin")
    assert spin and spin.GetNumTimeSamples() >= 2


def test_reauthor_replaces_previous(tmp_path):
    user = _user(tmp_path)
    turntable.author_turntable(user, pivot=[0, 0, 0], radius=100, frames=10,
                               data_root=str(tmp_path / "data"))
    layer = turntable.author_turntable(user, pivot=[0, 0, 0], radius=100, frames=24,
                                       data_root=str(tmp_path / "data"))
    info = scan_stage(user, (layer,))
    assert info.end_time == 23                                          # not the stale 10-frame orbit


def test_reauthor_succeeds_while_layer_is_held_in_registry(tmp_path):
    """pxr's layer registry keeps the edits layer ALIVE while composed stages reference
    it — re-authoring must reuse-and-clear, not CreateNew (which throws 'already exists')."""
    from pxr import Sdf
    user = _user(tmp_path)
    layer_path = turntable.author_turntable(user, pivot=[0, 0, 0], radius=100, frames=10,
                                            data_root=str(tmp_path / "data"))
    held = Sdf.Layer.FindOrOpen(layer_path)            # simulate a composed scan holding it
    assert held is not None
    layer_path2 = turntable.author_turntable(user, pivot=[0, 0, 0], radius=100, frames=24,
                                             data_root=str(tmp_path / "data"))
    info = scan_stage(user, (layer_path2,))
    assert info.end_time == 23                         # re-author replaced the 10-frame rig


def test_remove_turntable_deletes_layer_and_allows_reauthor(tmp_path):
    from pxr import Sdf
    import os
    user = _user(tmp_path)
    layer_path = turntable.author_turntable(user, pivot=[0, 0, 0], radius=100, frames=10,
                                            data_root=str(tmp_path / "data"))
    held = Sdf.Layer.FindOrOpen(layer_path)
    assert turntable.remove_turntable(user, data_root=str(tmp_path / "data")) is True
    assert not os.path.exists(layer_path)
    # and the cycle restarts cleanly
    again = turntable.author_turntable(user, pivot=[1, 2, 3], radius=50, frames=12,
                                       data_root=str(tmp_path / "data"))
    assert scan_stage(user, (again,)).end_time == 11


def test_camera_less_stage_auto_frames_the_asset(tmp_path, monkeypatch):
    """Opening a stage with NO cameras must frame the default prim's bounds — not stare
    at the world origin (the whole point of the turntable feature's target use case)."""
    from dev_variant_presenter.config import Settings
    from dev_variant_presenter.models import StageInfo
    from dev_variant_presenter.render.runtime import OpenStage, RenderRuntime

    asset = tmp_path / "asset.usda"
    asset.write_text("""#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
)
def Xform "World" {
    def Mesh "Box" {
        float3[] extent = [(-50, -50, -50), (50, 50, 50)]
        point3f[] points = [(-50, -50, -50), (50, 50, 50)]
    }
}
""", encoding="utf-8")
    rt = RenderRuntime(Settings())
    monkeypatch.setattr(rt, "_reopen", lambda: None)
    monkeypatch.setattr(rt, "_write_camera", lambda: None)
    monkeypatch.setattr(rt, "_classify_async", lambda *a, **k: None)
    info = StageInfo(str(asset), "World", "Y", 0, 0, 24, (), ())   # no cameras
    ident = tuple(float(v) for v in __import__("numpy").eye(4).reshape(-1))
    rt._do_open(OpenStage(user_usd=str(asset), selection=(), camera_path="",
                          camera_xform=ident, focus_distance=0.0, stage_info=info))
    import numpy as np
    eye = rt._camera.to_xform()[3, :3]
    assert np.linalg.norm(eye) > 50                  # moved off the origin to a framing orbit
    fwd = -rt._camera.to_xform()[2, :3]
    to_center = -eye / np.linalg.norm(eye)           # box is centered at origin
    assert float(np.dot(fwd, to_center)) > 0.99      # and it is LOOKING at the asset


def test_camera_clip_at_and_loop_time():
    from dev_variant_presenter.sequence.timeline import Timeline, camera_clip_at, loop_stage_time
    tl = Timeline.from_dict({"duration_s": 10.0, "fps": 30.0, "tracks": [
        {"kind": "camera", "clips": [
            {"value": "/Cam/A", "start_s": 0.0, "duration_s": 2.0},
            {"value": "/Turntable/Camera", "start_s": 4.0, "duration_s": 4.0}]}]})
    assert camera_clip_at(tl, 1.0) == ("/Cam/A", 0.0)
    assert camera_clip_at(tl, 3.0) == ("/Cam/A", 0.0)            # held through the gap
    assert camera_clip_at(tl, 5.5) == ("/Turntable/Camera", 4.0)  # clip-relative origin
    assert camera_clip_at(tl, 9.9) == ("/Turntable/Camera", 4.0)  # held after it ends
    # 1.5s into a 30fps stage with range 0..119 -> frame 45; loops past the end
    assert loop_stage_time(1.5, 30.0, 0, 119) == 45.0
    assert loop_stage_time(5.0, 30.0, 0, 119) == 30.0            # 150 % 120
    assert loop_stage_time(2.0, 30.0, 0, 0) == 0                 # no range -> default


def test_camera_is_animated_walks_ancestors(tmp_path):
    """The turntable rig animates the PARENT pivot, not the camera prim — detection
    must walk up the hierarchy."""
    from dev_variant_presenter.config import Settings
    from dev_variant_presenter.render.runtime import RenderRuntime
    user = _user(tmp_path)
    layer = turntable.author_turntable(user, pivot=[0, 0, 0], radius=100, frames=24,
                                       data_root=str(tmp_path / "data"))
    rt = RenderRuntime(Settings())
    rt._user_usd = layer                       # stage containing the rig
    assert rt._camera_is_animated(turntable.CAMERA_PATH) is True
    rt2 = RenderRuntime(Settings())
    rt2._user_usd = user                       # static stage, no rig
    assert rt2._camera_is_animated("/World/Asset") is False


def test_composite_root_inherits_content_timebase(tmp_path):
    """A 60tps rig sublayered under a bare 24tps root gets an implicit layer-offset
    SCALE (24/60): the spin finishes by frame 48 and HOLDS for the rest of the range
    — exactly the 'preview spin freezes' bug. The composite root must hoist the
    content's time metadata so root time == content time."""
    from pxr import Sdf, Usd, UsdGeom
    from dev_variant_presenter.models import QualitySpec
    from dev_variant_presenter.render.composer import build_composite
    user = _user(tmp_path)
    layer = turntable.author_turntable(
        user, pivot=[0, 0, 0], radius=300.0, frames=120, fps=60.0,
        data_root=str(tmp_path / "data"))
    out = build_composite(
        user, (), camera_path=turntable.CAMERA_PATH,
        render_product_path="/Render/Product", quality=QualitySpec(),
        out_path=str(tmp_path / "composite.usda"),
        extra_sublayers=(layer,), use_stage_camera=True)
    root = Sdf.Layer.FindOrOpen(out)
    assert root.timeCodesPerSecond == 60.0
    assert (root.startTimeCode, root.endTimeCode) == (0.0, 119.0)
    stage = Usd.Stage.Open(out)
    spin = stage.GetPrimAtPath(turntable.PIVOT_PATH).GetAttribute("xformOp:rotateY:spin")
    assert spin.GetTimeSamples() == [0.0, 120.0]      # NOT [0.0, 48.0] (scaled)
    # and the rig actually orbits in root time: positions differ across the revolution
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(turntable.CAMERA_PATH))
    p30 = xf.ComputeLocalToWorldTransform(Usd.TimeCode(30.0)).ExtractTranslation()
    p90 = xf.ComputeLocalToWorldTransform(Usd.TimeCode(90.0)).ExtractTranslation()
    assert (p30 - p90).GetLength() > 100.0


def test_exact_pose_framing_survives_the_spin(tmp_path):
    """camera_world: frame 0 reproduces the user's EXACT composed view (including pan
    offset — the camera need not look at the pivot), and the pivot point holds the same
    camera-space position for the whole revolution (the pan framing never snaps out)."""
    from pxr import Gf, Usd, UsdGeom
    user = _user(tmp_path)
    pivot = [10.0, 50.0, -20.0]
    # a deliberately panned view: orbit pose pushed sideways so the pivot is OFF-center
    m = turntable.orbit_matrix(pivot, radius=300.0, height=40.0, theta=0.7)
    m[3, :3] += m[0, :3] * 55.0          # pan: slide the eye along its right vector
    layer = turntable.author_turntable(
        user, pivot=pivot, radius=300.0, height=40.0, frames=120, fps=60.0,
        start_deg=33.0, camera_world=tuple(m.reshape(-1)),
        data_root=str(tmp_path / "data"))
    stage = Usd.Stage.Open(layer)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(turntable.CAMERA_PATH))
    w0 = xf.ComputeLocalToWorldTransform(Usd.TimeCode(0.0))
    assert np.allclose(np.array([[w0[i][j] for j in range(4)] for i in range(4)]),
                       m, atol=1e-9)     # frame 0 == the composed view, pan included
    pv = Gf.Vec3d(*pivot)
    cam0 = w0.GetInverse().Transform(pv)             # pivot in camera space at frame 0
    for t in (30.0, 60.0, 90.0):
        wt = xf.ComputeLocalToWorldTransform(Usd.TimeCode(t))
        camt = wt.GetInverse().Transform(pv)
        assert (camt - cam0).GetLength() < 1e-6      # pivot pinned to the same pixel
        # and the camera genuinely orbits (positions differ across the revolution)
    p30 = xf.ComputeLocalToWorldTransform(Usd.TimeCode(30.0)).ExtractTranslation()
    p90 = xf.ComputeLocalToWorldTransform(Usd.TimeCode(90.0)).ExtractTranslation()
    assert (p30 - p90).GetLength() > 100.0


def test_snap_with_clip_time_poses_the_rig(tmp_path):
    """Timeline scrub: snapping to an animated camera with at_s must pose the viewer at
    the rig's clip-relative stage time — the turntable rotates under the playhead
    instead of holding frame 0."""
    from pxr import Usd, UsdGeom
    from dev_variant_presenter.config import Settings
    from dev_variant_presenter.models import CameraInfo, StageInfo
    from dev_variant_presenter.render.runtime import RenderRuntime, SnapToCamera
    user = _user(tmp_path)
    layer = turntable.author_turntable(user, pivot=[0, 50, 0], radius=300.0,
                                       frames=120, fps=60.0,
                                       data_root=str(tmp_path / "data"))
    rt = RenderRuntime(Settings())
    rt._user_usd = layer                       # the live composite stand-in (holds the rig)
    rt._stage_info = StageInfo(layer, "", "Y", 0, 119, 60.0, (),
                               (CameraInfo(turntable.CAMERA_PATH, "Camera", animated=True),))
    ident = tuple(float(v) for v in np.eye(4).reshape(-1))
    # 0.5s into the clip at 60fps -> stage frame 30 -> a quarter revolution
    rt._do_snap(SnapToCamera(camera_xform=ident, focus_distance=0.0,
                             camera_path=turntable.CAMERA_PATH, at_s=0.5))
    stage = Usd.Stage.Open(layer)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(turntable.CAMERA_PATH))
    w30 = xf.ComputeLocalToWorldTransform(Usd.TimeCode(30.0))
    expected = np.array([[w30[i][j] for j in range(4)] for i in range(4)])
    assert np.allclose(rt._camera.to_xform(), expected, atol=1e-6)
    # and frame 0 of the spin differs (it genuinely rotated away from the start pose)
    w0 = xf.ComputeLocalToWorldTransform(Usd.TimeCode(0.0))
    assert not np.allclose(rt._camera.to_xform(),
                           np.array([[w0[i][j] for j in range(4)] for i in range(4)]), atol=1.0)


def test_first_camera_clip_governs_the_lead_in():
    """A camera clip that doesn't start at 0 must still own the frames before it —
    falling back to 'whatever camera is live' rendered surprise lead-ins."""
    from dev_variant_presenter.sequence.timeline import Timeline, camera_clip_at, state_at
    tl = Timeline.from_dict({"duration_s": 10.0, "fps": 30.0, "tracks": [
        {"kind": "camera", "clips": [
            {"value": "/TurntableRig/Turntable", "start_s": 3.0, "duration_s": 4.0}]}]})
    assert camera_clip_at(tl, 0.0) == ("/TurntableRig/Turntable", 3.0)
    _, cam = state_at(tl, 0.0, ())
    assert cam == "/TurntableRig/Turntable"


def test_rig_info_roundtrip(tmp_path):
    """The UI rehydrates its pivot gizmo from the authored rig after a reload."""
    user = _user(tmp_path)
    assert turntable.rig_info(user, data_root=str(tmp_path / "data")) is None
    turntable.author_turntable(user, pivot=[1.5, 99.0, -7.25], radius=300.0,
                               frames=240, fps=60.0, start_deg=141.8,
                               data_root=str(tmp_path / "data"))
    info = turntable.rig_info(user, data_root=str(tmp_path / "data"))
    assert info["pivot"] == [1.5, 99.0, -7.25]
    assert info["frames"] == 240
    assert info["fps"] == 60.0
    assert abs(info["start_deg"] - 141.8) < 1e-4
