# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI control plane: REST endpoints + /events WebSocket + static frontend.

Handlers validate and enqueue commands to the RenderRuntime; they never touch ovrtx.
"""
from __future__ import annotations

import asyncio
import queue as _queue
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dev_variant_presenter.batch.engine import ExplosionError, count_permutations, guard_count
from dev_variant_presenter.batch.jobs import BatchJob, MatrixMode
from dev_variant_presenter.config import Settings
from dev_variant_presenter.models import QualitySpec, StageInfo, VariantChoice
from dev_variant_presenter import mirror, session, store, turntable
from dev_variant_presenter.api.folder_picker import pick_folder
from dev_variant_presenter.post import processing as post
from dev_variant_presenter.render.runtime import (
    CancelJob, CaptureFraming, GetCameraState, MoveCamera, OpenStage, PickFocus,
    PickPoint, ProbeOcclusion, RestartStream, RunBatch, RunTimeline, SetCamera,
    SetCameraState, SetPlayback, SetQuality, SetResolution, SetSelection, SnapToCamera,
)
from dev_variant_presenter.scan.variants import read_camera, scan_stage
from dev_variant_presenter.sequence.timeline import Timeline, TimelineError, frame_times, validate

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


# ----- request models -----
class ChoiceModel(BaseModel):
    prim_path: str
    set_name: str
    variant: str

    def to_choice(self) -> VariantChoice:
        return VariantChoice(self.prim_path, self.set_name, self.variant)


class OpenBody(BaseModel):
    usd_path: str
    camera_path: str | None = None
    selections: list[ChoiceModel] | None = None
    looks: dict = {}      # per-camera optics to seed at open (project restore: one reopen)
    xforms: dict = {}     # per-camera framing overrides to seed at open


class VariantBody(BaseModel):
    selections: list[ChoiceModel]


class SnapBody(BaseModel):
    camera_path: str
    reset: bool = False        # drop this camera's transform override -> back to the authored framing
    at_s: float | None = None  # seconds into the camera's timeline clip — scrubbing an ANIMATED
                               # camera poses its rig at that stage time (turntable rotates)


class QualityModel(BaseModel):
    mode: str = "RealTimePathTracing"
    samples_per_pixel: int = 64
    max_bounces: int = 4
    resolution: tuple[int, int] = (1280, 720)

    def to_spec(self) -> QualitySpec:
        return QualitySpec(self.mode, self.samples_per_pixel, self.max_bounces, tuple(self.resolution))


class RenderModeBody(BaseModel):
    quality: QualityModel = QualityModel()


class BatchJobModel(BaseModel):
    mode: str
    base_selection: list[ChoiceModel] = []
    included: dict[str, list[str]] = {}
    cameras: list[str] = []
    quality: QualityModel = QualityModel()
    frame_mode: Literal["single", "animation_range"] = "single"
    out_dir: str
    curated: list[list[ChoiceModel]] = []
    confirm: bool = False
    frame_start: int | None = None
    frame_end: int | None = None
    frame_step: int = 1

    def to_job(self) -> BatchJob:
        return BatchJob(
            mode=MatrixMode(self.mode),
            base_selection=tuple(c.to_choice() for c in self.base_selection),
            included={k: tuple(v) for k, v in self.included.items()},
            cameras=list(self.cameras), quality=self.quality.to_spec(),
            frame_mode=self.frame_mode, out_dir=self.out_dir,
            curated=tuple(tuple(c.to_choice() for c in row) for row in self.curated),
            frame_start=self.frame_start, frame_end=self.frame_end,
            frame_step=max(1, self.frame_step))


class BatchRequest(BaseModel):
    job: BatchJobModel


class OverlayReq(BaseModel):
    out_dir: str


class VideoReq(BaseModel):
    out_dir: str
    fps: int = 24


class CompressReq(BaseModel):
    video_path: str


class TimelineRenderBody(BaseModel):
    timeline: dict
    quality: QualityModel = QualityModel()
    out_dir: str


class DisplayBody(BaseModel):
    resolution: tuple[int, int] | None = None
    focal_length: float | None = None
    f_stop: float | None = None
    focus_distance: float | None = None
    exposure: float | None = None
    iso: float | None = None


class PickBody(BaseModel):
    nx: float
    ny: float


class ProjectBody(BaseModel):
    points: list[list[float]]


class LookAtBody(BaseModel):
    target: list[float]
    radius: float
    height: float = 0.0


class OcclusionBody(BaseModel):
    point: list[float]
    nx: float
    ny: float


class PlaybackBody(BaseModel):
    playing: bool
    fps: float = 0.0          # 0 -> stage fps


class TurntableBody(BaseModel):
    pivot: list[float]
    radius: float
    height: float = 0.0
    frames: int = 120
    fps: float = 0.0          # 0 -> stage fps
    focal_length: float = 35.0
    start_deg: float = 0.0    # spin start azimuth (frame 0 = the view the user composed)
    camera_world: list[float] | None = None   # 16 floats: author this EXACT pose as frame 0
                                              # (pan offsets preserved); None -> look-at-pivot


class SaveViewBody(BaseModel):
    name: str
    timeline: dict
    project: str = ""         # the project this view belongs to (views are project-scoped)


class NameBody(BaseModel):
    name: str
    project: str = ""         # for view delete: which project the view lives in


class ProjectSaveBody(BaseModel):
    name: str
    base_selection: list[ChoiceModel] = []
    display: dict = {}
    timeline: dict | None = None   # the editor strip's working timeline (workspace, not library)
    camera: str = ""               # the selected camera at save time


class CameraStateBody(BaseModel):
    looks: dict = {}          # {camera_path: {focal_length/f_stop/focus_distance/iso}}
    xforms: dict = {}         # {camera_path: {m: [16 floats], dist: float}}


def _stage_info_dict(info: StageInfo) -> dict:
    return {
        "usd_path": info.usd_path,
        "default_prim": info.default_prim,
        "up_axis": info.up_axis,
        "start_time": info.start_time,
        "end_time": info.end_time,
        "fps": info.fps,
        "variant_sets": [
            {"set_name": vs.set_name, "prim_path": vs.prim_path,
             "variants": list(vs.variants), "current": vs.current}
            for vs in info.variant_sets
        ],
        "cameras": [{"path": c.path, "name": c.name, "animated": c.animated}
                    for c in info.cameras],
        "meters_per_unit": info.meters_per_unit,
    }


def create_app(runtime, settings: Settings, events: "_queue.Queue | None" = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        loop = asyncio.get_running_loop()

        async def broadcaster():
            while True:
                evt = await loop.run_in_executor(None, _app.state.events.get)
                if evt.get("type") == "ready":          # first frame of the open stage exists
                    _app.state.stage_ready = True
                elif evt.get("type") in ("stage_open", "resolution"):
                    _app.state.stage_ready = False      # frameless until the reopen's first frame
                for ws in list(_app.state.ws_clients):
                    try:
                        await ws.send_json(evt)
                    except Exception:
                        _app.state.ws_clients.discard(ws)
        task = asyncio.create_task(broadcaster())
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(title="Dev Variant Presenter", lifespan=lifespan)
    app.state.events = events or _queue.Queue()
    app.state.ws_clients = set()
    app.state.stage_info = None
    app.state.user_usd = None
    app.state.stage_ready = False   # server has produced a frame of the open stage
    app.state.live_selection = []   # mirror of the last selection sent to the renderer
    app.state.active_camera = ""    # last authored camera selected (open or snap)

    @app.post("/api/open")
    def open_stage(body: OpenBody):
        usd_path = body.usd_path
        app.state.source_url = body.usd_path if mirror.is_url(body.usd_path) else ""
        if mirror.is_url(usd_path):
            # remote stage: mirror it (download-once, cached) and open the local copy —
            # pxr has no http resolver, and the composite never writes the source anyway
            def prog(count, name):
                app.state.events.put({"type": "mirror_progress", "downloaded": count, "file": name})
            usd_path = mirror.ensure_local(usd_path, "data", progress=prog)
        import os as _os
        edit = turntable.edit_layer_path(usd_path)
        extras = (edit,) if _os.path.isfile(edit) else ()
        info = scan_stage(usd_path, extras)
        app.state.stage_info = info
        app.state.user_usd = usd_path
        app.state.edit_layer = extras[0] if extras else ""
        # crash recovery: a server restart wipes the in-memory session (committed looks +
        # framings, viewer pose, stream resolution). When the SAME stage reopens without
        # explicit state — the client's auto-reopen after a watchdog relaunch, or a manual
        # re-open — merge the disk checkpoint back in. Explicit body state (a project
        # load) always wins; the checkpoint is recovery, not a second project store.
        ck = session.load() or {}
        restore = bool(ck) and ck.get("usd") == usd_path
        looks = body.looks or (ck.get("looks") or {} if restore else {})
        xforms = body.xforms or (ck.get("xforms") or {} if restore else {})
        ck_cam = (ck.get("camera") or "") if restore else ""
        if ck_cam and not any(c.path == ck_cam for c in info.cameras):
            ck_cam = ""
        cam = body.camera_path or ck_cam or (info.cameras[0].path if info.cameras else "")
        xf, focus = read_camera(usd_path, cam, extras) if cam else (tuple([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]), 0.0)
        if body.selections is not None:
            sel = tuple(c.to_choice() for c in body.selections)
        elif restore and ck.get("selection"):
            sel = tuple(VariantChoice(c["prim_path"], c["set_name"], c["variant"])
                        for c in ck["selection"]
                        if isinstance(c, dict) and {"prim_path", "set_name", "variant"} <= c.keys())
        else:  # default to the stage's current selections
            sel = tuple(VariantChoice(vs.prim_path, vs.set_name, vs.current) for vs in info.variant_sets)
        app.state.live_selection = [
            {"prim_path": c.prim_path, "set_name": c.set_name, "variant": c.variant} for c in sel]
        app.state.active_camera = cam
        app.state.stage_ready = False   # warming until the runtime emits ready
        resp = _stage_info_dict(info)
        resp["source_url"] = app.state.source_url   # the user-facing identity of a mirrored stage
        resp["camera"] = cam                        # so a bare reopen reflects a restored session
        resp["selection"] = app.state.live_selection
        resp["turntable"] = turntable.rig_info(usd_path)   # rehydrate the pivot UI
        runtime.post(OpenStage(user_usd=usd_path, selection=sel, camera_path=cam,
                               camera_xform=xf, focus_distance=focus, stage_info=info,
                               up_axis=info.up_axis, looks=looks, xforms=xforms,
                               edit_layer=app.state.edit_layer))
        if restore:
            res = ck.get("resolution")
            cur = getattr(runtime, "_settings", settings).stream_resolution
            if res and tuple(res) != tuple(cur):   # streamer rebuild — only when it differs
                runtime.post(SetResolution(width=int(res[0]), height=int(res[1])))
            pose = ck.get("pose") or {}
            if pose.get("m") and len(pose["m"]) == 16:   # exact viewer pose, not just framing
                runtime.post(MoveCamera(matrix=tuple(float(v) for v in pose["m"]),
                                        focus_distance=float(pose.get("dist") or 0.0)))
        return resp

    @app.get("/api/stage")
    def current_stage():
        # what's open RIGHT NOW — lets a reloaded tab attach to the running session
        if app.state.stage_info is None:
            return {"open": False}
        return {"open": True, "usd_path": app.state.user_usd,
                "source_url": getattr(app.state, "source_url", ""),
                "info": _stage_info_dict(app.state.stage_info),
                "selection": app.state.live_selection,
                "camera": app.state.active_camera,
                "ready": app.state.stage_ready,
                # current stream size — the client syncs its resolution select + aspect
                # to this on attach (after a server restart they'd otherwise disagree)
                "display": {"resolution": list(getattr(runtime, "_settings", settings).stream_resolution)},
                # authored turntable rig (if any) — the client rehydrates its pivot
                # gizmo + tools from this after a reload/project load
                "turntable": turntable.rig_info(app.state.user_usd) if app.state.user_usd else None,
                # racy reads of render-thread state — fine for dropdown badges
                "look_cams": [k for k, v in dict(getattr(runtime, "_looks", {})).items() if v],
                "framed_cams": [k for k, v in dict(getattr(runtime, "_xforms", {})).items() if v]}

    @app.post("/api/variant")
    def set_variant(body: VariantBody):
        app.state.live_selection = [
            (c.model_dump if hasattr(c, "model_dump") else c.dict)() for c in body.selections]
        runtime.post(SetSelection(tuple(c.to_choice() for c in body.selections)))
        return {"ok": True}

    @app.post("/api/camera/save-framing")
    def save_framing():
        runtime.post(CaptureFraming())
        return {"ok": True}

    @app.post("/api/camera/snap")
    def camera_snap(body: SnapBody):
        usd = app.state.user_usd
        if not usd:
            return JSONResponse({"error": "no stage open"}, status_code=400)
        extras = (app.state.edit_layer,) if getattr(app.state, "edit_layer", "") else ()
        xf, focus = read_camera(usd, body.camera_path, extras)
        app.state.active_camera = body.camera_path
        runtime.post(SnapToCamera(camera_xform=xf, focus_distance=focus,
                                  camera_path=body.camera_path, reset=body.reset,
                                  at_s=body.at_s))
        return {"ok": True}

    @app.post("/api/render-mode")
    def render_mode(body: RenderModeBody):
        runtime.post(SetQuality(body.quality.to_spec()))
        return {"ok": True}

    @app.post("/api/pick-focus")
    def pick_focus(body: PickBody):
        if app.state.stage_info is None:
            raise HTTPException(400, "no stage open")
        reply: "_queue.Queue" = _queue.Queue()
        runtime.post(PickFocus(nx=body.nx, ny=body.ny, reply=reply))
        try:
            return reply.get(timeout=15)
        except Exception:  # noqa: BLE001
            return {"error": "pick timed out"}

    @app.post("/api/pick-point")
    def pick_point(body: PickBody):
        if app.state.stage_info is None:
            raise HTTPException(400, "no stage open")
        reply: "_queue.Queue" = _queue.Queue()
        runtime.post(PickPoint(nx=body.nx, ny=body.ny, reply=reply))
        try:
            return reply.get(timeout=15)
        except Exception:  # noqa: BLE001
            return {"ok": False, "error": "pick timed out"}

    @app.get("/api/camera-pose")
    def camera_pose():
        # the LIVE viewport camera's world pose (racy read — fine for one-shot UI derivations)
        m = runtime._camera.to_xform()
        return {"eye": [float(m[3][0]), float(m[3][1]), float(m[3][2])],
                "right": [float(m[0][0]), float(m[0][1]), float(m[0][2])],
                "up": [float(m[1][0]), float(m[1][1]), float(m[1][2])],
                "m": [float(v) for v in m.reshape(-1)]}   # full pose: exact turntable framing

    @app.post("/api/project")
    def project(body: ProjectBody):
        # gizmo overlay: racy read of live camera state is fine at overlay rates
        if app.state.stage_info is None:
            raise HTTPException(400, "no stage open")
        return {"screen": runtime.project_points(body.points)}

    @app.post("/api/camera/look-at")
    def camera_look_at(body: LookAtBody):
        # snap the FREE viewport camera onto the suggested turntable orbit around `target`,
        # keeping the current azimuth (no 180-degree whip) — this IS frame 0 of the
        # camera-to-be, so Add is WYSIWYG from this moment
        import math as _math
        info = app.state.stage_info
        up = (info.up_axis if info else "Y") or "Y"
        m = runtime._camera.to_xform()
        off = [float(m[3][i]) - body.target[i] for i in range(3)]
        theta = _math.atan2(off[1], off[0]) if up.upper() == "Z" else _math.atan2(off[0], off[2])
        pose = turntable.orbit_matrix(body.target, body.radius, body.height, theta, up)
        d = float(body.radius ** 2 + body.height ** 2) ** 0.5
        runtime.post(MoveCamera(matrix=tuple(float(v) for v in pose.reshape(-1)),
                                focus_distance=d))
        return {"ok": True, "start_deg": _math.degrees(theta),
                "radius": body.radius, "height": body.height}

    @app.post("/api/probe-occlusion")
    def probe_occlusion(body: OcclusionBody):
        if app.state.stage_info is None:
            raise HTTPException(400, "no stage open")
        reply: "_queue.Queue" = _queue.Queue()
        runtime.post(ProbeOcclusion(point=tuple(body.point), nx=body.nx, ny=body.ny, reply=reply))
        try:
            return reply.get(timeout=10)
        except Exception:  # noqa: BLE001
            return {"ok": False, "error": "probe timed out"}

    @app.post("/api/playback")
    def playback(body: PlaybackBody):
        runtime.post(SetPlayback(playing=body.playing, fps=body.fps))
        return {"ok": True}

    @app.post("/api/turntable")
    def add_turntable(body: TurntableBody):
        usd = app.state.user_usd
        info = app.state.stage_info
        if not usd or info is None:
            raise HTTPException(400, "no stage open")
        layer = turntable.author_turntable(
            usd, pivot=body.pivot, radius=body.radius, height=body.height,
            frames=body.frames, fps=body.fps or info.fps or 30.0,
            focal_length=body.focal_length, up_axis=info.up_axis,
            start_deg=body.start_deg, camera_world=body.camera_world)
        # re-open WITH the new edits layer, seeded with the current camera state so
        # dialed looks/framing survive (same pattern as project open)
        reply: "_queue.Queue" = _queue.Queue()
        runtime.post(GetCameraState(reply=reply))
        try:
            state = reply.get(timeout=5)
        except Exception:  # noqa: BLE001
            state = {}
        new_info = scan_stage(usd, (layer,))
        app.state.stage_info = new_info
        app.state.edit_layer = layer
        app.state.stage_ready = False
        cam = turntable.CAMERA_PATH
        xf, focus = read_camera(usd, cam, (layer,))
        sel = tuple(VariantChoice(c["prim_path"], c["set_name"], c["variant"])
                    for c in app.state.live_selection)
        app.state.active_camera = cam
        runtime.post(OpenStage(user_usd=usd, selection=sel, camera_path=cam,
                               camera_xform=xf, focus_distance=focus, stage_info=new_info,
                               up_axis=new_info.up_axis, looks=state.get("looks"),
                               xforms=state.get("xforms"), edit_layer=layer))
        resp = _stage_info_dict(new_info)
        resp["source_url"] = getattr(app.state, "source_url", "")
        resp["camera"] = cam
        return resp

    @app.post("/api/turntable/remove")
    def remove_turntable_ep():
        usd = app.state.user_usd
        info = app.state.stage_info
        if not usd or info is None:
            raise HTTPException(400, "no stage open")
        turntable.remove_turntable(usd)
        reply: "_queue.Queue" = _queue.Queue()
        runtime.post(GetCameraState(reply=reply))
        try:
            state = reply.get(timeout=5)
        except Exception:  # noqa: BLE001
            state = {}
        new_info = scan_stage(usd)                      # no extras: the rig is gone
        app.state.stage_info = new_info
        app.state.edit_layer = ""
        app.state.stage_ready = False
        cams = [c.path for c in new_info.cameras]
        cam = app.state.active_camera if app.state.active_camera in cams else (cams[0] if cams else "")
        xf, focus = read_camera(usd, cam) if cam else (tuple([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]), 0.0)
        sel = tuple(VariantChoice(c["prim_path"], c["set_name"], c["variant"])
                    for c in app.state.live_selection)
        app.state.active_camera = cam
        runtime.post(OpenStage(user_usd=usd, selection=sel, camera_path=cam,
                               camera_xform=xf, focus_distance=focus, stage_info=new_info,
                               up_axis=new_info.up_axis, looks=state.get("looks"),
                               xforms=state.get("xforms"), edit_layer=""))
        resp = _stage_info_dict(new_info)
        resp["source_url"] = getattr(app.state, "source_url", "")
        resp["camera"] = cam
        return resp

    @app.post("/api/display")
    def set_display(body: DisplayBody):
        # only act on fields the client actually sent (camera params None = clear, so we must
        # not touch unspecified ones)
        data = (body.model_dump if hasattr(body, "model_dump") else body.dict)(exclude_unset=True)
        if data.get("resolution"):
            w, h = data["resolution"]
            runtime.post(SetResolution(width=int(w), height=int(h)))
        cam = {k: data[k] for k in ("focal_length", "f_stop", "focus_distance", "exposure", "iso") if k in data}
        if cam:
            runtime.post(SetCamera(params=cam))
        return {"ok": True}

    @app.get("/api/config")
    def config():
        # The browser knows the control port (its own URL) but must be told the
        # WebRTC signaling port, which may have been auto-shifted off the default.
        return {"signal_port": settings.signal_port,
                "stream_resolution": list(settings.stream_resolution)}

    @app.post("/api/batch")
    def post_batch(req: BatchRequest):
        info = app.state.stage_info
        if info is None:
            raise HTTPException(400, "No stage open")
        job = req.job.to_job()
        count = count_permutations(job, info.variant_sets)
        try:
            guard_count(count, confirm=req.job.confirm)
        except ExplosionError as e:
            raise HTTPException(409, str(e))   # client re-sends with confirm=true
        reply: "_queue.Queue" = _queue.Queue()
        runtime.post(RunBatch(job=job, reply=reply))
        return {"count": count}

    @app.post("/api/batch/cancel")
    def post_batch_cancel():
        # Direct flag set, not a queued command: the render thread is inside run_batch
        # and isn't draining the queue, so a queued CancelJob would never be seen.
        runtime.request_cancel()
        return {"ok": True}

    @app.get("/api/results")
    def get_results(dir: str = Query(...)):
        return {"permutations": post.list_results(dir)}

    @app.get("/api/frame")
    def get_frame(path: str = Query(...)):
        p = Path(path)
        if p.suffix.lower() != ".png" or not p.is_file():
            raise HTTPException(404, "Frame not found")
        return FileResponse(str(p), media_type="image/png")

    # ----- post-processing (pure PIL/OpenCV/ffmpeg; off the render thread, in the executor) -----
    @app.post("/api/browse-folder")
    async def browse_folder():
        # the SERVER is a local desktop process — it can open a real folder dialog and
        # return the absolute path (a browser page can't, by security design). All Tk
        # work is pinned to one dedicated thread (see folder_picker.py) so a second Browse
        # click can't trip the cross-thread Tcl_AsyncDelete crash that drops the viewport.
        loop = asyncio.get_running_loop()
        try:
            return {"path": await loop.run_in_executor(None, pick_folder)}
        except RuntimeError as exc:
            # No Tk / no display (headless host, or Linux without python3-tk). Degrade to
            # the same shape the frontend already handles — empty path == "cancelled" —
            # so Browse is a no-op and the user types the path, instead of a 500.
            return {"path": "", "reason": str(exc)}

    @app.post("/api/post/cutsheet")
    async def post_cutsheet(req: OverlayReq):
        loop = asyncio.get_running_loop()
        path = await loop.run_in_executor(None, post.make_cut_sheet, req.out_dir)
        return {"path": path}

    @app.post("/api/post/overlay")
    async def post_overlay(req: OverlayReq):
        loop = asyncio.get_running_loop()
        return {"count": await loop.run_in_executor(None, post.overlay_all, req.out_dir)}

    @app.post("/api/post/video")
    async def post_video(req: VideoReq):
        loop = asyncio.get_running_loop()
        return {"count": await loop.run_in_executor(
            None, lambda: post.convert_all_to_videos(req.out_dir, req.fps))}

    @app.post("/api/post/compress")
    async def post_compress(req: CompressReq):
        loop = asyncio.get_running_loop()
        return {"path": await loop.run_in_executor(None, post.compress_video, req.video_path)}

    @app.get("/api/video")
    def get_video(path: str = Query(...)):
        p = Path(path)
        if p.suffix.lower() != ".mp4" or not p.is_file():
            raise HTTPException(400, "not an MP4 file")
        return FileResponse(str(p), media_type="video/mp4")

    # ----- timeline (NLE) -----
    # Live SCRUB intentionally has NO endpoint: the frontend computes state_at client-side and
    # posts the resulting selection to /api/variant (+ /api/camera/snap on camera change).
    @app.post("/api/timeline/render")
    def timeline_render(body: TimelineRenderBody):
        try:
            tl = Timeline.from_dict(body.timeline)
            validate(tl)
        except TimelineError as e:
            raise HTTPException(422, str(e))
        frames = len(frame_times(tl))
        reply: "_queue.Queue" = _queue.Queue()
        runtime.post(RunTimeline(timeline=body.timeline, quality=body.quality.to_spec(),
                                 out_dir=body.out_dir, reply=reply))
        return {"frames": frames}

    @app.post("/api/timeline/cancel")
    def timeline_cancel():
        runtime.request_cancel()
        return {"ok": True}

    # ----- track views: a view lives INSIDE its project (clips reference the stage's
    # variant sets + cameras and lean on the project's per-camera overrides, so a view is
    # only valid for its own project — never a global pool). All ops are project-scoped. -----
    @app.get("/api/timelines")
    def list_timeline_views(project: str = Query("")):
        return {"views": [{"name": n} for n in store.project_view_names(project)] if project else []}

    @app.post("/api/timelines/save")
    def save_timeline_view(body: SaveViewBody):
        if not body.project:
            raise HTTPException(400, "open or save a project first — track views live in a project")
        if not store.save_project_view(body.project, body.name, body.timeline):
            raise HTTPException(404, f"project not found: {body.project}")
        return {"ok": True, "name": body.name}

    @app.get("/api/timelines/load")
    def load_timeline_view(name: str = Query(...), project: str = Query("")):
        rec = store.load_project_view(project, name) if project else None
        if rec is None:
            raise HTTPException(404, "saved view not found in this project")
        return rec

    @app.post("/api/timelines/delete")
    def delete_timeline_view(body: NameBody):
        return {"ok": store.delete_project_view(body.project, body.name) if body.project else False}

    # ----- projects: bundle USD + base look + display + all its timelines -----
    @app.get("/api/projects")
    def list_projects_ep():
        return {"projects": store.list_projects()}

    @app.post("/api/projects/save")
    def save_project_ep(body: ProjectSaveBody):
        # a mirrored stage's identity is its URL (re-opens resolve via the marker in ~1s);
        # the local junction path is plumbing and shouldn't leak into project files
        usd = getattr(app.state, "source_url", "") or getattr(app.state, "user_usd", "") or ""
        base = [(c.model_dump if hasattr(c, "model_dump") else c.dict)() for c in body.base_selection]
        reply: "_queue.Queue" = _queue.Queue()       # per-camera looks + framing live server-side
        runtime.post(GetCameraState(reply=reply))
        try:
            state = reply.get(timeout=5)
        except Exception:  # noqa: BLE001
            state = {}
        # timelines=None PRESERVES the project's existing track-view library (managed via
        # /api/timelines/save) — a workspace re-save must not wipe it
        store.save_project(body.name, usd, base, body.display, timelines=None,
                           looks=state.get("looks", {}), xforms=state.get("xforms", {}),
                           timeline=body.timeline, camera=body.camera)
        return {"ok": True, "name": body.name}

    @app.post("/api/restart")
    def restart_server():
        # LAST-RESORT escalation: native stream state can wedge beyond a streamer rebuild
        # (observed: 5 rebuilds without recovery; a process restart always clears it). The
        # run_server.ps1 watchdog relaunches us; the client auto-reopens its last stage.
        import os as _os
        import threading as _threading
        _threading.Timer(0.5, lambda: _os._exit(43)).start()
        return {"ok": True, "restarting": True}

    @app.post("/api/stream/restart")
    def stream_restart():
        # ghost-session eviction + dry-pipe escalation. BLOCKS until the render thread has
        # actually rebuilt the streamer — during a stage warmup a single step can take many
        # seconds, and a client that handshakes before the rebuild lands just gets torn
        # down by it (observed as a reconnect loop for the whole warmup).
        reply: "_queue.Queue" = _queue.Queue()
        runtime.post(RestartStream(reply=reply))
        try:
            reply.get(timeout=30)
            return {"ok": True, "rebuilt": True}
        except Exception:  # noqa: BLE001 - render thread busy past the timeout
            return {"ok": True, "rebuilt": False}

    @app.post("/api/camera-state")
    def set_camera_state_ep(body: CameraStateBody):
        runtime.post(SetCameraState(looks=body.looks or {}, xforms=body.xforms or {}))
        return {"ok": True}

    @app.get("/api/projects/load")
    def load_project_ep(name: str = Query(...)):
        rec = store.load_project(name)
        if rec is None:
            raise HTTPException(404, "project not found")
        return rec   # the project's track views ride along in rec["timelines"] (project-scoped)

    @app.post("/api/projects/delete")
    def delete_project_ep(body: NameBody):
        return {"ok": store.delete_project(body.name)}

    @app.websocket("/events")
    async def events_ws(ws: WebSocket):
        await ws.accept()
        app.state.ws_clients.add(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            app.state.ws_clients.discard(ws)

    if (WEB_DIR / "index.html").is_file():
        # Serve index.html with no-cache: the browser would otherwise hold on to a stale
        # copy of the page and keep loading the script/stylesheet versions it names.
        @app.get("/")
        def _index():
            return FileResponse(str(WEB_DIR / "index.html"),
                                headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    else:
        @app.get("/")
        def _root():
            return {"status": "ok", "note": "frontend missing (web/index.html)"}

    return app
