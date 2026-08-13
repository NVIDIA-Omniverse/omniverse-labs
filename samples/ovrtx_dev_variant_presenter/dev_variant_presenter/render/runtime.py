# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RenderRuntime — the sole owner of the ovrtx.Renderer.

One continuous, never-blocking render thread: drain+coalesce commands, then step()
+ submit a frame (or submit_last when idle). All ovrtx calls happen here; the API
and ovstream callbacks only enqueue. Variant switching takes the cheapest correct
route: the effect classifier's replayable attribute writes when it can, and a full
stage reload as the universal fallback.
"""
from __future__ import annotations

import glob
import os
import queue
import tempfile
import threading
import time

os.environ.setdefault("OVRTX_SKIP_USD_CHECK", "1")

import numpy as np

from dev_variant_presenter.config import Settings
from dev_variant_presenter.models import QualitySpec, Selection, StageInfo
from dev_variant_presenter.render import composer
from dev_variant_presenter.render.camera import CameraController
from dev_variant_presenter.render.commands import (  # noqa: F401 — re-export for routes/app
    CancelJob, CaptureFraming, GetCameraState, MoveCamera, OpenStage, PickFocus,
    PickPoint, PrerenderThumbnails, ProbeOcclusion, RestartStream, RunBatch,
    RunTimeline, SetCamera, SetCameraState, SetCameraXform, SetPlayback, SetQuality,
    SetResolution, SetSelection, Shutdown, SnapToCamera,
)
from dev_variant_presenter.render.stage_session import StageSession
from dev_variant_presenter.usd_guard import USD_LOCK


class _RuntimeRenderBackend:
    """Adapts this runtime's `StageSession` + `ovrtx.Renderer` to
    `batch.engine.RenderBackend` — the batch/timeline render loop's only view of the
    render thread, so it never calls `Renderer.open_usd` / `update_from_usd_time` /
    `step` directly. Render-thread only; stateless (holds refs, not its own state)."""

    def __init__(self, session: StageSession, renderer):
        self._session = session
        self._renderer = renderer

    def open_composite(self, path: str) -> None:
        self._session.populate_usd(path)

    def set_time(self, time_code: float) -> None:
        self._session.update_from_usd_time(time_code)

    def reset(self) -> None:
        self._renderer.reset()

    def step_product(self, rp: str, dt: float):
        return self._renderer.step(render_products={rp}, delta_time=dt,
                                   ordinal=self._session.committed_ordinal)


def _value_to_tensor(v):
    """Shader-input value -> the tensor StageSession.write_attribute (ovstage) expects.
    None for unsupported types.

    float -> a plain (1,) float32 array: one element, one lane, 4 bytes.

    color3f (and any vector) -> ONE multi-lane element, declared explicitly. A plain numpy
    dtype cannot express `lanes=3`, so the layout is overridden the same way
    StageSession.write_omni_xform declares its 16-lane matrix. Passing a bare (1,3) array
    instead — which is what this did while the project was pinned to ovrtx 0.3 — is read as
    `lanes=1, bytesPerRow=12`, and ovstage rejects it against the existing 3-lane Fabric
    column: "existing attribute 'inputs:diffuse_reflection_color' has a different type".
    Every shader-input variant switch then fell back to a full reload (~1.6s), silently,
    because the fallback still renders the correct frame.
    """
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return np.array([float(v)], dtype=np.float32)
    try:
        comps = [float(x) for x in v]
    except (TypeError, ValueError):
        return None
    if not comps:
        return None
    from ovstage import DLDataType, DLDataTypeCode, make_dltensor
    # make_dltensor keeps the numpy buffer alive via the returned tensor, so the caller
    # only has to hold the tensor until the write completes.
    return make_dltensor(
        np.array(comps, dtype=np.float32),
        dtype=DLDataType(code=DLDataTypeCode.kDLFloat, bits=32, lanes=len(comps)),
        shape=[1])


class RenderRuntime:
    def __init__(self, settings: Settings, on_event=None):
        self._settings = settings
        self._emit = on_event or (lambda e: None)
        self._q: "queue.Queue" = queue.Queue()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

        self._renderer = None
        self._session: StageSession | None = None
        self._streamer = None
        self._composite_seq = 0
        self._composites: list[str] = []   # rolling window; see _composite_path
        self._user_usd: str | None = None
        self._camera_path: str = ""
        self._live_selection: Selection = ()
        self._stage_info: StageInfo | None = None
        self._quality = QualitySpec(resolution=settings.stream_resolution)
        self._looks: dict[str, dict] = {}   # per-camera display overrides, keyed by authored camera
                                            # path -> {focal_length / f_stop / focus_distance / iso}.
                                            # The "active" look is the one for self._camera_path.
        self._xforms: dict[str, dict] = {}  # per-camera transform overrides (your live framing),
                                            # keyed by camera path -> {"m": 16 floats, "dist": float}.
        self._camera = CameraController()
        self._has_stage = False
        self._ready_sent = False
        self._actions = None        # scan/effects classifier output; None until classified (reload-only)
        self._edit_layer: str = ""                 # sidecar edits layer (turntable camera)
        self._play = None                          # live playback: {fps, start, end, t0} or None
        self._live_composite: str | None = None   # last composite path ovrtx has open (for pick bbox)
        self._pick_stage = None                    # cached pxr stage of _live_composite (click-to-focus)
        self._pick_stage_path: str | None = None

        # input accumulator, mutated by the ovstream callback thread
        self._cam_in = {"yaw": 0.0, "pitch": 0.0, "dolly": 0.0, "px": 0.0, "py": 0.0,
                        "lx": None, "ly": None, "lb": False,           # left button = orbit
                        "mx": None, "my": None, "mb": False,           # middle button = pan
                        "dirty": False}

    # ----- public API (any thread) -----
    def post(self, cmd) -> None:
        self._q.put(cmd)

    def request_cancel(self) -> None:
        """Set the cancel flag directly (any thread). Needed for mid-batch cancel: while
        run_batch blocks the render thread the command queue isn't drained, so a queued
        CancelJob would never be seen. threading.Event is safe to set cross-thread."""
        self._cancel.set()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="render", daemon=True)
        self._thread.start()

    # ----- ovstream callbacks (SDK threads) -----
    def _on_input(self, event) -> None:
        import ovstream
        st = self._cam_in
        if event.type != ovstream.InputEventType.MOUSE:
            return
        mo = event.mouse
        if mo.type == ovstream.MouseEventType.BUTTON and mo.data == ovstream.MouseButton.LEFT:
            st["lb"] = (mo.button_state == ovstream.KeyState.DOWN)
            if st["lb"]:
                st["lx"], st["ly"] = mo.x, mo.y
        elif mo.type == ovstream.MouseEventType.BUTTON and mo.data == ovstream.MouseButton.MIDDLE:
            st["mb"] = (mo.button_state == ovstream.KeyState.DOWN)
            if st["mb"]:
                st["mx"], st["my"] = mo.x, mo.y
        elif mo.type == ovstream.MouseEventType.MOVE and st["lb"]:
            if st["lx"] is not None:
                st["yaw"] -= (mo.x - st["lx"]) * 0.005
                st["pitch"] += (mo.y - st["ly"]) * 0.005
                st["dirty"] = True
            st["lx"], st["ly"] = mo.x, mo.y
        elif mo.type == ovstream.MouseEventType.MOVE and st["mb"]:
            if st["mx"] is not None:
                k = max(self._camera.distance, 1.0) * 0.0015      # world units per pixel ~ zoom level
                st["px"] -= (mo.x - st["mx"]) * k                 # grab-the-scene: drag right -> scene right
                st["py"] += (mo.y - st["my"]) * k
                st["dirty"] = True
            st["mx"], st["my"] = mo.x, mo.y
        elif mo.type == ovstream.MouseEventType.WHEEL:
            st["dolly"] -= (mo.scroll_y or 0.0) * (max(self._camera.distance, 1.0) * 0.05)
            st["dirty"] = True

    def _on_connection(self, connected) -> None:
        self._emit({"type": "connection", "connected": bool(connected)})

    # ----- render thread -----
    def _run(self) -> None:
        import ovrtx
        from dev_variant_presenter.render.stream import Streamer
        self._ovrtx = ovrtx
        self._renderer = ovrtx.Renderer()
        self._session = StageSession("dev_variant_presenter")
        self._session.create_and_attach(self._renderer)
        w, h = self._settings.stream_resolution
        self._streamer = Streamer(w, h, self._settings.signal_port,
                                  on_input=self._on_input, on_connection=self._on_connection)
        self._streamer.start()
        self._emit({"type": "warmup"})
        dt = 1.0 / 60.0
        while True:
            if self._drain():
                break
            if self._has_stage and (self._streamer.is_client_connected or not self._ready_sent
                                    or self._play):
                # _play keeps stepping with no viewer: preview spin must advance stage time
                # across WebRTC reconnects (and headless probes) — stop resets the gate.
                # step while a viewer is attached OR until the first frame lands (_ready_sent):
                # a freshly opened stage MUST be stepped through composition/shader compile even
                # headless, or the next populate_usd (e.g. a batch composite) wedges the renderer.
                if self._play:
                    # live playback = a smooth camera animator: evaluate the active
                    # camera's rig at wall-clock stage time (pxr, on the live composite)
                    # and fabric-write the result onto the viewer camera — the exact
                    # mechanism free navigation uses, so it streams at full rate.
                    # update_from_usd_time + reset() is WRONG here: a reset discards the
                    # in-flight frame, and resetting per time change starves the stream
                    # to a slideshow (~6fps measured). Renders (timeline/batch) still use
                    # stage-time evaluation — they converge per frame and can afford it.
                    p = self._play
                    span = max(1.0, p["end"] - p["start"] + 1.0)
                    tc = p["start"] + ((time.monotonic() - p["t0"]) * p["fps"]) % span
                    m = self._camera_world_at(self._camera_path, tc)
                    if m is not None:
                        self._write_matrix(m)
                else:
                    self._apply_camera_input()   # free-nav input pauses while playing
                try:
                    products = self._renderer.step(
                        render_products={self._settings.render_product_path}, delta_time=dt,
                        ordinal=self._session.committed_ordinal)
                    self._submit(products)
                except Exception as e:  # noqa: BLE001
                    self._emit({"type": "error", "message": f"step: {e}"})
                    self._streamer.submit_last()
            elif self._has_stage:
                # warmed up but no WebRTC viewer: don't path-trace into the void (79% GPU /
                # 422 W idle measured). Keep the cached frame warm; full rate resumes on connect.
                self._streamer.submit_last()
                time.sleep(0.1)
            else:
                self._streamer.submit_last()
                time.sleep(0.03)
        self._session.detach_and_destroy(self._renderer)
        self._streamer.stop()

    def _drain(self) -> bool:
        cmds = []
        try:
            while True:
                cmds.append(self._q.get_nowait())
        except queue.Empty:
            pass
        latest_sel = latest_xf = None
        latest_cam = None          # merged SetCamera params (coalesce a burst -> one reopen)
        latest_res = None          # latest SetResolution
        latest_snap = None         # latest SnapToCamera (a scrub burst -> one snap+reopen)
        restart_req = False        # RestartStream bursts collapse to ONE streamer rebuild
        restart_replies = []       # every requester gets unblocked by that one rebuild
        state_reply = None         # a GetCameraState in this batch — answered AFTER deferred applies
        shutdown = False
        for cmd in cmds:
            try:   # a single bad command must NEVER kill the render thread (-> permanent "Reconnecting")
                if isinstance(cmd, Shutdown):
                    shutdown = True
                elif isinstance(cmd, SetSelection):
                    latest_sel = cmd            # coalesce — keep only the latest
                elif isinstance(cmd, SetCameraXform):
                    latest_xf = cmd
                elif isinstance(cmd, OpenStage):
                    self._do_open(cmd)
                elif isinstance(cmd, SnapToCamera):
                    latest_snap = cmd           # coalesce — a scrub burst collapses to one snap
                elif isinstance(cmd, GetCameraState):
                    state_reply = cmd.reply   # defer: must reflect this batch's deferred SetCamera
                elif isinstance(cmd, SetCameraState):
                    self._do_set_camera_state(cmd)
                elif isinstance(cmd, SetQuality):
                    self._do_quality(cmd)
                elif isinstance(cmd, SetCamera):
                    latest_cam = {**(latest_cam or {}), **cmd.params}   # merge a burst
                elif isinstance(cmd, SetResolution):
                    latest_res = cmd
                elif isinstance(cmd, PickFocus):
                    self._do_pick_focus(cmd)
                elif isinstance(cmd, PickPoint):
                    self._do_pick_point(cmd)
                elif isinstance(cmd, SetPlayback):
                    self._do_playback(cmd)
                elif isinstance(cmd, MoveCamera):
                    self._camera.snap_to(np.array(cmd.matrix, dtype=np.float64).reshape(4, 4),
                                         focus_distance=cmd.focus_distance or None)
                    self._write_camera()
                elif isinstance(cmd, CaptureFraming):
                    if self._camera_path and self._camera_is_animated(self._camera_path):
                        # an animated camera's pose is MOTION owned by its rig — renders
                        # ignore framing overrides on it, so don't pretend to save one
                        self._emit({"type": "framing_skipped", "message":
                                    "This camera is animated — its motion comes from its rig. "
                                    "Reframe the turntable with 'Update camera from this view'."})
                    else:
                        self._capture_xform()
                        self._emit({"type": "framing_saved", "camera": self._camera_path})
                elif isinstance(cmd, ProbeOcclusion):
                    self._do_probe_occlusion(cmd)
                elif isinstance(cmd, RestartStream):
                    restart_req = True
                    if cmd.reply is not None:
                        restart_replies.append(cmd.reply)
                elif isinstance(cmd, CancelJob):
                    self._cancel.set()
                elif isinstance(cmd, RunBatch):
                    self._do_batch(cmd)
                elif isinstance(cmd, RunTimeline):
                    self._run_timeline(cmd)
                else:
                    self._emit({"type": "error",
                                "message": f"command not implemented: {type(cmd).__name__}"})
            except Exception as e:  # noqa: BLE001
                self._emit({"type": "error", "message": f"{type(cmd).__name__} failed: {e}"})
        if restart_req:              # one rebuild even if escalation queued several
            try:
                self._do_restart_stream()
            except Exception as e:  # noqa: BLE001
                self._emit({"type": "error", "message": f"stream restart failed: {e}"})
            for q in restart_replies:
                q.put(True)          # rebuild done (or failed) — requesters may handshake
        if latest_res is not None:   # one stream-resize per drain
            try:
                self._do_set_resolution(latest_res)
            except Exception as e:  # noqa: BLE001
                self._emit({"type": "error", "message": f"resolution failed: {e}"})
        if latest_snap is not None:  # one snap (+ reopen if the look changes) per scrub burst
            try:
                self._do_snap(latest_snap)
            except Exception as e:  # noqa: BLE001
                self._emit({"type": "error", "message": f"camera snap failed: {e}"})
        if latest_cam is not None:   # one reopen for a whole burst of camera tweaks
            try:
                self._do_set_camera(SetCamera(params=latest_cam))
            except Exception as e:  # noqa: BLE001
                self._emit({"type": "error", "message": f"camera failed: {e}"})
        if latest_sel is not None:
            try:
                self._do_selection(latest_sel)
            except Exception as e:  # noqa: BLE001
                self._emit({"type": "error", "message": f"selection failed: {e}"})
        if latest_xf is not None:
            try:
                self._write_matrix(np.array(latest_xf.matrix, dtype=np.float64).reshape(4, 4))
            except Exception as e:  # noqa: BLE001
                self._emit({"type": "error", "message": f"camera move failed: {e}"})
        if state_reply is not None:   # answer AFTER deferred SetCamera/snap so the snapshot is current
            state_reply.put({"looks": self._snapshot_looks(), "xforms": self._snapshot_xforms()})
        return shutdown

    # ----- command handlers (render thread) -----
    def _composite_path(self) -> str:
        """Path for the next composite layer. Each composition gets its own file.

        Every reopen authors a DIFFERENT document (a different variant selection), and a
        distinct path keeps that unambiguous for anything downstream that keys on the
        composed-scene path: the renderer, and our own `_pick_stage`, which keeps a pxr
        stage open on the current composite to measure bboxes for click-to-focus.
        Re-authoring one fixed path underneath a reader is the failure mode this avoids.

        One file per reopen would otherwise accumulate for the life of the process, so we
        keep a rolling window of our own: the new composite, plus the previous one, which
        is still the open stage until `populate_usd` below replaces it. The PID is in the
        name because two instances would otherwise both claim `_live_1.usda`.

        A hard native crash (which this stack does have) skips the rolling delete, so the
        first call in a process also sweeps day-old leftovers. That age check is what
        keeps the sweep from deleting a concurrent instance's live composite.
        """
        d = os.path.join(tempfile.gettempdir(), "dev_variant_presenter")
        os.makedirs(d, exist_ok=True)
        if self._composite_seq == 0:
            cutoff = time.time() - 86400.0
            for stale in glob.glob(os.path.join(d, "_live_*.usda")):
                try:
                    if os.path.getmtime(stale) < cutoff:
                        os.remove(stale)
                except OSError:
                    pass          # best-effort: a locked or vanished leftover is harmless
        self._composite_seq += 1
        path = os.path.join(d, f"_live_{os.getpid()}_{self._composite_seq}.usda")
        self._composites.append(path)
        while len(self._composites) > 2:      # current + the one still open behind it
            try:
                os.remove(self._composites.pop(0))
            except OSError:
                pass
        return path

    def _reopen(self) -> None:
        # Pump a last-frame heartbeat while we wait on USD_LOCK + build + populate_usd, so a
        # reload-path variant switch (or quality change) doesn't stall the render loop past
        # the WebRTC ~7s liveness window and drop the client.
        self._with_usd_heartbeat(self._reopen_locked)

    def _reopen_locked(self) -> None:
        # USD_LOCK: build_composite (pxr authoring) + populate_usd (ovstage recomposition) must
        # not race the background classifier's pxr authoring — that crashes the process.
        with USD_LOCK:
            path = composer.build_composite(
                self._user_usd, self._live_selection,
                camera_path=self._camera_path,
                render_product_path=self._settings.render_product_path,
                quality=self._quality, out_path=self._composite_path(),
                viewer_camera_path=self._settings.viewer_camera_path,
                camera=self._effective_cam(),
                extra_sublayers=(self._edit_layer,) if self._edit_layer else ())
            self._session.populate_usd(path)   # already advances the write floor
            self._live_composite = path   # pick-to-focus measures bboxes against this composition
        self._has_stage = True

    def _with_usd_heartbeat(self, fn) -> None:
        """Run a blocking render-thread stage op (lock wait + build + populate_usd) while a
        daemon thread re-streams the last frame every ~2s, so the WebRTC client survives a
        reload (~7s liveness). Safe: the render thread is blocked in fn (not submitting),
        and the pump is joined before the render loop resumes — stream_video is never
        called from two threads at once."""
        stop = threading.Event()

        def pump():
            while not stop.wait(2.0):
                try:
                    self._streamer.submit_last()
                except Exception:  # noqa: BLE001
                    pass

        t = threading.Thread(target=pump, name="reopen-heartbeat", daemon=True)
        t.start()
        try:
            fn()
        finally:
            stop.set()
            t.join(timeout=3.0)

    def _do_open(self, cmd: OpenStage) -> None:
        self._user_usd = cmd.user_usd
        self._live_selection = cmd.selection
        self._camera_path = cmd.camera_path
        # seed per-camera state BEFORE the (single) reopen so a project's first frame is already
        # the saved look/framing — not the authored view that then settles override-by-override
        self._looks = {cam: dict(look) for cam, look in (cmd.looks or {}).items() if look}
        self._xforms = {cam: dict(xf) for cam, xf in (cmd.xforms or {}).items() if xf}
        self._ready_sent = False    # re-warm the new stage even with no viewer attached
        self._play = None
        self._edit_layer = cmd.edit_layer or ""
        self._stage_info = cmd.stage_info
        self._camera = CameraController(up_axis=cmd.up_axis or "Y")
        ov = self._xforms.get(self._camera_path)
        if ov:   # the camera's saved framing wins over the authored xform
            self._camera.snap_to(np.array(ov["m"], dtype=np.float64).reshape(4, 4),
                                 focus_distance=ov.get("dist") or None)
        else:
            self._camera.snap_to(np.array(cmd.camera_xform, dtype=np.float64).reshape(4, 4),
                                 focus_distance=cmd.focus_distance)
        self._reopen()
        self._write_camera()
        if not cmd.camera_path:
            self._auto_frame(cmd.stage_info)   # camera-less stage: do not stare at the origin
        self._emit({"type": "stage_open"})
        self._emit({"type": "camera_params", "params": dict(self._look())})   # sync the sliders
        # classify variants off the render thread; until ready, switching uses reload
        self._actions = None
        threading.Thread(target=self._classify_async, args=(cmd.user_usd, cmd.stage_info),
                         name="classify", daemon=True).start()

    def _classify_async(self, user_usd, stage_info) -> None:
        try:
            from dev_variant_presenter.scan import effects
            acts = effects.classify_variants(user_usd, stage_info)
            self._actions = acts  # atomic rebind; the render thread only reads it
            self._emit({"type": "classified",
                        "fast_sets": sorted(n for n, a in acts.items() if a.kind == "shader-input"),
                        "swatches": {n: a.swatches for n, a in acts.items() if a.swatches}})
        except Exception as e:  # noqa: BLE001
            self._emit({"type": "error", "message": f"variant classify failed (reload-only): {e}"})

    def _do_selection(self, cmd: SetSelection) -> None:
        new = cmd.selection
        cur = {c.set_name: c.variant for c in self._live_selection}
        changed = [c for c in new if cur.get(c.set_name) != c.variant]
        if not changed:
            return
        # fast path only for shader-input sets that are classified; anything else -> reload
        fast_writes = []
        reload_needed = self._actions is None
        if not reload_needed:
            for c in changed:
                act = self._actions.get(c.set_name)
                if act is not None and act.kind == "shader-input" and c.variant in act.per_variant:
                    fast_writes.extend(act.per_variant[c.variant])
                else:
                    reload_needed = True
                    break
        self._live_selection = new
        if reload_needed:
            self._reopen()
            self._write_camera()    # restore the user's current view across the reload
        else:
            try:
                self._apply_shader_writes(fast_writes)
                self._renderer.reset()   # clear accumulation; the live loop re-converges
            except Exception as e:  # noqa: BLE001 — fall back to the always-correct reload
                self._emit({"type": "error", "message": f"fast-path write failed, reloading: {e}"})
                self._reopen()
                self._write_camera()
                reload_needed = True
        self._emit({"type": "selection", "fast": not reload_needed,
                    "selections": [{"prim_path": c.prim_path, "set_name": c.set_name, "variant": c.variant}
                                   for c in new]})

    def _apply_shader_writes(self, writes) -> None:
        """Fast-path shader-input writes (material swaps classified shader-input) via ovstage:
        one query per distinct prim, one shared ordinal for the whole batch, one advance —
        so a multi-attribute variant switch lands atomically. UPSERT (session default) since
        these prims already exist in the live composite."""
        if not writes:
            return
        by_prim: dict[str, list] = {}
        for w in writes:
            by_prim.setdefault(w.prim, []).append(w)
        ordinal = self._session.next_ordinal()
        queries = []
        try:
            for prim, prim_writes in by_prim.items():
                query = self._session.query_from_paths([prim])
                queries.append(query)
                for w in prim_writes:
                    t = _value_to_tensor(w.value)
                    if t is None:
                        raise ValueError(f"unsupported write value for {w.attr}: {w.value!r}")
                    self._session.write_attribute(query, w.attr, t, is_array=False,
                                                  ordinal=ordinal, advance=False)
        finally:
            for q in queries:
                self._session._release_query(q)
        self._session.advance(ordinal)

    def _do_batch(self, cmd: RunBatch) -> None:
        """Run a batch synchronously on the render thread (sole ovrtx owner), then
        resume the live look. The live stream is frozen during the batch — run_batch
        calls the heartbeat (streamer.submit_last) to hold the ~7s WebRTC window."""
        self._cancel.clear()
        from dev_variant_presenter.batch import engine
        try:
            if not self._user_usd or self._stage_info is None:
                raise RuntimeError("no stage open")
            self._play = None       # renders own stage time; stop the live playback
            # Hold USD_LOCK for the whole batch: run_batch authors a composite + opens it
            # (via the backend -> ovstage) per permutation; the classifier (if still running)
            # pauses between sets until the batch releases. Fast-path classification is
            # irrelevant during a batch.
            with USD_LOCK:
                out_dir = engine.run_batch(
                    cmd.job, self._live_selection, self._stage_info,
                    backend=_RuntimeRenderBackend(self._session, self._renderer),
                    render_product_path=self._settings.render_product_path,
                    emit=self._emit, is_cancelled=self._cancel.is_set,
                    composer=composer, user_usd=self._user_usd,
                    viewer_camera_path=self._settings.viewer_camera_path,
                    heartbeat=self._streamer.submit_last,
                    camera_look=self._render_look,
                    apply_camera_override=self._apply_xform_override,
                    camera_is_animated=self._camera_is_animated,
                    camera_world_at=self._camera_world_at,
                    write_camera=self._write_matrix,
                    extra_sublayers=(self._edit_layer,) if self._edit_layer else ())
            cmd.reply.put({"ok": True, "out_dir": out_dir})
        except Exception as e:  # noqa: BLE001
            self._emit({"type": "error", "message": f"batch: {e}"})
            cmd.reply.put({"ok": False, "error": str(e)})
        finally:
            if self._user_usd:
                self._reopen()        # resume the live composite (viewer cam, stream res, mode)
                self._write_camera()  # restore the user's view across the LIVE<->BATCH switch

    def _run_timeline(self, cmd: RunTimeline) -> None:
        """Render a timeline: walk frame_times, compose state_at over the live base, render each
        frame as a still (reusing engine._converge/_save_frame), then assemble timeline.mp4 via
        post.frames_to_video. LIVE->BATCH->LIVE under USD_LOCK, heartbeated, cancellable."""
        import os
        from dev_variant_presenter.batch import engine
        from dev_variant_presenter.post import processing
        from dev_variant_presenter.sequence.timeline import (
            Timeline, camera_clip_at, frame_times, loop_stage_time, state_at,
        )
        self._cancel.clear()
        self._play = None           # renders own stage time; stop the live playback
        try:
            if not self._user_usd:
                raise RuntimeError("no stage open")
            tl = Timeline.from_dict(cmd.timeline)
            base = self._live_selection
            times = frame_times(tl)
            total = len(times)
            frames_dir = os.path.join(cmd.out_dir, "_timeline")
            os.makedirs(frames_dir, exist_ok=True)
            rp = self._settings.render_product_path
            max_steps = max(cmd.quality.samples_per_pixel, 40)
            backend = _RuntimeRenderBackend(self._session, self._renderer)
            # Timeline state only changes at clip boundaries, so consecutive (and recurring)
            # frames are pixel-identical — render each unique state ONCE and copy the PNG for
            # repeats (a 60s slideshow @30fps is ~10 renders + 1790 copies, not 1800 renders).
            # NOTE: if a scene-time animation track is ever added, time must join this key.
            import json as _json
            import shutil as _shutil
            rendered: dict[str, str] = {}   # state key -> first rendered frame's png path
            anim_cams: dict[str, bool] = {}  # camera path -> has animated xform (incl. parents)
            with USD_LOCK:   # build_composite + populate_usd must not race the classifier
                warned_missing: set[str] = set()
                for i, t in enumerate(times):
                    if self._cancel.is_set():
                        break
                    selection, camera_path = state_at(tl, t, base)
                    known = {c.path for c in self._stage_info.cameras} if self._stage_info else set()
                    if camera_path and known and camera_path not in known:
                        # a clip naming a camera that no longer exists (renamed/removed rig)
                        # must FAIL LOUDLY — silently rendering another camera burns trust
                        if camera_path not in warned_missing:
                            warned_missing.add(camera_path)
                            self._emit({"type": "error", "message":
                                        f"Timeline clip camera not found: {camera_path} — "
                                        "re-add the clip from the camera list. Rendering the "
                                        "active camera instead."})
                        camera_path = None
                    cam_path = camera_path or self._camera_path
                    look = self._render_look(cam_path)   # fills DOF focus for an animated turntable cam
                    ov = self._xforms.get(cam_path)        # your saved framing overrides the authored xform
                    # ANIMATED cameras (turntable rigs, authored camera moves) play their
                    # animation clip-relatively: stage time = clip-relative seconds * stage
                    # fps, looping over the stage range. Static shots stay at default time
                    # (keeps the frame dedup that makes variant slideshows cheap).
                    if cam_path not in anim_cams:
                        anim_cams[cam_path] = self._camera_is_animated(cam_path) if cam_path else False
                    animated = anim_cams[cam_path]
                    tc = None
                    if animated and self._stage_info is not None:
                        cc = camera_clip_at(tl, t)
                        rel = max(0.0, t - cc[1]) if cc else t   # lead-in holds frame 0
                        tc = loop_stage_time(rel, self._stage_info.fps or 24.0,
                                             self._stage_info.start_time, self._stage_info.end_time)
                    frame_png = os.path.join(frames_dir, f"{i:05d}.png")
                    key = _json.dumps([sorted((c.prim_path, c.set_name, c.variant) for c in selection),
                                       cam_path, look, ov,
                                       round(tc, 4) if tc is not None else None], sort_keys=True)
                    src = rendered.get(key)
                    if src is not None:
                        _shutil.copyfile(src, frame_png)
                    else:
                        composite = os.path.join(cmd.out_dir, f"_tl_{i:05d}.usda")
                        composer.build_composite(
                            self._user_usd, selection,
                            camera_path=cam_path,
                            render_product_path=rp, quality=cmd.quality, out_path=composite,
                            viewer_camera_path=self._settings.viewer_camera_path,
                            camera=look,
                            extra_sublayers=(self._edit_layer,) if self._edit_layer else ())
                        backend.open_composite(composite)
                        if tc is not None:
                            # scene-side time (if the stage animates geometry/lights) — via
                            # ovstage, never Renderer.update_from_usd_time
                            backend.set_time(float(tc))
                            backend.reset()             # discard prior-time accumulation
                        if animated and tc is not None:
                            # ovstage's update_from_usd_time does NOT re-evaluate time-sampled
                            # XFORMS in this ovrtx build (GPU-verified: the rig rendered at
                            # default time for every frame). Evaluate the camera in pxr and
                            # fabric-write it onto the viewer camera (via the session) — the
                            # mechanism live playback proved out. Remove this CPU pxr push if a
                            # future ovstage/ovrtx build re-evaluates time-sampled xforms itself.
                            m = self._camera_world_at(cam_path, float(tc))
                            if m is not None:
                                self._write_matrix(m)
                        elif not animated:
                            self._apply_xform_override(cam_path)   # saved framing beats the authored xform
                        engine._converge(backend, rp, max_steps=max_steps,
                                         delta_time=1.0 / 60.0, heartbeat=self._streamer.submit_last)
                        engine._save_frame(backend, rp, frame_png, delta_time=1.0 / 60.0)
                        try:
                            os.unlink(composite)
                        except OSError:
                            pass
                        rendered[key] = frame_png
                    self._emit({"type": "timeline_progress", "frame": i + 1, "total": total})
                    self._streamer.submit_last()
            out_mp4 = os.path.join(cmd.out_dir, "timeline.mp4")
            # mp4 assembly over hundreds of PNGs blocks the render thread well past the ~7s
            # WebRTC liveness window -> heartbeat through it (same pattern as reopen/converge),
            # or the viewer drops to "Reconnecting" right as the render finishes.
            self._with_usd_heartbeat(
                lambda: processing.frames_to_video(frames_dir, out_mp4,
                                                   fps=int(round(tl.fps)) or 24))
            self._emit({"type": "timeline_done", "frames": total, "video": out_mp4,
                        "cancelled": self._cancel.is_set()})
            cmd.reply.put({"ok": True, "frames": total, "video": out_mp4})
        except Exception as e:  # noqa: BLE001
            self._emit({"type": "error", "message": f"timeline: {e}"})
            cmd.reply.put({"ok": False, "error": str(e)})
        finally:
            if self._user_usd:
                self._reopen()
                self._write_camera()

    @staticmethod
    def _pick_ndc_rect(nx: float, ny: float, w: int, h: int) -> tuple[float, float, float, float]:
        """Map UI click (nx, ny) in [0, 1] top-left origin to an ovrtx 0.4 pick rectangle.

        ovrtx 0.4 ``enqueue_pick_query`` takes NDC edges (not pixels): for pixel (x, y) use
        ``[x/w, y/h, (x+1)/w, (y+1)/h]``. Passing raw pixels raises ValueError
        (``invalid NDC rectangle``) because values like 640 fall outside [0, 1].
        """
        px = max(0, min(int(nx * w), w - 1))
        py = max(0, min(int(ny * h), h - 1))
        return px / w, py / h, (px + 1) / w, (py + 1) / h

    def _do_pick_focus(self, cmd: PickFocus) -> None:
        """Click-to-focus: ray-pick the clicked pixel, resolve the hit prim, and set the focus
        distance to that prim's surface. Prefer pick `worldPositionM` when non-zero (0.4 NDC
        pick); otherwise fall back to the prim world-bbox distance (0.3-era workaround still
        needed if the hit position stays zero — peer apps on this stack don't rely on it
        either)."""
        import numpy as np
        if not self._has_stage:    # stepping a stageless renderer corrupts the sensor
            cmd.reply.put({"ok": False, "reason": "no stage open"})   # scheduler (native
            return                                                    # crash ~2min later)
        ovrtx = self._ovrtx
        rp = self._settings.render_product_path
        w, h = self._settings.stream_resolution
        left, top, right, bottom = self._pick_ndc_rect(cmd.nx, cmd.ny, w, h)
        try:
            self._renderer.enqueue_pick_query(rp, left, top, right, bottom)
            products = self._renderer.step(render_products={rp}, delta_time=1.0 / 60.0,
                                           ordinal=self._session.committed_ordinal)
            frame = products[rp].frames[0]
            rvar = frame.render_vars[ovrtx.OVRTX_RENDER_VAR_PICK_HIT]
            world_pos = None
            with rvar.map(device=ovrtx.Device.CPU) as mapped:
                hit_count = int(np.from_dlpack(mapped.params["hitCount"]).copy().reshape(-1)[0])
                prim_id = int(np.from_dlpack(mapped["primPath"]).copy().reshape(-1)[0]) if hit_count else 0
                if hit_count:
                    try:
                        if "worldPositionM" in mapped.params:
                            world_pos = np.from_dlpack(mapped.params["worldPositionM"]).copy().reshape(-1)
                        else:
                            world_pos = np.from_dlpack(mapped["worldPositionM"]).copy().reshape(-1)
                    except Exception:  # noqa: BLE001
                        world_pos = None
            prim_path = self._renderer.resolve_prim_path_id(prim_id) if prim_id else ""
            if not prim_path:
                cmd.reply.put({"ok": False, "reason": "nothing under the cursor — click on the model"})
                return
            if world_pos is not None and world_pos.size >= 3 and float(np.linalg.norm(world_pos[:3])) > 1e-8:
                eye = self._camera.to_xform()[3, :3]
                dist = float(np.linalg.norm(world_pos[:3] - eye))
            else:
                dist = self._prim_world_distance(prim_path)
            if dist is None:
                cmd.reply.put({"ok": False, "reason": "couldn't measure that surface — try another spot"})
                return
            self._active_look()["focus_distance"] = dist     # focus sets the current camera's look
            if self._has_stage:
                self._write_camera_optics()
            self._emit({"type": "camera_params", "params": dict(self._active_look())})
            self._emit({"type": "focus_picked", "distance": dist, "prim": prim_path})
            cmd.reply.put({"ok": True, "distance": dist, "prim": prim_path})
        except Exception as e:  # noqa: BLE001
            self._emit({"type": "error", "message": f"pick failed: {e}"})
            cmd.reply.put({"ok": False, "error": f"{type(e).__name__}: {e}"})

    def _camera_is_animated(self, cam_path: str) -> bool:
        """Does the camera (or ANY ancestor — turntable rigs animate the parent pivot)
        carry time-sampled xform ops? Checked on the live composite via the pick-stage."""
        from pxr import Usd, UsdGeom
        path = self._live_composite or self._user_usd
        if not path or not cam_path:
            return False
        with USD_LOCK:
            if self._pick_stage is None or self._pick_stage_path != path:
                self._pick_stage = Usd.Stage.Open(path)
                self._pick_stage_path = path
            prim = self._pick_stage.GetPrimAtPath(cam_path)
            while prim and prim.IsValid() and prim.GetPath() != "/":
                xf = UsdGeom.Xformable(prim)
                if xf:
                    for op in xf.GetOrderedXformOps():
                        if op.GetNumTimeSamples() > 1:
                            return True
                prim = prim.GetParent()
        return False

    def _camera_world_at(self, cam_path: str, tc: float):
        """The camera prim's world transform at stage time `tc`, evaluated on the live
        composite (pick-stage cache — same timebase the renderer composes). Row-vector
        4x4 numpy, or None if the prim doesn't resolve. Drives live playback."""
        from pxr import Usd, UsdGeom
        path = self._live_composite or self._user_usd
        if not path or not cam_path:
            return None
        with USD_LOCK:
            if self._pick_stage is None or self._pick_stage_path != path:
                self._pick_stage = Usd.Stage.Open(path)
                self._pick_stage_path = path
            prim = self._pick_stage.GetPrimAtPath(cam_path)
            if not prim or not prim.IsValid():
                return None
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode(float(tc)))
        return np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)

    def _prim_world_bbox(self, prim_path: str):
        """(center[3], size[3]) of a prim's world AABB on the live composite, or None."""
        from pxr import Usd, UsdGeom
        path = self._live_composite or self._user_usd
        if not path:
            return None
        with USD_LOCK:
            if self._pick_stage is None or self._pick_stage_path != path:
                self._pick_stage = Usd.Stage.Open(path)
                self._pick_stage_path = path
            prim = self._pick_stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                return None
            cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                useExtentsHint=True)
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            return None
        mn, mx = rng.GetMin(), rng.GetMax()
        center = [float((mn[i] + mx[i]) / 2.0) for i in range(3)]
        size = [float(mx[i] - mn[i]) for i in range(3)]
        return center, size

    def _prim_world_distance(self, prim_path: str) -> float | None:
        """World-space distance (scene units) from the viewer camera to a prim's bounding box —
        the closest point on its world AABB, which approximates the clicked surface. Caches a pxr
        stage of the live composite (current variant selections applied) keyed on its path, so
        repeated picks don't reopen; opens read-only, so the shared user layer is never mutated."""
        import numpy as np
        from pxr import Usd, UsdGeom
        path = self._live_composite or self._user_usd
        if not path:
            return None
        with USD_LOCK:
            if self._pick_stage is None or self._pick_stage_path != path:
                self._pick_stage = Usd.Stage.Open(path)
                self._pick_stage_path = path
            prim = self._pick_stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                return None
            cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                useExtentsHint=True)
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            return None
        cam = np.array(self._camera.to_xform()[3][:3], dtype=np.float64)
        mn, mx = rng.GetMin(), rng.GetMax()
        closest = np.array([min(max(cam[i], mn[i]), mx[i]) for i in range(3)], dtype=np.float64)
        near = float(np.linalg.norm(closest - cam))          # camera outside bbox -> near face
        mp = rng.GetMidpoint()
        mid = float(np.linalg.norm(np.array([mp[0], mp[1], mp[2]], dtype=np.float64) - cam))
        dist = near if near > 1.0 else mid                   # inside/degenerate -> bbox center
        return dist if (np.isfinite(dist) and dist > 0) else None

    def _pick_prim(self, nx: float, ny: float) -> str:
        """Ray-pick the pixel and resolve the hit prim path ('' = nothing under the cursor).
        MUST NOT run stageless: stepping a renderer with no opened USD poisons the native
        sensor scheduler ('Sensor with handle 0 not found' flood -> process death). A stale
        browser tab's gizmo timer hitting a freshly relaunched server did exactly that."""
        if not self._has_stage:
            raise RuntimeError("no stage open")
        import numpy as np
        ovrtx = self._ovrtx
        rp = self._settings.render_product_path
        w, h = self._settings.stream_resolution
        left, top, right, bottom = self._pick_ndc_rect(nx, ny, w, h)
        self._renderer.enqueue_pick_query(rp, left, top, right, bottom)
        products = self._renderer.step(render_products={rp}, delta_time=1.0 / 60.0,
                                       ordinal=self._session.committed_ordinal)
        frame = products[rp].frames[0]
        rvar = frame.render_vars[ovrtx.OVRTX_RENDER_VAR_PICK_HIT]
        with rvar.map(device=ovrtx.Device.CPU) as mapped:
            hit_count = int(np.from_dlpack(mapped.params["hitCount"]).copy().reshape(-1)[0])
            prim_id = int(np.from_dlpack(mapped["primPath"]).copy().reshape(-1)[0]) if hit_count else 0
        return self._renderer.resolve_prim_path_id(prim_id) if prim_id else ""

    def _do_probe_occlusion(self, cmd: ProbeOcclusion) -> None:
        """Is `point` behind/inside scene geometry as seen from the camera? Pick the surface
        under its projected pixel and compare camera distances. Used by the gizmo to shade
        the pivot 'submerged' while it is being dragged into an object."""
        import numpy as np
        if not self._has_stage:    # see _pick_prim: a stageless step is a native time bomb
            cmd.reply.put({"ok": True, "occluded": False})
            return
        try:
            prim = self._pick_prim(cmd.nx, cmd.ny)
            if not prim or prim.startswith("/Viewer"):
                cmd.reply.put({"ok": True, "occluded": False})
                return
            surf = self._prim_world_distance(prim)
            eye = np.array(self._camera.to_xform()[3][:3], dtype=np.float64)
            d_pivot = float(np.linalg.norm(np.array(cmd.point, dtype=np.float64) - eye))
            margin = max(1.0, d_pivot * 0.01)   # bbox-nearest underestimates big prims; bias
            occluded = surf is not None and surf < d_pivot - margin
            cmd.reply.put({"ok": True, "occluded": bool(occluded)})
        except Exception as e:  # noqa: BLE001
            cmd.reply.put({"ok": False, "error": f"{type(e).__name__}: {e}"})

    def _do_pick_point(self, cmd: PickPoint) -> None:
        """Pick a prim and reply with its world bbox center + size — the turntable pivot
        seed (the user refines with the gizmo nudges afterwards)."""
        if not self._has_stage:    # see _pick_prim: a stageless step is a native time bomb
            cmd.reply.put({"ok": False, "reason": "no stage open"})
            return
        try:
            prim_path = self._pick_prim(cmd.nx, cmd.ny)
            if not prim_path:
                cmd.reply.put({"ok": False, "reason": "nothing under the cursor"})
                return
            box = self._prim_world_bbox(prim_path)
            if box is None:
                cmd.reply.put({"ok": False, "reason": "couldn't measure that prim"})
                return
            center, size = box
            cmd.reply.put({"ok": True, "point": center, "size": size, "prim": prim_path})
        except Exception as e:  # noqa: BLE001
            cmd.reply.put({"ok": False, "error": f"{type(e).__name__}: {e}"})

    def project_points(self, pts) -> list:
        """Project world points through the CURRENT live camera -> normalized screen coords
        [[nx, ny, in_front], ...] for the gizmo overlay. Called from the API thread with a
        racy read of camera state — a frame of staleness is invisible at overlay rates."""
        import numpy as np
        m = self._camera.to_xform()
        f = float(self._look().get("focal_length") or 50.0)
        h_ap = float(getattr(composer, "LAST_APERTURE", 20.955) or 20.955)
        w, h = self._settings.stream_resolution
        v_ap = h_ap * h / w
        inv = np.linalg.inv(np.asarray(m, dtype=np.float64))
        out = []
        for pt in pts:
            pc = (np.array([pt[0], pt[1], pt[2], 1.0]) @ inv)[:3]
            z = -pc[2]
            if z <= 1e-6:
                out.append([0.5, 0.5, False])
                continue
            nx = 0.5 + (pc[0] * f / z) / h_ap
            ny = 0.5 - (pc[1] * f / z) / v_ap
            out.append([float(nx), float(ny), True])
        return out

    def _do_playback(self, cmd: SetPlayback) -> None:
        """Loop the stage's authored time range in the LIVE viewport (turntable spin
        preview). Wall-clock driven so the on-screen speed is the real animation speed
        at the stage fps. The active camera's rig is evaluated in pxr per iteration and
        written onto the viewer camera (see the _run playback branch) — no
        recomposition, no renderer reset, so starting/stopping is instant and the
        stream stays at full rate."""
        if not cmd.playing or self._stage_info is None:
            if self._play:
                self._play = None
                st = self._cam_in   # drop drag input queued during the spin: no jump on resume
                st["yaw"] = st["pitch"] = st["dolly"] = st["px"] = st["py"] = 0.0
                st["dirty"] = False
                if self._has_stage:
                    self._write_camera()      # restore the user's framing
            return
        fps = cmd.fps or (self._stage_info.fps if self._stage_info else 0) or 24.0
        self._play = {"fps": float(fps), "start": float(self._stage_info.start_time),
                      "end": float(self._stage_info.end_time), "t0": time.monotonic()}

    def _do_snap(self, cmd: SnapToCamera) -> None:
        # Switching the authored camera also switches which per-camera look + framing is active.
        # Restore this camera's transform override (your saved framing) if it has one; reset drops
        # it back to the authored camera. Reopen ONLY when the effective look changes (so an
        # un-dialed camera, or scrubbing between same-look cameras, stays a fast fabric xform write).
        old_eff = self._effective_cam() if self._has_stage else {}
        if cmd.camera_path:
            self._camera_path = cmd.camera_path
        if cmd.reset:
            self._xforms.pop(self._camera_path, None)
        ov = None if cmd.reset else self._xforms.get(self._camera_path)
        if ov:
            self._camera.snap_to(np.array(ov["m"], dtype=np.float64).reshape(4, 4),
                                 focus_distance=ov.get("dist") or None)
        else:
            self._camera.snap_to(np.array(cmd.camera_xform, dtype=np.float64).reshape(4, 4),
                                 focus_distance=cmd.focus_distance)
        if (cmd.at_s is not None and self._stage_info is not None
                and self._camera_is_animated(self._camera_path)):
            # timeline scrub inside an animated clip: pose the rig at the clip-relative
            # stage time (same pxr evaluate + write mechanism as live playback), so the
            # turntable rotates under the playhead instead of holding frame 0
            from dev_variant_presenter.sequence.timeline import loop_stage_time
            tc = loop_stage_time(float(cmd.at_s), self._stage_info.fps or 24.0,
                                 self._stage_info.start_time, self._stage_info.end_time)
            m = self._camera_world_at(self._camera_path, float(tc))
            if m is not None:
                self._camera.snap_to(m)
        if self._has_stage and self._effective_cam() != old_eff:
            self._reopen()
        self._write_camera()
        self._emit({"type": "camera_params", "params": dict(self._look())})

    def _do_quality(self, cmd: SetQuality) -> None:
        # keep the fixed stream resolution; change mode/samples/bounces via a reopen
        q = cmd.quality
        self._quality = QualitySpec(mode=q.mode, samples_per_pixel=q.samples_per_pixel,
                                    max_bounces=q.max_bounces,
                                    resolution=self._settings.stream_resolution)
        if self._has_stage:
            self._reopen()
            self._write_camera()

    def _do_set_camera(self, cmd: SetCamera) -> None:
        # display overrides (FOV / depth of field / exposure) -> authored on the viewer camera.
        # None clears; f_stop<=0 turns DOF OFF. Mutates the ACTIVE camera's look so each camera
        # keeps its own optics into the timeline.
        look = self._active_look()
        for k, v in cmd.params.items():
            if v is None or (k == "f_stop" and float(v) <= 0):
                look.pop(k, None)
            else:
                look[k] = v
        if not look:                       # fully cleared -> drop the entry (keeps _looks tidy)
            self._looks.pop(self._camera_path or "", None)
        if self._has_stage:
            if "iso" in cmd.params:
                # exposure:iso is a custom attribute (composer authors it with
                # CreateAttribute, not a UsdGeomCamera schema attr) — it has no defined
                # fallback value, so a live write can't "unset" it back to un-authored on
                # clear. focalLength/fStop/focusDistance/exposure ARE schema attrs with
                # defined fallbacks (see composer.build_composite), so those go live via
                # _write_camera_optics; iso still needs the always-correct reopen path.
                self._reopen()
                self._write_camera()
            else:
                self._write_camera_optics()
        self._emit({"type": "camera_params", "params": dict(self._look())})

    def _write_camera_optics(self) -> None:
        """Live scalar writes for the viewer camera's optics — focalLength/fStop/
        focusDistance/exposure are UsdGeomCamera schema attrs with defined fallback
        values (50 / 0 / 0 / 0, matching composer.build_composite's un-authored default),
        so a bare write reproduces what a reopen would author, without paying for one."""
        look = self._effective_look(self._look(), self._camera)
        attrs = {
            "focalLength": float(look.get("focal_length") or 50.0),
            "fStop": float(look.get("f_stop") or 0.0),
            "focusDistance": float(look.get("focus_distance") or 0.0),
            "exposure": float(look.get("exposure") or 0.0),
        }
        self._session.write_scalar_attrs(self._settings.viewer_camera_path, attrs)

    def _look(self) -> dict:
        """Read-only view of the active camera's look (never creates an entry)."""
        return self._looks.get(self._camera_path or "", {})

    def _active_look(self) -> dict:
        """The mutable look dict for the currently selected authored camera (created on demand —
        call only when about to write to it)."""
        return self._looks.setdefault(self._camera_path or "", {})

    @staticmethod
    def _effective_look(look: dict, controller=None) -> dict:
        """A look with an auto focus distance when DOF is on but none was set — focus on the orbit
        target (live camera distance) so the subject stays sharp. Timeline frames pass no
        controller (no orbit target per authored camera), so focus stays as authored/picked."""
        cam = dict(look)
        if cam.get("f_stop") and not cam.get("focus_distance"):
            dist = float(getattr(controller, "distance", 0) or 0) if controller is not None else 0.0
            if dist > 0:
                cam["focus_distance"] = dist
        return cam

    def _effective_cam(self) -> dict:
        return self._effective_look(self._look(), self._camera)

    def _render_look(self, cam_path: str) -> dict:
        """The look to render `cam_path` with. WYSIWYG: the camera you're currently viewing renders
        with the SAME effective look the live viewport shows — including the orbit controller's
        auto-focus when DOF (f_stop) is on but no focus was explicitly picked — so rendered DOF
        matches what you see. Other cameras render with their stored look (their own picked focus).
        Shared by the timeline and batch render paths."""
        controller = self._camera if cam_path == self._camera_path else None
        return self._effective_look(self._looks.get(cam_path, {}), controller)

    def _snapshot_looks(self) -> dict:
        """Deep copy of the non-empty per-camera looks (for project save)."""
        return {cam: dict(look) for cam, look in self._looks.items() if look}

    def _snapshot_xforms(self) -> dict:
        """Deep copy of the per-camera transform overrides (for project save)."""
        return {cam: dict(xf) for cam, xf in self._xforms.items() if xf}

    def _do_set_camera_state(self, cmd: SetCameraState) -> None:
        """Restore a project's per-camera looks + transform overrides, snap the active camera to
        its restored framing, then reopen so the live view reflects both."""
        self._looks = {cam: dict(look) for cam, look in (cmd.looks or {}).items() if look}
        self._xforms = {cam: dict(xf) for cam, xf in (cmd.xforms or {}).items() if xf}
        ov = self._xforms.get(self._camera_path)
        if ov:
            self._camera.snap_to(np.array(ov["m"], dtype=np.float64).reshape(4, 4),
                                 focus_distance=ov.get("dist") or None)
        if self._has_stage:
            self._reopen()
            self._write_camera()
        self._emit({"type": "camera_params", "params": dict(self._look())})

    def _do_restart_stream(self) -> None:
        # Half-open WebRTC / ghost sessions hold the single client slot; recreate ONLY the
        # ovstream.Server to drop them. NOT a full Streamer rebuild — that cycles the
        # process-global ovstream.initialize()/shutdown(), which degrades the NVENC encoder
        # and streams black (the every-reload-goes-black regression). See stream.py.
        self._streamer.rebuild_server()
        self._emit({"type": "stream_restarted"})

    def _do_set_resolution(self, cmd: SetResolution) -> None:
        # "real" stream resize: recreate the ovstream Server at the new buffer size (the
        # server is fixed-size at boot), then reopen the live composite to match. Uses
        # rebuild_server, NOT a full Streamer/global-context cycle (that wedges NVENC black).
        # The browser's WebRTC drops and reconnects (same signaling port).
        from dataclasses import replace
        # 4:2:0 H.264 cannot encode odd dimensions — an odd-height stream produces ZERO
        # frames forever (looks like a dead stream, immune to every reconnect/rebuild)
        w, h = int(cmd.width) & ~1, int(cmd.height) & ~1
        if (w, h) == tuple(self._settings.stream_resolution):
            return   # same size — a streamer rebuild would only drop the client for nothing
        self._settings = replace(self._settings, stream_resolution=(w, h))
        self._quality = QualitySpec(mode=self._quality.mode,
                                    samples_per_pixel=self._quality.samples_per_pixel,
                                    max_bounces=self._quality.max_bounces, resolution=(w, h))
        self._streamer.rebuild_server(w, h)
        self._ready_sent = False
        if self._has_stage:
            self._reopen()
            self._write_camera()
        self._emit({"type": "resolution", "width": w, "height": h})

    # ----- camera + framing -----
    def _apply_camera_input(self) -> None:
        st = self._cam_in
        if not st["dirty"]:
            return
        if st["yaw"] or st["pitch"]:
            self._camera.orbit(st["yaw"], st["pitch"])
        if st["dolly"]:
            self._camera.dolly(st["dolly"])
        if st["px"] or st["py"]:
            self._camera.pan(st["px"], st["py"])
        st["yaw"] = st["pitch"] = st["dolly"] = st["px"] = st["py"] = 0.0
        st["dirty"] = False
        self._write_camera()
        # NOTE: navigation does NOT auto-capture framing — free orbiting around a shot
        # camera must never rewrite it. Committing is explicit (CaptureFraming).

    def _capture_xform(self) -> None:
        """Record the active camera's current framing so it restores on switch, renders in the
        timeline, and persists in the project. 'dist' (orbit pivot distance) lets snap_to rebuild
        the controller exactly on restore; 'm' is the row-vector world xform ovrtx consumes."""
        if not self._camera_path:
            return
        self._xforms[self._camera_path] = {
            "m": [float(x) for x in self._camera.to_xform().reshape(-1)],
            "dist": float(self._camera.distance)}

    def _auto_frame(self, info) -> None:
        """A stage with NO authored cameras opens with the free camera framing the default
        prim's world bounds (a turntable-ready three-quarter view), not a view from origin."""
        try:
            from dev_variant_presenter.turntable import orbit_matrix
            root = "/" + info.default_prim if info and info.default_prim else "/"
            box = self._prim_world_bbox(root)
            if not box:
                return
            center, size = box
            ext = max(max(size), 1e-3)
            up = (info.up_axis if info else "Y") or "Y"
            m = orbit_matrix(center, ext * 2.0, ext * 0.5, 0.6, up)
            self._camera.snap_to(m, focus_distance=((ext * 2.0) ** 2 + (ext * 0.5) ** 2) ** 0.5)
            self._write_camera()
        except Exception:  # noqa: BLE001 - framing is best-effort, never block an open
            pass

    def _apply_xform_override(self, cam_path: str) -> None:
        """Write a camera's saved framing override (if any) over the freshly opened composite —
        shared by the live view, the timeline render, and the grid batch."""
        ov = self._xforms.get(cam_path)
        if ov:
            self._write_matrix(np.array(ov["m"], dtype=np.float64).reshape(4, 4))

    def _write_camera(self) -> None:
        self._write_matrix(self._camera.to_xform())

    def _write_matrix(self, m: np.ndarray) -> None:
        if not self._has_stage:
            return
        self._session.write_omni_xform(self._settings.viewer_camera_path, m)

    def _submit(self, products) -> bool:
        """Push the step's completed frame (if any) to the stream; True if one landed."""
        ovrtx = self._ovrtx
        import warp as wp
        for _name, product in products.items():
            for frame in product.frames:
                if "LdrColor" not in frame.render_vars:
                    continue
                with frame.render_vars["LdrColor"].map(device=ovrtx.Device.CUDA) as m:
                    self._streamer.submit(wp.from_dlpack(m))
                if not self._ready_sent:
                    self._ready_sent = True
                    self._emit({"type": "ready"})
                return True
        return False
