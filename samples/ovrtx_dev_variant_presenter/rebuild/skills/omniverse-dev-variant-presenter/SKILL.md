---
name: omniverse-dev-variant-presenter
description: >
  Build a browser-based, RTX-streamed app for exploring and rendering USD VARIANT
  PERMUTATIONS — a live ovstream/WebRTC viewport plus instant variant switching,
  cameras, a turntable rig, grid/batch stills, a multi-track timeline, projects, post,
  and remote-stage mirroring. Use when asked to build a "variant presenter / configurator"
  — or a "variant studio", the name used in the walkthrough video and earlier releases —
  a permutation render tool, or to add variant/timeline/batch features on top of an
  Omniverse realtime viewer.
license: Apache-2.0
metadata:
  author: NVIDIA Customer Success
  tags: [omniverse, usd, ovrtx, ovstream, variants, configurator, viewer]
  domain: ai-ml
  languages: [python, javascript]
---

# Dev Variant Presenter

This is the **orchestrator** for building a USD variant-permutation presenter. It composes
the NVIDIA **`omniverse-realtime-viewer`** seed skill (which owns the render/stream/
camera/picking base and the architectural non-negotiables) with the Variant Presenter
**domain layer** taught by the sibling skills below.

> **Read the seed skill FIRST.** Everything here assumes the seed's non-negotiables:
> server-side `ovrtx` only; the browser shows an `ovstream` WebRTC video + UI (NO
> client-side 3D — no three.js/babylon/model-viewer/r3f); user USD is never modified
> (viewer camera, render product, render vars, settings, selections live in
> session/composite/sidecar layers); ONE owner of `renderer.step()` + stage mutation.
> If you cannot find the seed skill, its key recipes are summarized in
> `references/seed-recap.md`.

## What you are building (definition of done)

A single Python process that streams a live path-traced USD viewport to a browser and
lets the user: open a stage (local path OR remote `http(s)://` URL), switch variants
live, snap/dial cameras, build a turntable rig, batch-render permutation stills, author
and render a multi-track timeline to MP4, save/reopen projects, and label/contact-sheet
results. Validate against the NVIDIA **ConceptCar** sample (13 variant sets, 18 cameras).

The 9 capabilities map to the sibling skills:

| Capability | Skill |
|---|---|
| App architecture + stability + control plane | **this skill** |
| Scan variant sets/cameras; classify fast-path vs reload | `usd-variant-scan-classify` |
| Live variant switching (attr-replay vs reopen) | `usd-variant-live-switching` |
| Grid / batch permutation stills | `ovrtx-grid-batch-render` |
| Multi-track timeline (NLE) → MP4 | `ovrtx-timeline-nle` |
| Turntable rig (orbit camera) | `ovrtx-turntable-camera` |
| Projects + crash-recovery session | `ovrtx-projects-session` |
| Remote `http(s)` stage mirroring | `usd-remote-stage-mirror` |
| Results browse + label + cut sheet | `ovrtx-post-cutsheet` |
| Frontend look + layout (theme, panels, chips, timeline strip) | `omniverse-dev-variant-presenter-ui` |

Build in this order: **stream a live stage → variant switching → cameras → batch →
timeline → turntable → projects/post/remote.** Get a frame on screen before anything
else; every later feature is a mutation of the live look.

**"Done" is production CADENCE, not the happy path.** After the core works, you MUST also build
the depth + robustness + UX in `references/production-parity-checklist.md` — the full control-plane
surface (folder-browse dialog, video playback, gizmo/occlusion/look-at/turntable-remove,
project-restore, track-view CRUD, stream/process restart), stream SELF-HEALING (half-open watchdog,
reconnect loop, ghost eviction, warming ticker), the full turntable (3D gizmo + nudge +
frame-from-current-view + remove + rehydrate), full project round-trip (open restores
look/optics/framings/timeline), in-page video review, real aspect ratios, camera badges +
per-camera optics restore, timeline editing depth (no-overlap snap, hide tracks, render-cancel,
result surfacing), the in-app `data-help` help layer, and Backdrops/environment actually
re-rendering. Read that checklist and implement ALL of it — a happy-path-only build is NOT done.

## Architecture (single process)

```
Browser (ovstream WebRTC video + control panels)
  │  WebRTC video + mouse input         │  REST + WebSocket (/events)
  ▼                                     ▼
ovstream Server (signaling :49100)   FastAPI / uvicorn (control port)
                                        │  validate + ENQUEUE commands
                                        ▼
                  RENDER THREAD  (owns ovstage.Stage + attached ovrtx.Renderer + ovstream)
                    • one continuous loop that ALWAYS submits a frame
                    • drains a coalesced command queue between frames
                    • populate_usd / live writes / step(ordinal=…)
                    • variant switch · camera · render-mode · batch · timeline · pick
                                        │ author composites (pxr) → ovstage.population
                                        ▼
              StageSession (ovstage)  ·  variant+camera scan  ·  batch/timeline/post
```

- **The render thread is the only thread that touches `ovstage` / `ovrtx`.** HTTP handlers never
  call the stage or renderer directly — they construct a command object and `runtime.post(cmd)`
  it. Commands that need a result carry a reply `queue.Queue`; the handler blocks on
  `reply.get(timeout=...)`.
- **`ovstage` is required.** The app owns an `ovstage.Stage` (via `StageSession`), attaches the
  renderer with `Renderer.attach_ovstage`, populates with `ovstage.population.open_usd`, and
  drives live writes through `ovstage.PathDictionary(stage)` / `Stage.write_attribute`. Do not
  use the deprecated `Renderer.open_usd` / `Renderer.write_attribute` / `Renderer.get_path_dictionary`
  path.
- **The loop never starves the stream.** Each iteration: drain+coalesce the queue, apply
  the net mutation, `step()` once, submit the frame. Long work (reopen, batch, timeline)
  must submit heartbeat frames (see "stability" below) so WebRTC never goes >~7 s without
  a frame.
- **Layering (never mutate user USD):** build a **composite** `.usda` that `subLayers` the
  user stage and adds `/Viewer/Camera`, a `RenderProduct`, ordered `LdrColor` RenderVar,
  `resolution`, and render settings. Apply variant **selections** in a session layer (or
  re-author the composite). The **turntable rig** lives in a separate sidecar layer
  sublayered above the user stage. The user's file is opened read-only.

## pxr + ovrtx live in ONE process (do NOT split into subprocesses)

`ovrtx` and `pxr` (usd-core) **coexist in a single interpreter** — keep the app
single-process as the architecture above shows. The trap that makes people wrongly split
them: constructing the `ovrtx.Renderer()` (or importing `ovrtx`) **before** pxr has loaded
its USD plugins causes a `TfType::AddAlias 'ParticleField'` duplicate-registration clash, and
then `pxr` stage ops die. Avoid it with **import order**:

- Import `pxr` at MODULE level in your composer/scanner/mirror modules (so usd-core's USD
  plugins register first, at app import time).
- Import `ovrtx` **lazily, on the render thread**, inside the function that builds the
  renderer — NOT at module top, and after the pxr modules are already imported.

```python
# composer.py / scan.py / mirror.py  — module level, loads at app import (pxr first)
from pxr import Usd, UsdGeom, Sdf, Gf
...
# runtime.py — ovrtx/ovstage imported lazily on the render thread, AFTER pxr is loaded
def _render_thread(self):
    import ovrtx                      # first ovrtx touch happens here, post-pxr
    self._renderer = ovrtx.Renderer()
    self._session = StageSession("dev_variant_presenter")   # owns the app's ovstage.Stage
    self._session.create_and_attach(self._renderer)  # ovstage.Stage(...) + renderer.attach_ovstage(stage)
```

With this order they share one process cleanly (verified with usd-core 26.x + ovrtx 0.4.x + ovstage 0.1.x).
**Do not** move pxr work into a worker subprocess — that adds an IPC protocol, and it is easy
to leave one pxr call (e.g. mirror dependency-walking, bbox for picking) behind in the ovrtx
process, which then clashes. If — and only if — you still hit an unavoidable plugin clash on
your exact toolchain, isolate **every** pxr call behind one subprocess (scan, classify,
composite authoring, mirror, bbox, camera-pose eval) with nothing pxr left in the ovrtx
process; but the single-process import-order path above is the supported design.

## Control plane (REST + WS) — expose exactly this surface

A thin FastAPI app. Handlers validate + enqueue; they do not render. (This is the contract
a UI and an automated checker both rely on — keep the names/shapes.)

```
GET  /api/config            -> {signal_port:int, stream_resolution:[w,h]}
POST /api/open              {usd_path} -> {variant_sets:[{set_name,prim_path,variants[],current}],
                                           cameras:[{path,name,animated}], up_axis, fps,
                                           start_time, end_time, source_url}
GET  /api/stage             -> {open:bool, ready:bool, selection[], camera}
POST /api/variant           {selections:[{prim_path,set_name,variant}]} -> {ok:true}
POST /api/camera/snap       {camera_path, reset?, at_s?} -> {ok:true}
POST /api/camera/save-framing  -> {ok:true}        # commit current view as this camera's framing
POST /api/render-mode       {quality:{mode,samples_per_pixel,max_bounces,resolution:[w,h]}} -> {ok:true}
POST /api/display           {focal_length?,f_stop?,focus_distance?,iso?,resolution?} -> {ok:true}
POST /api/pick-focus        {nx,ny} -> {focus_distance, ...}   # focus picker (returns a DISTANCE)
POST /api/pick-point        {nx,ny} -> {world:[x,y,z], prim_path}  # 3D pick for the TURNTABLE PIVOT (returns a POINT) — distinct from pick-focus; the pivot needs a world position, not a distance
POST /api/batch             {job:{mode,base_selection,included,cameras,quality,frame_mode,
                                  out_dir,confirm,frame_start?,frame_end?,frame_step?}} -> {count}  (409 over guard)
POST /api/batch/cancel      -> {ok:true}
GET  /api/results?dir=      -> {permutations:[...]}
GET  /api/frame?path=       -> image/png
POST /api/turntable         {pivot,radius,height,frames,fps,focal_length,start_deg,camera_world?} -> stage info
POST /api/playback          {playing,fps} -> {ok:true}   # live preview spin
POST /api/timeline/render   {timeline,quality,out_dir} -> {frames}
POST /api/timeline/cancel   -> {ok:true}
POST /api/post/overlay      {out_dir} -> {count}
POST /api/post/cutsheet     {out_dir} -> {path}
POST /api/post/video        {out_dir,fps} -> {count}
GET  /api/video?path=       -> video/mp4
GET  /api/projects          -> {projects:[...]}
POST /api/projects/save     {name,base_selection,display,camera,timeline?} -> {ok:true}
GET  /api/projects/load?name= -> project record
POST /api/projects/delete   {name} -> {ok}
WS   /events                -> JSON events: {type:"ready"}, {type:"stage_open"},
                               {type:"batch_progress",done,total,...}, {type:"batch_done",...},
                               {type:"mirror_progress",downloaded,file}, {type:"classified",...}
GET  /                      -> the viewer HTML (serves <video id="remote-video"> + ovstream client)
```

- **Auto-pick free ports** for both the control port and the signaling port (scan from a
  preferred value, then OS-ephemeral) so the app coexists with other local servers. The
  browser learns the control port from its own URL and the signaling port via
  `GET /api/config`. Print a launch banner: `http://127.0.0.1:<port>`.
- **`/api/stage` is "what's open right now"** so a reloaded tab re-attaches to the running
  session instead of re-opening.

- **The `/events` WebSocket needs a WS library installed.** FastAPI/uvicorn serve HTTP fine
  without one, but the `/events` WS handshake then FAILS (`No supported WebSocket library
  detected` at launch; `WebSocket connection ... failed` in the browser) — and since the live
  `ready`/`classified`/progress/`mirror_progress` events ride that socket, the status badge stays
  stuck on "warming", swatch dots never populate, and progress bars never move, even though the
  video streams fine. **Install `uvicorn[standard]` (or `websockets` / `wsproto`).** Don't ship a
  bare `uvicorn`.

## Testing (ship a runnable suite that covers ALL functionality)

The app MUST ship its own comprehensive, **runnable** test suite the user can execute to verify
every capability — this is a success criterion, not optional. Provide a single command (e.g. a
`run_tests` script or a documented sequence) and cover all three layers:

1. **Pure logic (no GPU):** variant classification (fast/reload split + swatch extraction), matrix
   counting + the explosion guard, timeline `state_at`/`frame_times`/presets, `{set}-{variant}`
   naming + label parsing, mirror cache-path sanitization. Plus the JS timeline mirror (node test).
2. **API contract (no GPU, FastAPI TestClient):** every `/api/*` endpoint returns its documented
   shape; the explosion guard 409s; `/api/projects` round-trips; `/api/open` of a ported
   `http://host:port/...` URL mirrors; **the `/events` WebSocket accepts and delivers events**
   (this is the layer that catches the missing-WS-library class of bug).
3. **Live / GPU smoke:** open the ConceptCar, assert a **well-lit** non-black frame, switch a
   fast-path variant (pixels move), snap a camera; **run a 1–2 permutation batch and then ASSERT
   the `{label}.png` file(s) physically exist on disk in `out_dir`** — author a turntable +
   confirm the camera appears, render a short timeline to MP4 **and assert the `.mp4` exists**.

> **The batch/timeline checks MUST assert files on disk, not a returned `{count}`/`{frames}`.**
> The render runs on the render thread; it can **silently write NOTHING while still returning a
> correct `count`** (a `_render_permutation` that no-ops, or `cv2.imwrite` returning `False`
> without raising — `imwrite` does NOT throw on failure). `count` is the *plan*; the PNG files are
> the *proof*. Checking only `count==N` lets a batch that produces zero stills — and a Results/post
> tab with nothing to show — pass as green. After the
> batch, also run `/api/post/overlay`+`/api/post/cutsheet` on those stills and assert the cut-sheet
> file exists (post is dead weight if batch wrote nothing). Make these HARD asserts in the suite —
> give them the same teeth as the well-lit-frame pixel assert.

Document which layers need the GPU. The README must show how to run the suite. The suite passing
is part of "done".

## STABILITY NON-NEGOTIABLES (each one cost a real debugging session — bake them in)

Read `references/stability-checklist.md` for the full why/how. The short list:

1. **Pin the WebRTC media candidate to loopback.** Set `ovstream.ServerConfig.webrtc_public_ip = "127.0.0.1"`. Without it, ovstream auto-detects an interface and on a multi-NIC/VPN box advertises an unreachable ICE candidate → signaling connects but **media never pairs → black viewport**, non-deterministically across launches. (Remote access needs the real LAN IP instead.)
2. **The browser `<video>` must be `muted` and you must call `.play()` on stream attach.** Chrome blocks autoplay of unmuted media without a gesture → the element sits paused on frame 0 → black, even though frames are arriving.
3. **Pin the host to IPv4.** Bind/advertise `127.0.0.1`, not `localhost` (Windows resolves `localhost`→IPv6 `::1` first for WebSockets; an IPv4-only server then refuses the WS). In the client, normalize `localhost`/`::1` → `127.0.0.1` for both the `/events` WS and the ovstream `server` host.
4. **Never construct two `ovrtx`/`ovstream` instances on one machine.** carb/ovstream has process-global state; a second `Renderer()`+`Server()` access-violates ~1.5 s into init AND wedges the first server's signaling. One renderer, one process. (Init ovstream once per process; on client evict, rebuild only the `Server`, not the global context.)
5. **Never `step()` a stageless renderer.** Stepping before any `ovstage.population.open_usd` (via your `StageSession.populate_usd`) floods `Invalid RenderProduct` / sensor errors and kills the process in minutes. Guard EVERY step/pick path on a `has_stage` flag; reject stageless pick/probe HTTP routes with 400.
6. **Serialize ALL pxr stage work behind one process-global lock** (`USD_LOCK = threading.RLock()`). pxr's Sdf change manager is process-global and NOT thread-safe; the background classifier authoring `SetVariantSelection` while the render thread composes/populates via `ovstage.population.open_usd` → access violation. Hold the lock around every open/compose/author (scan, classify per-set, reopen, batch, mirror). Use an **RLock** so render-thread nesting doesn't self-deadlock. Do NOT lock steady-state `renderer.step()` / `ovstage.Stage.write_attribute` (ovstage/ovrtx fabric writes, not the pxr change manager).
7. **Heartbeat through every blocking reopen.** A reload-class variant switch or `population.open_usd` runs synchronously on the render thread and sends no frames meanwhile. Run the blocking call while a short-lived daemon thread calls `streamer.submit_last()` every ~2 s (re-streams the cached BGRA buffer); join it before the loop resumes so `stream_video` is never called from two threads at once.
8. **Coalesce the command queue; defer reads until after writes.** Collapse a burst of camera/variant/resolution mutations and apply once per drain. Any command that READS mutable state (e.g. GetCameraState for a project save) must be answered AFTER the coalesced writes are applied, or a save races and misses the just-set value.
9. **Idle power — every no-work loop branch sleeps (GPU AND CPU), and MEASURE it.** No stage open → `sleep(~0.05)` per tick; stage but no client/job/playback → `submit_last()` + `sleep(~0.1)` instead of `step()` — otherwise the loop path-traces full-rate with no viewer (measured ~80% util / 420 W idle). Verify the server PID sits at ≤ ~1 core in BOTH idle states. Note: a CONNECTED tab renders full-rate by design (a forgotten tab = hours of heat; pausing the stream on `document.hidden` is a worthwhile enhancement). Full recipe: stability-checklist item 9.
10. **Stream reconnect must always recover; never wedge the control plane.** Rebuild the ovstream `Server` (synchronously) before EVERY browser connect; treat a same-stage Open as attach/reconnect and ask the server for state (don't trust a stale client `connected` flag); keep the step gate inclusive of "warmup incomplete"/"job running" so the first frame after a reopen renders and `ready` fires even mid-reconnect; and give EVERY control-plane reply-queue read a `timeout` so a busy render thread can't hang `/api/stage`. See the checklist.

## ovrtx/ovstage/ovstream quick reference (ovrtx 0.4.x + ovstage 0.1.x + ovstream 0.4.x)

The app owns ONE `ovstage.Stage` (a `StageSession` coordinator wrapping it — see
`references/seed-recap.md`); the renderer **attaches** to it (`renderer.attach_ovstage(stage)`). ALL
scene mutation — population, live attribute writes, USD-time — goes through `ovstage`, never through
the deprecated `Renderer.open_usd` / `Renderer.write_attribute` / `Renderer.update_from_usd_time`
(those production-path calls are gone; they still exist as legacy renderer-owned APIs but using them
means you skipped attaching ovstage).

- Frame to ovstream must be a **BGRA8 CUDA device buffer**; ovrtx `LdrColor` is RGBA8 → a
  mandatory GPU R↔B swizzle (Warp kernel). No CPU-pixel→encode path.
- Render mode is a USD attr on the RenderProduct: `omni:rtx:rendermode ∈
  {"RealTimePathTracing","PathTracing","Minimal"}` (camelCase, NO spaces — the code form,
  not the spaced prose form). Switch = an `ovstage.Stage.write_attribute` (via the session) +
  `renderer.reset()` + warm-up frames.
- Camera: ovrtx has no native camera input. Drive `omni:xform` via `ovstage.Stage.write_attribute`
  with **one 16-lane float64 matrix element** built with `ovstage.make_dltensor` (dtype
  `DLDataType(code=kDLFloat, bits=64, lanes=16)`, shape `[1]`) and
  `semantic=int(ovstage.AttributeSemantic.MATRIX)` — NOT the ovrtx-0.3-era `(1,4,4)` tensor +
  `Semantic.XFORM_MAT4x4`. Row-vector layout (translation in row 3) is unchanged. Match
  `verticalAperture = horizontalAperture * height/width`. Query the target prim via
  `ovstage.PathDictionary(stage).create_path_list_from_strings([path])` →
  `stage.query_from_path_list(...)` — `stage.get_path_dictionary()` returns a raw C bundle that does
  NOT have `create_path_list_from_strings`; using it directly breaks live writes.
- ‼ **Read the camera FROM the composed stage — never synthesize it from angles.** Seed the free/orbit
  camera from the authored camera's `UsdGeom.Xformable(camPrim).ComputeLocalToWorldTransform(Default)`
  (and `snap_to` the orbit controller from that matrix, built with the stage's real
  `UsdGeom.GetStageUpAxis`). The orbit **pivot = eye + forward·focusDistance** (`GetFocusDistanceAttr`);
  if unset, project the asset bbox centre onto the forward ray — **NEVER default the pivot to `(0,0,0)`**,
  and never gate it on size/name heuristics that can reject every prim and collapse to the origin. A blind
  build that rebuilt the camera from spherical `(yaw,pitch,distance,up_axis)` + a bbox heuristic broke
  THREE things at once: origin pivot, no animated-rig playback, car tilted vs the environment. Camera-less
  stages → auto-frame (up-axis-aware three-quarter view), never a bare controller default.
- **Turntable Preview / `/api/playback {playing:true}`** is a LIVE wall-clock animator: record
  `t0=monotonic()`, and per render tick compute `tc = start + ((monotonic()-t0)*fps) % span`,
  `m = Xformable(rigCam).ComputeLocalToWorldTransform(TimeCode(tc))` on the live composite (which sublayers
  the rig sidecar), then write `m` onto the viewer camera via the `ovstage` fabric write above.
  `ovstage.population.update_from_usd_time` does NOT re-evaluate time-sampled xforms in this
  ovrtx/ovstage build — a **remaining library gap**, confirmed on 0.4.0/0.1.0, not a 0.3-only quirk —
  so this CPU pxr-evaluate + fabric-write is still the required workaround. Only setting a `playing`
  flag while the render path keeps writing the static orbit matrix = the "Preview does nothing" bug.
- **Pick-point**: prefer the hit's `worldPositionM` when it comes back non-zero (ovrtx 0.4's NDC pick
  fix makes it usable — see the pick recipe below); when it stays zero, fall back to the resolved
  prim's bbox-centre world point + extent (render PICK var, `hitCount>0`). On a MISS, place NO pivot —
  never substitute the orbit target / a hardcoded point (that floats the gizmo in empty space).
- **Stream watchdog must not reconnect-storm:** measure liveness via `requestVideoFrameCallback`
  decoded-frame arrival (NOT `getVideoPlaybackQuality` polling), seed the baseline at `onStart` success
  (not module load), single-flight reconnect, LATCH escalation (one `/api/stream/restart` per dry streak,
  reset by a real frame). A healthy live stream must sit idle indefinitely with zero reconnects.
- **Depth of field (f-stop):** DoF blur is PHYSICAL — radius scales with aperture diameter =
  `focalLength / fStop` against the camera's sensor aperture. On the viewer `UsdGeomCamera` author
  `horizontalAperture`/`verticalAperture` (≈20.955mm full-frame, vertical = ×h/w) + `focalLength` (mm)
  ALONGSIDE `CreateFStopAttr` + `CreateFocusDistanceAttr` (world units) — `focalLength`/`fStop`/
  `focusDistance` are schema attrs with defined fallbacks, so apply them via **live `ovstage` scalar
  writes** (`write_scalar_attrs`, no reopen needed). fStop WITHOUT a correct aperture+focalLength →
  negligible blur (fStop alone measures a ~1% centre-edge drop, vs ~97% with aperture+focal
  authored). Do NOT fake it with a DOF_BOOST multiplier — it won't render as blur. `fStop ≤ 0` clears DoF.
- First `Renderer()`/`step()` is slow (shader compile, can be minutes cold; cached after).
  Use generous readiness timeouts; emit a `{type:"ready"}` event on the first real frame.
- **EXPOSURE — the viewport must be well-lit, not near-black.** RTX post/tonemap exposure
  (`/rtx/post/*`, e.g. `filmIso`) set BEFORE the renderer warms up is **reset by the render
  preset**, so the scene renders dark. Apply a sensible default exposure AFTER warmup (and after
  any rendermode switch / reopen, which re-apply the preset), or enable auto-exposure. Treat the
  per-camera ISO control the same way. **Self-check:** a rendered frame's mean luminance must be in
  a usable range (clearly lit — not ~0); if the first frames are near-black on a lit stage, the
  exposure is being reset — re-apply it post-warmup.
- **Don't surface the error-material ("all shaders red") frames.** On a cold open the MDL
  materials compile over the first frames; until they do, RTX shows a flat **red/magenta error
  material**. Gate the stream: keep a "warming…" overlay and DON'T flip to LIVE / surface frames
  until materials have resolved (a non-error frame). A simple, robust gate: warm up several
  frames before marking ready, and keep the warming overlay until the first frame whose pixels
  aren't dominated by the error color. Also warm after every reopen/rendermode switch.
- See the seed skill's `ovrtx-rendering` / `streaming-server` / `streaming-client` recipes.

## Picking (focus distance + pivot point) — concrete, build-verified recipe

Both the **focus picker** and the **turntable pivot** need a ray-pick from a clicked pixel.
Do NOT return `"pick unsupported"` — this exact recipe works on ovrtx 0.4 (render thread only,
guarded on `has_stage`). ‼ **`enqueue_pick_query` takes an NDC `[0,1]` top-left rectangle, NOT
pixels** — passing raw pixel integers raises `ValueError: invalid NDC rectangle` (the pre-0.4
pixel-rect habit is an easy carry-over). Map the clicked pixel to NDC edges first:

```python
def _pick_ndc_rect(nx, ny, w, h):
    px = max(0, min(int(nx * w), w - 1)); py = max(0, min(int(ny * h), h - 1))
    return px / w, py / h, (px + 1) / w, (py + 1) / h    # [left, top, right, bottom] in [0, 1]

def _pick_prim(self, nx, ny):
    if not self.has_stage:                      # stepping a stageless renderer corrupts the sensor
        return ""
    rp = self.render_product_path
    w, h = self.stream_resolution
    left, top, right, bottom = self._pick_ndc_rect(nx, ny, w, h)
    self.renderer.enqueue_pick_query(rp, left, top, right, bottom)   # NDC rect at the click
    products = self.renderer.step(render_products={rp}, delta_time=1/60,
                                  ordinal=self.session.committed_ordinal)
    rvar = products[rp].frames[0].render_vars[ovrtx.OVRTX_RENDER_VAR_PICK_HIT]
    with rvar.map(device=ovrtx.Device.CPU) as mapped:             # MAP first, THEN subscript
        hit = int(np.from_dlpack(mapped.params["hitCount"]).copy().reshape(-1)[0])   # .params for hitCount
        pid = int(np.from_dlpack(mapped["primPath"]).copy().reshape(-1)[0]) if hit else 0
    return self.renderer.resolve_prim_path_id(pid) if pid else ""
```

`ovrtx_pick_hit.worldPositionM` gives an exact hit position when it is non-zero, so prefer it and
fall back to the resolved prim's world bbox when it stays zero. **Budget your time accordingly: on
ovrtx 0.4.0 it has been observed coming back zero on every hit, with a correct NDC rect and both
pick flags tried** (the `GIZMO` flag makes everything miss outright). Treat the bbox/click-ray
fallback as the path that will actually run, build it first, and verify it independently — do not
spend a session trying to make `worldPositionM` report. Both routes satisfy the gate. On the cached pxr composite stage (`Usd.Stage.Open(composite)`), compute the prim's
world bbox (`UsdGeom.BBoxCache(...).ComputeWorldBound(prim).ComputeAlignedRange()`) for that fallback:
- **`/api/pick-focus`** → distance from the live camera eye to `worldPositionM` (or, on a zero hit
  position, the bbox center/nearest face) → set the active camera's `focus_distance` → a live
  `ovstage` scalar write so DOF updates → return `{focus_distance}`.
- **`/api/pick-point`** → the bbox CENTER as a world point (turntable pivot picks may keep the bbox
  convention either way) → return `{world:[x,y,z], prim_path}`
  for the turntable pivot gizmo.

Verify live: arm Pick, click the car → focus value changes; arm Pick pivot, click → a gizmo drops
on the model at a non-zero world point. (The seed skill's `native-picking-selection` /
`camera-picker` references cover the same API.)

## Frontend

A vanilla HTML/JS page is sufficient (no build step). **Read `omniverse-dev-variant-presenter-ui` for
the exact visual design + layout** — the GitHub-dark + NVIDIA-green theme tokens, the
header / viewport+timeline-strip / 330px right-side tabbed panel layout, and the component styles
(pill chips with swatches, segmented buttons, blocks, the NLE timeline strip). The app must look
and feel like the product, not just function — implement that design, don't invent a generic one.
Load the ovstream WebRTC client library (`omniverse-webrtc-streaming-library.js`) — a
raw `RTCPeerConnection` will NOT interoperate with ovstream signaling.

**Do NOT commit that library.** It is NVIDIA's own StreamSDK code under NVIDIA's terms, not
yours to redistribute. Have the launcher **fetch it on first run**, pinned and verified, and
add it to `.gitignore`:

- Source: `https://raw.githubusercontent.com/NVIDIA-Omniverse/ovstream/af7f1f9006d1037a3cc7b8eca73f39a6469b69c2/examples/webrtc_client/omniverse-webrtc-streaming-library.js`
  (pinned to a commit, never `HEAD`, so upstream cannot silently change what you fetch).
- Verify SHA256 `447A74830162B91CB92B0A636F02C0B3E668D835E2A4496F560E31E2B48E5C71`.
- Download to `<name>.partial` and only `Move-Item` it into place **after** the hash matches.
  Writing straight to the real filename means an interrupted download leaves a partial file
  that the next launch's "is it already there?" test silently accepts → a corrupt viewer.
- Keep the fetch OUT of the crash-relaunch loop (check-and-skip if the file exists) — a crash
  must relaunch instantly, not re-verify a 700 KB download.
- On a hash mismatch, delete the file and refuse to start; never leave a corrupt one behind.
- Guard the page too: if `window.OVWebRTC` is undefined, render a visible message with the URL
  to fetch by hand, instead of a dead-looking blank app.

**Client HTTP calls must be SAME-ORIGIN / relative** — `fetch("/api/...")`, not a
reconstructed absolute base like `http://${hostname}:${port}/api/...`. The IPv4 normalization
(`localhost`/`::1` → `127.0.0.1`) is for the **WebSocket and the ovstream signaling host
ONLY** — applying it to the HTTP fetch base turns a same-origin request into a CROSS-ORIGIN
one when the page is served on `localhost` (the rewrite changes the host), and the server has
no CORS headers → every `fetch` fails with "Failed to fetch". Keep REST relative; rewrite host
only for the WS/ovstream connect. **Wrap your app JS in an IIFE** (the streaming
library leaks minified globals like `el` into global scope; a top-level `const el` in your
code collides → a parse-time SyntaxError silently kills the whole file). Serve `index.html`
with `Cache-Control: no-cache, no-store, must-revalidate` so script changes take effect; no
`?v=N` query strings are needed on top of that. See `ovrtx-timeline-nle` for the timeline UI + the Python↔JS state mirror.

## Validation

Stream the ConceptCar, confirm `remote-video.videoWidth > 0` and a non-black pixel,
switch a fast-path variant (sub-second, no reopen) and a reload set (stream survives),
snap a camera, render a 1-permutation batch to PNG, render a 2-clip timeline to MP4,
author a turntable and preview the spin, mirror a remote URL with the source untouched.
Capture this evidence before claiming done (seed skill `validation.md`).
