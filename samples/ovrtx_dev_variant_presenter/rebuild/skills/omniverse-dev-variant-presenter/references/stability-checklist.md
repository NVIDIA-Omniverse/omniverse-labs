# Stability checklist — the bugs that wear one face ("black viewport / stream lost")

Each item below was a distinct, costly failure on a real RTX workstation. A streaming
USD viewer that skips these *appears* to work in a quick test and then fails
intermittently. Implement all of them.

## 0. Windows MAX_PATH: a deep `.venv` breaks Slang shader compilation and crashes the renderer natively
**Read this one FIRST — it bites during setup, before any of your code runs.**
**Symptom:** the app installs and imports fine, then the renderer dies natively on first open, or
every frame comes back blank/black. The logs show Slang/shader build failures rather than anything
resembling a Python error, so it reads as "ovrtx is broken on this machine" instead of a path problem.
**Cause:** the total path length to shader build artefacts under your virtualenv exceeds Windows'
260-character `MAX_PATH`. This is easy to hit when the project lives under a long directory (a nested
scratch tree, a synced folder, a per-session temp path). **Junction/symlink tricks do NOT help** —
carb and Slang canonicalise the path back to the long real one before opening it.
**Fix:** put the virtualenv somewhere physically short — e.g. create it at `C:\<short>\<name>env` and
point the launcher at it — rather than letting `uv sync` place `.venv` beside a deeply nested project.
Long-path support being enabled in Windows is not sufficient on its own here.
**Verify:** render one frame and confirm it is non-blank before building anything on top; if the venv
path plus roughly 120 characters exceeds 260, relocate before debugging anything else.

## 1. WebRTC media never pairs → black (the #1 intermittent killer)
**Symptom:** signaling (TCP) connects, the page loads, the badge says LIVE — but the
viewport is black and `remote-video.videoWidth === 0` / `bytesReceived` stays 0. Works
some launches, not others.
**Cause:** `ovstream` auto-detects which host IP to advertise as its media (ICE) candidate.
On a multi-NIC / Wi-Fi+Ethernet+VPN box it picks an unreachable/link-local address
(observed: media UDP bound to `127.0.0.238`), non-deterministically per launch.
**Fix:** `cfg.webrtc_public_ip = "127.0.0.1"` in the `ServerConfig` (localhost-only viewer).
For LAN/remote use the host's reachable IP instead.
**Diagnostic:** `remote-video.videoWidth` (>0 = media flowing). `Get-NetUDPEndpoint
-OwningProcess <pid>` shows the bind address.

## 2. Chrome autoplay pauses the <video> → black (frames ARE arriving)
**Symptom:** black viewport but `videoWidth` is 1920 and a canvas pixel sample shows the
car — the element is just `paused`. Recurs across fresh servers AND fresh tabs.
**Cause:** `<video autoplay>` that is NOT muted is blocked by Chrome without a user gesture.
**Fix:** element `muted` + call `v.play()` on stream attach. The render stream is
video-only so muting costs nothing. `paused===true` + content-present = this bug.

## 3. localhost → IPv6 ::1 trap (WebSocket refuses)
**Symptom:** page + app.js load, then `/events` WS and ovstream signaling fail with a bare
`WebSocket connection failed` / `WinError 1225`.
**Cause:** uvicorn bound to `127.0.0.1` is IPv4-only; Chrome-on-Windows resolves
`localhost` → IPv6 `::1` first for WebSockets.
**Fix:** advertise/connect `127.0.0.1`. Client shim: normalize host before use —
`s.replace(/^localhost/i,'127.0.0.1').replace(/^\[?::1\]?/,'127.0.0.1')` — for BOTH the
`/events` WS and the ovstream `server` host. Real hostnames pass through.

## 4. Never two ovrtx/ovstream instances on one box
**Symptom:** launching a second renderer process access-violates (~0xC0000005) ~1.5 s into
init AND momentarily wedges the FIRST server's signaling.
**Cause:** carb/ovstream process-global state; pxr + ovrtx share one `usd_ms.dll`.
**Fix:** exactly one renderer in one process. Before launching, assert no other instance is
up. `ovstream.initialize()/shutdown()` run ONCE per process (ref-counted); on client evict
rebuild only the `Server`, not the global context. Don't run headless verification scripts
that build a renderer alongside the live server.

## 5. Never step a stageless renderer (the crash-storm root cause)
**Symptom:** `[ovrtx] Unable to find render product prim` + `Invalid USD RenderProduct
Prim` + a flood of `omni.sensorscheduling ... handle 0 not found` (~2000/s) → native death
~2.5 min later.
**Cause:** `renderer.step(render_products={rp})` before any `ovstage.population.open_usd`
(your `StageSession.populate_usd`). A stale browser tab's gizmo/probe timer hammering a freshly
relaunched server before the stage reopened re-arms this every restart.
**Fix:** a `has_stage` flag; guard ALL step/pick/probe paths on it; reject stageless
pick/probe HTTP routes with 400; gate client-side polling timers on server readiness.

## 6. Serialize all pxr stage work behind one process-global RLock
**Symptom:** `UsdVariantSet::SetVariantSelection → Sdf_ChangeManager::_OpenChangeBlock →
TfNotice::_Send` access violation when a background variant classifier authors selections
while the render thread composes/populates via `ovstage.population.open_usd`.
**Cause:** pxr's Sdf change manager is process-global and not thread-safe; two threads
composing/authoring ANY stages at once corrupt it.
**Fix:** `USD_LOCK = threading.RLock()`. Hold it around every stage open / compose / author:
the scanner, each classifier set (release between sets so the render thread waits ≤1 set),
the reopen (build composite + `population.open_usd`), the whole batch, the mirror, bbox reads
for picking. **RLock** so render-thread nesting (batch → reopen) doesn't self-deadlock. Do NOT
lock steady-state `renderer.step()` / `ovstage.Stage.write_attribute` (ovstage/ovrtx fabric,
not the pxr change manager).
**Belt-and-suspenders (strongest):** `ovstage.population.open_usd` itself composes a USD stage
internally and shares that same process-global Sdf change manager — so a background classifier
daemon thread can corrupt it EVEN with lock discipline if a reopen slips through. The most robust
pattern is to (a) wrap every `population.open_usd`/`populate_usd` in `USD_LOCK` too, and (b) run
classification **SYNCHRONOUSLY on the render thread BEFORE firing `ready`** — not in a background
daemon — so no reopen can ever race it. That removes the race by construction; it costs ~the
classify time (~10 s) added to first warmup, which is acceptable. (A background classifier WITH
careful locking also works and gives a faster `ready`, but synchronous-before-`ready` is the
crash-proof default.)

## 7. Heartbeat through every blocking reopen
**Symptom:** a reload-class variant switch (visibility/transform/structural, e.g. a
lighting/Backdrops set) drops the stream → stuck "Reconnecting…".
**Cause:** `_reopen` (build composite + `populate_usd` / `population.open_usd`) runs
synchronously in the command loop; no frames/heartbeats during it; a heavy reload + any lock
wait exceeds ovstream's ~7 s liveness window → client dropped.
**Fix:** run the blocking call via a helper that spawns a short-lived daemon thread calling
`streamer.submit_last()` every ~2 s (re-streams the cached BGRA buffer — safe from another
thread while the render thread is in `populate_usd`); join it before the loop resumes.

## 8. Coalesce the queue; defer reads until after writes
**Symptom:** a display change + a project save in the same drain → the save snapshots state
BEFORE the deferred camera write applied → the just-set value is missing from the save.
**Fix:** the loop coalesces a burst of mutations (camera/variant/resolution collapse to one
apply). Any command that READS mutable state collects a reply and is answered at the very
END of the drain, after the coalesced writes are applied. (Unit tests that call methods
directly miss this — only a live HTTP probe hammering write+read together exposes it.)

## 9. Idle power — EVERY no-work loop branch must sleep (GPU **and** CPU), and MEASURE it
**Symptom:** ~80% GPU util / ~420 W with no viewer connected; or the box running hot on CPU
for hours while apparently idle.
**Cause:** the loop calls `step()` every iteration whenever a stage is open — or ANY idle
branch (including the **no-stage** one) spins without a sleep.
**Fix:** two throttled idle branches, both mandatory:
- **no stage open** → `sleep(~0.05)` per tick, nothing else;
- **stage open, no client, no batch/timeline/playback** → `submit_last()` (keeps the cached
  frame warm for instant reconnect) + `sleep(~0.1)` instead of `step()`.
Full rate only while a client is attached or a job/playback runs.
**Verify by MEASURING, not reading:** the server process must sit at ≤ ~1 core with GPU
near-idle in BOTH states (sample the PID's CPU delta over ~10 s).
**Operational note:** a CONNECTED client renders at FULL RATE by design — a browser tab left
open keeps the GPU+CPU hot indefinitely (measured ~16 cores × 4 h from one forgotten tab).
Worthwhile enhancement: pause/disconnect the stream on `document.hidden`, re-attach on
visibility — the cached-frame + reconnect machinery already makes this cheap.

## 10. Stream reconnect must always work; never let a reopen wedge the control plane
**Symptom:** after the WebRTC client drops once (a reopen, a resize, a network blip, or the
stage being re-opened underneath it), re-opening from the UI does NOT bring the video back
(`remote-video.srcObject` stays null); repeated reopen/restart cycling leaves the render loop
not stepping (GPU idle) and `/api/stage` hanging.
**Causes + fixes:**
- **Connect only ONCE the server has frames (`ready`), and do NOT rebuild the ovstream `Server` on a
  normal open/connect** — rebuilding the Server tears down a HEALTHY live WebRTC session (the client
  ends at "stream error" / a frameless session drops). Rebuild the Server ONLY to evict a *confirmed*
  ghost — i.e. on an explicit reconnect AFTER a real drop, or via `/api/stream/restart`. Build the
  Server ONCE at startup (`bootstrap`, item 16) and reuse it; the browser connects when `ready` fires.
  (Earlier guidance to "rebuild before every connect" was too aggressive — it killed live sessions;
  a freshly-rebuilt Server's first AV1 keyframe can also arrive after the client's `maxReconnects`
  budget, ending at "stream error".)
- **Treat a same-stage Open as ATTACH/RECONNECT**, and ASK THE SERVER for current state rather
  than trusting stale client `connected` flags — trusting the client flag is exactly why
  "re-Open does nothing" after a drop. NOTE a same-stage re-open ATTACH emits NO new `ready` event,
  so reconcile the client status from `/api/stage` (don't wait forever for a `ready` that won't come).
- **A reopen must keep the loop alive** (heartbeat, item 7) and must not leave the loop
  idle-gated when it should render — include "warmup incomplete" / "a job/playback running" in
  the step gate, not only `is_client_connected`, so the first frame after a reopen always
  renders and `ready` fires even if the client is mid-reconnect.
- **Give EVERY control-plane reply-queue read a timeout** (`reply.get(timeout=...)` + a default)
  so a busy/blocked render thread can never hang `/api/stage`, `/api/projects/save`, etc. A
  hung status endpoint makes the whole app look dead.
- **`/api/stage` must NEVER raise** — wrap every helper it calls (e.g. `rig_info` opening a turntable
  sidecar via `Sdf.Layer.FindOrOpen`, units/metadata reads) in try/except that degrades to a default.
  A single malformed sidecar / unreadable layer that throws on the status path 500s `/api/stage` → the
  client's readiness poll never succeeds → the app looks permanently "warming"/unready and EVERY
  render-dependent feature fails. The status endpoint is load-bearing; it must be exception-proof.

## 11. Serve the page and its scripts from the SAME base — or the whole UI is DOA (silent)
**Symptom:** the page loads, the chrome looks right, but **clicking Open does nothing** and no control
works. `remote-video.videoWidth===0`, no chips, status stuck at the initial HTML text. No server error.
It looks like "the stream won't connect" but the frontend JS **never executed at all**.
**Cause:** `index.html` is served at `/` and references its scripts root-/document-relative
(`<script src="app.js">` or `"./app.js"` → resolves to `/app.js`), but the static assets are mounted
under a PREFIX (`app.mount("/web", StaticFiles(...))` → only `/web/app.js` exists). Every `<script>`
404s (FastAPI returns a 22-byte `{"detail":"Not Found"}`); the browser silently fails to load them. If
the frontend is one IIFE, NOTHING runs: no `onclick` handlers bind, no streaming-lib global, no `boot()`.
**Fix:** the base the HTML references MUST match where assets are served. Pick ONE and be consistent:
- mount `StaticFiles(directory=WEB, html=True)` at **`/`** (register ALL `/api/*` + `/events` routes
  FIRST so they win), and reference scripts at root (`/app.js`), **or**
- keep the `/web` mount and reference every asset under it (`/web/app.js`, `/web/<lib>.js`).
The fetched WebRTC lib (see the orchestrator SKILL — downloaded by the launcher on first run,
pinned + SHA256-verified, gitignored, NOT committed), `timeline-core.js`, `app.js`, and any CSS
all must resolve to 200.
**Diagnostic (do this every build):** `curl` each `<script src>` and assert **200 + a JS content-type**,
not 404. In-browser: `typeof document.getElementById('open-btn').onclick === 'function'` and
`typeof window.OVWebRTC === 'object'`. (A mismatch like `./app.js` vs a `/web` mount leaves every API
test green and the app 100% dead in the browser.)

## 12. Self-verify must LOAD THE PAGE, not just diff the HTML string (the verification trap)
**Symptom:** the build's own live test passes, yet a human opens the app and nothing works.
**Cause:** a self-verify that drives `/api/*` with `requests`/TestClient and "checks the frontend" by
asserting the index.html **text** contains `"app.js"` / `"remote-video"`. That string is present even
when `/app.js` 404s, the WS lib is the wrong global, or app.js has a syntax error — none of which a
string match or an API-only probe can see — which is exactly how a DOA frontend reports as green.
**Fix:** the GPU/live layer of the suite MUST, after the server is up:
1. fetch EVERY asset `index.html` references (`<script src>`, `<link href>`) and assert each is **200**
   with a script/CSS content-type — fail loudly on any 404;
2. ideally load `/` in a real headless browser and assert the JS actually executed — e.g.
   `window.OVWebRTC` exists AND `getElementById('open-btn').onclick` is a function AND, after a UI Open,
   `remote-video.videoWidth>0` with a non-black pixel sample. "HTTP 200 on `/api/*`" and "HTML mentions
   app.js" are NOT proof the app runs. Drive real controls on pixels (the production-parity sweep).

## 13. ovrtx/ovstage may still serve a stale look on a REUSED composite path (the recurring "variant/Backdrops doesn't change" bug)
**Symptom:** a reload-class variant switch (Doors, Backdrops/environment, anything that rebuilds the
composite + populates it) appears to do nothing — same materials/lighting — even though the new selection
IS authored and composes correctly in pxr. The Backdrops/env case is the usual place this bites.
**Cause (0.3-era):** ovrtx cached the opened stage keyed by the composite's FILE PATH; re-authoring
selections into a REUSED path and reopening it served the cached materials/lighting.
**Status on ovrtx 0.4 + ovstage 0.1:** `ovstage.population.open_usd`/`open_usd_from_string` semantically
**replaces** the stage's prior content at each call (unlike the old renderer-owned open), so the caching
itself is no longer the reason to avoid a reused path. The reason that remains is ownership: each
recomposition is a DIFFERENT document, and re-authoring one fixed path underneath anything still reading
it — the renderer, or your own pxr stage held open to measure bboxes for click-to-focus — is its own bug.
**Give every recomposition a UNIQUE composite path** — `composite_<hash>_<counter>.usda` — and prune
leftovers from earlier runs at startup, since these accumulate one file per reopen. **This is not
composite-only** — it
would bite ANY fixed-path reopen, including the remote-mirror root (see the mirror skill: a child
sublayer skipped on first parse stays cached child-less if the root path is reused). Verify: switch
Backdrops → whole-frame pixels move (Δ large); switch a reload variant twice → it actually changes
each time.

## 14. An async route that blocks on the render-thread reply wedges the WHOLE event loop
**Symptom:** during a cold open (shader compile, tens of seconds to minutes) EVERY endpoint hangs —
`/api/stage` polls time out, the page looks dead — not just the open call.
**Cause:** an `async def` FastAPI handler that does a blocking `reply_queue.get(timeout=...)` to wait on the
single render thread. A blocking call inside the async event loop stalls ALL concurrent requests, not just
that one.
**Fix:** run the blocking render-thread reply-wait OFF the event loop — `await asyncio.to_thread(...)` /
`loop.run_in_executor(...)` (or make the route `def`, not `async def`, so Starlette threadpools it).
Verify: hammer `/api/stage` during a cold open and confirm it answers with 0 timeouts throughout.

## 15. ovstream `is_client_connected()` can stay False while RTP actually flows → gate streaming on the /events WS, not that flag
**Symptom:** the browser shows live video (frames decoding) but the server thinks no client is attached, so
the idle-GPU throttle (item 9) starves the decoder mid-session → stutter / "stream error".
**Cause:** in this ovstream build `Server.is_client_connected()` does not reliably reflect a live WebRTC
media session (observed: stays False while RTP flows; negotiated codec is AV1).
**Fix:** track UI presence via the **`/events` WebSocket client count** (`ui_clients>0`) and step at full
rate whenever a page is open (or a job/playback runs), rather than trusting `is_client_connected()`. Keep
streaming continuously while any WS is open so a freshly-rebuilt Server's first AV1 keyframe isn't starved
before the client's reconnect budget (`maxReconnects`) runs out.

## 16. `ovstream.initialize()` must run BEFORE warp / ovrtx / full pxr import, or it dies with WinError 127
**Symptom:** the render thread silently dies during startup (no traceback that surfaces) — the app
"hangs" warming forever, the stream never comes up. Or `ovstream.initialize()` raises
`OSError: [WinError 127] The specified procedure could not be found`.
**Cause:** importing `warp` / `ovrtx` / a full `pxr` set first loads DLLs that shadow/clash with the
ones `ovstream.initialize()` needs; initializing ovstream after them fails to resolve a procedure.
**Fix:** initialize ovstream **first**, in a tiny `bootstrap.py` that the entrypoint imports BEFORE
uvicorn/app/pxr/warp/ovrtx — `import ovstream; ovstream.initialize(); server = build_server(...)`,
THEN import the rest. (ovstream is process-global + ref-counted; init once — see item 4.) Wrap the
render thread so an init failure is LOGGED loudly, not swallowed into a silent hang.

## 17. The explosion guard must reject SYNCHRONOUSLY in the HTTP handler — never defer it to the render thread
**Symptom:** `POST /api/batch` of an over-limit matrix returns **504** ("enqueue timed out") instead of
a fast **409** — intermittently, right after an open (when the render thread is warming). The guard
*logic* is correct (returns 409 the instant the render thread is idle), so it passes a naive unit test.
**Cause:** the over-limit count was checked ON the render thread (the batch command's reply-wait, e.g.
`request("batch", timeout=20)`); while the render thread is busy with the post-open warmup/convergence,
the batch command queues behind it and the reply-wait elapses → 504.
**Fix:** the explosion guard needs NO stage/GPU — it's `prod(len(included[set]))` vs the threshold.
Compute it **synchronously in the request handler** from the job payload and return 409 BEFORE enqueuing
any render-thread work. Only enqueue the actual render once the guard (and `confirm`) pass. Verify with
a realistic full-cartesian (e.g. 1210 perms) posted IMMEDIATELY after open, not just a warm idle server.

## 18. The native folder dialog (Tk) must run on ONE dedicated thread + be single-flight, or a 2nd Browse kills the process
**Symptom:** clicking a 📁 **Browse** button a second time (e.g. twice in a row, or after
cancelling the first with no selection) drops the viewport — the stream dies and the watchdog
relaunches. In the logs: repeated `main: thread_init: already added for thread`, then a native
abort, then the `/events` WebSocket `connection closed`.
**Cause:** the `/api/browse-folder` handler opened the dialog by creating a fresh `tk.Tk()` per
request on a **transient `run_in_executor` worker thread** (default `ThreadPoolExecutor`, rotating
threads). Tkinter/Tcl is **not thread-safe**: a second dialog on a different worker — plus Python
finalizing the first Tcl interpreter on a thread other than the one that created it — trips the
native `Tcl_AsyncDelete: async handler deleted by the wrong thread` abort, which kills the whole
process (and with it ovstream → the viewport).
**Fix (two parts, both required):**
- **Pin ALL Tk work to ONE dedicated, long-lived thread with a single reused, never-destroyed
  `Tk()` root.** Start it once (lazily), marshal each `askdirectory` request onto it via a queue,
  and return the result. No per-call `Tk()` create/destroy, so nothing is ever created or finalized
  across threads. (The route still hands off via `run_in_executor` so the event loop never blocks —
  but the *Tk calls* all happen on the pinned thread, not the pool thread.)
- **Single-flight the Browse action** so a dialog can never be opened twice at once: front-end guard
  (`browseFolder._busy`, shared across all 📁 buttons — only one OS dialog can exist) that ignores a
  click while one is outstanding; and a back-end non-blocking lock in the pick path that returns `""`
  (treated as cancel) for a concurrent request instead of QUEUING a second dialog behind the first.
**Enforcement:** a native modal cannot be driven by the headless browser/HTTP gates, so this is gated
by a **required pytest regression** — assert every dialog runs on one thread id with one root created
once, and that a concurrent pick returns `""` opening only ONE dialog — plus code review. Keep the
picker a **stdlib-only module** (threading + queue + a lazily-imported tkinter; NO app/render/USD
imports) so the test exercises it in isolation — you do NOT stand up the FastAPI app or import `pxr`
to test a folder dialog. (`tests/test_folder_picker.py`; see `SPEC-FUNCTIONAL.md` `/api/browse-folder`
+ `production-parity-checklist.md`.)

## Animated-camera gotcha (turntable/timeline) — read before rendering motion
`ovstage.population.update_from_usd_time(t)` does **NOT** re-evaluate time-sampled xforms on
ovrtx 0.4 + ovstage 0.1 (with or without `renderer.reset()`) — this is a **confirmed remaining
library gap** on the installed 0.4.0/0.1.0 build, not a 0.3-only quirk. Shooting through an
animated stage camera renders its default-time pose every frame. The ONLY working mechanism:
evaluate `UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(Usd.TimeCode(tc))` in pxr on the
composed stage, then write it onto the viewer/render-product camera's `omni:xform` via
`ovstage.Stage.write_attribute` (the `StageSession.write_omni_xform` helper) per frame. Used for
live preview spin, timeline scrub, and batch animation_range alike. **Verify motion
GEOMETRICALLY** (pick-prim sweeps or eyeballing quarter-lap frames), never by pixel-hash
uniqueness — RT2 noise differs every frame regardless.

Also: the composite ROOT layer must hoist `timeCodesPerSecond` (and start/end) from the
user stage. USD scales sublayer time by `root_tps/sublayer_tps`; a 60 fps rig under a bare
24 fps root composes its spin at the wrong rate.

**Camera POSE must stay a LIVE per-frame fabric write — never bake it (the orbit-killer).** The
viewer camera pose is driven by `ovstage.Stage.write_attribute` on `omni:xform` — one **16-lane
float64 matrix element** built with `ovstage.make_dltensor` (`DLDataType(code=kDLFloat, bits=64,
lanes=16)`, shape `[1]`) and `semantic=int(ovstage.AttributeSemantic.MATRIX)` (NOT the ovrtx-0.3-era
`(1,4,4)` tensor + `Semantic.XFORM_MAT4x4`) — written through a query built from
`ovstage.PathDictionary(stage)`, EVERY frame from the ovstream-input→orbit-controller path, each
write followed by `advance_write_floor`. This is what makes viewport mouse-drag orbit/pan/dolly work
live. **"Baking" the camera pose into the composite (reopen-per-move) breaks live orbit outright
— the viewport stops responding to the mouse, while everything else still looks healthy.** If the
`omni:xform` write STATUS_BREAKPOINT-crashes, the cause is almost always a wrong tensor shape/dtype/
semantic, a `Stage.get_path_dictionary()` raw-bundle query instead of `ovstage.PathDictionary(stage)`
(a malformed write is a crash/poison vector — see the fast-path note), OR stepping a stageless
renderer (item 5) — FIX THAT, don't abandon the live write. Separately: camera OPTICS
(focal/f-stop/focus/exposure) apply via **live `ovstage` scalar writes** (schema attrs with defined
fallbacks — no reopen needed); `exposure:iso` (a custom attribute with no fallback) still needs the
always-correct reopen path. Baking OPTICS via a reopen is fine when you must; baking the POSE never is.
