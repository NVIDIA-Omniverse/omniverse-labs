# Build: Dev Variant Presenter (single-shot) — spec-driven, gated by an executable grader AND a real-browser verifier

You are an expert Omniverse / USD / Python engineer and a careful frontend developer. Build a
complete, runnable **Dev Variant Presenter** that matches a PRODUCTION app in function, UX, AND
robustness. Work autonomously to completion. Don't stop to ask.

## ‼‼ THE REQUIREMENT LEDGER — 18 subtle behaviours, ALL gated
Every item below is easy to get wrong, easy to *look* right in a smoke test, and dead in a user's
hands when wrong. The SPEC + skills describe the correct behavior; these 18 are pulled out here
because they are the subtle ones, and each is machine-checked by one of the two gates. Get every one
right the first time.

**Live camera, orbit, and the turntable rig**

1. **Orbit direction is PINNED.** Convention: `azimuth -= Δx·0.005`, `elevation += Δy·0.005` (Y-up,
   azimuth from +Z toward +X): dragging right DECREASES the eye azimuth about the pivot; dragging
   down RAISES the eye. A sign slip here inverts ALL camera controls at once, and the gate checks the
   sign. (SPEC-UX viewport.)
2. **Turntable Preview must spin the camera in a CONTINUOUS orbit around the pivot — not a one-off
   jerk, not a frozen frame.** The preview is a per-tick WALL-CLOCK camera animator: each render-loop
   iteration compute `tc = start + ((monotonic()-t0)*fps) % span`, evaluate the rig's world pose with
   pxr `ComputeLocalToWorldTransform(tc)` on the live composite, then write it onto the VIEWER camera
   via `ovstage.Stage.write_attribute("omni:xform", ...)` (a `StageSession.write_omni_xform` helper —
   one 16-lane float64 matrix element via `make_dltensor`), then `step(..., ordinal=committed_ordinal)`
   at full rate. **NEVER `population.update_from_usd_time` for the live spin** (it does not move
   time-sampled xforms in this ovrtx/ovstage build — a remaining gap, not a 0.3-only quirk — so this
   CPU pxr-evaluate + fabric-write is still the required workaround) and **NEVER `reset()`/reopen per
   frame** (that starves the stream to a ~6fps slideshow, which reads as a jerk rather than a spin).
   The orbit is only visible when the animated rig is the ACTIVE render camera; activating it reopens
   (the stream reconnects briefly) — keep the play state alive across that reconnect. See the
   `ovrtx-turntable-camera` skill's "CONTINUOUS orbit" section.
3. **Preview-spin STOP restores the user's EXACT pre-spin framing** — sample-and-hold the free
   camera pose at spin start; write THAT back on Stop. Anything else snaps to an arbitrary pose.
4. **Preview-Stop restores the pose SAMPLE-AND-HELD at spin start** — hold it in a dedicated slot
   when playback starts; never restore "the current free-camera state" (mid-spin writes leak into it
   → Stop lands somewhere weird). Must hold for long spins and repeated preview/stop cycles.
5. **The pivot gizmo must LOOK like an axis gizmo: three colored axis LINES (X red / Y green /
   Z blue, dark-outlined) through the pivot + a white center dot — NOT just a cluster of dots.**
   Endpoints = `pivot ± L·axis` projected via `/api/project`, re-projected ~every 300 ms; fat
   invisible hit-lines carry `.handle[data-ax]`. (SPEC-UX "Turntable + pick gizmo";
   `ovrtx-turntable-camera` skill.)
6. **Gizmo drags move the pivot through WORLD SPACE by pure screen→world math — NEVER by
   re-picking geometry.** Axis drag = mouse delta projected onto the axis's screen direction ×
   world-units-per-pixel, applied along the WORLD axis; center-dot drag = the same in the camera
   right/up plane. Re-picking under the cursor on every drag-move makes the pivot STICK to the car's
   surface instead of tracking the drag. `/api/pick-point` fires ONLY for the armed "Pick pivot"
   click — the gate asserts ZERO pick-point calls during gizmo drags.
7. **The pivot gizmo HIDES while the spin plays** (returns on Stop) — WYSIWYG preview; it also
   re-projects against the frozen free-camera pose during playback, so leaving it up paints a
   stale artifact over the spin. (`ovrtx-turntable-camera` skill.)

**Variants — the fast path**

8. **Fast-path variant switches must be IMMEDIATELY VISIBLE — including from the timeline.** The
   failure mode: clicking Carpaint changes NOTHING until an unrelated reopen (switching a Backdrop,
   or a Preview spin) flushes it — and timeline material clips are equally dead while
   geometry/environment clips work. The fast path must push its writes through
   **`ovstage.Stage.write_attribute` (via a `PathDictionary(stage)` query — never the deprecated
   `Renderer.write_attribute`)** on the classified shader inputs, advance the write floor, **and then
   `renderer.reset()`** (clear accumulation so the live loop re-converges). Do NOT self-test only
   before classification finishes (everything reloads then, hiding the bug) — the gate waits for the
   `classified` event and switches a fast set TWICE, requiring a pixel change within seconds each
   time. (SPEC-FUNCTIONAL `/api/variant`; `usd-variant-live-switching` skill.)
9. **The classifier must NEVER hang silently — `classified` within ~120 s on EVERY stage, including
   the 11 GB mirrored S3 stage.** A classifier that hangs there emits no event and no error → the
   fast path never engages → every chip AND timeline switch degrades to a slow reload, which reads to
   the user as "variants don't work." Time-bound per set; emit `classified` (even with reduced
   `fast_sets`) + an `error` on trouble. (SPEC-FUNCTIONAL `/api/variant`;
   `usd-variant-live-switching` skill.)

**Timeline**

10. **An ANIMATED camera clip (Turntable rig) must ANIMATE under the timeline playhead** — scrub/play
    across a camera-track clip re-poses the rig at the clip-relative stage time (`at_s` →
    `loop_stage_time` → pxr-evaluate → viewer-camera write). Snapping camera clips at frame 0 yields
    zero rotation in the timeline. (`ovrtx-timeline-nle` skill; SPEC-UX Timeline.)
11. **The timeline-render MP4 must be BROWSER-PLAYABLE: H.264 (`libx264`) + `yuv420p` via
    imageio-ffmpeg, at the current display resolution — NEVER `cv2.VideoWriter mp4v`** (which writes
    a valid-looking file that `/api/video` serves fine and Chrome silently cannot decode). Render at
    the CURRENT display resolution, never a hardcoded 640×360. (`ovrtx-timeline-nle` skill.)

**Stream lifecycle** (these surface on a remote/mirrored stage in particular — the browser gate has
an S3 lane that exercises them)

12. **ATTACH must fully revive the client — `ready` never re-fires.** A page (re)load onto an
    already-open stage (or Open on the same path) gets NO `ready` event: the client MUST reconcile
    from `/api/stage` and become FULLY interactive — overlays cleared, orbit input armed, chips
    posting, panels populated. Gating any of that on the live `ready` event alone produces a zombie
    page on every attach (stuck "downloading…" overlay, video decoding but orbit dead and chip clicks
    silently dropped). **AND: input re-binding requires a server-side stream REBUILD on attach** —
    ovstream is single-client, and a new page connecting over a dead ghost session gets video but NOT
    input (bound to the ghost forever). Treat attach-with-existing-session exactly like your
    resolution-change path: rebuild the ovstream session so video AND input bind to the new client.
    (SPEC-UX "Stream self-healing".)
13. **The boot overlay must HAND OFF** — download → "compiling shaders / warming" → GONE the moment
    frames flow. The trap is leaving "downloading stage…" painted over a fully live stream. (SPEC-UX.)

**UI layout + readability**

14. **Tab labels must be READABLE in every state.** Filling `.tab.active` solid green leaves the
    label invisible. Active tab = green TEXT + green underline on the dark bar (any styling must keep
    text/background contrast ≥ 3:1; the gate measures computed styles). (ui skill.)
15. **Grid tab: the camera selector is a COMPACT DROPDOWN, not a tall per-camera checklist.** A stage
    with ~18 cameras (ConceptCar) must not push the "Include sets" / variant chips off-screen. Use a
    multi-select dropdown (or a dropdown button opening a checkbox popover) under id `#grid-cameras`,
    still able to select one OR several cameras. (SPEC-UX "Grid / batch".)
16. **Timeline VIEWS within a project are a first-class feature — not just Save/Open Project.** With a
    project open, reveal `#tlv-block` ("Saved track views"): `#tlv-save` names the current timeline as a
    view in `#tlv-list`, `#tlv-load` REPLACES the working timeline with a saved view, `#tlv-del` removes
    one; MULTIPLE named views coexist per project. Hidden (with `#proj-hint`) when no project is open.
    (SPEC-UX "Projects".)

**Process robustness**

17. **IDLE POWER — every render-loop branch with no work MUST sleep, and MEASURE it.** No stage →
    ~50 ms sleep per tick; stage-but-no-client (and no play/job) → `submit_last()` + ~100 ms sleep
    (never path-trace into the void). Verify the server process sits at ≤ ~1 core with GPU near-idle
    in both states. Also note: a CONNECTED client renders at full rate BY DESIGN — a forgotten open
    tab keeps the box hot for hours (measured ~16 cores × 4 h); pausing the stream on
    `document.hidden` is a worthwhile enhancement. (SPEC-UX "IDLE POWER".)
18. **The native folder dialog (📁 Browse → `/api/browse-folder`) must NOT crash the process on a
    second click.** Running the Tk dialog with a fresh `tk.Tk()` per request on a rotating
    `run_in_executor` worker thread trips the native `Tcl_AsyncDelete` abort on a 2nd Browse (twice in
    a row, or after cancelling the first) → the whole process dies and the viewport drops.
    Required: pin ALL Tk work to ONE dedicated long-lived thread with a single reused root, AND
    single-flight the Browse action (front-end `browseFolder._busy` shared across every 📁 button;
    back-end non-blocking lock returning `""` for a concurrent request — never queue a 2nd dialog).
    A native modal can't be driven by the browser/HTTP gates, so ship the **pytest regression** that
    asserts one-thread/one-root + concurrent-pick-returns-`""`. (stability-checklist item 18;
    SPEC-FUNCTIONAL `/api/browse-folder`.)

## ‼ THE CONTRACT IS A SPEC, AND IT IS EXECUTABLE — read this first
Two documents define exactly WHAT to build (the binding contract); the skills define HOW (recipes,
gotchas, architecture so you don't re-discover the traps). Throughout this prompt, `...\rebuild\X`
means the path `X` inside this bundle — substitute the absolute location of the `rebuild\` folder
you were handed for the `...` prefix:
- **`...\rebuild\skills\omniverse-dev-variant-presenter\references\SPEC-FUNCTIONAL.md`**
  — every `/api/*` endpoint's request/response/status + the WS `/events` event set + the file-output
  contract + the safety gates.
- **`...\rebuild\skills\omniverse-dev-variant-presenter\references\SPEC-UX.md`** — the browser/interaction
  contract: the **DOM-id contract** (you MUST expose these exact element ids), the tab-contextual dock,
  the live optics sliders + their value boxes + render effects, the pivot gizmo, timeline clip
  editing + scrub + live transport, the results dock player, project restore, help popover.

**The definition of "DONE" is TWO machine-checked gates, BOTH shipped to you. A build is NOT done until
BOTH are green.**

### Gate 1 — the HTTP grader (server + file contract)
A real grader is shipped at **`...\rebuild\acceptance\grade_http.py`** (+ `remote_mirror_probe.py`).
Run it against your own running server and iterate until:

> **`python ...\rebuild\acceptance\grade_http.py --url http://127.0.0.1:<port> --usd
> <local ConceptCar mirror>\Concept_Car.usd --render
> --json grade.json` reports every area `full` EXCEPT `area2`/`area3` (inherently HTTP-only, allowed
> `partial`). ZERO `fail`. ZERO `partial` outside area2/3.**

`<local ConceptCar mirror>` is the folder your own app mirrors the public ConceptCar stage into —
see **Stage data** under Environment for how to produce it and where it lands.

That includes `area5_files` (batch writes PNG files on disk), `area8` (post/cutsheet), `area4`
(turntable response `cameras[]` INCLUDES the rig camera), `area7` (projects list returns names
VERBATIM), `area9` (remote mirror), `area2_reload_survive`, `area5_guard` (409), `area6`,
`frontend_assets`, `G5_events_ws`, `G3`, and the GPU picks `area3_pickfocus`/`area4_pickpoint`.

> **Picks idle-throttle:** `area3_pickfocus`/`area4_pickpoint` drive a GPU pick on the render thread,
> which times out if the stream is idle-throttled with no client connected. KEEP A BROWSER CLIENT
> CONNECTED while you run `grade_http --render` (open the app in Chrome and leave it streaming — the
> same client Gate 2 uses), OR make your pick handler service requests even while the stream idles.
> `pick-focus` returns `{ok, distance|focus_distance, prim}` and `pick-point` returns
> `{ok, point|world, size, prim}` — the grader accepts either field name.

### Gate 2 — the REAL-BROWSER verifier (the interactive UX) — the hard one
A build-agnostic browser verifier is shipped at **`...\rebuild\acceptance\verify_browser.cjs`**.
It drives a REAL mouse/keyboard in headful Chrome and asserts EVERY `SPEC-UX.md [vb]` clause on the
RENDERED PIXELS + your DOM. **It is keyed on the SPEC-UX DOM-id contract — expose those exact ids or it
cannot drive your build.** Run it:

> `npm i puppeteer-core` (a system Google Chrome is required), then run it **TWICE — both lanes must
> print `0 failed`**:
> 1. **LOCAL lane:** `node ...\rebuild\acceptance\verify_browser.cjs http://127.0.0.1:<port>
>    <local ConceptCar mirror>\Concept_Car.usd`
> 2. **S3 lane** (the stage arg is the remote URL — the gate auto-extends its budgets):
>    `node ...\rebuild\acceptance\verify_browser.cjs http://127.0.0.1:<port>
>    https://omniverse-content-production.s3.us-west-2.amazonaws.com/Samples/Showcases/2023_2_1/ConceptCar/Concept_Car.usd`
>    The FIRST S3 open cold-mirrors ~11 GB (many minutes, with your `mirror_progress` overlay); later
>    opens hit the cache and are fast. Production usage is S3-FIRST, and classification, attach and
>    overlay-handoff can all pass the local lane and still fail against a remote stage — so a
>    local-only pass is not a pass.

It asserts (each of these is a behaviour a server-side grader cannot see, which is why it is driven
here through a real browser):
- viewport **left-drag ORBITS** the camera (pixels move) — the camera pose MUST be a LIVE per-frame
  `ovstage.Stage.write_attribute("omni:xform", ...)` fabric write (never the deprecated
  `Renderer.write_attribute`); **NEVER bake the pose into the composite** (baking kills
  orbit). Esc cancels an armed pick.
- each Display slider has a **live value box** that updates on a keyboard step (ISO/focal numeric,
  f-stop shows `"off"` at 0) AND a render effect: **ISO** low→high brightens (no auto-gain may override a
  manual ISO), **focal** wide→tele changes FOV, **f-stop** onchange posts `/api/display{f_stop}` and
  produces DoF.
- focus-pick fills `#disp-fd`; camera-select repopulates the sliders.
- a variant (Carpaint) chip click MOVES the car pixels (fast path).
- **pivot-pick → an interactive `#gizmo` appears** (`.handle[data-ax]` axis lines + center dot) +
  `#tt-pivot` updates → dragging an axis handle MOVES `#tt-pivot` → nudge buttons → `#tt-frames-s`
  readout → Create → the rig camera appears in `#camera-select`.
- resolution change → the video AUTO-RECONNECTS (no manual reload) + Grid W/H sync.
- a 1-perm Grid batch writes a PNG on disk + the estimate/guard text shows → the Results media plays
  **OVER the viewport** (where the live video renders), NOT in a separate dock below the video.
- timeline clip **select** (`.sel` + `#tl-del-clip` enables) / **drag-move** / **edge-resize** /
  **`.clip-var` ▾ change-variant** / **Delete**; **playhead drag-scrub** moves `#tl-playhead`+`#tl-playtime`
  and jumps the viewport to that clip's variant; **live Play** switches the car across a clip boundary;
  Space/Home/End keys; `#tl-loop` toggles; the three layout resize drags (`#panel-resize`,
  `#tl-resize`, `#tl-gutter-resize`).
- project save → appears in `#proj-list` VERBATIM → open RESTORES the selection + per-camera ISO; the
  track-view save round-trip; the help popover on hover; a collapsible Configure block toggles.

Run the grader, read the scorecard, fix what isn't right, re-run — that loop IS the task. Both gates
are build-agnostic: they only drive your `/api/*` + the DOM and read your output files, and they
reveal no reference implementation, so they can be re-run independently against a fresh server. Pass
them by building the real feature, not by special-casing the test.

## ‼ BUILD FRESH — write the whole app yourself
Create everything from scratch in `...\rebuild\build`: `uv venv` + your own `.venv`, write every
source file yourself. Do NOT copy/robocopy/hard-link/symlink another build's output (venv, `web\`,
`dev_variant_presenter\`, `node_modules`, mirror cache) into it.

## Read your skills (the HOW) after the spec
START at `...\rebuild\skills\omniverse-dev-variant-presenter\SKILL.md`, then siblings, then
`references\stability-checklist.md` (items 10–18), `seed-recap.md`, `production-parity-checklist.md`.
Working root = `...\rebuild\build`. Do NOT read the reference `dev_variant_presenter\`/`web\` or any
other build's sources.

## The highest-risk clauses (silent, expensive to get wrong — get these exactly right)
1. **Live camera pose** (orbit): drive the viewer camera from ovstream mouse input via a per-frame
   `ovstage.Stage.write_attribute("omni:xform", ...)` fabric write (through the app-owned
   `StageSession` / `ovstage.PathDictionary(stage)` — NEVER the deprecated `Renderer.write_attribute`),
   one 16-lane float64 matrix element via `make_dltensor` + `advance_write_floor`. Optics (ISO/focal/
   f-stop/focus) apply via **live `ovstage` scalar writes** (`focalLength`/`fStop`/`focusDistance`/
   `exposure` are schema attrs with defined fallbacks — write them directly, no reopen); `exposure:iso`
   is a custom attribute with no fallback, so it still needs the always-correct reopen path. The POSE
   must NEVER be baked into the composite — baking it kills viewport orbit outright, and the app still
   looks alive, so it is the single most expensive mistake in this list. (skills
   `usd-variant-live-switching`, stability-checklist.)
2. **Slider value boxes + render effects**: `#disp-iso`/`#disp-fl`/`#disp-fs` each have a sibling
   `#disp-*-v` span updated on `oninput`; `onchange` POSTs `/api/display`. No auto-gain may cancel a
   manual ISO. f-stop `"off"` at 0.
3. **Batch file-output** (`area5_files`): a single-frame job MUST write `out_dir/{label}.png` per
   permutation. `cv2.imwrite` returns `False` WITHOUT raising — check its bool return, count files.
4. **Remote/S3 "red car"** (`area9` + live): follow the PROVEN mirror recipe (`usd-remote-stage-mirror`).
5. **Projects list VERBATIM** (`area7`); **turntable rig camera in the response** (`area4`).
6. **Tab-contextual dock + LIVE playback transport** (SPEC-UX, `[vb]`): timeline strip ONLY on the
   Timeline tab; the Results still/clip plays **OVER the viewport** (the panel where the video renders),
   NOT a dock below the video; Play advances on a rAF wall-clock and switches the active
   variant/camera LIVE.
7. **Pick rectangle is NDC `[0,1]`, top-left origin — never pixels.** `Renderer.enqueue_pick_query`
   (ovrtx 0.4) takes normalized `[left,top,right,bottom]`; for a clicked pixel `(x,y)` in a `w×h`
   stream send `[x/w, y/h, (x+1)/w, (y+1)/h]`. Passing raw pixel integers raises `ValueError: invalid
   NDC rectangle` (the pre-0.4 pixel-rect habit is an easy carry-over). Prefer the pick hit's
   `worldPositionM` for focus distance when it comes back non-zero; fall back to the prim's world
   bbox (closest point / centre) when it doesn't. Turntable pivot picks may keep the bbox-centre
   convention either way.

## Don't regress the rest (all specified)
Serving contract (HTML asset base == static mount; every `<script>`/`<link>` 200). Explosion guard
SYNCHRONOUS in the handler (fast 409, not 504). `ovstream.initialize()` bootstrap-first (WinError 127).
Connect on `ready`; don't rebuild the Server on a normal connect. ovrtx caches a stage by FILE PATH →
unique composite paths. Async routes never block the loop; gate streaming on the `/events` WS count.
ConceptCar is cm → authored camera + warm ≥24, never a near-black first frame. Backdrops switch
re-renders sky/lighting. User USD never modified. Every JSON route returns 400 on malformed JSON.
Emit the full `/events` set (SPEC-FUNCTIONAL): `warmup, ready, stage_open, resolution, classified,
mirror_progress, batch_progress, batch_done, timeline_progress, timeline_done, camera_params,
framing_saved/skipped, focus_picked, error`. Implement ALL of SPEC-UX.md.

## Environment
Windows 11 + RTX. Python `>=3.11,<3.12` (ovrtx ships no cp312+ distribution). uv + a `.venv` in
`...\rebuild\build`. `uv pip install "ovrtx>=0.4,<0.5" "ovstage>=0.1,<0.2" "ovstream>=0.4,<0.5" usd-core warp-lang
numpy pillow "opencv-python<5" imageio-ffmpeg natsort fastapi "uvicorn[standard]" websockets httpx`
All of these resolve from PyPI, so no extra index needs configuring — BUT the install needs network
access to **`pypi.nvidia.com`** as well as `pypi.org`: the `ovrtx` package on PyPI is only a
`wheel-stub` sdist whose build backend downloads the real platform wheel from NVIDIA's index. If that
host is unreachable the install fails at build time with no obvious explanation. (`ovstage` and
`ovstream` publish real wheels to PyPI directly.)
The app OWNS the `ovstage.Stage` (see `references/seed-recap.md` "Render path") — `ovstage` is a
first-class dependency, not optional. Watchdog `run_server.ps1` uses `& $py ... *>> $log` and MUST set
`$ErrorActionPreference = "Continue"` — with `"Stop"`, the FIRST native stderr line (Warp/carb
warnings) terminates the watchdog loop under Windows PowerShell 5.1, so one crash leaves a
permanently dead app. Re-exec children via
`sys.executable`. For Gate 2: `npm i puppeteer-core` in `...\rebuild\build` (system Chrome at the
default path; or set the CHROME env var).

**Seed skill.** This bundle is the DOMAIN layer on top of NVIDIA's **`omniverse-realtime-viewer`**
seed skill (https://github.com/NVIDIA/skills), which owns the render/stream/camera/picking base and
the architectural non-negotiables — the orchestrator skill opens with "Read the seed skill FIRST".
If you cannot reach it, `...\rebuild\skills\omniverse-dev-variant-presenter\references\seed-recap.md`
in this package recaps its key recipes and is a sufficient fallback.

**Stage data.** No stage file ships in this bundle — you MIRROR one with the app itself, and that
mirror IS the `<local ConceptCar mirror>` both gates take as their local stage argument. Start your
server, `Open` the public ConceptCar URL
`https://omniverse-content-production.s3.us-west-2.amazonaws.com/Samples/Showcases/2023_2_1/ConceptCar/Concept_Car.usd`
once and let the mirror run to completion (~11 GB, many minutes — your `mirror_progress` overlay
reports it). Per the `usd-remote-stage-mirror` recipe the cache lands under the working root at
`data\_mirror\<url authority, with ':' → '_'>\<url path>` — here
`...\rebuild\build\data\_mirror\omniverse-content-production.s3.us-west-2.amazonaws.com\Samples\Showcases\2023_2_1\ConceptCar\Concept_Car.usd`
— with a sibling `<file>.mirror_complete` marker meaning the dependency closure is closed. (On
Windows, when that absolute path would blow past MAX_PATH the mirror also exposes it through a short
`C:\ovml\<hash>` junction; either path opens the same stage.) Point Gate 1 and Gate 2's LOCAL lane
at that mirrored `Concept_Car.usd`; Gate 2's S3 lane takes the URL above directly.

## Self-verify (sole GPU tenant), then STOP the server (watchdog FIRST, then the python proc)
1. `python run_tests.py` + `--gpu` (your own honest suite) and `node web/timeline-core.test.cjs`.
2. **Gate 1:** launch your server, run `grade_http.py ... --usd <local ConceptCar mirror>\Concept_Car.usd
   --render` — iterate until 0 fail / 0 non-design partial.
3. **Gate 2, LOCAL lane:** keep your server running (single-client — close any other browser/client),
   run the shipped `verify_browser.cjs` with that same mirrored `Concept_Car.usd` — iterate until
   `0 failed`.
4. **Gate 2, S3 lane:** run `verify_browser.cjs` again with the S3 URL as the stage arg (cold mirror
   the first time) — iterate until `0 failed`. This lane exercises the remote-stage paths (mirror,
   classification, attach, overlay handoff) that the local lane cannot reach; do not skip it.

## Deliverables in `...\rebuild\build\`
Full app + `.venv`; `pyproject.toml`; `README.md`; `run_tests.py`; **`NOTES.md`** with the final
`grade.json` rollup + BOTH `verify_browser` lane summaries (`N passed, 0 failed` each) pasted in +
honest gaps. **Write `NOTES.md` last, once all three rollups are green**, so it reports the final
state of the build rather than an intermediate one.
