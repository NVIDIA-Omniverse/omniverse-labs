# SPEC-FUNCTIONAL — the binding control-plane contract (the WHAT)

This is the **contract** your build MUST satisfy. The skills tell you HOW (recipes, gotchas,
architecture); this tells you exactly WHAT every endpoint does. **It is executable:** the shipped
acceptance suite (`acceptance/grade_http.py` + `acceptance/verify_browser.cjs` +
`acceptance/remote_mirror_probe.py`) checks these clauses. Each clause is tagged with the check that
verifies it, e.g. `[grade:area5_files]`. **"Done" = the acceptance bar at the bottom is GREEN** — not
"it looks complete," not "count returned 200." Where a clause says MUST, a build that violates it is
wrong even if everything else passes.

Convention: all requests/responses are JSON unless noted. Bind/advertise IPv4 `127.0.0.1` (never
`localhost`). Control port auto-picks (8080…); ovstream signal port 49100. The HTML's asset base MUST
match the static mount (`/` with all `/api/*`+`/events` registered first, assets at root) — every
`<script>`/`<link>` MUST serve 200 with a JS/CSS content-type. `[grade:frontend_assets]`

---

## Core lifecycle

### `GET /api/config` → `{signal_port, stream_resolution, default_stage?, ...}`
Answers as soon as the HTTP server is up (BEFORE the renderer warms). MUST include `signal_port` and
`stream_resolution`. `[grade:contract]`

### `GET /events` (WebSocket)
MUST upgrade to HTTP 101 (needs `uvicorn[standard]`/`websockets` — a build that serves HTTP but fails
the WS handshake is broken). `[grade:G5_events_ws]` MUST emit these event types over the session (the frontend
REACTS to each — omitting one silently breaks the UX the verifier then drives): `warmup`, `ready`,
`stage_open`, `resolution {width,height}`, `classified {fast_sets,swatches}`, `mirror_progress
{downloaded,file}`, `batch_progress`, `batch_done`, `timeline_progress {frame,total}`, `timeline_done`,
`camera_params {params}`, `framing_saved`/`framing_skipped`, `focus_picked {distance}`, `error {message}`.
(`selection`, `connection`, `stream_restarted` may also be emitted but the reference client does not depend
on them.) `[parity; verify_browser]`

### `POST /api/open {usd_path, [resolution, selections, display, looks, xforms, camera]}` → stage info
Opens a local path OR an `http(s)`/S3 URL (mirrors remote first, emitting `mirror_progress`). Response
MUST contain `variant_sets: [{set_name, prim_path, variants:[...], current}]` and
`cameras: [{name, path, animated}]`, plus `source_url`, `up_axis`, `meters_per_unit`, frame span.
`[grade:open]`
- For a remote URL, `source_url` MUST equal the requested URL verbatim, and the user's source files
  MUST be byte-unchanged. `[grade:area9]`
- MUST mirror the FULL closure incl. `.mdl`-internal textures/modules so materials resolve (no red
  car). Verify on the REAL S3 ConceptCar. `[parity "red car"; live]`
- Restore path: when `selections`/`display`/`looks`/`xforms`/`camera` are passed, the first warmed
  frame MUST reflect them. `[parity projects]`

### `GET /api/stage` → live state
MUST return `{open, ready, selection:[{set_name,variant,prim_path}], camera, look_cams, framed_cams,
turntable, stream_resolution, cameras?, variant_sets?}`. `ready` flips true only on a real warmed
frame. Reply-queue reads MUST have a timeout so a busy render thread can't hang this. `[grade:open,
area2_reload_survive; parity]`

---

## Variants

### `POST /api/variant {selections:[{set_name,variant,prim_path}]}` → `{ok, path:"fast"|"reload"}`
Applies the selection. Fast path (shader-input-classified sets) writes attributes with no reopen;
reload path rebuilds a unique composite + re-applies fast overrides after `ovstage.population.open_usd`
(the app's `StageSession.populate_usd`).
`[grade:area2]` (HTTP accept; the **pixel move** is asserted by `verify_browser`)
- **‼ EVERY switch — fast OR reload — MUST be VISIBLE in the viewport IMMEDIATELY, with no other
  action.** The fast path must push its writes through **`ovstage.Stage.write_attribute`** (via a
  `StageSession`/`ovstage.PathDictionary(stage)` query on the classified shader inputs, advancing
  the write floor) — NEVER the deprecated `Renderer.write_attribute` — **and then `renderer.reset()`**
  (clear accumulation so the live loop re-converges on the new look). A fast path that edits USD/pxr
  state without an ovstage write, or writes without resetting accumulation, shows NOTHING until the
  next unrelated reopen — the user clicks Carpaint and the car doesn't change until they touch a
  Backdrop or the turntable. This is hard-gated: the verifier
  waits for `classified`, then switches a **fast** set twice and requires a pixel change within
  seconds each time.
- **The flip-to-fast regression:** before classification finishes every switch reloads (visible);
  after `classified` flips a set to the fast path, its switches MUST STILL be visible. Do not
  self-test only in the pre-classified window.
- **Timeline scrub/play/edit apply state through this same endpoint** — material/look (fast) sets
  MUST update the viewport from the timeline exactly like geometry/environment (reload) sets.
- **‼ The `classified {fast_sets,swatches}` event MUST be emitted within a DEADLINE (~120 s of ready) on
  EVERY stage — including a MIRRORED S3 stage — and the classifier must NEVER hang silently.** A build's
  classifier silently hung forever on the 11 GB mirrored ConceptCar: no event, no error, fast path never
  engaged, and every switch (chips AND timeline clips) was a multi-second reload that read as "variants
  don't work." Requirements: classify off the render thread; time-bound it (if a set can't classify in
  budget, classify it reload-only and move on); on failure/timeout, STILL emit `classified` (possibly
  with fewer/empty `fast_sets`) plus an `error` event — the UI must never be left ambiguous. ConceptCar
  (local or mirrored) classifies 8 shader-input sets incl. Carpaint. `[vb: classified arrives ≤120 s,
  on the local AND the mirrored-S3 stage]`
- **MUST survive a reload:** switch a fast set, then a reload set → `/api/stage.selection` MUST still
  carry BOTH selections. `[grade:area2_reload_survive]`
- Backdrops/environment switches MUST re-render sky/lighting (whole-frame pixels move). `[parity; live]`

---

## Cameras + optics + picking

### `POST /api/camera/snap {camera_path, [at_s, reset]}` → 200
Snaps the free camera to the authored camera's pose; emits `camera_params` with that camera's optics.
`reset:true` clears optics to defaults (posting explicit nulls). `[grade:area3]`
### `POST /api/display {iso, focal_length, f_stop, focus_distance, [resolution]}` → 200
Applies optics; a `resolution` change rebuilds the stream at that shape +
`verticalAperture = horizontalAperture*h/w`. `[grade:area3]`
### `POST /api/camera/look-at {target, radius, height}` → 200 · `POST /api/camera/save-framing` → `{ok}`
### `GET /api/camera-pose` → `{eye,right,up,m}` · `POST /api/camera-state {looks,xforms}` → 200
Live pose for the gizmo; push-back of per-camera optics/framings for project restore. `[parity]`
### `POST /api/pick-focus {nx,ny}` → `{focus_distance:<float>, prim_path?}`
MUST return a numeric `focus_distance`. `[grade:area3_pickfocus]`
### `POST /api/pick-point {nx,ny}` → `{ok:true, world:[x,y,z], size:[dx,dy,dz], prim_path}` on a HIT | `{ok:false}` on a MISS
MUST return a world point **ON the picked geometry**: resolve the hit prim from the render PICK var
(`hitCount>0`). `enqueue_pick_query` takes an NDC `[0,1]` top-left rect, NOT pixels — with that fixed,
`worldPositionM` returns a real non-zero position on a hit; the turntable pivot MAY use it directly, or
keep the bbox-centre convention (return that prim's world-AABB midpoint via
`BBoxCache(Default,[default_,render],useExtentsHint=True)` on the live composite; `size` = that AABB
extent) — either satisfies the contract. **On a MISS (`hitCount==0` / empty bbox) it MUST return `{ok:false}`
and place NO pivot — NEVER substitute the orbit target, a hardcoded point (e.g. `[0,1,0]`), or any
click-independent fallback** (masking a miss as `ok:true` with an arbitrary point lands the
gizmo in empty space). Two different on-car clicks MUST return different points.
`[grade:area4_pickpoint; verify_browser]`
### `POST /api/project {points}` → `{screen:[[x,y,visible],...]}` · `POST /api/probe-occlusion {point,nx,ny}` → `{occluded:bool}`
For the turntable gizmo re-projection + occluded shading. `[parity; verify_browser]`

---

## Optics + camera — how they MUST AFFECT THE RENDER (the dead-slider trap)
Returning 200 from `/api/display` is NOT enough — the change MUST be visible. Sliders that post 200
and move nothing are the classic failure here, and they are gated on pixels. Contract:
- **ISO** authored as `exposure:iso` on the viewer `UsdGeomCamera` — a custom attribute with no
  defined USD fallback, so clearing it back to "unauthored" still needs a reopen (applied via a
  reopen). Changing ISO
  MUST change frame brightness. **No auto-exposure / auto-gain may OVERRIDE a manual ISO** — if you add an
  auto-gain so the first frame isn't near-black, it must not cancel the user's ISO (e.g. only auto-gain
  until the user touches ISO, or fold ISO into the gain). `[verify_browser px: low vs high ISO → mean luma differs]`
- **f_stop** authored via `CreateFStopAttr` (`f_stop ≤ 0` ⇒ DoF OFF, attribute cleared). f_stop > 0 MUST
  produce **clearly visible** depth-of-field blur (with `focus_distance`, or the live orbit distance if
  unset). `[px: off vs f/2 background sharpness differs]`
  - ‼ **DoF STRENGTH IS PHYSICAL — fStop ALONE is not enough.** ovrtx's DoF blur radius scales with the
    physical aperture **diameter = focalLength / fStop** measured against the camera's **sensor aperture**.
    The viewer `UsdGeomCamera` MUST author `horizontalAperture` + `verticalAperture` (≈ **20.955 mm**
    full-frame, or the source camera's, with `verticalAperture = horizontalAperture * h/w`) AND
    `focalLength` (mm) alongside `fStop` (+ `focusDistance` in **stage/world units**). Without a correct
    aperture+focalLength on the camera, `fStop` produces **negligible blur** (fStop alone measures a
    ~1% centre-edge drop; authoring aperture+focal drops ~97%). **Do NOT fake
    DoF with a "DOF_BOOST"/aperture-multiplier hack — it does not render as blur.** A near `focus_distance`
    on a far subject at f/2–f/4 must collapse the subject's high-frequency detail. `[verify_browser px:
    reset optics → focus near → off vs f/4 → centre-region edge energy drops ≥30%]`
- **focal_length** via `CreateFocalLengthAttr` (mm) — changes FOV. **focus_distance** via
  `CreateFocusDistanceAttr`. `[px]`
- **Optics changes (focal_length/f_stop/focus_distance/exposure) apply via a LIVE `ovstage` scalar
  write** — `focalLength`/`fStop`/`focusDistance`/`exposure` are `UsdGeomCamera` schema attrs with
  defined fallback values, so `ovstage.Stage.write_attribute` reproduces what a reopen would author,
  with no reopen needed (`exposure:iso` is the one exception — see above, it still reopens); **camera
  POSE changes apply via a LIVE per-frame `ovstage.Stage.write_attribute("omni:xform", ...)` fabric
  write** — NEVER bake the pose into the composite (baking kills live viewport orbit), and NEVER call
  the deprecated `Renderer.write_attribute`. Driving the camera from viewport mouse drag = ovstream
  input event → orbit controller → the `ovstage` fabric write (one 16-lane float64 matrix element via
  `make_dltensor` + `AttributeSemantic.MATRIX`, row-vector layout). `[verify_browser: drag viewport →
  pose/pixels move]`
- ‼ **THE CAMERA TRANSFORM IS ALWAYS READ *FROM THE COMPOSED STAGE* — NEVER synthesized from angles +
  heuristics.** (Rebuilding the camera from spherical `(yaw,pitch,distance,up_axis)` +
  a bbox/name heuristic breaks THREE things at once: the orbit pivot collapses to the world origin
  `(0,0,0)`, the animated turntable rig can't play, and the car tilts/sits wrong vs the environment.)
  Contract:
  - On open/snap, seed the free camera from the authored camera's **composed world matrix**
    (`UsdGeom.Xformable(camPrim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())`) and seed the
    orbit controller from THAT matrix (`snap_to`). The controller is constructed with the source stage's
    true `up_axis` (`UsdGeom.GetStageUpAxis`, default `"Y"`).
  - **The orbit PIVOT** = `eye + forward * focusDistance` where `focusDistance` is read from the camera
    (`UsdGeom.Camera.GetFocusDistanceAttr().Get()`); if unset, project the asset bbox centre onto the
    forward ray. **NEVER default the pivot to `(0,0,0)`**, and never gate it on size/name heuristics that
    can reject every prim and collapse to the origin. `[verify_browser: orbit → recovered pivot is inside
    the car AABB, >50 units from origin]`
  - **Camera-less stages MUST be auto-framed** up-axis-aware (a three-quarter view of the default-prim
    bounds), never left at a bare controller default. `[grade/verify on a camera-less fixture]`
  - **Defense-in-depth:** the composer SHOULD also set the composite's up-axis to the source's
    (`UsdGeom.SetStageUpAxis(composite, GetStageUpAxis(user))`).

## Additional routes (complete the surface)
`POST /api/render-mode {quality}` (reopen w/ new `omni:rtx:rendermode`) · `POST /api/playback
{playing,fps}` — **a LIVE wall-clock camera animator, no recompose**: on `playing:true` record
`t0=monotonic()`+`start/end/fps`, and on EVERY render tick while playing compute
`tc = start + ((monotonic()-t0)*fps) % span`, evaluate the active animated camera's world transform
`UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(TimeCode(tc))` on the live composite (which MUST
sublayer the turntable sidecar), and write that 4×4 onto the viewer camera's `omni:xform` via
`ovstage.Stage.write_attribute` (never the deprecated `Renderer.write_attribute`). Do NOT rely on
`ovstage.population.update_from_usd_time` (confirmed frozen at frame 0 on this ovrtx 0.4 + ovstage 0.1
build — a remaining library gap, not a 0.3-only quirk) and do NOT just set a `playing` flag while
the render path keeps writing the static orbit matrix (that's the "Preview does nothing" bug). `playing:false`
restores framing; keep stepping headless so a gate sees motion. `[verify_browser: Preview → eye orbits]`
· `POST /api/camera-state {looks,xforms}` (project
restore; reopen) · `POST /api/camera/look-at {target,radius,height}` · `POST /api/probe-occlusion
{point,nx,ny}`→`{occluded}` · `POST /api/post/video {out_dir,fps}` · `POST /api/post/compress
{video_path}`→`{path}` · `POST /api/stream/restart`→`{ok,rebuilt}` (rebuild ovstream Server only — NOT
the global context) · `POST /api/restart` (exit 43 → watchdog relaunch) · `POST /api/browse-folder`→`{path}`
(native Tk dialog — see the pinned-thread + single-flight requirement at §`/api/browse-folder`).
All JSON routes return 400 on malformed body.

---

## Grid / batch  (see skill `ovrtx-grid-batch-render`)

### `POST /api/batch {job}` → `{count}`  (HTTP returns the PLAN; the render runs async)
`job = {mode:"one_at_a_time"|"full_cartesian"|"curated", base_selection, included:{set:[variants]},
cameras, quality:{mode,samples_per_pixel,resolution,...}, frame_mode:"single"|"animation_range",
out_dir, confirm, [frame_start,end,step,curated]}`.
- `count` MUST equal: one_at_a_time = Σ|variants|; full_cartesian = Π|variants|; curated = len(curated).
  `[grade:area5_oaat, area5_cart]`
- **Explosion guard (SYNCHRONOUS in the handler):** if `count > 500` and not `confirm`, return **HTTP
  409** and queue NOTHING; `confirm:true` then runs it. MUST NOT 504. `[grade:area5_guard]`
- **★ FILE-OUTPUT CONTRACT (a silent trap):** a `frame_mode:"single"` job MUST write
  `out_dir/{label}.png` at the TOP LEVEL, one per permutation, where `label = permutation_name` =
  `{set}-{variant}` joined by `_` (folder-safed). A returned `count` is the PLAN — **the PNG files on
  disk are the proof.** `cv2.imwrite` returns `False` WITHOUT raising, so a broken save writes nothing
  silently: check its bool return (`raise` on False) and count files written. `[grade:area5_files]`
- `animation_range` writes `out_dir/{label}/{frame:04d}.png` + assembles `out_dir/{label}.mp4`.
- `POST /api/batch/cancel` → flips a bool the loop checks between permutations. `[parity]`

---

## Timeline  (see skill `ovrtx-timeline-nle`)

### `POST /api/timeline/render {timeline, quality, out_dir}` → `{frames:<int>}`
`frames` MUST be computed IN THE HANDLER (`duration_s*fps`, e.g. 4 s @ 2 fps → 8) and returned
synchronously; the MP4 render is fire-and-forget on the render thread. `[grade:area6]`
- MUST write `out_dir/timeline.mp4` (assert the file on disk, not just `frames`). On completion emit
  `timeline_done {path}`. `POST /api/timeline/cancel` cancels. `[parity; verify_browser]`
- Project-scoped track views: `GET /api/timelines?project=` · `POST /api/timelines/save` ·
  `GET /api/timelines/load` · `POST /api/timelines/delete`. `[parity]`

---

## Turntable  (see skill `ovrtx-turntable-camera`)

### `POST /api/turntable {pivot, radius, height, frames, fps, focal_length, start_deg, [camera_world]}` → stage info
Authors the rig in a SIDECAR layer (user source byte-unchanged), rescans, and the response MUST be the
refreshed stage info **whose `cameras[]` INCLUDES the rig camera** (name/path containing "turntable").
`[grade:area4]` — *(a `200` whose `cameras[]` omits the rig camera is NOT full.)*
- Rig metadata stored as a JSON **string** in `customLayerData`; `rig_info` MUST be exception-proof so
  a malformed sidecar can't 500 `/api/stage`. `[parity; stability]`
### `POST /api/turntable/remove` → deletes rig+sidecar and reopens. `[parity]`

---

## Projects  (see skill `ovrtx-projects-session`)

### `POST /api/projects/save {name, base_selection, display, camera, [looks,xforms,timeline]}` → `{ok}`
Bundles base selection + display/resolution + per-camera looks + framing xforms + camera + working
timeline + track views.
### `GET /api/projects` → `{projects:[name,...]}`
MUST list every saved project **by its name VERBATIM** (do NOT normalize/strip characters — a name
saved as `_x` MUST appear as `_x`). `[grade:area7]` — *(stripping a leading `_`, so the
saved name isn't found in the list, is NOT full.)*
### `GET /api/projects/load?name=` → the full record · `POST /api/projects/delete {name}` → `{ok}`
Open MUST RESTORE ALL of it (seed looks/xforms/display/camera into the reopen + re-apply the timeline),
not just base-selection+camera. `[grade:area7; parity]`

---

## Results / media / post  (see skill `ovrtx-post-cutsheet`)

### `GET /api/results?dir=` → `{permutations:[{name,kind:still|sequence|video,path,frames?,frame_count?}]}`
Enumerates top-level stills, frame-folder sequences, and videos. `[grade:area8]`
### `POST /api/post/overlay {out_dir}` → `{count}` · `POST /api/post/cutsheet {out_dir}` → `{path}`
Overlay burns labels onto stills (originals MUST stay byte-unchanged); cutsheet writes a labeled
contact sheet to disk (`path` MUST be an existing file). `[grade:area8]`
### `GET /api/video?path=` → MP4 (FileResponse) · `GET /api/frame?path=` → PNG · `POST /api/browse-folder` → `{path}` (native Tk dialog)
`[parity; verify_browser]`
- **`/api/browse-folder` MUST be crash-safe under repeated/concurrent use.** Tkinter/Tcl is not
  thread-safe: opening the dialog with a fresh `tk.Tk()` per request on a rotating `run_in_executor`
  worker thread lets a 2nd Browse (twice in a row, or after cancelling the first) trip the native
  `Tcl_AsyncDelete` abort → the whole process dies and the viewport drops. Required: **pin ALL Tk work
  to ONE dedicated, long-lived thread with a single reused, never-destroyed root** (marshal each
  `askdirectory` onto it via a queue; the route hands off through `run_in_executor` so the event loop
  never blocks, but the Tk calls all run on the pinned thread), and **single-flight the request** — a
  concurrent `/api/browse-folder` returns `{path:""}` (cancel) rather than queuing a 2nd dialog.
  `[gated by a required pytest regression — a native modal can't be driven by verify_browser; stability-checklist item 18]`

---

## Recovery
`POST /api/restart` (process restart, exit 43 → watchdog relaunch) · `POST /api/stream/restart`
(rebuild ovstream `Server`, blocking until rebuilt). `[parity stream self-heal]`

---

## Safety GATES (non-negotiable)
- **G3 — user USD never modified:** the source file's sha256 MUST be unchanged across the whole
  session (turntable/edits go to sidecars). `[grade:G3]`
- **G1/G2/G4** — server owns ovrtx; no client-side 3D; ONE render thread is the sole stepper/owner.
  `[code review + verify_browser]`

---

## ★ ACCEPTANCE BAR — this is the definition of "done"
Run the SHIPPED, build-agnostic grader against your running server and a LOCAL ConceptCar, and meet
ALL of:
1. **`python acceptance/grade_http.py --url http://127.0.0.1:<port> --usd <local ConceptCar> --render
   --json out.json` → every area `full` EXCEPT `area2` and `area3`** (those two are inherently
   HTTP-only "pixel change confirmed live" and are allowed `partial`). **ZERO `fail`. ZERO `partial`
   outside area2/3.** In particular `area5_files`, `area8`, `area4`, `area7`, `area9`,
   `area2_reload_survive`, `area5_guard`, `area6`, `frontend_assets`, `G5_events_ws`, `G3` MUST ALL be
   `full`. This grader IS the acceptance instrument — it is shipped to you ON PURPOSE; passing it is
   the contract, not cheating. (It only drives `/api/*` + reads files; it does not reveal any
   reference implementation, so it can be re-run independently against a fresh server.)
2. **Browser/pixel layer:** the SHIPPED `acceptance/verify_browser.cjs` (headful Chrome, NO
   `--use-fake-ui-for-media-stream`; needs `puppeteer-core` + a system Chrome) drives a REAL
   mouse/keyboard against your running server and asserts every `SPEC-UX.md` item tagged `[vb]` —
   media pair (`videoWidth>0`, frames decoded > 0), non-black pixels, viewport orbit-drag moves pixels,
   each slider's live value box + render effect (ISO→luma, focal→FOV, f-stop→DoF), focus/pivot pick →
   gizmo → gizmo drag, timeline clip select/drag/resize/▾-edit/delete, playhead scrub, live Play moves
   the car at a clip boundary, resolution-change auto-reconnect, Results dock player, project restore,
   help popover, the tab-contextual dock (timeline strip hidden on Configure). It is keyed on the
   `SPEC-UX.md` DOM ids + rendered pixels (never on `/api` field names), so it is build-agnostic. It
   takes the STAGE as its second argument and MUST be run in **BOTH lanes — each printing `0 failed`**
   (see `PROMPT.md`): the **LOCAL lane**
   `node acceptance/verify_browser.cjs http://127.0.0.1:<port> <local ConceptCar mirror>\Concept_Car.usd`,
   then the **S3 lane** with the remote stage URL as that argument (the gate auto-extends its budgets;
   the first open cold-mirrors ~11 GB). A local-only pass is NOT a pass — production usage is S3-first,
   and classification, attach and overlay-handoff can all pass locally and still fail on a mirrored
   stage. Both lanes run this exact file; the app is not done until the interactive UX actually works.
3. Every other `SPEC-UX.md` item present (gizmo, browse dialog, help popover, badges) — self-attest
   with concrete evidence in NOTES.md.

If any gradeable area is below `full` (or any `[vb]` check fails), the build is NOT done — fix it and
re-run. **Do not declare done on a partial.** THIS SAME `grade_http.py` can be re-run independently
against a fresh server; a build that passes it here passes there.
