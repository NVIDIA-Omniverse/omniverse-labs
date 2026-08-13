# SPEC-UX — the binding interaction contract (the WHAT, client side) — EXHAUSTIVE

The UX half of the contract. `SPEC-FUNCTIONAL.md` pins the server; this pins the browser app, in
full. **"Self-attest" is abolished** — a feature that isn't gated is a feature that ships broken.
Dead sliders, dead camera-drag, dead timeline editing and a dead gizmo all answer 200 to an HTTP
grader, so only a real browser can tell them from working ones. **Every clause here is checkable, and the
`[vb]` clauses MUST be exercised by `verify_browser.cjs` driving a REAL mouse/keyboard in headful
Chrome** — not by reading the DOM string, not by the builder's word. `[dom]` = assert via DOM
state/structure. `[px]` = assert by sampling rendered viewport pixels.

## DOM CONTRACT — use these exact element ids (so the gate is build-agnostic)
A build MUST expose these ids so the shipped browser verifier can drive any compliant build. (Names
taken from the reference app.)

> **These ids are a TEST SEAM, not a UI design prescription.** They exist for exactly one reason: so
> `acceptance/verify_browser.cjs` can drive ANY compliant build with the same script, keyed on the DOM
> and rendered pixels rather than on one implementation's internals. They say nothing about how the UI
> should look, how elements should be nested, what framework to use, or how components should be
> factored — only what the gate must be able to grab. If you rename or restructure these ids, the gate
> must be updated in lockstep or it can no longer drive the app.

Header: `#usd-path`, `#open-btn`, `#status`, `#overlay`. Viewport:
`#remote-video`, `#pick-overlay`, `#gizmo`, `#hint`. Tabs: `.tab[data-pane="configure|grid|timeline|results"]`,
`.pane`. Display: `#camera-select`, `#disp-iso`+`#disp-iso-v`, `#disp-fl`+`#disp-fl-v`,
`#disp-fs`+`#disp-fs-v`, `#disp-fd`, `#disp-pick`, `#disp-res`, `#disp-save-framing`, `#disp-reset`,
`.mode-btn[data-mode]`. Variants: `.vcard`, `.chip`. Turntable: `#tt-pick`, `#tt-pivot`,
`#tt-step`, `#tt-frames`, `#tt-frames-s`, `.tt-nudge button[data-ax][data-d]`, `#tt-add`,
`#tt-preview`, `#tt-remove`. Grid: `.grid-mode-btn`, `#grid-w/#grid-h/#grid-spp`, `#grid-anim`,
`#grid-fstart/#grid-fend/#grid-fstep`, `#grid-cameras`, `#grid-sets`, `#grid-estimate`, `#grid-out`,
`#grid-browse`, `#grid-render`, `#grid-cancel`, `#grid-bar`, `#grid-status`. Timeline: `#timeline-strip`,
`#tl-playtime`, `#tl-dur`, `#tl-mode-stack/#tl-mode-playhead`, `#tl-slideshow/#tl-mixer/#tl-clear`,
`#tl-del-clip`, `#tl-clip`, `#tl-fps`, `#tl-rulerbar`, `#tl-playhead`, `#tl-scroll`, `#tl-gutter-resize`,
`.tl-track`, `.tl-add-clip`, `.tl-clip` (+ `.rs` resize handle + `.clip-var` dropdown + `.sel` selected),
transport `#tl-to-start/#tl-step-back/#tl-play/#tl-step-fwd/#tl-to-end/#tl-loop`, `#tl-render/#tl-cancel/#tl-bar/#tl-status/#tl-out/#tl-browse`. Results: `#results-dir`, `#results-browse`,
`#results-refresh`, `#results-select`, `#results-slider`, `#results-frame-label`, `#results-img`,
`#results-video`, `#post-overlay`, `#post-cutsheet`. Projects: `#proj-name`, `#proj-save`, `#proj-list`,
`#proj-open`, `#proj-del`, `#proj-msg`, track-views `#tlv-*`. Help: `#help-pop`, `[data-help]`, `.info`.

Additional ids the verifier keys on (readouts, containers, sub-elements — expose these too):
Variants: `#variant-cards` (root), `.vcard`+`.name`(set name)+`.prim`, `.chip.on` (selected), `.swatch` (color dot),
`#vset-count`. Grid: `#grid-count` (total renders badge), `.grid-sets .vcard .check.setinc input` (per-set include),
`#grid-frame-range`. Turntable: `#tt-tools` (shown after a pivot pick), `#tt-msg`, gizmo handles
`#gizmo .handle[data-ax="0|1|2"]` (axis lines) + `#gizmo .handle.dot[data-ax="screen"]` (center). Timeline:
`#tl-tracks` (rows root), `#tl-ruler`, `#tl-transport` (the scrub-excluded gutter holding the transport buttons),
`#tl-resize` (strip top-edge resize), per row `.tl-track-label`+`.tl-hide`+`.name`+`select`(track variant)+`.tl-add-clip`,
per clip `.clab`(label)+`.caret`+`.clip-var-wrap`+`.sw`(swatch). Results: `#results-frame-controls` (multi-frame slider
container), `#results-empty`, `#post-log`. Projects: `#panel-resize` (panel resize). Configure: `.block-caret`
(collapsible toggle). The reference app exposes ALL of these verbatim — a compliant build MUST too.

---

## Viewport / live camera  (easy to ship DEAD — top priority)
- Single WebRTC `<video id=remote-video muted>`; call `.play()` on attach. `[vb: videoWidth>0, decoded>0]`
- Well-lit, never near-black. `[px: center-patch mean ≫ 0]`
- **Left-drag on the viewport ORBITS the camera; right/middle-drag PANS; wheel DOLLIES — LIVE.** This is a
  per-frame `ovstage.Stage.write_attribute("omni:xform", ...)` fabric write (through the app's
  `StageSession`/`ovstage.PathDictionary` — never the deprecated `Renderer.write_attribute`) driven by
  ovstream input events → the orbit controller → the viewer camera. The camera pose MUST stay a live
  fabric write — **do NOT bake the pose into the composite** (baking silently kills orbit). The orbit MUST rotate around the
  authored **pivot on the asset** (camera look-at / focus point) — **NOT the world origin** (reconstructing
  the camera from spherical angles + a bbox heuristic ends up orbiting `(0,0,0)`). `[vb: drag the
  video, assert pixels change AND the pivot recovered from before/after camera-pose is inside the car
  AABB / >50 units from origin — not the world centre]`
- **‼ Orbit DIRECTION convention (a sign slip inverts ALL controls at once — it is pinned):** with the
  orbit state as azimuth/elevation about the pivot (Y-up: eye offset `[cosE·sinA, sinE, cosE·cosA]`,
  azimuth measured from +Z toward +X), per mouse-move deltas in stream pixels:
  `azimuth -= Δx * 0.005` and `elevation += Δy * 0.005` (elevation clamped ±~85°). Observable:
  **dragging RIGHT decreases the eye's azimuth about the pivot; dragging DOWN raises the eye**
  (you end up looking more down onto the car). Pan follows the drag (content moves WITH the cursor);
  wheel-forward dollies IN. `[vb: rightward drag → the eye azimuth about the recovered pivot DECREASES
  (sign check via /api/camera-pose before/after)]`
- Esc cancels an armed pick. `[vb]`

## Tabs + the tab-contextual dock
- Tabs Configure/Grid/Timeline/Results; one `.pane.active` at a time. `[vb: click each, pane toggles]`
- **‼ Tab labels must be READABLE in every state — especially `.tab.active`.** Styling the
  active tab as a solid NVIDIA-green block hides its own label (foreground ≈ background). Text color vs
  background-color must clear a real contrast ratio (≥ 3:1 minimum; aim 4.5:1) for active AND inactive
  tabs — e.g. near-black text on the green active tab, or green text on the dark bar. `[vb: computed
  color/background contrast on every .tab, active + inactive]`
- **Tab-contextual surfaces:** (a) the **dock BELOW the viewport** (`#timeline-strip`) shows the Timeline
  strip ONLY on the Timeline tab — empty/hidden on Configure/Grid/Results. (b) the **Results media takes
  over the MAIN VIEWPORT PANEL** (NOT a dock below): `#results-img`/`#results-video` are children of the
  `.viewport` section, OVERLAYING `#remote-video`, shown (`.show`) ONLY on the Results tab. So selecting a
  render replaces the live viewport with the rendered still/video IN PLACE — it does NOT play in a separate
  box below the viewport (a dock-below player is the common wrong reading of this clause: results media
  belongs IN the viewport panel).
  `[vb: strip hidden on Configure, shown on Timeline; on Results, the selected `#results-img`/`#results-video`
  is `.show` AND its bounding box overlaps `#remote-video`'s (i.e. it's in the viewport, not below it)]`
- Collapsible Configure blocks (caret ▾/▸, persist to localStorage). `[dom]`

## Configure — Display optics  (easy to ship with missing readouts + dead ISO/f-stop)
For EACH of ISO (`#disp-iso`+`#disp-iso-v`), Focal-mm (`#disp-fl`+`#disp-fl-v`), f-stop
(`#disp-fs`+`#disp-fs-v`):
- A **live numeric value box** sits next to the slider and updates on `oninput` AS YOU DRAG (f-stop shows
  `"off"` at 0, else 1-decimal). `[vb: drag each slider, assert its value box text changes live]`
- On `onchange` it POSTs `/api/display`. The change has a VISIBLE RENDER EFFECT: `[px]`
  - **ISO → frame brightness changes** (no auto-gain/auto-exposure may override it — if you add an
    auto-gain to avoid near-black, it MUST NOT cancel a manual ISO change). `[px: ISO low vs high → mean luma differs]`
  - **Focal mm → FOV changes** (14=wide, 200=tele). `[px: framing changes]`
  - **f-stop → depth-of-field blur changes** (0=off/all-sharp; small f-stop = shallow DoF). `[px: background sharpness differs off vs f/2]`
- Focus: `#disp-fd` number ("auto" placeholder); `#disp-pick` arms focus-pick → click viewport →
  `/api/pick-focus` → `#disp-fd` fills + plane moves. `[vb]`
- Resolution `#disp-res` (16:9 720/1080 · 1:1 1080² · 9:16 1080×1920 · 4:3 1440×1080 · 2.39:1 1920×804):
  on change, stream rebuilds at that shape, Grid W/H sync, AND **the client AUTO-RECONNECTS so video
  returns within ~15 s with NO manual reload** (the easy failure is a permanently black video). `[vb: change res, assert decoded
  frames resume without reload]`
- `#camera-select`: per-camera badges ↻ animated · ◆ saved framing · ✱ optics (from `look_cams`/`framed_cams`),
  label `name (path)`; switching snaps the camera + repopulates the sliders from `camera_params`. `[vb]`
- `#disp-save-framing` → "Framing saved" overlay; `#disp-reset` → sliders to defaults + clears optics. `[vb]`
- `.mode-btn[data-mode=RealTimePathTracing|PathTracing]` toggles render mode. `[dom + px]`
- `#stage-info`: human units (m/cm/mm/in/ft) + fps + frame count. `[dom]`

## Configure — Variants
- `.vcard` per set (name + prim path); `.chip` per variant with a classifier `.swatch` color dot;
  clicking a chip sets `.on` + POSTs `/api/variant` and **the car changes in the viewport IMMEDIATELY**
  (fast path, no reload flicker for shader sets). **This must hold AFTER classification completes** —
  a fast switch that writes without resetting accumulation applies SILENTLY (nothing changes until the
  user touches a Backdrop or the turntable forces a reopen): the fast path must write to the renderer AND reset
  accumulation (SPEC-FUNCTIONAL `/api/variant`). `[vb: wait for the `classified` event, pick a set from
  its `fast_sets` (Carpaint), click a chip → car pixels move within seconds; click a SECOND variant on
  the same set → pixels move again]`
- Backdrops/env switch re-renders sky/lighting (whole-frame change). `[px]`

## Turntable + pick gizmo  (pick is easy to ship DEAD)
- `#tt-pick` arms pivot-pick (button `.armed`, `#pick-overlay` shows) → click viewport → `/api/pick-point`
  → **an interactive 3D `#gizmo` appears** (X red / Y green / Z blue draggable axis lines + white center
  dot), re-projected via `/api/project`+`/api/camera-pose`, occluded shading via `/api/probe-occlusion`.
  The pivot MUST land **ON the picked geometry** (the hit prim's bbox centre) — clicking the **car** places
  the pivot ON the car; clicking **empty background** places NO pivot (server returns `{ok:false}`). A
  server that returns an arbitrary/click-independent point on a miss leaves the pivot floating in space.
  `[vb: click the car → #gizmo visible AND #tt-pivot ≈ car centre (not (0,0,0)/[0,1,0]); two different
  car clicks give different pivots; a corner/background click places no pivot]`
- **Gizmo LOOK: three colored axis LINES through the pivot** (X red / Y green / Z blue, each drawn as a
  visible line segment with a dark outline, endpoints projected via `/api/project`) **plus a white center
  dot — NOT just a cluster of dots.** Dots alone, with no axis lines, are unusable. The fat
  invisible hit-lines are the `.handle[data-ax]` drag targets; re-project ~every 300 ms so the gizmo
  tracks camera moves. `[vb: #gizmo contains ≥2 rendered LINE elements ≥30 px long]`
- **Gizmo DRAG is pure screen-space→world math — it must NOT re-pick geometry.** Drag an axis handle →
  the pivot slides **along that world axis, freely through space** (mouse delta projected onto the axis's
  projected screen direction × world-units-per-pixel from the live projection); drag the center dot →
  the pivot moves in the **camera plane** (camera right/up from `/api/camera-pose`). Both update
  `#tt-pivot` live. **`/api/pick-point` is for the ARMED "Pick pivot" click ONLY** — re-picking geometry
  on every drag-move makes the pivot stick to the car's surface instead of moving
  through world space. `[vb: drag an axis handle → #tt-pivot changes AND zero /api/pick-point requests
  during the drag; drag the center dot → #tt-pivot changes, again zero pick-point requests]`
- `.tt-nudge` ± per axis + `#tt-step`; `#tt-frames` → `#tt-frames-s` live "N frames ÷ fps = S s/rev". `[vb]`
- `#tt-add` "Create camera from this view" (captures the exact 16-float pose as frame 0; relabels to
  "Update camera"); `#tt-remove` → `/api/turntable/remove`. Rehydrate pivot/frames from `stage.turntable`.
  `[vb: create → rig camera appears in #camera-select]`
- `#tt-preview` **spins the camera LIVE around the authored pivot — a CONTINUOUS revolution, not a
  one-off jerk.** It re-authors from the current view first (WYSIWYG), **makes the animated rig the active
  render camera**, then `/api/playback {playing:true}` runs a per-tick wall-clock animator (evaluate the
  rig pose → fabric-write the viewer camera → step, at full rate; **NO reset/reopen-per-frame and NO
  `update_from_usd_time`** — those cause the frozen/jerky slideshow; see the `ovrtx-turntable-camera`
  skill). Stop returns to the framing. A Preview that flips a flag while the view never moves — or that
  jerks once and freezes, or never orbits the pivot — is BROKEN.
  `[vb: click Preview → the stream may briefly RECONNECT (activating an animated camera reopens); wait for
  status to return to "live", THEN sample viewport PIXELS over several seconds → the frame changes
  CONTINUOUSLY (the orbit); Stop → motion stops. ‼ /api/camera-pose reports the FREE camera, NOT the
  fabric orbit pose — do NOT verify the spin via camera-pose eye (it stays frozen); use rendered pixels
  after the stream is live.]`
- **‼ STOP restores the user's EXACT pre-spin framing — regardless of spin duration and across REPEATED
  preview/stop cycles.** SAMPLE-AND-HOLD the free camera's pose at the moment the spin STARTS (not "the
  last written pose" — mid-spin fabric writes must never leak into the restore target); on Stop: clear
  the play state, DROP any drag input queued during the spin, and re-write that held pose. Any other
  restore target snaps to an arbitrary pose on Stop. `[vb: sample the frame before Preview →
  spin → Stop → after reconvergence the frame matches the pre-preview frame (small pixel delta); then
  Preview → Stop a SECOND time → still restores]`
- **The pivot gizmo HIDES while the spin plays** (Preview is a WYSIWYG render preview; an authoring
  overlay on top defeats it — and it re-projects against the frozen free-camera pose anyway, so it sits
  at a stale screen spot while the scene orbits). Hide `#gizmo` on play, restore it on Stop. (Intentional
  improvement over the reference implementation, like the Grid dropdown — the SPEC and prompt bind builds; the
  gate reports it softly so the reference stays green.) `[vb-soft: #gizmo not visible while playing]`

## Grid / batch
- `.grid-mode-btn` (one-at-a-time/full-cartesian); `.grid-q-btn`; `#grid-w/#grid-h` (synced from
  `#disp-res`); `#grid-spp`; `#grid-anim` toggles the `#grid-frame-range` row (`#grid-fstart/fend/fstep`,
  default to stage span); `#grid-cameras` is a **COMPACT camera dropdown** (a multi-select dropdown, or a
  dropdown button opening a checkbox popover — it still supports picking one OR several cameras, each
  producing its own output per permutation) — **NOT a tall per-camera checklist**: a stage with ~18
  cameras (e.g. ConceptCar) must NOT push `#grid-sets`/the variant chips off-screen. Keep `#grid-cameras`
  as the id (the selected camera values must be readable from it). `#grid-sets` per-set include + per-variant chip
  de-select. `#grid-estimate` = perms × cams × frames + rough time + the >500 guard warning shown
  BEFORE send. `#grid-out` + `#grid-browse` (📁 → `/api/browse-folder`); auto-prefill Results dir.
  `#grid-render` (disabled until count>0) / `#grid-cancel`; `#grid-bar` + `#grid-status` from
  `batch_progress`/`batch_done`. `[vb: run a 1–2 perm batch → PNGs on disk (see SPEC-FUNCTIONAL area5_files); estimate + guard shown]`

## Timeline NLE + LIVE transport  (clip drag/select/edit + playhead-scrub are easy to ship DEAD)
- One `.tl-track` per variant set + a camera track; sticky label (hide `−`, name, per-track variant
  `select`, `.tl-add-clip` "Append"). `#tl-gutter-resize` drags the label column; `#tl-show-track`
  un-hides. `[vb]`
- **Clips** `.tl-clip`: `[vb] each of —`
  - **click-select** → `.sel` green outline + `#tl-del-clip` enables; `[vb]`
  - **drag body → moves** (snaps to whole seconds, clamps no-overlap); `[vb: drag a clip, assert start_s changed]`
  - **drag `.rs` right edge → resizes** duration (min ~0.25 s, clamps); `[vb]`
  - **`.clip-var` `▾` dropdown → change to ANY variant** (not cycle); clip relabels + viewport updates; `[vb]`
  - select + **Delete/Backspace** or `#tl-del-clip` removes it (NOT dbl-click). `[vb]`
- **‼ ANY timeline edit re-applies the composed state at the playhead — NO scrub required.** `renderTimeline()`
  MUST end by re-pushing the state at the current playhead (e.g. `postStateAt(tlPlayT)` when a stage is open
  on the Timeline tab), so adding a clip, changing a track's variant, or `▾`-editing a clip updates the
  viewport IMMEDIATELY. Refreshing only when the ruler is scrubbed (or only for the *selected*
  clip) reads to the user as "variants don't update until I move the time marker." `[vb: with the playhead on a clip,
  ▾-change that clip's variant → car pixels move WITHOUT touching the ruler]`
- **Clip swatch (`.sw`) is a small COLORED dot shown ONLY when the variant has a classifier swatch color.**
  No color ⇒ NO swatch element (do NOT render a default/placeholder box, e.g. `#ffffff44`, which shows as a
  stray white square on every clip). `[vb: a clip on a non-color set (e.g. Doors) has no `.sw`; the `.sw`, if
  present, has a real (non-placeholder) background]`
- **Playhead**: `#tl-rulerbar` **drag-to-scrub** moves `#tl-playhead`, updates `#tl-playtime`, and the
  **viewport jumps to that clip's variant/camera** (debounced). `[vb: drag the ruler, assert playhead + pixels move]`
- **LIVE transport** `#tl-play` (Space) advances on a rAF wall-clock and **switches the active
  variant/camera LIVE in the viewport** as it crosses clips (coalesce → POST `/api/variant` only on a
  change); `#tl-to-start`(Home)/`#tl-step-back`(←)/`#tl-step-fwd`(→)/`#tl-to-end`(End); `#tl-loop` toggles
  `.active` (wrap at end); `#tl-playtime` / `#tl-dur` readouts; auto-scroll-follow. `[vb: 2-clip paint
  timeline, Play → car color changes at the boundary; scrub mid-clip → stream jumps to that variant;
  Home/End/Space keys move #tl-playhead + #tl-playtime; #tl-loop toggles .active]`
- **‼ An ANIMATED camera clip (the Turntable rig, or any authored camera move) ANIMATES under the
  playhead.** Scrubbing or playing across a camera-track clip whose camera is animated must re-pose
  the rig at the **clip-relative stage time** (client: send `at_s = playhead − clip.start` with the
  camera snap; server: `loop_stage_time(at_s)` → pxr-evaluate the rig's world pose at that timecode →
  write onto the viewer camera — the exact live-playback mechanism). Treating camera clips as
  static frame-0 snaps makes the turntable clip show NO rotation in the timeline. `[vb: put a Turntable
  clip on the camera track, scrub to two different times INSIDE it → the view angle differs (pixels)]`
- Append mode `#tl-mode-stack`/`#tl-mode-playhead`; presets `#tl-slideshow/#tl-mixer/#tl-clear` (confirm
  before replace); `#tl-clip`(new-clip secs)/`#tl-fps`. Render: `#tl-out`+`#tl-browse`, `#tl-render`/
  `#tl-cancel`, `#tl-bar`/`#tl-status`; on `timeline_done` auto-point Results at the MP4 + switch tab.
  Logic in `timeline-core.js` (node-tested). `[vb + dom]`
- **‼ The rendered MP4 must be BROWSER-PLAYABLE: H.264 (`libx264`) + `yuv420p`, at the current display
  resolution.** Encode via imageio-ffmpeg — **NEVER `cv2.VideoWriter` with `mp4v`** (MPEG-4 Simple
  Profile): the file writes fine, `/api/video` serves it fine, and Chrome's `<video>` silently shows
  NOTHING (ffprobe the output to confirm the codec). `[vb: render a short timeline → open it in
  Results → `#results-video.videoWidth > 0` and `currentTime` advances]`

## Results
- The side PANEL holds the controls + the RENDERS list: `#results-dir`+`#results-browse`+`#results-refresh`,
  and `#results-select` listing stills/sequences `(Nf)`/videos `▸` (a list of the folder's outputs — good).
- **Every 📁 Browse button (`#results-browse`/`#grid-browse`/`#tl-browse` → `/api/browse-folder`) is
  SINGLE-FLIGHT: a click while a folder dialog is already open is a no-op — it must NEVER open (or queue)
  a second native dialog.** Guard the shared handler with `browseFolder._busy` across all three buttons.
  This is not cosmetic: clicking Browse twice used to crash the whole process and drop the viewport
  (Tkinter is not thread-safe — see stability-checklist item 18 + SPEC-FUNCTIONAL `/api/browse-folder`).
- **The chosen media takes over the MAIN VIEWPORT PANEL** (it replaces the live viewport in place — NOT a
  separate player below the viewport): `#results-img` (still) / `#results-video` (`<video controls loop>`
  via `/api/video`) are children of the `.viewport` section overlaying `#remote-video`, shown only on the
  Results tab. The multi-frame `#results-slider`+`#results-frame-label` (`N / total`, server frame list)
  control it from the side panel. `#post-overlay` + `#post-cutsheet` (auto-select + show result).
  `[vb: run a batch → Results tab → select a render → `#results-img`/`#results-video` is `.show` AND its
  bounding box overlaps `#remote-video`'s (it occupies the viewport panel, not a box below); cutsheet writes a file]`

## Projects — full round-trip + dirty state
- `#proj-name`/`#proj-save` (`.dirty` green dot; empty-name Save updates the open project),
  `#proj-list`/`#proj-open`/`#proj-del`. **Open RESTORES ALL** (selection + display + per-camera
  looks/xforms + camera + working timeline) — first frame shows the saved state. Track-views `#tlv-*`
  project-gated (`#tlv-block` hidden + `#proj-hint` shown when no project). `[vb: save a project with a
  distinct ISO + a non-default variant; it appears in #proj-list VERBATIM; change state; open it →
  #disp-iso-v + the .chip.on selection restore to the saved values]`
- **Timeline VIEWS within a project (named track-views) — a FIRST-CLASS feature, not just project
  save/open.** With a project open, `#tlv-block` is REVEALED (hidden + `#proj-hint` "Open a project to
  manage its track views" shown otherwise): a "Saved track views" panel where `#tlv-name`+`#tlv-save`
  saves the CURRENT timeline arrangement as a named view into `#tlv-list`; selecting one + `#tlv-load`
  REPLACES the working timeline with that view's clips; `#tlv-del` removes it. MULTIPLE named views
  coexist in one project (saved inside the project, distinct from — and in addition to — save/open
  project). Save/open of whole projects alone, with no per-project named timeline views,
  is INCOMPLETE. `[vb: open a project → #tlv-block visible; save view "A" → appears in #tlv-list; change
  the timeline (add/remove a clip); save "B" → both listed; load "A" → the timeline is restored to A;
  delete "B" → gone. And: with NO project open, #tlv-block is HIDDEN and #proj-hint is shown.]`

## Layout — resizable docks (drag handles)
- `#panel-resize` (drag → the right `.panel` width changes), `#tl-resize` (drag the strip's top edge →
  `#timeline-strip` height changes), `#tl-gutter-resize` (drag → the track-label column width / `--label-w`
  changes). Each is a real pointer-drag. `[vb: drag each handle, assert the target size changed]`

## Stream self-healing (mandatory — ovstream DOES wedge, but MUST NOT reconnect-storm)
- `requestVideoFrameCallback` half-open watchdog: LIVE but no decoded frame ~10 s → reconnect →
  `/api/stream/restart` → `/api/restart`. Single-flight `connecting` guard + ~15 s connect watchdog;
  connect on `ready`; treat a **resolution change AND a same-stage Open as ATTACH/reconnect** (don't wait
  for a `ready` that won't re-fire); `localStorage` last-stage + auto-reopen after a watchdog restart;
  status vocabulary: no stage·opening·warming·connecting·attaching·reconnecting·batch·live·error.
  `[vb: after a resolution change, video returns with no reload]`
- ‼ **A HEALTHY LIVE STREAM MUST STAY CONNECTED — no disconnect/reconnect storm.** The watchdog that
  exists to recover a wedge MUST NOT fire on a healthy stream. All five of the following are required
  to avoid the storm: (1) measure liveness via the **`requestVideoFrameCallback` decoded-frame arrival**
  (reset a `lastVideoFrame` timestamp in the rVFC tick), NOT a `getVideoPlaybackQuality().totalVideoFrames`
  poll (returns 0 on the fallback path → never advances → false stall); (2) **seed `lastVideoFrame` in the
  `onStart` success handler** (at connect), never once at module load; (3) gate the watchdog on
  `streamLive && serverStageReady` with a ~10 s tolerance; (4) reconnect is **single-flight** (`recoveryTimer`
  / `connecting` guard, ~9 s backoff, `terminate()` before backoff); (5) **escalation is LATCHED** — at most
  ONE `/api/stream/restart` per dry streak (reset by a real frame), full `/api/restart` only after repeated
  rebuild failure. `[vb: open → live, sit idle 60 s; assert 0 `/api/stream/restart` + 0 `/api/restart` calls,
  decoded frames increase monotonically, exactly 1 connect / 0 disconnects, status stays `live`]`
- ‼ **ATTACH MUST FULLY REVIVE THE CLIENT — `ready` never re-fires.** The server emits `ready` ONCE per
  cold open; a page (re)load onto an already-open stage, or Open on the same path, gets NO `ready`. The
  client MUST reconcile from **`/api/stage`** (`open`/`ready` true ⇒ go straight to connect/live) and then
  be FULLY interactive: overlays cleared, orbit input armed, variant chips posting, panels populated. A
  client that gates all of that solely on the live `ready` event turns every attach (and any cold open whose
  `ready` got lost during a long mirror) into a zombie page: stuck "downloading…" overlay, video decoding but
  orbit dead and chips silently no-op. `[vb: with the stage open, page.goto() the app again + click Open →
  within ~30 s: no overlay visible, status live, a chip click changes the server-side selection]`
  **‼ Input re-binding on attach REQUIRES a stream rebuild server-side.** ovstream is single-client: when
  a new page connects while a dead ghost session lingers, VIDEO flows to the new client but INPUT can stay
  bound to the ghost — orbit dead forever (measured >3 min on the reference implementation; this is why the
  reference only soft-passes this). Builds MUST treat an attach that finds an existing session exactly
  like the resolution-change path they already implement: tear down / rebuild the ovstream session so
  video AND input bind to the NEW client. `[vb-soft today (reference-implementation limitation), but REQUIRED of
  builds: a viewport drag after attach changes pixels]`
- ‼ **The boot overlay must HAND OFF, never linger over live video.** During a cold open the overlay
  progresses download → "compiling shaders / warming" → gone. The moment decoded frames flow, NO
  mirror/progress overlay may remain over the viewport. `[vb: at first decoded frame, assert the download
  overlay is not visible]`
- ‼ **IDLE POWER: every render-loop branch with no work MUST sleep.** No stage open → sleep ~50 ms per
  tick; stage open but no client and no playback/job → `submit_last()` (keep the cached frame warm) +
  sleep ~100 ms, never path-trace into the void (measured 79% GPU / 422 W without this). Verify by
  MEASURING: the server process at ≤ ~1 core and GPU near-idle in both states. **Operational note: a
  CONNECTED client renders at FULL RATE by design** — a browser tab left open keeps the GPU/CPU hot for
  hours (measured ~16 cores × 4 h from one forgotten tab). A worthwhile enhancement (not in the reference
  implementation): pause/disconnect the stream on `document.hidden` and re-attach on visibility.

## Cross-cutting
- In-app help: `[data-help]` on controls + `.info` ⓘ badges → ONE viewport-clamped `#help-pop` + native
  `title`. `[vb: hover a [data-help] control → #help-pop shows its text]`
- Mirrored stage: rewrite path field to `source_url`; **"Downloading stage… N files · <name>" overlay on a
  COLD open** (driven by `mirror_progress {downloaded,file}`). On a cache-complete re-open NO progress shows
  (correct — same as a warm reference). This was never gated, so it only *looked* broken; it's now gated.
  index.html served no-cache/no-store. `[grade:area9_mirror_progress — delete the *.mirror_complete
  marker, open a cold http(s)/S3 URL, assert ≥1 mirror_progress with non-decreasing `downloaded` + the
  overlay shows "Downloading stage… N files"; warm re-open emits zero]`

---
## ACCEPTANCE (UX half) — gated, not self-attest
The **shipped `acceptance/verify_browser.cjs`** (headful Chrome, real mouse/keyboard, keyed ONLY on the DOM
ids above + rendered pixels — never on `/api` response field names, so it is build-agnostic) drives and
asserts EVERY `[vb]` clause above:
- viewport orbit-drag moves pixels; Esc cancels an armed pick;
- each Display slider's live value box updates on a keyboard step (ISO/focal numeric, f-stop `"off"` at 0)
  AND its `[px]` render effect (ISO low vs high → luma differs with NO auto-gain override; focal wide vs
  tele → frame changes; f-stop off vs on → frame changes); focus-pick fills `#disp-fd`;
- camera-select repopulates the sliders; save-framing overlay; reset clears optics;
- a variant (Carpaint) chip click moves car pixels;
- pivot-pick → `#gizmo` appears + `#tt-pivot` updates → axis-handle drag moves `#tt-pivot` → nudge button →
  `#tt-frames-s` readout → Create → rig camera appears in `#camera-select`;
- resolution change → decoded frames resume with NO reload + Grid W/H sync;
- a 1-perm Grid batch writes a still + the estimate/guard text shows → Results dock shows it (`#results-img`
  `.show` on the Results tab only);
- timeline clip select (`.sel` + `#tl-del-clip` enables) / drag-move / edge-resize / `.clip-var` ▾-edit /
  Delete; playhead drag-scrub moves `#tl-playhead`+`#tl-playtime`; live Play switches the car at a clip
  boundary; Space/Home/End keys; `#tl-loop` toggles; the three layout resize drags;
- project save → appears in `#proj-list` verbatim → open restores ISO + variant; track-view save round-trip;
- help popover on hover; a collapsible block toggles.

A `[vb]` clause with no assertion in this file = NOT done. The same verifier runs against any compliant
build AND against the reference app; "passes here" = "passes there." There is no "self-attest" tier.
