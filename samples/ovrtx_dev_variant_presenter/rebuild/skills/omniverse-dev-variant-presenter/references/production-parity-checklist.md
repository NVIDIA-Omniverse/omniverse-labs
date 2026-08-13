# Production-parity checklist — build ALL of this, not just the happy path

A build that opens a stage, switches variants, streams live, and renders grid stills is only the
HAPPY PATH. Production cadence also requires the DEPTH + ROBUSTNESS + UX below. Every item here is
something the reference app does; implement all of them.

## Full control-plane surface (don't ship a subset)
Beyond the core contract, the reference exposes — implement these:
- `POST /api/browse-folder` → `{path}` — opens a NATIVE folder dialog (Tk on the server, which is a
  local desktop process) and returns the chosen absolute path. Wire a 📁 button next to EVERY folder
  field (Grid out, Timeline out, Results dir). Typing Windows paths by hand is not acceptable.
  **Crash-safety (mandatory): pin ALL Tk to ONE dedicated thread with a single reused root, and
  single-flight the Browse action** — a fresh `tk.Tk()` per request on a rotating `run_in_executor`
  worker thread crashes the whole process on the 2nd Browse (native `Tcl_AsyncDelete`), dropping the
  viewport. See stability-checklist item 18.
- `GET /api/video?path=` → streams an MP4 (FileResponse). The Results UI MUST be able to PLAY rendered
  videos (a `<video controls loop>`), not just stills.
- `POST /api/project {points}` → `{screen}` — project 3D points to screen for the turntable gizmo.
- `GET /api/camera-pose` → the live camera world pose (eye/right/up/m) for gizmo re-projection.
- `POST /api/probe-occlusion {point,nx,ny}` → whether the pivot is behind geometry (gizmo dashed/occluded shading).
- `POST /api/camera/look-at {target,radius,height}` → snap the free camera onto the turntable orbit (no 180° whip).
- `POST /api/turntable/remove` → delete the rig + sidecar and reopen.
- `POST /api/camera-state {looks,xforms}` → push per-camera optics + framings (project restore).
- `GET /api/timelines?project=` · `POST /api/timelines/save` · `GET /api/timelines/load` · `POST /api/timelines/delete`
  — project-scoped track-view CRUD.
- `POST /api/restart` (process restart, exit 43) and `POST /api/stream/restart` (rebuild the ovstream Server,
  blocking until rebuilt) — stream-recovery escalation.
- `GET /api/stage` MUST also return `look_cams` + `framed_cams` (which cameras have optics/framing) and the
  `turntable` rig info, so the client can badge cameras + rehydrate the pivot.
- WS `/events` MUST emit: `camera_params` (per-camera optics on snap → repopulate the Display sliders),
  `framing_saved`/`framing_skipped`, `timeline_done` (→ surface the MP4 in Results), plus `ready`,
  `classified`, `mirror_progress`, `batch_progress`/`batch_done`.

## Stream self-healing (this build's ovstream DOES wedge — recovery is mandatory)
- **Half-open watchdog:** use `requestVideoFrameCallback` — if LIVE but no decoded frame for ~10 s, reconnect;
  escalate to `/api/stream/restart` (ghost eviction), then `/api/restart`.
- **Reconnect loop:** a single-flight `connecting` guard + a ~15 s connect watchdog; rebuild the ovstream
  `Server` BEFORE every connect (evict the ghost single-client session).
- **Warming ticker:** an overlay with elapsed seconds while composing; self-heal a missed `ready` by polling
  `/api/stage`.
- **Rich status vocabulary:** no stage · opening · warming up · connecting · attaching · reconnecting · batch · live · error.
- **Same-stage Open = ATTACH, not recompose** (ask `/api/stage`; don't re-warm ~30 s on a re-open).
- **localStorage last-stage** + auto-reopen after a watchdog restart; merge the session checkpoint on a bare reopen.

## Turntable — full rig (not a static dot)
- Pick pivot arms an overlay (Esc cancels), posts `/api/pick-point`, and drops an **interactive 3D gizmo**:
  colored X(red)/Y(green)/Z(blue) draggable axis lines + a center screen-plane drag dot, re-projected as the
  camera moves (`/api/project` + `/api/camera-pose`), with occluded/dashed shading via `/api/probe-occlusion`.
- **Nudge** ± buttons per axis + a `step` field (auto-sized from the picked object).
- **Create from THIS view:** capture the exact current 16-float camera pose as frame 0 (pan offsets preserved),
  derive radius/height/start_deg from the live view; relabel the button to "Update camera" once the rig exists.
- **Preview spin** re-authors from the current view first (WYSIWYG), uses stage fps, toggles to "Stop".
- **Frames** field shows a live "frames ÷ fps = N s/revolution" readout.
- **Remove** calls `/api/turntable/remove` (deletes rig + sidecar + reopen) — not just a client-side clear.
- **Rehydrate** the pivot/frames from `stage.turntable` after a reload / project open.

## Projects — full round-trip
- Save bundles base selection + display/resolution + per-camera looks + framing xforms (pulled from the render
  thread via `/api/camera-state`/`GetCameraState`) + selected camera + the working timeline + project-scoped views.
- **Open RESTORES ALL of it** — seed looks/xforms/display/camera into `/api/open` so the first frame shows the
  saved state, and re-apply the working timeline. (Restoring only base-selection+camera is the bug.)
- **Dirty indicator:** track workspace changes; green dot on Save; placeholder flips to `re-save "<name>"`;
  empty-name Save updates the open project. Confirm before a project Open replaces the current look.
- **Track views:** project-gated section (hidden with a hint when no project open); per-view Save/Load/Delete
  via `/api/timelines/*`.

## Results & post — review the output
- Folder field + **📁 browse** + Refresh. Render list with frame-count `(Nf)` + `▸` for videos. These
  controls live in the side PANEL; the chosen media itself renders in the **main viewport area** (next item).
- **Play videos in-page** (`<video controls loop>` via `/api/video`) — MP4s must be reviewable. The video
  AND its scrubber/transport render in the **main viewport area, overlaying `#remote-video`, NOT in the
  side panel**. (A side-panel MP4 scrubber is the bug.) The gate measures this as overlap with the
  viewport element, so a player parked below the viewport FAILS it.
- Frame slider ONLY for multi-frame sequences, using the server-provided frame list (not a hardcoded `0000.png`),
  with `N / total` — with the image, not in the side panel.
- Overlay labels (disable when no images), Cut sheet (auto-select + show the result), media cache-bust per refresh.

## Display — real aspect ratios
16:9 720p/1080p · **1:1 1080×1080 · 9:16 1080×1920 · 4:3 1440×1080 · 2.39:1 1920×804**. Changing it rebuilds the
stream at that shape (reconnect choreography) + sets `verticalAperture = horizontalAperture*h/w` + syncs Grid W/H.

## Cameras — badges + optics restore
- Per-camera badges: ↻ animated · ◆ saved framing · ✱ optics overrides (from `look_cams`/`framed_cams`); label `name (path)`.
- On camera switch, repopulate the Display sliders from that camera's saved look (`camera_params` event).
- Reset camera clears optics to defaults AND posts explicit nulls (don't keep the dialed ISO/focal/f-stop/focus).

## Timeline NLE — editing depth
- **The timeline strip is shown ONLY when the Timeline tab is active** — hidden on Configure/Grid/Results
  even with a stage open (it is NOT an always-on fixture below the viewport). See the ui skill's
  tab-contextual dock. (Leaving the strip up on every tab is wrong.)
- Clips **snap to whole seconds and clamp against neighbors (NO overlaps)**; per-clip `▾` opens a dropdown to
  pick ANY variant directly (not cycle-to-next); select + a dedicated **Delete clip** button (+ Delete/Backspace),
  NOT double-click-to-delete.
- **Hide/show tracks** (per-track `−` + a "show hidden…" dropdown); a **gutter resize** for the label column.
- **Playback control = a real LIVE transport:** Play advances the playhead on a `requestAnimationFrame`
  wall-clock and **switches the active variant/camera LIVE in the viewport** as it crosses clips (same
  `state_at` path as scrub; coalesce so you only POST `/api/variant` on a change) — pressing Play visibly
  changes the car in the stream, not just a moving graphic. Pause/Stop/step/loop + Space/arrows/Home-End;
  **auto-scroll-follow**; live `mm:ss / length` readouts. (Separate from the Results MP4 player.) Verify on
  pixels: a 2-clip paint timeline, Play → the car color changes as the playhead crosses the boundary.
- **Render to MP4 has a Cancel button** (`/api/timeline/cancel`) + progress; on `timeline_done`, auto-point
  Results at the MP4, select it, switch to Results. Animated-camera clips auto-size to one lap.
- Presets confirm before replacing existing clips; share `timeline-core.js` logic (node-tested).

## Grid — completeness
- Hide the frame-range row until "Animation range" is checked; default start/end to the stage's frame span.
- Per-variant chips always visible (greyed until the set is included), toggle live; estimate multiplies
  perms × cameras × frames and shows a rough time + the 500 guard warning BEFORE sending.
- Auto-prefill the Results dir from the batch out_dir on start.

## Cross-cutting UX
- **In-app help:** `data-help` on dozens of controls + `info` "ⓘ" badges on section headings, driven by ONE
  JS-positioned, viewport-clamped popover (`#help-pop`); plus native `title` tooltips. (README promises "every
  control explains itself".)
- Stage-info shows a human units label (meters/cm/mm/in/ft from metersPerUnit) + frame count; rich `data-help`.
- Mirrored stage: rewrite the path field to the canonical `source_url`; show `Downloading stage… N files · <name>` overlay.
- Esc cancels an armed pick overlay. Serve index.html `no-cache, no-store, must-revalidate` — that is
  the whole cache story; do NOT add `?v=N` query strings to the asset tags, they are redundant with it.

## Backdrops / environment (the recurring bug)
Switching Backdrops MUST change the rendered sky/lighting (whole-frame pixels move). Verify the variant selection
actually composed on its deep prim AND the renderer re-applies the dome on reopen (don't reuse cached lighting).
See `usd-variant-live-switching` → "ENVIRONMENT / lighting / Backdrops variants MUST visibly re-render".

## Remote stages — MATERIALS must resolve, not just geometry (the "red car" bug)
A remote `http(s)`/S3 stage can open with the environment lit but the **car rendered as the flat red error
material** + a paint switch moves ZERO pixels — because mirroring the USD closure is NOT enough. An `.mdl`
carries its OWN deps (textures it samples + `import ::Module` it pulls) that live INSIDE the `.mdl`, invisible
to `UsdUtils`/`Sdf`; and deep collected trees exceed Windows MAX_PATH so ovrtx's native MDL resolver can't open
them. The PROVEN fix (see `usd-remote-stage-mirror`): **bulk-download the S3 prefix** (gets the `.mdl`-internal
textures/modules pxr can't see), heal the legitimately-nested `<stage_dir>/<host>/` copy, and expose the mirror
through a **short `mklink /J` junction** so native code stays under 260 chars. Verify on the **REAL public
ConceptCar S3 URL**: the car renders correct (non-red) paint AND a fast paint switch moves the car pixels.

---
**"Done" = the happy path AND every item above, verified on a full LIVE pixel/interaction sweep** — clicking real
controls (browse, add-clip, de-select, gizmo, play video, switch sky dome, change aspect, save/open project,
**open the REAL S3 ConceptCar and confirm the car paint resolves**), not just HTTP responses. The dock below the
viewport is tab-contextual (Timeline strip on Timeline; the video player + scrubber on Results) — never always-on.
