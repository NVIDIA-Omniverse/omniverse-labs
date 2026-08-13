---
name: ovrtx-timeline-nle
description: >
  Build a multi-track, non-linear timeline (NLE) for USD variant + camera changes over time:
  a pure state-resolution engine (tracks/clips, state_at(t), frame sampling, mixer/slideshow
  presets), a client-side JS mirror for instant scrubbing, and a render-to-MP4 path on an
  ovrtx renderer. Use when adding a timeline/sequencer to a variant presenter.
license: Apache-2.0
metadata:
  author: NVIDIA Customer Success
  tags: [usd, ovrtx, timeline, nle, variants, animation]
  domain: ai-ml
  languages: [python, javascript]
---

# Timeline (NLE)

A flagship feature: tracks = variant sets (+ one camera track), clips = "this variant for
this duration". Scrubbing drives the live viewport; rendering writes an MP4.

## Pure engine (no ovrtx, no pxr — a test should assert that)

```
Clip  = {value, start_s, duration_s}      # value = variant name, or camera path on the camera track
Track = {kind: "variant_set"|"camera", set_name, prim_path, clips:[...]}
Timeline = {duration_s, fps, tracks:[...]}
```

Core functions:
- `validate(timeline)` — at most one track per variant set; clips have positive duration and
  don't overlap within a track.
- `state_at(timeline, t, base_selection) -> (Selection, camera_path|None)` — compose the
  full selection + active camera at time `t`. Per variant-set track: the covering clip's
  value, else the **previous** clip's value (a gap HOLDS the last value), else base. Untracked
  sets stay pinned to base. Camera track → active camera; **before the first camera clip the
  FIRST clip already governs** (don't fall back to "whatever's live" — that renders surprise
  lead-ins).
- `frame_times(timeline) -> [t...]` — `n = round(duration_s * fps)`, times `i/fps` for
  `i in range(n)`. (So 4 s @ 2 fps → 8 frames.)
- Presets: `make_mixer(sets)` = parallel tracks cycling every set in lockstep (shorter tracks
  hold their last value); `make_slideshow(sets)` = one track stepping through every
  (set, variant) one change at a time.

Keep this module import-pure (only dataclasses/typing + your models). The render thread and
the API both consume `state_at`/`frame_times`; the browser mirrors it in JS.

## Client-side JS mirror (instant scrub, no round-trip)

Mirror `state_at`/`frame_times`/`_value_at` in `timeline-core.js` and PIN IT with a node
test (`timeline-core.test.cjs`) that checks the JS against the same cases as the Python.
**Scrub has NO server endpoint:** on playhead move the frontend computes `state_at`
client-side and POSTs the resulting selection to `/api/variant` (+ `/api/camera/snap` when
the camera changes). This keeps scrubbing smooth and the server authoritative only for
render.

‼ **`renderTimeline()` MUST end by re-applying the state at the playhead** (`postStateAt(playhead)` when a
stage is open on the Timeline tab) so ANY edit — append a clip, change a track's variant, `▾`-edit a clip,
move/resize/clear — updates the viewport IMMEDIATELY. If you only push on scrub (or only for the *selected*
clip), the user sees "variants don't update until I move the time marker."

‼ **Material (fast-path) sets MUST update from the timeline exactly like geometry/environment sets.**
The timeline posts through the same `/api/variant` — if the server's fast path applies silently (writes
never reach the renderer, or `renderer.reset()` missing after them; see `usd-variant-live-switching`),
the user sees Doors/Backdrops clips update the viewport while **Carpaint clips change
nothing**. Verify timeline pixel behavior on a FAST set specifically, AFTER the `classified` event
(pre-classification everything reloads and the bug hides).

‼ **Clip swatch (`.sw`) is added ONLY when a swatch color exists** for that variant: `const sw =
swatches[set]?.[clip.value]; if (sw) { el.style.background = sw; append }` — do NOT always create the swatch
with a placeholder background (e.g. `#ffffff44`); that paints a stray white square on every clip.

> JS-editing trap: do not generate JS containing `${...}` template literals through a bash
> `python -c "..."` heredoc — bash expands `${...}` before Python sees it and silently strips
> them. Write JS with a file tool and `node --check` it.

## ‼ PLAYBACK CONTROL — a real LIVE transport, not just scrub
Pressing **Play** must actually play the assembled timeline in the LIVE viewport, not merely move a
playhead graphic. Required:
- **Real-time transport:** play advances the playhead on a `requestAnimationFrame` WALL-CLOCK
  (`rel_s += (now-last)/1000`), looping at `duration_s` when loop is on; **Pause/Stop** halt it;
  **step ±1 frame**, **to-start/to-end**, **loop toggle**; Space=play/pause, arrows=step, Home/End=ends
  (ignore when typing in an input).
- **The viewport updates LIVE as it plays** — at each tick compute `state_at(playhead)` and drive the
  SAME path as scrub (POST `/api/variant` on a variant change; `/api/camera/snap` on a camera change;
  for an animated/turntable camera evaluate its pose at the looped stage time). Playing must visibly
  switch paints/doors/cameras in the stream in real time — WYSIWYG. Coalesce: only POST when the
  computed selection actually CHANGES (don't spam `/api/variant` every frame).
- **Playhead + readouts move:** the green playhead line tracks the time, with a live `mm:ss / length`
  readout, and the ruler **auto-scroll-follows** under horizontal scroll.
- This is SEPARATE from playing a rendered MP4 in **Results** — that's the dock's `<video controls loop>`
  with its own scrubber/transport BELOW the video (ui skill). Both must work: live timeline playback in
  the viewport, AND rendered-video playback in the Results dock.
**Verify on pixels:** build a 2-clip paint timeline, press Play, and confirm the car color CHANGES in the
stream as the playhead crosses the clip boundary (not just the playhead moving); scrub mid-clip and
confirm the stream jumps to that clip's variant.

## ‼ ANIMATED camera clips ANIMATE under the playhead
A camera-track clip whose camera is ANIMATED (the Turntable rig, or an authored camera move) must
re-pose the rig at the **clip-relative stage time** on every scrub/play tick — not snap frame 0 once:
- client (`applyStateAt`): when the state's camera is animated, send `at_s = playhead − clip.start`
  with the camera snap (`/api/camera/snap {camera_path, at_s}`), EVERY time (not only on camera change);
- server (`_do_snap`): with `at_s` + an animated camera, `tc = loop_stage_time(at_s, fps, start, end)`
  → pxr `ComputeLocalToWorldTransform(tc)` on the live composite → write onto the viewer camera (the
  live-playback mechanism). Snapping camera clips statically makes the turntable clip show ZERO
  rotation while scrubbing/playing the timeline. Verify: scrub to two times inside a Turntable clip —
  the view ANGLE must differ.

## `/api/timeline/render` contract (get `frames` RIGHT — a 0 means a parse bug)

Body = `{ "timeline": {duration_s, fps, tracks:[...]}, "quality": {...}, "out_dir": str }`.
The handler MUST `Timeline.from_dict(body["timeline"])` then return
`{ "frames": len(frame_times(tl)) }` where `frames = round(duration_s * fps)`. A 4 s @ 2 fps
timeline MUST return **8** — if you get 0, you parsed the wrong field (e.g. read `duration`/`fps`
off the top-level body instead of `body["timeline"]`, or defaulted them to 0). Compute frames
from the timeline's own `duration_s`/`fps`, not from the (possibly empty) tracks. **Compute `frames`
SYNCHRONOUSLY in the HTTP handler (pure `frame_times` logic) and return it immediately, then kick the
render FIRE-AND-FORGET on the render thread** (emit `timeline_done` with the MP4 path when finished) —
do NOT block the handler on a render-thread reply for `frames` (that reply-wait flakes/times out and
returns a spurious 0/504). The frames count needs no GPU; only the MP4 assembly does. Verify: post a
2-clip 4 s @ 2 fps timeline → `{frames:8}` instantly → an `.mp4` lands in `out_dir` shortly after.

## Render to MP4 (render thread)

`/api/timeline/render {timeline, quality, out_dir}` validates, computes `frames =
len(frame_times(tl))`, returns `{frames}`, and queues a render-thread job that for each
frame time: resolves `state_at`, applies the selection (fast path or reopen) + camera pose
(for an animated/turntable camera, evaluate its world pose in pxr at the looped stage time —
`loop_stage_time(rel_s, fps, start, end)` — and write it onto the viewer camera via
`ovstage.Stage.write_attribute("omni:xform", ...)` / `session.write_omni_xform` — never the
deprecated `Renderer.write_attribute`), converges, saves a frame, then assembles the MP4 (reuse
the batch `frames_to_video`). Runs LIVE↔BATCH under `USD_LOCK` + heartbeat; cancel via the direct
flag. Pass the per-camera look/framing into each frame's composite (a timeline render that forgets
the camera overrides renders the wrong optics).

**‼ MP4 codec: H.264 (`libx264`) + `pix_fmt yuv420p` via imageio-ffmpeg — NEVER `cv2.VideoWriter`
with `mp4v`.** MPEG-4 Simple Profile writes a valid file that `/api/video` serves fine, but Chrome's
`<video>` cannot decode it → the Results player silently shows NOTHING (ffprobe the output if
unsure). Render frames at the CURRENT DISPLAY RESOLUTION, never a hardcoded 640×360.

## UI MUST let the user ADD clips to tracks (not just render a pre-made timeline)

The timeline is useless if there's no way to put a clip on a track. There is **one track per
variant set + one camera track**, and EACH track must expose a way to add a clip:
- a per-track **picker** of that set's variants (or, on the camera track, the authored cameras) +
  an **Append / Add-clip** button → appends a clip of the chosen value;
- an **append-mode** toggle in the toolbar (Stack after the last clip, or drop At-playhead);
- once a clip exists: **drag** to move, drag the right edge to resize, a **▾** on the clip to
  change its value, and Delete to remove it;
- **presets** (Mixer / Slideshow) auto-fill all tracks, and **Clear** empties them — but presets
  are NOT a substitute for manual per-track add.
Build a clip's value list from the open stage's variant sets + cameras (the camera track lists
`cameras[]`). Verify by clicking a track's picker, choosing a variant, clicking Append, and
seeing a clip appear on that track; do the same on the camera track.

**The per-clip `▾` must be a REAL, appended dropdown that picks ANY value directly** — actually
append a `<select>`/menu element to the clip and wire its change handler; do NOT (a) create the
control and forget to append it, (b) make it `double-click-to-cycle-to-the-next-variant`, or (c)
rely on dblclick at all. The user must be able to jump straight to an arbitrary variant. Selection +
a **dedicated Delete-clip button** (also Delete/Backspace) — not double-click-to-delete. (A `▾` that
is created but never appended, with a dblclick-cycle fallback, does not meet this bar.)

**Track management depth (don't skip):** per-track **hide/show** (a `−` on each track header + a
"show hidden…" menu to bring them back) and a **draggable label-gutter resize** for the track-label
column. These are part of "editing depth," not optional polish.

## UI
A timeline strip in the below-viewport dock (keep the viewport visible for scrub feedback) shown
**ONLY while the Timeline tab is active** (hidden on Configure/Grid/Results even with a stage open —
see the ui skill's tab-contextual dock; it is NOT an always-on fixture). Contents: drag/resize/delete
clips, a ruler scrub (debounce ~90 ms), Mixer/Slideshow/Clear presets, a transport (play/pause/step/loop
+ Space / arrows / Home-End), and Render-to-MP4. Clip fills can reuse the per-variant swatch colors from
the classifier.
