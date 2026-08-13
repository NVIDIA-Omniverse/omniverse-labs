---
name: ovrtx-turntable-camera
description: >
  Author a turntable camera that orbits a chosen pivot one full revolution, into a SIDECAR
  USD layer (never touching the source), preserving the user's exact framed pose as frame 0,
  and play/scrub/render it correctly in ovrtx (where time-sampled xforms are NOT moved by
  update_from_usd_time — you must pxr-evaluate + fabric-write per frame). Use when adding a
  turntable / orbit-camera feature to a variant presenter.
license: Apache-2.0
metadata:
  author: NVIDIA Customer Success
  tags: [usd, ovrtx, camera, turntable, animation]
  domain: ai-ml
  languages: [python]
---

# Turntable camera

## Rig shape (authored into a sidecar layer — source untouched)

Author into a sidecar `.usda` (e.g. `data/_edits/<sha1-of-stage>/turntable.usda`) that is
sublayered ABOVE the user stage. The source USD is never modified.

```
/TurntableRig            Xform   # pivot; animated spin about the up axis (time samples 0..frames)
/TurntableRig/Turntable  Camera  # child camera; leaf name "Turntable" = its dropdown label
```

Spin the pivot, not the camera: the camera is a static child, so it rigidly orbits as the
pivot rotates. Author the pivot's `xformOp:rotateZYX` (or rotate about up) with a time sample
per frame over `[0, frames)`, plus `focalLength` and aperture on the camera.

## Frame 0 = the user's EXACT current pose (WYSIWYG)

When the user clicks "create from this view", author their current world pose as frame 0 so
the spin starts from exactly what they framed (pan offsets preserved, pivot pinned in camera
space through the lap):

```python
# world = the camera's current world matrix (16 floats, row-vector)
# place the camera as a child of the pivot so that at start_deg the composed world == `world`
R0 = rotation_about_up(start_deg)            # pivot's frame-0 rotation
local = world @ inverse(R0 @ translate(pivot))
# author `local` as the camera's local xform; author the pivot's animated rotation 0..360
```

Without a supplied `camera_world`, fall back to a look-at-pivot orbit at a given
radius/height. Expose `rig_info(usd_path)` to read pivot/frames/fps/start_deg back so the UI
rehydrates the pivot gizmo after a reload (the pivot lives IN the rig).

## ‼ Store rig metadata as a JSON STRING in customLayerData — and rig_info MUST tolerate a bad sidecar
Two regressions that BOTH manifest as "the whole app is permanently unready / `/api/stage` 500s":
1. **Do NOT store the rig dict as a nested USD dictionary with bare Python lists** — e.g. writing
   `layer.customLayerData = {"turntable": {"pivot": [0,0,0], "frames": 48, ...}}`. USD serializes the
   Python `list` to an **untyped** `pivot = [0, 0, 0]`, which USD's OWN text parser then CANNOT read back
   (`Unrecognized value typename 'pivot'`) — the sidecar becomes permanently unparseable. **Store the whole
   rig dict as a single JSON STRING** (a `string` value always round-trips):
   `layer.customLayerData = {"turntable_json": json.dumps({"pivot":[...], "frames":..., ...})}`, and
   `rig_info` does `json.loads(...)`. (If you must use a typed USD dict instead, every value needs a type —
   `double3 pivot = (0,0,0)`, `double[] = [0,0,0]` — never a bare untyped list. The JSON string is simpler
   and proven.)
2. **`rig_info` (and ANY sidecar/layer read on the `/api/stage` path) MUST be wrapped in try/except** and
   return `None` on any parse/open failure — `Sdf.Layer.FindOrOpen(bad_sidecar)` RAISES `pxr.Tf.ErrorException`,
   and an unguarded raise on the status path 500s `/api/stage` → the readiness poll never succeeds → the app
   looks permanently "warming"/unready and EVERY render-dependent feature fails. A malformed sidecar must
   degrade to "no rig," never take down the status endpoint.

**Pivot picking needs a 3D POINT, not a focus distance.** The "Pick pivot" flow must return a
world-space point to orbit around — wire it to `POST /api/pick-point {nx,ny} -> {world:[x,y,z]}`
(NOT `/api/pick-focus`, which returns a scalar distance). Server side, resolve the picked prim
(ovrtx pick → `resolve_prim_path_id`) and take its world-bbox point nearest the click ray, OR the
pick hit's `worldPositionM` directly once the NDC-rect fix is in place (both are acceptable; the
bbox convention is simplest to keep consistent with the focus picker). The frontend arms
the viewport overlay, captures the normalized click, POSTs `/api/pick-point`, drops the gizmo at
the returned point, and lets the user nudge it. Without a pick-point endpoint the pivot can't be
set and "Pick pivot" silently does nothing. **Verify:** arm Pick pivot, click the car, assert a
gizmo appears at a non-zero world point.

**The gizmo must be INTERACTIVE, not a static drawing.** Draggable colored axis handles —
X(red)/Y(green)/Z(blue) — plus a center screen-plane drag dot, re-projected as the camera moves
(`/api/project` for 3D→screen AND `/api/camera-pose` for the live basis), with occluded segments
dashed via `/api/probe-occlusion`. The gizmo layer must receive pointer events and move the pivot on
drag; ALSO offer per-axis nudge ± buttons + a `step` field as the keyboard/precise path. Do NOT ship
the gizmo as a `pointer-events:none` SVG that only *shows* where the pivot is while the ± buttons are
the only way to move it — that does not meet the bar. **This is a plain
client-side SVG/DOM overlay drawn from projected screen coordinates — do NOT switch it to `ovui` (or
any native in-viewport widget kit); ovui is for a different, in-renderer overlay pipeline this app
doesn't use, and it can't receive pointer events from a browser drag anyway.** Keep `#gizmo` as SVG.

**‼ The gizmo must LOOK like an axis gizmo and DRAG through world space (two easy misses):**
1. **Look = three colored axis LINES through the pivot, not a cluster of dots.** Project the pivot
   plus the six axis endpoints (`pivot ± L·axis`, L ≈ 4×step) via `/api/project`, then per axis draw:
   a dark outline line (4px, ~55% black), the colored core line (2px; X `#ff5252`, Y `#76b900`,
   Z `#4f8cff`), and an INVISIBLE fat hit line (~14px, opacity 0) carrying `class="handle"
   data-ax="0|1|2"` — plus a white center circle and an invisible ~12px `.handle.dot
   [data-ax="screen"]`. Endpoint dots alone, with no lines, make an unusable
   gizmo. Re-project ~every 300 ms so it tracks camera moves.
2. **Drag = pure screen→world math; NEVER re-pick geometry during a drag.** On pointerdown compute,
   from the CURRENT projection, each active axis's screen direction + world-units-per-pixel
   (`wpp = L / screen_len`); on pointermove, `t = (mouse Δ · axis screen dir) * wpp` and move the
   pivot `t` along the WORLD axis (center dot: the same with camera `right`/`up` from
   `/api/camera-pose` → screen-plane move). The pivot slides freely through space — through air,
   inside geometry, anywhere. `/api/pick-point` is ONLY for the armed "Pick pivot" click: re-picking
   under the cursor on every drag-move makes the pivot STICK to the car's surface
   (and break entirely over background). Zero `/api/pick-point` calls may fire during a gizmo drag.
3. **Scope the ~300 ms reprojection timer to the Configure tab** (the gizmo is a Configure-tab
   authoring tool — hide it and no-op the redraw elsewhere). A timer left POSTing
   `/api/project` on EVERY tab once a rig exists starves the server's async worker pool and
   lags unrelated endpoints (Results listing, project/view round-trips) past usable — it presents
   as "Results/views are broken" when it is really the gizmo flooding. Also NAMESPACE the gizmo's
   client state: storing its projection on a variable that collides with the app's PROJECT NAME
   corrupts project/view saves to `[object Object]`.

**Surface the rig camera.** After authoring the sidecar, the new `Turntable` camera must SHOW UP
for the user: re-scan the stage WITH the sidecar layer composed in (`scan_stage(usd, extras=(sidecar,))`)
so the camera appears in `/api/open`/`/api/stage` `cameras[]`, return the updated stage info from
`/api/turntable`, and select the rig camera. If the rig authors but the camera list isn't
re-scanned with the sidecar, the Turntable camera never appears in the dropdown (a common miss —
status 200 but no visible camera).

## Time metadata (critical)
Hoist `timeCodesPerSecond` + start/end onto the composite ROOT from the user stage. A 60 fps
rig under a bare 24 fps composite root composes its spin at `24/60` the rate (spins ~2.5×
fast then holds). `frames ÷ stage_fps = seconds per revolution` (240 @ 60 fps = 4 s).

**‼ The sidecar's time RANGE never propagates by itself — a composed stage takes start/end
timecodes ONLY from its root/session layer, never from a SUBLAYER.** So on a stage with no (or a
shorter) authored animation, a rescan after authoring the rig still reports the OLD range (e.g.
`0/0`), the app's `start_time`/`end_time` stay 0, the playback span collapses to ~0, `tc` pins at
frame 0 — and the preview "jerks once and freezes" even with a perfect per-tick animator. That exact
signature almost always means this, not the animator. After authoring + rescan, ADOPT the rig's range into the app's
stage info (`start=0, end=frames-1, fps`) — or hoist `max(user_range, rig_range)` onto the composite
ROOT — BEFORE the reopen that activates the rig.

## Playback / scrub / render — the ovrtx animated-camera rule

`ovstage.population.update_from_usd_time(t)` does **NOT** re-evaluate time-sampled xforms on ovrtx
0.4 + ovstage 0.1 (with or without `renderer.reset()`) — it only drives scene-side time. This is a
**confirmed remaining library gap** on the installed 0.4.0/0.1.0 build, not a 0.3-only quirk.
Shooting through the rig camera renders its default-time pose every frame. The ONLY working
mechanism:

```python
stage = Usd.Stage.Open(composite)            # cached pxr stage of the live composite
xf = UsdGeom.Xformable(stage.GetPrimAtPath(CAMERA_PATH))
m = xf.ComputeLocalToWorldTransform(Usd.TimeCode(tc))   # rig pose at this stage time
session.write_omni_xform(viewer_camera, as_float64_rowvector(m))   # ovstage.Stage.write_attribute,
# one 16-lane float64 matrix element via make_dltensor + AttributeSemantic.MATRIX — NEVER the
# deprecated Renderer.write_attribute / the ovrtx-0.3-era (1,4,4) + Semantic.XFORM_MAT4x4 tensor.
```

- **Live preview spin** = wall-clock `tc` advancing each loop iteration → evaluate → fabric
  write. Streams at full rate, zero start latency (same path as free navigation). Gate the
  render loop on `playing` so a headless probe of preview behavior doesn't no-op.
  **WYSIWYG: re-author the rig from the CURRENT free-orbit view before spinning** (capture the live
  16-float pose as frame 0, derive radius/height/start_deg from it), so preview starts from exactly
  what the user is looking at — not from a stale last-authored pose. A no-op `pass` where that
  re-author should be (spinning the old rig instead) fails the WYSIWYG requirement.
- **Timeline scrub / batch animation_range** = the same evaluate-then-write, at the looped
  stage time for the frame.

## ‼ The preview must be a CONTINUOUS orbit, not a jerk (the #1 turntable regression)
The classic failure is: the rig is authored and playback is posted, yet the preview **just jerks once
and freezes / the camera never orbits the pivot**. Authoring the rig is not enough — the *playback
loop* is where it breaks. Get ALL of these right or the spin dies:
1. **The spin is a per-tick wall-clock ANIMATOR, driven from the render loop** — every iteration:
   `tc = start + ((monotonic()-t0) * fps) % span`, evaluate `ComputeLocalToWorldTransform(tc)` on the
   live composite, fabric-write onto the VIEWER camera, then `renderer.step()`. Setting `playing=True`
   must NOT itself pose one frame and return — it sets a `_play={fps,start,end,t0}` state the loop reads
   forever until stopped. Posing once = the "jerk."
2. **NEVER `reset()` / reopen / recompose per frame (or per time change).** A reset discards the
   in-flight frame and starves the stream to a ~6 fps slideshow that reads as a stutter/jerk, not a
   smooth revolution. NEVER use `ovstage.population.update_from_usd_time` for the live spin (it does
   not move time-sampled xforms on this ovrtx/ovstage build → the camera holds frame 0 → looks
   frozen — a confirmed remaining gap, not a 0.3-only quirk). The spin does NO recomposition.
3. **The orbit is only VISIBLE when the animated rig is the ACTIVE render camera.** Preview must ensure
   the rig camera is active (switch to it). Switching to an animated camera changes the effective
   camera/look → a stage REOPEN → the WebRTC stream briefly reconnects (seconds). Keep the `_play` state
   ALIVE across that reconnect and keep stepping (don't gate stepping solely on `is_client_connected`),
   so the orbit resumes streaming once the frame lands — otherwise the user sees the reconnect blip then
   a frozen frame.
4. On **stop**: clear `_play`, drop any drag input queued during the spin, and **re-write the user's
   EXACT pre-spin framing** — restoring the wrong saved camera snaps to an arbitrary pose on Stop.
   **SAMPLE-AND-HOLD the free camera's pose at the moment the spin STARTS**
   (a dedicated `_preplay_pose`, captured once in `_do_playback(playing=True)`) and write THAT back on
   stop — never "the current free-camera state" (mid-spin fabric writes / snap side-effects leak into
   it and the restore lands somewhere weird). Must hold for arbitrarily long spins and
   across REPEATED preview/stop cycles.
5. **Hide the pivot gizmo while the spin plays; restore it on Stop.** Preview is a WYSIWYG render
   preview — and during playback the gizmo re-projects against the FROZEN free-camera pose (the pose
   API doesn't track the fabric orbit), so it sits at a stale screen spot over the spinning scene.
   (Intentional improvement over the reference implementation, like the Grid camera dropdown.)

## Verify GEOMETRICALLY, never by pixel hash
RT2 noise + variant changes make pixels differ every frame regardless, so a pixel-hash
"uniqueness" check is worthless for proving camera motion. Verify by sweeping a pick-prim
across quarter-lap frames or by eyeballing 0/90/180/270° frames. Confirm the source file's
sha is unchanged (only the sidecar was written).
