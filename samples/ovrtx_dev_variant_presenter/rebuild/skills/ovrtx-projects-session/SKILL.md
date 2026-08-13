---
name: ovrtx-projects-session
description: >
  Persist a Variant Presenter workspace: named PROJECTS (bundle a stage with its base variant
  selection, display/optics, selected camera, per-camera looks/framing, and saved timeline
  views) and an automatic crash-recovery SESSION checkpoint that restores the live state
  after a server restart. Use when adding save/reopen + resilience to a variant presenter.
license: Apache-2.0
metadata:
  author: NVIDIA Customer Success
  tags: [usd, ovrtx, projects, persistence, session]
  domain: ai-ml
  languages: [python]
---

# Projects + session recovery

Two separate persistence concerns. Keep them distinct.

## Projects (explicit, named, user-managed)

A project bundles everything needed to re-open a setup:
```
{ name, usd,                       # stage identity (see below)
  base_selection: [{prim_path,set_name,variant}],
  display: {resolution, focal_length, f_stop, focus_distance, iso, ...},
  camera,                          # selected camera at save time
  looks:  {camera_path: {focal_length,f_stop,focus_distance,iso}},   # per-camera optics
  xforms: {camera_path: {m:[16 floats], dist}},                      # per-camera framing
  timelines: {name: timeline_dict} }                                 # project-scoped track views
```

- **Stage identity for a mirrored remote stage is its URL**, not the local junction path —
  re-opens resolve the URL via the mirror cache in ~1 s, and the local path is plumbing that
  shouldn't leak into project files.
- Per-camera `looks`/`xforms` live server-side on the render thread; fetch them for a save via
  a `GetCameraState` reply-queue command — and remember the **defer-reads-after-writes** rule
  (answer it at the END of the command drain, after coalesced writes, or a save races and
  misses a just-set value).
- **GUARD the reply-queue read.** A `reply.get(timeout=...)` can time out (the render thread
  is mid-cold-open or mid-batch) or return nothing → your read returns `None`. Code that then
  does `state.get(...)` raises `AttributeError` → a **500 on project save**. Always default:
  ```python
  try:
      state = reply.get(timeout=5)
  except Exception:
      state = {}
  state = state or {}            # never assume the reply arrived
  ```
  The save must still succeed (with whatever explicit body fields it has) when the render
  thread doesn't answer in time — persistence must not depend on a live render-thread reply.
- `/api/projects/save` with `timelines=None` PRESERVES the existing track-view library (a
  workspace re-save must not wipe named views). `/api/projects/load` returns the whole record;
  its `timelines` ride along.
- **Track views are PROJECT-SCOPED.** A view's clips reference the stage's variant sets +
  cameras and lean on the project's per-camera overrides, so a view is only valid inside its
  own project — never a global pool (that cross-contaminates across stages). No project open
  ⇒ no named views (the workspace timeline still saves).

## Session checkpoint (implicit, automatic, recovery-only)

A background checkpointer writes the LIVE state (open stage, selection, camera, per-camera
looks/xforms, viewer pose, stream resolution) to `data/session.json` every ~2 s. On the SAME
stage reopening without explicit state — a client auto-reopen after a watchdog relaunch, or a
manual re-open — merge the disk checkpoint back in so a server death costs ~2 s of dialing,
not the whole session. **Explicit body state (a project load) always wins**; the checkpoint
is recovery, not a second project store. Reconcile: if the checkpoint's camera no longer
exists in the stage, drop it; only restore a resolution that differs (a streamer rebuild is
costly).

## Watchdog / restart
`/api/restart` → schedule `os._exit(43)` and let an external watchdog relaunch; the client
auto-reopens its last stage and the checkpoint restores the dialed state. `exit=43` is the
app restarting ITSELF (a stream-wedge escalation), distinct from a native crash (`exit=-1`).
(When a coding agent launches the server as a background task, a `Tee`-based watchdog may not
see the exit cleanly — prefer launching the bare process and relaunching on the exit
notification, or wait on the process handle, not a pipe.)
