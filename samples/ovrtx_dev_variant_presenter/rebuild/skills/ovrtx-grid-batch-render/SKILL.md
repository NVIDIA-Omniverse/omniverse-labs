---
name: ovrtx-grid-batch-render
description: >
  Batch-render combinatorial USD variant permutations to disk with an ovrtx renderer:
  one-at-a-time and full-Cartesian matrix modes, a combinatorial-explosion guard, a
  {set}-{variant} naming convention, optional animation-range frame sequences, progress
  events, and cancel. Use when adding a grid/matrix/batch still-render feature to a variant
  presenter.
license: Apache-2.0
metadata:
  author: NVIDIA Customer Success
  tags: [usd, ovrtx, batch, rendering, variants]
  domain: ai-ml
  languages: [python]
---

# Grid / batch permutation render

Pure expansion/counting logic (no ovrtx) + a render loop that runs on the render thread.

## Job model

```
BatchJob:
  mode           : "one_at_a_time" | "full_cartesian" | "curated"
  base_selection : the pinned live look; every set NOT swept stays here
  included       : {set_name: [variant, ...]}   # cherry-picked variants per swept set
  cameras        : [camera_path, ...]            # one output per camera ("" = current view)
  quality        : {mode, samples_per_pixel, max_bounces, resolution:[w,h]}
  frame_mode     : "single" | "animation_range"
  out_dir        : output folder
  curated        : [[choice,...], ...]           # explicit selections (mode == curated)
  frame_start/frame_end/frame_step               # animation_range only
```

## Counting (compute BEFORE rendering; this is what `/api/batch` returns)

- `one_at_a_time` → `sum(len(variants) for each included set)` — vary ONE set at a time, the
  rest pinned to base.
- `full_cartesian` → `prod(len(variants) for each included set)` — every combination.
- `curated` → `len(curated)`.

```python
def count_permutations(job, sets):
    swept = [v for name, v in job.included.items() if name in {s.set_name for s in sets}]
    if job.mode == "curated":        return len(job.curated)
    if job.mode == "full_cartesian": return math.prod(len(v) for v in swept) if swept else 1
    if job.mode == "one_at_a_time":  return sum(len(v) for v in swept)
```

## Explosion guard

```python
THRESHOLD = 500
def guard_count(count, confirm=False, threshold=THRESHOLD):
    if count > threshold and not confirm:
        raise ExplosionError(count, threshold)   # the route maps this to HTTP 409
    return count
```
`/api/batch` returns `409` when the count exceeds the guard without `confirm`; the client
re-sends with `confirm:true`. The ConceptCar's full Cartesian is ~30 billion — the guard is
essential.

**Evaluate the guard SYNCHRONOUSLY in the `/api/batch` handler — do NOT defer it to the render
thread.** The count is pure logic (`prod(len(included[set]))`), needs no stage/GPU; compute it in the
handler and return 409 BEFORE enqueuing any render-thread work. If you instead enqueue the job and let
the render thread run `guard_count`, the reply-wait races the post-open warmup and times out → a
**504 instead of 409** right after an open (a real bug a warm/idle unit test misses). See
stability-checklist item 17. Verify by POSTing a 1210-perm cartesian IMMEDIATELY after open.

## Expansion — every permutation carries the FULL look

Each expanded selection is the full `base_selection` with the swept set(s) overridden, so a
render is complete and order-independent. The **label names only what VARIES**:
`Carpaint-Sakura` (one-at-a-time), `Carpaint-Sakura_Wheels-Black` (cartesian).

```python
def _folder_safe(t): 
    for ch in '/\\ :*?"<>|': t = t.replace(ch, "_")
    return t
def permutation_name(selection):           # {set}-{variant} joined by "_"
    return "_".join(f"{_folder_safe(c.set_name)}-{_folder_safe(c.variant)}" for c in selection) or "default"
```

**Parsing rule (shared with post-processing):** set names contain `_` (e.g. `Wheel_Colors`),
so a label parser must NOT split on `_`. The boundary between `{set}-{variant}` pairs is the
`_` BETWEEN a `-variant` and the next `set-`; accumulate `_`-segments until a `-`.

## Grid UI: include a set AND cherry-pick its variants (both directions)

`included` is `{set_name: [variants]}` — so the UI must let the user choose WHICH variants of an
included set participate, not just "all of it". Required affordances in the Grid "Include sets"
list:
- a **per-set toggle** to include/exclude the whole set (unticked sets stay pinned to the base look);
- within an included set, **each variant chip toggles individually** (click to drop/add it from
  `included[set]`) — the user MUST be able to **de-select** variants from a selected set, and the
  permutation estimate updates live as they do.
Without per-variant toggles the grid can only sweep entire sets, which is a real gap. Verify:
include a set, then click one of its variant chips OFF → it leaves `included[set]` and the count
drops.

## Render loop (render thread)

For each (permutation, camera): build a composite with that selection + camera + quality →
populate via `ovstage.population.open_usd` (the `StageSession.populate_usd` helper) → apply
per-camera framing/look overrides → converge → save PNG. Emit
`{type:"batch_progress", done, total, name, phase}` per step and `{type:"batch_done", ...}`
at the end. Reuse the live composer.

- **Converge** (don't save a noisy single step): step until a 64×64 downsample hash is
  stable N times or `max_steps` (≈ `max(samples_per_pixel, 40)`). RT2 settles fast; PT runs
  to its sample budget. Heartbeat (`submit_last()`) every few steps so a long convergence
  doesn't drop the stream (the whole batch runs LIVE↔BATCH under `USD_LOCK`).
- **animation_range:** per frame `f` in `[start,end,step]`: `session.update_from_usd_time(f)`
  (`ovstage.population.update_from_usd_time`) + `renderer.reset()`; if the camera is animated,
  evaluate its world pose in pxr at `f` and write it via `ovstage.Stage.write_attribute`
  (`session.write_omni_xform`) — never the deprecated `Renderer.write_attribute`
  (`population.update_from_usd_time` does NOT move time-sampled xforms — a confirmed remaining
  gap on ovrtx 0.4 + ovstage 0.1, see the orchestrator stability note); converge; save
  `{f:04d}.png` into a per-permutation folder; then assemble an MP4 at `fps/step`.
- **Cancel** via a direct flag (`request_cancel()` sets a bool the loop checks between
  permutations) — NOT a queued command, because the render thread is inside `run_batch` and
  isn't draining the queue.

## Results / output layout (must be discoverable)
- **A single-frame permutation writes `out_dir/{label}.png` at the TOP LEVEL** (not in a
  per-permutation subfolder) — `list_results`, the Results UI, the overlay/cut-sheet post step,
  and any external check all look there. Burying single stills in subfolders makes them look
  "missing".
- animation_range writes `out_dir/{label}/{frame:04d}.png` + assembles `out_dir/{label}.mp4`.
- `list_results(dir)` enumerates both for the Results UI.
- **Verify the render actually writes files:** run a 1–2 permutation job and confirm the PNG(s)
  appear in `out_dir` (a returned `{count}` is the plan, not proof the render ran — the job runs
  on the render thread; make sure it isn't dropped when the stage is mid-reopen, and that it
  completes even with no browser connected).
- **Make a silent save FAIL LOUDLY.** `cv2.imwrite(path, img)` returns `False` on failure (bad
  array shape/dtype, bad path) and does **NOT** raise — so a broken save writes nothing, throws
  nothing, and the batch still reports its `count`. Check the return: `if not cv2.imwrite(p, bgr):
  raise RuntimeError(f"imwrite failed: {p}")`. And track a `written` counter across the
  permutation loop; if a batch finishes with `written == 0`, **log it / emit an error event** —
  never let a job "complete" having produced zero files. (The failure looks like this: batch returns
  `count:2`, raises nothing, writes zero PNGs — while the live render + timeline-MP4 paths keep
  working, so ONLY an on-disk batch-stills assert catches it.)
