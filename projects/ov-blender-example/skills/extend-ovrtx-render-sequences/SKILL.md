---
name: extend-ovrtx-render-sequences
description: Extend the OVRTX Blender add-on with persistent-session animated rendering, atomic scene updates, bounded refinement, one readback per frame, resume records, and temporal validation. Use when adding sequence capture or Blender final-render animation support.
license: "Apache-2.0"
metadata:
  author: "Max Bickley"
  version: "0.1"
  team: "omniverse"
  domain: "physical-ai"
  tags:
    - blender
    - omniverse
    - ovrtx
    - rendering
---
# Extend OVRTX render sequences

## When to Use

Use when adding sequence capture or Blender final-render animation support.

## Boundary

Implement orchestration in a feature branch of the distributed source checkout.
Reuse the documented session lifecycle for Blender operators, queues, manifests,
and frame writers.

## Code map

- `addon/ovrtx_blender_example/engine.py`: `OvrtxRenderEngine.render` is
  the Blender final-render entrypoint; `_run_final_render_job` owns one render
  job's progress, cancellation, and result publication.
- `addon/ovrtx_blender_example/render_requests.py`: `RenderRequest` and
  `snapshot_key` define frame state versus session identity.
- `addon/ovrtx_blender_example/ovrtx_session.py`: `reuse_decision`
  determines whether adjacent frames may share a session.
- `addon/ovrtx_blender_example/ovrtx_session_controller.py`:
  `OvrtxSessionController.ensure` and `.render` serialize ensure/update/readback
  on the owner thread. Extend this layer; do not call the runtime client from a
  Blender UI thread.
- `addon/ovrtx_blender_example/runtime_scheduler.py`:
  `RuntimeScheduler.tick_viewport` is the model for ordered update/step/render
  work and diagnostics.
- `addon/ovrtx_blender_example/ovrtx_session_controller.py`: keep selected
  product and render-variable forwarding centralized in raw acquisition.
- Add a small `render_sequence.py` coordinator rather than embedding a
  multi-frame loop inside `OvrtxRenderEngine.render`. Give it an injectable
  frame-state adapter and writer so lifecycle tests do not require Blender/GPU.

## Instructions

1. Define the sequence contract: frame range, time mapping, camera, resolution,
   refinement target, reset policy, output format, cancellation, and resume.
2. Keep one session for compatible adjacent frames. Apply each frame's complete
   scene/camera update atomically, wait for the declared completion boundary,
   and perform one accepted readback.
3. Make re-key, reset, dropped-frame, retry, cancellation, and resume boundaries
   explicit. Never disguise independently warmed shards as one continuous take.
4. Write frames atomically and record source, add-on, worker, client, GPU,
   camera, frame/time, completed samples, timings, and checksums.
5. Test lifecycle state transitions without a GPU, then run a short animated
   camera fixture through the real runtime. Inspect first/middle/last, a contact
   sheet, flicker, temporal history, and render time.

## Required tests

Extend `tests/test_final_render_on_rpc_thread.py`,
`test_offthread_session_lifecycle.py`, `test_ovrtx_session.py`,
`test_ovrtx_session_controller.py`, and `test_render_requests.py`. Add
`test_render_sequence.py` for frame ordering, one ensure per compatible run,
one readback per completed frame, cancellation, failed atomic writes, resume,
and explicit re-key boundaries.

## Client/server escalation gate

Probe the official API for persistent session reuse, time/camera updates,
completion signaling, readback, and cancellation. Escalate only when one of
those primitives is absent or observably incorrect in a minimal direct-client
test. Per-frame process startup caused by add-on request identity, scheduling,
or state management remains add-on work.

## Handoff

Return the changed files, tests run, one representative native sequence result,
and any add-on/runtime boundary finding. Commit or build review bundles only
when the user requests them.
