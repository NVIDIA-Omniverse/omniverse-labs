# Live edits on the public SDK path

The target is the existing add-on's complete edit contract, with one external
renderer and one native OVStage retained across accepted Blender edits. This
spike uses the existing generation owner, typed edit planner, serialized render
controller, and main-thread Blender presentation. No NVIDIA SDK is imported by
Blender.

## Edit routing

- Typed values use direct OVStage attribute writes. Validate the complete batch
  before writing, preserve schema types and semantic annotations, advance the
  write ordinal, notify the renderer, and invalidate accumulated history.
- Structural edits use the public population API on the existing stage. Add
  the accepted immutable source reference, remove its predecessor, and apply
  those source changes together. Removed prims and removed attributes must
  disappear; layering every generation indefinitely is incorrect.
- A failed population change restores the preceding accepted source before
  returning an error. The generation owner may reuse its accepted specification
  during rejection, so restoring that source is the worker's responsibility.
- Successful structural changes advance the controller's publication revision
  without changing its simulation ID. Frames from the preceding revision must
  be rejected. Retained values and authoritative physics poses are reconciled
  through the existing generation owner.
- Visual-only changes can preserve a running physics controller when the full
  composed physics inputs, configuration, and initial conditions are unchanged.
  Uncertain dependencies disable that shortcut. Explicit physics reset remains
  an explicit reset.

Direct value writes are the preferred path. Source population is the fallback
for structural changes, including material graph and geometry changes; it does
not inherently require a new native stage or renderer. Runtime binding changes,
explicit restart, and recovery after process death have separate lifecycle
semantics and must not masquerade as ordinary edit reuse.

### Native camera-sequence visibility and mesh ownership

For the native camera-sequence consumer, existing mapped mesh hide/show changes
compile to a single `visibility` token. Render restrictions, viewport restrictions
and the per-view-layer eye state use the same accepted visibility contract.
Visibility observations preserve queued dependency-graph geometry/transform/shading
flags; they must not turn known flags into unknown through an empty context.
Unknown output, collection membership and unsupported instance structure retain
the existing population fallback.

Assigning another Mesh datablock to an existing Object uses the existing typed
mesh compiler when its representation is supported. `direct_object_updates.py`
repairs only the nominated owners' data-user buckets; it does not rediscover
owners by traversing the scene. Shared users and pre-acknowledgment reversals are
preserved. The accepted index advances only after complete native acknowledgment.
Mixed edits keep proven visibility-only owners out of mesh extraction/export,
even when another owner requires population. This is an add-on routing change;
SDK versions and viewport/F12 consumer admission are unchanged.

## Evidence and limits

`run_ovstage_scene_updates_smoke.py` uses the installed public SDKs and checks
unchanged process, stage, and renderer identities across mesh addition,
deletion, topology change, and material source change. It compares a red/green
material region against a settled no-edit image; accepted writes alone do not
pass. It also verifies removed attributes and injects an apply failure after
native mutation to verify predecessor restoration.

`blender_public_physics_generation_reuse.py` checks an actual advancing Blender
physics session across a visual topology edit, including exact state, worker
PID, step count, and simulation time. `probe_public_sdk_physics_mutation.py`
separately probes native body additions and deletions on an existing stage;
that native result is not a claim that every Blender physics edit is integrated.

`prove_public_sdk_ui.py` is the integration gate: it captures the real Blender
window and requires visible changes plus unchanged worker PID, runtime epoch,
controller simulation ID, worker load count, and native stage ID. Its GIF is a
sequence of original window captures, not a continuous recording. A native
probe pass does not substitute for this UI gate.

Frame ownership and bounded shared memory are described in
[`public_sdk_frame_transport.md`](public_sdk_frame_transport.md). Full feature
parity and performance require their separate acceptance evidence.

## Bounded Blender proof

Final-code `ui-final-12` captured twelve successful ordinary Blender edits: transform,
material, light, add/delete, geometry, camera, coat/IOR, emission on/off, and
texture connect/disconnect. Each retained PID 666120, epoch 1, native Stage
`057687dfffcb491e95f6a5fff1243e6b`, simulation ID and one worker load. Its result
is `out/public-sdk-coverage/ui-final-12/private/result.json`. Warm texture
connect/disconnect took 3.45/3.20 seconds. UI10's first cold texture connection
took 94.76 seconds with the UI/previous frame retained, and remains valid
evidence. This validates the tested graph transition and
ownership, not full-speed texture editing or complete lobe/map coverage.

`ui-final-07` separately rendered a final image and resumed the viewport with
the same worker/Stage. Its clean explicit restart created a new worker and
resumed the accepted edited scene; those lifecycle results remain valid.

UI10's subsequent forced-crash failure (`teardown_unconfirmed`) is historical:
teardown now confirms shutdown before clearing references, and failed close
can be retried. UI12 repeated owned renderer death followed by a material
write, which produced a broken pipe and an uncertain update. Explicit restart
then passed in 7.98 seconds: new PID 667422 and Stage
`d85bf30aaa7f4746994f7cc40468bb76`, geometry/transforms preserved, new red material
visible. The old worker was reaped and all three recorded old resources were
absent. See `ui-final-12/private/ui-edits/crash-restart-result.json` and
`crash-resource-check.json`. No automatic recovery was observed in this case;
this is evidence for explicit restart, not automatic retry of uncertain writes.
Concurrent-physics restart and broader cancellation races remain unmeasured.

For confirmed idle renderer death, acknowledged state may be replayed before a
new, unsent value/resize/source edit. After-send failures remain uncertain and
are not automatically retried. Physics never uses that renderer recovery path.
