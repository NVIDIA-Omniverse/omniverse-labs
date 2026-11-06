# Camera sequences on the public SDK backend

`bpy.ops.ovrtx_example.render_timeline('EXEC_DEFAULT', selected_only=True)`
captures the selected camera markers in frame order. Each frame must have exactly
one camera marker. Use Solid viewports during capture; camera batches do not
compete with a rendered viewport for the SDK worker.

The operator returns `RUNNING_MODAL` promptly. Main-thread timers prepare each
frame, set the frame before assigning its camera, and save the resulting image.
A single native owner thread and controller remain alive between successive
sequences in the same Blender scene. The public SDK's supported scene-update
path decides whether a changed source can update the existing native session;
unsupported changes retain its normal replacement behavior. Blender file load,
add-on unregister and exit cancel active work and close the owner. Failed or
cancelled captures retire their owner and never produce a success receipt.

Native accumulation must reach `max_samples` before saving. HDR frames use the
owned scene-linear RGBA16F plane and Blender's image display transform;
RGBA8 is not required for HDR. Camera exposure and color management follow the
existing final-render presentation contract. This API does not recreate retired
SIL exposure controls or the project-specific standalone `hdr_still` operator.

`render_sequence.active_for_scene(scene)` returns the active sequence, exposing
`written`, `error`, `receipts` and `runtime_diagnostics()`. After completion,
`render_sequence.last_result()` returns status, frames and error. Each saved
frame emits an `OVRTX_SEQUENCE_FRAME` JSON record with the Blender camera name,
USD camera prim path, frame, output path, SHA256, completed samples, dimensions,
elapsed time, native timings and runtime diagnostics. A separate `.sequence.json`
contains this receipt. Completion or failure emits `OVRTX_SEQUENCE_COMPLETE`.

When `semantic_output_path` is requested, typed semantic outputs must validate
before the RGB frame is saved. Per-frame masks and native capture metadata use
the RGB output stem, so frames cannot overwrite another frame's evidence. Their
`.capture.json` remains distinct from the sequence receipt. Missing or invalid
semantic output fails the frame; callers must not promote a stale RGB bundle.
The semantic transport and generic semantic identity modules are separate
integration requirements.

## Textured World support

The active World Output may use a Background linked directly to an Environment
Texture, or a chain of unclamped MixRGB MIX selectors whose unlinked evaluated
factors are exactly zero or one. Saved simple drivers can set those factors.
The selected disk or packed texture is materialized through the existing texture
cache. Unsupported blends, procedural factors, interpolation and cycles are
rejected by the converter rather than approximated as a single HDR.

The selected texture and Background strength are authored on one canonical
`/World/StudioDome`, and other stock-exported dome lights are deactivated by
separate opinions. The stock source is preserved. A zero strength remains zero.
Object visibility uses the existing source-generation opinions; the SDK backend
must reset history for visibility changes as well as texture/intensity changes.

## Standalone HDR still compatibility

`bpy.ops.ovrtx_example.render_hdr_still()` synchronously captures the current
scene camera asynchronously, returning `RUNNING_MODAL` promptly. It reports
completion through the same sequence status API after completed native samples,
requested semantic outputs, display PNG and scene-linear 16-bit EXR are saved. It uses the
same durable owner as camera sequences. Select OVRTX, scene-linear HDR, PNG and a
scene camera, and keep viewports Solid. Native failures return `CANCELLED` and
retire the owner. File format and bit depth are restored after EXR saving.

The generic module exposes `PUBLIC_SDK_DURABLE_HDR_API = 1` for callers that must
select this API instead of legacy F12 interception. `render_hdr_still(bpy, scene)`
returns the completed PNG path. `last_capture(scene)` supplies its hash-bound
camera/frame receipt, PNG/EXR paths and runtime evidence. `last_image(scene)`
returns the retained owned scene-linear float image; callers can save another
display format with its `save_render(path, scene=scene)` method. This avoids
relying on Blender's transient F12 `Render Result`. The function preserves the
authored display transform and white balance while enforcing exposure zero,
gamma one and no display curves, as required for native scene-linear output.
Standalone semantic raw capture uses the explicit `semantic_output_path` stem;
timeline capture instead isolates each frame by its output stem.

Camera capture reserves its own native lane before source activation. Attached
rendered viewports are rejected; detached preparation is drained and closed on
the capture thread before the independent controller activates. Hidden
preparation cannot reactivate that retired runtime during capture. Returning to
a rendered viewport rejects active captures. For an idle retained camera owner,
it schedules native teardown without waiting in the viewport callback, redraws
when teardown completes, and then creates the normal viewport runtime. Native
teardown failures retain a blocked ownership barrier. Source-generation
retained values and candidate acceptance use the normal generation adapter.
