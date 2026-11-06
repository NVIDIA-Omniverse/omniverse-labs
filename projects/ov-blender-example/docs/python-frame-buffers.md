# Experimental native frame buffers

Set `OVRTX_PYTHON_FRAME_BUFFER=1` before starting Blender to enable the
experimental LDR frame-buffer path. The default remains the existing bytes
path. A compatible native client must advertise `frame_buffer_protocol=True`
and export its non-instantiable `FrameBuffer` type; older clients retain the
legacy path automatically. No runtime is installed or upgraded by this flag.

The add-on asks the native LDR decoder for a read-only buffer that owns its
received response. It authenticates the exact native exporter type, byte
format, contiguity, and read-only status before retaining a view. A generic
read-only memoryview is not sufficient because another owner could mutate its
backing storage. Frame-slot replacement, cancellation, and client shutdown do
not recycle storage held by an outstanding result.

`RenderResult.rgba8_origin` records the row order: legacy bytes are
`bottom_left`, while retained native pixels are `top_left`. Viewport drawing
flips its rectangle vertically instead of allocating reversed CPU rows.
`RenderResult.rgba8_bytes()` materializes canonical bottom-up bytes for final
render output; `rgba8_bytes(origin="top_left")` supplies PNG scanlines. HDR
payload handling, transfer format, color transforms, and native decoder
defaults are unchanged.

This is not end-to-end zero-copy. The current Blender GPU buffer API can expose
read-only input storage as writable. Therefore the candidate makes one
independent `bytearray` staging copy for UBYTE GPU upload. Its diagnostics report
`texture_buffer_storage="owned_mutable_staging"` and
`texture_buffer_copy_bytes`. Float-upload fallback still allocates float pixels
and reports its conversion time. No Blender patch is required, and immutable
native response memory is never passed to that writable GPU buffer API.

For a 1280x720 RGBA8 frame, the candidate removes the 3,686,400-byte native
Python-bytes copy and CPU row-reversal allocations, while adding one
3,686,400-byte upload staging copy. A retained view pins the entire native
response, including metadata and any other columns—not only its selected pixel
payload. Queue depths are unchanged; this flag does not bound caller-retained
results or promise a particular frame-rate improvement.

## Validation

The focused pure suite is `tests/test_frame_buffer_presentation.py`. Run the
included GPU fixture in a fresh, caller-owned Blender process, not an existing
artist session:

```text
blender --factory-startup --python scripts/probe_frame_buffer_gpu.py -- --output /caller/chosen/gpu-boundary.json
```

Inspect the JSON `status`; process exit alone is not a pass. This fixture uses
synthetic asymmetric 2x2 pixels, checks exact framebuffer readback through new
texture, texture-update and float-fallback routes, and proves mutable staging
writes cannot change its immutable source. It does not claim these pixels came
from the renderer. Real native ownership, worker shutdown, and retained-response
tests must be run separately against the matching native client.
