# Public SDK semantic capture

The public SDK path can return color, `SemanticSegmentation` and
`SemanticIdMap` from one native RenderProduct frame. Integer IDs remain uint32
through an owned `r32u` raw plane in the existing OVStream SHM carrier. They
never enter display-color conversion. The bounded uint8 ID-map tensor is copied
into owned base64 control metadata from the same SDK frame.

`RenderResult.semantic_capture` is a typed immutable `SemanticCapture` carrying
dimensions, origin, product/camera identity, native stage epoch/ordinal, requested
simulation time, SDK frame start/end times and both lossless payloads. The client
rejects missing requested products, wrong types/dimensions, inconsistent frame
metadata and an ID map from another SHM epoch. Existing held-copy acknowledgments,
chunk assembly, failed-copy cancellation and worker ownership apply to the entire
color-plus-ID transfer.

Set `scene.ovrtx_example.semantic_output_path` to enable final capture. The typed
request authors the two additional RenderVars on each declared product. Viewport
translation keeps this final-output control disabled. The output prefix is not a
persistent scene identity; enabling products changes composition, while changing
only the destination does not.

Objects can carry `ovrtx_semantic_class`, `ovrtx_semantic_name`, and
`ovrtx_semantic_source_id` string properties. Class IDs may not contain semicolons
or newlines. These author `SemanticsAPI:class` on the mapped object prim, inherited
by its geometry. Class/name conflicts fail explicitly. Numeric renderer IDs are
runtime output rather than stored object identity. Mesh-generation input
fingerprints include semantic identity, so changing a class updates its generated
opinion instead of silently reusing a prior label.

Consumers snapshot `semantic_identity.scene_labels(scene)` before native submit.
After the owned result returns, call:

```python
receipt = semantic_output.save_capture(
    output.with_suffix(""), result, labels,
    camera=request.camera_prim_path, frame=frame_number,
)
```

The helper rejects missing or incoherent products, unmapped IDs and classes
outside the supplied inventory. It writes `.ids.npy` and `.capture.json`, and
returns `verified`, the paths and metadata. The caller retains its own RGB save
and completion receipt. `semantic_aov.finalize_sidecar(rgb_path, capture_prefix)`
binds these native IDs to a stable class palette, `.semantic.png` and a
`.semantic.json` containing the exact RGB checksum. Palette generation is derived
presentation; `.ids.npy` preserves the original integer IDs.

## Validation

Run the included generic two-class/occluder fixture through both synchronous
rendering and native asynchronous tickets:

```text
python scripts/run_public_sdk_semantic_smoke.py --python <sdk-python> --gpu 0 --output <local-output>
```

GB10 ARM64 SDK0.5.0.377615 / OVStage0.2.0.377349 returned exact HDR float16 and
semantic uint32 pixels, and the uint8 packed ID map, through the real worker/client
SHM path in6.47seconds. Both frames used one worker, one native epoch, two copy
acknowledgments and no restarts; shutdown returned0 with no cleanup errors. This
proves the generic transport and class mapping. Blender timeline/F12 integration
and scene-specific coverage require their separate acceptance fixtures.

The route supports the two named semantic products only. Other AOVs, point
clouds, boxes and multi-sensor result aggregation remain unsupported; no color
payload is substituted for an unavailable typed product.
