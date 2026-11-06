# Windows render output support

The public SDK route preserves one selected RenderProduct using `LdrColor`
(display-encoded RGBA8) or `HdrColor` (scene-linear RGBA16F). The request,
wire descriptor and owned `RenderResult.product_id` identify the same product.
Missing or mismatched identity fails before pixels are consumed. Multiple
declared/requested products and additional AOVs fail before loading a scene or
allocating renderer resources.

Blender Render Image (F12), Render Animation, HDR/EXR image output, color
management, sample/RTPT settings, the baseline DLSS preset control, and physics
timeline/playback/baking remain the supported routes. The DLSS checkbox retains
the baseline default versus Performance preset behavior; unchecked does not
claim that DLSS is completely disabled. HDR presentation applies Blender's view
transform once; LDR viewport presentation uses the renderer's display encoding.
The owned HDR payload retains the renderer's exact alpha bits. Blender HDR
presentation and F12/EXR retain the baseline opaque-alpha policy (alpha 1);
scene-linear RGB remains exact. Raw transport alpha and saved presentation alpha
are separate contracts.

The following candidate additions are explicitly unavailable, pending separate
admission. Their retained lower-level authoring/storage tests are not evidence
that the public workflows are supported:

- Multiple render products, depth/normal/semantic/instance AOVs, LiDAR/radar and XR.
- The candidate `render_timeline` camera-marker convenience operator. Use normal
  Blender Render Animation; physics timeline and baking are separate unchanged
  workflows.
- The candidate `render_hdr_still` convenience operator. Use F12 with
  scene-linear HDR and Blender's EXR output settings.
- The candidate DLSS quality enum beyond `INHERIT`. Use the baseline DLSS preset
  checkbox. Saved candidate files must reset `dlss_quality` to `INHERIT`.
- Candidate native scene/per-camera auto exposure or nondefault white-point
  controls. Use Blender color-management exposure for HDR. Saved candidate files
  must disable native auto exposure and restore its white-point scale to 10.

Candidate capture entrypoints fail before scene mutation, worker launch, or
output-file creation, and their Blender operators are not registered. Candidate
properties remain readable to diagnose saved files; they are not advertised as
admitted controls in the render panel. Runtime requests reject nondefault
candidate settings explicitly rather than ignoring them.

## Edge anti-aliasing (supersampling)

The OVRTX RTPT worker renders silhouette and shadow edges aliased. These do not
change that on the public SDK route: `omni:rtx:post:aa:op` (taa/fxaa/rtxaa),
`omni:rtx:rtpt:cfgpuCameraJitter`, `omni:rtx:pt:pixelFilter:*`,
`omni:rtx:rt:subpixel:*`, sample count (1 and 512 samples give the same edge
profile), `manualScaling` above 1, and the DLSS execMode (auto equals native
scale here; Quality/Balanced/Performance improve edge error by only 8-11
percent). The add-on therefore supersamples: `final_supersample` (default 2x2)
renders F12 at twice the width and height and box-averages back down (HDR in
linear, LDR in linear light); `viewport_supersample` (default 1x) renders the
viewport at a multiple of its resolution and lets the linear display filter
average it. Measured on Classroom 960x540 (edge RMSE vs a 4x reference): F12
0.0511 -> 0.0291, HDR viewport 0.0510 -> 0.0291, LDR viewport 0.0588 -> 0.0330;
cost: F12 59.6 s -> 73.0 s (+22 percent), viewport at 2x about 2x frame time
(render 51 ms -> 106 ms, cadence 123 ms -> 203 ms on this heavy scene), so the
viewport default stays 1x.

