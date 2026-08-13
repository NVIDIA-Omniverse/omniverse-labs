# Seed-skill recap (fallback if `omniverse-realtime-viewer` is unavailable)

Prefer the real seed skill and its focused references. This recap is a safety net only.

## Non-negotiables (the seed's architectural rules)
- `ovrtx` does ALL USD/3D rendering, server-side. The browser shows an `ovstream` WebRTC
  video stream + UI and does **no** client-side 3D (no WebGL/three.js/babylon/
  model-viewer/r3f/playcanvas/aframe).
- User USD files are never modified. Viewer cameras, render products, render vars, render
  settings, selection metadata, and runtime state live in session/composite/sidecar layers.
- One owner for `renderer.step()`, stage mutation, picking, and live attribute writes.
- If the GPU/runtime is absent, scaffold the `ovrtx` path and document the requirement —
  never add a browser-renderer fallback.

## Render path (ovrtx 0.4.x + ovstage 0.1.x — the app OWNS the stage)
- The app owns ONE `ovstage.Stage` for its whole process lifetime (a `StageSession` coordinator):
  `r = ovrtx.Renderer()`; `stage = ovstage.Stage("dev_variant_presenter")`; `r.attach_ovstage(stage)`.
  Never construct the stage AFTER the renderer without attaching it, and never call the
  deprecated renderer-owned `Renderer.open_usd` / `Renderer.write_attribute` /
  `Renderer.update_from_usd_time` in production paths — they still exist on the SDK but bypass
  ovstage ownership.
- **Populate** via `ovstage.population.open_usd(stage, composite_path, ordinal=N,
  domains=population.PopulationDomain.RENDERING)`, then `stage.advance_write_floor(N,
  ovstage.Scope.ALL)` — never `step()` above the write floor you've advanced to, and never
  `step()` before the first populate (a stageless step corrupts the sensor scheduler).
- Loop: `products = r.step(render_products={rp_path}, delta_time=dt,
  ordinal=stage.committed_ordinal)`; `frame = products[rp_path].frames[0]`; `with
  frame.render_vars["LdrColor"].map(device=ovrtx.Device.CUDA) as m: ...`. Copy CPU pixels
  INSIDE the `with` (the mapping unmaps on exit).
- Compose a stage pipeline: a Camera prim → a RenderProduct (camera rel + ordered
  `LdrColor` RenderVar + `resolution` GfVec2i) → render settings. Author in a composite
  `.usda` that `subLayers` the user file, then `populate_usd` it (never `Renderer.open_usd`).
- **Live attribute writes** go through `stage.write_attribute(query, attr, ordinal, tensor,
  is_array=..., semantic=..., prim_mode=ovstage.PrimMode.UPSERT)`, where `query` comes from
  `ovstage.PathDictionary(stage).create_path_list_from_strings([...])` →
  `stage.query_from_path_list(...)` (NOT the raw bundle from `stage.get_path_dictionary()`).
  `omni:xform` is ONE 16-lane float64 matrix element via `ovstage.make_dltensor` +
  `AttributeSemantic.MATRIX` — not ovrtx 0.3's `(1,4,4)` + `Semantic.XFORM_MAT4x4`. Scalar
  optics (`focalLength`/`fStop`/`focusDistance`/`exposure`) are plain float32 `(1,)` tensors on
  schema attrs with defined fallbacks, so they apply live with no reopen; `exposure:iso` (a
  custom attribute, no fallback) still needs a reopen to clear back to "unauthored."
- Render mode = `omni:rtx:rendermode` token on the RenderProduct
  (`RealTimePathTracing`/`PathTracing`/`Minimal`); change = an `ovstage` write + `renderer.reset()` +
  warm-up. RT2 needs ~40 accumulation steps to converge; PT converges in one long step.
- Shut down in order: `renderer.detach_ovstage()` before `stage.destroy()`; don't destroy the
  stage while attached, and don't exit the interpreter without an ordered detach.

## Streaming path (ovstream 0.4.x, Windows wheel)
- `ovstream.initialize()` (once/process) → `Server(ServerType.WEBRTC)` → set
  `on_connection/on_message/on_input/on_unicode` → `start(ServerConfig(width, height,
  webrtc_signal_port=49100, webrtc_public_ip="127.0.0.1"))` → per-frame
  `stream_video(VideoFrame.from_cuda_array(bgra_buf, sync=CudaSync(...)))` → `stop()/close()`.
- Callbacks run on SDK threads (keep short). The frame must be a **BGRA8 CUDA device
  buffer** (ovrtx LdrColor is RGBA8 → GPU R↔B swizzle via a Warp kernel). WebRTC is
  single-client; resolution is fixed per session (stop/start to change).
- `on_input` mouse coords are already render-pixel — do not re-map. `MouseButton`:
  NONE=0, LEFT=1, MIDDLE=2, RIGHT=3.

## Browser client
- Use `omniverse-webrtc-streaming-library.js` (plain HTML/JS, no build) — fetched by the
  launcher on first run, pinned + SHA256-verified, gitignored, **not committed** (see the
  orchestrator SKILL for the recipe) — or
  `@nvidia/ov-web-rtc`. `AppStreamer.connect({streamSource: StreamType.DIRECT, streamConfig:
  {videoElementId, server, signalingPort, fps, onStart/onStop/onUpdate}})`. The `<video>`
  must have focus to forward input. Disconnect a verification browser before handing off
  the URL (single-client).

## Camera control
- ovrtx has no native camera. Author `omni:xform` on the camera prim via `ovstage.Stage.write_attribute`
  (float64 row-vector 4×4, packed as ONE 16-lane matrix element via `make_dltensor`; translation in
  row 3). Keep an orbit/pan/dolly controller in app state that emits matrices on mouse input. Match
  `verticalAperture = horizontalAperture * h/w`.

## Picking (eye-dropper / focus)
- `enqueue_pick_query(rp, left, top, right, bottom, flags=0)` — ‼ **NDC `[0,1]` top-left rect, NOT
  pixels**: for pixel `(x,y)` in a `w×h` stream, send `[x/w, y/h, (x+1)/w, (y+1)/h]`; raw pixel
  integers raise `ValueError: invalid NDC rectangle` on ovrtx 0.4. → next `step(...,
  ordinal=stage.committed_ordinal)` → render var `ovrtx_pick_hit`. Map FIRST (`mapped =
  rvar.map(device=CPU)`), THEN subscript the mapped object (`np.from_dlpack(mapped["primPath"])`).
  With the NDC rect fixed, `worldPositionM`/`worldNormal` on ovrtx 0.4 come back with a real hit
  position when there's a hit — prefer it (non-zero norm) for an exact focus point/distance; when it
  stays zero, fall back to `resolve_prim_path_id(int(primPath))` → the prim path → its world bbox in
  pxr → the closest AABB point to the camera.
