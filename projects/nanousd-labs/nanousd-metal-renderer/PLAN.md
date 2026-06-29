# nanousd-metal-renderer — Port Plan

Living document tracking the Vulkan → Metal port. Read this top-to-bottom before resuming work in a fresh session.

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Skeleton & build (sibling dir, `gpu_metal.mm` stubs, CMake) | Done |
| 2 | Raster basics (mesh shaders, headless triangle) | Done |
| 3 | Headless readback (`gpu_screenshot`, `gpu_readback_pixels`) | Done — folded into Phase 2 |
| 4 | Hardware ray tracing (BLAS/TLAS + intersector kernel) | Done — MVP |
| 5 | Tiled multi-camera RT (with depth/seg/normal output) | Done — MVP |
| 6 | Raycast (LiDAR/radar) | Pending |
| 7 | IBL (HDR env + SH irradiance + BRDF LUT) | Done — MVP |
| 7b | MaterialX MSL backend (codegen path; per-mat pipelines TBD) | Done — codegen only |
| 11.A | BasisCurves via HW RT (procedural AABBs + IFT) | Done — MVP |
| Newton viewer | `viewer_metal.py` ViewerBase adapter for IsaacLab on Mac | Done |
| 8 | Metal 4 IFB migration | Stretch (deferred — see below) |
| 9 | MetalFXTemporalDenoisedScaler ("DLSS" slot) | Stretch (deferred — see below) |

**Why Phase 8 (IFB) is deferred:** Intersection Function Buffers are Metal 4's path to dispatching many distinct intersection functions cheaply. We have exactly one (`curve_isect`); the IFT/IFL machinery already in place handles a single function with no measurable cost. IFB pays off when you have e.g. per-curve-type kernels, hair vs. ribbon vs. spline shading, or many procedural primitive variants. Revisit if Phase 11.B (Catmull-Rom curves) or volumetric primitives land.

**Why Phase 9 (denoiser) is deferred:** `MetalFXTemporalDenoisedScaler` is Metal 4 / macOS 15+ — adopting it would force the deployment target up from the current 13.0. The denoiser also requires a G-buffer (motion vectors + albedo + roughness MRTs) we haven't built. Most importantly, our deterministic single-bounce ray-traced output is already noise-free — there's nothing to denoise. Phase 9 is only relevant once we add path tracing or stochastic shadow sampling, neither of which is on the roadmap.

Current source sizes (informational):
- `src/gpu_metal.mm` ~4800 lines
- `src/shaders/raytrace.metal` ~2600 lines (rt_render, rt_render_tiled, curve_isect, IBL helpers)
- `src/shaders/mesh.metal` (Phase 2)
- `test/test_headless_render.c` ~750 lines (5 RT + raster tests)
- `python/nusd_renderer/viewer_metal.py` retired stub (raises on use; viewer path is OVRTX-only)
- `test/test_viewer_metal.py` ~125 lines

End-to-end smoke tests pass on Apple M3:
- Raster (`test_triangle`): `render_output.ppm` with sky-blue clear and a cyan-blue triangle.
- RT MVP (`test_triangle_rt`): `render_output_rt.ppm` with synthetic 3-point lighting (center ~`(85, 162, 198)` sRGB-encoded), warm-horizon sky gradient (`~(239, 233, 230)`), checker-like ground tint below the horizon. The test also translates the triangle by +100 in X and verifies the center pixel becomes sky/ground — exercises the `gpu_update_tlas` refit path. Note the values shifted from `(23, 92, 144)` linear → `(85, 162, 198)` sRGB after `color_target` was switched to `MTLPixelFormatRGBA8Unorm_sRGB` for Vulkan parity.
- Tiled RT MVP (`test_tiled_rt`): 2-camera 1×2 grid (1600×600). Cam 0 looks at the triangle, Cam 1 rotated 180° around Y looks at sky/ground. Verifies left-half center is the triangle (`(23,92,144)` linear — tiled path stays linear regardless of color_target format), right-half is sky/ground (`(242,224,199)`); checks depth (=3.0), segmentation (=mesh_id+1), and normal (≈(0,0,±1)) at the cam-0 hit; then translates the triangle and re-renders to exercise the **inline** TLAS refit path inside an in-flight tiled command buffer; finally verifies cam-0 sensors flipped to miss state (depth=-1, seg=0).
- **Python suite (115 tests)**: every Python test in `test/*.py` passes — covering tiled rendering correctness, transform updates, edge cases, stress tests, training simulation, pixel-validation single-vs-tiled equivalence (with sRGB conversion), and the new `test_viewer_metal.py` (4 cases) for the Newton ViewerBase adapter. The Python tests historically used **column-vector convention** (translation at flat[3]/[7]/[11]) but `nu_set_transforms` expects **row-vector USD convention** (translation at flat[12]/[13]/[14], matching what the USD scene parser produces). All test files were corrected; nine column-vector usages flipped to the canonical row-vector form. The doc on `nu_set_transforms` in `nusd_renderer.h` now spells out the convention.
- **Curves RT (`test_curves_rt`)**: Loads `test/basicCurves.usda` (8-point linear curve → 6 segments, 0.05 radius) and renders via the curve intersection function table; reports 282 warm-tinted pixels (R > B + 30) confirming visible cylinder geometry.
- **IBL (`test_ibl_rt`)**: Loads `~/OpenUSD-install/resources/Lights/table_mountain.hdr` via `nu_load_environment`, renders the same triangle with and without IBL, expects > 1% pixels to differ; in practice 149,021/160,000 pixels change (93%) at scale=2.11×, center-pixel synthetic `(187,188,192)` → env-lit `(234,231,219)`. Skips silently if the HDR isn't at the expected path.
- **MaterialX MSL codegen**: Build with `-DNUSD_ENABLE_MATERIALS=ON` succeeds (links pre-installed MaterialX 1.39.4 from `~/OpenUSD-install/`); the GenMsl codegen path runs end-to-end without crashes.
- **Texture sampling end-to-end (`test_textured_cube_rt`)**: Loads `test/textured_quad.usda` (a quad with UVs [0,0]–[1,1] bound to a 2×2 BMP `quad4color.bmp` containing four distinct colors — red/green/blue/yellow), renders, and asserts each of the four screen quadrants resolves to a unique dominant-channel signature. Discriminating: if textures don't bind, all four anchors render the same base_color and the test fails with < 3 distinct signatures. Currently passes with 4/4 distinct: top-left blue (92,63,182), top-right yellow (178,187,59), bottom-left red (178,63,58), bottom-right green (90,187,10). Skips silently if `NUSD_ENABLE_MATERIALS=OFF`.

## High-level architecture

The project is a hard fork of `nanousd-vulkan-renderer/` cleanly split across three layers:

```
include/nusd_renderer.h     — Public C API, identical to Vulkan port
src/
  renderer.c                — Platform-agnostic API impl (uses gpu.h)
  scene.c                   — USD loading via nanousd  
  camera.c                  — View / projection matrices
  material.cpp              — MaterialX → MSL pipeline (Phase 7)
  material_stub.cpp         — No-op replacement when -DNUSD_ENABLE_MATERIALS=OFF
  gpu.h                     — RHI interface (Vulkan + Metal share this)
  gpu_metal.mm              — Metal backend (this is the port deliverable)
  shaders/
    *.glsl                  — Reference shaders from Vulkan source-of-truth
    *.metal                 — Metal Shading Language ports (filled in by phase)
```

**`renderer.c`, `scene.c`, `camera.c`, `material.cpp`, and `nusd_renderer.h` were ported unchanged** (modulo dropping the GLFW dependency in `renderer.c` and switching MaterialX backend in `material.cpp`). All Metal-specific code lives in `gpu_metal.mm` and the `*.metal` shader files.

## What was dropped vs the Vulkan port

| Removed | Why |
|---------|-----|
| `gpu_vulkan.c` (10K lines) | Replaced by `gpu_metal.mm` |
| All `*_dlss.glsl` shaders | DLSS is NVIDIA-only; MetalFX would be a separate feature |
| `python/nusd_renderer/_cuda_interop.py` | No CUDA on macOS since 2019 |
| `python/nusd_renderer/viewer_vulkan.py` | X11/Vulkan-specific viewer |
| Volk + shaderc + glslc deps in CMake | Replaced by Metal frameworks |
| GLFW dep | Headless on Metal needs no window — `MTLCreateSystemDefaultDevice()` works without one |
| GLFW calls in `renderer.c` | Same |

CUDA-Vulkan zero-copy interop and DLSS are **permanently stubbed** — see `gpu_interop_available`, `gpu_dlss_available`, etc. in `gpu_metal.mm`. They return 0 / NULL / -1 unconditionally.

## Build & test

Prerequisites:
- macOS 13.0 or later
- AppleClang (Command Line Tools — full Xcode is **not** required for now)
- nanousd (point CMake at a built `nanousd` install or checkout)

```bash
cd nanousd-metal-renderer

cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DNANOUSD_INCLUDE_DIR=../nanousd/include \
  -DNANOUSD_STATIC=../nanousd/build/Release/libnanousdapi.dylib \
  -DNANOUSD_SHARED=../nanousd/build/Release/libnanousd.dylib

cmake --build build -j$(sysctl -n hw.logicalcpu)

DYLD_LIBRARY_PATH=../nanousd/build/Release \
  ./build/test_headless_render
# Expect:
#   PASS: triangle test
#   PASS: triangle RT test
ls -la render_output.ppm render_output_rt.ppm  # both 1.4 MB, valid P6 PPM
```

`-DNUSD_ENABLE_MATERIALS=ON` enables MaterialX (Phase 7). Defaults ON; pass `-DNUSD_ENABLE_MATERIALS=OFF` to build without MaterialX.

## Architectural decisions made

### Objective-C++ + ARC, opaque structs hold `id<MTL...>` strong references
`gpu_metal.mm` is compiled with `-fobjc-arc`. The `Gpu`, `GpuBuffer_s`, and `GpuPipeline_s` structs are C++ structs (allowed with ARC for Objective-C++) holding `id<MTLDevice>`, `id<MTLBuffer>`, etc. directly. Allocation is `new`/`delete` (NOT `calloc`/`free`) — `new Gpu()` value-initializes, and the compiler-generated destructor releases the ARC fields. **Never `memset` over the struct** — it would clobber ARC's strong refs.

### Runtime MSL compilation, not AOT metallib
`xcrun metal` requires full Xcode (not bundled with Command Line Tools). We sidestep by reading `.metal` source files at `gpu_init` time and compiling via `[device newLibraryWithSource:options:error:]`. The CMake `renderer_shaders` target stages `.metal` files into `${SHADER_DIR}` (the build dir) where the runtime loader finds them.

If full Xcode becomes available, switch to AOT via `xcrun -sdk macosx metal -c ... -o ....air` then `xcrun -sdk macosx metallib ... -o nusd_shaders.metallib` and load via `[device newLibraryWithURL:]`. The CMake commented-out `xcrun metal` block is a starting point.

### Pipeline → MSL function-name mapping (no SPV bytes)
`renderer.c` calls `gpu_create_pipeline(GpuPipelineDesc{...vert_spv, frag_spv...})`. The Metal backend **ignores the SPV bytes** and selects MSL functions by hard-coded name from a single `MTLLibrary` compiled at init:

| `renderer.c` call | Metal vertex fn | Metal fragment fn |
|---|---|---|
| `gpu_create_pipeline` | `vertex_mesh` | `fragment_mesh` |
| `gpu_create_shadow_pipeline` | `vertex_mesh` | `fragment_mesh_shadow` |
| `gpu_create_material_pipeline` (Phase 7) | (delegates to `gpu_create_pipeline`; dedicated material-pipeline shaders never built) | |
| `gpu_create_dlss_pipeline` (stubbed) | (delegates to `gpu_create_pipeline`) | |

This keeps `renderer.c` and `gpu.h` unchanged. `GpuPipelineDesc.vert_spv`/`frag_spv` become unused on Metal; the `vertex_stride` and `attribs` fields are also currently ignored (the vertex descriptor is hard-coded to renderer.c's 48-byte layout). If we ever need flexibility, we can read `attribs`.

### Storage modes
- `MTLStorageModeShared` for: vertex/index buffers, push-constant equivalents, readback buffer. Apple Silicon makes Shared zero-copy for CPU access.
- `MTLStorageModePrivate` for: render target color/depth textures, BLAS/TLAS backing (Phase 4), tiled output image (Phase 5).

For the "direct SSBO write from raygen → CUDA zero-copy" pattern in the Vulkan version, the Metal equivalent on Apple Silicon is just a Shared `MTLBuffer` — no CUDA / no external-memory-fd plumbing needed. Whoever reads it (Python ctypes, PyTorch MPS, etc.) reads `[buffer contents]` directly.

### Push constants → `setBytes`
Vulkan `vkCmdPushConstants(160 bytes)` → Metal `[encoder setVertexBytes:length:160 atIndex:1]` + same for fragment. Limit 4 KB; all our push-constant structs are well under.

### No GLFW
`renderer.c` no longer creates a hidden window. `gpu_init`'s first parameter (the window pointer) is ignored — kept in the signature only because `gpu.h` is shared with the Vulkan port.

## Known pitfalls / things that bit me

1. **MSL forbids fragment-fn-calling-fragment-fn.** Refactor shared code into `static inline` helpers. (`fragment_mesh_shadow` initially called `fragment_mesh` and the compiler rejected it.)
2. **Vulkan `vec * mat` syntax matches Metal `vec * mat`.** The C side stores row-major matrices; Metal `float4x4` is column-major in storage. The reinterpret = transpose, and `vec*mat` semantics in both languages compensate. Net effect: GLSL `vec(p,1) * pc.mvp` ports verbatim to MSL `float4(p,1) * pc.mvp`.
3. **Vertex layout is 48 bytes (12 floats).** `renderer.c` packs `[pos(3) normal(3) pad(3) uv(2) matID(1)]`. Phase 4+ shaders that access uv/matID need to know this stride.
4. **`MTLPackedFloat4x3` ≠ `VkTransformMatrixKHR` despite both being "3x4 float".** `vk[r*4+c]` is element (r,c) (row-major 3x4 with translation at indices 3/7/11). `mt.columns[c][r]` is element (r,c). Don't `memcpy` — transpose explicitly. See the `vk3x4_to_mtl4x3` helper in `gpu_metal.mm`.
5. **`MTLAccelerationStructureSizes` field names** are `buildScratchBufferSize` and `refitScratchBufferSize`, not the Vulkan-style `buildScratchSize` / `refitScratchSize`.
6. **`intersection_result.object_to_world_transform` requires the `world_space_data` tag** in the intersector template. Without it, the field doesn't exist and you have to multiply via the instance buffer manually. We pay the (small) extra latency and keep the tag.
7. **`MTLTextureUsageShaderWrite`** must be added to any texture the kernel writes to. The Phase 1-3 `color_target` only had `RenderTarget | ShaderRead`; the RT kernel needed the additional flag (added — permissive, raster path unaffected).
8. **`useResource:usage:MTLResourceUsageRead` on every BLAS** — needed even though we bind the TLAS via `setAccelerationStructure:`. The TLAS internally references BLAS objects; without the `useResource` they can be evicted and the TLAS dereferences garbage memory.
9. **macOS-13 vs newer-26 link warning.** `libnanousdapi.dylib` was built for macOS 26; we target 13. Non-fatal. Bump `CMAKE_OSX_DEPLOYMENT_TARGET` if it bothers you, but doing so loses Metal-3 RT compat with older M-series.
10. **Don't `pkill cmake` cavalierly.** Stuck FetchContent git clones can be force-killed with `kill -9 <pid>` on the worker procs (`git index-pack`, `git remote-https`).
11. **Old PPM may persist.** `render_output.ppm` and `render_output_rt.ppm` from a previous run stay around; to verify a fresh run, `rm` them before re-running.
12. **Encoder state does NOT carry across encoder swaps.** Inside a single command buffer, ending a compute encoder and opening a new one (or going compute → AS → compute for an inline refit) loses **all** of `setComputePipelineState` / `setBuffer` / `setBytes` / `setAccelerationStructure` / `useResource` bindings. Factor a single bind helper and call it from both `begin_frame_tiled_rt` and the post-AS-refit reopen (`tiled_bind_compute_resources` does this). Forgetting `useResource` on BLASes after the swap is silent — the dispatch reads garbage.
13. **`uchar4(float3, uchar)` does not compile** — MSL has only `uchar4(uchar, uchar3)` / `uchar4(uchar3, uchar)` / `uchar4(uchar2, uchar2)`. Go through `uchar3(float3 * 255 + 0.5)` first and then `uchar4(c3, 255)`.
14. **`[device newLibraryWithSource:]` does not resolve user `#include` paths.** Only the system `<metal_stdlib>`, `<metal_raytracing>`, etc. work. To share helpers between kernels, put both kernels in one `.metal` file (or read a header at runtime and prepend it before compilation). We picked the single-file approach for `rt_render` + `rt_render_tiled`.
15. **Renderer-side USD-to-VkTransformMatrix conversion mismatch (was a real bug).** `renderer.c::nu_render_tiled` originally did a flat 12-float copy where the single-camera RT path uses `usd4x4_to_vk3x4` (transpose + drop last column). Without the transpose, USD-style transforms with translation in the last row of the 4×4 silently become identity. Fixed in Phase 5; the `nu_cast_rays` / `nu_cast_rays_async` paths had the same flat-copy bug and were also fixed in Phase 5b sync. The Vulkan port's `nu_cast_rays{,_async}` still have the bug — push the fix back upstream when raycast lands.
16. **MSL push-constant struct size != C struct size when matrices are involved.** MSL aligns any struct containing a `float4x4` to 16 bytes and rounds the struct size up. `GpuRtPushConstants` is 152 B in C but 160 B in MSL. `setBytes:length:152` is rejected by Metal API Validation. Fix: pad the host send into a 160-byte aligned stack buffer and `setBytes:length:160`. `GpuRtTiledPushConstants` is unaffected because the MSL `TiledPushConstants` has no `float4x4` (cameras live in a separate SSBO).
17. **`synchronizeResource:` is invalid on Shared-storage buffers.** Phase 3's `blit_color_to_readback` had a `[blit synchronizeResource:gpu->readback_buffer]` call guarded only by macOS version. Metal API Validation rejects it because synchronizeResource only applies to Managed storage. Apple Silicon's Shared buffers are CPU-coherent without it; just remove the call.
18. **`intersection_result.instance_id` is the instance index, not the BLAS index.** For segmentation output we use `hit.instance_id + 1` and rely on the fact that `mesh_to_blas[m] == m` for unique prototypes — so today instance_id = mesh_id. The instant a USD with prototype-instances arrives (multiple meshes referencing the same BLAS, `prototype_idx != m`), `instance_id` will still be the position in `MTLInstanceAccelerationStructure`'s instance array — which IS the renderer-side mesh_id, NOT the BLAS index. So the segmentation contract `seg = mesh_id + 1` continues to hold. Verify on the first prototype-bearing scene before claiming Phase 5 covers it.
19. **`nu_set_transforms` convention is row-vector USD, not column-vector numpy.** The C API doc originally said "row-major" without specifying row-vector vs column-vector. The C-level USD scene parser produces row-vector convention (translation at flat[12]/[13]/[14]); a typical numpy user assumes column-vector convention (translation at flat[3]/[7]/[11]). Many of the Python tests (carried over from the Vulkan port) used column-vector convention and "passed" tautologically (e.g. identity == identity-after-no-op-move) while only the tests that actually checked motion failed. Fixed in Phase 5b: doc made explicit, all Python tests rewritten to row-vector convention, push the test-corrections back upstream to the Vulkan repo separately.
20. **`MTLPixelFormatRGBA8Unorm_sRGB` for `color_target` matches the Vulkan swapchain.** Vulkan picks `VK_FORMAT_B8G8R8A8_SRGB` for the swapchain image, so the single-camera path's CPU readback returns sRGB-encoded pixels. Metal mirrors this by using `MTLPixelFormatRGBA8Unorm_sRGB` for the offscreen color target — the kernel writes linear floats and Metal hardware encodes to sRGB on store. The tiled RT path bypasses this (writes packed RGBA8 to a Shared `MTLBuffer`) and stays linear; callers convert via the `use_srgb` push-constant flag if they want display-ready pixels. Without the format change, `test_pixel_by_pixel_single_vs_tiled` failed with max diff ~72 because the test expected `single == sRGB(tiled)` to within ±1.
21. **`gpu_get_allocated_memory` should count the readback buffer.** Vulkan's `gpu_init` allocates `readback_buf` and that counts toward `allocated_bytes`, so a downstream test calling `add_mesh` then querying `gpu_memory_used` sees a nonzero result even before the first render. Metal's `recreate_render_targets` originally allocated `readback_buffer` without counting it. Fixed: `recreate_render_targets` now subtracts the previous size and adds the new size; `gpu_get_allocated_memory` reports a nonzero number after `gpu_init` (1.8 MB at 800×600).

## Phase 4 — Hardware ray tracing (DONE — MVP)

Single-camera RT path is wired end-to-end via `intersector<triangle_data, instancing, world_space_data>` in `src/shaders/raytrace.metal`. Exercised by `test_triangle_rt` in `test/test_headless_render.c`, including a refit smoke test (translate the triangle, re-render, verify it disappears).

**MVP scope:**

- One BLAS per unique prototype, built in a single command buffer with one shared scratch buffer.
- TLAS uses `MTLAccelerationStructureUsageRefit`; `gpu_update_tlas` refits in place rather than rebuilding (validated by the test).
- Closest-hit branch: GLSL `fast_mode` path (face normal + diffuse) and the no-IBL/no-materials/no-lights synthetic-3-point fallback (Cook-Torrance for key + rim with shadow rays, fill-only diffuse, hemisphere ambient, ACES tonemap).
- Miss branch: no-IBL sky gradient + checkered ground plane with shadow rays + fog-to-horizon.
- Shadow rays via `intersection_query<instancing>` with `accept_any_intersection(true)` and `force_opacity::opaque`.
- Output: existing `color_target` texture (added `MTLTextureUsageShaderWrite`); `gpu_screenshot`'s blit reuses verbatim.
- `rt_library` and `rt_pipeline` are persistent — `gpu_destroy_rt_scene` only blows away per-scene state (BLAS/TLAS/instance/scene-data buffers + mesh_to_blas).

**Bindings (kernel `rt_render` in `src/shaders/raytrace.metal`):**

| Slot | Type | Contents |
|---|---|---|
| `texture(0)` | `texture2d<float, access::write>` | output (= `color_target`) |
| `buffer(0)` | `instance_acceleration_structure` | TLAS, via `setAccelerationStructure:` |
| `buffer(1)` | `constant PushConstants&` | `GpuRtPushConstants` (148 B) via `setBytes` |
| `buffer(2)` | `device const SceneHeader*` | scene header at offset 0 of `scene_data_buf` |
| `buffer(3)` | `device const MeshData*` | per-mesh array at offset 16 of `scene_data_buf` |
| `buffer(4)` | `device const float*` | shared vertex buffer (12-float stride) |
| `buffer(5)` | `device const uint*` | shared index buffer |

Plus `useResource:MTLResourceUsageRead` on **each BLAS** before dispatch.

**SceneData layout** (host writes in `gpu_build_rt_scene`):
```
SceneHeader {
    uint  vertex_stride;   // 12 (floats per vertex)
    uint  has_materials;   // 0 in MVP
    float env_mip_levels;  // 0 in MVP
    uint  _pad;
}                         // 16 bytes
MeshData {
    uint  vertex_offset;   // in vertices, into shared VB
    uint  index_offset;    // in uint32, into shared IB
    uint  _pad0;
    uint  _pad1;
    float color_r, color_g, color_b, color_a;
}                         // 32 bytes per mesh
```

**Deferred — straight ports once their dependencies land:**

- BLAS compaction (`writeCompactedSize:` + `copyAndCompactAccelerationStructure:`). Vulkan reference saves ~30-40% memory; defer until scenes are big enough to care.
- IBL (`envMap` / `irrMap` lookups). The shader's `scene.env_mip_levels == 0` short-circuits all `if (envMipLevels > 0)` branches — drop-in once `gpu_load_environment` is real (Phase 7).
- MaterialBuffer + textures (Phase 7). Shader's `scene.has_materials == 0` so `mat_buf.materials[materialId]` branch is dead.
- Scene lights SSBO. Shader's synthetic 3-point fallback runs unconditionally; wire `gpu_upload_lights` and add a `device const GpuLight*` binding when scene lights become useful.
- Depth / segmentation / normals output buffers. Push-constant fields (`depth_enabled`, etc.) wired but the kernel ignores them — primary use is Phase 5 tiled.
- Glass / clearcoat. Single-bounce glass needs a continuation `intersector.intersect(...)` call; clearcoat is just an extra GGX lobe. Both straight ports of the GLSL once a warehouse-glass-shelves use case shows up.
- Real `gpu_update_tlas_inline`. Currently delegates to `gpu_update_tlas` (which submits its own command buffer + waits). Phase 5 needs a true inline refit recorded into the in-flight tiled command buffer (no commit/wait).

### Reference reading for future RT work

- **WWDC23 session 10128** "Your Guide to Metal Ray Tracing" — `intersector<>` tags, refit vs rebuild, multi-level instancing. https://developer.apple.com/videos/play/wwdc2023/10128/
- **WWDC20 session 10012** "Discover Ray Tracing with Metal" — original `intersector<triangle_data, instancing>` syntax intro. https://developer.apple.com/videos/play/wwdc2020/10012/
- **Apple — Accelerating ray tracing using Metal** — docs page that ties it all together. https://developer.apple.com/documentation/metal/accelerating-ray-tracing-using-metal
- **Apple Metal Shader Converter sample** — single-kernel RT pattern. https://developer.apple.com/metal/shader-converter/
- MSL header: `/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/*/Libraries/lib/clang/*/include/metal/metal_raytracing` — definitive source for what tags expose what fields on `intersection_result`.
- Vulkan reference: `~/nanousd-vulkan-renderer/src/gpu_vulkan.c` for `gpu_build_rt_scene` (line 2382), `gpu_update_tlas` (3667), `gpu_update_scene_colors` (3924), `gpu_destroy_rt_scene` (4003), `gpu_begin_frame_rt` (4071), `gpu_cmd_trace_rays` (4124), `gpu_end_frame_rt` (4140).
- GLSL reference shaders: `src/shaders/raytrace.rgen.glsl` (40 lines), `.rmiss.glsl` (189), `.rchit.glsl` (758).

## Phase 5 — Tiled multi-camera RT (DONE — MVP)

Multi-camera tiled RT path is wired end-to-end via a second kernel `rt_render_tiled` in **the same** `src/shaders/raytrace.metal` file as Phase 4's `rt_render`. The two kernels share `shade_hit` / `shade_miss` / `trace_shadow` / `sky_gradient` helpers (refactored to take `fast_mode`/`ground_y`/`scene_scale` as scalars rather than a `constant PushConstants&`, so both push-constant struct types can call them). Exercised by `test_tiled_rt` in `test/test_headless_render.c`.

**Single-file shader, not a separate `raytrace_tiled.metal`.** The runtime MSL compiler (`[device newLibraryWithSource:options:error:]`) doesn't resolve `#include` paths from the in-memory source string, so a separate "common header" approach would have required either a custom prepend step or out-of-band concatenation. Both kernels in one file is simpler — `ensure_rt_pipeline` and `ensure_tiled_rt_pipeline` share the same compiled `MTLLibrary` and just look up different functions by name.

**MVP scope (single-buffered, single command buffer per frame):**

- One Shared output buffer (`tiled_color_buf`) at total_w*total_h*4 bytes (`device uchar4*`); kernel writes packed RGBA8 directly. No 2D-image output mode, no double-buffering — `gpu_get_last_tiled_slot` always returns 0 and `gpu_map_tiled_staging_slot(_, _, 1)` returns NULL. The slot API is exercised in the test via slot 0.
- Single-buffered Shared depth/seg/normals buffers, sized total_w*total_h*4/4/12 bytes. Always allocated in `gpu_tiled_init`, always bound; the kernel skips the writes when the corresponding push-constant flag is 0.
- Camera SSBO at slot 6: pairs of (view_inv, proj_inv) row-major C floats, reinterpreted as column-major `float4x4` by Metal — the C-row-major-stores-the-math-matrix convention mirrors what the existing single-camera RT kernel does.
- Inline TLAS refit (`gpu_update_tlas_inline`): when called inside a tiled frame, ends the current compute encoder, opens an AS encoder, refits in place, ends the AS encoder, reopens a fresh compute encoder, and re-binds everything via `tiled_bind_compute_resources`. **Encoder state does NOT survive an encoder swap** — every `setBuffer` / `setAccelerationStructure` / `useResource` call has to be re-issued on the new encoder. A single helper covers both `gpu_begin_frame_tiled_rt` and the post-refit rebind.
- Synthesized cmd buffer: `gpu_end_frame_tiled_rt` commits but does NOT wait. The wait happens in `gpu_map_tiled_staging` / `gpu_readback_tiled_*` / `gpu_wait_tiled_complete` (all share `tiled_wait_inflight`). Single-buffered means at most one in-flight cmdbuf; `gpu_begin_frame_tiled_rt` waits on it before starting a new one.

**Bindings (kernel `rt_render_tiled` in `src/shaders/raytrace.metal`):**

| Slot | Type | Contents |
|---|---|---|
| `buffer(0)` | `instance_acceleration_structure` | TLAS, via `setAccelerationStructure:` |
| `buffer(1)` | `constant TiledPushConstants&` | `GpuRtTiledPushConstants` (152 B) via `setBytes` |
| `buffer(2)` | `device const SceneHeader*` | scene header at offset 0 of `scene_data_buf` |
| `buffer(3)` | `device const MeshData*` | per-mesh array at offset 16 of `scene_data_buf` |
| `buffer(4)` | `device const float*` | shared vertex buffer (12-float stride) |
| `buffer(5)` | `device const uint*` | shared index buffer |
| `buffer(6)` | `device const float4x4*` | camera SSBO ([view_inv, proj_inv] pairs) |
| `buffer(7)` | `device uchar4*` | output rgba8 (Shared, total_w*total_h elements) |
| `buffer(8)` | `device float*` | depth output (total_w*total_h floats) |
| `buffer(9)` | `device uint*` | segmentation output (total_w*total_h uints) |
| `buffer(10)` | `device float*` | normals output (total_w*total_h*3 floats) |

Plus `useResource:MTLResourceUsageRead` on each BLAS before dispatch.

**TiledPushConstants layout** (must match `GpuRtTiledPushConstants` in `gpu.h`, 152 B): `tile_w/tile_h/num_cols/num_cameras` at offsets 0–12; `use_direct_out`/`use_per_env_layout`/`use_srgb` at 16/20/24 (renderer.c writes these into `_pad[0..2]`); padding through 127; then `ground_y` at 128, `scene_scale` at 132, `fast_mode/depth_enabled/segmentation_enabled/normals_enabled` at 136/140/144/148. Same `ground_y`/`scene_scale` offsets as the single-camera `PushConstants` so the shared `shade_miss` helper (which takes those as scalars) is fed identical values from either kernel.

**Renderer.c bug fix folded in:** the tiled path's transform conversion (`nu_render_tiled` line ~1156) was a flat 12-float copy where the single-camera path used `usd4x4_to_vk3x4` (a transpose-and-truncate that pulls translation from indices 12/13/14 of the row-major USD 4x4 into VkTransformMatrixKHR's `[3]/[7]/[11]` slots). Without the transpose, any USD-style transform with translation in the last row gets silently dropped, and `nu_set_transforms` looks like a no-op for tiled frames. Now uses the same helper as the single-camera path.

**Deferred (future Phase 5b / Phase 6+):**

- Double-buffered `tiled_color_buf[2]` with two-slot inflight tracking — would let the CPU read slot N while the GPU renders slot N+1. Trivial extension once a downstream user needs the overlap; the `last_slot` / `map_staging_slot` API surface is already in place.
- Per-env contiguous output layout (`use_per_env_layout` flag) — a CUDA optimization. The push-constant slot exists but the Metal kernel ignores it (always tiled flat).
- `gpu_wait_previous_tiled_complete` — currently returns 0 (no previous frame). Pairs with the double-buffered upgrade.
- `gpu_set_skip_staging_copy` / `gpu_set_direct_write` — Vulkan-side optimizations that don't apply to Metal (Shared buffers ARE the staging).

## Phase 6 — Raycast (LiDAR)

Reference: `src/shaders/raycast.comp.glsl` (130 lines).

Already a compute shader in Vulkan; the Metal port is mostly mechanical. Use Metal's **`intersection_query`** (the `GL_EXT_ray_query` analogue) for the inline trace. Vulkan version uses `rayQueryEXT` to walk one ray at a time inside a compute thread; Metal `intersection_query<>` is the same idea.

Layout matches Vulkan exactly: split input layout `[origins[N*3] | directions[N*3]]`, split output `[distances[N] | normals[N*3] | hit_positions[N*3]]`. All buffers Shared storage on Apple Silicon.

Drop:
- `gpu_raycast_get_interop_fds` — stubbed permanently
- `gpu_cast_rays_gpu` — no CUDA-from-Metal interop. (If the user wants CUDA-style PyTorch interop, use MPS via `torch.from_numpy(np.frombuffer(buffer.contents, ...))`.)

Estimated: ~200 lines `gpu_metal.mm` + ~150 lines `raycast.metal`.

## Phase 7 — Materials (MaterialX MSL backend) + IBL

### Materials

Currently `material.cpp` uses `MaterialX::GenGlsl::GlslShaderGenerator` and runs `shaderc_compile_into_spv` to produce SPIR-V. The Metal port:

1. Pin MaterialX to `v1.39.4` or later — `MaterialXGenMsl` reached parity with `MaterialXGenGlsl` in 1.39.4 (Sep 2025). The CMake fetch already targets `v1.39.4`.
2. Swap `GenGlsl` includes for `GenMsl`:
   ```cpp
   #include <MaterialXGenMsl/MslShaderGenerator.h>
   auto shader_gen = MaterialX::MslShaderGenerator::create();
   ```
3. Replace `shaderc_compile_into_spv` with `[device newLibraryWithSource:options:error:]` — same runtime-compile pattern as our raster shaders.
4. Update `material.h::MaterialShader` to hold `id<MTLLibrary>` (or `void*` strong-ref) instead of `uint32_t* vert_spv` / `frag_spv`. Or: keep the C struct, but stash a function name + library handle pair.
5. Wire `gpu_create_material_pipeline` to use the named function from the per-material library.

If an older Vulkan-only MaterialX build is reachable, it may be missing `GenMsl`. Force a fresh MaterialX fetch on first Metal build.

### IBL

`raytrace.rmiss.glsl` samples an equirectangular env map via `texture(envMap, dirToEquirect(dir))`. Metal port:

1. `gpu_load_environment(hdr_path)`: load HDR via `stb_image.h` (already in the tree), upload as `MTLTextureType2D` with `MTLPixelFormatRGBA32Float` or `MTLPixelFormatRG11B10Float`. Generate mipmaps via `[blit generateMipmaps:texture]`.
2. Bind the env texture into the RT compute kernel via argument buffer or direct binding.
3. Implement `gpu_create_env_bg_pipeline` / `gpu_draw_env_background` for raster fallback.
4. BRDF LUT: optional. The Vulkan path generates a BRDF integration LUT for IBL specular. For Phase 7 MVP, skip — fall back to `Lambert` only. Enable IBL specular as Phase 7b.

Estimated: ~400 lines `material.cpp` rework + ~300 lines IBL code in `gpu_metal.mm`.

## Reference implementations (cribbing list, prioritized)

1. **Apple Metal Shader Converter sample** — single-kernel RT pattern. https://developer.apple.com/metal/shader-converter/
2. **OpenUSD HgiMetal** — production-grade RHI patterns (resource bindings, command encoding, push-constants-via-setBytes). RT not used in HdStorm-Metal but the RHI patterns are gold. https://github.com/PixarAnimationStudios/OpenUSD/tree/release/pxr/imaging/hgiMetal
3. **MaterialX `MaterialXGenMsl`** — production shader emitter we're going to use directly. https://github.com/AcademySoftwareFoundation/MaterialX/tree/main/source/MaterialXGenMsl
4. **WWDC23 10128**, **WWDC22 10105**, **WWDC20 10012** — Metal RT design talks.
5. **`~/nanousd-vulkan-renderer/`** — the Vulkan source-of-truth. For each Metal function, the corresponding Vulkan implementation in `gpu_vulkan.c` is the spec. Differences should be tracked here in PLAN.md.

## visionOS note

VisionOS does not expose Metal directly to apps; rendering goes through RealityKit `ShaderGraphMaterial` (which is MaterialX under the hood). A headless RT renderer can't target visionOS without going through RealityKit's content pipeline. Stick with macOS Metal headless on M-series; if visionOS becomes a target later, the MaterialX node graphs from Phase 7 are reusable, the rest is not.

## Resuming in a new session — checklist

1. Open this PLAN.md and read top-to-bottom — it's the bootstrapping doc.
2. `cmake --build build && DYLD_LIBRARY_PATH=../nanousd/build/Release ./build/test_headless_render` — confirm Phases 1–5 still work (`PASS: triangle test`, `PASS: triangle RT test`, `PASS: tiled RT test`).
2b. Python suite: `DYLD_LIBRARY_PATH=../nanousd/build/Release:./build PYTHONPATH=python /tmp/nusd-venv/bin/python -m pytest test/ --no-header -q` runs ~115 tests (some `@unittest.skip`/`skipTest` on environments without RT) and should report no failures. Set up the venv with `python3.12 -m venv /tmp/nusd-venv && /tmp/nusd-venv/bin/pip install pytest numpy` if it's not there.
3. Read `src/gpu_metal.mm` end-to-end (~4800 lines). Pay attention to the `Gpu` struct layout. Phase 5 added: `tiled_rt_pipeline`, `tiled_camera_buf`, `tiled_color_buf`, `tiled_depth_buf`, `tiled_seg_buf`, `tiled_normals_buf`, `tiled_camera_capacity`, `tiled_inflight_cmd`, `in_frame_rt_tiled`.
4. Read `src/shaders/raytrace.metal` (~2600 lines) — both `rt_render` (single-camera) and `rt_render_tiled` (multi-camera) with shadow rays via `intersection_query<instancing>` and shared `shade_hit` / `shade_miss` / `trace_shadow` helpers (taking scalar `fast_mode` / `ground_y` / `scene_scale`).
5. Read the Phase 5 implementation in `gpu_metal.mm` for: `ensure_tiled_rt_pipeline`, `tiled_ensure_output_buffers`, `tiled_bind_compute_resources`, `gpu_tiled_init` / `_upload_cameras`, `gpu_begin_frame_tiled_rt` / `gpu_cmd_trace_rays_tiled` / `gpu_end_frame_tiled_rt`, `gpu_update_tlas_inline` (the in-tiled-frame branch with encoder swap). That's the working pattern — Phase 6+ can mirror it.
6. For Phase 6 (raycast), reference `src/shaders/raycast.comp.glsl` (130 lines) and `~/nanousd-vulkan-renderer/src/gpu_vulkan.c` (`gpu_build_raycast_pipeline`, `gpu_cast_rays`). MSL `intersection_query<instancing>` works the same way as inside the Phase 4/5 kernels' `trace_shadow` helper. Watch out for the `nu_cast_rays`/`nu_cast_rays_async` flat-copy transform bug noted in Known pitfalls #15 — needs the same `usd4x4_to_vk3x4` fix once that path goes live.
7. Skim Vulkan reference for any further phases via `~/nanousd-vulkan-renderer/src/gpu_vulkan.c`.

## Open questions to revisit

- **Should we drop GLFW from `gpu.h`'s `gpu_init` signature?** Currently the first param is `void*` and ignored. Cleaner to drop, but breaks compat with Vulkan port. Defer to Phase 7 cleanup.
- **Should `gpu_create_pipeline` actually parse the SPV bytes?** Currently ignored. Keeping the parameter unused is fine, but consider documenting in `gpu.h`.
- **Should we offer a precompiled `.metallib` build path when full Xcode is detected?** ~10-50ms startup savings. Defer.
- **MetalFX for the "DLSS" slot?** Apple's `MTLFXTemporalScaler` / `MTLFXSpatialScaler`. Drop-in conceptually for the offscreen-MRT-then-upscale pattern in `gpu_metal.mm`. Defer to a Phase 8.
- **PyTorch MPS interop pattern.** The `gpu_get_cuda_interop_info` API is stubbed; the Metal-friendly equivalent would be `gpu_get_mps_buffer_info(slot)` returning the buffer's `[contents]` pointer + size for ctypes wrap. Decide on API shape when a downstream user materializes.
