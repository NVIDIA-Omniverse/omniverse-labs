# AGENTS.md - AI Agent Guidelines for ovrtx-interop

> IMPORTANT: Update this file after significant changes to behavior, file layout, or conventions.

## Project Purpose

`vulkan-interop` demonstrates how to display OVRTX-rendered frames in Vulkan by:

- Rendering a USD scene through OVRTX
- Mapping OVRTX output as CUDA arrays
- Copying into Vulkan-exported images imported by CUDA
- Synchronizing CUDA and Vulkan with an external timeline semaphore
- Presenting with async double-buffering (ping-pong images)

This sample intentionally keeps OVRTX integration in `src/main.cpp` (no wrapper class).

## Current Platform Support

- Linux and Windows are both implemented.
- Platform-specific interop handle types are selected in `src/cuda/cuda_kernel.*` and `src/vk/vulkan_context.*`.

## Source Layout (Current)

```
ovrtx-interop/
├── CMakeLists.txt
├── README.md
├── AGENTS.md
├── shaders/
│   ├── fullscreen.vert
│   ├── fullscreen.frag
│   ├── overlay.vert
│   └── overlay.frag
├── src/
│   ├── main.cpp
│   ├── camera/
│   │   ├── orbit_camera.hpp
│   │   └── orbit_camera.cpp
│   ├── cuda/
│   │   ├── cuda_kernel.hpp
│   │   └── cuda_kernel.cpp
│   ├── glsl/
│   │   └── spirv_loader.hpp
│   └── vk/
│       ├── vulkan_context.hpp
│       ├── vulkan_context.cpp
│       ├── sampled_image.hpp
│       ├── sampled_image.cpp
│       ├── shader.hpp
│       ├── shader.cpp
│       ├── command_buffer.hpp
│       └── command_buffer.cpp
└── tests/ (optional; built only with ENABLE_TESTS=ON)
```

## Build and Run

### Configure + build

Linux:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Windows:

```pwsh
cmake -B build
cmake --build build --config Release
```

### Run

Linux:

```bash
./build/ovrtx-interop
```

Windows:

```pwsh
.\build\Release\ovrtx-interop.exe
```

## Command Line Interface (Matches `main.cpp`)

```text
Usage: ovrtx-interop [options]

Options:
  --usd, -u <path>              USD file path or URL
                                default: robot-ovrtx sample URL
  --render-product, -r <path>   Render product prim path
                                default: /Render/Camera
  --up-axis, -a <Y|Z>           Scene up axis (default: Z)
  --units <meters|centimeters>  Scene units (default: meters)
  --num-frames, -n <N>          Render N frames, save out.png, and exit
  --help, -h                    Show help
```

## Runtime Flow (Current Implementation)

Initialization:

1. `ovrtx_initialize` -> `ovrtx_create_renderer`
2. Load USD with `ovrtx_open_usd_from_file` and wait via `ovrtx_wait_op`
3. Call `cuda_init(&cuda_uuid)` to use the CUDA context/device selected by OVRTX
4. Run one OVRTX step to detect output type and dimensions
5. Create Vulkan context using CUDA UUID for device matching
6. Create two exportable sampled images (`SHARED_IMAGE_COUNT = 2`)
7. Export Vulkan image memory and import into CUDA as surface-backed arrays
8. Export Vulkan timeline semaphore and import into CUDA
9. Prime first frame into buffer 0

Main loop:

1. Poll window/events and acquire swapchain image
2. If prior CUDA work is done (`cuEventQuery(cuda_frame_done_event)`):
   - update `read_timeline_value`
   - swap `read_idx` and `write_idx`
3. If camera moved, write transform to OVRTX with `ovrtx_write_attribute`
4. If no CUDA work pending:
   - `ovrtx_step` -> `ovrtx_fetch_results` -> `ovrtx_map_render_var_output`
   - wait on `rendered_output.cuda_sync.wait_event` (if provided)
   - enqueue pending `ovrtx_enqueue_pick_query` before stepping, if the user clicked or dragged in the viewport
   - `ovrtx_step` -> `ovrtx_fetch_results` -> `ovrtx_map_render_var_output`
   - if a pick query was submitted, map `OVRTX_RENDER_VAR_PICK_HIT` on CPU, print resolved prim paths, and update selection outline groups
   - wait on `rendered_output.cuda_sync.wait_event` (if provided)
   - copy OVRTX `CUarray` -> CUDA-imported Vulkan image at `write_idx`
   - signal external timeline semaphore from CUDA
   - unmap with `ovrtx_unmap_render_var_output` using `copy_done_event`
   - destroy results via `ovrtx_destroy_results`
5. Vulkan draws fullscreen triangle sampling image `read_idx`, plus the marquee overlay line strip while dragging
6. Submit/present with Vulkan waiting on `read_timeline_value`

## OVRTX Integration Contract

The core OVRTX frame lifecycle used by this sample:

1. `ovrtx_step(...)` enqueues a frame
2. `ovrtx_fetch_results(...)` waits for completion
3. `ovrtx_map_render_var_output(...)` maps output as CUDA array (`OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY`)
4. Consume `rendered_output.tensors[0].dl->data` as `CUarray`
5. If `rendered_output.cuda_sync.wait_event != 0`, wait on it in CUDA stream
6. After copy/compute, call `ovrtx_unmap_render_var_output(...)` and pass completion event
7. Call `ovrtx_destroy_results(...)`

Do not skip unmap/destroy; the sample treats these as required per-frame cleanup.

## Output Type and Format Mapping

Output detection:

- Search render vars for `HdrColor` first, then `LdrColor`.
- `HdrColor` is preferred when both exist.

Format mapping used in this sample:

| OVRTX output | Vulkan format | CUDA image format | bytes/pixel |
|---|---|---|---|
| HdrColor | `VK_FORMAT_R16G16B16A16_SFLOAT` | `CudaImageFormat::Half4` | 8 |
| LdrColor | `VK_FORMAT_R8G8B8A8_SRGB` | `CudaImageFormat::UInt8_4` | 4 |

## Synchronization Model

- OVRTX -> CUDA: per-frame CUDA wait event from mapped output (`cuda_sync.wait_event`)
- CUDA -> Vulkan: external timeline semaphore; CUDA signals monotonically increasing value
- CPU side: frame completion is polled by `cuEventQuery(cuda_frame_done_event)` to avoid blocking
- Double buffering:
  - CUDA writes `write_idx`
  - Vulkan samples `read_idx`
  - indices swap only after CUDA frame is complete

## Camera and Scene Updates

- Orbit camera input is handled through GLFW callbacks.
- On change, camera transform is written to `/World/Camera` attribute `omni:fabric:localMatrix`.
- `OVRTX_SEMANTIC_TRANSFORM_4x4` is used with a 16-lane float64 DLTensor payload.

## Picking and Selection

- Right mouse drag rotates the orbit camera; mouse wheel dollies.
- Left click enqueues a 1x1 pick query for the clicked RenderProduct pixel.
- Left drag enqueues a marquee pick query using the drag rectangle and draws the bounds through `overlay.vert` / `overlay.frag`.
- Any scene used with picking must restrict the picked RenderProduct to CUDA-visible GPU 0 with `uint[] deviceIds = [0]`.
- Pick results are returned as `OVRTX_RENDER_VAR_PICK_HIT`, mapped on CPU, and resolved to string prim paths with `ovrtx_get_path_dictionary` plus `path_dictionary_get_tokens_from_paths`.
- Selection outlines are enabled at renderer creation with `ovrtx_config_entry_selection_outline_enabled(true)`.
- Selection fill styling is enabled at renderer creation with `ovrtx_config_entry_selection_fill_mode(OVRTX_SELECTION_FILL_MODE_GROUP_FILL_COLOR)`.
- Group `1` is styled with `ovrtx_set_selection_group_styles()` to set a custom outline color and translucent fill color.
- Current selection is rendered by writing group `1` through `ovrtx_set_selection_outline_group`; group `0` clears prior selection.

## Dependencies

System requirements (found via CMake):

- CUDA Toolkit (`find_package(CUDAToolkit REQUIRED)`)
- Vulkan SDK/runtime (`find_package(Vulkan REQUIRED)` + `glslc`)

Fetched automatically by CMake:

- OVRTX via `ovrtx_fetch()`
- GLFW (if not found locally)
- GLM
- volk
- unordered_dense
- GoogleTest (only when `ENABLE_TESTS=ON`)

## Notes for Future Agents

- Prefer keeping OVRTX interop behavior concentrated in `src/main.cpp` unless a refactor is intentional.
- Keep README and this file aligned with actual CLI flags/defaults.
- If you change sync behavior, update both:
  - `OVRTX Integration Contract`
  - `Synchronization Model`
- If you add/remove files or targets, update `Source Layout` and `Build and Run`.
Options:
  --up Y|Z       Up axis for scene coordinate system (default: Y)
  --units m|cm   Scene units for camera distance scaling (default: cm)
```

### Camera Distance Scaling

The orbit camera's initial distance is scaled based on the `--units` option:
- **Centimeters (default)**: Distance = ~1732 cm (appropriate for cm-scale scenes)
- **Meters**: Distance = ~17.3 m (1/100th of cm distance)

### Up Axis Support

The `--up` option configures the orbit camera's vertical axis:
- **Y-up (default)**: Standard for many DCC tools (Maya, Blender default)
- **Z-up**: Common in CAD/engineering applications (AutoCAD, SolidWorks)

## ovrtx Integration

The ovrtx library provides USD scene rendering via CUDA. Integration is handled directly in `main.cpp`:
- Uses ovrtx C API directly (no wrapper class)
- Supports two output types (prefers HdrColor, falls back to LdrColor):
  - **HdrColor**: RGBA 16-bit float (half-precision), linear color space
  - **LdrColor**: RGBA 8-bit uint, sRGB color space
- Output type is detected at startup and determines Vulkan/CUDA formats
- Output is provided as CUarray via `OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY`
- `cuda_copy_array_to_surface(buffer_idx, ..., format, ...)` copies to double-buffered image
- Error handling via `ovrtx_get_last_error()` function
- Transform updated via `ovrtx_write_attribute()` with `OVRTX_SEMANTIC_XFORM_MAT4x4`

### Output Format Mapping

| Output Type | Vulkan Format | CUDA Format | Bytes/Pixel |
|-------------|--------------|-------------|-------------|
| HdrColor | `VK_FORMAT_R16G16B16A16_SFLOAT` | `CU_AD_FORMAT_HALF` | 8 |
| LdrColor | `VK_FORMAT_R8G8B8A8_SRGB` | `CU_AD_FORMAT_UNSIGNED_INT8` | 4 |

The `CudaImageFormat` enum in `cuda_kernel.hpp` encapsulates this:
- `CudaImageFormat::Half4` - For HdrColor (16-bit float × 4 channels)
- `CudaImageFormat::UInt8_4` - For LdrColor (8-bit uint × 4 channels)

Data flow (async double-buffered):
```
OrbitCamera::update() → ovrtx_write_attribute(transform)
      ↓
ovrtx_step() → ovrtx_fetch_results() → ovrtx_map_render_var_output()
      ↓
cuStreamWaitEvent(ovrtx.wait_event)
      ↓
cuda_copy_array_to_surface(write_idx, CUarray)
      ↓
cuda_signal_timeline(frame_counter)  ← async signal
      ↓
ovrtx_unmap_render_var_output()
      ↓
[Meanwhile] Vulkan renders image[read_idx]
```

## Orbit Camera Control

Interactive camera control via mouse:
- Right-click-and-drag rotates camera around target point
- Scroll wheel dollies camera in/out (adjusts distance to target)
- Uses spherical coordinates (azimuth, elevation, distance)
- `OrbitCamera` class in `src/camera/orbit_camera.hpp`
- GLFW callbacks track mouse state
- Camera transform sent to ovrtx via `ovrtx_write_attribute()`

## Multi-GPU Notes

This project is designed for systems with multiple GPUs:
- CUDA is initialized first to establish the target device
- Vulkan device selection matches the CUDA device UUID
- UUID comparison uses VkPhysicalDeviceIDProperties

## Platform Support

Currently Linux-only. Windows support planned.
