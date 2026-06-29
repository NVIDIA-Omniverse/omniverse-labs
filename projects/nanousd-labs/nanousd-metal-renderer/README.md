# nanousd-metal-renderer

> **Part of [`nanousd-labs`](..)** — an experimental fleet that generates USD implementations and stacks from the [USD Core Specification](https://github.com/aousd/specifications-public/tree/main/core). This component is the **Metal renderer backend** (Apple Silicon). The [fleet README](..) has the full picture — the stack, how the repos fit together, and the skillgraph that drives it.

Headless Metal hardware ray tracing renderer built on [nanousd](../nanousd). Mac-native port of [nanousd-vulkan-renderer](../nanousd-vulkan-renderer); same USD pipeline, Metal 3 backend instead of Vulkan KHR ray tracing.

The public Python renderer contract is OVRTX-compatible: import `ovrtx.Renderer`
from `python/ovrtx`. The `nusd_renderer` and `nusd_renderer_metal` packages are
private backend bindings used by that facade.

> **Status:** Active macOS port. Raster, hardware RT, tiled multi-camera RT, Python bindings, MaterialX MSL codegen, the MDL material bridge (`src/mdl_bridge.cpp`), Gaussian splatting (`src/gs_scene.c`, `src/shaders/gaussian_rt.metal`, the `nu_gs_*` C API, and `test/test_gaussian_render.c`), and procedural primitives (`src/primitive_geom.c`) are in tree; see `PLAN.md` for the current phase matrix and remaining deferred items.

## Differences vs the Vulkan renderer

| Capability | Vulkan | Metal |
|---|---|---|
| Headless rendering | Hidden GLFW window + offscreen | Offscreen `MTLTexture`, no window needed |
| Hardware ray tracing | Raygen / miss / chit pipeline + SBT | Single compute kernel + `intersector<triangle_data, instancing>` |
| Raster | GLSL → SPIR-V (shaderc) | MSL via SPIRV-Cross (port-time) → metallib |
| Materials | MaterialX `GenGlsl` → SPIR-V | MaterialX `GenMsl` (≥1.39.4) → MTLLibrary |
| Push constants | `vkCmdPushConstants` | `[encoder setBytes:length:atIndex:]` |
| CUDA zero-copy interop | External fd + CUDA import | **Removed** — no CUDA on macOS. Apple Silicon unified memory makes `MTLStorageModeShared` already-zero-copy for CPU readers; PyTorch MPS interop is a future option. |
| DLSS upscaling | NGX SDK (NVIDIA) | **Removed** — MetalFX would be a future replacement |
| IsaacLab/Newton-Warp integration | Supported (Linux) | Not applicable (no CUDA/Warp on Mac) |

## Target hardware

M4 / M5 (MacBook Pro / Mac Studio) — hardware-accelerated ray tracing introduced on M3, expanded on M4.

The renderer requires macOS 13.0+ (Metal 3 ray tracing API).

## Architecture

```
include/nusd_renderer.h       C API (~100+ functions, identical to Vulkan port)
src/
  renderer.c                  API impl, scene management, BLAS instancing
  gpu_metal.mm                Metal backend (RT pipeline, BLAS/TLAS)
  scene.c                     USD loading via nanousd
  camera.c                    View/projection matrices
  material.cpp                Material pipeline (Phase 7)
  material_stub.cpp           Build placeholder (Phase 1–6)
  shaders/
    *.glsl                    Reference shaders ported from Vulkan source-of-truth
    *.metal                   Metal Shading Language ports (filled in by phase)
python/
  ovrtx/                       public OVRTX-compatible facade
  nusd_renderer/
    _bindings.py              private ctypes wrapper (looks for .dylib on macOS)
  nusd_renderer_metal/         private backend selector for the OVRTX facade
```

## Building

```bash
cd nanousd-metal-renderer
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DNANOUSD_INCLUDE_DIR=/path/to/nanousd/include \
  -DNANOUSD_STATIC=/path/to/libnanousdapi.dylib \
  -DNANOUSD_SHARED=/path/to/libnanousd.dylib
cmake --build build -j$(sysctl -n hw.logicalcpu)
```

To build with materials (Phase 7+):
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DNUSD_ENABLE_MATERIALS=ON ...
```

### Prerequisites

- macOS 13.0 or later (Metal 3 ray tracing)
- **Command Line Tools** (`xcode-select --install`) — full Xcode is not required. Shaders are compiled from `.metal` source at runtime via `[MTLDevice newLibraryWithSource:options:error:]` (`src/gpu_metal.mm`), so there is no `metal`/`metallib` AOT build step (see `CMakeLists.txt`).
- [nanousd](../nanousd) built locally (the `nanousd-vulkan-renderer` README has the build steps).

## Usage

### C API

```c
#include "nusd_renderer.h"

NuRenderer* r = nu_renderer_create(&(NuRendererConfig){
    .width = 128, .height = 128, .enable_rt = 1
});

int mesh_id = nu_add_mesh(r, &(NuMeshDesc){
    .positions = positions, .indices = indices,
    .nvertices = nverts, .nindices = nidx
});

int instance_id = nu_add_mesh_instance(r, mesh_id, transform);

nu_render_tiled(r, vp_inv_matrices, num_envs, w, h, NU_RENDER_RT);
nu_fetch_pixels_tiled(r, pixels, num_envs, w, h);

nu_renderer_destroy(r);
```

### Python

```python
import numpy as np
import ovrtx

renderer = ovrtx.Renderer(ovrtx.RendererConfig())
renderer.open_usd("scene.usda")
outputs = renderer.step({"/Render/Products/Product"}, 1.0 / 60.0)

with outputs["/Render/Products/Product"].frames[-1].render_vars["LdrColor"].map(ovrtx.Device.CPU) as mapped:
    pixels = np.from_dlpack(mapped)
```

## Render modes

| Mode | Description | Requirements |
|------|-------------|-------------|
| `NU_RENDER_RASTER` | Three-point lit rasterization | Any Metal GPU |
| `NU_RENDER_SHADOW` | Raster + intersection-query shadows | M1+ (any Metal RT) |
| `NU_RENDER_RT` | Full hardware ray tracing | M3+ for HW acceleration |

## See also

- [nanousd-vulkan-renderer](../nanousd-vulkan-renderer) — Linux/Windows source-of-truth, full feature set
- [Apple — Accelerating ray tracing using Metal](https://developer.apple.com/documentation/metal/accelerating-ray-tracing-using-metal)
- [MaterialX `MaterialXGenMsl`](https://github.com/AcademySoftwareFoundation/MaterialX/tree/main/source/MaterialXGenMsl)

## Contributing

This project is currently not accepting contributions.
