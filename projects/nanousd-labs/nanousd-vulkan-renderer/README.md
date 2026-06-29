# nanousd-vulkan-renderer

> **Part of [`nanousd-labs`](..)** — an experimental fleet that generates USD implementations and stacks from the [USD Core Specification](https://github.com/aousd/specifications-public/tree/main/core). This component is the **Vulkan renderer backend** (headless hardware ray tracing). The [fleet README](..) has the full picture — the stack, how the repos fit together, and the skillgraph that drives it.

Headless Vulkan hardware ray tracing renderer built on [nanousd](../nanousd).

The public Python renderer contract is now OVRTX: import `ovrtx.Renderer` from
`python/ovrtx` and select this implementation with
`NANOUSD_OVRTX_BACKEND=vulkan`. The historical `nusd_renderer` package no
longer exports a public `NuRenderer` facade; its submodules are private backend
plumbing for the OVRTX implementation.

## Quick start

```bash
./build.sh     # clean release build (C lib + Python bindings into workspace .venv)
./run.sh       # runs the Vulkan-RT smoke + correctness ctest
```

Both scripts accept `--help` for flags. The same `build.sh`/`run.sh` shape is used across the fleet repos — `cd` into any repo and run those two commands.

## Features

- **Headless GPU rendering** — no display, no Omniverse
- **Hardware ray tracing** — `VK_KHR_ray_tracing_pipeline`, BLAS/TLAS, batched compaction, inline TLAS rebuild
- **CUDA-Vulkan zero-copy interop** — SSBO direct write from raygen, double-buffered, DLPack to Warp
- **Per-env SSBO layout** — `[env, H, W, 4]`, no de-tiling kernel
- **BLAS instancing** — identical geometry shares BLAS; the proto-mesh table is hash-keyed for O(1) lookup at warehouse scale
- **MaterialX PBR pipeline** — 1.39.4, per-texture sRGB selection (color slots vs normal/roughness/metallic data slots are correctly differentiated)
- **Fast mode shading** — flat diffuse for RL sensors, no shadows, no PBR
- **Raster fallback** when RT hardware unavailable
- **Multi-mode rendering** — raster, ray-query shadows, full hardware RT
- **Depth, segmentation, normals SSBO output** — for sensor simulation and downstream RL
- **LiDAR / radar internals** — async + GPU-interop raycast primitives behind the OVRTX-facing path
- **USD scene loading** via nanousd (USDA / USDC) — meshes, point instancers, lights, BasisCurves loader (Phase 11.A.1)
- **Python bindings** (ctypes)

## Architecture

```
include/nusd_renderer.h         private backend ABI
src/
  renderer.c                    API impl, scene/curve registries, BLAS instancing
  gpu_vulkan.c                  Vulkan backend (RT pipeline, BLAS/TLAS, interop)
  scene.c                       USD loader (meshes, lights, BasisCurves)
  scene.h                       SceneMesh, SceneLight, SceneCurve, SceneCurveSegment/Aabb
  camera.c                      View/projection matrices
  material.cpp                  MaterialX → GLSL → SPIR-V via shaderc
  shaders/
    raytrace_tiled.rgen.glsl    Tiled multi-camera raygen (SSBO direct write)
    raytrace.rchit.glsl         Closest-hit (fast_mode + PBR paths)
    raytrace.rmiss.glsl         Miss shader (sky/ground, shadow queries)
    mesh.{vert,frag}.glsl       Raster fallback
python/
  ovrtx/                        public OVRTX-compatible facade
  nusd_renderer/
    _bindings.py                private ctypes wrapper
    _cuda_interop.py            CUDA-Vulkan zero-copy (cuImportExternalMemory)
test/
  test_headless_render.c        Triangle smoke test (raster + RT)
  test_scene_curves.c           BasisCurves loader smoke test (Phase 11.A.1)
  bench_*.py / test_*.py        Performance + correctness suite
```

## Building

```bash
cd nanousd-vulkan-renderer
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cmake --install build --prefix ../.local/
```

Prerequisites: Vulkan SDK with `glslc` and `libshaderc`, [nanousd](../nanousd) (branch `main`). If `glslc` is not on `PATH`, point CMake at an installed Vulkan SDK copy:

```bash
cmake -B build -DGLSLC=/path/to/VulkanSDK/bin/glslc
```

The build symlinks `libnanousd.so` (the runtime backend nanousd loads via `dlopen`) into `build/` automatically, so test binaries run from the build tree without `LD_LIBRARY_PATH`.

### Native Windows smoke test

Use an x64 MSVC developer shell, or run the repo's `bldenv.bat` first if you
have the fleet build environment installed. Install the Vulkan SDK and make
sure `VULKAN_SDK` points at it.

One-command smoke path:

```powershell
.\tools\windows_smoke.ps1
```

The script uses CMake's default Windows generator. To force Ninja, pass
`-Generator Ninja`; to force Visual Studio, pass the corresponding CMake
generator name.

Manual equivalent:

```powershell
cd C:\src\github\nanousd-labs\nanousd-vulkan-renderer
$env:VULKAN_SDK = "C:\VulkanSDK\1.4.350.0"

cmake -S ..\nanousd -B ..\nanousd\build-win `
  -DCMAKE_BUILD_TYPE=Release `
  -DNANOUSD_BUILD_TESTS=OFF
cmake --build ..\nanousd\build-win --config Release --target nanousd nanousdapi

cmake -S . -B build-win `
  -DCMAKE_BUILD_TYPE=Release `
  -DNANOUSD_DIR=..\nanousd `
  -DGLSLC="$env:VULKAN_SDK\Bin\glslc.exe" `
  -DNUSD_ENABLE_OPENMP=OFF
cmake --build build-win --config Release --target test_headless_render

.\build-win\test_headless_render.exe
```

On Windows, `NU_BUILD_GS_RT` and `NUSD_ENABLE_PTEX` default to `OFF` for this
minimal path. The build also looks for `shaderc_shared.lib` under
`%VULKAN_SDK%\Lib` and copies discovered `nanousd.dll` / `nanousdapi.dll`
runtime dependencies into `build-win`.

To test USD scene loading after the triangle passes:

```powershell
.\build-win\test_headless_render.exe .\tests\assets\capsule_z.usda capsule.ppm
```

If MaterialX fetches are slow or unreliable on your network, pre-populate a
local MaterialX checkout/tarball extraction and add
`-DFETCHCONTENT_SOURCE_DIR_MATERIALX=C:\path\to\MaterialX-1.39.4` to the
renderer configure command.

## Usage

```python
import numpy as np
import ovrtx

renderer = ovrtx.Renderer(ovrtx.RendererConfig())
renderer.open_usd_from_string("""#usda 1.0
(
    subLayers = [@scene.usda@]
)

def "Render" {
    def RenderProduct "Camera" {
        int2 resolution = (1280, 720)
        rel camera = </Camera>
        rel orderedVars = [<LdrColor>]
        def RenderVar "LdrColor" { string sourceName = "LdrColor" }
    }
}
""")

outputs = renderer.step({"/Render/Camera"}, 0.0)
with outputs["/Render/Camera"].frames[-1].render_vars["LdrColor"].map(ovrtx.Device.CPU) as mapped:
    pixels = np.from_dlpack(mapped)
```

Use `open_usd()` / `open_usd_from_string()` for the root stage. Use
`add_usd_reference()` / `add_usd_reference_from_string()` only for additive
references that need removable handles.

### IsaacLab integration

IsaacLab should not import `nusd_renderer` directly. Use IsaacLab's normal
`ovrtx_renderer` preset and place the nanousd OVRTX facade ahead of the
official OVRTX wheel:

```bash
export PYTHONPATH=/path/to/nanousd-vulkan-renderer/python:/path/to/nanousd-opengl-renderer/python:$PYTHONPATH
export NANOUSD_OVRTX_BACKEND=vulkan   # or opengl/metal when those backend packages are present
```

The IsaacLab side remains OVRTX-shaped. Backend selection is owned by the
facade, so public IsaacLab can run unchanged as long as its `isaaclab_ov`
package is installed and imports `ovrtx.Renderer`.

## Runtime environment

```bash
export NANOUSD_BACKEND=/path/to/nanousd/build/Release/libnanousd.so
export NANOUSD_OVRTX_BACKEND=vulkan
export DISPLAY=:1   # required for headless Vulkan on DGXC machines
```

## Render modes

The OVRTX facade selects the best native mode for the backend: Vulkan/Metal use
hardware RT when available, and OpenGL uses raster output. Render products
choose the outputs (`LdrColor`, depth/distance, normals, segmentation, etc.).

## Testing

```bash
./build/test_headless_render                                            # triangle, raster + RT
./build/test_scene_curves tests/assets/native_instance_curve_root.usda  # BasisCurves loader

# Microbenchmarks (Python)
python test/bench_render.py
python test/bench_zerocopy_breakdown.py
python test/bench_tiled_comprehensive.py
python test/bench_tlas_update.py
```

## Contributing

This project is currently not accepting contributions.
