<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->


# ovrtx + ovphysx Livestream

Live RTX viewport with physics simulation streaming from a separate process.

**Status:** Working

---

## Overview

This sample shows how to build a desktop application that combines RTX rendering
(`ovrtx`) with physics simulation (`ovphysx`). The two libraries cannot run in
the same Python process after USD has loaded (Carbonite plugin conflicts), so the
sample uses a **two-process architecture**: the viewer holds the GPU for ovrtx
while a worker process runs PhysX on CPU and streams world-pose JSON lines back
over stdout.

The sample ships as two sibling Python packages:

| Package | Purpose |
|---------|---------|
| `ov-libaries-livestream/` | Six standalone tutorials — minimal render, depth map, prim clone, PhysX smoke test, tensor bindings, and contact forces. Each runs without the viewer. |
| `usd-viewer-example/` | PySide6 desktop viewer with **ovrtx → File** and **ovphysx → Simulation** menus that wire each tutorial to a Play button. |

---

## Libraries used

| Library | Role in this sample |
|---------|---------------------|
| `ovrtx` | Renders the USD scene offscreen via RTX ray tracing; `open_usd()` loads the stage, `step()` produces color and depth buffers, `map_attribute()` applies PhysX transforms |
| `ovphysx` | Runs PhysX simulation (articulation, rigid body, contact forces) in a child process; streams world-pose JSONL on stdout |
| `pyside6` | Desktop window, ~30 Hz Qt render loop, menus, and log panel |
| `pillow` | Saves PNG frames in the standalone tutorial scripts |
| `numpy` | Pixel buffer conversion and transform math |

---

## Architecture

ovrtx and ovphysx cannot coexist in the same Python process once USD is loaded.
The viewer keeps the GPU for ovrtx; each PhysX worker is spawned as a child with
`PhysX(device="cpu")` so the two stacks never share CUDA or Carbonite plugins.

```
Viewer process  (ovrtx + PySide6)
  open_usd(path)
  _on_frame() @ ~30 Hz:
    apply transforms via map_attribute(omni:xform)
    renderer.step()
    map HdrColor → numpy → QImage → display
              ▲
              │  JSONL on stdout  (version-1 pose schema)
              │
PhysX worker  (ovphysx only, device="cpu")
  physx.add_usd(path)
  loop: physx.step() → read tensors → print JSON line → stdout
```

**Transform order matters:** apply transforms first, then call `renderer.step()`
—the same pattern as the `planet-system` sample with Warp, but with PhysX
running out-of-process instead of in-process.

The shared pose wire format (`physx_pose_utils.py`) serialises world 4×4 matrices
from PhysX tensors into a versioned JSON object. The viewer converts to local
transforms before calling `map_attribute`.

---

## Prerequisites

- Python 3.10–3.13
- [uv](https://docs.astral.sh/uv/) package manager
- A CUDA-capable NVIDIA GPU with drivers suitable for ovrtx
- Windows (tested); Linux not verified on this sample

---

## Setup

1. Clone the repository (if not already):

   ```bash
   git clone https://github.com/nvidia-omniverse/omniverse-labs.git
   cd omniverse-labs/samples/ovrtx_ovphysx_livestream/livestream-files/examples/python
   ```

2. Install the viewer environment:

   ```bash
   cd usd-viewer-example
   uv sync
   ```

3. Install the standalone tutorials (required for PhysX menu items inside the viewer, and to run tutorials independently):

   ```bash
   cd ../ov-libaries-livestream
   uv sync
   ```

Both environments use committed `uv.lock` files; no network resolution is
needed after the first install.

---

## Usage

### USD Viewer

```bash
cd examples/python/usd-viewer-example
uv run main.py
```

The default scene is the Robot-OVRTX remote URL. First run compiles and caches
shaders — expect a noticeable delay.

Open a local file from **File → Open…** or pass a path on the command line:

```bash
uv run main.py path/to/scene.usda
```

Force a specific render product when auto-detection is wrong:

```bash
uv run main.py --render-product /Render/Camera path/to/scene.usda
```

#### Simulation menu

| Menu item | What it does | Required stage |
|-----------|-------------|----------------|
| **ovphysx → Articulation** | Streams articulation link poses; viewer applies them live | `links_chain_sample.usda` |
| **ovphysx → Tensor Binding** | Streams DOF targets + link poses via tensor API | `links_chain_sample.usda` |
| **ovphysx → Contact Binding** | Shows falling cubes (rigid stream) or contact-force log | `boxes_falling_on_groundplane.usda` |
| **ovphysx → Clone** | Streams PhysX clone + rigid-body poses | `basic_simulation.usda` |
| **ovrtx → Clone** | Clones the robot prim in-renderer and offsets it | Robot-OVRTX URL |
| **ovrtx → Depth map** | Renders depth AOV to a false-colour PNG | `robot_with_depth.usda` |

Use **Simulation → Reload scene** to drop in-memory PhysX transforms and reload
the current USD without restarting the viewer.

---

## Standalone tutorials

Each tutorial in `ov-libaries-livestream/` runs independently without the viewer.

### Minimal render to PNG

```bash
cd examples/python/ov-libaries-livestream
uv run example.py
```

Loads the Robot-OVRTX remote URL, steps the renderer once, saves `LdrColor`
to `last_render.png`, and opens it.

### Depth map

```bash
uv run depth_map_example.py
```

Renders `DepthSensorDistance` from `robot_with_depth.usda`, applies optional
hole-fill, and saves a false-colour depth PNG.

### Prim clone

```bash
uv run clone_example.py
```

Loads the Robot-OVRTX scene, clones `/World` to a new prim, offsets it by +X
with uniform scale, and saves a rendered PNG showing both copies.

### PhysX smoke test

```bash
uv run hello_world_physx.py
```

Loads `links_chain_sample.usda` into `PhysX(device="cpu")` and runs a single
simulation step. Useful for verifying ovphysx is installed correctly — no GPU
required.

### Tensor bindings

```bash
uv run tensor_bindings.py
```

Demonstrates the `TensorType.ARTICULATION_LINK_POSE` API against
`links_chain_sample.usda`. Pass `--viewer-stream` to emit JSONL on stdout for
the viewer to consume.

### Contact binding

```bash
uv run contact_binding.py --usd boxes_falling_on_groundplane.usda
```

Applies `PhysicsContactReportAPI` and logs per-frame contact forces.

---

## Assets

The `ov-libaries-livestream/` directory includes purpose-built USDA stages:

| File | Used by |
|------|---------|
| `links_chain_sample.usda` | Articulation and tensor-binding tutorials; has a `RenderProduct` for the viewer |
| `boxes_falling_on_groundplane.usda` | Rigid-body / contact-binding tutorials |
| `robot_with_depth.usda` | Depth-map tutorial; includes an `OmniSensorDepthSensorSingleViewAPI` prim |
| `robot_camera_renderer_depth.usda` | Alternate depth stage with camera render product |
| `basic_simulation.usda` | Clone + PhysX clone tutorials; includes a dome light |

All stages require an **OVRTX-valid `RenderProduct`** with `HdrColor` and/or
`LdrColor` to display in the viewer. See
[`usd-viewer-example/STAGE_SETUP.md`](livestream-files/examples/python/usd-viewer-example/STAGE_SETUP.md)
for the authoring checklist.

---

## Known Limitations

- **No same-process co-loading.** Running `ovphysx.PhysX()` and `ovrtx.Renderer()`
  in the same process after USD has loaded causes Carbonite plugin errors
  (`IUsdPhysics`, "PhysX plugins could not be loaded"). The subprocess
  architecture is the correct workaround; in-process PhysX would require a full
  Kit/Isaac extension stack with extensions loaded in a specific order.
- **Local USD only for PhysX workers.** Workers require a filesystem path;
  `http(s)://` and `omniverse://` URIs are not supported by `ovphysx.add_usd()`.
- **Windows only (tested).** The sample has not been validated on Linux.
- **`var.tensor` deprecation.** In ovrtx 0.3.x, `render_var.tensor.numpy()` is
  deprecated in favour of `np.from_dlpack(render_var)`. The tutorials still use
  the old form and will emit a `DeprecationWarning`; it will be removed in a
  future ovrtx release.
- **Depth output is sparse by default.** `DepthSensorDistance` from the Kit
  viewport sensor often has unfilled pixels. The `depth_map_example.py` tutorial
  includes a `dilate_depth_holes` helper, but aggressive dilation blurs edges.
  See the in-file comments for guidance.

---

## License

[Apache 2.0](../../LICENSE).  
Third-party dependency notices: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
