# USD Viewer (ovrtx viewport)

Small **PySide6** application that uses **ovrtx** to render an **OVRTX-ready** USD stage and shows the result in a live window. It maps **`HdrColor` first** (float linear → sRGB uint8, same idea as [vulkan-interop](../../c/vulkan-interop/)), then falls back to **`LdrColor`**.

This is not an embedded ovrtx widget: ovrtx renders offscreen; the app displays the raster result (same idea as [minimal](../minimal/), but continuous instead of `Image.show()`).

For **GPU-present**, **swapchain display**, and **interactive orbit** at higher frame rates, see the C sample [vulkan-interop](../../c/vulkan-interop/).

## Prerequisites

- Python 3.10–3.13
- [uv](https://docs.astral.sh/uv/)
- A **GPU** and environment suitable for running ovrtx (see the main repo README)

## USD requirements

The stage must contain at least one **OVRTX-valid** `RenderProduct` with **`HdrColor` and/or `LdrColor`**. By default this app **auto-picks** a path after each load by **probing** common products. For **remote** URLs and unknown files it tries **`/Render/Camera`** first (same as [minimal](../minimal/)), then typical **Kit** `HydraTextures` paths. For a **local `.usda`** file that defines the Kit viewport RenderProduct but **no** `RenderProduct "Camera"` (like `three_spheres.usda`), it probes **Kit first** so ovrtx does not log sensor errors for a non-existent `/Render/Camera`. If none respond with a color buffer, it falls back to **`/Render/Camera`**; use **`--render-product` / `-r`** to force a path. For exotic stages you can try **`--use-stage-dump`** (runs `ovrtx_debug_dump_stage`; slower and **unstable on some Windows builds**). Many scenes (including the default Robot-OVRTX URL) use **`HdrColor`**; **`LdrColor` alone may look black** if the stage does not populate it.

Arbitrary USD files without a matching `RenderProduct` and camera setup will not work until you extend the scene (for example with an inline layer as in [`skills/loading-usd`](../../../skills/loading-usd/SKILLS.md)). For a short guide on authoring or grafting that setup, see **[`STAGE_SETUP.md`](STAGE_SETUP.md)**.

## Performance note

Each frame copies the chosen color buffer to host memory for Qt. That is simple and portable but can limit interactive **FPS** compared to the Vulkan + CUDA path in **vulkan-interop**.

## Troubleshooting

| Symptom | Things to check |
|--------|-------------------|
| **Black viewport**, load succeeds | Often fixed by displaying **`HdrColor`** (this app prefers it over `LdrColor`). If you still see black, confirm the chosen render product (stderr: `Using render product:`) matches a prim that outputs color, or set **`-r`** explicitly. |
| **`Invalid USD RenderProduct` / sensor prim** | The stage’s `RenderProduct` must be OVRTX-valid (typically Kit **`…/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0`** with **`Render/Vars/LdrColor`**). A minimal `def RenderProduct "Camera"` under `/Render/Camera` is usually **not** enough. |
| **Dialog: no HdrColor/LdrColor** | Your render product’s `orderedVars` / outputs may use different names; inspect the stage or use `ovrtx_debug_dump_stage` patterns from other examples. |
| **Crash in `QImage::copy`** | Do not use mapped ovrtx memory after the `map()` context exits; keep a **numpy `.copy()`** inside `with` (this example does). |
| **Native crash right after `USD loaded.`** | Default auto-detect avoids `ovrtx_debug_dump_stage` for the first step; do not pass **`--use-stage-dump`** unless you need it. Force **`-r /Render/Camera`** for Robot-like scenes if probing still misbehaves. |

## PhysX (ovphysx) in a separate process

**Simulation → Run PhysX in separate process…** starts [`physx_subprocess_sim.py`](physx_subprocess_sim.py) with **`sys.executable`**, so **only ovphysx** loads in that child. The main window stays **ovrtx-only**, avoiding Carbonite / USD plugin clashes between the two SDKs.

Do **not** run `ovphysx.PhysX()` in the same process as `ovrtx.Renderer()` after USD has loaded: PhysX Carbonite plugins (e.g. `omni.physx.plugin`, tensor APIs) typically fail to resolve (`IUsdPhysics`, `IPhysx`, “PhysX plugins could not be loaded”). Real-time stepping in the viewer would need a different architecture (IPC to a dedicated PhysX worker, or a full Kit/Isaac stack with extensions loaded in order).

Requirements:

- The stage must be a **local file path** (not `http(s)://` or `omniverse://`). Use **File → Open…** or pass a path on the command line.
- The child uses the same Python environment as the viewer (e.g. `uv run main.py`).

On success, the worker writes a **JSON** payload: link prim paths, **world** 4×4 matrices from PhysX, and an **`articulation_world_matrix4d`** (defaults to identity for a fixed articulation root with no authored root xform). The viewer converts to **local** transforms and applies them with **`map_attribute`** on **`omni:xform`** (`Semantic.XFORM_MAT4x4`), because writing **`omni:fabric:worldMatrix`** is deprecated in current ovrtx builds. Use **Simulation → Reload scene (clear PhysX transforms)** to reload the current source and drop in-memory overrides. This avoids **`add_usd_layer`** with a `path_prefix` under an existing articulation prim (see [`skills/loading-usd`](../../../skills/loading-usd/SKILLS.md)).

### Live PhysX (articulation stream)

**Simulation → Live PhysX (articulation stream)…** spawns [`physx_live_worker.py`](physx_live_worker.py) in a **separate process** (ovphysx only). The worker loads the same local USD as the viewer, runs `PhysX.step` in a loop like [`hello_world_physx.py`](../ov-libaries-livestream/hello_world_physx.py), and prints **one compact JSON line per step** on stdout (same **version 1** schema as the batch worker). Each frame, the viewer parses the latest complete line, applies poses via **`map_attribute`**, then calls **`renderer.step`**—the same **update transforms → step renderer** order as [planet-system](../planet-system/), with transforms sourced from PhysX tensors instead of Warp.

- Requires a **local** stage file (e.g. [`links_chain_sample.usda`](../ov-libaries-livestream/links_chain_sample.usda)); default articulation root in the dialog is `/World/articulation` when that file is open.
- Toggle the menu item off (or reload USD) to stop the worker. Do not run **live** and **batch** PhysX subprocesses at the same time.

Minimal PhysX-only smoke test (no viewer):

```bash
cd examples/python/ov-libaries-livestream
uv run hello_world_physx.py
```

Or run the worker directly:

```bash
cd examples/python/usd-viewer-example
uv run python physx_subprocess_sim.py --usd ../ov-libaries-livestream/links_chain_sample.usda --steps 120 --out-json physx_poses.json
```

To exercise the live worker from the command line (JSONL on stdout):

```bash
cd examples/python/usd-viewer-example
uv run python physx_live_worker.py --usd ../ov-libaries-livestream/links_chain_sample.usda --max-steps 5
```

### Contact Binding

**Simulation → Run Contact Binding…** behavior depends on the open stage:

- **[`boxes_falling_on_groundplane.usda`](../ov-libaries-livestream/boxes_falling_on_groundplane.usda)** — Spawns [`physx_rigid_live_worker.py`](physx_rigid_live_worker.py) (ovphysx in a child process). It streams **version 1** rigid-body poses on stdout; the viewer applies them each frame like **Live PhysX (articulation)** so you can **see the cubes fall**. Use **Simulation → Reload scene** (or close the app) to stop the stream.
- **Any other file** — Runs the contact-force tutorial via [`contact_binding_subprocess.py`](contact_binding_subprocess.py) → [`../ov-libaries-livestream/contact_binding.py`](../ov-libaries-livestream/contact_binding.py) (log output only; no viewport motion).

```bash
# Falling boxes (same worker the viewer uses for that USDA)
cd examples/python/usd-viewer-example
uv run python physx_rigid_live_worker.py --usd ../ov-libaries-livestream/boxes_falling_on_groundplane.usda --rigid-body-paths /World/Cube1 /World/Cube2 --max-steps 10

# Contact-force tutorial only
uv run python contact_binding_subprocess.py --usd ../ov-libaries-livestream/boxes_falling_on_groundplane.usda
```

## Running

```bash
cd examples/python/usd-viewer-example
uv run main.py
```

Default scene: Robot-OVRTX (remote URL). First run may take noticeable time while shaders compile and cache.

A tiny **local** example (three spheres) lives in **`three_spheres.usda`**. The viewer **auto-detects** a usable `RenderProduct` after each load (probes common paths), so you can use **File → Open…** after starting on the default Robot URL without passing `-r`:

```bash
uv run main.py
# then File → Open… → three_spheres.usda
```

Override detection when needed:

```bash
uv run main.py --render-product "/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0"
```

When you use **File → Open…**, the app **`remove_usd`s** the previous layer, then calls **`reset_stage()`** and **`reset()`** before loading the next file so RTX does not keep stale render-product state. (Those calls are **not** run before the very first load—doing so on an empty renderer could crash during the first `step()`.) If problems persist, restart the viewer once.

Open a local file from **File → Open…**, or pass a path on the command line:

```bash
uv run main.py path/to/scene.usda
```

Force a specific render product (disables auto-detect):

```bash
uv run main.py --render-product /Render/Camera path/to/scene.usda
```

## Package index

`ovrtx` and `ovphysx` are available on [PyPI](https://pypi.org). No additional index configuration is required; `uv run` resolves them automatically.
