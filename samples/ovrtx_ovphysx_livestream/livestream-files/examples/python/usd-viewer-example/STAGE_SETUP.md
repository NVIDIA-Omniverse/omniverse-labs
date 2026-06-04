<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Setting up USD for the ovrtx USD viewer

This viewer only displays stages that **ovrtx** can render: you need a valid **`RenderProduct`** and at least one color **`RenderVar`** named **`HdrColor`** or **`LdrColor`**. Generic Hydra or GL-only USD often will not work until you add that render graph.

## Checklist

1. **Camera** — A `UsdGeom.Camera` (or equivalent) somewhere on the stage, with a sensible transform.
2. **Render product** — A prim typed as **`RenderProduct`** (or Kit export equivalent) that ovrtx treats as an RTX view. It must **`rel camera`** point at your camera prim.
3. **Color output** — Under that render product (or via `rel orderedVars`), wired **`RenderVar`** prims whose **`uniform string sourceName`** is **`HdrColor`** and/or **`LdrColor`**. The viewer reads those names only.
4. **Optional: `RenderSettings`** — Many Kit-style scenes set **`rel products`** on a `RenderSettings` prim to the same render product you will step.

If any of this is missing, load may succeed but you get errors, a black image, or “no HdrColor/LdrColor”.

## Two patterns that work

### A. `/Render/Camera` (simple Robot-style)

Used by the default **Robot-OVRTX** sample and [minimal](../minimal/) example.

- Path is typically **`/Render/Camera`** (a `RenderProduct` named `Camera` under `/Render`).
- **`rel camera`** targets your shot camera (e.g. `</World/Camera>`).
- **`rel orderedVars`** (or nested vars) include **`HdrColor`** and/or **`LdrColor`** with matching `sourceName`.

Your **geometry and lights** live under something like `/World`; the **render prims** live under `/Render`. See Omniverse / OVRTX sample content for full detail.

### B. Kit viewport render product (local “three spheres” style)

Many Omniverse Kit exports use a long path such as:

`/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0`

That prim is still a **`RenderProduct`**, with **`rel camera`**, **`rel orderedVars`** (often pointing at `/Render/Vars/LdrColor` or similar), plus RTX-related attributes. Copy the structure from:

- **[`three_spheres.usda`](three_spheres.usda)** in this folder (small, self-contained), or  
- **[`examples/c/minimal/torus-plane.usda`](../../c/minimal/torus-plane.usda)** (commented alternate shows a minimal `/Render/Camera` block you can adapt).

**Important:** If you only use pattern B, do **not** invent **`/Render/Camera`** unless you also define that `RenderProduct`. Stepping a non-existent camera render product produces sensor / scheduler errors in the log.

## Opening your file in the viewer

- **Auto-detect** probes common paths (`/Render/Camera`, then typical Kit paths, with a special case for local `.usda` files that look Kit-only—see [README](README.md)).
- If auto-detect picks the wrong product or your path is unusual, pass **`-r` / `--render-product`** with the full Sdf path to your `RenderProduct`.

```bash
uv run main.py -r "/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0" ./my_scene.usda
```

## Adding render setup without editing the original asset

You can **`subLayer`** or **`add_usd_layer`** extra USDA that only adds `/Render` (and optionally a camera). See [`skills/loading-usd`](../../../skills/loading-usd/SKILLS.md) for `add_usd`, `add_usd_layer`, and `path_prefix`.

## Quick sanity check

After load, stderr should show something like:

`Using render product: '/Render/...'`

If the viewport is black, confirm that product actually exposes **`HdrColor`** (preferred by the viewer) or **`LdrColor`** for your RTX settings—some stages only fill one of them.
