---
name: combining-scene-with-runtime-prims
description: Loading a base USD scene with additional runtime prims in tests. Use when a doc test needs to combine a scene file with RenderProducts or other prims that aren't in the original scene.
---

# Combining a Scene with Runtime Prims

## Overview

ovrtx only supports one root layer per renderer. Documentation tests often need a base scene (for cameras, lights, geometry, and materials) plus doc-specific prims such as `RenderProduct` or `RenderVar` definitions. For that case, use `open_usd_from_string()` / `ovrtx_open_usd_from_string()` with an inline root USDA layer that contains a `subLayers` composition arc to the base scene and authors the extra prims in the same root layer.

This inline sublayer pattern is supported and is the preferred docs-test pattern when the code example needs to demonstrate a single composed stage. Use `add_usd_reference*` only when the example specifically needs to add or remove referenced content after a root stage is already open.

## The Pattern

Build one inline root USDA string that:

1. Uses `subLayers = [@/path/to/base_scene.usda@]` to bring in the actual scene content.
2. Authors the extra doc-specific prims in the same inline root layer.
3. Is loaded with `open_usd_from_string()` in Python or `ovrtx_open_usd_from_string()` in C.

> **Python source:** `tests/docs/python/test_sensor_configuration.py` snippet `doc-add-render-config-layer`
>
> **C source:** `tests/docs/c/test_sensor_configuration.cpp` snippet `doc-add-render-config-layer-c`

## Why This Works

USD `subLayers` is a composition arc that layers multiple USD files into a single stage. The inline USDA becomes the "stronger" layer (opinions win), and the sublayered scene file provides the base content. ovrtx sees this as one root layer containing everything.

## Key Details

- **No `defaultPrim` or `prefix_path` is required for this pattern.** Those are reference-addition concerns, not root-layer sublayer concerns.
- **The inline root can author prims at absolute paths.** For example, it can sublayer `simple_camera.usda` and add `/Render/Camera` in the stronger layer.
- **Camera paths must match the scene.** If the scene has `/Camera0`, the RenderProduct's `rel camera` must point to `/Camera0`, not `/World/Camera`.
- **The shared `renderer` fixture does not reset automatically.** Use `open_usd*` to replace the root stage for each test, or call `reset_stage()` explicitly before additive reference tests.

## Common Pitfalls

- **Don't call `open_usd()` and `open_usd_from_string()` separately to combine layers.** The second call replaces the root stage, discarding the first.
- **Don't use removed APIs.** `add_usd()`, `add_usd_layer()`, and `ovrtx_add_usd()` are old names. Use `open_usd*` for root layers and `add_usd_reference*` for additive references.
- **Don't switch docs tests to `add_usd_reference*` just to combine base scenes with RenderProducts.** Inline `subLayers` keep the snippet as one root stage and match the documented pattern.
- **Check camera prim paths.** Test scenes use paths like `/Camera0`, not `/World/Camera`. If the RenderProduct points at the wrong path, the step will produce no frames.
