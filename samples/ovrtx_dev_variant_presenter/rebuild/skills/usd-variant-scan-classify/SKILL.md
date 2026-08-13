---
name: usd-variant-scan-classify
description: >
  Scan a USD stage for variant sets and cameras, and classify each variant set by HOW its
  variants differ (shader-input = live "fast path" vs transform/visibility/binding/
  structural = needs a reopen) so a live viewer can switch most variants without reloading.
  Also extract a representative color swatch per variant. Use when building variant
  switching / a configurator on top of an ovrtx viewer.
license: Apache-2.0
metadata:
  author: NVIDIA Customer Success
  tags: [usd, variants, pxr, classification]
  domain: ai-ml
  languages: [python]
---

# Scan + classify variant sets

Two pure-`pxr` passes (no ovrtx). Both run under the process-global `USD_LOCK` (see the
orchestrator's stability checklist) because they open/compose stages.

## 1. Scan (`scan_stage(usd_path, extra_sublayers=()) -> StageInfo`)

Open the user stage read-only (compose with any extra sublayers, e.g. the turntable
sidecar). Walk it and collect:

- **Variant sets:** for each prim with variant sets, one record `{set_name, prim_path,
  variants: [names], current: selected}`. A prim can hold several sets; a set name can
  recur on different prims — key by `(prim_path, set_name)`.
- **Cameras:** every `UsdGeom.Camera` → `{path, name (leaf), animated}`. `animated` =
  the camera's xform (or an ancestor's) has time samples, OR it sits under an animated rig.
- **Stage metadata:** `up_axis`, `meters_per_unit`, `timeCodesPerSecond` (fps),
  `startTimeCode`/`endTimeCode`. **Hoist the timebase from the user stage** so the renderer
  and UI agree (a bare composite root defaults to 24 fps and silently rescales sublayer
  time — see the stability checklist's animated-camera note).

Return this as the `/api/open` and `/api/stage` payload.

## 2. Classify (`classify_variants(usd_path) -> {set_name: VariantEffect}`)

Run **off the render thread** (a daemon thread, emitting a `classified` WS event when done)
— it composes the stage ~once per variant and can take tens of seconds on a big stage.
Switches that arrive before it finishes take the reload path, then flip to fast.

For each variant set, compute the **union-absolute** effect:

1. For each variant of the set, select it and record the authored opinions it produces.
2. Take the **union** of `(prim_path, attribute)` that change across ANY variant of the set.
3. For each variant, record the **absolute** value it authors for every attr in that union
   (or the base value if that variant doesn't touch it) — so switching to a variant writes
   the FULL state, including reverting attrs a previous variant changed.

Classify the set by the KIND of the changed attrs:

| Kind | Trigger | Switch mechanism |
|---|---|---|
| **shader_input** (FAST) | only `inputs:*` on shader prims change (e.g. `inputs:diffuse_reflection_color`, `inputs:coat_color`, `inputs:metalness`) | attr-replay (no reopen) |
| **transform** | `xformOp:*` / `omni:xform` changes | reopen (or write `omni:xform` + reset) |
| **visibility** | `visibility` changes | reopen (or write `visibility` + reset) |
| **binding** | `material:binding` changes | reopen |
| **structural** | sublayers / lights / references / activation change | reopen |

A set is **fast-path** only if EVERY changed attr across its variants is a convertible numeric
shader input. **Accept the full numeric family — `color3f`/`float3`, `float`, AND `float2`/`Vec2f`
(and `float4`)** — NOT just color3f/float. A Carpaint/Wheel set drives inputs like
`inputs:roughness_range_position` (a `float2`/`Vec2f`); if the classifier only allows color3f/float it
mis-classifies that set as reload (wrong 8/5 split, sluggish switches). Write layout: a vector input is
ONE multi-lane element (`lanes=3` for color3f/float3, `2` for float2/Vec2f, `4` for float4, always
`shape=[1]`); a scalar float is a bare `(1,)` array. See `usd-variant-live-switching` — a bare `(1,3)`
array is the ovrtx-0.3 form and silently drops every switch to reload. Anything non-numeric
(bool/token/asset/string) → reload. Record per set:
`{kind, fast: bool, writes: {variant: [(prim_path, attr, value), ...]}}`.

On the ConceptCar this yields **8 of 13 fast** (Carpaint, Int_Trim_Color, Int_leather,
Int_leather_dash, Light_strip_color, Screen_Color, Stitch_Color, Wheel_Colors) and 5 reload
(Doors, Wheel_Turns, Headrests, frontLicensePlate, Backdrops).

## 3. Swatches (representative color per variant) — free, from the classification

For a chip UI, pick the `color3f` shader input that **varies MOST across the set's
variants** (not just `diffuse_reflection_color` — for some sets diffuse is constant while
`diffuse_tint`/`coat_color`/`emissive_color` is what changes). Name-hint by
color/emissi/tint/diffuse/coat/base. Convert **linear→sRGB** before emitting the hex (USD
shader colors are linear; skipping this renders dark paints as near-black mud). Ship
`swatches: {set_name: {variant: "#rrggbb"}}` in the `classified` event.

## Gotchas
- Hold `USD_LOCK` per set and release between sets (and for pure-Python work) so the render
  thread waits at most one set's worth of time.
- Classify against a freshly composed stage, not the live render stage.
- Write layout (float32): every VECTOR input is one multi-lane element — `lanes=len(components)`,
  `shape=[1]`, built with `ovstage.make_dltensor`; a scalar float is a bare `(1,)` array. Accept the
  whole numeric family on the fast path; only `bool`/token/asset/string/unconvertible inputs force
  reload. (A `float2` like `inputs:roughness_range_position` IS fast — don't drop it to reload.)
  Bare `(N,C)` numpy arrays are the ovrtx-0.3 form: they declare `lanes=1`, ovstage rejects the write,
  and the fast path falls back to reload while still rendering correctly, so the loss is invisible.
