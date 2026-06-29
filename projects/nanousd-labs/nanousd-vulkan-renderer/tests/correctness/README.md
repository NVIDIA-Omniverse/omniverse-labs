# nanousd-vulkan-renderer correctness tests

Regression-style golden-image tests. Each test renders an asset twice
(Vulkan + ovrtx) and compares against checked-in golden images.

## Tests

| Asset | Modes tested |
|---|---|
| `chess_king_black` | rt, shadow, raster — OpenChessSet via MaterialX, 17 meshes |
| `agibot_scanner` | rt, shadow, raster — agibot SimReady scanner via OmniPBR/MDL, 4682 tri |
| `agibot_material_tray` | rt, shadow, raster — agibot SimReady tray via OmniPBR/MDL |

Each asset is exercised in all three Vulkan render modes:

| Mode | Pipeline | Lighting |
|---|---|---|
| `rt` (`NU_RENDER_RT`) | Hardware ray tracing pipeline | Path-traced + IBL (MaterialX / OmniPBR materials) |
| `shadow` (`NU_RENDER_SHADOW`) | Raster + ray-query shadows | Three-point lit + RT-traced shadow rays |
| `raster` (`NU_RENDER_RASTER`) | Classic raster pipeline | Three-point lit (key + fill + rim) |

Test labels are `<asset>_<mode>` (e.g. `agibot_scanner_shadow`).
9 tests total. Suite finishes in ~30s.

Each test verifies three thresholds:

| Comparison | Default RMS limit | What it catches |
|---|---|---|
| **Vulkan vs Vulkan-golden** | 2.0 | Regressions in our renderer |
| **ovrtx vs ovrtx-golden** | 2.0 | Drift in the ovrtx wrapper / GPU driver |
| **Vulkan vs ovrtx (current)** | 60-100 | Sanity check vs reference reality |

Per-test thresholds in `run_tests.py::TESTS` override the cross-renderer
limit where ovrtx can't render the asset correctly (chess: ovrtx UJITSO
fails on multi-node MaterialX → uniform-red fallback, so the
cross-renderer threshold is effectively disabled).

## Goldens

Stored in `golden/` and checked into the repo. Per-asset ovrtx golden is
shared across the three modes (ovrtx is always path-traced, regardless of
which Vulkan pipeline we're testing). Layout:

```
golden/
  chess_king_black_rt_vulkan.png        (RT mode)
  chess_king_black_shadow_vulkan.png    (shadow mode)
  chess_king_black_raster_vulkan.png    (raster mode)
  chess_king_black_ovrtx.png            (ovrtx, shared across modes)
  agibot_scanner_{rt,shadow,raster}_vulkan.png
  agibot_scanner_ovrtx.png
  agibot_material_tray_{rt,shadow,raster}_vulkan.png
  agibot_material_tray_ovrtx.png
```

After a deliberate change (new shader, loader feature, etc.), regenerate
goldens with `python3 run_tests.py --update`. Inspect the diff visually
before committing.

## Reproducibility

```bash
cd nanousd-vulkan-renderer
python3 tests/correctness/run_tests.py             # run + verify
python3 tests/correctness/run_tests.py --update    # rewrite goldens
python3 tests/correctness/run_tests.py --only chess  # filter
```

Returns nonzero on FAIL, suitable for CI. Suite finishes in ~30s.

## Determinism

When the test was first run, agibot_material_tray showed RMS 87 against
its own freshly-captured Vulkan golden — i.e. running the same render
twice produced different outputs. Root cause:
`material.cpp::collect_root_layer_reference_dirs` cached anchor
directories keyed by `NanousdStage` pointer, and stage handles get
recycled across scenes (each `nu_load_usd` creates and later frees one),
so the cache returned chess's anchors for material_tray. Fixed by
keying the cache on the root-layer path string instead.

The current suite is byte-identical across runs (RMS 0.00 for all 3
Vulkan-vs-golden checks).

## What this rig does NOT cover

- Direct lighting (no analytic lights in the test scenes)
- Material variants
- Multi-asset scenes (1 asset per test here)
- Animation / time-varying state
