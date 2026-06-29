# Backend comparison: chess

## What is compared

- **OVRTX 0.3** (reference, **reused**): NVIDIA OVRTX path tracer. The `*_ovrtx.png` reference frames are reused (committed), not rendered by this harness.
- **Metal RT**: local `nusd_renderer` `NuRenderer(enable_rt=True)`, `render(NU_RENDER_RT)` — hardware ray tracing on Metal.
- **Metal Raster**: local `nusd_renderer` `NuRenderer(enable_rt=False)`, `render(NU_RENDER_RASTER)` — rasterizer.

- **Resolution**: 768x768 (**square**, for camera parity). The native Metal backend treats `fov_degrees` as the vertical FOV and derives horizontal FOV from the aspect; OVRTX derives its projection from focal_length + horizontal/vertical aperture (authored equal). At a square aspect (1.0) hfov==vfov in both, so the subjects co-register — which is what makes the reused OVRTX references valid.
- **Cameras**: two angles per asset, set programmatically on the Metal backend. Chess and the Apple assets use bbox-framed angles (camA front three-quarter, camB higher/opposite). The warehouse uses explicit interior look-at cameras at forklift/eye height.
- **Lighting rig (shared)**: constant-color `DomeLight` (no HDR) + Key + Fill `SphereLight` positioned from the asset bbox. The wrapper *sub-layers* the asset (so material bindings survive) — byte-identical to the wrapper that produced the reused OVRTX references.

![contact sheet](contact_sheet.png)

## Metrics vs OVRTX reference

RMS / MAE are over 8-bit sRGB pixels; silhouette IoU compares foreground masks (background-delta) between each Metal backend and the OVRTX reference.

| Asset | Cam | RT RMS | RT MAE | RT IoU | Raster RMS | Raster MAE | Raster IoU | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| chess_set | camA | 19.9 | 5.7 | 0.977 | 28.7 | 23.6 | 0.979 | ok |
| chess_set | camB | 14.5 | 4.1 | 0.993 | 25.3 | 20.4 | 0.993 | ok |

### Mean RGB (black-frame sanity)

| Asset | Cam | OVRTX mean RGB | Metal RT mean RGB | Metal Raster mean RGB |
| --- | --- | --- | --- | --- |
| chess_set | camA | (166.9, 170.1, 175.4) | (163.8, 167.2, 171.7) | (171.7, 182.7, 203.7) |
| chess_set | camB | (155.6, 158.9, 163.1) | (155.2, 158.9, 162.4) | (164.0, 174.5, 193.5) |

## Per-asset comparisons

### chess_set

_MaterialX OpenChessSet (SideFX/ASWF)_  (up axis: Y)

**camA** — camera eye (1.21623590295, 0.728387501949, 1.43086576818), target (0, 0.0946376550198, 0), FOV 35 deg

![chess_set camA](chess_set_camA_compare.png)

**camB** — camera eye (-1.27055723837, 1.27564531724, 0.952917928776), target (0, 0.0946376550198, 0), FOV 35 deg

![chess_set camB](chess_set_camB_compare.png)

## Notes

The wrapper *sub-layers* the chess set, so the nanousd loader keeps its MaterialX material bindings (board + translucent-marble pieces + gold trim) under the shared constant-color dome + Key/Fill rig.
_Populate backend-specific shading observations from the rendered compare strips after a run._

_See [../README.md](../README.md) for the cross-set write-up and caveats._

## Repro steps

macOS + Metal only. All commands assume the repo at `$HOME/nanousd-labs/nanousd-metal-renderer`.

### 1. Build the Metal renderer library

```bash
cd $HOME/nanousd-labs/nanousd-metal-renderer
./build.sh
```

This produces `build/libnusd_renderer.dylib`, discovered automatically by the
`nusd_renderer` ctypes bindings (or point at it explicitly with
`NUSD_RENDERER_LIB=/path/to/libnusd_renderer.dylib`).

### 2. Python environment

You need a Python with **OpenUSD (`pxr`)**, `numpy` and `Pillow`. The harness
imports `pxr` for wrapper generation + bbox framing and `nusd_renderer` from
`$HOME/nanousd-labs/nanousd-metal-renderer/python` (added to `sys.path` automatically).

```bash
python -c "import pxr, numpy, PIL"   # must succeed
```

If the Metal renderer dlopens a separate nanousd USD-parsing backend on your
build, point at it with `NANOUSD_BACKEND=/path/to/libnanousd.dylib`.

### 3. Assets

- **OVRTX references are reused** — the committed
  `comparisons/<set>/frames/<asset>_<cam>_ovrtx.png` files. This harness does NOT
  render OVRTX. (They were rendered by the Vulkan renderer's comparison harness
  from the identical wrapper + camera.)
- **Chess (MaterialX)**: set `NUSD_CHESS_USD=/path/to/OpenChessSet/chess_set.usda`
  (or place it at `comparisons/.assets/chess/chess_set.usda`).
- **Warehouse (Isaac Sim `Simple_Warehouse/full_warehouse.usd`)**: set
  `NUSD_WAREHOUSE_USD=/path/to/full_warehouse.usd` (fetch the whole
  `Simple_Warehouse/` dir incl. its `Materials/` + `Props/` subtrees).
- **Apple USDZ**: downloaded automatically into `comparisons/.assets/apple/`
  (git-ignored) from `https://developer.apple.com/augmented-reality/quick-look/models/<dir>/<file>.usdz`.

### 4. Run the harness

```bash
cd $HOME/nanousd-labs/nanousd-metal-renderer
python comparisons/render_backend_comparison.py --set all
```

Use `--set chess|apple|warehouse` for a single set, or `--gate` for a quick
chess camA black-frame pre-flight. `--readme-only` regenerates the
READMEs/contact sheets from an existing `metrics.json` (no render).

The harness regenerates the co-located sub-layer wrapper next to each asset's
root layer at run time (`<asset_dir>/_nusd_backend_compare_wrapper_<label>.usda`)
— required so the nanousd material loader's `.mtlx`/texture scan (keyed off the
root layer's directory) finds the asset's materials. The copy committed under
`<set>/wrappers/<label>.usda` is a record of the generated text.

