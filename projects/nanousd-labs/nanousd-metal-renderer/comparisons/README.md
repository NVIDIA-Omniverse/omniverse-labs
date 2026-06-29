# Renderer backend comparison: OVRTX 0.3 (reused) vs Metal RT vs Metal Raster

This tree compares the Metal renderer's two backends against the OVRTX 0.3
path-traced reference, on three asset sets, from two camera angles each, with a
**shared lighting rig that every backend reads** and a **square output so the
subjects co-register**. It is produced by
[`render_backend_comparison.py`](render_backend_comparison.py).

It is the Metal counterpart of the Vulkan renderer's `comparisons/` harness and
is modeled directly on it ("be like Vulkan"): it emits **two** candidate renders
per frame — `*_metal_rt.png` and `*_metal_raster.png` — each diffed against the
OVRTX reference (`*_diff_rt.png`, `*_diff_raster.png`), exactly as the Vulkan
harness emits `*_vk_rt.png` / `*_vk_raster.png`.

## What is compared

| Backend | How it is driven | Role |
| --- | --- | --- |
| **OVRTX 0.3** | NVIDIA OVRTX path tracer — **reused reference images**, not rendered here | reference |
| **Metal RT** | local `nusd_renderer` `NuRenderer(enable_rt=True)` → `render(NU_RENDER_RT)` (hardware ray tracing on Metal) | candidate |
| **Metal Raster** | local `nusd_renderer` `NuRenderer(enable_rt=False)` → `render(NU_RENDER_RASTER)` (rasterizer) | candidate |

### Reused OVRTX reference (the key difference from a from-scratch harness)

This harness **does not render OVRTX**. The `*_ovrtx.png` reference frames are
reused — committed under each `<set>/frames/` directory, copied from the Vulkan
renderer's comparison set, which renders the **identical** sub-layer wrapper and
camera. That reuse is only valid because the framing and lighting are kept
byte-identical (see *Camera parity* and *Wrapper* below); if you change the
camera math, resolution, or light rig, the references must be re-rendered.

This removes the OVRTX venv dependency entirely — the harness only needs OpenUSD
(`pxr`), `numpy`/`Pillow`, and the built Metal renderer.

- **Resolution**: **768×768 (square)** — required for camera parity (below).
- **Cameras**: two angles per asset, set programmatically on the Metal backend
  (no authored camera). Chess and the Apple assets use bbox-framed angles —
  `camA` (front three-quarter) and `camB` (higher, opposite side). The
  **warehouse uses explicit interior look-at cameras** at forklift/eye height.
  The up-axis is read per asset (Y-up for chess and the Apple assets, **Z-up for
  the warehouse**) and the camera basis matches.
- **Lighting rig (shared)**: a constant-color `DomeLight` (no HDR texture) plus a
  Key and a Fill `SphereLight` positioned from the asset bbox, authored into the
  wrapper. This is byte-identical to the wrapper that produced the reused OVRTX
  references.

## Camera parity (why square output)

The native Metal backend treats `fov_degrees` as the **vertical** FOV and derives
the horizontal FOV from the aspect ratio; OVRTX instead derives its projection
from `focal_length` + horizontal/vertical aperture (which the harness authors
equal). At a non-square aspect those conventions disagree and OVRTX frames the
subject larger/offset. At a **square** aspect (1.0), `hfov == vfov == fov_degrees`
in both, so the subjects co-register and the per-pixel diffs are **real shading
deltas, not silhouette ghosting**. This is an exact identity at aspect 1.0 — no
per-asset aperture tuning needed — and it is what makes the reused OVRTX
references line up with the Metal renders.

## How the wrapper is composed

Each probe wrapper **sub-layers** the asset's root layer and authors only the
light rig at root scope:

```
#usda 1.0
(
    defaultPrim = "<asset root prim>"
    subLayers = [ @./<asset root layer>@ ]
)
def DomeLight  "CompareDome" { ... }
def SphereLight "CompareKey"  { ... }
def SphereLight "CompareFill" { ... }
```

Sub-layering (not a `references` arc) keeps the asset's prims at their original
paths so **material bindings survive**. The operative wrapper is written
*co-located* next to the asset's root layer at run time
(`_nusd_backend_compare_wrapper_<label>.usda`), because the nanousd material
loader discovers `.mtlx` side-cars and relative textures by scanning the root
layer's directory. The copy under `<set>/wrappers/<label>.usda` is a record of
the generated text.

## Statistics

For each (asset, camera, backend) the harness reports, vs the OVRTX reference:

- **RMS** and **MAE** over 8-bit sRGB pixels (lower = closer to the reference).
- **silhouette IoU** — intersection-over-union of the foreground masks
  (background-delta) of the candidate and the reference (1.0 = identical
  silhouette / perfect co-registration).
- **mean RGB** per backend — a black-frame sanity check; a near-zero luma flags a
  backend (or the reference) that came back black, in which case RMS/MAE/IoU for
  that row are not a fidelity signal.

These land in each set's `metrics.json` and the generated `<set>/README.md`
table, with a 5-panel `*_compare.png` strip
(OVRTX | Metal RT | Metal Raster | diff RT ×4 | diff Raster ×4) and a
`contact_sheet.png` per set.

## Asset sets

- **chess** — MaterialX OpenChessSet (SideFX/ASWF); MaterialX materials.
- **apple** — six Apple AR Quick Look USDZ assets (teapot, toy drummer, robot,
  Fender Stratocaster, soccerball, pancakes); UsdPreviewSurface texture PBR.
  Downloaded automatically into `.assets/apple/` (git-ignored).
- **warehouse** — Isaac Sim `Simple_Warehouse/full_warehouse.usd`; Z-up interior,
  local OmniPBR/MDL materials.

## Status

The **apple set is rendered** on Metal (Apple M3 Pro, hardware RT) and its real
results are committed: Metal RT/Raster frames, diffs, 12 compare strips, the
contact sheet, `metrics.json` and the generated [`apple/README.md`](apple/README.md).
Headline: the subjects **co-register** with the reused OVRTX references
(silhouette IoU **0.89–0.99**), and the large RT-vs-Raster RMS gap is a
**dome/background-handling** difference (Metal RT renders the constant DomeLight
near-white; Raster fills a blue-ish hemisphere ambient), not a subject error —
see `apple/README.md`.

The **chess and warehouse sets are also rendered** on Metal — their Metal
RT/Raster frames, compare strips, contact sheets and `metrics.json` are committed
alongside the reused OVRTX references (the assets — SideFX/ASWF OpenChessSet;
Isaac Sim `Simple_Warehouse` — were sourced locally per the recipe below).
Re-render with:

```bash
NUSD_CHESS_USD=/path/to/chess_set.usda \
  python comparisons/render_backend_comparison.py --set chess
NUSD_WAREHOUSE_USD=/path/to/full_warehouse.usd \
  python comparisons/render_backend_comparison.py --set warehouse
```

### Verified build/run recipe (macOS, Apple silicon)

The apple set above was produced with exactly these steps on this machine:

```bash
# 1. nanousd must be on a commit that has the API the metal renderer needs
#    (NANOUSD_COMPOSITION_ARC_DIRECT, nanousd_rel_metadatas — on nanousd main,
#    NOT the eslavin/color-spaces commit the superproject currently pins).
git -C ../nanousd checkout main && git -C ../nanousd reset --hard origin/main
cmake -S ../nanousd -B ../nanousd/build -DCMAKE_BUILD_TYPE=Release -DNANOUSD_BUILD_TESTS=OFF
cmake --build ../nanousd/build --parallel

# 2. build the Metal renderer (FetchContent builds MaterialX on first run)
./build.sh                                   # -> build/libnusd_renderer.dylib

# 3. Python 3.10+ venv with OpenUSD + numpy + Pillow + NVIDIA FLIP + scipy + scikit-image
#    (bindings require >=3.10; flip-evaluator drives the FLIP diff; scipy + scikit-image
#    power the gate's GMSD structure and CIEDE2000 colour legs — without them the gate
#    SKIPS rather than runs, so they are required, not optional)
python3.12 -m venv ../.venv
../.venv/bin/pip install usd-core numpy pillow flip-evaluator scipy scikit-image

# 4. run (the bindings find python/ via the harness; point at the dylib + nanousd)
NUSD_RENDERER_LIB="$PWD/build/libnusd_renderer.dylib" \
DYLD_FALLBACK_LIBRARY_PATH="$PWD/../nanousd/build/Release" \
  ../.venv/bin/python comparisons/render_backend_comparison.py --set apple
```

The non-render half (metrics, diff/compare/contact compositing,
`metrics.json`/README generation, `--readme-only` roundtrip) is additionally
covered by a standalone check against the reference images. See **Repro steps**
in the per-set README or in `render_backend_comparison.py` for the full detail
(env vars, asset download recipes).

## Repro (summary)

```bash
cd ..                                   # nanousd-metal-renderer
./build.sh                              # build/libnusd_renderer.dylib
# Python with pxr + numpy + Pillow on PATH; assets available:
NUSD_CHESS_USD=/path/to/chess_set.usda \
NUSD_WAREHOUSE_USD=/path/to/full_warehouse.usd \
  python comparisons/render_backend_comparison.py --set all
```

## FLIP diff pipeline (automated Metal-vs-OVRTX testing)

[`flip_pipeline.py`](flip_pipeline.py) is an automated regression + diagnosis
harness that scores every Metal output against its OVRTX golden using NVIDIA
**ϜLIP** (perceptual difference), via the calibrated metric in
[`flip_compare.py`](flip_compare.py). RMS/MAE/IoU stay in `metrics.json` (IoU
still answers the geometry/coverage question FLIP can't); FLIP is the perceptual
diff. **Lower FLIP = closer to the golden.**

Calibration matters here: naive whole-image FLIP is fooled by the Metal studio
backgrounds (white/blue vs the golden's grey) and by subject size. So studio
scenes (apple, chess) composite the subject onto the golden's background and
average FLIP over the foreground; the full-scene warehouse uses plain FLIP. The
reference is always `*_ovrtx.png`.

```bash
PY=../.venv/bin/python
$PY comparisons/flip_pipeline.py                 # analyse committed frames -> flip/REPORT.md + heatmaps
$PY comparisons/flip_pipeline.py --check         # regression gate vs flip/baseline.json (exit 1 on regress)
$PY comparisons/flip_pipeline.py --update-baseline   # freeze current scores as the baseline
$PY comparisons/flip_pipeline.py --render        # re-render Metal frames first (needs assets)
```

Committed artifacts: `flip/baseline.json` (regression reference),
`flip/results.json` (latest scores + diagnoses), `flip/REPORT.md` (human
summary). The heatmap strips (`flip/*.png`) are gitignored — regenerate on demand.
The CTest target `flip_metal_vs_ovrtx` runs `--check` and **skips** (return 77)
when `flip-evaluator` is absent from the test interpreter, mirroring
`ovrtx_python_metal`.

---

`--set chess|apple|warehouse` runs one set; `--gate` does a quick chess camA
pre-flight; `--readme-only` regenerates the READMEs/contact sheets from an
existing `metrics.json` without rendering. Full details (env vars, asset
download recipes) are in the generated per-set README and the script's
`REPRO_BLOCK`.
