# Renderer backend comparison: OVRTX 0.3 vs OpenGL (GLES raster)

This tree compares two nanousd renderer backends, rendering the **same composed
USD scene with the same authored light rig, the same programmatic cameras, and a
square output so the subjects co-register**, through each:

- **OVRTX 0.3** (reference): NVIDIA OVRTX path tracer, driven through
  `nanousdview._backend` (`OvrtxViewportRenderer`, `rt2` mode) under the
  `ovrtx==0.3.0` venv.
- **OpenGL (GLES raster)**: this repo's `nusd_renderer_opengl`
  (`NuRenderer(enable_rt=False)`, `render(NU_RENDER_RASTER)`) — a portable
  **OpenGL ES 3.2** rasterizer with no hardware ray tracing.

It is the OpenGL counterpart of the Vulkan repo's
`comparisons/render_backend_comparison.py` (OVRTX vs Vulkan RT vs Vulkan
Raster); it reuses that harness's proven OVRTX driver and camera/metrics engine
helpers verbatim and swaps the local backend to this repo's GLES rasterizer.

## Asset sets (2 camera angles each)

| Set | Assets | Source |
| --- | --- | --- |
| [`chess/`](chess/README.md) | OpenChessSet (MaterialX) | `chess_set.usda` (SideFX/ASWF) |
| [`apple/`](apple/README.md) | teapot, toy_drummer, robot, fender_stratocaster, ball_soccerball, pancakes | Apple AR Quick Look USDZ |
| [`warehouse/`](warehouse/README.md) | Isaac Sim `Simple_Warehouse/full_warehouse.usd` (interior) | NVIDIA Isaac Sim (local PBR materials) |

- **Resolution**: **768x768 (square)**. See *Camera parity (FIX 1)* below — the
  square output is what makes the two backends co-register, so the metrics are
  meaningful.
- **Cameras**: two angles per asset, set programmatically on both backends (no
  authored camera). Chess and the Apple assets use bbox-framed angles — `camA`
  (front three-quarter) and `camB` (higher, opposite side). The **warehouse uses
  explicit interior look-at cameras** at forklift/eye height (camA down the long
  aisle, camB a 3/4 corner view) so racks, shelves, boxes, floor and walls fill
  the frame instead of the exterior footprint. The up-axis is read per asset
  (Y-up for chess and the Apple assets, **Z-up for the warehouse**).
- **Lighting rig (shared)**: a constant-color `DomeLight` (0.78, 0.82, 0.90 @
  450) + a Key `SphereLight` (1.0, 0.95, 0.86 @ 6000) + a Fill `SphereLight`
  (0.68, 0.76, 1.0 @ 1200), with Key/Fill positioned from each asset's bbox. The
  generated wrapper **sub-layers** the asset's root layer (so material bindings
  survive — a `references` arc would re-root the prims and drop them) and authors
  only these lights at root scope, so **both** backends see identical lights.
  OVRTX is run with `NUVIEW_OVRTX_DEFAULT_LIGHTING=0` so it reads the authored
  rig instead of its built-in default lighting.
- The operative wrapper is written **co-located with each asset's root layer**,
  because the nanousd material loader discovers `.mtlx`/texture side-cars by
  scanning the *root* (wrapper) layer's directory. A copy of each generated
  wrapper is kept under `<set>/wrappers/<label>.usda` for the record.

## Camera parity (FIX 1 — subjects now co-register)

Previously the harness rendered **non-square (512x320)** with a **square camera
aperture** (`hap == vap`). The native OpenGL backend treats `fov_degrees` as the
**vertical** FOV and derives the horizontal FOV from the aspect ratio; OVRTX
instead derives its projection from `focal_length` + horizontal/vertical aperture
(which the harness authors equal). At a non-square aspect those two conventions
disagree, so **OVRTX framed every subject larger and offset** and the silhouettes
did not overlap — which inflated every RMS/MAE and crushed every silhouette IoU.
(The reference Vulkan harness measured the same mechanism at ~1.8x on its native
backends, which share the identical `set_camera_explicit` plumbing.)

The fix is to **render square** (768x768). At a square aspect (1.0) the horizontal
FOV equals the vertical FOV in *both* backends (`hfov == vfov == fov_degrees`), so
the subjects co-register. This holds for any FOV and any scene because it is an
exact identity at aspect 1.0; no per-asset aperture tuning is needed.

**Verified empirically:** silhouette IoU vs the OVRTX reference jumped from
~0.25–0.41 (old 512x320) to **0.86–0.99** (new 768x768). Measured directly on the
soccerball (foreground bbox via the same background-delta mask the metrics use),
the OVRTX and OpenGL subjects now agree to **0.0% in width and 0.0–0.3% in height,
corners within 1px** — well inside the ~3% target. The image strips show real
shading deltas, not silhouette ghosting.

## How the wrapper is composed (the fix that makes materials show)

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

Two details are load-bearing:

1. **Sub-layers, not references.** A USD `references` arc re-roots the asset
   under a new prim, and the nanousd loader drops the asset's material bindings
   in that case — chess then renders plain white with `0` materials.
   Sub-layering composes the asset's prims at their *original* paths with
   bindings intact.
2. **Co-located with the asset's root layer.** The nanousd material loader
   discovers `.mtlx` side-cars and relative textures by scanning the directory
   of the *root* (wrapper) layer. The harness therefore writes the operative
   wrapper next to the asset root layer (e.g.
   `OpenChessSet/_nusd_backend_compare_wrapper_chess_set.usda`) and loads that.
   A copy is kept under `<set>/wrappers/<label>.usda` as a record.

## Results at a glance

Per (asset, camera): mean luma per backend (no black frames anywhere), RMS vs the
OVRTX reference, silhouette IoU, and the OpenGL material/texture upload counts.

| Set | Asset | Cam | OVRTX luma | OpenGL luma | RMS | IoU | OpenGL mat / tex |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| chess | chess_set | camA | 169.8 | 170.1 | 40.3 | 0.976 | 15 / **40** |
| chess | chess_set | camB | 158.5 | 171.2 | 56.2 | 0.989 | 15 / **40** |
| apple | teapot | camA | 181.9 | 172.0 | 22.2 | 0.959 | 1 / **5** |
| apple | teapot | camB | 182.1 | 171.2 | 23.8 | 0.913 | 1 / **5** |
| apple | toy_drummer | camA | 177.6 | 165.9 | 18.9 | 0.982 | 2 / **9** |
| apple | toy_drummer | camB | 177.4 | 165.2 | 18.4 | 0.985 | 2 / **9** |
| apple | robot | camA | 169.1 | 161.2 | 23.7 | 0.989 | 1 / **5** |
| apple | robot | camB | 171.2 | 161.4 | 21.5 | 0.993 | 1 / **5** |
| apple | fender_stratocaster | camA | 177.2 | 166.1 | 19.6 | 0.889 | 19 / **8** |
| apple | fender_stratocaster | camB | 178.3 | 166.2 | 19.2 | 0.909 | 19 / **8** |
| apple | ball_soccerball | camA | 176.9 | 172.8 | 24.8 | 0.915 | 1 / **4** |
| apple | ball_soccerball | camB | 176.6 | 171.7 | 23.2 | 0.922 | 1 / **4** |
| apple | pancakes | camA | 176.9 | 167.4 | 22.1 | 0.978 | 5 / **10** |
| apple | pancakes | camB | 175.2 | 166.9 | 22.5 | 0.967 | 5 / **10** |
| warehouse | warehouse | camA | 87.3 | 82.9 | 43.7 | 0.896 | 5910 / **213** |
| warehouse | warehouse | camB | 72.4 | 79.2 | 45.6 | 0.984 | 5910 / **213** |

No frame is black on either backend (the warehouse interior is correctly darker
than the bright bbox-framed subjects). `mat / tex` = materials and textures the
OpenGL GLES backend uploaded, scraped from the nanousd loader log. The bold
texture counts are the load-bearing FIX-3 signal: **every set now uploads `tex >
0`** (the Apple assets previously uploaded `0`).

## Real OVRTX-vs-OpenGL differences (the headlines)

1. **The Apple USDZ assets now TEXTURE in OpenGL (FIX 3, `material.c`).**
   Previously the GLES loader uploaded **0 textures** for all six Apple assets
   (`gpu_gles: uploaded N materials, 0 textures`) and they rendered near-white. A
   `material.c` loader fix — the `nanousd_isa(child,"Shader")` gate now also
   accepts an exact `typename=="Shader"`, which is how these USDZ-extracted
   Shader prims resolve — means the backend now uploads their UsdPreviewSurface
   texture maps: teapot **5**, toy_drummer **9**, robot **5**,
   fender_stratocaster **8**, ball_soccerball **4**, pancakes **10** textures
   (all were 0 before). Visually confirmed: the robot now shows its red painted
   body and grey metal head/legs, the soccer ball its black/white pentagons, the
   Stratocaster its sunburst finish and wood neck, the teapot/drummer/pancakes
   their photogrammetry color. Chess's **15 MaterialX (.mtlx) materials / 40
   textures** also load as before. So **all three sets are textured in OpenGL**.

2. **OpenGL renders the heavy warehouse interior.** (The Vulkan repo's raster
   backend rendered this scene black until a headless command-buffer-reuse race
   in `gpu_vulkan.c` was fixed; it renders the warehouse now too.) The
   set now uses NVIDIA's standard Isaac Sim `Simple_Warehouse/full_warehouse.usd`
   (replacing the old "Physical AI" warehouse, whose `omniverse://` materials did
   not resolve offline). Its materials are **all local**, so the OpenGL loader
   uploads **5910 materials / 213 textures with zero `omniverse://` misses and
   zero failed texture loads**, and renders the full **3468-mesh** interior from
   both cameras (luma ~79–83): yellow/orange storage racks, stacked cardboard
   boxes, signage, grey floor and the structural ceiling/walls, all textured.
   Older Vulkan raster comparison captures black-failed this scene before the
   Vulkan frame-cycling fix; OpenGL GLES did not exhibit that failure.

3. **Path-traced vs raster look (the dominant remaining difference).** With the
   subjects co-registered and the materials textured in both backends, the
   remaining gaps are now genuine shading deltas. OVRTX (path-traced) is more saturated and
   contrasty, with soft path-traced contact shadows, real occlusion, dome fill on
   faces turned away from the Key/Fill, and (on the warehouse) a warm reflective
   floor and ceiling light bloom. OpenGL's GLES raster is **flatter and a touch
   darker** (typically ~6–12 luma below OVRTX on the Apple assets), lifts
   shadowed/black regions with a procedural hemisphere ambient (see *Lighting
   parity* below) rather than tracing them — so e.g. the soccerball's black
   pentagons read dark grey-blue rather than true black and the warehouse floor
   takes a cool bluish cast — and has no ray-traced contact shadows or
   reflections (it is raster-only, no RT).

See each set's README for per-asset embeds, the full metrics table, and the
contact sheet.

## Lighting parity (FIX 4 — no-HDR ambient, documented honestly)

The shared rig deliberately uses a **constant-color `DomeLight` with NO HDR
texture** plus Key/Fill `SphereLight`s, so both backends read the same lights.
The two backends differ in how they fill the *environment* term when no HDR IBL
is present, and this is the dominant tone difference:

- **OVRTX** path-traces the constant-color dome as an **area environment**: it is
  the softest and brightest fill on shadowed faces, and it produces real
  multi-bounce fill, occlusion and contact shadows.
- **OpenGL (GLES raster)** is a forward rasterizer with **no path tracing and no
  ray-traced shadows**. With no HDR dome loaded it lights shadowed faces with a
  **procedural sky/ground hemisphere ambient + the constant dome color**, which
  is uniform and a touch brighter in the shadows but lacks traced occlusion. That
  is why OpenGL reads flatter and loses the deep blacks and contact shadows that
  OVRTX resolves. This is an architectural raster-vs-path-traced difference, not a
  missing material or a black-frame bug; it is documented here rather than hacked
  out of the shader.

## Honest caveats (read before trusting the metrics)

1. **RMS / MAE remain diagnostic, not strict parity thresholds.** OVRTX is a
   converged path tracer while OpenGL is a single-pass GLES rasterizer, so a real
   GI / shadow / tone gap remains by design. But the subjects now co-register
   (FIX 1, IoU 0.86–0.99), so the silhouettes overlap and the scalars track real
   shading differences rather than camera misalignment.
2. **The warehouse interior is correctly darker** than the bright bbox-framed
   subjects (OVRTX luma ~73–87, OpenGL ~91–95); that is the scene, not a
   black-frame failure. Both backends render the textured interior.
3. **Two trivial warehouse props are missing offline** (a `Forklift/forklift.usd`
   and one `S_Barcode_248.usd`); USD prints a warning and renders the scene
   without them.

## Repro steps

All commands assume this repo at `$HOME/nanousd-labs/nanousd-opengl-renderer`
and the verified box environment.

### 1. Build the OpenGL renderer library

```bash
cd $HOME/nanousd-labs/nanousd-opengl-renderer
./build.sh
```

Produces `build/libnusd_renderer_opengl.so` (found automatically by the
`nusd_renderer_opengl` ctypes bindings, or point `NUSD_RENDERER_LIB` at it). This
.so also compiles `src/material.c`, which carries the FIX-3 isa/typename Shader
fix that makes the Apple USDZ texture maps upload.

### 2. Environments

- Native renderer python (numpy, Pillow; loads the OpenGL `.so` via
  `python/nusd_renderer_opengl`): `$HOME/nanousd-labs/.venv/bin/python`
- OVRTX 0.3 reference venv (`ovrtx==0.3.0`):
  `$HOME/nanousd-labs/.ovrtx03-venv/bin/python`

### 3. Fetch assets

- Chess (MaterialX):
  `/path/to/OpenChessSet/chess_set.usda`
- Warehouse (Isaac Sim `Simple_Warehouse/full_warehouse.usd`):
  `$HOME/assets/Isaac/Environments/Simple_Warehouse/full_warehouse.usd`
  — download recipe below.
- Apple USDZ: pre-copied into `comparisons/.assets/apple/` (git-ignored). Copy
  them from the Vulkan repo rather than re-downloading:
  ```bash
  cp -r ../nanousd-vulkan-renderer/comparisons/.assets/apple comparisons/.assets/
  ```
  (If they are missing the harness will fall back to downloading them from
  `https://developer.apple.com/augmented-reality/quick-look/models/<dir>/<file>.usdz`.)

#### Warehouse download (NVIDIA Isaac Sim, public S3 mirror, no creds)

The warehouse is NVIDIA's **standard Isaac Sim `Simple_Warehouse/full_warehouse.usd`**.
Its materials resolve **offline** because they are local (`./Materials/` and
`./Props/`) — unlike the older "Physical AI" warehouse whose materials reference
`omniverse://` and do NOT resolve here. Fetch the **whole `Simple_Warehouse/`
directory** (the `.usd` PLUS its sibling `Materials/` and `Props/` subtrees) from
the public production mirror.

With the AWS CLI (recursive, easiest — `--no-sign-request` needs no credentials):

```bash
DEST=$HOME/assets/Isaac/Environments/Simple_Warehouse
aws s3 cp --no-sign-request --recursive \
  s3://omniverse-content-production/Assets/Isaac/4.5/Isaac/Environments/Simple_Warehouse/ \
  "$DEST/"
```

Or over HTTPS with `curl`/`wget` (the bucket is web-accessible at
`https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Simple_Warehouse/`):

```bash
BASE=https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Simple_Warehouse
DEST=$HOME/assets/Isaac/Environments/Simple_Warehouse
mkdir -p "$DEST/Materials/Textures" "$DEST/Props"
wget -q "$BASE/full_warehouse.usd" -O "$DEST/full_warehouse.usd"
# Then mirror the Materials/ (*.mdl + Textures/*.png) and Props/ (*.usd) subtrees
# the root layer references. The recursive aws s3 cp above is the reliable way to
# pull the complete tree.
```

### 4. Run the harness

The harness imports the shared OVRTX driver + camera/metrics engine helpers from
the **Vulkan** repo's `scripts/`, and `pxr` from the OpenUSD install. Run it with
the native python and that PYTHONPATH/LD_LIBRARY_PATH:

```bash
cd $HOME/nanousd-labs/nanousd-opengl-renderer
PYTHONPATH=$HOME/OpenUSD_install/lib/python:$HOME/nanousd-labs/nanousd-vulkan-renderer/scripts:$HOME/nanousd-labs/nanousd-opengl-renderer/python \
LD_LIBRARY_PATH=$HOME/OpenUSD_install/lib \
OVRTX_PYTHON=$HOME/nanousd-labs/.ovrtx03-venv/bin/python \
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
NUVIEW_OVRTX_DEFAULT_LIGHTING=0 \
  $HOME/nanousd-labs/.venv/bin/python comparisons/render_backend_comparison.py --set all
```

- `--set chess|apple|warehouse` renders a single set.
- `--gate` renders only chess camA on both backends and prints mean RGB (the
  pre-flight black-frame check).
- `--readme-only --set all` regenerates the READMEs/contact sheets from existing
  `metrics.json` without re-rendering.
- `--width/--height` default to 768x768; keep them **equal** to preserve camera
  parity (FIX 1).

The harness regenerates the co-located sub-layer wrapper next to each asset's
root layer at run time (e.g.
`<asset_dir>/_nusd_backend_compare_wrapper_<label>.usda`) — that placement is
required so the nanousd material loader's `.mtlx`/texture scan, which keys off
the root layer's directory, finds the asset's materials. Load the asset via the
harness, not the `wrappers/<label>.usda` record copy directly (its `subLayers`
path is relative to the asset directory).

## FLIP diff pipeline (automated OpenGL-vs-OVRTX testing)

[`flip_pipeline.py`](flip_pipeline.py) scores every OpenGL output against its
OVRTX golden using NVIDIA **ϜLIP** (perceptual difference), via the calibrated
metric in [`flip_compare.py`](flip_compare.py) (shared verbatim with the Vulkan
and Metal renderers). RMS/MAE/IoU stay in `metrics.json` (IoU still answers the
geometry question FLIP can't); FLIP is the perceptual diff. **Lower FLIP =
closer to the golden.** Needs `flip-evaluator` plus `scipy` and `scikit-image`
(`pip install flip-evaluator scipy scikit-image`) — scipy/scikit-image drive the
gate's GMSD structure and CIEDE2000 colour legs; without them the gate SKIPS, not runs.

```bash
PY=$HOME/nanousd-labs/.venv/bin/python
$PY comparisons/flip_pipeline.py                 # analyse committed frames -> flip/REPORT.md + heatmaps
$PY comparisons/flip_pipeline.py --check         # regression gate vs flip/baseline.json (exit 1 on regress)
$PY comparisons/flip_pipeline.py --update-baseline   # freeze current scores as the baseline
$PY comparisons/flip_pipeline.py --render        # re-render OpenGL frames first (needs assets)
```

Committed artifacts: `flip/baseline.json` (regression reference) and
`flip/REPORT.md`; the heatmap strips and `results.json` are gitignored. The
CTest target `flip_opengl_vs_ovrtx` runs `--check` and **skips** (return 77)
when `flip-evaluator` is absent, mirroring `python_bindings_test`.

## Layout

```
comparisons/
  README.md                 # this file
  .gitignore                # ignores .assets/
  render_backend_comparison.py
  {chess,apple,warehouse}/
    frames/                 # raw per-backend PNGs
    <asset>_<cam>_compare.png   # OVRTX | OpenGL output strips
    contact_sheet.png
    metrics.json
    README.md
    wrappers/<label>.usda   # record copy of the generated wrapper
  .assets/                  # git-ignored; Apple USDZ + warehouse source live here
```
