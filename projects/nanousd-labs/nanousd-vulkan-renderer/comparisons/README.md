# Renderer backend comparison: OVRTX 0.3 vs Vulkan RT vs Vulkan Raster

This tree compares three nanousd renderer backends on three asset sets, from two
camera angles each, with a **shared lighting rig that every backend reads** and a
**square output so the subjects co-register across backends**. It is produced by
[`render_backend_comparison.py`](render_backend_comparison.py).

## What is compared

| Backend | How it is driven | Role |
| --- | --- | --- |
| **OVRTX 0.3** | NVIDIA OVRTX path tracer via `nanousdview._backend` (`OvrtxViewportRenderer`, `rt2` / RealTimePathTracing) | reference |
| **Vulkan RT** | local `nusd_renderer` `NuRenderer(enable_rt=True)` → `render(NU_RENDER_RT)` (hardware ray tracing) | candidate |
| **Vulkan Raster** | local `nusd_renderer` `NuRenderer(enable_rt=False)` → `render(NU_RENDER_RASTER)` (rasterizer) | candidate |

- **Resolution**: **768x768 (square)**. See *Camera parity* below — square output
  is what makes the three backends co-register, so the metrics are meaningful.
- **Cameras**: two angles per asset, set programmatically on every backend (no
  authored camera). Chess and the Apple assets use bbox-framed angles — `camA`
  (front three-quarter) and `camB` (higher, opposite side). The **warehouse uses
  explicit interior look-at cameras** at forklift/eye height (camA down the long
  aisle, camB a 3/4 corner view) so racks, shelves, boxes, floor and walls fill
  the frame instead of the exterior building footprint. The up-axis is read per
  asset (Y-up for chess and the Apple assets, **Z-up for the warehouse**) and the
  camera basis is chosen to match.
- **Lighting rig (shared, authored into each wrapper layer)**: a constant-color
  `DomeLight` (NO HDR texture) plus a Key and a Fill `SphereLight` positioned from
  the asset bounding box. **All three backends read this same authored rig** —
  OVRTX is run with `NUVIEW_OVRTX_DEFAULT_LIGHTING=0` so it uses the authored
  lights instead of its built-in default lighting.

## Camera parity (FIX 1 — subjects now co-register)

Previously the harness rendered **non-square (512x320)** with a **square camera
aperture** (`hap == vap`). The native Vulkan backends treat `fov_degrees` as the
**vertical** FOV and derive the horizontal FOV from the aspect ratio; OVRTX
instead derives its projection from `focal_length` + horizontal/vertical aperture
(which the harness authors equal). At a non-square aspect those two conventions
disagree, so **OVRTX framed every subject ~1.8x larger** and the silhouettes did
not overlap — which inflated every RMS/MAE and crushed every silhouette IoU.

The fix is to **render square** (768x768). At a square aspect (1.0) the horizontal
FOV equals the vertical FOV in *both* backends (`hfov == vfov == fov_degrees`), so
the subjects co-register. This holds for any FOV and any scene because it is an
exact identity at aspect 1.0; no per-asset aperture tuning is needed.

**Verified empirically on the soccerball** (foreground bbox via the same
background-delta mask the metrics use):

| | OVRTX vs RT subject bbox | width Δ | height Δ | bbox corners |
| --- | --- | ---: | ---: | --- |
| **Before** (512x320, non-square) | OVRTX area 2.81x larger | 70.7% | 43.4% | OVRTX spills to full 512 width |
| **After** (768x768, square) | aligned | 0.0% | 0.3% | match within 1px (x0,y0,x1,y1) |

Because the cameras now align, the **RMS / MAE / silhouette-IoU metrics are
finally meaningful** — the image strips show real shading deltas, not silhouette
ghosting. The subjects are co-registered.

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
   A copy of the generated text is kept under `<set>/wrappers/<label>.usda` as a
   record.

## Sets

| Set | Assets | README | Contact sheet |
| --- | --- | --- | --- |
| Chess (MaterialX) | OpenChessSet | [chess/README.md](chess/README.md) | [chess/contact_sheet.png](chess/contact_sheet.png) |
| Apple USDZ | teapot, toy_drummer, robot, fender_stratocaster, soccerball, pancakes | [apple/README.md](apple/README.md) | [apple/contact_sheet.png](apple/contact_sheet.png) |
| Warehouse (Isaac Sim) | Simple_Warehouse/full_warehouse.usd (interior) | [warehouse/README.md](warehouse/README.md) | [warehouse/contact_sheet.png](warehouse/contact_sheet.png) |

## Real backend differences observed (aligned subjects, matched lights)

With the subjects co-registered, materials loading in every backend, and the same
authored lights driving all three, the remaining differences are genuine renderer
differences:

- **Materials render in all three backends.** Chess loads its MaterialX
  materials/textures (marble board checker, green/black/white pieces, gold trim);
  the Apple assets apply their UsdPreviewSurface base-color / roughness / metallic
  / normal maps; the new warehouse applies its OmniPBR/MDL materials (textured
  racks, boxes, signs, floor, walls).
- **No-HDR ambient is the dominant tone difference (FIX 3 — see below).** OVRTX
  path-traces the constant dome as an area environment (softest fill on shadowed
  faces). Both Vulkan paths instead add a *procedural sky/ground hemisphere*
  ambient when no HDR dome is loaded. The few-light chess/apple path still has
  an RT-vs-raster ambient scale asymmetry; the warehouse now uses a separate
  many-light branch that tones down high/vertical raster fill and warms the RT
  bounce. The remaining gap is mostly missing path-traced occlusion, floor
  reflections, and multi-bounce transport.
- **Reflections.** RT places sharp, physically-correct reflections on glossy
  surfaces (soccerball, guitar body, chess board). Raster's specular is
  approximate; OVRTX's are soft and path-traced.
- **Camera framing now matches** (FIX 1): the subjects are the same size and
  position in all three backends.

## Lighting parity (FIX 3 — the cause of "raster is brighter")

The original write-up attributed raster's extra brightness to the raster
`mesh.frag` adding a procedural sky/ground + ambient fallback "that RT does not
get". On closer inspection of the shaders the real story is more precise and is
documented here honestly:

- **Both** Vulkan paths add a procedural hemisphere ambient when no HDR IBL is
  loaded: `src/shaders/mesh.frag.glsl` (raster) and
  `src/shaders/raytrace.rchit.glsl` (RT) each have a `skyColor`/`groundColor`
  hemisphere fallback in their `else` (no-IBL) branch.
- The **difference** in the few-light authored rig is the scaling under authored
  lights: the **RT** fallback is multiplied by `vec3(0.32, 0.34, 0.38)` when
  `sceneLights.nlights > 0`, while raster uses its brighter legacy fallback.
  The warehouse has 41 authored lights, so it takes the `many*NoIbl` branches
  instead; those branches are now tuned separately from the chess/apple path.
- **Is there a clean toggle to disable it for parity?** No. Grepping the env
  handling (`renderer.c`), `mesh.frag.glsl`, and `gpu_vulkan.c` finds no
  env/`#define`/flag that disables only the raster procedural-ambient fallback
  while keeping authored lights (the one related var,
  `NUSD_VISIBLE_DOME_FALLBACK_INTENSITY`, controls the *visible dome* fallback,
  not the per-surface ambient). The `ibl_params.z` negative markers select the
  branch but cannot zero the fallback.
- **Decision:** the shared constant `DomeLight` remains the common environment
  for all backends. The warehouse gap was closed with a narrow many-light/no-IBL
  shader adjustment rather than a harness lighting change, so chess/apple
  few-light behavior is not pulled along with the warehouse.

## Honest caveats (read before trusting the metrics)

These are real backend realities observed in this run, not invented:

1. **Vulkan Raster previously rendered the warehouse pure black — now fixed.**
   On heavy scenes whose *first* frame is slow to draw (the warehouse), a
   command-buffer-reuse-while-in-flight race in the headless present path
   (`gpu_begin_frame`) re-recorded command buffer 0 while the GPU was still
   executing that first frame, blanking the framebuffer to black — a silent
   black (the render call returned OK). Chess and the Apple assets finish their
   first frame fast enough to never hit it, which is why only the warehouse was
   affected (and why my earlier "instancer / scene-scale raster limitation"
   guesses were wrong — it is a frame-pacing GPU-sync race, not geometry).
   Fixed in `src/gpu_vulkan.c` by cycling `current_image` with the frame index
   so the command buffer, its framebuffer, the readback image and the guarding
   fence rotate in lockstep. All three backends now render the textured
   warehouse interior.
2. **RMS / MAE remain diagnostic, not strict parity thresholds.** OVRTX is a
   converged path tracer while the Vulkan backends are single-pass RT / raster, so
   a real GI/shadow/tone gap remains by design. But the subjects now co-register
   (FIX 1), so the silhouettes overlap and the scalars track real shading
   differences rather than camera misalignment.

## Global repro steps

All paths assume this repo at
`$HOME/nanousd-labs/nanousd-vulkan-renderer` on the verified box.

### 1. Build the renderer library

```bash
cd $HOME/nanousd-labs/nanousd-vulkan-renderer
NANOUSD_DIR=$HOME/nanousd-labs/nanousd \
  PATH=$HOME/blender/lib/linux_x64/shaderc/bin:$PATH \
  ./build.sh
```

This produces `build/libnusd_renderer.so`. The renderer also dlopens the nanousd
USD-parsing backend; this run pointed it at the existing
`nanousd/build/Debug/libnanousd.so` via the `NANOUSD_BACKEND` env var (the
harness sets this default itself; override it if your backend lives elsewhere).

### 2. Environments

- Native renderer python (has `nusd_renderer`, numpy, Pillow):
  `$HOME/nanousd-labs/.venv/bin/python`
- OVRTX 0.3 reference venv (has `ovrtx==0.3.0`):
  `$HOME/nanousd-labs/.ovrtx03-venv/bin/python`

### 3. Fetch assets

- Chess (MaterialX):
  `/path/to/OpenChessSet/chess_set.usda`
- Warehouse (Isaac Sim `Simple_Warehouse/full_warehouse.usd`):
  `$HOME/assets/Isaac/Environments/Simple_Warehouse/full_warehouse.usd`
  — download recipe below.
- Apple USDZ: downloaded automatically by the harness into
  `comparisons/.assets/apple/` (git-ignored) from
  `https://developer.apple.com/augmented-reality/quick-look/models/<dir>/<file>.usdz`,
  then the root layer of each `.usdz` is unpacked and sub-layered.

#### Warehouse download (NVIDIA Isaac Sim, public S3 mirror, no creds)

The warehouse is NVIDIA's **standard Isaac Sim `Simple_Warehouse/full_warehouse.usd`**.
Its materials resolve **offline** because they are local (`./Materials/` and
`./Props/`) — unlike the older "Physical AI" warehouse whose materials reference
`omniverse://` and do NOT resolve here (that is why the previous run showed a flat
placeholder slab; this asset is the correct one). Fetch the **whole
`Simple_Warehouse/` directory** (the `.usd` PLUS its sibling `Materials/` and
`Props/` subtrees) from the public production mirror.

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
# Root layer:
wget -q "$BASE/full_warehouse.usd" -O "$DEST/full_warehouse.usd"
# Then mirror the Materials/ (*.mdl + Textures/*.png) and Props/ (*.usd) subtrees
# the root layer references. The recursive aws s3 cp above is the reliable way to
# pull the complete tree (the .usd references many sibling files).
```

Two trivial props are missing offline (a `Forklift/forklift.usd` and one
`S_Barcode_248.usd`); USD prints a warning and the scene renders without them.

### 4. Run the harness

```bash
cd $HOME/nanousd-labs/nanousd-vulkan-renderer
PYTHONPATH=$HOME/OpenUSD_install/lib/python:$PWD/scripts \
LD_LIBRARY_PATH=$HOME/OpenUSD_install/lib \
OVRTX_PYTHON=$HOME/nanousd-labs/.ovrtx03-venv/bin/python \
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
NUVIEW_OVRTX_DEFAULT_LIGHTING=0 \
  $HOME/nanousd-labs/.venv/bin/python \
  comparisons/render_backend_comparison.py --set all
```

Useful flags:

- `--gate` — render only chess `camA` through all three backends and print mean
  RGB per backend (the pre-flight black-frame check).
- `--set chess|apple|warehouse` — render a single set.
- `--readme-only` — regenerate READMEs / contact sheets from the existing
  `metrics.json` without re-rendering.
- `--width/--height` — default 768x768; keep them **equal** to preserve camera
  parity (FIX 1).

## FLIP diff pipeline (automated Vulkan-vs-OVRTX testing)

[`flip_pipeline.py`](flip_pipeline.py) scores both Vulkan backends (RT and
raster) against their OVRTX golden using NVIDIA **ϜLIP** (perceptual difference),
via the calibrated metric in [`flip_compare.py`](flip_compare.py) (shared
verbatim with the OpenGL and Metal renderers). RMS/MAE/IoU stay in `metrics.json`
(IoU still answers the geometry question FLIP can't); FLIP is the perceptual
diff. **Lower FLIP = closer to the golden.** Needs `flip-evaluator` plus `scipy`
and `scikit-image` (`pip install flip-evaluator scipy scikit-image`) — scipy/scikit-image
drive the gate's GMSD structure and CIEDE2000 colour legs; without them the gate SKIPS, not runs.

```bash
PY=/path/to/.venv/bin/python   # numpy + Pillow + flip-evaluator
$PY comparisons/flip_pipeline.py                 # analyse committed frames -> flip/REPORT.md + heatmaps
$PY comparisons/flip_pipeline.py --check         # regression gate vs flip/baseline.json (exit 1 on regress)
$PY comparisons/flip_pipeline.py --update-baseline   # freeze current scores as the baseline
$PY comparisons/flip_pipeline.py --render        # re-render Vulkan frames first (needs assets)
```

Unlike Metal/OpenGL (which run brighter than the golden), Vulkan RT lands
*darker* than the path-traced reference — the data-driven report headline
reflects that. Committed artifacts: `flip/baseline.json` + `flip/REPORT.md`; the
heatmap strips and `results.json` are gitignored. The CTest target
`flip_vulkan_vs_ovrtx` runs `--check` and **skips** (return 77) without
flip-evaluator.

## Output layout

```
comparisons/
  README.md                      <- this file
  render_backend_comparison.py   <- the harness
  .gitignore                     <- ignores .assets/ (downloaded source USDZ)
  chess/  apple/  warehouse/      <- one dir per set, each containing:
    frames/                       <- raw per-asset per-backend PNGs (committed)
    <asset>_<cam>_compare.png     <- OVRTX | RT | Raster
    contact_sheet.png
    metrics.json
    README.md
    wrappers/<label>.usda         <- record copy of the generated sub-layer wrapper
                                     (the operative copy is co-located with the asset)
```
