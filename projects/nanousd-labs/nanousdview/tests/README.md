# nanousdview Golden Image Tests

`render_goldens.py` renders scenes through the nanousdview screenshot path and
compares the captured viewport pixels against checked-in golden images.

Goldens are backend-specific by default:

```text
tests/golden/
  vulkan/<case>.ppm
  vulkan/depth/<case>.ppm
  vulkan/normals/<case>.ppm
  vulkan/segmentation/<case>.ppm
  opengl/<case>.ppm
  metal/<case>.ppm
```

This avoids false failures from legitimate backend differences. If you want
cross-backend parity checks, compare every backend against one backend's
goldens with `--reference-backend`.

## First-Time Capture

Create goldens on machines that can run each backend:

```bash
cd nanousdview
python3 tests/render_goldens.py --backend vulkan --update
python3 tests/render_goldens.py --backend opengl --update  # opengles is an alias
python3 tests/render_goldens.py --backend ovrtx --update   # NVIDIA OVRTX runtime
python3 tests/render_goldens.py --backend metal --update   # macOS only
```

Review image diffs before committing refreshed goldens.
The current nanousdview OVRTX viewport path captures color only; requested
depth, normals, segmentation, and environment-map captures are reported as
unsupported and skipped unless strict AOV handling is enabled.

## Regression Runs

```bash
python3 tests/render_goldens.py --backend vulkan
python3 tests/render_goldens.py --backend vulkan --aov color,depth,normals,segmentation
python3 tests/render_goldens.py --backend vulkan,opengl,ovrtx,metal
python3 tests/render_goldens.py --backend opengl,metal --reference-backend vulkan
python3 tests/render_goldens.py --backend opengl --include-opengl-renderer-assets
```

Unavailable backends are skipped by default. Unsupported requested AOVs are
also skipped by default, which lets a mixed backend run request
`--aov color,depth,normals,segmentation` while OpenGL remains color-only. Use
`--strict-backends` or `--strict-aovs` when CI must fail instead.

Outputs, `summary.json`, `report.html`, and `contact_sheet.png` are written to
`tests/output/` and ignored by git. When a golden exists, the harness also
writes `<case>.diff.ppm` beside the captured image. The report converts PPMs to
PNG thumbnails using the Python standard library, so no Pillow dependency is
required.

The harness uses the same Qt viewport path as the interactive application, so
GPU backends generally need a real display server. Run it from a desktop
session, under a CI display wrapper such as `xvfb-run`, or pass explicit display
settings:

```bash
python3 tests/render_goldens.py --backend vulkan --display :1
python3 tests/render_goldens.py --backend vulkan --qt-platform xcb --display :99
```

CTest registers a cheap help/smoke test by default. To register the full golden
comparison in CTest, configure with:

```bash
cmake -S . -B build \
  -DNANOUSDVIEW_ENABLE_GOLDEN_TESTS=ON \
  -DNANOUSDVIEW_GOLDEN_BACKENDS=vulkan,opengl
ctest --test-dir build --output-on-failure -L golden
```

Extra harness arguments can be passed as a semicolon-separated CMake list:

```bash
cmake -S . -B build \
  -DNANOUSDVIEW_ENABLE_GOLDEN_TESTS=ON \
  '-DNANOUSDVIEW_GOLDEN_CTEST_ARGS=--only;test_cube;--strict-backends'
```

The golden CTest uses return code `77` for all-skipped runs, so Linux can
register Metal handoff coverage without failing local test runs. CTest labels
include `golden`, `renderer`, `viewer`, and optionally `gpu`, `reference`,
`asset`, and `slow` based on configured backends and arguments.

## Deterministic Capture Options

The harness forwards these options to `nanousdview`. At the moment only color
AOV captures are implemented; unsupported requests are skipped by default.

```bash
python3 tests/render_goldens.py \
  --backend vulkan \
  --aov normals \
  --render-mode rt \
  --frame 12 \
  --camera /World/Camera \
  --envmap path/to/studio.hdr
```

`nanousdview` accepts the same flags directly with `--screenshot`.

## Scenes

The default scene is `test_cube.usda`, which is checked into this repo. Add
more cases without editing the script:

```bash
python3 tests/render_goldens.py \
  --case cube=test_cube.usda \
  --case scanner=../nanousd-vulkan-renderer/tests/correctness/assets/agibot_scanner.usda
```

For convenience, `--include-renderer-correctness-assets` adds the checked-in
Vulkan renderer correctness scenes when that sibling repo is present.
`--include-opengl-renderer-assets` adds portable OpenGL renderer fixtures
covering mesh, material, texture, PBR, and BasisCurves paths.
