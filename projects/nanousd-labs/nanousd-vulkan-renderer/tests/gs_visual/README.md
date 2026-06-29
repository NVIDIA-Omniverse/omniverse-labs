# VK3DGRT visual regression tests

Visual regression harness for the splat (3D Gaussian Ray Tracing) path.
Mirrors `tests/correctness/run_tests.py` but exercises `nu_gs_render`
instead of the mesh-RT path.

Each test renders a deterministic Gaussian-splat scene, dumps the RGBA
PNG under `output/`, and diffs it against a checked-in `golden/<label>.png`
via per-pixel RMS (threshold = 2.0 LSB by default — splat path has no
jitter / no Monte Carlo, so drift across drivers should stay within a
single LSB unless something has actually regressed).

## What this catches

- particle SSBO upload / staging layout
- `gs_as_build` push-constant transpose (Slang column-major vs C
  row-major) — a flipped transpose would shift every splat
- TLAS instance-custom-index plumbing — a swap re-orders the grayscale
  gradient `gs_3dgrt.rchit.slang` paints
- `rt_image → swapchain` blit inside `gpu_gs_render_tile`
- SBT entry order / handle alignment

## Running

```bash
# Build the renderer first (NU_BUILD_GS_RT=ON).
./build.sh

# Run all tests.
NUSD_RENDERER_LIB="$(pwd)/build/libnusd_renderer.so" \
  PYTHONPATH=python \
  python3 tests/gs_visual/run_tests.py

# Run one.
python3 tests/gs_visual/run_tests.py --only grid12

# Regenerate goldens (after an intentional shader / push-constant change).
python3 tests/gs_visual/run_tests.py --update
```

## When to regenerate goldens

The golden was captured with the **Pragmatist rgen** (paints by
`InstanceID() / particle_count`). When the K-buffer trace loop
replaces the rgen body, the test will fail until you run with `--update`
and check in the new golden.

## Adding a test

Append to `TESTS` in `run_tests.py`:

```python
TESTS = [
    ("grid12",  256, 256, 5),
    ("my_test", 512, 512, 50),  # label, w, h, min unique R values
]
```

Then add a corresponding scene-builder helper alongside `_make_grid12()`
and call it from `_render()`. Keep scenes small + deterministic — these
are regression tests, not benchmarks.
