---
keywords:
  - verification
  - junk shop
  - benchmark
  - render
---

# Junk Shop verification workload

A single reproducible render you can run to check that an installed
`ov-blender-example` is working, and to compare your machine against the
reference results. The manual run below needs only Blender, the installed
release and the public scene file.

## The scene

| | |
| --- | --- |
| Scene | Blender's *The Junk Shop* splash (Alex Treviño / Blender Studio, CC BY) |
| Fixture id | `perf_junk_shop_1280x720` |
| File | `thejunkshopsplashscreen-35a35553b3dd4f8c8fb5a6ccc5065ff1.blend` from [download.blender.org](https://download.blender.org/demo-files/archives/art-gallery/blender-splash-screens/blender-2-81/thejunkshopsplashscreen-35a35553b3dd4f8c8fb5a6ccc5065ff1.blend) (also recorded in `tests/fixtures/perf_junk_shop_1280x720/spec.json`) |
| sha256 | `33191670ef370dcaa5fa2483c7d75abc9b8ce106ecea33cd155dbfc95ff649d2` |
| Size | 412,530,532 bytes |

**Check the hash before you draw any conclusion from a number.** There is more
than one Junk Shop `.blend` in circulation and they are not interchangeable — two
of this project's own measurement lanes once ran on different copies and their
results were compared as if they matched. Every artifact below records the hash
it actually opened, so a mismatch is visible rather than silent.

```bash
sha256sum thejunkshopsplashscreen-35a35553b3dd4f8c8fb5a6ccc5065ff1.blend
```

## Render settings

| Setting | Value |
| --- | --- |
| Engine | `OVRTX_EXAMPLE` |
| Camera | `SplashScreen Camera` (**not** `Original Camera`, which is also in the file) |
| Resolution | 1280 × 720 at 100 % |
| Samples | min 1, max 64 |
| Presentation | `scene_linear_hdr` |

The `.blend` ships at 2000 × 1000 with Cycles and its own camera. Override all
three — a run at the file's native settings is not comparable with anything here.

## Running it

1. Open the `.blend` in the Blender you installed the release into.
2. In the Render properties, select the `OVRTX_EXAMPLE` engine and set the
   samples to min 1, max 64.
3. In the Output properties, set the resolution to 1280 × 720 at 100 %.
4. Make `SplashScreen Camera` the active scene camera.
5. Render with **Render > Render Image**.

Compare the result with the approved reference frame,
[`goldens/perf_junk_shop_1280x720/single-frame/frame.png`](goldens/perf_junk_shop_1280x720/single-frame/frame.png).
Sampling noise means the two never match exactly, but the composition, lighting
and overall brightness should agree.

The automated render harness, the scene-linear comparison tool and the
performance-gate runner that produced the numbers below belong to the upstream
build repository and are not part of this project. The tables document what
they measured on the reference system.

## Expected output

A path-traced Junk Shop render at 64 samples. Sampling noise means two runs are
never bit-identical, so the upstream comparison of the scene-linear HDR buffer
against the reference accepts a tolerance, not equality:

| Check | Threshold |
| --- | --- |
| Mean scene-linear luma ratio, full frame | within ±5 % of 1.0 |
| Same, on the central ROI | within ±5 % of 1.0 |
| Pearson correlation of the two luma fields | ≥ 0.95 |
| New scene-black pixels | ≤ 0.1 % of the frame |
| Non-finite / outlier pixels | no worse than the reference |

The correlation check is the one that matters most and it is not redundant. Mean
luma is invariant under pixel permutation: a **left-right mirrored frame scores a
perfect 1.00000 luma ratio**. Correlation separates them — a genuine re-render
scores ~0.996, a mirror ~0.16.

Observed run-to-run on one machine: luma ratios 0.998978 and 1.001287,
correlations 0.9968 and 0.9976. **Treat ±1 % as normal variance.**

## Performance checkpoints

Reference system: NVIDIA RTX A6000 (48 GiB, driver 597.06), AMD Ryzen
Threadripper PRO 5965WX (24 cores), 256 GiB RAM, Windows 11 build 26200, Vulkan.

| Gate | Measured | Budget |
| --- | --- | --- |
| G1 — cold init to worker ready | 1.50 s | 300 s |
| G2 — first render (scene generation + USD export + cold-cache frame) | 27.87 s | 180 s |
| Full 64-sample render | 30.85 s | — |
| G3 — steady-state navigation frame time | 40.98 ms | 15 ms — **currently failing** |

Two of these deserve context rather than being read off the table:

- **G3 is a known failure**, not a fault in your setup. It is measured by a
  separate navigation harness, and closing the gap is open work. If you reproduce
  ~41 ms you have reproduced the current state correctly.
- **G2 is dominated by USD export** (~18 s of the 27.87 s), which is CPU-bound
  scene conversion rather than rendering.

The upstream gate runner reports each gate's pass/fail state and fails overall
today because of G3.

## Failure logs

There are **two** logs and they contain different things. Reading only the first
is the most common way to misdiagnose this stack.

| Log | Where | Contains |
| --- | --- | --- |
| Renderer | `<runtime>/runtime/ovrtx-bridge-server/ovrtx_<timestamp>.log` | the renderer's own output: shader compilation, GPU/driver detection, scene-graph errors |
| Worker | wherever `OV_BLENDER_EXAMPLE_WORKER_LOG` points | the bridge's output: stage load, render-path selection, per-step drive lines |

`<runtime>` is the extension's runtime store, the `ovrtx-sdk` folder in the
Blender settings folder you installed into (or the folder named by its
`native-root.json`). The worker log is **not** written by default — the worker
inherits Blender's stdout. Set it before starting Blender:

```bash
export OV_BLENDER_EXAMPLE_WORKER_LOG=/path/to/worker.log   # Windows: set / $env:
```

### Reading them

- `Shader caches are missing from the application` — the runtime's
  `cache/shadercache` is incomplete. It is part of the install; restore the
  runtime tree rather than trying to rebuild it.
- `ovphysx_grpc_server exited before readiness` — usually the OS holding the
  port, not a broken worker. See [runtime boundary](runtime-boundary.md); the
  add-on names the reserving range when it can.
- `external Fabric not found for UsdStageId ...` — the renderer and the staging
  library are bound to different Fabric runtimes. A packaging problem, not a
  scene problem.
- **No errors but no output** — check the *worker* log. A render that keeps
  stepping without finishing looks identical to a hang in the renderer log.

## What a result does and does not tell you

A render takes one of two paths, `ovstage` or `direct_usd`; the worker log's
render-path selection line says which. Both are valid and both pass the checks
above, but they are different code paths, and a number from one is not evidence
about the other. Say which one you measured, and which release package you
installed.
