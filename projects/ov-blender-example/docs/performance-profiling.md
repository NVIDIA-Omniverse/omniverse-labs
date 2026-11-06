# Blender-side performance profiling

The add-on has opt-in Blender-process profiling for development and performance
investigation. It combines:

- one pyinstrument call tree per add-on thread;
- exact inclusive wall timers based on `time.perf_counter_ns()`; and
- one JSON index containing count, total, mean, min, recent p50/p95, max, and
  thread attribution for every named scope.

Profiling is disabled unless `OV_BLENDER_EXAMPLE_PYINSTRUMENT_DIR` is present
when the add-on is imported. Disabled builds do not import pyinstrument, and
decorated non-Blender boundaries retain their original functions.

## Setup

pyinstrument is a development dependency and is not bundled into the extension.
Install it for the same Python ABI as Blender, or into an isolated target. For a
Blender build using CPython 3.13 on Windows:

```powershell
uv pip install --target .tmp\pyinstrument `
  --python-version 3.13 --python-platform windows pyinstrument==5.1.3

$env:OV_BLENDER_EXAMPLE_PYINSTRUMENT_DIR = `
  (Resolve-Path _build).Path + '\results\pyinstrument'
$env:OV_BLENDER_EXAMPLE_PYINSTRUMENT_PATH = `
  (Resolve-Path .tmp\pyinstrument).Path
```

`OV_BLENDER_EXAMPLE_PYINSTRUMENT_PATH` is optional when pyinstrument is already
importable by Blender. `OV_BLENDER_EXAMPLE_PYINSTRUMENT_INTERVAL` changes the
sampling interval in seconds; the default is `0.001`.

Launch Blender and exercise the workload after setting the variables. Keep the
output directory outside the source tree. The add-on writes:

- `ovrtx-blender-performance.json`: exact timer summary and capture index;
- `*.html`: interactive pyinstrument call trees;
- `*.txt`: full text call trees;
- `*.json`: machine-readable pyinstrument trees; and
- `*.pyisession`: reusable pyinstrument sessions.

Separate captures cover Blender's main thread, each viewport render thread,
final-render jobs, authoring preparation, runtime service/install/remove work,
and physics pose production when those paths run.

## Reading the numbers

Timer scopes are inclusive. A parent such as `render_thread.iteration` contains
children such as `render_thread.runtime_tick` and
`render_thread.acquire_sample`; do not add nested scopes together. The p50 and
p95 values describe up to the most recent 4096 samples, while count, total,
mean, min, and max cover the full capture.

pyinstrument is a wall-clock sampling profiler, so native calls and waits remain
visible under the Python call that initiated them. Profiled throughput is
expected to be slower than an uninstrumented gate. Use an unprofiled repetition
for release performance claims and the profiled repetition for attribution.

If profiling was requested but pyinstrument cannot be imported, the JSON index
records the import error and exact scope timers remain available.
