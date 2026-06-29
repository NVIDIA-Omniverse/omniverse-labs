# nanousd Backend Benchmarks

`harness/nanousd_backend_benchmark.py` runs nanousd's C API stage-load
benchmark across a backend matrix. It is intended to compare the generated
skillgraph backend against nanousd's default backend and, when available, the
OpenUSD backend.

The generated backend now implements the stage-open slice needed by this
benchmark for supported local USDA fixtures: open/close, validity, direct local
dependency opening through `usd-composed-stage-open`, prim traversal,
attribute-count queries, diagnostics, and root-layer/layer enumeration. Timing
comparisons remain optional until separate performance contracts add gates.

## Build nanousd Benchmark Tool

Build nanousd with a shared `nanousdapi` dispatch library. This matters because
the static API build links the default backend into the executable, which can
short-circuit `NANOUSD_BACKEND` selection before the environment-provided
backend is loaded.

```powershell
& "C:\Users\eslavin\AppData\Local\Programs\CMake\bin\cmake.exe" `
  -S C:\tmp\nanousd `
  -B C:\tmp\nanousd\build-shared-api `
  -DNANOUSD_SHARED_API=ON `
  -DNANOUSD_BUILD_TESTS=ON

& "C:\Users\eslavin\AppData\Local\Programs\CMake\bin\cmake.exe" `
  --build C:\tmp\nanousd\build-shared-api `
  --config Release `
  --target benchmark_stage_load
```

On Linux or macOS, the equivalent command is:

```bash
cmake -S /tmp/nanousd -B /tmp/nanousd/build-shared-api \
  -DCMAKE_BUILD_TYPE=Release \
  -DNANOUSD_SHARED_API=ON \
  -DNANOUSD_BUILD_TESTS=ON
cmake --build /tmp/nanousd/build-shared-api --target benchmark_stage_load
```

## Run a Baseline

The default backend is selected by leaving `NANOUSD_BACKEND` unset:

```powershell
py harness\nanousd_backend_benchmark.py `
  --targets benchmarks\nanousd-backend\stage-load.json `
  --benchmark-cmd C:\tmp\nanousd\build-shared-api\Release\benchmark_stage_load.exe `
  --backend nanousd_default=DEFAULT
```

## Compare Backends

Pass additional backends as `id=path`. The runner sets `NANOUSD_BACKEND` for
each non-default backend before invoking `benchmark_stage_load`.

```powershell
py harness\nanousd_backend_benchmark.py `
  --targets benchmarks\nanousd-backend\stage-load.json `
  --benchmark-cmd C:\tmp\nanousd\build-shared-api\Release\benchmark_stage_load.exe `
  --backend nanousd_default=DEFAULT `
  --backend generated=generated\cpp\nanousd_readonly_backend.dll `
  --json-out reports\nanousd-backend\stage-load.json
```

On Linux:

```bash
python3 harness/nanousd_backend_benchmark.py \
  --targets benchmarks/nanousd-backend/stage-load.json \
  --benchmark-cmd /tmp/nanousd/build-shared-api/benchmark_stage_load \
  --backend nanousd_default=DEFAULT \
  --backend generated=generated/cpp/libnanousd_readonly_backend.so \
  --json-out reports/nanousd-backend/stage-load.json
```

The first target file intentionally marks generated-vs-default ratios as
optional. The generated backend is expected to complete the fixture runs, but
the ratios become gates only after we add explicit performance contracts.

## Report Results

Use `harness/nanousd_backend_report.py` to turn the benchmark JSON into a
Markdown review artifact and an optional SVG chart:

```powershell
py harness\nanousd_backend_report.py `
  reports\nanousd-backend\stage-load.json `
  --markdown-out reports\nanousd-backend\stage-load.md `
  --svg-out reports\nanousd-backend\stage-load-open-ms.svg `
  --svg-metric open_ms `
  --baseline nanousd_default `
  --candidate generated
```

The Markdown report includes best-run tables, declared comparison status, and
generated-vs-baseline pressure ratios. The SVG is a quick visual for one metric;
`open_ms`, `traverse_ns_per_prim`, and `rss_delta_bytes_per_prim` are the most
useful starting points.

## Signals

The runner parses these signals from `benchmark_stage_load`:

- `open_ms`
- `traverse_ms`
- `traverse_ns_per_prim`
- `prim_handle_ns_per_prim`
- `nattribs_ns_per_prim`
- `freeprim_ns_per_prim`
- `diagnostics_ms`
- `rss_delta_bytes`
- `rss_delta_bytes_per_prim`
- `rss_peak_bytes`
- `total_prims`
- `total_attributes`

Use regressions to decide which graph contract needs pressure:

- high `open_ms`: layer-open, composition, or stage-population contracts
- high traversal cost: stage indexes and prim-handle lifetime
- high `nattribs_ns_per_prim`: document-model property storage
- high RSS per prim: path, token, value, or adapter JSON retention
