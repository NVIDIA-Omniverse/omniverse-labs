---
name: nanousd-benchmark-reporting
description: Use this skill when presenting benchmark results that compare the generated skillgraph nanousd backend against nanousd's default backend or another baseline backend.
metadata:
  author: NVIDIA
---

# nanousd-benchmark-reporting

Use this skill when presenting benchmark results that compare the generated
skillgraph nanousd backend against nanousd's default backend or another
baseline backend.

## Provides

- Benchmark result reporting for `harness/nanousd_backend_benchmark.py`
- Markdown tables for best-run metrics and declared comparison ratios
- Generated-vs-baseline pressure tables for metrics that suggest contract work
- Optional SVG charts for visual inspection of one metric across targets and
  backends

## Contract

This skill owns:

- `contracts/capabilities/nanousd-benchmark-reporting.json`
- `harness/nanousd_backend_report.py`
- the reporting section of `docs/nanousd-backend-benchmarks.md`

The reporter consumes JSON emitted by `harness/nanousd_backend_benchmark.py`.
It must not rerun benchmarks, parse benchmark stdout, or silently change the
selection metric chosen by the benchmark runner. The benchmark runner owns
measurement and best-run selection; the report owns presentation.

Reports should include:

- benchmark suite and measurement id
- source benchmark JSON path
- a best-run table by target and backend
- declared comparison status, ratio, threshold, and required/optional state
- generated-vs-baseline pressure ratios for open, traversal, attribute count,
  and RSS signals
- an optional chart when a visual signal makes regression review easier

## Reporting Guidance

Use Markdown as the durable review artifact. SVG charts are useful for quick
visual comparison, but they should supplement tables rather than replace them.

The default chart metric is `open_ms`, because open cost is currently the
highest-pressure signal for the generated backend. Use
`rss_delta_bytes_per_prim` or `traverse_ns_per_prim` when reviewing memory or
traversal-specific work.

Optional `SLOW` comparisons are performance pressure, not PR failure. Required
comparisons should be treated as gates only when a performance contract makes
that requirement explicit.

## Boundary Guards

Do not track generated report outputs under `reports/`; they are local review
artifacts.

Do not conflate nanousd default backend attribute counts with generated authored
attribute counts unless a correctness contract says they should match. Report
the observed values and let the relevant backend contract decide whether a
difference is meaningful.

Do not hide missing or failed backend results. A report may still render partial
data, but it must make missing backend comparisons visible.

## Test Obligations

- reporter can read benchmark JSON produced by
  `harness/nanousd_backend_benchmark.py --json-out`
- Markdown output includes best-run and comparison tables
- SVG output uses only inline SVG and no external assets
- generated-vs-baseline pressure rows preserve metric units and ratios
