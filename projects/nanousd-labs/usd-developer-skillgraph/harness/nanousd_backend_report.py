#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render reports from nanousd backend benchmark JSON."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any


DEFAULT_REPORT_METRICS = [
    "total_ms",
    "open_ms",
    "traverse_ms",
    "traverse_ns_per_prim",
    "nattribs_ns_per_prim",
    "rss_delta_bytes",
    "rss_delta_bytes_per_prim",
]

PRESSURE_METRICS = [
    "open_ms",
    "traverse_ns_per_prim",
    "nattribs_ns_per_prim",
    "rss_delta_bytes_per_prim",
]

METRIC_LABELS = {
    "total_ms": "total ms",
    "open_ms": "open ms",
    "traverse_ms": "traverse ms",
    "traverse_ns_per_prim": "traverse ns/prim",
    "prim_handle_ns_per_prim": "prim handle ns/prim",
    "nattribs_ns_per_prim": "nattribs ns/prim",
    "freeprim_ns_per_prim": "freeprim ns/prim",
    "rss_delta_bytes": "RSS delta",
    "rss_delta_bytes_per_prim": "RSS delta bytes/prim",
    "rss_peak_bytes": "RSS peak",
    "total_prims": "prims",
    "total_attributes": "attributes",
}

BACKEND_COLORS = [
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def benchmark_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in data.get("results", []) if item.get("ok") and isinstance(item.get("best"), dict)]


def ordered_backends(results: list[dict[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for result in results:
        seen.setdefault(str(result["backend"]), None)
    return list(seen)


def ordered_targets(results: list[dict[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for result in results:
        seen.setdefault(str(result["target"]), None)
    return list(seen)


def by_target_backend(results: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(result["target"]), str(result["backend"])): result for result in results}


def best_metric(result: dict[str, Any], metric: str) -> float | None:
    value = result.get("best", {}).get(metric)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def format_metric(metric: str, value: Any) -> str:
    if value is None:
        return "-"
    if metric.endswith("_bytes") or metric == "rss_delta_bytes":
        return format_bytes(float(value))
    if metric.endswith("_bytes_per_prim"):
        return f"{float(value):.1f}"
    if metric.endswith("_ms"):
        return f"{float(value):.3f}"
    if metric.endswith("_ns_per_prim"):
        return f"{float(value):.1f}"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def format_bytes(value: float) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    magnitude = float(value)
    unit = units[0]
    for unit in units:
        if abs(magnitude) < 1024.0 or unit == units[-1]:
            break
        magnitude /= 1024.0
    if unit == "B":
        return f"{int(value)} B"
    return f"{magnitude:.2f} {unit}"


def format_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    if math.isinf(value):
        return "inf"
    return f"{value:.3f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def report_rows(results: list[dict[str, Any]], metrics: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for result in results:
        best = result["best"]
        row = [
            str(result["target"]),
            str(result["backend"]),
            str(best.get("total_prims", "-")),
            str(best.get("total_attributes", "-")),
        ]
        row.extend(format_metric(metric, best.get(metric)) for metric in metrics)
        rows.append(row)
    return rows


def comparison_rows(comparisons: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for comparison in comparisons:
        rows.append([
            str(comparison.get("id", "")),
            str(comparison.get("status", "")),
            str(comparison.get("target", "")),
            METRIC_LABELS.get(str(comparison.get("metric", "")), str(comparison.get("metric", ""))),
            format_ratio(comparison.get("ratio")),
            format_ratio(comparison.get("max_ratio")),
            "yes" if comparison.get("required") else "no",
        ])
    return rows


def pressure_rows(
    results: list[dict[str, Any]],
    baseline: str,
    candidate: str,
) -> list[list[str]]:
    lookup = by_target_backend(results)
    rows: list[list[str]] = []
    for target in ordered_targets(results):
        base = lookup.get((target, baseline))
        cand = lookup.get((target, candidate))
        if not base or not cand:
            continue
        for metric in PRESSURE_METRICS:
            base_value = best_metric(base, metric)
            candidate_value = best_metric(cand, metric)
            if base_value is None or candidate_value is None:
                continue
            if base_value == 0.0:
                ratio = 1.0 if candidate_value == 0.0 else math.inf
            else:
                ratio = candidate_value / base_value
            rows.append([
                target,
                METRIC_LABELS.get(metric, metric),
                format_metric(metric, base_value),
                format_metric(metric, candidate_value),
                format_ratio(ratio),
            ])
    return rows


def relative_link(from_file: Path | None, to_file: Path) -> str:
    if from_file is None:
        return str(to_file).replace("\\", "/")
    try:
        return str(to_file.relative_to(from_file.parent)).replace("\\", "/")
    except ValueError:
        return str(to_file).replace("\\", "/")


def render_markdown(
    data: dict[str, Any],
    *,
    source: Path,
    markdown_out: Path | None,
    svg_out: Path | None,
    svg_metric: str,
    baseline: str,
    candidate: str,
) -> str:
    results = benchmark_results(data)
    metrics = DEFAULT_REPORT_METRICS
    measurement = data.get("measurement_model", {})
    title = data.get("suite", "nanousd-backend-stage-load")

    lines = [
        f"# {title}",
        "",
        f"Source: `{source}`",
        "",
    ]
    if measurement.get("id"):
        lines.extend([f"Measurement: `{measurement['id']}`", ""])
    if measurement.get("description"):
        lines.extend([str(measurement["description"]), ""])
    if svg_out is not None:
        lines.extend([
            f"![{METRIC_LABELS.get(svg_metric, svg_metric)} chart]({relative_link(markdown_out, svg_out)})",
            "",
        ])

    headers = ["target", "backend", "prims", "attributes"]
    headers.extend(METRIC_LABELS.get(metric, metric) for metric in metrics)
    lines.extend([
        "## Best Runs",
        "",
        markdown_table(headers, report_rows(results, metrics)),
        "",
    ])

    comparisons = data.get("comparisons", [])
    if comparisons:
        lines.extend([
            "## Declared Comparisons",
            "",
            markdown_table(
                ["id", "status", "target", "metric", "ratio", "max ratio", "required"],
                comparison_rows(comparisons),
            ),
            "",
        ])

    pressure = pressure_rows(results, baseline, candidate)
    if pressure:
        lines.extend([
            f"## {candidate} vs {baseline}",
            "",
            markdown_table(
                ["target", "metric", baseline, candidate, "ratio"],
                pressure,
            ),
            "",
        ])

    lines.extend([
        "## Reading The Signals",
        "",
        "- high open time points at layer-open, composition, parser, or stage-population cost",
        "- high traversal cost points at stage indexes and handle lifetime behavior",
        "- high attribute-count cost points at property storage and lookup shape",
        "- high RSS points at path, token, value, or adapter JSON retention",
        "",
    ])
    return "\n".join(lines)


def render_svg(data: dict[str, Any], metric: str) -> str:
    results = benchmark_results(data)
    targets = ordered_targets(results)
    backends = ordered_backends(results)
    lookup = by_target_backend(results)
    values = [
        best_metric(result, metric)
        for result in results
        if best_metric(result, metric) is not None
    ]
    max_value = max(values) if values else 1.0
    if max_value <= 0.0:
        max_value = 1.0

    left = 86
    top = 56
    right = 28
    bottom = 92
    chart_height = 260
    bar_width = 28
    bar_gap = 8
    group_gap = 38
    group_width = max(1, len(backends)) * bar_width + max(0, len(backends) - 1) * bar_gap
    width = max(720, left + right + len(targets) * group_width + max(0, len(targets) - 1) * group_gap)
    height = top + chart_height + bottom
    axis_y = top + chart_height

    def x_for(target_index: int, backend_index: int) -> int:
        return left + target_index * (group_width + group_gap) + backend_index * (bar_width + bar_gap)

    def y_for(value: float) -> float:
        return axis_y - (value / max_value) * chart_height

    title = f"nanousd backend comparison: {METRIC_LABELS.get(metric, metric)}"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="28" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#111827">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{axis_y}" x2="{width - right}" y2="{axis_y}" stroke="#111827" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{axis_y}" stroke="#111827" stroke-width="1"/>',
    ]

    for tick in range(0, 5):
        value = max_value * tick / 4.0
        y = y_for(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#374151">{html.escape(format_metric(metric, value))}</text>')

    for target_index, target in enumerate(targets):
        group_center = left + target_index * (group_width + group_gap) + group_width / 2.0
        parts.append(f'<text x="{group_center:.1f}" y="{axis_y + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">{html.escape(target)}</text>')
        for backend_index, backend in enumerate(backends):
            result = lookup.get((target, backend))
            value = best_metric(result, metric) if result else None
            if value is None:
                continue
            x = x_for(target_index, backend_index)
            y = y_for(value)
            bar_height = max(1.0, axis_y - y)
            color = BACKEND_COLORS[backend_index % len(BACKEND_COLORS)]
            parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{color}"/>')
            parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#111827">{html.escape(format_metric(metric, value))}</text>')

    legend_y = height - 36
    legend_x = left
    for backend_index, backend in enumerate(backends):
        color = BACKEND_COLORS[backend_index % len(BACKEND_COLORS)]
        x = legend_x + backend_index * 170
        parts.append(f'<rect x="{x}" y="{legend_y}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{x + 18}" y="{legend_y + 11}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{html.escape(backend)}</text>')

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_json", type=Path, help="JSON output from harness/nanousd_backend_benchmark.py")
    parser.add_argument("--markdown-out", type=Path, help="Write a Markdown report instead of stdout")
    parser.add_argument("--svg-out", type=Path, help="Write a grouped bar-chart SVG")
    parser.add_argument("--svg-metric", default="open_ms", help="Metric to chart when --svg-out is provided")
    parser.add_argument("--baseline", default="nanousd_default", help="Baseline backend id for ratio tables")
    parser.add_argument("--candidate", default="generated", help="Candidate backend id for ratio tables")
    args = parser.parse_args()

    data = load_json(args.benchmark_json)
    if args.svg_out:
        write_text(args.svg_out, render_svg(data, args.svg_metric))

    report = render_markdown(
        data,
        source=args.benchmark_json,
        markdown_out=args.markdown_out,
        svg_out=args.svg_out,
        svg_metric=args.svg_metric,
        baseline=args.baseline,
        candidate=args.candidate,
    )
    if args.markdown_out:
        write_text(args.markdown_out, report)
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
