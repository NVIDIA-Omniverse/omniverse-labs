#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Automated OpenGL-vs-OVRTX renderer-diff pipeline driven by NVIDIA ϜLIP.

Compares every OpenGL output against its OVRTX golden using
the calibrated FLIP metric in ``flip_compare`` and produces:

  * a per-pair FLIP heatmap strip (``golden | metal | FLIP``) under ``flip/``,
  * ``flip/results.json``  — machine-readable scores + diagnoses,
  * ``flip/REPORT.md``     — human summary with per-scene tables and worst offenders,
  * a pass/fail verdict against ``flip/baseline.json`` (regression gate, exit code).

Reference is always the OVRTX golden (``*_ovrtx.png``). Lower FLIP == closer to golden.

Usage:
  python flip_pipeline.py                 # analyse committed frames -> report + heatmaps
  python flip_pipeline.py --check         # also gate vs baseline; exit 1 on regression
  python flip_pipeline.py --update-baseline   # freeze current scores as the baseline
  python flip_pipeline.py --render        # re-render OpenGL frames first (needs assets)
  python flip_pipeline.py --sets warehouse    # restrict to one scene set

Run with the project venv (numpy, Pillow, flip-evaluator):
  NUSD_RENDERER_LIB / NUSD_*_USD env vars are only needed for --render.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    import flip_compare as fc  # noqa: E402  (pulls in numpy, Pillow, flip-evaluator)
except ModuleNotFoundError as exc:  # graceful CTest skip when the venv deps are absent
    print(f"SKIP: Python dependency missing: {exc.name} (pip install flip-evaluator)")
    raise SystemExit(77) from exc

SETS = ("chess", "apple", "warehouse")
RENDERER = "OpenGL"
BACKENDS = ("opengl",)  # tag -> golden is "<base>_ovrtx.png"
# Regression gate. The score is studio = mean FLIP over the foreground subject,
# fullscene = 95th-percentile FLIP (the mean is gameable on full scenes — a
# tone-matched-but-murky render beats a good one — so we pool the worst regions).
# A render may not get worse than its per-key baseline by more than REGRESSION_TOL,
# and may never exceed the (mode-specific) absolute ceiling. Several orthogonal legs
# stop a render passing the gate by exploiting any single axis:
#   - GMSD (full): brightness-invariant structure (catches crushed/over-smoothed
#     geometry that a tone offset hides);
#   - crushed-black guard (full): test may not black out salient geometry beyond golden;
#   - loose exposure backstop (full): catch gross mis-exposure only;
#   - CIEDE2000 colour (both modes): exposure-suppressed hue/chroma drift — catches a
#     'right brightness, wrong colour' regression (e.g. a floor that fades blue->grey)
#     that FLIP's lightness-heavy term and GMSD's brightness-invariant structure miss.
#     Pooled by MEAN, not p95: deltaE's worst regions are the legit rack/GI colour gaps,
#     so a localised colour fault shows in the mean, not the (saturated) tail.
REGRESSION_TOL = 0.03            # score regression tolerance (studio mean / fullscene p95)
ABSOLUTE_CEILING_STUDIO = 0.95   # studio masked-mean backstop (loose; relative gate is primary)
ABSOLUTE_CEILING_FULL = 0.99     # fullscene p95 backstop (good single-pass ~0.74-0.96; relative gate is primary)
GMSD_TOL = 0.03                  # fullscene structure-regression tolerance
BLACK_FRAC_TOL = 0.05            # fullscene crush guard: test black-fraction over golden's
EXPOSURE_CEILING_PCT = 80.0      # fullscene loose gross-mis-exposure backstop (|luma_delta_pct|)
DELTAE_TOL = 0.75                # CIEDE2000 colour-regression tolerance (both modes; ~0.75 ΔE units)


def _discover_pairs(frames_root: Path, sets: tuple[str, ...]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for set_name in sets:
        frames = frames_root / set_name / "frames"
        if not frames.is_dir():
            continue
        for golden in sorted(frames.glob("*_ovrtx.png")):
            base = golden.name[: -len("_ovrtx.png")]  # e.g. robot_camA / warehouse_camA
            for backend in BACKENDS:
                test = frames / f"{base}_{backend}.png"
                if test.exists():
                    pairs.append(
                        {"set": set_name, "base": base, "backend": backend,
                         "golden": golden, "test": test}
                    )
    return pairs


def _render(sets: tuple[str, ...]) -> None:
    """Re-render the OpenGL frames via the existing comparison harness (needs assets)."""
    for set_name in sets:
        print(f">>> re-rendering OpenGL frames: {set_name}")
        subprocess.run(
            [sys.executable, str(HERE / "render_backend_comparison.py"), "--set", set_name],
            check=True,
        )


def run(frames_root: Path, out_dir: Path, sets: tuple[str, ...]) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for p in _discover_pairs(frames_root, sets):
        res = fc.flip_vs_golden(p["golden"], p["test"])
        key = f"{p['set']}/{p['base']}/{p['backend']}"
        strip = out_dir / f"{p['set']}_{p['base']}_{p['backend']}_flip.png"
        fc.compare_strip(p["golden"], p["test"], res, strip)
        records.append({
            "key": key,
            "set": p["set"],
            "asset": p["base"].rsplit("_cam", 1)[0],
            "cam": "cam" + p["base"].rsplit("_cam", 1)[1] if "_cam" in p["base"] else "",
            "backend": p["backend"],
            "flip": round(res.score, 4),
            "mode": res.mode,
            "gmsd": round(res.gmsd, 4),
            "deltae": round(res.deltae, 3),
            "black_frac": round(res.black_frac, 4),
            "black_frac_golden": round(res.black_frac_golden, 4),
            "diagnosis": res.diagnosis(),
            "luma_delta_pct": round(res.tonal.get("luma_delta_pct", 0.0), 1),
            "rb_delta": round(res.tonal.get("rb_delta", 0.0), 1),
            "worst_region": res.worst_region,
            "strip": strip.name,
        })
        print(f"  FLIP {res.score:.3f}  {key:42s} :: {res.diagnosis()}")
    records.sort(key=lambda r: r["key"])
    return records


def gate(records: list[dict[str, Any]], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    """Annotate each record pass/fail. A render must clear EVERY applicable leg:
    absolute (mode ceiling; fullscene crush + gross-exposure) and relative-to-baseline
    (score regression; fullscene GMSD structure). Failing any one fails the gate, so a
    render cannot pass by gaming a single axis (the FLIP-mean exploit). Baseline entries
    are {"flip","gmsd","deltae"}; a bare float is tolerated as a legacy flip-only baseline,
    and a missing "deltae" key (legacy baseline) simply skips the colour leg."""
    for r in records:
        base = baseline.get(r["key"])
        if isinstance(base, dict):
            base_flip, base_gmsd, base_deltae = base.get("flip"), base.get("gmsd"), base.get("deltae")
        elif base is None:
            base_flip = base_gmsd = base_deltae = None
        else:
            base_flip, base_gmsd, base_deltae = float(base), None, None
        r["baseline"] = base_flip
        full = r.get("mode") == "fullscene"
        ceiling = ABSOLUTE_CEILING_FULL if full else ABSOLUTE_CEILING_STUDIO
        fails: list[str] = []
        # --- absolute legs (no baseline needed) ---
        if r["flip"] > ceiling:
            fails.append(f"exceeds {r.get('mode','?')} ceiling {ceiling}")
        if full:
            if r["black_frac"] > r["black_frac_golden"] + BLACK_FRAC_TOL:
                fails.append(f"crushed-black {r['black_frac']:.2f} > golden {r['black_frac_golden']:.2f}+{BLACK_FRAC_TOL}")
            if abs(r["luma_delta_pct"]) > EXPOSURE_CEILING_PCT:
                fails.append(f"gross mis-exposure {r['luma_delta_pct']:+.0f}%")
        # --- relative legs (need a baseline) ---
        if base_flip is not None and r["flip"] > base_flip + REGRESSION_TOL:
            fails.append(f"FLIP regressed +{r['flip'] - base_flip:.3f} vs {base_flip:.3f}")
        if full and base_gmsd is not None and r["gmsd"] > base_gmsd + GMSD_TOL:
            fails.append(f"GMSD structure regressed +{r['gmsd'] - base_gmsd:.3f} vs {base_gmsd:.3f}")
        if base_deltae is not None and r.get("deltae", 0.0) > base_deltae + DELTAE_TOL:
            fails.append(f"colour ΔE2000 regressed +{r['deltae'] - base_deltae:.2f} vs {base_deltae:.2f}")

        if fails:
            r["status"] = "fail"
            r["reason"] = "; ".join(fails)
        elif base is None:
            r["status"] = "new"
            r["reason"] = "no baseline entry"
        else:
            r["status"] = "pass"
            r["reason"] = ""
    return records


def run_semantic(frames_root: Path, sets: tuple[str, ...], records: list[dict[str, Any]],
                 top: int) -> list[dict[str, Any]]:
    """Run the VLM semantic diff (``semantic_diff.py``) on the `top` worst-FLIP frames.

    Returns ``[{key, mode, flip, diff}]`` for the report. A triage/report layer, never a gate: a VLM
    is non-deterministic and can hallucinate. Degrades gracefully — a missing anthropic SDK, missing
    API key, or per-frame API error is logged and skipped, never fatal."""
    try:
        import semantic_diff as sd  # noqa: E402  (optional; needs the anthropic SDK)
    except ModuleNotFoundError:
        print("  semantic: anthropic SDK not installed — skipping VLM diff (pip install anthropic)")
        return []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  semantic: ANTHROPIC_API_KEY unset — skipping VLM diff")
        return []
    paths = {f"{p['set']}/{p['base']}/{p['backend']}": p for p in _discover_pairs(frames_root, sets)}
    out: list[dict[str, Any]] = []
    for r in sorted(records, key=lambda r: -r["flip"])[: max(0, top)]:
        p = paths.get(r["key"])
        if p is None:
            continue
        try:
            diff = sd.semantic_diff(str(p["golden"]), str(p["test"]))
        except Exception as exc:  # noqa: BLE001 — network/API/quota: report and continue
            print(f"  semantic: {r['key']} failed: {exc}")
            continue
        out.append({"key": r["key"], "mode": r["mode"], "flip": r["flip"], "diff": diff})
        print(f"  semantic {r['key']:42s} :: {diff.get('verdict')} "
              f"[{'BUG' if diff.get('is_bug') else 'ok'}]")
    return out


def write_report(records: list[dict[str, Any]], out_dir: Path, gated: bool,
                 semantic: list[dict[str, Any]] | None = None) -> None:
    lines = [f"# {RENDERER} vs OVRTX — ϜLIP diff report", ""]
    lines.append(
        f"Each {RENDERER} backend output is compared to its OVRTX golden with the calibrated "
        "FLIP metric. Studio scenes: subject composited onto the golden background "
        "and foreground-masked, scored by mean FLIP. Full scenes: 95th-percentile FLIP "
        "(the mean is gameable — a tone-matched-but-murky render can beat a good one), "
        "plus orthogonal GMSD (structure), CIEDE2000 (exposure-suppressed colour), "
        "crushed-black and exposure gate legs. "
        "**Lower = closer to the golden.** Single-pass backends sit well above the "
        "path-traced golden even when they look right — read the *relative* ordering and "
        "the heatmaps, and judge full scenes by eye, not the scalar alone."
    )
    lines.append("")
    # Systematic finding headline — data-driven so it reads correctly for any
    # renderer (single-pass backends lack path-traced occlusion, but whether they
    # land brighter or darker than the golden is backend-specific).
    if records:
        nb = sum(1 for r in records if r["luma_delta_pct"] > 0)
        nd = len(records) - nb
        wf = max(records, key=lambda r: r["flip"])
        lines.append(
            f"**Systematic finding:** vs the OVRTX golden, {nb} {RENDERER} renders are brighter "
            f"and {nd} darker (single-pass lacks path-traced occlusion). Largest perceptual gap: "
            f"`{wf['key']}` at FLIP {wf['flip']:.3f} ({wf['diagnosis']})."
        )
        lines.append("")
    if gated:
        fails = [r for r in records if r["status"] == "fail"]
        new = [r for r in records if r["status"] == "new"]
        verdict = "✅ PASS" if not fails else f"❌ FAIL ({len(fails)} regressed)"
        lines += [f"**Regression gate: {verdict}**  "
                  f"(tol +{REGRESSION_TOL}; ceilings studio {ABSOLUTE_CEILING_STUDIO} / "
                  f"fullscene {ABSOLUTE_CEILING_FULL}; +GMSD/crush/exposure legs on full scenes; "
                  f"{len(new)} new)", ""]
        if fails:
            lines.append("Regressions:")
            for r in fails:
                lines.append(f"- `{r['key']}` — FLIP {r['flip']:.3f} — {r['reason']}")
            lines.append("")

    worst = sorted(records, key=lambda r: -r["flip"])[:5]
    lines.append("## Largest diffs from golden")
    lines.append("")
    lines.append("| pair | FLIP | GMSD | ΔE2000 | mode | diagnosis |")
    lines.append("|---|---|---|---|---|---|")
    for r in worst:
        lines.append(f"| `{r['key']}` | {r['flip']:.3f} | {r.get('gmsd', 0.0):.3f} | {r.get('deltae', 0.0):.2f} | {r['mode']} | {r['diagnosis']} |")
    lines.append("")

    for set_name in SETS:
        rows = [r for r in records if r["set"] == set_name]
        if not rows:
            continue
        lines.append(f"## {set_name}")
        lines.append("")
        hdr = "| pair | backend | FLIP |"
        if gated:
            hdr += " status | baseline |"
        hdr += " diagnosis |"
        lines.append(hdr)
        lines.append("|---|---|---|" + ("---|---|" if gated else "") + "---|")
        for r in sorted(rows, key=lambda r: (r["asset"], r["cam"], r["backend"])):
            label = f"{r['asset']} {r['cam']}".strip()
            row = f"| {label} | {r['backend']} | {r['flip']:.3f} |"
            if gated:
                b = f"{r['baseline']:.3f}" if r.get("baseline") is not None else "—"
                row += f" {r['status']} | {b} |"
            row += f" {r['diagnosis']} |"
            lines.append(row)
        lines.append("")

    if semantic:
        lines.append("## Semantic diff (VLM)")
        lines.append("")
        lines.append(
            "Claude vision describing, in words, how the worst-FLIP renders deviate from the golden — "
            "splitting *expected* cross-renderer tone/GI differences from *likely bugs* (the one call "
            "no per-pixel metric can make). Triage/explanation only — non-deterministic, not a gate."
        )
        lines.append("")
        for s in semantic:
            d = s["diff"]
            flag = "⚠ likely bug" if d.get("is_bug") else "no bug"
            lines.append(f"### `{s['key']}` — {d.get('verdict', '?')} ({flag}) — FLIP {s['flip']:.3f}")
            obs = d.get("observations") or []
            if not obs:
                lines.append("- _(no meaningful difference reported)_")
            for o in obs:
                tag = "🐞 likely-bug" if o.get("classification") == "likely-bug" else "expected"
                lines.append(f"- **{o.get('category')}** @ {o.get('region')} "
                             f"(sev {o.get('severity')}, {tag}): {o.get('detail')}")
            lines.append("")

    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenGL-vs-OVRTX FLIP diff pipeline")
    ap.add_argument("--frames-root", default=str(HERE), help="dir containing <set>/frames/")
    ap.add_argument("--out", default=str(HERE / "flip"), help="output dir for heatmaps/report")
    ap.add_argument("--sets", default="all", help="comma list of sets, or 'all'")
    ap.add_argument("--render", action="store_true", help="re-render OpenGL frames first (needs assets)")
    ap.add_argument("--check", action="store_true", help="gate vs baseline; exit 1 on regression")
    ap.add_argument("--baseline", default=str(HERE / "flip" / "baseline.json"))
    ap.add_argument("--update-baseline", action="store_true", help="write current scores as baseline")
    ap.add_argument("--semantic", action="store_true",
                    help="run the VLM semantic diff (Claude vision) on the worst-FLIP frames into REPORT.md "
                         "(needs anthropic SDK + ANTHROPIC_API_KEY; report-only, never gates)")
    ap.add_argument("--semantic-top", type=int, default=3, help="how many worst-FLIP frames to semantically diff")
    args = ap.parse_args()

    sets = SETS if args.sets == "all" else tuple(s.strip() for s in args.sets.split(","))
    frames_root = Path(args.frames_root)
    out_dir = Path(args.out)

    if args.render:
        _render(sets)

    records = run(frames_root, out_dir, sets)
    if not records:
        # No committed comparison frames (e.g. partial checkout) — skip rather than fail.
        print(f"SKIP: no OVRTX-golden/{RENDERER} frame pairs found under", frames_root)
        return 77

    baseline_path = Path(args.baseline)
    if args.update_baseline:
        baseline = {r["key"]: {"flip": r["flip"], "gmsd": r["gmsd"], "deltae": r["deltae"]} for r in records}
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f">>> wrote baseline ({len(baseline)} entries) -> {baseline_path}")

    gated = args.check
    baseline = {}
    if gated and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if gated:
        gate(records, baseline)

    semantic = run_semantic(frames_root, sets, records, args.semantic_top) if args.semantic else None

    (out_dir / "results.json").write_text(
        json.dumps(records, indent=2, default=str) + "\n", encoding="utf-8"
    )
    write_report(records, out_dir, gated, semantic)
    print(f">>> wrote {len(records)} results -> {out_dir/'results.json'} + REPORT.md")

    if gated:
        fails = [r for r in records if r["status"] == "fail"]
        if fails:
            print(f">>> REGRESSION GATE FAILED: {len(fails)} pair(s) regressed", file=sys.stderr)
            for r in fails:
                print(f"    {r['key']}: FLIP {r['flip']:.3f} — {r['reason']}", file=sys.stderr)
            return 1
        print(">>> regression gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
