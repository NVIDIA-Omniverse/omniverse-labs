#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VLM semantic visual-diff: explain, in words, how a renderer output deviates from the OVRTX golden.

A qualitative complement to FLIP. FLIP gives a scalar + heatmap; this says *what* differs, *where*,
and — the move no per-pixel metric can make — separates EXPECTED cross-renderer tone/exposure/GI
differences from LIKELY BUGS (crushed geometry, wrong materials, missing objects). It is a
human-facing / triage layer, NOT a deterministic CI gate: a VLM is non-deterministic and can
hallucinate, so keep FLIP+legs (see ``flip_pipeline.py``) as the gate and use this to explain a
failure or surface a defect the scalar missed.

Wiring: ``flip_pipeline.py --semantic`` runs this on the worst-FLIP frames and appends the results
to REPORT.md. Standalone:

  ANTHROPIC_API_KEY=... python semantic_diff.py golden.png test.png [--json]

Requires the Anthropic SDK (``pip install anthropic``) and an API key; both are optional — the
pipeline degrades gracefully (prints a note, skips the section) when either is absent.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

MODEL = "claude-opus-4-8"

# Closed renderer-difference taxonomy. Each observation is bucketed as an EXPECTED cross-renderer
# difference (single-pass vs path-traced: tone, white balance, missing GI/occlusion) or a LIKELY BUG
# (a real defect). That split is the value FLIP can't provide.
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "is_bug", "observations"],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["match", "faithful-lower-fidelity", "broken"],
            "description": "match = essentially identical; faithful-lower-fidelity = only expected "
                           "cross-renderer tone/GI differences; broken = a real structural/material defect.",
        },
        "is_bug": {"type": "boolean", "description": "true if any observation is a likely bug, not just tone."},
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["category", "region", "detail", "severity", "classification"],
                "properties": {
                    "category": {"type": "string", "enum": [
                        "exposure-tone", "color-cast", "crushed-shadows", "clipped-highlights",
                        "missing-or-added-geometry", "crushed-geometry", "texture-material-mismatch",
                        "missing-GI-occlusion", "noise-denoiser-artifact"]},
                    "region": {"type": "string", "description": "e.g. 'floor', 'racks upper-center', 'global'."},
                    "detail": {"type": "string", "description": "wrong-vs-expected, concrete ('grey not blue')."},
                    # NB: structured-output schemas don't support numerical constraints (minimum/maximum),
                    # so the 1-5 range lives in the description, not as min/max (which would 400).
                    "severity": {"type": "integer", "description": "1 (barely noticeable) to 5 (severe defect)."},
                    "classification": {"type": "string", "enum": ["expected-cross-renderer", "likely-bug"]},
                },
            },
        },
    },
}

PROMPT = (
    "Image 1 is the GOLDEN path-traced reference. Image 2 is a single-pass renderer's TEST output of the "
    "SAME scene and camera. Describe how the TEST deviates FROM the golden — NOT which looks better.\n"
    "Single-pass renderers legitimately differ from a path tracer in overall exposure/white-balance and in "
    "lacking global illumination / contact occlusion; classify those as 'expected-cross-renderer'. Classify "
    "as 'likely-bug' only real defects: crushed-to-black or blown-out salient geometry, lost product/detail, "
    "wrong material/albedo, missing/added objects. Name the region and the wrong-vs-expected specifics. "
    "If there is no meaningful difference, return verdict 'match' with an empty observations list."
)


def _img_block(path: str) -> dict:
    data = base64.standard_b64encode(Path(path).read_bytes()).decode("utf-8")
    media = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}


def semantic_diff(golden: str, test: str) -> dict:
    """Return a structured semantic diff of `test` against the golden reference. Raises on SDK/API error."""
    from anthropic import Anthropic  # lazy: optional dependency, only needed when this runs

    client = Anthropic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": [
            _img_block(golden), _img_block(test), {"type": "text", "text": PROMPT}]}],
    )
    # output_config.format guarantees a text block containing schema-valid JSON (after any thinking blocks).
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


def format_diff(r: dict) -> str:
    """One-line verdict + bulleted observations, bug-flagged. Used by the CLI and the pipeline report."""
    out = [f"verdict: {r['verdict']}  (bug={r['is_bug']})"]
    for o in r["observations"]:
        flag = "BUG" if o["classification"] == "likely-bug" else " · "
        out.append(f"  [{flag}] {o['category']} @ {o['region']} (sev {o['severity']}): {o['detail']}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="VLM semantic visual-diff vs the OVRTX golden")
    ap.add_argument("golden")
    ap.add_argument("test")
    ap.add_argument("--json", action="store_true", help="emit raw JSON instead of the formatted summary")
    a = ap.parse_args()
    r = semantic_diff(a.golden, a.test)
    print(json.dumps(r, indent=2) if a.json else format_diff(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
