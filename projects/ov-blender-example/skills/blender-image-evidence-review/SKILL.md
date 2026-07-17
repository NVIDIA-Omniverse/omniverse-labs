---
name: blender-image-evidence-review
description: Review Blender, Cycles, and OVRTX render evidence with deterministic image checks, contact sheets, crops, metrics, and an honest visual verdict. Use for smoke tests, hero renders, AOV review, renderer comparisons, or contributor acceptance artifacts.
license: "Apache-2.0"
metadata:
  author: "Max Bickley"
  version: "0.1"
  team: "omniverse"
  domain: "physical-ai"
  tags:
    - blender
    - omniverse
    - ovrtx
    - validation
---
# Blender image evidence review

Review artifacts, not API claims. Keep native source images immutable and write
review derivatives to a caller-selected output directory. OVRTX services and
clients use documented interfaces; this skill only consumes their documented output and reports.

## When to Use

Use for smoke tests, hero renders, AOV review, renderer comparisons, or contributor acceptance artifacts.

## Instructions

1. Load the image/result sidecars and verify existence, nonzero size, dimensions,
   channels, orientation, frame/product identity, and checksums.
2. Verify the image is from the claimed renderer and color mode. A screenshot,
   cached blit, Cycles render, or postprocessed overlay must not be labeled
   native OVRTX.
3. Compute near-black/near-white fractions, finite-value checks, mean/percentile
   luma, alpha statistics, and (when available) HDR range. Preserve the metric
   thresholds and tool/version in a JSON report.

## Review products

- Build a label-free contact sheet for sequence or A/B inspection.
- Use fixed, documented crops for material, camera/framing, lighting, and
  contact regions; never move a crop until after measuring its projection.
- Generate false-color diffs only for aligned, label-free images and report the
  alignment method. For different cameras, call the result look review rather
  than pixel parity.
- Keep semantic/ID/depth overlays separate from beauty images and retain the
  original native arrays beside any colored visualization.

## Verdicts

Lead with one of: `matches enough for smoke`, `ready for review`, `visual
mismatch`, `material gap suspected`, `light/tone-map gap suspected`, `frame or
product mismatch`, or `artifact invalid`. Explain the evidence and next action.
Do not let a single scalar score conceal black frames, stale outputs, missing
objects, double color transforms, or inconsistent camera identity.

Return the report, contact sheet, crops/diffs, source references, and explicit
native/converted/postprocessed classification.
