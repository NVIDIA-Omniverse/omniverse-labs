---
name: blender-ui-evidence-capture
description: Capture readable, truthful Blender UI and viewport evidence for add-on workflows, OVRTX interactive edits, and contributor reviews. Use for before/after screenshots, short UI movies, or deterministic frame-stepped captures without depending on a specific remote-desktop deployment.
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
# Blender UI evidence capture

Capture the Blender application window or a named viewport, not a browser stream
or an unverified cached image. This is UI evidence: it does not replace native
OVRTX render-product or sensor evidence.

## When to Use

Use for before/after screenshots, short UI movies, or deterministic frame-stepped captures without depending on a specific remote-desktop deployment.

## Instructions

1. Choose an isolated output directory and record Blender/add-on revision,
   `.blend` hash, display geometry, frame range, source/output FPS, camera, and
   requested OVRTX sample/refinement boundary.
2. Verify the visible editor is the intended `VIEW_3D`/camera view, the add-on
   panel is readable, and Blender is GPU-backed when the claim is interactive
   performance. Do not assume an encoder's GPU proves Blender rendering.
3. Set UI scaling and window geometry through the user's Blender profile only
   when requested; record the effective values. Avoid mutating an artist's
   session for a capture without saving a copy.

## Capture modes

- Use real-time server/display capture when interaction cadence is the evidence.
- Use deterministic frame stepping for exact frame-count deliverables. For each
  sample, evaluate the scene/subframe, wait for the documented viewport or OVRTX
  readiness signal, capture one lossless frame, validate it, then acknowledge
  before advancing. Sleep duration alone is not readiness.
- For OVRTX, require the add-on's frame/session identity and completed sample;
  a static screenshot is not proof of a live update.

## Validate and hand off

Check dimensions, decodeability, unique-frame ratio, timestamps, nonblank motion
crop, and absence of stale/duplicate/out-of-order frames. Preserve the raw
master and make a normalized review movie only as a derivative. Return lead
frames, the movie/PNG sequence, capture logs, and a manifest mapping output
indices to Blender frames/subframes and OVRTX session/sample identity.

Never include credentials, restricted runtime paths, or implementation details in the
evidence bundle. Pair with `blender-image-evidence-review` for visual verdicts
and `blender-addon-extension-development` for feature acceptance.
Before sharing, inspect screenshots for unrelated applications, notifications,
user identity, proprietary names, and sensitive scene metadata; crop or redact
them and sanitize logs, paths, and manifests with
`blender-sanitized-support-bundle`. Preserve the raw master locally only.
