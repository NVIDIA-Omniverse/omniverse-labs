---
name: ovrtx-render-parity-validation
description: Diagnose and validate Blender-versus-OVRTX render differences with controlled fixtures, native readback, stage and camera checks, global response tests, and material-family comparisons. Use when extending the add-on or investigating a visual mismatch
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
    - rendering
    - validation
---
# OVRTX render parity validation

Parity is a layered investigation, not a single image score. Preserve the
source scene and use the add-on/example probes with the installed
runtime. Do not modify undocumented runtime components or call a fallback renderer an
OVRTX pass.

## When to Use

Use when extending the add-on or investigating a visual mismatch.

## Instructions

1. Copy the source scene and record Blender, add-on, runtime, GPU, camera,
   resolution, frame, samples, color mode, and source hash. Keep an explicit
   native OVRTX output directory and a separate Blender/Cycles control.
2. Use the same authored camera and scene identity where possible. If a USD
   fixture is involved, record the default prim, camera/product paths, units,
   axis, material bindings, and dependency hashes.
3. Render a neutral gray/chrome control and a minimal light/world control before
   testing complex material graphs. Save raw LdrColor/HdrColor and sidecars.

## Triage in layers

Classify the first failing layer and stop changing later layers until it is
resolved:

1. **Runtime/readback** — session readiness, completed samples, finite pixels,
   dimensions, orientation, and non-stale frame identity.
2. **Stage/camera** — prim identity, transforms, clipping, product camera,
   visibility, units, and composition/dependency closure.
3. **Global response** — world/dome, light type/units, exposure, tone map,
   display-transform owner, and alpha/compositing.
4. **Material conversion** — binding, UV/texture paths and color spaces, then
   supported OpenPBR/UsdPreviewSurface lobe behavior.
5. **Fine fidelity** — sampling, denoising, DOF, shadows, and renderer-specific
   effects after all preceding evidence passes.

Use the smallest probe that isolates the layer. Do not compensate for a camera,
world, or color-management defect by editing albedo or roughness.

## Report and gates

Require numeric image checks (finite values, dimensions, luma/highlight ranges),
fixed aligned crops or an explicit look-review classification, a contact sheet,
and a JSON evidence report. Include the exact effective settings and unsupported
fields. A successful add-on write or worker RPC is not visual proof. Mark the
result `runtime-unavailable`, `stage-mismatch`, `global-response-gap`,
`material-gap`, `matches-for-smoke`, or `parity-not-established` with next
action. Pair with `blender-image-evidence-review` and
`ovrtx-color-management` for display-safe comparisons.
