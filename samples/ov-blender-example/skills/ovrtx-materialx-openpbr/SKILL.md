---
name: ovrtx-materialx-openpbr
description: Prepare Blender materials for the add-on's OVRTX MaterialX/OpenPBR conversion and diagnose failures without renderer source access. Use for source textures, OpenPBR graphs, black/red/flat OVRTX materials, missing bindings, UV/normal-map issues, or Blender-versus-OVRTX material comparisons.
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
    - materialx
    - materials
---
# OVRTX MaterialX/OpenPBR

Prepare Blender materials for the add-on's OVRTX MaterialX/OpenPBR conversion
and diagnose failures through documented add-on and runtime interfaces.

## When to Use

Use for source textures, OpenPBR graphs, black/red/flat OVRTX materials, missing bindings, UV/normal-map issues, or Blender-versus-OVRTX material comparisons.

## Boundary and inputs

- Work through the installed add-on, Blender APIs, documented probes, and the
  supported runtime interfaces.
- Preserve source textures and relative paths. Do not bake a comparison image
  and call it a material conversion.
- Record Blender, add-on, runtime, material, texture, UV, and color-space
  identities before changing the graph.

## Instructions

1. Inventory only materials assigned to renderable polygons and verify evaluated
   geometry, slots, polygon assignment, UV layer, and texture paths.
2. Build or repair a reachable graph from the active material output. Prefer
   named Principled/OpenPBR-adjacent inputs and source image textures over
   disconnected diagnostic nodes.
3. Validate base color, roughness, metallic/specular, normal, transmission,
   opacity, emission, coat, sheen, and texture color-space intent separately.
4. Confirm the add-on's generated MaterialX/OpenPBR binding and geomprop/UV name
   in the resolved output. Treat USD Preview fallback as a separate lane.
5. Render a neutral control and one material-specific fixture through Blender and
   OVRTX. Compare scene-linear/HDR data when available; use display PNGs only as
   a separately labeled presentation comparison.
6. Classify each used material as `faithful`, `approximate`, `unbound`, or
   `runtime-unavailable`, with the cause and next action.

## Failure ladder

Diagnose in this order: runtime/readback, stage/material identity and binding,
UV/texture path and color space, global lighting/exposure/display transform,
then individual lobe conversion. Do not change albedo to hide a light or
tone-map problem. Do not reject a scene because an unreachable unsupported node
exists.

## Handoff

Summarize the material changed, conversion class, native result, and any
unsupported node or texture. Add hashes, mapping tables, fixture renders, or
raw/display comparisons only when the caller needs a reproducible handoff or
review artifact. Pair with `ovrtx-render-settings` for probed scene
controls and a contributor parity skill for graph-family regression tests.
