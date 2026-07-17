---
name: texture-uv-material-workflow
description: Create and validate Blender UVs, atlases, PBR image materials, decals, and bake targets for Blender and OVRTX. Use when textures stretch, colors are wrong, alpha is black, baking fails, or a material must survive USD/MaterialX handoff. The OVRTX runtime/client is installed and not a required source checkout.
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
    - materials
    - workflow
---
# Texture and UV material workflow

Keep geometry, UV layout, image interpretation, and renderer conversion as
separate checks. A successful node edit or Blender preview does not prove that
the OVRTX MaterialX/OpenPBR result is faithful.

## When to Use

Use when textures stretch, colors are wrong, alpha is black, baking fails, or a material must survive USD/MaterialX handoff. The OVRTX runtime/client is installed and not a required source checkout.

## Instructions

1. Copy the source `.blend` and textures into a caller-owned working/output
   directory. Record image hashes, dimensions, intended color space, and mesh
   identity. Use stable UV-layer and material names.
2. Choose unwrap, projection, or atlas layout by surface type. For a reference
   plane or logo, register the front-facing UVs to measured image bounds; for a
   closed asset, inspect front, back, and side coverage separately.
3. Use `sRGB` for base color/emissive color images and `Non-Color` for roughness,
   metallic, normal, bump, AO, and masks. Connect normal images through a
   Normal Map/Bump node, not directly to color.
4. For atlases, map each mesh region into its own declared rectangle. Do not
   apply a whole atlas to every part. For alpha, connect image alpha and enable
   the documented blend/clip mode; reject black backing planes.
5. For baking, use the supported engine and an active image target with a valid
   UV map. Preserve cage, margin, resolution, and color-space settings in the
   report; save the bake as a derived artifact.

## Validation loop

- Apply a checker or numbered UV test and inspect a camera render for stretching,
  flips, seams, overlap, and atlas registration.
- Confirm each polygon has the intended material slot and active UV layer; check
  evaluated geometry if modifiers change topology.
- Render a neutral Blender control, then use `ovrtx-materialx-openpbr` with the
  same camera/lighting to classify each used material as faithful, approximate,
  unbound, or runtime-unavailable.
- Diagnose in order: runtime/readback → material identity/binding → texture
  path/UV/color space → light/exposure/display transform → individual lobe.
  Do not alter albedo to conceal a tone-map or lighting problem.

## Handoff

Return source and output hashes, UV-layer/material mapping, texture color-space
table, checker/bake previews, exported dependency closure, and native versus
Blender/postprocessed classifications. Keep source references and image planes
out of final USD/GLB unless requested. Use documented add-on diagnostics for
material-conversion failures.
