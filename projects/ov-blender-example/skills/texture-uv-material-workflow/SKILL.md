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

Read `references/bpy-recipes.md` before authoring or repairing UVs and image
materials. It contains Blender 5.1 patterns for context-free planar UVs,
context-prepared unwraps, reachable Principled materials, normal maps, alpha,
polygon material assignment, and bake targets. Run each pattern as its own
`blender-python-execution` transaction and verify its JSON postcondition.

Use `blender-community-skill-bootstrap` to install upstream
`blender-uv-texturing` and `blender-materials` when detailed optional recipes
are needed. Execute them through `blender-python-execution` so active-object,
mode, color-space, and output-path state are explicit and checked.

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

Run the read-only `scripts/audit_material_uv.py` through Blender Python after
authoring and before render/export. Require `status: pass` for the visible meshes
in scope. A pass requires at least one renderable mesh, valid reachable material
outputs and image files, role-appropriate image color spaces, assigned polygon
materials, required UV layers, finite UV coordinates, no non-UDIM coordinates
outside the unit tile, and no nonzero-area face collapsed to zero UV area.
Treat orphan image nodes and unused material slots as cleanup warnings; inspect
them even though they do not fail the audit. Declare UDIM use with a reachable
tiled image or the object boolean property `uv_allow_udim`.

For MCP, prepend `MATERIAL_UV_AUDIT_REQUEST = {}` and append the complete audit
script in the same execution call; require the printed JSON to pass. For a
saved caller-owned scene, use `blender --background scene.blend --python
scripts/audit_material_uv.py`; a failing audit exits with status 2.

## Handoff

Summarize the objects/materials changed, whether the audit passed, and any
remaining color-space, UV, or dependency issue. Include hashes, the complete
JSON audit, checker/bake previews, or a dependency inventory only when the
caller needs a reproducible handoff or review artifact. Keep source references
and image planes out of final USD/GLB unless requested.
