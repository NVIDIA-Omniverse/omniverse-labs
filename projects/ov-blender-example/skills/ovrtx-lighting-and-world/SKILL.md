---
name: ovrtx-lighting-and-world
description: Author and debug Blender lights and world/HDRI lighting through the OVRTX add-on, including live value edits, dome conversion, light-unit checks, and render evidence. Use when a scene is dark, overexposed, flat, or differs from a Blender reference.
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
    - lighting
---
# OVRTX lighting and world

Use Blender as the authoring surface and the add-on's typed conversion/update
path. Use the documented runtime interfaces, and do not hand-edit generated
transport USD to fix lighting.

## When to Use

Use when a scene is dark, overexposed, flat, or differs from a Blender reference.

## Author a controlled lighting setup

1. Name the active camera, key/fill/rim lights, emissive meshes, and world.
   Record type, color/temperature, energy, size/shape, transform, shadow intent,
   and units. Keep a neutral gray control object in the scene.
2. Start with one supported light family at a time (point/spot/sun/area) and a
   known world. Render a low-sample baseline, then add lights incrementally.
3. Use the add-on's documented value conversion. The current example maps
   Blender light energy to USD light intensity and records the conversion policy
   and emitter area. Temperature is baked into color where supported; unsupported
   shadow/falloff/node fields must be reported, not approximated silently.
4. For a simple World, use flat color or one unlinked Background color/strength.
   The world conversion maps effective scene-linear RGB to a DomeLight
   color/intensity. Linked environment textures, ambiguous node graphs, and
   world topology changes require a scene refresh or are unsupported; classify
   them explicitly.

## Instructions

Value edits (energy, color, transform, supported world background values) should
be made in Blender and propagated through the warm add-on session. Type and
topology changes require the documented scene-generation refresh. After every
edit, render one completed sample and compare mean RGB/luma plus a screenshot.

Diagnose in this order: active OVRTX engine and readback, camera/product,
transform/scale, world/dome presence, light type/units, exposure/tone map, then
materials. A Cycles image or viewport screenshot is a useful control, never
proof that OVRTX applied a light edit.

## Closeout

Return the authored light/world table, conversion policy/version, supported and
rejected fields, baseline/after images, numeric deltas, source hash, effective
camera/render settings, and runtime logs. Keep HDR/LDR ownership explicit via
`ovrtx-color-management`.
