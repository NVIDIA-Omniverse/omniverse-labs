---
name: blender-workflow-routing
description: Plan and route a Blender task across MCP, modeling, materials, cameras, lighting, animation, rendering, USD export, and the OVRTX add-on. Use for multi-step scene work, “make a complete scene” requests, or when a user is unsure which Blender workflow to follow. The OVRTX/OVPhysX service and native client are installed dependencies.
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
    - workflow
---
# Blender workflow routing

Use this as the entry point for scene work. It provides a portable plan that
works with any supported Blender MCP provider and the installed OVRTX add-on.

## When to Use

Use for multi-step scene work, “make a complete scene” requests, or when a user is unsure which Blender workflow to follow. The OVRTX/OVPhysX service and native client are installed dependencies.

## Instructions

1. Clarify the deliverable: `.blend`, still, animation/movie, USD package,
   SimReady asset, OVPhysX replay, or sensor/AOV data.
   Route externally supplied or otherwise untrusted scene content through
   `blender-content-safety-and-privacy` before opening or importing it.
2. Run `blender-mcp-setup` for an agent-controlled session and
   `ovrtx-addon-install-and-preflight` when OVRTX output is requested. A live
   MCP connection does not prove OVRTX readiness.
3. Preserve existing user scenes. Save a copy before destructive operations,
   choose a caller-owned output directory, and use stable object/material names.
4. For a normal scene, chain: blockout → camera/composition → lighting →
   geometry → materials/UVs → final render → export. Lock camera and values
   early; do not polish detail before a cheap preview reads correctly.
5. Load only the focused skill for each phase:

| Intent | Route |
|---|---|
| Procedural mesh | `geometry-nodes-for-ovrtx` |
| Image/template reconstruction | `reference-to-3d-reconstruction` |
| Textures, UVs, atlas, baking | `texture-uv-material-workflow` |
| Keyframes or motion | `animation-quality-and-frame-range` |
| OVRTX still/live scene | `ovrtx-current-scene-workflow` |
| OVRTX settings/materials | `ovrtx-render-settings`, `ovrtx-materialx-openpbr` |
| USD handoff | `usd-copy-and-flatten` |
| SimReady/physics | `simready-addon-install-and-authoring`, `ovphysx-simulation-workflow` |

## Validation and handoff

- Inspect Blender scene state and take a viewport/render screenshot after each
  meaningful mutation; numerical success is not visual proof.
- Keep Blender/Cycles/EEVEE previews distinct from native OVRTX products.
- Record source path or “unsaved,” camera, frame, engine, output paths,
  add-on/runtime status, and unsupported features.
- Before sharing logs, reports, screenshots, or manifests, sanitize them with
  `blender-sanitized-support-bundle`; keep unsanitized artifacts local.
- Use documented add-on diagnostics. If a runtime feature
  is unavailable, report the exact preflight failure and stop that branch.
