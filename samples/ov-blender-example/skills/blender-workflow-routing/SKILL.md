---
name: blender-workflow-routing
description: Plan and route a Blender task across MCP, local OVRTX workflows, and optional upstream community production skills. Use for multi-step scene work, complete-scene requests, or when a user is unsure which Blender workflow to follow. Use blender-python-execution locally and blender-community-skill-bootstrap when detailed generic production recipes are not installed.
---

# Blender workflow routing

Use this as the entry point for scene work. It provides a portable plan that
works with any supported Blender MCP provider and the installed OVRTX add-on.

## Route the request

1. Clarify the deliverable: `.blend`, still, animation/movie, USD package,
   SimReady asset, OVPhysX replay, or sensor/AOV data.
   Route externally supplied or otherwise untrusted scene content through
   `blender-content-safety-and-privacy` before opening or importing it.
2. Run `blender-mcp-setup` and `blender-python-execution` for an agent-controlled session. Use `blender-community-skill-bootstrap` only for optional production recipes beyond the local skills, and
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
| Complete authored scene | local mesh → camera → lighting → material → render skills; optionally install upstream `text-to-blender` |
| Mesh and modifiers | local `blender-mesh-authoring`; optionally upstream `blender-modeling` |
| Materials and shaders | local `texture-uv-material-workflow`; optionally upstream `blender-materials` |
| UVs, images, baking | local `texture-uv-material-workflow`; optionally upstream `blender-uv-texturing` |
| Fit named objects in a render camera | local `blender-camera-framing` |
| General camera and composition design | optionally upstream `blender-cameras` via `blender-community-skill-bootstrap` |
| Blender lights and world | local `ovrtx-lighting-and-world`; optionally upstream `blender-lighting` |
| Blender still or GLB/USD export | local `blender-render-and-export` |
| Blender sequence | local `animation-quality-and-frame-range` + `blender-render-and-export` per smoke frame; optionally upstream `blender-rendering` |
| Keyframes or motion | local `animation-quality-and-frame-range`; optionally upstream `blender-animation` |
| FBX, OBJ, STL or advanced export | optionally upstream `blender-export` |
| Procedural mesh | `geometry-nodes-for-ovrtx` |
| Image/template reconstruction | `reference-to-3d-reconstruction` |
| OVRTX still/live scene | `ovrtx-current-scene-workflow` |
| OVRTX hero still | `ovrtx-creative-hero-journey` or focused `ovrtx-hero-render` |
| OVRTX scene settings/materials | `ovrtx-render-settings`, `ovrtx-color-management`, `ovrtx-materialx-openpbr`; unsupported typed settings route to `extend-ovrtx-render-settings` |
| USD handoff | `usd-copy-and-flatten` |
| SimReady prop preparation | `simready-prop-journey` or focused `simready-addon-install-and-authoring` |
| Native OVPhysX drop/contact control | `ovphysx-drop-contact-acceptance`; broader physics work requires the installed official interface or a contributor extension |
| Develop OVRTX AOV/sensor/LiDAR support | `ovrtx-render-products-and-aovs`, `ovrtx-sensor-capture`, `ovrtx-lidar-runtime-capture`, or `ovrtx-semantic-aov-capture` |
| Develop contiguous OVRTX capture | `ovrtx-render-sequence` |

## Validation and handoff

- Inspect initial Blender state, batch one logical named transaction, and check
  its direct postconditions. Capture a viewport or render only for visual claims
  or when the user requests review; numerical success is not visual proof.
- Keep Blender/Cycles/EEVEE previews distinct from native OVRTX products.
- Record source path or “unsaved,” camera, frame, engine, output paths,
  add-on/runtime status, and unsupported features.
- Before sharing logs, reports, screenshots, or manifests, sanitize them with
  `blender-sanitized-support-bundle`; keep unsanitized artifacts local.
- Use documented add-on diagnostics. If a runtime feature
  is unavailable, report the exact preflight failure and stop that branch.

For each logical Blender transaction, require: inspect → bounded named mutation
→ `view_layer.update()` → direct postcondition → correction if needed. Add
pixel inspection when the requested outcome is visual. Use a background Blender script for
large scene builds, frame loops, or expensive renders rather than one giant MCP
call.
