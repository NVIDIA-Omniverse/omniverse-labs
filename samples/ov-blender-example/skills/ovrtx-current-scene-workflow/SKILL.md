---
name: ovrtx-current-scene-workflow
description: Build, edit, preview, and render the current Blender scene through the OVRTX Blender add-on. Use when a user wants OVRTX output from a new or existing `.blend`, wants to drive Blender through MCP and see live viewport updates, or needs a reproducible still from the current scene. The workflow uses the add-on's Blender boundary and an installed runtime
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
# OVRTX current-scene workflow

The current Blender scene is the authoring boundary. It may be saved or unsaved,
created through Blender's UI, imported from USD, or assembled by Blender MCP.
OVRTX builds a managed scene generation from that state; use the documented
current-scene workflow rather than pre-exporting a separate USD.

## When to Use

Use when a user wants OVRTX output from a new or existing `.blend`, wants to drive Blender through MCP and see live viewport updates, or needs a reproducible still from the current scene. The workflow uses the add-on's Blender boundary and an installed runtime.

## Instructions

1. Use `blender-mcp-setup` to verify the Blender control loop, then use the normal Blender MCP tools or UI to inspect the scene.
2. Save a working copy when the scene is valuable or edits are destructive. Record the `.blend` path, active camera, resolution, frame, world, lights, and renderable objects.
3. Set the scene camera explicitly. Check clipping, lens/orthographic scale, transform, resolution, and aspect ratio before debugging OVRTX.
4. For imported USD, retain the source path and import settings. For a generated scene, use stable object/material/light names so later edits can be resolved.
5. Do not delete existing objects or reset the world unless the user requested a clean rebuild.

## 2. Select OVRTX and configure a first render

After add-on preflight passes, set the scene render engine to `OVRTX_EXAMPLE` through Blender's Render Properties or Blender Python. The add-on settings are available as `scene.ovrtx_example`:

- `min_samples`: first progressive sample count (use 1 for a smoke test);
- `max_samples`: completion target (raise after the first valid frame);
- `color_presentation`: `scene_linear_hdr` when the consumer owns display conversion, or `ldr_rgba8_display_passthrough` when OVRTX returns display-encoded pixels (the UI shows these as Scene Linear HDR and LDR Display Passthrough);
- `sync_viewport_camera`: map Blender viewport orbit/pan/zoom to the OVRTX preview camera when the scene camera mapping is valid.

Use Blender's Render Image/Render Animation action or the MCP's Blender-Python execution to request the frame.

## 3. Validate the first frame

Before increasing quality, verify all of the following:

- the OVRTX render engine is still active (not a silent Cycles/Eevee fallback);
- the expected camera is bound and the subject is in frame;
- the output dimensions and orientation are correct;
- the image is not black, flat, mirrored, or dominated by a missing world/material;
- the requested samples advance and the output file exists at the user-selected path.

Use a viewport screenshot plus structured Blender scene inspection. Keep raw OVRTX output separate from any presentation-grade crop or color grade.

## 4. Interactive refinement

In a rendered 3D View, enable camera synchronization and orbit/pan/zoom normally. The add-on reuses a warm OVRTX session and progressively refines stable views. If the viewport is stale, use the add-on's **Restart Viewport Session** action, then wait for a new first sample before judging quality.

Supported value edits (including transforms, camera view, and the add-on's mapped light/material/world values) should be made through Blender and allowed to propagate in-session. Structural or topology changes can request a new scene generation; wait for the replacement before comparing images. Do not edit the generated USD overlay by hand.

## 5. Color and output ownership

Choose exactly one display-transform owner:

- With scene-linear HDR, preserve linear values and apply Blender's intended view transform once in the consumer or presentation step.
- With LDR passthrough, treat OVRTX pixels as already display encoded and do not apply a second Blender transform.

Record render mode, sample range, resolution, camera, color mode, output path, and whether the frame is raw or graded. A visually pleasing screenshot without this provenance is not a reproducible render.

## Recovery

- **Black/empty frame:** check camera binding, runtime status, world/light inputs, and session logs; retry at one sample.
- **Wrong view:** reset the viewport camera, make the intended camera active, and confirm `sync_viewport_camera` and projection.
- **Stale edits:** restart the viewport session after a structural change; do not compare a cached frame with a newly authored scene.
- **No render engine:** return to `ovrtx-addon-install-and-preflight`.
- **MCP disconnected:** return to `blender-mcp-setup`; do not diagnose an OVRTX worker as an MCP failure.

## Closeout

Report the source `.blend` (or “unsaved”), camera, engine, sample range, color presentation, output files, session status, and any unsupported or approximate features. Distinguish OVRTX-rendered artifacts from Blender fallback renders and from postprocessed images.
