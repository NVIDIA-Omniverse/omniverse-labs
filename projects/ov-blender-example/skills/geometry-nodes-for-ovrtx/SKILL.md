---
name: geometry-nodes-for-ovrtx
description: Author, inspect, and prepare Blender Geometry Nodes assets for predictable OVRTX rendering and USD handoff. Use for procedural geometry, instances, fields, modifiers, or geometry that must survive add-on evaluation through documented add-on and runtime interfaces.
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
    - geometry-nodes
---
# Geometry Nodes for OVRTX

Geometry Nodes is a procedural authoring layer; OVRTX consumes the evaluated
Blender result through the add-on boundary. Preserve the node graph for
editable `.blend` work, but prove that the evaluated result is renderable and
exportable before claiming compatibility.

## When to Use

Use for procedural geometry, instances, fields, modifiers, or geometry that must survive add-on evaluation through documented add-on and runtime interfaces.

## Instructions

1. Save a copy and record Blender/add-on/runtime versions, frame, seed, object
   identity, modifier stack, and node-group inputs. Use deterministic seeds and
   avoid machine-local asset paths.
2. Build a small representative graph first. Give the modifier and node group
   stable names; expose scale, density, material, and seed controls as typed
   inputs. Keep simulation-time dependencies and frame range explicit.
3. Inspect the evaluated dependency graph at the intended frame. Verify finite
   transforms, expected bounds/vertex counts, material assignment, UV or named
   attributes, normals, instance realization policy, and visibility.
4. Test a low-sample Blender preview and then the same scene through
   `ovrtx-current-scene-workflow`. Never call a viewport screenshot proof of
   native OVRTX geometry without checking the active engine and session status.
5. For USD or SimReady handoff, use the add-on's documented export path. Apply
   modifiers or realize instances only when the target contract requires it;
   retain the procedural source `.blend` and record the bake/evaluation choice.
6. Change topology or node inputs in a controlled copy, restart scene generation
   when required, and compare object/prim identity and bounds before and after.

## Compatibility checks

- Avoid relying on unsupported custom attributes, anonymous attributes,
  unbounded recursion, or Blender-only Python callbacks in a final handoff.
- Check that generated materials are reachable from the active output and that
  texture/UV attributes have the names the add-on documents.
- Treat a missing mesh, stale instance, wrong frame, or empty render as a
  conversion/evaluation failure, not a reason to substitute a manually modeled
  mesh without disclosure.
- Keep collision proxies and visual geometry separate for OVPhysX; validate
  evaluated collider coverage with `simready-addon-install-and-authoring`.

## Closeout

Report source `.blend`, node-group/modifier names, frame and seed, evaluated
geometry counts/bounds, material/attribute notes, engine, output paths, and
which outputs are procedural, baked, native OVRTX, Blender preview, or
postprocessed. Escalate unsupported nodes through the add-on's documented issue or
diagnostic path; Use the documented setup and runtime diagnostics.
