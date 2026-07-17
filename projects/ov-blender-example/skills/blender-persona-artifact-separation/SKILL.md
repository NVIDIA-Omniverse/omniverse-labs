---
name: blender-persona-artifact-separation
description: Keep Blender creative-authoring and simulation-readiness workflows separate in scenes, reports, and artifacts. Use when a task mixes lookdev, camera, materials, physics, semantics, collision, or downstream validation.
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
# Persona and artifact separation

Choose a persona before creating evidence:

- `creative_artist`: composition, cameras, lighting, materials, renderer fidelity, and live visual edits.
- `simready`: semantics, collision, mass, joints, physics, sensors, and downstream runtime behavior.

## When to Use

Use when a task mixes lookdev, camera, materials, physics, semantics, collision, or downstream validation.

## Instructions

1. Use persona-scoped output roots (for example `.../creative_artist/` and `.../simready/`) and put `persona` in every manifest/report.
2. Keep beauty/render settings and simulation metadata in separate report sections. Do not use visual similarity as physics proof or physics success as lookdev proof.
3. When one task has both lanes, author shared scene inputs once, then produce explicit derivatives/overlays for each persona; record the source hash and derivative relationship.
4. State gaps by persona and label native OVRTX/OVPhysX, Blender, USD, and postprocessed evidence independently.

If a feature needs a runtime capability unavailable through the add-on, report that dependency and use its documented diagnostics.
