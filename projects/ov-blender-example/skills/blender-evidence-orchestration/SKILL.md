---
name: blender-evidence-orchestration
description: Plan and review reproducible Blender and OVRTX evidence runs for renders, viewport captures, simulations, and exported USD. Use when a task has multiple capture lanes, needs provenance, or must distinguish native output from Blender or postprocessed evidence.
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
    - validation
---
# Blender evidence orchestration

Use this optional coordinator when a Blender task produces more than one artifact or needs a defensible pass/fail claim. It works through the documented add-on and runtime interfaces.

## When to Use

Use when a task has multiple capture lanes, needs provenance, or must distinguish native output from Blender or postprocessed evidence.

## Instructions

1. Save or duplicate the source `.blend`; choose an output directory such as `.cache/<topic>-YYYY-MM-DD/`.
2. Write a manifest before launching: source/add-on versions, scene and fixture identity, camera, frame range, resolution, renderer, requested products, and runtime prerequisites.
3. Give each lane an owner (authoring, fixed render, viewport/UI, simulation, or USD export). Do not run competing Blender/runtime owners against the same session.
4. Use the smallest smoke input first. Promote to final resolution, samples, or animation only after dimensions, nonblank structure, camera, and engine identity pass.

## Evidence contract

Keep `manifest.json`, logs, reports, checksums, and first/middle/last frames for motion. Label every product as `native-ovrtx`, `native-ovphysx`, `blender`, `usd-export`, or `postprocessed-review`. A crop, overlay, grade, or contact sheet is review evidence, not a native render.

For interactive claims, capture the Blender UI with the intended viewport, camera, timeline, and visible result. For simulation claims, retain initial conditions, mass/collision/solver settings, poses or telemetry, contact/settling metrics, and the rendered proof. For USD claims, retain source-to-prim mapping and unresolved-reference checks.

## Closeout

Inspect artifacts rather than trusting exit code. Report exact paths, metrics, status (`pass`, `partial`, `failed`, or `blocked-*`), and the smallest blocker. Keep generated files out of source control unless deliberately requested. Never claim a runtime or server feature was tested by reading its source; test only through documented add-on capabilities and installed dependencies.
