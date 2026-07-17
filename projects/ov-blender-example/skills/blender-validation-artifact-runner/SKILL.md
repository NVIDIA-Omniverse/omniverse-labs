---
name: blender-validation-artifact-runner
description: Execute an already-decided Blender, OVRTX, OVPhysX, USD, or image-comparison validation command and collect concise evidence. Use for bounded execution and artifact checking after another agent or user has chosen the experiment.
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
# Validation artifact runner

This is execution support, not root-cause analysis. Do not change source, scene lookdev, cameras, or settings while running a decided validation.

## When to Use

Use for bounded execution and artifact checking after another agent or user has chosen the experiment.

## Instructions

1. Confirm command, working directory, inputs, expected outputs, timeout, and runtime owner. Use the add-on and installed runtime only.
2. Record command, start/end time, source/add-on versions, and output directory. Poll long jobs instead of blocking indefinitely.
3. Read the structured report and smallest relevant log signature. Verify expected PNG/EXR/NPY/USD/movie artifacts exist and have nonzero size; inspect dimensions and checksums where available.
4. For movies, check duration/frame rate and first/middle/last frames. For images, check nonblank structure and requested renderer/product identity. For USD, check layer/dependency closure.

Return command, exit code, elapsed time, report path, artifact paths, key metrics, and one blocker line if evidence is incomplete. Preserve failures as failures or `blocked-*`; do not silently substitute a prior render or postprocessed image.
