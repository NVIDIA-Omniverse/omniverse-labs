---
name: isaac-robotics-validation
description: Validate any robot task in an Isaac Sim scene derived from Blender without mutating the source asset. Use for Blender-to-Isaac robotics handoffs involving robot placement, support surfaces, manipulated objects, navigation, inspection, pick-and-place, controllers, runtime semantics, robot-mounted sensors, or direct RTX evidence.
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
    - robotics
    - validation
---
# Isaac robotics validation

Start from a user-supplied or Blender-exported USD derivative. Keep robots,
controllers, sensors, collision helpers, task proxies, and temporary semantic
labels in a session layer or separate derivative. Never mutate a protected source.

## When to Use

Use for Blender-to-Isaac robotics handoffs involving robot placement, support surfaces, manipulated objects, navigation, inspection, pick-and-place, controllers, runtime semantics, robot-mounted sensors, or direct RTX evidence.

## Instructions

1. Inspect dependency closure, units, up-axis, roots, cameras, support geometry,
   task objects, targets, and clearances. Measure transforms from the supplied
   stage; never reuse coordinates from another scene.
2. Record source, add-on, export, and Isaac versions; source and derivative
   hashes; prim paths; world poses; robot/controller configuration; timestep;
   tolerances; and output directory.
3. Choose a robot and controller available in the installed Isaac version. The
   integration must reset deterministically, advance one control step, report
   phase and terminal state, expose the required base/end-effector/sensor pose,
   and stop on non-finite transforms, controller errors, or timeout.
4. Prove placement before motion: validate robot root, task-object and target
   poses, support clearance, collision assumptions, and observer-camera framing.
5. Run the smallest deterministic action or route. Capture direct Replicator RGB
   or required sensor products plus first, action, and final frames and controller
   telemetry. A fixed observer camera is required for review; robot-mounted
   sensors are additional evidence.
6. Apply objective gates appropriate to the claim: finite transforms, expected
   phase transitions, target tolerance, collision/contact evidence or a disclosed
   kinematic attachment, contiguous frames, no forbidden source edits, and a
   clean terminal state.
7. Encode only validated contiguous frames. Record frame count, frame rate,
   duration, renderer identity, controller markers, checksums, and simulation
   shortcuts.

## Claim boundaries

- A kinematic proxy following an end effector proves controller and camera
  choreography, not a physical grasp.
- Runtime semantic labels prove an Isaac overlay, not Blender semantic export.
- A successful movie does not prove collision, dynamics, or controller behavior
  without matching numeric evidence.
- Keep appearance parity, physical-task validation, and UI interaction evidence
  as separate acceptance gates.

## Handoff

Return the derivative USD, manifest, frames or movie, logs, telemetry, checksums,
and status. Label Blender, native Isaac, OVRTX, and postprocessed outputs
separately. Missing Isaac/runtime assets are blockers, not evidence that the
Blender scene is invalid.
