---
name: ovphysx-drop-contact-acceptance
description: Author and accept a reproducible OVPhysX rigid-body drop or contact test from Blender, including collision proxies, authoritative pose readback, settling, and visible contact evidence. Use when a user needs to prove that a SimReady prop falls, collides, and comes to rest through the installed OVRTX/OVPhysX runtime rather than a Blender-only animation.
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
    - ovphysx
---
# OVPhysX drop and contact acceptance

Use this skill to answer a narrow question: did the authored bodies undergo a
real OVPhysX simulation and make the expected contact without tunneling,
floating, or losing identity? Blender and the add-on author the scene; OVPhysX
owns simulation state. OVRTX may render an authoritative pose replay.

## When to Use

Use when a user needs to prove that a SimReady prop falls, collides, and comes to rest through the installed OVRTX/OVPhysX runtime rather than a Blender-only animation.

## Runtime boundary

The add-on source and included scripts are available to the user. The
OVPhysX service, OVRTX worker, protocol definitions, and native client are
installed dependencies. Never ask the user to clone, build, inspect, or
patch those components. A missing service/client/GPU is `blocked`, not a reason
to substitute Blender rigid bodies or hand-authored keyframes.

## Instructions

1. Work from a copy of the `.blend` or USD and write artifacts to a unique
   caller-selected directory such as `.cache/ovphysx-drop-YYYYMMDD-HHMMSS/`.
   Hash the source before and after the run.
2. Choose one support (floor, stair, shelf, or collider) and one or more
   dynamic roots. Record stable Blender object names, exported USD prim paths,
   intended mass, initial transforms, units, up-axis, gravity, fixed `dt`,
   duration, solver/substeps, and the camera used for review.
3. Validate finite transforms and initial non-overlap. Keep each independently
   moving visual root as a separate body. Exclude cameras, lights, rigs, and
   decorative helpers unless they are explicitly simulated.
4. Build collision geometry from evaluated meshes. Use supported convex or
   compound primitives for concave or thin props and hide proxies only in the
   beauty render. Record proxy bounds and the approximation made; do not use a
   broad invisible box to conceal bad contact.

## Run and Read Back Native Poses

Use the add-on's documented OVPhysX controls or included probe after checking
its local `--help` and README. A fresh checkout may expose
`python3 scripts/run_ovphysx_drop_probe.py --require-real`; treat that command
as versioned and stop if it is not present.

Create the simulation through the supported API, apply only the authored
initial state/impulse, and step for a bounded interval. At every sample retain
the complete body pose set (body identity, timestamp, position, orientation,
and velocities when available). Reject missing, duplicate, reordered, NaN, or
infinite values. Compute a normalized state hash over the ordered pose set so
reruns can be compared without treating timestamps as identity.

Replay poses onto matching visual roots only through the documented shared-stage
or pose-replay path. Do not infer physics from a render image or Blender's own
rigid-body cache. Label each product as native OVRTX, Blender replay, or
diagnostic/viewport output.

## Acceptance gates

Pass only when all applicable gates hold:

- preflight and native simulation creation succeeded;
- expected body count and stable identity match the authored manifest;
- every sampled pose is finite and has a traceable timestamp/state hash;
- the dynamic body moves from its authored start and records a support contact;
- visible mesh-to-support distance is within the chosen scene tolerance after
  contact (a gap larger than about 1 cm at metre scale is normally a failure);
- no unexpected tunneling, persistent floating, or support penetration appears;
- a quiet/settled window satisfies the declared velocity and hysteresis limits;
- replay mapping is complete and source hashes are unchanged.

For a non-contact scenario, explicitly mark contact/settling gates as
`not_applicable`; never silently omit them. A short-lived contact followed by
continued unbounded motion is a failure, not a settled result.

## Evidence package

Write `ovphysx-drop-report.json` beside the captures. Include at least:

```json
{
  "status": "pass|blocked|fail",
  "source": {"path": "...", "sha256": "..."},
  "runtime": {"addon": "...", "ovrtx": "...", "ovphysx": "...", "gpu": "..."},
  "settings": {"dt": 0.0166667, "duration_s": 4.0, "gravity": [0, 0, -9.81]},
  "bodies": {"expected": 0, "discovered": 0, "identity": "pass|fail"},
  "poses": {"samples": 0, "finite": "pass|fail", "state_hash": "..."},
  "contact": {"events": 0, "settled_fraction": 0.0, "visible_gap_m": null},
  "replay": {"mapping": "pass|fail", "render_class": "ovrtx|blender|diagnostic"},
  "artifacts": [], "blockers": [], "limitations": []
}
```

Retain pose samples, collision/proxy audit, initial/contact/settled previews,
camera metadata, output hashes, and the exact command or UI actions. If native
readback or the runtime is unavailable, return `blocked` with the first exact
status/log message and do not generate a passing movie from fake poses.
