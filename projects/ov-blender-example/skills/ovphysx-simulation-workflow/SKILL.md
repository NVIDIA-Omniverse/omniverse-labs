---
name: ovphysx-simulation-workflow
description: "Prepare a Blender or USD scene for native OVPhysX, run a reproducible rigid-body simulation through the installed OVRTX/OVPhysX add-on, read authoritative poses, and replay them into a render or validation capture. Use for drops, contacts, settling, prop motion, and physics-ready SimReady assets. The service and native clients are installed dependencies: users need the add-on source and documented runtime installation."
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
    - simulation
    - workflow
---
# OVPhysX Simulation Workflow

## When to Use

Use for drops, contacts, settling, prop motion, and physics-ready SimReady assets. The service and native clients are installed dependencies: users need the add-on source and documented runtime installation.

## Contract

OVPhysX owns simulation state. Blender authors scene intent and visuals; OVRTX
may render a replay of poses published by OVPhysX. Do not replace the physics
step with Blender rigid-body simulation, hand-authored keyframes, or ballistic
math. Do not label an EEVEE/Cycles render as OVRTX output.

Use a copied scene and a unique `.cache/ovphysx-YYYYMMDD-HHMMSS/` output
directory. Keep the source `.blend`/USD immutable and hash it before and after
the run. The OVRTX/OVPhysX server, runtime bundle, and native client are
installed dependencies. Use the add-on's documented controls and included probe
scripts when present.

## Instructions

1. Confirm Blender version, add-on version, runtime bundle/client version,
   selected GPU, and the supported service endpoint/profile. A failed import,
   unavailable service, or missing native capability is a `blocked` preflight,
   not permission to fake poses.
2. Choose the scene/fixture and identify the camera, frame rate, frame domain,
   floor/supports, dynamic roots, visual-only objects, and exclusions. Record
   object/prim identities and source hashes.
3. Establish units, up-axis, gravity, fixed time step, solver/substep settings,
   and an explicit initial frame. Preserve authored positions; do not lift a
   prop merely to make a drop look better.
4. Check initial overlaps and finite transforms. Separate independently moving
   visual islands into independent bodies. Keep characters, rigs, lights,
   cameras, and render helpers out of the body set unless explicitly requested.

## 2. Make collision geometry agree with visuals

- Use evaluated Blender geometry and evaluated transforms, not only object
  AABBs. Record source bounds, component count, and proxy bounds.
- Use one rigid body per visually independent root. Use a compound of local
  boxes/capsules or supported convex parts for multipart rigid objects.
- For thin sheets, coils, hollow meshes, and concave props, subdivide or use
  supported compound pieces; never span a void with one box if it changes
  contact. Preserve floor-facing bottoms and functional openings.
- Keep proxy objects hidden from beauty renders but visible to the validator.
  Fit floors, shelves, and supports to actual surfaces; avoid invisible broad
  containment geometry.
- Record proxy coverage/volume assumptions and minimum thickness. Omit tiny
  decorative islands only with a stated threshold and accounting.

## 3. Create and run the native simulation

Use the add-on's documented OVPhysX workflow to create the simulation, configure
the scene, and apply initial state. If supported, apply impulses/velocities at
the authored pose through the add-on's write-state action. Prefer bounded,
mass-aware impulses and modest angular velocity so the test measures contact,
not numerical explosion.

Step with the configured fixed time step for a bounded duration (a few seconds
is typical for a drop). At every sample, read the complete body pose set using
the add-on's supported readback path. Retain body identity, timestamp, position,
orientation, linear/angular velocity when available, and a normalized state
hash. Never infer a pose from a render image or a partial body sample.

For a fresh checkout of the distributed example, its documented OVPhysX probe may
be used as a smoke test (for example, a command named
`run_ovphysx_drop_probe.py`). Treat command names and arguments as versioned
included interfaces: inspect `--help` and the local README, and stop if the
installed checkout does not expose the requested capability.

## 4. Readback and replay

Require all expected bodies to be discovered and all returned values to be
finite. Preserve authored exclusions and identity. A useful drop should show
motion, orientation change where expected, contact, and a stable/quiet phase.

To render, replay each authoritative pose onto the corresponding visual root
using the add-on's shared-stage or documented pose-replay path. Keep a mapping
from body ID to visual object/prim and reject missing, duplicate, or reordered
identity. Record whether the output is:

- native OVRTX rendering from the shared stage;
- Blender rendering of replayed poses; or
- a diagnostic viewport/overlay.

Render a small frame preview/contact sheet before a long sequence. Check
visible mesh-to-support distance, tunneling, floating, clipping, coil/contact
behavior, and final readability. Measure visible geometry rather than proxy
centers alone. For contact-critical floor props, a conspicuous gap (roughly
more than a centimeter at scene scale) or penetration is a failure unless the
asset itself justifies it.

## Acceptance gates

Pass only when:

- preflight, service connection, and native simulation creation succeeded;
- expected body count and identity match the authored manifest;
- all sampled poses are finite and include timestamps/state hashes;
- at least one contact event and a bounded settled/quiet phase are observed
  when the scenario expects them;
- no unexpected tunneling, persistent floating, or support penetration is
  visible;
- replay mapping is complete and the render class is correctly labeled;
- source hash is unchanged and all outputs are traceable.

Fail a run for missing native readback, unbounded motion, body identity loss,
or a physics substitute. Return `blocked` for missing runtime/add-on
capabilities, GPU/service unavailability, or unsupported schema. Never turn
either condition into a passing demo.

## Report and artifacts

Write `ovphysx-report.json` with at least:

```json
{
  "status": "pass|blocked|fail",
  "source": {"path": "...", "sha256": "..."},
  "runtime": {"addon": "...", "ovphysx": "...", "ovrtx": "...", "gpu": "..."},
  "settings": {"dt": 0.0166667, "duration_s": 4.0, "gravity": [0, 0, -9.81]},
  "bodies": {"expected": 0, "discovered": 0, "identity_check": "pass"},
  "poses": {"samples": 0, "finite": "pass", "state_hash": "..."},
  "contact": {"events": 0, "settled_fraction": 0.0, "visible_gap_m": null},
  "replay": {"mapping": "pass", "render_class": "ovrtx|blender|diagnostic"},
  "artifacts": [],
  "blockers": [],
  "limitations": []
}
```

Keep pose samples, proxy audit, preview/contact sheet, render frames, and the
report together under the cache directory. Publish only artifacts whose
licenses and source provenance are resolved.
