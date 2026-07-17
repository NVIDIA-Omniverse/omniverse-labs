---
name: simready-addon-install-and-authoring
description: Install or verify the SimReady Blender add-on, author a physics- and semantics-ready asset, run the add-on's validators, and export a traceable SimReady USD package. Use when a user wants Blender content that can be consumed by OVRTX, OVPhysX, Isaac, or other SimReady-aware tools. The OVRTX/OVPhysX runtime is an installed dependency
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
    - simready
    - addon-development
---
# SimReady Add-on Install and Authoring

## When to Use

Use when a user wants Blender content that can be consumed by OVRTX, OVPhysX, Isaac, or other SimReady-aware tools. The OVRTX/OVPhysX runtime is an installed dependency.

## Scope and safety

The SimReady add-on owns SimReady operators, schema details, and export
semantics. Use the version and documentation supplied with the user's
add-on distribution. Do not emulate its metadata by adding arbitrary Blender
custom properties, and do not claim SimReady conformance from a plain Blender
USD export. OVRTX/OVPhysX services and clients are installed dependencies;
invoke the add-on/runtime normally through documented interfaces and record versions.

Always work on a copy under `.cache/simready-YYYYMMDD-HHMMSS/`. Hash the input
`.blend` and prove it is unchanged at closeout.

## Install and preflight

1. Confirm Blender meets the add-on's supported version and that the user has
   the official add-on package, its license/asset dependencies, and any
   SimReady Foundations content required by that release. Never download
   unpublished content or use a similarly named replacement.
2. Install through Blender Preferences (or the add-on's documented extension
   workflow) in an isolated Blender profile. Enable the add-on and restart if
   requested. Capture Blender, add-on, and Foundations versions.
3. Run the add-on's registration/configuration smoke on a disposable scene:
   create one mesh, assign one supported material, add one semantic/physics
   record, invoke the documented validator, and export a tiny USD. If any
   operator, panel, validator, or exporter is absent, stop with a blocked
   preflight instead of falling back to custom properties.
4. If the add-on requires a project configuration, set it using the shipped
   UI/template and record the path and content hash. Keep credentials,
   absolute home paths, and machine-local caches out of published manifests.

## Instructions

### 1. Inventory and isolate

- Record object and material names, parent/collection structure, transforms,
  units, linked libraries, textures, and intended static/dynamic roles.
- Localize only the asset being prepared. Separate independently movable pieces
  into distinct roots before assigning rigid-body data.
- Preserve a source-object-to-exported-prim map. Reject duplicate or ambiguous
  identity; do not silently rename a user's hierarchy without recording it.

### 2. Apply SimReady data through the add-on

Use the add-on's documented panels/operators to author:

- semantic class, asset role, and any required labels;
- nonvisual material family and physical properties (for example friction,
  restitution, density) in the units required by that release;
- rigid-body ownership, mass/inertia, and static versus dynamic intent;
- collision representation: simple primitives or justified compound/convex
  pieces that cover contact surfaces without filling functional openings;
- joints, unibody relationships, or articulation only where the asset needs
  them.

Keep collision proxies hidden from beauty renders but discoverable to the
validator. For a dynamic object, set an explicit mass and a finite authored
pose. Keep floors/supports static and avoid broad invisible ledges that hide
bad collision geometry.

### 3. Validate before export

Run each named validator available in the installed add-on and retain its
individual result. At minimum check:

- registration/configuration and supported schema version;
- semantic root/type and unique object identity;
- material assignment and nonvisual physical properties;
- finite transforms, units, and asset scale;
- rigid-body ownership, explicit mass, and static supports;
- collider existence, bounds/contact coverage, and hidden presentation;
- texture and external dependency closure;
- export eligibility and default prim.

A focused validator passing does not imply the full suite passes. Record full,
focused, and export-eligibility results separately.

### 4. Export and reopen

Export using the SimReady add-on's own USD operator. Keep the authored `.blend`,
exported USD, copied dependencies, validator results, and source/prim map in a
single package. Reopen the exported stage with an installed USD inspector or
the target application and verify schemas, material bindings, prim paths,
default prim, colliders, mass, and semantic metadata survived composition.

If the next step is an OVRTX render, use the installed add-on integration and
label the render provenance. If it is an OVPhysX run, hand off the exported USD
to the supported fixture/runtime path; an authoring pass alone is not physics
evidence.

## Required report

Write `simready-report.json` containing:

```json
{
  "status": "pass|blocked|fail",
  "source": {"path": "...", "sha256": "..."},
  "dependencies": {"blender": "...", "simready_addon": "...", "foundations": "..."},
  "outputs": {"blend": {"path": "...", "sha256": "..."}, "usd": {"path": "...", "sha256": "..."}},
  "checks": {"registration": "pass", "validators": [], "export": "pass", "reopen": "pass"},
  "source_to_prim_map": "source-to-prim.json",
  "blockers": [],
  "limitations": []
}
```

Success requires every required validator to pass, an exported USD that
reopens, finite and traceable physics/semantic data, and an unchanged source
hash. If the official add-on or a required validator/exporter is unavailable,
return `blocked` with the missing version/capability and do not manufacture
SimReady labels or a success report.

## Evidence mode (optional)

When a user needs visible proof of add-on use, capture a controlled before
state, the real add-on interaction, and a saved/reopened after state. Include
the validator panel/result relevant to the claim. Keep UI screenshots as
supporting evidence; the machine-readable report and reopened USD are the
authoritative artifacts.
