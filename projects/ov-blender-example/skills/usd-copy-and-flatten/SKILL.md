---
name: usd-copy-and-flatten
description: Make a reproducible, self-contained USD handoff from a Blender scene or existing USD layer by copying inputs, closing relative dependencies, and flattening into a derived USDA/USDC. Use when preparing an OVRTX, OVPhysX, SimReady, or downstream DCC handoff; never mutate the user's source asset. Requires the installed Blender/USD tooling and supported add-on/runtime.
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
    - usd
---
# USD Copy and Flatten

## When to Use

Use when preparing an OVRTX, OVPhysX, SimReady, or downstream DCC handoff; never mutate the user's source asset. Requires the installed Blender/USD tooling and supported add-on/runtime.

## Purpose and boundaries

Create a derived package that another tool can open without reaching back into
the author's workspace. The source `.blend`, `.usd[a|c|z]`, textures, and
referenced assets remain untouched. The add-on and OVRTX/OVPhysX clients are
installed dependencies: invoke their supported UI/CLI entry points and
record versions, but do not inspect, rebuild, or assume access to their RPC
schemas or source trees.

Use this skill for a handoff, archival snapshot, bug reproduction, or a
flattened input to an OVRTX/OVPhysX fixture. Do not use flattening as a fix for
missing materials, bad transforms, or a broken add-on export; preserve those as
separate findings.

## Instructions

### 1. Freeze source and destination

1. Choose an empty, unique caller-owned handoff directory and refuse to write inside the source
   tree. Copy the source `.blend` or USD file and all explicitly supplied
   textures/references into `source/`.
2. Record the input identities, Blender/add-on/USD tool versions, and requested
   stage/default prim needed to reproduce the handoff. Keep absolute paths local
   and use relative paths in the dependency manifest.
3. If Blender is the source, save a copy first. Export USD with the user's
   stated axis, unit, frame range, material, animation, and visibility options;
   do not silently rotate or rescale geometry.
4. Build a dependency inventory by following USD sublayers, references,
   payloads, clips, variant selections, asset paths, and texture paths. Resolve
   paths relative to each layer before copying. Report unresolved or external
   paths as blockers.

### 2. Produce localized and flattened layers

Keep both forms:

- **Localized layer**: a readable USDA (or original USDC) with the same
  composition structure and paths rewritten to the copied `source/` package.
- **Flattened layer**: a derived USDC/USDA containing composed opinions for
  consumers that cannot resolve the original layer stack.

Prefer the USD distribution's `usdcat --flatten` (or the equivalent command
provided by the installed tool) for flattening. If the command is not
available, use the add-on's documented export/flatten action or a downstream
USD application. Do not implement a new flattening algorithm in Python and do
not hand-edit binary USDC. Preserve authored metadata such as `upAxis`,
`metersPerUnit`, time codes, frame rate, default prim, render settings,
materials, lights, cameras, and semantic/physics schemas.

Flattening can change asset paths, variant composition, and opinions. Compare
the localized and flattened stages after writing; a successful command alone is
not proof of equivalence.

### 3. Inspect and validate

Run the strongest available read-only checks (`usdchecker`, `usdcat -o`, an
installed DCC's stage inspector, or the add-on's validation action). Check:

- stage opens with no unresolved sublayers/references/payloads or textures;
- exactly one intended default prim and stable prim paths;
- finite transforms and sensible axis/unit metadata;
- expected frame range/time samples and animation interpolation;
- material bindings, texture color spaces, cameras, lights, and render products;
- physics schemas/colliders if this is an OVPhysX handoff;
- no accidental source-machine absolute paths or generated-cache paths.

For a render handoff, open the flattened stage through the installed OVRTX
integration and capture one small image. Label the image as an OVRTX render
only when the add-on reports a successful OVRTX render; Blender/EEVEE/Cycles
images are useful comparisons but are not interchangeable evidence.

### 4. Write the handoff manifest

Write `manifest.json` beside the outputs. Include:

```json
{
  "status": "pass|blocked|fail",
  "source": {"path": "...", "sha256": "..."},
  "localized": {"path": "...", "sha256": "..."},
  "flattened": {"path": "...", "sha256": "..."},
  "dependencies": [{"path": "...", "sha256": "...", "kind": "texture|usd|other"}],
  "metadata": {"default_prim": "...", "up_axis": "Z", "meters_per_unit": 1.0},
  "checks": {"opens": "pass", "paths_closed": "pass", "usdchecker": "pass"},
  "blockers": []
}
```

Hash the final files after all path rewrites. Keep the manifest, localized
layer, flattened layer, and copied dependencies together when publishing.
Before sharing, sanitize reports and diagnostic artifacts with
`blender-sanitized-support-bundle` and confirm the package contains no
source-machine absolute paths, credentials, or restricted URLs.

## Acceptance and failure handling

Success requires a source hash that still matches, a localized layer with all
dependencies present, a readable flattened layer, and zero unresolved path or
metadata blockers. If a dependency, USD utility, add-on operation, or OVRTX
runtime is unavailable, return `blocked` with the exact missing capability and
do not fabricate a flattened file or render result. If flattening opens but
changes visible content, retain both variants and report the affected prims;
do not overwrite the source or silently choose one.
