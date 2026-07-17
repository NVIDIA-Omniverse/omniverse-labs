---
name: simready-prop-journey
description: "Take one Blender prop through a concise SimReady happy path: preserve a safe working copy, author and validate with the official SimReady add-on, review native OVPhysX behavior, and produce a traceable USD handoff. Use when an artist wants an end-to-end prop workflow without loading every specialist skill at once."
---

# SimReady prop journey

Use this as the short orchestrator. Delegate detailed work and acceptance checks to the named public skills; do not duplicate or weaken their gates.

## Happy path

1. Run `blender-content-safety-and-privacy`, then create a unique working directory, copy the supplied source into it, hash the original and copy, and leave the original unchanged. Keep dependencies with the working copy and stop if the source or required assets cannot be handled safely.
2. Run `simready-addon-install-and-authoring`. Inventory the prop, author semantics and physics through the official add-on, run every required validator, export with the add-on, reopen the USD, and retain its source-to-prim map. Stop as `blocked` when the required add-on capability is unavailable; never manufacture a passing label or export.
3. Run `ovphysx-drop-contact-acceptance` for the smallest behavior test that represents the prop's intended use. Review authoritative motion, contact, settling, collision fit, and stable identity. Keep diagnostic, Blender, and native OVRTX imagery clearly distinguished.
4. Run `usd-copy-and-flatten` only when the recipient needs a dependency-complete or flattened package. Then run `usd-inspect-and-provenance` on the exact handoff candidate.
5. Deliver the validated `.blend`, localized and optional flattened USD, copied dependencies, validator and OVPhysX reports, source-to-prim map, checksums, and a concise limitations list. Use relative paths in shareable manifests and include only cleared assets.

## Completion gate

Call the journey complete only when the SimReady validators and reopened export pass, the applicable OVPhysX behavior gates pass, the handoff opens with resolved dependencies, and all artifacts trace back to the unchanged source. Otherwise return `blocked` or `fail` with the first actionable reason and preserve the partial evidence.
