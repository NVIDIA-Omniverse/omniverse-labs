---
name: usd-composed-stage-open
description: Use this skill when implementing runtime stage-open orchestration for composed USD stages.
metadata:
  author: NVIDIA
---

# usd-composed-stage-open

Use this skill when implementing runtime stage-open orchestration for composed
USD stages.

Normative references:

- `aousd/specifications-public@v1.0.1:specification/composition/README.md`
- `aousd/specifications-public@v1.0.1:specification/glossary`

Owned contracts:

- `contracts/handles/composed-stage-open.handle.json`
- `contracts/spec-coverage/composed-stage-open.coverage.json`

## Responsibilities

- Open the root layer stack through `usd-layer-stack-open`.
- Discover direct external reference asset identifiers from opened layer specs.
- Support the optional adapter `payloadInclusion` scenario control: open and
  supply direct external payload assets for `include`; omit payload opening and
  the payload opinion source for `omit`.
- Recursively discover reference assets through opened referenced layer stacks
  so bounded recursive reference sources receive their supplied layers.
- When optional payload inclusion is requested, discover and open reference
  assets authored inside loaded payload layer stacks so the payload source can
  expand bounded nested references introduced by a loaded payload.
- Resolve those dependency assets through the graph's bounded local resource
  policy and open them through graph layer-open/layer-stack-open boundaries.
- Build reference, payload, relocates, inherits, specializes, variant, and
  composition-arbitrator sources from opened graph-owned objects.
- Populate the stage through `usd-stage-population`.
- Keep every opened layer and source alive for the lifetime of the stage.

## Non-Responsibilities

- Do not perform LIVERPS arbitration locally; call `usd-composition-arbitrator`.
- Do not build traversal indexes locally; call `usd-stage-population`.
- Do not parse USDA text directly or retain parser-local syntax trees.
- Do not claim full recursive composition until recursive payloads,
  variant-contained arcs, payload load masks/mutation, and asset resolver
  contracts exist.

## Current Bounds

The generated implementation should load direct local reference dependency
assets. As optional support for the AOUSD population-mask payload flag, for
`payloadInclusion=include` it should also load direct local payload dependency
assets that can be discovered from opened layer specs. It should then continue
opening reference dependencies discovered in referenced and loaded payload layer
stacks until a bounded fixed point is reached. For `payloadInclusion=omit`, it
should not open payload assets and should omit the payload opinion source so
`payload_loading=absent`.

It may still report composition as `partial` because recursive payload arcs,
variant body arcs, schema fallbacks, full value resolution, and full
instancing/prototypes are not yet covered.
