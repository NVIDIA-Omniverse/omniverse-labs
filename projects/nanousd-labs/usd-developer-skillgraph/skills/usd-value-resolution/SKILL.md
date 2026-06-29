---
name: usd-value-resolution
description: Use this skill when implementing or verifying USD value resolution over already composed opinion stacks.
metadata:
  author: NVIDIA
---

# usd-value-resolution

Use this skill when implementing or verifying USD value resolution over already
composed opinion stacks.

## Spec Sources

- `aousd/specifications-public@v1.0.1:core/1.0.1/core_spec.md` section
  `Value Resolution`
- `aousd/specifications-public@v1.0.1:core/1.0.1/core_spec.md` section
  `Fallback Value Resolution for Attributes`

Pinned tag / commit: `v1.0.1`

## Provides

- A `value_resolution` capability
- A resolver boundary over strong-to-weak typed opinion stacks
- Attribute defaults, blocks, time samples, splines, clips, and schema fallbacks
- Metadata resolution for ordinary fields, special fields, dictionaries, and
  generic listOps
- Relationship target forwarding and raw attribute connection target queries

## Contract

This skill owns `contracts/handles/value-resolution.handle.json` and
`contracts/spec-coverage/value-resolution.coverage.json`.

The resolver consumes target-native opinion records. It must not build
composition indexes, inspect parser-local USDA syntax, parse canonical JSON
dumps as durable storage, or populate stages. Composition, stage population,
and value resolution are separate graph nodes.

When resolving attribute values, walk the composed opinion stack in strength
order and apply AOUSD value-resolution rules. Default-time queries look for
authored `default` values and then schema fallbacks. Timed queries apply
retiming, query authored `timeSamples`, then splines, defaults, value clips, and
fallbacks according to their AOUSD strength and precedence rules.

`valueBlock` and blocked time samples stop weaker authored values. If schema
fallback acquisition produces a fallback, return it; otherwise return the
unauthorable empty sentinel.

Metadata resolution returns the strongest authored value for ordinary fields,
except for fields with special AOUSD rules. Implement `custom`, `specifier`,
`typeName`, `variability`, recursive dictionaries, generic listOps,
root-layer-only layer metadata, relationship target forwarding, and raw
connection target resolution.

## Boundary Guards

Do not consult OpenUSD behavior.

Do not perform composition, namespace remapping, asset resolution, parser
syntax handling, or stage population in this skill.

Do not claim value-resolution behavior from a source provider unless the source
provider supplies the strength-ordered opinions, schema fallback context, layer
retiming, and clip context needed by this resolver.

## Test Obligations

- default value resolution walks weaker opinions when stronger opinions lack a
  default value
- value blocks and blocked time samples stop weaker values
- timed queries cover exact samples, held interpolation, linear interpolation,
  default fallback, and retiming
- schema fallback acquisition covers typed schemas, applied schemas, and the
  empty sentinel/default-metadata fallback
- metadata covers ordinary strongest-authored fields, `custom`, `specifier`,
  `typeName`, `variability`, dictionaries, listOps, and root-layer metadata
- spline cases cover held, linear, curve, value-block, and extrapolated segments
- value clips cover strength, active clips, times mapping, manifest defaults,
  and missing-clip interpolation
- relationship targets cover raw and forwarded targets; connections remain raw
- all cases in `goldens/unit/usd-value-resolution/value-resolution.json`
