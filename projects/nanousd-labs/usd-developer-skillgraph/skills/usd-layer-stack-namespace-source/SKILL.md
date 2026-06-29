---
name: usd-layer-stack-namespace-source
description: Use this skill when implementing or verifying the first composed namespace source backed by a root document-model layer and its already-opened recursive sublayers.
metadata:
  author: NVIDIA
---

# usd-layer-stack-namespace-source

Use this skill when implementing or verifying the first composed namespace
source backed by a root document-model layer and its already-opened recursive
sublayers.

## Spec Sources

- `specification/composition/README.md` sublayer and LIVERPS layer-stack rules
- `specification/stage_population/README.md` ordered children and properties
- `specification/document_data_model/README.md` subLayers, subLayerOffsets,
  population fields, and specifier fields
- `specification/foundational_data_types/README.md` list operation combination
  rules

Pinned tag / commit: `v1.0.1`

## Provides

- A `NamespaceSource` capability backed by a root layer stack
- Recursive sublayer stack construction from already-opened layers
- Layer-stack capability declaration as `layer_stack`
- Strong-to-weak layer stack inspection
- Composed prim existence by path across stack entries
- Composed property existence by path across stack entries
- Population field selection for stage population
- Authored listOp field composition across contributing specs when field types
  match
- Ordered child prim names merged across contributing specs
- Ordered property names merged across contributing specs
- Diagnostics for unresolved, duplicate, or unsupported sublayer inputs
- Diagnostics for cycles across recursive sublayers

## Contract

This skill owns `contracts/handles/layer-stack-namespace-source.handle.json`
and implements the general namespace-source capability described by
`contracts/capabilities/namespace-source.json`.

The first sublayer composition node is deliberately in-memory. It consumes a
root document-model `Layer` and a map or list of already-opened sublayer
`Layer`s supplied by an adapter, caller, or future loader. It does not resolve
asset identifiers, open files, load packages, or apply resolver search paths.
Those concerns belong to loader/resolver skills.

Layer strength is:

1. the root layer
2. recursive sublayers in depth-first authored `subLayers` order, strongest to
   weakest; nested sublayers appear immediately after the layer that authored
   them and before later sibling sublayers

The provider exposes a `NamespaceSource` whose capabilities are:

- `mode`: `layer_stack`
- `composition`: `partial`
- `value_resolution`: `partial`
- `payload_loading`: `absent`
- `instancing`: `absent`
- `schema_fallbacks`: `absent`
- `child_ordering`: `composed`
- `property_ordering`: `composed`

`partial` composition means sublayer stack merging only. It does not include
references, payloads, variants, inherits, specializes, relocates, value clips,
or schema fallback injection.

`partial` value resolution means only the field opinions needed by population
and namespace queries are selected or summarized. Attribute value interpolation,
time sample resolution, dictionary merging, clips, and fallback value resolution
remain separate skills.

## Composition Rules

Build a layer stack from the root layer's authored `subLayers` field and the
authored `subLayers` fields of supplied recursive sublayers. Each listed asset
must correspond to exactly one already-opened layer supplied to the provider.
Missing sublayers or cycles produce diagnostics and prevent a successful source
unless an adapter explicitly requests a diagnostic-only dry run.

`subLayerOffsets` are paired with `subLayers` and reported in layer-stack
inspection. They are not applied to time samples, timecode-valued fields, or
attribute values in this skill.

A prim or property path exists if any stack layer contributes a spec at that
path. The source must retain the ordered contributing spec list, strongest to
weakest, or an equivalent target-native representation sufficient to answer
field and child/property queries without rescanning all layer specs on every
query.

For ordinary scalar/token/bool population fields, expose the strongest authored
field opinion. For same-typed authored listOp fields, combine all contributing
listOp opinions in strong-over-weak layer-stack order using the standard listOp
combination rules from `foundational_data_types`. For `specifier`, expose an
effective population specifier:

- `class` if the strongest contributing defining specifier is `class`
- `def` if any contributing specifier is `def` and no stronger `class` applies
- `over` when no contributing `def` or `class` exists

This allows stronger `over` specs to contribute overrides without erasing a
weaker definition from stage-population query flags.

Child prim names are composed by iterating contributing specs from weakest to
strongest. After each spec contribution, append that spec's `primChildren`,
remove duplicate names while preserving first occurrence, and apply that same
spec's sparse `primOrder`. Relocates remain deferred and must not be simulated.

Property names are the union of contributing spec `properties` lists. Sort the
union with `usd-paths` path-element ordering, then apply only the strongest
authored `propertyOrder` opinion. Schema fallback properties remain absent.

Property field maps expose the strongest authored scalar field per field name
and the composed authored listOp field for matching listOp-typed contributions.
Do not combine dictionaries, time samples, splines, or non-listOp values across
layers in this skill.

## Boundary Guards

Consume `usd-document-model` for each already-opened layer. Do not duplicate
layers into independent JSON spec maps as durable storage.

Consume `usd-paths` for path identity, parent/name path construction, path-kind
checks, and path-element ordering. Adapter string paths may be parsed at the
boundary only.

Consume `usd-tokens` and the document-model field-token vocabulary for field
names, child names, property names, and token-valued fields.

Consume the `NamespaceSource` capability shape rather than adding a parallel
stage-population API. This provider supplies source inputs; it does not create
stage nodes, apply population masks, prune inactive descendants, or build stage
traversal indexes.

Do not perform asset resolution, layer opening, reference/payload loading,
inherits/specializes/variant composition, relocates, schema fallback evaluation,
model hierarchy traversal, instancing, or attribute value resolution.

## Test Obligations

- source capability declaration reports `layer_stack`
- stack summary reports root and recursive sublayers in depth-first strength order
- missing sublayer inputs produce diagnostics
- recursive sublayer cycles produce diagnostics
- stronger root fields override weaker sublayer fields
- same-typed authored listOp fields combine across layer-stack specs
- weaker sublayers supply prim and property specs absent from stronger layers
- stronger `over` plus weaker `def` remains effectively defined
- child names merge from weakest to strongest with `primOrder` after each merge
- property names union across layers, sorted by path element order, then use the
  strongest `propertyOrder`
- property field maps use strongest authored scalar fields and composed authored
  listOp fields per field key
- `subLayerOffsets` are paired and reported but do not retime values
- all cases in
  `goldens/unit/usd-layer-stack-namespace-source/layer-stack-namespace-source.json`
