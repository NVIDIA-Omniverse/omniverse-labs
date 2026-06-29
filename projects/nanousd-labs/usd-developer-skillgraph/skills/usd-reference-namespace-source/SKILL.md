---
name: usd-reference-namespace-source
description: Use this skill when implementing or verifying the first reference-composition namespace source backed by a base namespace source and already-opened referenced layers.
metadata:
  author: NVIDIA
---

# usd-reference-namespace-source

Use this skill when implementing or verifying the first reference-composition
namespace source backed by a base namespace source and already-opened referenced
layers.

## Spec Sources

- `specification/composition/README.md` references and composition strength
- `specification/file_formats/README.md` `ReferencesMetadata` and
  `ReferenceListItem`
- `specification/document_data_model/README.md` `references:
  listop<Reference>` and `defaultPrim`
- `specification/stage_population/README.md` namespace source consumption

Pinned tag / commit: `v1.0.1`

## Provides

- A `NamespaceSource` capability with direct reference composition
- Direct reference discovery from the base source's exposed `references` field
- Path-only internal references within the base namespace source
- Reference target mapping from an external asset layer into the referring prim
- Asset-only reference target selection through the referenced layer
  `defaultPrim`
- Bounded recursive and chained reference expansion through supplied referenced
  layers
- Local opinions stronger than referenced opinions
- Same-reference-family ordering by namespace depth and source-visible sibling
  list order
- Reference-family-only prim/property opinion views, or equivalent separable
  provenance, for downstream composition consumers such as relocates
- Path-valued field contents such as relationship `targetPaths` and attribute
  `connectionPaths` remapped into the referring namespace
- Arc-local `layerRelocates` mappings authored in referenced layer stacks and
  carried by nested reference records
- Composed child and property names across local and referenced specs
- Diagnostics for missing referenced layers and missing target prims

## Contract

This skill owns `contracts/handles/reference-namespace-source.handle.json` and
implements the shared namespace-source capability described by
`contracts/capabilities/namespace-source.json`. It also follows
`contracts/capabilities/semantic-runtime-types.json`.

The first version is an in-memory bounded reference provider. It consumes a base
namespace source, such as `usd-layer-stack-namespace-source`, plus a map of
already-opened reference layers keyed by authored asset identifier. It does not
resolve asset identifiers, open files, evaluate payloads, or perform full value
resolution.

The source exposes capabilities:

- `mode`: `composed_namespace`
- `composition`: `partial`
- `value_resolution`: `partial`
- `payload_loading`: `absent`
- `instancing`: `absent`
- `schema_fallbacks`: `absent`
- `child_ordering`: `composed`
- `property_ordering`: `composed`

`partial` composition means external references, path-only internal references,
and bounded recursive references authored on supplied referenced target
subtrees. Reference target descendants are mapped under the referring prim, and
nested references compose their parent and child namespace mappings.

## Composition Rules

Reference arcs are read from the base source's source-visible `references`
field. This first contract consumes the source-exposed listOp value and
preserves `explicit` entries when present, otherwise `prepend` followed by
`append`. Cross-layer reference listOp composition remains deferred to a later
composition contract.

External reference arcs have an authored asset identifier. The referenced layer
is supplied by the caller or adapter. If an arc also has a target path, that
prim is the reference root. If the arc has only an asset, use the referenced
layer's `defaultPrim`; missing defaultPrim is a diagnostic. Path-only internal
references use the base namespace source as the referenced namespace and must
name an explicit target prim path.

The reference target prim maps to the referring prim path. Descendant prims and
properties below the target are mapped by replacing the target prefix with the
referring prim path.

Nested references authored inside supplied referenced target subtrees must be
expanded transitively while their authored asset layers are supplied. Compose
the accumulated namespace mapping at every hop: a nested target prim contributes
at the parent exposed prim plus the nested source suffix. Detect cycles by
authored asset identifier, referenced target path, referring prim path, and
mapping context. Cycles are diagnostics and must never recurse without bound or
crash generated implementations.

If the referenced layer stack that authors a nested reference also authors
`layerRelocates`, carry that relocate namespace mapping on the nested reference
record. The mapping must hide relocated source paths, expose relocated target
paths, suppress stale target collisions from the referenced target layer, map
remote-source reads back to the relocated source specs, prune deleted source
subtrees, and remap path-valued fields through the accumulated reference plus
relocate mapping.

Path-valued field contents authored under referenced specs are mapped the same
way before fields are emitted. Each ObjectPath under the referenced target
subtree is projected under the referring prim path. For `targetPaths` and
`connectionPaths`, ObjectPaths outside the referenced target subtree are pruned
by this bounded external-reference provider rather than emitted in
referenced-layer coordinates. For path-only internal references, ObjectPaths
outside the referenced target subtree remain in the shared composed namespace.

Local base-source opinions are stronger than referenced opinions. Stronger
`over` specs may override weaker referenced `def` specs without erasing the
effective definition.

Keep referenced-family opinions separable from local base-source opinions even
when the provider also exposes a flattened NamespaceSource view for stage
population tests. Downstream providers such as relocates must be able to consume
referenced opinions at a path without accidentally consuming local opinions
authored at that same path.

When multiple reference arcs contribute to the same exposed path, order
reference-family opinions strong-to-weak by namespace depth first: deeper
authored reference arcs are stronger than ancestor reference arcs. For sibling
reference arcs authored at the same prim, preserve the source-visible authored
list order.

Child prim names are composed from referenced contributors first, then local
base-source contributors, applying local ordering after local contribution.
Property names are the union of referenced and local properties, sorted with
`usd-paths` path-element ordering, then reordered by the strongest local
`propertyOrder` if present.

## Boundary Guards

Consume `usd-layer-stack-namespace-source` or another namespace source for base
opinions. Do not bypass the source boundary to read parser-local structures.

Reference source runtime records must retain referring prim paths, referenced
target paths, and mapped contributor paths as path handles/references. String
spellings are acceptable for asset identifiers, diagnostics, and adapter
summaries, but not as the authoritative identity for USD object paths.
Contributing source provenance must be stored as typed records containing a
source/layer identifier and a semantic spec path. Do not store or later parse
composite strings such as `asset.usda:/Prim` in domain logic; those are adapter
or diagnostic renderings only.

Consume `usd-document-model` for already-opened referenced layers. Do not store
or compose canonical layer dump JSON as durable state.

Consume `usd-paths` and `usd-tokens` for path and token identity. Adapter
strings are boundary inputs only.

Do not perform asset resolution, layer opening, payload loading, inherits,
specializes, variant composition, root-layer relocates, value clips, schema fallback
evaluation, stage population, model hierarchy traversal, instancing, or
attribute value interpolation.

## Test Obligations

- source capability declaration reports `composed_namespace`
- direct external references contribute prim, property, and child opinions
- local opinions remain stronger than referenced opinions
- namespace-deeper reference arcs are stronger than ancestor reference arcs
- sibling reference arcs preserve source-visible list order
- asset-only references use the referenced layer `defaultPrim`
- missing reference layers fail clearly
- nested/chained references expand when their layers are supplied
- nested references authored in layer stacks with relocates remap children,
  properties, targetPaths, and connectionPaths through that arc-local mapping
- recursive reference cycles are diagnosed without crashes
- all cases in
  `goldens/unit/usd-reference-namespace-source/reference-namespace-source.json`
