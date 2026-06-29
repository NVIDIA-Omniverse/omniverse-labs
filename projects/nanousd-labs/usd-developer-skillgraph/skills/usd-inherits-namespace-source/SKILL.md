---
name: usd-inherits-namespace-source
description: Use this skill when implementing or verifying the first inherits-composition namespace source backed by a composed base namespace source.
metadata:
  author: NVIDIA
---

# usd-inherits-namespace-source

Use this skill when implementing or verifying the first inherits-composition
namespace source backed by a composed base namespace source.

## Spec Sources

- `aousd/specifications-public@v1.0.1:specification/composition/README.md`
  inherits, namespace mapping, implied inherits, and strength ordering
- `aousd/specifications-public@v1.0.1:specification/glossary/README.md`
  `Inherits` and `LIVERPS`
- `aousd/specifications-public@v1.0.1:specification/document_data_model/README.md`
  `inheritPaths`
- `aousd/specifications-public@v1.0.1:specification/file_formats/README.md`
  `InheritsMetadata`
- `aousd/specifications-public@v1.0.1:specification/stage_population/README.md`
  stage population consumption

Pinned tag: `v1.0.1`

Pinned tag / commit: `v1.0.1`

## Provides

- A `NamespaceSource` capability with direct inheritPaths composition
- Inherit path discovery from the base source's exposed `inheritPaths` field
- Inherited source prim and descendant mapping onto the inherit-authoring prim
- Recursive direct inherit expansion through inherited source prims
- Implied inherit arcs from upstream layer-stack sites at the same inherited
  source path
- Path-valued field contents such as relationship `targetPaths` and attribute
  `connectionPaths` remapped into the inherit-authoring namespace
- Same-path local layer-stack target opinions stronger than inherited opinions
- Implied upstream inherited opinions stronger than weaker reference-introduced
  target opinions
- Inherited opinions stronger than variants, relocates, references, and
  payloads by LIVERPS strength placement
- Composed child and property names across local and inherited specs
- Diagnostics for invalid inherit paths and missing inherited prims

## Contract

This skill owns `contracts/handles/inherits-namespace-source.handle.json` and
implements the shared namespace-source capability described by
`contracts/capabilities/namespace-source.json`. It also follows
`contracts/capabilities/semantic-runtime-types.json`.

The first version is an in-memory direct inherits provider. It consumes a
composed base namespace source, currently `usd-relocates-namespace-source` in
the bounded graph, and applies direct `inheritPaths` entries exposed on base
prim inputs. It does not open files, resolve assets, evaluate inherits authored
inside selected variants, or perform full value resolution.

The source exposes capabilities:

- `mode`: `composed_namespace`
- `composition`: `partial`
- `value_resolution`: `partial`
- `payload_loading`: `absent`
- `instancing`: `absent`
- `schema_fallbacks`: `absent`
- `child_ordering`: `composed`
- `property_ordering`: `composed`

`partial` composition means source-visible direct inherit paths plus recursive
direct inherit paths reachable through inherited source prims, and implied
inherit arcs for upstream layer-stack sites represented by the base source.
Inherits authored inside selected variants remain inert until a selected-variant
recursive composition contract exists.

## Composition Rules

Inherit arcs are read from the base source's source-visible `inheritPaths`
field. This first contract consumes the source-exposed listOp value and
preserves `explicit` entries when present, otherwise `prepend` followed by
`append`. Cross-layer inherit listOp composition remains owned by the base
source or a later composition contract.

Each inherit path item must be a prim path and must not be a property path. If
the inherited prim does not exist in the base namespace source, source
construction fails with a diagnostic naming both the inherit-authoring prim path
and inherited source path.

The inherited source prim maps to the inherit-authoring prim path. Descendant
prims and properties below the inherited source are mapped by replacing the
inherited source prefix with the inherit-authoring prim path.

Direct inherit expansion is transitive. If an inherited source prim also has
`inheritPaths`, those inherited source opinions contribute through the same
mapping, weaker than the source prim's own opinions. Implementations must track
semantic inherit-authoring path, inherited source path, and mapping context for
cycle detection; cycles are diagnostics and must not recurse unboundedly.

Inherited source resolution is source-visible and layer-stack aware. When a
referenced asset authors an inherit to an absolute class path, stronger
upstream layer stacks that contribute specs at that class path create implied
inherit contributors. Do not resolve the target only in the asset layer when
the applicable composed/layer-stack context has stronger opinions at that same
source path.
Do not flatten weaker reference-introduced target fields ahead of those implied
upstream inherited fields; the provider or composition arbitrator must retain enough
provenance to apply the LIVERPS placement.

Path-valued field contents authored under inherited source specs are mapped the
same way before fields are emitted. Each ObjectPath under the inherited source
subtree is projected under the inherit-authoring prim path. For `targetPaths`
and `connectionPaths`, ObjectPaths outside the inherited source subtree remain
in the shared composed namespace for direct inherits rather than being pruned or
remapped.

Inherit source runtime records must retain inherit-authoring prim paths,
inherited source paths, and mapped contributor paths as path handles/references.
String spellings are acceptable for adapter summaries, diagnostics, and
temporary path construction, but not as authoritative USD object identity.
Contributing source provenance must be stored as typed records containing a
source/layer identifier and a semantic spec path. Do not store or later parse
composite strings such as `asset.usda:/Prim` in domain logic; those are adapter
or diagnostic renderings only.

Same-path local layer-stack target opinions are stronger than inherited
opinions. Stronger `over` specs may override weaker inherited `def` specs
without erasing the effective definition. Weaker reference-introduced target
opinions do not automatically outrank implied upstream inherited opinions.

Child prim names are composed from inherited contributors first, then local
target contributors, applying local ordering after local contribution. Property
names are the union of inherited and local properties, sorted with `usd-paths`
path-element ordering, then reordered by the strongest local `propertyOrder` if
present.

Inherits are stronger than variants, relocates, references, payloads, and
specializes in LIVERPS strength order. The bounded generated graph evaluates
inherits after relocates so the provider can see source opinions supplied by
the preceding bounded providers. Payload opinions remain separate until
`usd-composition-arbitrator` merges them with the terminal composed namespace source.
Specializes consumes the inherits provider when direct specializes are in
scope, and variants consume specializes as the terminal non-variant source:

```text
usd-layer-stack-namespace-source
  -> usd-reference-namespace-source
  -> usd-relocates-namespace-source
  -> usd-inherits-namespace-source
  -> usd-specializes-namespace-source
  -> usd-variant-namespace-source
```

Graph order here is evaluation order, not field strength order.

## Boundary Guards

Consume `usd-relocates-namespace-source` or another composed namespace source
for base opinions. Do not bypass the source boundary to read parser-local
structures.

Consume `usd-paths` and `usd-tokens` for path and token identity. Adapter
strings are boundary inputs only.

Do not perform asset resolution, layer opening, recursive specializes
expansion, recursive variant expansion, value clips, schema fallback
evaluation, stage population, model hierarchy traversal, instancing, or
attribute value interpolation.

## Test Obligations

- source capability declaration reports `composed_namespace`
- direct inherits contribute prim, property, and child opinions
- chained inherits contribute transitive source opinions
- implied upstream inherit opinions contribute across represented reference
  layer-stack sites
- same-path local layer-stack target opinions remain stronger than inherited
  opinions
- weaker reference-introduced target opinions do not override implied upstream
  inherited opinions
- missing inherited prims fail clearly
- the integration golden demonstrates stage population consuming the inherits
  namespace source
- all cases in
  `goldens/unit/usd-inherits-namespace-source/inherits-namespace-source.json`
