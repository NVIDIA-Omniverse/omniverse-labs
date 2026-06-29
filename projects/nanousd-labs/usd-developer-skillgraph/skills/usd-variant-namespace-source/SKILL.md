---
name: usd-variant-namespace-source
description: Use this skill when implementing or verifying the first variant-composition namespace source backed by a base namespace source.
metadata:
  author: NVIDIA
---

# usd-variant-namespace-source

Use this skill when implementing or verifying the first variant-composition
namespace source backed by a base namespace source.

## Spec Sources

- `aousd/specifications-public@v1.0.1:specification/composition/README.md`
  variants, variant selection, namespace mapping, and strength ordering
- `aousd/specifications-public@v1.0.1:specification/glossary/README.md`
  `LIVERPS`, `Variant`, `Variant Set`, and `VariantSets`
- `aousd/specifications-public@v1.0.1:specification/document_data_model/README.md`
  `variantSetChildren`, `variantSetNames`, and `variantSelection`
- `aousd/specifications-public@v1.0.1:specification/file_formats/README.md`
  `VariantStatement`, `VariantSetStatement`, and metadata field mapping
- `aousd/specifications-public@v1.0.1:specification/stage_population/README.md`
  stage population consumption

Pinned tag: `v1.0.1`

Pinned tag / commit: `v1.0.1`

## Provides

- A `NamespaceSource` capability with direct selected variant composition
- Variant selection discovery from the non-variant base source's
  strength-ordered `variantSelection` opinions
- Variant set discovery from source-visible `variantSetNames` or
  `variantSetChildren`
- Selected variant path construction through `usd-paths`
- Selected variant root and descendant mapping onto the variant-owning prim
- Path-valued field contents such as relationship `targetPaths` and attribute
  `connectionPaths` remapped into the variant-owning namespace
- Multiple direct selected variant sets ordered by the final source-visible
  `variantSetNames` list
- Base source local/layer-stack and inherit opinions stronger than selected
  variant opinions
- Selected variant opinions weaker than inherits and stronger than relocates,
  references, payloads, and specializes by LIVERPS strength placement
- Composed child and property names across base and selected variant specs
- Diagnostics for missing variant set specs and selected variant specs

## Contract

This skill owns `contracts/handles/variant-namespace-source.handle.json` and
implements the shared namespace-source capability described by
`contracts/capabilities/namespace-source.json`. It also follows
`contracts/capabilities/semantic-runtime-types.json` and consumes the abstract
input boundary described by
`contracts/capabilities/non-variant-namespace-source-input.json`.

The first version is an in-memory direct selected-variant provider. It consumes
a non-variant base namespace source that satisfies the non-variant input
capability. It uses that source to compute variant selections and to inspect
variant-selection paths. It recursively discovers selected variants authored
inside selected variant namespaces. It does not open files, evaluate references
or payloads inside selected variants, or perform full value resolution.

The current bounded graph wires this input to `usd-specializes-namespace-source`
because that is the terminal non-variant provider available today. That concrete
edge is an evaluation dependency, not a semantic requirement that variants
depend on specializes.

The source exposes capabilities:

- `mode`: `composed_namespace`
- `composition`: `partial`
- `value_resolution`: `partial`
- `payload_loading`: `absent`
- `instancing`: `absent`
- `schema_fallbacks`: `absent`
- `child_ordering`: `composed`
- `property_ordering`: `composed`

`partial` composition means selected-variant namespace composition, including
recursive selected variants, without evaluating non-variant arcs authored inside
selected variants.

## Composition Rules

Variant selections are computed after non-variant composition arcs have
supplied their opinions. For a given variant set, read the strength-ordered
opinion list for the current prim and select the strongest source-visible
`variantSelection` entry for that set. The field fallback is `{}`. Missing
selections and empty selection values are inert unless a selection for the same
variant set was already decided while composing the current selected-variant
branch; in that case, reuse the previously decided selection for the later
matching set.

Declared variant sets are read from the composed `variantSetNames` list when
present, otherwise from `variantSetChildren`. The resulting source-visible
variant set order is the selected variant set strength order; the first
selected variant set in that order is stronger than later selected variant
sets. A `variantSelection` entry for a set that is not declared on the prim is
a diagnostic. A declared set whose selected variant spec is absent is also a
diagnostic.

The selected variant path is constructed by applying the selected variant set
and variant name to the owning prim path, for example
`/World{modelingVariant=high}`. The selected variant root maps to the owning
prim path. Descendant prims and properties below the selected variant root are
mapped by replacing the selected variant prefix with the owning prim path.
When a selected variant contributes nested variant sets, discover those sets
from the composed selected-variant namespace. Candidate nested variant specs may
live under the already-selected source variant path, even though the resulting
opinions are exposed at composed scene paths.

Path-valued field contents authored inside selected variant specs are mapped
the same way before fields are emitted. Each ObjectPath under the selected
variant subtree is projected under the variant-owning prim path. For
`targetPaths` and `connectionPaths`, ObjectPaths outside the selected variant
subtree are pruned by this bounded selected-variant provider rather than
emitted with variant-selection paths.

Variant source runtime records must retain owning prim paths, selected variant
paths, and mapped contributor paths as path handles/references. Variant set names
and selected variant names used for identity, lookup, and selected-path
construction must be retained as token/name handles or equivalent semantic name
atoms. A selected variant name is not necessarily a prim/property identifier, so
the chosen name handle must support the AOUSD variant-name grammar. String
spellings are acceptable for adapter summaries, diagnostics, and temporary path
construction, but they must not be the authoritative stored identity in
composition records.
Contributing source provenance must be stored as typed records containing a
source/layer identifier and a semantic spec path. Do not store or later parse
composite strings such as `asset.usda:/Prim` in domain logic; those are adapter
or diagnostic renderings only.

Base-source local and inherits opinions are stronger than selected variant
opinions. Base-source specializes opinions are weaker than selected variant
opinions.
Selected variant opinions are weaker than inherits and stronger than relocates,
references, payloads, and specializes opinions in the full LIVERPS strength
order. Payload-aware final arbitration belongs to `usd-composition-arbitrator`;
generated implementations must not derive this by flattening the base field map
alone.
the base source must preserve enough contributing-opinion identity to insert
selected variant opinions at their LIVERPS position.

Child prim names are composed with specializes contributors weaker than selected
variant contributors, and local/inherits contributors stronger than selected
variant contributors, applying the strongest available ordering after the
strongest contribution. Property names are the union of specializes, selected
variant, and stronger base properties, sorted with `usd-paths` path-element
ordering, then reordered by the strongest non-specializes `propertyOrder` if
present.
Among selected variant contributors for the same exposed prim or property,
variant arcs authored deeper in namespace are stronger than shallower variant
arcs; for selected variants authored at the same prim, the final
`variantSetNames` order remains the same-set strength order.

The graph should place this source after the terminal non-variant composition
provider. In the current bounded graph, that concrete provider chain is:

```text
usd-layer-stack-namespace-source
  -> usd-reference-namespace-source
  -> usd-relocates-namespace-source
  -> usd-inherits-namespace-source
  -> usd-specializes-namespace-source
  -> usd-variant-namespace-source
```

This reflects AOUSD v1.0.1 evaluation: selected variants are computed after
other composition arcs are available, then inserted at their LIVERPS strength
position. Graph order here is evaluation order, not field strength order.
The concrete terminal provider may change as new non-variant composition
providers are added; the variant contract should continue to consume the
abstract non-variant namespace-source input.

## Boundary Guards

Consume a NamespaceSource that satisfies
`contracts/capabilities/non-variant-namespace-source-input.json` for base
opinions. Do not depend on payload-specific APIs and do not bypass the source
boundary to read parser-local structures.

Query selected variant specs through source-visible variant-selection paths
validated by `usd-paths`; stage population must receive only composed scene
paths with selected variant prefixes projected away.

Consume `usd-paths` and `usd-tokens` for path and token identity. Adapter
strings are boundary inputs only.

Do not perform asset resolution, layer opening, external variant fallback-map
policy, references or payloads inside selected variants, inherits or
specializes, relocates, value clips, schema fallback evaluation, stage
population, model hierarchy traversal, instancing, or attribute value
interpolation.

## Test Obligations

- source capability declaration reports `composed_namespace`
- selected variants contribute prim, property, and child opinions
- local and inherits opinions remain stronger than selected variant opinions
- specializes opinions remain weaker than selected variant opinions
- missing or empty selections remain inert
- recursive selected variants contribute prim, property, and child opinions
- previously decided selections are reused for later matching variant sets in
  the current selected-variant branch
- missing selected variants fail clearly
- the integration golden demonstrates stage population consuming the variant
  namespace source
- all cases in
  `goldens/unit/usd-variant-namespace-source/variant-namespace-source.json`
