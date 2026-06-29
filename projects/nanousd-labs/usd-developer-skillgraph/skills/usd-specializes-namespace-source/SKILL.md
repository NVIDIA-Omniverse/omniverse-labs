---
name: usd-specializes-namespace-source
description: Use this skill when implementing or verifying the first specializes-composition namespace source backed by a composed base namespace source.
metadata:
  author: NVIDIA
---

# usd-specializes-namespace-source

Use this skill when implementing or verifying the first specializes-composition
namespace source backed by a composed base namespace source.

## Spec Sources

- `aousd/specifications-public@v1.0.1:specification/composition/README.md`
  specializes, namespace mapping, and strength ordering
- `aousd/specifications-public@v1.0.1:specification/glossary/README.md`
  `Specializes` and `LIVERPS`
- `aousd/specifications-public@v1.0.1:specification/document_data_model/README.md`
  `specializes`
- `aousd/specifications-public@v1.0.1:specification/file_formats/README.md`
  `SpecializesMetadata`
- `aousd/specifications-public@v1.0.1:specification/stage_population/README.md`
  stage population consumption

Pinned tag: `v1.0.1`

Pinned tag / commit: `v1.0.1`

## Provides

- A `NamespaceSource` capability with direct specializes composition
- Specialize path discovery from the base source's exposed `specializes` field
- Specialized source prim and descendant mapping onto the specialize-authoring
  prim
- Recursive direct specializes expansion through specialized source prims
- Implied specializes arcs from upstream layer-stack sites at the same
  specialized source path
- Path-valued field contents such as relationship `targetPaths` and attribute
  `connectionPaths` remapped into the specialize-authoring namespace
- Target prim opinions stronger than specialized opinions
- Specialized opinions globally weaker than local, inherits, variants,
  relocates, references, and payloads by LIVERPS strength placement
- Composed child and property names across target and specialized specs
- Diagnostics for invalid specialize paths and missing specialized prims

## Contract

This skill owns `contracts/handles/specializes-namespace-source.handle.json` and
implements the shared namespace-source capability described by
`contracts/capabilities/namespace-source.json`. It also follows
`contracts/capabilities/semantic-runtime-types.json`.

The first version is an in-memory direct specializes provider. It consumes a
composed base namespace source, currently `usd-inherits-namespace-source` in
the bounded graph, and applies direct `specializes` entries exposed on base
prim inputs. It does not open files, resolve assets, evaluate specializes
authored inside selected variants, or perform full value resolution.

The source exposes capabilities:

- `mode`: `composed_namespace`
- `composition`: `partial`
- `value_resolution`: `partial`
- `payload_loading`: `absent`
- `instancing`: `absent`
- `schema_fallbacks`: `absent`
- `child_ordering`: `composed`
- `property_ordering`: `composed`

`partial` composition means source-visible direct specialize paths plus
recursive direct specialize paths reachable through specialized source prims,
and implied specializes arcs for upstream layer-stack sites represented by the
base source. Specializes authored inside selected variants remain inert until a
selected-variant recursive composition contract exists.

## Composition Rules

Specialize arcs are read from the base source's source-visible `specializes`
field. This first contract consumes the source-exposed listOp value and
preserves `explicit` entries when present, otherwise `prepend` followed by
`append`. Cross-layer specialize listOp composition remains owned by the base
source or a later composition contract.

Each specialize path item must be a prim path and must not be a property path.
If the specialized prim does not exist in the base namespace source, source
construction fails with a diagnostic naming both the specialize-authoring prim
path and specialized source path.

The specialized source prim maps to the specialize-authoring prim path.
Descendant prims and properties below the specialized source are mapped by
replacing the specialized source prefix with the specialize-authoring prim
path.

Direct specializes expansion is transitive. If a specialized source prim also
has `specializes`, those specialized source opinions contribute through the same
mapping, weaker than the source prim's own opinions and still globally weakest
against non-specializes opinions on the specialize-authoring prim.
Implementations must track semantic specialize-authoring path, specialized
source path, and mapping context for cycle detection; cycles are diagnostics
and must not recurse unboundedly.

Specialized source resolution is source-visible and layer-stack aware. When a
referenced asset authors a specializes arc to an absolute class or base path,
stronger upstream layer stacks that contribute specs at that path create
implied specializes contributors. Do not resolve the target only in the asset
layer when the applicable composed/layer-stack context has stronger opinions at
that same source path.

Path-valued field contents authored under specialized source specs are mapped
the same way before fields are emitted. Each ObjectPath under the specialized
source subtree is projected under the specialize-authoring prim path. For
`targetPaths` and `connectionPaths`, ObjectPaths outside the specialized source
subtree remain in the shared composed namespace for direct specializes rather
than being pruned or remapped.

Specialize source runtime records must retain specialize-authoring prim paths,
specialized source paths, and mapped contributor paths as path
handles/references. String spellings are acceptable for adapter summaries,
diagnostics, and temporary path construction, but not as authoritative USD
object identity. Contributing source provenance must be stored as typed records
containing a source/layer identifier and a semantic spec path.

Specializes use the same namespace mapping as inherits, but they do not use the
same strength behavior. All opinions on the specialize-authoring prim are
stronger than opinions introduced by the specializes arc. This includes selected
variant opinions and other composition arc opinions introduced on the
specialize-authoring prim. Generated implementations must therefore not flatten
specializes fields into the base field map in a way that makes them stronger
than later selected variants.

Child prim names are composed from specialized contributors first, then target
contributors, applying target ordering after target contribution. Property names
are the union of specialized and target properties, sorted with `usd-paths`
path-element ordering, then reordered by the strongest target `propertyOrder` if
present.

The bounded generated graph evaluates specializes after inherits so the
provider can see source opinions supplied by the preceding bounded providers.
Variants then consume the specializes provider as the terminal non-variant
source:

```text
usd-layer-stack-namespace-source
  -> usd-reference-namespace-source
  -> usd-relocates-namespace-source
  -> usd-inherits-namespace-source
  -> usd-specializes-namespace-source
  -> usd-variant-namespace-source
```

Graph order here is evaluation order, not field strength order. Selected
variants are evaluated after the terminal non-variant source so selection can
observe specializes opinions, then selected variant opinions are inserted above
specializes opinions in LIVERPS strength.

## Boundary Guards

Consume `usd-inherits-namespace-source` or another composed namespace source for
base opinions. Do not bypass the source boundary to read parser-local
structures.

Consume `usd-paths` and `usd-tokens` for path and token identity. Adapter
strings are boundary inputs only.

Do not perform asset resolution, layer opening, recursive variant expansion,
value clips, schema fallback evaluation, stage population, model hierarchy
traversal, instancing, or attribute value interpolation.

## Test Obligations

- source capability declaration reports `composed_namespace`
- direct specializes contribute prim, property, and child opinions
- chained specializes contribute transitive source opinions
- implied upstream specializes opinions contribute across represented reference
  layer-stack sites
- target opinions remain stronger than specialized opinions
- selected variants remain stronger than specialized opinions when variants are
  in scope
- missing specialized prims fail clearly
- the integration golden demonstrates stage population consuming the
  specializes namespace source
- all cases in
  `goldens/unit/usd-specializes-namespace-source/specializes-namespace-source.json`
