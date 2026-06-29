---
name: usd-relocates-namespace-source
description: Use this skill when implementing or verifying the first relocates-composition namespace source backed by a composed base namespace source.
metadata:
  author: NVIDIA
---

# usd-relocates-namespace-source

Use this skill when implementing or verifying the first relocates-composition
namespace source backed by a composed base namespace source.

## Spec Sources

- `aousd/specifications-public@v1.0.1:specification/composition/README.md`
  relocates, namespace mapping, and strength ordering
- `aousd/specifications-public@v1.0.1:specification/glossary/README.md`
  `LIVERPS` and `Relocates`
- `aousd/specifications-public@v1.0.1:specification/document_data_model/README.md`
  `layerRelocates`
- `aousd/specifications-public@v1.0.1:specification/file_formats/README.md`
  `RelocatesMetadata` and layer relocates metadata
- `aousd/specifications-public@v1.0.1:specification/stage_population/README.md`
  stage population consumption

Pinned tag: `v1.0.1`

Pinned tag / commit: `v1.0.1`

## Provides

- A `NamespaceSource` capability with direct `layerRelocates` composition
- Relocate discovery from the base source's source-visible root/layer fields
- Source-to-target prim subtree projection
- Empty relocate target deletion/removal behavior
- Validation of the direct relocate restrictions listed by AOUSD v1.0.1
- Target-local opinions stronger than relocated source opinions
- Recoverable diagnostics for local-layer-stack opinions at relocate source
  paths without dropping otherwise-valid relocate entries
- Recoverable diagnostics for nested relocate sources authored beneath an
  already-relocated ancestral source path
- Requirements for reference and loaded payload providers to carry arc-local
  relocate namespace mappings authored in the layer stacks that introduce
  nested arcs
- Composed child and property names across target-local and relocated source
  specs
- Diagnostics for invalid relocate entries

## Contract

This skill owns `contracts/handles/relocates-namespace-source.handle.json` and
implements the shared namespace-source capability described by
`contracts/capabilities/namespace-source.json`. It also follows
`contracts/capabilities/semantic-runtime-types.json`.

The first version is an in-memory direct relocates provider. It consumes a
composed base namespace source, currently `usd-reference-namespace-source` in
the bounded graph, and applies direct `layerRelocates` entries exposed at the
base root/layer input. `usd-composition-arbitrator` may supply a richer
local/reference/payload base source. This provider does not open files, resolve
assets, automatically rewrite invalid ancestral relocate source paths, or
perform full value resolution. Arc-local relocate mappings authored in
referenced or loaded payload layer stacks are applied by the reference and
payload providers that introduce those layer stacks.

The source exposes capabilities:

- `mode`: `composed_namespace`
- `composition`: `partial`
- `value_resolution`: `partial`
- `payload_loading`: `absent`
- `instancing`: `absent`
- `schema_fallbacks`: `absent`
- `child_ordering`: `composed`
- `property_ordering`: `composed`

`partial` composition means direct relocate entries over the currently composed
base source plus bounded arc-local mappings for references and loaded payload
nested-reference records. Automatic source-path rewriting for invalid ancestral
relocates, and recursive composition of relocate mappings into inherit or
specialize arcs, remain future contracts.

## Composition Rules

Read relocates from the source-visible `layerRelocates` field on the base
source's root/layer input. The domain model is a native map from source prim
path handle to optional target prim path handle. Adapter JSON may represent that
map as a keyed object or an array of `{source,target}` records, but runtime
records must store path identity as semantic path handles/references.

Validate direct entries before applying them. A source path and non-empty target
path must be prim paths and must not be `/` or the pseudo-root. Reject duplicate
source paths, duplicate non-empty target paths, identity mappings, target paths
below source paths, and mappings that place a prim at the path of an existing
ancestor of the source path. An empty target is not an error; it removes the
source subtree from the composed namespace.

A non-empty relocate maps the source subtree to the target subtree. For a prim
or property under the target, query the corresponding source path by replacing
the target prefix with the source prefix. For source paths and descendants,
hide the relocated source subtree from composed namespace queries.

Target-local opinions are stronger than relocated source opinions in this first
provider. Stronger `over` specs may override weaker relocated `def` specs
without erasing the effective definition. Child names are composed from
relocated source contributors first, then target-local base contributors,
applying target-local ordering after target-local contribution. Property names
are the union of relocated source and target-local properties, sorted with
`usd-paths` path-element ordering, then reordered by the strongest target-local
`propertyOrder` if present.

AOUSD requires source-local opinions at a relocate source path to be a
composition error and not contribute through that source path. The error is
recoverable: retain and apply the otherwise-valid relocate entry, diagnose the
ignored source-local opinions, and relocate only non-local source opinions. If
the base source has already combined local and reference-family opinions at the
source path, consume a reference-family-only source view or equivalent
per-family provenance rather than relocating a flattened local-plus-remote
field map.

If a relocate source is authored beneath an already-relocated source path, the
entry is invalid because the authored path is not the ancestral relocated path.
Generated implementations must diagnose that as a recoverable composition
error, ignore that invalid nested entry, continue applying other valid entries,
and never crash.

When references or loaded payloads are authored inside a layer stack with
`layerRelocates`, the arc provider must carry that layer stack's relocate
namespace mapping with the nested record. That mapping hides source paths,
exposes target paths, redirects remote-source reads to relocated source specs,
prunes deleted sources, suppresses stale target collisions, and remaps
path-valued `targetPaths` and `connectionPaths`.

Relocates sit between variants and references in LIVERPS strength order. The
bounded generated graph evaluates references before relocates so the relocates
provider can see referenced source opinions that need to move. Payload opinions
are produced by a separate payload opinion source until `usd-composition-arbitrator`
merges them with the terminal composed namespace source. When no stronger
non-variant provider is present, variants may consume relocates directly. With
inherits and specializes in scope, inherits consumes relocates, specializes
consumes inherits, and variants consume specializes as the terminal non-variant
source:

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

Consume `usd-reference-namespace-source` in the current bounded graph, or a
composed namespace source supplied by `usd-composition-arbitrator`, for base
opinions. Do not bypass the source boundary to read parser-local structures.

Relocates source runtime records must retain source paths, target paths, and
mapped contributor paths as path handles/references. String spellings are
acceptable for adapter summaries, diagnostics, and temporary path construction,
but not as authoritative USD object identity.
Contributing source provenance must be stored as typed records containing a
source/layer identifier and a semantic spec path. Do not store or later parse
composite strings such as `asset.usda:/Prim` in domain logic; relocate
source-path opinion checks must compare the typed fields directly.

Consume `usd-paths` and `usd-tokens` for path and token identity. Adapter
strings are boundary inputs only.

Do not perform asset resolution, layer opening, automatic ancestral relocate
source rewriting, inherit/specialize arc-local namespace mapping composition,
recursive variant expansion, value clips, schema fallback evaluation, stage
population, model hierarchy traversal, instancing, or attribute value
interpolation.

## Test Obligations

- source capability declaration reports `composed_namespace`
- direct relocates move a referenced child prim and property to the target path
- target-local opinions remain stronger than relocated source opinions
- empty target removes the source subtree
- invalid ancestral relocate source paths are recoverable diagnostics
- nested references in referenced and loaded payload layer stacks receive
  arc-local relocate mappings
- invalid direct relocate entries fail clearly
- the integration golden demonstrates stage population consuming the relocates
  namespace source
- all cases in
  `goldens/unit/usd-relocates-namespace-source/relocates-namespace-source.json`
