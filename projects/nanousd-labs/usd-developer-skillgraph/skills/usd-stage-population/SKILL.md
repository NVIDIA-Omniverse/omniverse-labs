---
name: usd-stage-population
description: Use this skill when implementing or verifying population of a stage hierarchy from a namespace source.
metadata:
  author: NVIDIA
---

# usd-stage-population

Use this skill when implementing or verifying population of a stage hierarchy
from a namespace source.

## Spec Sources

- `specification/stage_population/README.md`
- `specification/document_data_model/README.md` population fields
- `specification/path_grammar/README.md` path element ordering

Pinned tag / commit: `v1.0.1`

## Provides

- Target-native `Stage` populated from a `NamespaceSource`
- Root scene object at `/`
- Prim traversal and path lookup for populated prims
- Parent and child prim queries
- Property enumeration for populated prims
- Population mask handling
- Active descendant pruning
- Default prim lookup
- Stage query flags for active, loaded, defined, abstract, and instanceable
- Portable model hierarchy traversal and query state for `kind`
- Bounded instance summaries and shared representation IDs for instanceable
  prims backed by supported composition-arc provenance
- Diagnostics passthrough from the namespace source

## Contract

This skill owns `contracts/handles/stage-population.handle.json` and the
coverage ledger in `contracts/spec-coverage/stage-population.coverage.json`.
It also follows `contracts/capabilities/semantic-runtime-types.json`.

Stage population consumes the generic namespace-source capability rather than a
particular layer or parser. The first provider is
`usd-single-layer-namespace-source`, where every path has at most one authored
opinion. Future composition nodes should provide the same namespace-source
capability at a stronger capability level.

The populated stage is a target-native scene graph/query surface. It must not
be a canonical JSON dump, a parser-local USDA tree, or the document-model layer
itself. JSON appears only at conformance adapter boundaries. After population,
the public Stage API is read-only: do not expose mutable maps, vectors, indexes,
or caches that allow callers to replace populated prims/properties or reorder
traversal.

Population starts at `/`, retrieves source child names, applies population-mask
filtering when provided, applies active pruning, and creates populated prim
nodes for reachable paths. Traversal exposed by this skill returns populated
non-root prims in stage order.

For a single-layer source, child ordering is:

1. start with the source `primChildren` order
2. apply the source `primOrder` sparse order for the parent
3. leave names absent from `primOrder` in source order

For properties, apply path element ordering to the source property names, then
apply the source `propertyOrder` sparse order. Future composed providers may
expose merged children/properties with `child_ordering=composed` and
`property_ordering=composed`; in that case stage population preserves provider
order and remains responsible only for population-mask filtering, active
pruning, and the query indexes built over populated nodes.

## Query Policy

The first contract uses source-visible field opinions plus explicit fallbacks:

- `active` falls back to `true`
- `loaded` is `true` when the source declares payload loading absent
- `specifier` falls back to `over`
- `defined` is true only if the prim and all populated ancestors have
  specifier `def` or `class`
- `abstract` is true if the prim or any populated ancestor has specifier
  `class`
- `instanceable` falls back to `false`
- `typeName`, `kind`, and `defaultPrim` fall back to empty token/string

Model hierarchy uses only portable AOUSD kind values. `group` and `assembly`
are included and continue traversal. `component` is included but terminates
traversal, so descendants of a component are not part of the model hierarchy.
`subcomponent`, empty, and unknown kind values are excluded and terminate
continuity. Custom kind aliases are not portable and must not be enabled unless
a separate contract defines that policy.

The `instanceable` flag remains the source-visible field opinion with fallback
`false`. A populated prim can claim `instance=true` only when it is marked
instanceable and the consumed composed namespace source supplies supported
composition-arc provenance. Matching shared-representation IDs include both the
supported arc provenance and the relative population-mask context, because the
spec notes that masks can prevent otherwise matching instances from sharing.
When a matching shared representation has already been populated, the later
instance root remains populated but duplicate descendants are not added to
stage traversal.

Attribute and relationship field maps are exposed as source-visible authored
or schema-defined fields from the consumed namespace source. Do not resolve
attribute values across opinions or time; that belongs to a value-resolution
skill. Those field maps still preserve the typed
inspection and checked-read semantics supplied by the namespace source and
document model; stage population must not degrade them into adapter JSON.

## Boundary Guards

Consume `NamespaceSource`. Do not read parser-local USDA structures or assume
the source is a document-model layer. If an implementation bulk-enumerates a
source for performance, the semantic boundary is still the namespace-source
contract.

Consume `usd-paths` for path identity, parent/child construction, prefix checks
for masks, and path element ordering. Do not implement a parallel string-only
path model.

Consume `usd-tokens` or equivalent token identity for names, field keys,
specifier/typeName tokens, and ordering fields.

Stage indexes must retain paths, property names, child names, typeName/kind
tokens, and field keys as semantic handles. String paths and names may be
materialized for adapter summaries, diagnostics, and reports, but they must not
be the authoritative values used for stage lookup or traversal.

Use a private builder or equivalent construction-only path for internal
population mutation. The generated public surface may expose const queries and
const views, but it must not expose mutable ownership of path-to-prim maps,
path-to-property maps, traversal arrays, property arrays, or ordering caches.

Do not perform composition arc evaluation, resource loading, asset resolution,
schema fallback evaluation, value interpolation, or value clips. If a
schema-aware namespace source supplies schema-defined properties, stage
population may populate those properties, but it must not compute prim
definitions itself.

For instancing, consume graph-owned composition provenance or
instance-filtered descendant inputs supplied by the composed namespace source.
Do not infer instance matching from OpenUSD behavior, parser-local syntax, or
same-looking local child topology. Do not treat local descendant opinions under
an instanceable prim as instance source opinions. Do not add implementation-specific
prototype query APIs unless a later contract cites a pinned AOUSD
requirement for them.

Do not reapply single-layer `primOrder` or `propertyOrder` rules to a source
that declares composed child/property ordering. Those rules belong to the
composition provider for composed sources.

Do not present the resulting Stage as full USD stage conformance when the
consumed namespace source declares composition or value resolution absent.

## Performance Model

`contracts/performance/stage-population.performance.json` defines the initial
measurement model. The important shape is that construction tracks populated
paths and exposed properties, lookup does not scan traversal results, ordering
is cached or bounded per prim/property list, and the populated stage does not
retain adapter JSON or duplicate full path strings unnecessarily.

## Test Obligations

- hierarchy traversal from source child names
- `primOrder` and `propertyOrder`
- default prim lookup
- population masks include ancestors and exclude unrelated branches
- active=false prunes descendants
- path lookup and parent/child queries
- query flags for active, loaded, defined, abstract, and instanceable
- model hierarchy traversal and summaries for portable `kind` values
- instance summaries, shared representation IDs, population-mask separation,
  and descendant filtering for supported non-local composition arcs
- property summaries for attributes and relationships
- preservation of schema-defined properties when supplied by a schema-aware
  namespace source
- diagnostics passthrough from the source
- all cases in `goldens/unit/usd-stage-population/stage-population.json`
- all cases in
  `goldens/integration/usd-instancing-model-stage-population/instancing-model-stage-population.json`
