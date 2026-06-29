# Stage Population Contract

The stage population work is split into two graph nodes:

- `usd-single-layer-namespace-source`
- `usd-stage-population`

This avoids making population a single-layer dead end. Stage population consumes
a generic `NamespaceSource` capability. The first provider wraps one
document-model `Layer`; later composition skills can provide the same capability
after evaluating sublayers, references, payloads, variants, inherits,
specializes, relocates, and load state.

## Namespace Source

`contracts/capabilities/namespace-source.json` defines the reusable provider
boundary. It exposes path-addressed scene-object inputs, child names, property
names, field opinions, source diagnostics, and a capability declaration.

`contracts/handles/single-layer-namespace-source.handle.json` is the first
provider. Its capability declaration is deliberately limited:

- `mode`: `single_layer`
- `composition`: `absent`
- `value_resolution`: `absent`
- `payload_loading`: `absent`
- `instancing`: `absent`
- `schema_fallbacks`: `absent`

That provider exposes authored document-model fields. It does not prune
inactive prims, apply masks, compose listOps, inject schema fallback
properties, or resolve values. Exposed field opinions remain typed
document-model values; JSON dumps are adapter output, not the namespace-source
or stage-population field representation.

## Stage Population

`contracts/handles/stage-population.handle.json` owns the reusable stage query
surface. It populates a target-native stage from any namespace source and owns:

- root scene object creation
- population mask filtering
- active descendant pruning
- prim traversal order
- property traversal order
- path-to-prim lookup
- parent/child queries
- default prim lookup
- basic query flags
- portable model hierarchy traversal from resolved `kind`
- bounded instance summaries and shared representation IDs
- diagnostics passthrough
- property population for schema-defined properties when those properties are
  already supplied by the consumed namespace source

The populated stage should be read-only to callers. Implementations may use
mutable builders internally, but public APIs should expose only const queries or
const views of populated prims, properties, traversal, and lookup results.

Namespace sources declare whether their child/property name lists are raw
single-layer lists or already composition-ordered lists. Stage population orders
raw single-layer lists itself, but preserves provider order for composed
sources.

For the first source, traversal is still meaningful but scoped: it is a stage
view over one layer with no composition. When composition providers arrive, the
same population skill should consume their richer namespace source.

## Instancing And Models

The stage population contract now covers the portable model hierarchy rules for
`group`, `assembly`, `component`, and `subcomponent`. `component` prims are
included in the model hierarchy but terminate model traversal, while
`subcomponent`, empty, and unknown kind values are excluded and terminate
continuity. Custom aliases and validation diagnostics remain implementation
dependent.

For instancing, stage population consumes graph-owned composition provenance
from a composed namespace source. It can assign a shared representation ID when
two instanceable prims have matching supported composition-arc provenance and
compatible population-mask context. Local descendant opinions under supported
instance roots are excluded when the composition source supplies
instance-filtered descendant inputs for non-local composition arc families.

## Deferred Work

The current contract does not implement value resolution, schema fallback
evaluation, payload load policy, implementation-specific prototype query APIs,
non-portable model hierarchy aliases, or a nanousd C API backend.
Schema-defined properties may be populated only when a schema-aware namespace
source has already supplied them.

The performance contract in
`contracts/performance/stage-population.performance.json` defines the target
shape for later adapters: construction should scale with populated prims and
properties, lookup should use an index, ordering should not be recomputed from
raw strings on every query, and the stage should not retain adapter JSON as its
native representation.

## Planned Validation

The planned unit scopes are:

```powershell
py harness/regen_graph.py --scope usd-single-layer-namespace-source-unit --target python --validate
py harness/regen_graph.py --scope usd-stage-population-unit --target python --validate
```

The same scopes should be available for C++ once generated artifacts exist.
