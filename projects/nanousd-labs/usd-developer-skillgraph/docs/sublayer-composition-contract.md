# Sublayer Composition Contract

The first composition node is `usd-layer-stack-namespace-source`.

It intentionally sits between document-model layers and stage population:

- input: a root `Layer` plus already-opened recursive sublayer `Layer`s
- output: a `NamespaceSource` with `mode=layer_stack`
- consumer: the existing `usd-stage-population` skill, once a stage scope is
  wired to this richer provider

This keeps the composition contract focused on layer-stack semantics instead
of asset resolution, resource loading, or file I/O.

## Included

The layer-stack provider owns:

- recursive `subLayers` interpretation across already-opened layers
- `subLayerOffsets` pairing and reporting, without retiming values
- root stronger than recursive sublayers
- recursive sublayers in depth-first authored order, strongest to weakest
- prim/property existence across contributing specs
- effective population `specifier` for stage query flags
- strongest field selection for scalar/token/bool population fields
- authored listOp field composition across contributing specs when field types
  match
- child prim name merging from weakest to strongest, applying `primOrder` after
  each contributing spec
- property name union, path-element sorting, and strongest `propertyOrder`
- strongest scalar property fields and composed authored listOp property fields
- diagnostics for missing or duplicate supplied sublayers

The provider declares `child_ordering=composed` and
`property_ordering=composed`, so stage population preserves the provider's name
order and still owns masks, active pruning, traversal, and lookup indexes.

## Deferred

This contract does not resolve asset identifiers or open files. The sublayer
loader consumes `usd-layer-open` and `usd-resource-protocol` to turn authored
`subLayers` into opened layers.

The following also remain deferred:

- relocates
- references and payloads
- inherits, specializes, and variants
- dictionary composition
- attribute value resolution and time sample interpolation
- applying `subLayerOffsets` to time-varying values
- schema fallback properties
- payload load policy and instancing

## Planned Validation

The unit scope is:

```powershell
py harness/regen_graph.py --scope usd-layer-stack-namespace-source-unit --target python
```

Once generated artifacts exist, validation should be:

```powershell
py harness/regen_graph.py --scope usd-layer-stack-namespace-source-unit --target python --validate
```

The first integration follow-up should add a stage-population scope that uses
the same `NamespaceSource` contract with the layer-stack provider.

Actual recursive sublayer resource loading is covered separately by
`docs/layer-stack-open-contract.md`.
