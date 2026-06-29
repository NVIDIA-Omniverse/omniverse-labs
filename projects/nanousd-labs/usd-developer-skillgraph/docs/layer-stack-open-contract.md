# Layer Stack Open Contract

`usd-layer-stack-open` is the first contract that reads actual sublayer
resources.

It bridges two existing pieces:

- `usd-layer-open`: opens one resolved layer resource into a document-model
  `Layer`
- `usd-layer-stack-namespace-source`: composes an in-memory root layer plus
  already-opened recursive sublayers into a `NamespaceSource`

The node opens the root layer, reads its authored `subLayers`, resolves those
asset identifiers with a bounded local policy, opens sublayers recursively, and
hands the opened layers to the layer-stack namespace source. Recursive entries
are ordered depth-first at the authored subLayer site before later sibling
sublayers.

## Local Policy

This is not full USD asset resolution. The first loader resolves local relative
sublayer identifiers against the containing directory of the layer that authored
the `subLayer`. Both `asset.usda` and `./asset.usda` are treated as local
resources relative to that authoring layer for this prototype.

Unsupported URI schemes fail instead of falling back to filesystem paths.
Cycles in the recursive stack are diagnosed by resolved layer location and do
not recurse unboundedly. Package sublayers, resolver contexts, search paths,
and remote protocols are left for later contracts.

## Boundary

The loader owns resource orchestration, not composition:

- it must use `usd-layer-open` for root and recursively discovered sublayers
- it must not parse USDA directly
- it must not reimplement child/property merge rules
- it must construct `usd-layer-stack-namespace-source` with opened layers

The output is a native layer-stack-open result containing either a
`LayerStackNamespaceSource` or diagnostics. JSON is only for handle adapters.

## Planned Validation

```powershell
py harness/regen_graph.py --scope usd-layer-stack-open-unit --target python
py harness/regen_graph.py --scope usd-layer-stack-open-unit --target cpp
```
