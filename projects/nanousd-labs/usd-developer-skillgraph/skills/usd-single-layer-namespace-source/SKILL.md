---
name: usd-single-layer-namespace-source
description: Use this skill when implementing or verifying the degenerate namespace source that wraps one document-model layer for stage population.
metadata:
  author: NVIDIA
---

# usd-single-layer-namespace-source

Use this skill when implementing or verifying the degenerate namespace source
that wraps one document-model layer for stage population.

## Spec Sources

- `specification/stage_population/README.md`
- `specification/document_data_model/README.md`

Pinned tag / commit: `v1.0.1`

## Provides

- A `NamespaceSource` capability backed by exactly one document-model `Layer`
- Layer field lookup for root layer metadata
- Prim input lookup by absolute prim path
- Property input lookup by property path
- Child prim names from `primChildren`
- Property names from `properties`
- Source capability declaration as `single_layer`
- Ordering declaration as raw single-layer child/property lists
- Diagnostics passthrough from the layer/opening boundary

## Contract

This skill owns `contracts/handles/single-layer-namespace-source.handle.json`
and implements the general namespace-source capability described by
`contracts/capabilities/namespace-source.json`.

The point of this node is to avoid making `usd-stage-population` depend
directly on document-model storage. Stage population consumes a namespace view;
today that view is backed by one layer, and later a composition node can provide
the same capability after evaluating sublayers, references, payloads, variants,
inherits, specializes, relocates, and load state.

A single-layer namespace source is intentionally degenerate:

- each path has zero or one local spec opinion
- field values are authored layer values
- child and property lists come from the document-model child-list fields
- no listOp composition is performed
- no value resolution is performed
- no schema fallback properties are injected
- no inactive filtering is performed
- child and property names are raw layer lists for stage population to order

The source must declare those limits explicitly through its capability
inspection surface. A generated target must not silently use the single-layer
source as evidence that composition or value resolution is implemented.

## Boundary Guards

Consume `usd-document-model` for layer/spec/field storage. Do not duplicate the
layer as a separate map of JSON specs.

Consume `usd-paths` for all path identity and path-kind checks. Adapter string
paths may be parsed at the boundary, but target code should pass path handles or
equivalent path references.

Consume `usd-tokens` and the document-model field-token vocabulary for field
names, child names, property names, and token-valued fields. Do not introduce a
provider-local string identity domain.

Do not perform stage population. This skill exposes source inputs; it does not
create stage nodes, apply population masks, prune inactive descendants, or
build traversal/lookup indexes.

Do not perform composition, value resolution, asset resolution, schema fallback
evaluation, model hierarchy traversal, or instancing.

## Test Obligations

- source capability declaration reports `single_layer`
- root layer field lookup
- prim lookup for existing and missing paths
- property lookup for attributes and relationships
- child/property name enumeration follows document-model stored lists
- active=false is exposed as a field and not interpreted locally
- authored composition arc fields remain inert source-visible fields
- all cases in `goldens/unit/usd-namespace-source/single-layer-namespace-source.json`
