---
name: usd-layer-stack-open
description: Use this skill when implementing or verifying recursive local sublayer loading from resolved layer resources into a layer-stack namespace source.
metadata:
  author: NVIDIA
---

# usd-layer-stack-open

Use this skill when implementing or verifying recursive local sublayer loading from
resolved layer resources into a layer-stack namespace source.

## Spec Sources

- `specification/composition/README.md` sublayer and layer-stack inputs
- `specification/document_data_model/README.md` subLayers and subLayerOffsets
- USD Core Spec Resource Interface sections covering resolved resources,
  local file resources, URI references, and package resource spelling

Pinned tag / commit: `v1.0.1`

## Provides

- Opening a root layer resource
- Reading `subLayers` from the opened root layer and recursively opened local sublayers
- Resolving local sublayer identifiers relative to the layer that authored them
- Opening recursively discovered local sublayer resources through `usd-layer-open`
- Detecting recursive sublayer cycles by resolved layer location
- Construction of a `LayerStackNamespaceSource` from opened layers
- Layer-stack-open diagnostics for missing, unsupported, failed, or cyclic sublayers

## Contract

This skill owns `contracts/handles/layer-stack-open.handle.json` and bridges
the single-layer opener with the in-memory layer-stack namespace source.

This is a bounded recursive local sublayer loader. It accepts a resolved root
resource location, opens it with `usd-layer-open`, reads each opened layer's
authored `subLayers` field, resolves each local sublayer using the prototype
policy below, opens recursively discovered sublayers with `usd-layer-open`, and
passes the opened layers to `usd-layer-stack-namespace-source`.

The output is a target-native result containing either a
`LayerStackNamespaceSource` or diagnostics. It must not be a canonical JSON
dump, a parser-local USDA tree, or a separate JSON spec store. JSON appears only
in conformance adapters and diagnostic serialization.

## Local Resolution Policy

This is not full USD asset resolution. It is a bounded local policy for the
first actual-sublayer contract:

- local filesystem layer resources may resolve relative sublayer identifiers
  against the containing directory of the layer that authored the subLayer
- local `file` URI roots may resolve relative sublayer identifiers against the
  containing file URI path of the authoring layer
- explicit local `file` URI sublayers may be passed through as resolved
  locations
- unsupported URI schemes fail clearly
- package resource sublayers remain unsupported in this contract
- resolver search paths, resolver callbacks, contexts, layer identifiers, and
  non-local protocols remain deferred

For this prototype, plain relative asset identifiers such as `asset.usda` and
anchored relative identifiers such as `./asset.usda` are both resolved relative
to the layer that authored the subLayer. This is a local resolver policy, not a
claim of full asset resolution.

## Boundary Guards

Consume `usd-layer-open` to open the root and every recursive sublayer. Do not read
files directly and do not invoke USDA parsers from this node.

Consume `usd-resource-protocol` for resolved-location classification and local
resource rules. Do not add ad hoc URI parsing, package splitting, or unsupported
scheme fallback to filesystem paths.

Consume `usd-layer-stack-namespace-source` to construct the composed namespace
source from opened layers. Do not duplicate child/property composition,
effective specifier rules, or field-strength merging in the loader.

Consume the document-model layer returned by `usd-layer-open`; do not retain
canonical layer dump JSON as source storage.

Do not perform references, payloads, inherits, specializes, variant composition, relocates, schema
fallback evaluation, value resolution, stage population, or instancing.

## Test Obligations

- root layer resource opens through `usd-layer-open`
- recursively discovered local sublayers open through `usd-layer-open`
- opened layers are passed to `usd-layer-stack-namespace-source`
- stack summary preserves root plus recursive sublayers and subLayerOffsets in depth-first strength order
- composed source queries work after resource loading
- missing sublayer resources fail clearly
- unsupported sublayer URI schemes fail clearly
- recursive sublayer cycles fail clearly and do not recurse unboundedly
- all cases in `goldens/unit/usd-layer-stack-open/layer-stack-open.json`
