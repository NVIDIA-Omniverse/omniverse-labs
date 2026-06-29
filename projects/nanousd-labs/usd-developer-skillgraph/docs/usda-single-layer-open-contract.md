# USDA Single-Layer Open Contract

The current topmost graph scope is `usda-single-layer`: open one resolved local
USDA text layer, parse it, store the result in the document model, and emit the
canonical single-layer dump through the validation adapter.

This deliberately stops before composition, value resolution, stage population,
schemas, USDC, and USDZ.

## Contract Stack

- `contracts/handles/layer-open.handle.json`
  - generic resource open and format dispatch
- `contracts/handles/usda-layer-open.handle.json`
  - USDA text format handler and UTF-8 decode boundary
- `contracts/handles/usda-spec-parser.handle.json`
  - USDA spec parsing into `usd-document-model`
- `contracts/spec-coverage/usda-single-layer.coverage.json`
  - spec production coverage ledger, ownership, positive/negative goldens, and
    quality-floor notes
- `contracts/handles/usda-value-parser.handle.json`
  - USDA RHS value parsing through foundational values, paths, and listOps
- `contracts/usda-productions/*.contract.json`
  - production-family obligations for values, metadata, layer structure, layer
    metadata, prims, attributes, relationships, variants, and parser
    diagnostics
- `contracts/capabilities/usda-lexical-format.json`
  - USDA text lexical substrate through the identifier scanner

`goldens/integration/usda-single-layer/basic.json` is the current integration
golden for this stack. It asserts that generated targets populate document-model
storage and return a target-native `Layer` before the dump command serializes
canonical layer/spec/field JSON.

Additional production-family goldens live beside it:

- `common-metadata.json`
- `value-syntax.json`
- `attribute-time-samples.json`
- `relationship-listops.json`
- `property-listop-dispatch.json`
- `metadata-grammar-contexts.json`
- `prim-metadata-composition.json`
- `variant-specs.json`
- `layer-metadata.json`
- `parser-diagnostics.json`

The parser, USDA layer-format handler, and generic layer opener must not return
`Json`, Python `dict`, tagged JSON, or canonical dump trees as their domain
result. JSON is only an information boundary for validation, diagnostics
serialization, and explicit dump commands.

## Scope

In scope:

- local filesystem resource opening
- USDA text header and layer content parsing
- layer metadata, prims, attributes, relationships, variants, child lists, and
  order fields
- authored storage for sublayers, relocates, references, payloads, inherits,
  specializes, variant sets, and variant selections
- duplicate spec path diagnostics

Out of scope:

- asset resolver search paths and dependent layer loading
- sublayer/reference/payload/inherits/specializes/variant composition semantics
- value resolution
- stage population
- schemas and fallback evaluation
- USDC and USDZ

Composition is out of scope, but authored opinions that would later feed
composition are still layer data. The parser contract therefore requires storing
those fields inertly on the consumed layer rather than silently discarding them.
