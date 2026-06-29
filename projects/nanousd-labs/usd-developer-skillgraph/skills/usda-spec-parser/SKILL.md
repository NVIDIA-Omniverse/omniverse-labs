---
name: usda-spec-parser
description: Use this skill when implementing or verifying the USDA single-layer parser.
metadata:
  author: NVIDIA
---

# usda-spec-parser

Use this skill when implementing or verifying the USDA single-layer parser.

## Spec Sources

- `specification/file_formats/README.md` sections `Layer`, `Prim Specs`,
  `Attribute Specs`, `Relationship Specs`, `Common Metadata`, and variant
  statements
- `specification/document_data_model/README.md`

Pinned tag / commit: `v1.0.1`

The pinned specification is normative. This skill, its contracts, and its
goldens exist to decompose that specification and add representation,
diagnostic, API, and performance constraints needed for a usable generated
library. They must not relax the grammar or make document-model storage fields
stand in for grammar productions.

## Provides

- Parse a USDA layer into the document model
- Map layer metadata to layer fields
- Map standalone layer doc strings to the layer documentation field
- Map prim declarations to prim specs
- Map authored prim metadata, including references, payloads, inherits,
  specializes, variants, and variantSets, to stored fields without evaluating
  composition
- Map attribute declarations and values to attribute specs
- Map relationship declarations and targets to relationship specs
- Map variant set and variant declarations to stored specs
- Map sublayer and relocate metadata to stored layer fields without loading or
  applying dependent layers
- Maintain child list fields
- Map `reorder rootPrims`, `reorder nameChildren`, and `reorder properties`
- Return the populated document-model `Layer`
- Support canonical layer dump JSON only through an adapter boundary

## Contract

This skill owns `contracts/handles/usda-spec-parser.handle.json` and provides
the `usda_spec_parser` capability consumed by USDA layer opening. It consumes
lexical parsing, value parsing, document-model storage, and path construction
from graph dependencies instead of owning alternate versions of those systems.
It also follows `contracts/capabilities/semantic-runtime-types.json`.

Production-family obligations are factored into:

- `contracts/spec-coverage/usda-single-layer.coverage.json`
- `contracts/usda-productions/common-metadata.contract.json`
- `contracts/usda-productions/layer-structure.contract.json`
- `contracts/usda-productions/layer-metadata.contract.json`
- `contracts/usda-productions/prim-specs.contract.json`
- `contracts/usda-productions/attribute-specs.contract.json`
- `contracts/usda-productions/relationship-specs.contract.json`
- `contracts/usda-productions/variant-specs.contract.json`
- `contracts/usda-productions/parser-diagnostics.contract.json`

The parser accepts one USDA file and emits the canonical dump described in
`harness/dump_contract.md` only through an adapter. The domain parser result is
a target-native result containing either a populated document-model `Layer` or
diagnostics. It must not be a `Json`, Python `dict`, tagged JSON, or canonical
dump tree.

Parsing is grammar-first and storage-second. Match a production valid for the
current parser context, then map the result to document-model fields. Do not
parse metadata by accepting an arbitrary key, looking up a core field token, and
then deciding what value grammar to use. Field names are storage labels; they do
not authorize syntax.

Every created spec must be stored through `usd-document-model`. All spec paths
must be constructed by `usd-paths`; do not keep a parser-local path type.
Field keys, child/property names, type names, variant set names, and selected
variant names that flow into document-model or composition-facing records must
be converted to target-native token/name identity before durable storage.

JSON materialization is allowed only when a validation adapter or dump command
serializes the returned `Layer`.

Metadata parsing helpers should either mutate the document-model `Layer`
directly or return target-native field assignments keyed by document-model field
tokens. They must not return `map<string, Json>`, Python `dict`, tagged JSON,
or any other serialized object tree. Attribute and relationship field assembly
follows the same rule: build `FieldValue`-style typed values, not adapter JSON
that is immediately converted back into storage.

Known document-model fields must use the core field tokens provided by
`usd-document-model`; do not route known names like `propertyOrder`,
`primOrder`, `payload`, `references`, `defaultPrim`, `documentation`,
`targetPaths`, or `typeName` through extension-field string lookup.

Metadata key dispatch must classify the authored USDA spelling once at the
parser boundary with the current metadata context before selecting storage or
value typing. The generic `KeyValueMetadata` and `ListOpMetadata` productions
use `Identifier`, which excludes USDA keywords. Keywords such as `payload`,
`references`, `inherits`, `specializes`, `variantSets`, `variants`, `kind`,
`subLayers`, and `relocates` are legal only through the context-specific
productions that mention them. Do not accept a prim-only keyword as generic
attribute or relationship metadata simply because the document model has a core
field token with that spelling. Only unknown non-keyword extension metadata may
use extension-field token lookup.

Layer metadata blocks may contain a standalone documentation string. Parse that
as document-model `documentation`; do not route it through the generic
identifier-based metadata entry path.

The metadata context must distinguish at least layer, prim, attribute, and
relationship metadata. A boolean `layer`/`non-layer` split is not sufficient:
it accepts prim-only keyword productions on attributes and relationships. For
variant statements, use the prim metadata production family for grammar
dispatch, then store the fields on the variant spec.

Reference and payload metadata must store `listOp<Reference>` and
`listOp<Payload>` using native arc-record items. Do not store those items as
parser dictionaries, bare asset strings, or bare path strings; convert them to
adapter JSON only during dump/validation serialization.

Schema metadata must preserve the field's specified listOp element type.
`apiSchemas` applies to prim specs and stores authored applied-schema
identifiers as `listOp<token>`. This is distinct from `variantSetNames`, which
is `listOp<string>`. The parser records these fields as inert authored data and
must not inject schema fallback properties or evaluate applied-schema semantics.

Other specialized document-model values must also stay native when parsed:
`specifier`, `variability`, `Retiming[]`, `Relocates`, `TimeSamples`, and
`Spline`. Their canonical dump shapes may contain strings and objects, but the
parser/domain boundary must pass enum/domain values, path references,
numeric-time maps, and spline records rather than plain strings or dictionary
payloads.

Layer `relocates` metadata accepts both path-to-path entries and path-to-None
entries. `None` is the deleted-relocate target sentinel for composition
providers and must not be rejected by the parser or stored as a string literal.

Property parsing must handle shared listOp prefixes explicitly. The USDA grammar
allows both `ListOpRelationship` and `ListOpConnect`; after greedily consuming
an optional listOp and optional `custom`, dispatch on the following declaration
tokens so `append rel target = ...` and `append float attr.connect = ...` are
both accepted without parser backtracking. A listOp prefix alone must not imply
the property is a relationship.

Composition fields may be parsed as inert metadata only if needed by a golden,
and the production contracts now treat their authored storage as in scope. This
skill must not implement composition semantics, but it must not drop authored
composition arc fields simply because composition itself is deferred.

## Boundary Guards

Defer identifiers to `usd-identifiers-and-names`.

Defer values to `usda-value-parser`.

Defer paths to `usd-paths`.

Defer listOps to `usd-listops-authored`.

Defer storage to `usd-document-model`.

Do not implement sublayer loading, reference/payload resolution, inherit or
specialize composition, variant selection composition, asset resolution, value
resolution, or stage population.

## Test Obligations

- all cases in `goldens/integration/usda-single-layer/basic.json`
- all production entries marked `implemented` in
  `contracts/spec-coverage/usda-single-layer.coverage.json`
- positive and negative goldens for context-sensitive grammar decisions,
  especially metadata keywords and shared-prefix productions
- parser domain API returns a document-model `Layer`, not JSON
- duplicate spec path halts processing
- last authored order field wins
- child lists reflect parsed specs
- inactive prims remain present in the layer
- authored composition arc metadata is stored as layer data without evaluation
- variant set and variant specs are stored without performing variant selection
- sublayer and relocate metadata are stored without loading or applying
  dependent layers
- standalone layer doc strings parse as documentation metadata
