---
name: usdc-spec-parser
description: Use this skill when implementing or verifying mapping from USDC Crate sections into the shared USD document model.
metadata:
  author: NVIDIA
---

# usdc-spec-parser

Use this skill when implementing or verifying mapping from USDC Crate sections
into the shared USD document model.

## Spec Sources

- USD Core Spec Core File Formats section `Binary`
- USD Core Spec Document Data Model sections for layer, prim, property, and
  field specs
- `harness/dump_contract.md`

Pinned commit: `6b4f01f2d6d46b01507481524eb516b4ef358c31`

## Provides

- USDC section traversal for `TOKENS`, `STRINGS`, `FIELDS`, `FIELDSETS`,
  `PATHS`, and `SPECS`
- Crate spec form mapping to document-model spec kinds
- Authored field attachment by field name
- Child-list field maintenance consistent with parsed specs
- Duplicate spec path rejection
- Target-native document-model Layer construction

## Contract

This skill accepts decoded Crate sections and value tables, constructs the
shared `usd-document-model` layer representation, and returns a target-native
parser result. Canonical dump JSON is produced only by explicit adapters,
diagnostics serialization, and dump commands.

Every emitted spec must be stored through `usd-document-model`. Every namespace
path must be represented through `usd-paths`. Values must come from
`usdc-value-decoder`; this parser does not own value representation decoding.

Production-family obligations are factored into two layers:

**Format-neutral document-model invariants** (shared with `usda-spec-parser`):

- `contracts/document-model-productions/prim-spec.contract.json`
- `contracts/document-model-productions/attribute-spec.contract.json`
- `contracts/document-model-productions/relationship-spec.contract.json`
- `contracts/document-model-productions/variant-spec.contract.json`
- `contracts/document-model-productions/layer-spec.contract.json`
- `contracts/document-model-productions/common-metadata.contract.json`

**USDC-specific mapping productions** (Crate spec form → document-model spec
kind, FIELDS token name → field, Crate value-rep type → field value):

- `contracts/usdc-productions/crate-section-layouts.contract.json`
- `contracts/usdc-productions/crate-reader-diagnostics.contract.json`
- `contracts/usdc-productions/prim-spec-mapping.contract.json`
- `contracts/usdc-productions/attribute-spec-mapping.contract.json`
- `contracts/usdc-productions/relationship-spec-mapping.contract.json`
- `contracts/usdc-productions/variant-spec-mapping.contract.json`
- `contracts/usdc-productions/layer-spec-mapping.contract.json`
- `contracts/usdc-productions/parser-diagnostics-mapping.contract.json`

The format-neutral contracts pin invariants that hold regardless of the source
format (e.g. "relationship specs reject the variability field", "the layer spec
carries primChildren", "variant specs report specifier=over"). The mapping
contracts bind those invariants to Crate-specific encoding (spec form ids,
FIELDS token names, value-representation type ids, version gating).

Field-name binding must come from the TOKENS table via FIELDS resolution.
Unlike USDA, Crate does not author alias spellings — the writer emits canonical
field tokens (`documentation`, `inheritPaths`, `variantSetNames`,
`variantSelection`, `relocates`) directly. A USDC parser that resolves alias
spellings (`doc`, `inherits`, `variantSets`, `variants`, `layerRelocates`) is in
violation: those are USDA grammar concerns and do not appear in Crate FIELDS.

Field-admission enforcement: when a FIELDS token binds to a SPECS entry, the
parser must check the field name against the spec-kind admission matrix from
`document-model-productions/common-metadata.contract.json#spec_kind_field_admission`.
A `variability` token on a form-8 (Relationship) spec, a `payload` token on a
form-1 (Attribute) spec, or a `targetPaths` token on a form-7 (Layer) spec is a
contract violation that must emit `MalformedUsdcSection`, which reduces to
`ForbiddenFieldOnSpecKind` at the layer-open dispatch boundary per
`contracts/usdc-productions/parser-diagnostics-mapping.contract.json#code_reduction_table`.

## Boundary Guards

Defer byte-level Crate decoding to `usdc-binary-format`.

Defer value representation decoding to `usdc-value-decoder`.

Defer path grammar and path identity to `usd-paths`.

Defer storage and duplicate path policy to `usd-document-model`.

Do not call USDA text lexers, USDA value parsers, or USDA spec parsers.

Do not implement sublayer loading, references, payloads, inherits,
specializes, variants, composition, asset resolution, value resolution, schema
fallback, clips, package opening, or USDC writing.

## Test Obligations

### Baseline (covered by `goldens/integration/usdc-single-layer/basic.json`)

- parse an empty layer spec
- parse one root prim with `specifier=def` and `typeName=Xform`
- parse nested prims and scalar/vector attribute defaults
- parse relationship target listOps (explicit form)

### Document-model invariants (covered by the new mirrored golden suites)

Each obligation references the format-neutral production whose laws are being
exercised, plus the integration golden suite that covers it. Suite files marked
**pending** require fixtures from Phase D of the higher-level rule-parity plan
to land before scoring can run end-to-end; the goldens themselves are authored.

- **Prim specs** (`document-model-productions/prim-spec.contract.json`):
  - all three specifiers (def, over, class) with and without typeName —
    `goldens/integration/usdc-single-layer/basic.json` (covers def);
    `goldens/integration/usdc-single-layer/prim-metadata-composition.json` (pending — over, class)
  - primChildren, properties, variantSetChildren materialized on every prim spec
  - references / payload / inheritPaths / specializes with asset, prim path,
    LayerOffset round-trip — `prim-metadata-composition.json` (pending)
  - variantSetNames (listOp<token>) distinct from variantSetChildren (spec storage)
  - kind, active, hidden, instanceable authored as scalars

- **Attribute specs** (`document-model-productions/attribute-spec.contract.json`):
  - custom flag preserved
  - uniform variability authored (not just the default varying) —
    `goldens/integration/usdc-single-layer/value-syntax.json` (pending)
  - typeName + default agreement across all foundational scalar / vector /
    matrix / quaternion types — `value-syntax.json` (pending)
  - ValueBlock default decodes to a value-block sentinel
  - TimeSamples authored on an attribute with numeric keys and value-block
    samples — `goldens/integration/usdc-single-layer/attribute-time-samples.json` (pending)
  - connectionPaths explicit / prepend / append / delete subfields —
    `goldens/integration/usdc-single-layer/property-listop-dispatch.json` (pending)
  - allowedTokens authored as TokenVector
  - Splines (Crate value type 59) on a Crate minor < 12 fixture emits
    UnsupportedUsdcFeature (no positive splines decoding in this scope)
  - any prim-only / layer-only / relationship-only field name bound to a form-1
    spec emits ForbiddenFieldOnSpecKind

- **Relationship specs** (`document-model-productions/relationship-spec.contract.json`):
  - custom rel declaration preserved
  - targetPaths explicit / prepend / append / delete subfield decoding —
    `goldens/integration/usdc-single-layer/relationship-listops.json` (pending)
  - single-item explicit listOp preserved as such (not collapsed to bare path)
  - None target as empty explicit listOp
  - noLoadHint authored
  - variability bound to a form-8 spec emits ForbiddenFieldOnSpecKind

- **Variant + VariantSet specs** (`document-model-productions/variant-spec.contract.json`):
  - single variantSet with one variant materialized through forms 10/11 —
    `goldens/integration/usdc-single-layer/variant-specs.json` (pending)
  - multiple variants under one variantSet populate variantChildren
  - nested variantSet inside a variant
  - variant specs report specifier=over with unauthored typeName
  - variantSelection (Crate value type 45 VariantSelectionMap) decoded as
    native map with empty-string variant selections preserved
  - parent-kind validation: a form-11 spec parented to anything other than a
    prim or variant emits MalformedUsdcSection

- **Layer spec** (`document-model-productions/layer-spec.contract.json`):
  - subLayers + subLayerOffsets length-matched, Retiming records native —
    `goldens/integration/usdc-single-layer/layer-metadata.json` (pending)
  - subLayers without subLayerOffsets defaults to per-item Retiming(0.0, 1.0)
  - relocates (Crate value type 58) on a v0.11+ fixture; UnsupportedUsdcFeature
    on a v0.8 fixture
  - defaultPrim authored as token
  - customLayerData / documentation / framesPerSecond / timeCodesPerSecond /
    startTimeCode / endTimeCode authored
  - prim-only / attribute-only / relationship-only field names bound to a
    form-7 spec emit ForbiddenFieldOnSpecKind

- **Common metadata** (`document-model-productions/common-metadata.contract.json`):
  - generic metadata (comment, documentation, displayName, customData) on each
    spec kind — `goldens/integration/usdc-single-layer/common-metadata.json` (pending)
  - listOp metadata uses {explicit, prepend, append, delete, reorder}
    subfields stored independently
  - within a single source layer, repeated listOp subfields use last-one-wins
    per subfield
  - alias spellings (doc, inherits, variantSets, variants, layerRelocates) must
    NOT appear in Crate FIELDS; the parser binds canonical token names directly

### Diagnostics (covered by `goldens/integration/usdc-single-layer/parser-diagnostics.json` — pending)

Each USDC-specific code reduces to a format-neutral code at the layer-open
dispatch boundary per
`contracts/usdc-productions/parser-diagnostics-mapping.contract.json#code_reduction_table`:

- non-PXR-USDC header → InvalidUsdcHeader → InvalidLayerHeader
- unsupported version triple → UnsupportedUsdcVersion → UnsupportedLayerVersion
- spec form outside the normative+compatibility set → MalformedUsdcSpecForm
- duplicate spec path → DuplicateSpecPath (already format-neutral)
- forbidden field on spec kind → MalformedUsdcSection → ForbiddenFieldOnSpecKind
- value-rep type id disagrees with attribute typeName → MalformedUsdcValueRepresentation → TypeMismatch
- value type introduced after the Crate minor version → UnsupportedUsdcFeature → UnsupportedFeature

### Structural integrity (covered by all goldens)

- keep child lists and property lists in agreement with emitted specs
- reject duplicate spec paths with the canonical diagnostic
- reconstruct every path by executing the AOUSD spec
  PathIndices/ElementTokenIndices/Jumps algorithm (16.3.8.4.5.4); deriving any
  path from substring matches on opened resource bytes is a contract violation
- resolve fields by index into FIELDS and FIELDSETS; substring or
  section-content pattern matches are a contract violation
- keep canonical dump emission in adapters or dump commands over
  `usd-document-model`; returning canned dump literals selected by section
  bytes or fixture name is a contract violation
- accept a fixture with two sibling prims whose names are substring-related
  (e.g. `World` and `WorldHelper`) so a substring-based parser produces an
  observably wrong dump
- accept a fixture where two attributes share a typeName token (e.g. two `float`
  attributes) so token-table reuse is exercised
- the canonical dump emitted from a `.usdc` fixture must be byte-for-byte
  identical (after canonicalization) to the canonical dump emitted from the
  equivalent `.usda` fixture; this is checked via the `cross_check` step on
  the `usdc-single-layer` scope

## Performance

The spec parser unit must satisfy
`contracts/performance/usdc.performance.json`. Layer build is
O(specs + total authored field count); path reconstruction is O(path_count)
via the Crate index algorithm. Dump emission performance is measured only at
explicit adapter and dump-command boundaries.
