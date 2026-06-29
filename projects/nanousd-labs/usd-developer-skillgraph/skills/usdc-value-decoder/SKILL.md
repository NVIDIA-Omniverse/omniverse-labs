---
name: usdc-value-decoder
description: Use this skill when implementing or verifying USDC Crate value representation decoding.
metadata:
  author: NVIDIA
---

# usdc-value-decoder

Use this skill when implementing or verifying USDC Crate value representation
decoding.

## Spec Sources

- USD Core Spec Core File Formats section `Binary`
- USD Core Spec Document Data Model field and value representation sections
- `contracts/value.schema.json`
- `docs/usdc-spec-errata.md` — pinned interpretations where the AOUSD prose is
  ambiguous or diverges from real `.usdc` content. In particular: the
  Array / Inlined / Compressed flag bits live in the high bits of byte 7 of
  the 8-byte LE value-representation word (masks `0x80 / 0x40 / 0x20`); a
  decoder must not also accept the low-bit layout as a fallback.

Pinned commit: `6b4f01f2d6d46b01507481524eb516b4ef358c31`

## Provides

- Crate value representation decoding
- Token, string, and path index resolution
- Inlined scalar values
- Offset-backed scalar and array values
- Dictionaries
- Vectors used by first fixtures
- Authored listOp values
- Specifier and variability values

## Contract

This skill converts low-level Crate value representations into the existing
canonical value shape used by `contracts/value.schema.json`.

It depends on `usdc-binary-format` for byte and buffer decoding, on
`usd-tokens` for token identity, on `usd-foundational-values` for canonical
value shapes, on `usd-paths` for namespace path strings, and on
`usd-listops-authored` for listOp boundaries.

It may use decoded Crate token, string, and path tables supplied by
`usdc-spec-parser`, but it must not own Crate section traversal or document
spec storage.

Production-family obligations are factored into:

- `contracts/usdc-productions/crate-value-representations.contract.json` — wire
  format of the 8-byte value-representation word and the type table
- `contracts/usdc-productions/attribute-spec-mapping.contract.json` — shared
  ownership of TimeSamples (type 46), VariantSelectionMap (type 45), and
  LayerOffsetVector (type 49) decoding, called out as
  `co_owners: ["usdc-value-decoder"]` in that contract
- `contracts/usdc-productions/parser-diagnostics-mapping.contract.json` —
  mapping of value-level diagnostics (`MalformedUsdcValueRepresentation` →
  `TypeMismatch`, `UnsupportedUsdcFeature` → `UnsupportedFeature`) at the
  layer-open dispatch boundary
- `contracts/document-model-productions/attribute-spec.contract.json` — the
  format-neutral value-shape rules for TimeSamples (native time→FieldValue
  map; no string-keyed JSON), Spline (native Spline record; no Python dict
  payloads), default-vs-typeName agreement, and value-block sentinel semantics

## Boundary Guards

Defer byte-level bounds, endian, bootstrap, TOC, compression, and integer-array
decoding to `usdc-binary-format`.

Defer section-to-document mapping and spec creation to `usdc-spec-parser`.

Defer storage to `usd-document-model` through `usdc-spec-parser`.

Do not call USDA text lexers, USDA value parsers, or USDA spec parsers.

Do not implement composition, value resolution, schema fallback, clips, package
opening, or USDC writing.

## Test Obligations

### Baseline (covered by `goldens/unit/usdc-value-decoder/value-decoder.json`)

- decode token and string values used by layer metadata and type names
- decode `int`, `float`, `double`, and `double3` values used by attributes
- decode specifier and variability values
- decode dictionaries used by layer metadata fixtures
- decode relationship target `PathListOp` values
- preserve the canonical JSON value shape already used by USDA dumps
- decode one inlined scalar and one offset-backed scalar of the same target
  type so the inlined/offset distinction is exercised
- decode at least one offset-backed `token[]` array sourced from token indices
  in TOKENS (not inlined)

### Document-model value-shape compliance

Each obligation references the format-neutral value-shape rule in
`contracts/document-model-productions/attribute-spec.contract.json` plus the
USDC-specific mapping in
`contracts/usdc-productions/attribute-spec-mapping.contract.json`:

- **TimeSamples (Crate type 46)** — decode as a native time→FieldValue map
  keyed by numeric time ordinates. String-keyed JSON storage is a contract
  violation per `attribute-spec.contract.json#fields.timeSamples`. None-valued
  samples (Crate ValueBlock type 51) decode to value-block sentinels.
  Exercised by `goldens/integration/usdc-single-layer/attribute-time-samples.json`
  (pending Phase D fixture).
- **VariantSelectionMap (Crate type 45)** — decode the wire format (uint64
  count + alternating uint32 string-index key + uint32 string-index value) as
  a native variantSelection map; empty-string values are preserved, not
  dropped. Exercised by `goldens/integration/usdc-single-layer/variant-specs.json`
  (pending Phase D fixture).
- **LayerOffsetVector (Crate type 49)** — decode as a sequence of native
  Retiming records (offset:double, scale:double), one per subLayer item; raw
  dictionaries / JSON payloads are forbidden per
  `layer-spec.contract.json#fields.subLayerOffsets`. Exercised by
  `goldens/integration/usdc-single-layer/layer-metadata.json` (pending Phase D).
- **PathListOp (Crate type 34)** subfield semantics — explicit / prepend /
  append / delete subfields are decoded independently from the listOp bitmask
  byte; the parser must not collapse a single-item explicit listOp to a bare
  path. Used by relationship `targetPaths` and attribute `connectionPaths`;
  format-neutral rule lives in
  `common-metadata.contract.json#listop_subfields`.
- **Relocates (Crate type 58)** — at Crate minor ≥ 11, decode as a native
  source-PathRef → target-PathRef map. At minor < 11, emit
  `UnsupportedUsdcFeature` (which reduces to `UnsupportedFeature` at dump
  boundary per `parser-diagnostics-mapping.contract.json`).
- **default vs typeName agreement** — when decoding a `default` field on an
  attribute spec, the value-representation type id must match the attribute's
  typeName per the type-id table in
  `attribute-spec-mapping.contract.json#storage_mapping.default_field_type_table_reference`.
  Mismatch emits `MalformedUsdcValueRepresentation`, which reduces to
  `TypeMismatch`.

### Structural integrity (universal)

- read every value representation as a single 8-byte little-endian word and
  dispatch on the 1-byte type and 1-byte flags; multiple per-byte reads,
  per-field-name dispatch, or section byte string scans are contract
  violations
- the Array / Inlined / Compressed flag bits live in the HIGH bits of byte 7
  (masks `0x80 / 0x40 / 0x20`) per
  `crate-value-representations.contract.json#flags` and
  `docs/usdc-spec-errata.md#1`. Accepting the low-bit layout (0x01/0x02/0x04)
  as a fallback is a contract violation.
- emit a clear `UnsupportedUsdcFeature` diagnostic for value types introduced
  at Crate 0.10.0 or later (Splines, Relocates, PathExpression) when
  encountered under a lower minor; placeholder JSON is a contract violation
- must not dispatch values by field-name substrings or by section byte
  string scans
- the value-decoder source code must be a single shared implementation
  consumed by both `usdc_parser.cpp` and `usdc_value_decoder_adapter.cpp`; a
  parser-internal duplicate decoder is a contract violation enforced by the
  `no_inline_value_decoder_in_parser` lint rule in
  `contracts/lint/usdc-single-layer.lint.json`

## Performance

The value-decoder unit must satisfy
`contracts/performance/usdc.performance.json`. Value representations are read
as a single 8-byte little-endian word and dispatched in O(1) on type and
flags. Offset-backed scalar and array values are returned as non-owning views
into the opened resource byte view until canonicalization runs.
