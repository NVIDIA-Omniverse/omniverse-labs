# `usdc-value-decoder` — Implementation Scope

This doc enumerates exactly what the regen agent must produce for the
`usdc-value-decoder` C++ unit to satisfy its contracts and unit goldens. It
exists because the current `generated/cpp/usdc_values.{h,cpp}` is empty
modulo a vestigial enum, and the adapter `usdc_value_decoder_adapter.cpp` is
the 111-byte stub that emits `{"ok":true,"diagnostics":[],"results":[]}`.

## Contracts that constrain the unit

| Source | Path | What it requires |
|---|---|---|
| Handle | `contracts/handles/usdc-value-decoder.handle.json` | 5 declared operations (`decode_value_representation`, `decode_scalar`, `decode_vector`, `decode_dictionary`, `decode_path_listop`); model entries `ValueRepresentationWord`, `OffsetValueView`; 9 laws |
| Production | `contracts/usdc-productions/crate-value-representations.contract.json` | Full 8-byte word layout, flag combinations, type table, inlined/offset/array/compressed dispatch, dictionary with negative offsets, half-precision (1+5+10), feature gating |
| Diagnostics | `contracts/usdc-productions/crate-reader-diagnostics.contract.json` | `MalformedUsdcValueRepresentation`, `UnsupportedUsdcFeature` |
| Performance | `contracts/performance/usdc.performance.json` | `decode_value_representation`: O(1) word read + type/flags dispatch |
| Unit golden | `goldens/unit/usdc-value-decoder/value-decoder.json` | 16 cases (see breakdown below) |

## What the unit golden exercises (16 cases → required code paths)

For each case the adapter must dispatch a `decode_value_representation` op
with these inputs and emit the canonical tagged value the contract specifies.

| Case id | Type code | Flags | Required code path |
|---|---|---|---|
| inlined-scalar-bool-true | 1 (bool) | Inlined | inlined bool from payload byte 0 |
| inlined-scalar-int-and-float | 3 (int), 8 (float) | Inlined | inlined int32; inlined float via uint32→float bit_cast |
| inlined-token-by-index | 11 (token) | Inlined | uint32 token index → tokens table lookup |
| specifier-and-variability-inlined | 42, 44 | Inlined | enum-value mapping for specifier (def/over/class), variability (uniform/varying) |
| offset-backed-double | 9 (double) | none | 8-byte IEEE 754 double at file offset |
| offset-backed-double3-vector | 23 (double3) | none | 3 contiguous doubles at file offset |
| offset-backed-token-array | 11 (token) | Array | uint64 count + uint32 token indices at file offset; tokens table lookup |
| dictionary-with-negative-offset | 31 (dictionary) | none | uint64 count + (uint32 key_token + int64 value_offset) entries; signed offset; recursive value-rep decoding |
| inlined-plus-array-flag-rejected | any | Inlined+Array | early diagnostic `MalformedUsdcValueRepresentation` |
| inlined-plus-compressed-rejected | any | Inlined+Compressed | same |
| compressed-flag-on-quaternion-rejected | 17 (quatf) | Array+Compressed | reject (quats may not be compressed) |
| path-list-op-relationship-targets | 34 (PathListOp) | none | header bitmask + 6 ordered subfield arrays (add explicit / add / prepend / append / delete / reorder); paths table lookup |
| value-block-sentinel-inlined | 51 (ValueBlock) | Inlined | sentinel; canonical `{type: valueBlock, value: null}` |
| half-vec2h-inlined | 21 (half2) | Inlined | two 16-bit Imath halfs from payload bytes 0-3 |
| path-expression-deferred-v0.10 | 57 (PathExpression) | any | when version.minor < 10 → `UnsupportedUsdcFeature` |
| splines-deferred-v0.12 | 59 (Splines) | any | when version.minor < 12 → `UnsupportedUsdcFeature` |

## Required C++ surface (target: `generated/cpp/usdc_values.{h,cpp}`)

### Public header

Namespace `usdsg::usdc_values`.

A **tagged value type** that the parser stores and the adapter renders to
canonical JSON. The contract requires "the implementation must not collapse
to JSON before the document-model boundary". One viable shape:

```cpp
struct DecodedValue {
  // Tag matches the spec's type-table entry (or a sentinel for valueBlock).
  std::string type;          // "bool", "int", "float", "double", "token", "double3",
                             // "token[]", "specifier", "variability", "valueBlock",
                             // "dictionary", "half2", "listop<ObjectPath>", ...
  // One-of payload (use std::variant or a tagged union per target style):
  bool bool_v = false;
  std::int64_t int_v = 0;
  double double_v = 0.0;
  std::string token_v;       // resolved through tokens table; canonical text
  std::vector<double> vec_v; // for vectors / vec-arrays
  std::vector<std::string> token_array_v;
  std::map<std::string, DecodedValue> dict_v;
  // PathListOp: 6 ordered subfields (explicit, add, prepend, append, delete, reorder).
  // Storage shape mirrors usd-listops-authored.handle.json.
  struct PathListOp {
    std::vector<std::string> explicit_;
    std::vector<std::string> prepend;
    std::vector<std::string> append;
    std::vector<std::string> delete_;
  } path_listop_v;
};
```

A **tables view** the decoder consumes:

```cpp
struct ValueTables {
  std::span<const std::string> tokens;   // token table (index 0 = sentinel ;-) )
  std::span<const std::string> strings;  // STRINGS-resolved strings (indices into tokens)
  std::span<const std::string> paths;    // PATHS-reconstructed canonical path strings
};
```

A **version triple** the decoder receives for feature gating:

```cpp
struct CrateVersion { unsigned major, minor, patch; };
```

A **decode result** that the contract requires to distinguish "decoded value"
from "diagnostic":

```cpp
struct DecodeResult {
  bool ok = false;
  DecodedValue value;
  std::string diagnostic_code;   // empty when ok
  std::string diagnostic_message; // empty when ok
};
```

The **primary entry point**:

```cpp
DecodeResult decode_value_representation(
    std::uint64_t word,
    std::span<const std::uint8_t> payload,    // bytes of the opened payload; used for offset-backed values
    const ValueTables& tables,
    CrateVersion version);
```

The word is read by the caller (or by a thin overload taking
`std::span<const std::uint8_t>` of 8 bytes) and dispatched here. Internally
this function:
1. Decomposes the word: `type = byte[6]`, `flags = byte[7]`, `payload = word & 0x0000_FFFF_FFFF_FFFF`.
2. Validates flag combinations:
   - `Inlined+Array`, `Inlined+Compressed`, `Inlined+Array+Compressed` ⇒ `MalformedUsdcValueRepresentation`.
   - `Compressed` on quat*/matrix*/dictionary/listop ⇒ same diagnostic.
3. Feature-gates type codes 57 (PathExpression) / 58 (Relocates) / 59 (Splines) against `version.minor`. Emit `UnsupportedUsdcFeature` when out-of-range.
4. Dispatches by type code into one of the per-type decoders below.

### Per-type sub-decoders (internal anonymous namespace)

These are the routines `decode_value_representation` calls. They are the
testable hot path. Each is small and obvious from the byte layout.

| Function | Signature | Notes |
|---|---|---|
| `decode_inlined_scalar_bool` | `(u32) → DecodedValue` | nonzero ⇒ true |
| `decode_inlined_scalar_int` | `(u32) → DecodedValue` | reinterpret_cast / bit_cast to int32 |
| `decode_inlined_scalar_uint` | `(u32) → DecodedValue` | direct |
| `decode_inlined_scalar_float` | `(u32) → DecodedValue` | std::bit_cast<float>(u32) |
| `decode_inlined_token_by_index` | `(u32, tokens) → DecodedValue` | resolves through table; bounds-check; emit diagnostic on OOB |
| `decode_inlined_specifier` | `(u32) → DecodedValue` | 0=def, 1=over, 2=class |
| `decode_inlined_variability` | `(u32) → DecodedValue` | 0=varying, 1=uniform (or vice versa per spec) |
| `decode_inlined_value_block` | `() → DecodedValue` | always sentinel |
| `decode_inlined_half2` | `(u32) → DecodedValue` | two 16-bit halfs in payload bytes 0-3 |
| `decode_offset_scalar_double` | `(payload, offset) → DecodedValue` | 8 bytes IEEE 754 |
| `decode_offset_scalar_int64` | `(payload, offset) → DecodedValue` | |
| `decode_offset_vector_double3` | `(payload, offset) → DecodedValue` | 3 doubles contiguous |
| `decode_offset_array_token` | `(payload, offset, tokens) → DecodedValue` | uint64 count + N×uint32 indices; tokens lookup per element |
| `decode_offset_dictionary` | `(payload, offset, tokens, version) → DecodedValue` | uint64 count + (u32 key_token + i64 value_offset) entries; recurses into value reps |
| `decode_offset_path_listop` | `(payload, offset, paths) → DecodedValue` | 1B subfield bitmask, then per-set-bit a uint64 count + N×uint32 path indices, in order: AddExplicit/Add/Prepend/Append/Delete/Reorder |
| `decode_half` | `(u16) → double` | Imath 1+5+10 with subnormals, ±0, ±∞, NaN |

For the goldens' scope, decoders for the following can be **deferred** with a
clear `UnsupportedUsdcFeature` (or returned as not-yet-implemented diagnostic
for value types declared in the contract but not yet exercised by a golden):
matrices, quaternions other than quatf, vectors other than double3/half2,
string/asset by-index, ReferenceListOp, PayloadListOp, integer ListOps,
TimeSamples, Splines, Relocates, PathExpression.

The unit golden's `inlined-plus-array-flag-rejected` and
`compressed-flag-on-quaternion-rejected` cases force the flag-validation
code path to exist even though no positive quat decode is exercised.

### Adapter (target: `generated/cpp/usdc_value_decoder_adapter.cpp`)

The adapter reads the golden case input from `argv[1]`, dispatches each op,
and emits the result envelope. Structure:

```cpp
int main(int argc, char** argv) {
  // 1. Read input JSON from argv[1]
  // 2. For each op in input.ops:
  //    a. Parse word_hex into uint64
  //    b. Build ValueTables from inline tables.{tokens, paths, strings}
  //    c. If payload_bytes_at_offset_hex present, splice bytes into a synthetic payload at payload_offset
  //    d. Build CrateVersion from version (default {0,8,0})
  //    e. Call decode_value_representation
  //    f. Convert DecodeResult into canonical tagged JSON
  // 3. Emit {"ok": all_ok, "diagnostics": [...], "results": [...]} to stdout
}
```

The adapter uses a tiny inline JSON reader (the rest of the repo's adapters
avoid taking a JSON dependency). For the case shapes in the unit golden, a
hand-rolled parser of ~150 LOC suffices since the input grammar is fixed.

## Out-of-scope for this unit (deferred to other units)

- Section table / TOKENS / STRINGS / PATHS construction → owned by `usdc-spec-parser` via `usdc-binary-format`.
- LZ4 buffer decoding and compressed integer array decoding → owned by `usdc-binary-format`.
- Document-model spec construction and dump emission → owned by `usdc-spec-parser` + `usd-document-model`.
- listOp item type for token/string/int variants → defer to `usd-listops-authored` once `usdc-value-decoder` can emit the matching tagged JSON.

## Implementation size estimate

| File | Approx LOC |
|---|---|
| `generated/cpp/usdc_values.h` | ~80 (DecodedValue, ValueTables, CrateVersion, DecodeResult, primary entry point) |
| `generated/cpp/usdc_values.cpp` | ~350 (16 per-type decoders + flag/version validation + half-precision conversion) |
| `generated/cpp/usdc_value_decoder_adapter.cpp` | ~250 (tiny JSON reader + op dispatch + canonical JSON serializer) |
| **Total** | **~680 LOC** |

This is comparable to the size of `usd_document_model.cpp` after its recent
expansion (the dump builder is ~130 LOC of pure serialization plus structured
field helpers). The decoder is broader in surface but each piece is a small
direct translation of the spec.

## Acceptance signal

The unit is complete when:

1. `python3 harness/regen_graph.py --scope usdc-value-decoder-unit --target cpp --validate` passes — all 16 cases in `goldens/unit/usdc-value-decoder/value-decoder.json` match expected output.
2. `harness/contract_lint.py` reports no new violations (this unit has no specific forbidden patterns today, but the canned-literal rule applies to the adapter's stdout too).
3. The 4 USDC integration goldens in `goldens/integration/usdc-single-layer/basic.json` still pass after `usdc-spec-parser` regenerates to consume the real value-decoder (rather than the current `bytes.size()` switch cheat).

## Why this can be cheated today (and why the unit golden closes the cheat)

The current `usdc_values.{h,cpp}` is empty because the spec-parser layer
above never invokes it — `usdc_parser.cpp:106-112` dispatches by file-size
into canned `*_layer()` builders that fabricate the dump from
hand-authored fields. The value-decoder unit golden runs the **adapter
directly**, not through the spec-parser; the stub adapter returns
`results: []` for every case, which fails all 16 expected-output diffs the
moment the harness can actually execute the binary (which the exec-bit fix
in the same batch enables).
