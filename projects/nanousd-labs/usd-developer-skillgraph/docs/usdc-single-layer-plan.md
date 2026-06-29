# USDC Single-Layer Plan

> **Status note.** This document covers the original `usdc-single-layer`
> target shape, the initial fixture set, and the C++ implementation order.
> The higher-level representation-rule parity work that brought USDC's
> document-model contracts in line with USDA's lives in
> `docs/document-model-productions.md`. New `.usdc` fixtures, golden suites,
> and document-model lint rules added during rule-parity work follow the
> layout there.

## Goal

Add a `usdc-single-layer` target that opens one resolved local USDC Crate layer
resource and emits the same canonical layer dump JSON used by
`usda-single-layer`.

Unlike the first `usda-single-layer` prototype, this should be implemented as a
C++ target first. The target should prove that binary Crate decoding can compose
with the existing resource protocol, layer-open dispatch, document model, paths,
tokens, values, and authored listOp boundaries.

The representation rules a parsed USDC layer must satisfy — what fields each
spec kind carries, how listOps subdivide, which field names are forbidden on
which spec kind, how diagnostics reduce to format-neutral codes — live in the
format-neutral document-model productions at
`contracts/document-model-productions/` and the USDC-side mapping productions
at `contracts/usdc-productions/*-mapping.contract.json`. See
`docs/document-model-productions.md` for the full contract layout.

## Non-Goals

- USDC writing
- USDZ package opening
- `.usd` forwarding format dispatch
- sublayer/reference/payload loading
- composition, value resolution, stage population, clips, and schema fallback
- full historical Crate compatibility before the AOUSD 1.0.1 supported range

## Target Shape

Add these graph nodes:

- `usdc-binary-format`
  - Owns Crate byte reader primitives.
  - Reads little-endian integers/floats, validates `PXR-USDC`, reads version,
    bootstrap, table of contents, LZ4 buffers, and compressed integer arrays.
  - Does not map decoded data into USD specs.

- `usdc-value-decoder`
  - Owns Crate value representation decoding.
  - Depends on `usdc-binary-format`, `usd-tokens`, `usd-foundational-values`,
    `usd-paths`, and `usd-listops-authored`.
  - Decodes token/string/path indices, inlined values, offset values, arrays,
    dictionaries, vectors, listOps, specifiers, and variability values required
    by the first fixture set.

- `usdc-spec-parser`
  - Owns Crate section mapping into the document model.
  - Depends on `usdc-binary-format`, `usdc-value-decoder`,
    `usd-document-model`, and `usd-paths`.
  - Reads `TOKENS`, `STRINGS`, `FIELDS`, `FIELDSETS`, `PATHS`, and `SPECS`,
    constructs document-model specs, attaches authored fields, and emits the
    canonical layer dump.

- `usdc-layer-open`
  - Owns the concrete USDC layer-format handler.
  - Depends on `usdc-spec-parser`.
  - Provides `usdc_layer_format`.
  - Accepts opened resource bytes from `usd-layer-open` and returns canonical
    dump JSON.

Update `usd-layer-open`:

- depend on `usdc-layer-open`
- consume `usdc-layer-open.usdc_layer_format`
- dispatch `.usdc` to the USDC handler
- keep `.usda` dispatch through `usda-layer-open`
- keep `.usdz` unsupported until package support exists

## Durable Files To Add Or Update

Add:

- `skills/usdc-binary-format/SKILL.md`
- `skills/usdc-value-decoder/SKILL.md`
- `skills/usdc-spec-parser/SKILL.md`
- `skills/usdc-layer-open/SKILL.md`
- `graph/scopes/usdc-single-layer.yaml`
- `goldens/integration/usdc-single-layer/basic.json`
- `benchmarks/usdc/targets.json`
- `benchmarks/fixtures/*.usdc`

Update:

- `graph/skillgraph.json`
- `graph/skillgraph.yaml`
- `skills/usd-layer-open/SKILL.md`
- `harness/dump_contract.md`
- `harness/score.py`
- `harness/benchmark.py`
- `docs/graph-driven-regeneration.md`
- `README.md`

Generated C++ artifacts for the first implementation:

- `generated/cpp/usdc_binary.h`
- `generated/cpp/usdc_binary.cpp`
- `generated/cpp/usdc_values.h`
- `generated/cpp/usdc_values.cpp`
- `generated/cpp/usdc_parser.h`
- `generated/cpp/usdc_parser.cpp`
- `generated/cpp/usdc_layer_open.h`
- `generated/cpp/usdc_layer_open.cpp`
- update `generated/cpp/dump_layer.exe`

## Harness Changes

Generalize `harness/score.py` from USDA-only inline text to format-neutral case
inputs.

Support these case input forms:

```json
{"input": {"text": "...", "extension": "usda"}}
```

```json
{"input": {"fixture": "benchmarks/fixtures/minimal.usdc"}}
```

Rules:

- Inline `text` cases write a temporary file using `extension`.
- `fixture` cases pass the fixture path directly to the dump command.
- Existing `input.usda` cases remain supported during migration.
- Expected output comparison stays unchanged.

Generalize `harness/dump_contract.md` wording:

- replace `USDA file path` with `single layer resource path`
- state that current scopes may provide USDA text or USDC binary resources
- keep the JSON schema unchanged

`harness/benchmark.py` can already pass fixture paths directly. Add
`benchmarks/usdc/targets.json` rather than changing the benchmark runner unless
USDC needs a different measurement model.

## Fixture Strategy

Start with checked-in binary fixtures instead of generating Crate files during
the test run. The reader is the target under test; fixture authoring should be a
separate controlled step.

Initial fixture set:

- `minimal_empty.usdc`
  - layer spec only
  - proves header, bootstrap, TOC, required sections, layer form, and empty
    child lists

- `one_root_prim.usdc`
  - `/World` prim with `specifier=def` and `typeName=Xform`
  - proves `TOKENS`, `FIELDS`, `FIELDSETS`, `PATHS`, `SPECS`, token values, and
    spec form mapping

- `nested_prims_attributes.usdc`
  - mirrors the USDA golden for nested prims and scalar/vector defaults
  - proves numeric value representations and property specs

- `relationship_targets.usdc`
  - relationship with explicit target listOp
  - proves path values and `PathListOp`

Useful source of truth for fixture creation:

- author equivalent USDA files
- convert them to USDC with a trusted external tool outside the harness
- check in the resulting small `.usdc` files
- keep expected JSON authored manually or copied from matching USDA golden after
  confirming the converted binary represents the same layer opinions

## Implementation Order

1. Add `usdc-single-layer` scope with only planned nodes and no generated
   artifacts. Confirm `regen_graph.py --scope usdc-single-layer --target cpp`
   prints missing artifacts rather than failing on an unknown scope.

2. Add USDC skill contracts and graph nodes. Keep boundary guards strict:
   binary decoding belongs to `usdc-*`, resource I/O stays in
   `usd-resource-protocol`, layer dispatch stays in `usd-layer-open`, and spec
   storage stays in `usd-document-model`.

3. Generalize `harness/score.py` input handling. Confirm existing
   `usda-single-layer` validation still passes before adding USDC fixtures.

4. Add minimal USDC fixtures and goldens. Start with `minimal_empty.usdc` and
   `one_root_prim.usdc`.

5. Implement `generated/cpp/usdc_binary.{h,cpp}`.
   Required first pass:
   - byte cursor with bounds checks
   - little-endian numeric readers
   - header and version validation
   - bootstrap and TOC reading
   - LZ4 buffer decompression
   - compressed integer array decoding

6. Implement `generated/cpp/usdc_values.{h,cpp}`.
   Required first pass:
   - decode value representation payload/type/flags
   - support token, string, int, float, double, double3, specifier,
     variability, dictionaries used by fixtures, and path/listOp values used by
     relationships
   - return existing canonical value shape from `contracts/value.schema.json`

7. Implement `generated/cpp/usdc_parser.{h,cpp}`.
   Required first pass:
   - read tokens, strings, fields, field sets, paths, and specs
   - map Crate spec forms to document-model spec kinds
   - attach decoded fields by field name
   - keep child lists in agreement with specs
   - reject duplicate spec paths

8. Implement `generated/cpp/usdc_layer_open.{h,cpp}` and update the C++
   `dump_layer.exe` target dispatch.

9. Record generated artifacts:

```bash
python3 harness/regen_graph.py --scope usdc-single-layer --target cpp --record-existing
```

10. Validate both scopes:

```bash
python3 harness/regen_graph.py --scope usda-single-layer --target python --validate
python3 harness/regen_graph.py --scope usdc-single-layer --target cpp --validate
```

## Acceptance Criteria

- `usda-single-layer` continues to pass unchanged.
- `usdc-single-layer` passes all initial USDC goldens.
- Unsupported `.usdz` still fails clearly through `usd-layer-open`.
- Resource handling remains owned by `usd-resource-protocol`.
- USDC binary parsing does not call USDA lexer, USDA value parser, or
  `usda-spec-parser`.
- The graph status for `usdc-single-layer` is `ready` after recording generated
  artifacts.

## Risks And Follow-Ups

- LZ4 may require a C++ dependency, vendored library, or a small local adapter.
  If dependency installation is not acceptable, fixtures can initially avoid
  compressed value data only if the Crate format permits that for the required
  sections; the plan should otherwise add an explicit dependency policy.
- The current document model is minimal and may need expansion for variant sets,
  time samples, extension fields, and unknown fields.
- `usd-resource-protocol` currently classifies package resources but does not
  read package entries. That is fine for `usdc-single-layer`; it will matter for
  `usdz-single-layer`.
- The canonical dump contract wording is USDA-specific even though the schema is
  format-neutral. Update wording before adding USDC goldens.
