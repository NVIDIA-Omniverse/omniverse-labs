# Regeneration Tasks

This file is the operator/agent prompt for creating disposable generated target
implementations under `generated/`. It is not an executable generator.

Use `harness/regen_graph.py` first to inspect graph order and artifact status.
Then generate missing or stale files in dependency order, run the scope
validation, and record the generated manifest only after the target is accepted.

## USDA Single-Layer Reader

Generate a target implementation under `generated/<target-name>/` that opens a
resolved local USDA layer resource and emits the canonical layer dump JSON to
stdout.

### Inputs

Read these durable artifacts:

- `graph/skillgraph.yaml`
- `graph/skillgraph.json`
- `graph/scopes/usda-single-layer.yaml`
- all `skills/*/SKILL.md` files in the required subgraph
- `contracts/handles/resource.handle.json`
- `contracts/*.schema.json`
- `harness/dump_contract.md`
- `goldens/integration/usda-single-layer/basic.json`
- pinned spec excerpts under `spec/pinned/` if present

If `spec/pinned/` is absent, stop and ask the operator to materialize the pinned
spec — `materialize-spec.sh` on Linux/macOS, `materialize-spec.ps1` on Windows
(see `spec/README.md`).

### Output

Create a dump command. It must accept a USDA file path as its final argument and
write JSON matching `contracts/layer-dump.schema.json` to stdout.

Suggested target names:

- `generated/python/dump_layer.py`
- `generated/cpp/dump_layer.exe`
- `generated/rust/target/debug/dump_layer`

### Validation

First inspect the graph plan:

```bash
python3 harness/regen_graph.py --scope usda-single-layer --target python
```

Run:

```bash
python3 harness/score.py \
  --goldens goldens/integration/usda-single-layer/basic.json \
  --dump-cmd "<dump command>"
```

Then run benchmarks if fixtures are present:

```bash
python3 benchmarks/make_fixtures.py
python3 harness/benchmark.py \
  --targets benchmarks/targets.json \
  --dump-cmd "<dump command>"
```

Once the generated artifacts are accepted, record and validate the whole
subgraph:

```bash
python3 harness/regen_graph.py --scope usda-single-layer --target python --record-existing --validate
```

### Rules

Do not edit generated output by hand after scoring. If output fails, update the
skill, contract, golden, or generation prompt, then regenerate the missing or
stale node and any dependent nodes in graph order.

Do not implement composition, sublayer/reference/payload asset resolution, value
resolution, stage population, USDC, or USDZ for this target.

Do not create alternate local versions of resource protocol handling, paths,
values, listOps, or layer storage inside the parser. Use the dependency
contracts.

## USDC Single-Layer Reader

Generate a C++ target implementation under `generated/cpp/` that opens one
resolved local USDC Crate layer resource and emits the same canonical layer
dump JSON used by `usda-single-layer`.

This target proves that binary Crate decoding composes with the existing
resource protocol, layer-open dispatch, document model, paths, tokens, values,
and authored listOp boundaries. It must not route through USDA parsing code.

### Inputs

Read these durable artifacts:

- `docs/usdc-single-layer-plan.md`
- `graph/skillgraph.yaml`
- `graph/skillgraph.json`
- `graph/scopes/usdc-single-layer.yaml`
- `skills/usdc-binary-format/SKILL.md`
- `skills/usdc-value-decoder/SKILL.md`
- `skills/usdc-spec-parser/SKILL.md`
- `skills/usdc-layer-open/SKILL.md`
- all dependency `skills/*/SKILL.md` files in the required subgraph
- `contracts/handles/resource.handle.json`
- `contracts/*.schema.json`
- `harness/dump_contract.md`
- `goldens/integration/usdc-single-layer/basic.json`
- `benchmarks/usdc/targets.json`
- checked-in binary fixtures under `benchmarks/fixtures/*.usdc`
- pinned spec excerpts under `spec/pinned/` if present

If Crate details are not available in `spec/pinned/`, use the checked-in AOUSD
spec PDF or ask the operator to materialize the relevant spec excerpts before
implementing binary decoding.

### Output

Generate these C++ artifacts:

- `generated/cpp/usdc_binary.h`
- `generated/cpp/usdc_binary.cpp`
- `generated/cpp/usdc_values.h`
- `generated/cpp/usdc_values.cpp`
- `generated/cpp/usdc_parser.h`
- `generated/cpp/usdc_parser.cpp`
- `generated/cpp/usdc_layer_open.h`
- `generated/cpp/usdc_layer_open.cpp`

Also ensure the complete C++ target has the shared dependencies and entrypoint
declared by `graph/skillgraph.json`, including:

- `generated/cpp/usd_identifiers.{h,cpp}`
- `generated/cpp/usd_tokens.{h,cpp}`
- `generated/cpp/usd_values.{h,cpp}`
- `generated/cpp/usd_paths.{h,cpp}`
- `generated/cpp/usd_listops.{h,cpp}`
- `generated/cpp/usd_document_model.{h,cpp}`
- `generated/cpp/usd_resource_protocol.{h,cpp}`
- `generated/cpp/usda_*` artifacts required by the current `usd-layer-open`
  dependency closure, unless the graph is updated to dispatch USDC without that
  dependency
- `generated/cpp/dump_layer.exe`

The dump command must accept a `.usdc` fixture path as its final argument and
write JSON matching `contracts/layer-dump.schema.json` to stdout.

### Implementation Order

First inspect the graph plan:

```bash
python3 harness/regen_graph.py --scope usdc-single-layer --target cpp
```

Generate missing or stale artifacts in graph order. For the USDC-specific
nodes, implement:

1. `usdc-binary-format`: byte cursor, little-endian reads, `PXR-USDC` header and
   version validation, bootstrap, TOC, LZ4 buffer boundary, and compressed
   integer arrays.
2. `usdc-value-decoder`: Crate value representation decoding for tokens,
   strings, paths, scalars, vectors, dictionaries, specifier, variability, and
   relationship target listOps required by the fixtures.
3. `usdc-spec-parser`: `TOKENS`, `STRINGS`, `FIELDS`, `FIELDSETS`, `PATHS`, and
   `SPECS` section mapping into `usd-document-model`, with canonical dump
   output and duplicate spec path rejection.
4. `usdc-layer-open`: concrete `.usdc` layer-format handler.
5. `usd-layer-open`: generic dispatch that keeps `.usda` routed through
   `usda-layer-open`, routes `.usdc` through `usdc-layer-open`, and keeps
   `.usdz` unsupported.

### Validation

Run the USDC golden suite directly:

```bash
python3 harness/score.py \
  --goldens goldens/integration/usdc-single-layer/basic.json \
  --dump-cmd "generated/cpp/dump_layer.exe"
```

Run the USDC benchmark smoke suite:

```bash
python3 harness/benchmark.py \
  --targets benchmarks/usdc/targets.json \
  --dump-cmd "generated/cpp/dump_layer.exe"
```

Then record and validate the full C++ subgraph:

```bash
python3 harness/regen_graph.py --scope usdc-single-layer --target cpp --record-existing --validate
```

Finally, rerun the USDA scope to make sure generic layer-open dispatch did not
regress USDA behavior:

```bash
python3 harness/regen_graph.py --scope usda-single-layer --target python --validate
```

If a C++ USDA target is present and accepted, validate it too:

```bash
python3 harness/regen_graph.py --scope usda-single-layer --target cpp --validate
```

### Acceptance criteria for the regenerated parser

These behavioral gates are enforced by durable goldens/harness (not by the
generated code) and define what a correct regeneration must satisfy. They can
fail on a stale generated tree even when the contracts and goldens are correct;
a regeneration is not accepted until `regen_graph --scope usdc-single-layer
--target cpp --validate` is green, including these:

- **Child-list ordering must agree across formats.** The USDC parser must emit
  `primChildren` / `propertyChildren` / `variantSetChildren` in the same order
  the USDA parser produces — the authored order when the Crate spec authors the
  children field, and the document-model child order when materializing it.
  `cross_format_check` (USDA fixture vs USDC fixture of the same scene) and
  `synthesized_scenes` (USDC parse vs the alphabetizing emitter) fail on any
  ordering disagreement.
- **`PathListOp` subfields must decode.** `usdc-value-decoder` must populate the
  explicit / prepend / append / delete subfields of a `PathListOp` (Crate type
  34); `targetPaths` and `connectionPaths` depend on it. The
  `all-four-subfields-on-targetpaths` golden enforces this.

### Rules

Do not edit generated output by hand after scoring. If output fails, update the
skill, contract, golden, or generation prompt, then regenerate the missing or
stale node and any dependent nodes in graph order.

Keep ownership boundaries strict:

- resource I/O belongs to `usd-resource-protocol`
- generic layer dispatch belongs to `usd-layer-open`
- Crate byte decoding belongs to `usdc-binary-format`
- Crate value representation decoding belongs to `usdc-value-decoder`
- section-to-document mapping belongs to `usdc-spec-parser`
- layer storage belongs to `usd-document-model`
- paths belong to `usd-paths`
- tokens belong to `usd-tokens`
- authored listOps belong to `usd-listops-authored`

Do not call USDA lexer, USDA value parser, or USDA spec parser from USDC
decoding.

Do not implement USDC writing, USDZ package opening, `.usd` forwarding format
dispatch, sublayer/reference/payload loading, composition, value resolution,
stage population, clips, or schema fallback for this target.

## USDZ Single-Layer Reader

Generate a C++ target implementation under `generated/cpp/` that opens one
resolved local USDZ package layer resource and emits the same canonical layer
dump JSON used by `usda-single-layer` and `usdc-single-layer`.

The target sits on top of the USDA and USDC handlers. It must open the outer
package through `usd-resource-protocol`, parse ZIP/USDZ structure only through
`usdz-package-format`, select either the first physical package entry or an
explicit `outer.usdz[entry]` path, and dispatch `.usda` entries to
`usda-layer-open` and `.usdc` entries to `usdc-layer-open`.

### Inputs

Read these durable artifacts:

- `graph/skillgraph.yaml`
- `graph/skillgraph.json`
- `graph/scopes/usdz-single-layer.yaml`
- `skills/usdz-package-format/SKILL.md`
- `skills/usdz-layer-open/SKILL.md`
- `skills/usd-layer-open/SKILL.md`
- all dependency `skills/*/SKILL.md` files in the required subgraph
- `contracts/handles/usdz-package-format.handle.json`
- `contracts/handles/usdz-layer-open.handle.json`
- `contracts/usdz-productions/*.contract.json`
- `contracts/lint/usdz-single-layer.lint.json`
- `goldens/integration/usdz-single-layer/basic.json`
- `goldens/unit/usdz-package-format/package-format.json`
- `goldens/unit/usdz-layer-open/layer-open.json`
- `benchmarks/usdz/targets.json`
- USDZ fixtures under `benchmarks/fixtures/usdz/`

Regenerate deterministic USDZ fixtures with:

```bash
python3 benchmarks/make_usdz_fixtures.py
```

### Output

Generate these C++ artifacts:

- `generated/cpp/usdz_package.h`
- `generated/cpp/usdz_package.cpp`
- `generated/cpp/usdz_package_adapter.cpp`
- `generated/cpp/usdz_layer_open.h`
- `generated/cpp/usdz_layer_open.cpp`
- `generated/cpp/usdz_layer_open_adapter.cpp`
- updated `generated/cpp/dump_layer.cpp`

The dump command must accept either `package.usdz` or
`package.usdz[inner.usda|inner.usdc]` as its final argument.

### Validation

Run:

```bash
python3 harness/regen_graph.py --scope usdz-package-format-unit --target cpp --validate
python3 harness/regen_graph.py --scope usdz-layer-open-unit --target cpp --validate
python3 harness/regen_graph.py --scope usdz-single-layer --target cpp --validate
```

Then run the USDA and USDC integration scopes to check dispatch regressions:

```bash
python3 harness/regen_graph.py --scope usda-single-layer --target cpp --validate
python3 harness/regen_graph.py --scope usdc-single-layer --target cpp --validate
```

### Rules

Do not extract package entries to temporary files.

Do not parse USDA or USDC directly from `usdz-layer-open`; dispatch through the
concrete layer handlers.

Do not forward the outer package `file_backed_path` to an inner USDC handler
unless it names the same byte extent. USDZ entries should use non-owning byte
views over the opened outer package.

Do not implement Zip64, nested packages, package writes, `.usd` forwarding,
composition, dependent layer loading, asset resolution, value resolution, or
stage population for this target.
