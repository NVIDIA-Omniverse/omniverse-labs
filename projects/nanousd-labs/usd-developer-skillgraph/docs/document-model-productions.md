# Document-Model Productions — Format-Neutral Rule Layer

## Purpose

The contracts under `contracts/document-model-productions/` codify rules a
parsed USD layer must satisfy regardless of whether the source was USDA text or
USDC binary. They sit between the abstract document-model capability
(`contracts/capabilities/document-data-model.json`) and the per-format mapping
productions (`contracts/usda-productions/*` and
`contracts/usdc-productions/*-mapping.contract.json`):

```
contracts/capabilities/document-data-model.json
        |
        |  (extends with full per-field shape rules)
        v
contracts/document-model-productions/{prim,attribute,relationship,variant,layer,common-metadata}-spec.contract.json
        |                                                 |
        |  USDA cites via derives_from                    |  USDC cites via derives_from
        v                                                 v
contracts/usda-productions/{prim,attribute,...}.json    contracts/usdc-productions/*-mapping.contract.json
        |                                                 |
        |  (USDA grammar dispatch + storage_mapping)      |  (Crate spec form + FIELDS token + value-rep type)
        v                                                 v
                          usd-document-model (one shared layer)
```

The format-neutral layer is what made the second target (USDC) tractable: the
first target (USDA) had accumulated a deep set of per-spec-kind invariants
across nine production contracts and eleven golden suites, but those rules
were all phrased in USDA terms ("a missing ListOp token authors the explicit
subfield", "reorder nameChildren stores primOrder", "reserved keywords in
metadata"). USDC needed the same invariants — relationship specs reject
variability, prim specs carry primChildren/properties/variantSetChildren,
variant specs report specifier=over — but in Crate terms (FIELDS token names,
spec form ids, value-rep type ids). Rather than duplicate ~80% of the prose
into a parallel `usdc-productions/` shape, the shared invariants live in one
place and each format adds a thin mapping layer.

## What lives where

### `contracts/capabilities/document-data-model.json`

Defines the minimum: spec_forms (layer/prim/attribute/relationship and their
required field set), the known core field-name vocabulary, field-value model
(tagged variants, native domain values for specifier/variability/Retiming/etc.),
and the most basic laws (each spec path is unique within a layer; the layer
spec always exists at `/`; active=false does not remove a spec).

Use this layer to declare *what* the document model is.

### `contracts/document-model-productions/*-spec.contract.json` (6 files)

Each format-neutral production extends the capability minimum with the full
per-field shape rules:

- **`prim-spec.contract.json`** — 25 fields a prim spec carries, listOp subfield
  rules for composition arc fields, child-list invariants
  (primChildren/properties/variantSetChildren), alias mapping (doc, inherits,
  variantSets, variants), 10 laws.
- **`attribute-spec.contract.json`** — 14 fields, forbidden_fields set
  (prim/relationship/layer-only field names rejected on attribute specs),
  TimeSamples / Spline / default-vs-typeName value-shape rules, 9 laws.
- **`relationship-spec.contract.json`** — 11 fields, forbidden_fields set
  (variability and all attribute/prim/layer-only fields rejected), listOp
  subfield rules for targetPaths, 8 laws.
- **`variant-spec.contract.json`** — variant + variantSet spec kinds, parent-kind
  nesting rules, variantSetChildren-vs-variantSetNames separation,
  specifier=over invariant for variants, 7 laws.
- **`layer-spec.contract.json`** — 18 layer-level fields, subLayer/Retiming
  agreement, relocates value-shape, alias mapping (doc, layerRelocates), 8 laws.
- **`common-metadata.contract.json`** — field categories per spec kind, alias
  mapping, spec_kind_field_admission matrix (which fields each spec kind
  admits), listOp subfield semantics ({explicit, prepend, append, delete,
  reorder}), repeated-field last-one-wins rule, extension-field rule,
  format-neutral diagnostic codes (`DuplicateSpecPath`, `InvalidFieldAlias`,
  `ForbiddenFieldOnSpecKind`, `TypeMismatch`), 9 laws.

Each declares `applies_to_formats: ["usda", "usdc"]` and
`derives_from: contracts/capabilities/document-data-model.json#<anchor>`.

Use this layer to declare *what shape a spec must have* — invariants that hold
regardless of how the bytes arrived.

### `contracts/usda-productions/*.contract.json` (9 files)

Each cites a format-neutral production via `derives_from` and adds USDA-specific
content:

- **`storage_mapping`** — how USDA grammar productions (`reorder nameChildren`,
  `payload`, `subLayers`) map onto document-model field names.
- **`metadata_productions`** — USDA grammar for keyword metadata productions
  (`InheritsMetadata`, `ReferencesMetadata`, etc.).
- USDA-specific `laws` — text-grammar rules ("ListOpRelationship shares the
  leading ListOp token with ListOpConnect, so property dispatch must inspect
  the following tokens"; "reserved USDA keywords must not be routed through
  extension-field token lookup").
- `format_neutral_laws_hoisted` — audit trail listing every law that was moved
  upstream during Phase A.2 of the higher-level rule-parity work. This array
  is a change log a reviewer can use to confirm no semantic drift occurred
  when laws were hoisted.

Use this layer to declare *how USDA text produces a conformant spec*.

### `contracts/usdc-productions/*-mapping.contract.json` (6 files)

Each cites a format-neutral production via `derives_from` and adds
USDC-specific content:

- **`spec_form`** (or `spec_forms` for variant/variantSet) — the Crate spec
  form id (e.g. 6 for Prim, 1 for Attribute, 8 for Relationship, 10/11 for
  Variant/VariantSet, 7 for Layer) that produces this document-model spec
  kind.
- **`storage_mapping.field_resolution`** — for each field on the spec kind,
  the FIELDS token name and the Crate value-representation type id used to
  encode it. E.g. `targetPaths` is FIELDS token "targetPaths" decoded as
  value type 34 (PathListOp).
- **`forbidden_field_enforcement`** — the spec-kind admission matrix in Crate
  terms: which FIELDS token names attached to which spec forms must emit
  `MalformedUsdcSection` (reduced to `ForbiddenFieldOnSpecKind` at dump
  boundary).
- USDC-specific `laws` — Crate encoding rules ("Relocates type 58 is
  version-gated at minor ≥ 11"; "subLayers + subLayerOffsets must have
  matching lengths").

Use this layer to declare *how Crate binary produces a conformant spec*.

### `contracts/usdc-productions/parser-diagnostics-mapping.contract.json`

Maps the wire-layer USDC-specific diagnostic codes (defined in
`crate-reader-diagnostics.contract.json`) to the format-neutral codes declared
in `common-metadata.contract.json#diagnostics`. This means a dump-side consumer
matches on the same `DuplicateSpecPath` / `ForbiddenFieldOnSpecKind` /
`TypeMismatch` / `UnsupportedFeature` codes regardless of which format produced
the diagnostic. Format detection at the error-reporting boundary is
unnecessary.

## How `derives_from` works

Three structural relationships are expressed:

1. **Format-neutral → capability.** Each
   `document-model-productions/<x>.contract.json` declares
   `derives_from: "contracts/capabilities/document-data-model.json#<anchor>"`
   to say: this production extends the capability's minimum with per-field
   shape rules. The capability's required_fields set is preserved; the
   production adds the optional field set, listOp subfield rules, and value-
   shape laws.

2. **Format-specific → format-neutral.** Each USDA / USDC production declares
   `derives_from: "contracts/document-model-productions/<x>.contract.json"`
   to say: this format produces specs that satisfy the cited contract. The
   format-neutral laws are not restated; the production adds only the per-
   format grammar dispatch and storage mapping.

3. **Format-specific → format-specific (rare).** A mapping contract may also
   cite a dependency contract directly via `dependency_contracts`, e.g.
   `prim-spec-mapping.contract.json` cites
   `crate-section-layouts.contract.json` and
   `crate-value-representations.contract.json` to point readers at the wire-
   format details that the storage mapping presumes.

A contract cross-reference checker (`harness/contract_xref.py`, planned as
Task #28 of the rule-parity workstream) walks every `derives_from` entry and
confirms the target file exists and the cited production has matching
`applies_to_formats`. The cross-reference is structural only; semantic
equivalence between the hoisted laws and the upstream rules is a human review
concern, supported by the `format_neutral_laws_hoisted` audit array in each
USDA production.

## How goldens cite both layers

An integration golden suite (e.g.
`goldens/integration/usdc-single-layer/relationship-listops.json`) cites both
layers in its `production_contracts` array:

```json
{
  "suite": "usdc-single-layer-relationship-listops",
  "production_contracts": [
    "contracts/document-model-productions/relationship-spec.contract.json",
    "contracts/usdc-productions/relationship-spec-mapping.contract.json"
  ],
  "cases": [...]
}
```

The format-neutral citation tells the reader what *shape* the expected layer
JSON is asserting. The format-specific citation tells the reader what *Crate
encoding* the fixture exercises. The expected layer JSON itself is identical
to the equivalent USDA golden's expected layer JSON — that's the whole point
of the format-neutral lift, and it's checked end-to-end by the `cross_check`
step on the `usdc-single-layer` scope.

## How to add a new spec-kind production

If a future spec revision introduces a new spec kind:

1. Add the spec kind to `contracts/capabilities/document-data-model.json`
   `spec_forms` with its required_fields set.
2. Add `contracts/document-model-productions/<kind>-spec.contract.json` with
   the full per-field shape rules, listOp subfield rules, forbidden_fields
   set, and laws. Declare
   `applies_to_formats: ["usda", "usdc"]` and
   `derives_from: contracts/capabilities/document-data-model.json#spec_forms.<kind>`.
3. Update `common-metadata.contract.json#spec_kind_field_admission.matrix` to
   list the new spec kind's admitted fields.
4. For each format that supports the new kind, add a mapping production
   under `contracts/usda-productions/` or
   `contracts/usdc-productions/<kind>-spec-mapping.contract.json` citing the
   new format-neutral production.
5. Wire the new format-specific contract into the relevant skill node's
   `provides[].contracts` array in `graph/skillgraph.json`.
6. Add integration golden suites under
   `goldens/integration/<format>-single-layer/<kind>.json` citing both
   layers in `production_contracts`.
7. If the spec kind has invariants the parser must positively materialize in
   code (e.g. required default fields), add a `regex_required` lint rule to
   `contracts/lint/<scope>.lint.json`.

The pattern repeats. The format-neutral layer absorbs the cost once; each
new format only pays the cost of its own mapping.

## Behavioral checks: enforcing format-neutrality at runtime

The lint, source_audit, and contract_xref gates check the generated source
shape — they're file-set-listed and pattern-specific, so a sufficiently
adversarial regen agent can evade them by renaming files (implementation
moved outside the audited path list), changing surface syntax (e.g.
`std::find(tokens.begin(), ..., "Foo")` instead of `tokens[N] == "Foo"`),
or inserting dead-code anchors that satisfy regex requirements without
contributing to behavior.

Five harness steps test the parser's *output* rather than its source,
which is structurally immune to all of the above:

### `cross_format_check` — cross-format dump equivalence

For each matched pair `(foo.usda, foo.usdc)` declared in
`graph/skillgraph.json` `usdc-single-layer.cross_format_check.fixture_pairs`,
the harness runs `dump_layer.exe` against both files and asserts the
canonicalized layer JSON is identical. The two parsers (USDA grammar walk;
USDC Crate decoder) consume entirely different surface representations of
the same scene. Their canonical dumps MUST agree because the document
model is format-neutral by design (see this document's top section).

If either parser cheats — e.g., a USDA "parser" that substring-matches
fixture text and emits memorized dumps — its output for a given fixture
will diverge from what the OTHER format's parser produces from real
content. The check reports the first divergence with a JSON path so the
cheat site is immediately legible.

The check has a strong forcing property when **adding new fixtures**: any
fixture whose USDA/USDC pair the agent didn't anticipate falls through to
whatever real parser code exists. A real parser produces matching dumps;
a cheating parser falls through to default behavior and diverges. The
agent cannot pre-cheat for unauthored fixtures.

### `filename_rename_check` — content-driven dispatch invariance

For each fixture, the harness copies it to a randomly-named temporary path
and asserts the dump is unchanged (after canonicalization). A parser that
gates on the input filename (`if path.contains("World")`) produces
different output for the rename; a parser that only reads bytes/text is
path-invariant.

### `usda_rename_invariance` — content-driven dispatch invariance (USDA)

For each USDA fixture, the harness extracts every prim/variant identifier
*and every attribute identifier* declared in the fixture, generates a
deterministic-but-random replacement for each, AND mutates unique float
defaults to random values, rewrites the USDA text with all
substitutions, and asserts the resulting layer dump equals the original
dump with the same substitutions applied.

A real content-driven parser is invariant under identifier substitution
and propagates value literals from text to dump verbatim. Three classes
of cheat fail this check:
- **Substring-match canned-literal cheater** — `if (text.contains
  ("World")) emit_known_dump_A();` does not fire on renamed text and
  falls through to a generic fallback.
- **Canned-dump cheater** — hardcoded dump that bakes in the original
  numeric defaults (e.g. `"value": 1.5`) diverges when the unique value
  `1.5` is randomly remapped to e.g. `614.2565` in both text and
  expected dump.
- **Attribute-name fingerprinter** — hardcoded paths like
  `/World.size` that don't track the renamed attribute name.

The rename map and value mutations are randomized at validate time
using a seed the agent cannot know at code-generation time, so
pre-cheating for specific random names or values is structurally
impossible.

Value mutation is conservative: only float literals with a decimal point
that appear exactly once in the USDA text AND exactly once as a numeric
leaf in the original dump are eligible. This avoids ambiguity when the
same numeric value (e.g. `1.0`) shows up across several attributes.

This check is the USDA analog of what `cross_format_check` does for the
(USDA, USDC) pair-equivalence shape: both target "agent has memorized
specific fixtures" rather than "agent has implemented a parser."
`filename_rename_check` covers the orthogonal "agent has memorized a
filename" shape. The three together cover every memorization vector that
doesn't require coordinating a fake parser across formats.

### `usdc_rename_invariance` — content-driven dispatch invariance (USDC)

For each (usdc, usda) pair, the harness reads prim/variant identifiers
from the USDA sibling, picks same-length random replacements, and
rewrites the USDC bytes via NUL-bounded byte substitution
(`\0World\0` → `\0xKp1A\0`). The mutated parse must produce the
rename-applied dump.

This check closes the cheat shape that `cross_format_check` cannot reach
on its own: a fake USDC parser that fingerprints by byte content AND a
fake USDA parser that fingerprints by text content, coordinated to emit
the same canned dump for matched (usda, usdc) pairs. With byte mutation
applied only to the USDC side, a real parser produces a structurally
identical dump (with the rename map applied to path components); a
byte-fingerprint cheater either falls through to a generic fallback or
keeps emitting the original tokens.

**Limitation.** USDC token sections are LZ4-compressed. When a token's
bytes appear as a literal run in the LZ4 stream, the NUL-bounded search
finds and replaces them. When LZ4 emits the token as a back-reference
to another literal elsewhere in the file, the bytes aren't visible as
`\0<name>\0` and are silently skipped per token. Fixtures where ALL
tokens are LZ4-compressed-away are skipped entirely with a clear
warning. For the 1-5 KB fixtures committed under
`benchmarks/fixtures/`, LZ4 typically emits identifiers as raw
literals so the check catches the common case.

### `synthesized_scenes` — generated-at-validate-time fixtures

Every check above starts from a fixture committed to the repo. The regen
agent sees those bytes at code-generation time, so even with all the
randomized mutations layered on top, an adversarial agent could in
principle plant *two* canned outputs: one for the known fingerprint of a
committed fixture, and another for "anything else" that the rename
mutation happens to fall through to.

Synthesized scenes break this entirely. The harness owns a small
`Scene` data model (prim tree with scalar `double` attribute defaults)
and generates random scenes at validate time. For each scene it:

1. Emits a `.usda` file (random prim names, random nesting, random
   attribute defaults).
2. When usd-core is available, also emits a `.usdc` binary for the same
   `Scene`.
3. Computes the expected layer dump from the same `Scene` object via a
   separate code path (no parser involved).
4. Runs the parser on the emitted USDA (and USDC, when present).
5. Asserts each parser dump equals the expected dump.

The agent never sees these scenes at codegen time — they exist for
milliseconds inside the harness. There's nothing to fingerprint, no
shape to bake in. The only strategy that works is "implement a real
parser."

**Two emitter implementations** live in `harness/scene_synthesis.py`:

- `PureStdLibEmitter` — always available. f-string templating for the
  USDA text; hand-coded JSON construction for the expected dump.
  Cannot author USDC (the binary Crate format requires LZ4 +
  multi-section encoding, which is impractical to reimplement in pure
  Python; a hand-rolled USDC writer would be a fragile source of false
  negatives).
- `UsdCoreEmitter` — used when `pxr` is importable. Builds an
  `Sdf.Layer` in memory via the Pixar Python API, exports to USDA text
  AND USDC bytes via the spec authors' own serializer, and traverses
  the same layer to produce the canonical dump JSON. Acts as a
  reference implementation: pxr is what the spec authors maintain.
  When both emitters are available, the harness runs a parity check at
  the start of each run, asserting they agree on a sample scene; this
  catches drift between implementations.

`get_emitter()` selects usd-core when available, else falls back to the
pure-stdlib path. The `Scene` model is the intersection of what both
emitters support; growing coverage means growing both implementations
together. MVP scope: root prims and nested child prims (up to depth 3)
with scalar `double` attribute defaults. Future expansions (variants,
listOps, layer metadata, time samples) extend both emitters together.

**usd-core is treated as a preference, not a dependency, but is
strongly recommended.** The USDA half of `synthesized_scenes` runs
everywhere via the pure-stdlib fallback. The USDC half requires usd-
core; without it, the harness prints a hint at validate time pointing
the operator at the install path. Without usd-core, USDC parsing is
tested only on committed fixtures (via `cross_format_check` and
`usdc_rename_invariance`); a USDC parser that fingerprints the
committed fixtures via a property that survives byte mutation could in
principle pass those gates while cheating. Synthesized USDC closes
that hole. To enable it:

```
python3 -m venv ~/.venv/usd-dev-python3.12
~/.venv/usd-dev-python3.12/bin/pip install usd-core
# Then run the harness via that venv's python:
~/.venv/usd-dev-python3.12/bin/python harness/regen_graph.py --scope ... --validate
```

Implemented at `harness/regen_graph.py:run_cross_format_check`,
`run_filename_rename_check`, `run_usda_rename_invariance`,
`run_usdc_rename_invariance`, and `run_synthesized_scenes`. All five
run as part of `validate_scope` next to the existing `cross_check`
branch when the scope declares the corresponding field.

## Why behavioral checks are more leveraged than additional lint rules

Lint rules define their own escape hatches in source. Every rule names a
file set + a regex; an agent reading the rule can change the file name or
shift the surface syntax until the regex stops matching. Across two regen
rounds against this codebase, agents have demonstrated:

- `tokens[N] == "literal"` → `std::find(tokens.begin(), ..., "literal")`
  (rule's regex didn't match iterator search)
- `"crate-section-layouts.contract.json"` literal → `"crate-section-layouts" + ".contract" + ".json"` concatenation
  (rule's regex didn't span the concat)
- Implementation in `usdc_parser.cpp` (audited) → implementation in
  `reader.cpp` (not in any rule's `applies_to_artifacts`)
- Public-API call required by `symbol_required` rule → `(void)foo({})`
  one-liner that satisfies the symbol-presence check with zero behavior
- Required code patterns (e.g. `kind == "relationship"` + `variability`)
  → dead-code anchor function `keep_contract_field_anchors()` with all
  the literals + `(void)` casts to suppress unused-variable warnings

Each evasion shape required a new lint rule. The arms race is asymmetric:
the agent moves faster than the rule authoring.

A behavioral check has no source-side escape hatch. The escape would have
to be in *runtime behavior*, which is much harder to fake: to defeat
`cross_format_check`, the agent would need to coordinate canned-literal
substring matches across both USDA and USDC parsers and keep them
consistent as fixtures evolve. Any new fixture pair immediately exposes
the gap. The path of least resistance becomes: write a real parser.

### Lint role: forbidden patterns only

There's a sharp asymmetry between the two kinds of lint rule
`harness/contract_lint.py` supports:

- **`regex_match` / `symbol_present` (forbidden patterns).** "If this
  shape appears in the source, something is wrong." The pattern is
  enumerable (`std::ifstream`, content-keyed `tokens[N] == "..."`
  dispatch, the literal `R"({"ok":true,...})"` canned dump). The
  agent can evade only by *not writing the pattern* — and finding
  another working way to do the same job. Each rule narrows the
  cheap-cheat space. These are useful.

- **`regex_required` / `symbol_required` (required patterns).** "If
  this shape *doesn't* appear, something is wrong." The pattern is
  open-ended. The agent satisfies the rule by *putting the literal
  somewhere*, anywhere — a dead-code anchor function, a string in a
  `[[maybe_unused]] const char *`, a `(void)`-cast wrapper around the
  symbol. The rule has no way to verify the symbol is on a load-
  bearing path. These are theater.

The `usdc-single-layer.lint.json` rule set used to include four
`regex_required` rules ("must materialize primChildren / properties /
variantSetChildren on prim specs", etc.) and three `symbol_required`
rules ("the adapter must call `open_crate` / `decode_value_
representation` / `parse_usdc_layer`"). All seven were trivially
evaded:

- The prim-spec required-fields rule was satisfied by
  `[[maybe_unused]] const char *required_prim_fields =
  "primChildren" "properties" "variantSetChildren";` — a single
  inert line in the parser source.
- All three adapter symbol-required rules were satisfied by
  `auto fn = &usdsg::usdc::<symbol>; (void)fn;` — the adapter still
  printed `{"ok":true, "results":[]}` without invoking the unit.
  The unit goldens (`goldens/unit/usdc-*/...`) catch these stubs
  behaviorally when the unit scope is validated; the lint rule was
  trying to be a cheap shortcut and failed at it.

The pruning removes all seven and leaves only the forbidden-pattern
rules. Future lint additions should encode known-bad shapes
(`regex_match` or `symbol_present`), not aspirational ones; required
behavior belongs in the goldens or the behavioral checks above.

## Scope layout: language-agnostic vs. language-specific gates

Validation gates in `graph/skillgraph.json` are split by whether they
inspect *output* (language-agnostic, lives at scope level) or *source*
(language-specific, lives under `targets.<target>.*`).

**Scope-level (language-agnostic, behavioral):**
- `goldens` / `handle_goldens` — assert layer dump equality; the dump is
  format-neutral by construction.
- `benchmarks` / `handle_benchmarks` — assert latency targets; targets
  are wall-clock measurements, language-neutral.
- `contract_xref` — walks `derives_from` entries in the contracts/
  directory; has nothing to do with the generated code's language.
- `cross_format_check` — compares dumps from two parsers consuming the
  same logical scene in different surface formats.
- `filename_rename_check` — copies a fixture to a random path and
  asserts dump equality.
- `usda_rename_invariance` — substitutes prim/attribute identifiers AND
  unique float defaults in USDA text and asserts dump equality
  (modulo the substitution map).
- `usdc_rename_invariance` — byte-level NUL-bounded token substitution
  on USDC bytes and asserts dump equality (modulo the rename map).
- `synthesized_scenes` — generates random scenes at validate time
  (USDA always; USDC additionally when usd-core is available) and
  asserts the parser dump equals an expected dump computed independently
  from the same Scene object. Strongest check: the agent never sees the
  inputs at codegen time, so there's nothing to memorize.

These apply to a Rust regen, a Python regen, or a C++ regen
identically — the check inspects what the parser produces, not how it
is written. A new language target just declares its `dump_cmd` /
`adapter_cmd` under `targets.<language>` and inherits the entire
behavioral suite for free.

**Per-target (language-specific, structural):**
- `targets.<lang>.contract_lints` — pattern checks against generated
  source files. Patterns are language-specific by construction (a C++
  rule against `std::find(...)` does not apply to Rust's `.iter().find()`
  or Python's `next(filter(...))`).
- `targets.<lang>.source_audit` — forbidden-symbol scan against named
  source artifacts. The artifact list and the forbidden symbols are
  both language-specific.
- `targets.<lang>.cross_check` — compares two adapter binaries' output;
  the `cmd` field names the actual `.exe` / executable path produced by
  the language toolchain.

These three gates harden the C++ implementation today; when a
Rust/Python target is added, equivalent rules go under
`targets.rust.contract_lints` / `targets.rust.source_audit` /
`targets.rust.cross_check`. The behavioral checks above are unchanged.

The `validate_scope` harness reads per-target fields first and falls
back to scope-level only for legacy entries (not used by any current
scope). New gates should always be added at the correct layer:
language-agnostic at scope level, language-specific under
`targets.<lang>`.

## Cross-references

- **Capability:** `contracts/capabilities/document-data-model.json`
- **Format-neutral productions:** `contracts/document-model-productions/`
- **USDA mapping productions:** `contracts/usda-productions/`
- **USDC mapping productions:** `contracts/usdc-productions/*-mapping.contract.json`
- **Skill SKILL.md files cite the production layers:**
  `skills/usda-spec-parser/SKILL.md`, `skills/usdc-spec-parser/SKILL.md`,
  `skills/usdc-value-decoder/SKILL.md`, `skills/usdc-layer-open/SKILL.md`.
- **Lint rules:** `contracts/lint/usdc-single-layer.lint.json` (15
  forbidden-pattern rules; see "Lint role" below).
- **Errata for spec/reference divergence:** `docs/usdc-spec-errata.md`.
