# Graph-Driven Regeneration

The initial prototype had a dependency list in `graph/skillgraph.yaml`, but the
generated target was a single file. That proved correctness for one target, but
it did not exercise the intended skillgraph workflow.

The executable model is now `graph/skillgraph.json`.

## Node Model

Each node declares:

- `skill`: the durable instruction contract for that node
- `depends_on`: direct dependency edges
- `provides`: named capabilities owned by the node
- `consumes`: required capability consumption obligations from dependency nodes
- `targets.<language>.artifacts`: generated outputs owned by that node

`depends_on` is an availability and ordering edge. It says a provider is in
scope before the dependent node is generated. `consumes` is stricter: it says a
dependent node must use the provider-owned semantic boundary rather than
silently reimplementing it.

For example, `usd-paths` depends on `usd-identifiers-and-names` and consumes the
`identifier_scanner` capability. Path parsing still owns path delimiters,
relative-path structure, variant punctuation, and path-specific diagnostics, but
UTF-8 decoding and XID identifier validity remain owned by
`usd-identifiers-and-names`.

Consumption rules may declare a performance exception. That exception is not a
new semantic owner. It permits a target-local fast path only when the provider
entrypoint would violate a declared performance target, and only if the local
path still passes the provider conformance subset, the consumer conformance
tests, and the consumer performance targets. The preferred fix is to improve the
provider capability API first.

For the Python prototype, the first single-layer target now maps nodes to
separate generated modules:

- `usd-identifiers-and-names` -> `generated/python/usd_identifiers.py`
- `usd-foundational-values` -> `generated/python/usd_values.py`
- `usd-paths` -> `generated/python/usd_paths.py`
- `usd-listops-authored` -> `generated/python/usd_listops.py`
- `usd-document-model` -> `generated/python/usd_document_model.py`
- `usda-lexical-format` -> `generated/python/usda_lex.py`
- `usda-value-parser` -> `generated/python/usda_values.py`
- `usda-spec-parser` -> `generated/python/usda_parser.py`
- `usd-resource-protocol` -> `generated/python/usd_resource_protocol.py`
- `usda-layer-open` -> `generated/python/usda_layer_open.py`
- `usd-layer-open` -> `generated/python/dump_layer.py`

The executable dump command remains a thin generic layer-open entrypoint:

```text
generated/python/dump_layer.py
```

For the current scope, that generic entrypoint has one registered format
handler: `usda-layer-open`. Future USDC or USDZ handlers should be added as
separate format nodes instead of expanding the USDA handler.

The topmost durable contract for this stack is
`contracts/handles/layer-open.handle.json`. It depends on the resource protocol,
USDA layer-format, USDA spec-parser, document-model, value, path, listOp, token,
and identifier contracts through graph edges.

The `usdc-single-layer` scope adds the planned USDC format nodes:

- `usdc-binary-format` -> `generated/cpp/usdc_binary.{h,cpp}`
- `usdc-value-decoder` -> `generated/cpp/usdc_values.{h,cpp}`
- `usdc-spec-parser` -> `generated/cpp/usdc_parser.{h,cpp}`
- `usdc-layer-open` -> `generated/cpp/usdc_layer_open.{h,cpp}`

The first USDC implementation is C++-first. Until those artifacts are generated
and recorded, `regen_graph.py --scope usdc-single-layer --target cpp` should
report missing C++ artifacts rather than an unknown scope.

The same graph also has a `cpp` target. Its generated modules live under
`generated/cpp/`, and the entrypoint is `generated/cpp/dump_layer.exe`.

`usd-paths-handle` is a smaller scope that validates only the `usd-paths`
subgraph through `contracts/handles/path.handle.json`. It uses generated
adapter commands rather than a layer dump command.

`usd-tokens-handle` does the same for `usd-tokens` through
`contracts/handles/token.handle.json`.

`usd-identifiers-unit` validates the `usd-identifiers-and-names`
`identifier_scanner` capability through
`contracts/handles/identifier.handle.json`. The shared goldens assert validated
text and matched scanner spans rather than numeric offsets, because offsets are
target-native units and differ between Python character indexes and C++ byte
indexes for UTF-8 text.

`usd-foundational-values-unit` validates the foundational data type catalog and
value model through `contracts/handles/foundational-values.handle.json`. That
scope intentionally has no identifier dependency; it consumes only token
identity for the token scalar.

`usd-listops-authored-unit` validates native authored listOp storage through
`contracts/handles/listops-authored.handle.json`. It consumes foundational value
domains and path validation, but does not perform cross-layer listOp
composition.

`usd-single-layer-namespace-source-unit` validates the first provider of the
generic namespace-source capability. It wraps one document-model `Layer` and
declares composition, value resolution, payload loading, instancing, and schema
fallbacks absent.

`usd-stage-population-unit` validates the reusable stage-population consumer
over that namespace source. The stage-population node owns traversal, masks,
active pruning, query flags, and stage indexes, while `usd-composition-arbitrator`
or another composed namespace source can provide richer composed inputs without
replacing the population skill.

`usd-layer-stack-namespace-source-unit` validates the partial composition
provider. It consumes a root document-model layer plus already-opened recursive
sublayers, exposes the same `namespace_source` capability, and declares
composed child/property ordering. It does not resolve or open sublayer assets.

`usd-layer-stack-open-unit` validates the resource-loading bridge for recursive
local sublayers. It uses `usd-layer-open` for the root and each recursively
discovered sublayer, then constructs the layer-stack namespace source from the
opened document-model layers. Full asset resolver policy remains deferred.

`usd-specializes-namespace-source-unit` validates the first direct specializes
composition provider. It consumes the inherits namespace source, maps
specialized source subtrees onto the specialize-authoring prim, and exposes
specialized opinions as globally weakest composition opinions before variants
are evaluated.

`usd-variant-namespace-source-unit` validates the first selected-variant
composition provider. It consumes the non-variant namespace source after
reference, relocates, inherits, and specializes providers have supplied their
opinions, computes selected variants from strength-ordered prim opinions, then
inserts selected variant opinions at their LIVERPS strength position. Stage
population remains behind the same namespace-source boundary.

`usd-composition-arbitrator` is the terminal composition contract. It consumes the
bounded local and arc-family opinion sources, builds strong-to-weak prim and
property opinion stacks, applies LIVERPS arbitration in one place, and exposes a
normal namespace-source facade for stage population and nanousd backends.

## Regeneration Workflow

Use `harness/regen_graph.py` to ask for a node or scope. The harness expands the
transitive closure and prints the topological generation order.

Graph scope commands use `{python}` as a portable placeholder for the active
Python interpreter running the harness. This keeps the graph independent of
platform-specific launcher names such as `py` or `python3`.

```powershell
py harness/regen_graph.py --scope usda-single-layer --target python
```

For a single node:

```powershell
py harness/regen_graph.py --node usda-value-parser --target python
```

The status model is:

- `missing`: declared artifact does not exist
- `unrecorded`: artifact exists, but no generated manifest has accepted it
- `stale`: artifact exists, but the skill/dependency fingerprint or artifact
  digest changed
- `ready`: artifact exists and matches the recorded manifest

After a generated subgraph is accepted, record it:

```powershell
py harness/regen_graph.py --scope usda-single-layer --target python --record-existing
```

Then run full scope validation:

```powershell
py harness/regen_graph.py --scope usda-single-layer --target python --validate
```

Or do both in one command:

```powershell
py harness/regen_graph.py --scope usda-single-layer --target python --record-existing --validate
```

The format-neutral scorer accepts legacy inline USDA text, explicit inline text
with an extension, and checked-in fixture paths:

```json
{"input": {"usda": "#usda 1.0\n"}}
```

```json
{"input": {"text": "#usda 1.0\n", "extension": "usda"}}
```

```json
{"input": {"fixture": "benchmarks/fixtures/minimal_empty.usdc"}}
```

## Open Design Gap

This still does not invoke an LLM by itself. It plans and validates regeneration
using concrete graph edges and generated artifact ownership. The next layer
would be a target generator that consumes this plan, regenerates only `missing`
or `stale` nodes in order, and writes the manifest only after the complete
scope passes validation.
