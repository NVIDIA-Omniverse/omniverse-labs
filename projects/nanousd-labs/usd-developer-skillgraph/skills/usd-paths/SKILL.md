---
name: usd-paths
description: Use this skill when implementing or verifying USD path syntax, construction, and ordering.
metadata:
  author: NVIDIA
---

# usd-paths

Use this skill when implementing or verifying USD path syntax, construction, and
ordering.

## Spec Sources

- `specification/path_grammar/README.md`
- `specification/document_data_model/README.md` section `Speculative ObjectPath as a Core Specialized Type`
- `specification/file_formats/README.md` section `Paths and Asset References`

Pinned tag / commit: `v1.0.1`

## Provides

- Absolute root path `/`
- Absolute prim paths
- Property paths
- Relative paths accepted by the normative `Path` grammar
- Variant-selection paths accepted by the normative `Path` grammar
- Path references from USDA angle brackets
- Parent/name path construction for prims and properties
- Canonical path component identity through the graph's component identity dependency
- Path element ordering

## Contract

Paths are the unique identifiers for specs in a layer, and the USDA file format
uses the normative `Path` production for angle-bracket path references. The
document data model also defines a narrower `ObjectPath` specialized type for
absolute prim and property targets.

This prototype graph currently uses `usd-tokens` as the dependency that provides
a registered-string/component-identity capability. That does not require a path
implementation to expose or store concrete Token values. It may use any
target-native representation that satisfies the same identity constraints:
stable equality, stable hashing, and string recovery for reusable path
components. The required property is that hot path operations do not repeatedly
scan or duplicate raw component strings when canonical component identity is
available from the graph.

This skill consumes the `identifier_scanner` capability from
`usd-identifiers-and-names`. That consumption is stronger than merely depending
on the node. Path parsing owns path syntax such as `/`, `.`, relative-parent
segments, variant-selection punctuation, canonical path assembly, and
path-specific errors. UTF-8 decoding, XID start/continue policy, identifier
component validity, and namespaced property-name component validity remain owned
by `usd-identifiers-and-names`.

If the provider scanner interface would violate a declared path performance
target, a target may use a declared local fast path only as a performance
exception. That fast path must remain semantically bound to the identifier
scanner contract, pass the provider conformance subset it consumes, pass the
path conformance and performance targets, and be visible in the graph or target
report. Prefer improving the provider API or adding a provider-owned fast
entrypoint before forking local scanner logic.

Parsing must follow the normative `Path` grammar from the pinned spec for the
non-legacy grammar: absolute paths, relative paths, property-only relative
paths, namespaced properties, variant selections, whitespace allowed inside
variant selections, and UTF-8/XID identifiers. Compatibility-mode legacy path
extensions remain out of scope unless a golden explicitly adds them.

This skill also owns the Path handle contract in
`contracts/handles/path.handle.json`. That contract is the stable semantic
boundary for dependent skills and conformance tests. A `PathHandle` is an
opaque reference owned by a target-specific context; it is not a required class,
struct layout, pointer type, string representation, or product API.

This skill also owns the Path performance contract in
`contracts/performance/path.performance.json`. That contract is intentionally
limited to operations on pre-created path handles so parsing and string
validation costs do not hide whether the durable representation is structured.

Targets may implement whole paths as interned values, arena indices, parsed
segments, parent-linked nodes, compact IDs, strings with cached component
structure, or another representation. Reusable component names should be backed
by the graph's component identity dependency rather than an unrelated
parser-local string scheme. Targets must expose a conformance adapter for the
handle operations required by the contract:

- parse and canonical stringify
- kind query for root, prim, property, relative, and variant paths
- parent and final-name query
- child and property path construction
- equality and stable total ordering
- release of test-harness handles

## Implementation Quality Floor

Path equality, hashing, prefix checks, parent/name derivation, and component
iteration should be able to operate on canonical structure or component identity
instead of reparsing or rescanning complete path strings. Stringification may be
cached or assembled from components, but it should not be the only durable path
representation for implementations that depend on this skill.

The durable path representation should avoid retaining the same text at multiple
levels. Parser-local strings are acceptable while recognizing syntax, and a
single cached canonical path string may be retained for lookup or
stringification. Path nodes should not additionally retain raw component strings
when component identity handles are already available; prefer parent id, final
component identity, optional property identity, kind, depth, and cached hash.

The conformance benchmark for this skill compares shallow and deep path handles
for equality, hashing, parent lookup, final-name lookup, and fixed-prefix
checks. Passing the benchmark does not prescribe the native data structure, but
it does require the hot operations to avoid complete path-string rescans or
complete path-data copies once the handle exists.

## Boundary Guards

Defer identifier character-class validation to `usd-identifiers-and-names`.
Consume the `identifier_scanner` capability for identifier recognition embedded
inside path parsing. Do not locally define UTF-8 decoding policy, XID
start/continue policy, or identifier validity unless declaring and justifying a
performance exception.

Defer component identity, equality, hashing, and string conversion for repeated
path names to the graph-provided component identity dependency; do not create a
separate parser-local atom table for path component strings unless the target
explicitly declares why that domain is isolated and non-comparable with the
dependency contract.

Do not implement composition namespace mapping, anchoring across layer stacks,
asset resolution, relocates, or stage population.

Do not create a parser-local path representation inside `usda-spec-parser`.
Use the Path handle contract or generated target-native path dependency
instead.

## Test Obligations

- `/`
- `/Root`
- `/Root/Child`
- `/Root.attr`
- `/Root.namespaced:attr`
- `.`
- `.property`
- `..`
- `../..`
- `../.points`
- `Descendant`
- `/City{ selection = NewYork }` canonicalized as `/City{selection=NewYork}`
- `/City/Street{selection=5thAvenue}`
- `/City/Street{selection=}`
- UTF-8/XID identifiers such as `M\u00fcnchen`
- UTF-8/XID identifiers inside absolute paths, property names, and variant
  selections, such as `/M\u00fcnchen/Cr\u00e8me.caf\u00e9` and `/\u6771\u4eac{lod=\u9ad8}`
- invalid parent traversal in absolute path contexts
- element ordering for root/child/property names
- path component names represented through graph-provided canonical identity
- all cases in `goldens/unit/usd-paths/path-handle.json`
- all budgets in `benchmarks/path/targets.json`
