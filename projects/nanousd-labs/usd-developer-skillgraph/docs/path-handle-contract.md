# Path Handle Contract

`contracts/handles/path.handle.json` is the first prototype of a handle-shaped
skill boundary.

The important distinction is that `PathHandle` is not a required native type.
It is an opaque reference owned by a target-specific context during a
conformance run. A target can implement paths as strings, interned IDs, arena
indices, parsed segment vectors, pointers, or another representation.

`usd-paths` depends on a component identity capability. In this prototype graph
that provider is `usd-tokens`, but the path contract is not a requirement to
store or expose concrete Token values. A target may use any native
representation that provides stable component equality, hashing, and string
recovery.

`usd-paths` also consumes the identifier scanner capability from
`usd-identifiers-and-names`. That is an architectural contract, not just a
generation-order dependency: path parsing should not independently define UTF-8
decoding or XID identifier validity. A target-local scanner fast path is allowed
only as a declared performance exception, and it remains semantically bound to
the identifier scanner contract.

## Context

In this prototype, `Context` means the state owned by the adapter invocation. It
creates handles, maps test names to implementation-owned values, and releases
test handles. It could become an allocator, interner, registry, or no-op wrapper
in a real target.

The product library does not need to expose this context as its public API.

## Contract Shape

The contract tests behavior through operations over opaque handles:

- `parse`
- `to_string`
- `kind`
- `parent`
- `name`
- `append_child`
- `append_property`
- `equal`
- `compare`
- `release`

The stable API is therefore a conformance adapter, not a native path class.
Dependent skills can rely on the same semantic capability without dictating the
internal path representation.

The implementation quality floor is stricter than the adapter shape. Path
components should be canonicalized, and hot operations such as equality,
hashing, prefix checks, parent/name derivation, and component iteration should
not require reparsing, rescanning complete path strings, or copying complete
path data. The generated prototype uses the graph's component identity provider
and a whole-path identity registry while keeping canonical path text available
for stringification and ordering. Durable path nodes retain compact identity
fields rather than a second copy of raw component strings; parser strings are
temporary recognition state.

`contracts/performance/path.performance.json` captures that quality floor as a
separate performance contract. The benchmark operates only on pre-created
handles so parse and validation cost does not mask the durable representation.

## Validation

The path handle scope is:

```powershell
py harness/regen_graph.py --scope usd-paths-handle --target python --validate
py harness/regen_graph.py --scope usd-paths-handle --target cpp --validate
```

This now runs both the handle goldens and `benchmarks/path/targets.json`. Both
generated targets currently pass the handle goldens and path hot-operation
budgets:

```text
OVERALL: 6/6 (100.0%)
```

The existing single-layer parser goldens still pass for both targets after the
Path handle work:

```text
OVERALL: 6/6 (100.0%)
```

## Current Scope

This handle contract now covers the pinned spec's non-legacy `Path` grammar for
root, absolute prim/property paths, relative paths, property-only relative
paths, namespaced properties, variants, and UTF-8 identifiers. Construction
helpers remain intentionally narrower: child/property construction is for
absolute prim/property ObjectPath-style handles. Compatibility-mode legacy
extensions, relational attributes, mapper expressions, path expressions, and
composition-time namespace mapping remain deferred.
