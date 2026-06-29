# Foundational Values Contract

`contracts/capabilities/foundational-data-types.json` captures the pinned
spec's foundational type catalog: scalar types, dimensioned types, semantic
aliases, arrays, dictionaries, value blocks, and list operation element-type
eligibility.

`contracts/handles/foundational-values.handle.json` is the unit conformance
surface for generated targets. It intentionally uses the same JSON adapter shape
as handle tests, but it does not require persistent handles.

## Dependencies

The foundational type catalog itself has no identifier dependency. Type names
such as `float3`, `matrix4d`, and `dictionary` are fixed spec names; identifier
scanning only belongs to textual parsers that read those names from a file.

The value model does depend on `usd-tokens` for the `token` scalar. The
foundational data types spec defines `token` as a UTF-8 string with optimized
comparison and hashing expectations, but the identity domain and performance
contract are owned by `usd-tokens`.

Tagged JSON is only the conformance adapter and serialization shape. Generated
targets should keep foundational values in native typed storage, keep dictionary
values as typed native entries, and materialize the tagged JSON objects only
when a test adapter or dump operation asks for canonical output.

The native storage shape is intentionally implementation-defined. A target may
use `std::variant`, a tagged union, enum-plus-payload storage, an arena-backed
value object, or a backend-native value container. The observable requirement is
that every value preserves one active USD type identity, exposes that type
without materializing adapter JSON, and supports checked typed reads.

Typed reads must either return the requested payload when the requested type
agrees with the stored type or report a type mismatch. Generated targets must
not recover type identity by reparsing a canonical JSON/string dump.

Numeric storage is also observable at the contract level. Authored numeric USD
values should be held as native numeric payloads with the appropriate
signed/unsigned/floating semantics. JSON number text is an adapter concern; it
must not be the backing representation that typed reads parse from.

The typed payload catalog is likewise observable. Implementations may share
storage containers internally, but inspection and checked reads must still
distinguish every foundational scalar domain and width/precision, dimensioned
shape, matrix shape/order, quaternion component order, semantic alias role and
underlying type, homogeneous array element type, dictionary entries, and the
value block sentinel. In particular, `half`, `float`, `double`, `timecode`,
`quatf`, `matrix4d`, `point3f`, and `point3f[]` cannot collapse to an
untyped number or array surface.

## Scope

The current unit scope is intentionally exhaustive over the bounded catalogs in
this skill. It validates:

- every scalar, dimensioned, semantic alias, dictionary, and opaque catalog
  entry
- every allowed array type and the disallowed opaque/dictionary array surfaces
- semantic role catalog entries for color, normal, point, vector, frame,
  texCoord, and group
- semantic alias agreement with each alias's underlying type in both
  directions, including valid alias arrays
- scalar canonical values and integer boundary ranges
- every dimensioned and non-opaque semantic alias value shape
- every allowed array value shape
- asset string C0/C1 control-code rejection
- value block tagging
- recursive dictionary combining
- storage model inspection for on-demand JSON materialization
- storage policy inspection for implementation-defined typed value containers
- typed value inspection and checked typed reads, including mismatch
  diagnostics
- native numeric payload storage for scalar, dimensioned, and array numeric
  values
- complete typed payload catalog inspection for scalars, vectors, quaternions,
  matrices, semantic aliases, arrays, dictionaries, and value blocks

It defers full list operation algebra to `usd-listops-authored` and all
specialized document-model types to their owning skills.

```powershell
py harness/regen_graph.py --scope usd-foundational-values-unit --target python --validate
py harness/regen_graph.py --scope usd-foundational-values-unit --target cpp --validate
```
