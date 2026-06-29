---
name: usd-foundational-values
description: Use this skill when implementing or verifying USD value storage independent of USDA syntax.
metadata:
  author: NVIDIA
---

# usd-foundational-values

Use this skill when implementing or verifying USD value storage independent of
USDA syntax.

## Spec Sources

- `specification/foundational_data_types/README.md`
- `specification/document_data_model/README.md` section `Metadata Fields > Allowed Types`

Pinned tag / commit: `v1.0.1`

## Provides

- Scalar values: bool, uchar, int, uint, int64, uint64, half, float, double,
  timecode, token, string, asset
- Dimensioned values: 2/3/4-tuples, matrices, quaternions
- Semantic alias agreement with underlying types
- Arrays of scalar and dimensioned values
- Dictionaries with string keys and heterogeneous values
- Value block sentinel

## Contract

This skill owns the foundational value model contract in
`contracts/handles/foundational-values.handle.json` and the foundational data
type catalog in `contracts/capabilities/foundational-data-types.json`.

Represent field values independently from their USDA spelling. Values emitted in
the canonical dump use the tagged representation from
`contracts/value.schema.json`.

Tagged JSON is an adapter and serialization representation, not durable value
storage. Implementations must store foundational values and parser raw values in
native typed payloads. Dictionaries should retain typed native entries and
materialize tagged JSON only when canonical output is requested.

Raw and durable values must preserve one observable active USD value type and
payload. The storage shape is implementation-defined: C++ targets may use
`std::variant`, a tagged union, enum-plus-payload storage, an arena-backed value
object, or a backend-native value container. Other targets should use the local
equivalent. The contract constrains typed inspection and checked reads, not the
physical memory layout.

Every value must expose its stored USD type identity without requiring adapter
JSON materialization. A typed read must either return the requested payload when
the requested type agrees with the stored type, or report a type mismatch. The
implementation must not recover type identity by reparsing a canonical
JSON/string dump.

Numeric foundational values must use native numeric payloads as their
authoritative typed storage. Integer types preserve signedness and width/range
semantics, floating types preserve numeric floating payloads, and dimensioned
numeric values/arrays preserve numeric element payloads. JSON number literals
may be rendered as strings inside adapter JSON machinery, but a generated value
model must not store numeric authored values as decimal strings and later parse
them back for typed reads.

The generated value model must preserve the complete foundational payload
catalog as typed data, even when several catalog entries share one backend
container. Inspection and checked reads must distinguish scalar domains and
width/precision (`uchar`, `int`, `uint`, `int64`, `uint64`, `half`, `float`,
`double`, `timecode`), dimensioned element type and shape, matrix shape/order,
quaternion component order, semantic alias role and underlying type,
homogeneous array element metadata, dictionary typed entries, and the
`valueBlock` sentinel.

The pure foundational data type catalog has no dependency on identifier parsing:
the type names are fixed spec names, not user-authored identifiers. Textual
parsers that recognize those names may depend on `usd-identifiers-and-names`,
but this value model should not.

This prototype still depends on `usd-tokens` for the `token` scalar because the
spec gives tokens registered-string identity and constant-time equality/hash
expectations. A foundational value implementation may store tagged token values,
token arrays, and token dictionary leaves, but token identity and performance
remain owned by `usd-tokens`.

For this prototype, value implementation must be sufficient for USDA
single-layer parsing and dumping. It does not need to evaluate splines or resolve
values across layer opinions.

## Boundary Guards

Do not depend on identifier scanning or keyword validation for the foundational
type catalog. Identifier grammar belongs to textual parsers.

Defer token interning, token equality/hash behavior, and Token handle
conformance to `usd-tokens`.

Defer authored list operation structure to `usd-listops-authored`.

Defer path value syntax and path equality to `usd-paths`.

Do not store `Json`, tagged JSON, or other adapter object trees as the backing
representation for raw or durable foundational values.

Do not make type inspection or typed reads depend on reparsing serialized
adapter JSON.

Do not store numeric foundational values as decimal strings in the durable value
model. Stringified numbers are permitted only at text/JSON serialization
boundaries.

Do not collapse dimensioned values, quaternions, matrices, semantic aliases, or
typed arrays to untyped numeric arrays for durable storage or typed reads.

Do not implement value resolution, interpolation, asset resolution, or schema
fallback evaluation.

## Test Obligations

- Exhaustive type catalog entries for scalar, dimensioned, array, dictionary,
  opaque, and semantic alias types
- Semantic role catalog entries for color, normal, point, vector, frame,
  texCoord, and group
- Semantic alias agreement with every alias's underlying type in both
  directions, including valid array aliases
- Numeric range and coercion cases covering every scalar type and integer
  boundary
- Tuple arity checks and valid values for every dimensioned type
- Arrays of every scalar, dimensioned, and non-opaque semantic alias type
- Nested dictionaries
- Recursive dictionary combining
- Storage model inspection proving tagged JSON is materialized on demand rather
  than retained as the authoritative raw or durable value representation
- Typed value inspection and checked typed reads, including type mismatch
  diagnostics
- Storage inspection proving numeric typed values are native numeric payloads,
  not string payloads
- Typed payload catalog inspection covering scalar domains, vector shapes,
  quaternions, matrices, semantic aliases, homogeneous arrays, dictionaries,
  and value blocks
- Asset strings rejecting C0/C1 control code points
- Value block sentinel for `None`
- all cases in `goldens/unit/usd-foundational-values/foundational-values.json`
