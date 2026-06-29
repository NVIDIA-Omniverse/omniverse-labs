---
name: usd-document-model
description: Use this skill when implementing or verifying USD layer/spec/field storage.
metadata:
  author: NVIDIA
---

# usd-document-model

Use this skill when implementing or verifying USD layer/spec/field storage.

## Spec Sources

- `specification/document_data_model/README.md`
- `specification/file_formats/README.md` sections that map USDA syntax to
  document-model fields

Pinned tag / commit: `v1.0.1`

The document model owns layer/spec/field storage after syntax has been accepted
by a format parser. It must not be used as a substitute grammar oracle: the fact
that a core field exists, such as `payload` or `references`, does not make that
field name valid in every USDA metadata context.

## Provides

- Layer object
- Layer spec at `/`
- Prim specs
- Attribute specs
- Relationship specs
- Field storage by `(spec path, field name)`
- Tokenized field-name vocabulary for known core metadata fields
- Required field authoring for parsed specs
- Child list maintenance
- Duplicate spec path rejection

## Contract

This skill owns `contracts/handles/document-model.handle.json` and the
document-model capability contract in
`contracts/capabilities/document-data-model.json`. It also follows the
cross-cutting semantic runtime type contract in
`contracts/capabilities/semantic-runtime-types.json`.

A layer is a collection of specs. Each spec except the layer spec has a unique
absolute path. The layer spec is addressed as `/`.

The parser must keep child-list fields in agreement with created specs:

- layer and prim `primChildren`
- prim `properties`
- prim `variantSetChildren`

For this prototype, variant set/spec storage can be deferred until goldens add
it.

Metadata field names are not just incidental strings. The document data model
specification calls out field-name redundancy as something performant formats
should exploit. Generated targets should therefore intern field names through the
graph's token identity capability or an equivalent field atom mechanism. Known
core, reserved, and deprecated metadata field names from the document data model
must be predeclared as field tokens. Unknown extension fields may still be
accepted, but they must be validated as field names and interned into the same
field-name identity domain.

Format parsers are responsible for validating whether a field spelling is legal
in the current file-format production before storing it. The document model can
store valid layer data from any format, but it does not authorize USDA keyword
syntax or context-specific metadata productions.

Tagged `token` values, `token[]` values, and child/name-list fields such as
`primChildren`, `properties`, `variantSetChildren`, `primOrder`, and
`propertyOrder` should also be held through token identity internally. Parser and
JSON conformance boundaries may expose string spellings, but the durable layer
model should not use raw strings as the only identity for these values.

Do not use JSON as the durable field value model. Generated targets may expose
canonical tagged JSON in adapters and dumps, but specs should store a
target-native `FieldValue`-style variant or equivalent value container that can
represent foundational data types, token/token-array values, dictionaries, value
blocks, and the specialized document-model field values in scope. Conversion to
JSON belongs at the conformance or serialization boundary.

Document-model field lookup must preserve typed readability. A caller must be
able to inspect whether an authored field exists, retrieve the field's stored
USD/document-model type identity, and perform a checked typed read without
round-tripping through canonical JSON. A typed field read succeeds only when the
requested type agrees with the stored value type; otherwise it reports a type
mismatch.

Specialized document-model values in this scope are domain values, not generic
string/dictionary payloads:

- `specifier`: native enum/domain value for `def`, `over`, or `class`
- `variability`: native enum/domain value for `varying` or `uniform`
- `Retiming` / `Retiming[]`: native layer-offset records with offset and scale
- `Relocates`: native map from source path reference to target path reference
- `TimeSamples`: native numeric-time map to field values/value blocks
- `Spline`: native spline record with extrapolation, loop, knot, tangent, and
  interpolation members

Do not retain a `Json` value as the fallback/raw payload for unmodeled field
types. That needlessly increases resident memory and couples the layer model to
the adapter representation. If a target needs a generic fallback for values not
yet modeled as dedicated variants, it should store a small native payload tree or
typed byte/value representation and materialize JSON only when dumping or
serving an adapter response.

The contract does not require a particular backing layout. `std::variant`,
tagged unions, enum-plus-payload containers, arena-owned values, and
backend-native field containers are all acceptable when they preserve the typed
read semantics above.

Numeric field values must preserve native numeric payloads inherited from the
foundational value model. The document model may serialize numbers as JSON
number literals when dumping, but it must not retain stringified numeric
payloads as the authoritative field storage used by typed reads or composition
helpers such as layer offsets.

Foundational field values must preserve the complete foundational typed payload
catalog. A backend may use one compact value container, but inspection and
checked reads must still distinguish scalar domains and widths, `half`,
`float`, `double`, `timecode`, dimensioned values, quaternions, matrices,
semantic aliases, homogeneous arrays, dictionaries, and `valueBlock` sentinels.
These values must not be collapsed to untyped arrays or adapter JSON before
composition, stage population, or generated backend APIs consume them.

Domain-facing APIs should also surface token-typed inputs as token values where
the spec says the value is a token. For example, prim `typeName` and child
property names should be interned before durable storage; a parser or adapter may
accept strings and convert them to tokens before calling the domain API.

Spec identity must consume the path capability directly. Domain-facing document
model APIs should accept and store path handles/references, not plain strings.
Adapters and parsers can parse textual paths at their boundary and stringify
paths for diagnostics or canonical dumps. Path parsing, stringification, and
parent/name construction belong to `usd-paths`; do not add `Layer` operations
that wrap those concepts as if they were document-model semantics.

The same rule applies to runtime records owned by downstream composition,
stage-population, or backend code. If a value is semantically a USD path, store
the path handle/reference as the authoritative value. If a value is a field key,
USD token value, child/property name, schema/type name, variant set name, or
selected variant name used for identity decisions, store a token/name handle or
equivalent target-native atom as the authoritative value. Keeping a string for
adapter output or diagnostics is fine only when it is derived from those
semantic handles and does not become the value used for lookup, comparison, or
composition.

## Boundary Guards

Defer path identity and path construction to `usd-paths`.

Defer token identity to `usd-tokens`.

Defer non-token field value representation to `usd-foundational-values` and
`usd-listops-authored`.

The foundational data types spec defines list operation eligibility and algebra,
but this graph keeps authored listOp structure in `usd-listops-authored`.
Document-model implementations store listOp values as fields; they do not own
listOp construction or combination.

Do not filter inactive prims. `active` has no semantic effect at the layer
level.

Do not perform composition, value resolution, asset resolution, or stage
population.

## Test Obligations

- layer spec exists at `/`
- root prim child list
- nested prim child list
- property child list
- field-name token vocabulary and token-backed child/name lists
- typed field existence/type inspection and checked field reads
- native numeric payload storage for numeric field values
- complete foundational typed payload catalog preservation inside field values
- required spec fields from USDA declarations
- duplicate spec path error
- missing or invalid parent error without partial mutation
- all cases in `goldens/unit/usd-document-model/document-model.json`
