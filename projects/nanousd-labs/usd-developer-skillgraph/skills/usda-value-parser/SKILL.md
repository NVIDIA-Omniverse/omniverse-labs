---
name: usda-value-parser
description: Use this skill when implementing or verifying USDA right-hand-side values and attribute type validation.
metadata:
  author: NVIDIA
---

# usda-value-parser

Use this skill when implementing or verifying USDA right-hand-side values and
attribute type validation.

## Spec Sources

- `specification/file_formats/README.md` sections `Values`, `Common Metadata`,
  `Attribute Values`, and `Attribute Types and Value Validation`
- `specification/foundational_data_types/README.md`

Pinned tag / commit: `v1.0.1`

The pinned specification is normative for value grammar. This skill may add
target-native representation, diagnostics, and performance constraints, but it
must not broaden where a value form is legal. Consumers decide whether `None`,
PathRef, AssetRef, listOp, time sample, or spline syntax is permitted in their
current production context.

## Provides

- Atomic values
- Tuple values
- List values
- Dictionary values
- Path refs
- Asset refs
- `None` as value block where appropriate
- Type-directed validation and coercion for attribute defaults

## Contract

This skill owns `contracts/handles/usda-value-parser.handle.json` and provides
the `usda_value_parser` capability consumed by the USDA spec parser. It consumes
USDA lexical services, foundational value storage, path references, and authored
listOp storage from the graph rather than defining local equivalents. It also
follows `contracts/capabilities/semantic-runtime-types.json`.

Production-family obligations are factored into:

- `contracts/usda-productions/value-syntax.contract.json`

The value parser returns target-native semantic values for the spec parser:
typed raw values, dictionaries with typed entries, path handles, authored
listOps, time-sample maps, spline maps, and value-block sentinels as
appropriate for the target language. It must not use `Json`, tagged JSON,
Python `dict`, `std::map<string, Json>`, or canonical dump object trees as the
domain value representation. Validation adapters may serialize these values to
`contracts/value.schema.json`, but that serialization is not the parser
boundary.

Value parsing supplies production building blocks; it does not decide higher
level grammar context. For example, parsing a path reference is valid only when
the caller's production allows a PathRef, and parsing `None` is valid only when
the caller maps it to a value block or empty authored listOp.

When parsing an attribute declaration and default assignment, validate the
parsed value against the declared type if the type is known in the foundational
type table.

Unknown attribute types have undefined validation behavior in the spec. For this
prototype, preserve the value and emit no type-validation error unless a golden
requires otherwise.

## Boundary Guards

Defer text lexing to `usda-lexical-format`.

Defer scalar, array, dictionary, semantic alias, and value-block storage to
`usd-foundational-values`.

Defer path parsing to `usd-paths`.

Defer authored listOp storage to `usd-listops-authored`.

Path spellings, token spellings, and name spellings parsed from USDA text are
boundary strings. Convert them to target-native path, token, or semantic name
values before returning domain data to `usda-spec-parser`.

Do not return JSON-shaped values to `usda-spec-parser`; return target-native
typed value objects or apply a clearly typed adapter only at test/dump output.

Do not implement value resolution or spline evaluation.

Time sample maps and spline maps are specialized authored values. They should
be returned as target-native domain records: time samples keyed by numeric time
ordinates with typed field values/value blocks, and splines with typed
extrapolation, loop, knot, tangent, and interpolation members. Do not return
Python dict/list payloads, `std::map<string, Json>`, tagged JSON, or canonical
dump object trees as the domain representation for these values.

## Test Obligations

- scalar defaults
- tuple defaults
- arrays
- dictionaries
- string dictionaries
- path references and asset references
- `None` value block for attribute defaults
- time sample maps as stored authored values
- spline maps as stored authored values
- type mismatch diagnostic
- numeric coercion where allowed by attribute type validation
