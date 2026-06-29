---
name: usd-tokens
description: Use this skill when implementing or verifying USD token values.
metadata:
  author: NVIDIA
---

# usd-tokens

Use this skill when implementing or verifying USD token values.

## Spec Sources

- `specification/foundational_data_types/README.md` section `Scalar Types`
- `specification/foundational_data_types/README.md` section `List Operations`
- `specification/glossary/README.md` entry `Token`
- `specification/document_data_model/README.md` section `Value Ranges`
- `specification/file_formats/README.md` sections `Values` and Crate `token`

Pinned tag / commit: `v1.0.1`

## Provides

- Token values as UTF-8 encoded strings
- A first-class target-native Token value abstraction or equivalent semantic type
- Empty token values
- Token interning or equivalent registered-string behavior
- Equality and hashing suitable for token list operations
- Conversion from token handle to string text

## Contract

The `token` fundamental type has the same value domain as `string`: a UTF-8
encoded string. The token type exists because runtimes are expected to optimize
commonly used strings for hashing and equality. Tokens often serve the role of
enumerated values, but the type itself does not restrict values to identifiers,
keywords, non-empty strings, or strings without whitespace.

This skill owns the Token handle contract in
`contracts/handles/token.handle.json`. A `TokenHandle` is an opaque reference
owned by a target-specific context; it is not a required class, struct layout,
pointer type, intern-table index, string representation, or product API.

Targets may implement tokens as interned strings, atom table indices, arena
entries, reference-counted string handles, or plain strings where that target
still satisfies the conformance and performance expectations for its scope.
Those are storage strategies, not the public semantic shape of the generated
library. A target must keep `token` distinguishable from `string` in the USD
value model and must expose a named Token abstraction or equivalent first-class
value type for implementation code. Raw primitive IDs may be used internally,
but they must not be the primary token abstraction generated for the library.

Token equality and hashing are defined within a compatible token identity
domain. Implementations may realize that domain with a process-wide registry,
runtime-owned registry, sharded registry, pointer-stable string pool,
target-language atom facility, or another strategy. A token identity must not be
merely an ID from an unrelated local table unless tokens from that table cannot
escape or be compared with tokens from another table.

This skill also owns the performance contract in
`contracts/performance/token.performance.json`. Equality, hashing, and
assignment/copy of existing token handles must be flat with respect to token
text length. Interning from arbitrary text is deliberately measured separately
because construction must inspect the input text.

## Implementation Quality Floor

The generated implementation should make `Token` visible as a domain concept.
For targets with nominal types, `Token` should be a distinct class, struct,
newtype, alias with type-checking value, or equivalent native abstraction. For
dynamic targets, `Token` should be represented by a dedicated value object or
tagged value rather than an unadorned integer or string.

Token values should be cheap to copy or assign. Equality and hashing should use
registered identity where available, not re-scan token text. String conversion
may consult the registry/table/context that owns the token text.

Token values are immutable after creation and may be copied, assigned,
compared, and hashed concurrently without caller synchronization. Interning into
a shared token identity domain must either be safe under concurrent calls or the
target must explicitly declare a weaker single-threaded threading model for this
skill.

## Boundary Guards

Do not impose identifier, keyword, schema, or field-specific restrictions in the
token type. Those restrictions belong to the caller, metadata field, schema, or
USDA syntax layer.

Do not implement token listOp composition here. This skill only supplies token
identity, string conversion, equality, and hashing behavior needed by listOps.

Do not make USDA spelling decisions here. Parsing whether a token came from an
identifier or quoted string belongs to `usda-value-parser`.

## Test Obligations

- Empty token
- ASCII token
- Unicode token
- Token containing whitespace
- Token equal/hash behavior for repeated text
- Token inequality for distinct text
- First-class Token abstraction preserved separately from raw string/integer storage
- Token identity is canonical within the domain where Token values are comparable
- Token identity remains canonical when comparable tokens are produced through independent conformance contexts or facades
- Threading model for token values and interning is declared
- Release of test-harness handles
- performance targets in `benchmarks/token/targets.json`
