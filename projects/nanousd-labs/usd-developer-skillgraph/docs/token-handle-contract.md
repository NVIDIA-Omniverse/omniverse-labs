# Token Handle Contract

`contracts/handles/token.handle.json` is the second handle-shaped skill
boundary in the prototype.

## Spec Translation

The pinned specification says:

- `token` is a scalar foundational type whose fundamental type is a UTF-8
  encoded string.
- `token` and `string` have the same value specification, but runtimes may use
  different representations because token is optimized for expected usage.
- `token` remains a distinct USD value type even when the stored value domain
  is string-like.
- Tokens are expected to be commonly used strings optimized for hashing and
  equality.
- The glossary describes Token as a handle for a registered string that can be
  compared, assigned, and hashed in constant time.
- `token` is one of the fundamental types that must be supported by list
  operations.
- Field-specific restrictions, such as `allowedTokens`, are not enforced at the
  layer document level.

This means the token type itself should not reject empty strings, whitespace,
reserved USDA keywords, punctuation, or Unicode strings solely because they are
not identifiers. Those restrictions belong to callers, metadata fields, schema
logic, or USDA spelling rules.

## Contract Shape

`TokenHandle` is an opaque reference owned by a target-specific context. It is
not a required native type, pointer, intern-table index, string wrapper, or
product API.

Separately, the generated implementation should still expose Token as a native
domain concept. A target may store Token as an interned ID, pointer, atom table
slot, string handle, or another compact representation, but implementation code
should see a named Token abstraction or equivalent first-class value type rather
than raw primitive IDs as the main token representation.

Token identity is specified in terms of a compatible identity domain rather than
a mandated global registry. A process-wide registry, runtime-owned registry,
sharded registry, pointer-stable string pool, or native atom facility can all
satisfy the contract. The important constraint is that tokens which can be
compared by the implementation have canonical identity in that domain; an ID
from an unrelated local table is not enough unless those tokens cannot escape
and be compared elsewhere.

The threading floor is also explicit: created Token values are immutable and
safe to copy, assign, compare, and hash concurrently. Interning into a shared
identity domain must be safe under concurrent calls, or the target must declare
a weaker single-threaded model.

The conformance operations are:

- `intern`
- `intern_in_fresh_context`
- `to_string`
- `is_empty`
- `equal`
- `hash_equal`
- `release`

The generated adapters use synchronized shared intern registries in both Python
and C++ for this prototype, but that is an implementation choice. A target may
choose another representation if it satisfies the same observable contract,
identity-domain rules, threading model, and performance expectations.

## Validation

The token handle scope is:

```powershell
py harness/regen_graph.py --scope usd-tokens-handle --target python --validate
py harness/regen_graph.py --scope usd-tokens-handle --target cpp --validate
```

Both generated targets currently pass:

```text
OVERALL: 3/3 (100.0%)
```

The golden cases cover empty tokens, keywords, whitespace, Unicode, punctuation
that is not a valid identifier, case-sensitive equality, repeated interning,
hash agreement for equal tokens, and canonical identity for comparable tokens
created through independent conformance contexts.

## Performance

`contracts/performance/token.performance.json` captures the spec requirement
that existing token handles support constant-time equality, assignment, and
hashing. The benchmark target is `benchmarks/token/targets.json`.

The primary gate is a flatness ratio: long-token operation cost divided by
short-token operation cost for the same operation. Construction and lookup from
arbitrary text are intentionally excluded from this first performance gate.

Current generated targets pass both correctness and performance:

```text
Python: OVERALL 4/4, all flatness ratios under 1.5
C++:    OVERALL 4/4, all flatness ratios under 1.5
```

The ratio gate is the important evidence: operation cost stays flat when token
text grows from 8 bytes to 4096 bytes. Exact nanosecond timings vary by run and
machine.
