---
name: usd-listops-authored
description: Use this skill when implementing authored list operation values as they appear in a single layer.
metadata:
  author: NVIDIA
---

# usd-listops-authored

Use this skill when implementing authored list operation values as they appear
in a single layer.

## Spec Sources

- `specification/foundational_data_types/README.md` section `List Operations`
- `specification/file_formats/README.md` section `A General Note on Listops`

Pinned tag / commit: `v1.0.1`

## Provides

- Explicit listOp items
- Prepended listOp items
- Appended listOp items
- Deleted listOp items
- `None` as empty listOp outside attribute value assignment
- Last-one-wins behavior for repeated same listOp subfield

## Contract

This skill owns `contracts/handles/listops-authored.handle.json` and stores
authored list operations. It does not combine list operations across layer
opinions.

Implementations should keep listOps in native authored storage with one item
list per authored subfield: explicit, prepend, append, and delete. Tagged JSON
is only the adapter/dump representation and should be materialized on demand.

The core listOp value model stores typed items, not parsed strings. The native
item representation is:

- ObjectPath: path handle/reference from `usd-paths`
- Reference: native reference arc record with optional asset reference, optional
  prim path handle, offset, and scale
- Payload: native payload arc record with optional asset reference, optional
  prim path handle, offset, and scale
- token: token handle/reference from `usd-tokens`
- string: native string
- int/int64/uint/uint64: fixed-width integer values

Canonical dump shape:

```json
{
  "type": "listop<ObjectPath>",
  "value": {
    "explicit": ["/A"],
    "prepend": [],
    "append": [],
    "delete": []
  }
}
```

## Boundary Guards

Defer item equality for paths to `usd-paths`.

Defer token identity to `usd-tokens`.

Defer scalar eligibility and integer width/range rules to
`usd-foundational-values`.

Do not implement cross-layer listOp composition or value resolution.

Do not retain tagged JSON as the backing listOp representation.

Do not parse USDA text syntax in this capability. USDA syntax belongs to the
USDA parser skills.

Do not make JSON parsing or JSON-number/string coercion part of the core listOp
model. JSON coercion belongs only in the handle adapter.

Do not store ObjectPath, token, or numeric listOp items as durable strings.
Strings are the native storage type only for `listOp<string>`.

Do not store `listOp<Reference>` or `listOp<Payload>` items as dictionaries,
bare asset strings, or bare path strings. Those shapes are parser input or
dump-adapter output only; the domain item is a typed arc record.

## Test Obligations

- implicit explicit listOp from a single authored list
- append/prepend/delete syntax
- repeated same subfield last-one-wins
- empty listOp defaults
- object path item validation through `usd-paths`
- reference and payload arc item storage as typed native records, not strings or
  dictionaries
- foundational element type validation for token, string, int, int64, uint, and
  uint64
- all cases in `goldens/unit/usd-listops-authored/listops-authored.json`
