---
name: usd-identifiers-and-names
description: Use this skill when implementing or verifying USD identifier and name handling.
metadata:
  author: NVIDIA
---

# usd-identifiers-and-names

Use this skill when implementing or verifying USD identifier and name handling.

## Spec Sources

- `specification/document_data_model/README.md` sections `Specs > Names` and `Metadata Fields > Allowed Names`
- `specification/file_formats/README.md` sections `Keywords` and `Identifiers`

Pinned tag / commit: `v1.0.1`

## Provides

- Field names
- Prim names
- Variant set names
- Variant names
- Property names
- Namespaced property names
- ASCII prim type names
- Keywordless USDA identifiers
- Embedded-grammar identifier scanner capability

## Contract

Identifiers are UTF-8 / Unicode code point sequences constrained by the spec's
XID start/continue rules. ASCII-only shortcuts may be used only where the spec
requires ASCII, such as USDA prim type names.

Property names may contain namespace separators (`:`). Attribute and
relationship specs sharing a parent may not have the same property name.

This skill owns the reusable identifier scanner capability described by
`contracts/capabilities/identifier-scanner.json`. Dependent parsers should be
able to ask this skill to consume one identifier, one identifier-continuation
code point, or one namespaced property name from an offset in caller-owned text.
The scanner may return spans, offsets, views, or target-native name values; it
does not need to allocate strings.

The scanner capability owns UTF-8 decoding validity, XID start/continue policy,
keywordless identifier policy when requested, and namespaced-name component
validity. Callers own their surrounding grammar delimiters and diagnostics.

## Boundary Guards

Defer path parsing to `usd-paths`.

Defer USDA tokenization details such as comments, strings, numbers, and asset
refs to `usda-lexical-format`.

Do not silently accept reserved USDA keywords where the grammar requires a
keywordless identifier.

## Test Obligations

- Valid ASCII and Unicode identifiers
- Embedded UTF-8 identifier scanner spans
- Namespaced properties
- Keyword rejection in keywordless identifier contexts
- ASCII-only prim type names
- Scanner behavior for embedded identifiers, including returned end offsets and
  failures without consuming caller grammar delimiters
- all cases in `goldens/unit/usd-identifiers/identifier-scanner.json`
