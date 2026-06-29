---
name: usda-lexical-format
description: Use this skill when implementing or verifying the USDA text-format substrate.
metadata:
  author: NVIDIA
---

# usda-lexical-format

Use this skill when implementing or verifying the USDA text-format substrate.

## Spec Sources

- `specification/file_formats/README.md` section `Text`

Pinned tag / commit: `v1.0.1`

## Provides

- UTF-8 input handling
- `#usda` layer header parsing
- Spaces, CR/LF/CRLF, and statement separators
- Python-style and C/C++-style comments
- Single-line and multi-line strings
- Number literals including `inf`, `-inf`, and `nan`
- USDA identifiers
- Asset references
- Path reference delimiters

## Contract

This skill owns `contracts/capabilities/usda-lexical-format.json` and provides
the `usda_lexical_format` capability consumed by USDA value and spec parsing.
It consumes the `identifier_scanner` capability from
`usd-identifiers-and-names` for identifier validity; it should not maintain an
independent UTF-8 or XID policy.

The lexer/parser substrate should preserve enough location information to emit
line/column diagnostics, but exact diagnostic wording is controlled by goldens.

The substrate may be a lexer, parser-combinator layer, or direct recursive
descent helpers. The contract is behavioral, not architectural.

## Boundary Guards

Defer identifier UTF-8/XID scanning to `usd-identifiers-and-names`.

Defer path grammar after stripping angle brackets to `usd-paths`.

Defer value typing/coercion to `usda-value-parser`.

Do not map syntax to document-model fields here; that belongs to
`usda-spec-parser`.

## Test Obligations

- valid layer header
- comments in all whitespace positions
- strings and escapes
- numbers
- asset references
- invalid header diagnostic
