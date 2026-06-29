---
name: usdc-binary-format
description: Use this skill when implementing or verifying the low-level USDC Crate binary reader.
metadata:
  author: NVIDIA
---

# usdc-binary-format

Use this skill when implementing or verifying the low-level USDC Crate binary
reader.

## Spec Sources

- USD Core Spec Core File Formats sections `Binary` and `Crate`
- USD Core Spec supported Crate version range for AOUSD 1.0.1
- `docs/usdc-spec-errata.md` — pinned interpretations where the AOUSD prose is
  ambiguous or diverges from real `.usdc` content. In particular: the LZ4
  envelope uses `TfFastCompression` framing (u8 chunk count, per-chunk i32
  size), not the spec text's u64+u64 framing.

Pinned commit: `6b4f01f2d6d46b01507481524eb516b4ef358c31`

## Provides

- Byte cursor with strict bounds checks
- Little-endian integer and floating-point reads
- `PXR-USDC` header and version validation
- Crate bootstrap and table-of-contents reads
- LZ4 buffer decompression boundary
- Compressed integer array decoding

## Contract

This skill owns only Crate byte-level decoding. It accepts an already-opened
resource byte view from a layer-format handler and returns typed low-level Crate
sections, buffers, and primitive arrays for higher-level USDC nodes.

The reader must reject malformed sizes, offsets, unsupported versions,
truncated data, integer overflows, invalid compression metadata, and unknown
required encodings with diagnostics before higher-level section mapping starts.

This skill does not map decoded bytes into USD specs, fields, paths, tokens, or
canonical layer dumps.

## Boundary Guards

Defer resource opening to `usd-resource-protocol` through `usd-layer-open`.

Defer token, string, path, field, field-set, and spec interpretation to
`usdc-value-decoder` and `usdc-spec-parser`.

Do not call USDA text lexers, USDA value parsers, or USDA spec parsers.

Do not implement package extraction, `.usd` forwarding dispatch, sublayer
loading, composition, value resolution, schema fallback, or USDC writing.

## Test Obligations

- reject non-`PXR-USDC` data with a clear diagnostic
- reject unsupported Crate versions
- accept Crate version triples with major 0, minor in 8 through 12, any patch; reject everything else before any section interpretation
- reject section offsets and sizes outside the payload
- expose validated section bytes as non-owning views into the opened resource
- address the six standard TOC sections (TOKENS, STRINGS, FIELDS, FIELDSETS, PATHS, SPECS) by section kind in constant time after `read_toc`
- decode uncompressed and LZ4-compressed byte buffers used by fixture sections
- decode at least one multi-chunk LZ4 buffer so single-chunk-only implementations fail the unit
- decode compressed integer arrays used by fixture sections
- decode at least one compressed integer array that exercises 2-bit codepoints 01, 10, and 11 in the same payload
- read every numeric field with a single little-endian word read; byte-at-a-time scalar loops in the hot path are a contract violation
- must not expose a helper that searches the full opened resource as text, since such helpers exist only to bypass section traversal

## Performance

The binary-format unit must satisfy
`contracts/performance/usdc.performance.json`. Header and TOC validation must
be O(section_count); standard section lookup must be O(1); section bytes must
be non-owning views; LZ4 and compressed integer arrays must run at decoder
throughput, not at byte-at-a-time interpretation rates.
