# USDC Spec Errata

The AOUSD 1.0.1 Core Specification documents the USDC Crate binary format in
§16.3. In two places, the published text is ambiguous or inconsistent with what
real `.usdc` files (those produced by the reference USD writer) contain on disk.
This document pins the interpretation a conformant AOUSD 1.0.1 reader must use
in this repository, and records the source of the divergence so a future spec
revision can address it upstream.

Pinned commit: `6b4f01f2d6d46b01507481524eb516b4ef358c31`.

## 1. Value-representation flag bits live in the high bits of byte 7

### What the spec says

§16.3.9 ("Value Representations") describes the 8-byte value-representation
word and the three flag bits in this prose:

> The possible bit masks are
>
> | Description           | Value             |
> |-----------------------|-------------------|
> | Value is an Array     | The last bit      |
> | Value is Inlined      | The second last bit |
> | Value is Compressed   | The third last bit  |

"Last bit" is positionally ambiguous: it can be read as the least-significant
bit (bit 0) or as the most-significant bit (bit 63) of the 8-byte
little-endian word. The contract previously took the LSB reading and listed
`array_bit = bit 0 (least significant)` etc.

### What real `.usdc` files contain

The reference USD writer's `pxr/usd/usd/crateFile.h` defines the value
representation type as:

```cpp
struct _ValueRep {
    constexpr static uint64_t _IsArrayBit      = 1ull << 63;
    constexpr static uint64_t _IsInlinedBit    = 1ull << 62;
    constexpr static uint64_t _IsCompressedBit = 1ull << 61;
    // ...
};
```

When the value-representation word is read as a little-endian `uint64`, byte 7
is the most-significant byte. Pixar's three flag bits land in the top three
bits of byte 7. Read as masks on the flags byte (`flags = word >> 56`):

| Flag                  | Bit position in the LE word | Mask on byte 7 |
|-----------------------|------------------------------|----------------|
| `Value is an Array`   | bit 63                       | `0x80`         |
| `Value is Inlined`    | bit 62                       | `0x40`         |
| `Value is Compressed` | bit 61                       | `0x20`         |

Every checked-in `.usdc` fixture in this repository (and every `.usdc` file
produced by the reference writer) uses this layout. The spec's "last bit"
phrasing therefore refers to bit 63, not bit 0.

### What conformant readers must do

A reader of AOUSD 1.0.1 `.usdc` files in this repository:

- treats `flags & 0x80` as the array bit, `flags & 0x40` as the inlined bit,
  and `flags & 0x20` as the compressed bit, where `flags` is byte 7 of the
  8-byte little-endian value-representation word;
- must not also accept `flags & 0x01 / 0x02 / 0x04` as a fallback —
  "tolerantly accepting both layouts" is itself a contract violation because
  it hides the discrepancy and lets one decoder pass two contradictory test
  suites;
- rejects the flag combinations `Inlined + Array`, `Inlined + Compressed`,
  and `Inlined + Array + Compressed` with `MalformedUsdcValueRepresentation`
  before any payload decoding runs (the combinations themselves are unchanged
  by this errata; only the bit positions differ).

`contracts/usdc-productions/crate-value-representations.contract.json`
describes the high-bit layout as normative and cites this errata. The unit
golden `goldens/unit/usdc-value-decoder/value-decoder.json` uses the high-bit
encoding for every `word_hex` value.

## 2. LZ4 envelope uses `TfFastCompression` framing, not the spec's u64+u64 framing

### What the spec says

§16.3.7.1 ("LZ4 Compression") describes the envelope as:

> | Description       | Size    | Value                       |
> |-------------------|---------|-----------------------------|
> | Number of Chunks  | 8 bytes | Unsigned 64-bit integer     |
> | Chunk Size .      | 8 bytes | Unsigned 64-bit integer     |
> | Chunk Data        | …       | LZ4 stream                  |

Read literally, this says the chunk count is a `uint64` and each chunk size is
a `uint64`.

### What real `.usdc` files contain

The reference USD writer uses `pxr_lz4::CompressToBuffer` /
`DecompressFromBuffer` from `pxr/base/tf/fastCompression.{h,cpp}`. That format
is:

- 1 byte `uint8` chunk count;
- if chunk count == 0, the remainder of the buffer is one LZ4 frame;
- if chunk count > 0, for each chunk: 4 bytes `int32` compressed chunk size,
  then that many LZ4-compressed bytes.

Every checked-in `.usdc` fixture in this repository uses this `u8 + i32`
layout for the LZ4 envelope wrapping the TOKENS data, FIELDS value-rep blob,
and the LZ4-compressed inner payload of each compressed integer array. The
spec's u64+u64 framing is not what the reference writer produces.

### What conformant readers must do

A reader of AOUSD 1.0.1 `.usdc` files in this repository:

- accepts the `TfFastCompression` framing (`u8` chunk count, per-chunk
  `int32` compressed size) when decompressing any LZ4-compressed section
  payload or any compressed-integer-array inner LZ4 stream;
- must not accept the spec-text u64+u64 framing as a fallback for the same
  reasons given in §1 above;
- treats the outer 8-byte `compressed_size` prefix of a compressed integer
  array (per `crate-compression.contract.json#compressed_integer_array.outer_envelope`)
  as the only 8-byte length in the structure; the LZ4 frame *inside* that
  prefix uses `TfFastCompression` framing.

`contracts/usdc-productions/crate-compression.contract.json` describes the
`TfFastCompression` framing as normative for `.usdc` content read by this
repository and cites this errata.

## Spec resolution

Both items are reader-only divergences. Writers in this repository remain out
of scope (`usdc-single-layer.yaml#excluded` lists `USDC writing`), so there is
no asymmetric-format risk. If a future AOUSD spec revision moves either
clause to match the reference implementation, this errata document and the
contract cross-references can be removed; the code paths the contracts pin
will already match.
