---
name: usdz-package-format
description: Use this skill when implementing or verifying USDZ package indexing and entry byte access.
metadata:
  author: NVIDIA
---

# usdz-package-format

Use this skill when implementing or verifying USDZ package indexing and entry
byte access.

## Spec Sources

- OpenUSD USDZ File Format Specification v1.3: USDZ is an uncompressed,
  unencrypted ZIP archive; file data begins at 64-byte-aligned offsets; opening
  `package.usdz` uses the first file in the package as the default layer.
- USD Core Spec Resource Interface sections covering package resources.
- `docs/usdz-spec-errata.md` pins three reader-only positions where the AOUSD
  §16.4 package text diverges from the reference writer or is relaxed by
  §16.4.2: (1) 64-byte alignment applies to entry *data*, not the local file
  header; (2) a non-zero EOCD comment is tolerated; (3) the default layer is the
  first *physical* entry.

Pinned AOUSD commit: `6b4f01f2d6d46b01507481524eb516b4ef358c31`

## Provides

- `UsdzArchiveIndex`
- `UsdzEntry`
- `PackageEntryByteView`
- EOCD, central directory, and local header validation for standard ZIP archives
- stored/no-encryption checks
- 64-byte entry data alignment checks
- normalized entry names and duplicate-entry diagnostics
- default-layer selection and explicit-entry lookup

## Contract

This skill owns `contracts/handles/usdz-package-format.handle.json` and
`contracts/usdz-productions/`.

The input is already-opened outer package bytes from `usd-resource-protocol`.
This skill never opens files, extracts package entries to temporary paths, or
hands out owned copies of selected entries when the outer package bytes remain
available. It returns non-owning byte views over the package storage.

The archive reader covers standard non-Zip64 USDZ. It rejects encrypted entries,
compressed entries, unsafe paths, duplicate normalized paths, out-of-bounds
headers or payloads, and entries whose file data does not start on a 64-byte
boundary.

Opening a whole package selects the first physical file entry by local-header
offset. It must not scan later package entries looking for a `.usda` or `.usdc`
file.

## Boundary Guards

Defer outer resource reads and package spelling split to `usd-resource-protocol`.

Defer `.usda` and `.usdc` layer parsing to `usdz-layer-open` and the concrete
format handlers it dispatches to.

Do not implement USDA parsing, USDC Crate parsing, nested package opening,
package writes, composition, or asset resolution in this skill.

## Test Obligations

- index a valid stored, aligned package
- select the first physical entry as the default layer
- look up an explicit normalized entry
- reject invalid ZIP, compressed entries, encrypted entries, misaligned entries,
  duplicate names, unsafe paths, and missing explicit entries
- expose selected entry bytes through a non-owning view
