# USDZ Spec Errata

The AOUSD 1.0.1 Core Specification documents the USDZ package format in §16.4
("Package"). USDZ is a constrained ZIP archive: stored (uncompressed),
unencrypted, non-Zip64, with a fixed alignment rule and a restricted
end-of-central-directory record.

Two of those clauses are either inconsistent with what real `.usdz` files (those
produced by the reference USD writer) contain on disk, or are deliberately
relaxed by §16.4.2 ("Validation and Out of Spec USDZ files"), which states that
readers are *not required* to validate a package and *may* accept out-of-spec
files. This document pins the interpretation the USDZ reader in this repository
uses, and records the source of each divergence so a future spec revision can
address it upstream.

These are all **reader-only** positions. No USDZ writer is in scope
(`graph/scopes/usdz-single-layer.yaml#excluded` lists `USDZ writing`), so there
is no asymmetric-format risk: the reader is lenient where the spec text and the
reference implementation disagree, and never emits a non-conformant package.

Pinned commit: `6b4f01f2d6d46b01507481524eb516b4ef358c31`.

## 1. 64-byte alignment applies to entry data, not the local file header

### What the spec says

§16.4.1.3 ("Data Layout"):

> USD Packages must be aligned to 64 byte blocks, such that every **file header**
> starts at a multiple of 64 bytes for efficient reading.

Read literally, this requires each entry's *local file header* to begin at an
offset that is a multiple of 64.

### What real `.usdz` files contain

The reference USD writer (`pxr/usd/usd/zipFile.cpp`,
`UsdZipFileWriter::AddFile`) aligns the **file data**, not the header, to 64-byte
boundaries. It does this by padding the *extra* field of the local file header so
that the bytes of the entry's content begin at a 64-byte multiple. The header
itself is free to start at an unaligned offset. This is intentional: USD memory-
maps and reads the asset *data* directly, so it is the data payload that must be
aligned for efficient (and mmap-friendly) access.

A package written this way routinely has entries whose local-header offset is
*not* 64-aligned while the data offset *is*. For example, in
`benchmarks/fixtures/usdz/explicit_root_usda.usdz` the second entry
(`scenes/root.usda`) has `local_header_offset = 83` (not a multiple of 64) and
`data_offset = 192` (= 3 × 64). A header-aligned reading would reject this valid
reference-writer package; the data-aligned reading accepts it.

The fixture generator encodes the same convention:
`benchmarks/make_usdz_fixtures.py::local_padding` computes
`pad = (-(local_header_offset + 30 + len(name))) % 64` and emits it as the local
extra field, aligning the *data* offset.

### What conformant readers must do

The USDZ reader in this repository:

- treats the 64-byte alignment requirement as applying to each entry's **data
  offset** (`data_offset % 64 == 0` in `generated/cpp/usdz_package.cpp`), where
  `data_offset = local_header_offset + 30 + local_name_len + local_extra_len`;
- computes that offset from the **local** file header's name and extra lengths
  (which carry the alignment padding), not the central directory's, since the
  two extra fields legitimately differ;
- does **not** require the local-header offset itself to be 64-aligned;
- emits `MisalignedUsdzEntryData` when (and only when) the data offset is not
  64-aligned.

`contracts/usdz-productions/package-layout.contract.json#usdz_entry_constraints`
states "file data offset is 64-byte aligned" and the
`usdz-package-format` skill lists "64-byte entry data alignment checks"; both
cite this errata. The unit golden
`goldens/unit/usdz-package-format/package-format.json` proves both directions:
`select-default-first-physical` accepts a data-aligned entry whose header is not
64-aligned, and `reject-misaligned-entry` rejects an unaligned data offset.

## 2. EOCD comment-length leniency

### What the spec says

§16.4.1.4 ("End of Central Directory record Restrictions"):

> the End of Central Directory record must be at the very end of the zip file,
> with no padding following it.

and, for the comment-length field of the EOCD:

> Comment Length — 2 bytes — This must be zero as no comment is allowed. The
> comment length must be the final set of bytes in the file with nothing after
> it.

A strictly conformant *writer* must therefore emit a comment length of zero.

### What this reader does

§16.4.2 explicitly anticipates readers diverging here:

> Implementations of this specification are not required to validate a USDZ file
> when reading from it. … if a USDZ file does have a comment … it may be still
> easily read by a range of zip libraries.

The reader takes that latitude. `find_eocd` in
`generated/cpp/usdz_package.cpp` searches the standard ZIP comment window
backward from the end of the buffer and accepts the first EOCD signature whose
recorded comment length is *consistent* with the file size — that is, where
`eocd_offset + 22 + comment_length == file_size`. It does **not** reject a
non-zero comment length. The "EOCD at the very end with no trailing padding"
restriction is still enforced (the comment, if present, must run exactly to
end-of-file).

This is a pure reader leniency: it widens the set of packages this repository
will *open*, and never affects what it would write (writing is out of scope).

### What conformant readers must do

The reader in this repository accepts a USDZ package with a non-zero,
consistent EOCD comment and otherwise treats it identically to a comment-free
package. The fixture `benchmarks/fixtures/usdz/commented_eocd.usdz` carries a
non-empty comment; `index-commented-eocd-leniency`
(`goldens/unit/usdz-package-format`),
`open-default-commented-eocd-leniency` (`goldens/unit/usdz-layer-open`), and
`commented-eocd-leniency`
(`goldens/integration/usdz-single-layer`) pin that the package opens
successfully end to end.

## 3. Default layer is the first physical entry

### What the spec says

§16.4.1.2 ("Contents"):

> A USD Package must use the first file within the Zip file as its root layer to
> be read. The file must be the first entry in the Zip Central Directory … the
> file must be the first file provided from the beginning of the Zip file …

For a conformant *writer*, the root layer is simultaneously the first central-
directory entry and the first entry physically (lowest local-header offset);
these coincide.

### What this reader does

The reader selects the default layer purely by **physical order**:
`parse_usdz_archive` sorts indexed entries by ascending `local_header_offset`,
and `default_layer_entry` returns the first one
(`generated/cpp/usdz_package.cpp`). It does **not** independently verify that the
first physical entry is also the first central-directory entry, and it does
**not** scan past the first entry looking for a `.usda`/`.usdc` extension — the
first physical entry *is* the root layer, whatever its extension (an unsupported
first entry yields `UnsupportedPackageLayerFormat` / `NestedUsdzUnsupported`
rather than silently skipping ahead).

This matches the reference behavior of opening the package's first file and is
codified by `package-layout.contract.json#entry_ordering` ("default-layer
selection returns the first ordered entry without scanning for a later USD
extension"). The unit golden `select-default-first-physical` proves it: in a
package whose first physical entry is `assets/notes.txt`, the default layer is
that `.txt` entry, not the later `.usda`.

## Spec resolution

All three items are reader-only positions:

- Item 1 is a genuine spec/reference divergence (the spec text says "file
  header"; the reference writer and every real `.usdz` align the data). The
  data-aligned reading is normative for this repository.
- Items 2 and 3 are leniencies that §16.4.2 explicitly permits.

If a future AOUSD spec revision rewords §16.4.1.3 to say "file data" and/or
tightens §16.4.2, this document and the contract cross-references can be
revisited; the reader paths the contracts pin already match the reference
implementation's behavior.
