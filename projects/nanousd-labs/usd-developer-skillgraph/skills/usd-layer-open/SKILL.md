---
name: usd-layer-open
description: Use this skill when implementing or verifying the generic boundary that opens a resolved USD layer resource and dispatches it to a registered layer-format handler.
metadata:
  author: NVIDIA
---

# usd-layer-open

Use this skill when implementing or verifying the generic boundary that opens a
resolved USD layer resource and dispatches it to a registered layer-format
handler.

## Spec Sources

- USD Core Spec Resource Interface sections covering resolving identifiers to
  resource locations, resource protocol interaction, extension resolution, and
  packaged resources
- USD Core Spec Document Data Model section `Formats`
- USD Core Spec Core File Formats sections `Text`, `Binary`, and `Package`

Pinned tag / commit: `v1.0.1`

## Provides

- Root layer resource opening after identifier resolution
- Format dispatch from a resolved resource location to declared format handlers
- Target-native layer-open result ownership
- Canonical layer-open dump adapter ownership
- Clear diagnostics for unsupported resource schemes or unsupported formats
- A boundary where USDA, USDC, and USDZ open paths compose with
  the same resource protocol

## Contract

This skill owns `contracts/handles/layer-open.handle.json`. Layer opening is the
integration boundary between resource protocol I/O and layer-format handlers. It
accepts an already-resolved resource location or a root input that is treated as
a concrete local resource by the target. It classifies the input through
`usd-resource-protocol`, selects a handler from the resolved location when the
extension is decisive, opens the resource through
`usd-resource-protocol`, and dispatches the opened bytes or local file-backed
path to the selected format handler.

Resolved `.usda`, `.usdc`, and `.usdz` extensions are decisive fast paths when
the corresponding handler is registered. A `.usda` resource dispatches to
`usda-layer-open`, a `.usdc` resource dispatches to `usdc-layer-open`, and a
`.usdz` resource dispatches to `usdz-layer-open`,
without reading resource content to inspect a text header, binary magic, or other
signature for the dispatch decision. Content signatures may be used only as a
fallback for ambiguous resources with no registered extension match. They must
not override an explicit `.usda`, `.usdc`, or `.usdz` resolved extension.

For the current `usda-single-layer` target, the only registered handler is
`usda-layer-open`, so the generic opener can only dispatch USDA text layers and
must report unsupported-format diagnostics for other formats. The domain output
is a target-native layer-open result containing either a document-model `Layer`
or diagnostics. It must not be a `Json`, Python `dict`, tagged JSON, or
canonical dump tree.

The `usdc-single-layer` target adds `usdc-layer-open` as a separate concrete
handler for `.usdc` resources while preserving the same target-native
layer-open result boundary and dump-adapter separation.

The `usdz-single-layer` target adds `usdz-layer-open` as a package-layer handler
for `.usdz` resources and `Package` resolved locations. Resource protocol owns
package spelling split into outer resource plus entry. USDZ package parsing,
default-layer selection, and entry byte views belong to `usdz-package-format`
through `usdz-layer-open`, not to this generic opener.

The generated dump command is a separate adapter. It may convert the native
layer-open result into the canonical layer dump JSON described by
`harness/dump_contract.md` for goldens, diagnostics, and benchmarks.

The opener does not own USDA grammar, USDC decoding, USDZ package parsing,
document-model storage, USD namespace paths, authored value representation,
listOps, or field semantics. It only owns resource-to-format dispatch and the
target entrypoint shape.

When the resource cannot be opened or the format is unsupported, diagnostics
should fail before parser-specific layer state is emitted.

## Handle Contract

This skill owns the entrypoint contract in
`contracts/handles/layer-open.handle.json`. That contract codifies the
ResolvedLocation-first dispatch flow: convert identifier to
`ResolvedLocation` through `usd-resource-protocol`, split package-typed
locations through `split_package`, select `.usda`/`.usdc`/`.usdz` handlers by resolved
extension before any content sniffing, read through `read_resource`, then invoke
the selected handler with the opened `ResourceReadResult` (including
`file_backed_path` when present for standalone file handlers). If there is no decisive extension match,
implementations may read the resource once and use a signature fallback before
choosing a handler.

## Boundary Guards

Defer resource location typing, local file URI conversion, resource reads,
resource writes, and unsupported resource protocol diagnostics to
`usd-resource-protocol`.

Defer USDA text-layer handling to `usda-layer-open`.

Defer canonical layer dump construction to an explicit adapter over the returned
document-model `Layer`.

Defer USDC binary parsing to `usdc-layer-open` and its `usdc-*` dependency
closure when that handler is in the active target scope.

Defer USDZ package parsing, default-layer selection, and package-entry lookup to
`usdz-layer-open` and `usdz-package-format`.

Defer USD namespace path grammar and spec-path identity to `usd-paths` through
the parser/document-model dependency closure.

Do not implement text parsing, binary parsing, package parsing, sublayer
loading, reference/payload loading, value clip loading, composition, asset
resolution search paths, value resolution, or stage population in this generic
opener.

Do not call `usd-resource-protocol.read_resource` from concrete handlers; the
entrypoint owns resource I/O and hands the opened `ResourceReadResult` to the
concrete handler. Concrete handlers that invoke resource protocol reads
directly with a raw identifier string are a contract violation.

## Test Obligations

- dispatch a local filesystem USDA layer to `usda-layer-open` and emit the
  canonical layer dump through the adapter
- dispatch a local filesystem USDC layer to `usdc-layer-open` when that handler
  is present in the active target scope and emit the canonical layer dump
  through the adapter
- dispatch a supported local `file` URI USDA layer to `usda-layer-open` and emit
  the same canonical layer dump through the adapter when file URI cases are
  added
- expose a domain API whose success value is a document-model `Layer`, not a
  JSON object
- dispatch a supported local `file` URI USDC layer to `usdc-layer-open` when
  that handler is present in the active target scope
- the entrypoint converts the input identifier to a `ResolvedLocation` through
  `usd-resource-protocol` before format dispatch; raw-identifier suffix matching
  before resolution is a contract violation
- format dispatch reads `ResolvedLocation` metadata first and may consult an
  opened resource signature only as an ambiguous-resource fallback; suffix
  matching on the raw identifier before resolution is a contract violation
- `.usda` and `.usdc` resolved extensions dispatch automatically to
  `usda-layer-open` or `usdc-layer-open` when the handler is registered; this
  extension fast path must not read resource content to inspect a header or magic
  value before selecting the handler
- content-signature dispatch is only a fallback for ambiguous resources with no
  registered extension match; mismatched opened bytes under an explicit `.usda`
  or `.usdc` extension are handled by the selected handler's validation
  diagnostics, not by redispatching to another handler
- when `usd-resource-protocol` exposes `file_backed_path` on the opened
  `ResourceReadResult`, the entrypoint forwards that path to the concrete
  handler for standalone resources so format-specific fast paths (e.g. Crate
  mmap) can run
- package-syntax inputs are split through `usd-resource-protocol.split_package`
  before any handler is invoked; `outer.usdz[inner]` reaches `usdz-layer-open`
  as opened outer package bytes plus the explicit inner entry
- `.usdz` local resources dispatch to `usdz-layer-open`; whole-package opens use
  the package default layer selected by `usdz-package-format`
- Uri-typed locations that no registered handler accepts must produce
  `UnsupportedScheme`, not `ResourceOpenFailed`
- reject unsupported resource schemes before parser state is emitted
- reject unsupported formats clearly
- preserve the handler's exact diagnostics for malformed USDA content after the
  resource has been opened
- all cases in `goldens/integration/usda-single-layer/basic.json`
