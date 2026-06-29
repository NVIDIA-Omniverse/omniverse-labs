---
name: usdz-layer-open
description: Use this skill when implementing or verifying opening a single layer from a USDZ package.
metadata:
  author: NVIDIA
---

# usdz-layer-open

Use this skill when implementing or verifying opening a single layer from a USDZ
package.

## Spec Sources

- OpenUSD USDZ File Format Specification v1.3 default-layer and package layout
  requirements
- USD Core Spec Resource Interface package resource model
- USD Core Spec Core File Formats sections `Text`, `Binary`, and `Package`

Pinned AOUSD commit: `6b4f01f2d6d46b01507481524eb516b4ef358c31`

## Provides

- USDZ layer-format handler over already-opened package bytes
- default-layer selection through `usdz-package-format`
- explicit package entry selection for `outer.usdz[entry]`
- dispatch of `.usda` entries to `usda-layer-open`
- dispatch of `.usdc` entries to `usdc-layer-open`
- target-native `LayerOpenResult` pass-through

## Contract

This skill owns `contracts/handles/usdz-layer-open.handle.json`.

The generic `usd-layer-open` entrypoint opens the outer resource through
`usd-resource-protocol` and passes those package bytes here. This handler uses
`usdz-package-format` to index and select the package entry. It then invokes the
registered concrete layer-format handler for the selected entry extension:
`.usda` routes to `usda-layer-open`, and `.usdc` routes to `usdc-layer-open`.

For package entries, the standalone `file_backed_path` from the outer package
must not be forwarded to the inner handler. The path names the entire archive,
not the selected byte extent. Use the non-owning `PackageEntryByteView` bytes
instead.

Unsupported `.usd`, nested `.usdz`, missing default layer, missing explicit
entry, and package-layout diagnostics fail before parser-specific layer state is
emitted.

## Boundary Guards

Defer ZIP/USDZ structure parsing to `usdz-package-format`.

Defer resource classification, package spelling split, and outer package reads
to `usd-resource-protocol` and `usd-layer-open`.

Defer USDA text handling to `usda-layer-open` and USDC binary handling to
`usdc-layer-open`.

Do not implement `.usd` forwarding, nested package opening, package writes,
composition, dependent layer loading, stage population, or asset resolution.

## Test Obligations

- open a default `.usda` layer from a package
- open a default `.usdc` layer from a package
- ignore later non-layer assets when the default entry is already selected
- open explicit `.usda` and `.usdc` package entries
- preserve exact package diagnostics for invalid ZIP, compression, encryption,
  alignment, duplicate names, unsafe paths, missing default layer, missing
  explicit entry, unsupported `.usd`, and nested `.usdz`
- return the same target-native layer result shape as USDA and USDC handlers
