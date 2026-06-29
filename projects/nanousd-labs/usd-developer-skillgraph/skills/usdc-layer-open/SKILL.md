---
name: usdc-layer-open
description: Use this skill when implementing or verifying the concrete USDC layer-format handler used by generic layer opening.
metadata:
  author: NVIDIA
---

# usdc-layer-open

Use this skill when implementing or verifying the concrete USDC layer-format
handler used by generic layer opening.

## Spec Sources

- USD Core Spec Document Data Model section `Formats`
- USD Core Spec Core File Formats sections `Binary` and `Crate`
- `harness/dump_contract.md`

Pinned commit: `6b4f01f2d6d46b01507481524eb516b4ef358c31`

## Provides

- `usdc_layer_format`
- USDC extension recognition for layer-open dispatch
- Opened-byte handoff to `usdc-spec-parser`
- Target-native layer result pass-through for USDC inputs
- Clear diagnostics for non-USDC or unsupported USDC resources

## Contract

This skill is the concrete USDC format handler. It accepts an already-opened
binary resource byte view from `usd-layer-open`, verifies that the resolved
resource is a USDC candidate, and calls `usdc-spec-parser` to build a
target-native document-model Layer result. Canonical layer dump JSON is produced
only by explicit adapters, diagnostics serialization, and dump commands.

The handler may inspect the resolved resource display name or extension to
answer whether it is a USDC candidate for dispatch. A resolved `.usdc` extension
is sufficient for `usd-layer-open` to select this handler without reading
resource content to inspect the `PXR-USDC` magic header. It does not own resource
classification, local file URI conversion, package splitting, unsupported
resource-scheme diagnostics, or generic signature sniffing.

## Boundary Guards

Defer resource location typing, local file URI conversion, resource reads, and
unsupported resource protocol diagnostics to `usd-resource-protocol` through
`usd-layer-open`.

Defer generic format dispatch and command-line entrypoint ownership to
`usd-layer-open`.

Defer Crate byte decoding to `usdc-binary-format`, value decoding to
`usdc-value-decoder`, and document-model construction to `usdc-spec-parser`.

Do not call USDA text lexers, USDA value parsers, or USDA spec parsers.

Do not implement USDZ package parsing, `.usd` forwarding dispatch, sublayer
loading, composition, asset resolution, value resolution, schema fallback,
clips, or USDC writing.

## Test Obligations

- accept an opened local filesystem `.usdc` layer and return a target-native layer result
- expose a `.usdc` resolved-extension candidate predicate that lets
  `usd-layer-open` select this handler before content is read for magic-header
  inspection
- when `usd-layer-open` selects this handler by extension, validate the opened
  resource bytes through `usdc-binary-format`; non-`PXR-USDC` bytes produce USDC
  diagnostics rather than being redispatched to another handler
- reject non-`PXR-USDC` data before parser state is emitted
- preserve parser diagnostics for malformed Crate content after dispatch
- keep `.usdz` unsupported through `usd-layer-open` until package support exists
- pass the target-native layer result from `usdc-spec-parser`; returning canned dump JSON selected by file content or section names is a contract violation
- pass non-`PXR-USDC`, unsupported-version, and parser-level diagnostics through from `usdc-binary-format` and `usdc-spec-parser` without re-running format detection or local opened-byte string scans
- the handler accepts opened resource bytes from `usd-layer-open`; calling `usd-resource-protocol.read_resource` or `read_binary_file` directly with a raw identifier string is a contract violation
- `open_usdc_layer` is called only after `usd-layer-open` has received a
  successful `ResourceReadResult`; open-failure diagnostics are emitted by
  `usd-layer-open` before bytes reach this handler
- when the opened resource exposes `file_backed_path`, the handler forwards it to `usdc-binary-format` so the Crate decoder can open a memory-mapped region instead of consuming an owned byte vector
- when a canonical dump is emitted by an adapter or dump command for a `.usdc`
  fixture, it must match byte-for-byte after JSON canonicalization whether the
  layer is opened through the full `dump_layer.exe` pipeline or through
  `usdc_layer_open_adapter.exe`. Any divergence indicates that the USDC
  pipeline is taking a shortcut around the production decoder or violating one
  of the format-neutral document-model invariants in
  `contracts/document-model-productions/`.

## Performance

The layer-open handler must satisfy
`contracts/performance/usdc.performance.json` by deferring all decode and
layer-build work to `usdc-binary-format`, `usdc-value-decoder`, and
`usdc-spec-parser`. The handler itself adds only constant-time dispatch and
diagnostic pass-through.
