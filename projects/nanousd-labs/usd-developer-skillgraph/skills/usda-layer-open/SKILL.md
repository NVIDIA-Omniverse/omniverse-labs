---
name: usda-layer-open
description: Use this skill when implementing or verifying the USDA text-layer format handler used by layer opening.
metadata:
  author: NVIDIA
---

# usda-layer-open

Use this skill when implementing or verifying the USDA text-layer format
handler used by layer opening.

## Spec Sources

- USD Core Spec Core File Formats section `Text`
- USD Core Spec Document Data Model section `Formats`

Pinned tag / commit: `v1.0.1`

## Provides

- USDA text layer format recognition for layer-open dispatch
- UTF-8 text-resource handoff to `usda-spec-parser`
- Target-native layer result pass-through for USDA inputs
- Canonical single-layer dump adapter support for USDA inputs
- Clear diagnostics for invalid text decoding before parser state is emitted

## Contract

This skill owns `contracts/handles/usda-layer-open.handle.json` and is the
concrete USDA format handler. It accepts already-opened USDA resource bytes
from `usd-layer-open`, decodes it as UTF-8 text, and calls `usda-spec-parser` to
build a document-model `Layer`.

Use `opened resource bytes`, `bytes`, or a target-native byte view for this
boundary. Do not name this input `payload`; in USD, payload refers to payload
composition arcs, which are explicitly out of scope here.

Implementations should avoid copying the entire opened resource when a safe byte
span, string view, memory map, or equivalent view can be handed to the parser.
In C++ this means prefer `std::span<const std::byte>`/`std::string_view`-style
interfaces over constructing a new `std::string` from the whole resource.

The domain output is a target-native layer-open result containing either the
document-model `Layer` or diagnostics. It must not be a `Json`, Python `dict`,
tagged JSON, or canonical dump tree. Canonical dump JSON belongs only to an
explicit adapter over the successful layer result.

In the graph, this skill consumes the `usda_spec_parser` capability. The handler
may decode bytes, select/diagnose the USDA format, and pass parser output
through, but it must not independently parse USDA grammar, construct paths,
mutate document-model state, or parse values.

The handler may inspect the resolved resource display name or extension to
answer whether it is a USDA candidate for dispatch. A resolved `.usda` extension
is sufficient for `usd-layer-open` to select this handler without reading
resource content to inspect a USDA text header. It does not own resource
classification, local file URI conversion, package splitting, unsupported
resource-scheme diagnostics, or generic signature sniffing.

The handler does not own generic layer opening or future format dispatch among
USDA, USDC, USDZ, or other layer formats. It owns only the USDA text path.

## Handle Contract

This skill owns the handler contract in
`contracts/handles/usda-layer-open.handle.json`. That contract requires that
the handler accept opened resource bytes or an equivalent byte view from a
successful `ResourceReadResult` supplied by `usd-layer-open` and pass it through
to `usda-spec-parser` without performing resource I/O itself.

## Boundary Guards

Defer resource location typing, local file URI conversion, resource reads, and
unsupported resource protocol diagnostics to `usd-resource-protocol` through
`usd-layer-open`.

Defer generic format dispatch, entrypoint argument handling, and unsupported
format diagnostics to `usd-layer-open`.

Defer USDA grammar, document-model storage, USD namespace paths, authored value
representation, listOps, and field semantics to
`usda-spec-parser` and its dependency closure.

Defer canonical layer dump materialization to an explicit adapter over the
returned document-model `Layer`.

Do not implement USDC parsing, USDZ package parsing, sublayer loading,
reference/payload loading, composition, asset resolution search paths, value
resolution, or stage population.

## Test Obligations

- accept an opened local filesystem USDA layer and return a successful
  document-model `Layer` result
- accept an opened supported local `file` URI USDA layer and emit the same
  canonical layer dump through the adapter when file URI cases are added
- expose a domain API whose success value is a document-model `Layer`, not a
  JSON object
- expose a `.usda` resolved-extension candidate predicate that lets
  `usd-layer-open` select this handler before content is read for header
  inspection
- when `usd-layer-open` selects this handler by extension, validate the opened
  bytes as USDA text after resource I/O; do not redispatch based on opened byte
  content
- reject undecodable UTF-8 before parser state is emitted
- preserve the parser's exact diagnostics for malformed USDA content after the
  resource has been opened and decoded
- the handler accepts opened resource bytes from `usd-layer-open`;
  calling `usd-resource-protocol.read_text_file_or_empty` or any other resource
  I/O directly with a raw path or URI string is a contract violation
- when the opened resource exposes `file_backed_path`, the handler forwards it
  to `usda-spec-parser` so future file-backed strategies can compose
- empty-resource inputs produce a successful empty `Layer` result, not
  `ResourceOpenFailed`; open-failure is reported by `usd-layer-open` before
  bytes reach this handler
- all cases in `goldens/integration/usda-single-layer/basic.json`
