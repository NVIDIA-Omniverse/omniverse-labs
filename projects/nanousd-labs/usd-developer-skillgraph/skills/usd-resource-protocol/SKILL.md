---
name: usd-resource-protocol
description: Use this skill when implementing or verifying USD resource protocol support after asset resolution.
metadata:
  author: NVIDIA
---

# usd-resource-protocol

Use this skill when implementing or verifying USD resource protocol support
after asset resolution.

## Spec Sources

- USD Core Spec Resource Interface sections covering resource identifiers,
  URI references, resolved resource locations, resource interaction, file-scheme
  support, packages, and security considerations
- USD Core Spec USDZ package requirements for locating the root layer inside a
  package
- RFC 3986 URI reference syntax and relative-reference resolution, as
  incorporated by the resource interface
- RFC 8089 local file URI behavior, as incorporated by the resource interface

Pinned tag / commit: `v1.0.1`

## Provides

- Separation between authored asset identifiers, resolved resource locations,
  resource protocol I/O, file-format parsing, and package-entry access
- A typed resolved-location boundary for resource I/O
- Local filesystem resource reads
- Local filesystem resource writes and whole-resource saves
- Local `file` URI handling for the supported RFC 8089 subset
- Unsupported URI/protocol diagnostics without adding network dependencies
- Package resource addressing for `outer[inner]`
- Resource read results that can expose bytes or a local file-backed fast path
- Resource write results that preserve clear success and diagnostic state

## Contract

Asset resolution and resource interaction are separate steps. Resolver output
must be converted to a resolved-location value before resource I/O so authored
identifiers, resolved locations, local paths, URI destinations, and package
entries are not confused.

The resolved-location value should distinguish at least these cases:

- local filesystem resources, including decoded local `file` URIs;
- URI or protocol resources that are retained as typed unsupported destinations
  unless a future skill explicitly adds that protocol;
- package resources, represented as an outer resolved location plus a normalized
  package-entry path.

Use `Type` as the resolved-location discriminator name. Avoid `Kind` because
USD already uses kind terminology for model classification.

Filesystem behavior is the default, dependency-free path. Local paths should
continue to support efficient file-backed parsing where a target benefits from
it, such as memory mapping a local USDC file. Layer-format selection may use the
`ResolvedLocation` display/path extension before content is read for signature
inspection, but invoking a concrete handler still goes through
`read_resource(const ResolvedLocation&)` and passes the resulting byte view or
file-backed provenance to the relevant file-format parser.

URI syntax and anchored relative URI behavior must follow an RFC 3986 helper or
equivalent structured parser. Do not split URI strings ad hoc in parser, writer,
package, or resource call sites. Relative URI resolution and dot-segment removal
belong at the resolution boundary, not in resource protocol I/O.

Anchored relative authored identifiers such as `./asset.usda` and
`../asset.usda` resolve against the authored document. Non-anchored relative
identifiers such as `asset.usda` remain application or search-path identifiers
until the resolver produces a concrete location.

The supported local `file` URI subset maps local file URIs to decoded filesystem
paths. Query-bearing or fragment-bearing file URIs, and file URIs with
non-local authorities, must not silently become local paths. Empty authority and
`localhost` authority may be treated as local. Other authorities must remain
typed URI destinations or fail with clear unsupported diagnostics.

Windows drive paths, drive-relative paths, UNC paths, root-relative paths, and
backslash-containing paths are non-normative filesystem path forms, not URI
schemes. In particular, single-character scheme-looking prefixes such as `C:`
must not be classified as URI protocols.

Package resource syntax such as `outer.usdz[scenes/root.usda]` is not an
ordinary filesystem path or URI path. Split package syntax before protocol I/O,
read the outer resource through the resource layer, and then locate the inner
entry inside the package namespace.

This skill stops at the outer resource plus normalized entry spelling boundary.
It does not parse ZIP central directories, validate USDZ layout, select default
layers, or produce package entry byte views; those responsibilities belong to
`usdz-package-format`.

Writes and saves are conservative. Support local filesystem paths and decoded
local `file` URIs. Unsupported URI schemes must fail clearly, for example with a
diagnostic that names the unsupported scheme. Package-entry writes are out of
scope unless package mutation is explicitly added by a later skill or scope.

## Implementation Quality Floor

The production read API must be shaped around resolved locations:
`read_resource(const ResolvedLocation&)` returning a `ResourceReadResult`.
Internal read APIs that accept raw identifier strings after resolution are a
contract violation. Public convenience APIs may continue to accept authored
identifier strings, but they must convert through `classify_resolved` at the
Stage/Compose or equivalent boundary before calling `read_resource`.

For LocalFile production reads, the `ResourceReadResult` must populate
`file_backed_path` with a usable filesystem path so downstream format handlers
can elect file-backed fast paths (memory-mapped Crate decoding in particular).
Loose "should populate" wording does not satisfy this contract — LocalFile reads
that omit `file_backed_path` are a contract violation.

Production reads must distinguish empty-resource success from open-failure: an
empty file is `ok=true` with zero-length bytes, not a diagnostic. Read APIs that
return an empty buffer to signal "could not open" (the prior
`read_text_file_or_empty` shape) are a contract violation.

Resource write results should report the resolved destination and any
diagnostic text without falling back to treating unsupported URI strings as
filesystem paths.

For local whole-resource saves, prefer an atomic local save when it is simple:
write a sibling temporary file, flush and close it, then rename or replace the
destination. Preserve existing parent-directory creation behavior unless a
scope explicitly changes it.

Do not add network protocol support, TLS stacks, credential handling, remote
cache behavior, or remote write semantics as part of this skill.

## Boundary Guards

Defer authored identifier anchoring, search-path policy, and resolver callback
behavior to the asset resolver or caller. This skill opens or saves an already
resolved location.

Defer USDA lexical parsing to `usda-lexical-format` and USDA layer construction
to `usda-spec-parser`.

Defer USD path grammar and spec-path identity to `usd-paths`.

Defer value representation to `usd-foundational-values` and authored listOp
representation to `usd-listops-authored`.

Do not fold resource I/O into the resolver. Resolution may produce a URI-shaped,
path-shaped, package-shaped, or promised location; protocol I/O decides whether
that resolved location can be read or saved.

Do not use OpenUSD implementation behavior as the source of truth unless a task
explicitly asks for compatibility analysis. This skill is grounded in the AOUSD
resource interface and its referenced URI standards.

## Test Obligations

- local path resource reads for filesystem-backed USDA and USDC inputs
- local `file` URI resource reads for filesystem-backed USDA and USDC inputs
- local path and local `file` URI writes
- unsupported non-file URI reads and writes fail with clear diagnostics
- RFC 3986 anchored relative URI resolution
- RFC 8089 local file URI conversion, including percent-decoding at the local
  filesystem boundary
- rejection or unsupported typing for query-bearing, fragment-bearing, and
  remote-authority file URIs
- Windows drive, drive-relative, UNC, root-relative, and backslash-containing
  path-like inputs are not mistaken for URI protocols
- package resource reads for an outer package and an explicit inner entry
- package-relative asset values preserve the outer resource identity and resolve
  inner paths inside the package namespace
- package-entry writes fail clearly unless package mutation is explicitly in
  scope
- production reads expose a `ResourceReadResult` that distinguishes
  empty-resource success (`ok=true`, zero-length bytes) from open-failure
- LocalFile production reads populate `file_backed_path` with a usable
  filesystem path so format handlers can elect a file-backed fast path
- the production read API accepts a `ResolvedLocation` value, not a raw
  identifier string; raw-string read APIs after resolution are a contract
  violation
