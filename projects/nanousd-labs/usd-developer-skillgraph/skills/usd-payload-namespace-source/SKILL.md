---
name: usd-payload-namespace-source
description: Use this skill when implementing or verifying the first direct loaded payload opinion source backed by supplied payload arc sites and already-opened payload layers.
metadata:
  author: NVIDIA
---

# usd-payload-namespace-source

Use this skill when implementing or verifying the first direct loaded payload
opinion source backed by supplied payload arc sites and already-opened payload
layers.

## Spec Sources

- `specification/composition/README.md` payloads and composition strength
- `specification/file_formats/README.md` `PayloadMetadata` and `PayloadList`
- `specification/document_data_model/README.md` `payload: listOp<Payload>`
- `specification/stage_population/README.md` optional payload population flag

Pinned tag / commit: `v1.0.1`

## Provides

- A payload arc opinion source that can expose mapped payload opinions through
  the `NamespaceSource` shape for conformance tests
- Direct payload discovery from supplied payload arc sites
- Path-only internal payloads within the payload arc layer namespace
- Payload target mapping from an external asset layer into the payload-bearing
  prim
- Asset-only payload target selection through the payload layer `defaultPrim`
- Payload-contributed prim and property opinions in composed scene coordinates
- Path-valued field contents such as relationship `targetPaths` and attribute
  `connectionPaths` remapped into the payload-bearing namespace
- Bounded references authored inside loaded payload target subtrees while
  their referenced layers are already supplied
- Arc-local `layerRelocates` mappings authored in loaded payload layer stacks
  and carried by nested reference records
- Diagnostics for missing payload layers and missing target prims

## Contract

This skill owns `contracts/handles/payload-namespace-source.handle.json` and
implements the shared namespace-source shape described by
`contracts/capabilities/namespace-source.json` and the arc-source split in
`contracts/capabilities/composition-opinion-source.json`. It also follows
`contracts/capabilities/semantic-runtime-types.json`.

The first version is an in-memory direct loaded-payload opinion source. It
consumes payload-bearing prim paths and their source-visible payload listOp
values, plus a map of already-opened payload layers keyed by authored asset
identifier. It does not resolve asset identifiers, open files, recurse through
payload arcs, implement load policy, merge with local/reference opinions, or
perform full value resolution. Optional runtime direct payload asset opening
for the adapter payload inclusion scenario is owned by
`usd-composed-stage-open`.

The payload source is not the LIVERPS composition arbitrator. Generated domain
constructors must not require `ReferenceNamespaceSource` or another concrete
prior composition provider. `usd-composition-arbitrator` should consume local,
reference, payload, and other arc opinion sources and produce the final
composed namespace source for stage population.

The source exposes capabilities:

- `mode`: `composed_namespace`
- `composition`: `partial`
- `value_resolution`: `partial`
- `payload_loading`: `partial`
- `instancing`: `absent`
- `schema_fallbacks`: `absent`
- `child_ordering`: `composed`
- `property_ordering`: `composed`

`partial` payload loading means the source composes direct external payload
arcs from supplied layers and direct path-only internal payloads as loaded.
Explicit unloaded state, load rules, population load masks, and payload asset
opening are outside this source; optional bounded runtime asset opening lives in
`usd-composed-stage-open`.

## Composition Rules

Payload arcs are read from supplied payload arc sites. Each site carries the
payload-bearing prim path and the source-visible `payload` listOp value. This
first contract consumes that supplied listOp value and preserves `explicit`
entries when present, otherwise `prepend` followed by `append`. Cross-layer
payload listOp composition before arc-site construction remains a later
composition contract.

External payload arcs have an authored asset identifier. The payload layer is
supplied by the caller or adapter. If an arc also has a target path, that prim
is the payload root. If the arc has only an asset, use the payload layer's
`defaultPrim`; missing defaultPrim is a diagnostic. Path-only internal payloads
use the payload arc layer as the payload namespace and must name an explicit
target prim path.

The payload target prim maps to the payload-bearing prim path. Descendant prims
and properties below the target are mapped by replacing the target prefix with
the payload-bearing prim path.

Path-valued field contents authored under payload specs are mapped the same way
before fields are emitted. Each ObjectPath under the payload target subtree is
projected under the payload-bearing prim path. For `targetPaths` and
`connectionPaths`, ObjectPaths outside the payload target subtree are pruned by
this bounded external-payload provider rather than emitted in payload-layer
coordinates. For path-only internal payloads, ObjectPaths outside the payload
target subtree remain in the shared composed namespace.

References authored inside a loaded payload target subtree are expanded while
their referenced layers are already supplied. Treat these nested opinions as
payload-family contributions because the payload arc introduced that loaded
layer stack into the composed stage. If the loaded payload layer stack authors
`layerRelocates`, carry that relocate namespace mapping on the nested record:
hide relocated source paths, expose relocated target paths, suppress stale
target collisions, prune deleted sources, and remap path-valued fields through
the accumulated payload/reference plus relocate mapping.

This source returns only payload-contributed opinions. Base-source opinions are
stronger than payload opinions in full composition, but that arbitration belongs
to `usd-composition-arbitrator`. Stronger local/reference `over` specs
overriding weaker payload `def` specs is not evaluated by this payload source.

Child prim names and property names come from mapped payload specs only.

## Boundary Guards

Consume supplied payload arc sites and supplied already-opened payload layers.
Do not consume `usd-reference-namespace-source` or another concrete prior
composition provider in the payload source constructor.

Payload source runtime records must retain payload-bearing prim paths, payload
target paths, and mapped contributor paths as path handles/references. String
spellings are acceptable for asset identifiers, diagnostics, and adapter
summaries, but not as the authoritative identity for USD object paths.
Contributing source provenance must be stored as typed records containing a
source/layer identifier and a semantic spec path. Do not store or later parse
composite strings such as `asset.usda:/Prim` in domain logic; those are adapter
or diagnostic renderings only.

Consume `usd-document-model` for already-opened payload layers and field values
on supplied arc sites. Do not store or compose canonical layer dump JSON as
durable state.

Consume `usd-paths` and `usd-tokens` for path and token identity. Adapter
strings are boundary inputs only.

Do not perform asset resolution, layer opening, recursive payload-arc
expansion, payload load policy, local/reference/payload LIVERPS merging,
inherits, specializes, variant composition, root-layer relocates, value clips,
schema fallback evaluation, stage population, model hierarchy traversal,
instancing, or attribute value interpolation.

## Test Obligations

- source capability declaration reports `composed_namespace`
- direct external payloads contribute prim, property, and child opinions
- asset-only payloads use the payload layer `defaultPrim`
- missing payload layers fail clearly
- references inside loaded payload layers expand when their layers are supplied
- nested payload arcs inside payload layers remain deferred
- composition-arbitrator tests, not payload-source tests, should prove local or
  reference opinions remain stronger than payload opinions
- all cases in
  `goldens/unit/usd-payload-namespace-source/payload-namespace-source.json`
