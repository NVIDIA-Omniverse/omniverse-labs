---
name: usd-composition-arbitrator
description: Use this skill when implementing or verifying the central composition arbitrator that forms composed prim and property opinion stacks before stage population.
metadata:
  author: NVIDIA
---

# usd-composition-arbitrator

Use this skill when implementing or verifying the central composition arbitrator that
forms composed prim and property opinion stacks before stage population.

## Spec Sources

- `aousd/specifications-public@v1.0.1:specification/composition/README.md`
  composition algorithm, composition arcs, namespace mapping, variants, and
  strength ordering
- `aousd/specifications-public@v1.0.1:specification/glossary/README.md`
  composition operators, LIVERPS, local opinions, and specializes
- `aousd/specifications-public@v1.0.1:specification/stage_population/README.md`
  stage population consumption
- `aousd/specifications-public@v1.0.1:specification/document_data_model/README.md`
  composition fields and typed field values

Pinned tag: `v1.0.1`

Pinned tag / commit: `v1.0.1`

## Provides

- A central `composition_arbitrator` capability
- A final `NamespaceSource` facade for stage population
- Strong-to-weak `CompositionOpinion` stacks for composed prim and property
  paths
- LIVERPS family ordering across local, inherits, selected variants,
  relocates, references, loaded payloads, and specializes
- Typed composition provenance for source identifiers and semantic spec paths
- Instance-filtered descendant inputs for supported non-local composition arc
  opinions under instanceable prims
- Source capability reporting that accurately marks this first layer as
  partial composition, partial value resolution, and full instancing

## Contract

This skill owns `contracts/handles/composition-arbitrator.handle.json` and follows
`contracts/capabilities/composition-opinion-stack.json`. It also implements the
shared namespace-source boundary described by
`contracts/capabilities/namespace-source.json`.

This is the first graph node that should form composed prims and properties.
The existing arc-specific namespace sources remain useful as bounded sources of
mapped opinions, but they are not the final authority for opinion strength
across all arc families.

Stage population consumes the namespace-source facade produced by this layer.
It must not evaluate references, payloads, inherits, variants, relocates, or
specializes itself.

## Composition Rules

Collect all available opinions for a composed prim or property into typed
`CompositionOpinion` records before choosing population-facing fields. Each
record needs at least:

- arc family identity: local, inherits, variants, relocates, references,
  payloads, or specializes
- source identifier and semantic source spec path
- exposed composed scene path
- namespace mapping used to project the source opinion
- field map with typed values and token field keys
- strength keys for layer strength, arc family, namespace depth,
  authored/implied status when represented, and sibling order when represented

Path-valued field contents in emitted opinions must be in composed scene
coordinates. Relationship `targetPaths` and attribute `connectionPaths`
listOp<ObjectPath> elements are namespace-mapped through the opinion's mapping;
elements outside the mapped source subtree are pruned unless the source opinion
comes from a same-namespace arc and the original path is already in composed
coordinates. This remains required even while dictionary merging, listOp value
composition, and attribute value resolution are partial.

Order available opinions strong-to-weak by AOUSD v1.0.1 LIVERPS:

```text
Local > Inherits > Variants > Relocates > References > Payloads > Specializes
```

Graph order is not strength order. Constructor dependencies can exist because
one bounded source needs another source to discover opinions, but the
composition arbitrator must compare explicit strength keys when forming the final
stack.

Variant selections are computed after other composition arcs have supplied the
source-visible opinions needed to choose the selection. Selected variant
opinions are then inserted at variant strength, weaker than local and inherits
and stronger than relocates, references, payloads, and specializes.

Specializes opinions are globally weakest. If a prim specializes another prim,
all opinions on the specializing prim are stronger than opinions introduced by
the specializes arc, including opinions introduced by arcs on the specialized
prim.

Payload opinions may be included only when a loaded payload opinion source has
supplied them. If no payload source is available, report payload loading absent
or partial and do not claim loaded payload composition.

Recursive reference opinions and recursive or implied inherit/specialize
opinions supplied by bounded arc sources must retain source-visible layer
strength, namespace depth, source-visible order, namespace mapping, and
authored/implied status where applicable until the final opinion stack is
formed. Do not flatten those contributors into an opaque field map before
comparing them against local, variant, reference, payload, and specializes
opinions.

For this first bounded contract, non-special fields may expose the strongest
authored field value per field token. Do not claim dictionary merging, listOp
value composition, time-sample interpolation, value clips, or full value
resolution until later contracts define those behaviors.

For instanceable prim descendants, expose a bounded instance-filtered input
surface that excludes local descendant specs and includes only supported source
opinions introduced by composition arcs authored on the instanceable prim or
recursively on the source prim of such arcs. The supported non-local arc families
are inherits, selected variants, relocates, references, loaded payloads, and
specializes. This boundary must not require implementation-specific prototype
query APIs unless a later contract cites a pinned AOUSD requirement for them.

## Boundary Guards

Consume target-native namespace/opinion sources and typed document-model
objects. Do not inspect parser-local USDA syntax or canonical JSON dumps.

Use `usd-paths` for every path identity and path operation. Source spec paths,
composed paths, variant-selection paths, arc authoring paths, relocate source
and target paths, and property paths must be semantic path values in domain
records.

Use `usd-tokens` or semantic name handles for field keys, arc family names,
child names, property names, variant set names, selected variant names, and
schema/type names used for identity decisions. Adapter strings are boundary
renderings only.

Do not concatenate source identifiers and paths into composite strings for
later lookup, comparison, prefix checks, or composition decisions.

Do not perform stage population, asset resolution, layer opening, schema
fallback evaluation, implementation-specific prototype query APIs, value clips,
or attribute value interpolation.

## Test Obligations

- source capability declaration reports `composed_namespace`,
  `composition=partial`, `value_resolution=partial`, `instancing=full`, and
  composed child/property ordering
- local opinions are stronger than all arc opinions
- inherits are stronger than selected variants
- selected variants are stronger than relocates, references, payloads, and
  specializes
- payload opinions participate only when supplied
- adapter conformance cases can set `includePayloadSource=false` to verify that
  payload loading is reported absent and payload opinions do not participate
- specializes remain globally weakest
- `prim_input` and `property_input` are derived from the same opinion stacks
  exposed by `prim_opinions` and `property_opinions`
- stage population can consume the namespace-source facade without evaluating
  composition arcs
- instance-filtered descendant inputs exclude local descendant opinions under a
  supported instanceable prim and include non-local arc-family opinions
