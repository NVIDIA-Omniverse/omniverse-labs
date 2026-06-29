# Specializes Composition Contract

This document describes the first bounded specializes composition contract in
the skill graph. The normative source is `aousd/specifications-public@v1.0.1`
only, specifically `specification/composition/README.md`,
`specification/glossary`, `specification/document_data_model/README.md`, and
`specification/file_formats/README.md`.

## Scope

`usd-specializes-namespace-source` wraps a composed base `NamespaceSource` and
applies direct `specializes` entries exposed through base-source prim inputs.
It returns a normal composed namespace source for stage population.

The first contract includes:

- reading source-visible `specializes`
- validating specialized target paths as prim paths
- mapping a specialized source prim subtree onto the specialize-authoring prim
- recursively expanding direct specializes arcs authored on specialized source
  prims
- applying implied specializes arcs from upstream layer-stack sites that
  contribute opinions at the same specialized source path
- remapping path-valued field contents such as relationship `targetPaths` and
  attribute `connectionPaths` into the specialize-authoring namespace
- all target opinions stronger than specialized opinions
- composed child and property name output
- diagnostics for missing specialized prims

The first contract defers:

- unbounded recursive specialize expansion without cycle diagnostics
- full cross-layer specialize listOp composition beyond the base source
- specializes authored inside selected variants
- dictionary merging and value resolution

## Graph Placement

Specializes are globally weakest in LIVERPS strength order:

```text
Local > Inherits > Variants > Relocates > References > Payloads > Specializes
```

The concrete generated graph evaluates references, relocates, and inherits
before specializes so the specializes provider can see the composed source
opinions it needs. Payload opinions remain separate until `usd-composition-arbitrator`
merges them with the terminal composed namespace source. Variants consume
specializes as the terminal non-variant source so selection can observe
specializes opinions:

```text
usd-layer-stack-namespace-source
  -> usd-reference-namespace-source
  -> usd-relocates-namespace-source
  -> usd-inherits-namespace-source
  -> usd-specializes-namespace-source
  -> usd-variant-namespace-source
```

This is evaluation order, not field-strength order. The variant provider must
insert selected variant opinions above specializes opinions even though it
evaluates after the specializes source.

## Runtime Model

The domain model is a list of direct specialize arcs. Each arc stores the
specialize-authoring prim path and specialized source prim path as semantic path
handles/references.

For a prim or property at the target path, the provider queries the
corresponding specialized source path by replacing the target prefix with the
source prefix. For example, if `/World` specializes `/_Class`,
`/_Class/Geom.points` contributes to `/World/Geom.points`.

Direct specializes expansion is transitive. If `/_Class` specializes
`/_Base`, opinions from `/_Base` also contribute through the original mapping,
weaker than `/_Class`'s own opinions and still globally weakest against
non-specializes opinions on the specializing prim. Implementations must track
the semantic specialize-authoring path, specialized source path, and mapping
context to diagnose cycles and avoid unbounded recursion.

Specialized source lookup is source-visible and layer-stack aware. If a
specializes arc authored in a referenced asset names `</class_Asset>`, stronger
upstream layer stacks that provide opinions at `/class_Asset` also participate
through implied specializes arcs. Those implied specialized opinions remain
specializes-family opinions, but stronger upstream layer-stack sites at the
specialized source path are stronger than weaker specialized-source sites.

Path-valued field contents authored under the specialized source prim are
projected in the same direction before fields are emitted. ObjectPaths outside
the specialized source subtree remain in the shared composed namespace for
direct specializes. This is namespace mapping of semantic path values, not full
listOp composition or value resolution.

Target opinions are stronger than specialized source opinions in this first
provider. Child names are composed from specialized contributors first, then
target contributors. Property names are unioned, sorted by path ordering, and
then adjusted by the strongest target `propertyOrder`.

## Validation

The first provider rejects direct entries that violate the AOUSD v1.0.1
`specializes` value range or cannot be found in the bounded base source:

- specialize path item is not a prim path
- specialize path item is a property path
- specialized source prim is missing

Recursive specializes cycles are diagnostics. Missing upstream implied
specializes contributors are not errors when no stronger layer-stack site
contributes at the specialized source path.
