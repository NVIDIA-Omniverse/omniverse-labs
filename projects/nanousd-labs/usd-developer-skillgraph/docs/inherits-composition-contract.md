# Inherits Composition Contract

This document describes the first bounded inherits composition contract in the
skill graph. The normative source is `aousd/specifications-public@v1.0.1` only,
specifically `specification/composition/README.md`, `specification/glossary`,
`specification/document_data_model/README.md`, and
`specification/file_formats/README.md`.

## Scope

`usd-inherits-namespace-source` wraps a composed base `NamespaceSource` and
applies direct `inheritPaths` entries exposed through base-source prim inputs.
It returns a normal composed namespace source for stage population.

The first contract includes:

- reading source-visible `inheritPaths`
- validating inherited target paths as prim paths
- mapping an inherited source prim subtree onto the inherit-authoring prim
- recursively expanding direct inherit arcs authored on inherited source prims
- applying implied inherit arcs from upstream layer-stack sites that contribute
  opinions at the same inherited source path
- remapping path-valued field contents such as relationship `targetPaths` and
  attribute `connectionPaths` into the inherit-authoring namespace
- same-path local layer-stack target opinions stronger than inherited opinions
- implied upstream inherited opinions stronger than weaker reference-introduced
  target opinions
- composed child and property name output
- diagnostics for missing inherited prims

The first contract defers:

- unbounded recursive inherit expansion without cycle diagnostics
- full cross-layer inherit listOp composition beyond the base source
- inherits authored inside selected variants
- interleaved inherits and specializes expansion beyond the ordered providers
  in this graph
- dictionary merging and value resolution

## Graph Placement

Inherits sit directly below local opinions in LIVERPS strength order:

```text
Local > Inherits > Variants > Relocates > References > Payloads > Specializes
```

The concrete generated graph evaluates references and relocates before
inherits so the inherits provider can see the composed source opinions it
needs. Payload opinions remain separate until `usd-composition-arbitrator` merges
them with the terminal composed namespace source. Specializes then consumes inherits
when direct specializes are in scope, and variants consume specializes as the
terminal non-variant source:

```text
usd-layer-stack-namespace-source
  -> usd-reference-namespace-source
  -> usd-relocates-namespace-source
  -> usd-inherits-namespace-source
  -> usd-specializes-namespace-source
  -> usd-variant-namespace-source
```

This is evaluation order, not field-strength order. The inherits contract still
records the LIVERPS placement and requires implementations not to infer strength
from graph order.

## Runtime Model

The domain model is a list of direct inherit arcs. Each arc stores the
inherit-authoring prim path and inherited source prim path as semantic path
handles/references.

For a prim or property at the target path, the provider queries the
corresponding inherited source path by replacing the target prefix with the
source prefix. For example, if `/World` inherits `/_Class`,
`/_Class/Geom.points` contributes to `/World/Geom.points`.

Direct inherit expansion is transitive. If `/_Class` itself inherits
`/_Base`, opinions from `/_Base` also contribute through the original mapping,
weaker than `/_Class`'s own opinions but still in the inherits family.
Implementations must track the semantic inherit-authoring path, inherited
source path, and mapping context to diagnose cycles and avoid unbounded
recursion.

Inherited source lookup is source-visible and layer-stack aware. If an inherit
arc authored in a referenced asset names `</class_Asset>`, stronger upstream
layer stacks that provide opinions at `/class_Asset` also participate through
implied inherit arcs. Those implied inherited opinions are stronger than weaker
reference-introduced local opinions on the inheriting prim because inherits are
above references in LIVERPS. Implementations must therefore keep enough
provenance to avoid flattening reference-local target fields ahead of stronger
implied upstream inherited fields.

Path-valued field contents authored under the inherited source prim are
projected in the same direction before fields are emitted. ObjectPaths outside
the inherited source subtree remain in the shared composed namespace for direct
inherits. This is namespace mapping of semantic path values, not full listOp
composition or value resolution.

Same-path local layer-stack target opinions are stronger than inherited source
opinions in this first provider. Weaker target opinions introduced by
references or payloads are not automatically stronger than implied upstream
inherited opinions. Child names are composed from inherited contributors first,
then target-local contributors. Property names are unioned, sorted by path
ordering, and then adjusted by the strongest target-local `propertyOrder`.

## Validation

The first provider rejects direct entries that violate the AOUSD v1.0.1
`inheritPaths` value range or cannot be found in the bounded base source:

- inherit path item is not a prim path
- inherit path item is a property path
- inherited source prim is missing

Recursive inherit cycles are diagnostics. Missing upstream implied inherit
contributors are not errors when no stronger layer-stack site contributes at
the inherited source path.
