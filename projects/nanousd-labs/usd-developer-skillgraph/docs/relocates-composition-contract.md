# Relocates Composition Contract

This document describes the first bounded relocates composition contract in the
skill graph. The normative source is `aousd/specifications-public@v1.0.1` only,
specifically `specification/composition/README.md`, `specification/glossary`,
`specification/document_data_model/README.md`, and
`specification/file_formats/README.md`.

## Scope

`usd-relocates-namespace-source` wraps a composed base `NamespaceSource` and
applies direct `layerRelocates` entries exposed through the base source. It
returns a normal composed namespace source for stage population.

The first contract includes:

- reading source-visible `layerRelocates`
- validating direct source-to-target entries
- moving a source prim subtree to a target prim subtree
- deleting/removing a source subtree when the target is empty
- target-local opinions stronger than relocated source opinions
- recoverable diagnostics for local-layer-stack opinions at relocate source
  paths while still applying the otherwise-valid relocate entry
- recoverable diagnostics for relocate sources authored beneath an already
  relocated ancestral source path
- arc-local relocate namespace mappings for references and loaded payload layer
  stacks that introduce nested references
- composed child and property name output
- diagnostics for invalid relocate entries

The first contract defers:

- automatic source-path rewriting for invalid ancestral relocate authoring
- full recursive composition of relocate namespace mappings into inherit and
  specialize arcs
- inherits, specializes, recursive variants, value clips, and value resolution

## Graph Placement

Relocates sit between variants and references in LIVERPS strength order:

```text
Local > Inherits > Variants > Relocates > References > Payloads > Specializes
```

The concrete generated graph evaluates direct references before relocates so
the relocates provider can see referenced source opinions that need to be
moved. Payload opinions are produced by a separate payload opinion source until
`usd-composition-arbitrator` merges them with the terminal composed namespace source.
When no stronger non-variant provider is present, variants may consume relocates
directly. With inherits and specializes in scope, inherits consumes relocates,
specializes consumes inherits, and variants consume specializes as the terminal
non-variant source:

```text
usd-layer-stack-namespace-source
  -> usd-reference-namespace-source
  -> usd-relocates-namespace-source
  -> usd-inherits-namespace-source
  -> usd-specializes-namespace-source
  -> usd-variant-namespace-source
```

This is evaluation order, not field-strength order. The relocates contract
still records the LIVERPS placement and requires implementations not to infer
strength from graph order.

## Runtime Model

The domain model is a native map from source prim path to optional target prim
path. Empty target means deletion/removal of the source subtree. Runtime records
must hold source paths, target paths, and mapped contributor paths as semantic
path handles/references. Adapter strings are boundary input and output only.

For a non-empty mapping, the provider projects a source subtree onto a target
subtree by replacing the source prefix with the target prefix. Properties are
mapped with the same owning-prim prefix replacement.

Target-local opinions are stronger than relocated source opinions in this first
provider. Child names are composed from relocated source contributors first,
then target-local contributors. Property names are unioned, sorted by path
ordering, and then adjusted by the strongest target-local `propertyOrder`.

## Validation

The first provider rejects direct entries that violate AOUSD v1.0.1 relocate
restrictions:

- source or target is pseudo-root or root
- source or non-empty target is not a prim path
- duplicate source or duplicate non-empty target
- target equals source
- target is below source
- target places a prim at an existing ancestor path of the source

Source-local authored opinions at the relocate source path are a composition
error and must not contribute through that source path. This error is
recoverable: the relocate entry is still retained and applied. When local and
referenced opinions coexist at the source path, the implementation must consume
reference-family-only source opinions or equivalent per-family provenance rather
than relocating a flattened local-plus-remote field map.

When references or loaded payloads are authored inside a layer stack that also
authors `layerRelocates`, the arc provider must carry that layer stack's
relocate mapping with the nested arc. The mapping hides source paths, maps
relocated target paths back to the remote source specs, prunes deleted sources,
remaps path-valued `targetPaths` and `connectionPaths`, and reports invalid
ancestral source paths as recoverable diagnostics.
