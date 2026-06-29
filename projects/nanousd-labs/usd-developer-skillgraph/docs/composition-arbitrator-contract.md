# Composition Arbitrator Contract

This document describes the central composition-arbitrator contract. The normative
source is `aousd/specifications-public@v1.0.1` only, especially
`specification/composition/README.md`, `specification/glossary`, and
`specification/stage_population/README.md`.

## Purpose

`usd-composition-arbitrator` is the place where generated implementations should
evaluate composition opinions to form composed prims and properties. The
stage-population skill remains a consumer: it receives a `NamespaceSource` and
builds traversal/query indexes, but it does not choose between references,
payloads, inherits, variants, relocates, specializes, or local opinions.

This fills the gap in the current generated shape, where several arc-specific
namespace sources can be stacked but no single contract says where all opinions
are compared to form a prim.

## Model

The layer builds strong-to-weak opinion stacks for composed prim and property
paths. Each opinion stores typed provenance:

- arc family token: local, inherits, variants, relocates, references, payloads,
  or specializes
- source identifier separate from semantic spec path
- exposed composed scene path
- namespace mapping paths
- typed field map keyed by field tokens

Recursive reference opinions and recursive or implied inherit or specialize
opinions supplied by bounded arc sources remain individual composition
opinions. Their source-visible layer strength, namespace depth, source-visible
order, namespace mapping, and authored/implied status where applicable must
survive until the composition arbitrator forms the final opinion stack.

Path-valued field contents in those typed field maps must already be in
composed scene coordinates before the namespace-source facade reaches stage
population or a generated backend. That includes relationship `targetPaths` and
attribute `connectionPaths` listOp<ObjectPath> entries. Elements that cannot be
mapped by the relevant namespace mapping are pruned for external or
variant-selection source namespaces, but same-namespace arcs may preserve paths
that are already in composed coordinates.

The output is both an inspection surface, such as `compose_prim_index`, and a
normal `NamespaceSource` facade for stage population.

For instanceable prim descendants, the composition arbitrator also exposes bounded
instance-filtered prim and property indexes. These exclude local descendant
opinions and include supported non-local composition arc opinions that come from
arcs authored on the instanceable prim, or recursively on source prims of those
arcs.

## Strength

The contract requires AOUSD v1.0.1 LIVERPS family order:

```text
Local > Inherits > Variants > Relocates > References > Payloads > Specializes
```

Graph evaluation order is separate from this field-strength order. For example,
variants may be evaluated after specializes so the selection can observe all
non-variant opinions, but selected variant opinions must still be inserted
stronger than specializes in the final stack.

Specializes remain globally weakest. Loaded payload opinions participate only
when a payload opinion source supplies them.

## Deferred

This first layer is intentionally bounded. It does not claim recursive arc
discovery beyond recursive references and recursive/implied inherit and
specialize opinions supplied by bounded arc sources, payload asset loading,
unloaded payload policy, variant fallbacks, full value resolution, value clips,
schema fallback properties, or implementation-specific prototype query APIs.
Namespace remapping of path-valued field contents and instance-descendant
filtering for supported non-local composition arc families are still in scope;
they are not full listOp composition or value resolution.
Generated sources should declare `composition=partial` and
`value_resolution=partial` until later contracts cover those behaviors.
