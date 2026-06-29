# Variant Composition Contract

`usd-variant-namespace-source` is the first variant-composition node.

Normative references for this contract are only from
`aousd/specifications-public@v1.0.1`:

- `specification/composition/README.md`, especially `Variants`,
  `Computing Variant Selection`, `Namespace Mapping`, and `Strength Ordering`
- `specification/glossary/README.md`, especially `LIVERPS`, `Variant`,
  `Variant Set`, and `VariantSets`
- `specification/document_data_model/README.md`, especially
  `variantSetChildren`, `variantSetNames`, and `variantSelection`
- `specification/stage_population/README.md` for stage population consumption

It sits above the non-variant composition namespace source:

- input: a non-variant base `NamespaceSource` satisfying
  `contracts/capabilities/non-variant-namespace-source-input.json`
- output: a richer `NamespaceSource` with `mode=composed_namespace`
- consumer: `usd-stage-population` through the existing namespace-source
  boundary

This keeps variant selection out of stage population while matching the spec:
variant selection is computed after other composition arcs are available, then
selected variant opinions are inserted at their LIVERPS strength position.
Graph order is evaluation order, not field strength order.

In the current bounded graph, `usd-specializes-namespace-source` is the concrete
terminal non-variant provider, so the generated adapter wires layer stack,
reference, relocates, inherits, and specializes sources before constructing
variants.
Payload opinions remain separate until `usd-composition-arbitrator` merges them into
the terminal composed source. That is not a specializes-specific semantic
dependency of the variant contract.

## Included

The variant provider owns:

- direct selected variants authored on base-source prims
- variant set discovery through `variantSetNames` or `variantSetChildren`
- variant selection from strength-ordered non-variant prim opinions
- selected variant path construction
- mapping the selected variant subtree onto the owning prim path
- recursive selected variants discovered from composed selected-variant
  namespaces
- reuse of a previously decided selection for a matching later variant set in
  the current selected-variant branch
- the `variantSelection` fallback value `{}` producing no selection when no
  previous selection exists
- remapping path-valued field contents such as relationship `targetPaths` and
  attribute `connectionPaths` into the variant-owning namespace
- missing or empty variant selections remaining inert
- multiple direct selected variant sets ordered by final `variantSetNames`
  strength, or `variantSetChildren` when `variantSetNames` is absent
- local and inherits opinions stronger than selected variant opinions
- specializes opinions weaker than selected variant opinions
- effective population `specifier` across base and selected variant specs
- composed child and property name lists
- diagnostics for missing selected variant sets and selected variant specs

The provider declares `child_ordering=composed` and
`property_ordering=composed`, so downstream consumers preserve provider
ordering while still owning masks, active pruning, traversal, and lookup
indexes.

Path-valued field contents authored inside selected variant specs are mapped
the same way as specs: each ObjectPath under the selected variant subtree is
projected under the variant-owning prim path. For `targetPaths` and
`connectionPaths`, ObjectPaths outside that subtree are pruned by this bounded
selected-variant provider rather than emitted with variant-selection paths.

Recursive selected variants are discovered from the composed selected-variant
namespace. A nested variant set can therefore be stored under an already
selected variant path while contributing opinions at the exposed composed prim.
Within the variant arc family, deeper selected variant contributors are stronger
than shallower contributors; same-prim selected variant sets keep the final
`variantSetNames` order.

## Deferred

This contract does not evaluate references and payloads authored inside selected
variants. Later contracts should cover those cases as separate provider layers.

The following also remain deferred:

- external variant fallback maps or stage/session policies not defined by the
  AOUSD v1.0.1 core composition text
- specializes, inherits, or relocates authored inside selected variants
- dictionary and value resolution across variants
- applying offsets or time-varying value resolution
- schema fallback properties
- instancing

## Planned Validation

The unit scope is:

```powershell
py harness/regen_graph.py --scope usd-variant-namespace-source-unit --target python
```

Once generated artifacts exist, validation should be:

```powershell
py harness/regen_graph.py --scope usd-variant-namespace-source-unit --target python --validate
```

The integration scope verifies that the composed selected variant source feeds
stage population:

```powershell
py harness/regen_graph.py --scope usd-variant-stage-population-integration --target cpp --validate
```
