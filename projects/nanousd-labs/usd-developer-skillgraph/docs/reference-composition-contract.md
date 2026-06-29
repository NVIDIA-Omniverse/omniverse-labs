# Reference Composition Contract

`usd-reference-namespace-source` is the first reference-composition node.

It sits above an existing namespace source:

- input: a base `NamespaceSource` plus already-opened referenced `Layer`s or
  bounded referenced layer-stack roots
- output: a richer `NamespaceSource` with `mode=composed_namespace`
- consumer: `usd-stage-population`, through the existing namespace-source
  boundary

This keeps the first reference contract focused on namespace composition rather
than resolver policy or file I/O.

## Included

The reference provider owns:

- direct external references authored on base-source prims
- path-only internal references within the base namespace source
- supplied already-opened reference layers keyed by authored asset identifier
- asset-plus-path target references
- asset-only target selection through referenced layer `defaultPrim`
- mapping the target prim subtree under the referring prim path
- bounded recursive and chained references authored inside supplied referenced
  target subtrees
- composed namespace mappings through chained reference targets
- remapping path-valued field contents such as relationship `targetPaths` and
  attribute `connectionPaths` into the referring namespace
- local opinions stronger than referenced opinions
- same-reference-family strength ordering by namespace depth, then
  source-visible authored list order for sibling arcs
- separable reference-family-only opinion views, or equivalent typed
  provenance, for downstream composition consumers such as relocates
- arc-local `layerRelocates` mappings authored in referenced layer stacks and
  carried by nested reference records
- effective population `specifier` across local and referenced specs
- composed child and property name lists
- diagnostics for missing reference layers, missing target prims, and recursive
  reference cycles

The provider declares `child_ordering=composed` and
`property_ordering=composed`, so stage population preserves provider ordering
and still owns masks, active pruning, traversal, and lookup indexes.

Path-valued field contents authored under referenced specs are mapped the same
way as specs: each ObjectPath under the referenced target subtree is projected
under the referring prim path. For `targetPaths` and `connectionPaths`,
ObjectPaths outside that subtree are pruned by this bounded external-reference
provider rather than emitted in referenced-layer coordinates. For path-only
internal references, paths outside the referenced target subtree stay in the
shared composed namespace.

Nested references inside supplied referenced target subtrees are expanded
transitively while their referenced layers are available. Each nested mapping is
composed with the parent mapping, so a target reached through multiple reference
arcs contributes at the final exposed scene path. Implementations must bound
that traversal, diagnose cycles by authored asset, target path, referring prim
path, and mapping context, and continue without unbounded recursion or crashes.

If the layer stack that authors a nested reference also authors
`layerRelocates`, the nested reference record must carry that relocate namespace
mapping. The provider hides relocated source paths, exposes relocated target
paths, suppresses stale target collisions from the referenced target layer,
maps remote-source reads back to the relocated source specs, and remaps
path-valued fields through the accumulated reference plus relocate mapping.

When multiple reference arcs contribute to the same exposed scene path, arcs
authored deeper in namespace are stronger than arcs authored higher in
namespace. Sibling reference arcs authored at the same prim follow the
source-visible authored list order.

Even when a generated target exposes a flattened NamespaceSource view for
testing, it must retain enough reference-family opinion data for downstream
providers to consume referenced opinions without also consuming local
base-source opinions at the same path.

## Deferred

This contract does not resolve asset identifiers or open files. A later
reference loader should consume `usd-layer-open` and `usd-resource-protocol` to
turn authored reference asset identifiers into opened layers.

The following also remain deferred:

- payload loading and payload load policy
- inherits, specializes, variants, and root-layer relocates
- cross-layer reference listOp composition
- dictionary and value resolution across references
- applying layer offsets to time-varying values
- schema fallback properties
- instancing

## Planned Validation

The unit scope is:

```powershell
py harness/regen_graph.py --scope usd-reference-namespace-source-unit --target python
```

Once generated artifacts exist, validation should be:

```powershell
py harness/regen_graph.py --scope usd-reference-namespace-source-unit --target python --validate
```

The integration scope verifies that the composed reference source feeds stage
population rather than stopping at source-level inspection:

```powershell
py harness/regen_graph.py --scope usd-reference-stage-population-integration --target cpp --validate
```

Actual reference resource loading should be covered separately by a future
reference-open contract.
