# Payload Composition Contract

`usd-payload-namespace-source` is the first direct loaded payload opinion
source.

It does not sit above an existing reference namespace source. It consumes
payload arc sites and already-opened payload layers:

- input: payload-bearing prim paths and source-visible payload listOp values,
  plus already-opened payload `Layer`s
- output: mapped payload opinions in composed scene coordinates
- consumer: `usd-composition-arbitrator`, which merges local, reference,
  payload, and other opinions into a final namespace source

This keeps the first payload contract focused on namespace composition rather
than resolver policy, file I/O, or the optional population-mask payload
inclusion control.

## Included

The payload provider owns:

- direct external payloads supplied as payload arc sites
- path-only internal payloads within the payload arc layer namespace
- supplied already-opened payload layers keyed by authored asset identifier
- asset-plus-path target payloads
- asset-only target selection through payload layer `defaultPrim`
- mapping the target prim subtree under the payload-bearing prim path
- remapping path-valued field contents such as relationship `targetPaths` and
  attribute `connectionPaths` into the payload-bearing namespace
- payload-contributed prim and property opinions
- payload child and property name lists
- bounded references authored inside loaded payload target subtrees while their
  referenced layers are already supplied
- arc-local `layerRelocates` mappings authored in loaded payload layer stacks
  and carried by nested reference records
- diagnostics for missing payload layers and missing target prims

The provider may expose the namespace-source shape for conformance tests, but
it is not responsible for LIVERPS arbitration against local or reference
opinions.

Path-valued field contents authored under payload specs are mapped the same
way as specs: each ObjectPath under the payload target subtree is projected
under the payload-bearing prim path. For `targetPaths` and `connectionPaths`,
ObjectPaths outside that subtree are pruned by this bounded external-payload
provider rather than emitted in payload-layer coordinates. For path-only
internal payloads, paths outside the payload target subtree stay in the shared
composed namespace.

References authored inside a loaded payload target subtree are expanded while
their referenced layers are already supplied. These nested opinions remain
payload-family contributions because the payload arc introduced the loaded layer
stack into the composed stage. If the loaded payload layer stack authors
`layerRelocates`, the nested record carries that relocate namespace mapping so
source paths are hidden, relocated target paths are exposed, stale target
collisions are suppressed, deleted sources are pruned, and path-valued fields
are remapped through the accumulated payload/reference plus relocate mapping.

## Deferred

This contract does not resolve asset identifiers or open files.
`usd-composed-stage-open` owns the optional runtime path that turns authored
direct payload asset identifiers into opened layer stacks for the adapter
`payloadInclusion=include` scenario. This support is optional under AOUSD Core
1.0.1 because stage population may provide a payload-inclusion flag; it is not a
baseline compliance requirement.

The following also remain deferred:

- explicit unloaded payload state, payload load masks, and load/unload mutation
- recursive and chained payloads
- inherits, specializes, variants, and root-layer relocates
- merging payload opinions with local/reference opinions
- cross-layer payload listOp composition before arc-site construction
- dictionary and value resolution across payloads
- applying layer offsets to time-varying values
- schema fallback properties
- instancing

## Planned Validation

The unit scope is:

```powershell
py harness/regen_graph.py --scope usd-payload-namespace-source-unit --target python
```

Once generated artifacts exist, validation should be:

```powershell
py harness/regen_graph.py --scope usd-payload-namespace-source-unit --target python --validate
```

Stage population should consume a final composed namespace source once
`usd-composition-arbitrator` consumes those payload opinions. Actual payload resource
loading and the optional bounded adapter payload inclusion scenario are covered
separately by `docs/composed-stage-open-contract.md`.
