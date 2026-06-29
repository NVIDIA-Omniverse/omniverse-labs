# Value Resolution Contract

`contracts/handles/value-resolution.handle.json` defines the independent
value-resolution boundary. It consumes a strength-ordered stack of typed
opinions and returns resolved metadata, relationship targets, connection
targets, or attribute values without evaluating composition arcs or stage
population.

The AOUSD v1.0.1 core specification is the only normative source for this
contract. OpenUSD behavior is not used to fill gaps.

## Dependencies

Composition owns the opinion stack. The resolver assumes its input has already
been ordered strongest to weakest and that path-valued authored fields have
already been remapped into composed scene coordinates where composition requires
that.

The document model owns typed field storage. The resolver works on target-native
field values, tokens, paths, dictionaries, authored listOps, value blocks, time
samples, splines, schema fallback records, and value clip contexts. Tagged JSON
appears only at the adapter boundary.

Stage population owns stage traversal and property handles. Stage query code may
call this resolver, but it must not reimplement value resolution locally.

## Covered Rules

This PR covers the AOUSD value-resolution pipeline:

- default-time attribute resolution over authored `default` opinions
- value blocks and blocked time samples stopping weaker values
- schema fallback acquisition from prim definitions and applied schemas
- timed queries over authored `timeSamples`
- exact, held, and linear time-sample resolution with layer-offset retiming
- spline evaluation for held, linear, curve, block, and extrapolated segments
- value clip resolution, including manifests, active clips, times mapping,
  missing-value handling, and clip strength
- ordinary strongest-authored metadata fields
- `custom`, `specifier`, `typeName`, and `variability` special metadata rules
- recursive dictionary metadata combining
- generic listOp metadata combining, including `targetPaths` and
  `connectionPaths`
- layer metadata resolution from the root layer spec only
- raw and forwarded relationship target queries
- raw attribute connection queries

```powershell
py harness/regen_graph.py --scope usd-value-resolution-unit --target cpp --validate
```
