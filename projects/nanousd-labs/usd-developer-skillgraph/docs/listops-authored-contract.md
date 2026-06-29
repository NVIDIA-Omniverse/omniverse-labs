# Authored ListOps Contract

`contracts/handles/listops-authored.handle.json` defines the standalone unit
surface for generated authored list operation values.

`usd-listops-authored` depends on:

- `usd-tokens` for token item identity
- `usd-foundational-values` for eligible scalar element domains
- `usd-paths` for `ObjectPath` item validation

The current scope covers single-layer authored storage only. It stores explicit,
prepend, append, and delete item lists natively, applies repeated authored
subfields with last-one-wins behavior, and materializes the tagged JSON listOp
shape only for adapter/dump output.

The core value model stores typed native items:

- `ObjectPath` items are path handles/references
- `Reference` items are native composition arc records with optional asset,
  optional path, offset, and scale
- `Payload` items are native composition arc records with optional asset,
  optional path, offset, and scale
- `token` items are token handles/references
- `string` items are native strings
- integer items are fixed-width integer values

JSON test inputs and future USDA text syntax are boundary concerns. The handle
adapter may coerce JSON strings/numbers into typed native items, and JSON
objects into native `Reference`/`Payload` arc records. The USDA parser may map
authored syntax into the same typed API, but neither parsing surface is part of
the core listOp value model.

It intentionally does not perform cross-layer listOp composition or value
resolution.

```powershell
py harness/regen_graph.py --scope usd-listops-authored-unit --target python --validate
py harness/regen_graph.py --scope usd-listops-authored-unit --target cpp --validate
```
