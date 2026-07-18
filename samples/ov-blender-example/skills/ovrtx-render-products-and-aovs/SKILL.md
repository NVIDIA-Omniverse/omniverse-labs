---
name: ovrtx-render-products-and-aovs
description: Probe, customize, and validate OVRTX render-product and RenderVar/AOV readback in the public add-on and native client. Use when developers add beauty, HDR, depth, ID, semantic, or other products, or when diagnosing product identity, selection, iteration, dimensions, and decoding. Availability is capability-specific, not implied by this skill.
---

# OVRTX render products and AOVs

Start from the public implementation in `addon/`. Do not add a
second bridge or infer a product from an image file.

## Baseline contract

- `sensor_paths` declare render products at simulation creation.
- `selected_sensor_paths` select products for a request/session.
- RenderVar reads use exact paths such as `<product>/LdrColor`.
- The native client exposes typed builders such as
  `build_ReadWorldState_ldr_color` and, when supported,
  `build_ReadWorldState_hdr_color`, plus matching decode helpers.
- Iterator pages, terminal status, completed-sample identity, dimensions, and
  per-product identity are part of the acceptance contract.

Run the focused public tests before customization:

```text
python3 -m pytest -q \
  tests/test_ovrtx_runtime_client.py \
  tests/test_ovrtx_session.py
```

For a real-runtime diagnostic, begin with an existing probe under `scripts/`
that uses the same `ovrtx_runtime_client` and `ovrtx_session` boundary. Extend
it only through the typed product builders and decoders under test, using
explicit native-client path, worker command, fixture USD, sensor/product path,
RenderVar path, and caller-owned output. A successful LdrColor read proves only
that selected product and decoder; it does not prove arbitrary AOV families.

For a missing product family, route implementation to the matching
`extend-ovrtx-*` skill. Require a typed builder/decoder, exact product identity,
finite shape-correct data, terminal/error propagation, cleanup, and focused
unit plus real-runtime coverage before advertising it as supported.
