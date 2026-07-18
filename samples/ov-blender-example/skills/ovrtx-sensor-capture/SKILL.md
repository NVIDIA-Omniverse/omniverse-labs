---
name: ovrtx-sensor-capture
description: Customize and validate OVRTX sensor declaration, selection, stepping, and native readback for one or multiple render products. Use when developers add camera-like sensors or debug sensor identity and output routing in the add-on. This skill probes capabilities and does not claim every sensor family is already implemented.
---

# OVRTX sensor capture

Use the public client in `addon/` and its tests. Sensor identity is
the USD product path, not a Blender object name or output filename.

1. Run the focused `tests/test_ovrtx_runtime_client.py` and
   `tests/test_ovrtx_session.py` controls, then inspect a compatible native
   probe under `scripts/` before adapting it.
2. Supply an explicit fixture, worker, native-client path, sensor paths, matching
   RenderVar paths, dimensions, time, and caller-owned output JSON.
3. Require successful worker start, simulation creation with every requested
   sensor, one bounded sample step, exact per-RenderVar reads, decoded frames,
   finite dimensions/data, deletion, and shutdown.
4. Run `test_ovrtx_runtime_client.py` and `test_ovrtx_session.py`; retain tests
   for deduplication, changed-sensor session identity, selected-product routing,
   iterator pages, and terminal errors.

The current public probes cover LdrColor camera products. For depth,
semantic, LiDAR, or another sensor schema, first probe native capabilities and
then use the corresponding extension skill. Do not synthesize output from mesh
sampling or postprocessing while labeling it native sensor readback.
