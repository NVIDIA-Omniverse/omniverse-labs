---
name: ovrtx-lidar-runtime-capture
description: Develop and validate a native OVRTX LiDAR capture path in the public add-on, including sensor/product identity, typed raw-return decoding, timing, coordinate frames, and truthfulness gates. Use when developers customize the add-on for LiDAR. The current multi-sensor LdrColor probe validates routing only and must not be presented as LiDAR output.
---

# OVRTX LiDAR runtime capture

Use `ovrtx-sensor-capture` to establish multi-sensor creation, selection,
stepping, iteration, and cleanup. Then use `extend-ovrtx-lidar-capture` for the
implementation work.

## Required technical surface

1. Add a documented sensor/product schema and a typed native builder/decoder for
   raw returns. Probe capabilities before calling it.
2. Preserve sensor path, RenderVar path, simulation ID, sample time, return ID,
   coordinate frame, units, and valid/invalid status through every layer.
3. Decode finite positions plus any supported range, intensity, ring/channel,
   timestamp, or return index without inventing absent fields.
4. Validate an isolated target with measured pose and bounds before a complex
   scene. Check axis/unit conversion and first-surface behavior numerically.
5. Test no-return, timeout, partial/iterator, terminal error, reset, and cleanup
   paths. Run one real native capture after unit tests.

Rendered depth, RGB edges, mesh vertices, random samples, and interpolated point
clouds are diagnostics or derived approximations, never native LiDAR. Summarize
the implemented capability and exact unsupported fields; create visualization
or report artifacts only when requested.
