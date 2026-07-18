---
name: ovrtx-render-sequence
description: Develop and validate contiguous OVRTX sequence capture with stable session identity, explicit frame/sample boundaries, native readback, reset behavior, and deterministic output ordering. Use when developers customize the add-on for animation or multi-frame capture; the public render-spike baseline is one frame and does not itself prove sequence support.
---

# OVRTX render sequence

Use `animation-quality-and-frame-range` for Blender-side motion and
`extend-ovrtx-render-sequences` for implementation. Establish the one-frame
native control first with `ovrtx-current-scene-workflow` and require the render
result to identify engine `OVRTX_EXAMPLE`.

## Sequence contract

1. Keep one compatible session alive across adjacent frames; define which
   changes require reset/recompose.
2. Apply one explicit Blender frame/subframe and all corresponding scene updates
   atomically. Carry source frame, simulation time, session ID, product path,
   and requested/completed samples into readback identity.
3. Wait on the runtime's readiness/sample boundary, not a sleep duration. Read
   each product once, validate exact dimensions/dtype/finite pixels, then
   acknowledge and advance.
4. Bound queues and backpressure. Never drop, reorder, duplicate, or silently
   overwrite a frame to maintain cadence.
5. Test first/middle/last frames, structural reset, cancellation, timeout,
   terminal output, resume policy, and cleanup. Compare captured frame identity
   to evaluated Blender motion; a successful API call is not sequence proof.

Run relevant pinned session, viewport-handoff, and runtime-client tests before a
real three-frame native smoke. Add encoding only after the lossless ordered
frames pass.
