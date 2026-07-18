---
name: ovrtx-semantic-aov-capture
description: Develop and validate semantic or ID RenderVar capture through the public OVRTX add-on and native client. Use when developers customize semantic class, instance, segmentation, or ID AOV support. This skill requires typed native readback and durable USD semantics; it does not treat colored postprocessing as native evidence.
---

# OVRTX semantic AOV capture

Use `ovrtx-render-products-and-aovs` for product/readback mechanics and
`extend-ovrtx-semantic-aovs` for implementation.

1. Define durable semantic class/instance ownership in USD separately from the
   visualization. Preserve stable prim identity through composition.
2. Add an explicit product/RenderVar path and typed native builder/decoder.
   Probe the installed native module's capabilities; fail as unsupported when
   the semantic builder or decoder is absent.
3. Preserve dtype, dimensions, background/unlabeled value, class/instance
   mapping, sample time, product identity, and terminal status.
4. Test duplicate labels, missing labels, hidden/excluded prims, remapped prim
   paths, iterator pages, wrong dtype/shape, terminal errors, and cleanup.
5. Validate a tiny fixture with two known labeled objects and one unlabeled
   background before a real scene. Require exact pixel IDs at known regions and
   stable mapping across repeated reads.

Colorizing an ID array is a derivative for inspection. Keep the raw typed array
and mapping authoritative. Do not claim support from a beauty image, material
color substitution, or screen-space postprocess.
