---
name: renderer-shader-host-layout-sync
description: Keep host C/C++ structs and the shaders that read them byte-for-byte in sync — SSBO/UBO/push-constant layout, binding indices, and shared constants across GLSL/MSL/SPIR-V. Use when Codex adds or changes a struct shared between host code and a shader, wires descriptor/binding layouts, shares a push-constant block across pipelines, or debugs "wrong pixels with no crash" layout-drift bugs.
---

# Renderer Shader/Host Layout Sync

## Purpose

A renderer is two programs — host code and shader code — reading the same bytes through two compilers that pad, align, and lay out independently. When their view of a struct, a binding index, or a constant drifts, the symptom is the *worst kind* of bug: no crash, just wrong pixels, because `array[i]` reads from the wrong offset for every `i >= 1`. It reads like a material or shading bug; it is a layout bug. This skill makes the host↔shader contract enforced at build time instead of discovered in an image. It is the GPU-side companion to `contract-first-ffi` (which owns the host↔language-binding boundary) and `renderer-backend-port` (cross-language alignment differences).

## Pin the layout at the host translation unit, not in your head

- Any struct shared host↔shader must be byte-identical. The shading language applies its own alignment (a member needing 16-byte alignment grows the whole struct), so a struct that "looks identical" can mismatch. Bake the expected `sizeof` as a named constant and `static_assert` it — plus `offsetof` for every load-bearing field — in the backend translation unit, so adding or reordering a field fails the *build*, not the GPU. (Canonical war story: a host material struct at 192 bytes vs a shader struct at 112 rendered instanced geometry with the wrong material — silent until the size assert was added.)
- Treat the assert as the authority; do not transcribe the numbers anywhere else. The point is that one place fails loudly when the layout moves.

## Changing a shared struct is a checklist, not an edit

- Bumping a shared struct requires, together: update the host struct, update the shader's mirror struct, update the size/offset asserts, and **clear the persisted pipeline / SPIR-V cache** — a stale cache loaded against new shader bytecode produces wrong or crashing pipelines. A "how to change this safely" comment at the struct should name every step.

## One file owns the binding layout; everything mirrors it

- With many SSBO/UBO/AS bindings shared across raygen/hit/miss/compute shaders, pick **one file as the authoritative source** of binding numbers and have the host build its descriptor-set layout at the *same* indices. A second pass (deferred shading, a debug view) must mirror the producer's bindings exactly, or it binds — and reads — the wrong buffer. The same rule covers shared *constants* (a K-buffer depth, a workgroup size): one definition, mirrored with a guard comment, never two literals. A compute shader's `local_size_x` and the host's `ceil(count / local_x)` dispatch divisor are the same number in two places — drive both from one build-time define.

## A shared push-constant block has fixed offsets — declare the whole block

- Reusing one push-constant struct across pipelines (single-camera, tiled, curve) lets shared closest-hit/miss code read common fields (a ground plane, a scene scale) at the *same byte offsets* regardless of which raygen ran. Each consuming shader must declare the **full** block even if it uses two fields, or the offsets shift; pad explicitly to the alignment the shared fields need.
- (Surprising, and worth a comment where it lives:) flags are sometimes smuggled into the unused float slots of a shared matrix push-constant and read back via `floatBitsToUint()`. The trap is the storage convention — a GLSL `mat4` is column-major, so the host's row-major struct order maps to specific `[col][row]` indices; get the mapping wrong and you read a different flag.

## Know your block-layout rule and don't change it silently

- `std430` vs scalar block layout (`GL_EXT_scalar_block_layout`) changes how a `vec3` packs with a following scalar; the renderer's structs may rely on scalar packing to stay 16-byte aligned without explicit padding. Switching the layout qualifier silently reinterprets every buffer — it is part of the contract, not a free toggle.
- Bind by reflected size: when a backend binds a uniform/argument struct by reflection or a fixed byte length, the shading-language padding makes the required buffer *larger* than the live fields; allocate and send the **full reflected size** (e.g. pad a 20-byte payload to 48 because a `uint3`/`float4x4` member forces 16-byte alignment) or the bind is rejected or undersized.

## Validate the layout, not just the image

- A reduced fixture that reads element `[1]` (not just `[0]`) of every shared array exposes stride/offset drift that a single-element scene hides. Capture the relevant term AOV (material ID, base color) and confirm element N matches authored data before trusting beauty. Keep the size/offset asserts in every build configuration.

## Anti-Patterns

- A shared struct with no compile-time size/offset assert ("it looks the same").
- Bumping a shared struct without clearing the pipeline/SPIR-V cache.
- Binding indices or shared constants duplicated as literals in two files.
- A shader that declares only the push-constant fields it uses (offset shift).
- Flipping `std430`↔scalar, or sending only the live-field size of a reflected struct.
- Diagnosing a stride/offset bug as a material or shader-math bug.

## Handoff

Report which structs/bindings/constants are shared host↔shader, where each is asserted, the block-layout rule in force (std430/scalar), the pipeline-cache-clear step if a struct changed, and the element-`[1]` AOV evidence that the layout is correct. Cross-reference `renderer-backend-port` for cross-language alignment and `contract-first-ffi` for the host↔binding boundary.
