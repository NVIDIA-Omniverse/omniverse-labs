---
name: renderer-backend-port
description: Port an existing nanoUSD renderer to a new GPU backend or API (e.g. Vulkan to Metal, DX12, or WebGPU) or stand up a second backend behind the shared renderer contract. Use when Codex adds a backend, translates shaders and RHI calls across graphics APIs, decides what to share versus reimplement versus stub, keeps backends at ABI and feature parity, or hits cross-API transform, struct-alignment, command-encoder, or resource-residency pitfalls.
---

# Renderer Backend Port

## Purpose

A renderer reaches a new platform by swapping the backend under a stable seam, not
by forking the whole renderer. The port's job is to keep the platform-agnostic core
and the public contract identical, reimplement only the RHI and the shaders, and
report capability differences honestly instead of faking them. A port that edits the
core or the public ABI has found a leak in the seam — fix the seam, not the core.

## Workflow

1. **Identify the three layers and the seam.** Public C ABI on top → platform-agnostic
   core (API impl, scene extraction, camera, material setup) → RHI seam (an opaque
   GPU handle plus buffer/pipeline/frame/trace verbs) → backend implementation +
   shaders. Only the bottom two move. In the nanoUSD renderers the seam is `gpu.h`,
   the core is `renderer.c`/`scene.c`/`camera.c`/`material.cpp`, and a backend is
   `gpu_<api>.{c,mm}` plus its shader set. Read the existing seam before porting; do
   not invent a parallel one.

2. **Keep the public ABI identical and include it verbatim.** The new backend
   satisfies `renderer-capability-abi`: same canonical header, same capability
   bitmask, a backend-info query that names the new backend. Do not redefine flags
   locally. Differences are expressed as *cleared capability bits*, not as ABI edits.

3. **Scope by capability; stub the rest explicitly.** Enumerate the features the
   target genuinely cannot do (vendor-specific interop, vendor upscalers, windowing
   assumptions) and the ones it does differently. Each unsupported entry point returns
   the unsupported result and clears its capability bit — never a no-op success. Keep
   a short "what was dropped and why" ledger; it feeds `renderer-feature-validation-matrix`
   and `legacy-retirement-clean-history`.

4. **Translate shaders deliberately.** Decide ahead-of-time versus runtime shader
   compilation up front — it dictates how kernels share helpers and whether includes
   resolve. Keep the reference shaders beside the port as the spec. Two recurring
   traps: cross-language struct and push-constant *alignment* (a struct that is N
   bytes in the host language may be padded larger by the shading language — size the
   host upload to the shading language's layout, not the host's); and runtime
   compilers that do not resolve user includes (put co-dependent kernels in one
   translation unit). A shared shader-codegen library (e.g. MaterialX) can have a
   mature generator for one target and an immature one for another — pin to the
   version where *your* target's generator reached parity and force a fresh fetch, or
   an older shared build silently lacks your backend.

5. **Honor cross-API invariants — the ones that silently corrupt.** Frame each as an
   invariant plus a failure signature, because the APIs differ but the failures
   repeat:
   - *Transform convention.* Agree on row- versus column-vector and where translation
     lives; a flat element copy across that boundary silently drops translation and
     the object renders at the origin (and per-frame transform updates look like
     no-ops). Use one named conversion at the GPU boundary and verify with a
     translate-and-re-render test, not a static frame. This exact bug shipped in more
     than one nanoUSD backend's raycast path. Know precisely *where* the storage
     reinterpret cancels: storing a row-major host matrix into a column-major shader
     `mat4` is an implicit transpose that the shader's `vec * mat` multiply order
     cancels — the same host bytes are correct in GLSL `M * v` and MSL `v * M` with no
     host conversion — but a matrix consumed by fixed-function hardware (an
     acceleration-structure instance transform) does NOT cancel and needs an explicit
     element transpose. Treating both as one rule corrupts one path.
   - *Command-encoder/buffer state.* On some APIs, ending and reopening an encoder
     (e.g. for a mid-frame acceleration-structure refit) drops every binding. Re-bind
     through a single shared helper used by both the initial and the post-refit path;
     a forgotten rebind reads garbage with no error.
   - *Acceleration-structure residency.* A TLAS may reference BLASes the API will not
     keep resident automatically; mark them used or they are evicted and rays
     dereference freed memory.
   - *Resource storage modes, texture usage flags, and memory accounting* differ;
     mirror the reference's semantics (including which buffers count toward reported
     memory) so telemetry and downstream tests stay comparable.
   - *Output color-space/encoding.* Match the reference's swapchain/target encoding or
     pixel-equality tests fail by a fixed, mystifying delta.

6. **Prove parity; do not assert it.** Use `renderer-image-comparison-testing` with
   backend-specific goldens and, where parity is a contract, cross-backend diffs on
   the reduced corpus. A port phase is done when its gate matches the reference
   backend on a fixture, not when it compiles and renders something.

7. **When the port reveals a bug in shared-core logic, fix it at the source of truth
   and propagate.** Ports surface latent bugs in the supposedly-shared code (a wrong
   transform copy, a test that passed tautologically). Fix it in the canonical core
   and push the fix to every backend in the same change. Patching only the new backend
   is how "shared" files drift thousands of lines apart and how a fix found in one
   backend never reaches the others.

## Anti-Patterns

- Forking the core per backend instead of sharing it behind the RHI seam — it drifts
  (observed: a scene loader diverging thousands of lines between two backends).
- Faking an unsupported feature as no-op success instead of clearing its capability
  bit and returning the unsupported result.
- `memcpy` across a matrix-convention or struct-alignment boundary "because both are
  3x4 floats."
- Re-deriving bindings after an encoder swap by guesswork instead of one shared bind
  helper.
- Fixing a shared-core bug only in the backend that surfaced it.
- Calling "it builds and renders a triangle" parity.

## Handoff

Report the layers touched (should be only the RHI and shaders), the capabilities
supported versus stubbed with reasons, parity evidence against the reference backend
on named fixtures, and any shared-core fixes that must be propagated upstream to the
other backends.
