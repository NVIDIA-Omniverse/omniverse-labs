---
name: renderer-gpu-interop
description: Share renderer output zero-copy across GPU APIs and runtimes (Vulkan↔CUDA, Metal/MPS, DLPack to Warp/PyTorch) and own GPU resource lifetime and teardown across that boundary. Use when Codex wires external-memory/-semaphore interop, hands a render target to a training/compute consumer without a CPU bounce, or debugs interop fd ownership, cross-API sync, double-buffered overlap, ARC/handle lifetime, or teardown-order crashes.
---

# Renderer GPU Interop

## Purpose

The headline value of a headless sensor renderer is often that its pixels never touch the CPU between raygen and the consumer (a Warp/PyTorch training tensor). That payoff lives entirely in the interop seam — and the seam is where the scariest, *silent* bugs live: leaked kernel fds, use-after-free across the API boundary, desynced double buffers, and teardown-order crashes that only fire at process exit. This skill is the knowledge for crossing that boundary safely. Pairs with `renderer-realtime-sensor` (the sensor-output side) and `renderer-simulation-loop` (the per-step update side).

## Ownership is a transfer protocol — name the exact transfer point

- An exported memory fd (`VK_KHR_external_memory_fd` → `cuImportExternalMemory`, or the reverse) is *consumed* by exactly one call **on success**; **on failure the producer still owns it and must `close()` it.** Define and document that single transfer point, or you leak a kernel fd every frame (or double-close → `EBADF`).
- The consumer's view — a DLPack tensor wrapping interop VRAM — does **not** own the memory: its deleter must be a **no-op**. The producing allocation owns the buffer and outlives the per-frame view; a real deleter frees memory the renderer still owns → use-after-free next frame.
- On a **unified-memory backend** (Apple Silicon), the whole external-memory dance collapses: a `Shared` buffer *is* the staging buffer and the consumer reads its `contents` pointer directly. Do not port the fd/external-memory plumbing onto UMA — stub it and record why. (And a host-sync call like `synchronizeResource:` is *invalid* on Shared storage.)

## Sync has modes that are not interchangeable

- A CPU-fence wait serializes producer and consumer. An **imported timeline semaphore** waited on the consumer's stream is what lets the GPU wait *overlap* with host work. Choose by whether you need overlap — the semaphore import is cheap; the memory import is the expensive part.

## Double-buffered overlap: one producer index, readers derive theirs

- The overlap invariant (consumer reads frame N−1 while the producer writes N) holds only if there is **one** write index owned by the producer and every reader computes `read = 1 − write_idx` — never its own counter (a second counter desyncs the buffers). A frame-skip path must leave that index *untouched* so stale readers re-see the last good slot.
- Know and *document* tolerated asymmetries instead of hiding them: e.g. under caller-owned ("inverted") memory the set-A/set-B alternation may drop every other frame's writes — ship single-buffered there and state why it is harmless (the latest writes to slot 0 are always visible) rather than pretending overlap works.

## Inverted ownership dodges a sync slow-path — at the cost of strict teardown order

- Holding active consumer-side imports of producer-allocated memory can trip a slow mode (e.g. in `cudaStreamSynchronize`). Inverting it — the consumer allocates and exports, the renderer imports — sidesteps that, **but** the renderer must then be torn down (or re-pointed at fresh fds) *before* the caller frees its allocation. Reversed free order → the GPU dereferences freed memory.

## Resource lifetime and teardown is a dependency graph, not a list

- Destroy consumers before producers, and wait-idle first: descriptor sets / pipelines that *reference* a buffer before the buffer; cached command buffers before the command pool; memory pools before the device. Annotate every destroy with its "must come before X because…" reason inline — an out-of-order destroy is a use-after-free that usually only crashes at *exit*.
- Make destroys **idempotent** and call them unconditionally — a resource that was uploaded but never built still leaks if its free is gated on the "built" flag.
- Any state change that recreates descriptor sets invalidates cached command buffers recorded against the old sets — invalidate the command-buffer cache in the *same* call.
- After you drop the CPU mirror (a `finalize` step), defend the GPU buffers against a spurious rebuild request — refuse to re-upload from nothing and nuke correct state.
- **Handle-runtime hazards differ by API.** Under ARC (Objective-C++), allocate handle structs with `new`/`delete` (value-init nils the `id<MTL…>` fields; the destructor releases them) and **never `memset` the whole struct** (it clobbers ARC's strong refs) — yet `memset` *is* correct on POD GPU-input structs (instance descriptors, push-constant scratch). The discriminator is "does this struct hold a managed handle," not "is it a struct." Release is assignment-to-nil, not an explicit call.
- Split teardown into **persistent** (device / shader library / pipeline — scene-independent and expensive) vs **per-scene** (BLAS/TLAS/scene buffers); the per-scene destroy must not touch persistent state, or you re-pay shader compilation per scene swap.

## Process-global coexistence hazards

- The NVIDIA driver may refuse a fresh EGL context **after Vulkan-WSI has loaded in the same process** — create the GL context first and have the renderer reuse it.
- OpenGL binds one context per thread *implicitly* (unlike explicit Metal/Vulkan handles): own GL on a dedicated worker thread, serialize all GL calls through it, and never mutate the embedder's main-thread GL state.

## Validate as a boundary

- Reproduce with a tiny producer→consumer loop: assert the consumer reads the producer's *latest* frame with zero CPU copies on the hot path; watch the fd table (growth per frame is the leak signature); and test teardown explicitly — construct, hand off, free in the documented order *and* the wrong order, under a sanitizer.

## Anti-Patterns

- Leaving fd ownership undefined → leak on failure or double-close.
- A DLPack / consumer view with a real (non-no-op) deleter.
- A second reader-side buffer counter instead of deriving from the producer's index.
- Porting external-memory fd plumbing onto a unified-memory backend.
- `memset` over a struct holding ARC/managed GPU handles.
- Destroying producers before consumers; gating a free on a "built" flag.
- A fresh EGL context after Vulkan in-process; touching the embedder's GL state off the worker thread.

## Handoff

Report the interop path (which API exports/imports memory + semaphore), the ownership/transfer points and free order, the buffering scheme and any tolerated asymmetry, what runs on the device vs CPU, and the teardown-order + fd-leak evidence. Defer the physics→render contract to `renderer-simulation-loop`.
