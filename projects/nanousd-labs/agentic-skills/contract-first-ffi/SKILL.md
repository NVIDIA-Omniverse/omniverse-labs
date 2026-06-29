---
name: contract-first-ffi
description: Maintain C ABIs and language bindings as exact, testable contracts. Use when evolving public headers, ctypes/cffi/pybind/Rust/JS bindings, shared libraries, struct layouts, GPU/native APIs, or any cross-language boundary where drift can cause crashes or silent corruption.
---

# Contract-First FFI

## Purpose

Make native APIs boring to call from other languages. The binding is not a second implementation; it is a strict projection of the public contract.

## Contract Workflow

1. Start from the canonical native API.
   - Identify the public header or IDL.
   - Treat it as the source of truth for structs, enums, function signatures, ownership, and lifecycle.
   - Remove duplicate local declarations where possible.

2. Match layouts explicitly.
   - For every struct crossing the boundary, verify field order, width, signedness, alignment, and array length.
   - Add native `static_assert(sizeof(...))` and `offsetof(...)` checks when possible.
   - In dynamic languages, keep struct declarations adjacent to comments naming the native equivalent.

3. Make symbols required unless the product supports mixed versions.
   - If the library and binding ship together, bind symbols directly and fail at import/init time when missing.
   - Reserve `hasattr`, `dlsym` optionality, and feature probes for genuinely optional runtime capabilities.
   - Delete “older build” guards when creating a clean version line.

4. Keep ownership explicit.
   - Document who allocates and frees buffers, handles, strings, images, GPU resources, and borrowed pointers.
   - Make borrowed-handle APIs say how long the owner must keep the object alive.
   - Avoid returning raw pointers without a matching unmap/free/close function.

5. Test the boundary, not just internals.
   - Import the binding.
   - Create and destroy the native object.
   - Call at least one function that exercises each important struct.
   - Run one smoke path through the same binding used by real clients.

## Binding Discipline

- Keep binding names close to native names unless idiomatic wrappers add value.
- Wrap errors consistently: native status -> binding exception or result type.
- Never silently ignore failed optional symbol binding unless absence is a supported capability.
- Avoid per-call marshaling for large arrays; pass contiguous buffers and pointers.
- Keep high-frequency data paths zero-copy or single-copy by design.

## Native API Design

Prefer:

- Opaque handles for resources.
- Plain old data structs for descriptors.
- `int`/enum status codes plus `get_last_error`.
- Explicit `create`/`destroy`, `map`/`unmap`, `enable`/`fetch`.
- Separate capability queries for hardware/runtime features.

Avoid:

- ABI-visible C++ types.
- Caller-visible ownership ambiguity.
- Structs that change size without version strategy.
- Global mutable state hidden behind per-instance APIs.
- Binding-only behavior not present in native API.

## Version Policy

Decide before adding compatibility code:

- **Version-locked:** binding and library are built together. Missing symbols are fatal. Delete stale guards.
- **Version-tolerant:** binding supports several deployed library versions. Feature probes are explicit, documented, and tested.
- **Plugin ecosystem:** negotiate ABI version at load time before binding symbols.

Do not mix these policies accidentally.

## Review Checklist

- Public header and binding signatures agree.
- All struct sizes and offsets are verified or manually audited.
- Optional paths are truly optional.
- Errors propagate with actionable text.
- Build scripts install/update bindings after native rebuilds.
- Smoke test uses the installed binding, not an in-tree import accident.
