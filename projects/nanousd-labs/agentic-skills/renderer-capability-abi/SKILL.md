---
name: renderer-capability-abi
description: Define and govern the shared renderer C ABI and backend capability contract so multiple backends stay interchangeable and parity is machine-checked. Use when Codex designs or changes the public renderer header, adds a capability/AOV/entry point, reports what a backend supports versus stubs, or wires renderer ABI conformance into CI. Pairs with contract-first-ffi (general FFI hygiene) and renderer-feature-validation-matrix (per-feature status).
---

# Renderer Capability ABI

## Purpose

Multiple backends behind one renderer are only interchangeable if "what this
backend can do" is a single, runtime-queryable, machine-checked contract — not
folklore spread across copied headers. This skill is the renderer-specific
governance layered on top of `contract-first-ffi`: one canonical public header, a
capability bitmask, a versioned backend-info query, an explicit unsupported result,
and a conformance check that fails the build on drift. Defer struct-layout
versioning, symbol-export, and null-safety mechanics to `contract-first-ffi`; this
skill is about the capability *contract* and keeping it singular.

Teach and enforce the **shape, invariants, and source of truth** below — never a
frozen copy of the current flag list or struct fields. The canonical header is the
source of truth; a transcribed list in prose is just a second copy that will drift.

## Principles

1. **One canonical header, included verbatim.** There is exactly one public
   renderer header. Every backend repo consumes *that file* (vendored, symlinked, or
   submoduled), not a hand-copied edition. Name the source of truth explicitly and
   point every backend at it.

2. **Capabilities are a runtime contract, not a compile-time assumption.** Model
   capability as a bitmask — one bit per renderer capability, grouped by concern
   (render modes, AOVs, load paths, scene queries, mutation verbs, interop) — plus a
   versioned, size-stamped backend-info struct and a single query entry point. A
   backend advertises only what it truly does. Callers branch on the queried bits,
   never on a backend name or string.

3. **Unsupported is a first-class result.** Provide a dedicated "unsupported" error
   code. A stubbed entry point returns it and leaves its capability bit clear. This
   is what lets `renderer-feature-validation-matrix` mark a row `degraded`/`stubbed`/
   `blocked` honestly, and what lets `renderer-backend-port` drop a feature without
   lying. A stub that returns success is a correctness bug.

4. **Parity is machine-checked in CI.** A conformance checker parses every backend
   header against the canonical one — required capability macros, the backend-info
   struct layout, the unsupported code, the query signature — and fails on any drift
   or missing symbol, including in the canonical header itself. Wire it as a test
   that gates merges and treat a red conformance check like a red build, not a
   warning. (The nanoUSD parent repo ships exactly this: `tools/check_renderer_abi.py`,
   registered as the `renderer_abi_conformance` CTest — run it before claiming a
   backend conforms.)

5. **Evolve the contract in one place, then propagate in the same change.** Add or
   rename a capability or struct field in the canonical header first, bump the
   info-struct version, then update every backend to match. Size/version stamps let
   older callers stay safe. A backend must never get ahead of the canonical contract
   silently — the live failure mode this skill exists to stop is a capability ABI
   that landed in one backend while the canonical header (and the conformance test)
   stayed behind.

## Anti-Patterns

- A "shared" header that is actually N hand-copied editions — they drift, and one
  ends up ahead of canonical while the checker goes red.
- Branching on backend name/string instead of a queried capability bit.
- Stubs that return success; capabilities advertised but not implemented, or
  implemented but not advertised.
- Adding a capability bit inside a backend instead of the canonical header.
- Transcribing the full flag list into docs/skills/plans instead of pointing at the
  canonical header (a fresh drift source).
- Treating the conformance checker as advisory while it is failing.

## Handoff

Report the canonical header location, the capabilities added or changed (bit +
backend-info version), each backend's supported/stubbed set, and the conformance-check
status (and the exact command that produced it).
