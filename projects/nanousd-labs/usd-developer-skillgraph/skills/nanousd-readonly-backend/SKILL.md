---
name: nanousd-readonly-backend
description: Use this skill when implementing or verifying a generated read-only backend for nanousd's C API.
metadata:
  author: NVIDIA
---

# nanousd-readonly-backend

Use this skill when implementing or verifying a generated read-only backend for
nanousd's C API.

## Sources

- `include/nanousd/nanousdapi.h`
- `include/nanousd/nanousd_backend.h`

The source headers define a broad backend vtable. This skill owns only the
first read-only slice of that surface.

## Provides

- Read-only `NanousdBackend_v1` vtable partition
- Exported `nanousd_create_backend_v1` entry point with C linkage
- Complete non-null `NanousdBackend_v1` function pointer population
- Total crash-safe behavior for all C ABI slots, including unsupported slots
- Runtime `nanousd_open` and traversal behavior for supported local USDA
  fixtures through `usd-composed-stage-open`
- Stage and prim handles backed by generated stage-population results
- C API attribute readers backed by document-model checked reads
- Typed value conversion for scalar, vector, matrix, quaternion, array, token,
  string, and asset values
- Relationship target reads
- Composition diagnostics exposed through `nanousd_diagnostics` and
  `nanousd_diagnostics_json`
- Time-sample key and exact-sample reads
- Path and listOp utility delegation
- Deterministic unsupported stubs for mutating C API entries

## Contract

This skill owns:

- `contracts/capabilities/nanousd-readonly-c-api-backend.json`
- `contracts/handles/nanousd-readonly-backend.handle.json`
- `contracts/performance/nanousd-readonly-backend.performance.json`
- `contracts/capabilities/semantic-runtime-types.json` as a cross-cutting
  dependency

The backend must consume the graph's stage, document-model, foundational value,
path, token, and listOp capabilities. It must not keep a separate parser-local
stage or infer typed values by reparsing canonical JSON dumps.

Do not store stage paths, prim paths, property paths, field keys,
child/property names, typeName/kind tokens, variant set names, or selected
variant names as raw strings in backend query state. The nanousd C API exposes
C strings at the ABI boundary, but generated backend state should retain graph
path, token, name, field, and value abstractions internally.

Generated backend code must export `nanousd_create_backend_v1`, return a
non-null `NanousdBackend_v1*`, and assign every vtable field. Read-only and
utility fields point to generated implementations. Mutating or deferred fields
point to deterministic unsupported stubs; they must not be left null.

Generated C ABI functions must never crash the caller. Treat null handles,
invalid handles, null input pointers, malformed text, missing values,
out-of-range indices, type mismatches, and unsupported features as ordinary
failure cases with deterministic nanousd return values. Do not let C++
exceptions cross the C ABI boundary. Non-owning `const char*` return slots must
return a stable empty string on failure rather than null.
Owning string APIs, including `nanousd_write_usda_string` and
`nanousd_diagnostics_json`, must return null on failure or memory compatible
with `nanousd_free_string`; never return a borrowed string literal from those
slots.

Path, listOp, and math slots are utility slots. Do not satisfy them with a
generic unsupported fallback. Generate typed path handles, standalone listOp
handles, and stateless math functions with explicit null and invalid-input
behavior. Opaque handle factories must return either null or a fully initialized
handle whose companion functions are total.

Generated C++ output must include portable CMake build infrastructure. The
CMake project should build the nanousd backend shared library and adapter when
`NANOUSD_INCLUDE_DIR` resolves to headers containing
`nanousd/nanousd_backend.h`. Do not rely on the Windows-only `build.ps1` path
as the sole build mechanism.

The runnable benchmark slice must open supported local USDA files through
`usd-composed-stage-open`, receive a graph-owned populated stage, and serve
`benchmark_stage_load` calls through the `NanousdBackend_v1` vtable. Do not
construct reference or payload namespace sources with empty dependency maps when
`usd-composed-stage-open` can discover and open direct local dependency assets.
Do not keep a parser-local shadow stage to answer `nprims`, `prim`, `nattribs`,
diagnostics, or layer path queries.

Diagnostic callbacks must report diagnostics collected by
`usd-composed-stage-open` and stage population. The array result must be
allocated for `nanousd_free_diagnostics`, string fields must be owned by that
array, and `nanousd_diagnostics_json` must return `nanousd_free_string`
compatible memory. Relocate diagnostics use `arc_type = 8` and invalid
relocate category `8` until a richer warning/error taxonomy is contracted.

The backend may interpret shaped foundational payloads through a generic array
container when the field's stored type identity and payload metadata are
available. Dedicated `Vec`, `Quat`, `Matrix`, or typed-span storage can be added
later for performance, but is not required by this first contract.

For quaternion reads, USD payload order is `i,j,k,w`. The nanousd C API exposes
quaternions as `w,i,j,k`; generated backend code must reorder explicitly.

Zero-copy array access is not required in this first contract. `arraydata*`
functions may return null until a later performance contract requires
contiguous native array storage.

## Boundary Guards

Do not implement mutation through this skill. Setters, creation, clear/block,
composition arc writes, variant writes, schema registration, and file writers
must be unsupported stubs in the read-only backend.

Do not bypass the namespace-source/stage-population boundary to recover prims
from parser-local state.

Do not infer type identity by looking at JSON serialization or value text.
Consume document-model typed reads and foundational payload metadata.

Do not claim schema fallback, instancing/prototype, clip, spline, xformOp, or
full diagnostic parity behavior until those contracts exist in the graph.

## Test Obligations

- C API function partition into read-only, utility, unsupported stub, and
  deferred behavior
- exported `nanousd_create_backend_v1` entry point
- all `NanousdBackend_v1` function pointer slots populated with non-null
  generated implementations or unsupported stubs
- generated CMake build metadata for the C++ target
- CMake-built nanousd backend shared library and adapter
- nanousd C API backend benchmark runs through
  `harness/nanousd_backend_benchmark.py`
- composed stage open, validity, traversal, attribute-count, diagnostics, and
  layer-path behavior through the C vtable
- stage/prim/attribute read mapping to graph-owned handles and values
- scalar conversion for half/float/double/timecode and integer domains
- vector and semantic alias reads
- matrix3d and matrix4d row-major flattening
- quaternion order conversion from USD `i,j,k,w` to C API `w,i,j,k`
- homogeneous array copy reads
- exact authored time-sample reads
- deterministic unsupported write stubs
- all cases in `goldens/unit/nanousd-readonly-backend/nanousd-readonly-backend.json`
