# nanousd Read-Only Backend Contract

`contracts/capabilities/nanousd-readonly-c-api-backend.json` defines the first
bounded contract for generating a backend for nanousd's C API. This now covers
the exported backend ABI entry point and complete vtable fill, but it is still
a read-only slice that consumes the current
skillgraph composed-stage-open, stage-population, document-model,
foundational-value, path, token, and listOp contracts.

The target shape is a backend that exports `nanousd_create_backend_v1`, returns
a non-null `NanousdBackend_v1`, and populates every function pointer slot.
Read-only and utility vtable entries point to generated implementations.
Mutating and deferred entries are still present, but they are deterministic
unsupported stubs that return failure without changing stage state.

Crash-safety is part of the contract. A generated backend may fail a compliance
assertion because a feature is unsupported or incomplete, but it must not crash
the calling process. Every C ABI function is total for null handles, invalid
handles, null input pointers, malformed text, out-of-range indices, missing
values, type mismatches, and unsupported features. No C++ exception may cross
the C ABI boundary.

## Backend Entry Point

The generated backend must expose the nanousd ABI surface directly:

- `nanousd_create_backend_v1` has C linkage.
- The returned `NanousdBackend_v1*` is non-null and stable for the backend
  lifetime.
- Every vtable function pointer is non-null.
- Unsupported mutating operations and deferred read behavior are represented by
  real function pointers to deterministic stubs, never by null slots.
- Non-owning `const char*` return slots never return null. Failure or missing
  string results return a stable empty string unless the nanousd C API
  explicitly defines an owning null-pointer failure.
- Owning string results such as `nanousd_write_usda_string` and
  `nanousd_diagnostics_json` return either null on failure or memory compatible
  with `nanousd_free_string`. They must not return string literals or
  handle-local buffers.
- Opaque handle factories either return null or a fully initialized generated
  handle whose companion functions are null-safe.
- `nanousd_open(NULL)` and `nanousd_open("")` return null. Failed non-null
  opens may return an invalid stage handle with an error string, but all stage
  functions must safely accept null and invalid handles.

The generated C++ target also emits `generated/cpp/CMakeLists.txt`. That CMake
project builds the nanousd read-only backend shared library and metadata
adapter when `NANOUSD_INCLUDE_DIR` points at a checkout containing
`nanousd/nanousd_backend.h`.

Use CMake as the authoritative build path for this backend. The Windows
`build.ps1` helper may still exist for older adapter-only checks, but the
backend shared library used by nanousd benchmarks is built by the generated
CMake target.

## Read-Only Surface

The first backend contract covers:

- stage open/close/valid/error and root-layer path reads
- contributing layer enumeration when the generated source can report it
- stage metadata reads for authored numeric, string, and token-like fields
- prim traversal by index, path, default prim, child index, and child name
- prim path, name, typeName, kind, active, defined/abstract, and instanceable
  queries
- attribute enumeration, typeName reads, and authored default reads
- scalar, vector, quaternion, matrix, string, token, asset, and homogeneous
  array attribute reads
- exact authored time-sample key/value reads for supported value shapes
- relationship and connection target reads
- path utility operations by delegating to `usd-paths`
- standalone listOp create/introspection/combine/free utilities and authored
  prim listOp introspection by delegating to `usd-listops-authored`
- stateless math utilities that operate only on caller-provided arrays

Path, listOp, and math entries are classified as utility functions, not
deferred behavior. They must be implemented as typed generated functions with
explicit null and invalid-input handling; a generic pointer-shaped fallback stub
is not sufficient for these slots.

The contract intentionally does not require first-class vector, quaternion, or
matrix payload structs in the generated backend. A backend may interpret the
document-model payload array using the field's stored type identity and
foundational payload metadata. The observable requirement is that typed C API
readers behave as if they were reading the USD type named by the field.

## Runtime Stage-Open Slice

This contract now includes a runnable stage-open backend for supported local
USDA fixtures. The generated `nanousd_open` path uses graph-owned
`usd-composed-stage-open` orchestration, then serves the benchmark's first
read-only calls through the `NanousdBackend_v1` vtable:

- `open`, `close`, `isvalid`, and `error`
- `nprims`, `prim`, `primpath`, `defaultprim`, and `freeprim`
- prim path/name/typeName/kind/active/defined/abstract/instanceable queries
- `nattribs`, `attribname`, `hasattrib`, and `attribtype`
- diagnostics, root layer path, layer count, and layer path queries

The bounded runtime slice remains partial. Recursive/chained sublayer,
reference, payload, and variant-contained dependency expansion behind
`nanousd_open` is still deferred, but direct local dependency opening must go
through `usd-composed-stage-open` and final prim/property arbitration must go
through `usd-composition-arbitrator`.

## Typed Value Mapping

The backend must not infer shapes from an untyped JSON array. It consumes the
stored USD type, checked-read agreement rules, and payload metadata from the
foundational and document-model contracts.

Important mapping rules:

- `half` and half-shaped values may be read through float C API readers by
  converting at the C boundary.
- semantic aliases such as `point3f`, `normal3f`, `color3f`, `frame4d`, and
  their array forms may satisfy readers for their underlying typed payload.
- matrix readers flatten row-major payloads into `double[9]` or `double[16]`.
- USD quaternion payloads use `i,j,k,w`; nanousd C API quaternion readers return
  `w,i,j,k`, so backend code must reorder components explicitly.
- generic heterogeneous arrays and dictionaries do not satisfy numeric array
  readers.
- zero-copy `nanousd_arraydata*` is optional and may return null until a later
  performance contract requires contiguous native array storage.

## Deferred

The following remain outside this first PR:

- all setters, creation, clear/block, composition arc writes, variant writes,
  schema registration, and file serialization
- USDC/USDZ parsing or writing
- full asset resolver behavior beyond the current resource/layer-open graph
- recursive runtime sublayer, reference, and payload composition behind `nanousd_open`
- schema fallback values, schema registry semantics, and full `isa`/`hasapi`
  behavior
- instancing and prototype queries
- xformOp stack evaluation
- interpolation, clips, and spline evaluation
- complete nanousd diagnostic category parity

## Test Shape

`goldens/unit/nanousd-readonly-backend/nanousd-readonly-backend.json` is a
contract suite. It establishes the expected C API partition and typed reader
mapping, verifies the exported entry point and non-null vtable population
policy, and checks that a tiny USDA fixture can be opened and traversed through
the generated backend vtable.

Future generated adapters should validate this scope with a command like:

```powershell
py harness/regen_graph.py --scope nanousd-readonly-backend-unit --target cpp --validate
```

Backend performance comparison is covered separately by
`contracts/performance/nanousd-readonly-backend.performance.json` and
`harness/nanousd_backend_benchmark.py`. The generated backend must now complete
the supported fixture runs; generated-vs-baseline timing ratios remain optional
until we add performance gates.
