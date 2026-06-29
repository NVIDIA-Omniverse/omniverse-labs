# CLAUDE.md — nanousd

This file governs the **`nanousd` component** of the nanousd-labs fleet. It sits in its own directory within the `omniverse-labs` container; the container's root `AGENTS.md` governs the repository as a whole. Paths and commands below are relative to this component's directory (the CMake source dir when you build `nanousd`).

## Source of Truth

The **USD Core Specification** (https://github.com/aousd/specifications-public/tree/main/core) is the sole authoritative reference for nanousd's implementation. All design decisions, naming conventions, data model semantics, and behavioral details must be derived from the specification itself.

**Do not consult OpenUSD (pxr) source code or documentation** unless explicitly asked to. OpenUSD is one implementation of the specification and contains implementation-specific details (e.g. `Usd`-prefixed type names, internal APIs) that do not apply to nanousd. When in doubt, go back to the spec.

## Testing Requirements

**All changes to nanousd source code must be accompanied by tests in both test suites, and both suites must pass.**

### Two test suites

1. **`nanousd_tests`** — C++17 backend tests, links `nanousd_core` + `nanousdapi`
   - Source: `tests/test_main.cpp`
   - Exercises parser internals, stage operations, C++ API, write support

2. **`compliance_test`** — Pure C11 compliance tests, links only `nanousdapi`
   - Source: `tests/compliance/compliance_test.c`
   - Exercises only the public C API (`nanousdapi.h`) — no implementation internals

### Running tests

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DNANOUSD_BUILD_TESTS=ON
cmake --build build --parallel
cd build && ctest --output-on-failure
```

Tests must be run from `${CMAKE_SOURCE_DIR}` (this component's root) to find test data files in `tests/usda/`, `tests/usdc/`, and `tests/compliance/usda/`. The `CMakeLists.txt` configures this via `WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}` in all `add_test()` calls.

### CI

There is no in-tree CI for this component: it lives as a directory in the `omniverse-labs` container, so the standalone GitHub Actions workflow does not run here. Run both suites locally with `ctest` (above) before committing.

### When adding new C API functions

1. Implement the backend function in `src/nanousd_backend.cpp`
2. Declare it in `include/nanousd/nanousdapi.h`
3. Add a test in `tests/compliance/compliance_test.c` using only the C API
4. Add a C++ test in `tests/test_main.cpp` if backend-level coverage is useful
5. Ensure `ctest` passes before committing

## Architecture

- `nanousd_core` — static lib (C++17), all parsing/writing/stage logic
- `nanousd_backend.so` — shared lib, implements the vtable ABI (`nanousd_backend.h`)
- `libnanousdapi.so` / `libnanousdapi.dylib` — shared lib, public C API (`nanousdapi.h`), dynamically loads backend
- `nanousd_core` must be compiled with `POSITION_INDEPENDENT_CODE ON` (already set in `CMakeLists.txt`) because it is linked into `nanousd_backend.so`

## Key Implementation Notes

### PathPool lifecycle

`PathPool` (in `src/path_pool.cpp`, introduced in commit `a6c5ea7`) is a process-singleton that interns `PathData` nodes for the parent-pointer-tree representation of `Path`. Each call to `Path::AppendChild`, `Path::AppendProperty`, or `Path::AppendVariantSelection` allocates a new `PathData*` via `new` and stashes it in `PathPool::entries_`.

**Important:** The pool currently has no destructor. `PathData*` nodes leak across the process lifetime — ASAN run on `compliance_test` (108 USD files) reports `~112 KB / 520 allocs` from `PathPool::Intern` at `src/path_pool.cpp:92`. The header documents this behavior:

> Stage-singleton ownership is *not* used: paths from a discarded stage stay interned, same convention as `Token`'s process-singleton intern pool. Tooling that loads many scenes can call `PathPool::Clear()` between runs.

`PathPool::Clear()` (path_pool.h:97) frees all entries except the root. Per-stage scoping is not yet implemented; if you build long-running tooling that loads many scenes, call `Clear()` at appropriate boundaries.

### Compose regressions

`TestComposeSubLayers()` (`tests/test_main.cpp:2162`) is currently failing as of `a6c5ea7` — the `compose.cpp` overhaul that introduced lazy primindex composition + inherits resolution regressed sublayer composition for at least one fixture. Logged in workspace `ASAN_REPORT.md` Bug 3.

### USDC Writer
- Path element tokens must be pre-added to the token table during `CollectLayer` (before `WriteTokensSection` is called). This is done via the second pass in `CollectLayer` that iterates `paths_.paths` and calls `tokens_.AddOrGet(p.GetName())`. Without this, the TOKENS section would be written before path element tokens are known, causing the parser to fail the token bounds check during path reconstruction.
- The PATHS section uses jump-tree DFS encoding (spec §16.3.8.4.5). Jump semantics: -2=leaf, -1=has child at x+1 only, 0=has sibling at x+1 only, >0=has child at x+1 AND sibling at x+jump.

### USDC Parser
- `CrateTypeId::Token` values are decoded as `Value(Token(...))`, preserving token-typed semantics through write-read roundtrips.
- `CrateTypeId::Specifier` → decoded as `Value(Int)`, handled by `ApplyField`.
