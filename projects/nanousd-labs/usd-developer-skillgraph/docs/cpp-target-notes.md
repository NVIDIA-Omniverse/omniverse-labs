# C++ Target Notes

The C++ target was generated as a second implementation of
`usda-single-layer`. It uses the same graph nodes as the Python target, with
per-node generated modules under `generated/cpp/`.

## Generated Node Mapping

- `usd-identifiers-and-names` -> `generated/cpp/usd_identifiers.{h,cpp}`
- `usd-foundational-values` -> `generated/cpp/usd_values.{h,cpp}`
- `usd-paths` -> `generated/cpp/usd_paths.{h,cpp}`
- `usd-listops-authored` -> `generated/cpp/usd_listops.{h,cpp}`
- `usd-document-model` -> `generated/cpp/usd_document_model.{h,cpp}`
- `usda-lexical-format` -> `generated/cpp/usda_lex.{h,cpp}`
- `usda-value-parser` -> `generated/cpp/usda_values.{h,cpp}`
- `usda-spec-parser` -> `generated/cpp/usda_parser.{h,cpp}`
- `usda-layer-open` -> `generated/cpp/usda_layer_open.{h,cpp}`
- `usd-layer-open` -> `generated/cpp/dump_layer.exe`

The generated entrypoint is:

```text
generated/cpp/dump_layer.exe
```

## Build System

The C++ target should generate portable CMake build infrastructure alongside
the source artifacts:

```text
generated/cpp/CMakeLists.txt
```

The CMake project builds a shared core library, all adapter executables, the
layer dump command, and the nanousd read-only backend targets when nanousd
headers are available. It writes executable artifacts back to `generated/cpp/`
by default so existing graph `adapter_cmd` paths continue to work on Windows
and Linux. Linux executables intentionally keep the `.exe` suffix because the
graph command paths are platform-neutral strings, and a Unix executable may use
that filename without issue.

Configure with an explicit nanousd checkout when it is not in a default probe
location:

```powershell
cmake -S generated/cpp -B generated/cpp/build -DNANOUSD_INCLUDE_DIR=C:/tmp/nanousd/include
cmake --build generated/cpp/build --config Release
```

The older `generated/cpp/build.ps1` path remains a local Windows convenience,
but it should not be the only generated build route.

## Validation Result

Command:

```powershell
py harness/regen_graph.py --scope usda-single-layer --target cpp --validate
```

Correctness:

```text
OVERALL: 6/6 (100.0%)
```

CLI smoke benchmark:

```text
PASS tiny_layer_parse:       best=9.938 ms   target<=150.000 ms
SLOW large_flat_prim_parse:  best=316.797 ms target<=300.000 ms
PASS many_attributes_parse:  best=326.228 ms target<=400.000 ms
```

The generated C++ artifact manifest was not recorded as accepted because the
full graph validation did not pass the current shared benchmark gate.

## Difference From Python

The C++ target has much lower process startup overhead than the Python target,
which is visible in the tiny fixture. The large flat-prim workflow is slower in
this first C++ generation because the generated representation builds a generic
JSON tree using ordered maps and then serializes the full document model.

That is a useful signal for the skillgraph approach: different target languages
can satisfy the same correctness contract while exposing different data
structure pressure. The next C++ regeneration should probably push ownership of
performance-sensitive layout into the document-model skill rather than letting
the parser regenerate a generic JSON representation.
