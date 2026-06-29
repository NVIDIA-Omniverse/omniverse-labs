# nanousd-python

> **Part of [`nanousd-labs`](..)** — an experimental fleet that generates USD implementations and stacks from the [USD Core Specification](https://github.com/aousd/specifications-public/tree/main/core). This component is the **Python bindings** over the `nanousd` C ABI. The [fleet README](..) has the full picture — the stack, how the repos fit together, and the skillgraph that drives it.

`nanousd-python` is the Python package for `nanousd`. It provides a small
native API over the public `nanousd` C API and a compatibility shim for Python
code that imports Pixar-style `pxr` modules.

The goal is practical OpenUSD-shaped interoperability for lightweight tools:
load stages, inspect prims, read and author common attributes, walk composition
metadata, exercise variants and arcs, and run selected OpenUSD Python tests
against `nanousd` without depending on a full OpenUSD build.

It is not a full OpenUSD Python implementation. The compatibility layer mirrors
the parts that `nanousd`, `nanousdview`, and the current tests need, and it
keeps unsupported areas explicit instead of pretending to be complete.

## What is in this component

- `nanousd.Stage`, `nanousd.Prim`, `nanousd.Path`, and `nanousd.ListOp`: a
  `nanobind` extension that wraps the public `nanousd` C API directly.
- `nanousd.pxr_compat`: the extracted `nanousdview` `pxr` compatibility shim,
  cleaned up so it lives as a standalone package.
- `from pxr import Usd, Sdf, Gf, ...`: registered automatically when importing
  `nanousd` or `nanousd.pxr_compat`.
- A copied OpenUSD Python test corpus under `tests/openusd_compat/upstream`,
  with a focused harness that runs the subset expected to pass today.

## Current API Shape

The native API is intentionally small and direct:

```python
import nanousd

stage = nanousd.Stage.open("scene.usda")

for prim in stage.traverse():
    print(prim.path, prim.type_name, prim.attribute_names())

cube = stage.get_prim_at_path("/World/Cube")
if cube is not None and cube.has_attribute("size"):
    print(cube.read_double("size"))
```

For authoring:

```python
import nanousd

stage = nanousd.Stage.create()
stage.set_metadata_token("upAxis", "Z")

cube = stage.define_prim("/World/Cube", "Xform")
cube.create_attribute("size", "double")
cube.set_double("size", 2.0)

cube.create_attribute("points", "point3f[]")
cube.set_vec3f_array("points", [
    -1.0, -1.0, -1.0,
     1.0, -1.0, -1.0,
     1.0,  1.0, -1.0,
])

stage.write_usda("out.usda")
```

For OpenUSD-shaped imports:

```python
import nanousd  # registers the synthetic pxr package

from pxr import Usd, Sdf, Gf

stage = Usd.Stage.Open("scene.usda")
path = Sdf.Path("/World/Cube")
v = Gf.Vec3d(1.0, 2.0, 3.0)
```

## Native Binding Coverage

The `nanobind` layer currently exposes:

- Stage open/create, traversal, prim lookup, default prim lookup, root and used
  layer inspection, sublayer paths and offsets.
- Stage metadata, time metadata, USDA/USDC export, USDA string export, and
  diagnostics.
- Prim identity, hierarchy, type checks, API schema application/removal, active
  and instanceable state, specifier, and removal.
- Attribute creation, typed reads/writes, generic `get`/`set`, authored checks,
  interpolation, array values, transforms, defaults, blocks, and time samples.
- Relationships, connections, references, payloads, inherits, specializes, and
  list-op inspection.
- Variant set creation, variant creation, variant selection, and variant
  inspection.
- `Sdf.Path`-style path helpers, list-op helpers, vector math, matrix math, and
  quaternion helpers exposed from the `nanousd` C API.

The native layer should stay close to `nanousdapi.h`. If the C API cannot do
something, the Python package should not invent a large private model for it.

## pxr Compatibility Layer

`nanousd.pxr_compat` is a Python shim that lets selected code continue to use
imports such as:

```python
from pxr import Usd, Sdf, Gf, UsdGeom, UsdShade
```

The shim is useful for tools and tests that only need a read-mostly subset of
OpenUSD behavior. It includes defensive handling for missing C symbols, missing
namespace modules, common `Gf` math and camera helpers, `Tf.Notice` stubs, stage
and prim convenience methods, variant-set accessors, and compatibility fixes
driven by copied OpenUSD tests and `nanousdview` usage.

This layer is a compatibility bridge, not a claim of OpenUSD compliance. It is
acceptable for it to be permissive where tool code needs to keep running, but
the README and tests should stay honest about what is actually implemented.

## Build

From this component:

```bash
python -m pip install -e .
```

From the fleet root (repos checked out as siblings):

```bash
./nanousd-python/build.sh
```

or:

```bash
python -m pip install -e ./nanousd-python
```

When building outside the workspace, point CMake at a `nanousd` checkout or
install:

```bash
python -m pip install -e . --config-settings=cmake.define.NANOUSD_DIR=/path/to/nanousd
```

Requirements:

- Python 3.10 or newer.
- `nanobind` and `scikit-build-core` for the extension build.
- `numpy` for parts of the `pxr` compatibility layer.
- A built or discoverable `nanousd` C API library and headers.

## Tests

Run the package tests with:

```bash
python -m pytest -q
```

The tests cover both layers:

- Native `nanobind` bindings over real `nanousd` stage creation, authoring,
  traversal, attributes, relationships, variants, composition arcs, export, and
  reopen.
- `pxr` compatibility imports and selected API behavior.
- A curated OpenUSD Python corpus copied from a local OpenUSD checkout. The
  copied files keep their upstream license headers and are not all run by
  default. `tests/test_openusd_copied_subset.py` imports selected copied tests
  that are expected to pass against `nanousd-python` today.

At the time this README was written, the focused OpenUSD subset includes full
module runs for:

- `pxr/usd/sdf/testenv/testSdfPath.py`
- `pxr/usd/sdf/testenv/testSdfListOp.py`
- `pxr/usd/sdf/testenv/testSdfAssetPath.py`
- `pxr/usd/usd/testenv/testUsdTimeCode.py`

It also runs the URL-encoded identifier case from:

- `pxr/usd/usd/testenv/testUsdStage.py`

## Design Rules

- Prefer direct `nanobind` bindings for stable, useful `nanousd` C API
  features.
- Keep the native API Pythonic but mechanically close to the C API.
- Use `pxr_compat` for compatibility with existing OpenUSD-shaped Python code.
- Do not imply full OpenUSD compliance until copied OpenUSD tests and real tool
  workloads justify that claim.
- Add tests whenever a new shim method is introduced. A copied OpenUSD test is
  preferred when the behavior comes from OpenUSD.

## Known Gaps

- This package does not replace Pixar OpenUSD Python bindings.
- Many upstream OpenUSD tests are present only as compatibility corpus material;
  they are intentionally not all part of default pytest discovery.
- Some shim behavior is permissive to support tool startup and inspection paths.
  Permissive stubs should be replaced with real `nanousd` C API bindings when
  those features matter.
- Variant selection updates may require reopening or recomposition-safe handle
  refresh in some paths because `nanousd_recompose()` invalidates outstanding
  prim handles.
- Full writable `Sdf.Layer`, `Usd.StageCache`, advanced composition editing,
  complete schema behavior, and complete time-sample semantics are still areas
  to expand.

## Repository Layout

```text
.
├── CMakeLists.txt
├── pyproject.toml
├── src/
│   ├── bindings.cpp
│   └── nanousd/
│       ├── __init__.py
│       └── pxr_compat/
│           ├── __init__.py
│           └── _native_compat.py
└── tests/
    ├── test_native_api.py
    ├── test_pxr_compat_api.py
    ├── test_openusd_copied_subset.py
    └── openusd_compat/
```

## License

Apache-2.0. Copied OpenUSD test files retain their upstream copyright and
license headers.

## Contributing

This project is currently not accepting contributions.
