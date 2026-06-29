# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Third-Party Notices

This is the consolidated third-party software inventory for `ovfmi`. It covers
Python runtime dependencies declared by `apps/ov-fmi`, Python packages installed
by the setup scripts, optional NVIDIA runtime packages, external build/runtime
prerequisites, bundled USD/material/texture assets, and standards referenced by
the demo code and documentation.

NVIDIA's own license terms do not override the upstream licenses of these
components. Use of each component is subject to its own license. Where a
component is resolved from PyPI or NVIDIA's package index at install time, the
authoritative license text is the license file distributed with the resolved
wheel or source archive. Preserve those license files when redistributing a
resolved environment, generated FMU/SSP archive, wheel, SDK bundle, or native
binary package.

This file is not a lockfile-generated SBOM. `ovfmi` has unpinned dependencies
and transitive dependencies resolved by package installers, so downstream
redistributors should generate a fresh dependency notice from the exact
resolved environment they ship.

If you discover a missing or incorrect attribution, please open an issue or
pull request against this repository.

---

## 1. Python runtime dependencies

These Python packages are declared directly by `apps/ov-fmi/pyproject.toml` and
are installed when the app package is installed. They are resolved from PyPI at
install time and are not vendored in this repository.

| Package | Required version | License | Scope | Source |
|---|---|---|---|---|
| `fmpy` | `==0.3.25` | BSD-2-Clause, with bundled component notices | Runtime FMI/SSP parsing and simulation | <https://github.com/CATIA-Systems/FMPy> |
| `numpy` | `>=2.2` | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Runtime array/tensor handling | <https://github.com/numpy/numpy> |
| `glfw` / pyGLFW | `>=2.8` | MIT | Runtime OpenGL window/input handling | <https://github.com/FlorianRhiem/pyGLFW> |
| `PyOpenGL` | `>=3.1` | BSD-style | Runtime OpenGL Python bindings | <https://github.com/mcfletch/pyopengl> |

Notes:
- `fmpy==0.3.25` declares additional runtime dependencies such as `attrs`,
  `jinja2`, `lark`, `lxml`, `msgpack`, `nbformat`, and `numpy`. These are
  resolved transitively by the Python installer and are not declared directly
  by this repository. Include their license files if redistributing a resolved
  Python environment.
- FMPy includes separate notices for FMI header/schema files, SUNDIALS/CVODE,
  co-simulation wrapper binaries, remoting binaries, bundled icons/materials,
  Bootstrap Icons, Font Awesome Free Icons, and MPack. Preserve the license
  files contained in the resolved FMPy wheel.
- NumPy wheels include several embedded third-party components under permissive
  licenses. Preserve all NumPy `LICENSE*`, `COPYING`, and bundled component
  license files from the resolved wheel or source archive.
- PyOpenGL's published metadata identifies a BSD license. The upstream
  repository license text attributes Michael C. Fletcher and contributors and
  includes notices for code inherited from the PyOpenGL 2.x series.

### License and notice locations

| Package | License / notice file to preserve from resolved distribution |
|---|---|
| `fmpy` | `fmpy-*.dist-info/licenses/LICENSE.txt`; `fmpy/cswrapper/license.txt`; `fmpy/fmucontainer/documentation/LICENSE.txt`; `fmpy/remoting/license.txt` |
| `numpy` | `numpy-*.dist-info/licenses/LICENSE.txt` and bundled license files under `numpy/` |
| `glfw` | `glfw-*.dist-info/LICENSE.txt` |
| `PyOpenGL` | Upstream `license.txt` from <https://github.com/mcfletch/pyopengl> |

---

## 2. Python packages installed by setup scripts

The Windows and Linux setup scripts install these packages in addition to the
declared `ov-fmi` runtime package. They are resolved from PyPI or NVIDIA's
package index at setup time and are not vendored in this repository.

| Package | Version/source in setup | License | Scope |
|---|---|---|---|
| `ovrtx` | `ovrtx==0.3.0.312915` from `https://pypi.nvidia.com`, unless `OVRTX_DIR` is supplied | `LicenseRef-NvidiaProprietary` / NVIDIA Software License Agreement | Required renderer runtime |
| `ovphysx` | `ovphysx==0.4.9` from `https://pypi.nvidia.com`, unless skipped | `LicenseRef-NVIDIA-Omniverse` | Optional physics runtime |
| `usd-core` | Unpinned PyPI package installed into isolated `apps/ov-fmi/.usd_venv` | `LicenseRef-TOST-1.0` | USD parsing fallback subprocess |
| `cuda-python` | Unpinned PyPI package installed only when `INSTALL_CUDA_PYTHON=1` or `-InstallCudaPython` is used | `LicenseRef-NVIDIA-SOFTWARE-LICENSE` | Optional CUDA/OpenGL zero-copy display path |

Notes:
- `ovrtx` is an NVIDIA package, not an OSS dependency. The installed
  `ovrtx==0.3.0.312915` package includes its own `THIRD-PARTY-NOTICES.txt`
  and package license files for separate components. Preserve those files if
  redistributing the package or a bundle containing it.
- `ovphysx==0.4.9` package metadata reports license files `LICENSE.txt` and
  `ovphysx-LICENSES.zip`. Preserve those files if redistributing the package
  or a bundle containing it.
- `usd-core` package metadata reviewed for version 26.5 reports
  `LicenseRef-TOST-1.0`, the Tomorrow Open Source Technology License 1.0 used
  by OpenUSD. Preserve the license file from the exact resolved `usd-core`
  package.
- `cuda-python` is separately licensed NVIDIA software. It is optional and is
  installed only for the CUDA interop display path.

### License and notice locations

| Package | License / notice file to preserve from resolved distribution |
|---|---|
| `ovrtx` | `ovrtx/THIRD-PARTY-NOTICES.txt`; `ovrtx-*.dist-info/licenses/*`; package `LICENSE` files |
| `ovphysx` | `LICENSE.txt`; `ovphysx-LICENSES.zip` |
| `usd-core` | `usd_core-*.dist-info/licenses/LICENSE.txt` |
| `cuda-python` | `cuda_python-*.dist-info/licenses/LICENSE` |

---

## 3. Source/reference submodules

This repository declares a pinned `ovrtx` source/reference submodule for
matching source/reference code and offline override workflows. Normal setup
installs the `ovrtx` wheel and does not build the submodule from source.

| Component | Pinned ref | License | Distribution | Source |
|---|---|---|---|---|
| `ovrtx` | `556ceb6a41e4bf6f4ff7b4526943a9366386543c` | `LicenseRef-NvidiaProprietary` | In-tree submodule at `third-party/ovrtx` when initialized | <https://github.com/NVIDIA-Omniverse/ovrtx> |

Notes:
- The `ovrtx` submodule's top-level metadata identifies the project as NVIDIA
  proprietary software governed by the NVIDIA Software License Agreement and
  product-specific NVIDIA terms.
- If the submodule is initialized and redistributed, include its top-level
  `LICENSE` file and any third-party notices included by the checked-out
  commit.

---

## 4. External build and runtime prerequisites

These tools and system components are required or optionally used to build and
run the demos. They are not vendored in this repository and are not
redistributed by `ovfmi`, but downstream users may need them installed.

| Prerequisite | Version/source in project | License |
|---|---|---|
| Python | Python 3.10 through 3.13 documented in `README.md` | Python Software Foundation License |
| Git and Git LFS | External prerequisites documented in `README.md` | GPL-2.0-only and other notices, depending on distribution |
| C++17 compiler | Visual Studio 2022 Build Tools, `g++`, or `clang++` for demo FMU builds | Toolchain-specific |
| CMake | Required only for optional `apps/ov-fmi/fmi_usd_helper` build | Apache-2.0 and BSD notices, depending on distribution |
| OpenUSD headers | Optional `fmi_usd_helper` uses OpenUSD v25.11 headers | `LicenseRef-TOST-1.0` |
| OpenGL / X11 runtime | Runtime display prerequisites documented in `README.md` | Vendor/system-specific |
| NVIDIA driver | Runtime prerequisite for RTX rendering | NVIDIA driver license |
| CUDA Toolkit | Optional prerequisite for CUDA/OpenGL zero-copy display | NVIDIA CUDA Toolkit EULA |
| ovphysx SDK | Optional input for `fmi_usd_helper` linking experiments | NVIDIA Omniverse / package-specific terms |

Notes:
- `apps/ov-fmi/fmi_usd_helper/CMakeLists.txt` can clone OpenUSD v25.11 headers
  and link against USD libraries from an ovphysx SDK. That helper is optional
  and version-sensitive.
- The generated demo FMU/SSP archives may contain compiled binaries produced by
  the local C++ toolchain. Preserve any applicable toolchain/runtime notices if
  redistributing generated archives.
- CUDA Toolkit and the NVIDIA driver are NVIDIA proprietary prerequisites, not
  OSS components.

---

## 5. Optional test and developer dependencies

These Python packages are documented or referenced for tests and optional
developer workflows. They are resolved from PyPI and are not vendored in this
repository.

| Package | Version/source in project | License | Scope |
|---|---|---|---|
| `pytest` | Unpinned optional install documented in `README.md` | MIT | Test runner for `apps/ov-fmi/tests` |
| `Pillow` | Unpinned optional install mentioned by `main.py --png` help | MIT-CMU | Optional PNG frame output |

Notes:
- `pytest` package metadata reviewed for version 9.1.1 reports MIT.
- `Pillow` package metadata reviewed for version 12.2.0 reports MIT-CMU and
  contains bundled third-party notices in its license file. Preserve the
  package license file if redistributing.

### License and notice locations

| Package | License / notice file to preserve from resolved distribution |
|---|---|
| `pytest` | `pytest-*.dist-info/licenses/LICENSE` |
| `Pillow` | `pillow-*.dist-info/licenses/LICENSE` |

---

## 6. Bundled USD, MDL, mesh, image, and texture assets

The repository includes demo USD stages, MDL material files, meshes, screenshots,
and textures used by the `ovfmi` examples. These assets are redistributed as
part of this project when the repository content is distributed.

| Asset group | Files | Notice |
|---|---|---|
| ov-fmi demo USD stages and screenshots | `usd/ov-fmi/`, `fmu-ball-test.png`, `physx-sim-test.png` | NVIDIA project/demo material unless a file states otherwise |
| Conveyor demo USD, MDL, mesh, and texture assets | `usd/conveyor/` | Includes Omniverse/MDL materials and generated USD layers. No separate OSS asset license file was present in the sparse checkout when reviewed. |
| Generated FMU and SSP inputs | `fmu/`, `ssp/` | NVIDIA project/demo material unless a file states otherwise. Generated archives may include compiled binaries and copied FMI/SSP metadata. |

Known provenance strings visible in checked-in USD layers include:

- `E:\Simready\Collected_Conveyor3\ConveyorScene.usd`
- `../simready_usd/sm_box_corrugated_brown_b13_01.usd`
- `../simready_usd/sm_box_flat_a13_01.usd`

Notes:
- Treat `usd/conveyor` assets as NVIDIA Omniverse/SimReady or otherwise
  separately licensed assets unless a future source file identifies a more
  specific OSS asset license.
- Verify redistribution rights for these assets before publishing them outside
  the intended repository or package.
- USD layers reference MDL materials such as `OmniPBR` and `OmniGlass`; those
  material definitions are provided by the NVIDIA Omniverse runtime packages
  and are subject to their own package terms.

---

## 7. Standards and specification materials

The project implements and documents interoperability with the following
standards. The standards themselves are not vendored as standalone documents in
this repository, but related schema/header materials may be present in FMPy or
generated FMU/SSP archives.

| Standard | Project usage | Notice |
|---|---|---|
| FMI, Functional Mock-up Interface | Demo FMUs, `modelDescription.xml` files, FMPy runtime usage, and USD-FMI schema docs | FMPy includes copies of FMI header and model-description schema files from the FMI standard downloads and carries the associated notice from the MODELISAR consortium and Modelica Association Project "FMI". Preserve FMPy's license files if redistributing FMPy or copied FMI materials. |
| SSP, System Structure and Parameterization | Demo `.ssd` files and SSP runtime behavior | The project references SSP 1.0 and uses FMPy's SSP support. Preserve notices from any SSP standard files or tooling if copied into a distribution. |

---

## 8. Redistribution checklist

Before distributing `ovfmi` or a built package that includes dependencies:

1. Include this file or an updated equivalent.
2. Generate a fresh dependency notice from the exact resolved Python
   environment, including transitive dependencies.
3. Include full license files from every redistributed wheel, source archive,
   native SDK, generated FMU/SSP archive, and binary bundle.
4. Include `ovrtx` and `ovphysx` license and third-party notice files if those
   packages or their contents are redistributed.
5. Verify redistribution rights for `usd/conveyor` assets and any generated
   FMU/SSP archives that contain copied third-party binaries, schemas, or SDK
   files.
6. Do not treat NVIDIA proprietary package licenses as OSS licenses.

---

## 9. Sources reviewed

- `README.md`
- `apps/ov-fmi/pyproject.toml`
- `apps/ov-fmi/setup.sh`
- `apps/ov-fmi/setup.ps1`
- `apps/ov-fmi/fmi_usd_helper/CMakeLists.txt`
- `docs/USD-FMI-SCHEMA.md`
- `.gitmodules` from repository `HEAD`
- Package metadata and license files for representative dependency wheels
  reviewed on 2026-06-24.
