<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Third-Party Notices

This file records the third-party software and content used by `ovfmi`. It
covers direct Python dependencies, packages handled by the setup scripts,
optional runtime components, external prerequisites, bundled demo assets, and
standards used by the sample code.

The ovfmi Apache-2.0 license does not replace the upstream licenses of
third-party components. Packages resolved at installation time carry their
authoritative license texts in their wheels or source archives. Preserve those
texts when redistributing a resolved environment, generated FMU/SSP archive,
SDK bundle, or native binary package.

This inventory is not a lockfile-generated SBOM. Several dependencies are
unpinned, and package installers resolve transitive dependencies. A distributor
must generate notices from the exact environment and artifacts it distributes.

## 1. Direct Python dependencies

The following packages are declared by `pyproject.toml` and
`demoapp/pyproject.toml`. They are installed from a configured Python package
index and are not vendored in this repository.

| Package | Project constraint | License | Use | Upstream |
|---|---|---|---|---|
| `FMPy` | `==0.3.25` | BSD-2-Clause, plus bundled component notices | FMI/SSP parsing and simulation | <https://github.com/CATIA-Systems/FMPy> |
| `NumPy` | `>=2.2` | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Array and tensor handling | <https://github.com/numpy/numpy> |
| `glfw` / pyGLFW | `>=2.8` | MIT | OpenGL window and input handling | <https://github.com/FlorianRhiem/pyGLFW> |
| `PyOpenGL` | `>=3.1` | BSD-style, with bundled component notices | OpenGL Python bindings | <https://github.com/mcfletch/pyopengl> |
| `pytest` | `>=8` | MIT | Test runner installed with the demo application | <https://github.com/pytest-dev/pytest> |

Important license locations in installed distributions include:

| Package | Files to preserve |
|---|---|
| `FMPy` | `fmpy-*.dist-info/licenses/LICENSE.txt`, `fmpy/cswrapper/license.txt`, `fmpy/fmucontainer/documentation/LICENSE.txt`, and `fmpy/remoting/license.txt` |
| `NumPy` | `numpy-*.dist-info/licenses/` and license files shipped inside `numpy/` |
| `glfw` | `glfw-*.dist-info/licenses/LICENSE.txt` |
| `PyOpenGL` | The upstream `license.txt` and bundled `OpenGL/DLLS/*COPYING*` files |
| `pytest` | `pytest-*.dist-info/licenses/LICENSE` |

FMPy and NumPy ship additional component notices. Preserve their complete
license directories instead of copying only the primary license file.

## 2. NVIDIA runtime packages

Users install these packages in the base Python environment before running
setup. The project validates their presence but does not select an index, pin a
version, or redistribute them.

| Package | License reported by reviewed package metadata | Use |
|---|---|---|
| `ovrtx` | NVIDIA Proprietary Software | RTX rendering |
| `ovstage` | NVIDIA Proprietary Software | Shared USD data-plane stage |
| `ovphysx` | `LicenseRef-NVIDIA-Omniverse` | Optional rigid-body physics |

Preserve each resolved package's own license and notice files when
redistributing it. Reviewed packages contain these locations:

| Package | Files to preserve |
|---|---|
| `ovrtx` | `ovrtx/THIRD-PARTY-NOTICES.txt`, package license files, and component `PACKAGE-LICENSES/` directories |
| `ovstage` | `ovstage/THIRD-PARTY-NOTICES.txt` and package license files |
| `ovphysx` | `ovphysx-*.dist-info/licenses/LICENSE.txt`, `ovphysx-*.dist-info/licenses/ovphysx-LICENSES.zip`, and notices shipped with its documentation |

Any dependencies resolved by these packages, such as `packaging`, must also be
included in a distribution-specific license inventory.

## 3. Setup, build, and optional Python packages

| Package | Project use | License |
|---|---|---|
| `setuptools>=77` | PEP 517 build backend declared by the root project | MIT |
| `usd-core` | Unpinned package installed in `demoapp/.usd_venv` for isolated USD parsing | `LicenseRef-TOST-1.0` |
| `cuda-python` | Optional CUDA/OpenGL zero-copy display support | NVIDIA Software License |
| `Pillow` | Optional PNG output selected with `--png` | MIT-CMU, with bundled component notices |

Preserve the license directory from the resolved `setuptools`, `usd-core`,
`cuda-python`, and Pillow distributions when those packages are redistributed.
Pillow's full license file includes notices for bundled image-codec components.

## 4. Transitive Python dependencies

FMPy 0.3.25 declares dependencies including `attrs`, `jinja2`, `lark`, `lxml`,
`msgpack`, `nbformat`, and `numpy`. Pytest and the NVIDIA runtime packages also
resolve their own dependencies. These packages are not declared directly by
all ovfmi manifests and may vary by platform and installer state. Include every
resolved distribution and its license files in a distribution-specific SBOM
and notice bundle.

## 5. External build and runtime prerequisites

The following components are used from the host system and are not vendored by
this repository:

| Prerequisite | Use | License family |
|---|---|---|
| Python 3.10-3.13 | Application runtime | Python Software Foundation License |
| Git and Git LFS | Source and asset checkout | GPL-2.0-only and distribution-specific notices |
| Visual Studio 2022 Build Tools, GCC, or Clang | C++17 demo FMU compilation | Toolchain-specific |
| OpenGL and X11 runtime libraries | Live display | Vendor/distribution-specific |
| NVIDIA display driver | RTX rendering | NVIDIA driver license |
| CUDA Toolkit | Optional CUDA/OpenGL interop | NVIDIA CUDA Toolkit EULA |
| CMake | Optional `demoapp/fmi_usd_helper` build | Apache-2.0 and bundled component notices |
| OpenUSD 25.11 headers | Optional `fmi_usd_helper` build | `LicenseRef-TOST-1.0` |
| ovphysx SDK | Optional libraries for `fmi_usd_helper` | NVIDIA Omniverse/package-specific terms |

Generated FMU and SSP archives can contain binaries produced by the local C++
toolchain. Include applicable compiler runtime notices when distributing those
archives.

## 6. Bundled USD, MDL, mesh, image, and texture assets

The repository includes demo USD stages, MDL materials, meshes, screenshots,
and textures:

| Asset group | Paths | Notice |
|---|---|---|
| ovfmi-authored demo stages and screenshots | `demoapp/usd/*.usda`, `fmu-ball-test.png`, `physx-sim-test.png` | NVIDIA project/demo material unless a file states otherwise |
| Conveyor and box assets | `demoapp/usd/conveyor/` | Includes NVIDIA Omniverse/SimReady-derived USD, MDL, mesh, and texture content |
| Demo model sources | `demoapp/fmu/`, `demoapp/ssp/` | NVIDIA project/demo material unless a file states otherwise |

The conveyor layers contain SimReady provenance metadata and references to
NVIDIA Omniverse/Isaac content locations. No separate OSS asset license file is
present in the checked-in asset tree. Treat these assets as separately licensed
NVIDIA content and verify redistribution rights for the intended release
channel. USD layers also reference runtime-provided MDL materials, which remain
subject to the licenses of the packages that provide them.

## 7. Standards and specification materials

| Standard | Project use | Notice |
|---|---|---|
| FMI (Functional Mock-up Interface) | Demo FMU metadata, runtime behavior, and USD-FMI mappings | FMPy carries notices for FMI headers and schemas from the MODELISAR consortium and Modelica Association Project FMI. Preserve those notices when redistributing FMPy or copied specification materials. |
| SSP (System Structure and Parameterization) | Demo `.ssd` files and SSP runtime behavior | Preserve notices associated with any SSP specification files or tooling included in a distribution. |

The checked-in `demoapp/fmu/fmi2/fmi2_minimal.h` is an NVIDIA-authored minimal
set of declarations and carries its own NVIDIA SPDX header; it is not a copied
FMI SDK header.

## 8. Redistribution checklist

Before distributing `ovfmi` or a package containing its dependencies:

1. Include this file or an updated equivalent.
2. Generate an SBOM and dependency notice from the exact resolved environment.
3. Include complete license files from redistributed wheels, source archives,
   native SDKs, generated model archives, and binary bundles.
4. Include the NVIDIA runtime packages' own third-party notices when those
   packages or their contents are redistributed.
5. Confirm redistribution rights for `demoapp/usd/conveyor/` content.
6. Do not classify NVIDIA proprietary package licenses as open-source licenses.

## 9. Sources reviewed

- `pyproject.toml`
- `demoapp/pyproject.toml`
- `demoapp/setup.ps1`
- `demoapp/setup.sh`
- `demoapp/fmi_usd_helper/CMakeLists.txt`
- `README.md`
- `docs/USD-FMI-SCHEMA.md`
- checked-in asset provenance strings and SPDX headers
- installed package metadata and license inventories reviewed on 2026-07-22
  for FMPy 0.3.25, NumPy 2.4.3, glfw 2.10.2, PyOpenGL 3.1.10, pytest 9.1.0,
  Pillow 12.2.0, usd-core 26.5, ovrtx 0.4.0.346409, ovstage 0.1.0.346039,
  and ovphysx 0.5.9
