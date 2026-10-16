# Third-party notices

This inventory covers the source sample at version 0.3.0, its checked-in
material, and the optional dependencies in `requirements-ovui.txt`. It was
checked on 2026-10-01 against the Linux x86-64 and Windows x86-64 release
wheels listed below. Third-party components retain their own copyrights and
licenses. The sample's Apache-2.0 license does not relicense them.

## Release scope

| Material or dependency | How it is used | License / notices |
| --- | --- | --- |
| Service, simulator, shared client and console | Python standard library; `pyproject.toml` declares no runtime dependencies | Sample code: [LICENSE](LICENSE). The separately installed Python interpreter supplies its own [PSF license and notices](https://docs.python.org/3/license.html). |
| `ovui==0.2.0` | Separately installed optional dashboard runtime on Linux and Windows; native libraries, fonts and image assets are in the upstream wheel | NVIDIA proprietary runtime; third-party copyrights and full license texts are collected in [ovui notices](third_party_licenses/ovui-0.2.0.txt). NVIDIA runtime terms are linked below. |
| `nvidia-cuda-runtime-cu12==12.9.79` | Separately installed runtime required by the optional ovui client on Linux and Windows | [Complete license and notices from the wheel](third_party_licenses/nvidia-cuda-runtime-cu12-12.9.79.txt). |
| NVIDIA Kit SDK and extensions, including OpenUSD (`pxr`) | Separately installed SDK selected with `KIT_ROOT` / `KIT_EXECUTABLE`; extensions are declared in `apps/reference.kit` and the client `extension.toml` | Versions are supplied by the user's Kit installation, not pinned or bundled here. Its NVIDIA runtime terms and version-specific OpenUSD and other third-party licenses/notices accompany that SDK. Retain those notices if redistributing it. |
| `setuptools>=61` | Optional Python package build backend, declared in `pyproject.toml`; not used by the checkout launchers or bundled in the sample | MIT; [upstream license](https://github.com/pypa/setuptools/blob/main/LICENSE). Build tooling and its vendored dependencies retain the notices supplied with the installed version. |
| `wheel>=0.38.4` | Python package build tool; minimum version ensures that wheel builds honor the declared license files | MIT; [upstream license](https://github.com/pypa/wheel/blob/main/LICENSE.txt). Not bundled in the sample. |

This source release includes sample Python and launch scripts, configuration
and example JSON, a synthetic USD scene, documentation, SVG diagrams, and
screenshots / emitted example data under `artifacts/reference/`. No third-party
source tree, copied validation schema, font file, texture file, Python wheel,
SDK binary, or submodule is checked in. The scene has no external asset
references. Screenshots illustrate the separately installed Kit and ovui
clients; runtime and font attributions are retained here.

The [Composable Bindings white paper](https://aka.ms/ComposableBindings) is
referenced by the README and architecture documentation; it is not bundled.
OpenUSD API documentation is linked from the USD guide and is not bundled.

The pinned ovui wheel declares `numpy` and `cuda-python` only for its `examples`
extra. This sample does not request that extra. It also does not install ovui
widgets, data adapters, developer skills, or their dependencies. Those packages
are outside this inventory. In particular, this checkout has no `usd-core`,
OpenTelemetry, protobuf, jsonschema, pytest, or CloudEvents schema dependency.

## ovui 0.2.0

Copyright (c) 2025 NVIDIA Corporation (package metadata).
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved
(wheel notices).

The runtime is governed by the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and [Product Specific Terms for NVIDIA Omniverse](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-omniverse/),
as stated in the [ovui 0.2.0 release metadata](https://pypi.org/project/ovui/0.2.0/).

The following inventory is derived from `omni/ui/THIRD_PARTY_NOTICES.md` in
both audited wheels and `omni/ui/resources/fonts/PROVENANCE.md` and `OFL.txt`.
The two wheels' notice texts match after line-ending normalization. Full
copyrights, permission terms and disclaimers are in
[third_party_licenses/ovui-0.2.0.txt](third_party_licenses/ovui-0.2.0.txt).

| Component | Version / revision recorded by the wheel | License |
| --- | --- | --- |
| Dear ImGui | Vendored; version not specified in the wheel notices | MIT |
| Boost.Preprocessor | `c4ea7e40d365` | BSL-1.0 |
| FreeType | `VER-2-13-2` build fallback; system version otherwise | FreeType License (FTL option) |
| GLAD and Khronos headers / specifications | `1ecd45775d96` | MIT and Apache-2.0 |
| GLFW | `3.4` build fallback; system `glfw3` otherwise | Zlib |
| GLM | `bf71a8349481` | MIT option; additional Sun Microsystems notice for `glm/gtc/ulp.inl` |
| md4c | `729e6b8b320c` | MIT |
| stb | `31c1ad374564` | MIT or Unlicense |
| pybind11 inline code in native modules | Version not specified in the wheel notices | BSD-3-Clause |
| Noto Sans, eight styles | `v2.015`, per font provenance | SIL Open Font License 1.1; Copyright 2022 The Noto Project Authors |
| Open Sans SemiBold | Apache-licensed font cut identified by the wheel provenance | Apache-2.0; digitized data copyright 2010-2011 Google Corporation |
| Roboto Medium | Apache-licensed font cut identified by the wheel provenance | Apache-2.0; Copyright 2011 Google Inc. |
| Twemoji atlas | `v15.1`, per wheel notices; graphics adapted into an atlas | CC-BY-4.0; Copyright 2020 Twitter, Inc and other contributors |

Portions of the ovui runtime are copyright (c) The FreeType Project
(www.freetype.org). All rights reserved.

The license collection supplements abbreviated wheel notices with complete
FreeType, GLAD, GLFW, GLM, md4c and stb license files from the revisions above,
the complete Noto OFL and Apache-2.0 texts from the wheel, and the complete
CC-BY-4.0 text from the Twemoji v15.1.0 release. Each section identifies its
source. Both GLM license alternatives are preserved; ovui identifies MIT as
its chosen option. The Noto font-specific OFL supplies the attribution for
the actual v2.015 files.

The wheel also lists external Python, CUDA, OpenGL and Vulkan prerequisites
and build tools. Host interpreters, drivers, SDKs and system libraries are
not included in this source release; use the license and notice files
provided with the installed versions.

## CUDA runtime 12.9.79

The [CUDA runtime license file](third_party_licenses/nvidia-cuda-runtime-cu12-12.9.79.txt)
is copied without modification from
`nvidia_cuda_runtime_cu12-12.9.79.dist-info/licenses/License.txt`. It is identical
in the audited Linux and Windows wheels and includes NVIDIA terms and
upstream third-party copyrights and licenses. That upstream file covers the
broader CUDA Toolkit; its inclusion does not mean this sample distributes
every toolkit component listed there. This sample installs only the pinned
runtime package and does not bundle its binaries.

## Audited release artifacts

The SHA-256 values below were verified against downloaded files and the
PyPI release metadata for [ovui 0.2.0](https://pypi.org/pypi/ovui/0.2.0/json)
and [CUDA runtime 12.9.79](https://pypi.org/pypi/nvidia-cuda-runtime-cu12/12.9.79/json).

| Wheel | SHA-256 |
| --- | --- |
| `ovui-0.2.0-cp312-cp312-manylinux_2_35_x86_64.whl` | `de8656af7c17b1d2ae68726e8b918796da299d3e372a417df3baf1ac6863f07b` |
| `ovui-0.2.0-cp312-cp312-win_amd64.whl` | `df371fb416adfce6f3e83a36938b72dfec0d92d2a308c632db04b22aba6269f7` |
| `nvidia_cuda_runtime_cu12-12.9.79-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl` | `25bba2dfb01d48a9b59ca474a1ac43c6ebf7011f1b0b8cc44f54eb6ac48a96c3` |
| `nvidia_cuda_runtime_cu12-12.9.79-py3-none-win_amd64.whl` | `8e018af8fa02363876860388bd10ccb89eb9ab8fb0aa749aaf58430a9f7c4891` |

When changing dependencies, bundled material, target platforms, or release
packaging, review the actual resulting artifacts and update this inventory
and its license texts as part of the [IP review procedure](docs/reference/IP_REVIEW.md).
