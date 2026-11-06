# Dependency and asset notices

The project license is in [LICENSE](LICENSE); its scope is explained in
[licensing and distribution](docs/LICENSING.md). Third-party components retain
their original terms and are not relicensed under Apache-2.0. NVIDIA trademark
rights are separate. Preserve the complete notices in each asset directory.

| Component | Source and terms | Included notices |
| --- | --- | --- |
| ABB IRB 4600-60/2.05 visual geometry | ROS-Industrial ABB package; Apache-2.0. Source dimensions and joint offsets retained. | [Attribution](data/models/irb4600/ATTRIBUTION.md), [license](data/models/irb4600/LICENSE.txt), [source manifest](data/models/irb4600/manifest.json). |
| Concrete, plywood and cardboard PBR surfaces | Rob Tuytel / Poly Haven; ambientCG / Lennart Demes; CC0-1.0. | [Attribution](data/surfaces/ATTRIBUTION.md), [license](data/surfaces/CC0-1.0.txt), [source manifest](data/surfaces/manifest.json). |
| ABB IRB 120 visual geometry | ROS-Industrial ABB package; Apache-2.0. Original author and adaptation details are preserved. | [Attribution](data/models/irb120/ATTRIBUTION.md), [license](data/models/irb120/LICENSE.txt), [source manifest](data/models/irb120/manifest.json). |
| Car Concept | Eric Chadwick / Darmstadt Graphics Group GmbH, via Khronos glTF Sample Assets; CC BY 4.0. | [Attribution and changes](data/models/car/ATTRIBUTION.md), [source license notice](data/models/car/LICENSE.md), [complete license](data/models/car/CC-BY-4.0.txt), [trademark notice](data/models/car/LicenseRef-LegalMark-Khronos.txt). |
| NVIDIA eye and wordmark | Official NVIDIA brand source; NVIDIA trademarks. | [Brand attribution](data/brand/ATTRIBUTION.md). |

The public SDK distributions are installed separately: ovrtx, ovphysx, ovstage,
ovui, ovstorage, ovstream, ovnewton, ovpackage, Newton, Warp, and OpenUSD retain
their distribution terms. This source package contains no SDK wheels or DLLs.
The app-owned Python bootstrap uses Astral uv (MIT/Apache-2.0) and CPython; keep
their installed notices. Bootstrap and dependency hashes are in config/.

## NVIDIA SimReady exhibit props

Selected assets from [NVIDIA SimReady Assets](https://huggingface.co/datasets/nvidia/simready-assets), licensed CC BY 4.0. Source revision, per-file hashes, derivative recipe and retained notices are described in [SimReady assets](docs/SIMREADY_ASSETS.md). Setup installs a checksum-pinned archive of prepared USD models and textures, retaining the original CC BY 4.0 license and attribution. No mesh-processing tool is installed or executed on customer machines. PyMeshLab 2025.7 was used privately to author some of these derivatives; its package, installer and processing scripts are excluded from this source distribution. Historical authoring use remains disclosed in the release-review evidence.

## Navigation cursor icons

The hand and hand-grab paths are from Lucide, copyright (c) 2026 Lucide Icons and Contributors, licensed ISC. The source vectors, pinned provenance, full license, and details of the outline/color adaptation are included in [the cursor attribution](data/cursors/ATTRIBUTION.md).

The sidebar info icon also comes from Lucide, with ISC and MIT notices for its
Feather origin. The original SVG, PNG, pinned provenance and full licenses are
included in [the icon attribution](data/icons/ATTRIBUTION.md).

## Pinned SDK and native component notices

Licenses must be checked against the installed versions, not a newer repository
branch. In particular, the pinned OVUI 0.2.0 wheel identifies itself as NVIDIA
Proprietary even though the newer OVUI source release is Apache-licensed.
The pinned OVRTX, OVStage, OVPhysX and OVStream distributions have NVIDIA-specific
terms, and usd-core reports TOST-1.0. Their complete component notices remain in
the downloaded distributions. Preserve those notices if making an offline bundle.
OVStream's distribution also includes codec components even though this sample
uses local shared-memory frames rather than its encoded network transports.

Campus plaque lettering is generated from Noto Sans SemiBold, licensed under
the SIL Open Font License 1.1. The unmodified source font, complete license,
pinned provenance and outline-generation attribution are included in
[the wayfinding attribution](data/wayfinding/ATTRIBUTION.md). No Windows font
file or Segoe-derived vector outlines are included. Older fixed-text PNG signs
were rendered with Windows Segoe UI; those images are retained as artwork,
not redistributed font software.

## Direct application, setup and included authoring dependencies

These are the pinned dependency versions. They do not relicense downloaded SDKs.
Complete original license/notice texts remain with their distributions; exact
Python hashes are in `config/requirements*.lock`. The table includes
directly pinned worker dependencies and developer tools used by included scripts.
Resolved Python dependencies are also recorded in those locks.

| Dependency | Current version/ref | Reported license / terms | Integration |
| --- | --- | --- | --- |
| ovrtx | 0.5.1.385782 | NVIDIA proprietary distribution terms | Renderer and sensors |
| ovphysx | 0.6.4.72118723 | NVIDIA Omniverse terms | Native physics |
| ovstage | 0.2.1.385922 | NVIDIA proprietary distribution terms | Native stage operations |
| ovui | 0.2.0 | NVIDIA proprietary; Apache starts at 0.3 | Native UI; no automatic relicense of current wheel |
| ovstorage | 0.3.0 | Apache-2.0 | Storage API |
| ovstream | 0.5.6 | NVIDIA proprietary and bundled notices | Shared-memory frame API |
| ovnewton | 0.2.0 | Apache-2.0 | Physics worker |
| newton | 1.6.1 | Apache-2.0 | Simulation through OVNewton |
| ovpackage[local] | 0.2.1 | Apache-2.0 | Packaging worker |
| usd-core | 26.8 | LicenseRef-TOST-1.0 | OpenUSD APIs |
| warp-lang | 1.17.0 | Apache-2.0 plus bundled notices | Native GPU/CPU computation |
| numpy | 2.2.6 main; 2.5.3 Newton | BSD-3-Clause plus bundled component terms | Numeric data |
| Pillow | 13.0.0 | MIT-CMU plus bundled notices | Image processing |
| scipy | 1.18.1 | BSD-3-Clause plus bundled component terms | Newton environment |
| coacd | 1.0.14 | MIT | Collision preparation |
| fast-simplification | 0.2.0 | MIT | Geometry preparation |
| newton-usd-schemas | 0.5.0 | Apache-2.0 | Newton USD schema support |
| physx-usd-schemas | 25.11.1 | NVIDIA Software License Agreement and Omniverse Product Specific Terms (see below) | Physics USD schema support |
| Astral uv | 0.12.18 | MIT OR Apache-2.0, bundled notices | Downloaded bootstrap executable |
| CPython standalone | 3.12.14; build 20260901 | PSF terms plus bundled notices | Downloaded interpreter; archive and installed-file hashes verified (see below) |
| sharp | 0.35.5 in local authoring environment | Apache-2.0 plus libvips/native notices | Optional included icon-authoring scripts; not app runtime |
| Node.js | Developer-selected; no project pin | MIT plus bundled notices | Interpreter for optional icon-authoring scripts; not app runtime |


### Physics USD schema terms

`physx-usd-schemas 25.11.1` is governed by the
[NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and [Product Specific Terms for Omniverse](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-omniverse).
The Omniverse terms URL redirects to the consolidated
[Product Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/).
The pinned wheel declares `LicenseRef-NVIDIA-Omniverse` but does not contain
the full agreement text; use these agreement references together with the
wheel's original metadata.

### Python bootstrap distribution

The Windows x64 bootstrap uses CPython 3.12.14, python-build-standalone build
20260901, selected by uv 0.12.18. The original
[interpreter archive](https://github.com/astral-sh/python-build-standalone/releases/download/20260901/cpython-3.12.14%2B20260901-x86_64-pc-windows-msvc-install_only_stripped.tar.gz)
has SHA-256 `7c45c9622400d578709a9b2cddbe8124cc21d382409d9f13406d706d28e31b14`.
The archive checksum matches the pinned uv catalog, and its interpreter and
license files match the installed runtime. Preserve `LICENSE.txt` and bundled
component notices with the downloaded distribution.

Optional Node.js/sharp icon authoring is not part of normal app installation.
Keep bundled-component notices with any redistributed binary or offline bundle.
Third-party asset notices and nested license files are listed above; these SDK
and tool terms are separate from the project's Apache-2.0 software license.


## Separately delivered campus and guide

The campus and robot guide are NVIDIA-supplied artwork archives, identified
by archive and manifest hashes in `config/scene-release.json`. The campus
`manifest.json` and guide `asset-manifest.json` inventory their contents.
The archives are delivered separately from this source package; original
ownership, asset licenses, attribution and trademark notices remain applicable.
The sample's Apache-2.0 software license does not relicense this artwork.

The campus contains Poly Haven Oak Veneer 01 (Jenelle van Heerden), Forest Floor
(eye-candy.xyz) and Rocky Terrain 03 (Amal Kumar) textures under CC0-1.0. Their
[attribution, original source records and adaptation details](data/campus-provenance/polyhaven/ATTRIBUTION.md)
are also included inside the campus archive under `provenance/polyhaven/`, with
the full CC0-1.0 license. These terms apply to the named textures.

See [scene asset setup](docs/SCENE_ASSETS.md) for archive delivery and installation.
