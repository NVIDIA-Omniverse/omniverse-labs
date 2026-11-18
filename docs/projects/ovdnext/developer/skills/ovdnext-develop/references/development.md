<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# OVD Viewer source development — v0.2.21

## What is in the package and source

The runtime ZIP supplies the application, its Python/native modules and
dependencies. It is not a development checkout or the separate PhysX SDK
`ovd-inspect` command-line tool. The headless inspection skill works directly
with that runtime; compiling or changing OVD Viewer requires source.

The runtime packages were built from
`cd83af8e7c6211be71cc2a82541738e0b1f2018f`. Labs imports its approved
source-publication selection `8ae2da30d3d78d9039a94f5ddb84ee810e64775f`
under `projects/ovdnext/`. This contains the same application code,
with license corrections and restricted/unverified files excluded. Read
`SOURCE_DISTRIBUTION.md` for the resulting test-fixture limits. This is a pinned snapshot: make application changes in
the development repository, then import an explicitly selected reviewed
revision through the Labs process. The [published source](https://github.com/NVIDIA-Omniverse/omniverse-labs/tree/main/projects/ovdnext) is available to inspect and copy
into an editable development checkout. Maintainer integrations still follow
the release's reviewed-source import process.
Keep builds, dependencies, temporary
captures and Python bytecode outside the snapshot. Do not update its manifest
to conceal a local edit. Documentation in this companion is Labs-owned.

## Architecture map

| Location in application source | Responsibility |
| --- | --- |
| `src/PVDCore/` | Capture parsing, recorded state and the existing native engine |
| `src/PVDSession/Session.*` | Native session, file/live state, seeking and bounded inspection |
| `src/PVDSession/Presentation.*` | Coherent immutable views, inspector projections and availability |
| `src/PVDSession/*Bindings.cpp`, `bindings.cpp` | Python binding signatures and limits |
| `python/ovd_ui/ingestion.py` | File preparation, OVD snapshot identity, load jobs and resource lifetime |
| `python/ovd_ui/adapters.py` | Native data presented to the frontend |
| `python/ovd_ui/widgets/properties.py` | Property inspector and array presentation |
| `python/ovd_ui/workflow.py`, `recordings.py` | Capture/recording lifecycle |
| `python/ovd_ui/renderer.py` | ModernGL viewport |
| `python/ovd_ui/application.py` | OvUI desktop integration |
| `src/OvdData/`, `src/OvdDataOmniPvd/` | Separate data-model experiments and probe; not the application's authoritative PvdDom path |
| `ci/` and `.github/workflows/` | Dependency provisioning, builds, checks and packaging in the development repository |

The PhysX SDK supplies both the OmniPVD writer and reader classes in
`physx/pvdruntime/`. PhysX's `PxOmniPvd` integration and sampler use the writer
to record simulation state into an `.ovd` file or a live stream. OVD Viewer is
the viewer/inspector application that consumes that data using the SDK's
reader classes; it does not provide those writer classes or run the simulation.

In OVD Viewer, `src/PVDCore/OvdFileLoader.cpp` and `OvdLiveSource.cpp` call
`createOmniPvdReader()`, bind an input stream, then use `startReading()` and
`getNextCommand()` to decode it. The SDK's `physx/pvddom/` applies those commands
to the recorded object model, which OVD Viewer exposes through its native Session,
Presentation and UI. Saving a received TCP recording retains the producer's
bytes for later inspection. OVD support is the scope of this release; older
PVD/PVD3 code and documentation do not establish tested support.

## Prerequisites and access

| Input | Pinned value / requirement |
| --- | --- |
| Application | Commit above, v0.2.21 |
| Physics | `a12f79ce585ffb79e5ffa4848cb45efc3de2f632`; `physx/pvdruntime` and `physx/pvddom` |
| OvUI | `a81db103a4d99ff2d75368a444e167889536ff42`; runtime 0.2.0 |
| GoogleTest | `6910c9d9165801d8827d628cb72eb7ea9dd538c5` |
| Python | CPython 3.12.16, platform-specific prepared environment |
| Native Python binding | pybind11 3.1.0 |
| Runtime/build inputs | Expat 2.9.0 and source-built Pillow 13.0.0; pinned requirements and OvUI helper packages |
| Windows | Visual Studio 2022 x64 C++ tools; prepared dependencies |
| Linux | Ubuntu 22.04, C++ compiler, CMake/Ninja, Info-ZIP `unzip`; prepared dependencies |

Physics and some build inputs require NVIDIA repository/mirror access. An
Apache application license does not supply that access or relicense other
components. A clean external source build is **not** established by these
instructions without those inputs. If access is unavailable, use the
qualified runtime for the documented inspection workflow and report the
specific missing prerequisite. Do not substitute an arbitrary newer SDK,
runtime or PyPI package while claiming v0.2.21 reproduction.

The matching source's `python/README.md` and `.github/workflows/ovui.yml`
define full environment provisioning. `requirements-runtime.txt` alone is
not a build environment. Provisioning with `ci/setup_ovui.sh` and packaging
with `ci/package_ovui_portable.py` belong in a separate development checkout;
they create outputs and derive provenance from the enclosing Git repository.

## Build a prepared Labs snapshot

With the exact dependencies provisioned outside Labs, substitute real paths
for each placeholder below. These are native source verification commands,
not a release/package build or an installation recipe for missing inputs.
For application edits, use your editable development checkout as the CMake
`-S` path instead of `<labs>/projects/ovdnext`; keep `-B` outside that checkout.

```text
cmake -S <labs>/projects/ovdnext -B <external-build> -DPVD3_BUILD_PYTHON=ON -DPVD3_BUILD_TESTS=ON -DPVD3_REQUIRE_OVD_SUPPORT=ON -DPython_EXECUTABLE=<prepared-python> -Dpybind11_DIR=<prepared-pybind11-cmake> -DPVDRUNTIME_ROOT=<physics>/physx/pvdruntime -DPVDDOM_ROOT=<physics>/physx/pvddom -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=<pinned-googletest>
cmake --build <external-build> --config RelWithDebInfo --parallel 4
ctest --test-dir <external-build> -C RelWithDebInfo --no-tests=error --output-on-failure
```

On Windows also pass `-G "Visual Studio 17 2022" -A x64` at configure time.
On Linux pass `-DCMAKE_BUILD_TYPE=RelWithDebInfo`. Historical `PVD3_` option
names are build identifiers, not a claim of PVD3 release support.

For source launch with the prepared environment, run from a separate working
directory with `PYTHONDONTWRITEBYTECODE=1`. Windows uses
`<prepared-python> -B <source>/ci/run_ovui_windows.py --native-root <build>/python/RelWithDebInfo --check`,
then the same launcher with `-- <absolute-capture.ovd>` for the viewer.
Linux uses `bash <source>/ci/run_ovui.sh <absolute-capture.ovd>` with `OVD_PYTHON` and
`OVD_NATIVE_ROOT` pointing to the prepared interpreter and native output.
Viewer launch also needs the graphics/driver prerequisites in the package
README. Headless source tests and graphical qualification are separate.

## Choose verification for a change

| Change | Relevant evidence |
| --- | --- |
| Session/property handling | Native CTest plus `python/tests/test_native_session.py`, `test_native_records.py`, `test_native_presentation.py`, `test_property_arrays.py` and `test_array_repage.py` in the prepared environment |
| Capture loading/lifetime | Ingestion tests and real OVD fixtures; distinguish invalid input from cancellation and resource cleanup |
| Display/geometry | Native geometry/presentation tests plus visual inspection of the actual viewer |
| TCP selection | `docs/TCP_SELECTION.md`, capture-control tests and an explicitly provisioned compatible SDK sender; a skipped sender test is not interoperability evidence |
| Evidence JSON | `docs/EVIDENCE_JSON.md`, evidence tests and the documented field/identity semantics |
| Labs-owned docs/import metadata | Labs regression, snapshot/index verification and public-release hygiene |

Run Python/native suites through the source's prepared test environment and
documented CI entry points (`ci/test_ovui.py` for application tests). A bare
system `pytest` command is not equivalent. For example, with the native
extension and fixture executables already built:

```text
<prepared-python> -B <source>/ci/test_ovui.py --native-root <native-output> --fixture-root <fixture-executables> -- <source>/python/tests/test_property_arrays.py <source>/python/tests/test_array_repage.py
```

Replace placeholders with absolute paths and quote paths containing spaces
(PowerShell needs `&` before a quoted interpreter path). Missing fixture
executables are a prerequisite failure, not a passing test.
Report passed, failed, skipped and
unexecuted coverage separately. A source build or green PR check does not
replace package qualification or distribution approval.

Maintainer-only Labs import verification lives outside the public snapshot.
Use the source checks above for an editable development checkout; the
public release does not include the internal import-manifest tools.

Application changes follow the development repository's `AGENTS.md` and
changelog/version rules. This documentation companion does not change the
application version, its imported source, runtime dependencies or scan ZIPs.
When a future code change produces new packages, record their own provenance
and qualification rather than inheriting this candidate's results.
