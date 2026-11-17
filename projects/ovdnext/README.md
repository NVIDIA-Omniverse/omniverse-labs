# OVD Viewer

OVD Viewer is an OvUI/ModernGL application for inspecting OVD captures, backed by the
C++ PVDCore/PvdDom runtime and a CMake build system. Builds, tests and packages
require no Qt, Qwt or Boost SDK. Live TCP uses a small native Winsock/POSIX helper.

See [Python frontend setup and source launch](python/README.md) for Windows and
Linux. Scene/property inspection, playback, click highlighting, double-click
centering, Vis/Trsp controls and independently scaled gizmos share the existing
native capture presentation. Public OvUI 0.2.0 still needs CUDA runtime 12.9.79
even on its OpenGL path; application Qt removal does not remove that dependency.

The supported release scope is the bundled viewer and documented headless OVD
functions. PVD3 support is untested and outside this release scope. PhysX records
OVD using its OmniPVD writer classes; the viewer consumes recordings through
the OmniPVD reader classes.

The source-publication corrections retain v0.2.21 application behavior and
the qualified runtime archives. See [source distribution](SOURCE_DISTRIBUTION.md)
for source selection, dependency prerequisites and fixture limitations.
The complete historical release log remains in [CHANGELOG.md](CHANGELOG.md).

## Contributions

This project is currently not accepting contributions.

## Before downloading dependencies

NOTICE AND DISCLAIMER: This software automatically retrieves, accesses or interacts with external materials. Those retrieved materials are not distributed with this software and are governed solely by separate terms, conditions and licenses. You are solely responsible for finding, reviewing and complying with all applicable terms, conditions, and licenses, and for verifying the security, integrity and suitability of any retrieved materials for your specific use case. This software is provided "AS IS", without warranty of any kind. The author makes no representations or warranties regarding any retrieved materials, and assumes no liability for any losses, damages, liabilities or legal consequences from your use or inability to use this software or any retrieved materials. Use this software and the retrieved materials at your own risk.

## Rendering defaults

Face culling defaults to **None**. On upgrading from v0.2.9 or earlier, saved **Normal** culling resets to **None** because those versions also saved the implicit default. Saved **None** and **Inverse** are preserved. To restore Normal deliberately, choose **General settings > Rendering > Face culling: Normal** and Apply; choices saved by v0.2.10 onward are preserved. If OvdNext v0.2.9 or earlier later saves rendering settings to the same preferences file, **Normal** resets to **None** again on the next start of v0.2.10 or later; choose it again.

## Live capture workflow

Start OvdNext before the PhysX producer; it listens for live OVD streams on TCP port 5425 automatically. Run producers repeatedly without using **File > New** between runs. Each connection becomes a timestamped session clip in the **Recordings** tab. Clips stay in memory until removed or the viewer closes. A useful incoming capture is shown live; empty or rejected connections leave the previous useful capture available.

Activate an older clip to inspect and scrub it while new connections arrive in the background; choose **Follow latest** to resume automatic display. **Save as** exports a selected completed clip to `.ovd`. **Remove** releases its memory without deleting saved files. **Auto-save to disk** defaults OFF and saves each useful completed capture when enabled; explicit preferences are remembered. Empty or rejected connections remain memory-only and cannot be exported. Disk-save failures do not discard session clips or stop listening; unsaved clips remain available only while the viewer is open. The size column reports retained stream bytes; parsed scenes also consume RAM, so remove unneeded clips during long sessions.

An active connection appears as a **Live** row even with autosave off. The main Save command validates the destination before freezing and exporting the displayed live capture. Removing a saved memory clip leaves its file untouched; files in the recording library remain listed, where **Delete** deletes the file.

Freezing a stream mid-command can leave a viewable clip that cannot be exported, even if its row says **Complete** and Save is enabled. Let the producer finish cleanly when an export is required. Automatic disk saving remains synchronous and can briefly pause the UI for large captures; the next listener is armed before that copy begins.

**Accept live connections** starts ON every time; turn it OFF to let the current producer finish and block later reconnects. **File > New** clears the display while preserving session clips and saved files. **File > Attach to live...** selects a different port, **Detach live (freeze)** stops the active producer for inspection, and **Open folder** reveals automatically saved files. Unsaved clips are lost when the viewer closes. The **Clips** tab trims or splits frame ranges within the selected capture; overlaying separate captures is not supported.

See [source distribution](SOURCE_DISTRIBUTION.md) for the selected source, dependency prerequisites and excluded-fixture limits.

## Native runtime development

The recommended native build has no Qt dependency. It exposes the existing
PVDCore/PvdDom engine through `_pvd_native` for OvUI integration. PvdDom remains
the authoritative OVD capture store; the standalone OvdData experiments are
separate targets and are not linked into the native application.

With Python 3.12 and pybind11 available, configure from the repository root:

```bash
cmake -S . -B build-native -DPVD3_BUILD_PYTHON=ON -DPVD3_BUILD_TESTS=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_Qt5=ON \
  -DPython_EXECUTABLE=/path/to/python \
  -Dpybind11_DIR=/path/to/pybind11/share/cmake/pybind11 \
  -DPVDRUNTIME_ROOT=/path/to/physics/physx/pvdruntime \
  -DPVDDOM_ROOT=/path/to/physics/physx/pvddom
cmake --build build-native --config Release --target _pvd_native checked_output_tests
ctest --test-dir build-native -C Release -R CheckedOutputTest --output-on-failure
```

On Windows use Visual Studio 2019 x64 and Windows paths (PowerShell uses a
backtick for line continuation). The module is under `build-native/python`
on Linux, or `build-native/python/Release` for Visual Studio. Independent
reference results are preserved as hash-checked test data; no reference SDK
or executable build is required.
Native runtime integration does not by itself enable release publication.

Checked exports create an exclusive temporary file beside the destination and
publish only after successful writes, flush and close. No-overwrite publication
is atomic; failed exports preserve existing files. POSIX overwrite preserves
the destination mode, without claiming complete ACL or power-loss durability.

## Windows native verification

GitHub's `windows-2022` native verification uses MSVC `14.29.30133`, Windows SDK
`10.0.17763.0`, checksum-verified CMake `3.25.1` and hash-locked Ninja `1.10.2.4`.
Both root and standalone lanes use the same toolkit-free provisioning. The OvUI
application builds separately with its documented Python/runtime pins.

From a configured x64 developer shell, run `ci/build_native.py` with the same
source/build/Physics arguments as the Linux example below. Both checkout SHAs
are checked before configuring; `--source-commit` is mandatory. The tracked
`ci/run_windows_ci.bat verify` wrapper activates the CI-provisioned toolchain.

### Qt-free OvdData contract tests

```powershell
cmake -S src/OvdData -B build_ovd -G "Visual Studio 16 2019" -A x64 -DOVDDATA_BUILD_TESTS=ON
cmake --build build_ovd --config RelWithDebInfo --parallel
& .\build_ovd\RelWithDebInfo\ovddata_tests.exe
& .\build_ovd\RelWithDebInfo\ovddata_ingest_tests.exe
ctest --test-dir build_ovd -C RelWithDebInfo --no-tests=error --output-on-failure
```

This path does not discover or load Qt, Python, OvUI, OpenGL, PhysX, `pvdruntime`, or `pvddom`.

### Qt-free OvdNext real-capture probe

Configure the standalone OmniPVD adapter with the lightweight `PVDRuntime` source tree, then build and run the probe against a capture:

```powershell
cmake -S src/OvdDataOmniPvd -B build_ovd_file -G "Visual Studio 17 2022" -A x64 `
  -DPVDRUNTIME_ROOT=D:/path/to/physics/physx/pvdruntime `
  -DOVDDATAOMNIPVD_BUILD_TESTS=ON
cmake --build build_ovd_file --config RelWithDebInfo --parallel
& .\build_ovd_file\RelWithDebInfo\ovdnext_data_probe.exe .\test\data\test_scene.ovd
ctest --test-dir build_ovd_file -C RelWithDebInfo --no-tests=error --output-on-failure
```

`OvdData` is Qt-free and presentation-neutral. `OvdDataOmniPvd` adds only the lightweight `PVDRuntime` reader and validated-input implementation; it does not link `PVDCore`, Qt, OvUI, OvRTX, or a renderer. The probe loads all property payloads so it exercises the complete real-capture path. Other consumers can use a selective `LoadPlan` to reduce retained OvdData memory, but the current `PVDRuntime` reader still reads every command payload before the adapter can discard excluded bytes. Source-I/O skipping through seeking/indexing and lazy blobs remains future work.

### Test the OvdNext data path

Run this checkpoint from an x64 Developer PowerShell with Ninja available on `PATH`:

```powershell
cmake -S src/OvdDataOmniPvd -B build-ovddata -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPVDRUNTIME_ROOT=D:/dev/git_21/physics/physx/pvdruntime
cmake --build build-ovddata --target ovddata_tests ovddata_ingest_tests ovddata_omnipvd_tests ovdnext_data_probe ovdnext_data_probe_tests
ctest --test-dir build-ovddata -C RelWithDebInfo --no-tests=error --output-on-failure
.\build-ovddata\ovdnext_data_probe.exe .\test\data\test_scene.ovd
```

This presentation-neutral data path coexists with the OvUI application; developers can build and run both from the same checkout.

### Current data-path limits

This is a correctness-first data foundation, not yet a large-capture performance implementation. Materializing a view replays retained history, and the probe requests every frame in sequence; that can make a complete probe run quadratic in capture length. Unique-list reconstruction uses linear value searches, and each completed frame copies the live-object map. Indexed histories, incremental playback, and source-level payload skipping remain future work.

Only one sequential START/STOP frame scope is supported at a time. Simultaneous scene frame scopes are rejected; independently resetting scene frame numbers within a segment is not generally supported. On a projected view, `isAlive()` describes the selected subset, not a separate global-liveness query. Deferred commands after the final completed frame do not create a publication, and aggregate ingest counters can include such commands. Recorded object descriptors are retained only for creations in completed frames, even if those objects were subsequently destroyed.

The tracked fixture compatibility matrix is enforced in `OvdFileIngestTests.cpp`: seven fixtures load, including the intentionally zero-frame message-only recording; five reject unresolved-object attribute writes with named `eSET_ATTRIBUTE` / `UnknownObject` diagnostics. These rejections are explicit compatibility limits, not silently skipped tests or a claim of full legacy-loader parity.

## Linux native verification

Ubuntu 22.04 with glibc 2.35 is the oldest supported Linux package baseline.
Native verification needs a C++ compiler, CMake, Ninja, Python 3, Git and binutils,
not a desktop toolkit. With the Physics checkout at the pinned revision:

```bash
python3 ci/build_native.py \
  --source-root "$PWD" \
  --build-root "$PWD/_ci/build/linux-x86_64" \
  --physics-root /path/to/physics \
  --source-commit "$(git rev-parse HEAD)" \
  --physics-ref a12f79ce585ffb79e5ffa4848cb45efc3de2f632
```

This builds the native test targets, captures per-case CTest/JUnit results and
checks the fail-closed inventory. For GUI testing and portable packaging use
the OvUI workflow and [Python setup](python/README.md), including Xvfb, xauth,
xclip, Mesa/GL/EGL, xkbcommon, CUDA loader prerequisites and Info-ZIP `unzip`.

## GitHub verification

The GitHub workflows separate native core verification, the standalone data path and the OvUI runtime. Protected `main` requires these seven checks:

- `verify-root-windows`
- `verify-root-linux`
- `verify-ovdnext-data-windows`
- `verify-ovdnext-data-linux`
- `build (windows-2022, windows)`
- `build (ubuntu-22.04, linux)`
- `verify-release-version`

Trusted same-repository automation must configure access to the private Physics source. The approved existing read/write personal access token may be stored as the encrypted repository secret `OVDNEXT_CI_PHYSICS_TOKEN`; it retains its real read/write permissions because workflow `permissions` narrow only `github.token`, not a PAT. An optional, preferred alternative is a GitHub App with read-only access to `NVIDIA-Omniverse/physx-internal`, configured through the repository secrets `OVDNEXT_CI_APP_ID` and `OVDNEXT_CI_APP_PRIVATE_KEY`; the App is not mandatory. Each Physics checkout prefers the App token when configured, otherwise uses `OVDNEXT_CI_PHYSICS_TOKEN`, and falls back to `github.token` only for a future anonymously readable public Physics source. Repository secrets are not supplied to fork pull requests, and the workflow must not use `pull_request_target`.

The read/write CI token is an explicitly accepted security tradeoff: a repository writer who can change a workflow can access repository secrets and exercise the token's real permissions. Same-repository event guards, `persist-credentials: false`, and workflow `permissions` do not remove that exposure. A separate Physics-only read credential remains the recommendation; the current owner has chosen to keep the existing credential.

`OVDNEXT_CI_APP_ID` stores the GitHub App client ID for the native workflow's `client-id` input; the OvUI workflow's pinned token action accepts it through `app-id`.

Branch protection requires up-to-date PRs and enforces the checks for administrators too. Every PR advances the changelog and matching application version. Automatic private publication additionally requires `OVDNEXT_RELEASE_PUBLISH_ENABLED=true` and a protected `release` environment restricted to `main`, without a manual per-release approval step. Customer-distribution approval is separate. Repository settings must be configured and verified; naming an environment in YAML alone does not protect it.

Both root platforms use [the native driver](ci/build_native.py); Windows also has a [toolchain wrapper](ci/run_windows_ci.bat). Standalone data-path drivers remain [Windows](ci/build_ovdnext_data_windows.ps1) and [Linux](ci/build_ovdnext_data_linux.sh). Each root platform discovers 288 tests: 283 enabled and five existing disabled diagnostics. The [current inventory and policy](ci/baselines/qt-free/ovdnext-ci-parity.json) permit no platform exclusions; [source distribution](SOURCE_DISTRIBUTION.md) explains the public selection and tests that need separately authorized fixtures. These counts describe the complete internal v0.2.21 test inputs, not a test pass on the filtered public source.

Both root platforms allow at most these same three explicitly named runtime skips: `OvdInputValidationTest.FileLoaderAcceptsLargestAvailableCorpusCapture`, `OvdMultiSceneTest.FactoryChildrenHaveNoOwningScene`, and `OvdMultiSceneDeformableTest.DeformablesPartitionAcrossScenes`. In contrast, standalone verification requires all four data tests: `ovddata_tests`, `ovddata_ingest_tests`, `ovddata_omnipvd_tests`, and `ovdnext_data_probe_tests`.

The OvUI workflow builds and tests preview ZIPs on same-repository PRs. This change adds `verify-package (windows-2022, windows)` and `verify-package (ubuntu-22.04, linux)`: separate fresh-runner jobs that download the exact build ZIPs without rebuilding, verify archive/payload integrity and provenance, and exercise the relocated bundled runtime. They report PR status; the seven required branch-protection checks above remain unchanged.

After this workflow change is merged, a merged PR triggers fresh Windows/Linux builds of its `main` commit and publishes that version only after both builds and both package checks pass. PRs and direct pushes without an associated merged PR never publish releases. Existing published assets are never overwritten.

Release descriptions contain only the newest semantic-version section. Both
portable packages retain the cumulative changelog and exact source SHA. Download
from [GitHub Releases](https://github.com/NVIDIA-dev/omniverse-pvd3_ovd/releases)
or the [latest release](https://github.com/NVIDIA-dev/omniverse-pvd3_ovd/releases/latest).
Native verification artifacts retain build metadata and test diagnostics for
seven days; they are not application packages.

## Project structure

```text
CMakeLists.txt              Root build
src/OvdData/                Presentation-neutral recorded model and frame views
src/OvdDataOmniPvd/         Validated capture adapter and command-line probe
src/PVDCore/                Data model and binary protocol parsing
src/PVDSession/             Native Python session binding
python/ovd_ui/              OvUI workspace and ModernGL viewport
src/PxShared/               PhysX foundation and PVD SDK
test/                       GoogleTest suite and test captures
ci/                         Dependency, build, package, release, and contract scripts
cmake/                      CMake modules
```

## Perforce origins

Sources were internalized from the original PVD3, PxShared, zlib/minizip, Qwt, Boost, and PVD test-data depots.

## Qt-free OvUI packaging

Use uv 0.13.7 and pinned CPython 3.12.16. Prepare the development venv from
`python/requirements.txt`, plus `glcontext==3.0.0`,
`nvidia-cuda-runtime-cu12==12.9.79`, `setuptools==80.9.0`, `wheel==0.45.1`
and `cmake==3.25.2`; Linux also needs `ninja==1.10.2.4` and a C++ compiler,
while Windows needs Visual Studio 2022 with the x64 C++ tools. Install the three
OvUI helper packages from the pinned source checkout as in
`.github/workflows/ovui.yml`. `python/requirements-runtime.txt` records the
runtime closure; it does not provision the complete build environment.
The Linux source quickstart is
`bash ci/setup_ovui.sh`; it builds the headless native module and performs an
import smoke only. Full verification uses `ci/build_ovui.py` with `--work-root`,
`--physics-root` and `--ovui-root`, followed by `ci/test_ovui.py`. Native regression
tests and five writer fixtures build without a toolkit SDK. Camera, formatting,
geometry and profiling comparisons use frozen, hash-checked reference corpora.
On Linux, install Info-ZIP `unzip` (`sudo apt-get install unzip`) before running
`ci/build_ovui.py`; its extracted-package probe uses it to preserve executable modes.

Portable builds require the hash-pinned interpreter from
`ci/install_ovui_python.py --work-root NEW_PYTHON_INPUT`. GitHub CLI must be
authenticated for the private build-input mirror. Create the application venv
using the interpreter path printed by that command, then install the build and
application dependencies above. On Windows, run the prepared venv's
`python -B ci/install_ovui_expat.py --work-root NEW_EXPAT_BUILD` to rebuild both
XML modules against Expat 2.9.0. Linux already includes that parser. Existing
venvs based on the older runtime must be replaced before portable packaging;
the packager checks the input provenance and measured XML parser version.
On WSL, place `NEW_PYTHON_INPUT` on the Linux filesystem; the runtime includes
case-sensitive terminfo links that cannot be extracted onto a Windows-mounted
filesystem. Prepare the venv as above before running the source quickstart there.
Alternatively, set `OVUI_PYTHON_INPUT_ROOT` to a new directory on the Linux
filesystem before running `bash ci/setup_ovui.sh`.

Before running `ci/build_ovui.py` or `ci/package_ovui_portable.py`, replace the
stock Pillow wheel with the source build below. Use that venv's `python` and
put its `bin/` (Linux) or `Scripts/` (Windows) directory first on `PATH`.
Choose a new work directory; the optional source cache reuses hash-verified
archives. Network access is needed when an archive is absent from the cache.

```text
python -B ci/install_ovui_pillow.py --work-root NEW_PILLOW_BUILD --source-cache PILLOW_SOURCE_CACHE
```

Packaging rejects the stock `pillow==13.0.0` wheel. `ci/setup_ovui.sh` already
performs this preparation on Linux using `_ci/pillow-sources` as the cache.
The historical `ci/package_ovui_linux.py` wheelhouse route is retired; use the
portable packager below.

Create a portable archive from a fresh native build using an existing runtime:

```text
python -B ci/package_ovui_portable.py --native /absolute/path/to/_pvd_native<ABI-suffix> --physics-root /pinned/physics --ovui-root /pinned/ovui --output /new/package-directory
```

On Windows also pass `--msvc-runtime` with an existing complete x64
`Microsoft.VC*.CRT` redistributable directory compatible with the bundled
interpreter. Packaging requires no Qt argument. It checks native dependencies
with dumpbin or readelf/ldd, rejects Qt/Qwt/PySide/PyQt runtime identities, and
retains license notices, font hashes, runtime provenance and native inventory.

Extract the entire new archive, then run `OvdNext.cmd` on Windows or
`./OvdNext.sh` on Linux. `--check capture.ovd` (or a PXD2 file) exercises
bundled Python and native loading with development search paths removed.
Public OvUI/ModernGL still needs an appropriate OpenGL driver and its pinned
CUDA runtime; Linux also needs host libcuda. Bundled component licenses and
notices are included in the package.

The separate OvUI workflow retains private-source, same-repository, successful
Windows and Linux validation, protected-main, explicit
`OVDNEXT_RELEASE_PUBLISH_ENABLED`, and release-environment gates.
Diagnostic preview artifacts from failed checks never authorize publication.

## License

NVIDIA-owned OvdNext software is licensed under [Apache-2.0](LICENSE). Third-party components retain their own licenses; see [NOTICE](NOTICE) and the packaged component notices.
