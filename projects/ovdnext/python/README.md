# OvdNext OvUI frontend

The existing Python OvUI workspace and ModernGL viewport use the Qt-free
`_pvd_native` PVDCore/PvdDom runtime. PvdDom is the authoritative capture store;
independent regression references are frozen, hash-checked test data.

Build the native module using the repository README. Use Python 3.12 and the
pinned runtime requirements already provisioned for your platform. OvUI 0.2.0
still requires CUDA runtime 12.9.79 even when rendering through OpenGL; the
Qt-free application does not imply a CUDA-free OvUI wheel.

Windows source launch (the launcher verifies the native module and CUDA DLL):

```powershell
python -B ci/run_ovui_windows.py --native-root build-native/python/Release --check
python -B ci/run_ovui_windows.py --native-root build-native/python/Release -- capture.ovd
```

Linux source launch with an existing runtime environment:

```bash
OVD_PYTHON=/path/to/venv/bin/python OVD_NATIVE_ROOT=/path/to/build-native/python bash ci/run_ovui.sh capture.ovd
```

No Qt path is an application argument. Inspector is disabled by default. A
source checkout or wheel is not by itself a portable runtime package; packaging
and clean-machine verification are separate steps. Preserve existing preferences
when launching normally; automated tests use isolated APPDATA/XDG settings.

## Capture viewing tips

- **Contacts:** enable them in Display. OVD contacts are shown on Pre-sim frames when recorded. On a Post-sim frame, **Go to Pre-sim** pauses playback and selects the matching preceding frame, without leaving the current clip. **All** stepping is preserved; **Post-sim** stepping switches to **Pre-sim**. Nothing switches automatically. Missing recorded contacts, selection/visibility filters and scene selection can still leave no contacts to draw.
- **Planes:** the ground grid represents the captured plane; selecting it highlights the surface. `PxPlaneGeometry` has no recorded parameters: its local normal is +X, transformed by the shape's `localPose` and actor's `globalPose`.
- **Analysis:** Profile, Memory View, Error Stream and Thread Profiler apply to legacy PVD streams and are hidden for OVD captures or when no capture is open. OVD diagnostics use Capture messages; property plots remain available.
- **Opening files:** the picker shows Modified dates and offers newest/oldest sorting. **Open capture** remembers the sort choice when the picker closes, including Cancel, across application runs. Desktop drag-and-drop is not supported by the currently pinned standalone OvUI runtime; use **Open capture**.

## Qt-free OvUI packaging

For the current Windows/Linux ZIP contract, driver baseline, release assets and
remaining customer-shipment gates, see [source and runtime distribution](../SOURCE_DISTRIBUTION.md).

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
tests and five writer fixtures build without a toolkit SDK. Independent camera,
formatting, geometry and profiling results use frozen, hash-checked test data.
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
Alternatively, set `OVUI_PYTHON_INPUT_ROOT` to a directory on the Linux
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

Extract the entire new ZIP (`unzip` on Linux to retain executable modes), then run `OvdNext.cmd` on Windows or
`./OvdNext.sh` on Linux. `--check capture.ovd` (or a PXD2 file) exercises
bundled Python and native loading with development search paths removed.
Public OvUI/ModernGL still needs an appropriate OpenGL driver and its pinned
CUDA runtime; Linux also needs host libcuda. Bundled component licenses and
notices are included in the package.

The separate OvUI workflow retains private-source, same-repository, successful
Windows and Linux validation, protected-main, explicit
`OVDNEXT_RELEASE_PUBLISH_ENABLED`, and release-environment gates.
Diagnostic preview artifacts from failed checks never authorize publication.
