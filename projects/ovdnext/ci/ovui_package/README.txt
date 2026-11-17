PVD on OVUI — Linux runtime wheelhouse
====================================

Historical fixture for the retired wheelhouse format. These bundle scripts
remain for regression tests; current packages use ci/package_ovui_portable.py.
The instructions below describe that historical format.

Component licenses and notices accompany this bundle.

Run ./install.sh from a compatible Linux x86_64 host with CPython 3.12 and uv
already available, then ./run.sh /path/to/capture.ovd. Installation uses only
the hash-locked local wheelhouse. An optional install.sh argument selects a new
venv directory; launch its bin/pvd-ovui command directly. Do not move a created
venv: move/extract this bundle and install a fresh one instead.

This package contains the native extension and Python resources, not a Python
interpreter, system libraries, or GPU drivers. Qt is not required. Follow the source repository
README for host and desktop graphics prerequisites. native-ldd.txt/native-elf.txt
record this build's dynamic library and symbol requirements. This is a Linux
CPython wheel, not a manylinux or Windows compatibility claim.

manifest.json records source and dependency revisions, staged source hashes,
wheel hashes, and the license paths retained in every dependency wheel. The
installed ovd_ui/notices directory contains cumulative CHANGELOG.md and source
notices, including the pinned OVUI and Physics terms. Font licenses and source
hashes remain beside the packaged fonts. SHA256SUMS verifies the bundle files.
The canonical colors.json is packaged for first-use palette creation and reset.

Build/test tools, the repository, sample recordings, and Inspector are not
installed into the runtime venv. python -I probe.py --capture FILE validates
the installed assets, imports and native file reads without opening a window.
This probe does not establish visual or whole-application performance parity.
