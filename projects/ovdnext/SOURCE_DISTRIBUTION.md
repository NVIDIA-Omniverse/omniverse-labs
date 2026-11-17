# OVD Viewer source distribution

This source-publication candidate is based on v0.2.21 application commit
`cd83af8e7c6211be71cc2a82541738e0b1f2018f`. Its changes are licensing headers,
notices and source-release documentation. Runtime behavior, dependency pins
and the qualified Windows/Linux ZIPs remain unchanged. Source-review approval
and permission to distribute the packages are separate release decisions.

## Included source and terms

The selection retains the native PVDCore/PVDSession, OvdData and OmniPVD-reader
integration, supported PxShared platform code, Python viewer and headless
functions, build scripts, test source, user documentation and UI assets.
NVIDIA-authored code carries Apache-2.0 SPDX headers. Vendored PxShared keeps
its original terms; Geist and DM Mono fonts keep their OFL terms. See
[LICENSE](LICENSE), [NOTICE](NOTICE) and
[third-party notices](licenses/THIRD_PARTY_NOTICES.txt).

The exact included/excluded paths, Git blob IDs, modes, byte counts and
SHA-256 values are recorded in the accompanying source-review inventory.
Publication must use that reviewed selection rather than the full internal
repository. Source publication remains subject to the recorded approvals.

## Excluded material and test limits

The public selection excludes restricted console-platform files, internal
automation/security coordination, internal planning/release notes, compiled
legacy vendor documentation, all 15 recorded OVD/PXD2 captures under
`test/data/`, and historical calibration/reference fixture files. Their
redistribution rights have not been verified. No recorded capture is claimed
redistributable by this candidate. The internal repository retains them.

Tests that load those fixtures require separately authorized recordings and
are not claimed to pass against the filtered source tree. Build with
`PVD3_BUILD_TESTS=OFF` for the viewer if those test inputs are unavailable.
Tests that generate their own data retain their source; no test result is
inferred merely from including it in the export.

## Build prerequisites

The source checkout alone is not a standalone build environment. Native OVD
support requires PvdDom and PVDRuntime from the approved Physics input;
the frontend requires the approved OvUI and Python runtime dependencies.
Some pinned inputs require separately authorized access. This source
publication does not grant access to them or promise an anonymous build.
The qualified prebuilt ZIPs include the supported runtime and do not require
a compiler or source build. Read the external-material notice in README.md
before running build/test commands that download dependencies.

## Qualified runtime delivery

The source compliance changes do not produce a new runtime version or ZIP.
The mandatory matching third-party licensing/source companion must be
published beside both Windows and Linux archives. Do not replace the
binary/source provenance of those archives with this source-review commit.
