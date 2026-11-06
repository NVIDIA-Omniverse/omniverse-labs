# Licensing and distribution scope

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

The sample uses the Omniverse Labs policy in [LICENSE](../LICENSE):
**Apache-2.0 for NVIDIA-authored software**, and **CC-BY-4.0 for documentation
and other NVIDIA-authored non-software content**. A file's own license and
third-party notices take precedence for that component.

The software headers use NVIDIA's SPDX copyright statement and
`SPDX-License-Identifier: Apache-2.0`. Existing third-party files are not given
NVIDIA authorship or relicensed. Preserve their original notices and descriptions
of adaptations. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## What the source license covers

The application, library integration code, examples, setup scripts and validation
tools authored for this project are software. The root policy also distinguishes
their documentation from software. Downloaded SDKs, native binaries, datasets,
third-party models, textures, icons and fonts retain their own licenses.
NVIDIA names, the eye sculpture and wordmark retain their trademark rights;
the source license grants no permission to use them as another product's branding.

The source distribution contains application source and selected attributed art.
The installer obtains the pinned runtime and assets separately. The source ZIP
does not bundle SDK wheels, DLLs, the separately supplied campus, or the avatar.
An installer download is still a dependency and must be disclosed in release
review. A future offline bundle needs its own complete inventory and notices.

## Dependencies requiring separate terms

The pinned OVRTX, OVStage, OVPhysX, OVStream and OVUI 0.2.0 distributions carry
NVIDIA-specific terms. OVUI's newer Apache-licensed source release does not, by
itself, change the license of the pinned 0.2.0 wheel. OpenUSD's pinned distribution
reports TOST-1.0 terms. Do not describe all installed dependencies as Apache-2.0.

SimReady exhibit models are installed from a verified archive of prepared assets,
with their original CC-BY-4.0 license and attribution. Customer setup and repair
do not install or execute PyMeshLab. The package and private mesh-authoring
scripts are excluded from the public sample. Historical use of PyMeshLab
2025.7 to prepare derivatives is disclosed separately in release-review evidence.

## External-material notice

NOTICE AND DISCLAIMER: This software automatically retrieves, accesses or
interacts with external materials. Those retrieved materials are not distributed
with this software and are governed solely by separate terms, conditions and
licenses. You are solely responsible for finding, reviewing and complying with
all applicable terms, conditions, and licenses, and for verifying the security,
integrity and suitability of any retrieved materials for your specific use case.
This software is provided "AS IS", without warranty of any kind. The author makes
no representations or warranties regarding any retrieved materials, and assumes
no liability for any losses, damages, liabilities or legal consequences from your
use or inability to use this software or any retrieved materials. Use this software
and the retrieved materials at your own risk.

## Contributions

This project is currently not accepting contributions. See
[CONTRIBUTING.md](../CONTRIBUTING.md). This policy does not change the permissions
granted by the source and asset licenses.
