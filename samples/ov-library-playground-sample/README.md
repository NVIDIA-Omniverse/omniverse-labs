---
name: "Omniverse Libraries Playground"
description: >
  Explore Omniverse Libraries, experiment with real features, and discover
  how they can help solve simulation, robotics, visualization, and interactive application use cases.
version: "0.14.0-preview"
type: example
complexity: advanced
omniverse_libraries:
  - ovrtx
  - ovphysx
  - ovstage
  - ovui
  - ovstorage
  - ovstream
  - ovnewton
  - ovpackage
omniverse_library_version_min: null
tested_platforms:
  - Windows
tags:
  - usd
  - interactive
  - robotics
  - manufacturing
assets_included: true
skills_included: false
author: "Oliver Seifert, NVIDIA"
maintainer: "oseifert@nvidia.com"
---

# Omniverse Libraries Playground

**Status:** Working

**Release:** 0.14 preview.

## Overview

**Explore the Omniverse Libraries, experiment with their features, and discover
how they can help solve your use cases.** Walk through a virtual campus and try
interactive examples of simulation, robotics, visualization, and application
development. The robot guide introduces the experiences, and each room provides
controls, explanations, and example code you can build upon.

Start with robot sensors in OVRTX, carton dynamics in OVPhysX, or a shared robot
fleet in OVStage. Use the [source map](docs/SOURCE_MAP.md) to find the code behind
each demonstration.

### Libraries used

The launcher installs these versions:

| Library | Version | What you can explore |
| --- | --- | --- |
| ovrtx | 0.5.1.385782 | Moving-robot LiDAR and front/rear fisheye cameras; car and robot-arm materials and lighting. |
| ovphysx | 0.6.4.72118723 | Carton drops, stacking, impulses, and measured friction. |
| ovstage | 0.2.1.385922 | Reusable robot instances and synchronized scene views. |
| ovui | 0.2.0 | Buttons, sliders, and shared value models controlling a pallet stacker, lighting, and inspection status. |
| ovstorage | 0.3.0 | Save, compare, and restore scene revisions. |
| ovstream | 0.5.6 | Transfer rendered frames and send commands through local shared memory. |
| ovnewton | 0.2.0 | Simulate an articulated gantry and compare suspended-load sway. |
| ovpackage | 0.2.1 | Publish and install robot assets, verify files, detect edits, and restore a revision. |

Supporting dependencies include Newton 1.6.1, Warp 1.17.0, and OpenUSD
(`usd-core`) 26.8. Complete dependency versions and checksums are in the
[main](config/requirements-win.lock), [Newton](config/requirements-newton-win.lock),
and [Package](config/requirements-package-win.lock) lock files.

## Prerequisites

- Windows x64 and an NVIDIA RTX GPU.
- 20 GB of free disk space for libraries, scene assets, and caches.
- Internet access to GitHub, PyPI, pypi.nvidia.com, and Hugging Face during setup.

The launcher installs Python and all dependencies. No preinstalled Python or
developer tools are required. Hardware validation currently covers Windows 11
with an RTX PRO 5000 Blackwell laptop GPU and 24 GB VRAM; other GPU configurations
have not been validated.

## Setup

1. Open `samples/ov-library-playground-sample/` in your cloned repository, or
   extract the sample to a writable folder.
2. Double-click `scripts/launch_ov_library_lab.bat`.
3. Wait while the launcher downloads the pinned libraries and scene assets,
   verifies them, and prepares RTX shaders. It displays progress and recovery
   options. Later launches reuse the installed files.

The campus, robot guide and prepared exhibits download automatically from
versioned [GitHub Release assets](https://github.com/NVIDIA-Omniverse/omniverse-labs/releases/tag/ov-library-playground-assets-v0.14.0-rc.2). No GitHub account is needed.
See [scene asset setup](docs/SCENE_ASSETS.md) for checksums and manual downloads.

Dependencies and saved data are stored in `.ov_library_lab_runtime/`. Setup
does not change your system Python or PATH. Later launches reuse the installed
files. Keep the runtime folder when updating to preserve saved revisions.

See [first visit and troubleshooting](docs/CUSTOMER_GUIDE.md) for help.

## Usage

| Control | Action |
| --- | --- |
| WASD | Walk |
| Drag or arrow keys | Look around |
| Shift | Walk faster |
| Home | Return to the atrium |
| Library names at the top | Jump to a room |
| F1 or settings cog | Adjust quality and text size; access diagnostics |

Click the scene to resume navigation after using a control. In each room, tabs
switch experiment pages, selectors choose options, and action buttons run the
experiment. Use the information button beside the library title for explanations
and example code. **Reset room** restarts an experiment; **Start over** returns
to the atrium while preserving saved files.

## Architecture

OVUI provides the application window and controls. The application authors a USD
scene, OVStage manages its rendered representation, and OVRTX produces the camera
and sensor outputs. OVPhysX simulates rigid bodies and updates their displayed
poses. OVStorage saves scene revisions, and OVStream transfers rendered frames
and return commands. OVNewton and OVPackage run in separate worker processes
with their own dependency environments.

| Folder | Contents |
| --- | --- |
| source/ | Application, scene builders, and library integrations |
| scripts/ | Launcher and runnable examples |
| data/ | Models, materials, and scene graphics |
| config/ | Dependency versions and asset manifests |
| tools/ | Setup, diagnostics, and validation tools |
| docs/ | Usage, development, and troubleshooting guides |

## Extending this sample

Start with the [runnable examples](scripts/examples/README.md) or use the
[source map](docs/SOURCE_MAP.md) to find a room's implementation. Under
`source/nvidia/ov_library_lab/`:

- `content.py` defines the library catalog and descriptions.
- `scene.py` and the room modules create the exhibits.
- `panels.py` connects the controls to library operations.
- `layout.py` defines navigation and exhibit placement.

Floor-plan changes require corresponding navigation and placement updates.
See the [development guide](docs/DEVELOPMENT.md) for architecture details and
validation commands.

## Assets

The campus and robot guide use separate scene archives. The sample includes
Car Concept and ABB robot models, CC0 surface textures, and application-created
graphics. Setup also installs prepared NVIDIA SimReady models and textures from a verified archive.

Asset sources and license terms are listed in [third-party notices](THIRD_PARTY_NOTICES.md).
Preserve the accompanying attribution when reusing assets.

## Known Limitations

- First launch can take several minutes to compile shaders and prepare sensors.
- Physics and sensor examples use simplified properties and illustrative calibration.
- Fleet routes and pallet-stacker movement are scripted.
- Storage uses a local file backend; streaming uses local shared memory.
- The robot guide uses predefined conversation choices.

Saved revisions, logs, and diagnostics stay in the runtime folder. The application
does not upload them automatically.

## License

NVIDIA-authored software is licensed under **Apache-2.0**. Documentation and
other NVIDIA-authored non-software content use **CC-BY-4.0**. See [LICENSE](LICENSE)
and [licensing details](docs/LICENSING.md). Downloaded SDKs and third-party assets
retain their own terms. NVIDIA trademarks are not covered by the source license.

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

Report vulnerabilities through [SECURITY.md](SECURITY.md).

## Contributions

This project is currently not accepting contributions.
See [CONTRIBUTING.md](CONTRIBUTING.md).
