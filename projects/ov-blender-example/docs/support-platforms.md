# Support Platforms

`ov-blender-example` intends to align its release platform targets with OVRTX
platform targets. This document separates OVRTX-declared platform facts from
this repository's user release support status.

## Provenance

Inspected OVRTX source:

- Project: [`NVIDIA-Omniverse/ovrtx`](https://github.com/NVIDIA-Omniverse/ovrtx)
- Commit:
  [`29d11037fbcaed0f0f53e7f32d17bd0486fd453b`](https://github.com/NVIDIA-Omniverse/ovrtx/tree/29d11037fbcaed0f0f53e7f32d17bd0486fd453b)
- OVRTX version:
  [`0.3.0`](https://github.com/NVIDIA-Omniverse/ovrtx/blob/29d11037fbcaed0f0f53e7f32d17bd0486fd453b/VERSION.md)
- Inspected files:
  - [`README.md` release platform list](https://github.com/NVIDIA-Omniverse/ovrtx/blob/29d11037fbcaed0f0f53e7f32d17bd0486fd453b/README.md#releases)
  - [`VERSION.md`](https://github.com/NVIDIA-Omniverse/ovrtx/blob/29d11037fbcaed0f0f53e7f32d17bd0486fd453b/VERSION.md)
- System requirements source linked by the OVRTX README:
  [Omniverse Technical Requirements](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html)

## OVRTX Release Platform Targets

OVRTX publishes binary releases for these OS and CPU architecture pairs. These
define the intended `ov-blender-example` release platform set.

| OVRTX platform | `ov-blender-example` platform |
| --- | --- |
| Linux x86_64 | `linux-x64` |
| Windows x86_64 | `windows-x64` |
| Linux aarch64 | `linux-aarch64` |

Windows ARM64 is not an intended release platform because OVRTX does not
publish a Windows ARM64 binary release.

## Omniverse Runtime Requirements

The runtime requirements linked by OVRTX constrain hosts for the release
platforms above. They do not define additional release platforms.

| OS | Minimum version |
| --- | --- |
| Linux | Ubuntu 22.04 |
| Windows | Windows 11 |

The linked requirements list a GeForce RTX 3070 as the minimum GPU for
Omniverse Kit. The following table records validated driver minimums by GPU
generation and type; it is not a list of minimum GPU models.

| OS | GPU generation | GPU type | Minimum validated driver |
| --- | --- | --- | --- |
| Linux | Blackwell | GeForce or Workstation | `570.169` |
| Linux | Blackwell | Data Center | `580.95.05` |
| Linux | Ada, Ampere, or Turing | GeForce or Workstation | `570.169` |
| Linux | Ada, Ampere, or Turing | Data Center | `570.158.01` |
| Windows | Blackwell | GeForce | `581.42` |
| Windows | Blackwell | Workstation | `573.42` |
| Windows | Blackwell | Data Center | `581.42` |
| Windows | Ada, Ampere, or Turing | GeForce | `581.42` |
| Windows | Ada, Ampere, or Turing | Workstation | `573.42` |
| Windows | Ada, Ampere, or Turing | Data Center | `573.39` |

## ov-blender-example Support Matrix

Rows marked Supported are this repository's current user support claim. Release
assets may be added or replaced while active platform release builds are in
progress.

| Support platform | OVRTX target | Support status | Evidence |
| --- | --- | --- | --- |
| `linux-x64` | Yes | Supported | Published `linux-x64` release |
| `windows-x64` | Yes | Supported | Published `windows-x64` release; current platform release build is active |
| `linux-aarch64` | Yes | Supported | Linux aarch64 platform release build is active |

## Maintaining This Matrix

Update this document when OVRTX platform provenance or this repo's release
status changes.

For each row, keep:

- OVRTX binary release platform evidence
- Omniverse runtime requirements separate from release platform targets
- support status
- release evidence, active release work, or the concrete blocker for
  unsupported rows
