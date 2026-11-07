# Platforms and runtime versions

Use **Blender 5.2.1 LTS** with the matching add-on and runtime package. Install
Blender separately. The installer accepts Blender 5.2 or newer, but newer
versions may not have been tested.

## Package platforms

| Platform | Availability |
| --- | --- |
| Windows x64 | Add-on and runtime package. |
| Linux x64 | Add-on and runtime package. Viewport and hardware testing is limited. |
| Windows ARM64 / Linux ARM64 | Unsupported. |
| macOS | Unsupported. |

## Graphics and operating system

You need a compatible **NVIDIA RTX GPU and driver**. Tested configurations include:

- Windows: RTX A6000, driver 597.16.
- Ubuntu 22.04: NVIDIA L40, driver 595.71.05, glibc 2.35.

These are tested examples, not minimum hardware or driver requirements.

Linux runtime packages target **glibc 2.35**. Rendering also needs a display
session with Vulkan access to the GPU; Xvfb alone is insufficient.

## Runtime versions

The 0.2 package uses the versions below. Upgrade by installing a complete matched
package. Do not replace individual SDKs or install them in Blender's Python.

| Component | Packaged version |
| --- | --- |
| Blender reference | 5.2.1 LTS, installed separately |
| External CPython | 3.13.13 |
| `ovrtx` | 0.6.0.1791947691 |
| ovstage | 0.3.0.1791986420 |
| ovstream | 0.5.7 |
| ovphysx | 0.7.0.74035321 |
| PhysX USD schemas | 25.11.2.72696984 |
| Warp | 1.18.0 |
| NumPy | 2.5.3 |
| packaging | 23.2 |

## Physics devices

Physics uses the **CPU by default**, independently of GPU rendering. Start with
this setting.

GPU physics supports joints attached to World with unrotated local joint frames.
Unsupported configurations report an error rather than switching to CPU. See the
[physics guide](physics-guide.md#add-a-joint-or-a-bone-body) for joint setup and limitations.

## Installation paths

Follow the [installation guide](install-guide.md) to install or remove the package.
Paths may contain spaces and non-ASCII characters. On Windows, the installer may
place the native runtime in an ASCII-only folder under
`%ProgramData%\ovrtx-blender\<hash>`. Its location is recorded in
`ovrtx-sdk\native-root.json`. Use **Remove Runtime** to remove it correctly.
