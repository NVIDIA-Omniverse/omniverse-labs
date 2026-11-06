# Support platforms

This release has Windows x64 and Linux x64 qualification with
stock Blender 5.2 LTS (measured on 5.2.1). Measured hosts are an NVIDIA RTX A6000 with driver 597.16
on Windows and an NVIDIA L40 with driver 595.71.05 on Ubuntu 22.04 (glibc 2.35).
These are observed environments, not claimed minimum GPU/driver requirements.

| Platform | Qualification |
| --- | --- |
| Windows x64 | Representative installed renderer, viewport, production physics and recovery/cleanup checks; exhaustive benchmarking deferred. |
| Linux x64 | Native-host local package/install, LDR/HDR rendering, renderer recovery/cleanup, CPU/GPU physics and numeric physics-to-render synchronization. Exhaustive viewport, performance and hardware coverage remain unqualified. |
| Linux ARM, Windows ARM, macOS | Outside this Windows MVP; no qualification or release-support claim. |

The runtime tuple is pinned in `runtime/public-sdk-lock.json`: OVRTX
`0.6.0.1791947691`, OVStage `0.3.0.1791986420`, OVStream `0.5.7`, Warp `1.18.0`,
OVPhysX `0.7.0.74035321` and its PhysX USD schemas `25.11.2.72696984`, with
external CPython 3.13. The lock also fixes supporting Python
packages and native artifacts. This project consumes the locked SDK sidecar; it
does not rebuild an internal native bridge or infer compatibility from unrelated
public release versions.

Profile and addon/runtime archive paths may contain spaces and non-ASCII
characters. The Windows SDK loaders and caches need narrow-path (ASCII) folders, a
measured limitation, so a profile whose `ovrtx-sdk` path is not ASCII keeps the
native runtime, its caches and setup logs in `%ProgramData%\ovrtx-blender\<hash>`
(or `<system drive>\ovrtx-blender\<hash>`), named in `ovrtx-sdk\native-root.json`.
That folder follows a profile moved as a whole, also to an ASCII path; Remove
Runtime deletes it, and a failed install removes a folder it created. Installation
is refused only when no such folder can be created, or for an explicit external
interpreter in a non-ASCII folder.

Install with the release package installer described in the
[README](../README.md#install). A portable contract test or Windows-produced
Linux layout does not establish a Linux runtime pass.
