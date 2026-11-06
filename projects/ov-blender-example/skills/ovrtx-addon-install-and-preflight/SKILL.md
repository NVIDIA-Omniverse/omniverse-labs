---
name: ovrtx-addon-install-and-preflight
description: Install, enable and preflight the ovrtx Blender Example release package (add-on plus locked external SDK runtime) in stock Blender 5.2 or newer. Use for installation, upgrades, a missing render engine or runtime integrity diagnostics.
license: "Apache-2.0"
metadata:
  author: "Max Bickley"
  version: "0.1"
  team: "omniverse"
  domain: "physical-ai"
  tags:
    - blender
    - omniverse
    - ovrtx
    - lighting
    - addon-development
---
# OVRTX addon installation and preflight

Install from the release package ZIP for the platform (`windows-x64` or
`linux-x64`). It pairs the add-on, the locked external SDK runtime, their
distribution manifest and the installer. Stock Blender 5.2 or newer loads the
Python add-on and transport reader; native SDKs execute only in the external
runtime. Follow the package's `INSTALL.txt` and this project's README.

## Install the release package

Extract the package, close Blender, and run the installer from the extracted
folder with an explicit Blender executable: `install.bat BLENDER_EXE` on Windows,
`sh install.sh BLENDER_EXECUTABLE` on Linux. It installs into that Blender's own
settings folder, uses Blender's Python, and warms up the renderer's shader caches
(exit code 3 means the add-on is installed but the warm-up failed). For a
separate configuration, pass a new empty `SETTINGS_DIR` to both the installer and
`launch.bat`/`launch.sh`. Re-running a newer package's installer is the upgrade
path; it removes the previous add-on and runtime first.
Settings paths may contain spaces and non-ASCII characters; on Windows a non-ASCII
settings folder keeps the native runtime in the folder named by
`ovrtx-sdk\native-root.json` (`%ProgramData%\ovrtx-blender\<hash>`), which Remove
Runtime deletes. Keep the package files together, install only the package the
user selected, and preserve existing user settings and scene files.

The installed ready selector chooses the verified sidecar interpreter and worker
script. Native bridge client/module overrides and online release discovery are
not part of this port's supported setup. Do not substitute another runtime when
an integrity or version check fails.

## Readiness checks

- Stock Blender 5.2 or newer, enabled installed extension and `OVRTX_EXAMPLE` engine.
- Installed add-on/runtime pair from one package, complete locked tuple and exact shipped file hashes.
- Ready selector and external interpreter/worker within the settings folder.
- Camera, visible geometry and lighting/world in the current scene.
- Actual finite/nonblank rendered pixels and natural owned-worker/resource cleanup.

Smoke-test by selecting `OVRTX_EXAMPLE`, switching the viewport to Rendered
shading and rendering a still of a scene with a camera, geometry and lights. For a
fixed reference, follow the project's `docs/verification-workload.md`. The
automated installed-package verifier belongs to the upstream build repository and
is not part of this project. Preserve the first error.

LDR arrives pre-corrected and bypasses Blender color correction. Only scene-linear
HDR receives Blender's configured display transform. Do not hide a color
ownership error by changing material, light or exposure values.

Release qualification used representative runs; exhaustive benchmarking and long
stress work are deferred. See the project's `docs/support-platforms.md` for the
measured Windows and Linux scope.

## Troubleshoot and report

For a missing engine, inspect the extension/profile registration error. For an
unavailable runtime, retain the exact manifest/interpreter/path/version diagnostic
and re-run the installer from the complete, unmodified package. For blank pixels, inspect camera, geometry,
lighting and Rendered viewport shading before altering scene colors. A Cycles or
EEVEE frame does not establish OVRTX rendering. Return the selected Blender/settings folder,
package identity, readiness result, actual smoke result and first actionable
blocker. Keep logs local and sanitize credentials before any authorized sharing.
