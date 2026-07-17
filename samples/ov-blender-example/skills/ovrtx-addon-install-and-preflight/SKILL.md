---
name: ovrtx-addon-install-and-preflight
description: Install, enable, update, and preflight the OVRTX Blender add-on and its runtime bundle. Use when installing an extension ZIP or developer checkout, diagnosing a missing OVRTX render engine, or verifying runtime readiness through the documented setup.
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
# OVRTX add-on install and preflight

Treat the add-on and the OVRTX/OVPhysX runtime as two distribution layers. The add-on source is inspectable and may be extended. The native worker and client are installed or supplied as runtime components through the supported installer/configuration surface.

## When to Use

Use when installing an extension ZIP or developer checkout, diagnosing a missing OVRTX render engine, or verifying runtime readiness through the documented setup.

## Instructions

1. Confirm Blender meets the version requirement in the included README and add-on manifest.
2. Install only a release the user selected from a documented distribution
   location. Verify its publisher and declared digest or signature when one is
   supplied. Obtain explicit consent before any network download or native
   component installation. In Blender, install the release extension ZIP
   through the Extensions/Add-ons “Install from Disk” flow, then enable `ovrtx
   Blender Example` and restart Blender if requested.
   Keep the selected Release page URL with that ZIP. Do not discover, infer, or
   substitute a “latest” release.
3. For contributor work, use the checked-out `addon/` package only through the documented development/install workflow. Do not copy individual Python modules into Blender's user scripts directory: the extension registration and manifest are a unit.
4. Confirm that the render engine `OVRTX_EXAMPLE` appears and that the OVRTX Example panel is registered. If it does not, collect the Blender console traceback and verify that the installed package matches the intended Blender version.

## Install or verify the runtime

Open the add-on Preferences and inspect the Runtime panel. **Install Runtime
From** intentionally starts empty. Paste the exact GitHub Release page URL that
supplied the add-on ZIP, or enter an absolute folder containing that Release's
external manifest and every component archive. The install action remains
disabled until a source is explicit. Never guess a repository, tag, or parent
folder. The add-on verifies the external manifest against its embedded digest
before acquiring components and materializes the verified platform runtime in
Blender extension storage.

Use **Reset to Installed Runtime** after experimenting with advanced overrides. Advanced fields (worker command, native client path, and module name) are diagnostic escape hatches, not ordinary artist prerequisites.

Use the documented setup and runtime diagnostics. If an organization supplies a preinstalled runtime, use the documented path/module values or leave them at the installed defaults.

## Preflight gates

Run the add-on's own preflight/status view and classify each result:

- **Blender/add-on:** Blender version, add-on enabled, and `OVRTX_EXAMPLE` engine registered.
- **Runtime bundle:** installed state, manifest match, platform, and verified component files.
- **OVRTX worker:** configured command is resolved and launchable by the add-on.
- **Native client:** configured path exists and the installed module exposes the installed client surface expected by the add-on.
- **Scene boundary:** a current `.blend` or imported USD is available; no developer fixture catalog is required for a user scene.

Only claim “ready” when all required checks pass. A missing runtime, client import error, incompatible platform, or unavailable GPU is a blocked preflight; report the check label and message exactly as shown.

## Smoke test

1. Start Blender with the add-on enabled and open a disposable scene or a copy of the user's `.blend`.
2. Set the render engine to OVRTX Example, choose a camera, and request a one-sample or low-sample render.
3. Confirm that an image or rendered viewport appears, then raise samples only after the first frame is valid.
4. Keep the add-on session logs and the output path. If the smoke fails, use the panel's log-folder action and preserve the first error; do not mask it with repeated restarts.

## Troubleshooting boundaries

- **Add-on absent:** reinstall the extension ZIP/check the package manifest and restart Blender.
- **Runtime missing or stale:** use Install/Verify/Retry, then Reset to Installed Runtime; do not hand-edit cached runtime files.
- **Worker/client check fails:** verify the installed runtime status and documented configuration. Treat the worker/client as installed components; escalate with the status message, platform, add-on version, and logs.
- **Engine registered but render fails:** continue with `ovrtx-current-scene-workflow` diagnostics and capture the session log. A Blender Cycles/Eevee render is not evidence that OVRTX rendered.
- **Offline environment:** download the complete paired Release asset set on an
  approved connected system, transfer it unchanged into one directory, and use
  that absolute directory as **Install Runtime From**. Do not invent an offline
  package or substitute a different renderer while claiming OVRTX success.

## Closeout report

Return Blender version, add-on version/source revision, runtime state, preflight checks, platform/GPU if available, smoke output path, and blockers. Never include credentials in the report.
Keep the full report local; use relative or sanitized paths and
`blender-sanitized-support-bundle` before sharing logs or diagnostics.
