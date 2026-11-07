# Blender integration 0.2 release notes

Version 0.2 adds rendering and rigid-body simulation through an external runtime.
Use Blender **5.2.1 LTS** and install the add-on and runtime together.

## Features

- Preview scenes in the viewport and render stills or animations with `ovrtx`.
- Add rigid bodies, collision shapes, friction, bounce and gravity. Preview the motion, reset it, then bake it into editable Blender animation.
- Connect bodies with joints. Physics runs on the CPU by default, separately from GPU rendering.
- Render with MaterialX through **F12**. Final anti-aliasing defaults to **2×2**; viewport anti-aliasing defaults to **1×**.
- Choose **LDR** for display-ready colors or **scene-linear HDR** for Blender color adjustments and HDR output.
- Reconnect or restart the worker from **ovrtx Session** when rendering stops responding. The panel reports errors and log locations.

## Install or upgrade

Extract the complete platform package and run its installer. Installation includes
renderer warm-up, which can take several minutes. A warm-up failure is reported
separately from installation success.

To upgrade an existing 0.2 installation, close Blender and run the new package's
installer. For an earlier integration, we recommend using a
[separate Blender settings folder](install-guide.md#use-a-separate-settings-folder).

See the [installation guide](install-guide.md) for commands and recovery, and
[platforms and runtime versions](support-platforms.md) for package details.

## Limitations

- Packages are available for Windows x64 and Linux x64. ARM and macOS are unsupported.
- Only a subset of Blender shaders is supported. Materials may look different from Cycles or EEVEE.
- Rendering produces one LDR or HDR image at a time. Extra passes such as depth and normals, multiple render products and transparent output backgrounds are unavailable.
- Skinned deformation in `ovrtx`, live joint motor updates, articulation control, cloth, fluids and deformable-body simulation are unsupported in this workflow.
- GPU physics has narrower joint support than CPU physics. See the [physics guide](physics-guide.md#add-a-joint-or-a-bone-body).

For code changes and testing, see the [developer guide](developer-guide.md).
