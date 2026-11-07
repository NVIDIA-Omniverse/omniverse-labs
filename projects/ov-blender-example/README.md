# `ovrtx` Blender Example

Render Blender scenes with NVIDIA `ovrtx` and simulate rigid bodies with OVPhysX. The add-on translates supported Blender scene data to
USD; rendering, simulation and native SDKs run in external workers.

![The Junk Shop rendered with NVIDIA ovrtx](docs/images/perf_junk_shop_1280x720.png)

*The Junk Shop by Alex Treviño, original concept by Anaïs Maamar,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Rendered with NVIDIA ovrtx.*

Download the Windows x64 or Linux x64 package from [Blender 0.2.0 release](https://github.com/NVIDIA-Omniverse/omniverse-labs/releases/tag/ovrtx-blender-example-0.2).

## Guides

| Start here | What you will do |
| --- | --- |
| [Install or upgrade](docs/install-guide.md) | Extract the matching platform package, install the add-on/runtime pair and verify first pixels. |
| [Render a scene](docs/render-guide.md) | Open Junk Shop, review edits in the viewport, render the active camera and save a still or animation. |
| [Add physics and bake motion](docs/physics-guide.md) | Configure a Dynamic prop and Static floor, inspect colliders, preview/reset, bake and render. |
| [Build and extend](docs/developer-guide.md) | Find the source components, runtime dependencies and verification steps. |

The [project page](https://nvidia-omniverse.github.io/omniverse-labs/projects/ov-blender-example/)
summarizes installation, rendering and physics setup. See
[platforms and runtime versions](docs/support-platforms.md) and
[0.2 release notes](docs/release-notes-0.2.md) for package scope. For a repeatable
reference render, use the [Junk Shop verification workload](docs/verification-workload.md).

## Features

- Rendered viewport navigation and supported scene edits through ovrtx.
- Active-camera stills and Blender Render Animation using one LDR or HDR product.
- OV body, collider and physics-material authoring with preview/reset.
- Supported joints, kinematic inputs and bone-backed workflows within the
  documented runtime/device restrictions.
- Simulation baking into editable Blender Actions and saved Blender playback.

This is an experimental integration with no production support SLA. It does not
reproduce every Blender feature or promise Cycles/EEVEE pixel parity. Material
translation is a subset; extra AOVs/multiple render products, `ovrtx` skinned
motion, cloth and fluids are outside the documented package workflows. Native
physics admission is specific to the exact platform/runtime pair.

## Runtime components

Blender owns authoring and UI. The external runtime uses OVStage for scene
representation, `ovrtx` for rendering, OVStream for frame transport and OVPhysX for
simulation. Use the package's locked runtime; installing unrelated SDK versions
into Blender is not a replacement for the matched sidecar.

## License

Project-authored source and documentation are licensed under Apache-2.0.
NVIDIA `ovrtx`, OVPhysX and other separately installed runtime components and
third-party assets retain their own terms. See [LICENSE](LICENSE),
[Licensing and Distribution](docs/licensing.md) and
[Third-Party Notices](THIRD_PARTY_NOTICES.md).
