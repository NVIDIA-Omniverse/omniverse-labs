# OVRTX Blender addon

Use NVIDIA RTX rendering in stock Blender 5.2 LTS through the locked external
OVRTX, OVStage, OVStream, Warp and OVPhysX runtime. Blender loads the Python addon
and transport reader; native SDKs run in separate owned workers.

The Windows MVP supports the current Blender scene, interactive navigation and
edits, final still/animation output, and production-admitted rigid-body physics
and bake workflows. One selected render product with LDR or HDR is admitted.
Multiple render products and additional AOVs are future work and are rejected
explicitly. Linux x64 also has local package/install, external rendering and
recovery, CPU/GPU physics, and numeric physics-to-render synchronization
qualification. See [support platforms](docs/support-platforms.md) for the measured
scope; human visual acceptance remains separate.

## Install

You need Blender 5.2 or newer (stock, unmodified), a Windows x64 or Linux x64
machine, and a supported NVIDIA RTX GPU and driver. Blender itself is not
included.

1. Download the release package ZIP for your platform (`windows-x64` or
   `linux-x64`) from the 0.2.0 release, tag `ovrtx-blender-example-0.2`, and
   extract it. It contains the add-on (`ovrtx_blender_example.zip`), the matching
   runtime (`ovrtx-runtime-<platform>.zip`), their distribution manifest, the
   installer, `INSTALL.txt`, `LICENSE` and a `licenses/` folder. Keep these
   files together.
2. Close Blender, then run the installer from the extracted folder with your
   Blender executable:

   ```text
   Windows:  install.bat "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
   Linux:    sh install.sh /path/to/blender-5.2/blender
   ```

   The add-on and its runtime go into that Blender's own settings folder
   (`%APPDATA%\Blender Foundation\Blender\<major.minor>` on Windows,
   `~/.config/blender/<major.minor>` on Linux). The installer uses Blender's own
   Python; no other Python, checkout or download is needed.
3. The installer then warms up the renderer once so its shader caches are built
   before your first render. On an empty cache this takes several minutes (about
   4-5 minutes on an RTX A6000). If the warm-up fails, the add-on is still
   installed, the installer exits with code 3, and the first render builds the
   caches instead.
4. Start Blender normally. The add-on is enabled and its runtime lives in the
   settings folder's `ovrtx-sdk` subfolder.

To keep a separate, self-contained Blender configuration, pass a new empty
settings folder to both the installer and the launcher:
`install.bat BLENDER_EXE SETTINGS_DIR` then `launch.bat BLENDER_EXE SETTINGS_DIR`
(`install.sh`/`launch.sh` on Linux).

**Upgrading:** run the new release's installer the same way. It removes the
previous copy of the add-on and its runtime first, so re-running the installer is
the upgrade path. **Removing:** use **Remove Runtime** in the add-on preferences,
then remove the add-on in Blender.

Settings folder paths may contain spaces and non-ASCII characters. On Windows a
settings folder whose path is not ASCII keeps the native runtime in
`%ProgramData%\ovrtx-blender\<hash>`, named in its `ovrtx-sdk\native-root.json`
and removed with the runtime.

Building the add-on and runtime packages from source uses the upstream build
repository's packaging and lock tooling, which is not part of this project. This
folder contains the add-on, runtime and installer source those packages are built
from.

## Render and color

Enable **ovrtx Blender Example**, select the `OVRTX_EXAMPLE` render engine, and
use Rendered viewport shading. Open a scene with visible geometry, lighting and
a camera, then start with a small sample budget. Use Blender's normal final
render or animation commands for camera output.

LDR pixels from OVStream are already color-corrected and pass through Blender
without another color transform. HDR pixels are scene-linear and receive
Blender's configured view/display transform. Renderer workers isolate Blender's
inherited `OCIO` variable while Blender keeps its own display configuration.
HDR float32 presentation staging defaults off. `OVRTX_HDR_GPU_FLOAT32=1` opts in
on Windows and adds a presentation plane while retaining exact binary16 pixels.

See [Public SDK frame transport](docs/public_sdk_frame_transport.md) for exact
payloads, owned copies, ACK/cancel ownership and bounded carrier/message limits.
See [Support platforms](docs/support-platforms.md) for the observed Windows
environment and deferred platforms.

## Verify and troubleshoot

To check an installation:

1. Start Blender and open a scene with visible geometry, lighting and a camera
   (the default startup cube scene is enough).
2. Select the `OVRTX_EXAMPLE` render engine and switch the viewport to Rendered
   shading. The viewport shows the OVRTX image after the renderer starts.
3. Render a still with **Render > Render Image**. The result appears in Blender's
   image editor.

For a fixed reference render with expected results and performance numbers, use
the [Junk Shop verification workload](docs/verification-workload.md).

If the engine is unavailable, confirm the add-on is enabled in the Blender you
installed into. If the runtime is unavailable, keep its exact
integrity/path/version error and re-run the installer from the complete,
unmodified release folder rather than substituting another SDK or native client.
If the viewport is blank, check camera, geometry, lights and Rendered shading
before changing materials or color settings. Keep the first error and the logs
described in the verification workload when diagnosing a failed scene update or
restart.

Release qualification used representative checks rather than exhaustive
performance, staging, memory or stress testing; see
[support platforms](docs/support-platforms.md) for the measured scope. The
automated installed-package verifier and qualification harnesses belong to the
upstream build repository and are not part of this project.

## License

Project-authored source and documentation are licensed under Apache-2.0.
NVIDIA OVRTX, NVIDIA OVPhysX, their separately installed runtime components,
and third-party assets retain their own license terms. See [LICENSE](LICENSE),
[Licensing and Distribution](docs/licensing.md), and
[Third-Party Notices](THIRD_PARTY_NOTICES.md).
