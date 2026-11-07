# Build and modify the integration

Use this guide to change the add-on's code. To use the add-on in Blender, start
with [installation](install-guide.md) and [your first render](render-guide.md).

## How the components fit together

The add-on sends scene updates to a separate process called the **worker**. The
worker runs rendering and physics, then sends images and object positions back
to Blender.

| Component | Role |
| --- | --- |
| Blender add-on | Read scene edits, manage the worker and display rendered images. |
| ovstage | Store the worker's scene and apply updates. |
| `ovrtx` | Render images. |
| ovstream | Transfer images through local shared memory. |
| ovphysx | Simulate rigid bodies and joints. |
| External runtime | Provide Python, the SDKs and their native dependencies. |

Keep the add-on and runtime from the same package together. Install native SDKs
in the external runtime, not in Blender's Python. See the
[packaged versions](support-platforms.md#runtime-versions).

## Find the code

| Task | Source |
| --- | --- |
| Convert scenes and track USD objects | [`scene_generation.py`](../addon/ovrtx_blender_example/scene_generation.py) |
| Order rendering and physics work | [`runtime_scheduler.py`](../addon/ovrtx_blender_example/runtime_scheduler.py) |
| Start and replace renderer sessions | [`ovrtx_session_controller.py`](../addon/ovrtx_blender_example/ovrtx_session_controller.py) |
| Define render requests | [`render_requests.py`](../addon/ovrtx_blender_example/render_requests.py) |
| Communicate with rendering and physics | [`ovrtx_runtime_client.py`](../addon/ovrtx_blender_example/ovrtx_runtime_client.py), [`ovphysx_runtime_client.py`](../addon/ovrtx_blender_example/ovphysx_runtime_client.py) |
| Implement SDK rendering and physics | [`public_sdk_backend.py`](../runtime/public_sdk_backend.py) |
| Transfer frames | [`ovstream_frames.py`](../runtime/ovstream_frames.py) |
| Match SDK versions | [`public-sdk-lock.json`](../runtime/public-sdk-lock.json) |
| Build paired packages | [`build_public_sdk_packages.py`](../scripts/build_public_sdk_packages.py) |
| Display viewport images | [`viewport_presentation.py`](../addon/ovrtx_blender_example/viewport_presentation.py) |

Read the [architecture](architecture.md) for session lifetimes and scene updates.
The [development skill](../skills/blender-addon-extension-development/SKILL.md)
maps common changes to their tests.

## Make and test a change

1. Start from source that matches your package. Keep its SDK versions and package manifest together.
2. Make the change in the relevant component. Keep Blender callbacks responsive; send native work through the existing worker scheduler.
3. Run the tests for the component you changed. With the project's test dependencies installed, run the Python suite from the project directory:

   ```bash
   env PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
   ```

4. Follow your source distribution's build instructions to create a matching add-on ZIP, runtime ZIP and distribution JSON.
5. [Install the rebuilt pair in a separate Blender settings folder](install-guide.md). Confirm Blender loads the rebuilt add-on and its matching runtime.
6. Open a simple scene. Check viewport updates, an F12 render, and worker reconnect/restart. For physics changes, also check motion, reset and baking.

Python tests do not exercise the native renderer or physics engine. Test those
features in Blender before accepting a change.

### Keep updates consistent

- Preserve object identities across scene edits. If replacing a scene fails, keep the last working scene.
- Keep rendering and physics operations in the existing serialized session.
- Preserve image dimensions, pixel format and frame ownership when handling output.
- Preserve scene units and complete object poses when handling physics updates. Check device restrictions before enabling a feature.

Changing SDK versions or worker adapters requires new runtime checks. Rebuild
and test the complete package instead of replacing individual installed files.
The retired SIL bridge build commands do not apply to this integration.

## Compare performance

Use the same scene, camera, resolution, samples and color settings on the same
GPU and driver. Measure worker startup, first image, final render and viewport
navigation separately. Record whether shader caches were already warm.

Use the [Junk Shop verification workload](verification-workload.md) for fixed
scene settings, expected output and reference timings.

For physics changes, also measure reset and restart. Keep a short record of the
package version, scene settings and results so you can repeat the comparison.

## Find errors and logs

Read **Render Properties → ovrtx Session** before restarting. Use **Open ovrtx
Logs** when file logging is enabled. A worker error includes a `Details:` path;
keep `worker.log` and `ovrtx.log` from that same session's `epoch-N` folder.

For installation errors, use the printed log paths. Transaction records are in
`.ovrtx-transactions` beside the Blender settings folder. Exit code **3** means
the installation succeeded but renderer warm-up failed; check `warmup.log` and
the reported worker log.

A bug report should include the package version, OS, GPU and driver, steps to
repeat the problem, and the first error. State whether reconnecting or restarting
helped. Remove private paths, credentials and scene data before sharing logs.
