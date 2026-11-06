# Your first visit

Open the sample folder and double-click `scripts/launch_ov_library_lab.bat`.
First setup creates an isolated Python environment, downloads the pinned libraries
and three scene archives, verifies their contents, and prepares RTX shaders.
The default scene includes the campus, robot guide and prepared exhibits.
The launcher downloads the archives automatically without a GitHub account;
see [scene asset setup](SCENE_ASSETS.md) for checksums and manual installation.

RTX starts with **Robot sensors**: a mobile robot, live LiDAR point cloud and
front/rear fisheye images. OVRTX renders all three from one scene revision.
Pause the scripted route, move the target pallet and compare their outputs.
Expand **Point-cloud display** for Distance/Height colors or a 3D/Top view.
The illustrative sensor profile is not a calibrated physical sensor or an
autonomous-navigation example. **Concept car** is the second subject; **Robot arms** is third.
The sensor is prepared during loading and uses additional GPU memory; this
mode has been verified on the documented 24 GB GPU. If its worker stops,
**Retry sensors** reconnects it while the rest of the lab remains available.

In RTX, select Concept car to change the paint, switch finish, and move the light. Try
Physics next: release the cartons, reset, and compare friction. Visit Stage to
turn one AMR into a fleet. Newton and Package offer a short numbered sequence:
select a motion and run it; or publish a robot revision and install it.

Storage and Package each emphasize one next-action button. Storage guides you
through save, change and restore, with the working rack on the left and saved
snapshot on the right. Package guides you through publish and install, then
verifies the installed files. Expand **More** for revision history and optional
editing rather than guessing which action comes next.

RTX's **Camera** tab also displays native distance, surface-normal and object
class outputs, with perspective or wide fisheye optics. Physics has optional
collision-shape and mass controls. OVUI's **In scene** panel shares its controls
with the sidebar; adjusting either changes the same equipment value.

Tabs change pages; radio controls select a mode; action buttons perform work.
The result identifies what ran. Reset room restarts an experiment; Start over
returns to the atrium and keeps your saved data.
The **i** beside each library name opens a separate learning page with explanations,
use cases and example code. **Back to controls** restores the same experiment
without resetting it or resizing the viewport. **Reset room** remains a small
utility action at the bottom of the controls.

WASD walks, arrows/drag look, Shift moves faster, Home returns to the atrium.
Press Esc to release movement and click the scene to resume. F1/settings lets
you increase text, reduce quality, retry a failed action, or save diagnostics.
Begin with Balanced quality and use Performance if navigation feels slow.

| Symptom | Next step |
| --- | --- |
| Download failed | Check access to GitHub, PyPI, and pypi.nvidia.com; Retry reuses cached files. |
| Version missing or wrong | Use Repair and retry in the launcher. |
| First image taking time | Check the preparation message; first shader compilation can take minutes. |
| Slow walking | Try Performance and include GPU, display scaling, and quality in feedback. |
| Small text | Use Settings > Text size; Windows scaling also applies. |
| Keys edit a control | Click the scene to restore navigation focus. |
| Experiment fails | Note the message, retry/reset, and save diagnostics if it repeats. |
| App exits | Use the launcher's View log, then Retry or Repair. |
| Newton/Package unavailable | Repair installs their isolated worker environments. |

Your runtime folder is `.ov_library_lab_runtime` beside the sample by default.
It contains logs, inspection images, storage revisions, Package workspaces, and
diagnostics. Saved files survive resets and app updates if the runtime is kept.
Review logs before sharing: they can contain local paths. Diagnostics are local;
the application uploads nothing. Use the [feedback template](FEEDBACK.md) to
report a reproducible issue or suggest improvements to a showcase.
