# Developing the public sample

Application source is in `source/nvidia/ov_library_lab/`. `app.py` owns the host,
`presentation.py` navigation and panels, `native_runtime.py` rendering and PhysX,
and room/worker modules own their real library operations. `content.py` is the
available catalog. Worker processes are closed through `owned_process.py`.

Use the [source map](SOURCE_MAP.md) to locate each library integration and
`scripts/examples/` for small starting points. Do not place dependencies,
SDK binaries, credentials, visitor data, or logs in source control. Preserve
asset attribution and keep generated scenes versioned separately.

With the installed main environment's Python, from the sample root:

```powershell
python tools/check_source.py
python tools/check_experience.py --runtime-root PATH_TO_RUNTIME --output deliverables/flows.json
python tools/check_extended_rooms.py --runtime-root PATH_TO_RUNTIME --output deliverables/extended
python tools/preflight.py --runtime-root PATH_TO_RUNTIME --output deliverables/preflight.json
```

Only one GPU application/check should run at a time. The flow checks cover
public room navigation, source viewers, real physics, error/retry, reset and
saved-file preservation, quality changes, and camera capture. Extended checks
exercise Newton motion and Package publish/install/repair. They are not a
substitute for unassisted installation or human usability sessions.

Keep walking/input responsive; defer preparation and long SDK operations to
loading or owned workers. A room change should not resize the viewport. Keep
selection and action controls distinct and acknowledge long actions visibly.
Use the same hardware/preset and warmed shaders for timing comparisons.

`storage_panel.py` and `package_flow.py` derive the next action from completed
library operations. `sensor_views.py` displays native RTX measurements and
preserves raw tensors for export. `equipment_panel.py` projects a screen overlay
beside the stacker and shares its value model with the sidebar. Preserve these
contracts when extending the experiments; a display color is not a measurement,
and scripted equipment motion is not a physics result.

`config/sample-files.json` lists the curated source inventory. Packaged exports
add a manifest identifying each source file by hash. `config/distribution.json`
records the audience and build identity. See the [release notes](RELEASE_NOTES.md)
for the supported experiences and current validation limits.
