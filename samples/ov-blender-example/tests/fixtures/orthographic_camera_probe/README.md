# Orthographic Camera Probe Fixtures

These USD fixtures are the shared input corpus for the orthographic camera
capability probes.

Each `profile-*` directory contains:

- `perspective.usda`: the control scene, with `/World/Camera` or `/Camera`
  authored as a perspective camera.
- `orthographic.usda`: the matching scene, with the selected camera authored as
  an orthographic camera.

The Blender reference probe renders these files with Blender's local renderer to
check that the colored cube scene and camera settings are sensible without
OVRTX. The OVRTX probe renders the same files through the worker/client render
product path. A runtime finding should only lean on an OVRTX failure when the
matching Blender reference fixture renders visible cubes as expected.

The Blender reference probe assigns local red, green, and blue materials by
imported cube object name after USD import. That keeps the local reference
focused on camera and geometry visibility because Blender 5.1 does not preserve
these `UsdPreviewSurface` material bindings when importing the fixtures.

The authored render product selected by both probes is:

`/Render/OmniverseKit/HydraTextures/ViewportTexture0`

The selected camera is `/World/Camera` for most profiles and `/Camera` for the
`profile-root-matrix` variant.

## License

These project-authored probe fixtures are licensed under Apache-2.0. See the
repository [LICENSE](../../../LICENSE).
