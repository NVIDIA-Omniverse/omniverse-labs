# Third-Party Notices

## Scope

This file records the direct third-party dependencies imported by the
project-authored Blender add-on source, which is licensed under the Apache
License, Version 2.0 (see `LICENSE`). It does not list Python's standard
library or the transitive dependencies of Blender, OpenUSD, NumPy, gRPC, or an
externally supplied runtime bundle.

The packaged runtime (the `ovrtx-runtime-<platform>.zip` sidecar) carries its
own `licenses/` folder. That folder is generated from the files the runtime
actually ships and lists every component with its exact version, its license,
and the license texts (`licenses/README.md`, `licenses/components.json`).

## Blender

- **Component:** Blender 5.2 or later, including the `bpy`, `bpy_extras`,
  `gpu`, `gpu_extras`, and `mathutils` Python APIs.
- **Use in this project:** Host application and Python API for the add-on.
- **Copyright:** Blender Foundation and contributors.
- **License:** GNU General Public License, version 3.
- **Source and license:** <https://github.com/blender/blender> and
  <https://www.blender.org/about/license/>.
- **Distribution status:** Supplied by the host Blender installation; not
  vendored in this repository.

## OpenUSD

- **Component:** OpenUSD Python bindings (`pxr`).
- **Use in this project:** USD scene composition, authoring, material, and
  physics APIs.
- **Copyright and license:** Determined by the OpenUSD version supplied by the
  host runtime. This repository does not pin that version or retain its notice;
  do not infer terms from a different OpenUSD release.
- **Source and license:** <https://github.com/PixarAnimationStudios/OpenUSD>.
- **Distribution status:** Supplied by the runtime environment; not vendored in
  this repository. Obtain the version-specific notice before distributing the
  host runtime.

## NumPy

- **Component:** NumPy (`numpy`).
- **Use in this project:** Optional render-frame array processing.
- **Copyright:** 2005-2025 NumPy Developers.
- **License:** BSD 3-Clause License.
- **Source and license:** <https://github.com/numpy/numpy>.
- **Distribution status:** Supplied by the Python/Blender environment; not
  vendored in this repository.

## gRPC

- **Component:** gRPC Python runtime (`grpc`).
- **Use in this project:** OVRTX control-plane RPCs. The bundled OVPhysX
  runtime lookup also recognizes gRPC 1.64.3 runtime library locations.
- **Copyright:** 2015 gRPC authors.
- **License:** Apache License, Version 2.0.
- **Source and license:** <https://github.com/grpc/grpc>.
- **Distribution status:** Supplied by the external runtime deployment; not
  vendored in this repository.

## Maintenance

Update this file whenever an add-on import changes or a dependency becomes
vendored. The runtime's `licenses/` folder is rebuilt with every runtime package
by `scripts/runtime_licenses.py`; the build fails when a shipped component has
no version or no license text.
