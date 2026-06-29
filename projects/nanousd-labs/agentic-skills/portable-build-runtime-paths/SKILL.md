---
name: portable-build-runtime-paths
description: Make native builds, command-line tools, viewers, plugins, shaders, test harnesses, and packaged artifacts portable across working directories, install trees, copied build trees, users, and machines. Use when Codex works on CMake/install rules, RPATH/RUNPATH, dynamic library lookup, shader or asset discovery, CLI resource paths, packaging, deployment, or bugs that only reproduce outside the repo root.
---

# Portable Build Runtime Paths

## Purpose

Treat relocation as a build contract. A produced binary should run from documented build and install layouts without relying on the developer's current directory, home directory, username, shell setup, or stale environment variables.

## Workflow

1. Define the runtime layout.
   - Name the supported layouts: build tree, install tree, copied artifact folder, package bundle, plugin folder, or source checkout.
   - List every runtime dependency: shared libraries, plugins, shaders/SPIR-V, kernels, USD files, textures, config, Python modules, licenses, and helper executables.
   - Decide which paths are user inputs and which are packaged resources.

2. Choose stable anchors.
   - Resolve packaged resources relative to the executable, shared library, plugin, package root, or an explicit `--resource-dir`/config value.
   - Resolve scene-authored assets relative to the authored layer or scene context, not process `cwd`.
   - A loader that *discovers* sidecar assets (materials, textures) by recursively walking the scene directory breaks when the scene lives in a shared system dir (e.g. `/tmp`): it ingests hundreds of unrelated files and a ~170 ms load becomes seconds. Prefer parsing explicit USD references; if you must scan, bound it to the scene directory, gate it behind a cheap "any candidate files here?" check, and provide an environment opt-out.
   - Use process `cwd` only for user-provided relative paths and document that policy in CLI help.

3. Make dynamic linking relocatable.
   - Prefer build/install RPATHs based on `$ORIGIN` on Linux, `@loader_path`/`@rpath` on macOS, and app-local DLL search layout on Windows.
   - Do not bake absolute home, workspace, conan-cache, VM, or username paths into release artifacts unless the artifact is explicitly nonportable.
   - Keep `LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`, and shell setup scripts as developer conveniences, not required runtime contracts.

4. Make generated resources real build outputs.
   - Compile shaders, codegen outputs, and copied resources through build-system targets with declared inputs, outputs, and dependencies.
   - Install or package those outputs beside the binary or under a predictable resource directory.
   - Avoid hard-coded build-tree strings such as `build/foo.spv`; compute the path from the chosen runtime anchor.

5. Fail loudly and usefully.
   - On missing resources or libraries, report the exact path tried, the runtime anchor used, and the override flag or environment variable if one exists.
   - Include `--help` output for resource-path overrides when the binary supports them.
   - Do not hide portability failures by silently falling back to source-tree paths unless that is an explicitly documented dev mode.

6. Prove relocation.
   - Run `--help` and one smoke command from a directory outside the repo.
   - Run at least one smoke after `cmake --install` or equivalent packaging.
   - Copy or move the install/artifact directory to a temporary path with a different parent name and run again.
   - For path bugs, test from repo root, build dir, `/tmp`, and a path that does not resemble the developer's checkout.

## Checks

Useful checks when available:

```bash
readelf -d path/to/binary
ldd path/to/binary
otool -L path/to/binary
cmake --install build --prefix /tmp/app-install
```

For CLI tools, prefer smoke commands shaped like:

```bash
(cd /tmp && /tmp/app-install/bin/tool --help)
(cd /tmp && /tmp/app-install/bin/tool --scene /absolute/scene.usda --output /tmp/out.ppm)
```

## Review Checklist

- Does the binary run when launched from a different `cwd`?
- Does the installed artifact run without source-tree files present?
- Are RPATH/RUNPATH entries relative to the artifact, not absolute to a user's checkout?
- Are shaders, plugins, kernels, and other generated resources declared as build and install outputs?
- Are user-provided relative paths resolved consistently and separately from packaged resources?
- Are scene assets resolved through the scene/layer context rather than accidental process `cwd`?
- Do errors print the resolved path that failed?

## Anti-Patterns

- Opening packaged files with bare relative paths like `build/foo.spv` or `shaders/foo.spv`.
- Fixing local runs by requiring users to launch from the repo root.
- Shipping binaries that only work with a developer's `LD_LIBRARY_PATH`.
- Treating RPATH repair tools as the primary build design instead of fixing CMake/install rules.
- Letting tests pass because source-tree fallback masks missing installed resources.
- Copying libraries from another machine without verifying embedded runtime paths.

## Handoff

Report:

- Supported runtime layouts.
- Resource anchor and override policy.
- RPATH/library lookup strategy.
- Install/package files added or changed.
- Relocation smoke commands and outcomes.
- Remaining nonportable assumptions, if any.
