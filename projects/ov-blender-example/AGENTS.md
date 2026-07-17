# AGENTS.md

Project-specific guidance for agents working in `ov-blender-example`.
Follow the workspace-level `AGENTS.md` as well.

## Skills

The distributable agent skills live in `skills/`. `skills/manifest.json` is the
complete catalog; every listed path must resolve inside that directory.

- Read a matching skill's complete `SKILL.md` before changing code or running a
  workflow. Use `skills/blender-workflow-routing/` when the request spans more
  than one consumer workflow.
- Use `skills/blender-content-safety-and-privacy/` before opening externally
  supplied or otherwise untrusted content. Keep originals unchanged and keep
  unsanitized diagnostics local.
- Use `skills/blender-sanitized-support-bundle/` before sharing logs, reports,
  screenshots, manifests, or reproduction artifacts.
- Start artist journeys with `skills/ovrtx-creative-hero-journey/` or
  `skills/simready-prop-journey/` when the request matches one of those personas.
- For source changes, start with
  `skills/blender-addon-extension-development/`, then use the narrowest matching
  `skills/extend-ovrtx-*/` skill for the implementation map and required tests.
- Resolve source and test paths in developer skills relative to this directory,
  such as `addon/ovrtx_blender_example/` and `tests/`.
- Use only the setup, scripts, source, and runtime interfaces documented in this
  distribution. If a capability is unavailable, preserve the diagnostic and
  report the gap rather than inventing another interface.
- Keep generated scenes, renders, logs, and reports out of the source tree
  unless a test fixture or documentation artifact is explicitly requested.
- When editing the skill catalog, keep each skill self-contained, update
  `skills/manifest.json`, and verify that the manifest names exactly match the
  `skills/*/SKILL.md` directories with no parent-directory paths.

## Full Test Sweep

When a user asks for a deterministic test sweep, run:

- `env PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest`

The validation commands consume one existing materialized runtime through
`--runtime-root`. Semantic suites are invoked
directly with `scripts/validate_suite.py <suite>`; `scripts/validate.py` and
`scripts/validate_extended.py` run the fixed Task and Extended compositions.
