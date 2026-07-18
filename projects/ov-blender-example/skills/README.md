# Blender/OVRTX Skills

This catalog contains workflows for users and contributors of the OVRTX Blender
add-on.

## Runtime boundary

Use Blender, the add-on source or release extension, and its supported runtime
installation through the documented setup and add-on interfaces.

## Current inventory

- Routing/onboarding: `blender-workflow-routing`, `blender-mcp-setup`, `blender-python-execution`, `blender-community-skill-bootstrap`, `ovrtx-addon-install-and-preflight`, `blender-content-safety-and-privacy`
- User journeys: `ovrtx-creative-hero-journey`, `simready-prop-journey`
- Safe sharing: `blender-sanitized-support-bundle`
- Scene/USD handoff: `ovrtx-current-scene-workflow`, `usd-copy-and-flatten`, `usd-inspect-and-provenance`
- SimReady/native physics: `simready-addon-install-and-authoring`, `ovphysx-drop-contact-acceptance`
- OVRTX rendering/lookdev: `ovrtx-current-scene-workflow`, `ovrtx-render-settings`, `ovrtx-color-management`, `ovrtx-materialx-openpbr`, `ovrtx-lighting-and-world`, `ovrtx-hero-render`
- Blender production: `blender-mesh-authoring`, `blender-camera-framing`, `blender-render-and-export`, `geometry-nodes-for-ovrtx`, `reference-to-3d-reconstruction`, `texture-uv-material-workflow`, `animation-quality-and-frame-range`; use `blender-community-skill-bootstrap` for optional complementary recipes directly from their upstream owner
- Add-on development entrypoint: `blender-addon-extension-development`
- Runtime customization: `ovrtx-render-products-and-aovs`, `ovrtx-sensor-capture`, `ovrtx-lidar-runtime-capture`, `ovrtx-semantic-aov-capture`, `ovrtx-render-sequence`
- Add-on development: `extend-ovrtx-render-settings`, `extend-ovrtx-render-sequences`, `extend-ovrtx-lidar-capture`, `extend-ovrtx-semantic-aovs`, `extend-ovrtx-materialx-openpbr`, `extend-ovrtx-interactive-edits`, `extend-ovrtx-scene-generation`, `extend-ovrtx-lighting-world`

The machine-readable inventory is `manifest.json`: 40 entries across
consumer, advanced-consumer, contributor, onboarding, and orchestration roles. Every skill should
pass the Codex quick validator, state its audience and runtime prerequisites,
distinguish native from cached/postprocessed evidence, and use current
add-on interfaces.

This directory is the distributable catalog shipped with the public add-on.
Keep it synchronized with the reviewed public skill source, while preserving
repository-specific paths and newer add-on interfaces maintained here.

Public production skills should provide an executable path, not just advice.
For operations they claim to own, require: named inputs; exact MCP, CLI, or
`bpy` invocation; version-sensitive API notes; bounded side effects and
rollback behavior; structured output; numeric acceptance gates; known failure
modes; and at least one representative Blender runtime test. The camera,
mesh authoring, render/export, reference registration, material/UV, Geometry
Nodes, SimReady authoring, OVRTX settings/hero rendering, and Python-execution
skills use this contract.

The catalog keeps generic user-outcome orchestrators when they compose public
technical skills without imposing a research evidence process. It excludes
host/session deployment, persona management, and project-specific validation
or evidence orchestration.

Runtime-product skills remain available to developers when they define an
honest capability probe and extension contract. They do not imply that every
installed runtime already supports LiDAR, semantic/ID, arbitrary sensors/AOVs,
or contiguous sequence capture.

## Optional community production skills

This catalog does not bundle the RobLe3 `cc-blender-skill` production stack.
`blender-community-skill-bootstrap` points Codex to the upstream repository,
known revision, license, and exact per-skill paths so an agent can ask the user
to install only the optional capabilities a request needs. The local
`blender-python-execution` skill remains the safety and validation overlay for
MCP/bpy work and contains no runtime bridge implementation.
