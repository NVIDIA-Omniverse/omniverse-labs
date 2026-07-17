# Blender/OVRTX Skills

This catalog contains workflows for users and contributors of the OVRTX Blender
add-on.

## Runtime boundary

Use Blender, the add-on source or release extension, and its supported runtime
installation through the documented setup and add-on interfaces.

## Current inventory

- Persona entry points: `ovrtx-creative-hero-journey`, `simready-prop-journey`
- Routing/onboarding: `blender-workflow-routing`, `blender-mcp-setup`, `ovrtx-addon-install-and-preflight`, `blender-content-safety-and-privacy`
- Safe sharing: `blender-sanitized-support-bundle`
- Scene/USD handoff: `ovrtx-current-scene-workflow`, `usd-copy-and-flatten`, `usd-inspect-and-provenance`
- SimReady/OVPhysX: `simready-addon-install-and-authoring`, `ovphysx-simulation-workflow`, `ovphysx-drop-contact-acceptance`
- Rendering/lookdev: `ovrtx-render-settings`, `ovrtx-materialx-openpbr`, `ovrtx-color-management`, `ovrtx-lighting-and-world`, `ovrtx-render-products-and-aovs`, `ovrtx-render-sequence`, `ovrtx-hero-render`
- Sensors/AOVs: `ovrtx-sensor-capture`, `ovrtx-semantic-aov-capture`, `ovrtx-lidar-runtime-capture`
- Blender production: `geometry-nodes-for-ovrtx`, `reference-to-3d-reconstruction`, `texture-uv-material-workflow`, `animation-quality-and-frame-range`, `blender-asset-library-integration`
- Evidence/contributor: `blender-image-evidence-review`, `blender-ui-evidence-capture`, `ovrtx-render-parity-validation`, `blender-addon-extension-development`
- Add-on development: `extend-ovrtx-render-settings`, `extend-ovrtx-render-sequences`, `extend-ovrtx-lidar-capture`, `extend-ovrtx-semantic-aovs`, `extend-ovrtx-materialx-openpbr`, `extend-ovrtx-interactive-edits`, `extend-ovrtx-scene-generation`, `extend-ovrtx-lighting-world`

The machine-readable inventory is `manifest.json`: 46 entries across
consumer, advanced-consumer, contributor, onboarding, and orchestration roles. Every skill should
pass the Codex quick validator, state its audience and runtime prerequisites,
distinguish native from cached/postprocessed evidence, and use current
add-on interfaces.
