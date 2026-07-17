---
name: blender-mcp-setup
description: Connect an agent to a running Blender session through the user's Blender MCP server and verify a safe, usable scene-control loop. Use when a user is new to Blender MCP, asks to create or edit a scene through MCP, or reports that Blender tools cannot connect. This covers the Blender-side MCP connection only; OVRTX and OVPhysX runtime setup belongs to ovrtx-addon-install-and-preflight.
license: "Apache-2.0"
metadata:
  author: "Max Bickley"
  version: "0.1"
  team: "omniverse"
  domain: "physical-ai"
  tags:
    - blender
    - omniverse
    - ovrtx
    - mcp
---
# Blender MCP setup

Use the Blender MCP tools exposed by the user's configured Blender setup to inspect and control Blender. Do not assume a particular MCP implementation, endpoint, port, operating system, or OVRTX runtime package.

## When to Use

Use when a user is new to Blender MCP, asks to create or edit a scene through MCP, or reports that Blender tools cannot connect. This covers the Blender-side MCP connection only; OVRTX and OVPhysX runtime setup belongs to ovrtx-addon-install-and-preflight.

## Prerequisites

- Blender 5.1 or newer is recommended for the OVRTX example.
- Blender is running with the user's Blender MCP add-on/server enabled.
- The agent has access to scene inspection, Blender-Python execution, and (when available) a viewport screenshot tool.

The MCP server and the OVRTX worker are separate services. A working MCP connection does not prove OVRTX is installed, and an OVRTX preflight failure is not fixed by changing MCP settings.

Blender Python and MCP mutation tools have local-code authority. Use a
user-scoped, access-controlled endpoint, expose it only as required by the
user's deployment, and do not connect to an endpoint whose owner is unknown.

## Instructions

1. Discover the Blender MCP tools available in this session. Prefer the provider's scene-info/status call before executing code.
   For an untrusted `.blend`, first follow
   `blender-content-safety-and-privacy`: use an isolated Blender profile and
   open with automatic script execution disabled.
2. Make one non-destructive probe: read the Blender version, current file path (if any), active render engine, active camera, and object count. Do not clear or alter the scene during setup.
3. If the provider exposes a screenshot call, capture the current viewport and report whether it is a 3D View, camera view, or another editor.
4. Confirm the control loop with a harmless read-only Blender expression (for example, `bpy.app.version_string`). Only after the user asks for scene work should you execute mutations.
5. Return a compact readiness report: connection status, Blender version, current `.blend` path or “unsaved,” engine, active camera, object count, and any missing capability.

## Working rules

- Execute Blender Python in small, independently verifiable chunks. Re-import modules in each call; do not rely on variables surviving between calls.
- Address objects and datablocks by stable names, not selection order. Preserve existing user content unless deletion is explicitly requested.
- After every meaningful mutation, inspect the resulting object/camera/render state and take a viewport or render screenshot when available.
- Save a user-requested copy before destructive operations. Keep generated renders in a user-selected output directory.
- If Blender MCP is unavailable, stop at diagnosis and tell the user to start Blender, enable the MCP add-on, and verify the endpoint configured by their MCP deployment. Do not install an unrelated server or guess a port.
- Use the documented setup and runtime diagnostics. Runtime binaries and native clients are prerequisites handled by the OVRTX add-on.
- Do not enable embedded scripts, handlers, scripted drivers, or unknown
  add-ons merely to make an untrusted scene load or evaluate.

## Common failures

- **Cannot connect:** Blender is closed, the MCP add-on is disabled, the endpoint is wrong, or another process owns the endpoint. Ask the user to check those items in their deployment.
- **Scene info works but code fails:** report the exact Blender traceback and retry with a smaller chunk; do not repeatedly resend a large script.
- **Screenshot is unavailable:** continue with structured scene inspection, but mark visual validation as unavailable.
- **MCP works but OVRTX does not render:** switch to `ovrtx-addon-install-and-preflight`; this is not an MCP connection problem.
