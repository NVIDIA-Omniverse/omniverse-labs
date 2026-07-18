---
name: blender-python-execution
description: Execute and validate Blender Python through a Blender MCP provider using small, context-safe, idempotent transactions. Use when an agent must inspect or mutate a Blender scene, translate a Blender task into bpy code, recover from Blender code execution errors, or prove that an MCP-driven edit actually changed the intended scene.
---

# Blender Python execution

Use the provider's Blender tools as a transaction loop: inspect, execute one
bounded change, inspect structured state, then inspect pixels when the result is
visual. This skill owns execution discipline, not modeling or look-development
decisions.

## Discover the adapter

The known adapter exposes these schemas:

```text
get_scene_info()
execute_blender_code(code)
get_object_info(object_name)
get_viewport_screenshot(max_size=800)
```

Tool prefixes and provider aliases are discoverable and may differ. Match tools
by schema and behavior; do not guess an endpoint or port. Read
`references/mcp-call-contract.md` before the first mutation.

## Run the transaction loop

1. Call `get_scene_info()` before executing code. Identify the current file,
   scene, mode, active object, camera, engine, and objects in scope. Preserve
   existing content unless the user explicitly asks to replace it.
2. Run the read-only code in `scripts/scene_probe.py` through
   `execute_blender_code(code)` when the provider's scene summary is incomplete.
   Use `scripts/scene_audit.py` before rendering, export, or handoff. To prove
   camera coverage, prepend
   `BLENDER_AUDIT_REQUEST = {"targets": ["ROOT-subject"], "margin": 0.08,
   "include_descendants": True, "include_instances": True}`
   when executing through MCP, or pass `--targets GEO-subject --margin 0.08`
   after Blender's `--` delimiter; use `--no-descendants` or `--no-instances`
   only when those bounds are intentionally excluded. A camera merely existing is not a framing
   pass.
3. Define one transaction with named inputs, stable target names, a narrow
   mutation, and an expected postcondition. Prefer Blender's data API. Use an
   operator only when its context is prepared explicitly.
4. Send a small, self-contained code string to
   `execute_blender_code(code)`. Every call has a fresh Python namespace:
   re-import modules and reacquire datablocks from `bpy.data` by name.
5. End the code with one JSON object on standard output. Include `ok`,
   `operation`, target names, changed values or counts, and any warnings. Do not
   print credentials, environment variables, or unrelated paths.
6. Call `get_object_info(object_name)` for every primary target and
   `get_scene_info()` after scene-wide changes. Check the expected state rather
   than treating a successful tool response as proof.
7. For visible changes, call `get_viewport_screenshot(max_size=800)` or inspect
   the requested render. Verify framing, geometry, materials, and lighting. A
   screenshot can be stale or produced by another engine; it does not by itself
   prove that OVRTX rendered or accepted an edit.

## Execution rules

- Never use `time.sleep()` or a polling loop inside Blender code. Return control
  to the agent and poll with provider/status tools outside Blender.
- Keep each call bounded. Split creation, materials, lighting, rendering, and
  export into independently verifiable transactions.
- Use stable names and explicit ownership. Do not rely on selection order,
  default names, or Python variables from a previous call.
- Make mutations idempotent: get-or-create owned datablocks, set requested
  values absolutely, and avoid duplicate modifiers, links, handlers, or nodes.
- Call `bpy.context.view_layer.update()` before reading evaluated transforms,
  bounds, or dependency-graph results.
- Prefer direct data access. When an operator is necessary, set mode, selection,
  and active object explicitly or use `bpy.context.temp_override(...)` with a
  verified area/region. Restore user-visible context when practical.
- Do not delete broadly, reset the scene or World, open another file, save over
  the source, enable scripts, install add-ons, or render to an arbitrary path
  unless the user requested that exact action.
- Treat external `.blend`, USD, archives, scripts, drivers, and add-ons as
  untrusted content. Opening or enabling them is outside this execution skill.

## Mutation shape

Use the patterns in `references/python-transaction-patterns.md`. A minimal
transaction has this form:

```python
import bpy, json

name = "GEO-requested-target"
obj = bpy.data.objects.get(name)
if obj is None:
    raise RuntimeError(f"missing target: {name}")

obj.hide_render = False
bpy.context.view_layer.update()
print(json.dumps({
    "ok": True,
    "operation": "set_render_visibility",
    "object": obj.name,
    "hide_render": obj.hide_render,
}, sort_keys=True))
```

## Failure handling

- On a traceback, report the exception and failing line, reduce the transaction,
  reacquire state, and retry only the failed operation.
- On timeout, do not resend the same large script. Inspect the scene, split the
  work, and remove blocking waits or expensive loops.
- If structured state changed but pixels did not, check camera, view layer,
  visibility, dependency-graph evaluation, active engine, and stale viewport
  state before repeating the mutation. Use the interactive redraw/frame pattern
  in `references/python-transaction-patterns.md`, return control, then capture a
  new screenshot; use a render when background mode or freshness matters.
- For reproducible render-camera coverage, route to `blender-camera-framing`.
  Viewport `view_selected` changes only `RegionView3D` and cannot prove that a
  render camera contains the subject.
- If the provider lacks code execution, do not use shell Blender as a hidden
  substitute for mutating the user's live session. Stop and report the missing
  capability. A background Blender script is allowed only when the user
  explicitly authorizes an offline/caller-owned derivative or the routed
  workflow already declares background execution for a large build, frame
  loop, render, or round-trip test; keep it isolated from the live source.

## Completion gate

Report success only when the requested postcondition passes structured
inspection and every visual claim has current visual evidence. State the active
engine and label screenshots, Blender renders, native renderer products, and
postprocessed images separately.
