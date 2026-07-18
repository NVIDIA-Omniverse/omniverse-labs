# Blender MCP call contract

## Known schemas

The known adapter uses:

```text
get_scene_info()
execute_blender_code(code)
get_object_info(object_name)
get_viewport_screenshot(max_size=800)
```

The fully qualified tool names may have a provider prefix. Other providers may
use aliases. Discover tools and match their argument schemas rather than
assuming a fixed prefix, network endpoint, or port.

## Call order

### Read-only readiness

1. `get_scene_info()`
2. `execute_blender_code(code)` with the contents of `scripts/scene_probe.py`
3. `get_viewport_screenshot(max_size=800)` when available

The probe must return JSON and must not mutate or save the scene.

### Mutation transaction

1. `get_scene_info()`
2. `get_object_info(object_name)` for existing targets
3. `execute_blender_code(code)` for one bounded transaction
4. `get_object_info(object_name)` for each primary target
5. `get_scene_info()` for scene-wide postconditions
6. `get_viewport_screenshot(max_size=800)` for visible postconditions

Do not batch these into a blind script. The inspection calls are transaction
boundaries and give the agent a chance to stop after an unexpected result.

## Code argument

`execute_blender_code(code)` receives a Python source string evaluated inside
Blender. Each call has a fresh Python namespace. Only Blender datablocks and
changes to the open scene persist. Every call must therefore:

- import `bpy`, `json`, `math`, `bmesh`, or `mathutils` again as needed;
- reacquire objects, collections, materials, and scenes from `bpy.data`;
- avoid depending on local functions or variables created by an earlier call;
- avoid `time.sleep()`, long polling, interactive input, and unbounded loops;
- finish with one machine-readable JSON object on standard output.

Example response contract:

```json
{
  "ok": true,
  "operation": "set_transform",
  "object": "GEO-requested-target",
  "location": [1.0, 2.0, 3.0],
  "warnings": []
}
```

Use `ok: false` only when code catches a narrow, expected failure and returns a
useful diagnostic. Unexpected failures should raise and preserve the Blender
traceback.

## Evidence interpretation

- A successful execution response proves only that Blender accepted the code.
- `get_object_info` and `get_scene_info` prove structured authored state, not
  visual correctness.
- A viewport screenshot proves visible presentation at capture time, but may be
  stale or may show Solid, Material Preview, EEVEE, Cycles, or another engine.
- An OVRTX claim requires its own active-engine/session/product evidence. Never
  infer native OVRTX output from a viewport screenshot alone.

If a screenshot tool is unavailable, continue with structured inspection but
mark visual validation unavailable. If code execution is unavailable, stop
before mutation.
