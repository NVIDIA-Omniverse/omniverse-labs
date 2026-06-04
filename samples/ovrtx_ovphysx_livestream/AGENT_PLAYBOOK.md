# Agent playbook: building ovrtx + ovphysx viewer examples

This document captures the **prompts, process, and architecture** used to create the livestream examples under:

- `examples/python/ov-libaries-livestream/` — standalone ovrtx / ovphysx tutorials
- `examples/python/usd-viewer-example/` — PySide6 desktop viewer that runs those tutorials

Use it to reproduce the workflow with Cursor (or another agent), the repository **skills** (`skills/*/SKILLS.md`), and optional MCP tooling.

---

## What was built

Two sibling Python projects that share USD assets and a **version-1 JSON pose wire format**:

| Layer | Role | Key APIs |
|-------|------|----------|
| **Viewer (parent process)** | Load USD in ovrtx, render to Qt, apply transforms | `Renderer`, `map_attribute`, `clone_usd` |
| **PhysX workers (child processes)** | Simulate in ovphysx only, stream poses | `PhysX`, `create_tensor_binding`, `TensorType.*` |
| **Shared contract** | Pose interchange | `physx_pose_utils.py` → JSON lines on stdout |

**Critical constraint discovered early:** do **not** call `PhysX()` in the same process as `Renderer()` after USD has loaded. Carbonite plugin clashes (`IUsdPhysics`, `IPhysx`, “PhysX plugins could not be loaded”) make co-loading unreliable. The working pattern is:

```
Viewer (ovrtx only)  ←── JSONL stdout ───  Child process (ovphysx only)
       │                                              │
       └── map_attribute(omni:xform) each frame       └── tensor read → JSON
       └── renderer.step()                            └── physx.step() loop
```

This mirrors [planet-system](../planet-system/) (Warp writes transforms → step renderer), but PhysX runs out-of-process instead of in-process Warp.

---

## Agent prerequisites

Before prompting an agent, point it at:

1. **[`AGENTS.md`](../../AGENTS.md)** — repo layout and skill index
2. **[`skills/application-flow/SKILLS.md`](../../skills/application-flow/SKILLS.md)** — renderer lifecycle
3. **Reference examples**
   - [`minimal/`](../minimal/) — smallest ovrtx render loop
   - [`planet-system/`](../planet-system/) — transform update → `renderer.step()` loop
   - [`usd-viewer-example/`](usd-viewer-example/) — final integrated viewer
4. **Branch context** — see root [`README.md`](../../README.md) section *Using this branch (`user/agoldstein/livestream`)*

Tell the agent to **read surrounding code before editing**, match licensing headers, and keep changes scoped.

---

## Recommended build order (phases)

Build incrementally. Each phase should run standalone before wiring the viewer.

### Phase 1 — ovrtx-only samples (`ov-libaries-livestream`)

| Step | Goal | Skill / reference |
|------|------|-------------------|
| 1a | Single-frame PNG from remote USD | `minimal/main.py`, `renderer-creation`, `reading-render-output` |
| 1b | Depth / sensor AOV from local USDA | `depth_map_example.py`, `robot_with_depth.usda`, `STAGE_SETUP.md` |
| 1c | Clone prims in ovrtx | `clone_example.py`, `skills/cloning-prims` |
| 1d | Standalone ovphysx smoke test | `hello_world_physx.py`, `links_chain_sample.usda` |

**USD rule:** every stage the viewer must open needs a valid **RenderProduct** with **`HdrColor` and/or `LdrColor`**. See [`usd-viewer-example/STAGE_SETUP.md`](usd-viewer-example/STAGE_SETUP.md).

### Phase 2 — PhysX subprocess + pose apply

| Step | Goal | Files |
|------|------|-------|
| 2a | Batch sim → JSON file | `physx_subprocess_sim.py` |
| 2b | Shared pose math | `physx_pose_utils.py` (world 4×4, quaternion layout) |
| 2c | Viewer applies JSON via `map_attribute` | `main.py` → `_apply_physx_pose_json` |
| 2d | Live stream (JSONL on stdout) | `physx_live_worker.py`, `physx_rigid_live_worker.py` |

Use **`PhysX(device="cpu")`** in workers so GPU stays available to ovrtx in the parent.

### Phase 3 — Wire tutorials into the viewer UI

| Menu / workflow | Opens USDA | Runs script |
|-----------------|------------|-------------|
| ovrtx → Clone | Robot default or `basic_simulation.usda` | `clone_example.py` (ovrtx `clone_usd`) |
| ovrtx → Depth map | `robot_with_depth.usda` | `depth_map_example.py` |
| ovphysx → Articulation | `links_chain_sample.usda` | `hello_world_physx.py` or live worker |
| ovphysx → Tensor Binding | `links_chain_sample.usda` | `tensor_bindings.py` (`--viewer-stream`) |
| ovphysx → Contact Binding | `boxes_falling_on_groundplane.usda` | `contact_binding.py` (logs) or rigid live worker (motion) |
| ovphysx → Clone | `basic_simulation.usda` | `clone.py` (`--viewer-stream`; PhysX `clone` + rigid poses) |

Pattern: **menu selects scene → Play spawns `QProcess` with `sys.executable`** → parse stdout → apply transforms → existing `_on_frame()` render loop.

### Phase 4 — Polish for presentation

- Inline comments in `main.py`, `clone.py`, `tensor_bindings.py`, etc. (“for viewers / live walkthroughs”)
- Log panel under viewport (avoid modal dialogs covering the stage)
- Slow down clone stream (`sleep(4 * dt)`) so motion is visible
- Root README run instructions for the branch

---

## Representative prompts (from actual development)

These are paraphrased from the agent sessions that produced this work. Adapt paths and names for your repo.

### Bootstrap the viewer

> Create a desktop USD viewer using ovrtx as the render backend. Use PySide6. Render offscreen, display `HdrColor` first then `LdrColor`. Follow `examples/python/minimal` for the step/map loop but run continuously on a timer.

> Add a monospace panel showing the root layer USDA text to the left of the viewport.

> When the window resizes, explain how ovrtx render resolution follows the viewport (and implement if missing).

### ovrtx tutorials

> Make `clone_example.py` as simple as possible following `skills/cloning-prims/SKILLS.md`. Clone the robot prim so we can view it in the USD viewer via a menu item.

> Implement the depth map example and make `robot_with_depth.usda` viewable in the viewer. Add comments so livestream viewers understand each section.

> Add a dome light to `basic_simulation.usda` and ensure it loads in the USD viewer.

### ovphysx integration (core architecture)

> Review `planet-system` for how ovrtx and simulation work together (Warp updates transforms, then steps the renderer). Implement the **same principle with ovphysx** — run `hello_world_physx.py` so ovrtx displays the PhysX simulation. **Do not put Warp in the viewer.**

> Open USD viewer, then a UI button runs simulation in a **separate process** so Carbonite plugins load independently.

> Extend the worker to **stream transforms** (JSON lines on stdout) instead of only writing a file. The main process applies poses with `map_attribute` each frame, then calls `renderer.step()` — same order as planet-system.

> Add live PhysX for rigid bodies (`boxes_falling_on_groundplane.usda`) so we **see cubes fall**, not just contact-force logs from `contact_binding.py`.

### Wire scripts to UI

> Change the Rendering tab to **ovrtx**. When the user selects Clone, open the robot scene and show a **Play** button that runs the clone script.

> Do the same for Depth map: open `robot_with_depth.usda`, Play runs the depth script.

> Turn Simulation into **ovphysx** with Articulation / Tensor Binding / Contact Binding / Clone — each loads the right USDA; Play runs the matching script.

> Put a note next to Play: “select an ovphysx scene to play the simulation.”

### Debugging prompts (when things break)

> `hello_world_physx.py` doesn't work when I press Play after ovphysx → Articulation. Review the old **Simulation → Live PhysX (articulation stream)** implementation and use the same principle with the Play button.

> Contact binding opens the USD but Play doesn't run the script. Review when **Simulation → Contact Binding** worked and match that behavior.

> I don't see prims move in the viewport. Investigate the ovrtx apply path — subprocess logs look fine but transforms aren't visible.

> Stage is black after loading `links_chain_sample.usda`. Inspect the USDA render setup (RenderProduct, lights, HdrColor vs LdrColor).

> Auto-detect picks `/Render/Camera` but this Kit-only stage only has a HydraTextures product. Fix probe order for local `.usda` files.

---

## Architecture checklist for new examples

When adding a tutorial, verify:

- [ ] **Process split:** ovphysx code runs in a child process (`QProcess` + `sys.executable`), not in the viewer process
- [ ] **CPU PhysX:** `PhysX(device="cpu")` in workers when parent uses GPU for ovrtx
- [ ] **Local USD:** PhysX workers require a filesystem path, not `https://` URLs
- [ ] **Render product:** stage has OVRTX-valid RenderProduct; viewer auto-detect or `-r` override works
- [ ] **Transform write path:** use `map_attribute` on `omni:xform` / `Semantic.XFORM_MAT4x4` (not deprecated `omni:fabric:worldMatrix`)
- [ ] **Frame order:** apply transforms → `renderer.step()` → map color buffer → display
- [ ] **Stdout contract:** diagnostics on **stderr**; one JSON object per line on **stdout** for live streams
- [ ] **Version-1 JSON:** reuse `physx_pose_utils.version1_*_payload` for consistency
- [ ] **Sibling paths:** scripts resolve `../ov-libaries-livestream/` and `../usd-viewer-example/` — keep folder layout
- [ ] **Reload:** “Reload scene” clears in-memory transform overrides without requiring disk USD changes

---

## JSON pose contract (version 1)

Shared between workers and viewer:

```json
{
  "version": 1,
  "articulation_world_matrix4d": [[...4x4 identity or root world...]],
  "prims": [
    {"path": "/World/articulation/link0", "matrix4d": [[...4x4 row-vector...]]}
  ]
}
```

- Matrices are **world** poses from PhysX (position + imaginary-first quaternion → row-vector 4×4)
- Viewer converts to **local** transforms before `map_attribute`
- Articulation: `TensorType.ARTICULATION_LINK_POSE` + `body_names`
- Rigid bodies: `TensorType.RIGID_BODY_POSE` + explicit prim path list

Implement once in `physx_pose_utils.py`; import from worker scripts via dynamic `sys.path` insert (avoids circular package deps).

---

## Skills map for common tasks

| Task | Skill |
|------|-------|
| Create renderer, first step | `skills/renderer-creation` |
| Load USD / layers | `skills/loading-usd` |
| Render loop | `skills/stepping-and-rendering` |
| Read camera / depth | `skills/reading-render-output` |
| Apply simulation transforms | `skills/writing-transforms`, `skills/mapping-attributes` |
| Clone environments | `skills/cloning-prims` |
| App lifecycle ordering | `skills/application-flow` |
| Python project setup (`pyproject.toml`, uv) | `skills/project-setup-python` |

**Prompt pattern:** attach or @-mention the relevant skill when asking the agent to implement a feature.

---

## MCP + skills narrative (for livestreams / docs)

The repository ships **task skills** (`skills/`) that encode ovrtx conventions. Cursor **MCP** servers (filesystem, terminal, etc.) let an agent read those skills, run `uv run`, and iterate on USD/scripts in one session.

Suggested demo flow for an audience of OpenUSD / robotics developers:

1. Show `AGENTS.md` + one skill (e.g. `application-flow`)
2. Agent reads `minimal` → implements a one-file render script
3. Extend to viewer (`usd-viewer-example/main.py` structure)
4. Agent reads `planet-system` → explains transform-then-step loop
5. Agent implements PhysX subprocess + JSONL stream (this playbook, Phase 2)
6. Run `uv run main.py`, menu → Play, show code + viewport together

---

## Template: start a similar project from scratch

Copy-paste starter prompt for a new agent session:

```
You are working in the ovrtx repository. Read AGENTS.md and skills/application-flow/SKILLS.md first.

Goal: Add [DESCRIPTION] that combines ovrtx rendering with ovphysx simulation.

Constraints:
- ovrtx stays in the main process; ovphysx runs in a subprocess (QProcess + sys.executable).
- PhysX uses device="cpu"; parent renderer may use GPU.
- Apply poses with map_attribute (omni:xform) before renderer.step(), like planet-system.
- Reuse physx_pose_utils version-1 JSON for any pose streaming.
- Ship a local .usda with valid RenderProduct (HdrColor/LdrColor) — see usd-viewer-example/STAGE_SETUP.md.
- Match existing example style: pyproject.toml with uv, NVIDIA license header, argparse CLI.

Reference:
- examples/python/minimal (render)
- examples/python/planet-system (transform loop)
- examples/python/usd-viewer-example (viewer + subprocess)
- examples/python/AGENT_EXAMPLE_PLAYBOOK.md (this file)

Deliverables:
1. Standalone script under examples/python/ov-libaries-livestream/
2. Viewer menu + Play wiring in usd-viewer-example/main.py
3. README snippet with uv run commands
4. Comments suitable for a live code walkthrough
```

---

## Common pitfalls (learned during development)

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Black viewport | Stepping wrong render product; missing `HdrColor` | Prefer `HdrColor`; probe Kit paths for local USDA; see README troubleshooting |
| PhysX plugins failed | ovphysx + ovrtx same process | Subprocess worker only |
| GPU PhysX warning, no motion | CUDA contention with ovrtx | `PhysX(device="cpu")` |
| Contact binding “success” but no motion | Script only prints forces | Use `physx_rigid_live_worker.py` for falling boxes |
| Clone too fast to see | Worker loop uncapped | Sleep `4 * dt` between JSON lines |
| `Invalid render product /Render/Camera` | Kit stage without Camera product | Auto-detect Kit path first |
| `ModuleNotFoundError: pxr` in worker logs | Normal in ovphysx-only venv | Ignore if simulation output is correct |
| Transforms not visible | Wrong attribute semantic or world vs local | Use `_apply_physx_pose_json` pattern; reload scene to reset |

---

## File map

```
examples/python/
├── AGENT_EXAMPLE_PLAYBOOK.md          ← this file
├── ov-libaries-livestream/
│   ├── pyproject.toml                 # ovrtx + ovphysx deps
│   ├── example.py                     # minimal ovrtx PNG
│   ├── depth_map_example.py           # DepthSensorDistance AOV
│   ├── clone_example.py               # ovrtx clone_usd (robot grid)
│   ├── hello_world_physx.py           # minimal PhysX step
│   ├── tensor_bindings.py             # DOF targets + link poses; --viewer-stream
│   ├── contact_binding.py             # contact forces (log only)
│   ├── clone.py                       # PhysX clone + rigid stream
│   └── *.usda                         # stages tuned for viewer + PhysX
└── usd-viewer-example/
    ├── pyproject.toml                 # + PySide6
    ├── main.py                        # Qt viewer, menus, QProcess workers
    ├── physx_pose_utils.py            # JSON version 1 builders
    ├── physx_subprocess_sim.py        # batch JSON export
    ├── physx_live_worker.py           # articulation JSONL stream
    ├── physx_rigid_live_worker.py     # rigid body JSONL stream
    ├── contact_binding_subprocess.py  # runpy wrapper into ov-libaries-livestream
    ├── README.md                      # run instructions + troubleshooting
    └── STAGE_SETUP.md                 # USD render graph checklist
```

---

## Packaging for another repo

```bash
python tools/package_livestream_export.py
```

Produces `dist/livestream-export.zip` with both folders and install README. Keep them as **siblings** under `examples/python/` in the target repo.

---

## Making a *better* example than this one

Ideas the original build deferred:

- **IPC instead of stdout JSONL** — lower latency, typed schema (gRPC, shared memory)
- **Single pyproject** — optional meta-package depending on both example trees
- **Automated smoke tests** — subprocess workers exit 0 + JSON schema validation in CI
- **Signed / public PyPI** — remove Artifactory-only blocks in `pyproject.toml`
- **In-process PhysX** — only viable inside full Kit/Isaac with correct extension load order
- **Dense depth** — tune USDA sensor params before aggressive hole-fill in `depth_map_example.py`

When extending, add or update a skill under `skills/` if you introduce a repeatable pattern.
