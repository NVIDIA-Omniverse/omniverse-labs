# NanoUSD agentic Gaussian real-to-sim

This experiment is a local Apple-Metal development oracle for turning a Gaussian
scene into a registered, interactive robotics scene. It is designed for Codex to
operate through deterministic tools and evidence, and for a later Tinker RLVR
environment to consume the same action traces and hard-gate reward vector.

The central invariant is a three-way separation:

- **Measured visual truth:** the immutable source PLY and stable source-row IDs.
- **Physical truth:** registered colliders, support edges, joints, voxel occupancy,
  and collision meshes.
- **Generated priors:** hidden-interior completion candidates that are always marked
  non-measured and require explicit acceptance.

No tool silently replaces the Gaussian scene with a mesh. Every node stores its
source-row selection and collider atomically, and every operation appends to
`trace/operations.jsonl`.

## What works locally

- PLY ingest and PlayCanvas SOG/LOD ingest through `splat-transform`.
- Native NanoUSD Metal RGB, depth, normal, and stable Gaussian-ID AOVs.
- AABB or render-mask object selection with immutable source provenance.
- Support-tree inference and explicit support authoring.
- DRAWER-style prismatic drawer and revolute door fitting with confidence and
  diagnostics.
- Hidden-interior Gaussian completion candidates with candidate/accepted/rejected
  lifecycle.
- Deterministic gravity settle, push, and articulation-sweep checks.
- Full-scene or per-node voxel occupancy and collision GLB generation.
- Explicit registration of `splat-transform` GLBs back into NanoUSD PLY coordinates,
  including winding repair and a residual gate.
- USDA compilation with USD Physics bodies, collisions, joints, source selections,
  and generated-completion provenance.
- NanoUSD proxy-scene render validation.
- Dependency-free interactive HTML preview with joint sliders, front/top/side
  projections, scene graph, hard gates, completion state, and embedded Gaussian
  evidence.
- Pinned local SuperSplat experience preview that streams the immutable original
  SOG/LOD hierarchy at a configurable live budget while showing the registered
  articulation oracle alongside it.
- A single-use `RealToSimEpisode` adapter with sequential `step(action)` calls,
  public observations, trainer-only reward vectors, anti-freezing interactivity
  requirements, and fail-closed terminal rewards.
- Bounded JSON agent plans; unknown actions fail closed and arbitrary shell commands
  are not part of the plan surface.

This local verifier is intentionally labeled as conservative AABB/voxel physics. It
does not claim PhysX contact fidelity. Isaac Sim/Isaac Lab remains the promotion
lane for high-speed dynamics, robot-task success, deformables, and final simulator
parity.

## Quick start on the M5

From `projects/nanousd-labs/nanousd-metal-renderer`:

```bash
export PYTHONPATH="$PWD/experiments/agentic_real_to_sim/src"
PYTHON=../.venv/bin/python

$PYTHON -m nanousd_rts doctor
$PYTHON -m nanousd_rts demo /tmp/nanousd-rts-drawer
open /tmp/nanousd-rts-drawer/preview/index.html
```

The demo creates a cabinet with a sliding drawer and hinged door, proposes hidden
interior completions, accepts the best sweep-safe candidates, renders the Gaussian
and USDA representations, and runs all local gates.

An isolated install also works:

```bash
uv run --project experiments/agentic_real_to_sim nanousd-rts doctor
uv run --project experiments/agentic_real_to_sim \
  nanousd-rts demo /tmp/nanousd-rts-drawer
```

## Home Scan

The supplied Home Scan is a 42.3M-Gaussian, six-LOD PlayCanvas asset. Start with its
671,787-Gaussian lowest LOD:

```bash
$PYTHON -m nanousd_rts ingest \
  "$HOME/Downloads/Home Scan (Creation process in description)" \
  /tmp/nanousd-home-scan-rts \
  --lod 5 --up-axis Y

$PYTHON -m nanousd_rts render /tmp/nanousd-home-scan-rts \
  --name top-full --width 1200 --height 800 \
  --eye -3.90055 15.5 -6.12868 \
  --target -3.90055 -1.55229 -6.12868 \
  --up 0 0 -1 --fov 60

$PYTHON -m nanousd_rts add-node /tmp/nanousd-home-scan-rts \
  --id environment_scan --label "Home Scan Gaussian environment" \
  --role background \
  --bounds -17.1 -3.21 -14.66 9.30 0.10 2.41 \
  --tag measured --tag gaussian-environment

$PYTHON -m nanousd_rts voxelize /tmp/nanousd-home-scan-rts \
  --voxel-size 0.15 --opacity-threshold 0.2 --mesh-shape faces

$PYTHON -m nanousd_rts author-home-kitchen /tmp/nanousd-home-scan-rts
$PYTHON -m nanousd_rts verify /tmp/nanousd-home-scan-rts
$PYTHON -m nanousd_rts experience-preview /tmp/nanousd-home-scan-rts \
  --budget 16
$PYTHON -m nanousd_rts serve-preview /tmp/nanousd-home-scan-rts \
  --budget 16 --open
```

The background role deliberately keeps the measured scan collider-free. The
voxel command produces a separate derived collision GLB, preserving the visual
source instead of pretending the scan's full AABB is valid physics.

`author-home-kitchen` installs the deterministic Home Scan kitchen profile: 28
articulated parts spanning both refrigerator doors, the oven door, upper and base
cabinet doors, and drawers. Each part keeps its measured front as an extracted
LOD0 asset, adds a world-attached generated cavity plus joint-attached generated
backside/interior, and receives an explicit collider and sweep-verified joint.

The oven also demonstrates full-cavity background replacement. Unobserved dark
source splats inside the aperture are opacity-masked only in the streamed
background derivative, while the measured door extraction remains untouched and
the accepted generated cavity/racks stay labeled `measured=false`. This is the
local procedural analogue of amodal completion; it is not a claim that the hidden
surfaces were reconstructed from scan evidence.

The high-fidelity preview deliberately does not use `file://`: streamed
`lod-meta.json` assets and their WebP chunks require local HTTP. It pins
`@playcanvas/supersplat-viewer`, materializes the static viewer into the workspace,
and symlinks the immutable original source instead of copying or transcoding it.
The NanoUSD LOD remains the deterministic selection/physics lane; it is not used as
the final beauty renderer.

For interior scenes, add `--external-fill 1.6`; after choosing a known-free seed
point, add `--carve 1.6 0.2 --seed X Y Z`. These settings materially change
occupancy, so they remain explicit agent actions and are recorded in the trace.

Object nodes can be authored from a canonical AABB:

```bash
$PYTHON -m nanousd_rts add-node /tmp/nanousd-home-scan-rts \
  --id cabinet --label "kitchen cabinet" --role static \
  --bounds MIN_X MIN_Y MIN_Z MAX_X MAX_Y MAX_Z \
  --collision-mode shell
```

Or from a segmentation mask resolved through a stable-ID render:

```bash
$PYTHON -m nanousd_rts add-node /tmp/nanousd-home-scan-rts \
  --id drawer --label "upper drawer" --role movable \
  --id-aov /tmp/nanousd-home-scan-rts/evidence/render/top-full/id.npy \
  --mask /absolute/path/to/drawer-mask.png \
  --tag drawer
```

Then:

```bash
$PYTHON -m nanousd_rts infer-support /tmp/nanousd-home-scan-rts
$PYTHON -m nanousd_rts fit-joint /tmp/nanousd-home-scan-rts --node drawer --kind auto
$PYTHON -m nanousd_rts propose-completions /tmp/nanousd-home-scan-rts --node drawer
$PYTHON -m nanousd_rts completions /tmp/nanousd-home-scan-rts
$PYTHON -m nanousd_rts accept-completion /tmp/nanousd-home-scan-rts \
  --completion drawer.interior.01
$PYTHON -m nanousd_rts sweep /tmp/nanousd-home-scan-rts --node drawer
$PYTHON -m nanousd_rts compile /tmp/nanousd-home-scan-rts
$PYTHON -m nanousd_rts render-usda /tmp/nanousd-home-scan-rts
$PYTHON -m nanousd_rts verify /tmp/nanousd-home-scan-rts
$PYTHON -m nanousd_rts preview /tmp/nanousd-home-scan-rts --open
```

## Agent action contract

`nanousd-rts tools` prints the machine-readable catalog. `run-plan` accepts only
those actions:

```bash
$PYTHON -m nanousd_rts run-plan /tmp/workspace examples/drawer-plan.json
```

The expected loop is:

1. Ingest and render AOV evidence.
2. Select measured source rows into semantic nodes.
3. Bind and visually inspect physical proxies.
4. Infer or author support.
5. Fit joints; override uncertain axes, origins, or limits explicitly.
6. Propose hidden completions; never relabel generated content as measured.
7. Sweep, settle, push, and voxelize.
8. Compile USDA and render the proxy scene through NanoUSD.
9. Run `verify`; do not promote a scene with a failed hard gate.
10. Inspect the interactive preview and package the evidence.

## Workspace contract

```text
workspace/
  scene.json
  source/source.ply
  selections/<node>.npy
  generated/completions/<node>/*.ply
  evidence/render/<name>/{rgb,depth,normal,id,...}
  evidence/sweeps/<node>/sweep.json
  evidence/verification/report.json
  exports/scene.usda
  exports/scene.manifest.json
  exports/voxel/<scope>/*
  preview/index.html
  preview/physics.html
  preview/visual/{index.html,index.css,index.js,settings.json,content}
  trace/operations.jsonl
  trace/plan-result.json
```

## RLVR/Tinker bridge

The JSON plan is the policy action sequence, `operations.jsonl` is the tool trace,
and `evidence/verification/report.json` is the deterministic reward input. A
multi-turn Tinker environment can expose the catalog as tools, feed compact AOV
summaries and failed gates back to the policy, and score:

- source/selection provenance;
- visual-collider registration;
- support-graph validity;
- joint sweep and forbidden-overlap gates;
- USDA load/render success;
- completion provenance and accepted-candidate linkage;
- task-specific robot evaluation from the external simulator lane.

Use paired fidelity and stability scores. Never reward stability alone, because an
agent can otherwise make the entire scene static.

The dependency-free episode seam can be imported directly by a cookbook
`MessageEnv` or tool environment:

```python
from nanousd_rts import EpisodeRequirements, RealToSimEpisode, Workspace

episode = RealToSimEpisode(
    Workspace.open("/tmp/nanousd-rts-task"),
    EpisodeRequirements(
        required_nodes=("cabinet", "drawer"),
        required_interactive_nodes=("drawer",),
    ),
)
initial = episode.public_observation()
turn = episode.step({"tool": "render", "name": "policy-view"})
```

Attach each turn's `trainer_reward.dense_score` to Kevin-style discounted future
credit. Use `terminal_reward` for submission; it is zero unless all local hard
gates, hidden semantic/interactivity requirements, and required artifacts pass.

## Verification

```bash
PYTHONPATH=experiments/agentic_real_to_sim/src \
  ../.venv/bin/python -m unittest discover \
  -s experiments/agentic_real_to_sim/tests -v

cmake --build build --target nusd_renderer test_gaussian_render --parallel
./build/test_gaussian_render
```

The repo-local operating procedure is
[`nanousd-real-to-sim`](../../../agentic-skills/nanousd-real-to-sim/SKILL.md).
