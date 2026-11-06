# Public SDK physics spike

The public mode runs OVPhysX and OVStage in an external Python worker. Blender
retains authoring, conversion, presentation, baking, and the existing capability
checks. Production physics calls use the public SDKs without the SIL client or
server. This document describes the measured Linux spike, not a platform release
or a legal assessment of the process boundary.

## Coverage and admission

The acceptance runner exercises the existing Blender fixtures against the public
worker. It runs each family on CPU and GPU, preserving the fixtures' numerical
assertions and structural restrictions.

| Family | Covered behavior |
| --- | --- |
| Body controls | Sphere, box, capsule and convex hull behavior; initial linear/angular velocity; damping; mass and density; center of mass, diagonal inertia and principal axes; friction and restitution; gravity; meter/centimeter equivalence. |
| Joints | Fixed, revolute, prismatic and D6 joints; admitted limits and force drives; enabled/disabled behavior and world/body endpoints within the existing device restrictions. |
| Kinematic input | Evaluated object and bone targets before stepping; a stationary control and a moving contact case; exact target pose readback; kinematics excluded from dynamic output. |
| Parented poses | World-space physics output applied correctly to parented Blender owners. |
| Bone mechanisms | Object/bone endpoint conversion, articulated motion, unit scaling and presentation. CPU includes rotated and body-to-body cases; GPU uses the existing aligned, world-connected subset. |
| Contact baking | Object and bone simulation baked to Blender animation, then reopened and checked in a fresh Blender process with public SDK selection removed. |

Acceleration joint drives remain unadmitted. Rotated or body-to-body joint frames
that the baseline admits only on CPU remain CPU-only. A successful SDK call does
not widen these limits.

Admission is a separate check from transport availability. The manifest
`public/addon/ovrtx_blender_example/public_sdk_physics_admission.json` binds CPU/GPU
and capability families to exact OVPhysX/OVStage versions, native library hashes,
platform/architecture, and the `PhysicsBackend` class AST hash. Missing or changed
identity fails closed. Renderer-only implementation changes do not alter the
physics class hash. Existing SIL binary-pair admission remains intact for the
legacy mode.

Physics workers do not hash files when they start. Native library hashes come
from the integrity receipt written when the runtime was installed
(`public-sdk-integrity.json`); the worker only checks that the native files on
disk are exactly the recorded set. The adapter identity is recorded in
`runtime/physics_adapter_identity.json`. After changing any adapter component,
regenerate it with `python runtime/physics_adapter_identity.py --write` and
requalify physics. Package builds and the unit suite reject a stale record.

The baseline has twelve acceptance jobs: six families on two devices. Live mass
edits require a distinct `live_mass` admission with two additional Blender
generation-reuse probes and native inertia/dynamics comparisons. Each recorded
result is tied to its fixture and result hashes. Candidate measurements run first;
the entire suite then runs again through production admission without the
measurement option.

## Which edits keep the current simulation?

| Change | Behavior |
| --- | --- |
| Visual-only scene generation with unchanged physics dependencies, configuration and initial conditions | Keep the controller, worker process, native OVStage, native PhysX instance, simulation time, poses and velocities. |
| Only positive authored mass changes on existing dynamic bodies | Keep those same identities and state; update authored mass plus effective native mass/inertia, then verify readback. Requires separate live-mass admission. |
| Other physics settings, body roster or collider topology changes | Use the existing validated simulation replacement path. |
| Explicit simulation reset | Always reset, even when the scene fingerprint is unchanged. |
| Uncertain mutation outcome | Stop physics and require an explicit reset. Do not silently replay steps or recreate initial state. |

The reuse fingerprint includes physics schemas/attributes, collider descendants,
relationship dependencies, ancestor data and scene units. It is conservative:
unsupported instancing, asset-valued physics data, time-varying data, or a failed
inspection disables reuse and leaves the normal replacement path available.
Legacy SIL activations do not perform this additional inspection.

The mass route admits only a complete mass-only delta on the existing dynamic
roster. It validates the whole batch before changing anything. OVStage mass edits
alone do not update effective actor mass in this SDK, so the route also writes the
public native mass column. Likewise, native mass writes alone leave automatic
inertia unchanged: automatic inertia is scaled by the mass ratio, while explicitly
authored inertia is preserved. Native mass/inertia readback and exact unchanged
position, orientation and both velocity columns are checked before acknowledgment.
The probe compares live edits with fresh mass-two simulations under a driven joint,
covering both automatic and explicit inertia rather than checking property writes
alone.

The controller drains its asynchronous physics producer before a live mass edit
and keeps its step/time position. A native error, partial acknowledgment, or lost
reply makes the outcome uncertain and prevents continuation. Failed mass edits
cannot trigger an implicit generation rollback that would silently reset physics.

Direct SDK experiments also demonstrated adding/removing referenced bodies while
retaining native objects. This is exploratory evidence only: arbitrary live body
topology edits are not integrated or admitted. Reopening an entire USD source into
an existing Stage reset dynamic state in the experiment, so preserving a Stage
object alone is insufficient evidence of preserving simulation state.

## Implementation details that affect parity

- Public body poses are world-space. The Blender adapter avoids applying the
  legacy parent-local conversion twice, including initial writes and bone owners.
- Kinematic targets go through OVStage before stepping; writing a displayed pose
  alone does not establish the required contact motion.
- Velocity writes accept the existing client payload and preserve angular unit
  conversion. Native velocity writes avoid kinematic actors.
- CPU initialization omits the SDK warmup step: measured rotated joint fixtures
  otherwise drifted before the first requested step. GPU initialization retains
  its required setup.
- Physics errors never use renderer-style recovery. Dynamic simulation state is
  not reconstructible from the initial scene and acknowledged authoring edits.

## Reproduce the developer check

Run from this add-on checkout. Keep the SDK interpreter symlink intact; resolving
it to the base interpreter can bypass its virtual environment. Select a free GPU
and run the native acceptance sequentially to limit memory use.

```bash
export OV_BLENDER_PUBLIC_SDK_PYTHON="/path/to/public-sdk/bin/python"
export BLENDER_COMMAND="/path/to/blender-5.2/blender"
export CUDA_VISIBLE_DEVICES=3
python3 public/scripts/validate_public_sdk_physics.py \
  --blender "$BLENDER_COMMAND" \
  --output out/physics-production
```

Run the generation preservation and live-mass check once per device:

```bash
"$BLENDER_COMMAND" \
  --background --factory-startup --python-exit-code 17 \
  --python public/tests/blender_public_physics_generation_reuse.py -- \
  --device cpu --output out/physics-mass-cpu
"$BLENDER_COMMAND" \
  --background --factory-startup --python-exit-code 17 \
  --python public/tests/blender_public_physics_generation_reuse.py -- \
  --device gpu --output out/physics-mass-gpu
```

When intentionally qualifying a changed SDK/backend identity, first run these
three commands into fresh measurement directories with `--measure-unadmitted`.
That option belongs to the test fixtures; it does not disable production checks.
Record only successful complete measurements:

```bash
python3 public/scripts/record_public_sdk_physics_admission.py \
  --evidence-root out/physics-measure \
  --live-mass-evidence out/physics-measure-mass-cpu out/physics-measure-mass-gpu \
  --destination public/addon/ovrtx_blender_example/public_sdk_physics_admission.json
```

Then rerun all production commands without `--measure-unadmitted`. The recorder
rejects incomplete, stale or mixed-identity evidence; do not edit identity hashes
by hand to admit an unmeasured runtime.

The fixtures now take `--qualification-mode measurement|candidate|production`
in place of `--measure-unadmitted`, which they reject. The current Windows
admission (below) was qualified one bounded cell at a time with the integration
repository's `scripts/run_physics_qualification_cell.py`. Body controls ran as
six case shards per device, combined by `scripts/aggregate_public_sdk_body_controls.py`.
Each bake-contact cell got a separate `--offline-only` replay. The recorder then
promoted the measured identity from the complete candidate rerun
(`--candidate-evidence-root`, `--candidate-live-mass-evidence`).

## Evidence for this machine

### Windows, current adapter (current admission)

Host: Windows 11 x64, NVIDIA RTX A6000, driver 597.16, the project's Blender 5.3.0
Alpha build. The locked runtime is unchanged from the earlier Windows row below:
lock SHA-256 `d3355c7b…a10ee`, 297 native files (native SHA-256 `86f48728…b40e4`),
CPython 3.13.13 with executable SHA-256 `b51e9a42…5af0b`. The fixtures ran with the
installed sidecar's interpreter. Only the physics adapter changed: the worker and
backend now time the exit-time unload and the worker's exit after its socket closes.
The adapter identity is now
`a1b21fe1b26ff9816b8a12d34a335c53e5c6e43de010d275bf4fc54272e90d48`, and the
admission identity SHA-256 is `423105af…2093e`.

Status: **PASS** on this host. All 14 cells passed measurement and then the
normal-admission candidate rerun for that one identity: the six families plus
live mass, each on CPU and GPU. The recorder appended one `win32` production row,
and the earlier rows are unchanged. A shipped-default production rerun of all 14
cells, with no local manifest, also passed. Linux qualification for this adapter
is recorded separately below; admission never transfers across platforms.

### Windows, upgraded runtime (earlier adapter)

Host: Windows 11 x64, NVIDIA RTX A6000, driver 597.16, stock Blender 5.2.1.
The locked runtime has OVPhysX `0.7.0.74035321`, OVStage `0.3.0.1791986420`,
OVRTX `0.6.0.1791947691`, OVStream `0.5.7`, physx-usd-schemas `25.11.2.72696984`,
warp-lang `1.18.0`, NumPy `2.5.3` and packaging `23.2`. The interpreter is CPython
3.13.13, executable SHA-256 `b51e9a42…5af0b`. The lock SHA-256 is `d3355c7b…a10ee`,
with 297 native files (native SHA-256 `86f48728…b40e4`). The physics adapter
identity is `f776abde1e397d497de56b2f019bd89825bd425c0531efab0253b54b96bb5891`,
and the admission identity SHA-256 is `d22d2903…66e8f`.

Status: **PASS** on this host. All 14 cells passed measurement and then the
normal-admission candidate rerun for that one identity: the six families plus
live mass, each on CPU and GPU. The recorder appended one `win32` production
row; earlier rows are unchanged. A shipped-default production spot check of
joints and kinematic on both devices also passed. The first measurement round,
with adapter `cde5c280`, failed the parented fixture on both devices. OVPhysX 0.7
composes an edited `omni:xform` with its ancestors, and the adapter had written
world poses there. The adapter now writes the parent-local matrix, and the
identity was measured again from fresh directories.

### Linux, current adapter

Host: Linux x86-64, NVIDIA L40, driver 595.71.05, stock Blender 5.2.1 LTS.
The unchanged locked OVPhysX `0.7.0.74035321` / OVStage `0.3.0.1791986420`
runtime has lock SHA-256 `d3355c7b…a10ee`, 410 native files
(native SHA-256 `fa66fb44…d1082`) and CPython executable SHA-256
`940551e8…30043`. The current adapter is `a1b21fe1…e90d48`;
its Linux admission identity SHA-256 is `b2cc8f7f…a1b43a4`.

The previous Linux 0.7 production row named adapter `f776abde…5891`, not the
current adapter. Exact identity admission therefore correctly refused physics;
this was not a missing 0.7 native binary hash. Fresh measurement and candidate
qualification passed all 14 cells (six families plus live mass on CPU/GPU).
Body controls used six bounded shards per device with full cross-case
assertions, and contact bakes used independent offline replays. The recorder
appended the new Linux production row without altering earlier measurements.
Evidence is retained in the integration repository under
`_build/linux-physics-fix/{measurement,candidate,production}`. Changed or mixed
identities remain refused; the existing CPU/GPU joint restrictions are unchanged.

### Linux, upgraded runtime (earlier adapter)

Host: Linux x86-64 (Ubuntu, kernel 5.15), NVIDIA L40, driver 595.71.05, stock
Blender 5.2.1. The locked runtime is the same lock as on Windows (`d3355c7b…a10ee`)
with its Linux wheels: 410 native files (native SHA-256 `fa66fb44…d1082`). The
interpreter is Blender's CPython 3.13.13, executable SHA-256 `940551e8…30043`.
The physics adapter identity is the same `f776abde…5891`, and the admission
identity SHA-256 is `a0dcba61…18a4c3`.

Status: **PASS** on this host. All 14 cells passed measurement and then the
normal-admission candidate rerun. The recorder appended one `linux` production
row, and the earlier rows are unchanged. The recorder was rerun on Windows
against the copied Linux evidence, and it produced the same file. Two fixes to the
qualification tooling were needed on Linux, and neither changed the adapter.
First, the cell runner had resolved the SDK venv's `bin/python` symlink to the
base interpreter. Second, the recorder compared executed-source snapshots with
host-specific path separators.

### Linux, earlier runtime (historical)

Host: Linux x86-64, NVIDIA L40, driver 570.158.01, Blender 5.2.0. Physics uses
`ovphysx==0.6.3` and `ovstage==0.2.0.377349` in the external environment shared with
`ovrtx==0.5.0.377615` and `ovstream==0.4.5`.

Frozen physics class AST SHA-256 as computed by the SDK interpreter and recorded
in admission:
`a0da74ac2896dd4b836050c56e9dd68654a6c19f6f68ecba59ab460b41f99d74`.
AST serialization can differ between Python versions; compare identities using
the selected SDK interpreter.

Final validation status: **PASS**. All twelve baseline jobs and both live-mass
probes passed in candidate measurement and again through production admission.
The final evidence root in the integration
checkout is `out/public-sdk-coverage/private/`; `physics-final.log` records the
measurement, admission and production sequence. Baseline reports are under
`physics-final-measure/` and `physics-final-production/`. The four corresponding
`physics-final-{measure,production}-mass-{cpu,gpu}/` directories contain generation
preservation results and nested `inertia-parity/results.json` reports. The final
success marker is `FINAL_PHYSICS_ALL_PASS`.

The production reuse probes preserved twenty completed steps and simulation time
333,333,320 ns exactly across a visual edit, then preserved the advancing state
and native Stage/PhysX identities across a mass edit. Both automatic and explicit
inertia cases matched freshly authored simulations within the probe's numerical
tolerances on CPU and GPU. These checks establish the stated bounded physics
coverage; they do not claim arbitrary live topology changes or platform coverage
beyond this host.
