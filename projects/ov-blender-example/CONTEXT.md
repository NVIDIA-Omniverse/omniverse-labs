# ovrtx Blender Example

Terms used by the ovrtx Blender example when describing render correctness,
navigation performance, and diagnostic artifacts.

## Language

**User**:
A person who installs a published add-on zip in Blender, uses the installed
add-on's runtime installer, renders the open Blender scene with the add-on,
and reads add-on runtime/preflight status.
_Avoid_: Source user, release validator, developer install

**Task validation**:
The fixed composition of unit, small golden image, OVRTX integration, Blender
integration, and small performance suites against one prepared add-on and
materialized runtime.
_Avoid_: CI trigger, comparative validation, nightly validation

**Source add-on test**:
A developer-selected direct test or semantic suite that imports the add-on from
the source checkout. It does not require packaging or installation.
_Avoid_: Package validation, complete CI inventory, development release test

**Installed-package test**:
A canonical test or semantic suite run from a disposable test tree containing
the exact packaged add-on and its prepared runtime. It shares test code and
suite definitions with source add-on tests and does not modify the checkout.
_Avoid_: Checkout substitution, package-only test, rebuilt add-on test

**Extended validation**:
The fixed composition of large golden image and large performance suites
against one materialized runtime.
_Avoid_: Nightly validation, release validation, product gate

**Software frame service latency**:
The elapsed monotonic time from immediately before one production viewport
render invocation until Blender's matching `POST_PIXEL` callback observes its
newly drawn publication. It includes render service, readback, publication,
redraw dispatch, texture upload, and viewport draw submission. It does not
claim GPU completion, buffer swap, compositor presentation, scan-out, or pixel
response. Fixed-window observation count and chronology may derive visible-frame
throughput and pacing without becoming additional captured measurements.
_Avoid_: End-to-end display latency, input latency, render time, physical frame latency

**Blender scene fixture**:
A pinned `.blend` test input opened as Blender scene data and materialized
through production scene generation. Generated USD is internal transport state,
not the fixture's durable source authority.
_Avoid_: Prepared USD fixture, exact-stage fixture, user scene file

**Golden image**:
The operator-approved fixture-specific image used as an absolute visual
reference by task validation. Capture environment is recorded as provenance
but does not select the validation environment.
_Avoid_: Acceptance frame, screenshot, baseline image, performance reference

**Golden record**:
One tracked golden image plus its case, capture, presentation, digest, and
approval metadata. It does not retain a raw runtime result bundle.
_Avoid_: Visual golden, nonblank visual check, smoke image check

**Render-flow parity**:
The requirement that viewport and final render present the same current scene
generation and interpret equivalent render inputs identically. The canonical
scene-camera intent uses one golden correctness reference; navigation compares
matching changed-camera poses across adapters. Callback, presentation, and
runtime lifetimes may differ without defining separate scene-generation or
input-selection policies.
_Avoid_: Per-flow golden, fixture-path parity, identical intermediate frames

**Demo-quality static render**:
An operator-accepted single OVRTX image that looks intentional enough for a
demo: readable subject, clear framing, useful lighting, and no obvious runtime
artifact. It is a presentation asset, not validation evidence.
_Avoid_: Pretty render, beauty render, final golden image

**Navigation responsiveness**:
The user-visible quality of an OVRTX viewport during camera movement: presented
views remain current with the latest Blender navigation input and update with
fluid cadence, then the final view refines promptly after movement stops.
_Avoid_: Maximum FPS, orbit throughput, renderer speed

**Latest-view navigation**:
The camera-motion policy where obsolete pending camera values and render
results may be superseded so the newest camera value controls the next
presented snapshot. After movement stops, the final camera value must render
and refine completely.
_Avoid_: Every-camera-value rendering, FIFO camera replay, async latest pose

**SOL performance**:
The theoretical performance bound for a fixed workload on specified hardware,
derived from required work and physical resource limits under ideal execution.
_Avoid_: Empirical SOL, measured ceiling, current performance, target

**Measured performance**:
An observed performance result for a specified workload, hardware environment,
and measurement scope. Component and end-to-end results must state their scope.
_Avoid_: Achieved performance, empirical SOL, measured ceiling

**SOL resource model**:
An analytical decomposition of SOL performance into required work and physical
limits for Blender dispatch, CPU, GPU, host/device transfer, serialization,
transport, viewport presentation, and stage overlap. The end-to-end theoretical
bound is derived from these explicit terms.
_Avoid_: Single SOL number, profiler report, measured performance ledger

**Blender signal**:
An input crossing from Blender into the add-on before it has been translated
into an add-on-owned payload. Render and viewport callbacks, scene/context
state, depsgraph updates, operator inputs, and selection state can be Blender
signals. A render signal is scoped to one render engine or viewport callback;
many active viewports produce many render signals. Its immutable callback
source records event provenance (`final_render`, `view_update`, or `view_draw`),
while its explicit render intent (`final_render` or `viewport`) selects render
and presentation policy. Viewport update/draw refresh and reuse remain callback
phase mechanics, not render intents. A Blender signal is not itself a render
request, interactive edit, scheduler tick, or worker message.
_Avoid_: Runtime signal, update stream, backend event, add-on payload

**Blender callback adapter**:
The add-on boundary that receives Blender lifecycle or callback inputs and
presents them as Blender signals before add-on runtime behavior runs. It names
the Blender-facing edge, not the render engine class, depsgraph bridge, or
translator policy.
_Avoid_: Adaptor, RenderEngine shell, registration shell, bridge

**Blender ID**:
The type-qualified Blender datablock identity reported as changed by a
depsgraph callback, such as `OBJECT`, `MATERIAL`, `LIGHT`, or `WORLD` plus its
session UID. For a current-scene edit it is the authoritative source resolved
through the active scene generation; current selection does not replace it.
_Avoid_: Selected object, object name, inferred edit owner

**Selection-driven operation**:
One depsgraph callback group whose captured selection evidence associates a
changed Blender ID with a selected source or explicitly redirected interaction
owner. Every selected source must be represented by the same supported
operation or the callback group is rejected atomically. Unrelated selection
does not make a direct data-block edit selection-driven, and separate callbacks
are never combined to complete a group.
_Avoid_: Any edit while something is selected, selection-owned data-block edit

**Viewport region resize**:
A change in the drawable Blender viewport area used to present the interactive
OVRTX result. It is the operator-facing resize event for viewport
responsiveness discussions.
_Avoid_: Window resize, render-session resize, resolution change

**Viewport session**:
The presentation lifetime for one interactive Blender viewport pane. It owns
that pane's refinement, GPU texture, pose-mirror presentation, and viewport
artifacts. In the current-scene route it attaches to an authoring session and
borrows that session's OVRTX controller and runtime scheduler; detaching the
pane stops neither. In the exact-stage route, which has no authoring session,
the viewport session owns one standalone scheduler. It is distinct from an
OVRTX session and from Blender's short-lived fixed-sample render callback.
While several panes borrow one controller, they present the active OVRTX
session's canonical output shape rather than replacing it for competing region
sizes; each pane still owns its camera, refinement progress, and texture draw.
_Avoid_: OVRTX client, final render session, Blender engine lifetime

**Authoring session**:
The scene-scoped in-memory add-on lifetime for one Blender scene. It owns that
scene's generations and, while runtime-active, its one runtime scheduler,
retained desired values, physics playback runtime, timeline state, and OVRTX
controller. Every attached viewport presentation shares that scheduler across
generation replacement and pane attachment or detachment. Several scene
sessions may be retained, but
only one owns the process runtime activation slot in the first implementation.
Add-on registration and completed file load eagerly create the current scene's
initial generation at the next safe main-thread opportunity. Viewport, final
render, and physics demand activate runtime state only when needed and reuse
that generation; callback demand remains the fallback when eager generation
has not completed or failed. The session does not persist across file loads; a
new session materializes the current Blender scene. Opinion records, generation
numbering, runtime values, and playback state remain transient. The session
does not imply durable USD export or runtime failure recovery.
_Avoid_: Viewport session, scene generation, Blender file, durable authoring session

**Viewport session end**:
The lifecycle event that finalizes one pane's viewport artifacts and detaches
its presentation because the render engine is destroyed, presentation leaves
OVRTX, or a restart is requested. It stops a standalone exact-stage scheduler,
but not a current-scene authoring runtime. Replacing an OVRTX session is not a
viewport session end.
_Avoid_: Client shutdown, render cleanup, output flush

**Dependent service process**:
An external process that the add-on depends on for the viewport/runtime
experience, such as an OVRTX or OVPhysX bridge. It needs a
service kind, endpoint, managed pid when known, launch/workspace identity,
liveness, protocol method/status diagnostics, logs, and exit diagnostics before the
add-on can safely report, restart, or clean it up.
_Avoid_: Render worker only, physics worker only, arbitrary process

**Multi-sensor OVRTX session**:
An OVRTX session configured to produce outputs for more than one declared
render product sensor, allowing the presented render product output to change
without treating the user's camera choice as a separate scene workflow.
_Avoid_: Multi-sensor render session, active camera switch, render-session restart, user-facing sensor mode

**Sensor path**:
The USD path of a render product declared as an OVRTX visual sensor for an
OVRTX session.
_Avoid_: Camera path, selected viewport, output alias

**Selected sensor paths**:
The sensor paths whose render product outputs are currently read back and
presented for Blender viewport panes or render results. A single visible pane
is represented as a one-element list; split panes can select several sensor
paths for one frame read.
_Avoid_: Sensor set, active camera object, configured sensor list

**Blender behavior parity**:
The expectation that choosing OVRTX preserves normal Blender camera, viewport,
selection, and render workflows while changing the render/runtime backend.
_Avoid_: OVRTX-specific workflow, custom camera mode, alternate operator behavior

**Selected Blender executable**:
The exact Blender program used by repository-owned developer and diagnostic
entrypoints when a caller chooses one instead of the system default.
_Avoid_: Default Blender, Blender installation, patched Blender branch

**Repeatable patched Blender build**:
A locally compiled Blender from a stable public backport branch whose
documented source procedure reproduces the expected build, installation, and
executable selection. It records the resolved Blender source commit, delegates
host dependencies to Blender, and does not promise byte-identical output.
_Avoid_: Reproducible build, portable Blender release, patched Blender release

**Stable Blender backport branch**:
A public branch with a fixed Blender release and API behavior contract that may
be rebased or force-pushed as its backport is maintained. A build records the
commit it resolved; branch stability does not imply immutable Git history.
_Avoid_: Immutable source revision, release tag, pinned commit

**Add-on preflight**:
A lightweight readiness report shown by the installed add-on for configured
local prerequisites. It confirms the installed runtime payload can load the
required native client surface, but it is not a scene selection check, render
smoke test, or dependent service health check.
_Avoid_: Install check, scene input status, render smoke test, module discovery

**Test fixture support**:
The source-only fixture preparation area for validation assets, fixture catalog
records, golden checks, performance checks, and demo artifacts. When
folded into this repo, it belongs directly under `tests/fixtures/`; do not
preserve the old prep repo or package name as a nested subtree. It must stay out
of the add-on package and runtime imports.
_Avoid_: Runtime package, user scene workflow, add-on source, fixture support

**User scene input**:
The open Blender document, saved or unsaved, which the add-on renders through
live Blender authoring for viewport and final render; the user never selects a
scene-input path. Exact-stage validation is an internal harness concern rather
than a user scene-selection route. This boundary does not require a fixture
catalog entry.
_Avoid_: Fixture catalog entry, USD test fixture, source asset, scene-input
path setting

**Current Blender scene**:
The scene Blender currently presents and edits. It is the sole user-facing
authoring and render boundary whether its content was created in Blender,
opened from a `.blend`, or converted through Blender's native USD import.
It includes native Blender data and registered add-on properties saved in the
`.blend`. Selecting OVRTX changes the render backend; it does not select
another scene.
_Avoid_: Source file, OVRTX scene, selected USD

**Scene generation**:
A transient, immutable USD content identity shared by OVPhysX and OVRTX. For
the ordinary render path its first identity composes current session identity
and add-on opinion layers over one complete stock export of the current Blender
scene. That UID-free stock base may be reused for an exact unchanged clean saved
source; its source lookup digest and stock content digest are not the session
generation digest or generation number. Later identities may compose exact
scene topology deltas for affected Blender IDs over that immutable base. Every
identity has a validated supported-object/material-to-current-USD-prim mapping
and current add-on USD opinion records. An exact-stage adapter may provide
equivalent generation evidence for fixtures or diagnostics.
The generation is private runtime state, not a user-selected input, durable
export, imported-source identity, or USD test fixture.
_Avoid_: Runtime scene materialization, authored scene composition, authored
scene generation, user scene input, exported scene, source USD

**Reusable stock base**:
A validated, immutable, process-independent stock Blender USD export and its
generated assets. It contains no Blender `session_uid`, add-on opinion, topology
delta, render request, or runtime state. A source lookup digest may find it, its
content digest identifies it, and a new session must resolve its stable mapping
metadata and compose fresh identity and opinion layers before use.
Its content digest covers the root USD and every generated artifact file;
session identity is added only to the scene generation digest.
_Avoid_: Scene generation ID, generation number, session cache, render cache,
saved runtime state

**Scene topology delta**:
The immutable affected-ID add, versioned replacement, and deletion contribution
between two current-Blender-scene generations. A replacement deactivates its
predecessor root and defines the selected stock-exported object and required
dependencies under a private generation-owned root; a deletion deactivates the
currently mapped root. The generation mapping makes private path changes
invisible to callers. This delta owns Blender-native topology and is distinct
from sparse add-on schema opinions.
_Avoid_: Whole-scene replacement, add-on USD opinion change, same-path field
diff, durable edit layer, runtime topology command

**Scene-camera render parity**:
The expectation that an OVRTX final-frame render uses the active Blender scene
camera's framed view, projection, and render aspect.
_Avoid_: Fixture-camera fallback, transform-only camera copy, viewport-only sync

**Camera projection proof**:
Runtime evidence that a rendered camera view matches the expected projection
semantics, such as perspective depth scaling or orthographic equal-scale
behavior. It is stronger than proving that a camera render returns any nonblank
frame.
_Avoid_: Nonblank smoke test, hash-only image difference, camera render check

**Scene lighting parity**:
The expectation that OVRTX preserves the Blender/Cycles lighting response for
the same scene, camera, authored lights, world, emissive materials, and display
contract.
_Avoid_: Lighting match, same look, add-on lighting

**Renderer comparison baseline**:
A reproducible evidence lane that preserves the authored Blender scene and USD
opinions so renderer differences can be measured before compatibility
compensation is judged. It is a measurement baseline, not an approval target.
_Avoid_: Truth render, raw screenshot, fair mode, golden render run

**OVRTX compatibility policy**:
The reusable, scene-agnostic rules that adapt Blender scene intent to OVRTX
renderer semantics while preserving Blender-authored values as the source of
intent. It explains renderer or interchange semantic differences across scenes;
it is not a fixture-specific tune.
_Avoid_: Lighting hack, fixture tweak, art direction, screenshot fix

**Renderer response residual**:
A measured OVRTX/Cycles image difference that remains after Blender-authored
values, display ownership, fixture preparation, and current OVRTX compatibility
policy controls have been isolated. It is evidence for renderer/library
follow-up unless a new reusable compatibility rule is proven.
_Avoid_: Brightness tweak, screenshot adjustment, hidden fixture compensation

**OVRTX color presentation**:
The requested output representation for an OVRTX frame, including render
variable, frame format, color interpretation, and display-transform ownership.
The Blender scene owns this input; a scene-free adapter supplies it explicitly.
Viewport and final render interpret the same configured value identically.
Scene-linear HDR is the default for active workloads; LDR remains a supported
explicit configuration selectable by the Blender user and saved with the
scene.
_Avoid_: Color presentation lane, flow default, brightness multiplier, display hack

**OVRTX final render postprocess boundary**:
The OVRTX final render callback publishes an OVRTX render product into Blender's
render result and does not run source-scene compositor postprocess. Source
`.blend` compositor, sequencer, or cached composite state is outside the OVRTX
render-product comparison lane unless explicitly enabled as a separate Blender
postprocess test.
_Avoid_: Compositor lighting fix, hidden scene postprocess, screenshot lift

**USD test fixture**:
A developer-facing render-ready USD stage used by fixture catalog tests,
golden validation, demos, and reproducible artifacts. Preparation
intermediates are not retained fixture identities; the catalog names only the
final render-ready stage.
_Avoid_: Source blend, current Blender scene, source USD, raw export

**Fixture presentation recipe**:
An explicit fixture-specific set of derived scene opinions and capture settings
chosen to reproduce an intentional presentation while leaving the source asset
unchanged. It is neither a scene-agnostic OVRTX compatibility policy nor an
accepted validation result.
_Avoid_: Source repair, export fidelity, compatibility rule, golden

**Shared runtime stage**:
The in-memory USD stage authority for the current demo scene. Rendering and
physics compose through this stage's state; the term does not imply same-process
or zero-copy access.
_Avoid_: Same memory, direct native sharing, file handoff, copied transform stream

**Shared runtime stage host**:
The runtime that owns the shared runtime stage and exposes it to OVRTX, OVPhysX,
and Blender coordination code. It is the scene authority, not the owner of
physics or render commands. The current demo-local host may be replaced by
ovstage when that library exists.
_Avoid_: Provider, Blender-owned stage, transform relay, sidecar cache

**Stage mutation authority**:
The lane responsible for a change to the shared runtime stage. A viewport
interaction can produce a physics-authored stage mutation when OVPhysX turns the
interaction into simulated pose changes.
_Avoid_: Direct Blender transform write, unowned stage change, render-owned body pose

**OVPhysX-to-OVRTX pose bridge**:
The add-on-owned conversion from OVPhysX-authored body poses to OVRTX transform
value updates. It is coordinator-mediated translation, not direct
service-to-service IPC, stage ownership, or render-command ownership by the
stage host.
_Avoid_: Direct OVPhysX-to-OVRTX IPC, transform relay, stage-host render projection

**OVPhysX simulation specification**:
The deeply immutable set of desired scene and service inputs for an OVPhysX
simulation. OVPhysX simulation reuse policy evaluates simulation
specifications; a specification is not runtime pose state, a physics body prim
set, or protocol resource identity.
_Avoid_: Physics config key, controller config, body set, simulation ID

**OVPhysX simulation reuse policy**:
The add-on-owned rules that determine whether desired OVPhysX simulation input
can continue using a live simulation or requires replacement. Explicit reset
and terminal recovery are replacement inputs; OVRTX session replacement is
not.
_Avoid_: Controller key, OVRTX session reuse, physics restart heuristic

**ovstage**:
The expected OV library that will provide shared runtime stage hosting for the
OVRTX and OVPhysX composition demo.
_Avoid_: Current demo host, file-backed stage workaround, local reimplementation

**Fixture catalog**:
The canonical testing catalog of pinned Blender scene fixtures and USD test
fixtures used by comparative validation, performance checks, demos, and
reproducible artifacts. It records immutable source identity, canonical
camera/render-product choices, and fixture defaults; it is not part of the
ordinary Blender render workflow.
_Avoid_: User scene manifest, runtime boundary, source manifest, prep output guess, inferred fixture metadata

**Render-pool fixture**:
A small USD test fixture for the composition demo where OVRTX renders
OVPhysX-driven rigid bodies. In this name, pool means a pool of body slots, not
water or fluid simulation.
_Avoid_: Water pool scene, fluid fixture, beauty fixture

**Composite physics render fixture**:
A USD test fixture for an interactive demo that combines a visually rich
OVRTX render scene with OVPhysX-authored rigid-body motion.
_Avoid_: Good-looking demo, physics asset mashup, Blender physics demo

**Physics presentation island**:
A self-contained collision area inside a composite physics render fixture where
the simulated bodies are meant to interact visibly and stay framed.
_Avoid_: Implicit scene collision, full-scene physics, collision proxy pass

**Ramp-drop fixture**:
A composite physics render fixture whose subject is rigid bodies dropping onto
an explicitly authored ramp inside a physics presentation island.
_Avoid_: Full-scene physics demo, background-object collision demo

**Stair-drop fixture**:
A ramp-drop fixture whose contact surface is a staircase made from explicit box
colliders, producing readable step impacts without relying on complex collision
meshes.
_Avoid_: Mesh-collider ramp, implicit staircase collision, full-scene collision

**Orange stair-drop fixture**:
A stair-drop fixture whose dynamic bodies are authored from the orange source
asset rather than procedural or placeholder rigid bodies.
_Avoid_: Procedural orange stress test, single-orange validation, orange-colored cube demo

**USD layer**:
A concrete USD composition layer or file that contributes opinions to a stage.
_Avoid_: Workflow category, ownership boundary

**Durable layer**:
The USD layer that owns the durable prim or authored opinion an edit is trying
to change. It can be a write target when a USD stage is composed from many
scene and asset layers.
_Avoid_: Originating USD layer, workflow category, department category,
USD test fixture path, runtime overlay, export destination

**Temporary USD layer**:
A session-scoped USD layer composed with a USD stage so a render
session can use presentation or preview opinions without changing durable
authored state. OVRTX scene composition generates its fixed material, camera,
and render-product presentation layers internally; an interactive edit does not
select one as a write target.
_Avoid_: Generic topology payload, interactive write target, durable layer,
originating USD layer, runtime update stream

**Add-on USD opinion record**:
An immutable, validated USD contribution compiled for one scene generation.
Each record contains the complete current add-on contribution rooted at one USD
prim and identifies every effective USD claim it authors. The scene-generation
owner persists and composes the complete current record set into the private
generation artifact. Typed Blender conversion occurs before this boundary; the
record does not classify topology as render or simulation data.
For a Blender-native object represented in the current scene generation, the
contribution may contain sparse add-on schema opinions on validated mapped prims
and complete definitions for add-on-only descendants beneath its root. It never
redefines Blender-native geometry or topology. A root that exists only to
represent an add-on concept may instead contain that concept's complete
definition. These are the same record contract, not separate source-backed and
add-on-only record kinds.
Changes spanning multiple root prims are validated and applied atomically as a
record set so runtimes never observe a partial topology change.
_Avoid_: Temporary USD opinion record, temporary layer payload, render topology
record, simulation topology record, raw Blender edit, generic topology
operation, topology event log

**USD schema opinion**:
A Blender-authored choice to inherit, apply, or remove an applied USD schema
for a prim in the current authoring session. Inherit contributes no stronger
schema opinion; apply and remove become native USD `apiSchemas` list operations
without changing a weaker source layer.
_Avoid_: Runtime capability, schema command, copied effective schema state,
custom attribute reflection

**Use source**:
The explicit authoring action that clears a current-session USD opinion so the
composed value or schema is resolved from weaker source layers again. Equality
with the current source value does not imply use source because an equal
stronger opinion may intentionally pin composition.
_Avoid_: Reset to default, automatic equality pruning, rollback

**Blocked authoring state**:
The condition where the current Blender scene cannot be materialized and
validated as a scene generation. Blender retains the source change, no
generation is created, and runtime presentation is withheld until a later edit
or undo produces valid source.
_Avoid_: Invalid generation, runtime activation failure, automatic rollback,
retry state

**Sparse add-on opinion change**:
The affected add-on USD opinion record replacements and root removals
calculated between consecutive scene generations. It is committed
output, not an authoring command. It describes the authored change independently
of whether a runtime applies it directly or replaces its session from the
complete scene generation.
_Avoid_: Sparse topology change, partial scene snapshot, runtime topology
command, render topology edit, physics topology edit

**OVRTX scene composition**:
The render-session preparation step that adds OVRTX-only camera, render-product,
or presentation opinions above a scene generation. Its output is the
exact USD path an OVRTX session opens and is an input to OVRTX session reuse
policy.
_Avoid_: Scene generation, shared-stage runtime composition, runtime update stream

**Material scene conversion**:
The pre-session translation of Blender-authored materials into material and
binding opinions that OVRTX scene composition can apply to a USD stage.
_Avoid_: Material overlay, OpenPBR overlay, material export

**OVRTX session**:
An OVRTX renderer session started from exact composed scene input and render
product configuration, with stable identity for reuse decisions. Final rendering
and viewport rendering both use OVRTX sessions; the viewport session owns a
broader interactive lifetime.
_Avoid_: Render session, runtime session, viewport session

**OVRTX simulation ID**:
The immutable caller-provided identifier used to create and address an OVRTX
simulation through the service protocol. It is distinct from the
server-normalized simulation name and from OVRTX session reuse policy.
_Avoid_: Session identifier, simulation name, session key

**OVRTX simulation name**:
The server-normalized resource name for a created OVRTX simulation. It is
distinct from the caller-provided simulation ID.
_Avoid_: Simulation ID, session identifier, render session name

**OVRTX session specification**:
The deeply immutable set of desired scene and service inputs for an
OVRTX session. OVRTX session reuse policy evaluates session specifications; a
specification is not resolved service state, a protocol resource identifier,
or a live session handle.
_Avoid_: Render request, session key, simulation ID

**OVRTX session reuse policy**:
The add-on-owned rules that determine whether desired OVRTX session input can
continue using a live session or requires session replacement. It is an
operational compatibility opinion, not resource identity or viewport lifetime.
_Avoid_: Session key, simulation ID, simulation name

**OVRTX camera pose source**:
The source of the camera pose used by an OVRTX session: either the composed
scene or a runtime update. Removing a runtime pose override currently requires
session replacement to restore the composed pose.
_Avoid_: Blender camera mode, fixture camera, viewport controls mode

**Composition replay**:
The best-effort reapplication of a concrete authored or generated USD layer
after runtime/session recreation or desync so topology or other
runtime-unsupported edit state remains visible. It changes composition identity
and is not the first application of a temporary USD layer, update replay, write
persistence by itself, or a guess from stale runtime records.
_Avoid_: Initial overlay application, runtime edit replay, runtime update stream, inferred topology replay, stale layer retry

**Shared-stage runtime composition**:
The scheduler-ordered runtime path that publishes OVPhysX-authored pose values
through the runtime stage host and applies them to an already-open OVRTX
session. It does not write USD layers or choose the scene input that starts the
OVRTX session.
_Avoid_: OVRTX scene composition, USD layer writing, export

**Update**:
The edit mechanism that changes the current authoring session without composing
new session input. It is distinct from a general notification, redraw, pose
publication, or scheduler tick.
_Avoid_: Runtime update when naming the mechanism, live edit, patch

**OVRTX value update**:
The atomic application of complete transform values or arbitrary USD attribute
values to an active OVRTX session under view data authority. It is a concrete
application operation, not a payload type or separate edit mechanism.
_Avoid_: OVRTX value record, render value record, generic value record,
OVRTX values payload

**Compose**:
The edit mechanism that creates new session input, causing session identity to
change before the edit can be visible in the authoring session.
_Avoid_: Session reset, rekey, reload, runtime update

**Update stream**:
The targeted edit mechanism that carries Blender-authored updates into the
current authoring session without re-exporting the composed USD stage or
rewriting a durable layer. Updates carry enough target identity and layer
provenance to be written explicitly later when applicable; the stream is not a
write target.
_Avoid_: Runtime update stream, runtime delta, runtime delta layer, patch,
sync message, live edit

**View update stream**:
The update stream for view-authoritative edits. It retains pending view edits
until an active OVRTX session can apply them; it does not own simulation state
or physics replay.
_Avoid_: OVRTX queue, render queue, view backend

**Sim update stream**:
The update stream for sim-authoritative edits. It owns pending and accepted
initial condition values across OVPhysX simulation replacement; the shared
runtime stage remains the authority for current playback poses.
_Avoid_: OVPhysX queue, physics backend, playback pose store

**Edit mechanism**:
The way an interactive edit reaches, or does not reach, the current authoring
session: update the current session, compose new session input, or no session
mechanism. It is separate from edit shape and persistence.
_Avoid_: Application path, feedback, preview, session application, live route

**Data authority**:
The part of the current session's data model that owns edited data after an
interactive edit is applied. View authority owns rendered/view state; sim
authority owns simulation state or simulation initial conditions. For
update-stream edits, the data authority determines the live service authority:
view updates apply through OVRTX value updates, while sim updates apply through
OVPhysX/shared-stage simulation value updates. Material, light, world, UV,
camera, pose, and similar names are edit targets or conversion details, not
separate authority names.
_Avoid_: Edit authority, user authority, backend, worker, lane, update route,
service authority, material route, light route, world route, UV route, camera
route, transport adapter, edit taxonomy

**Edit record**:
A diagnostic record of an interactive edit's target, mechanism, persistence,
result, and observable effects. It is artifact vocabulary, not Blender user
vocabulary, and should not shape core edit or runtime interfaces.
_Avoid_: Edit evidence, update evidence, capability evidence, payload evidence, proof

**Diagnostic artifact**:
A generated file for automation, regression diagnosis, validation, or
human debugging. Diagnostic artifacts may contain edit records, status, metrics,
logs, images, or timings, but the artifact vocabulary should stay at the
diagnostic boundary.
_Avoid_: Evidence when naming current code, user-facing behavior, or core edit/runtime interfaces

**Diagnostics**:
Supplemental details used to explain an operation after its result or status is
known. Prefer narrower names such as RPC status, startup status, metrics, logs,
or artifacts when the consumer is specific.
_Avoid_: Evidence, proof, diagnostics as a replacement for typed result fields

**Current desired state**:
The latest accepted Blender-authored value for each semantic runtime target,
retained by the authoring session so an inactive or replacement runtime can be
brought to the same state. It is target state, not callback history, an
application result, or a count of values superseded before application.
_Avoid_: Event queue, callback ledger, applied state, retry history

**Pending application**:
The current desired values that have not yet been accepted by their active
OVRTX or OVPhysX target. Repeated values for one semantic target replace one
another before application; the pending set does not preserve discarded
callback samples or decide runtime teardown and readiness.
_Avoid_: Callback backlog, edit history, retry ledger, discarded-event count

**Typed lifecycle outcome**:
The explicit status or result returned by a runtime activation, application,
tick, replacement, or teardown operation. Lifecycle policy uses this outcome
and the owning runtime's state, never an inferred pattern in diagnostic events.
_Avoid_: Handoff evidence, diagnostic success heuristic, event-ledger state

**Bounded recent diagnostics**:
A size-limited tail of recent runtime or authoring facts retained only to
explain typed outcomes. It may report what happened, but cannot authorize edit
admission, application retry, teardown, readiness, or presentation.
_Avoid_: Runtime ledger, replay queue, lifecycle authority, complete history

**Exact semantic reporting policy**:
The diagnostic rule that reports the current semantic targets and outcomes an
operation actually considered. It does not count superseded callback samples
or influence runtime behavior.
_Avoid_: Callback accounting, discarded-event counter, admission policy

**Values written**:
Diagnostic signal that targeted values were accepted by their owning runtime
service into the current authoring session. For a VIEW application it becomes
true only after OVRTX accepts the values; for a SIM application it becomes true
only after OVPhysX accepts them. It does not imply durable persistence,
rendered publication, or Blender presentation.
_Avoid_: Session values written, render state changed, transform updated

**Authored light form**:
The Blender-facing authoring identity of a light: POINT, SPOT, SUN, or AREA
with its Rect-style or Disk-style area shape. It is distinct from USD light
family because multiple authored light forms can share one USD family.
_Avoid_: USD family, light family when meaning Blender light type

**Edit replay**:
The interactive-authoring-loop lifetime reapplication of accepted
Blender-authored edit state after the relevant session or endpoint is
recreated. It is not viewport session state, write persistence, whole-scene
export, or fixture composition.
_Avoid_: Runtime edit replay, persistent runtime override, saved edit, runtime layer, auto-export

**Selected edit write**:
The internal exact-stage persistence step that writes selected interactive
edits to a resolved durable layer whose provenance remains authoritative. The
ordinary current-Blender-scene route has no selected edit write because Blender
data is its authority. This is not a whole-scene USD export, Blender stock USD
export, or a separate edit mechanism.
_Avoid_: Explicit export, export batch, asset export, auto-export

**Replay contract**:
The category-specific rule for whether and how an edit can be replayed,
including its identity, lifetime, ordering, pending behavior, terminal failures,
and recorded result.
_Avoid_: Generic retry policy, blanket replay, persistence rule

**Interactive edit planner**:
The decision point that turns a Blender-authored edit into update, compose,
or unsupported intent. Internal exact-stage tooling may additionally select
write persistence when authoritative layer provenance supplies a write target.
The planner names the edit owner, edit mechanism, persistence, and expected
result or reason before scheduling begins.
_Avoid_: Frontend edit resolver, validation-loop planner, scheduler edit handler

**Backend capability seam**:
The boundary that validates an OV worker/client path can apply an update to an
active render or physics session without re-exporting authored state.
_Avoid_: End-to-end interactive proof, viewport usability proof

**Interactive operator seam**:
The boundary from stock Blender viewport interaction to visible runtime
feedback, including picking, gizmo alignment, edit capture, runtime application,
and runtime pose mirroring.
_Avoid_: Backend capability seam, headless transform proof

**Rendered viewport presentation**:
The OVRTX image presented inside Blender's active 3D Viewport as that
viewport's rendered-view result. It is interpreted through the user's current
viewport view choice, not as an independent fixture-camera preview.
_Avoid_: Standalone preview, fixture-camera overlay, backend-only render

**Color-presentation validation probe**:
A validation run that records how OVRTX viewport frame bytes are classified
against Blender view settings. It is not a pixel-parity validation unless it also
compares rendered output.
_Avoid_: Color parity probe, display-transform parity proof

**Native viewport fallback**:
The use of Blender's ordinary 3D Viewport presentation when OVRTX cannot
faithfully render the user's current viewport view choice, including
orthographic user views. It preserves native viewport behavior instead of
showing an unrelated OVRTX image.
_Avoid_: Fixture-camera fallback, approximate OVRTX view, stale preview

**Blender selection source**:
A native Blender selection input, such as viewport picking, Outliner selection,
or programmatic object selection. It is the user's interaction surface, not the
backend identity that necessarily owns the edit.
_Avoid_: OVRTX pick target, generated proxy surface, viewport-only selection

**Edit owner**:
The authored or runtime identity that should receive an edit resolved from a
Blender selection source, such as a render prim, physics body, material, light,
or USD property owner.
_Avoid_: Selected object, picked mesh, Blender interaction object, viewport proxy

**Selection resolution**:
The workflow step that maps a Blender selection source to the nearest valid
edit owner before planning an edit.
_Avoid_: Proxy picking, direct object edit, viewport hit handling

**USD prim resolution**:
The matching of a Blender-authored source to the corresponding prim in the USD
stage so an edit target can be constructed. It resolves prim paths and matching
evidence, not write-target layers or live `pxr.Usd.Prim` objects.
_Avoid_: Identity resolution, USD owner resolution

**USD path resolution contract**:
The USD stage and fixture catalog conventions that let runtime code resolve
selection ownership from USD paths, schemas, hierarchy, and fixture defaults.
Stable USD path identity after Blender import or session USD setup is the
irreducible requirement. This is an internal ingest/runtime identity
requirement, not a demand that users provide USD test fixtures or fixture manifests;
direct `.blend` and USD-family flows provide the identity for the session.
Optional fixture hints belong only to non-derivable exceptions.
_Avoid_: Interaction metadata, name heuristic, fixture guess, implicit ownership

**Inferred interaction mapping**:
A conservative fallback mapping used only when a Blender selection source maps
obviously to one simple edit owner without complete USD path resolution.
Inferred mappings are preview-only until USD path resolution exists.
_Avoid_: USD path resolution contract, authoring ownership, promotion-ready edit

**Blender interaction object**:
A native Blender scene object used as the user's stock selection,
manipulation, and inspection surface for an edit owner. It should be visually
colocated with the rendered object it represents, including an understandable
stock Blender transform origin or gizmo, and is not a generated visible helper
shape whose only purpose is to fake picking.
_Avoid_: Viewport proxy, generated pick shape, OVRTX render texture, physics body, durable asset state

**Initial condition**:
The user-authored starting state used when a physics generation begins,
including pose, velocity, and angular velocity.
_Avoid_: Playback pose, current simulation pose, render-only transform

**Initial condition values**:
Accepted user-authored values that define the next physics generation's
starting state and are reapplied when the OVPhysX simulation is recreated.
_Avoid_: Pose overrides, persistent overrides, startup replay, retained poses

**Playback pose**:
The runtime body pose produced by physics playback for the current simulation
generation. It is mirrored to Blender interaction objects for inspection, but
is not the same as an initial condition.
_Avoid_: Initial pose, drop pose, authored transform

**Authoring attribute**:
A user-editable property on a resolved edit owner, such as a physics, semantic,
material, or light value.
_Avoid_: Runtime update, worker message, custom UI state

**Look attribute**:
An authoring attribute that affects rendered appearance, such as material or
light values.
_Avoid_: Physics attribute, shader topology, render-session setting

**Value edit conversion policy**:
Scene-agnostic rules that interpret Blender-authored value edit state as target
USD value attributes, including whether a source field is supported,
unsupported, non-rendering, or topology-changing. It applies only after an edit
owner can be resolved and does not resolve edit ownership, choose edit mechanism or
persistence, create topology, compose session input, or transport updates to a
runtime.
_Avoid_: Edit owner resolution, edit routing, topology conversion, transport
adapter, generic conversion policy

**USD value edit support contract**:
The runtime-neutral contract for supported USD value edits: which USD
attributes are valid value-edit targets and what USD value type each accepts.
It does not interpret Blender source state, choose runtime service authority,
build transport records, or perform conversion.
_Avoid_: Value edit conversion policy, update route, runtime adapter, OVRTX
record schema, USD value target types

**Material value edit**:
A look attribute edit on an already-resolved material that is part of the USD
stage, such as diffuse color, roughness, metallic, or alpha/opacity,
when it maps to an existing input on a USD material or shader prim. It is not a
material binding, new material, texture graph, or shader network edit.
_Avoid_: Material structure, shader graph edit, material binding edit

**UV value edit**:
A look attribute edit on one already-resolved USD Mesh whose existing
face-varying UV value array matches the Blender active UV layer loop count and
proven loop order. It is not UV set creation, indexed UV authoring,
interpolation change, or mesh topology.
_Avoid_: UV topology edit, UV set edit, indexed UV primvar, mesh topology edit

**Look-only attribute**:
A look attribute that does not affect physics inputs or playback pose for the
active physics generation.
_Avoid_: Physics material value, initial condition, playback pose

**Physics-affecting attribute**:
An authoring attribute that affects physics inputs or simulation state, such as
initial pose, velocity, angular velocity, mass, friction, restitution, or
collider settings.
_Avoid_: Look-only attribute, render-only setting

**Physics playback lock**:
A user-facing editing state while an active physics generation owns
physics-affecting state for a specific object or attribute. The user can select
and inspect locked physics-backed objects, but physics-affecting authoring waits
until the user returns to the initial condition. In the first version, the stock
Blender timeline frame 1 is the initial condition state; pause and
mid-simulation stop do not clear the lock.
_Avoid_: Playback-pose lock, auto-reset, teleport edit, silent pending physics edit

**Runtime pose mirror**:
The synchronization that keeps Blender interaction objects aligned with
runtime-authored render values or physics pose data when runtime playback owns
motion.
_Avoid_: Viewport proxy mirror, authoring export, Blender physics simulation, transform bake

**Live physics authoring**:
The Blender workflow for creating and editing physics topology and initial
conditions consumed by OVPhysX. OVPhysX owns simulation playback state, not the
user-authored inputs.
_Avoid_: OVPhysX authoring, playback-pose authoring, fixture editing

**Value edit**:
An edit that changes a value on an already-resolved owner without changing
USD prim, relationship, binding, shader, mesh, collider, or authored light form
topology. Material, light, transform, pose, velocity, semantic, and physics
property value edits are value edits
when their owner already exists.
_Avoid_: Attribute edit, topology edit

**Topology edit**:
An edit that changes USD or graph topology, such as adding or deleting prims,
changing object hierarchy, material bindings, shader networks, mesh topology,
collider topology, light creation, or authored light form. Current-Blender-scene
topology produces a scene topology delta and a new logical scene generation.
Explicit internal exact-stage tooling may compose or write when it retains
authoritative durable-layer provenance; a future upstream topology protocol may
provide another runtime application route without changing scene authority.
_Avoid_: Structural edit, existing value edit, mesh-only topology

**Physics body prim set**:
The add-on-owned set of authored rigid-body prim paths whose poses must be read
from OVPhysX and published as one complete physics pose set for shared-stage
composition. The set is fixed for one shared-stage composition lifetime; it is
a pose query/publication contract, not OVPhysX simulation topology or identity.
_Avoid_: Body-prim CLI args, hard-coded tracked prims, manifest body list

**Physics pose set**:
A complete set of OVPhysX-authored body poses for the physics body prim set at
one simulation time. It is valid only when every body in the set has the
required pose attributes.
_Avoid_: Pose snapshot, partial body state, render transform batch

**Physics pose publication**:
A complete physics pose set plus handoff metadata such as generation,
simulation time, sequence, source authority, and production timing. This is the
data contract between physics execution and stage/render coordination; it does
not imply that the stage host or future ovstage owns the producer schedule.
_Avoid_: Producer lane, stage tick, render frame, queued transform write

**Physics pose producer**:
The orchestration adapter that advances and reads OVPhysX according to playback
intent, then emits physics pose publications. It owns physics service calls and
runtime scheduling policy, not shared runtime stage hosting.
_Avoid_: Producer lane, ovstage scheduler, stage owner, render loop

**Applied pose revision**:
The shared runtime stage revision whose OVPhysX-authored body poses have been
applied to the active OVRTX session.
_Avoid_: Viewport snapshot, timeline frame, render frame

**Host profile**:
A named machine and GPU environment recorded with runtime and performance
results. The name identifies the host class, not an individual CUDA device
ordinal.
_Avoid_: GPU suffix, CUDA device suffix

**Golden image regression**:
The absolute visual comparison of a fresh designated-case render with its
tracked golden image.
_Avoid_: Performance golden regression

**Golden identity**:
The designated visual case identified by one golden record. Host is capture
provenance rather than part of identity.
_Avoid_: Golden name, baseline key

**Golden promotion**:
An explicit operator-reviewed Git change replacing a tracked golden image and
its metadata. The validator never performs promotion.
_Avoid_: Auto-update, refresh baseline

**Approval summary**:
The golden metadata recording who approved the reference, when, and why.
_Avoid_: Review note, approval blob

**Viewport refinement**:
A rendered viewport preview that starts from a low-sample OVRTX result and
replaces it with higher-sample results while the viewport request remains stable.
_Avoid_: Progressive rendering, accumulation, fixed-sample preview

**Viewport responsiveness**:
The perceived promptness and cadence of visible OVRTX viewport feedback while a
user changes the viewport camera. It prioritizes reacting to the latest
orbit/pan/zoom input over refining a stable viewport request to maximum samples.
_Avoid_: Progressive rendering performance, interaction FPS, refinement latency

**Stable viewport request**:
A viewport preview request whose rendered content, view, output size, and OVRTX
runtime inputs have not changed since the current refinement began.
_Avoid_: Same frame, unchanged scene, cache hit

**Viewport snapshot**:
One stable OVRTX viewport preview state that may be refined for visual quality.
It is not a timeline frame, animation sample, or live edit stream.
_Avoid_: Animation frame, motion frame, live frame

**Viewport draw**:
One Blender `view_draw` callback that presents the current viewport snapshot.
A draw presents the latest frame published by the session's render thread and
never performs render work or waits on one.
_Avoid_: Viewport frame, rendered frame, reused frame

**Sample range**:
The minimum and maximum sample counts requested for an OVRTX render result.
Equal endpoints mean the render has no refinement range.
_Avoid_: Sample target, fixed samples, quality setting

**Minimum samples**:
The lower endpoint of a sample range. In viewport refinement, it is the first
sample count used for a new viewport snapshot.
_Avoid_: Initial quality, preview samples, draft samples

**Maximum samples**:
The upper endpoint of a sample range.
_Avoid_: Final samples, target samples, quality cap

**Completed samples**:
The sample count represented by a render result. In viewport refinement, it is
scoped to the current stable viewport request. It is not the lifetime sample
count of a reused OVRTX session.
_Avoid_: Current quality, sample step, frame index

**Session completed samples**:
The lifetime sample count processed by a reused OVRTX session across multiple
viewport snapshots. It is diagnostic artifact data, not refinement state.
_Avoid_: Completed samples, total quality, global samples
