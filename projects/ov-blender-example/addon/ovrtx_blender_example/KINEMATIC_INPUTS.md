# Kinematic pose inputs

The admitted fixed CPU/GPU worker accepts pose writes to authored, enabled
kinematic bodies as solver targets. The add-on never changes a dynamic body's
pose to emulate this behavior. Input paths are separate from the complete
dynamic output roster; Object and PoseBone input owners remain authored.

`physics_kinematic_inputs.KinematicInputs` copies generation digest, evaluated
Blender frame and typed meter/XYZW poses. Main-thread evaluation resolves stable
Object/bone identities and includes evaluated owner transforms and the frozen
scaled body offset. Quaternions are normalized. Units, scale, role, identity or
body-offset changes require restart. Worker admission checks the exact complete
input roster, generation and actual composed kinematic schema before writing.

Capture evaluates every substep before its corresponding native step, including
pre-roll and unsampled frames. Live playback uses the existing session scheduler
and main-thread dispatch; the asynchronous output producer cannot advance past
uncopied inputs. Backward playback resets and replays solver history with each
historical target. Successful per-step input history also permits bake capture
to restore an interrupted live simulation without teleporting its dynamic bodies.

The supported input dependency model is keyed Object/PoseBone transforms with
ordinary authored parent hierarchies. Dynamic output ancestors are rejected.
Active constraints or drivers on an input owner/ancestor are explicitly
unavailable because this implementation does not prove absence of indirect
output feedback through arbitrary Blender dependency graphs. This is an add-on
support boundary, not an OVPhysX transport limitation. User constraints, drivers,
Actions and channels are never muted or rewritten. Animated scale, live joint
motors, articulation controls and OVRTX skinning are not admitted by this feature.

Behavioral fixtures in `public/tests`:

- `blender_native_kinematic.py`: held/moving Object and bone native targets,
  separate output roster, independent sphere contact on CPU and GPU.
- `blender_kinematic_capture.py`: evaluated animated targets with transformed
  owners, body offsets, meter/centimeter equivalence, five substeps, preserved
  authoring channels/frame/Action, full capture without input-body output samples.
- `blender_kinematic_inputs.py`: immutable capture, renames, identity, unit,
  scale, offset, role and feedback rejection.
- `test_physics_kinematic_inputs.py`: typed finite normalized values, stale or
  incomplete rosters, dynamic/native schema rejection and main-thread boundary.

The measured direct-contact control remains at x=0 m when its input is held;
moving the input from x=-1 to 0 m pushes it to approximately x=0.4 m. Animated
Object/bone capture agrees across meter/centimeter settings within 1e-4 m.
These results prove contact behavior rather than relying on successful RPC
acknowledgement or schema presence alone.
