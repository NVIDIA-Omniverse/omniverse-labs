# Validated rigid-body controls

These authoring controls compile private USD generations and take effect on an
explicit simulation restart. They do not stream live updates. Simulation admits
only the exact CPU/GPU server and client identities in `physics_capabilities.py`.
The executable acceptance is `public/tests/blender_native_body_controls.py`;
`blender_body_controls.py` checks composed output, units, reset and invalidation.

| Authoring field | Authored USD | Authoring units → USD units | Behavioral check |
| --- | --- | --- | --- |
| `mass_mode`, `mass_kg` | MassAPI `mass`; MASS clears explicit density when mode authored | kg | Same hinge with 1/2 kg changes angular response |
| Body `density` | MassAPI `density`, `mass=0` overrides weaker explicit mass | kg/BU³ → kg/m³ | Density inferred 2 kg agrees with explicit 2 kg |
| `center_of_mass` | MassAPI `centerOfMass` | local body-frame BU → m | Offset center changes gravity torque response |
| `diagonal_inertia` | MassAPI `diagonalInertia` | kg·BU² → kg·m² | Doubled axis inertia reduces angular acceleration |
| `principal_axes` | MassAPI `principalAxes` | normalized wxyz quaternion | Quarter turn rotates principal inertia response |
| `initial_linear_velocity` | RigidBodyAPI `velocity` | BU/s → m/s | Free body travels 4 m at 2 m/s over 2 s |
| `initial_angular_velocity` | RigidBodyAPI `angularVelocity` | rad/s → degree/s | Free body rotates 2 radians at 1 rad/s over 2 s |
| `linear_damping`, `angular_damping` | PhysxRigidBodyAPI damping fields | 1/s | Both translation and rotation decay relative to undamped control |
| Box `dimensions` | Cube unit size and local dimensions transform | BU (including owner scale) → m | Unit box settles with center 0.5 m above floor; cm/m agree |
| Capsule `radius_m`, `height_m` | Capsule radius and cylinder height, Z axis | m | Radius 0.2 m and cylinder height 0.6 m settle at 0.5 m |
| Material `density` | MaterialAPI `density` | kg/BU³ → kg/m³ | Inferred mass agrees with explicit body mass |
| Material static/dynamic friction | MaterialAPI friction attributes | unitless | Incline retention and sliding deceleration |
| Material restitution | MaterialAPI `restitution` | [0, 1] | High restitution rebounds above inelastic control |
| Scene gravity direction/magnitude | PhysicsScene gravity attributes | direction, m/s² | Half gravity halves early drop; X gravity moves along X |

Unset fields remain sparse and do not author stronger values. `mass_mode=INHERIT`
omits body mass/density so source/material defaults apply. Switching to DENSITY
authors zero explicit mass to remove its precedence. Inertia may be all zero for
automatic inference; partially zero or negative inertia is rejected. Geometry
and velocities are converted once by `PhysicsUnits`. COM is authored in the
rigid body's frame; it is independent of collider offsets. Sphere/capsule fields
ending `_m` remain meter-valued for compatibility, including under scene-unit changes.

Generated collider geometry stays invisible to rendering. Collider schema
INHERIT/REMOVE and visual visibility remain distinct. Shape changes replace the
private opinion; they never mutate source USD. New boxes/capsules switched from a
legacy mesh suppress the old mesh collision without deleting visual geometry.

CCD, sleep/stabilization thresholds, solver iterations, contact/rest offsets,
material combine modes, filtering, advanced cooking and telemetry are **not
admitted authoring controls**. They are absent from editable RNA/UI and return an
unavailable capability reason. Existing explicit SimReady convex-decomposition
conversion remains a compatibility path, not generic cooking admission.
