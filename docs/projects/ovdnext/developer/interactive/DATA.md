<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Physics lab data and inspection recipes

[Open the interactive guide](index.html). It reads a local, extracted data asset;
it does not execute PhysX, load a native library, stream a simulation or upload a
capture. Select **Fit selection** to inspect a small body or contact region,
and switch between 3D, Front and Top views. All four labs appear on the same
page with independent frame and selection controls. Key attributes stay visible
in each inspector; full lists and bounded raw previews are in a closed-by-default
**Raw PhysX attributes** disclosure below each scene. All other content stays open.
All controls work offline.

## What is recorded

The extract was read with the unchanged Windows **v0.2.21** package and its
native `Session` / `Presentation` API. Source:
`cd83af8e7c6211be71cc2a82541738e0b1f2018f`. The application archive hashes are
in the [developer guide](../README.md#start-with-your-package).

| Lab | Fixture under `test/data/` | Scene | Extracted zero-based frames |
| --- | --- | --- | --- |
| Rigid bodies and chain | `multi_geom.ovd` | 3 | 1, 2, then 10 through 240 in steps of 10 (26 samples; 242 capture frames) |
| Joint drives | `articulation.ovd` | 3 | 1, 2 (2 samples; 4 capture frames) |
| Contacts | `2026_04_28_15_24_52_000000a751246bf6.ovd` | 5 | 1, 2, 4, 10, 30, 60, 100, 200, 300, 400, 500, 600 (12 samples; 604 capture frames) |

Each lab displays its capture SHA-256. `assets/physics-data.js` contains those
identities, schema, live objects, unique typed-value records, frame projections
and native geometry. Identical values are deduplicated, not approximated.
The older four-frame availability exercise uses `test_scene.ovd` separately.
These IDs belong to these fixtures; discover IDs again for your own recording.

### Rigid bodies

Pick a dynamic body to inspect `globalPose`, `linearVelocity`,
`angularVelocity`, `mass`, `massSpaceInertiaTensor`, damping, sleep thresholds,
flags and iteration counts. Follow `shapes` to a `PxShape`, then `geom` to its
geometry attributes. Shape properties also expose `localPose`, `contactOffset`,
`restOffset`, filter data and material references when recorded.

`globalPose` is an actor pose. It is not automatically the center-of-mass pose:
inspect `cMassLocalPose` too. The displayed velocity arrow indicates the recorded
linear-velocity **direction**, with a normalized display length. Read the vector
in the inspector for its magnitude. Small nonzero components retain scientific
notation instead of being rounded to zero.

### Articulations

The chain has links **31, 34, 38, 42**, connected by joints **35, 39, 43**.
Joint 39 is prismatic: its Y translation has `motion = eLIMITED`, with
`limitLow = -0.1` and `limitHigh = 0.1` on that axis. The other joint sections and
object buttons let you compare links, revolute joints and joint-local frames.

The drive fixture's joint **15** has twist free and five axes locked.
Its recorded drive arrays contain stiffness **1**, damping **2** and target
approximately **0.22**. It also stores limits on free/locked axes. These are
recorded parameters, not evidence that those limits are enforced or that every
drive is active. Check `motion`, `driveType` and the effort/force limits together.

The six slots are TWIST, SWING1, SWING2, X, Y, Z. Rotational coordinates use
radians; translations use the producer's scene units. Limits apply to limited
axes. Parent and child frames come from native overlay extraction; X/Y/Z use
red/green/blue, parent axes are dashed and child axes solid. Glyph lengths are
normalized for readability. See the [PhysX joint API](https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/_api_build/class_px_articulation_joint_reduced_coordinate.html)
and [axis definitions](https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/_api_build/struct_px_articulation_axis.html).

### Contacts

At frame **2**, scene **5** has two pairs with six contact points each:
dynamic actor **14** against static **11**, then dynamic **14** against static
**8**. The shape pairs are **16/13** and **16/10**. Normals are `[0, 0, 1]`;
the first separation is approximately **0.004361406** scene units. Normal
impulses differ between points. At frame **1**, the arrays are empty.

The point and normal arrays are recorded as flattened `PxF32[]`, grouped into
XYZ triples for visualization. `pairsContactCounts` partitions them by pair.
Raw `pairsActors` and `pairsContactShapes` contain ordered 64-bit producer
handles. The extract maps these through `session.evidence_metadata()` identities,
accepting a mapping only when unique. It does **not** infer pair ordering from
`view.references()`, which can expose membership without full order or multiplicity.
For this fixture every displayed contact reference resolves uniquely.

This lab reads every contact array from the labeled inspection frame. The native
viewer requires the matching **Pre-sim frame selected or in focus** to display
contacts, with the Contacts overlay enabled. It reads the following Post-sim
report: native Pre-sim frame 1 displays the contact report stored at frame 2.
Use **Go to Pre-sim** when the current frame is Post-sim. Raw frame-1 arrays
being empty therefore does not imply an empty native frame-1 contact overlay.
This behavior is implemented in `PVDSession/Overlay.cpp::ovdContacts` and the
application’s contact-frame guidance. Do not compare an unlabeled screenshot with a
different inspection frame. A positive separation can be within contact distance;
normal impulse is not force. No timestep conversion or SI-unit assumption is made.

## Decode a property from a coherent view

Use the packaged interpreter and loader in the
[tested headless example](../skills/ovdnext-inspect/references/headless.md#runnable-example).
With the example's loaded `session`, and IDs discovered from your capture:

```python
import struct

view = session.presentation(scene_id, frame, inspect_ids=[object_id])
instance = next(i for i in view.instances(0, 512) if i.id == object_id)
properties = {p.name: p for p in view.properties(instance.class_id)}
velocity = properties["linearVelocity"]
status = view.property_availability(object_id, velocity.id)
if status == "present":
    assert velocity.type == "PxVec3"
    assert view.property_size(object_id, velocity.id) == 12
    xyz = struct.unpack("<3f", view.read(object_id, velocity.id, 0, 12))
    print("linearVelocity", xyz)
else:
    print("linearVelocity", status)  # preserve absence; do not invent a zero

shape_property = properties.get("shapes")
if shape_property is not None:
    print("shape membership", view.references(object_id, shape_property.id))
```

The bounded 512-object page above is an example; use the headless guide's paging
workflow for a larger capture. Request another `Presentation` when selecting a
new object or frame. The attribute list shows exact decoded values and a
bounded hexadecimal preview for each property directly on the page.

## Representation and limits

- Geometry uses `geometry_snapshot()`: native actor/shape IDs, row-major transforms,
  primitive dimensions and captured mesh vertices/indices. Box dimensions are
  half-extents. Sphere radius and capsule radius/half-height remain separate from
  transforms; the capsule's long axis is local X. Analytic primitives are tessellated
  only for display. Mesh scale is already in the native transform.
- The illustrative grid is not captured geometry. Infinite planes are omitted
  from the finite preview. Camera fitting ignores them. The inspector is the
  authority for recorded values, not the preview's apparent scale or shading.
- `PxTransform` bytes are quaternion XYZW, followed by translation XYZ. Known
  primitives decode little-endian; flags use their actual stored byte width.
  Unknown representations stay undecoded with raw bytes available.
- Availability and object liveness are retained. `absent_at_frame` is not zero;
  an empty present array is not an absent sample. No missing value is interpolated.
- Reads are bounded to 64 KiB per property; raw previews to 64 bytes. Truncation
  is marked. Scene summaries and contact/joint tables use seven significant digits;
  the attribute list shows exact decoded values and raw previews. Per-property producer sample stamps were unavailable
  for these projections, so the guide reports the selected inspection frame only.
- The light-theme screenshot is the actual v0.2.21 Windows application. The browser
  lab is a separate documentation visualization, not the application's renderer.
  These documentation assets do not change the application or scanned archives.

## Concrete workflows

The mass/contact-offset walkthrough uses `multi_geom.ovd`, scene 3, frame 2:
body 11 `mass = 10` (`PxF32`, present); `shapes` references shape 10, whose
`contactOffset = 0.019999999552965164` (`PxF32`, present) and `restOffset = 0`.
The action buttons restore that frame and select the body or shape in the first
lab. The inspector's body summary exposes `shapes` without opening the raw dump.

The TCP workflow is drawn from the pinned application's `docs/TCP_SELECTION.md`
and `python/ovd_ui/application.py` / `widgets/capture_selection.py`. These are
instructions for the native viewer; this page does not open a listener or send
recording-selection requests.

Evidence side loading follows `docs/EVIDENCE_JSON.md` and `docs/EVIDENCE_SHOWCASE.md`
at application source `cd83af8e7c6211be71cc2a82541738e0b1f2018f`. The displayed
`right-handover-2` finding is the documented CPU control observation: seven points
at frames 74, 76, 78, 80, 82, 84, 86, +8.9907% relative energy change and a 1.3452%
diagnostic threshold. It is separate from this guide's extracted fixtures. No
intermediate frames, new measurements or solver verdicts are generated here.

### TCP recording replay

`assets/workflow-data.js` contains a compact extraction of the retained Linux
`test_authentic_selected_then_all_ack_frames_and_saved_payloads` evidence.
It records SDK sender revision `ad5c86f498025752261284367f0f1908d392a6b1` and
capture SHA-256 `f14133f6775d4e34c1b8b9bb85f7724efca2bccf38947464ee62a77257b285d5`.
The evidence report retains receiver acknowledgements and independently decoded
SDK writes; the original OVD was not re-read for this documentation change.
This is earlier interoperability evidence, not a new v0.2.21 TCP run or a GPU claim.

Discovery contains three catalog entries. Request 1 selects box with Primary:
epoch 1, applied, no completed frame yet at acknowledgement. Its later segment
contains box motion but no ground or mass write. Request 2 selects All + Secondary:
epoch 2, applied; its later segment contains ground and box mass 1. The replay
shows the last retained pose/velocity write in each segment, not an invented
universal frame or sample at the acknowledgement instant. The source report's
full hash and sender binary hash are preserved in the asset.

### Contact distance, rest distance and projection

The contact-offset diagram uses `multi_geom.ovd`, scene 3, frame 2:
box shape **10** and ground shape **7** both record `contactOffset =
0.019999999552965164` and `restOffset = 0`. These values are copied directly
from the existing native-decoded dataset into `workflow-data.js` with the
capture hash. The pair’s contact distance is the sum of the contact offsets
(about 0.04 scene units); its target resting distance is the sum of the rest
offsets (0). See the [PhysX PxShape definitions](https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/_api_build/class_px_shape.html).

The slider changes an **illustrative gap**, not the recording or a PhysX
simulation. The schematic assumes this pair is enabled for collision and shows
contact-margin overlap, not enlarged collision shapes or measured motion.
Contact generation can begin while surfaces are separated. It does not fix
their final resting separation at the contact distance.

The HTML’s Python mass/contact-offset excerpt was executed verbatim with the
unchanged v0.2.21 package. An additional read verified mass `not_projected`
without `inspect_ids`, then `present` / 10 with object 11 requested at scene 3,
frame 2.

Producer-side agent routing uses the existing
[PhysX OVD skill index](https://github.com/NVIDIA-Omniverse/PhysX/blob/7d6f6ac8b83e1610bdf715f1f2f1d6b718aee586/physx/SKILLS.md), especially `ovd-record` v0.1.2 at
reference revision `038f8fc29c57f84072faefb546a642d80fe80882`. A matching local
SDK copy remains authoritative for the producer being used. SDK CLI capabilities,
OVD Viewer's packaged Python API and viewer evidence import are distinct interfaces.

### Recording and ingestion diagrams

The opening diagrams explain OmniPVD's recorded objects, schema, lifetimes,
attribute updates and frame markers. Producer initialization, safe-point sampling,
late capture and explicit stream finalization follow the linked PhysX `ovd-record`
skill. OVD Viewer ingestion follows the pinned `python/ovd_ui/ingestion.py`:
asynchronous `CaptureOpener` for the UI and synchronous `load_capture` for headless
callers, then native Session/Presentation reads. The worked mass/contact-offset
values are measured from `multi_geom.ovd`; the SDK's separate six-step recorder
example is described as a command recipe, not claimed as executed here or as that
fixture's generator. TCP direction and request/epoch examples match the retained
receiver observations above.
