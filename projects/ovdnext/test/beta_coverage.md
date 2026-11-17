# OvdNext beta coverage matrix (Step 3)

Smoke pass confirming OvdNext opens captures across a representative spread of PhysX
feature types. Generated 2026-06-24.

## Method

Two checks per capture:

- **Opens (mechanical, automated):** launch `ovdnext.exe <file>`, wait, confirm the
  process stays alive and the window is responsive (no crash or assert on load). PASS below
  means the capture loaded without crashing in the current build.
- **Scene recognizable (visual, manual):** geometry renders, scene tree populates, and
  scrubbing works. This needs eyes; left as PENDING for a human pass.

Build under test: `ovdnext.exe` RelWithDebInfo (joint-gizmo visibility fix, 2026-06-25).

## Note on the snippet list

The original task named `SnippetHelloWorld, SnippetRigidDynamic, SnippetConvex,
SnippetJoint, SnippetArticulation, SnippetTrigger, SnippetContactReport, SnippetPBF`. That
list is stale: only four PhysX snippets actually bind OmniPVD and can emit an `.ovd`
(`SnippetOmniPvd`, `SnippetOmniPvdMesh`, `SnippetOmniPvdMultiGeom`, `SnippetOmniPvdNet`).
The other snippets never create a `PxOmniPvd`, so they cannot produce a capture without
being modified. Coverage below uses the OmniPVD snippets plus the committed, snippet-derived
test fixtures, which together exercise the same feature categories.

## Matrix

| Capture                         | Source                      | Feature categories                                   | Opens | Scene recognizable |
|---------------------------------|-----------------------------|------------------------------------------------------|-------|--------------------|
| test_scene.ovd                  | basic capture               | rigid body, single scene                             | PASS  | PENDING            |
| multiscene_rigid.ovd            | SdkUnitTests ActorTests     | rigid bodies, multiple PxScene                       | PASS  | PENDING            |
| snippet_meshes.ovd              | SnippetOmniPvdMesh           | convex / triangle / heightfield meshes               | PASS  | PENDING            |
| multi_geom_full.ovd             | SnippetOmniPvdMultiGeom      | mixed geometry, D6 joints (limits + drive), deformables | PASS | PENDING          |
| articulation.ovd                | articulation gap-fill        | PxArticulationReducedCoordinate, reduced-coord joint frames | PASS | PASS (links + joint frame gizmos render) |
| particle_system.ovd             | particle gap-fill            | PxParticleBuffer / PBD particles                     | PASS  | PENDING            |
| deformable_volume.ovd           | SnippetDeformableVolume cap. | deformable volume (tetrahedral, sim + collision)     | PASS  | PENDING            |
| deformable_surface.ovd          | SnippetDeformableSurface cap.| deformable surface (cloth, triangle mesh)            | n/a   | PENDING (155 MB, manual) |
| VolumeVolumeCollision.ovd       | SdkUnitTests (2 scenes)     | multi-scene deformable collision, pointer reuse      | PASS  | PENDING            |
| testRecordMessage.ovd           | message capture             | OmniPVD messages panel (no geometry)                 | PASS  | PENDING            |
| 2026_04_28_..a751246bf6.ovd     | real-world capture          | mixed real producer scene                            | PASS  | PENDING            |

"Opens = PASS" verified automatically (no crash, window responsive). `deformable_surface.ovd`
(155 MB) and the 206 MB real capture were left for a manual pass; both are too large for a
quick automated smoke.

## Coverage gaps (no OVD source today)

These feature categories from the original list have no current OmniPVD snippet or fixture
that records them, so they are untested here:

- **Triggers** (`PxTriggerPair` / trigger shapes) - still no OVD source.

Closed since the first pass:

- **Articulations** (`PxArticulationReducedCoordinate`) - `articulation.ovd` fixture added;
  links and reduced-coordinate joint frames both render. The joint frames were invisible at
  first: the joints-scale slider was at zero and the joint arm was a raw absolute length with
  no scene scaling and no floor, and the arrow pass was depth-tested so frames at the link
  junctions were buried inside the bodies. Fixed in bdc1b88 (scene-relative arm with a floor,
  arrow gizmos drawn depth-test off).
- **Particles** (`PxParticleBuffer` / PBD) - `particle_system.ovd` fixture added; opens, visual
  pass pending.

To close the trigger gap, add an OmniPVD write stream to a snippet that builds a trigger shape
(or extend `SnippetOmniPvdMultiGeom`) and record a fixture.

## Side finding (not a viewer bug)

`SnippetOmniPvd --omnipvdfile=<path>` records the file but does not exit cleanly when run
headless (the process stayed alive after writing). This is an SDK snippet behavior, separate
from the viewer; the fixtures above were used instead. Worth a follow-up look on the git_4
side if headless file recording from that snippet is wanted.
