# Omniverse Libraries Playground 0.14 preview

Explore Omniverse libraries through interactive, real-world examples, then use
the source to build on what you learned.

## Highlights

- The initial release has eight active library demonstrations. The campus
  preview keeps two unused galleries with Coming soon plaques and removes the
  arcade cabinet. Source and installer inventories omit the deferred integrations.
- Updated Pillow to 13.0.0 and refreshed its Windows lock hashes and notices.
- Updated the application, setup window, guide greeting and procedural entrance
  banner to the Omniverse Libraries Playground name. Existing repository/package
  identifiers, launch scripts and saved-data locations remain compatible.
- Added a [source map](SOURCE_MAP.md) connecting all eight public library
  demonstrations to their runnable examples and implementation modules.
  OVStream now has a runnable frame-and-return-command example in its code viewer.
- Reduced redundant transform traversal while retaining inherited transforms,
  instance prototypes and the reset-transform contract. Topology changes invalidate
  cached membership; source geometry, materials and rendering quality are unchanged.
- Retained sensor preparation during loading after a fresh-install check exposed
  a stall when initialization overlapped live rendering. Initialization logs now
  identify the phase; recovery shows elapsed preparation time. There is one
  outstanding sensor request, and completed images are presented before further work.
- Sensor pose submission includes moving meshes, robot/target roots and sensors;
  it no longer resends every static intermediate node in the asset hierarchy.

## Release scope and practical limits

This source edition includes OVRTX, OVPhysX, OVStage, OVUI, OVStorage, OVStream,
OVNewton and OVPackage. Default setup uses the current R48 campus and robot
guide from separately versioned, checksum-pinned archives. They are not stored
in Git history. The launcher downloads and verifies these archives automatically
from the public GitHub release. Manual installation is described in
[scene asset setup](SCENE_ASSETS.md). Missing artwork stops setup instead of
selecting the old procedural scene.

Windows x64 with an NVIDIA RTX GPU is the validated platform. Testing has used
one RTX PRO 5000 Blackwell laptop with 24 GB of GPU memory. Lower-memory devices,
different GPU generations and physical mixed-monitor handoff remain unvalidated.
First launch can take time to prepare shaders and sensors. The multi-sensor
scene uses additional GPU memory. Live performance remains below a consistent
30 fps on the reference laptop at Balanced quality; recorded video cadence is
not an interactive frame-rate claim.

This is an educational sample. Robot routes and fleet movement are scripted,
sensor calibration is illustrative, streaming is local shared memory and storage
uses a local file backend. Physics experiments use their native solvers.

Keep the runtime folder when updating the source to retain preferences and saved
revisions. See the [customer guide](CUSTOMER_GUIDE.md) for setup and recovery.
The package's `config/distribution.json` records its build identity, and
`config/sample-files.json` lists the curated source inventory. Packaged exports
also include a `config/package-manifest.json` with each source file's SHA-256.

The libraries and artwork retain their respective terms. See
[licensing](LICENSING.md) and [third-party notices](../THIRD_PARTY_NOTICES.md).
