# Changelog

## v0.2.21 (2026-11-10)

- Show and inspect recorded contact vectors with correct counts.
- Compact single-shape actors to geometry rows; retain shape-property access and material links.
- Explain unresolved material references instead of showing empty material folders.
- License the viewer under Apache-2.0 and include its license and PVD source notice in packages.

## v0.2.20 (2026-11-09)

- Open OVD integration 4.0 recordings from current PhysX builds while retaining 3.x compatibility.
- Resolve joint overlays through core constraints and verify file, live TCP, and headless ingestion.
- Remove internal-preview boilerplate from package instructions while retaining component notices.
- Retire Qt/Qwt and obsolete minizip paths; retain native and OvUI regression coverage on Windows and Linux.
- Replace Boost with a small native Windows/Linux TCP helper and remove its vendored source tree.

## v0.2.19 (2026-11-09)

- Upgrade portable runtimes to CPython 3.12.16 and codec-minimal Pillow 13.0.0.
- Verify updated runtime provenance and licenses while retaining Expat 2.9.0 checks.

## v0.2.18 (2026-11-06)

- Choose live recording scope and Primary/Secondary policy from an explicit, paginated OvUI panel.
- Keep recording selection bound to the live producer while inspecting older captures; clear stale inspection state when a new segment is displayed.
- Verify particle, diffuse, and deformable selections against genuine GPU simulation payloads.
- Animate selected deformables and their velocity arrows while preserving legacy captures and shared-mesh isolation.
- Keep recordings usable in compact windows and run hidden-window GUI regressions in Windows CI.
- Isolate Linux GUI input and size hidden Windows test windows explicitly.
- Include the required MSVC runtime in legacy Windows packages.

## v0.2.17 (2026-11-06)

- Add bounded, capability-gated TCP recording-selection APIs and catalog discovery.
- Validate request acknowledgements and preserve every completed segment in saved live recordings.
- Verify native control against genuine pinned SDK producers on Windows and Linux; viewer controls follow separately.
- Avoid shutdown hangs when frame pacing expires during application exit.

## v0.2.16 (2026-11-03)

- Record bounded CPU rocking-block controls and export source-verified energy windows with explicit timestep sensitivity and diagnostic limits.

## v0.2.15 (2026-11-03)

- Inspect imported SDK comparison evidence and analytic energy windows beside the timeline.
- Navigate verified source samples with explicit capture roles, scenes and clips while retaining capture and sample provenance.

## v0.2.14 (2026-10-12)

- Reject malformed OVD payloads that use parser class aliases, normalized attribute names, or retained parent-link fields to bypass input validation.

## v0.2.13 (2026-10-11)

- Include verified Windows and Linux Python dependency notices, including the bundled hash implementations.
- Exclude the unused Linux Berkeley DB extension and Windows Tcl-only zlib DLL from portable packages.

## v0.2.12 (2026-10-09)

- Update portable Windows/Linux XML parsing to Expat 2.9.0 while retaining the Python 3.12 ABI required by OvUI.
- Pin and verify portable interpreter inputs, preserve runtime build provenance, and reject packages with an unexpected XML parser version.
- Keep Linux packages compact by excluding Python runtime debug data.
- Include the Expat license notice on both platforms.
- Make Linux source setup reusable and explain how to replace outdated or incomplete Python environments.

## v0.2.11 (2026-10-08)

- Prevent Windows startup from loading MIME mappings from untrusted default files; retain built-in mappings and Windows registry associations.
- Exclude unused OvUI agent and inspector command-line tools from portable packages, removing their permissive decoding of remote Base64 responses.

## v0.2.10 (2026-10-06)

- Disable face culling by default. Reset unmarked Normal culling saved before v0.2.10 to None; preserve None, Inverse and subsequent choices unless an older version rewrites rendering settings.
- Draw debug arrows over bodies and heightfields while keeping contacts depth-tested. Occluded arrows do not intercept clicks on bodies.
- Capture live streams in memory; auto-save only useful completed captures. Empty or rejected connections remain memory-only and cannot be exported.
- Keep listening through autosave failures and preference changes; report recordings-folder and startup errors without hiding earlier warnings.
- Run Windows launcher and Linux clipboard checks in CI; show failing test names and locations, setup errors, skip reasons and missing clipboard-tool diagnostics.
- Give Linux desktop checks enough virtual screen space for the viewer window.
- Check release versions against each PR's base, including stacked changes.

## v0.2.9 (2026-10-06)

- Update the portable Windows/Linux Python runtime to 3.12.15 for current security fixes.
- Build Pillow without optional external codec libraries, including OpenJPEG/JPEG2000, with pinned static zlib-ng for PNG; verify icons, RGBA conversion and color parsing in built and extracted packages. Built-in Pillow decoders remain.
- Reject unrecognized Pillow feature names during package verification.
- Retire the historical Linux wheelhouse packager; use the portable packager with the prepared application runtime.

## v0.2.8 (2026-09-28)

- Keep camera drags active across workspace panels and floating windows; retain the initiating mouse button and cancel safely on Escape, focus loss or capture replacement, without changing OvUI.

## v0.2.7 (2026-09-18)

- Explain OVD contact visibility and offer an explicit jump to the matching Pre-sim frame within the current clip, retaining All stepping or switching Post-sim stepping to Pre-sim.
- Hide legacy analysis tabs for OVD captures or when no capture is open; retain legacy analysis and generated property graphs.
- Add file modification dates and newest/oldest sorting to the capture picker; remember the Open capture sort choice across runs.
- Highlight selected planes visibly in both themes without changing picking, and explain their pose-derived normal in the inspector.

## v0.2.6 (2026-09-18)

- Gate releases on separate Windows/Linux jobs that verify and launch the downloaded ZIPs, including payload integrity, Unicode relocation, capture loading and OvUI rendering checks.

## v0.2.5 (2026-09-18)

- Fix Linux viewport startup on runtime-only systems without OpenGL development packages, retaining the existing EGL/GLX context.

## v0.2.4 (2026-09-18)

- Fix Windows startup from Unicode installation paths without changing numeric or date formatting.

## v0.2.3 (2026-09-17)

- Check for Linux's required unzip tool before starting portable-package builds; document safe draft-release recovery and strict version-gate protection.

## v0.2.2 (2026-09-17)

- Restored retained live-capture history in OvUI, with individual in-memory OVD exports and removal; automatic disk saving defaults OFF while preserving explicit preferences.
- Keep the active capture visible in Recordings with its current byte count and stable selection through completion; allow the main Save command to freeze and export it with autosave off.
- Exposed scene-owned joints in the OvUI inspector, preserving inherited properties, world endpoints, reused handles and frame lifetimes; corrected converted scalar-reference byte sizes.

## v0.2.1 (2026-09-17)

- Package the Qt-free OvUI application as portable Windows and Linux ZIPs with matching checksums, published size/provenance reports and Linux executable permissions.
- Require extracted-package GUI checks on both CI platforms and report missing host graphics-driver libraries before loading OvUI.
- Document the current-driver baseline and remaining customer-distribution approval and validation gates.
- Require a new matching version on every main-bound PR and allow interrupted private release uploads to resume without replacing published assets.

## v0.2.0 (2026-09-15)

- Kept layout controls usable after preference-save failures and live-follow clip lists stable across trim, split and scrub/follow cycles.
- Supported Unicode capture, backing and recording paths on Windows; kept Linux compiler runtimes host-provided for graphics-driver compatibility.
- Added Qt-free Windows and Linux internal preview packaging with native dependency checks, relocated runtime probes and separately scoped Qt oracle verification.

- Restored the existing OvUI workspace and ModernGL renderer on the Qt-free native runtime, retaining inspection, playback, selection and independent overlay controls.
- Added explicit Windows Unicode clipboard and desktop-handler boundaries without Qt; retained Linux desktop handling and separate Qt reference verification.
- Integrated the native PVDCore/PvdDom session runtime for OvUI without Qt application dependencies; the Qt frontend remains an explicit legacy verification option.
- Preserved checked atomic exports and in-memory live recording, including source-alias protection when saving captures.

## v0.1.16 (2026-09-15)

- Enabled legacy frame-range export from the selected scene, preserving the starting snapshot and recorded origin shifts.
- Publish saved clips only after checked output completes; cancelled or failed saves preserve existing destinations and release temporary clip data.

## v0.1.15 (2026-09-10)

- Reserve versioned publication for the OvUI workflow while retaining legacy verification and explicitly enabling its shared-data test targets.
- Enforced the released session-clip and headless UI regressions in both GitHub platform suites.
- Pinned GitHub Actions, Windows Python dependencies, and Qt archive bytes; aligned both Windows verification toolchains and retained complete success logs.
- Excluded unpublished object creations from the recorded catalog, made repeated identical class declarations idempotent, and named capture translation failures.
- Required explicit release-publication opt-in and made CI artifact retries and main-branch run scheduling reliable.
- Removed the internal-network toolchain requirement from GitHub Windows verification and packaging while retaining the pinned compiler and Qt versions.
- Added an encrypted repository-secret fallback for pinned private Physics checkouts in trusted GitHub automation.
- Configured GitHub Actions to gate Windows/Linux packages and opt-in releases on the full coexistence verification matrix.
- Preserved the recorded class, attribute, and object model in every immutable OvdData frame view.
- Added recorded class and attribute selectors for retaining only the property data a consumer requests.
- Made the real-capture probe report deterministic alive-object counts by exact recorded leaf class.
- Kept the presentation-neutral OvdNext data path alongside the existing Qt application path.
- Added a Qt-free OvdNext real-capture probe that loads validated `.ovd` files into immutable, presentation-neutral OvdData frame views without Qt, OvUI, or OvRTX.
- Retired GitLab pipelines in this GitHub tree; GitHub Actions continues the released OvdNext history.

## v0.1.14 (2026-09-09)

- Kept live captures as session clips in memory, with switching, inspection, removal, and individual OVD export; automatic disk saving now defaults OFF.
- Preserved earlier clips across New, reset connections, and rejected live previews.
- Added Windows and Linux regression coverage for repeated live captures, empty or rejected reconnects, recording retention, and listener rearming.

## v0.1.13 (2026-09-03)

- Rearmed live listening after every capture and replaced the displayed scene only when the next producer supplied a complete frame; File > New is no longer required between runs.
- Extended the connection switch to OVD so users can keep the current capture and block automatic producer reconnects.
- Added a persistent Recordings tab with one timestamped `.ovd` per socket connection, background capture while browsing history, and follow-latest selection.
- Added an ON-by-default Keep live recordings switch; empty or rejected connections are identified without replacing a useful scene.
- Updated the pinned PhysX/PvdDom dependency to current trunk.

## v0.1.12 (2026-08-31)

- Started listening for live OVD producers automatically on normal launch.
- Added File > New (Ctrl+N) to discard the current capture and rearm live listening.
- Kept fast live streams stable when schema or lifetime updates follow the final frame.
- Updated OVD file loading for the latest PhysX trunk OmniPVD stream API.

## v0.1.11 (2026-07-23)

- Kept each GitLab Release page focused on that version while retaining the complete cumulative changelog in both runtime packages.

## v0.1.10 (2026-07-23)

- Added cumulative release notes to GitLab Releases and both runtime packages.
- Automated semantic release tags from the changelog: patch bumps are the default, while an intentional minor bump advances the minor component and resets patch to zero.
- Retained the exact merge commit SHA in release metadata, checksums, and immutable package coordinates.

## 2026.07.21-mr4: MR !4 — Fix release API job-token authentication (2026-07-21)

- Switched Release and Release Link API operations to explicit job-token authentication, including paginated link discovery and credential-safe HTTPS redirects.

## 2026.07.21-mr3: MR !3 — Fix release jobs on Titan runners (2026-07-21)

- Made the pinned GitLab CLI image work on Titan by disabling inherited Kubernetes entrypoint handling.

## MR !2: Add OvdNext GitLab CI/CD pipeline (2026-07-21; merged milestone, no GitLab Release)

- Added pinned Windows/Linux verification, deterministic packages, checksums, individual test reporting, and automatic merged-MR publishing.

## MR !1: Harden multi-segment OVD/PVD ingestion (2026-07-21; merged milestone, no GitLab Release)

- Hardened file and live ingestion, separate live captures, segment-local references, stable live poses, resource budgets, archive confinement, cancellation, and malformed-input handling.

## v0.1.09 (2026-07-15)

### Live OVD attach

- Attach to a running PhysX producer over TCP and watch the capture grow live: ovdnext listens as a TCP server (File > Attach), accepts on a worker thread so the UI never blocks, follows the newest frame, and can Save-to-.ovd. Detach or the toolbar Disconnect button stops the stream.

### Multi-segment captures

- A recording window = one stopSampling/startSampling cycle; each restarts at frame 1 and re-snapshots full state. Since every window restarts at 1, the windows share the frame range and OVERLAY on one timeline (they do not concatenate) -- a two-window capture shows two PxScene nodes, both present across the timeline. The viewer resolves each window's references within that window (PxScene.actors / actor.shapes no longer collapse onto the last writer and pull in the other window's objects). File and live.

### Scene tree

- Each PxScene nests under its owning PxPhysics (resolved via the ancestor chain, correct even when a later segment omits PxPhysics.scenes).
- Opening a capture auto-selects and expands the first scene.
- PxArticulationJointReducedCoordinate shows as a child of its articulation.

### Camera and picking

- Clicks select in every format now: CPU ray pick replaces a GPU id read that returned the background on some NVIDIA drivers for PXD2/PVD3.
- Fixed detectUpVector aborting on every PXD2/PVD3 open.
- Double-click focus / Zoom-to-Selected pivot correctly on off-center convex fragments; pick-to-focus no longer repels the camera.
- Mouse-wheel zoom no longer teleports on large captures. Space = play/pause.

### Visualisation

- OVD joint frame gizmos are visible (scene-sized, depth-test off).

### Internals

- Segment-consistency + live multi-segment regression tests and a two-segment resample fixture. Suite: 85 passed, 1 skipped, 0 failed.

## v0.1.08 (2026-06-01)

### Scene tree / Visibility

- Per-scene visibility now correctly ISOLATES scenes for every object type: hiding one PxScene hides exactly its objects and leaves the others; hide all to clear. Owning scene is resolved from the capture's object tree (parse-time parent chain), not the PxScene "actors" handle lists -- so it stays correct when a capture reuses a PhysX pointer across scenes (two sims into one .ovd, actor moved scene-to-scene). v0.1.07 could only approximate this.
- Multi-scene captures overlay correctly: every distinct object from every scene renders on the shared timeline (a two-run capture shows both runs).

### Internals

- Added multi-scene unit tests + small committed fixture; fixed a stale aliveness test. Suite: 80 passed, 1 skipped, 0 failed.

## v0.1.07 (2026-05-21)

### Scene tree / Vis column

- Multi-scene captures: hiding one PxScene no longer over-hides actors shared between scenes (first-pass fix: render a deformable iff at least one owning scene is visible; superseded by v0.1.08's true isolation).
- Orphan-deformable filter for bodies not referenced by any scene.

### Camera

- Orbit-zoom "invisible wall" removed (dropped the hard orbit-radius floor); small-radius zoom-crawl still tamed via an additive step.

### Visualisation

- F (frame selection) and Zoom-to-Selected now work for deformable volumes and surfaces (their triangles now contribute world bounds).

## v0.1.06 (2026-05-15)

### Scene tree

- PxArticulationJointReducedCoordinate navigable as a child of its link.

### Visualisation

- Deformable volume simulation mesh renders solid (was wireframe).

### Camera

- Orbit zoom + rotation crawl fixed (min world-unit zoom step).

Note: the v0.1.05 Windows binary had shipped a day-stale build; 0.1.06 ships freshly rebuilt HEAD on both platforms.

## v0.1.05 (2026-05-13)

### Visualisation

- PxDeformableVolume renders BOTH meshes (collision solid, sim as wireframe), each with its own Vis-column handle.

### Scene tree

- Top-level order normalised: PxScene at top, PxPhysics at bottom.
- PxAggregate re-homed under PxPhysics (via aggregates list); PxOmniPvdMetaData manually re-homed under PxPhysics.

### Bug fixes

- Pre-sim frame leaked onto post-sim mode at file open -- fixed.
- Stray "ye" typo in instanceview.cpp removed.

## v0.1.04 (2026-05-08)

### Visualisation

- Per-actor world AABB from PxActor.worldBounds (broadphase-accurate), replacing the per-shape OBB boxes that tilted with rotation.
- Per-actor centre-of-mass cross marker (gated on the Bounding Box toggle).
- Wireframe ground grid renders before the OIT pass (correct through-glass).
- Transparency slider rewritten: fixed inverted WBOIT blend (monotonic fade), "(none)" hint for empty bitflags, tagged objects always route through OIT.
- Gizmo size scales with scene characteristic length, not tolerance length.
- Contact gizmos render at pre-sim with +1 lookahead (align with bodies).
- Configurable colour palette via live-reloaded colors.json.
- Pre/Post-sim frame-mode toggle for OVD captures.

### Property panel

- Full inheritance chain shown; enum-vs-bitflag disambiguation; PxTransform/PxBounds3 sub-rows; slider window for long arrays; paginated jump-lists for PxPhysics factory subtrees; spreadsheet-style column resize.

### Scene tree

- Live search field with prev/next match cycling; alive-set-signature short-circuit replaces the per-frame rebuild (snappier scrubbing).

## v0.1.03 (2026-04-28)

### Visualisation

- Per-object transparency ("Trsp" column + "Tagged" dropdown mode); Weighted-Blended OIT replacing sorted-alpha (no z-fighting / floor bleed); log transparency curve; scene-scaled auto near-plane; wider camera-speed range.

### Property panel

- "Type" column with PhysX type names; Translation/Rotation rename; float precision 3->6; fixed-size arrays expand per-element; ObjectRef resolves to "<name> (<Class>) [id]"; named shape-flag bits; wider column defaults.

### Scene tree

- Header row visible + left-aligned; Type column widened to 160 px.

### Behaviour / internals

- Up-vector detection skips empty placeholder scenes; contact keyframe-range lookup (fixes byte-identical contacts vanishing after frame 1); settings scope split to "ovdnext"; less console spam; 4 new pvddom tests.

## v0.1.02 (2026-04-23)

- Removed the duplicative bottom-left status-bar frame text (top-right LCD + Pre/Post parity button already carry it). First broadly-shared beta drop.

v0.1.01 and earlier predate the packaged READMEs; see git history.
