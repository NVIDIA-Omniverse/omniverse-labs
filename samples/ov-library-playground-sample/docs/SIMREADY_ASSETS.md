# SimReady exhibit assets

The Lab uses NVIDIA's public [SimReady dataset](https://huggingface.co/datasets/nvidia/simready-assets) for its warehouse props. Setup installs a verified archive of prepared models and textures before the UI starts. It does not download or convert assets when a visitor enters a room.

| Prop | Source asset | Adaptation |
| --- | --- | --- |
| Roller conveyors | conveyor_belt_a05 | Uniformly scaled complete sections; repeated to form lanes; a steel bridge with adjustable end supports |
| Cartons | box_cardboard_a01, box_corrugated_brown_b04 | Two packaging finishes; dimensioned to the experiment's collision envelopes |
| Pallet | pallet_wood_recycled_a05 | Dimensioned for the operator station |
| Warehouse rack | rack_large_empty_a01 | Grounded rack with measured shelf contact heights; empty, pick-ready and replenished stock configurations |
| Cargo robot | equipment_robot_trolley_b01 | Shared geometry; UV-aware simplification; detachable shelf-and-bin module |
| Operator station | vehicle_forklift_blue_a01 | Pallet stacker; kinematic fork control within the source's 1.51 m joint travel |

## Install and repair

The normal setup and Repair flows call `tools/setup_simready.py`. To install just the props, use Python 3.12 or the library environment's Python:

```powershell
& "$RuntimeRoot/libraries-env/Scripts/python.exe" tools/setup_simready.py --runtime-root $RuntimeRoot
```

Setup downloads the pinned ZIP from the public URL in `config/prepared-exhibit-release.json`. For manual installation, place that ZIP in `data/scene-archives/`, or pass `--archive PATH_TO_ZIP`. The archive is approximately 302 MiB, compared with approximately 840 MiB for the original source downloads. Extracted assets occupy approximately 323 MiB.

Setup verifies the archive, its manifest, and every installed file. Later launches reuse verified models. **Repair and retry**, or `--repair`, checks every file's SHA-256 and restores damaged or missing files from the same archive. Replacement is staged and verified before touching existing artwork. Prior files are preserved under `data/models/.simready-backups/`; a failed download leaves them untouched. The installer uses only the Python standard library and never installs or imports a mesh-processing tool.

## Provenance and adaptations

`config/simready-assets.json` records dataset revision `cd8a53956ab42cda6281896e4eacec0e43739252` and the original files' sizes and SHA-256 or Git blob digests. The prepared archive retains the original CC-BY-4.0 license, attribution, per-asset license notices and adaptation metadata. USD and textures install under `data/models/simready/`, which is excluded from Git.

The prepared cargo robot and conveyor preserve material groups and texture seams while reducing dense geometry. Other props retain their source geometry. Standard USD Preview Surface networks are retained; textures are capped at 1K for small props and 2K for equipment, including UDIM tiles. File-backed USD references preserve relative texture anchors. The model geometry and materials match the existing exhibits. Each model's
`assetInfo.identifier` records an immutable URL at the pinned dataset revision;
local authoring paths are not required to load it. Texture references remain relative.

PyMeshLab 2025.7 was used privately during asset authoring. Its package, lock file and conversion scripts are excluded from the public sample and are not invoked by customer installation, repair or runtime. Historical authoring use remains disclosed in the release-review inventory; the prepared models retain the source assets' licenses and attribution.

## Simulation and controls

Imported simulator APIs are removed from the visual derivatives. The Lab continues to author the bodies, masses, contact settings and simple collision envelopes used by its live experiments. A detailed carton shell follows the same 2 kg rigid body as its box collider. Roller sections keep circular rollers through uniform scaling. The operator stacker is a scene-control example driven by shared OVUI value models; its fork movement does not claim hydraulic or load-capacity simulation. Fleet travel remains scripted, with real scene instancing and updates.

OVPackage collects referenced USD files and all material dependencies, including tiled textures and attribution. It prepares the dependency bundle off the UI thread, then hands it to the library's existing publish/install/verify flow.

## Attribution

Source assets: NVIDIA, SimReady Assets, **CC BY 4.0**. Each selected source package declares this license. Setup retains the supplied license and records the source revision and modifications in the prepared asset directory. Packaged demonstrations include those notices. These permissions cover the selected props; they do not replace the separate campus-artwork or robot-guide distribution decisions.

For a diagnostic comparison with the earlier procedural props, set `OVLAB_SIMREADY=0` for that process only. Normal launches use the prepared assets.
