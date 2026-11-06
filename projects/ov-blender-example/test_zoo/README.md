# Blender → OVStage test zoo

This is a procedural regression and capability-discovery zoo. It uses the
existing add-on scene compiler, material conversion, light conversion, and
OVRTX composition interfaces. It does not supply another runtime or bridge.

The catalog currently covers mesh modeling, ten modifier families, linked and
collection instances, Geometry Nodes, curves/hair/text, Blender materials,
texture maps, primvars, direct USD OpenPBR, lights/world, cameras, animation,
visibility, rigid bodies/joints, color/presentation, and explicit challenges.
`catalog.py` is the executable catalog; `--list` prints the full manifest.
`existing_gates.json` maps the deeper physics, bake, action-slot, undo/persistence,
and SimReady workflows to their existing executable acceptance fixtures.

```sh
python3 public/test_zoo/run.py --list
python3 public/test_zoo/run.py \
  --blender /path/to/blender-5.2 \
  --addon-source /path/to/addon-checkout/public \
  --cases smoke --edits --output out/test-zoo/smoke
python3 public/test_zoo/run.py \
  --blender /path/to/blender-5.2 --cases all --edits \
  --output out/test-zoo/all --resume
```

Each case runs in a fresh Blender process, serially. Outputs contain the source
`.blend`, baseline and edited compiled/composed USD, Blender evaluation data,
converter diagnostics, a JSON report, and a local log. Every step names the
operation applied. Optional `--preview` produces a **Workbench authoring
preview**, never native OVRTX evidence. `--addon-source` permits testing another
checkout without copying or changing its runtime files.

For native frames and live generation edits, use the existing SDK environment:

```sh
/path/to/sdk/python public/test_zoo/native.py \
  --input out/test-zoo/all --output out/test-zoo/native \
  --addon-source /path/to/public-sdk-checkout/public --python /path/to/sdk/python \
  --cases mesh.cube,material.base_color,geometry_nodes.realized_instances --gpu 2
```

The native runner uses the add-on's typed client, records PID/epoch/stage identity,
retains every image, compares edits with an unchanged-scene noise control, and
records cleanup diagnostics. `native_frames_measured` means those measurements
completed; it does not certify every lobe's physical accuracy or physics behavior.

`export_passed` means the actual compiler and composition completed. It does
**not** mean the feature rendered correctly. `native_status` remains `not_run`
until a native measurement is recorded. A volume node, ignored socket, or
unsupported camera must not be treated as a pass merely because USD was saved.
`conversion_rejected`, compiler failures, and blocked fixtures retain exact
diagnostics; none silently substitutes a simpler feature.

## Material scope

`material.*` exercises Blender Principled and other Blender shader graphs
through the add-on converter. `texture.*` covers color/data image handling,
UVs, packed images, scalar channels, normals, coat normals, bump, displacement,
emission, opacity, and procedural graph probes.

`openpbr_direct.*` deliberately bypasses Principled conversion and authors an
explicit OpenPBR USD material on the exported subject. These cases cover
diffuse roughness, specular/anisotropy/IOR, transmission depth/scatter/dispersion,
subsurface, coat, fuzz, emission, thin film, opacity, thin-walled geometry,
coat normal, and selected combinations. They establish public renderer
capabilities separately from Blender mappings. Thin-film thickness uses the
OpenPBR micrometer unit; Blender's Principled socket uses its own convention.

## Limits and assets

The default frame is512×384; generated test maps are64×64. Downloads and
generated output each have a1GiB initial budget. Cases have bounded process
timeouts. Procedural cases need no downloaded model. A small optional CC0
studio HDRI can be fetched with `fetch_assets.py --output out/test-zoo/assets`;
the download is bounded to32MiB and records URL, license, size, and SHA-256.
No archive is extracted, and no external blend-file scripts are executed.

The existing `tests/blender_native_*`, bake, physics, and presentation scripts
remain authoritative for their detailed numerical gates. Catalog references
connect zoo scenes to those gates; zoo export alone does not replace them.

Primary documentation is indexed in `sources.json`. In particular, evaluated
Geometry Nodes output must be checked independently from its editable graph;
native renderer modes may support different OpenPBR lobes; and display color
changes must not be confused with changed scene-linear HDR values.

## Consolidated evidence and visual review

Preserve source runs in place: their USD compositions contain absolute paths.
Index corrected cases from a newer run without moving or overwriting the old
evidence. Later sources take precedence per case:

```sh
python3 public/test_zoo/collect.py \
  --sources out/test-zoo/all-root out/test-zoo/all-final \
  --assets out/test-zoo/assets/assets.json --output out/test-zoo/collection.json
/path/to/sdk/python public/test_zoo/native.py \
  --collection out/test-zoo/collection.json --output out/test-zoo/native-openpbr \
  --addon-source /path/to/public --python /path/to/sdk/python \
  --cases 'openpbr_direct.*' --gpu 2
python3 public/test_zoo/gallery.py \
  --native out/test-zoo/native-smoke out/test-zoo/native-openpbr \
  --collection out/test-zoo/collection.json --output out/test-zoo/gallery.html
/path/to/sdk/python public/test_zoo/contact_sheet.py \
  --input out/test-zoo/native-openpbr --kind native --limit 32 \
  --output out/test-zoo/openpbr-native-contact-sheet.png
```

The offline HTML gallery shows actual baseline and edit frames, per-edit image
differences, session identity checks and author/export gaps. Weak edit signals
are labeled for review. They are not automatically SDK failures: the fixture,
lighting, renderer mode, sampling noise and lobe support must be isolated first.
The image sheet carries the same qualification directly on each tile.

As measured on 2026-09-30, the consolidated 180-case authoring collection has
162 export passes, 5 compiler identity failures, 11 strict material-conversion
rejections, and 2 unimplemented dedicated fixtures (Grease Pencil/compositor).
All 32 direct OpenPBR cases, 3 representative mesh/material/Geometry Nodes
cases and 18 additional workflow cases produced native image sequences while
retaining the same process, epoch and native stage within each sequence.
Across these 53 cases, 33 have at least one edit whose difference did not exceed
the whole-frame noise threshold (19 OpenPBR and 14 other workflows). The gallery
marks these for review. The other 20 cases have measurable edit signals, which
still do not certify physical or feature correctness.

The additional workflow contact sheet visibly exposes several unresolved
cases: the Bezier subject is absent, hair is extremely thin, volume scenes
show a solid source-like cube, and hiding the object did not yield the intended
image change. Texture mapping, world lighting and camera controls also need
targeted checks. These are useful gaps, not native feature passes. The topology
and armature animation and modifier-stack sequences show stronger edit effects.
This is not a claim of full feature parity.

The connected Geometry Nodes bake and simulation graphs are authoring fixtures;
they do not claim a completed persistent disk bake. Rigid-body/joint zoo exports
must be paired with the numerical fixture gates in `existing_gates.json`.

## Render mode and weak-lobe interpretation

The existing add-on composition authors `RealTimePathTracing` on the render
product. `RenderRequest` has no selectable renderer-mode field. This zoo keeps
that baseline contract; it does not silently substitute Interactive Path Tracing.
New native reports record the actual request defaults, translated settings,
authored mode evidence and samples per call. Authoring reports also record the
composed render-product mode. Text-layer mode evidence from older exports is
explicitly distinguished from a native engine-mode readback.

The historical `dlss_enabled=False` request maps to the add-on's Performance
preset, not a proven DLSS-off state. Neither that name nor an accepted setting
write establishes the native renderer's effective configuration.

[NVIDIA's OpenPBR documentation](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/OpenPBR.html)
states that RTX Real-Time2.0 does not support sheen or thin film, that SSS needs
enabling, and that transmission may require more bounces. It describes full
OpenPBR support in Interactive Path Tracing. These are mode-specific documents,
not proof of an exposed alternate-mode route in the standalone SDK.

The direct fixtures already activate their parent coat, fuzz, transmission,
subsurface and thin-film weights. No inactive-parent-weight defect was found
that would justify rewriting the existing weak-signal cases. Stronger tests
should use independent versioned outputs, close framing, targeted illumination,
adequate sampling and the saved no-edit control image. Until measured, the
weak-signal cases remain review gaps rather than certified lobe support.
