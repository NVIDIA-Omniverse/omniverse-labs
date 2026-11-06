# Scene assets

The default experience uses the R48 campus, its matching navigation data, the
lighting corrections and the robot guide. Detailed exhibit models come from
NVIDIA's SimReady dataset. These archives contain artwork, not SDK installations
or visitor data.

## Automatic setup

Run `scripts/launch_ov_library_lab.bat`. The launcher downloads the three archives
from the versioned [GitHub release](https://github.com/NVIDIA-Omniverse/omniverse-labs/releases/tag/ov-library-playground-assets-v0.14.0-rc.2), verifies their SHA-256
hashes and manifests, and installs them before the app starts. No GitHub account,
Git LFS or geometry-conversion tools are required. Interrupted downloads resume;
later launches reuse verified local files.

The campus installs under `data/campus/`, its navigation revision is activated,
and the guide installs under the selected runtime's `assets/lab-guide/` folder.
Prepared exhibits install under `data/models/simready/`. These downloads and
extracted assets are ignored by Git. See [SimReady assets](SIMREADY_ASSETS.md)
for model provenance and repair behavior.

The exact filenames, sizes, URLs and archive/manifest hashes are recorded in
`config/scene-release.json` and `config/prepared-exhibit-release.json`.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| nvidia-campus-r48-portable-provenance.zip | 986658860 | ec6127d444580c72dccfe6892b76e273fc821450483c4e6f57a049db744c99a6 |
| lab-guide.zip | 17921100 | 26acec23057f3616e8cecc540b60732b2d67725c3a7da3834d3ffa8975c1baa0 |
| OV-Library-Lab-prepared-exhibits-portable.zip | 317011550 | 38a98cedb508def6492cd30346b3415404bb0a41ace1ffe46fbf55f676000f39 |


## Manual downloads

Download all three ZIP files from the release's **Assets** list. Place them in
`data/scene-archives/` under the sample, without extracting them, then run the
regular launcher. It verifies and installs these local archives using the same
checks as automatic downloads. If a checksum does not match, move that file
aside and download the pinned version again.

With Python 3.12 already available, the asset installers can also run directly:

```powershell
python tools/setup_scene.py --archive "PATH_TO_CAMPUS_ZIP" --guide-archive "PATH_TO_GUIDE_ZIP"
python tools/setup_simready.py --runtime-root .ov_library_lab_runtime --archive "PATH_TO_PREPARED_EXHIBITS_ZIP"
```

If using a different runtime folder, pass the same `--runtime-root` to the
installers and `-RuntimeRoot` to `tools/launch.ps1`. Existing edited campus and
guide artwork is preserved; verification reports changes instead of overwriting
them. Prepared-exhibit replacements preserve prior files in a backup directory.

## Provenance and authoring

The campus archive includes `provenance/polyhaven/`: the original provenance
records for Oak Veneer 01, Forest Floor and Rocky Terrain 03, the CC0-1.0 license,
attribution and texture adaptation details. Copies are included in the sample's
[campus provenance folder](../data/campus-provenance/polyhaven/ATTRIBUTION.md).
The archive manifest covers these records along with the artwork. Keep the
provenance folder with the campus when distributing or copying it.

This package preserves the scene, textures and navigation from R48. Its distinct
`nvidia-campus-r48-portable-provenance` installation directory lets
existing campus installations remain intact. Campus geometry-source labels use
project-relative provenance paths. Prepared models identify their source with
immutable dataset URLs; neither archive depends on an author's workstation.
Geometry, materials, texture paths and navigation are unchanged.

`tools/package_campus_provenance.py` adds the Poly Haven supplement to the original
R48 archive. `tools/package_portable_assets.py` accepts that reviewed archive and
the original prepared-exhibit archive, verifies their hashes, rewrites only the
reviewed provenance fields, and produces new archives and a verification report:

```powershell
python tools/package_portable_assets.py --campus PROVENANCE_R48_ZIP --exhibits PREPARED_EXHIBITS_ZIP --output NEW_OUTPUT_DIRECTORY
```

Run the portability tool with the sample's USD-enabled Python. It compares every
authored USD field and verifies every archive member before accepting the output.
It preserves the existing archives and all source-license and attribution records.

The application composes the corrections in `data/scene-overrides/` over the
installed campus. R51 closes five corridor-to-gallery floor gaps with stone
infills that reuse the campus floor material and texture scale. The original
archive, its checksum, and its navigation data remain unchanged. To regenerate
this layer from an installed R48 campus, run `python tools/author_entrance_floors.py`
with the main library environment's Python.


The campus and guide retain their own terms and notices, separate from the
source-code Apache-2.0 license.

For explicit development testing only, `OVLAB_SCENE=classic` selects the older
procedural scene. Normal launches use the campus above.
