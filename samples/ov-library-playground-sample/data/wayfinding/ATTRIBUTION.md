# Wayfinding lettering

The campus plaque and physics-bench lettering in `lettering.json` is derived
from **Noto Sans SemiBold**, copyright 2018 the Noto Project Authors,
licensed under the SIL Open Font License 1.1.

- Source: [Noto Sans SemiBold](https://github.com/notofonts/noto-fonts/blob/ffebf8c1ee449e544955a7e813c54f9b73848eac/hinted/ttf/NotoSans/NotoSans-SemiBold.ttf)
- Upstream commit: `ffebf8c1ee449e544955a7e813c54f9b73848eac`
- Unmodified font: [NotoSans-SemiBold.ttf](fonts/NotoSans-SemiBold.ttf)
- Complete license: [SIL OFL 1.1](fonts/OFL.txt)
- Download provenance and SHA-256 hashes: [manifest](fonts/manifest.json)

The authored signs use selected text outlines flattened into polygon contours,
normalized to the font's units per em, triangulated and rounded to six decimal
places. The font itself has not been modified. The generated artwork is not a
general-purpose font, and the runtime draws the baked polygons without loading
the font. The font and its license remain included for attribution and reproducibility.

NVIDIA's private authoring tool `tools/make_campus_lettering.py` generates these
outlines. The public sample contains the resulting artwork and font provenance.
The signs replace the previously generated Segoe UI vector outlines; no Windows
font or Segoe UI outline data is included in this directory.
