# Navigation cursor artwork

The open and closed hand paths come from [Lucide](https://lucide.dev/),
copyright (c) 2026 Lucide Icons and Contributors, under the ISC license.
The original SVG files and full upstream [license](LICENSE.txt) are included.
[manifest.json](manifest.json) records the exact upstream revision and hashes.

The Lab uses the original paths as small outline cursors, with a pale stroke,
a dark contrast edge, and transparent padding. Both states share a fixed hotspot.
The generated PNGs are 224 pixels square, downsampled to the display's DPI at
runtime. They add no runtime dependency beyond the existing Pillow installation.

To reproduce the PNGs, run `node tools/render_cursors.cjs` from the repository
with `sharp` 0.35.5 available on `NODE_PATH`. This is an authoring step only.
