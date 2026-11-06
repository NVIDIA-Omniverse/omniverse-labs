# Surface assets

The bundled 2K color, roughness and OpenGL normal maps are licensed under **CC0 1.0**, included in `CC0-1.0.txt`.

- [Concrete Floor 02](https://polyhaven.com/a/concrete_floor_02), Rob Tuytel / Poly Haven. The shader tones the diffuse map to a neutral sealed-concrete finish and reduces normal strength; the source images are unmodified.
- [Plywood](https://polyhaven.com/a/plywood), Rob Tuytel / Poly Haven. Applied to wooden pallet and bench surfaces.
- [Cardboard 004](https://ambientcg.com/a/Cardboard004), ambientCG / Lennart Demes. Applied to carton shells with reduced normal strength.

See the [Poly Haven license](https://polyhaven.com/license) and [ambientCG license](https://docs.ambientcg.com/license/). File URLs, authors and SHA-256 hashes are preserved in `manifest.json`. `tools/fetch_surface_assets.py` reproduces the download and extraction.

These are visual PBR surfaces. They do not supply measured friction, restitution or mass; the ovphysx test loads author those properties separately.
