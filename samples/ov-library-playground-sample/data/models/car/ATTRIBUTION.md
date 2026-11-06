# Car Concept

Model and textures by **Eric Chadwick**, copyright 2024 **Darmstadt Graphics Group GmbH**.
Licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
The complete license is included in `CC-BY-4.0.txt`.

Source: [Khronos glTF Sample Assets / Car Concept](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/723ffc6706725b618b8c14ceb82e3e6904b08a76/Models/CarConcept).
The original model metadata, license notice and download manifest accompany this file.

Adaptation for OV Library Lab: converted from glTF to USD using the Kit asset converter, preserving 97 meshes (213,347 triangles), UVs and PBR textures. Khronos and 3D Commerce logo material references were replaced with neutral surfaces; the unused logo texture is not distributed. Logos and trademarks are excluded from the source's copyright license; the original trademark notice is included for provenance. Displayed at its source metre dimensions with a 1:1 scale and rotated mount. Body paint, roughness, metal response, clearcoat and illumination can be changed interactively.

This attribution does not imply endorsement by the artist, DGG or Khronos. The car is a concept asset, not a specific production vehicle.

## Runtime material refinements / September 2026

The lab preserves the attributed source geometry, textures and UVs. Runtime
overrides in `automotive.py` reduce paint and tire normal strength, increase
tire roughness, add restrained glass and clearcoat roughness, and provide
Gloss/Satin/Matte paint presets. No claim is made that these presets reproduce
a measured production vehicle finish. Original attribution is retained above.
