---
name: nanousd-renderer-scene-extraction
description: Load USD scenes through NanoUSD and convert composed USD prims into renderer-ready geometry, material, light, camera, transform, instancing, texture, and diagnostic representations. Use when implementing, debugging, or validating NanoUSD-based scene ingestion, renderer scene structs, material/light extraction, texture path resolution, or loader parity against USD/OpenUSD semantics.
---

# NanoUSD Renderer Scene Extraction

## Purpose

Turn composed USD data into a renderer-owned scene representation without losing authored intent. Preserve paths, transforms, visibility, material bindings, texture identity, light units, instancing, and unsupported-feature diagnostics until the renderer deliberately consumes or rejects them.

## Workflow

1. Define the extraction contract first.
   - Name the output structs and ownership rules: geometry buffers, material table, texture table, light table, camera table, prototype/instance table, path maps, warnings, and unsupported features.
   - Record coordinate convention, matrix layout, time code, up-axis policy, units policy, material model coverage, and renderer capability gaps.
   - Keep CPU scene ingestion separate from GPU upload. Upload only after counts, bounds, indices, material bindings, and light records pass invariants.

2. Open and validate the stage.
   - Use `nanousd_open`, `nanousd_isvalid`, `nanousd_error`, and `nanousd_close`; for borrowed handles, make ownership explicit and never close caller-owned stages.
   - Capture root layer, layer stack, `upAxis`, `metersPerUnit`, `timeCodesPerSecond`, frame range, and current evaluation time.
   - Treat load/composition warnings as scene diagnostics, not stdout noise. Include counts for prims, meshes, materials, textures, lights, curves, instances, skipped prims, and unsupported features.

3. Traverse composed prims deliberately.
   - Iterate `nanousd_nprims` / `nanousd_prim` for the composed stage, then free each prim handle with `nanousd_freeprim`.
   - Skip inactive, undefined, abstract, invisible, guide, and proxy data unless the renderer explicitly requests those purposes.
   - Visibility is inherited: walk ancestors before deciding a child mesh is visible. Keep USD paths in every emitted record for debugging, selection, AOVs, and stable IDs.

4. Build transforms once and document conventions.
   - Use `nanousd_get_local_transform(prim, time, local, &resetXformStack)` and compose parent transforms while honoring `resetXformStack`.
   - The Vulkan loader currently uses row-major, row-vector transforms: `P_world = P_local * leaf * parent * ... * root`.
   - Apply `upAxis` conversion and `metersPerUnit` exactly once at the scene boundary, and apply it consistently to mesh bounds, curve widths, light positions/extents, cameras, and instance transforms.
   - Re-evaluate transforms when the renderer time changes; do not reuse default-time matrices for animated xformOps.

5. Extract mesh geometry with interpolation intact.
   - Read `points`, `faceVertexCounts`, and `faceVertexIndices`; prefer NanoUSD zero-copy array readers for speed, but copy data the renderer must retain because later NanoUSD calls can invalidate cached array data.
   - Validate index ranges before triangulation. Triangulate polygons deterministically and preserve the source path and face/primitive mapping if picking or material subsets need it.
   - Handle normals, `primvars:normals`, `primvars:st`, alternate UV names, `displayColor`, `displayOpacity`, orientation, double-sidedness, subdivision scheme, and interpolation metadata.
   - Distinguish authored opinions from USD schema fallbacks. For visibility-critical fields such as Mesh `doubleSided`, record whether the value was authored or defaulted, and validate primary, shadow, and instance traversal semantics with a reduced fixture before tuning light intensity or denoising.
   - Face-varying UVs or normals require vertex expansion. Generate smooth or flat normals only when absent, and label generated data as generated.
   - Curves, points, volumes, subdivision surfaces, and unsupported prim types must produce either a converted renderer record or a structured unsupported-feature record; do not silently drop them.

6. Preserve instancing and prototypes.
   - Use NanoUSD instance/prototype APIs (`nanousd_isinstance`, `nanousd_prototype`, `nanousd_ninstances`, `nanousd_instance`) and `PointInstancer` attributes (`prototypes`, `protoIndices`, `positions`, `orientations`, `scales`) to build prototype and instance tables.
   - Store prototype geometry once, then store per-instance transforms, visibility, material overrides, and stable IDs.
   - USD `quatf`/`quath` arrays read into NanoUSD float arrays as `(x, y, z, w)` in the current Vulkan loader pattern; verify ordering before composing instance transforms.
   - Keep prototype-only prims hidden from normal draw lists while still available for instance expansion.

7. Resolve materials and textures from USD graphs.
   - Resolve `material:binding` relationships on the prim and ancestors. If material subsets are supported, preserve face ranges and subset binding strength.
   - Follow shader connections with `nanousd_nconnections` / `nanousd_connection`; strip output suffixes to locate upstream shader or texture prims with `nanousd_primpath`.
   - Support renderer-known models explicitly, such as UsdPreviewSurface, OmniPBR, OmniGlass, OmniSurface, and sidecar MaterialX. Unknown nodegraphs should become unsupported material diagnostics, not valid white defaults.
   - Map base color, opacity, opacity threshold, metallic, roughness, IOR, clearcoat, normal, normal scale, emissive, occlusion/ORM, specular workflow, subsurface, transmission, texture transforms, and fallback values.
   - Deduplicate textures by resolved identity plus color-space/sampler semantics. Mark base color and emissive textures as sRGB; mark normal, roughness, metallic, AO, opacity, and ORM data textures as linear data.
   - Resolve asset paths relative to the authored layer or stage context, not process `cwd`. For sidecar scans, bound the search to the scene directory and make expensive scans opt-in or cached.

8. Extract lights and cameras as first-class scene data.
   - Handle DomeLight, RectLight, DistantLight, SphereLight, and any renderer-supported Disk/Cylinder variants. Count known unsupported light types.
   - Preserve `inputs:color`, `inputs:intensity`, `inputs:exposure`, `inputs:normalize`, dimensions, radius, angle, shaping controls, visibility, texture/HDR paths, and dome rotation.
   - Derive light world frames from the same transform pipeline as geometry. For RectLight, store center, normal, and half-extent axes; for DistantLight, store normalized direction and angle; for SphereLight, store radius and world position.
   - Extract authored cameras when needed: transform, projection, focal length, apertures, clipping range, focus distance, and active-camera selection. If auto-framing is a fallback, report it as a fallback.

9. Emit a renderer-ready scene with diagnostics.
   - Return geometry buffers, material/light/texture arrays, curves, camera records, bounds, path-to-index maps, prototype/instance maps, and unsupported-feature diagnostics.
   - Include stable path hashes or IDs for meshes, lights, materials, emitters, and instances so renderer AOVs and logs can tie pixels back to USD.
   - Track load timing and memory by stage open, material load, mesh extraction, instancing, curves, lights, texture decode, GPU upload, and acceleration-structure build.

10. Validate with focused fixtures before full assets.
   - Add reduced scenes for resetXformStack, up-axis plus metersPerUnit, face-varying UV seams, inherited visibility, material binding, texture connections, point instancing, DomeLight HDR, direct lights, emissive materials, and unsupported prims.
   - Compare against OpenUSD/usdview/Hydra or a known reference renderer when parity matters. Match camera, exposure, tone mapping, color space, light/environment, sample count, and seed.
   - Use AOVs to isolate ingest bugs: geometry IDs, normals, UVs, base color, material ID, roughness, metallic, opacity, emissive, direct light, shadow visibility, indirect, and depth.

## Reference Contract and Cross-Cutting Conventions

These are the parts an agent cannot infer and that have silently broken existing
backends. Treat the renderers' `scene.h` as the reference *shape*; do not freeze a
field list here — match the canonical struct. The durable content is the invariant
and the failure it prevents.

**Scene struct shape (point at `scene.h`, preserve these groups).** Every renderer
needs, and must keep until the backend consumes them: geometry (flat positions plus
optional normals/displayColor/UVs and triangulated indices — copy what the renderer
retains, since later NanoUSD calls can invalidate zero-copy arrays); identity (the
USD path on *every* record, so AOVs, picking, and logs tie pixels back to USD);
transform (one world matrix per record in a single documented convention);
bounds (world-space *and* object-space — object-space lets you bound expanded
instances without re-transforming every shared prototype vertex); material binding
(an index into a material table, not inlined material data); instancing (a prototype
index plus a "prototype-only" flag, so PointInstancer prototype subtrees are
materialized via instances and never drawn standalone); and scene-level data (up-axis,
dome/IBL metadata, light table, structured unsupported-feature list). Do not flatten
instancing or drop paths at ingest "to simplify" — that information cannot be
recovered later.

**Vertex layout (example — verify against the header).** An interleaved per-vertex
layout such as 48-byte / 12-float `[pos.xyz, normal.xyz, pad.xyz, uv.xy, matID]` is
one workable choice. The durable rule: the host writer and every shader/pipeline that
reads the buffer derive stride and offsets from ONE definition, and any later pass
that touches uv/matID must use the same stride. A mismatch yields garbage attributes
with no error.

**Transform convention and the silent translation-drop bug (invariant + signature).**
Pick one convention and apply one named conversion at the GPU boundary. The NanoUSD
loaders use row-vector USD (translation in the matrix's last row). A flat element copy
across a row-vector/column-vector boundary silently drops translation: the object
renders at the origin and per-frame transform updates look like no-ops. Verify with a
translate-and-re-render test — a static identity-vs-identity test passes tautologically
and hides exactly this. This bug shipped in more than one backend.

**Host/shader struct sync.** Any struct shared between host and shader (per-segment
geometry, push constants, material records) has two definitions in two languages and
they must change together. Keep them adjacent or generated, and note the duplication at
each site; a silent mismatch mis-decodes every record. See `renderer-backend-port` for
the cross-language alignment/padding trap.

## NanoUSD Entry Points

- Stage: `nanousd_open`, `nanousd_close`, `nanousd_isvalid`, `nanousd_error`, `nanousd_stage_get_root_layer_path`, `nanousd_stage_n_layers`, `nanousd_stage_layer_path`.
- Metadata: `nanousd_metadatas`, `nanousd_metadatad`, `nanousd_timecodes_per_second`, `nanousd_frames_per_second`, `nanousd_start_time`, `nanousd_end_time`.
- Traversal: `nanousd_nprims`, `nanousd_prim`, `nanousd_primpath`, `nanousd_defaultprim`, `nanousd_parent`, `nanousd_nchildren`, `nanousd_child`.
- Prim state: `nanousd_path`, `nanousd_name`, `nanousd_typename`, `nanousd_kind`, `nanousd_isactive`, `nanousd_isdefined`, `nanousd_isabstract`, `nanousd_isa`, `nanousd_hasapi`.
- Attributes: `nanousd_hasattrib`, `nanousd_attribtype`, scalar/vector readers, `nanousd_attribarraylen`, `nanousd_attribarrayf`, `nanousd_attribarrayi`, `nanousd_arraydataf`, `nanousd_arraydatai`, `nanousd_attrib_token`, `nanousd_attribasset`, `nanousd_attrib_interpolation`, `nanousd_attrib_authored`.
- Time samples: `nanousd_hassamples`, `nanousd_nsamplekeys`, `nanousd_samplekey`, `nanousd_sample*`.
- Relationships/connections: `nanousd_hasrel`, `nanousd_nreltargets`, `nanousd_reltarget`, `nanousd_hasconnections`, `nanousd_nconnections`, `nanousd_connection`.
- Transforms: `nanousd_get_local_transform`, `nanousd_mul_m4d`.
- Instancing: `nanousd_isinstance`, `nanousd_isprototype`, `nanousd_isinprototype`, `nanousd_prototype`, `nanousd_ninstances`, `nanousd_instance`.

## Failure Signatures

- Exploded or mirrored scene: transform order, reset stack, quaternion order, up-axis, or units applied twice.
- White or default-gray scene: material binding not inherited, shader connections not followed, unknown material treated as valid, or texture path resolved from `cwd`.
- Texture looks washed out or normal map is wrong: sRGB applied to data textures, missing scale/bias, channel swizzle mismatch, or ORM channels mapped incorrectly.
- Missing shelves, chess pieces, or repeated assets: prototype-only prims drawn or hidden incorrectly, PointInstancer expansion skipped, or native instances flattened without prototype lookup.
- Lights visible but no shadows or no direct light: light transform, intensity/exposure/normalize, visibility, unsupported light type, alpha cutout, or TLAS membership mismatch.
- Black glass, opaque cutouts, dark roof openings, or missing shadow bands: opacity, opacity threshold, transmission, IOR, authored/schema-fallback sidedness, or shadow opacity policy diverges between raster, primary RT, instance RT, and shadow rays.
- Correct first frame but wrong animation: default-time values cached despite time-varying transforms, geometry, materials, or visibility.

## Handoff

When finishing loader work, report the scene path, time code, root layer, counts, bounds, material/light/texture totals, instancing totals, unsupported features, key diagnostics, validation command, and any renderer-degraded semantics that remain.
