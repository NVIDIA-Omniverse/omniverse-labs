// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NUSD_SCENE_H
#define NUSD_SCENE_H

/*
 * scene.h — USD scene data extracted via the nanousd C API (nanousdapi.h).
 *
 * All mesh data is arena-allocated for one-shot free.
 * Uses zero-copy (nanousd_arraydataf/i) where possible.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* Vertex data — flat arrays, 3 floats per vertex */
    float*    positions;     /* float[nvertices * 3] */
    float*    normals;       /* float[nvertices * 3], may be NULL */
    float*    colors;        /* float[nvertices * 3], per-vertex displayColor, may be NULL */
    float*    texcoords;     /* float[nvertices * 2], UV coordinates, may be NULL */

    /* Index data — triangulated */
    uint32_t* indices;       /* uint32_t[nindices] */

    /* Authored Ptex surface color sampled once per triangulated corner.
     * Packed as RGBA8 in index order: three colors per output triangle. */
    uint32_t* ptex_tri_colors;
    int       ptex_tri_color_count;

    int       nvertices;
    int       nindices;

    /* World-space transform (row-major 4x4) */
    double    world_xform[16];

    /* Per-mesh world-space bounds (computed during load) */
    float     bounds_min[3];
    float     bounds_max[3];

    /* Object-space bounds. Used to bound expanded instances without
     * re-transforming every shared prototype vertex per instance. */
    float     local_bounds_min[3];
    float     local_bounds_max[3];

    /* Fallback displayColor (used if colors == NULL) */
    float     display_color[3];
    int       has_display_color;

    /* Material assignment (-1 = no material) */
    int       material_index;

    /* Debug identity: USD prim path when loaded from a live stage. Geometry
     * cache hits may leave this NULL because older cache files do not store
     * paths. */
    char*     path;

    /* Offsets into merged buffers (set by viewer) */
    uint32_t  vertex_offset;
    uint32_t  index_offset;

    /* Instancing: index of prototype mesh in meshes[] array.
     * Equal to own index if this IS the prototype or non-instanced.
     * When prototype_idx != own index, positions/normals/indices
     * are shared pointers (not separately allocated). */
    int       prototype_idx;

    /* Set to 1 when this mesh sits under a PointInstancer's prototype
     * subtree. Such meshes are kept in scene->meshes so the third-pass
     * PI expansion can copy their geometry to per-instance scene meshes,
     * but they must not be drawn standalone (USD semantics — the proto
     * subtree is materialized only via the PI). The renderer's
     * load loop treats this as a draw skip. */
    int       is_proto_only;

    /* Tier 3 lazy extraction (see docs/plans/TIER_3_LAZY_MESH.md).
     * Set by scene_load when NUSD_LAZY_MESH=1: the renderer's not-yet-
     * implemented extract-deferred worker (step 3) re-resolves the
     * prim via ``nanousd_prim(stage, lazy_prim_idx)`` and reads
     * attributes on demand. ``-1`` in eager-loaded scenes.
     *
     * Index (not opaque handle) so no nanousd_freeprim leak in step 2;
     * step 3 re-acquires the handle as needed and frees it after use.
     *
     * Lifetime caveat (activates with step 3): once step 3 begins to
     * use ``lazy_prim_idx``, the owning Scene MUST keep the nanousd
     * stage alive past the current scene_free-at-end-of-attach
     * behaviour. Today (step 2) nothing reads this field, so the
     * lifetime issue is dormant. */
    int       lazy_prim_idx;

    /* Preprocessed meshlet range — populated only by the geometry cache
     * (geo_cache.c) on a cache hit. meshlet_count 0 = none. Instances share
     * their prototype's range. The meshlets index Scene.meshlet_vertices /
     * meshlet_triangles. */
    uint32_t  meshlet_offset;
    uint32_t  meshlet_count;
} SceneMesh;

/* A meshoptimizer meshlet: a small triangle cluster with cull bounds, in the
 * mesh-shader-native compact layout — a per-meshlet table of unique vertex
 * indices plus 8-bit micro-indices into that table. Populated only by the
 * geometry cache (geo_cache.c) on a cache hit, which compacts the cache's
 * expanded index stream at load time; the USD parse path leaves
 * Scene.meshlets NULL. */
typedef struct {
    float     center[3];
    float     radius;
    float     cone_axis[3];
    float     cone_cutoff;
    uint32_t  vertex_offset;    /* into Scene.meshlet_vertices */
    uint32_t  vertex_count;     /* unique vertices used by this meshlet (<= 64) */
    uint32_t  triangle_offset;  /* byte offset into Scene.meshlet_triangles */
    uint32_t  triangle_count;   /* triangles; 3*triangle_count micro-indices */
} SceneMeshlet;

/* Light kinds */
#define SCENE_LIGHT_RECT     0
#define SCENE_LIGHT_DISTANT  1
/* SphereLight is a point/area emitter at `position` with radius stored in
 * u_axis[0] (v_axis and normal are unused). Kept here as a separate kind so
 * the shader can apply 1/r² falloff and area-normalize when normalize=1.
 * Required for closed-room USD scenes (e.g. poc/assets/box_room.usda) where
 * RectLight + DistantLight cannot reach the interior. */
#define SCENE_LIGHT_SPHERE   2

typedef struct {
    int       kind;             /* SCENE_LIGHT_* */

    /* World-space frame.
     *   RectLight/DiskLight/CylinderLight approximation: position is the
     *     emitter center; normal is the emit direction (-Z in light-local
     *     space → -row2 of world xform); u_axis/v_axis are world-space
     *     half-extent vectors. Disks are approximated as equal-area rects;
     *     cylinders as length × diameter rects.
     *   DistantLight: position unused; normal is the emit direction
     *     (light shines along -normal). u_axis/v_axis unused. */
    float     position[3];
    float     normal[3];
    float     u_axis[3];        /* half-extent along light-local +X */
    float     v_axis[3];        /* half-extent along light-local +Y */

    /* Radiometric properties (UsdLuxLight semantics).
     *   color:      linear sRGB tint (inputs:color).
     *   intensity:  inputs:intensity * 2^exposure.
     *   normalize:  inputs:normalize. When 1, RectLight intensity is
     *     interpreted as power (W) → divide by area; when 0, intensity
     *     is interpreted as radiance (W/sr/m^2).
     *   angle_deg:  DistantLight cone half-angle (degrees). */
    float     color[3];
    float     intensity;
    int       normalize;
    float     angle_deg;
} SceneLight;

/* USD UsdGeomBasisCurves topology. RT expands this compact CV/topology
 * representation to curve segments only when building acceleration
 * structures; raster consumes the same CV data through 4-CV tessellation
 * patches. Width-only curves are tubes/implicit ribbons; curves with
 * authored normals become Storm-style oriented ribbons in the RT path. */
typedef enum {
    CURVE_BASIS_BEZIER     = 0,
    CURVE_BASIS_BSPLINE    = 1,
    CURVE_BASIS_CATMULLROM = 2,
    CURVE_BASIS_LINEAR     = 3   /* type=linear, basis ignored */
} SceneCurveBasis;

typedef enum {
    CURVE_WRAP_NONPERIODIC = 0,
    CURVE_WRAP_PERIODIC    = 1,
    CURVE_WRAP_PINNED      = 2
} SceneCurveWrap;

typedef struct {
    /* Per-CV data in OBJECT space (Phase 11.5.2). The world matrix lives
     * on the per-curve TLAS instance; dynamic transforms via
     * nu_set_transforms update curves at the same cost as meshes — no
     * BLAS rebuild, no segment re-upload. The per-segment AABBs emitted
     * by scene_curve_to_segments() are likewise object-space. */
    float*    cvs;             /* float[nv * 3]  — object-space CV positions */
    float*    widths;          /* float[nv]      — per-CV after fan-out */
    float*    colors;          /* float[nv * 3]  — per-CV displayColor, may be NULL */
    float*    normals;         /* float[nv * 3]  — per-CV authored normals, may be NULL */
    int       nv;
    int       ncurves_in_prim; /* len(curveVertexCounts) — number of sub-curves */
    int*      curve_vertex_counts; /* int[ncurves_in_prim] */
    uint32_t* patch_indices;   /* uint32[npatches * 4] — raster tess patches */
    int       npatches;

    /* World transform (row-major 4x4). Used at TLAS-instance creation
     * to position the curve BLAS in world space. normal_xform is the
     * inverse-transpose upper 3x3 used for authored ribbon normals. */
    double    world_xform[16];
    double    normal_xform[16];

    /* World-space bounds. */
    float     bounds_min[3];
    float     bounds_max[3];

    /* Fallback displayColor (when colors == NULL). */
    float     display_color[3];
    int       has_display_color;
    int       has_normals;
    int       has_texcoords;
    int       texcoord_count;

    /* Topology metadata (USD attributes). */
    SceneCurveBasis basis;
    SceneCurveWrap  wrap;
    int             type_is_cubic;  /* 1 = cubic, 0 = linear */

    /* Material assignment (-1 = none). Reuses Scene's material pool. */
    int       material_index;
} SceneCurve;

/* Per-instance transform for the compact PointInstancer model. 12 floats =
 * 48 B, column-major affine matching the composed inst_world (columns 0..2 =
 * R*S basis vectors, column 3 = translation; the always-(0,0,0,1) bottom row is
 * dropped). Converts to VkTransformMatrixKHR (3x4 row-major) with one swizzle.
 * Replaces per-instance SceneMesh clones: PI placements live only here. */
typedef struct {
    float m[12];
} SceneInstanceTransform;

static inline void scene_instance_transform_to_vk3x4(
        const SceneInstanceTransform* t,
        float dst[12])
{
    /* SceneInstanceTransform stores USD row-vector affine columns compactly:
     * x' = p.x*m0 + p.y*m3 + p.z*m6 + m9. VkTransformMatrixKHR wants the
     * equivalent column-vector 3x4 rows with translation in elements 3/7/11. */
    dst[0]  = t->m[0]; dst[1]  = t->m[3]; dst[2]  = t->m[6]; dst[3]  = t->m[9];
    dst[4]  = t->m[1]; dst[5]  = t->m[4]; dst[6]  = t->m[7]; dst[7]  = t->m[10];
    dst[8]  = t->m[2]; dst[9]  = t->m[5]; dst[10] = t->m[8]; dst[11] = t->m[11];
}

typedef enum {
    SCENE_INSTANCE_SOURCE_POINT_INSTANCER = 0,
    SCENE_INSTANCE_SOURCE_NATIVE_INSTANCE = 1,
} SceneInstanceSourceKind;

/* Compact PI batch: one prototype sub-mesh instanced by a slice of
 * Scene.pi_transforms. Multi-mesh prototypes share one transform_offset.
 * Acceptance gate: pi_scene_mesh_clones == 0. */
typedef struct {
    int      prototype_mesh_idx;     /* index into Scene.meshes (a proto sub-mesh) */
    uint32_t transform_offset;       /* slice base into Scene.pi_transforms */
    uint32_t transform_count;
    int      source_prim_idx;        /* flat-prim index of the authoring PI */
    int      material_or_binding_id; /* -1 = inherit from prototype */
    int      source_kind;            /* SceneInstanceSourceKind */
} SceneInstanceBatch;

typedef struct {
    SceneMesh* meshes;
    int        nmeshes;

    /* BasisCurves prims, loaded as tubes (Phase 11). */
    SceneCurve* curves;
    int         ncurves;

    /* Compact PointInstancer batches. PI clones live ONLY here (Phase 1+),
     * never in meshes[]. Heap-allocated (not arena); freed in scene_free. */
    SceneInstanceBatch*     pi_batches;
    int                     npi_batches;
    SceneInstanceTransform* pi_transforms;
    uint64_t                npi_transforms;

    /* Scene bounds (world space) */
    float      bounds_min[3];
    float      bounds_max[3];

    /* Authored up axis: 0 = X, 1 = Y (USD default), 2 = Z. Geometry stays in
     * authored coordinates by default, matching OVRTX/USD composition
     * semantics; renderers use this to choose the default camera up vector. */
    int        up_axis;

    /* Lights — arena-allocated */
    SceneLight* lights;
    int         nlights;
    int         has_authored_light;  /* any active USD light prim, including DomeLight */

    /* Preprocessed meshlets — populated only by the geometry cache.
     * USD-parsed scenes leave these NULL/0. Compact layout: each SceneMeshlet
     * indexes a slice of meshlet_vertices (its unique global vertex indices)
     * and meshlet_triangles (its 8-bit micro-indices). */
    SceneMeshlet*  meshlets;
    int            nmeshlets;
    uint32_t*      meshlet_vertices;    /* per-meshlet vertex tables, concatenated */
    int            nmeshlet_vertices;
    unsigned char* meshlet_triangles;   /* per-meshlet micro-indices, concatenated */
    int            nmeshlet_triangles;

    /* USD UsdLuxDomeLight metadata (max one per scene — extras after the
     * first are ignored, matching Hydra's behaviour). The renderer's
     * nu_attach_scene plumbs textured domes through to HDR IBL and
     * textureless domes through to a constant-color IBL. */
    int        has_dome_light;
    char       dome_hdr_path[512]; /* empty = textureless DomeLight */
    float      dome_color[3];      /* inputs:color, default white */
    float      dome_intensity;     /* inputs:intensity * 2^exposure */
    float      dome_rotation_y;    /* degrees, around Y-up (default 0) */

    /* Arena memory (opaque, freed by scene_free) */
    void*      _arena;

    /* Keep stage handle alive for zero-copy pointers */
    void*      _stage;

    /* 1 = scene_free closes the stage; 0 = stage is borrowed (caller owns).
     * Borrowed stages are used by nu_load_usd_from_handle so a Python
     * pxr_compat shim and the renderer can share a single composed stage. */
    int        _owns_stage;

    /* 1 = scene->meshes is on the heap (malloc); 0 = inside _arena.
     * Set by scene_release_mesh_payloads when it lifts the mesh array out
     * of the arena before destroying the arena, so scene_free knows to
     * call free(scene->meshes) instead of leaving it dangling. */
    int        _meshes_heap;

    /* MaterialCollection* (opaque — declared in material.h, freed by scene_free) */
    void*      materials;

    /* Native-instance-curve gate. The compact-native part-(A) prototype DFS
     * scans every prototype subtree's typenames; it sets checked_proto_for_curves=1
     * and has_basis_curves=1 iff it ever sees a BasisCurves prim. Because an
     * instance proxy is a typename-copy of its prototype, this answers "does the
     * scene have any BasisCurves?" WITHOUT the per-instance proxy expansion that
     * scene_load_native_instance_curves would otherwise trigger (the DSX 7-min
     * stall: ~1M proxies walked to find zero curves). When checked but no curves,
     * the native-instance curve pass is skipped. Only gates when part-A ran
     * (compact-native active) so compact-native-off scenes keep old behavior. */
    int        checked_proto_for_curves;
    int        has_basis_curves;
} Scene;

/* Per-segment data emitted by scene_curve_to_segments() and consumed by
 * the GPU intersection shader. Layout matches RENDERER_BIG_PLAN.md
 * Phase 12.1 datacenter design: 32 B AoS, std430-friendly, single
 * cacheline-pair per hit.
 *
 * IMPORTANT: this layout is duplicated as a GLSL `CurveSegment` struct
 * in `src/shaders/raytrace_curve.{rint,rchit}.glsl`. Any change here
 * MUST be mirrored there or the intersection shader will silently
 * mis-decode segment data.
 *
 * Phase 11.A constraint: each segment is *constant radius* (`r0`).
 * Varying-radius input (per-CV widths) is realised by emitting one
 * segment per CV-pair with that pair's leading radius — Storm's
 * pattern for the same restriction. Cone-sphere intersection for
 * truly varying radius lands in Phase 11.B with a parallel `r1[]`
 * SSBO at binding 16.
 *
 * `mat_flags` is reserved for the material-id LUT in Phase 12.2. For now,
 * bit 31 marks an oriented ribbon, bit 30 allows a small endpoint join pad
 * for tessellated/multi-segment ribbons, and the low 30 bits pack a 15-bit
 * octahedral world-space ribbon normal. This keeps the hot segment buffer
 * at 32 B while matching Storm/HdSt's rule that curves with authored normals
 * render as ribbons rather than tubes. */
#define SCENE_CURVE_SEG_FLAG_RIBBON          0x80000000u
#define SCENE_CURVE_SEG_FLAG_RIBBON_JOIN_PAD 0x40000000u
#define SCENE_CURVE_SEG_OCT_MASK             0x00007fffu
#define SCENE_CURVE_SEG_OCT_SHIFT_Y          15u

typedef struct {
    float    p0[3];          /*  0 — segment start (WORLD-SPACE; baked from curve->cvs through curve->world_xform during extraction) */
    float    r0;             /* 12 — segment radius                                              */
    float    p1[3];          /* 16 — segment end (WORLD-SPACE)                                   */
    uint32_t mat_flags;      /* 28 — ribbon flag + packed normal, or 0 for tube segments         */
} SceneCurveSegment;          /* 32 B */

/* Per-segment AABB (matches VkAabbPositionsKHR layout: 6 floats).
 * Phase 12.x: AABBs are now generated GPU-side at BLAS-build time from
 * the segment buffer (see gpu_build_curve_aabbs_compute), so the host
 * never authors them. The struct definition is kept for any code that
 * still wants to introspect the GPU layout. */
typedef struct {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
} SceneCurveAabb;

/* Count linear segments emitted by a SceneCurve under Phase 11.A rules
 * (cubic curves are still loaded but emit zero linear segments — they'll
 * use cubic-Bezier AABB extraction in Phase 11.B). */
int scene_curve_count_segments(const SceneCurve* curve);

/* Materialise the SceneCurve as a flat segment array in a caller-provided
 * buffer. out_segments must have capacity for at least
 * scene_curve_count_segments(curve) entries. Returns the number written.
 *
 * Phase 12.x: AABBs are no longer authored host-side. They are computed
 * by a GPU compute pass at BLAS-build time from the segment data
 * (see gpu_build_curve_aabbs_compute in gpu_vulkan.c), eliminating the
 * ~825 MB host→device upload on tera-scale fixtures. */
int scene_curve_to_segments(const SceneCurve* curve,
                            SceneCurveSegment* out_segments);

/* Materialise authored per-CV displayColor into the same flat segment order as
 * scene_curve_to_segments(). Returns the number of colors written, or 0 when
 * the curve has no per-CV color payload. out_rgb must have capacity for
 * scene_curve_count_segments(curve) * 3 floats. */
int scene_curve_to_segment_colors(const SceneCurve* curve,
                                  float* out_rgb);

/* Load a USD file (USDA or USDC). Returns NULL on failure. */
Scene* scene_load(const char* filepath);

/* Set the time used by subsequent scene_load* calls when reading
 * USD xform attributes. Pass NaN for authored default time. */
void   scene_set_load_time(double t);

/* Enable/disable material extraction for subsequent scene_load* calls.
 * Geometry-only DSX benchmark paths disable this to avoid eager full-stage
 * material/texture processing before the renderer can show a frame. */
void   scene_set_load_materials(int enabled);

/* Build a Scene from an already-open NanousdStage handle. The caller keeps
 * ownership of `stage` — scene_free will NOT close it. Used to share a
 * single composed stage between the Python pxr_compat shim and the
 * renderer (Phase 5: stage handle sharing). `stage_label` is a short
 * string used in diagnostic messages; pass the file path or layer
 * identifier so log lines stay informative. */
Scene* scene_load_from_stage(void* stage, const char* stage_label);

/* Same as scene_load_from_stage, but skip prims whose flat-list index `i`
 * is NOT in the `wanted_prims` bitmap. Used by Tier 3 step 4 frustum
 * cull: nu_extract_deferred_visible walks the lazy scene's world AABBs
 * against a camera frustum, builds a bitmap of visible prim indices,
 * and re-extracts only those prims (eager passes 1 + 2 + 3 all observe
 * the filter on the prim index they enumerate; second-pass instance
 * children inherit the filter via the parent instance prim's bit).
 *
 * wanted_prims: byte-per-prim bitmap, length `nprims_in_bitmap`. Non-zero
 *               at index i = include prim i. NULL or `nprims_in_bitmap` <= 0
 *               disables filtering (matches scene_load_from_stage behavior).
 * nprims_in_bitmap: length of the bitmap (typically the result of
 *                   `nanousd_nprims(stage)` at lazy step 2 time). Indices >=
 *                   this bound default to "wanted" (defensive against stage
 *                   growth between lazy + eager passes). */
Scene* scene_load_from_stage_filtered(void* stage,
                                      const char* stage_label,
                                      const unsigned char* wanted_prims,
                                      int nprims_in_bitmap);

/* Free all scene memory + close the USD stage (if owned). */
void   scene_free(Scene* scene);

/* Release the per-mesh CPU geometry payload (positions/normals/colors/
 * texcoords/indices) and destroy the underlying arena. The renderer-owned
 * GPU vertex/index buffers are unaffected; only the host-side copy is
 * dropped. Returns 1 if the release ran, 0 if it was skipped (e.g. scene
 * has curves which still need their host-side data, or the scene is
 * already released).
 *
 * Safety: after this call, the caller MUST NOT add/remove meshes or
 * mutate any per-mesh geometry. The SceneMesh structs survive (for
 * material_index, transform, bounds), but their positions/normals/etc.
 * pointers are NULL. To preserve the pxr_compat shim path that may
 * still reference stage strings, this function does NOT close the
 * nanousd stage in v1 — only the geometry arena is destroyed. */
int    scene_release_mesh_payloads(Scene* scene);

#ifdef __cplusplus
}
#endif

#endif /* NUSD_SCENE_H */
