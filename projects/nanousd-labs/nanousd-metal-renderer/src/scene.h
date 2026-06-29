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

    int       nvertices;
    int       nindices;

    /* World-space transform (row-major 4x4) */
    double    world_xform[16];

    /* Per-mesh world-space bounds (computed during load) */
    float     bounds_min[3];
    float     bounds_max[3];

    /* Fallback displayColor (used if colors == NULL) */
    float     display_color[3];
    int       has_display_color;

    /* Material assignment (-1 = no material) */
    int       material_index;

    /* Offsets into merged buffers (set by viewer) */
    uint32_t  vertex_offset;
    uint32_t  index_offset;

    /* Instancing: index of prototype mesh in meshes[] array.
     * Equal to own index if this IS the prototype or non-instanced.
     * When prototype_idx != own index, positions/normals/indices
     * are shared pointers (not separately allocated). */
    int       prototype_idx;

    /* USD PointInstancer prototype subtree marker. Set to 1 during the
     * first-pass mesh load when an ancestor in the prim chain is a
     * PointInstancer — those meshes are kept in scene->meshes (so PI
     * expansion can copy their geometry) but the renderer skips them
     * when adding draw items: the visible copies come from PI expansion. */
    int       is_proto_only;

    /* USD prim path of the mesh that produced this entry. `nu_attach_scene`
     * forwards it to `desc.name` so `nu_get_mesh_name` returns the path
     * the viewer needs for click-to-pick. Empty string = unnamed
     * (programmatic mesh, PI-expanded child without its own prim). */
    char      path[512];

    /* Tier 3 lazy extraction. Set when NUSD_LAZY_MESH=1 so the renderer can
     * re-resolve the flat-stage prim later via nanousd_prim(stage, lazy_prim_idx).
     * Eager meshes gate on positions != NULL; this field is only meaningful for
     * metadata-only scene records. Lazy records for native USD instance
     * child meshes additionally store the concrete child prim path above, so
     * filtered extraction can materialize that child directly. */
    int       lazy_prim_idx;
} SceneMesh;

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
     *   RectLight: position is the rectangle center; normal is the emit
     *     direction (-Z in light-local space → -row2 of world xform);
     *     u_axis/v_axis are world-space half-extent vectors.
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

/* USD UsdGeomBasisCurves topology, loaded as tubes for RT.
 * Phase 11.A handles type=linear only; cubic basis variants are loaded
 * (so bounds + prim count are correct) but skipped at BLAS-build time
 * with a single dedup'd warning. Cubic intersection lands in Phase 11.B. */
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
    /* Per-CV data, baked to WORLD space at load time (renderer is static-scene-
     * focused; baking sidesteps per-instance transforms in TLAS for the MVP). */
    float*    cvs;             /* float[nv * 3]  — world-space CV positions */
    float*    widths;          /* float[nv]      — per-CV after fan-out */
    float*    colors;          /* float[nv * 3]  — per-CV displayColor, may be NULL */
    int       nv;
    int       ncurves_in_prim; /* len(curveVertexCounts) — number of sub-curves */
    int*      curve_vertex_counts; /* int[ncurves_in_prim] */

    /* World transform (row-major 4x4). Already baked into cvs[]; preserved
     * for future per-instance TLAS work. */
    double    world_xform[16];

    /* World-space bounds. */
    float     bounds_min[3];
    float     bounds_max[3];

    /* Fallback displayColor (when colors == NULL). */
    float     display_color[3];
    int       has_display_color;

    /* Topology metadata (USD attributes). */
    SceneCurveBasis basis;
    SceneCurveWrap  wrap;
    int             type_is_cubic;  /* 1 = cubic, 0 = linear */

    /* Material assignment (-1 = none). Reuses Scene's material pool. */
    int       material_index;
} SceneCurve;

/* Per-instance transform for compact PointInstancer / native-instance
 * batches. 12 floats = 48 B, column-major affine matching the inst_world
 * layout produced by scene.c's PI Third Pass (columns 0..2 = R*S basis
 * vectors, column 3 = translation, the always-(0,0,0,1) bottom row is
 * dropped). Both Metal raster (fill_raster_instance_data) and RT instance
 * descriptors (MTLAccelerationStructureInstanceDescriptor / VkTransformMatrixKHR)
 * consume this directly without a conversion pass. */
typedef struct {
    float m[12];
} SceneInstanceTransform;

typedef enum {
    SCENE_INSTANCE_SOURCE_POINT_INSTANCER = 0,
} SceneInstanceSourceKind;

/* Compact PI / native-instance batch record. Replaces the per-clone
 * SceneMesh row representation: 2.9M Moana PI instances cost 144 MiB of
 * SceneInstanceTransform vs ~2.6 GiB of SceneMesh+RendererMesh records
 * before. The acceptance gate is `pi_scene_mesh_clones == 0`. */
typedef struct {
    int      prototype_mesh_idx;     /* index into Scene.meshes (a proto sub-mesh) */
    uint32_t transform_offset;       /* slice of Scene.pi_transforms */
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

    /* Compact PI batches. PI clones live ONLY here, never in meshes[]. */
    SceneInstanceBatch*     pi_batches;
    int                     npi_batches;
    SceneInstanceTransform* pi_transforms;
    uint64_t                npi_transforms;

    /* Scene bounds (world space) */
    float      bounds_min[3];
    float      bounds_max[3];

    /* Lights — arena-allocated */
    SceneLight* lights;
    int         nlights;

    /* USD UsdLuxDomeLight metadata (max one per scene — extras after the
     * first are ignored, matching Hydra's behaviour). The renderer's
     * nu_attach_scene plumbs the HDR file path through to
     * gpu_load_environment so the scene's authored IBL replaces the
     * procedural-sky fallback. dome_hdr_path[0] == '\0' = not set. */
    char       dome_hdr_path[512];
    int        has_dome_light;     /* 1 when an active UsdLuxDomeLight was authored */
    float      dome_color[3];      /* inputs:color, default white */
    float      dome_intensity;     /* multiplier on the HDR sample (default 1.0) */
    float      dome_rotation_y;    /* degrees, around Y-up (default 0) */

    /* Arena memory (opaque, freed by scene_free) */
    void*      _arena;

    /* Keep stage handle alive for zero-copy pointers */
    void*      _stage;

    /* 1 = scene_free closes the stage; 0 = stage is borrowed (caller owns).
     * Borrowed stages are used by nu_load_usd_from_handle so a Python
     * pxr_compat shim and the renderer can share a single composed stage. */
    int        _owns_stage;

    /* MaterialCollection* (opaque — declared in material.h, freed by scene_free) */
    void*      materials;

    /* Fatal load failure recorded while streaming large scenes. A caller
     * should treat this as a hard error, not a partial/truncated scene. */
    int        load_failed;
    char       load_error[512];
} Scene;

/* Per-segment data emitted by scene_curve_to_segments() and consumed by
 * the GPU intersection shader. Layout is std430-friendly:
 * 8 floats = 32 B; tightly packed AoS so a single hit reads one
 * cacheline-pair (matches Phase 12.1 datacenter design from
 * RENDERER_BIG_PLAN.md). */
typedef struct {
    float    p0[3];          /*  0 — world-space segment start         */
    float    r0;             /* 12 — radius at start                   */
    float    p1[3];          /* 16 — world-space segment end           */
    float    r1;             /* 28 — radius at end                     */
} SceneCurveSegment;          /* 32 B */

/* Per-segment AABB (matches VkAabbPositionsKHR layout: 6 floats). */
typedef struct {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
} SceneCurveAabb;

/* Count linear segments emitted by a SceneCurve under Phase 11.A rules
 * (cubic curves are still loaded but emit zero linear segments — they'll
 * use cubic-Bezier AABB extraction in Phase 11.B). */
int scene_curve_count_segments(const SceneCurve* curve);

/* Materialise the SceneCurve as flat segment + AABB arrays in caller-
 * provided buffers. out_segments / out_aabbs must have capacity for at
 * least scene_curve_count_segments(curve) entries each. Returns the
 * number written. */
int scene_curve_to_segments(const SceneCurve* curve,
                            SceneCurveSegment* out_segments,
                            SceneCurveAabb*    out_aabbs);

/* Same as scene_curve_to_segments(), plus per-segment RGB colors. The color
 * buffer is float[segment_count * 3]. Vertex-varying displayColor is averaged
 * over each emitted segment; uniform displayColor and the grey fallback are
 * used when per-CV colors are absent. */
int scene_curve_to_segments_colored(const SceneCurve* curve,
                                    SceneCurveSegment* out_segments,
                                    SceneCurveAabb*    out_aabbs,
                                    float*             out_colors);

/* Set the time at which subsequent scene loads evaluate xform attributes.
 * NaN (the default) means "use authored default time". Called by
 * renderer.c before scene_load* so xformOp:translate.timeSamples (and
 * friends) resolve at the right frame. Cf. Vulkan port commit 18d341d. */
void   scene_set_load_time(double t);

/* Enable or disable material/UV extraction for subsequent scene_load* calls. */
void   scene_set_load_materials(int enabled);

/* Internal hook used by deferred visible extraction. When non-empty, the
 * native USD instance second pass materializes only these concrete instance
 * child paths for filtered roots, avoiding a full recursive expansion. */
void   scene_set_visible_instance_child_paths(const char* const* paths, int count);

/* Load a USD file (USDA or USDC). Returns NULL on failure. */
Scene* scene_load(const char* filepath);

/* Build a Scene from an already-open NanousdStage handle. The caller keeps
 * ownership of `stage` — scene_free will NOT close it. Used to share a
 * single composed stage between a Python pxr_compat shim and the renderer.
 * `stage_label` is a short string used in diagnostic messages; pass the
 * file path or layer identifier so log lines stay informative. */
Scene* scene_load_from_stage(void* stage, const char* stage_label);

/* Filtered variant used by lazy visible extraction. wanted_prims is a
 * byte-per-flat-prim bitmap; non-zero means extract that prim. NULL or an
 * empty bitmap disables filtering and matches scene_load_from_stage(). */
Scene* scene_load_from_stage_filtered(void* stage,
                                      const char* stage_label,
                                      const unsigned char* wanted_prims,
                                      int nprims_in_bitmap);

/* Last fatal scene-load error, suitable for forwarding through renderer APIs. */
const char* scene_last_error(void);

/* Internal hook used by deferred visible extraction. When set, PointInstancer
 * expansion conservatively materializes only prototype submeshes whose
 * instance-space AABB intersects at least one frustum plane set. `planes` is
 * float[num_cameras * 6 * 4], with normalized row-major planes. Passing NULL
 * or num_cameras <= 0 clears the hook. */
void   scene_set_point_instance_frusta(const float* planes, int num_cameras);

/* Free all scene memory + close the USD stage (if owned). */
void   scene_free(Scene* scene);

#ifdef __cplusplus
}
#endif

#endif /* NUSD_SCENE_H */
