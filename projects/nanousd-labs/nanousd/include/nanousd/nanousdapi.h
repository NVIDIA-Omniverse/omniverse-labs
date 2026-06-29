// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NANOUSDAPI_H
#define NANOUSDAPI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef NANOUSD_STATIC
#   define NANOUSD_API
#elif defined(_WIN32)
#   ifdef NANOUSD_BUILDING
#       define NANOUSD_API __declspec(dllexport)
#   else
#       define NANOUSD_API __declspec(dllimport)
#   endif
#else
#   define NANOUSD_API __attribute__((visibility("default")))
#endif

/* ============================================================
 * Opaque handles
 * ============================================================ */

typedef struct NanousdStage_s*     NanousdStage;
typedef struct NanousdPrim_s*      NanousdPrim;
typedef struct NanousdPath_s*      NanousdPath;
typedef struct NanousdListOp_s*    NanousdListOp;

/* Composition arc kind values. Kept in sync with nanousd::ArcType. */
enum {
    NANOUSD_ARC_SUBLAYER   = 0,
    NANOUSD_ARC_REFERENCE  = 1,
    NANOUSD_ARC_PAYLOAD    = 2,
    NANOUSD_ARC_NONE       = 3,
    NANOUSD_ARC_LOCAL      = 4,
    NANOUSD_ARC_INHERITS   = 5,
    NANOUSD_ARC_VARIANT    = 6,
    NANOUSD_ARC_SPECIALIZE = 7,
    NANOUSD_ARC_RELOCATE   = 8
};

enum {
    NANOUSD_FLAT_PRIM_INSTANCE       = 1 << 0,
    NANOUSD_FLAT_PRIM_PROTOTYPE      = 1 << 1,
    NANOUSD_FLAT_PRIM_IN_PROTOTYPE   = 1 << 2,
    NANOUSD_FLAT_PRIM_INSTANCE_PROXY = 1 << 3
};

typedef struct NanousdFlatPrim_s {
    int         struct_size;
    const char* path;
    const char* type_name;
    int         parent_index;
    int         depth;
    int         flags;
} NanousdFlatPrim;

enum {
    NANOUSD_COMPOSITION_ARC_DIRECT            = 1 << 0,
    NANOUSD_COMPOSITION_ARC_ANCESTRAL         = 1 << 1,
    NANOUSD_COMPOSITION_ARC_HAS_SOURCE_SPEC   = 1 << 2,
    NANOUSD_COMPOSITION_ARC_IDENTITY_MAPPING  = 1 << 3,
    NANOUSD_COMPOSITION_ARC_FALLBACK_IDENTITY = 1 << 4
};

typedef struct NanousdCompositionArc_s {
    int         struct_size;
    int         arc_type;
    int         flags;
    int         layer_index;
    double      offset;
    double      scale;
    const char* layer_path;
    const char* source_path;
    const char* target_path;
} NanousdCompositionArc;

/* ============================================================
 * Stage lifecycle
 * ============================================================ */

NANOUSD_API NanousdStage   nanousd_open(const char* filepath);
/* Open a stage with a population mask. The mask paths must be absolute prim
 * paths. Populated prims include mask paths and their ancestors. Passing
 * count=0 is equivalent to opening without a mask. */
NANOUSD_API NanousdStage   nanousd_open_masked(const char* filepath,
                                               const char* const* mask_paths,
                                               int mask_path_count);
NANOUSD_API void         nanousd_close(NanousdStage stage);
NANOUSD_API int          nanousd_isvalid(NanousdStage stage);
NANOUSD_API const char*  nanousd_error(NanousdStage stage);
NANOUSD_API const char*  nanousd_stage_get_root_layer_path(NanousdStage stage);

/* ----- Composed layer enumeration -----
 * Walk every layer that contributed opinions to the stage's composed
 * scene (root + every layer pulled in via reference / payload /
 * sublayer / inherits / specializes). Index 0 is the root layer.
 * Returns 0 on invalid stage. The returned strings are owned by the
 * stage and remain valid until nanousd_close. */
NANOUSD_API int          nanousd_stage_n_layers(NanousdStage stage);
NANOUSD_API const char*  nanousd_stage_layer_path(NanousdStage stage, int index);

/* ----- Asset resolution and resource reads -----
 * Resolve asset identifiers with the same default resolver used for
 * composition. Package-backed anchors return USDZ package identifiers such
 * as /abs/model.usdz[textures/diffuse.png].
 *
 * nanousd_read_asset_bytes reads an already-resolved location, including a
 * package identifier. Caller owns *out_data and must free it with
 * nanousd_free_bytes. */
NANOUSD_API int          nanousd_resolve_asset_path(const char* anchorLayerPath,
                                                    const char* assetPath,
                                                    char* out,
                                                    size_t out_size);
NANOUSD_API int          nanousd_stage_resolve_asset_path(NanousdStage stage,
                                                          const char* assetPath,
                                                          char* out,
                                                          size_t out_size);
NANOUSD_API int          nanousd_read_asset_bytes(const char* resolvedLocation,
                                                  unsigned char** out_data,
                                                  size_t* out_size);
NANOUSD_API void         nanousd_free_bytes(void* data);

/* ----- Per-layer spec / opinion queries -----
 * usdview's Layer-Stack and Composition tabs show, for a selected
 * prim or property, exactly which layers authored opinions on it.
 * These functions let callers reproduce that view without opening
 * each layer file as a separate stage.
 *
 * layerIdx selects from the same list nanousd_stage_layer_path
 * enumerates (index 0 = root layer). primPath is an absolute prim
 * path ("/World/Cube"); attrName is the bare attribute name (no
 * leading "."). Returns 0 / NULL on invalid args. */
NANOUSD_API int          nanousd_layer_has_prim_spec(NanousdStage stage, int layerIdx,
                                                     const char* primPath);
NANOUSD_API int          nanousd_layer_has_attr_opinion(NanousdStage stage, int layerIdx,
                                                        const char* primPath,
                                                        const char* attrName);
NANOUSD_API int          nanousd_layer_attr_nsamples(NanousdStage stage, int layerIdx,
                                                     const char* primPath,
                                                     const char* attrName);

/* Read a single layer's contribution to a list-op-typed field
 * (references / payload / inheritPaths / specializes / apiSchemas /
 * etc.) without combining across the opinion stack. Caller frees with
 * nanousd_listop_free. Returns NULL if the layer doesn't author an
 * opinion on that field for that prim. */
NANOUSD_API NanousdListOp nanousd_layer_prim_listop(NanousdStage stage, int layerIdx,
                                                    const char* primPath,
                                                    const char* field);

/* ----- Sublayer enumeration & per-layer time offset -----
 * The composed layer list (nanousd_stage_n_layers) is flat. These
 * accessors recover the sublayer tree (parent → children) and the
 * cumulative Sdf.LayerOffset for each layer in the stack. */
NANOUSD_API int          nanousd_layer_n_sublayers(NanousdStage stage, int layerIdx);
NANOUSD_API const char*  nanousd_layer_sublayer_path(NanousdStage stage, int layerIdx, int subIdx);
NANOUSD_API int          nanousd_layer_offset(NanousdStage stage, int layerIdx,
                                              double* offset, double* scale);

/* Recomposition is internal: composition-changing setters re-resolve and
 * refresh their handle before returning. There is no user-facing recompose. */

/* ============================================================
 * Stage metadata
 * ============================================================ */

NANOUSD_API double       nanousd_timecodes_per_second(NanousdStage stage);
NANOUSD_API double       nanousd_frames_per_second(NanousdStage stage);
NANOUSD_API double       nanousd_start_time(NanousdStage stage);
NANOUSD_API double       nanousd_end_time(NanousdStage stage);

/* Generic stage metadata (for domain-specific fields like metersPerUnit) */
NANOUSD_API double       nanousd_metadatad(NanousdStage stage, const char* key, int* ok);
NANOUSD_API const char*  nanousd_metadatas(NanousdStage stage, const char* key, int* ok);

/* Set stage metadata — returns 1 on success, 0 on failure */
NANOUSD_API int          nanousd_set_stage_metadatad(NanousdStage stage, const char* key, double value);
NANOUSD_API int          nanousd_set_stage_metadatas(NanousdStage stage, const char* key, const char* value);
NANOUSD_API int          nanousd_set_stage_metadata_token(NanousdStage stage, const char* key, const char* value);

/* ============================================================
 * Prim traversal
 * ============================================================ */

NANOUSD_API int          nanousd_nprims(NanousdStage stage);
NANOUSD_API NanousdPrim    nanousd_prim(NanousdStage stage, int index);
NANOUSD_API NanousdPrim    nanousd_primpath(NanousdStage stage, const char* path);
NANOUSD_API NanousdPrim    nanousd_defaultprim(NanousdStage stage);

/* Fill a flat depth-first snapshot of composed stage prims.
 * Instance prims are expanded with display-path instance proxy records flagged
 * with NANOUSD_FLAT_PRIM_INSTANCE_PROXY.
 * Returns the total record count. If out is NULL or max_count is 0, only the
 * count is returned. String pointers are owned by the stage and remain valid
 * until nanousd_close() or a composition-changing authoring call. */
NANOUSD_API int          nanousd_traverse_flat(NanousdStage stage,
                                               NanousdFlatPrim* out,
                                               int max_count);

NANOUSD_API int          nanousd_nchildren(NanousdPrim prim);
NANOUSD_API NanousdPrim    nanousd_child(NanousdPrim prim, int index);
NANOUSD_API NanousdPrim    nanousd_childname(NanousdPrim prim, const char* name);

/* ============================================================
 * Prim queries
 * ============================================================ */

NANOUSD_API const char*  nanousd_path(NanousdPrim prim);
NANOUSD_API const char*  nanousd_name(NanousdPrim prim);
NANOUSD_API const char*  nanousd_typename(NanousdPrim prim);
NANOUSD_API const char*  nanousd_kind(NanousdPrim prim);
NANOUSD_API int          nanousd_isactive(NanousdPrim prim);
NANOUSD_API int          nanousd_isdefined(NanousdPrim prim);
NANOUSD_API int          nanousd_isabstract(NanousdPrim prim);
NANOUSD_API int          nanousd_isinstanceable(NanousdPrim prim);
NANOUSD_API int          nanousd_prim_isvalid(NanousdPrim prim);

/* Schema queries */
NANOUSD_API int          nanousd_isa(NanousdPrim prim, const char* typeName);
NANOUSD_API int          nanousd_hasapi(NanousdPrim prim, const char* apiName);

/* ============================================================
 * Attribute access
 * ============================================================ */

NANOUSD_API int          nanousd_nattribs(NanousdPrim prim);
NANOUSD_API const char*  nanousd_attribname(NanousdPrim prim, int index);
NANOUSD_API int          nanousd_nauthored_attribs(NanousdPrim prim);
NANOUSD_API const char*  nanousd_authored_attribname(NanousdPrim prim, int index);
NANOUSD_API int          nanousd_hasattrib(NanousdPrim prim, const char* name);
NANOUSD_API const char*  nanousd_attribtype(NanousdPrim prim, const char* name);

/* Authored/composed property enumeration. Includes attributes and
 * relationships in composed property order. */
NANOUSD_API int          nanousd_nproperties(NanousdPrim prim);
NANOUSD_API const char*  nanousd_propertyname(NanousdPrim prim, int index);
NANOUSD_API int          nanousd_property_is_attribute(NanousdPrim prim, const char* name);
NANOUSD_API int          nanousd_property_is_relationship(NanousdPrim prim, const char* name);

/* Scalar reads */
NANOUSD_API float        nanousd_attribf(NanousdPrim prim, const char* name, int* ok);
NANOUSD_API double       nanousd_attribd(NanousdPrim prim, const char* name, int* ok);
NANOUSD_API int          nanousd_attribi(NanousdPrim prim, const char* name, int* ok);
NANOUSD_API int64_t      nanousd_attribi64(NanousdPrim prim, const char* name, int* ok);
NANOUSD_API int          nanousd_attribb(NanousdPrim prim, const char* name, int* ok);
NANOUSD_API const char*  nanousd_attribs(NanousdPrim prim, const char* name, int* ok);

/* Vector reads — returns 1 on success, 0 on failure */
NANOUSD_API int          nanousd_attribv2f(NanousdPrim prim, const char* name, float out[2]);
NANOUSD_API int          nanousd_attribv3f(NanousdPrim prim, const char* name, float out[3]);
NANOUSD_API int          nanousd_attribv4f(NanousdPrim prim, const char* name, float out[4]);
NANOUSD_API int          nanousd_attribv2d(NanousdPrim prim, const char* name, double out[2]);
NANOUSD_API int          nanousd_attribv3d(NanousdPrim prim, const char* name, double out[3]);
NANOUSD_API int          nanousd_attribv4d(NanousdPrim prim, const char* name, double out[4]);
NANOUSD_API int          nanousd_attribv2i(NanousdPrim prim, const char* name, int out[2]);
NANOUSD_API int          nanousd_attribv3i(NanousdPrim prim, const char* name, int out[3]);
NANOUSD_API int          nanousd_attribv4i(NanousdPrim prim, const char* name, int out[4]);

/* Matrix reads */
NANOUSD_API int          nanousd_attribm4d(NanousdPrim prim, const char* name, double out[16]);

/* Array attribute reads — returns number of elements written, -1 on error */
NANOUSD_API int          nanousd_attribarraylen(NanousdPrim prim, const char* name);
NANOUSD_API int          nanousd_attribarrayf(NanousdPrim prim, const char* name,
                                          float* out, int maxlen);
NANOUSD_API int          nanousd_attribarrayi(NanousdPrim prim, const char* name,
                                          int* out, int maxlen);
NANOUSD_API int          nanousd_attribarrayd(NanousdPrim prim, const char* name,
                                          double* out, int maxlen);
NANOUSD_API int          nanousd_attribarrayi64(NanousdPrim prim, const char* name,
                                             int64_t* out, int maxlen);

/* ============================================================
 * TimeSamples
 * ============================================================ */

NANOUSD_API int          nanousd_hassamples(NanousdPrim prim, const char* name);
NANOUSD_API int          nanousd_nsamplekeys(NanousdPrim prim, const char* name);
NANOUSD_API double       nanousd_samplekey(NanousdPrim prim, const char* name, int index);

NANOUSD_API float        nanousd_samplef(NanousdPrim prim, const char* name,
                                     double time, int* ok);
NANOUSD_API double       nanousd_sampled(NanousdPrim prim, const char* name,
                                     double time, int* ok);
NANOUSD_API int          nanousd_samplev3f(NanousdPrim prim, const char* name,
                                       double time, float out[3]);
NANOUSD_API int          nanousd_samplev3d(NanousdPrim prim, const char* name,
                                       double time, double out[3]);
NANOUSD_API int          nanousd_samplev2f(NanousdPrim prim, const char* name,
                                       double time, float out[2]);
NANOUSD_API int          nanousd_samplearrayf(NanousdPrim prim, const char* name,
                                          double time, float* out, int maxlen);
NANOUSD_API int          nanousd_samplearrayd(NanousdPrim prim, const char* name,
                                          double time, double* out, int maxlen);
NANOUSD_API int          nanousd_samplearrayi(NanousdPrim prim, const char* name,
                                          double time, int* out, int maxlen);

/* ============================================================
 * Relationships
 * ============================================================ */

NANOUSD_API int          nanousd_hasrel(NanousdPrim prim, const char* name);
NANOUSD_API int          nanousd_rel_authored(NanousdPrim prim, const char* name);
NANOUSD_API int          nanousd_nreltargets(NanousdPrim prim, const char* name);
NANOUSD_API const char*  nanousd_reltarget(NanousdPrim prim, const char* name, int index);

/* Read string/token metadata authored on a relationship property, e.g.
 * material:binding's bindMaterialAs. Returns "" and sets *ok=0 if missing.
 * Pointer is valid until the prim handle is freed or reused. */
NANOUSD_API const char*  nanousd_rel_metadatas(NanousdPrim prim,
                                                const char* relName,
                                                const char* key,
                                                int* ok);

/* ============================================================
 * Collections (CollectionAPI)
 * ============================================================ */

/* Query the evaluated membership of a collection instance on a prim.
 * instance_name is the CollectionAPI instance name, e.g. "rooms" for
 * CollectionAPI:rooms. Returned path strings are owned by the prim handle
 * and remain valid until the next collection query on that handle. */
NANOUSD_API int          nanousd_collection_nmembers(NanousdPrim prim,
                                                     const char* instance_name);
NANOUSD_API const char*  nanousd_collection_member(NanousdPrim prim,
                                                   const char* instance_name,
                                                   int index);
NANOUSD_API int          nanousd_collection_contains(NanousdPrim prim,
                                                     const char* instance_name,
                                                     const char* path);

/* ============================================================
 * Paths (spec Section 4)
 * ============================================================ */

NANOUSD_API NanousdPath    nanousd_path_parse(const char* text);
NANOUSD_API const char*  nanousd_path_str(NanousdPath path);
NANOUSD_API NanousdPath    nanousd_path_append_child(NanousdPath parent, const char* child);
NANOUSD_API NanousdPath    nanousd_path_append_property(NanousdPath prim, const char* prop);
NANOUSD_API NanousdPath    nanousd_path_parent(NanousdPath path);
NANOUSD_API const char*  nanousd_path_name(NanousdPath path);
NANOUSD_API int          nanousd_path_is_absolute(NanousdPath path);
NANOUSD_API int          nanousd_path_is_root(NanousdPath path);
NANOUSD_API int          nanousd_path_is_property(NanousdPath path);
NANOUSD_API int          nanousd_path_equal(NanousdPath a, NanousdPath b);
NANOUSD_API void         nanousd_path_free(NanousdPath path);

/* ============================================================
 * ListOps (spec Section 11)
 * ============================================================ */

NANOUSD_API NanousdListOp  nanousd_listop_create_explicit(const char** items, int count);
NANOUSD_API NanousdListOp  nanousd_listop_create(const char** prepend, int nprepend,
                                            const char** append,  int nappend,
                                            const char** delete_, int ndelete);
NANOUSD_API void         nanousd_listop_free(NanousdListOp op);

NANOUSD_API int          nanousd_listop_is_explicit(NanousdListOp op);
NANOUSD_API int          nanousd_listop_nitems(NanousdListOp op);
NANOUSD_API const char*  nanousd_listop_item(NanousdListOp op, int index);
NANOUSD_API int          nanousd_listop_nprepended(NanousdListOp op);
NANOUSD_API const char*  nanousd_listop_prepended(NanousdListOp op, int index);
NANOUSD_API int          nanousd_listop_nappended(NanousdListOp op);
NANOUSD_API const char*  nanousd_listop_appended(NanousdListOp op, int index);
NANOUSD_API int          nanousd_listop_ndeleted(NanousdListOp op);
NANOUSD_API const char*  nanousd_listop_deleted(NanousdListOp op, int index);

NANOUSD_API NanousdListOp  nanousd_listop_combine(NanousdListOp stronger, NanousdListOp weaker);

/* Read a listop field from prim metadata (e.g. "apiSchemas", "references") */
NANOUSD_API NanousdListOp  nanousd_prim_listop(NanousdPrim prim, const char* field);

/* ============================================================
 * Vec/Matrix/Quaternion utilities (spec-defined math)
 *
 * Stateless operations on raw arrays. No opaque handles needed.
 * ============================================================ */

/* Vec3 */
NANOUSD_API float        nanousd_dot3f(const float a[3], const float b[3]);
NANOUSD_API double       nanousd_dot3d(const double a[3], const double b[3]);
NANOUSD_API float        nanousd_length3f(const float v[3]);
NANOUSD_API double       nanousd_length3d(const double v[3]);
NANOUSD_API void         nanousd_normalize3f(const float v[3], float out[3]);
NANOUSD_API void         nanousd_normalize3d(const double v[3], double out[3]);
NANOUSD_API void         nanousd_cross3f(const float a[3], const float b[3], float out[3]);
NANOUSD_API void         nanousd_cross3d(const double a[3], const double b[3], double out[3]);

/* Matrix4d (row-major double[16]) */
NANOUSD_API void         nanousd_mul_m4d(const double a[16], const double b[16], double out[16]);
NANOUSD_API void         nanousd_transform_point3d(const double m[16], const double p[3],
                                                double out[3]);

/* Quaternion (double[4] = w, i, j, k) */
NANOUSD_API void         nanousd_quat_slerp(const double a[4], const double b[4],
                                         double t, double out[4]);
NANOUSD_API void         nanousd_quat_to_matrix(const double q[4], double out[16]);

/* ============================================================
 * Attribute write operations
 *
 * All setters return 1 on success, 0 on failure.
 * Writes mutate the in-memory composed layer.
 * ============================================================ */

/* Scalar setters */
NANOUSD_API int          nanousd_set_attribf(NanousdPrim prim, const char* name, float value);
NANOUSD_API int          nanousd_set_attribd(NanousdPrim prim, const char* name, double value);
NANOUSD_API int          nanousd_set_attribi(NanousdPrim prim, const char* name, int value);
NANOUSD_API int          nanousd_set_attribs(NanousdPrim prim, const char* name, const char* value);
NANOUSD_API int          nanousd_set_attribb(NanousdPrim prim, const char* name, int value);
NANOUSD_API int          nanousd_set_attribi64(NanousdPrim prim, const char* name, int64_t value);

/* Vector setters */
NANOUSD_API int          nanousd_set_attribv2f(NanousdPrim prim, const char* name, const float v[2]);
NANOUSD_API int          nanousd_set_attribv3f(NanousdPrim prim, const char* name, const float v[3]);
NANOUSD_API int          nanousd_set_attribv4f(NanousdPrim prim, const char* name, const float v[4]);
NANOUSD_API int          nanousd_set_attribv2d(NanousdPrim prim, const char* name, const double v[2]);
NANOUSD_API int          nanousd_set_attribv3d(NanousdPrim prim, const char* name, const double v[3]);
NANOUSD_API int          nanousd_set_attribv4d(NanousdPrim prim, const char* name, const double v[4]);
NANOUSD_API int          nanousd_set_attribv2i(NanousdPrim prim, const char* name, const int v[2]);
NANOUSD_API int          nanousd_set_attribv3i(NanousdPrim prim, const char* name, const int v[3]);
NANOUSD_API int          nanousd_set_attribv4i(NanousdPrim prim, const char* name, const int v[4]);

/* Matrix setter */
NANOUSD_API int          nanousd_set_attribm4d(NanousdPrim prim, const char* name, const double v[16]);

/* Array setters */
NANOUSD_API int          nanousd_set_attribarrayf(NanousdPrim prim, const char* name,
                                               const float* data, int count);
NANOUSD_API int          nanousd_set_attribarrayd(NanousdPrim prim, const char* name,
                                               const double* data, int count);
NANOUSD_API int          nanousd_set_attribarrayi(NanousdPrim prim, const char* name,
                                               const int* data, int count);

/* Time sample setters */
NANOUSD_API int          nanousd_set_samplef(NanousdPrim prim, const char* name,
                                          double time, float value);
NANOUSD_API int          nanousd_set_sampled(NanousdPrim prim, const char* name,
                                          double time, double value);
NANOUSD_API int          nanousd_set_samplev3f(NanousdPrim prim, const char* name,
                                            double time, const float v[3]);
NANOUSD_API int          nanousd_set_samplev3d(NanousdPrim prim, const char* name,
                                            double time, const double v[3]);
NANOUSD_API int          nanousd_set_samplev4f(NanousdPrim prim, const char* name,
                                            double time, const float v[4]);
NANOUSD_API int          nanousd_set_sampleqf(NanousdPrim prim, const char* name,
                                           double time, const float v[4]);
NANOUSD_API int          nanousd_set_sample_token(NanousdPrim prim, const char* name,
                                               double time, const char* value);
NANOUSD_API int          nanousd_set_samplearrayf(NanousdPrim prim, const char* name,
                                               double time, const float* data, int count);
NANOUSD_API int          nanousd_set_samplearrayi(NanousdPrim prim, const char* name,
                                               double time, const int* data, int count);
NANOUSD_API int          nanousd_set_samplearrayv3f(NanousdPrim prim, const char* name,
                                                 double time, const float* data, int count);

/* Double-precision time sample setters */
NANOUSD_API int          nanousd_set_samplev2d(NanousdPrim prim, const char* name,
                                            double time, const double v[2]);
NANOUSD_API int          nanousd_set_samplev4d(NanousdPrim prim, const char* name,
                                            double time, const double v[4]);
NANOUSD_API int          nanousd_set_samplem4d(NanousdPrim prim, const char* name,
                                            double time, const double v[16]);
NANOUSD_API int          nanousd_set_samplearrayd(NanousdPrim prim, const char* name,
                                               double time, const double* data, int count);
NANOUSD_API int          nanousd_set_samplearrayv3d(NanousdPrim prim, const char* name,
                                                 double time, const double* data, int count);

/* Clear/block operations */
NANOUSD_API int          nanousd_clear_default(NanousdPrim prim, const char* name);
NANOUSD_API int          nanousd_clear_samples(NanousdPrim prim, const char* name);
NANOUSD_API int          nanousd_block_attrib(NanousdPrim prim, const char* name);

/* Create attribute (author new spec if not present) — returns 1 on success */
NANOUSD_API int          nanousd_create_attrib(NanousdPrim prim, const char* name,
                                            const char* typeName);

/* ============================================================
 * Bulk array access (GPU-friendly)
 *
 * Zero-copy pointer access returns a read-only pointer directly into
 * the internal contiguous storage. No memcpy, no allocation.
 * The pointer is valid until the prim handle is freed or the value
 * is modified. Returns NULL if the attribute is not a typed array
 * of the requested type. Sets *count to number of elements.
 *
 * IMPORTANT: These functions only return stable pointers for authored
 * (written) array values. Schema-only fallback arrays are NOT supported
 * for zero-copy access — use the copying variants (nanousd_attribarrayf
 * etc.) instead. Schema registration (nanousd_register_schemas_json) does
 * not invalidate existing pointers.
 * ============================================================ */

NANOUSD_API const float*  nanousd_arraydataf(NanousdPrim prim, const char* name, int* count);
NANOUSD_API const double* nanousd_arraydatad(NanousdPrim prim, const char* name, int* count);
NANOUSD_API const int*    nanousd_arraydatai(NanousdPrim prim, const char* name, int* count);

/* Vec3 array reads — output is flat float/double buffer (3 components per vec3).
 * maxcount is number of vec3 elements (not floats). Returns count written, -1 on error. */
NANOUSD_API int          nanousd_attribarrayv3f(NanousdPrim prim, const char* name,
                                             float* out, int maxcount);
NANOUSD_API int          nanousd_attribarrayv3d(NanousdPrim prim, const char* name,
                                             double* out, int maxcount);

/* Vec3 array setters — input is flat buffer (3 components per vec3).
 * count is number of vec3 elements (not floats). */
NANOUSD_API int          nanousd_set_attribarrayv3f(NanousdPrim prim, const char* name,
                                                 const float* data, int count);
NANOUSD_API int          nanousd_set_attribarrayv3d(NanousdPrim prim, const char* name,
                                                 const double* data, int count);

/* ============================================================
 * Quaternion attribute read/write
 *
 * Quaternion layout: float[4] = {w, i, j, k} (real part first,
 * matching USD's USDA text format). Internal conversion to/from
 * the storage layout is handled automatically.
 * ============================================================ */

NANOUSD_API int          nanousd_attribqf(NanousdPrim prim, const char* name, float out[4]);
NANOUSD_API int          nanousd_attribqd(NanousdPrim prim, const char* name, double out[4]);
NANOUSD_API int          nanousd_set_attribqf(NanousdPrim prim, const char* name, const float v[4]);
NANOUSD_API int          nanousd_set_attribqd(NanousdPrim prim, const char* name, const double v[4]);

/* ============================================================
 * Relationship write operations
 * ============================================================ */

/* Create a relationship on a prim. Must be called before setting targets. */
NANOUSD_API int          nanousd_create_rel(NanousdPrim prim, const char* name);

/* Replace all targets — relationship must exist (see nanousd_create_rel).
 * targets is array of path strings. */
NANOUSD_API int          nanousd_set_reltargets(NanousdPrim prim, const char* name,
                                             const char** targets, int count);
/* Append a single target — relationship must exist (see nanousd_create_rel). */
NANOUSD_API int          nanousd_add_reltarget(NanousdPrim prim, const char* name,
                                            const char* target);
/* Remove all targets */
NANOUSD_API int          nanousd_clear_reltargets(NanousdPrim prim, const char* name);

/* ============================================================
 * Stage creation
 * ============================================================ */

/* Create an empty in-memory stage */
NANOUSD_API NanousdStage   nanousd_create(void);

/* ============================================================
 * Prim creation
 * ============================================================ */

/* Define a prim at the given path with optional type name.
 * Creates ancestor prims as needed. Returns prim handle or NULL.
 * Specifier defaults to "def". */
NANOUSD_API NanousdPrim    nanousd_define_prim(NanousdStage stage, const char* path,
                                          const char* typeName);

/* Define a prim with an explicit specifier ("def", "over", "class").
 * Equivalent to nanousd_define_prim followed by nanousd_set_specifier. */
NANOUSD_API NanousdPrim    nanousd_define_prim_s(NanousdStage stage, const char* path,
                                            const char* typeName,
                                            const char* specifier);

/* Set the specifier on a prim ("def", "over", "class").
 * Returns 1 on success, 0 on failure. */
NANOUSD_API int          nanousd_set_specifier(NanousdPrim prim, const char* specifier);

/* ============================================================
 * Schema application
 * ============================================================ */

/* Apply an API schema (e.g. "PhysicsRigidBodyAPI", "PhysicsLimitAPI:transX").
 * Returns 1 on success, 0 on failure. */
NANOUSD_API int          nanousd_apply_api(NanousdPrim prim, const char* schemaName);

/* ============================================================
 * Composition arc write operations
 *
 * These mutate the root layer and trigger full recomposition.
 * WARNING: After these calls, all existing NanousdPrim handles
 * (except the one passed in) become stale. Re-acquire them
 * with nanousd_primpath().
 * ============================================================ */

/* Add a reference to an asset, optionally rooted at a prim path.
 * assetPath: the asset to reference (e.g. "./model.usd"), or NULL for
 *            an internal (same-layer) reference.
 * primPath:  root prim in the referenced asset (e.g. "/Model"), or NULL
 *            to use the referenced asset's defaultPrim.
 * Prepends to the prim's references listop.
 * Returns 1 on success, 0 on failure. */
NANOUSD_API int          nanousd_add_reference(NanousdPrim prim, const char* assetPath,
                                            const char* primPath);

/* Add a payload arc. Same shape as nanousd_add_reference; a payload is
 * a "lazy reference" that is loaded on demand (spec §6.3.5). Prepends
 * to the prim's payload listop. Returns 1 on success. */
NANOUSD_API int          nanousd_add_payload(NanousdPrim prim, const char* assetPath,
                                          const char* primPath);

/* Add an inherits arc to the prim. primPath must be an absolute path
 * to a prim acting as a class (typically a "/_class_..." prim with
 * specifier=class). Prepends to the prim's inheritPaths listop
 * (spec §6.3.5.5). Returns 1 on success. */
NANOUSD_API int          nanousd_add_inherit(NanousdPrim prim, const char* primPath);

/* Add a specializes arc to the prim. primPath must be an absolute
 * path. Prepends to the prim's specializes listop (spec §6.3.5.6).
 * Returns 1 on success. */
NANOUSD_API int          nanousd_add_specialize(NanousdPrim prim, const char* primPath);

/* Remove the index-th item from the named composition-arc listop.
 *   field: one of "references", "payload", "inheritPaths", "specializes",
 *          "apiSchemas".
 *   listOpKind: 0=explicit, 1=prepended, 2=appended, 3=deleted — selects
 *               which sublist of the listop the index applies to.
 *   index:  0-based index into the chosen sublist.
 * Returns 1 on success, 0 on invalid arguments / out-of-range. */
NANOUSD_API int          nanousd_remove_listop_item(NanousdPrim prim, const char* field,
                                                  int listOpKind, int index);

/* Convenience wrappers: remove the index-th *prepended* item from
 * the corresponding listop. Equivalent to nanousd_remove_listop_item
 * with listOpKind=1. Returns 1 on success. */
NANOUSD_API int          nanousd_remove_reference(NanousdPrim prim, int index);
NANOUSD_API int          nanousd_remove_payload(NanousdPrim prim, int index);
NANOUSD_API int          nanousd_remove_inherit(NanousdPrim prim, int index);
NANOUSD_API int          nanousd_remove_specialize(NanousdPrim prim, int index);

/* ============================================================
 * Prim-state writers
 *
 * Mutate prim metadata that affects composition and traversal.
 * Each of these recomposes the stage, which invalidates EVERY
 * outstanding NanousdPrim handle for this stage -- not just the passed
 * prim and its descendants, but siblings and unrelated prims too. After
 * any of these calls the only safe operation on a handle you did not pass
 * in is nanousd_freeprim(); calling an accessor (nanousd_typename,
 * nanousd_child, ...) on it is undefined behavior. Re-acquire live handles
 * with nanousd_primpath().
 * ============================================================ */

/* Activate or deactivate a prim. An inactive prim and its descendants
 * are masked from composed traversal (spec §6.3.6). active=0 deactivates,
 * active=1 activates. Returns 1 on success.
 *
 * To reactivate, reuse the handle you deactivated with: a deactivated prim
 * is masked, so nanousd_primpath() cannot re-acquire it, but the handle
 * retains the path needed to author active=1 back through. */
NANOUSD_API int          nanousd_set_active(NanousdPrim prim, int active);

/* Mark a prim instanceable (instanceable=1) or not (instanceable=0).
 * An instanceable prim with composition arcs participates in native
 * instancing (spec §6.6). Returns 1 on success. */
NANOUSD_API int          nanousd_set_instanceable(NanousdPrim prim, int instanceable);

/* Remove an applied API schema from this prim's apiSchemas listop
 * (counterpart to nanousd_apply_api). schemaName is the full name
 * including instance suffix (e.g. "PhysicsLimitAPI:transX").
 * Returns 1 on success, 0 if not currently applied. */
NANOUSD_API int          nanousd_remove_api(NanousdPrim prim, const char* schemaName);

/* Remove the prim and all of its descendants from the root layer.
 * This recomposes the stage, which invalidates EVERY outstanding
 * NanousdPrim handle for this stage -- the passed prim, its descendants,
 * and siblings/unrelated prims alike. After this call the only safe
 * operation on any such handle is nanousd_freeprim(); calling an accessor
 * (nanousd_typename, nanousd_child, ...) on it is undefined behavior.
 * Re-acquire live handles with nanousd_primpath(). The caller still owns
 * every handle: free each with nanousd_freeprim() as usual.
 * Returns 1 on success. */
NANOUSD_API int          nanousd_remove_prim(NanousdPrim prim);

/* ============================================================
 * Variant set authoring
 *
 * A variant set is declared by adding its name to the prim's
 * variantSetNames field (spec §11.2). A variant within a set is
 * declared by creating a Variant-typed spec at the variant-selection
 * sub-path. Authoring into a variant body (so that subsequent
 * set_attrib_* / define_prim calls land inside the variant) requires
 * the edit-target stack which has not yet been ported — see the
 * branch design notes.
 * ============================================================ */

/* Declare a new variant set on a prim. Idempotent — duplicate names
 * are ignored. Returns 1 on success. */
NANOUSD_API int          nanousd_create_variantset(NanousdPrim prim, const char* setName);

/* Declare a new variant within a variant set on a prim. Creates the
 * variant set first if it does not exist. Idempotent — declaring the
 * same variant twice is a no-op. Returns 1 on success. */
NANOUSD_API int          nanousd_create_variant(NanousdPrim prim,
                                              const char* setName,
                                              const char* variantName);

/* ============================================================
 * P1 — Matrix3d read/write (9-element double[], row-major)
 * ============================================================ */

NANOUSD_API int          nanousd_attribm3d(NanousdPrim prim, const char* name, double out[9]);
NANOUSD_API int          nanousd_set_attribm3d(NanousdPrim prim, const char* name,
                                            const double v[9]);

/* ============================================================
 * P1 — String/token array read/write
 *
 * Covers string[], token[], and asset[] arrays.
 * ============================================================ */

/* Returns number of strings in the array, -1 on error */
NANOUSD_API int          nanousd_attribarrays_len(NanousdPrim prim, const char* name);

/* Returns string at index (valid until prim freed or value modified) */
NANOUSD_API const char*  nanousd_attribarrays(NanousdPrim prim, const char* name, int index);

/* Set a string/token array from an array of C strings */
NANOUSD_API int          nanousd_set_attribarrays(NanousdPrim prim, const char* name,
                                               const char** strings, int count);

/* ============================================================
 * P1 — Asset path read
 *
 * Read an asset-typed attribute as a string.
 * Returns the asset path string, NULL on error.
 * ============================================================ */

NANOUSD_API const char*  nanousd_attribasset(NanousdPrim prim, const char* name, int* ok);

/* ============================================================
 * XformOp — composed local transform (geometry spec: xformable.md)
 *
 * Computes the 4x4 local transform from a prim's xformOpOrder stack.
 * time: time code (pass NaN for default time).
 * out: 16-element row-major double array.
 * resetXformStack: if non-null, set to 1 when !resetXformStack! is present.
 * Returns 1 on success, 0 on failure.
 * ============================================================ */

NANOUSD_API int          nanousd_get_local_transform(NanousdPrim prim, double time,
                                                  double out[16],
                                                  int* resetXformStack);

/* ============================================================
 * Binary write
 * ============================================================ */

/* Serialise the in-memory stage to a USDC binary crate file.
 * Returns 1 on success, 0 on failure. */
NANOUSD_API int          nanousd_write_usdc(NanousdStage stage, const char* filepath);

/* Serialise the in-memory stage to a USDZ package file.
 * Writes the flattened stage as the first package entry. Returns 1 on success,
 * 0 on failure. */
NANOUSD_API int          nanousd_write_usdz(NanousdStage stage, const char* filepath);

/* ============================================================
 * USDA text write
 * ============================================================ */

/* Serialise the in-memory stage to a USDA text file.
 * Writes the root layer only. Returns 1 on success, 0 on failure. */
NANOUSD_API int          nanousd_write_usda(NanousdStage stage, const char* filepath);

/* Serialise the in-memory stage to a malloc'd USDA string.
 * Returns NULL on failure. Caller must free with nanousd_free_string(). */
NANOUSD_API const char*  nanousd_write_usda_string(NanousdStage stage);

/* Free a string returned by nanousd_write_usda_string(). */
NANOUSD_API void         nanousd_free_string(const char* str);

/* ============================================================
 * Token/Asset attribute readers and setters
 *
 * The standard nanousd_attribs() / nanousd_set_attribs() handle string-typed values.
 * These variants handle token-typed or asset-typed values, which are distinct
 * types in the USD spec. Token readers return NULL/ok=0 for string-typed values,
 * and string readers return ""/ok=0 for token-typed values.
 * ============================================================ */

/** Read a token-typed scalar attribute. Returns "" and sets *ok=0 on failure. */
NANOUSD_API const char*  nanousd_attrib_token(NanousdPrim prim, const char* name, int* ok);

NANOUSD_API int          nanousd_set_attrib_token(NanousdPrim prim, const char* name,
                                               const char* value);
NANOUSD_API int          nanousd_set_attrib_asset(NanousdPrim prim, const char* name,
                                               const char* value);

/* Token array setter — stores as token[] type */
NANOUSD_API int          nanousd_set_attribarraytokens(NanousdPrim prim, const char* name,
                                                    const char** values, int count);

/* Token array readers — parallel to nanousd_attribarrays_len/nanousd_attribarrays
 * but for token-typed arrays. Returns -1/NULL on type mismatch. */
NANOUSD_API int          nanousd_attribarraytokens_len(NanousdPrim prim, const char* name);
NANOUSD_API const char*  nanousd_attribarraytokens(NanousdPrim prim, const char* name, int index);

/* ============================================================
 * Schema registration
 * ============================================================ */

/* Register additional schemas from a JSON string. Uses the same format
 * as the built-in schema definitions. Schemas can be registered at any
 * time, even after stages are opened — prim definition caches are
 * lazily invalidated. Returns 1 on success, 0 on failure. */
NANOUSD_API int          nanousd_register_schemas_json(const char* json);

/* ============================================================
 * Prim metadata
 *
 * Generic prim metadata access for scalar fields (string, double, token).
 * These walk the opinion stack for reads and write to the root layer for
 * sets. ListOp-typed composition fields (references, inheritPaths, etc.)
 * are not supported — use the dedicated composition arc functions instead.
 * ============================================================ */

/* Read a string or token prim metadata field. Returns "" and sets *ok=0
 * if not authored. The returned pointer is valid until the prim handle
 * is freed or the value is modified. */
NANOUSD_API const char*  nanousd_prim_metadatas(NanousdPrim prim, const char* key, int* ok);

/* Read a numeric prim metadata field. Returns 0.0 and sets *ok=0 if not authored. */
NANOUSD_API double       nanousd_prim_metadatad(NanousdPrim prim, const char* key, int* ok);

/* Set a string-typed prim metadata field. Returns 1 on success, 0 on failure. */
NANOUSD_API int          nanousd_set_prim_metadatas(NanousdPrim prim, const char* key,
                                                 const char* value);

/* Set a double-typed prim metadata field. Returns 1 on success, 0 on failure. */
NANOUSD_API int          nanousd_set_prim_metadatad(NanousdPrim prim, const char* key,
                                                 double value);

/* Set a token-typed prim metadata field (e.g. "kind", "purpose").
 * Returns 1 on success, 0 on failure. */
NANOUSD_API int          nanousd_set_prim_metadata_token(NanousdPrim prim, const char* key,
                                                      const char* value);

/* ============================================================
 * Attribute metadata & connections
 * ============================================================ */

/* Returns interpolation string ("vertex", "faceVarying", etc.) or NULL */
NANOUSD_API const char*  nanousd_attrib_interpolation(NanousdPrim prim, const char* name);

/* Returns 1 if attribute is authored (has a spec), 0 otherwise */
NANOUSD_API int          nanousd_attrib_authored(NanousdPrim prim, const char* name);

/* Authored attribute colorSpace metadata. Returns "" and sets *ok=0 if absent. */
NANOUSD_API const char*  nanousd_attrib_colorspace(NanousdPrim prim, const char* name, int* ok);

/* Resolved attribute color space per the color chapter inheritance rules. */
NANOUSD_API const char*  nanousd_attrib_resolved_colorspace(NanousdPrim prim, const char* name);

/* Author or clear attribute colorSpace metadata. */
NANOUSD_API int          nanousd_set_attrib_colorspace(NanousdPrim prim, const char* name,
                                                    const char* colorSpace);
NANOUSD_API int          nanousd_clear_attrib_colorspace(NanousdPrim prim, const char* name);

/* Resolved prim color space from ColorSpaceAPI inheritance. */
NANOUSD_API const char*  nanousd_prim_resolved_colorspace(NanousdPrim prim);

/* Returns 1 if attribute has connection targets, 0 otherwise */
NANOUSD_API int          nanousd_hasconnections(NanousdPrim prim, const char* name);

/* Returns number of connection targets for the attribute */
NANOUSD_API int          nanousd_nconnections(NanousdPrim prim, const char* name);

/* Returns the Nth connection target path string */
NANOUSD_API const char*  nanousd_connection(NanousdPrim prim, const char* name, int index);

/* Returns parent prim handle (caller must free with nanousd_freeprim), NULL if root */
NANOUSD_API NanousdPrim    nanousd_parent(NanousdPrim prim);

/* Returns string at index from a string/token array attribute (alias for attribarrays
 * with token array fallback). Valid until prim freed or value modified. */
NANOUSD_API const char*  nanousd_attribarrays_elem(NanousdPrim prim, const char* name, int index);

/* ============================================================
 * Instancing (spec Section 11.3)
 * ============================================================ */

/* Returns the number of prototype roots composed for this stage. */
NANOUSD_API int          nanousd_stage_nprototypes(NanousdStage stage);

/* Returns the Nth prototype root (caller must free), NULL on error. */
NANOUSD_API NanousdPrim    nanousd_stage_prototype(NanousdStage stage, int index);

/* Returns 1 if the prim is an instance (shares a prototype), 0 otherwise */
NANOUSD_API int          nanousd_isinstance(NanousdPrim prim);

/* Returns 1 if the prim is a prototype root, 0 otherwise */
NANOUSD_API int          nanousd_isprototype(NanousdPrim prim);

/* Returns 1 if the prim is inside a prototype subtree, 0 otherwise */
NANOUSD_API int          nanousd_isinprototype(NanousdPrim prim);

/* Returns 1 if the prim is an instance proxy descendant, 0 otherwise. */
NANOUSD_API int          nanousd_isinstanceproxy(NanousdPrim prim);

/* Returns the prototype prim for an instance (caller must free), NULL if not an instance */
NANOUSD_API NanousdPrim    nanousd_prototype(NanousdPrim prim);

/* Map an instance root or instance-proxy descendant to its corresponding prim
 * in the prototype namespace (caller must free). Prototype prims map to
 * themselves. Returns NULL for ordinary non-instanced prims. */
NANOUSD_API NanousdPrim    nanousd_priminprototype(NanousdPrim prim);

/* Returns the number of instances sharing this prototype */
NANOUSD_API int          nanousd_ninstances(NanousdPrim prim);

/* Returns the Nth instance prim (caller must free), NULL on error */
NANOUSD_API NanousdPrim    nanousd_instance(NanousdPrim prim, int index);

/* Write a runtime instance-key diagnostic string for an instance root.
 * Returns the full required byte count excluding the trailing NUL. If out is
 * NULL or out_size is 0, only the required length is returned. The string is
 * for equality/debug grouping within the current nanousd implementation and
 * is invalidated by recomposition; callers must not persist its layout. */
NANOUSD_API int          nanousd_instance_key(NanousdPrim prim,
                                              char* out,
                                              size_t out_size);

/* Composition-source records for this composed prim, in strength order. */
NANOUSD_API int          nanousd_ncomposition_arcs(NanousdPrim prim);
NANOUSD_API int          nanousd_composition_arc(NanousdPrim prim,
                                                 int index,
                                                 NanousdCompositionArc* out);

/* ============================================================
 * Handle cleanup
 * ============================================================ */

NANOUSD_API void         nanousd_freeprim(NanousdPrim prim);

/* ============================================================
 * Composition diagnostics
 *
 * After nanousd_open() or nanousd_open_masked(), non-fatal composition issues
 * (missing sublayers, unresolvable references, etc.) are collected as
 * diagnostics rather than causing the stage to be invalid. Any diagnostic at
 * Warning or Error severity means the stage may not represent the scene
 * correctly.
 *
 * Severity: 0=Info, 1=Warning, 2=Error
 * Category: 0=MissingSublayer, 1=SublayerParseFail, 2=MissingReference,
 *           3=ReferenceParseFail, 4=MissingDefaultPrim, 5=MissingPayload,
 *           6=PayloadParseFail, 7=Other, 8=InvalidRelocate,
 *           9=MissingInheritTarget, 10=MissingSpecializeTarget,
 *           11=InvalidReferenceTarget, 12=InvalidPayloadTarget,
 *           13=InvalidRetimingScale
 * ArcType:  0=sublayer, 1=reference, 2=payload, 3=none
 * ============================================================ */

typedef struct NanousdDiagnostic_s {
    int         severity;
    int         category;
    const char* message;
    const char* prim_path;
    const char* layer_path;
    const char* asset_path;
    int         arc_type;
} NanousdDiagnostic;

/* Returns malloc'd array of diagnostics; sets *count. Caller frees with
 * nanousd_free_diagnostics(). Returns NULL and sets count=0 if none. */
NANOUSD_API NanousdDiagnostic* nanousd_diagnostics(NanousdStage stage, int* count);
NANOUSD_API void             nanousd_free_diagnostics(NanousdDiagnostic* diags, int count);

/* JSON array of all diagnostics. Caller frees with nanousd_free_string().
 * Returns "[]" (allocated) if no diagnostics. */
NANOUSD_API const char*      nanousd_diagnostics_json(NanousdStage stage);

/* ============================================================
 * Variants (spec Section 11.2)
 *
 * A variant set is a named list of alternative prim bodies (variants)
 * declared on a prim. Exactly one variant per set is selected, and that
 * variant's content is composed into the prim's namespace. Reads below
 * compose listop / dictionary fields across the opinion stack. Selection
 * writes target the layer at layerIndex, recompose internally, and refresh
 * the passed-in prim handle before returning.
 * ============================================================ */

/* Number of variant sets declared on this prim (composed across layers). */
NANOUSD_API int          nanousd_nvariantsets(NanousdPrim prim);

/* Name of the Nth variant set. Returns "" if index is out of range.
 * Pointer is valid until the prim handle is freed or modified. */
NANOUSD_API const char*  nanousd_variantsetname(NanousdPrim prim, int index);

/* 1 if the named variant set is declared on this prim, 0 otherwise. */
NANOUSD_API int          nanousd_hasvariantset(NanousdPrim prim, const char* setName);

/* Number of variants declared in the named set, unioned across all
 * layers that author the set. */
NANOUSD_API int          nanousd_nvariants(NanousdPrim prim, const char* setName);

/* Name of the Nth variant in the named set. Returns "" if out of range.
 * Pointer is valid until the prim handle is freed or modified. */
NANOUSD_API const char*  nanousd_variantname(NanousdPrim prim,
                                               const char* setName, int index);

/* The selected variant for the named set (strongest-wins across the
 * opinion stack). Returns "" if no selection is authored. Pointer is
 * valid until the prim handle is freed or modified. */
NANOUSD_API const char*  nanousd_variantselection(NanousdPrim prim,
                                                    const char* setName);

/* Author a variant selection on the layer at layerIndex. Pass
 * variantName="" or NULL to clear the selection. Returns 1 on success,
 * 0 on failure (invalid prim, out-of-range layer, or no writable spec).
 * The stage is recomposed internally and the passed-in prim handle is
 * refreshed before return, so the new selection is immediately visible to
 * queries. Other live handles into the stage become stale — re-acquire. */
NANOUSD_API int          nanousd_setvariantselection(NanousdPrim prim,
                                                       const char* setName,
                                                       const char* variantName,
                                                       int layerIndex);

#ifdef __cplusplus
}
#endif

#endif /* NANOUSDAPI_H */
