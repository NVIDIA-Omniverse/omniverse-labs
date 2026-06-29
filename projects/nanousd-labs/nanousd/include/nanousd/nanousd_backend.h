// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NANOUSD_BACKEND_H
#define NANOUSD_BACKEND_H

/*
 * Backend ABI contract for nanousd.
 *
 * A backend is a shared library that exports a single C entry point:
 *
 *     NanousdBackend_v1* nanousd_create_backend_v1(void);
 *
 * The returned struct contains function pointers that implement every
 * operation in the public C API (nanousdapi.h).  The front-end library
 * (nanousdapi) loads the backend at runtime via dlopen / LoadLibrary and
 * dispatches every public call through this vtable.
 *
 * Opaque handles (NanousdStage, NanousdPrim, NanousdPath, NanousdListOp) are
 * allocated and owned by the backend — the front-end treats them as
 * pass-through pointers.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward-declare the opaque handle types (same as nanousdapi.h) */
typedef struct NanousdStage_s*   NanousdStage;
typedef struct NanousdPrim_s*    NanousdPrim;
typedef struct NanousdPath_s*    NanousdPath;
typedef struct NanousdListOp_s*  NanousdListOp;
typedef struct NanousdDiagnostic_s NanousdDiagnostic;
struct NanousdFlatPrim_s;
struct NanousdCompositionArc_s;

/* ============================================================
 * Version 1 backend vtable
 * ============================================================ */

typedef struct NanousdBackend_v1 {
    /* --- Original v1 functions (do not reorder) --- */

    /* Stage lifecycle */
    NanousdStage  (*open)(const char* filepath);
    void        (*close)(NanousdStage stage);
    int         (*isvalid)(NanousdStage stage);
    const char* (*error)(NanousdStage stage);

    /* Stage metadata */
    double      (*timecodes_per_second)(NanousdStage stage);
    double      (*frames_per_second)(NanousdStage stage);
    double      (*start_time)(NanousdStage stage);
    double      (*end_time)(NanousdStage stage);

    /* Prim traversal */
    int         (*nprims)(NanousdStage stage);
    NanousdPrim   (*prim)(NanousdStage stage, int index);
    NanousdPrim   (*primpath)(NanousdStage stage, const char* path);
    NanousdPrim   (*defaultprim)(NanousdStage stage);
    int         (*nchildren)(NanousdPrim prim);
    NanousdPrim   (*child)(NanousdPrim prim, int index);
    NanousdPrim   (*childname)(NanousdPrim prim, const char* name);

    /* Prim queries */
    const char* (*path)(NanousdPrim prim);
    const char* (*name)(NanousdPrim prim);
    const char* (*type_name)(NanousdPrim prim);
    const char* (*kind)(NanousdPrim prim);
    int         (*isactive)(NanousdPrim prim);
    int         (*isdefined)(NanousdPrim prim);
    int         (*isabstract)(NanousdPrim prim);
    int         (*isinstanceable)(NanousdPrim prim);
    int         (*prim_isvalid)(NanousdPrim prim);

    /* Attribute access */
    int         (*nattribs)(NanousdPrim prim);
    const char* (*attribname)(NanousdPrim prim, int index);
    int         (*hasattrib)(NanousdPrim prim, const char* name);
    const char* (*attribtype)(NanousdPrim prim, const char* name);
    float       (*attribf)(NanousdPrim prim, const char* name, int* ok);
    double      (*attribd)(NanousdPrim prim, const char* name, int* ok);
    int         (*attribi)(NanousdPrim prim, const char* name, int* ok);
    const char* (*attribs)(NanousdPrim prim, const char* name, int* ok);

    /* Vector reads */
    int         (*attribv2f)(NanousdPrim prim, const char* name, float out[2]);
    int         (*attribv3f)(NanousdPrim prim, const char* name, float out[3]);
    int         (*attribv4f)(NanousdPrim prim, const char* name, float out[4]);
    int         (*attribv2d)(NanousdPrim prim, const char* name, double out[2]);
    int         (*attribv3d)(NanousdPrim prim, const char* name, double out[3]);
    int         (*attribv4d)(NanousdPrim prim, const char* name, double out[4]);
    int         (*attribv2i)(NanousdPrim prim, const char* name, int out[2]);
    int         (*attribv3i)(NanousdPrim prim, const char* name, int out[3]);
    int         (*attribv4i)(NanousdPrim prim, const char* name, int out[4]);

    /* Matrix reads */
    int         (*attribm4d)(NanousdPrim prim, const char* name, double out[16]);

    /* Array reads */
    int         (*attribarraylen)(NanousdPrim prim, const char* name);
    int         (*attribarrayf)(NanousdPrim prim, const char* name, float* out, int maxlen);
    int         (*attribarrayi)(NanousdPrim prim, const char* name, int* out, int maxlen);

    /* TimeSamples */
    int         (*hassamples)(NanousdPrim prim, const char* name);
    int         (*nsamplekeys)(NanousdPrim prim, const char* name);
    double      (*samplekey)(NanousdPrim prim, const char* name, int index);
    float       (*samplef)(NanousdPrim prim, const char* name, double time, int* ok);
    int         (*samplev3f)(NanousdPrim prim, const char* name, double time, float out[3]);
    int         (*samplev3d)(NanousdPrim prim, const char* name, double time, double out[3]);

    /* Relationships */
    int         (*hasrel)(NanousdPrim prim, const char* name);

    /* Handle cleanup */
    void        (*freeprim)(NanousdPrim prim);

    /* --- Extensions (appended to preserve v1 ABI) --- */

    /* Generic stage metadata */
    double      (*metadatad)(NanousdStage stage, const char* key, int* ok);
    const char* (*metadatas)(NanousdStage stage, const char* key, int* ok);
    int         (*set_stage_metadatad)(NanousdStage stage, const char* key, double value);
    int         (*set_stage_metadatas)(NanousdStage stage, const char* key, const char* value);

    /* Schema queries */
    int         (*isa)(NanousdPrim prim, const char* typeName);
    int         (*hasapi)(NanousdPrim prim, const char* apiName);

    /* Additional scalar reads */
    int64_t     (*attribi64)(NanousdPrim prim, const char* name, int* ok);
    int         (*attribb)(NanousdPrim prim, const char* name, int* ok);

    /* Additional array reads */
    int         (*attribarrayd)(NanousdPrim prim, const char* name, double* out, int maxlen);

    /* Additional time sample reads */
    double      (*sampled)(NanousdPrim prim, const char* name, double time, int* ok);

    /* Relationship targets */
    int         (*nreltargets)(NanousdPrim prim, const char* name);
    const char* (*reltarget)(NanousdPrim prim, const char* name, int index);

    /* Paths */
    NanousdPath   (*path_parse)(const char* text);
    const char* (*path_str)(NanousdPath path);
    NanousdPath   (*path_append_child)(NanousdPath parent, const char* child);
    NanousdPath   (*path_append_property)(NanousdPath prim, const char* prop);
    NanousdPath   (*path_parent)(NanousdPath path);
    const char* (*path_name)(NanousdPath path);
    int         (*path_is_absolute)(NanousdPath path);
    int         (*path_is_root)(NanousdPath path);
    int         (*path_is_property)(NanousdPath path);
    int         (*path_equal)(NanousdPath a, NanousdPath b);
    void        (*path_free)(NanousdPath path);

    /* ListOps */
    NanousdListOp (*listop_create_explicit)(const char** items, int count);
    NanousdListOp (*listop_create)(const char** prepend, int nprepend,
                                  const char** append, int nappend,
                                  const char** delete_, int ndelete);
    void        (*listop_free)(NanousdListOp op);
    int         (*listop_is_explicit)(NanousdListOp op);
    int         (*listop_nitems)(NanousdListOp op);
    const char* (*listop_item)(NanousdListOp op, int index);
    int         (*listop_nprepended)(NanousdListOp op);
    const char* (*listop_prepended)(NanousdListOp op, int index);
    int         (*listop_nappended)(NanousdListOp op);
    const char* (*listop_appended)(NanousdListOp op, int index);
    int         (*listop_ndeleted)(NanousdListOp op);
    const char* (*listop_deleted)(NanousdListOp op, int index);
    NanousdListOp (*listop_combine)(NanousdListOp stronger, NanousdListOp weaker);
    NanousdListOp (*prim_listop)(NanousdPrim prim, const char* field);

    /* Vec/Matrix/Quaternion utilities */
    float       (*dot3f)(const float a[3], const float b[3]);
    double      (*dot3d)(const double a[3], const double b[3]);
    float       (*length3f)(const float v[3]);
    double      (*length3d)(const double v[3]);
    void        (*normalize3f)(const float v[3], float out[3]);
    void        (*normalize3d)(const double v[3], double out[3]);
    void        (*cross3f)(const float a[3], const float b[3], float out[3]);
    void        (*cross3d)(const double a[3], const double b[3], double out[3]);
    void        (*mul_m4d)(const double a[16], const double b[16], double out[16]);
    void        (*transform_point3d)(const double m[16], const double p[3], double out[3]);
    void        (*quat_slerp)(const double a[4], const double b[4], double t, double out[4]);
    void        (*quat_to_matrix)(const double q[4], double out[16]);

    /* --- Write operations (appended for ABI compat) --- */

    /* Scalar setters — return 1 on success, 0 on failure */
    int  (*set_attribf)(NanousdPrim prim, const char* name, float value);
    int  (*set_attribd)(NanousdPrim prim, const char* name, double value);
    int  (*set_attribi)(NanousdPrim prim, const char* name, int value);
    int  (*set_attribs)(NanousdPrim prim, const char* name, const char* value);
    int  (*set_attribb)(NanousdPrim prim, const char* name, int value);
    int  (*set_attribi64)(NanousdPrim prim, const char* name, int64_t value);

    /* Vector setters */
    int  (*set_attribv2f)(NanousdPrim prim, const char* name, const float v[2]);
    int  (*set_attribv3f)(NanousdPrim prim, const char* name, const float v[3]);
    int  (*set_attribv4f)(NanousdPrim prim, const char* name, const float v[4]);
    int  (*set_attribv2d)(NanousdPrim prim, const char* name, const double v[2]);
    int  (*set_attribv3d)(NanousdPrim prim, const char* name, const double v[3]);
    int  (*set_attribv4d)(NanousdPrim prim, const char* name, const double v[4]);
    int  (*set_attribv2i)(NanousdPrim prim, const char* name, const int v[2]);
    int  (*set_attribv3i)(NanousdPrim prim, const char* name, const int v[3]);
    int  (*set_attribv4i)(NanousdPrim prim, const char* name, const int v[4]);

    /* Matrix setter */
    int  (*set_attribm4d)(NanousdPrim prim, const char* name, const double v[16]);

    /* Array setters */
    int  (*set_attribarrayf)(NanousdPrim prim, const char* name, const float* data, int count);
    int  (*set_attribarrayd)(NanousdPrim prim, const char* name, const double* data, int count);
    int  (*set_attribarrayi)(NanousdPrim prim, const char* name, const int* data, int count);

    /* Time sample setters */
    int  (*set_samplef)(NanousdPrim prim, const char* name, double time, float value);
    int  (*set_sampled)(NanousdPrim prim, const char* name, double time, double value);
    int  (*set_samplev3f)(NanousdPrim prim, const char* name, double time, const float v[3]);
    int  (*set_samplev3d)(NanousdPrim prim, const char* name, double time, const double v[3]);

    /* Clear/block operations */
    int  (*clear_default)(NanousdPrim prim, const char* name);
    int  (*clear_samples)(NanousdPrim prim, const char* name);
    int  (*block_attrib)(NanousdPrim prim, const char* name);

    /* Create attribute (author new spec if not present) */
    int  (*create_attrib)(NanousdPrim prim, const char* name, const char* typeName);

    /* --- Bulk array access (GPU-friendly, appended for ABI compat) --- */

    /* Zero-copy read-only pointer into internal contiguous storage.
     * Returns pointer to first element and sets *count to number of elements.
     * Returns NULL if attribute is not a typed array of the requested type.
     * The pointer is valid until the prim handle is freed or the value is modified. */
    const float*  (*arraydataf)(NanousdPrim prim, const char* name, int* count);
    const double* (*arraydatad)(NanousdPrim prim, const char* name, int* count);
    const int*    (*arraydatai)(NanousdPrim prim, const char* name, int* count);

    /* Vec3 array reads — flat float/double buffer, 3 components per element.
     * Returns number of vec3s written, -1 on error. maxcount is in vec3s (not floats). */
    int  (*attribarrayv3f)(NanousdPrim prim, const char* name, float* out, int maxcount);
    int  (*attribarrayv3d)(NanousdPrim prim, const char* name, double* out, int maxcount);

    /* Vec3 array setters — flat buffer, 3 components per element.
     * count is in vec3s (not floats). */
    int  (*set_attribarrayv3f)(NanousdPrim prim, const char* name, const float* data, int count);
    int  (*set_attribarrayv3d)(NanousdPrim prim, const char* name, const double* data, int count);

    /* --- P0 physics prerequisites (appended for ABI compat) --- */

    /* Quaternion read/write — float[4] = {w, i, j, k} in USDA convention */
    int  (*attribqf)(NanousdPrim prim, const char* name, float out[4]);
    int  (*attribqd)(NanousdPrim prim, const char* name, double out[4]);
    int  (*set_attribqf)(NanousdPrim prim, const char* name, const float v[4]);
    int  (*set_attribqd)(NanousdPrim prim, const char* name, const double v[4]);

    /* Relationship write */
    int  (*set_reltargets)(NanousdPrim prim, const char* name,
                            const char** targets, int count);
    int  (*add_reltarget)(NanousdPrim prim, const char* name, const char* target);
    int  (*clear_reltargets)(NanousdPrim prim, const char* name);

    /* Stage creation */
    NanousdStage (*create)(void);

    /* Prim creation — returns prim handle, NULL on failure */
    NanousdPrim  (*define_prim)(NanousdStage stage, const char* path, const char* typeName);

    /* Schema application */
    int  (*apply_api)(NanousdPrim prim, const char* schemaName);

    /* --- P1 extensions (appended for ABI compat) --- */

    /* Matrix3d read/write — double[9], row-major */
    int  (*attribm3d)(NanousdPrim prim, const char* name, double out[9]);
    int  (*set_attribm3d)(NanousdPrim prim, const char* name, const double v[9]);

    /* String/token array read/write.
     * attribarrays_len: returns number of strings, -1 on error.
     * attribarrays: returns string at index (valid until prim freed or value modified).
     * set_attribarrays: sets array from array of C strings. */
    int         (*attribarrays_len)(NanousdPrim prim, const char* name);
    const char* (*attribarrays)(NanousdPrim prim, const char* name, int index);
    int         (*set_attribarrays)(NanousdPrim prim, const char* name,
                                    const char** strings, int count);

    /* Asset path read — returns asset string, NULL on error */
    const char* (*attribasset)(NanousdPrim prim, const char* name, int* ok);

    /* XformOp — compute composed local transform (geometry spec: xformable.md).
     * time: time code (NaN for default). out: 16-element row-major double[].
     * resetXformStack: if non-null, set to 1 if !resetXformStack! is present.
     * Returns 1 on success, 0 on failure. */
    int  (*get_local_transform)(NanousdPrim prim, double time, double out[16],
                                 int* resetXformStack);

    /* --- Binary write (appended for ABI compat) --- */

    /* Serialise the in-memory stage to a USDC binary crate file.
     * Returns 1 on success, 0 on failure. */
    int  (*write_usdc)(NanousdStage stage, const char* filepath);

    /* --- Array time sample reads (appended for ABI compat) --- */

    /* Vec2f time sample read — out must be float[2].
     * Returns 1 on success, 0 if time not found or wrong type. */
    int  (*samplev2f)(NanousdPrim prim, const char* name, double time, float out[2]);

    /* Array time sample reads — out is caller-allocated flat buffer of maxlen elements.
     * Returns number of elements written on success, 0 if time not found, -1 on error. */
    int  (*samplearrayf)(NanousdPrim prim, const char* name, double time, float* out, int maxlen);
    int  (*samplearrayd)(NanousdPrim prim, const char* name, double time, double* out, int maxlen);
    int  (*samplearrayi)(NanousdPrim prim, const char* name, double time, int* out, int maxlen);

    /* --- Stage root layer path (appended for ABI compat) --- */

    /* Returns the resolved file path of the root layer, or "" for in-memory stages.
     * The returned pointer is valid for the lifetime of the stage handle. */
    const char* (*stage_get_root_layer_path)(NanousdStage stage);

    /* --- Prim specifier write (appended for ABI compat) --- */

    /* Define a prim with an explicit specifier. Equivalent to define_prim + set_specifier. */
    NanousdPrim  (*define_prim_s)(NanousdStage stage, const char* path, const char* typeName,
                                 const char* specifier);

    /* Set the specifier on an existing prim ("def", "over", "class").
     * Returns 1 on success, 0 on failure. */
    int  (*set_specifier)(NanousdPrim prim, const char* specifier);

    /* --- USDA text write (appended for ABI compat) --- */

    /* Serialize the in-memory stage to a USDA text file.
     * Returns 1 on success, 0 on failure. */
    int  (*write_usda)(NanousdStage stage, const char* filepath);

    /* Serialize the in-memory stage to a malloc'd USDA string.
     * Returns NULL on failure. Caller frees with nanousd_free_string(). */
    const char* (*write_usda_string)(NanousdStage stage);

    /* --- Token/Asset attribute setters (appended for ABI compat) --- */

    /* Set a token-typed attribute value. Unlike set_attribs (which stores
     * TypeId::String), this stores TypeId::Token so the USDA writer emits
     * the correct serialisation. Returns 1 on success, 0 on failure. */
    int  (*set_attrib_token)(NanousdPrim prim, const char* name, const char* value);

    /* Set an asset-typed attribute value. Stores TypeId::Asset so the USDA
     * writer wraps the value with @...@. Returns 1 on success, 0 on failure. */
    int  (*set_attrib_asset)(NanousdPrim prim, const char* name, const char* value);

    /* Set a token[] array attribute. Like set_attribarrays but forces
     * TypeId::Token regardless of the attribute's declared typeName.
     * Returns 1 on success, 0 on failure. */
    int  (*set_attribarraytokens)(NanousdPrim prim, const char* name,
                                   const char** values, int count);

    /* Token array readers — parallel to attribarrays_len/attribarrays but
     * for token-typed arrays (stored as vector<Token>). */
    int         (*attribarraytokens_len)(NanousdPrim prim, const char* name);
    const char* (*attribarraytokens)(NanousdPrim prim, const char* name, int index);

    /* --- Composition arc write operations (appended for ABI compat) --- */

    /* Add a reference to an asset with optional prim path (prepends to
     * references listop). assetPath may be NULL for internal (same-layer)
     * references. primPath may be NULL to use the referenced asset's
     * defaultPrim. Triggers recomposition — all other NanousdPrim handles
     * become stale and must be re-acquired via nanousd_primpath().
     * Returns 1 on success, 0 on failure. */
    int  (*add_reference)(NanousdPrim prim, const char* assetPath,
                           const char* primPath);

    /* --- Relationship creation (appended for ABI compat) --- */

    /* Create a relationship on a prim. The relationship must be created
     * before targets can be set via set_reltargets / add_reltarget.
     * Returns 1 on success, 0 on failure. */
    int  (*create_rel)(NanousdPrim prim, const char* name);

    /* --- Schema registration (appended for ABI compat) --- */

    /* Register schemas from a JSON string (same format as built-in schemas).
     * Bumps the schema generation counter so open stages lazily rebuild
     * their prim definition caches. Returns 1 on success, 0 on failure. */
    int  (*register_schemas_json)(const char* json);

    /* --- Prim metadata (appended for ABI compat) --- */

    /* Generic prim metadata getters — walk the opinion stack for scalar fields.
     * Returns "" / 0.0 and sets *ok=0 if the field is not authored. */
    const char* (*prim_metadatas)(NanousdPrim prim, const char* key, int* ok);
    double      (*prim_metadatad)(NanousdPrim prim, const char* key, int* ok);

    /* Generic prim metadata setters — write to the root layer.
     * Returns 1 on success, 0 on failure. */
    int  (*set_prim_metadatas)(NanousdPrim prim, const char* key, const char* value);
    int  (*set_prim_metadatad)(NanousdPrim prim, const char* key, double value);
    int  (*set_prim_metadata_token)(NanousdPrim prim, const char* key, const char* value);

    /* --- Token scalar reader (appended for ABI compat) --- */
    const char* (*attrib_token)(NanousdPrim prim, const char* name, int* ok);

    /* --- Attribute metadata & connections (appended for ABI compat) --- */
    const char* (*attrib_interpolation)(NanousdPrim prim, const char* name);
    int         (*attrib_authored)(NanousdPrim prim, const char* name);
    int         (*hasconnections)(NanousdPrim prim, const char* name);
    int         (*nconnections)(NanousdPrim prim, const char* name);
    const char* (*connection)(NanousdPrim prim, const char* name, int index);

    /* Parent prim traversal */
    NanousdPrim   (*parent)(NanousdPrim prim);

    /* --- Int64 array read (appended for ABI compat) --- */

    /* Read int64[] array — out is caller-allocated int64_t buffer of maxlen elements.
     * Returns number of elements written, -1 on error. */
    int  (*attribarrayi64)(NanousdPrim prim, const char* name, int64_t* out, int maxlen);

    /* Token-or-string array reader — returns string regardless of whether the
     * stored type is Token or String. Equivalent to attribarrays with token fallback. */
    const char* (*attribarrays_elem)(NanousdPrim prim, const char* name, int index);

    /* --- Time sample setters: extended types (appended for ABI compat) --- */

    int  (*set_samplev4f)(NanousdPrim prim, const char* name, double time, const float v[4]);
    int  (*set_sampleqf)(NanousdPrim prim, const char* name, double time, const float v[4]);
    int  (*set_sample_token)(NanousdPrim prim, const char* name, double time, const char* value);
    int  (*set_samplearrayf)(NanousdPrim prim, const char* name, double time, const float* data, int count);
    int  (*set_samplearrayi)(NanousdPrim prim, const char* name, double time, const int* data, int count);
    int  (*set_samplearrayv3f)(NanousdPrim prim, const char* name, double time, const float* data, int count);

    /* --- Double-precision time sample setters --- */
    int  (*set_samplev2d)(NanousdPrim prim, const char* name, double time, const double v[2]);
    int  (*set_samplev4d)(NanousdPrim prim, const char* name, double time, const double v[4]);
    int  (*set_samplem4d)(NanousdPrim prim, const char* name, double time, const double v[16]);
    int  (*set_samplearrayd)(NanousdPrim prim, const char* name, double time, const double* data, int count);
    int  (*set_samplearrayv3d)(NanousdPrim prim, const char* name, double time, const double* data, int count);

    /* --- Instancing (appended for ABI compat) --- */
    int       (*isinstance)(NanousdPrim prim);
    int       (*isprototype)(NanousdPrim prim);
    int       (*isinprototype)(NanousdPrim prim);
    NanousdPrim (*prototype)(NanousdPrim prim);
    int       (*ninstances)(NanousdPrim prim);
    NanousdPrim (*instance)(NanousdPrim prim, int index);

    /* --- Diagnostics (appended for ABI compat) --- */
    NanousdDiagnostic* (*diagnostics)(NanousdStage stage, int* count);
    void             (*free_diagnostics)(NanousdDiagnostic* diags, int count);
    const char*      (*diagnostics_json)(NanousdStage stage);

    /* --- Variant API (spec §11.2, appended for ABI compat) ---
     * nvariantsets / variantsetname: list variant set names declared on the
     *   prim (composed across the opinion stack).
     * hasvariantset: true if the named variant set is declared.
     * nvariants / variantname: list the variants declared in a given set,
     *   unioned across layers that author the set.
     * variantselection: the selected variant for a set (strongest-wins
     *   over the opinion stack). Returns "" if no selection is authored.
     * setvariantselection: writes a selection into the layer at layerIndex.
     *   Pass variantName="" to clear. Rebuilds the stage internally and
     *   refreshes the passed-in prim handle. Returns 1 on success, 0 on
     *   failure. */
    int         (*nvariantsets)(NanousdPrim prim);
    const char* (*variantsetname)(NanousdPrim prim, int index);
    int         (*hasvariantset)(NanousdPrim prim, const char* setName);
    int         (*nvariants)(NanousdPrim prim, const char* setName);
    const char* (*variantname)(NanousdPrim prim, const char* setName, int index);
    const char* (*variantselection)(NanousdPrim prim, const char* setName);
    int         (*setvariantselection)(NanousdPrim prim, const char* setName,
                                        const char* variantName, int layerIndex);

    /* --- Spec-correct typed stage metadata (appended for ABI compat) ---
     * Adding setters in the middle of the struct breaks every consumer
     * that linked against the old layout. Append-only to preserve ABI. */
    int         (*set_stage_metadata_token)(NanousdStage stage, const char* key, const char* value);

    /* --- Composed-layer enumeration (appended for ABI compat) --- */
    int          (*stage_n_layers)(NanousdStage stage);
    const char*  (*stage_layer_path)(NanousdStage stage, int index);

    /* --- Masked stage open (appended for ABI compat) --- */
    NanousdStage (*open_masked)(const char* filepath,
                                const char* const* maskPaths,
                                int maskPathCount);

    /* --- Color-space resolution (appended for ABI compat) --- */
    const char* (*attrib_colorspace)(NanousdPrim prim, const char* name, int* ok);
    const char* (*attrib_resolved_colorspace)(NanousdPrim prim, const char* name);
    int         (*set_attrib_colorspace)(NanousdPrim prim, const char* name,
                                         const char* colorSpace);
    int         (*clear_attrib_colorspace)(NanousdPrim prim, const char* name);
    const char* (*prim_resolved_colorspace)(NanousdPrim prim);

    /* --- USDZ package write (appended for ABI compat) --- */
    int         (*write_usdz)(NanousdStage stage, const char* filepath);

    /* --- CollectionAPI evaluation (appended for ABI compat) --- */
    int         (*collection_nmembers)(NanousdPrim prim, const char* instanceName);
    const char* (*collection_member)(NanousdPrim prim, const char* instanceName,
                                     int index);
    int         (*collection_contains)(NanousdPrim prim, const char* instanceName,
                                       const char* path);

    /* --- Property enumeration (appended for ABI compat) --- */
    int         (*nproperties)(NanousdPrim prim);
    const char* (*propertyname)(NanousdPrim prim, int index);
    int         (*property_is_attribute)(NanousdPrim prim, const char* name);
    int         (*property_is_relationship)(NanousdPrim prim, const char* name);

    /* --- Relationship authored-state query (appended for ABI compat) --- */
    int         (*rel_authored)(NanousdPrim prim, const char* name);

    /* --- Per-layer spec / opinion queries (usdview panel parity).
     * These let consumers walk the layer-stack view (which layers
     * authored a given prim/property), the composition view (multi-
     * level arc traversal), and the property layer-stack (per-layer
     * value opinions + sample counts) without re-opening files. */
    int          (*layer_has_prim_spec)(NanousdStage stage, int layerIdx,
                                        const char* primPath);
    int          (*layer_has_attr_opinion)(NanousdStage stage, int layerIdx,
                                           const char* primPath, const char* attrName);
    int          (*layer_attr_nsamples)(NanousdStage stage, int layerIdx,
                                        const char* primPath, const char* attrName);
    NanousdListOp (*layer_prim_listop)(NanousdStage stage, int layerIdx,
                                       const char* primPath, const char* field);

    /* --- Sublayer enumeration & per-layer time offset (Layer-Stack
     * tab nests sublayers under their parents; the offset/scale shows
     * up as a column when non-identity). */
    int          (*layer_n_sublayers)(NanousdStage stage, int layerIdx);
    const char*  (*layer_sublayer_path)(NanousdStage stage, int layerIdx, int subIdx);
    int          (*layer_offset)(NanousdStage stage, int layerIdx,
                                 double* offset, double* scale);

    /* --- Internal recomposition trigger.
     * Retained as an appended backend slot for ABI compatibility. Public
     * callers do not invoke recomposition directly; composition-changing
     * setters call this internally and refresh the passed-in handle. */
    int          (*recompose)(NanousdStage stage);

    /* --- Composition-arc authoring (appended for ABI compat).
     * payload, inheritPaths, specializes counterparts to add_reference.
     * remove_listop_item generalises removal across all five composition
     * arc fields (references, payload, inheritPaths, specializes,
     * apiSchemas). All trigger recomposition. */
    int          (*add_payload)(NanousdPrim prim, const char* assetPath,
                                const char* primPath);
    int          (*add_inherit)(NanousdPrim prim, const char* primPath);
    int          (*add_specialize)(NanousdPrim prim, const char* primPath);
    int          (*remove_listop_item)(NanousdPrim prim, const char* field,
                                       int listOpKind, int index);

    /* --- Prim-state writers (appended for ABI compat). */
    int          (*set_active)(NanousdPrim prim, int active);
    int          (*set_instanceable)(NanousdPrim prim, int instanceable);
    int          (*remove_api)(NanousdPrim prim, const char* schemaName);
    int          (*remove_prim)(NanousdPrim prim);

    /* --- Variant set authoring (appended for ABI compat). */
    int          (*create_variantset)(NanousdPrim prim, const char* setName);
    int          (*create_variant)(NanousdPrim prim, const char* setName,
                                   const char* variantName);

    /* --- Asset resolution and resource reads (appended for ABI compat). */
    int          (*resolve_asset_path)(const char* anchorLayerPath,
                                       const char* assetPath,
                                       char* out,
                                       size_t out_size);
    int          (*stage_resolve_asset_path)(NanousdStage stage,
                                             const char* assetPath,
                                             char* out,
                                             size_t out_size);
    int          (*read_asset_bytes)(const char* resolvedLocation,
                                     unsigned char** out_data,
                                     size_t* out_size);
    void         (*free_bytes)(void* data);

    /* --- Relationship metadata (appended for ABI compat).
     * Relationship fields such as material:binding's bindMaterialAs are
     * property metadata, not prim metadata. Renderers need this to match
     * USD/Hydra material-binding strength rules. */
    const char*  (*rel_metadatas)(NanousdPrim prim, const char* relName,
                                  const char* key, int* ok);

    /* --- Authored attribute enumeration (appended for ABI compat).
     * Counterparts to nattribs/attribname that skip schema-fallback names
     * and report only authored attributes. Appended (not placed next to
     * nattribs/attribname) so adding them does not shift any existing
     * vtable slot. */
    int          (*nauthored_attribs)(NanousdPrim prim);
    const char* (*authored_attribname)(NanousdPrim prim, int index);

    /* --- Flat traversal, instancing, and composition-source diagnostics
     * (appended for ABI compat). These expose OpenUSD-like semantic facts
     * without exposing nanousd's private PrimIndex / OpinionEntry layout. */
    int          (*traverse_flat)(NanousdStage stage,
                                  struct NanousdFlatPrim_s* out,
                                  int max_count);
    int          (*stage_nprototypes)(NanousdStage stage);
    NanousdPrim  (*stage_prototype)(NanousdStage stage, int index);
    int          (*isinstanceproxy)(NanousdPrim prim);
    NanousdPrim  (*priminprototype)(NanousdPrim prim);
    int          (*instance_key)(NanousdPrim prim, char* out, size_t out_size);
    int          (*ncomposition_arcs)(NanousdPrim prim);
    int          (*composition_arc)(NanousdPrim prim, int index,
                                    struct NanousdCompositionArc_s* out);

} NanousdBackend_v1;

/* The single entry point every backend must export */
#ifdef _WIN32
#   ifdef NANOUSD_BACKEND_BUILDING
#       define NANOUSD_BACKEND_API __declspec(dllexport)
#   else
#       define NANOUSD_BACKEND_API __declspec(dllimport)
#   endif
#else
#   define NANOUSD_BACKEND_API __attribute__((visibility("default")))
#endif

typedef NanousdBackend_v1* (*NanousdCreateBackendFn)(void);

NANOUSD_BACKEND_API NanousdBackend_v1* nanousd_create_backend_v1(void);

#ifdef __cplusplus
}
#endif

#endif /* NANOUSD_BACKEND_H */
