// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NUSD_MATERIAL_H
#define NUSD_MATERIAL_H

/*
 * material.h — Material system for the nanousd-vulkan-renderer.
 *
 * Extracts material data from USD scenes and generates SPIR-V shaders
 * via MaterialX GLSL code generation + shaderc runtime compilation.
 *
 * The C structs here are populated by material.cpp (C++ / MaterialX).
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_MATERIAL_TEXTURES 8

/* Texture slot indices matching UsdPreviewSurface inputs */
enum {
    TEX_DIFFUSE_COLOR  = 0,
    TEX_NORMAL         = 1,
    TEX_ROUGHNESS      = 2,
    TEX_METALLIC       = 3,
    TEX_EMISSIVE_COLOR = 4,
    TEX_OCCLUSION      = 5,
    TEX_OPACITY        = 6,
    TEX_DISPLACEMENT   = 7,
};

/* Per-material PBR parameters (GPU-uploadable, std140 friendly).
 *
 * Phase 7c added the Standard Surface trailing block (subsurface_*,
 * transmission_*, two extra texture slots). All zero = no contribution,
 * so existing UsdPreviewSurface-only assets render bit-for-bit identically. */
typedef struct {
    float base_color[4];      /* rgb + alpha */
    float emissive_color[4];  /* rgb + intensity */
    float metallic;
    float roughness;
    float opacity;
    float ior;
    float occlusion;
    float clearcoat;
    float clearcoat_roughness;
    float normal_scale;
    int   tex_indices[MAX_MATERIAL_TEXTURES]; /* -1 = no texture */
    int   use_vertex_color;   /* fallback to vertex color if no material textures */
    float udim_scale_u;       /* UDIM atlas UV divisor (cols), 1.0 = no UDIM */
    float udim_scale_v;       /* UDIM atlas UV divisor (rows), 1.0 = no UDIM */
    /* UsdPreviewSurface opacityThreshold. If > 0, alpha-cutout: when the
     * sampled alpha < opacity_threshold, the rchit shader recurses through
     * the surface as if it weren't there (alpha test, not blend). 0
     * disables the cutout entirely. */
    float opacity_threshold;
    /* === Phase 7c — MaterialX Standard Surface extensions ============== */
    float subsurface_color[4];   /* rgb (linear), w unused */
    float subsurface_radius[4];  /* rgb scattering radius (scene units), w unused */
    float transmission_color[4]; /* rgb tint (Beer-Lambert), w unused */
    float subsurface_weight;     /* mix into diffuse lobe ([0,1]) */
    float subsurface_scale;      /* mm-unit scaler applied to radius */
    float transmission_weight;   /* mix into glass lobe ([0,1]) */
    float transmission_ior;      /* 0 = use mp.ior */
    int   tex_subsurface_weight; /* texture slot for subsurface (-1 = none) */
    int   tex_transmission_weight; /* texture slot for transmission (-1 = none) */
    /* sss_color_authored = 1 iff the loader actually wrote a constant value
     * to subsurface_color (non-zero only when MaterialX `subsurface_color`
     * resolves as a constant — chess King wires it through a nodegraph the
     * side-loader can't sample, so it stays 0 and the shader falls back to
     * baseColor). Replaces the fragile `sssColor < 0.999` heuristic. */
    int   sss_color_authored;
    /* UsdPreviewSurface specular workflow.
     *   useSpecularWorkflow=0 (default): metallic workflow.
     *     F0 = mix(0.04, baseColor, metallic).
     *   useSpecularWorkflow=1: artist-authored specular.
     *     F0 = specularColor (RGB), metallic input ignored.
     *
     * After sss_color_authored at offset 184, use_specular_workflow at 188
     * leaves the next offset at 192 = 16-aligned, so specular_color (vec4)
     * fits without extra padding under std430. */
    int   use_specular_workflow;
    float specular_color[4];      /* rgb (linear), w unused */
    /* UsdUVTexture inputs:scale and inputs:bias for the normal-map slot
     * (TEX_NORMAL). Defaults to (2,2,2,1) / (-1,-1,-1,0) which reproduces
     * the implicit `nm = nm * 2 - 1` mapping used before this was wired —
     * scenes that don't author scale/bias on a UsdUVTexture keep
     * rendering bit-identically. Scenes authoring different values (e.g.
     * raw [-1,1]-encoded normal maps with scale=1/bias=0) now render
     * spec-correctly. */
    float normal_tex_scale[4];
    float normal_tex_bias[4];
    float mdl_uv_transform[4];     /* xy scale + zw bias for generated MDL UVs */
    int   v_flip;                 /* 1 = sample texture V as 1-v for MDL */
    float roughness_tex_scale;     /* sampled roughness = tex.g * scale + bias */
    float roughness_tex_bias;      /* generated MDL RoughnessMin/Max remap */
    int   _pad_v_flip;
} MaterialParams;

/* Texture image data (CPU side, before GPU upload) */
typedef struct {
    unsigned char* pixels;  /* RGBA8 */
    int width;
    int height;
    int udim_cols;          /* UDIM grid columns (0 = not UDIM) */
    int udim_rows;          /* UDIM grid rows (0 = not UDIM) */
    char path[512];         /* resolved file path */
    int is_srgb;            /* 1 = sRGB-encoded color (diffuse, emissive);
                               0 = linear data (normal, roughness, metallic, AO).
                               If a texture is referenced by mixed slots across
                               materials, prefer linear (0) — sampling sRGB data
                               as linear renders too dark; sampling linear data
                               as sRGB renders WRONG (corrupts normals + PBR). */
} MaterialTexture;

/* Generated SPIR-V shader */
typedef struct {
    uint32_t* vert_spv;
    uint32_t  vert_size;
    uint32_t* frag_spv;
    uint32_t  frag_size;
} MaterialShader;

/* Per-material data */
typedef struct {
    MaterialParams  params;
    MaterialShader  shader;
    char            name[256];
    char            prim_path[512];
    char            ptex_color_path[512]; /* resolved authored Ptex surfaceMap, if any */
    float           ptex_average_color[3];
    int             has_ptex_average_color;
    int             shader_index;   /* index into unique shaders array */
} SceneMaterial;

typedef struct {
    char prim_name[256];
    int  material_index;
} MaterialBindingHint;

/* Scene-level material collection */
typedef struct {
    SceneMaterial*  materials;
    int             nmaterials;

    MaterialTexture* textures;
    int              ntextures;

    /* Unique generated shaders (many materials may share one shader) */
    MaterialShader*  unique_shaders;
    int              nunique_shaders;

    /* Fast path for direct streamed instance meshes. Built from simple ASCII
     * USDA material-binding opinions keyed by bound prim leaf name. */
    MaterialBindingHint* binding_hints;
    int                  nbinding_hints;

    /* Material-object dedup: maps each raw material index to a CANONICAL index
     * (the first material with byte-identical MaterialParams). NULL = no dedup.
     * Omniverse authors one MaterialInstanceDynamic per prim, so thousands of
     * byte-identical materials get distinct indices; materials_find_binding
     * returns the canonical so same-geometry/same-material instances share an
     * index — which lets the geometry compaction (B6 guard) actually fuse them. */
    int*                 dedup_remap;
} MaterialCollection;

/*
 * Load all materials from a USD stage.
 * Generates GLSL via MaterialX and compiles to SPIR-V.
 * Returns NULL on failure (no materials found is not a failure — returns
 * a valid collection with nmaterials = 0).
 *
 * stage is an NanousdStage handle (from nanousdapi.h).
 * scene_dir is the directory containing the USD file (for resolving
 * relative texture paths).
 */
MaterialCollection* materials_load(void* stage, const char* scene_dir);
MaterialCollection* materials_load_filtered(void* stage, const char* scene_dir,
                                            const unsigned char* wanted_prims,
                                            int nprims_in_bitmap);

/*
 * Look up the material index for a mesh prim.
 * Returns -1 if the mesh has no material binding.
 */
int materials_find_binding(MaterialCollection* mc, void* stage,
                           void* mesh_prim);

int materials_find_binding_by_path(MaterialCollection* mc,
                                   const char* mesh_path);

/* Free all material data including generated shaders and texture pixels. */
void materials_free(MaterialCollection* mc);

/* Initialize the MaterialX library (load standard libraries).
 * Call once at startup. Returns 1 on success. */
int materialx_init(void);

/* Shutdown MaterialX. */
void materialx_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* NUSD_MATERIAL_H */
