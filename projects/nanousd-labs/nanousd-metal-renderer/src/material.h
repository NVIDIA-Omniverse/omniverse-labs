// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NUSD_MATERIAL_H
#define NUSD_MATERIAL_H

/*
 * material.h — Material system for the nanousd-metal-renderer.
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
#ifndef MAX_MTLX_PROC_NODES
#define MAX_MTLX_PROC_NODES 64
#endif

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

enum {
    MTLX_PROC_OP_NONE      = 0,
    MTLX_PROC_OP_CONST     = 1,
    MTLX_PROC_OP_POSITION  = 2,
    MTLX_PROC_OP_TEXCOORD  = 3,
    MTLX_PROC_OP_ADD       = 4,
    MTLX_PROC_OP_SUBTRACT  = 5,
    MTLX_PROC_OP_MULTIPLY  = 6,
    MTLX_PROC_OP_DIVIDE    = 7,
    MTLX_PROC_OP_DOT       = 8,
    MTLX_PROC_OP_SIN       = 9,
    MTLX_PROC_OP_POWER     = 10,
    MTLX_PROC_OP_MIX       = 11,
    MTLX_PROC_OP_CLAMP     = 12,
    MTLX_PROC_OP_ABS       = 13,
    MTLX_PROC_OP_MIN       = 14,
    MTLX_PROC_OP_MAX       = 15,
    MTLX_PROC_OP_FRACTAL3D = 16,
    MTLX_PROC_OP_CONVERT   = 17,
    MTLX_PROC_OP_COMBINE3  = 18,
    MTLX_PROC_OP_EXTRACT   = 19,
    MTLX_PROC_OP_INVERT    = 20,
    MTLX_PROC_OP_IFGREATER = 21,
    MTLX_PROC_OP_RAMPTB    = 22,
    MTLX_PROC_OP_NOISE3D   = 23,
    MTLX_PROC_OP_CELLNOISE = 24,
    MTLX_PROC_OP_TEXTURE   = 25,
    MTLX_PROC_OP_RGBTOHSV  = 26,
    MTLX_PROC_OP_HSVTORGB  = 27,
};

enum {
    MTLX_PROC_TYPE_FLOAT = 1,
    MTLX_PROC_TYPE_VEC3  = 2,
};

typedef struct {
    int   op;
    int   type;
    int   in0;
    int   in1;
    float value[4];
    int   in2;
    int   in3;
    int   _pad0;
    int   _pad1;
} MtlxProcNode;

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
     * disables the cutout entirely. Reuses the existing _pad slot — no
     * struct-size change. Port from Vulkan a619161. */
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
    int   sss_color_authored;    /* 1 = author set subsurface_color as a constant;
                                  * 0 = unauthored / nodegraph-driven → shader
                                  * falls back to baseColor instead of using
                                  * the (1,1,1) default. Replaces the fragile
                                  * (>0.99) sentinel detection (port from
                                  * Vulkan 8056f2e). */
    /* UsdPreviewSurface specular workflow (port from Vulkan 301d753):
     *   useSpecularWorkflow=0 (default): metallic workflow.
     *     F0 = mix(0.04, baseColor, metallic).
     *   useSpecularWorkflow=1: artist-authored specular.
     *     F0 = specularColor (RGB), metallic input ignored.
     * sss_color_authored at offset 184, use_specular_workflow at 188 leaves
     * the next offset at 192 = 16-aligned, so specular_color (4×float)
     * lands on a 16-byte boundary with no extra padding. */
    int   use_specular_workflow;
    float specular_color[4];     /* rgb (linear), w unused */
    /* UsdUVTexture inputs:scale and inputs:bias for the normal-map slot
     * (TEX_NORMAL). Defaults to (2,2,2,1) / (-1,-1,-1,0), reproducing
     * the implicit `nm = nm * 2 - 1` mapping for unauthored scenes while
     * allowing raw [-1,1] normal maps or custom remaps to render correctly. */
    float normal_tex_scale[4];
    float normal_tex_bias[4];
    float uv_transform[4];          /* xy texture scale, zw texture offset */
    float roughness_tex_transform[4]; /* x scale, y bias for sampled roughness */
    int   v_flip;                 /* 1 = sample texture V as 1-v for MDL */
    int   _pad_v_flip[3];
    /* MaterialX Standard Surface lobe controls beyond the chess-set subset.
     * standard_surface_lobes gates these fields so legacy UsdPreviewSurface
     * materials keep their old interpretation of specular_color. */
    float base_weight;
    float specular_weight;
    float sheen_weight;
    float sheen_roughness;
    float sheen_color[4];
    float thin_film_thickness;  /* nanometers; 0 disables */
    float thin_film_ior;
    float specular_anisotropy;
    int   standard_surface_lobes;
    /* Legacy narrow MaterialX procedural parameters, kept for ABI continuity
     * with the first marble implementation. New materials use the generic
     * procedural graph table below. */
    int   procedural_kind;
    int   procedural_base_color;
    int   procedural_subsurface_color;
    int   procedural_octaves;
    float procedural_color1[4];
    float procedural_color2[4];
    float procedural_params[4]; /* scale1, scale2, power, noise_amp */
    int   procedural_node_count;
    int   procedural_base_color_output;
    int   procedural_subsurface_color_output;
    int   procedural_roughness_output;
    int   procedural_normal_output;
    int   procedural_graph_flags;
    int   procedural_graph_pad0;
    int   procedural_graph_pad1;
    MtlxProcNode procedural_nodes[MAX_MTLX_PROC_NODES];
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

/* Returns 1 when this build links the real MaterialX-backed loader,
 * 0 when it links the no-op stub. */
int materials_backend_available(void);

#ifdef __cplusplus
}
#endif

#endif /* NUSD_MATERIAL_H */
