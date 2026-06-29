// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NUSD_MATERIAL_H
#define NUSD_MATERIAL_H

/*
 * material.h — Material system for the nanousd-opengl-renderer.
 *
 * Extracts material data from USD scenes and loads textures via stb_image.
 * No MaterialX, no shaderc, no SPIR-V — uses built-in PBR shaders.
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

/* Per-material PBR parameters (GPU-uploadable, std140 friendly) */
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
    int   v_flip;             /* 1 → sample at (u, 1-v); set for MDL materials.
                                 NVIDIA's MDL pipeline explicitly does
                                 `1.0 - texture_coordinate(0).y` because
                                 Omniverse stores textures top-V (DirectX),
                                 while UsdPreviewSurface assumes bottom-V
                                 (OpenGL). We can't tell which convention an
                                 asset uses from the data alone, so we
                                 conservatively flip only when the Shader
                                 prim references an MDL source asset. */
    float opacity_threshold;  /* UPS opacityThreshold. > 0 = alpha-cutout
                                 (frag discards when sample < threshold);
                                 0 = alpha-blend (no discard). Mirrors
                                 vulkan's MaterialParams field. */
    float _pad_a, _pad_b, _pad_c; /* std140 align next vec4 */
    float mdl_uv_transform[4];    /* xy texture scale + zw bias; zero means identity */
    float transmission_color[4]; /* MaterialX Standard Surface transmission tint */
    float transmission_weight;   /* MaterialX Standard Surface transmission weight */
    float transmission_ior;      /* 0 = use ior */
    float _pad_d, _pad_e;        /* std140 vec4 */
    int   use_specular_workflow; /* MDL/UsdPreviewSurface specular workflow */
    float _pad_f, _pad_g, _pad_h;
    float specular_color[4];     /* rgb F0 when use_specular_workflow != 0 */
    float roughness_tex_scale;    /* sampled roughness = tex.g * scale + bias */
    float roughness_tex_bias;     /* Isaac MDL RoughnessMin/Max remap */
    float _pad_i, _pad_j;         /* std140 align trailing block */
} MaterialParams;

/* Texture image data (CPU side, before GPU upload) */
typedef struct {
    unsigned char* pixels;  /* RGBA8 */
    int width;
    int height;
    int udim_cols;          /* UDIM grid columns (0 = not UDIM) */
    int udim_rows;          /* UDIM grid rows (0 = not UDIM) */
    char path[1024];        /* resolved file path */
    int is_srgb;            /* 1 = sRGB-encoded color (diffuse, emissive);
                               0 = linear data (normal, roughness, metallic, AO).
                               Determined by vote across material slots after
                               extraction: linear wins on tie or if any data-slot
                               reference exists. Sampling sRGB data as linear
                               renders too dark; sampling linear data as sRGB
                               corrupts normals + PBR. Mirrors vulkan. */
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

/* Scene-level material collection */
typedef struct {
    SceneMaterial*  materials;
    int             nmaterials;

    MaterialTexture* textures;
    int              ntextures;

    /* Unique generated shaders (many materials may share one shader) */
    MaterialShader*  unique_shaders;
    int              nunique_shaders;
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
/* Set max texture resolution (default 512). Call before materials_load(). */
void materials_set_max_tex_size(int size);

MaterialCollection* materials_load(void* stage, const char* scene_dir);

/*
 * Look up the material index for a mesh prim.
 * Returns -1 if the mesh has no material binding.
 */
/* Precompute every prim_path → mat_idx binding in the stage and assign
 * meshes[k].material_index in one pass. Replaces the O(meshes × ancestors
 * × nanousd_calls) `materials_find_binding` loop with two stage walks +
 * a sorted-array bsearch. Returns the number of meshes that resolved to
 * a material (rest get -1). For the warehouse, this is ~150× faster.
 * `meshes` is a SceneMesh*; void* used here to avoid pulling scene.h. */
int  materials_assign_bindings(MaterialCollection* mc, void* stage,
                               void* meshes, int nmeshes);

int materials_find_binding(MaterialCollection* mc, void* stage,
                           void* mesh_prim);

/* Free all material data including generated shaders and texture pixels. */
void materials_free(MaterialCollection* mc);

/* Initialize the MaterialX library (load standard libraries).
 * Call once at startup. Returns 1 on success. */
int materialx_init(void);

/* Shutdown MaterialX. */
void materialx_shutdown(void);

/* Texture loader shared between material.c (UsdShade path) and
 * material_mtlx.cpp (.mtlx path). Adds the resolved texture to mc and
 * returns its index, or -1 on failure. stage is the optional NanousdStage
 * used to look up textures inside USDZ archives. */
int find_or_add_texture(MaterialCollection* mc, const char* scene_dir,
                        const char* tex_path, void* stage);

/* Scan scene_dir for *.mtlx files, parse each, and append a
 * SceneMaterial per <surfacematerial> we find. Returns the number of
 * materials added. Provided by material_mtlx.cpp; no-op if MaterialX
 * isn't initialized (returns 0). */
int materials_scan_mtlx_directory(MaterialCollection* mc,
                                  const char* scene_dir,
                                  void* stage);

#ifdef __cplusplus
}
#endif

#endif /* NUSD_MATERIAL_H */
