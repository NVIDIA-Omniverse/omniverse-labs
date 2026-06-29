// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "material.h"
#include <nanousd/nanousdapi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); g_fail++; } \
    else { fprintf(stderr, "ok:   %s\n", msg); } \
} while (0)

static int nearly(float a, float b)
{
    return fabsf(a - b) < 0.0005f;
}

static void dirname_of(const char* path, char* out, size_t out_size)
{
    const char* slash = strrchr(path, '/');
    if (!slash) {
        snprintf(out, out_size, ".");
        return;
    }
    size_t n = (size_t)(slash - path);
    if (n >= out_size) n = out_size - 1;
    memcpy(out, path, n);
    out[n] = '\0';
}

static SceneMaterial* find_material(MaterialCollection* mc, const char* name)
{
    if (!mc || !name) return NULL;
    for (int i = 0; i < mc->nmaterials; i++) {
        if (strcmp(mc->materials[i].name, name) == 0)
            return &mc->materials[i];
    }
    return NULL;
}

static int texture_path_contains(MaterialCollection* mc, int tex_idx,
                                 const char* needle)
{
    if (!mc || !needle || tex_idx < 0 || tex_idx >= mc->ntextures)
        return 0;
    return strstr(mc->textures[tex_idx].path, needle) != NULL;
}

static void check_mdl_material(MaterialCollection* mc, const char* name,
                               const float base[3], const float spec[3],
                               float roughness, float metallic,
                               float opacity, float normal_scale)
{
    SceneMaterial* mat = find_material(mc, name);
    char msg[256];
    snprintf(msg, sizeof(msg), "%s material exists", name);
    CHECK(mat != NULL, msg);
    if (!mat) return;

    MaterialParams* p = &mat->params;
    snprintf(msg, sizeof(msg), "%s base color", name);
    CHECK(nearly(p->base_color[0], base[0]) &&
          nearly(p->base_color[1], base[1]) &&
          nearly(p->base_color[2], base[2]), msg);

    snprintf(msg, sizeof(msg), "%s roughness/metallic/opacity/normal", name);
    CHECK(nearly(p->roughness, roughness) &&
          nearly(p->metallic, metallic) &&
          nearly(p->opacity, opacity) &&
          nearly(p->normal_scale, normal_scale), msg);

    snprintf(msg, sizeof(msg), "%s specular workflow", name);
    CHECK(p->use_specular_workflow == 1 &&
          nearly(p->specular_color[0], spec[0]) &&
          nearly(p->specular_color[1], spec[1]) &&
          nearly(p->specular_color[2], spec[2]), msg);

    snprintf(msg, sizeof(msg), "%s MDL v-flip", name);
    CHECK(p->v_flip == 1, msg);
}

static void check_mask_selection_material(MaterialCollection* mc)
{
    SceneMaterial* mat = find_material(mc, "MaskSelectionMDL");
    CHECK(mat != NULL, "MaskSelectionMDL material exists");
    if (!mat) return;

    MaterialParams* p = &mat->params;
    CHECK(p->tex_indices[TEX_DIFFUSE_COLOR] >= 0,
          "MaskSelectionMDL has diffuse texture");
    CHECK(p->tex_indices[TEX_ROUGHNESS] >= 0,
          "MaskSelectionMDL has ORM roughness texture");
    CHECK(p->tex_indices[TEX_OCCLUSION] == p->tex_indices[TEX_ROUGHNESS],
          "MaskSelectionMDL shares ORM as occlusion");
    CHECK(p->tex_indices[TEX_METALLIC] == p->tex_indices[TEX_ROUGHNESS],
          "MaskSelectionMDL does not treat MaskSelection _M as metallic");
    CHECK(p->tex_indices[TEX_DIFFUSE_COLOR] >= 0 &&
          p->tex_indices[TEX_DIFFUSE_COLOR] < mc->ntextures &&
          strstr(mc->textures[p->tex_indices[TEX_DIFFUSE_COLOR]].path,
                 "mdl_baked_masked_albedo:") != NULL,
          "MaskSelectionMDL bakes ColorAlbedo/AlbedoTexture/MaskSelection");
    CHECK(p->tex_indices[TEX_EMISSIVE_COLOR] < 0,
          "MaskSelectionMDL does not inherit emissive source from sibling MDL export");
    CHECK(nearly(p->base_color[0], 1.0f) &&
          nearly(p->base_color[1], 1.0f) &&
          nearly(p->base_color[2], 1.0f) &&
          p->v_flip == 0,
          "MaskSelectionMDL baked albedo resets tint and V flip");
}

static void check_body_mask_material(MaterialCollection* mc)
{
    SceneMaterial* mat = find_material(mc, "BodyMaskMDL");
    CHECK(mat != NULL, "BodyMaskMDL material exists");
    if (!mat) return;

    MaterialParams* p = &mat->params;
    CHECK(p->tex_indices[TEX_DIFFUSE_COLOR] >= 0 &&
          p->tex_indices[TEX_DIFFUSE_COLOR] < mc->ntextures &&
          strstr(mc->textures[p->tex_indices[TEX_DIFFUSE_COLOR]].path,
                 "mdl_baked_body_masked_albedo:") != NULL,
          "BodyMaskMDL bakes Body/Handle/Cap/AlbedoTexture/MaskSelection");
    CHECK(nearly(p->base_color[0], 1.0f) &&
          nearly(p->base_color[1], 1.0f) &&
          nearly(p->base_color[2], 1.0f) &&
          p->v_flip == 0,
          "BodyMaskMDL baked albedo resets tint and V flip");
    CHECK(p->tex_indices[TEX_EMISSIVE_COLOR] < 0,
          "BodyMaskMDL does not inherit emissive source from sibling MDL export");
}

static void check_emissive_product_material(MaterialCollection* mc)
{
    SceneMaterial* mat = find_material(mc, "EmissiveProductMDL");
    CHECK(mat != NULL, "EmissiveProductMDL material exists");
    if (!mat) return;

    MaterialParams* p = &mat->params;
    CHECK(texture_path_contains(mc, p->tex_indices[TEX_DIFFUSE_COLOR],
                                "albedo_D.png"),
          "EmissiveProductMDL keeps diffuse albedo texture");
    CHECK(p->tex_indices[TEX_EMISSIVE_COLOR] >= 0 &&
          p->tex_indices[TEX_EMISSIVE_COLOR] < mc->ntextures &&
          strstr(mc->textures[p->tex_indices[TEX_EMISSIVE_COLOR]].path,
                 "mdl_baked_emissive_product:") != NULL,
          "EmissiveProductMDL bakes emissive color*mask product");
    CHECK(nearly(p->emissive_color[3], 3.0f),
          "EmissiveProductMDL preserves authored emissive scalar");
}

static void check_materialx_proc_graph_scene(const char* usd)
{
    NanousdStage stage = nanousd_open(usd);
    CHECK(stage && nanousd_isvalid(stage), "nanousd_open MaterialX graph fixture");
    if (!stage || !nanousd_isvalid(stage)) {
        if (stage) nanousd_close(stage);
        return;
    }

    char scene_dir[1024];
    dirname_of(usd, scene_dir, sizeof(scene_dir));
    MaterialCollection* mc = materials_load(stage, scene_dir);
    CHECK(mc != NULL, "materials_load MaterialX graph fixture");
    CHECK(mc && mc->nmaterials == 1, "one MaterialX graph material loaded");
    CHECK(mc && mc->ntextures == 3, "MaterialX graph loads three textures");

    SceneMaterial* mat = find_material(mc, "M_ProcGraph");
    CHECK(mat != NULL, "M_ProcGraph material exists");
    if (mat) {
        MaterialParams* p = &mat->params;
        CHECK(nearly(p->base_color[0], 0.25f) &&
              nearly(p->base_color[1], 0.50f) &&
              nearly(p->base_color[2], 0.75f),
              "MaterialX graph preserves interface base tint");
        CHECK(nearly(p->roughness, 0.42f),
              "MaterialX graph preserves interface roughness");
        CHECK(nearly(p->mdl_uv_transform[0], 2.0f) &&
              nearly(p->mdl_uv_transform[1], 3.0f),
              "MaterialX graph applies tiledimage uvtiling");
        CHECK(texture_path_contains(mc, p->tex_indices[TEX_DIFFUSE_COLOR],
                                    "base_color.png"),
              "MaterialX graph binds upstream base-color texture");
        CHECK(texture_path_contains(mc, p->tex_indices[TEX_ROUGHNESS],
                                    "roughness.png"),
              "MaterialX graph binds upstream roughness texture");
        CHECK(texture_path_contains(mc, p->tex_indices[TEX_NORMAL],
                                    "normal.png"),
              "MaterialX graph binds upstream normal texture");
    }

    materials_free(mc);
    nanousd_close(stage);
}

static void check_materialx_resource_search_scene(const char* usd)
{
    char scene_dir[1024];
    char search_root[1024];
    dirname_of(usd, scene_dir, sizeof(scene_dir));
    dirname_of(scene_dir, search_root, sizeof(search_root));
    setenv("MATERIALX_SEARCH_PATH", search_root, 1);

    NanousdStage stage = nanousd_open(usd);
    CHECK(stage && nanousd_isvalid(stage),
          "nanousd_open MaterialX resource-search fixture");
    if (!stage || !nanousd_isvalid(stage)) {
        if (stage) nanousd_close(stage);
        return;
    }

    MaterialCollection* mc = materials_load(stage, scene_dir);
    CHECK(mc != NULL, "materials_load MaterialX resource-search fixture");
    CHECK(mc && mc->nmaterials == 1,
          "one MaterialX resource-search material loaded");
    CHECK(mc && mc->ntextures == 3,
          "MaterialX resource-search loads three textures");

    SceneMaterial* mat = find_material(mc, "M_ResourceGraph");
    CHECK(mat != NULL, "M_ResourceGraph material exists");
    if (mat) {
        MaterialParams* p = &mat->params;
        CHECK(texture_path_contains(mc, p->tex_indices[TEX_DIFFUSE_COLOR],
                                    "materialx_proc_graph/base_color.png"),
              "MaterialX search path binds base-color texture");
        CHECK(texture_path_contains(mc, p->tex_indices[TEX_ROUGHNESS],
                                    "materialx_proc_graph/roughness.png"),
              "MaterialX search path binds roughness texture");
        CHECK(texture_path_contains(mc, p->tex_indices[TEX_NORMAL],
                                    "materialx_proc_graph/normal.png"),
              "MaterialX search path binds normal texture");
    }

    materials_free(mc);
    nanousd_close(stage);
}

static void check_usd_preview_interface_fallback_scene(const char* usd)
{
    NanousdStage stage = nanousd_open(usd);
    CHECK(stage && nanousd_isvalid(stage),
          "nanousd_open UsdPreview interface fallback fixture");
    if (!stage || !nanousd_isvalid(stage)) {
        if (stage) nanousd_close(stage);
        return;
    }

    char scene_dir[1024];
    dirname_of(usd, scene_dir, sizeof(scene_dir));
    MaterialCollection* mc = materials_load(stage, scene_dir);
    CHECK(mc != NULL, "materials_load UsdPreview interface fallback fixture");
    CHECK(mc && mc->nmaterials == 1,
          "one UsdPreview interface fallback material loaded");

    SceneMaterial* mat = find_material(mc, "LeafMat");
    CHECK(mat != NULL, "LeafMat material exists");
    if (mat) {
        int mat_idx = (int)(mat - mc->materials);
        MaterialParams* p = &mat->params;
        CHECK(nearly(p->base_color[0], 0.12f) &&
              nearly(p->base_color[1], 0.45f) &&
              nearly(p->base_color[2], 0.20f),
              "LeafMat uses Material inputs:baseColor when Ptex preview is unresolved");
        CHECK(nearly(p->opacity, 0.77f) &&
              nearly(p->base_color[3], 0.77f),
              "LeafMat maps Material inputs:alpha to opacity");
        CHECK(nearly(p->roughness, 0.83f) &&
              nearly(p->metallic, 0.09f) &&
              nearly(p->ior, 1.31f),
              "LeafMat preserves Material scalar controls");
        CHECK(nearly(p->clearcoat, 0.20f) &&
              nearly(p->clearcoat_roughness, 0.25f),
              "LeafMat maps Disney clearcoatGloss to clearcoat roughness");
        CHECK(p->tex_indices[TEX_DIFFUSE_COLOR] < 0 &&
              p->use_vertex_color == 0,
              "LeafMat keeps authored color instead of vertex-color fallback");

        CHECK(materials_find_binding_by_path(mc,
              "/World/NativeInstance/Leaf#0") == mat_idx,
              "USDA binding hint resolves direct native-instance leaf path");

        NanousdPrim mesh = nanousd_primpath(stage, "/World/Leaf");
        CHECK(mesh != NULL, "Leaf mesh prim exists");
        if (mesh) {
            CHECK(materials_find_binding(mc, stage, mesh) == mat_idx,
                  "relationship binding resolves LeafMat");
            nanousd_freeprim(mesh);
        }
    }

    materials_free(mc);
    nanousd_close(stage);
}

static void check_usd_preview_ptex_placeholder_scene(const char* usd)
{
    NanousdStage stage = nanousd_open(usd);
    CHECK(stage && nanousd_isvalid(stage),
          "nanousd_open UsdPreview Ptex placeholder fixture");
    if (!stage || !nanousd_isvalid(stage)) {
        if (stage) nanousd_close(stage);
        return;
    }

    char scene_dir[1024];
    dirname_of(usd, scene_dir, sizeof(scene_dir));
    MaterialCollection* mc = materials_load(stage, scene_dir);
    CHECK(mc != NULL, "materials_load UsdPreview Ptex placeholder fixture");
    CHECK(mc && mc->nmaterials == 2,
          "two UsdPreview Ptex placeholder materials loaded");

    SceneMaterial* trunk = find_material(mc, "TrunkMat");
    CHECK(trunk != NULL, "TrunkMat material exists");
    if (trunk) {
        MaterialParams* p = &trunk->params;
        CHECK(nearly(p->base_color[0], 1.0f) &&
              nearly(p->base_color[1], 0.0f) &&
              nearly(p->base_color[2], 0.0f),
              "TrunkMat keeps authored material baseColor");
        CHECK(strstr(trunk->ptex_color_path, "trunk_geo.ptx") != NULL,
              "TrunkMat records authored Ptex surfaceMap path");
    }

    SceneMaterial* frond = find_material(mc, "FrondMat");
    CHECK(frond != NULL, "FrondMat material exists");
    if (frond) {
        MaterialParams* p = &frond->params;
        CHECK(nearly(p->base_color[0], 0.18f) &&
              nearly(p->base_color[1], 0.18f) &&
              nearly(p->base_color[2], 0.18f),
              "FrondMat keeps neutral UsdPreview default until real Ptex is sampled");
        CHECK(strstr(frond->ptex_color_path, "frond_geo.ptx") != NULL,
              "FrondMat records authored Ptex surfaceMap path");
    }

    materials_free(mc);
    nanousd_close(stage);
}

int main(int argc, char** argv)
{
    const char* usd = (argc > 1)
        ? argv[1]
        : "test/fixtures/mdl_surface_outputs.usda";
    const char* mtlx_graph_usd = (argc > 2)
        ? argv[2]
        : "test/materialx_proc_graph/scene.usda";
    const char* mtlx_resource_usd = (argc > 3)
        ? argv[3]
        : "test/materialx_resource_graph/scene.usda";
    const char* usd_preview_interface_usd = (argc > 4)
        ? argv[4]
        : "test/fixtures/usd_preview_interface_fallback.usda";
    const char* usd_preview_ptex_placeholder_usd = (argc > 5)
        ? argv[5]
        : "test/fixtures/usd_preview_ptex_placeholder_fallback.usda";

    NanousdStage stage = nanousd_open(usd);
    CHECK(stage && nanousd_isvalid(stage), "nanousd_open mdl fixture");
    if (!stage || !nanousd_isvalid(stage)) {
        if (stage) nanousd_close(stage);
        return 1;
    }

    char scene_dir[1024];
    dirname_of(usd, scene_dir, sizeof(scene_dir));
    MaterialCollection* mc = materials_load(stage, scene_dir);
    CHECK(mc != NULL, "materials_load mdl fixture");
    CHECK(mc && mc->nmaterials == 6, "six MDL materials loaded");

    const float base_surface[3] = {0.21f, 0.32f, 0.43f};
    const float spec_surface[3] = {0.11f, 0.22f, 0.33f};
    check_mdl_material(mc, "SurfaceOutputMDL", base_surface, spec_surface,
                       0.73f, 0.17f, 0.66f, 0.44f);

    const float base_mdl[3] = {0.24f, 0.35f, 0.46f};
    const float spec_mdl[3] = {0.14f, 0.25f, 0.36f};
    check_mdl_material(mc, "MdlSurfaceOutputMDL", base_mdl, spec_mdl,
                       0.63f, 0.27f, 0.76f, 0.54f);

    const float base_mtlx[3] = {0.27f, 0.38f, 0.49f};
    const float spec_mtlx[3] = {0.17f, 0.28f, 0.39f};
    check_mdl_material(mc, "MtlxSurfaceOutputMDL", base_mtlx, spec_mtlx,
                       0.53f, 0.37f, 0.86f, 0.64f);

    check_mask_selection_material(mc);
    check_body_mask_material(mc);
    check_emissive_product_material(mc);

    materials_free(mc);
    nanousd_close(stage);

    check_materialx_proc_graph_scene(mtlx_graph_usd);
    check_materialx_resource_search_scene(mtlx_resource_usd);
    check_usd_preview_interface_fallback_scene(usd_preview_interface_usd);
    check_usd_preview_ptex_placeholder_scene(usd_preview_ptex_placeholder_usd);

    fprintf(stderr, "\n%s: %d failures\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
