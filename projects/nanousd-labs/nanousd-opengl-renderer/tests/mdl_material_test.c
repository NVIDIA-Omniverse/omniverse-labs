// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "material.h"
#include <nanousd/nanousdapi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

static int copy_file(const char* src, const char* dst)
{
    FILE* in = fopen(src, "rb");
    if (!in) return 0;
    FILE* out = fopen(dst, "wb");
    if (!out) {
        fclose(in);
        return 0;
    }

    char buf[8192];
    int ok = 1;
    for (;;) {
        size_t n = fread(buf, 1, sizeof(buf), in);
        if (n > 0 && fwrite(buf, 1, n, out) != n) ok = 0;
        if (n < sizeof(buf)) {
            if (ferror(in)) ok = 0;
            break;
        }
    }

    fclose(in);
    if (fclose(out) != 0) ok = 0;
    return ok;
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

    snprintf(msg, sizeof(msg), "%s MDL v-flip", name);
    CHECK(p->v_flip == 1, msg);

    snprintf(msg, sizeof(msg), "%s specular workflow", name);
    CHECK(p->use_specular_workflow == 1 &&
          nearly(p->specular_color[0], spec[0]) &&
          nearly(p->specular_color[1], spec[1]) &&
          nearly(p->specular_color[2], spec[2]), msg);
}

static void check_no_material_scene(const char* usd)
{
    NanousdStage stage = nanousd_open(usd);
    CHECK(stage && nanousd_isvalid(stage), "nanousd_open no-material fixture");
    if (!stage || !nanousd_isvalid(stage)) {
        if (stage) nanousd_close(stage);
        return;
    }

    char scene_dir[1024];
    dirname_of(usd, scene_dir, sizeof(scene_dir));
    MaterialCollection* mc = materials_load(stage, scene_dir);
    CHECK(mc != NULL, "materials_load no-material fixture");
    CHECK(mc && mc->nmaterials == 0, "no-material fixture returns empty collection");

    materials_free(mc);
    nanousd_close(stage);
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
    char test_dir[1024];
    dirname_of(usd, scene_dir, sizeof(scene_dir));
    dirname_of(scene_dir, test_dir, sizeof(test_dir));

    const char* tmp_base = getenv("TMPDIR");
    if (!tmp_base || !tmp_base[0]) tmp_base = "/tmp";
    char temp_template[1024];
    int n = snprintf(temp_template, sizeof(temp_template),
                     "%s/nusd_mtlx_resource_XXXXXX", tmp_base);
    CHECK(n >= 0 && (size_t)n < sizeof(temp_template),
          "MaterialX resource temp path fits");
    if (n < 0 || (size_t)n >= sizeof(temp_template)) return;

    char* temp_dir = mkdtemp(temp_template);
    CHECK(temp_dir != NULL, "MaterialX resource temp dir created");
    if (!temp_dir) return;

    const char* src_names[] = {"base_color.png", "roughness.png", "normal.png"};
    const char* dst_names[] = {"mx_env_base_color.png",
                               "mx_env_roughness.png",
                               "mx_env_normal.png"};
    char dst_paths[3][1024];
    int copied = 1;
    for (int i = 0; i < 3; i++) {
        char src[1024];
        n = snprintf(src, sizeof(src), "%s/materialx_proc_graph/%s",
                     test_dir, src_names[i]);
        if (n < 0 || (size_t)n >= sizeof(src)) copied = 0;
        n = snprintf(dst_paths[i], sizeof(dst_paths[i]), "%s/%s",
                     temp_dir, dst_names[i]);
        if (n < 0 || (size_t)n >= sizeof(dst_paths[i])) copied = 0;
        if (copied && !copy_file(src, dst_paths[i])) copied = 0;
    }
    CHECK(copied, "MaterialX resource textures staged outside scene tree");
    if (!copied) {
        for (int i = 0; i < 3; i++) remove(dst_paths[i]);
        rmdir(temp_dir);
        return;
    }

    const char* old_env = getenv("MATERIALX_SEARCH_PATH");
    char old_env_copy[4096];
    int had_old_env = old_env && old_env[0];
    if (had_old_env)
        snprintf(old_env_copy, sizeof(old_env_copy), "%s", old_env);
    setenv("MATERIALX_SEARCH_PATH", temp_dir, 1);

    NanousdStage stage = nanousd_open(usd);
    CHECK(stage && nanousd_isvalid(stage),
          "nanousd_open MaterialX resource-search fixture");
    if (!stage || !nanousd_isvalid(stage)) {
        if (stage) nanousd_close(stage);
        if (had_old_env) setenv("MATERIALX_SEARCH_PATH", old_env_copy, 1);
        else unsetenv("MATERIALX_SEARCH_PATH");
        for (int i = 0; i < 3; i++) remove(dst_paths[i]);
        rmdir(temp_dir);
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
                                    "mx_env_base_color.png"),
              "MaterialX search path binds base-color texture");
        CHECK(texture_path_contains(mc, p->tex_indices[TEX_ROUGHNESS],
                                    "mx_env_roughness.png"),
              "MaterialX search path binds roughness texture");
        CHECK(texture_path_contains(mc, p->tex_indices[TEX_NORMAL],
                                    "mx_env_normal.png"),
              "MaterialX search path binds normal texture");
    }

    materials_free(mc);
    nanousd_close(stage);
    if (had_old_env) setenv("MATERIALX_SEARCH_PATH", old_env_copy, 1);
    else unsetenv("MATERIALX_SEARCH_PATH");
    for (int i = 0; i < 3; i++) remove(dst_paths[i]);
    rmdir(temp_dir);
}

int main(int argc, char** argv)
{
    const char* usd = (argc > 1)
        ? argv[1]
        : "tests/fixtures/mdl_surface_outputs.usda";
    const char* no_material_usd = (argc > 2)
        ? argv[2]
        : "tests/fixtures/no_material_extra_point.usda";
    const char* mtlx_graph_usd = (argc > 3)
        ? argv[3]
        : "tests/materialx_proc_graph/scene.usda";
    const char* mtlx_resource_usd = (argc > 4)
        ? argv[4]
        : "tests/materialx_resource_graph/scene.usda";

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
    CHECK(mc && mc->nmaterials == 3, "three MDL materials loaded");

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

    materials_free(mc);
    nanousd_close(stage);

    check_no_material_scene(no_material_usd);
    check_materialx_proc_graph_scene(mtlx_graph_usd);
    check_materialx_resource_search_scene(mtlx_resource_usd);

    fprintf(stderr, "\n%s: %d failures\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
