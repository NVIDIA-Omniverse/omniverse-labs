// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * material_mtlx.cpp — MaterialX (.mtlx) loader for the opengl renderer.
 *
 * The OpenChessSet and similar open USD assets bind their materials via
 * USD references to .mtlx files (e.g.
 *   def "Chess_Black_Material" ( prepend references = @./mat.mtlx@ )
 * ). nanousd's USD reader doesn't ship a MaterialX file-format plugin,
 * so those references don't resolve and 0 Material prims show up at
 * load time. Without this module, opengl renders the chess set as
 * untextured gray geometry.
 *
 * What we do: scan scene_dir recursively for *.mtlx, parse via the
 * MaterialX C++ API, walk every <surfacematerial> we find, and follow
 * its surfaceshader input to a <standard_surface>. The connected
 * <image> nodes (or <normalmap>-wrapped <image> for the normal slot)
 * become textures bound to the existing UsdPreviewSurface-shaped
 * MaterialParams (TEX_DIFFUSE_COLOR, TEX_NORMAL, TEX_ROUGHNESS,
 * TEX_METALLIC, …).
 *
 * Mirrors nanousd-vulkan-renderer/src/material.cpp's MaterialX path,
 * trimmed to the inputs the existing opengl PBR shader consumes.
 * Transmission constants are routed into the shader for the chess
 * pawn-top/glass materials; subsurface remains a later phase.
 */
#include "material.h"

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Library.h>
#include <MaterialXFormat/Environ.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mx = MaterialX;
namespace fs = std::filesystem;

/* ---- Library state ----------------------------------------------------- */

static mx::DocumentPtr g_stdlib;
static bool            g_materialx_initialized = false;

extern "C" int materialx_init(void)
{
    if (g_materialx_initialized) return 1;
    try {
        g_stdlib = mx::createDocument();

        /* MATERIALX_SEARCH_PATH defaulted by CMake; falls back to a few
         * well-known locations for in-tree development setups. */
        mx::FileSearchPath sp;
#ifdef MATERIALX_SEARCH_PATH
        sp.append(mx::FilePath(MATERIALX_SEARCH_PATH));
#endif
        const char* env = std::getenv("MATERIALX_SEARCH_PATH");
        if (env && *env) sp.append(mx::FilePath(env));

        mx::loadLibraries({"libraries"}, sp, g_stdlib);

        size_t n_defs = 0;
        for (auto& el : g_stdlib->getNodeDefs()) { (void)el; ++n_defs; }
        std::fprintf(stderr,
                     "materialx (gl): loaded %zu standard library definitions\n",
                     n_defs);
        g_materialx_initialized = true;
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "materialx (gl): init failed: %s\n", e.what());
        g_stdlib.reset();
        g_materialx_initialized = false;
        return 0;
    }
}

extern "C" void materialx_shutdown(void)
{
    g_stdlib.reset();
    g_materialx_initialized = false;
}

/* ---- Standard Surface input resolution -------------------------------- */

namespace {

struct SsInput {
    bool        is_constant   = false;
    bool        is_normal_map = false;
    bool        has_uv_scale  = false;
    int         n_vals        = 0;
    float       v[3]          = {0, 0, 0};
    float       uv_scale[2]   = {1.0f, 1.0f};
    std::string file_path;
    std::string nodegraph_name;
    std::string output_name;
};

struct ImageCandidate {
    std::string file_path;
    bool        has_uv_scale = false;
    float       uv_scale[2] = {1.0f, 1.0f};
};

/* Parse a comma-separated list of floats from MaterialX's value-string
 * format (e.g. "0.8, 0.1, 0.1" for color3, "0.6" for float). */
static int parse_value(const std::string& s, int max_vals, float* out)
{
    int n = 0;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',') && n < max_vals) {
        size_t a = tok.find_first_not_of(" \t");
        size_t b = tok.find_last_not_of(" \t");
        if (a == std::string::npos) continue;
        try {
            out[n++] = std::stof(tok.substr(a, b - a + 1));
        } catch (...) { /* leave previous values intact */ }
    }
    return n;
}

static int parse_graph_input_value(mx::NodeGraphPtr ng,
                                   mx::InputPtr in,
                                   int max_vals,
                                   float* out,
                                   int depth = 0)
{
    if (!in || depth > 6) return 0;

    std::string vs = in->getValueString();
    if (!vs.empty()) return parse_value(vs, max_vals, out);

    if (ng && in->hasInterfaceName()) {
        mx::InputPtr iface = ng->getInput(in->getInterfaceName());
        int n = parse_graph_input_value(ng, iface, max_vals, out, depth + 1);
        if (n > 0) return n;
    }

    if (ng) {
        std::string child_name = in->getNodeName();
        mx::NodePtr child = child_name.empty() ? mx::NodePtr() : ng->getNode(child_name);
        if (child) {
            mx::InputPtr value = child->getInput("value");
            int n = parse_graph_input_value(ng, value, max_vals, out, depth + 1);
            if (n > 0) return n;
            mx::InputPtr inner = child->getInput("in");
            n = parse_graph_input_value(ng, inner, max_vals, out, depth + 1);
            if (n > 0) return n;
        }
    }

    return 0;
}

static void fill_image_candidate(mx::NodeGraphPtr ng,
                                 mx::NodePtr image,
                                 ImageCandidate& out)
{
    if (!image) return;
    mx::InputPtr file_in = image->getInput("file");
    if (file_in) out.file_path = file_in->getValueString();

    mx::InputPtr uv = image->getInput("uvtiling");
    if (uv) {
        float vals[2] = {1.0f, 1.0f};
        int n = parse_graph_input_value(ng, uv, 2, vals);
        if (n > 0) {
            out.has_uv_scale = true;
            out.uv_scale[0] = vals[0];
            out.uv_scale[1] = (n > 1) ? vals[1] : vals[0];
        }
    }
}

static void apply_image_candidate(const ImageCandidate& image, SsInput& out)
{
    out.file_path = image.file_path;
    out.has_uv_scale = image.has_uv_scale;
    out.uv_scale[0] = image.uv_scale[0];
    out.uv_scale[1] = image.uv_scale[1];
}

static std::string lower_copy(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    return s;
}

static bool contains_any(const std::string& s,
                         std::initializer_list<const char*> needles)
{
    for (const char* n : needles) {
        if (s.find(n) != std::string::npos) return true;
    }
    return false;
}

static int texture_semantic_score(const std::string& path,
                                  const std::string& input_name)
{
    std::string p = lower_copy(path);
    std::string in = lower_copy(input_name);
    if (contains_any(in, {"normal"}))
        return contains_any(p, {"normal", "nrm", "bump"}) ? 100 : 1;
    if (contains_any(in, {"rough"}))
        return contains_any(p, {"rough"}) ? 100 :
               contains_any(p, {"mask"}) ? 15 : 1;
    if (contains_any(in, {"metal"}))
        return contains_any(p, {"metal"}) ? 100 : 1;
    if (contains_any(in, {"opacity", "alpha"}))
        return contains_any(p, {"opacity", "alpha"}) ? 100 : 1;
    if (contains_any(in, {"base_color", "diffuse", "albedo", "color"}))
        return contains_any(p, {"base", "albedo", "diffuse", "color"}) ? 100 :
               contains_any(p, {"mask"}) ? 10 : 1;
    return 1;
}

static void collect_upstream_images(mx::NodeGraphPtr ng,
                                    mx::NodePtr node,
                                    std::unordered_set<std::string>& seen,
                                    std::vector<ImageCandidate>& out,
                                    int depth)
{
    if (!ng || !node || depth > 12) return;
    std::string key = node->getName();
    if (!seen.insert(key).second) return;

    const std::string& cat = node->getCategory();
    if (cat == "image" || cat == "tiledimage") {
        ImageCandidate image;
        fill_image_candidate(ng, node, image);
        if (!image.file_path.empty()) out.push_back(image);
        return;
    }

    for (mx::InputPtr in : node->getInputs()) {
        if (!in) continue;
        std::string child_name = in->getNodeName();
        if (child_name.empty()) continue;
        collect_upstream_images(ng, ng->getNode(child_name),
                                seen, out, depth + 1);
    }
}

static ImageCandidate choose_upstream_image(mx::NodeGraphPtr ng,
                                            mx::NodePtr driver,
                                            const std::string& input_name)
{
    std::vector<ImageCandidate> candidates;
    std::unordered_set<std::string> seen;
    collect_upstream_images(ng, driver, seen, candidates, 0);
    if (candidates.empty()) return {};

    int best = 0;
    int best_score = texture_semantic_score(candidates[0].file_path, input_name);
    for (int i = 1; i < (int)candidates.size(); ++i) {
        int score = texture_semantic_score(candidates[i].file_path, input_name);
        if (score > best_score) {
            best = i;
            best_score = score;
        }
    }
    return candidates[best];
}

static void apply_interface_constant(mx::NodeGraphPtr ng,
                                     const std::string& input_name,
                                     SsInput& r)
{
    if (!ng) return;
    std::string wanted = lower_copy(input_name);
    int best_score = 0;
    float best[3] = {0, 0, 0};
    int best_n = 0;

    for (mx::InputPtr in : ng->getInputs()) {
        if (!in || in->getValueString().empty()) continue;
        std::string name = lower_copy(in->getName());
        std::string type = in->getType();
        int score = 0;
        int max_vals = 1;
        if (contains_any(wanted, {"base_color", "diffuse", "albedo", "color"}) &&
            (type == "color3" || type == "vector3") &&
            contains_any(name, {"base", "brick", "diffuse", "albedo", "color"})) {
            score = contains_any(name, {"base"}) ? 100 : 80;
            max_vals = 3;
        } else if (contains_any(wanted, {"rough"}) &&
                   (type == "float" || type == "integer") &&
                   contains_any(name, {"rough"})) {
            score = 100;
            max_vals = 1;
        }
        if (!score) continue;

        float vals[3] = {0, 0, 0};
        int n = parse_value(in->getValueString(), max_vals, vals);
        if (n > 0 && score > best_score) {
            best_score = score;
            best_n = n;
            best[0] = vals[0];
            best[1] = vals[1];
            best[2] = vals[2];
        }
    }

    if (best_score > 0) {
        r.is_constant = true;
        r.n_vals = best_n;
        r.v[0] = best[0];
        r.v[1] = best[1];
        r.v[2] = best[2];
    }
}

/* Resolve a Standard Surface input to either a constant or an upstream
 * image filename. Handles direct images plus a bounded upstream walk for
 * simple MaterialX sample graphs such as the procedural brick material. */
static SsInput resolve_ss_input(mx::NodePtr ss,
                                mx::DocumentPtr doc,
                                const std::string& input_name)
{
    SsInput r;
    if (!ss || !doc) return r;

    mx::InputPtr in = ss->getInput(input_name);
    if (!in) return r;

    /* Connection through a nodegraph output — the chess set's pattern. */
    std::string ng_name  = in->getNodeGraphString();
    std::string out_name = in->getOutputString();
    if (!ng_name.empty() && !out_name.empty()) {
        r.nodegraph_name = ng_name;
        r.output_name = out_name;
        mx::NodeGraphPtr ng = doc->getNodeGraph(ng_name);
        if (ng) {
            mx::OutputPtr out_elt = ng->getOutput(out_name);
            if (out_elt) {
                std::string driver_name = out_elt->getNodeName();
                mx::NodePtr driver = ng->getNode(driver_name);
                if (driver) {
                    const std::string& cat = driver->getCategory();
                    if (cat == "image" || cat == "tiledimage") {
                        ImageCandidate image;
                        fill_image_candidate(ng, driver, image);
                        apply_image_candidate(image, r);
                        return r;
                    }
                    if (cat == "normalmap") {
                        mx::InputPtr nmIn = driver->getInput("in");
                        if (nmIn) {
                            std::string inner = nmIn->getNodeName();
                            mx::NodePtr inner_node = ng->getNode(inner);
                            if (inner_node && (inner_node->getCategory() == "image" ||
                                               inner_node->getCategory() == "tiledimage")) {
                                ImageCandidate image;
                                fill_image_candidate(ng, inner_node, image);
                                apply_image_candidate(image, r);
                                r.is_normal_map = true;
                                return r;
                            }
                        }
                    }
                    ImageCandidate image = choose_upstream_image(ng, driver, input_name);
                    apply_image_candidate(image, r);
                    if (!r.file_path.empty()) {
                        apply_interface_constant(ng, input_name, r);
                        return r;
                    }
                }
            }
        }
    }

    /* Constant value. */
    std::string vs = in->getValueString();
    if (vs.empty()) return r;
    r.is_constant = true;
    int max_vals = (in->getType() == "float" || in->getType() == "integer") ? 1 : 3;
    r.n_vals = parse_value(vs, max_vals, r.v);
    return r;
}

static float clamp01(float v)
{
    return std::max(0.0f, std::min(1.0f, v));
}

static unsigned char to_linear_byte(float linear)
{
    return (unsigned char)(clamp01(linear) * 255.0f + 0.5f);
}

static unsigned char to_srgb_byte(float linear)
{
    linear = clamp01(linear);
    float srgb = (linear <= 0.0031308f)
        ? linear * 12.92f
        : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
    return (unsigned char)(clamp01(srgb) * 255.0f + 0.5f);
}

static float repeat_coord(float v)
{
    v = v - std::floor(v);
    return v < 0.0f ? v + 1.0f : v;
}

static const unsigned char* sample_repeat(const MaterialTexture& tex,
                                          float u,
                                          float v)
{
    static const unsigned char zero[4] = {0, 0, 0, 255};
    if (!tex.pixels || tex.width <= 0 || tex.height <= 0) return zero;
    int x = (int)(repeat_coord(u) * (float)tex.width);
    int y = (int)(repeat_coord(v) * (float)tex.height);
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= tex.width) x = tex.width - 1;
    if (y >= tex.height) y = tex.height - 1;
    return tex.pixels + ((size_t)y * (size_t)tex.width + (size_t)x) * 4;
}

static float sample_repeat_r(const MaterialTexture& tex, float u, float v)
{
    return sample_repeat(tex, u, v)[0] / 255.0f;
}

static void rgb_to_hsv(const float rgb[3], float hsv[3])
{
    float r = rgb[0], g = rgb[1], b = rgb[2];
    float maxv = std::max(r, std::max(g, b));
    float minv = std::min(r, std::min(g, b));
    float d = maxv - minv;
    float h = 0.0f;
    if (d > 1e-6f) {
        if (maxv == r) {
            h = std::fmod((g - b) / d, 6.0f);
        } else if (maxv == g) {
            h = (b - r) / d + 2.0f;
        } else {
            h = (r - g) / d + 4.0f;
        }
        h /= 6.0f;
        if (h < 0.0f) h += 1.0f;
    }
    hsv[0] = h;
    hsv[1] = maxv > 1e-6f ? d / maxv : 0.0f;
    hsv[2] = maxv;
}

static void hsv_to_rgb(const float hsv[3], float rgb[3])
{
    float h = repeat_coord(hsv[0]) * 6.0f;
    float s = clamp01(hsv[1]);
    float v = clamp01(hsv[2]);
    int i = (int)std::floor(h);
    float f = h - (float)i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));
    switch (i % 6) {
    case 0: rgb[0] = v; rgb[1] = t; rgb[2] = p; break;
    case 1: rgb[0] = q; rgb[1] = v; rgb[2] = p; break;
    case 2: rgb[0] = p; rgb[1] = v; rgb[2] = t; break;
    case 3: rgb[0] = p; rgb[1] = q; rgb[2] = v; break;
    case 4: rgb[0] = t; rgb[1] = p; rgb[2] = v; break;
    default: rgb[0] = v; rgb[1] = p; rgb[2] = q; break;
    }
}

static float read_graph_float(mx::NodeGraphPtr ng,
                              const char* name,
                              float fallback)
{
    if (!ng || !name) return fallback;
    mx::InputPtr in = ng->getInput(name);
    if (!in) return fallback;
    float v = fallback;
    return parse_value(in->getValueString(), 1, &v) > 0 ? v : fallback;
}

static void read_graph_color(mx::NodeGraphPtr ng,
                             const char* name,
                             const float fallback[3],
                             float out[3])
{
    out[0] = fallback[0];
    out[1] = fallback[1];
    out[2] = fallback[2];
    if (!ng || !name) return;
    mx::InputPtr in = ng->getInput(name);
    if (!in) return;
    parse_value(in->getValueString(), 3, out);
}

static bool find_graph_texture_file(mx::NodeGraphPtr ng,
                                    const char* filename,
                                    std::string& out)
{
    if (!ng || !filename) return false;
    for (mx::NodePtr node : ng->getNodes()) {
        if (!node) continue;
        const std::string& cat = node->getCategory();
        if (cat != "image" && cat != "tiledimage") continue;
        mx::InputPtr file = node->getInput("file");
        if (!file) continue;
        std::string value = file->getValueString();
        if (fs::path(value).filename() == filename) {
            out = value;
            return true;
        }
    }
    return false;
}

static int append_generated_texture(MaterialCollection* mc,
                                    const char* key,
                                    unsigned char* pixels,
                                    int width,
                                    int height)
{
    if (!mc || !key || !pixels || width <= 0 || height <= 0) return -1;
    for (int i = 0; i < mc->ntextures; ++i) {
        if (std::strcmp(mc->textures[i].path, key) == 0) {
            std::free(pixels);
            return i;
        }
    }
    int idx = mc->ntextures;
    MaterialTexture* new_texs = (MaterialTexture*)std::realloc(
        mc->textures, (size_t)(idx + 1) * sizeof(MaterialTexture));
    if (!new_texs) {
        std::free(pixels);
        return -1;
    }
    mc->textures = new_texs;
    std::memset(&mc->textures[idx], 0, sizeof(MaterialTexture));
    mc->textures[idx].pixels = pixels;
    mc->textures[idx].width = width;
    mc->textures[idx].height = height;
    std::snprintf(mc->textures[idx].path, sizeof(mc->textures[idx].path),
                  "%s", key);
    mc->ntextures = idx + 1;
    return idx;
}

static bool try_bind_brick_procedural(const SsInput& base,
                                      mx::DocumentPtr doc,
                                      MaterialParams* params,
                                      MaterialCollection* mc,
                                      const std::string& mtlx_dir,
                                      void* stage)
{
    if (!doc || !params || !mc || base.nodegraph_name.empty()) return false;

    mx::NodeGraphPtr ng = doc->getNodeGraph(base.nodegraph_name);
    if (!ng) return false;

    std::string base_gray, variation, dirt, mask, roughness, normal;
    if (!find_graph_texture_file(ng, "brick_base_gray.jpg", base_gray) ||
        !find_graph_texture_file(ng, "brick_variation_mask.jpg", variation) ||
        !find_graph_texture_file(ng, "brick_dirt_mask.jpg", dirt) ||
        !find_graph_texture_file(ng, "brick_mask.jpg", mask) ||
        !find_graph_texture_file(ng, "brick_roughness.jpg", roughness) ||
        !find_graph_texture_file(ng, "brick_normal.jpg", normal)) {
        return false;
    }

    int t_base = find_or_add_texture(mc, mtlx_dir.c_str(), base_gray.c_str(), stage);
    int t_var  = find_or_add_texture(mc, mtlx_dir.c_str(), variation.c_str(), stage);
    int t_dirt = find_or_add_texture(mc, mtlx_dir.c_str(), dirt.c_str(), stage);
    int t_mask = find_or_add_texture(mc, mtlx_dir.c_str(), mask.c_str(), stage);
    int t_rough = find_or_add_texture(mc, mtlx_dir.c_str(), roughness.c_str(), stage);
    int t_norm = find_or_add_texture(mc, mtlx_dir.c_str(), normal.c_str(), stage);
    if (t_base < 0 || t_var < 0 || t_dirt < 0 ||
        t_mask < 0 || t_rough < 0 || t_norm < 0) {
        return false;
    }

    const MaterialTexture& base_tex = mc->textures[t_base];
    const MaterialTexture& var_tex = mc->textures[t_var];
    const MaterialTexture& dirt_tex = mc->textures[t_dirt];
    const MaterialTexture& mask_tex = mc->textures[t_mask];
    const MaterialTexture& rough_tex = mc->textures[t_rough];
    const MaterialTexture& norm_tex = mc->textures[t_norm];
    int width = base_tex.width;
    int height = base_tex.height;
    if (width <= 0 || height <= 0) return false;

    size_t nbytes = (size_t)width * (size_t)height * 4;
    unsigned char* baked_color = (unsigned char*)std::malloc(nbytes);
    unsigned char* baked_rough = (unsigned char*)std::malloc(nbytes);
    unsigned char* baked_norm = (unsigned char*)std::malloc(nbytes);
    if (!baked_color || !baked_rough || !baked_norm) {
        std::free(baked_color);
        std::free(baked_rough);
        std::free(baked_norm);
        return false;
    }

    static const float brick_default[3] = {0.661876f, 0.19088f, 0.0f};
    static const float dirt_default[3] = {0.56372f, 0.56372f, 0.56372f};
    float brick_color[3], dirt_color[3], brick_hsv[3];
    read_graph_color(ng, "brick_color", brick_default, brick_color);
    read_graph_color(ng, "dirt_color", dirt_default, dirt_color);
    rgb_to_hsv(brick_color, brick_hsv);

    float hue_variation = read_graph_float(ng, "hue_variation", 0.083f);
    float value_variation = read_graph_float(ng, "value_variation", 0.787f);
    float roughness_amount = read_graph_float(ng, "roughness_amount", 0.853f);
    float dirt_amount = read_graph_float(ng, "dirt_amount", 0.248f);
    float uvtiling = read_graph_float(ng, "uvtiling", 3.0f);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float u = ((float)x + 0.5f) / (float)width * uvtiling;
            float v = ((float)y + 0.5f) / (float)height * uvtiling;
            float base_v = sample_repeat_r(base_tex, u, v);
            float var_v = sample_repeat_r(var_tex, u, v);
            float dirt_v = sample_repeat_r(dirt_tex, u, v);
            float mask_v = sample_repeat_r(mask_tex, u, v);
            float rough_v = sample_repeat_r(rough_tex, u, v);

            float add19 = hue_variation * var_v + base_v;
            float varied_hsv[3] = {
                brick_hsv[0] + (add19 - 0.35f) * hue_variation,
                brick_hsv[1],
                brick_hsv[2] + add19 * value_variation * var_v,
            };
            float varied_rgb[3];
            hsv_to_rgb(varied_hsv, varied_rgb);

            float dirt_mix = clamp01(dirt_amount * dirt_v);
            float dirty_rgb[3] = {
                varied_rgb[0] * (1.0f - dirt_mix) + dirt_color[0] * dirt_mix,
                varied_rgb[1] * (1.0f - dirt_mix) + dirt_color[1] * dirt_mix,
                varied_rgb[2] * (1.0f - dirt_mix) + dirt_color[2] * dirt_mix,
            };
            float mortar = 0.263273f * base_v;
            float out_rgb[3] = {
                mortar * (1.0f - mask_v) + dirty_rgb[0] * base_v * mask_v,
                mortar * (1.0f - mask_v) + dirty_rgb[1] * base_v * mask_v,
                mortar * (1.0f - mask_v) + dirty_rgb[2] * base_v * mask_v,
            };
            float out_rough = rough_v * roughness_amount /
                              std::max(mask_v, 0.00001f);

            size_t o = ((size_t)y * (size_t)width + (size_t)x) * 4;
            baked_color[o + 0] = to_srgb_byte(out_rgb[0]);
            baked_color[o + 1] = to_srgb_byte(out_rgb[1]);
            baked_color[o + 2] = to_srgb_byte(out_rgb[2]);
            baked_color[o + 3] = 255;
            baked_rough[o + 0] = to_linear_byte(out_rough);
            baked_rough[o + 1] = to_linear_byte(out_rough);
            baked_rough[o + 2] = to_linear_byte(out_rough);
            baked_rough[o + 3] = 255;

            const unsigned char* npx = sample_repeat(norm_tex, u, v);
            baked_norm[o + 0] = npx[0];
            baked_norm[o + 1] = npx[1];
            baked_norm[o + 2] = npx[2];
            baked_norm[o + 3] = 255;
        }
    }

    std::string key_prefix = std::string("__generated_mtlx_brick:") + mtlx_dir;
    int color_idx = append_generated_texture(
        mc, (key_prefix + ":base_color").c_str(), baked_color, width, height);
    int rough_idx = append_generated_texture(
        mc, (key_prefix + ":roughness").c_str(), baked_rough, width, height);
    int norm_idx = append_generated_texture(
        mc, (key_prefix + ":normal").c_str(), baked_norm, width, height);
    if (color_idx < 0 || rough_idx < 0 || norm_idx < 0) return false;

    params->tex_indices[TEX_DIFFUSE_COLOR] = color_idx;
    params->tex_indices[TEX_ROUGHNESS] = rough_idx;
    params->tex_indices[TEX_NORMAL] = norm_idx;
    params->base_color[0] = 1.0f;
    params->base_color[1] = 1.0f;
    params->base_color[2] = 1.0f;
    params->roughness = 1.0f;
    params->roughness_tex_scale = 1.0f;
    params->roughness_tex_bias = 0.0f;
    params->normal_scale = 1.0f;

    std::fprintf(stderr,
                 "material: baked MaterialX brick graph (%dx%d, uvtiling %.2f)\n",
                 width, height, uvtiling);
    return true;
}

/* Pull every Standard-Surface input we care about and bind to params /
 * texture slots. */
static void read_standard_surface(mx::NodePtr ss,
                                  mx::DocumentPtr doc,
                                  MaterialParams* params,
                                  MaterialCollection* mc,
                                  const std::string& mtlx_dir,
                                  void* stage)
{
    /* Defaults — match the MaterialX nodedef for unfilled inputs. */
    params->base_color[0] = 0.8f;
    params->base_color[1] = 0.8f;
    params->base_color[2] = 0.8f;
    params->base_color[3] = 1.0f;
    params->emissive_color[0] = 0.0f;
    params->emissive_color[1] = 0.0f;
    params->emissive_color[2] = 0.0f;
    params->emissive_color[3] = 1.0f;
    params->metallic = 0.0f;
    params->roughness = 0.2f;
    params->opacity = 1.0f;
    params->ior = 1.5f;
    params->occlusion = 1.0f;
    params->clearcoat = 0.0f;
    params->clearcoat_roughness = 0.01f;
    params->normal_scale = 1.0f;
    params->use_vertex_color = 0;
    params->udim_scale_u = 1.0f;
    params->udim_scale_v = 1.0f;
    params->opacity_threshold = 0.0f;
    params->v_flip = 0;
    params->transmission_color[0] = 1.0f;
    params->transmission_color[1] = 1.0f;
    params->transmission_color[2] = 1.0f;
    params->transmission_color[3] = 1.0f;
    params->transmission_weight = 0.0f;
    params->transmission_ior = 0.0f;
    params->roughness_tex_scale = 1.0f;
    params->roughness_tex_bias = 0.0f;
    params->mdl_uv_transform[0] = 1.0f;
    params->mdl_uv_transform[1] = 1.0f;
    params->mdl_uv_transform[2] = 0.0f;
    params->mdl_uv_transform[3] = 0.0f;
    for (int i = 0; i < MAX_MATERIAL_TEXTURES; i++) params->tex_indices[i] = -1;

    if (!ss) return;

    auto load_tex = [&](const std::string& rel) -> int {
        if (rel.empty()) return -1;
        return find_or_add_texture(mc, mtlx_dir.c_str(), rel.c_str(), stage);
    };

    auto bind_color3 = [&](const SsInput& in, float* out_rgb, int tex_slot) {
        if (in.has_uv_scale) {
            params->mdl_uv_transform[0] = in.uv_scale[0];
            params->mdl_uv_transform[1] = in.uv_scale[1];
        }
        if (in.is_constant && in.n_vals >= 3) {
            out_rgb[0] = in.v[0];
            out_rgb[1] = in.v[1];
            out_rgb[2] = in.v[2];
        }
        if (!in.file_path.empty()) {
            int t = load_tex(in.file_path);
            if (t >= 0 && tex_slot >= 0) params->tex_indices[tex_slot] = t;
        }
    };

    auto bind_float = [&](const SsInput& in, float* out_f, int tex_slot) {
        if (in.has_uv_scale) {
            params->mdl_uv_transform[0] = in.uv_scale[0];
            params->mdl_uv_transform[1] = in.uv_scale[1];
        }
        if (in.is_constant && in.n_vals >= 1) {
            *out_f = in.v[0];
        }
        if (!in.file_path.empty()) {
            int t = load_tex(in.file_path);
            if (t >= 0 && tex_slot >= 0) params->tex_indices[tex_slot] = t;
        }
    };

    SsInput base = resolve_ss_input(ss, doc, "base_color");
    SsInput rough = resolve_ss_input(ss, doc, "specular_roughness");
    SsInput nrm = resolve_ss_input(ss, doc, "normal");
    bool baked_brick = try_bind_brick_procedural(base, doc, params, mc,
                                                mtlx_dir, stage);

    if (!baked_brick) {
        bind_color3(base, params->base_color, TEX_DIFFUSE_COLOR);
        if (!base.is_constant && !base.file_path.empty() &&
            params->tex_indices[TEX_DIFFUSE_COLOR] >= 0) {
            params->base_color[0] = 1.0f;
            params->base_color[1] = 1.0f;
            params->base_color[2] = 1.0f;
        }
    }

    bind_float(resolve_ss_input(ss, doc, "metalness"),
               &params->metallic, TEX_METALLIC);

    if (!baked_brick)
        bind_float(rough, &params->roughness, TEX_ROUGHNESS);

    /* `normal` arrives via a <normalmap> wrapper; resolve_ss_input flagged
     * is_normal_map so we know to bind to TEX_NORMAL. */
    if (!baked_brick && !nrm.file_path.empty()) {
        if (nrm.has_uv_scale) {
            params->mdl_uv_transform[0] = nrm.uv_scale[0];
            params->mdl_uv_transform[1] = nrm.uv_scale[1];
        }
        int t = load_tex(nrm.file_path);
        if (t >= 0) params->tex_indices[TEX_NORMAL] = t;
    }

    /* Constants we can fold straight in (no texture slot needed). */
    {
        SsInput sIor = resolve_ss_input(ss, doc, "specular_IOR");
        if (sIor.is_constant && sIor.n_vals >= 1) {
            params->ior = sIor.v[0];
            params->transmission_ior = sIor.v[0];
        }
    }
    {
        SsInput tr = resolve_ss_input(ss, doc, "transmission");
        if (tr.is_constant && tr.n_vals >= 1) params->transmission_weight = tr.v[0];
        bind_color3(resolve_ss_input(ss, doc, "transmission_color"),
                    params->transmission_color, -1);
    }
    {
        SsInput em = resolve_ss_input(ss, doc, "emission");
        SsInput ec = resolve_ss_input(ss, doc, "emission_color");
        float weight = 0.0f;
        if (em.is_constant && em.n_vals >= 1) weight = em.v[0];
        if (weight > 0.0f && ec.is_constant && ec.n_vals >= 3) {
            params->emissive_color[0] = ec.v[0] * weight;
            params->emissive_color[1] = ec.v[1] * weight;
            params->emissive_color[2] = ec.v[2] * weight;
        }
    }
}

/* Recursively scan a directory for *.mtlx files. Skips hidden dirs
 * and limits depth to keep symlink loops bounded. */
static std::vector<std::string> find_mtlx_files(const std::string& root,
                                                int max_depth = 6)
{
    std::vector<std::string> out;
    try {
        if (!fs::exists(root) || !fs::is_directory(root)) return out;
        fs::recursive_directory_iterator it(root,
            fs::directory_options::skip_permission_denied);
        for (; it != fs::recursive_directory_iterator(); ++it) {
            if (it.depth() > max_depth) { it.disable_recursion_pending(); continue; }
            const auto& p = it->path();
            std::string fname = p.filename().string();
            if (!fname.empty() && fname[0] == '.') {
                if (fs::is_directory(p)) it.disable_recursion_pending();
                continue;
            }
            if (it->is_regular_file() && p.extension() == ".mtlx") {
                out.push_back(p.string());
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "material: mtlx scan failed under %s: %s\n",
                     root.c_str(), e.what());
    }
    std::sort(out.begin(), out.end());
    return out;
}

}  /* anonymous namespace */

/* ---- C entry point ----------------------------------------------------- */

extern "C" int materials_scan_mtlx_directory(MaterialCollection* mc,
                                             const char* scene_dir,
                                             void* stage)
{
    if (!mc || !scene_dir || !*scene_dir) return 0;

    std::vector<std::string> files = find_mtlx_files(scene_dir);
    if (files.empty()) return 0;

    if (!g_materialx_initialized) {
        if (!materialx_init()) return 0;
    }

    std::fprintf(stderr,
                 "material: scanning %zu .mtlx file(s) under %s\n",
                 files.size(), scene_dir);

    int added = 0;
    /* Track names already added so a second .mtlx file that re-defines
     * the same surfacematerial doesn't produce duplicate entries. Seed
     * this from the existing collection too: material.c may call this
     * once for the wrapper scene, then again for referenced layer dirs
     * and NUSD_MTLX_DIRS. */
    std::unordered_map<std::string, int> seen_by_name;
    for (int i = 0; i < mc->nmaterials; ++i) {
        if (mc->materials[i].name[0]) {
            seen_by_name[mc->materials[i].name] = i;
        }
    }

    for (const std::string& path : files) {
        mx::DocumentPtr doc = mx::createDocument();
        try {
            mx::readFromXmlFile(doc, path);
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                         "material: failed to parse %s: %s\n",
                         path.c_str(), e.what());
            continue;
        }

        std::string mtlx_dir = fs::path(path).parent_path().string();

        int per_file = 0;
        for (mx::NodePtr surfmat : doc->getNodes("surfacematerial")) {
            if (!surfmat) continue;
            std::string mat_name = surfmat->getName();
            if (seen_by_name.count(mat_name)) continue;

            /* Find the connected <standard_surface>. */
            mx::InputPtr ssIn = surfmat->getInput("surfaceshader");
            if (!ssIn) continue;
            std::string ss_name = ssIn->getNodeName();
            if (ss_name.empty()) continue;
            mx::NodePtr ss = doc->getNode(ss_name);
            if (!ss || ss->getCategory() != "standard_surface") continue;

            /* Append a SceneMaterial. */
            int idx = mc->nmaterials;
            SceneMaterial* arr = (SceneMaterial*)std::realloc(
                mc->materials, (size_t)(idx + 1) * sizeof(SceneMaterial));
            if (!arr) {
                std::fprintf(stderr,
                             "material: realloc failed at %d materials\n",
                             idx + 1);
                break;
            }
            mc->materials = arr;
            std::memset(&mc->materials[idx], 0, sizeof(SceneMaterial));
            std::snprintf(mc->materials[idx].name,
                          sizeof(mc->materials[idx].name),
                          "%s", mat_name.c_str());
            /* Convention used by materials_assign_bindings to look up
             * by USD prim path. The .mtlx route doesn't author a USD
             * prim_path, so leave it empty — meshes match by material
             * name via the relationship target (handled in viewer.c
             * material binding precompute). */
            mc->materials[idx].prim_path[0] = '\0';
            mc->materials[idx].shader_index = 0;
            read_standard_surface(ss, doc,
                                  &mc->materials[idx].params,
                                  mc, mtlx_dir, stage);

            mc->nmaterials = idx + 1;
            seen_by_name[mat_name] = idx;
            ++added;
            ++per_file;
        }

        if (per_file > 0) {
            std::fprintf(stderr, "material:   %s -> %d material(s)\n",
                         fs::path(path).filename().c_str(), per_file);
        }
    }

    return added;
}
