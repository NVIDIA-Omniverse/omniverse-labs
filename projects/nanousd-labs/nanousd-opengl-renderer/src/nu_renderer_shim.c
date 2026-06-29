// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * nu_renderer_shim.c — translates the canonical `nu_*` C API
 * (nusd_renderer.h) onto this repo's existing `viewer_*` OpenGL renderer
 * surface (viewer.h).
 *
 * Goal: nanousdview can dlopen libnusd_renderer_opengl.so and drive it
 * via the same calls it uses for the Vulkan renderer. Where the canonical
 * surface assumes hardware ray tracing or features the GLES renderer
 * doesn't have (RT, multi-camera tiled, CUDA-Vulkan zero-copy interop),
 * this shim returns NU_ERROR_NO_RT or NU_ERROR.
 *
 * Per the workspace plan: Vulkan owns the canonical nu_* API; this shim
 * brings opengl into line. The header in include/nusd_renderer.h is a
 * byte-equal vendored copy of Vulkan's; CI parity check enforces sync.
 */

#include "nusd_renderer.h"
#include "scene.h"
#include "viewer.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

/* ---- Internal renderer state ---- */

struct NuRenderer {
    NuRendererConfig cfg;
    Viewer*  viewer;
    NuCameraDesc cam;
    int          cam_set;
    /* Optional override for the camera up vector. Set by
     * nu_set_camera_explicit; cleared by nu_set_camera so the renderer
     * falls back to the loaded scene's authored upAxis. */
    float        cam_up[3];
    int          cam_up_set;
    float        projection_shift_x;
    float        projection_shift_y;
    int          scene_up_axis;  /* 0=X, 1=Y, 2=Z */
    int          fallback_lights_active;

    NuExposureDesc exposure_desc;
    float        tone_exposure_scale;
    float        tone_sky_scale;
    float        tone_white_point;
    uint32_t     tone_flags;

    /* Internal RGBA8 readback buffer. */
    unsigned char* rgba;
    int            rgba_w;
    int            rgba_h;

    NuRenderMode last_mode;
    int          last_render_ok;
    double       current_time;

    char err[256];
};

/* ---- Helpers ---- */

static void nu_set_err(NuRenderer* r, const char* fmt, ...) {
    if (!r) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(r->err, sizeof r->err, fmt, ap);
    va_end(ap);
}

static int ensure_rgba_buffer(NuRenderer* r, int w, int h) {
    if (r->rgba && r->rgba_w == w && r->rgba_h == h) return 1;
    free(r->rgba);
    size_t bytes = (size_t)w * (size_t)h * 4u;
    r->rgba = (unsigned char*)malloc(bytes);
    if (!r->rgba) { r->rgba_w = r->rgba_h = 0; return 0; }
    memset(r->rgba, 0, bytes);
    r->rgba_w = w;
    r->rgba_h = h;
    return 1;
}

static int parse_float_env2(const char* name_a, const char* name_b, float* out)
{
    const char* s = getenv(name_a);
    if ((!s || !*s) && name_b) s = getenv(name_b);
    if (!s || !*s) return 0;
    char* end = NULL;
    float v = strtof(s, &end);
    if (end == s || !isfinite(v)) return 0;
    *out = v;
    return 1;
}

static int parse_bool_env2(const char* name_a, const char* name_b, int* out)
{
    const char* s = getenv(name_a);
    if ((!s || !*s) && name_b) s = getenv(name_b);
    if (!s || !*s) return 0;
    if (strcmp(s, "1") == 0 || strcasecmp(s, "true") == 0 ||
        strcasecmp(s, "yes") == 0 || strcasecmp(s, "on") == 0) {
        *out = 1;
        return 1;
    }
    if (strcmp(s, "0") == 0 || strcasecmp(s, "false") == 0 ||
        strcasecmp(s, "no") == 0 || strcasecmp(s, "off") == 0) {
        *out = 0;
        return 1;
    }
    return 0;
}

static int parse_float_attr_line(const char* text, const char* name, float* out)
{
    size_t name_len = strlen(name);
    const char* p = text;
    while ((p = strstr(p, name)) != NULL) {
        const char* line_end = strchr(p, '\n');
        if (!line_end) line_end = p + strlen(p);
        const char* q = p + name_len;
        while (q < line_end && *q != '=') q++;
        if (q < line_end && *q == '=') {
            char* end = NULL;
            float v = strtof(q + 1, &end);
            if (end != q + 1 && isfinite(v)) {
                *out = v;
                return 1;
            }
        }
        p += name_len;
    }
    return 0;
}

static int parse_bool_attr_line(const char* text, const char* name, int* out)
{
    size_t name_len = strlen(name);
    const char* p = text;
    while ((p = strstr(p, name)) != NULL) {
        const char* line_end = strchr(p, '\n');
        if (!line_end) line_end = p + strlen(p);
        const char* q = p + name_len;
        while (q < line_end && *q != '=') q++;
        if (q < line_end && *q == '=') {
            q++;
            while (q < line_end && (*q == ' ' || *q == '\t')) q++;
            if (q < line_end) {
                if (*q == '1' || strncasecmp(q, "true", 4) == 0) {
                    *out = 1;
                    return 1;
                }
                if (*q == '0' || strncasecmp(q, "false", 5) == 0) {
                    *out = 0;
                    return 1;
                }
            }
        }
        p += name_len;
    }
    return 0;
}

static char* read_usd_scan_text(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    const size_t max_bytes = 8u * 1024u * 1024u;
    char* text = (char*)malloc(max_bytes + 1u);
    if (!text) {
        fclose(f);
        return NULL;
    }
    size_t n = fread(text, 1u, max_bytes, f);
    fclose(f);
    text[n] = '\0';
    return text;
}

static void exposure_desc_from_usd_path(const char* path, NuExposureDesc* desc)
{
    memset(desc, 0, sizeof(*desc));
    char* text = read_usd_scan_text(path);
    if (text) {
        if (parse_float_attr_line(text, "exposure:fStop", &desc->exposure_f_stop))
            desc->flags |= NU_EXPOSURE_HAS_FSTOP;
        if (parse_float_attr_line(text, "exposure:responsivity", &desc->exposure_responsivity))
            desc->flags |= NU_EXPOSURE_HAS_RESPONSIVITY;
        if (parse_float_attr_line(text, "exposure:time", &desc->exposure_time))
            desc->flags |= NU_EXPOSURE_HAS_TIME;
        if (parse_bool_attr_line(text, "omni:rtx:autoExposure:enabled", &desc->auto_exposure_enabled))
            desc->flags |= NU_EXPOSURE_HAS_AUTO_EXPOSURE;
        if (parse_float_attr_line(text, "omni:rtx:autoExposure:whitePointScale", &desc->white_point_scale))
            desc->flags |= NU_EXPOSURE_HAS_WHITE_POINT;
        if (parse_float_attr_line(text, "rtx:post:tonemap:fNumber", &desc->tonemap_f_number) ||
            parse_float_attr_line(text, "omni:rtx:post:tonemap:fNumber", &desc->tonemap_f_number))
            desc->flags |= NU_EXPOSURE_HAS_TONEMAP_FNUM;
        if (parse_float_attr_line(text, "rtx:post:tonemap:cm2Factor", &desc->tonemap_cm2_factor) ||
            parse_float_attr_line(text, "omni:rtx:post:tonemap:cm2Factor", &desc->tonemap_cm2_factor))
            desc->flags |= NU_EXPOSURE_HAS_TONEMAP_CM2;
        free(text);
    }

    if (parse_float_env2("NUSD_EXPOSURE_FSTOP", "NUSD_OVRTX_EXPOSURE_FSTOP", &desc->exposure_f_stop))
        desc->flags |= NU_EXPOSURE_HAS_FSTOP;
    if (parse_float_env2("NUSD_EXPOSURE_RESPONSIVITY", "NUSD_OVRTX_EXPOSURE_RESPONSIVITY", &desc->exposure_responsivity))
        desc->flags |= NU_EXPOSURE_HAS_RESPONSIVITY;
    if (parse_float_env2("NUSD_EXPOSURE_TIME", "NUSD_OVRTX_EXPOSURE_TIME", &desc->exposure_time))
        desc->flags |= NU_EXPOSURE_HAS_TIME;
    if (parse_float_env2("NUSD_WHITE_POINT_SCALE", "NUSD_OVRTX_WHITE_POINT_SCALE", &desc->white_point_scale))
        desc->flags |= NU_EXPOSURE_HAS_WHITE_POINT;
    if (parse_float_env2("NUSD_TONEMAP_FNUMBER", "NUSD_OVRTX_TONEMAP_FNUMBER", &desc->tonemap_f_number))
        desc->flags |= NU_EXPOSURE_HAS_TONEMAP_FNUM;
    if (parse_float_env2("NUSD_TONEMAP_CM2_FACTOR", "NUSD_OVRTX_TONEMAP_CM2_FACTOR", &desc->tonemap_cm2_factor))
        desc->flags |= NU_EXPOSURE_HAS_TONEMAP_CM2;
    if (parse_bool_env2("NUSD_AUTO_EXPOSURE_ENABLED", "NUSD_OVRTX_AUTO_EXPOSURE_ENABLED", &desc->auto_exposure_enabled))
        desc->flags |= NU_EXPOSURE_HAS_AUTO_EXPOSURE;
}

static float exposure_clamp_scale(float v)
{
    if (!(v > 0.0f) || !isfinite(v)) return 1.0f;
    if (v < 1.0e-4f) return 1.0e-4f;
    if (v > 1.0e4f) return 1.0e4f;
    return v;
}

static void renderer_apply_exposure_desc(NuRenderer* r, const NuExposureDesc* desc)
{
    if (!r) return;
    if (desc) {
        r->exposure_desc = *desc;
    } else {
        memset(&r->exposure_desc, 0, sizeof(r->exposure_desc));
    }

    const NuExposureDesc* d = &r->exposure_desc;
    float scale = 1.0f;
    const uint32_t photo_flags = NU_EXPOSURE_HAS_FSTOP |
                                 NU_EXPOSURE_HAS_RESPONSIVITY |
                                 NU_EXPOSURE_HAS_TIME |
                                 NU_EXPOSURE_HAS_TONEMAP_FNUM |
                                 NU_EXPOSURE_HAS_TONEMAP_CM2;
    if ((d->flags & photo_flags) != 0u) {
        /* The existing shader tonemap was calibrated against the DSX/OVRTX
         * lighting-fixed reference camera (f/21, 1/50 s, responsivity
         * 1.1026709, RenderProduct cm2Factor ~= 0.5).  Treat that as scale
         * 1.0; using the raw OVRTX public f/5 default here double-darkens DSX.
         */
        const float ref_responsivity = 1.1026709f;
        const float ref_time = 0.02f;
        const float ref_f_number = 21.0f;
        const float ref_cm2_factor = 0.5f;
        const float ref_photo = ref_responsivity * ref_time /
                                (ref_f_number * ref_f_number) *
                                ref_cm2_factor;
        float f_number = ref_f_number;
        if ((d->flags & NU_EXPOSURE_HAS_FSTOP) && d->exposure_f_stop > 0.0f)
            f_number = d->exposure_f_stop;
        else if ((d->flags & NU_EXPOSURE_HAS_TONEMAP_FNUM) &&
                 d->tonemap_f_number > 0.0f)
            f_number = d->tonemap_f_number;
        float responsivity = ((d->flags & NU_EXPOSURE_HAS_RESPONSIVITY) &&
                              d->exposure_responsivity > 0.0f)
                           ? d->exposure_responsivity : ref_responsivity;
        float exposure_time = ((d->flags & NU_EXPOSURE_HAS_TIME) &&
                               d->exposure_time > 0.0f)
                            ? d->exposure_time : ref_time;
        float cm2_factor = ((d->flags & NU_EXPOSURE_HAS_TONEMAP_CM2) &&
                            d->tonemap_cm2_factor > 0.0f)
                         ? d->tonemap_cm2_factor : ref_cm2_factor;
        scale = (responsivity * exposure_time /
                 (f_number * f_number) * cm2_factor) / ref_photo;
        if ((d->flags & NU_EXPOSURE_HAS_AUTO_EXPOSURE) &&
            d->auto_exposure_enabled &&
            (d->flags & NU_EXPOSURE_HAS_WHITE_POINT) &&
            d->white_point_scale > 0.0f)
            scale /= d->white_point_scale;
    }

    float direct_scale = 1.0f;
    if (parse_float_env2("NUSD_EXPOSURE_SCALE", "NUSD_OVRTX_EXPOSURE_SCALE",
                         &direct_scale))
        scale *= direct_scale;

    r->tone_exposure_scale = exposure_clamp_scale(scale);
    r->tone_sky_scale = r->tone_exposure_scale;
    r->tone_white_point = ((d->flags & NU_EXPOSURE_HAS_WHITE_POINT) &&
                           d->white_point_scale > 0.0f)
                        ? d->white_point_scale : 1.0f;
    r->tone_flags = d->flags;

    if (r->viewer) {
        viewer_set_tone_mapping(r->viewer, r->tone_exposure_scale,
                                r->tone_sky_scale, r->tone_white_point,
                                r->tone_flags);
    }
}

static void make_lookat(const float eye[3], const float target[3],
                        const float up[3], float out[16]) {
    float f[3] = { target[0]-eye[0], target[1]-eye[1], target[2]-eye[2] };
    float fl = sqrtf(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
    if (fl > 1e-9f) { f[0]/=fl; f[1]/=fl; f[2]/=fl; }
    float s[3] = { f[1]*up[2] - f[2]*up[1],
                   f[2]*up[0] - f[0]*up[2],
                   f[0]*up[1] - f[1]*up[0] };
    float sl = sqrtf(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
    if (sl <= 1e-9f) {
        float alt[3] = {1.0f, 0.0f, 0.0f};
        if (fabsf(up[0]) > 0.5f) { alt[0] = 0.0f; alt[1] = 1.0f; }
        s[0] = f[1]*alt[2] - f[2]*alt[1];
        s[1] = f[2]*alt[0] - f[0]*alt[2];
        s[2] = f[0]*alt[1] - f[1]*alt[0];
        sl = sqrtf(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
    }
    if (sl > 1e-9f) { s[0]/=sl; s[1]/=sl; s[2]/=sl; }
    float u[3] = { s[1]*f[2] - s[2]*f[1],
                   s[2]*f[0] - s[0]*f[2],
                   s[0]*f[1] - s[1]*f[0] };
    /* Row-major bytes: rows are basis vectors, translation in column 3
     * ([3], [7], [11]). The downstream pipeline (mat4_mul, UBO layout)
     * is row-major; mirroring nanousd-vulkan-renderer/src/camera.c for
     * convention parity. The previous column-major layout produced
     * operand-reversed products in mat4_mul, collapsing geometry to a
     * diagonal stripe under any non-identity world transform. */
    out[0] =  s[0];  out[1] =  s[1];  out[2] =  s[2];   out[3]  = -(s[0]*eye[0]+s[1]*eye[1]+s[2]*eye[2]);
    out[4] =  u[0];  out[5] =  u[1];  out[6] =  u[2];   out[7]  = -(u[0]*eye[0]+u[1]*eye[1]+u[2]*eye[2]);
    out[8] = -f[0];  out[9] = -f[1];  out[10] = -f[2];  out[11] =  (f[0]*eye[0]+f[1]*eye[1]+f[2]*eye[2]);
    out[12] = 0.0f;  out[13] = 0.0f;  out[14] =  0.0f;  out[15] = 1.0f;
}

static void make_perspective(float fov_deg, float aspect, float zn, float zf,
                             float projection_shift_x, float projection_shift_y,
                             float out[16]) {
    float f = 1.0f / tanf(fov_deg * (3.14159265358979f / 360.0f));
    memset(out, 0, sizeof(float)*16);
    /* Row-major bytes for OpenGL-style clip space ([-1,+1] depth, +Y up).
     * Vulkan's variant in camera.c uses out[5] = -f for the Y-flip; on
     * OpenGL we keep +f. */
    out[0]  = f / aspect;
    out[5]  = f;
    out[2]  = projection_shift_x;
    out[6]  = projection_shift_y;
    out[10] = (zf + zn) / (zn - zf);
    out[11] = (2.0f * zf * zn) / (zn - zf);
    out[14] = -1.0f;
}

/* ============================================================
 *  Lifecycle
 * ============================================================ */

NuRenderer* nu_renderer_create(const NuRendererConfig* config) {
    NuRenderer* r = (NuRenderer*)calloc(1, sizeof(*r));
    if (!r) return NULL;
    if (config) r->cfg = *config;
    if (r->cfg.width  <= 0) r->cfg.width  = 1920;
    if (r->cfg.height <= 0) r->cfg.height = 1080;
    r->scene_up_axis = 1;
    r->last_mode = NU_RENDER_RASTER;
    r->current_time = NAN;
    renderer_apply_exposure_desc(r, NULL);
    return r;
}

void nu_renderer_destroy(NuRenderer* r) {
    if (!r) return;
    if (r->viewer) viewer_destroy(r->viewer);
    free(r->rgba);
    free(r);
}

static void nu_write_fixed_string(char* dst, size_t dst_size, const char* src) {
    if (!dst || dst_size == 0) return;
    snprintf(dst, dst_size, "%s", src ? src : "");
}

/* Capabilities this OpenGL/GLES backend genuinely implements. This is a
 * raster-only backend: there is no hardware ray tracing, no CUDA-Vulkan
 * interop, no multi-camera tiled path, and no GPU memory / phase-timing /
 * command-cache introspection (those shim entry points return 0 / zeroed
 * structs, so their bits stay clear — advertising them would lie). Honest set:
 *   - RASTER + color AOV (nu_render RASTER, nu_fetch_pixels RGBA8)
 *   - USD file load + scene/mesh queries (nu_load_usd, nu_get_scene_bounds,
 *     nu_get_mesh_count / _name / _transform)
 *   - per-mesh mutation (nu_set_transforms / _colors / _visibility)
 *   - current-time set, dome-light / IBL (nu_load_environment*)
 * Materials/textures/MaterialX are layered on only when the instance opted in
 * (enable_materials drives the viewer's MaterialX-capable PBR path).
 * Cleared deliberately: RT modes, RT-only AOVs (depth/normal/prim-id),
 * load-from-handle (the shim returns NU_ERROR), CUDA interop, async (the shim
 * just reissues a raster frame), timings, gpu-memory, command-cache stats,
 * meshlets, geometry cache, visible window, curves, multi-camera, and
 * analytic lights. */
static uint64_t nu_static_backend_capabilities(void) {
    return
        NU_CAP_RENDER_RASTER |
        NU_CAP_AOV_COLOR |
        NU_CAP_LOAD_USD_FILE |
        NU_CAP_SCENE_BOUNDS |
        NU_CAP_MESH_PATHS |
        NU_CAP_DOME_LIGHT |
        NU_CAP_SET_TRANSFORMS |
        NU_CAP_SET_COLORS |
        NU_CAP_SET_VISIBILITY |
        NU_CAP_SET_CURRENT_TIME;
}

/* contract-first-ffi: pin the compiled NuBackendInfo layout (the ABI checker
 * only compares field text). Any struct change must re-sync every backend. */
_Static_assert(sizeof(NuBackendInfo) == 208,
               "NuBackendInfo layout changed -- re-sync nusd_renderer.h across all backends");

NuResult nu_get_backend_info(NuRenderer* r, NuBackendInfo* out_info) {
    if (!out_info) return NU_ERROR;

    memset(out_info, 0, sizeof(*out_info));
    out_info->version = NU_BACKEND_INFO_VERSION;
    out_info->struct_size = (uint32_t)sizeof(*out_info);
    out_info->capabilities = nu_static_backend_capabilities();
    nu_write_fixed_string(out_info->backend_name,
                          sizeof(out_info->backend_name), "opengl");
    nu_write_fixed_string(out_info->backend_version,
                          sizeof(out_info->backend_version), "OpenGL ES 3.2");
    nu_write_fixed_string(out_info->renderer_name,
                          sizeof(out_info->renderer_name),
                          "nanousd-opengl-renderer");

    if (!r) return NU_OK;

    /* MaterialX PBR + textured rendering is only active when the instance
     * opted in (config.enable_materials → viewer MATERIAL mode). */
    if (r->cfg.enable_materials) {
        out_info->capabilities |= NU_CAP_MATERIALS |
                                  NU_CAP_TEXTURES |
                                  NU_CAP_MATERIALX;
    }

    return NU_OK;
}

/* contract-first-ffi: the public header declares these, so define them -- a
 * declared-but-undefined symbol is a latent dlsym/link crash for any consumer
 * that resolves it. Honest raster stubs: this backend exposes no window handle
 * (NU_CAP_VISIBLE_WINDOW is cleared) and no curve geometry (NU_CAP_CURVES is
 * cleared), so they report "nothing" rather than faking a value. */
void* nu_get_window(NuRenderer* r) { (void)r; return NULL; }
int   nu_get_curve_segment_count(NuRenderer* r) { (void)r; return 0; }

/* ============================================================
 *  Scene loading
 * ============================================================ */

int nu_load_usd(NuRenderer* r, const char* path) {
    if (!r || !path) return NU_ERROR;
    if (r->viewer) { viewer_destroy(r->viewer); r->viewer = NULL; }
    scene_set_load_time(r->current_time);
    r->viewer = viewer_create(path, r->cfg.width, r->cfg.height,
                              0, NULL, 1, r->cfg.enable_materials);
    if (!r->viewer) {
        nu_set_err(r, "viewer_create('%s') failed", path);
        return NU_ERROR;
    }
    r->scene_up_axis = viewer_get_scene_up_axis(r->viewer);
    r->fallback_lights_active = 1;
    viewer_set_overlay_enabled(r->viewer, 0);
    {
        NuExposureDesc exp;
        exposure_desc_from_usd_path(path, &exp);
        renderer_apply_exposure_desc(r, &exp);
    }
    return 1;  /* "number of meshes loaded" — we don't track count, return 1 = loaded. */
}

int nu_load_usd_from_handle(NuRenderer* r, void* stage, const char* label) {
    (void)stage; (void)label;
    if (r) nu_set_err(r, "load_usd_from_handle not supported on opengl backend");
    return NU_ERROR;
}

int nu_load_usd_from_handle_with_dir(NuRenderer* r, void* stage,
                                     const char* label, const char* scene_dir) {
    (void)stage; (void)label; (void)scene_dir;
    if (r) nu_set_err(r, "load_usd_from_handle_with_dir not supported on opengl backend");
    return NU_ERROR;
}

NuResult nu_build_accel(NuRenderer* r) {
    if (!r || !r->viewer) return NU_ERROR_NO_SCENE;
    return NU_OK;
}

/* ============================================================
 *  Configuration
 * ============================================================ */

NuResult nu_get_scene_bounds(NuRenderer* r, float out_min[3], float out_max[3]) {
    if (!r || !out_min || !out_max) return NU_ERROR;
    if (!r->viewer) return NU_ERROR_NO_SCENE;
    if (!viewer_get_scene_bounds(r->viewer, out_min, out_max)) return NU_ERROR;
    return NU_OK;
}

NuResult nu_set_render_size(NuRenderer* r, int width, int height) {
    if (!r || width <= 0 || height <= 0) return NU_ERROR;
    r->cfg.width = width;
    r->cfg.height = height;
    if (r->viewer) viewer_resize(r->viewer, width, height);
    return NU_OK;
}

void nu_get_render_size(NuRenderer* r, int* width, int* height) {
    if (!r) { if (width) *width=0; if (height) *height=0; return; }
    if (width)  *width  = r->cfg.width;
    if (height) *height = r->cfg.height;
}

NuResult nu_set_camera(NuRenderer* r, int cam_id, const NuCameraDesc* cam) {
    if (!r || !cam)        return NU_ERROR;
    if (cam_id != 0)       return NU_ERROR_BAD_ID;
    r->cam = *cam;
    r->cam_set = 1;
    /* Reset to scene-authored upAxis; nu_set_camera_explicit overrides it. */
    r->cam_up_set = 0;
    r->projection_shift_x = 0.0f;
    r->projection_shift_y = 0.0f;
    return NU_OK;
}

NuResult nu_set_camera_explicit(NuRenderer* r,
                                const float eye[3], const float target[3],
                                const float up[3], float fov_degrees,
                                float near_clip, float far_clip) {
    if (!r || !eye || !target || !up) return NU_ERROR;
    r->cam.eye[0]    = eye[0];    r->cam.eye[1]    = eye[1];    r->cam.eye[2]    = eye[2];
    r->cam.target[0] = target[0]; r->cam.target[1] = target[1]; r->cam.target[2] = target[2];
    r->cam.fov_degrees = fov_degrees > 0 ? fov_degrees : 60.0f;
    r->cam.near_clip   = near_clip > 0 ? near_clip : 0.1f;
    r->cam.far_clip    = far_clip > r->cam.near_clip ? far_clip : 10000.0f;
    r->cam_up[0] = up[0]; r->cam_up[1] = up[1]; r->cam_up[2] = up[2];
    r->cam_up_set = 1;
    r->cam_set = 1;
    r->projection_shift_x = 0.0f;
    r->projection_shift_y = 0.0f;
    return NU_OK;
}

NuResult nu_set_camera_explicit_window(NuRenderer* r,
                                       const float eye[3], const float target[3],
                                       const float up[3], float fov_degrees,
                                       float near_clip, float far_clip,
                                       float projection_shift_x,
                                       float projection_shift_y) {
    NuResult rc = nu_set_camera_explicit(r, eye, target, up, fov_degrees, near_clip, far_clip);
    if (rc != NU_OK) return rc;
    r->projection_shift_x = isfinite(projection_shift_x) ? projection_shift_x : 0.0f;
    r->projection_shift_y = isfinite(projection_shift_y) ? projection_shift_y : 0.0f;
    return NU_OK;
}

NuResult nu_get_camera(NuRenderer* r, int cam_id, NuCameraDesc* out_desc) {
    if (!r || !out_desc) return NU_ERROR;
    if (cam_id != 0)     return NU_ERROR_BAD_ID;
    *out_desc = r->cam;
    return NU_OK;
}

NuResult nu_set_exposure(NuRenderer* r, const NuExposureDesc* desc) {
    if (!r || !desc) return NU_ERROR;
    renderer_apply_exposure_desc(r, desc);
    return NU_OK;
}

/* ============================================================
 *  Render
 * ============================================================ */

NuResult nu_render(NuRenderer* r, int cam_id, NuRenderMode mode) {
    if (!r) return NU_ERROR;
    if (!r->viewer) return NU_ERROR_NO_SCENE;
    if (cam_id != 0) return NU_ERROR_BAD_ID;
    if (mode == NU_RENDER_RT || mode == NU_RENDER_SHADOW) {
        nu_set_err(r, "render mode %d unsupported on opengl (no RT)", (int)mode);
        return NU_ERROR_NO_RT;
    }
    if (!r->cam_set) {
        nu_set_err(r, "no camera set; call nu_set_camera first");
        return NU_ERROR;
    }
    if (!ensure_rgba_buffer(r, r->cfg.width, r->cfg.height)) return NU_ERROR;

    float view16[16], proj16[16];
    float default_up[3] = {0.f, 1.f, 0.f};
    if (r->scene_up_axis == 0) {
        default_up[0] = 1.f; default_up[1] = 0.f; default_up[2] = 0.f;
    } else if (r->scene_up_axis == 2) {
        default_up[0] = 0.f; default_up[1] = 0.f; default_up[2] = 1.f;
    }
    const float* up = r->cam_up_set ? r->cam_up : default_up;
    make_lookat(r->cam.eye, r->cam.target, up, view16);
    float aspect = (float)r->cfg.width / (float)r->cfg.height;
    float fov  = r->cam.fov_degrees > 0 ? r->cam.fov_degrees : 60.0f;
    float zn   = r->cam.near_clip   > 0 ? r->cam.near_clip   : 0.1f;
    float zf   = r->cam.far_clip    > zn ? r->cam.far_clip   : 10000.0f;

    /* Legacy workaround for the GLES rasterizer's depth precision. Off by
     * default because it changes the caller's projection and therefore does
     * not match OVRTX camera semantics. */
    const char* clip_env = getenv("NUSD_GL_DEPTH_HEURISTIC");
    if (clip_env && clip_env[0] && clip_env[0] != '0') {
        float bmin[3], bmax[3];
        if (viewer_get_scene_bounds(r->viewer, bmin, bmax)) {
            float cx = (bmin[0]+bmax[0])*0.5f;
            float cy = (bmin[1]+bmax[1])*0.5f;
            float cz = (bmin[2]+bmax[2])*0.5f;
            float dx = r->cam.eye[0]-cx, dy = r->cam.eye[1]-cy, dz = r->cam.eye[2]-cz;
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);
            float ex = bmax[0]-bmin[0], ey = bmax[1]-bmin[1], ez = bmax[2]-bmin[2];
            float diag = sqrtf(ex*ex + ey*ey + ez*ez);
            /* GLES rasterizer's depth precision is sensitive to both
             * ratio and absolute scale. Empirical heuristic that works
             * across small (pbr_materials, ~0.07 unit) and large
             * (Kitchen_set, ~800 unit) scenes:
             *   zn = max(dist * 0.05, diag * 0.01)
             *   zf = dist + diag * 5
             * Override the caller's values entirely — stageView's
             * defaults (near=diag*0.001, far=diag*100, ratio 100k:1)
             * collapse depth precision regardless of scene scale. */
            if (dist > 0 && diag > 0) {
                float zn_t = dist * 0.05f;
                float floor_t = diag * 0.01f;
                if (zn_t < floor_t) zn_t = floor_t;
                float zf_t = dist + diag * 5.0f;
                zn = zn_t;
                zf = zf_t;
            }
        }
    }
    make_perspective(fov, aspect, zn, zf, r->projection_shift_x, r->projection_shift_y, proj16);

    /* viewer_set_render_mode call — only if explicitly opting into MATERIAL.
     * Calling it every frame with the default (RASTER) appears to interfere
     * with the GLES pipeline; leave the mode alone unless we need to flip it. */
    if (r->cfg.enable_materials) {
        viewer_set_render_mode(r->viewer, VIEWER_MODE_MATERIAL);
    }

    int ok = viewer_render_to_rgba(r->viewer, r->cfg.width, r->cfg.height,
                                   view16, proj16, r->cam.eye, r->rgba);
    if (!ok) {
        nu_set_err(r, "viewer_render_to_rgba failed");
        r->last_render_ok = 0;
        return NU_ERROR;
    }
    r->last_mode = mode;
    r->last_render_ok = 1;
    return NU_OK;
}

NuResult nu_fetch_pixels(NuRenderer* r, void* out_pixels, NuPixelFormat format) {
    if (!r || !out_pixels) return NU_ERROR;
    if (!r->last_render_ok || !r->rgba) return NU_ERROR;
    if (format != NU_PIXEL_RGBA8) {
        nu_set_err(r, "format %d unsupported (opengl shim is RGBA8 only)", (int)format);
        return NU_ERROR;
    }
    size_t bytes = (size_t)r->rgba_w * (size_t)r->rgba_h * 4u;
    memcpy(out_pixels, r->rgba, bytes);
    return NU_OK;
}

NuResult nu_save_ppm(NuRenderer* r, const char* path) {
    if (!r || !path || !r->rgba) return NU_ERROR;
    FILE* f = fopen(path, "wb");
    if (!f) return NU_ERROR;
    fprintf(f, "P6\n%d %d\n255\n", r->rgba_w, r->rgba_h);
    for (int y = 0; y < r->rgba_h; ++y) {
        const unsigned char* row = r->rgba + (size_t)y * r->rgba_w * 4u;
        for (int x = 0; x < r->rgba_w; ++x) {
            fputc(row[x*4+0], f);
            fputc(row[x*4+1], f);
            fputc(row[x*4+2], f);
        }
    }
    fclose(f);
    return NU_OK;
}

/* ============================================================
 *  Capability + status
 * ============================================================ */

int nu_rt_available(NuRenderer* r) {
    (void)r;
    return 0;
}

const char* nu_get_last_error(NuRenderer* r) {
    return r ? r->err : "";
}

int nu_get_mesh_count(NuRenderer* r) {
    if (!r || !r->viewer) return 0;
    return viewer_get_mesh_count(r->viewer);
}

int nu_get_mesh_name(NuRenderer* r, int mesh_id, char* out_buf, int buf_cap) {
    if (!r || !r->viewer) return NU_ERROR_NO_SCENE;
    int len = viewer_get_mesh_name(r->viewer, mesh_id, out_buf, buf_cap);
    if (len < 0) {
        nu_set_err(r, "nu_get_mesh_name: invalid mesh id %d", mesh_id);
        return NU_ERROR_BAD_ID;
    }
    return len;
}

NuResult nu_get_mesh_transform(NuRenderer* r, int mesh_id, float out_mat4x4[16]) {
    if (!r || !r->viewer) return NU_ERROR_NO_SCENE;
    if (!viewer_get_mesh_transform(r->viewer, mesh_id, out_mat4x4)) {
        nu_set_err(r, "nu_get_mesh_transform: invalid mesh id %d", mesh_id);
        return NU_ERROR_BAD_ID;
    }
    return NU_OK;
}

/* ============================================================
 *  Stubs — features the canonical Vulkan API exposes that
 *  this shim doesn't implement. Each returns NU_ERROR /
 *  NU_ERROR_NO_RT so callers handle gracefully.
 * ============================================================ */

int nu_add_mesh(NuRenderer* r, const NuMeshDesc* d) {
    (void)d;
    if (r) nu_set_err(r, "nu_add_mesh not supported on opengl backend");
    return NU_ERROR;
}
int nu_add_mesh_instance(NuRenderer* r, int p, const float m[16]) {
    (void)p; (void)m;
    if (r) nu_set_err(r, "nu_add_mesh_instance not supported on opengl backend");
    return NU_ERROR;
}
NuResult nu_remove_mesh(NuRenderer* r, int mid) {
    (void)mid;
    if (r) nu_set_err(r, "nu_remove_mesh not supported on opengl backend");
    return NU_ERROR;
}
void nu_clear_scene(NuRenderer* r) { (void)r; }
NuResult nu_set_transforms(NuRenderer* r, const int* ids, const float* m, int n) {
    if (!r || !r->viewer) return NU_ERROR_NO_SCENE;
    if (!viewer_set_transforms(r->viewer, ids, m, n)) {
        nu_set_err(r, "nu_set_transforms: invalid arguments or mesh id");
        return NU_ERROR_BAD_ID;
    }
    return NU_OK;
}
NuResult nu_set_colors(NuRenderer* r, const int* ids, const float* c, int n) {
    if (!r || !r->viewer) return NU_ERROR_NO_SCENE;
    if (!viewer_set_colors(r->viewer, ids, c, n)) {
        nu_set_err(r, "nu_set_colors: invalid arguments or mesh id");
        return NU_ERROR_BAD_ID;
    }
    return NU_OK;
}
NuResult nu_set_visibility(NuRenderer* r, const int* ids, const int* v, int n) {
    if (!r || !r->viewer) return NU_ERROR_NO_SCENE;
    if (!viewer_set_visibility(r->viewer, ids, v, n)) {
        nu_set_err(r, "nu_set_visibility: invalid arguments or mesh id");
        return NU_ERROR_BAD_ID;
    }
    return NU_OK;
}
NuResult nu_set_current_time(NuRenderer* r, double t) {
    if (!r) return NU_ERROR;
    r->current_time = t;
    return NU_OK;
}
NuResult nu_load_environment(NuRenderer* r, const char* path) {
    if (!r || !path) return NU_ERROR;
    if (!r->viewer) return NU_ERROR_NO_SCENE;
    if (!viewer_load_environment(r->viewer, path)) {
        nu_set_err(r, "viewer_load_environment('%s') failed", path);
        return NU_ERROR;
    }
    return NU_OK;
}
NuResult nu_render_async(NuRenderer* r)                          { return nu_render(r, 0, NU_RENDER_RASTER); }
NuResult nu_fetch_async(NuRenderer* r, void* out)                { return nu_fetch_pixels(r, out, NU_PIXEL_RGBA8); }
NuResult nu_render_tiled(NuRenderer* r, const float* vp_inv, int n, int tw, int th, NuRenderMode m) {
    (void)vp_inv; (void)n; (void)tw; (void)th; (void)m;
    if (r) nu_set_err(r, "nu_render_tiled requires RT (vulkan/metal); not supported on opengl backend");
    return NU_ERROR_NO_RT;
}
void  nu_set_fast_mode(NuRenderer* r, int f)                     { (void)r;(void)f; }
void  nu_set_skip_staging(NuRenderer* r, int s)                  { (void)r;(void)s; }
void  nu_set_per_env_layout(NuRenderer* r, int e)                { (void)r;(void)e; }
void  nu_set_tiled_srgb(NuRenderer* r, int e)                    { (void)r;(void)e; }
void  nu_set_depth_enabled(NuRenderer* r, int e)                 { (void)r;(void)e; }
void  nu_set_segmentation_enabled(NuRenderer* r, int e)          { (void)r;(void)e; }
void  nu_set_normals_enabled(NuRenderer* r, int e)               { (void)r;(void)e; }
uint64_t nu_get_gpu_memory_used(NuRenderer* r) {
    return (r && r->viewer) ? viewer_get_gpu_memory_used(r->viewer) : 0;
}
int   nu_interop_available(NuRenderer* r)                        { (void)r; return 0; }

/* ---- More stubs: raycasting / tiled multi-cam / CUDA interop / GPU mapping
 * ---- (vulkan-only features; opengl backend doesn't support them but the
 * ---- vulkan _bindings.py looks up these symbols unconditionally) ---- */
#define OPENGL_NO_RT_STUB(fn) \
    do { if (r) nu_set_err(r, fn " requires RT (vulkan/metal); not supported on opengl backend"); \
         return NU_ERROR_NO_RT; } while(0)
NuResult nu_cast_rays(NuRenderer* r, const float* o, const float* d, int n, float md, float* od, float* on, float* op) {
    (void)o;(void)d;(void)n;(void)md;(void)od;(void)on;(void)op; OPENGL_NO_RT_STUB("nu_cast_rays");
}
NuResult nu_cast_rays_async(NuRenderer* r, const float* o, const float* d, int n, float md) {
    (void)o;(void)d;(void)n;(void)md; OPENGL_NO_RT_STUB("nu_cast_rays_async");
}
NuResult nu_cast_rays_gpu(NuRenderer* r, int n, float md) {
    (void)n;(void)md; OPENGL_NO_RT_STUB("nu_cast_rays_gpu");
}
NuResult nu_cast_rays_wait(NuRenderer* r, float* od, float* on, float* op) {
    (void)od;(void)on;(void)op; OPENGL_NO_RT_STUB("nu_cast_rays_wait");
}
NuResult nu_cast_rays_wait_fence(NuRenderer* r) {
    OPENGL_NO_RT_STUB("nu_cast_rays_wait_fence");
}
NuResult nu_fetch_depth_tiled(NuRenderer* r, float* o, int n, int tw, int th) {
    (void)o;(void)n;(void)tw;(void)th; OPENGL_NO_RT_STUB("nu_fetch_depth_tiled");
}
NuResult nu_fetch_normals_tiled(NuRenderer* r, float* o, int n, int tw, int th) {
    (void)o;(void)n;(void)tw;(void)th; OPENGL_NO_RT_STUB("nu_fetch_normals_tiled");
}
NuResult nu_fetch_pixels_cuda(NuRenderer* r, int* fd, uint64_t* sz, int* w, int* h, int* f) {
    (void)fd;(void)sz;(void)w;(void)h;(void)f;
    if (r) nu_set_err(r, "nu_fetch_pixels_cuda (CUDA-Vulkan interop) not supported on opengl backend");
    return NU_ERROR;
}
NuResult nu_fetch_pixels_tiled(NuRenderer* r, void* o, int n, int tw, int th) {
    (void)o;(void)n;(void)tw;(void)th; OPENGL_NO_RT_STUB("nu_fetch_pixels_tiled");
}
NuResult nu_fetch_segmentation_tiled(NuRenderer* r, uint32_t* o, int n, int tw, int th) {
    (void)o;(void)n;(void)tw;(void)th; OPENGL_NO_RT_STUB("nu_fetch_segmentation_tiled");
}
void     nu_get_cmd_cache_stats(NuRenderer* r, uint64_t* a, uint64_t* b, uint64_t* c, uint64_t* d) {
    (void)r; if(a)*a=0; if(b)*b=0; if(c)*c=0; if(d)*d=0;
}
NuResult nu_get_cuda_interop_info(NuRenderer* r, NuCudaInteropInfo* o, int n, int tw, int th) {
    (void)o; (void)n; (void)tw; (void)th;
    if (r) nu_set_err(r, "nu_get_cuda_interop_info (CUDA-Vulkan interop) not supported on opengl backend");
    return NU_ERROR;
}
int      nu_get_interop_prev_idx(NuRenderer* r)                  { (void)r; return -1; }
int      nu_get_interop_read_idx(NuRenderer* r)                  { (void)r; return -1; }
int      nu_get_last_tiled_slot(NuRenderer* r)                   { (void)r; return -1; }
NuResult nu_get_phase_timings_ms(NuRenderer* r, NuPhaseTimings* o) { (void)r; if(o) memset(o, 0, sizeof(*o)); return NU_OK; }
NuResult nu_load_environment_intensity(NuRenderer* r, const char* p, float i) {
    if (!r || !p) return NU_ERROR;
    if (!r->viewer) return NU_ERROR_NO_SCENE;
    if (!viewer_load_environment_intensity(r->viewer, p, i)) {
        nu_set_err(r, "viewer_load_environment_intensity('%s', %.3f) failed", p, i);
        return NU_ERROR;
    }
    return NU_OK;
}
void*    nu_map_pixels_gpu(NuRenderer* r)                        { (void)r; return NULL; }
const void* nu_map_tiled_pixels_raw(NuRenderer* r, int n, int tw, int th, int* ow, int* oh) {
    (void)r;(void)n;(void)tw;(void)th; if(ow)*ow=0; if(oh)*oh=0; return NULL;
}
const void* nu_map_tiled_pixels_raw_slot(NuRenderer* r, int n, int tw, int th, int s, int* ow, int* oh) {
    (void)r;(void)n;(void)tw;(void)th;(void)s; if(ow)*ow=0; if(oh)*oh=0; return NULL;
}
NuResult nu_raycast_get_interop_info(NuRenderer* r, int n, NuRaycastInteropInfo* o) {
    (void)n; (void)o;
    if (r) nu_set_err(r, "nu_raycast_get_interop_info requires RT (vulkan/metal); not supported on opengl backend");
    return NU_ERROR;
}
void     nu_unmap_pixels_gpu(NuRenderer* r)                      { (void)r; }
NuResult nu_wait_previous_tiled_complete(NuRenderer* r)          { (void)r; return NU_OK; }
NuResult nu_wait_tiled_complete(NuRenderer* r)                   { (void)r; return NU_OK; }
