// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nusd_renderer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    NuRendererConfig cfg = {0};
    cfg.width = 64;
    cfg.height = 64;
    cfg.enable_rt = 1;

    NuRenderer* r = nu_renderer_create(&cfg);
    if (!r) {
        fprintf(stderr, "nu_renderer_create failed\n");
        return 1;
    }

    const float eye[3] = {0.0f, 0.0f, -3.0f};
    const float target[3] = {0.0f, 0.0f, 0.0f};
    const float up[3] = {0.0f, 1.0f, 0.0f};
    if (nu_set_camera_explicit(r, eye, target, up, 45.0f, 0.01f, 100.0f) != NU_OK) {
        fprintf(stderr, "nu_set_camera_explicit failed\n");
        nu_renderer_destroy(r);
        return 1;
    }

    const float positions[3] = {0.0f, 0.0f, 0.0f};
    const float scales[3] = {0.5f, 0.5f, 0.5f};
    const float orientations[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    const float opacities[1] = {0.95f};
    const float sh[3] = {
        (1.0f - 0.5f) / 0.28209479177387814f,
        (0.2f - 0.5f) / 0.28209479177387814f,
        (0.1f - 0.5f) / 0.28209479177387814f,
    };

    NuGsDesc desc;
    desc.positions = positions;
    desc.scales = scales;
    desc.orientations = orientations;
    desc.opacities = opacities;
    desc.sh_coefficients = sh;
    desc.sh_degree = 0;
    desc.particle_count = 1;
    desc.prim_xform = NULL;

    if (nu_gs_set_particles(r, &desc) != NU_OK ||
        nu_gs_set_k(r, 8) != NU_OK ||
        nu_gs_render(r, 0) != NU_OK) {
        fprintf(stderr, "Gaussian render failed: %s\n", nu_get_last_error(r));
        nu_renderer_destroy(r);
        return 1;
    }

    uint8_t* pixels = (uint8_t*)calloc((size_t)cfg.width * (size_t)cfg.height * 4u, 1u);
    float* depth = (float*)calloc((size_t)cfg.width * (size_t)cfg.height, sizeof(float));
    if (!pixels || !depth) {
        free(pixels);
        free(depth);
        nu_renderer_destroy(r);
        return 1;
    }

    if (nu_fetch_pixels(r, pixels, NU_PIXEL_RGBA8) != NU_OK ||
        nu_gs_fetch_depth(r, depth) != NU_OK) {
        fprintf(stderr, "Gaussian readback failed\n");
        free(pixels);
        free(depth);
        nu_renderer_destroy(r);
        return 1;
    }

    uint64_t rgb_sum = 0;
    int depth_count = 0;
    for (int i = 0; i < cfg.width * cfg.height; i++) {
        rgb_sum += pixels[i * 4 + 0];
        rgb_sum += pixels[i * 4 + 1];
        rgb_sum += pixels[i * 4 + 2];
        if (depth[i] > 0.0f) depth_count++;
    }

    free(pixels);
    free(depth);
    nu_renderer_destroy(r);

    if (rgb_sum == 0 || depth_count == 0) {
        fprintf(stderr, "Gaussian render was blank (rgb_sum=%llu, depth_count=%d)\n",
                (unsigned long long)rgb_sum, depth_count);
        return 1;
    }

    printf("Gaussian render smoke OK (rgb_sum=%llu, depth_count=%d)\n",
           (unsigned long long)rgb_sum, depth_count);
    return 0;
}
