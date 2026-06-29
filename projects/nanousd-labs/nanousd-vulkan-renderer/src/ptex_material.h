// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NUSD_PTEX_MATERIAL_H
#define NUSD_PTEX_MATERIAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Sample an authored Ptex color map into one packed RGBA8 color per
 * triangulated output corner: three colors for each output triangle, in the
 * same order as indices[]. face_counts/face_vertex_index_count describe the
 * original USD face topology before fan triangulation. Returns the number of
 * colors written, or 0 when Ptex support is disabled/unavailable. */
int nusd_ptex_sample_face_tri_colors(const char* path,
                                     const int* face_counts,
                                     int face_count,
                                     int face_vertex_index_count,
                                     uint32_t* out_rgba8,
                                     int out_color_count);

/* Sample a representative average color from an authored Ptex map. This is
 * used for primitives without a face id domain, such as BasisCurves. */
int nusd_ptex_sample_average_color(const char* path,
                                   float out_rgb[3],
                                   int max_samples);

#ifdef __cplusplus
}
#endif

#endif /* NUSD_PTEX_MATERIAL_H */
