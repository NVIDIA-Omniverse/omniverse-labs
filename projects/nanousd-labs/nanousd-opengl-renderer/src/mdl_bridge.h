// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NUSD_MDL_BRIDGE_H
#define NUSD_MDL_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

#define NUSD_MDL_MAX_DECODED_TEXTURES 32

typedef struct NusdMdlDecodedTexture {
    char name_hint[128];
    char file_path[1024];
    char db_name[512];
} NusdMdlDecodedTexture;

typedef struct NusdMdlDecoded {
    int has_base_color;
    int has_emissive_color;
    int has_metallic;
    int has_roughness;
    int has_opacity;
    int has_ior;
    int has_clearcoat;
    int has_clearcoat_roughness;
    int has_normal_scale;
    int has_transmission_color;
    int has_transmission_weight;
    int has_transmission_ior;
    int has_specular_color;
    int has_specular_workflow;

    float base_color[4];
    float emissive_color[4];
    float metallic;
    float roughness;
    float opacity;
    float ior;
    float clearcoat;
    float clearcoat_roughness;
    float normal_scale;
    float transmission_color[4];
    float transmission_weight;
    float transmission_ior;
    float specular_color[4];
    int use_specular_workflow;
    int texture_count;
    NusdMdlDecodedTexture textures[NUSD_MDL_MAX_DECODED_TEXTURES];
} NusdMdlDecoded;

enum {
    NUSD_MDL_INPUT_FLOAT = 1,
    NUSD_MDL_INPUT_COLOR = 2,
    NUSD_MDL_INPUT_BOOL  = 3,
    NUSD_MDL_INPUT_INT   = 4,
};

typedef struct NusdMdlInput {
    const char* name;
    int kind;
    float values[4];
    int int_value;
} NusdMdlInput;

int nusd_mdl_bridge_available(void);
int nusd_mdl_bridge_decode(const char* mdl_source_asset,
                           const char* subidentifier,
                           const char* scene_dir,
                           NusdMdlDecoded* out);
int nusd_mdl_bridge_decode_with_inputs(const char* mdl_source_asset,
                                       const char* subidentifier,
                                       const char* scene_dir,
                                       const NusdMdlInput* inputs,
                                       int input_count,
                                       NusdMdlDecoded* out);

#ifdef __cplusplus
}
#endif

#endif /* NUSD_MDL_BRIDGE_H */
