// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * material_stub.cpp — No-op MaterialX stub for builds without -DNUSD_ENABLE_MATERIALS=ON.
 *
 * Phase 7 will swap this for material.cpp with MaterialX GenMsl backend.
 * Until then, materials_load() returns NULL and the renderer falls back to
 * per-mesh display colors.
 */

#include "material.h"
#include <cstdlib>

extern "C" {

MaterialCollection* materials_load(void* /*stage*/, const char* /*scene_dir*/)
{
    return nullptr;
}

int materials_find_binding(MaterialCollection* /*mc*/, void* /*stage*/, void* /*mesh_prim*/)
{
    return -1;
}

void materials_free(MaterialCollection* mc)
{
    free(mc);
}

int materialx_init(void)
{
    return 1;
}

void materialx_shutdown(void)
{
}

int materials_backend_available(void)
{
    return 0;
}

}
