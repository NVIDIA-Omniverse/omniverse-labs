/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "tst_pvdcorecase.h"
#include "tst_frame_export.h"
#include "tst_pvdcore_ue4_shooter.h"
#include "tst_pvdcore_apex_destruction.h"
#include "tst_pvd_loader.h"
#include "tst_native_loading.h"
#include "tst_ovd_input_validation.h"
#include "tst_ovd_loader.h"
#include "tst_ovd_multiscene.h"
#include "tst_ovd_resample.h"
#include "tst_ovd_live_resample.h"
#include "tst_ovd_memory_recording.h"
#include "tst_security_regressions.h"
#include "tst_renderer_security.h"
#include "tst_event_serializer_security.h"
#include "tst_object_model_allocation_security.h"

#include <gtest/gtest.h>

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
