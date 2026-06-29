// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nanousd/api.h"
#include "nanousd/stage.h"

namespace nanousd {

// ============================================================
// XformOp transform computation (geometry spec: xformable.md)
//
// Computes the composed local 4x4 transform from a prim's
// xformOpOrder and associated xformOp attributes.
//
// This is a geometry-domain utility, kept separate from core
// to match the pattern of geom_metrics.h / physics_metrics.h.
// ============================================================

// Compute the composed local transform for a prim from its xformOp stack.
// Returns identity if the prim has no xformOpOrder.
// If resetXformStack is non-null, sets it to true if !resetXformStack! is present.
NANOUSD_CORE_API GfMatrix4d ComputeLocalTransform(const UsdPrim& prim,
                                  UsdTimeCode time = UsdTimeCode::Default(),
                                  bool* resetXformStack = nullptr);

// Check if the prim's xformOpOrder begins with !resetXformStack!
NANOUSD_CORE_API bool HasResetXformStack(const UsdPrim& prim);

} // namespace nanousd
