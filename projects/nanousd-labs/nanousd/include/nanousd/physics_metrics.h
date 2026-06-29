// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nanousd/geom_metrics.h"

namespace nanousd {

// ============================================================
// Physics domain layer metadata (kilogramsPerUnit)
//
// These are convenience accessors for layer metadata fields
// defined by the physics schema domain, kept separate from
// core and geometry to avoid coupling domain-specific knowledge
// into the core Spec/Stage classes.
//
// The underlying data is stored as generic fields on the layer
// spec and parsed by the generic metadata fallback path.
// ============================================================

namespace PhysicsTokens {
    inline const Token kilogramsPerUnit{"kilogramsPerUnit"};
}

// --- kilogramsPerUnit ---

inline double GetKilogramsPerUnit(const Layer& layer) {
    return detail::GetFieldAsDouble(
        layer.GetLayerSpec().GetFields(), PhysicsTokens::kilogramsPerUnit, 1.0);
}

inline bool HasKilogramsPerUnit(const Layer& layer) {
    return layer.GetLayerSpec().HasField(PhysicsTokens::kilogramsPerUnit);
}

inline void SetKilogramsPerUnit(Layer& layer, double v) {
    layer.GetLayerSpec().SetField(PhysicsTokens::kilogramsPerUnit, Value(v));
}

inline double GetKilogramsPerUnit(const Stage& stage) {
    return detail::GetFieldAsDouble(
        stage.GetComposedLayerSpec().GetFields(), PhysicsTokens::kilogramsPerUnit, 1.0);
}

inline bool HasKilogramsPerUnit(const Stage& stage) {
    return stage.GetComposedLayerSpec().HasField(PhysicsTokens::kilogramsPerUnit);
}

} // namespace nanousd
