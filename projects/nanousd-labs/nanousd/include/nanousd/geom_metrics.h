// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nanousd/stage.h"

namespace nanousd {

// ============================================================
// Geometry domain layer metadata (metersPerUnit, upAxis)
//
// These are convenience accessors for layer metadata fields
// defined by the geometry schema domain, kept separate from
// core to avoid coupling domain-specific knowledge into the
// core Spec/Stage classes.
//
// The underlying data is stored as generic fields on the layer
// spec and parsed by the generic metadata fallback path.
// ============================================================

namespace GeomTokens {
    inline const Token metersPerUnit{"metersPerUnit"};
    inline const Token upAxis{"upAxis"};
}

namespace detail {
    // Generic metadata may be parsed as Int or Double depending on literal form.
    // This helper coerces either to double.
    inline double GetFieldAsDouble(const Fields& fields, const Token& name, double fallback) {
        auto* v = fields.Get(name);
        if (!v) return fallback;
        if (auto* d = v->Get<Double>()) return *d;
        if (auto* i = v->Get<Int>()) return static_cast<double>(*i);
        if (auto* f = v->Get<Float>()) return static_cast<double>(*f);
        return fallback;
    }
}

// --- metersPerUnit ---

inline double GetMetersPerUnit(const Layer& layer) {
    return detail::GetFieldAsDouble(
        layer.GetLayerSpec().GetFields(), GeomTokens::metersPerUnit, 0.01);
}

inline bool HasMetersPerUnit(const Layer& layer) {
    return layer.GetLayerSpec().HasField(GeomTokens::metersPerUnit);
}

inline void SetMetersPerUnit(Layer& layer, double v) {
    layer.GetLayerSpec().SetField(GeomTokens::metersPerUnit, Value(v));
}

inline double GetMetersPerUnit(const Stage& stage) {
    return detail::GetFieldAsDouble(
        stage.GetComposedLayerSpec().GetFields(), GeomTokens::metersPerUnit, 0.01);
}

inline bool HasMetersPerUnit(const Stage& stage) {
    return stage.GetComposedLayerSpec().HasField(GeomTokens::metersPerUnit);
}

// --- upAxis ---

inline Token GetUpAxis(const Layer& layer) {
    auto val = layer.GetLayerSpec().GetFields().GetAs<String>(GeomTokens::upAxis);
    return val ? Token(*val) : Token("Y");
}

inline bool HasUpAxis(const Layer& layer) {
    return layer.GetLayerSpec().HasField(GeomTokens::upAxis);
}

inline void SetUpAxis(Layer& layer, const Token& t) {
    layer.GetLayerSpec().SetField(GeomTokens::upAxis, Value(t.GetString()));
}

inline Token GetUpAxis(const Stage& stage) {
    auto val = stage.GetComposedLayerSpec().GetFields().GetAs<String>(GeomTokens::upAxis);
    return val ? Token(*val) : Token("Y");
}

inline bool HasUpAxis(const Stage& stage) {
    return stage.GetComposedLayerSpec().HasField(GeomTokens::upAxis);
}

} // namespace nanousd
