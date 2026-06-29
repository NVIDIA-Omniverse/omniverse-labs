// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"
#include "value.h"

#include <optional>
#include <vector>

namespace nanousd {

// ============================================================
// Spline evaluation (spec §7.4.2.4, §12.5.3)
//
// Public compute primitives for authored cubic splines. Separated
// into their own translation unit following the xform.h pattern so
// tools and tests can drive evaluation without constructing a Stage.
// `UsdAttribute::Get` consumes the same helpers internally to fill
// the spline slot in the per-opinion value-resolution walk.
//
// Scope: scalar numeric splines (half/float/double — all stored as
// double internally), Bezier and Hermite curve types, the four
// per-segment interpolation modes (none / held / linear / curve),
// dual-value knots (value / preValue), extrapolation modes including
// loop repeat/reset/oscillate, inner-loop replay (§12.5.3.4), and
// Bezier anti-regression correction (§12.5.3.5).
// ============================================================

// Evaluate `spline` at `time`. Returns the sampled value, or
// nullopt when no value is defined at that time:
//   - empty spline (no knots)
//   - query falls into a segment whose nextInterpolationMode is
//     `none` (value block per §7.4.2.4.2)
//   - query is before the first knot and preExtrapolationMode is
//     `none`, or after the last knot and postExtrapolationMode is
//     `none`
//
// The returned Value always holds a Double. Callers that need
// float/half must narrow at the boundary.
NANOUSD_CORE_API std::optional<Value> EvaluateSpline(const Spline& spline,
                                                     double time);

// Bake a spline to a timeSamples Dictionary at the provided sample
// times. Entries for which EvaluateSpline returns nullopt are
// omitted — the resulting dict faithfully represents "no value" at
// those times. Keys are the stringified sample times (matching the
// storage convention used elsewhere for timeSamples).
NANOUSD_CORE_API Dictionary BakeSplineToTimeSamples(
        const Spline& spline,
        const std::vector<double>& sampleTimes);

} // namespace nanousd
