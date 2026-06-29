// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanousd/spline.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace nanousd {

namespace {

// ------------------------------------------------------------
// Segment math (§7.4.2.4.2, §12.5.3)
// ------------------------------------------------------------
//
// Each segment runs from knots[i] (left) to knots[i+1] (right). The
// segment's interpolation mode lives on knots[i].nextInterpolationMode:
//   None   — value block across the segment; Evaluate returns nullopt
//   Held   — constant knots[i].value across the segment
//   Linear — lerp from knots[i].value at t_i to knots[i+1].preValue at t_{i+1}
//   Curve  — cubic Bezier or Hermite (per spline.curveType) with the
//            two knots' tangents
//
// `preValue` is the left-limit value at a knot (§12.5.3.3). The right
// endpoint of a segment therefore uses knots[i+1].preValue — this is
// what preserves dual-valued (discontinuous) behaviour at a knot.

double BezierScalar(double u, double p0, double p1, double p2, double p3) {
    const double um = 1.0 - u;
    return um*um*um*p0 + 3.0*um*um*u*p1 + 3.0*um*u*u*p2 + u*u*u*p3;
}

double SolveBezierU(double t, double t0, double t1, double t2, double t3) {
    if (t <= t0) return 0.0;
    if (t >= t3) return 1.0;
    double lo = 0.0, hi = 1.0;
    for (int i = 0; i < 64; ++i) {
        double mid = 0.5 * (lo + hi);
        double tm = BezierScalar(mid, t0, t1, t2, t3);
        if (tm < t) lo = mid;
        else        hi = mid;
        if (hi - lo < 1e-12) break;
    }
    return 0.5 * (lo + hi);
}

// Cubic Bezier segment with anti-regression (§12.5.3.5). If the
// combined tangent widths would push the curve to double back in
// time (preWidth + postWidth > dt), scale both proportionally so
// their sum equals dt — tangent directions are preserved, only
// lengths shrink, which keeps T(u) monotonic without changing the
// author's intent more than necessary.
double EvaluateBezierSegment(const SplineKnot& a, const SplineKnot& b,
                              double time) {
    const double dt = b.time - a.time;
    double postW = a.postTangentWidth;
    double preW  = b.preTangentWidth;
    if (dt > 0.0 && postW + preW > dt) {
        const double scale = dt / (postW + preW);
        postW *= scale;
        preW  *= scale;
    }

    const double t0 = a.time;
    const double t3 = b.time;
    const double t1 = a.time + postW;
    const double t2 = b.time - preW;

    const double v0 = a.value;
    const double v3 = b.preValue;
    const double v1 = v0 + a.postTangentSlope * postW;
    const double v2 = v3 - b.preTangentSlope  * preW;

    double u = SolveBezierU(time, t0, t1, t2, t3);
    return BezierScalar(u, v0, v1, v2, v3);
}

// --- Cubic Hermite --------------------------------------------------
//
// Parameterised directly by u = (t - t_i) / (t_{i+1} - t_i). Tangent
// basis per the standard cubic Hermite form; slopes are dv/dt, so
// value-space tangents are Δt * slope.
double EvaluateHermiteSegment(const SplineKnot& a, const SplineKnot& b,
                               double time) {
    const double dt = b.time - a.time;
    if (dt == 0.0) return a.value;  // degenerate (shouldn't happen)
    const double u  = (time - a.time) / dt;
    const double u2 = u * u;
    const double u3 = u2 * u;
    const double h00 =  2.0 * u3 - 3.0 * u2 + 1.0;
    const double h10 =        u3 - 2.0 * u2 + u;
    const double h01 = -2.0 * u3 + 3.0 * u2;
    const double h11 =        u3 -       u2;
    return h00 * a.value
         + h10 * dt * a.postTangentSlope
         + h01 * b.preValue
         + h11 * dt * b.preTangentSlope;
}

// ------------------------------------------------------------
// In-range evaluation: `time` is assumed to lie in
// [knots.front().time, knots.back().time]. Returns nullopt only for
// `none` interp segments.
// ------------------------------------------------------------

std::optional<double> EvaluateInRange(const Spline& spline, double time) {
    const auto& knots = spline.knots;

    // Largest index i with knots[i].time <= time. Scans all knots so
    // an exact-on-last-knot query lands on the final knot itself.
    size_t i = 0;
    for (size_t k = 0; k < knots.size(); ++k) {
        if (knots[k].time <= time) i = k;
    }
    // Right-continuous at knots.
    if (time == knots[i].time) return knots[i].value;
    if (i + 1 >= knots.size()) return knots[i].value;

    const SplineKnot& a = knots[i];
    const SplineKnot& b = knots[i + 1];
    switch (a.nextInterpolationMode) {
    case InterpolationMode::None:
        return std::nullopt;
    case InterpolationMode::Held:
        return a.value;
    case InterpolationMode::Linear: {
        const double dt = b.time - a.time;
        if (dt == 0.0) return a.value;
        const double u = (time - a.time) / dt;
        return a.value + (b.preValue - a.value) * u;
    }
    case InterpolationMode::Curve:
        if (spline.curveType == CurveType::Hermite) {
            return EvaluateHermiteSegment(a, b, time);
        }
        return EvaluateBezierSegment(a, b, time);
    }
    return std::nullopt;
}

// ------------------------------------------------------------
// Inner loops (§12.5.3.4.2) — fold `time` into the proto range if it
// falls in the numPreLoops copies before protoStart or the
// numPostLoops copies after protoEnd. valueOffset accumulates per
// copy for seamless joins (or whatever the author specified).
// ------------------------------------------------------------

struct InnerFold {
    double effectiveTime;  // remapped time inside [protoStart, protoEnd)
    double valueDelta;     // add to the evaluated value
};

std::optional<InnerFold> TryFoldInnerLoop(const Spline& s, double time) {
    const auto& lp = s.loopParameters;
    const double period = lp.protoEnd - lp.protoStart;
    if (period <= 0.0) return std::nullopt;
    if (lp.numPreLoops <= 0 && lp.numPostLoops <= 0) return std::nullopt;

    if (lp.numPostLoops > 0 && time > lp.protoEnd) {
        const double innerEnd = lp.protoEnd + lp.numPostLoops * period;
        if (time <= innerEnd) {
            const double tSince = time - lp.protoEnd;
            int n = static_cast<int>(std::floor(tSince / period));
            // Guard the exact-on-end boundary: stay in the last copy.
            if (n >= lp.numPostLoops) n = lp.numPostLoops - 1;
            const double frac = tSince - n * period;
            return InnerFold{lp.protoStart + frac,
                             static_cast<double>(n + 1) * lp.valueOffset};
        }
    }
    if (lp.numPreLoops > 0 && time < lp.protoStart) {
        const double innerStart = lp.protoStart - lp.numPreLoops * period;
        if (time >= innerStart) {
            // tSince is negative in the pre-loop region.
            const double tSince = time - lp.protoStart;
            int n = static_cast<int>(std::floor(tSince / period));  // <= -1
            if (n < -lp.numPreLoops) n = -lp.numPreLoops;
            const double frac = tSince - n * period;  // in [0, period)
            return InnerFold{lp.protoStart + frac,
                             static_cast<double>(n) * lp.valueOffset};
        }
    }
    return std::nullopt;
}

// ------------------------------------------------------------
// Outer extrapolation (§12.5.3.4.1)
// ------------------------------------------------------------

// Fold `time` into the authored period [tFirst, tLast] for LoopRepeat
// / LoopReset / LoopOscillate. Returns the evaluated in-range value
// plus the per-period value delta. Mirrors (oscillate) every other
// copy so ends meet seamlessly without value drift.
std::optional<double> EvaluateOuterLoop(const Spline& s, double time,
                                         ExtrapolationMode mode) {
    const auto& knots = s.knots;
    const double tFirst = knots.front().time;
    const double tLast  = knots.back().time;
    const double period = tLast - tFirst;
    if (period <= 0.0) return std::nullopt;

    const double tSince = time - tFirst;
    int n = static_cast<int>(std::floor(tSince / period));
    double frac = tSince - n * period;  // in [0, period)
    double effectiveTime = tFirst + frac;

    if (mode == ExtrapolationMode::LoopOscillate) {
        // Every other copy flips the curve in time. std::abs of n
        // because pre-loops (n<0) mirror on the same odd parity.
        const bool mirror = (std::abs(n) % 2) == 1;
        if (mirror) effectiveTime = tLast - (effectiveTime - tFirst);
    }

    auto v = EvaluateInRange(s, effectiveTime);
    if (!v) return std::nullopt;

    if (mode == ExtrapolationMode::LoopRepeat) {
        // Progressive: each copy adds (lastKnot.value - firstKnot.value)
        // so joins are value-continuous.
        *v += static_cast<double>(n) * (knots.back().value - knots.front().value);
    }
    return v;
}

std::optional<double> ExtrapolatePre(const Spline& s, double time) {
    const SplineKnot& edge = s.knots.front();
    switch (s.preExtrapolationMode) {
    case ExtrapolationMode::None:
        return std::nullopt;
    case ExtrapolationMode::Held:
        // Approaching from below a knot means the preValue limit.
        return edge.preValue;
    case ExtrapolationMode::Linear:
        return edge.preValue + edge.preTangentSlope * (time - edge.time);
    case ExtrapolationMode::Sloped:
        return edge.preValue + s.preExtrapolationSlope * (time - edge.time);
    case ExtrapolationMode::LoopRepeat:
    case ExtrapolationMode::LoopReset:
    case ExtrapolationMode::LoopOscillate:
        return EvaluateOuterLoop(s, time, s.preExtrapolationMode);
    }
    return std::nullopt;
}

std::optional<double> ExtrapolatePost(const Spline& s, double time) {
    const SplineKnot& edge = s.knots.back();
    switch (s.postExtrapolationMode) {
    case ExtrapolationMode::None:
        return std::nullopt;
    case ExtrapolationMode::Held:
        return edge.value;
    case ExtrapolationMode::Linear:
        return edge.value + edge.postTangentSlope * (time - edge.time);
    case ExtrapolationMode::Sloped:
        return edge.value + s.postExtrapolationSlope * (time - edge.time);
    case ExtrapolationMode::LoopRepeat:
    case ExtrapolationMode::LoopReset:
    case ExtrapolationMode::LoopOscillate:
        return EvaluateOuterLoop(s, time, s.postExtrapolationMode);
    }
    return std::nullopt;
}

} // anonymous namespace

std::optional<Value> EvaluateSpline(const Spline& spline, double time) {
    if (spline.knots.empty()) return std::nullopt;

    // §12.5.3.4.2 inner loops: if the query lies in one of the
    // authored pre/post replicas of the proto range, fold into the
    // proto and accumulate valueOffset. Inner-loop folding happens
    // before outer extrapolation — per spec ("Inner loops only
    // support Continue extrapolation method"), outer extrapolation
    // beyond the inner-loop region uses the ordinary outer modes,
    // just measured from the authored edges.
    double effectiveTime = time;
    double valueDelta = 0.0;
    if (auto fold = TryFoldInnerLoop(spline, time)) {
        effectiveTime = fold->effectiveTime;
        valueDelta    = fold->valueDelta;
    }

    const auto& knots = spline.knots;
    const double tFirst = knots.front().time;
    const double tLast  = knots.back().time;

    if (effectiveTime < tFirst) {
        auto v = ExtrapolatePre(spline, effectiveTime);
        if (!v) return std::nullopt;
        return Value(*v + valueDelta);
    }
    if (effectiveTime > tLast) {
        auto v = ExtrapolatePost(spline, effectiveTime);
        if (!v) return std::nullopt;
        return Value(*v + valueDelta);
    }

    auto v = EvaluateInRange(spline, effectiveTime);
    if (!v) return std::nullopt;
    return Value(*v + valueDelta);
}

Dictionary BakeSplineToTimeSamples(const Spline& spline,
                                    const std::vector<double>& sampleTimes) {
    Dictionary out;
    char buf[64];
    for (double t : sampleTimes) {
        auto v = EvaluateSpline(spline, t);
        if (!v) continue;
        std::snprintf(buf, sizeof(buf), "%s", std::to_string(t).c_str());
        out[buf] = std::move(*v);
    }
    return out;
}

} // namespace nanousd
