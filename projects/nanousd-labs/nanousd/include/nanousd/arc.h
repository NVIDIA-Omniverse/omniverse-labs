// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace nanousd {

// Composition arc kinds. Used in two places:
//
//  - OpinionEntry::arcType to record which arc introduced the opinion, so
//    later compose phases can enforce the LIVRPS strength ordering from
//    spec §10.5.
//
//  - Diagnostic::arcType to tell consumers of a composition diagnostic
//    which arc was being processed when the issue occurred.
//
// Sublayer is included even though it isn't one of the LIVRPS arcs — a
// sublayer's opinions compose into a prim's Local opinions per §10.3, but
// diagnostics still want to distinguish "failed to load a sublayer" from
// "failed to load a reference" for clarity. None covers diagnostics that
// aren't tied to any specific arc.
//
// Integer values for Sublayer / Reference / Payload / None are pinned for
// ABI: the public C API exposes Diagnostic::arc_type as an int, and
// consumers (see compliance tests and benchmark_stage_load) rely on the
// encoding 0=sublayer, 1=reference, 2=payload, 3=none.
enum class ArcType : uint8_t {
    Sublayer   = 0,
    Reference  = 1,
    Payload    = 2,
    None       = 3,
    Local      = 4,
    Inherits   = 5,
    Variant    = 6,
    Specialize = 7,
    Relocate   = 8,
};

// LIVRPS strength rank per spec §10.5 — lower rank is stronger.
// Used to stable-sort PrimIndex::entries so composition queries see
// opinions in strongest-to-weakest order. Sublayer shares Local's rank
// (sublayer opinions are part of the Local layer stack); None is used
// by diagnostics only and sorts after everything.
inline int ArcTypeStrength(ArcType t) {
    switch (t) {
        case ArcType::Local:      return 0;
        case ArcType::Sublayer:   return 0;
        case ArcType::Inherits:   return 1;
        case ArcType::Variant:    return 2;
        case ArcType::Relocate:   return 3;
        case ArcType::Reference:  return 4;
        case ArcType::Payload:    return 5;
        case ArcType::Specialize: return 6;
        case ArcType::None:       return 7;
    }
    return 7;
}

} // namespace nanousd
