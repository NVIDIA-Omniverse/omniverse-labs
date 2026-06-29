// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"
#include "arc.h"

#include <cstdint>
#include <string>
#include <vector>

namespace nanousd {

// ============================================================
// Composition diagnostic types
// ============================================================

// Severity levels for composition diagnostics.
// Warning and above mean the stage may not represent the scene correctly.
enum class DiagSeverity : uint8_t {
    Info    = 0,  // Informational, no correctness impact
    Warning = 1,  // Degraded but partially usable (e.g., missing payload)
    Error   = 2,  // Stage is likely incorrect at affected prims
};

// Category of composition issue.
enum class DiagCategory : uint8_t {
    MissingSublayer   = 0,  // Sublayer asset could not be resolved
    SublayerParseFail = 1,  // Sublayer file found but failed to parse
    MissingReference  = 2,  // Reference asset could not be resolved
    ReferenceParseFail= 3,  // Reference file found but failed to parse
    MissingDefaultPrim= 4,  // Reference/payload target has no defaultPrim
    MissingPayload    = 5,  // Payload asset could not be resolved
    PayloadParseFail  = 6,  // Payload file found but failed to parse
    Other             = 7,  // Catch-all
    InvalidRelocate         = 8,  // Relocate metadata violates composition rules
    MissingInheritTarget    = 9,  // Inherit target has no specs
    MissingSpecializeTarget = 10, // Specialize target has no specs
    InvalidReferenceTarget  = 11, // Reference target path is invalid
    InvalidPayloadTarget    = 12, // Payload target path is invalid
    InvalidRetimingScale    = 13, // Offset scale is <= 0 and ignored
};

// A single composition diagnostic.
struct Diagnostic {
    DiagSeverity severity = DiagSeverity::Error;
    DiagCategory category = DiagCategory::Other;
    std::string message;
    std::string primPath;    // Prim being composed (empty for layer-level issues)
    std::string layerPath;   // Layer that contained the arc
    std::string assetPath;   // The unresolvable/unparseable asset path
    ArcType arcType = ArcType::None;
};

// Collects diagnostics during composition.
// Not thread-safe — parallel sublayer gathering uses per-branch vectors
// that are merged single-threaded after join.
class NANOUSD_CORE_API DiagnosticCollector {
public:
    void Add(Diagnostic diag);
    void Merge(std::vector<Diagnostic>&& diags);

    const std::vector<Diagnostic>& GetAll() const { return diagnostics_; }
    size_t Count() const { return diagnostics_.size(); }
    bool Empty() const { return diagnostics_.empty(); }

    bool HasErrors() const;    // Any severity >= Error
    bool HasWarnings() const;  // Any severity >= Warning

    // Returns the message of the first Error-severity diagnostic, or empty.
    std::string GetFirstError() const;

    // JSON array of all diagnostics (no external deps).
    std::string ToJson() const;

private:
    std::vector<Diagnostic> diagnostics_;
};

// String conversion helpers (used by ToJson and C API).
NANOUSD_CORE_API const char* SeverityToString(DiagSeverity s);
NANOUSD_CORE_API const char* CategoryToString(DiagCategory c);
NANOUSD_CORE_API const char* ArcTypeToString(ArcType a);

} // namespace nanousd
