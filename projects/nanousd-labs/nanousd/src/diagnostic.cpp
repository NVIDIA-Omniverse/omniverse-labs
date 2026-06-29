// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanousd/diagnostic.h"

#include <sstream>

namespace nanousd {

// ============================================================
// String conversion
// ============================================================

const char* SeverityToString(DiagSeverity s) {
    switch (s) {
        case DiagSeverity::Info:    return "info";
        case DiagSeverity::Warning: return "warning";
        case DiagSeverity::Error:   return "error";
    }
    return "unknown";
}

const char* CategoryToString(DiagCategory c) {
    switch (c) {
        case DiagCategory::MissingSublayer:    return "missing_sublayer";
        case DiagCategory::SublayerParseFail:  return "sublayer_parse_fail";
        case DiagCategory::MissingReference:   return "missing_reference";
        case DiagCategory::ReferenceParseFail: return "reference_parse_fail";
        case DiagCategory::MissingDefaultPrim: return "missing_default_prim";
        case DiagCategory::MissingPayload:     return "missing_payload";
        case DiagCategory::PayloadParseFail:   return "payload_parse_fail";
        case DiagCategory::Other:              return "other";
        case DiagCategory::InvalidRelocate:    return "invalid_relocate";
        case DiagCategory::MissingInheritTarget:
            return "missing_inherit_target";
        case DiagCategory::MissingSpecializeTarget:
            return "missing_specialize_target";
        case DiagCategory::InvalidReferenceTarget:
            return "invalid_reference_target";
        case DiagCategory::InvalidPayloadTarget:
            return "invalid_payload_target";
        case DiagCategory::InvalidRetimingScale:
            return "invalid_retiming_scale";
    }
    return "unknown";
}

const char* ArcTypeToString(ArcType a) {
    switch (a) {
        case ArcType::Sublayer:  return "sublayer";
        case ArcType::Reference: return "reference";
        case ArcType::Payload:   return "payload";
        case ArcType::None:      return "none";
        case ArcType::Local:      return "local";
        case ArcType::Inherits:   return "inherits";
        case ArcType::Variant:    return "variant";
        case ArcType::Specialize: return "specialize";
        case ArcType::Relocate:   return "relocate";
    }
    return "unknown";
}

// ============================================================
// DiagnosticCollector
// ============================================================

void DiagnosticCollector::Add(Diagnostic diag) {
    diagnostics_.push_back(std::move(diag));
}

void DiagnosticCollector::Merge(std::vector<Diagnostic>&& diags) {
    for (auto& d : diags) {
        diagnostics_.push_back(std::move(d));
    }
}

bool DiagnosticCollector::HasErrors() const {
    for (const auto& d : diagnostics_) {
        if (d.severity >= DiagSeverity::Error) return true;
    }
    return false;
}

bool DiagnosticCollector::HasWarnings() const {
    for (const auto& d : diagnostics_) {
        if (d.severity >= DiagSeverity::Warning) return true;
    }
    return false;
}

std::string DiagnosticCollector::GetFirstError() const {
    for (const auto& d : diagnostics_) {
        if (d.severity >= DiagSeverity::Error) return d.message;
    }
    return {};
}

// ============================================================
// JSON serialization
// ============================================================

namespace {

void EscapeJson(std::string& out, const std::string& s) {
    out.push_back('"');
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    // Control character — encode as \u00XX
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x",
                             static_cast<unsigned>(static_cast<unsigned char>(c)));
                    out += buf;
                } else {
                    out.push_back(c);
                }
                break;
        }
    }
    out.push_back('"');
}

} // anonymous namespace

std::string DiagnosticCollector::ToJson() const {
    std::string json;
    json.reserve(256 * diagnostics_.size());
    json.push_back('[');
    for (size_t i = 0; i < diagnostics_.size(); ++i) {
        if (i > 0) json.push_back(',');
        const auto& d = diagnostics_[i];
        json += "{\"severity\":\"";
        json += SeverityToString(d.severity);
        json += "\",\"category\":\"";
        json += CategoryToString(d.category);
        json += "\",\"message\":";
        EscapeJson(json, d.message);
        json += ",\"primPath\":";
        EscapeJson(json, d.primPath);
        json += ",\"layerPath\":";
        EscapeJson(json, d.layerPath);
        json += ",\"assetPath\":";
        EscapeJson(json, d.assetPath);
        json += ",\"arcType\":\"";
        json += ArcTypeToString(d.arcType);
        json += "\"}";
    }
    json.push_back(']');
    return json;
}

} // namespace nanousd
