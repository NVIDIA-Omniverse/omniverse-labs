// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanousd/usda_writer.h"
#include "nanousd/resource.h"
#include "nanousd/spec.h"
#include "nanousd/value.h"
#include "nanousd/path.h"
#include "nanousd/types.h"

#include <algorithm>
#include <cassert>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace nanousd {

// ============================================================
// Value formatting
// ============================================================

namespace {

// Spec-compliant floating point formatter (section: Double-Precision Floating
// Point Representation when Writing USDA content).
// Rules:
//   - Special values: "nan", "inf", "-inf"
//   - Finite: shortest string that round-trips via stod/stof
//   - Scientific notation when exponent < -6 or exponent >= 15
//   - No trailing zeros, no trailing decimal point
template <typename T>
static std::string FormatFloatingPoint(T v) {
    if (std::isnan(v))  return "nan";
    if (std::isinf(v))  return v > 0 ? "inf" : "-inf";

    char buf[64];
    // to_chars in scientific mode gives the shortest round-trip digit string
    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), v,
                                   std::chars_format::scientific);
    if (ec != std::errc{}) {
        // fallback — should not happen in practice
        snprintf(buf, sizeof(buf), "%.17g", static_cast<double>(v));
        return buf;
    }
    *ptr = '\0';

    // Parse: [-]d[.ddd]e[+-]eee
    const char* p = buf;
    bool neg = (*p == '-'); if (neg) ++p;

    std::string digits;
    while (*p && *p != 'e' && *p != 'E') {
        if (*p != '.') digits += *p;
        ++p;
    }
    ++p; // skip 'e'/'E'

    int exp = 0;
    bool expNeg = false;
    if      (*p == '+') ++p;
    else if (*p == '-') { expNeg = true; ++p; }
    while (*p) exp = exp * 10 + (*p++ - '0');
    if (expNeg) exp = -exp;

    // Strip trailing zeros from significand
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();

    std::string out;
    if (neg) out += '-';

    if (exp < -6 || exp >= 15) {
        // Scientific: d[.ddd]eN  (no leading '+' on exponent per spec)
        out += digits[0];
        if (digits.size() > 1) { out += '.'; out += digits.substr(1); }
        out += 'e';
        char ebuf[16]; snprintf(ebuf, sizeof(ebuf), "%d", exp);
        out += ebuf;
    } else {
        // Decimal
        int nd = static_cast<int>(digits.size());
        int dotPos = exp + 1; // digits before the decimal point
        if (dotPos <= 0) {
            out += "0.";
            for (int i = 0; i < -dotPos; ++i) out += '0';
            out += digits;
        } else if (dotPos >= nd) {
            out += digits;
            for (int i = nd; i < dotPos; ++i) out += '0';
        } else {
            out += digits.substr(0, dotPos);
            out += '.';
            out += digits.substr(dotPos);
        }
    }
    return out;
}

static std::string FormatDouble(double v) { return FormatFloatingPoint(v); }
static std::string FormatFloat(float v)   { return FormatFloatingPoint(v); }

static std::string EscapeString(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out += '"';
    for (char c : s) {
        if      (c == '"')  { out += "\\\""; }
        else if (c == '\\') { out += "\\\\"; }
        else if (c == '\n') { out += "\\n"; }
        else if (c == '\r') { out += "\\r"; }
        else if (c == '\t') { out += "\\t"; }
        else                { out += c; }
    }
    out += '"';
    return out;
}

// Forward declaration
static std::string FormatValue(const Value& val, const std::string& typeName = "");

// Map TypeId to USDA type name for dictionary entries
static const char* TypeIdToUsda(TypeId id) {
    switch (id) {
        case TypeId::Bool:    return "bool";
        case TypeId::UChar:   return "uchar";
        case TypeId::Int:     return "int";
        case TypeId::UInt:    return "uint";
        case TypeId::Int64:   return "int64";
        case TypeId::UInt64:  return "uint64";
        case TypeId::Half:    return "half";
        case TypeId::Float:   return "float";
        case TypeId::Double:  return "double";
        case TypeId::String:  return "string";
        case TypeId::Token:   return "token";
        case TypeId::Asset:   return "asset";
        case TypeId::TimeCode:return "timecode";
        case TypeId::Half2:   return "half2";
        case TypeId::Half3:   return "half3";
        case TypeId::Half4:   return "half4";
        case TypeId::Float2:  return "float2";
        case TypeId::Float3:  return "float3";
        case TypeId::Float4:  return "float4";
        case TypeId::Double2: return "double2";
        case TypeId::Double3: return "double3";
        case TypeId::Double4: return "double4";
        case TypeId::Int2:    return "int2";
        case TypeId::Int3:    return "int3";
        case TypeId::Int4:    return "int4";
        case TypeId::Quath:   return "quath";
        case TypeId::Quatf:   return "quatf";
        case TypeId::Quatd:   return "quatd";
        case TypeId::Matrix2d:return "matrix2d";
        case TypeId::Matrix3d:return "matrix3d";
        case TypeId::Matrix4d:return "matrix4d";
        case TypeId::Dictionary: return "dictionary";
        default: return nullptr;
    }
}

// Check if a key needs quoting (contains ':' or other special chars)
static bool NeedsQuoting(const std::string& key) {
    for (char c : key) {
        if (c == ':' || c == ' ' || c == '.' || c == '/' || c == '"')
            return true;
    }
    return false;
}

// Format a dictionary entry key (quote if needed)
static std::string FormatDictKey(const std::string& key) {
    if (NeedsQuoting(key)) return EscapeString(key);
    return key;
}

// Forward declare the recursive dictionary formatter
static std::string FormatDictionary(const Dictionary& dict, const std::string& indent);

static std::string FormatScalar(const Value& val) {
    switch (val.GetTypeId()) {
        case TypeId::Bool:
            if (auto* v = val.Get<Bool>()) return *v ? "1" : "0";
            break;
        case TypeId::UChar:
            if (auto* v = val.Get<UChar>()) { char buf[16]; snprintf(buf,sizeof(buf),"%u",(unsigned)*v); return buf; }
            break;
        case TypeId::Int:
            if (auto* v = val.Get<Int>()) { char buf[32]; snprintf(buf,sizeof(buf),"%d",*v); return buf; }
            break;
        case TypeId::UInt:
            if (auto* v = val.Get<UInt>()) { char buf[32]; snprintf(buf,sizeof(buf),"%u",*v); return buf; }
            break;
        case TypeId::Int64:
            if (auto* v = val.Get<Int64>()) { char buf[32]; snprintf(buf,sizeof(buf),"%lld",(long long)*v); return buf; }
            break;
        case TypeId::UInt64:
            if (auto* v = val.Get<UInt64>()) { char buf[32]; snprintf(buf,sizeof(buf),"%llu",(unsigned long long)*v); return buf; }
            break;
        case TypeId::Half:
            if (auto* v = val.Get<Half>()) return FormatFloat(static_cast<float>(*v));
            break;
        case TypeId::Float:
            if (auto* v = val.Get<Float>()) return FormatFloat(*v);
            break;
        case TypeId::Double:
        case TypeId::TimeCode:
            if (auto* v = val.Get<Double>()) return FormatDouble(*v);
            break;
        case TypeId::String:
            if (auto* v = val.Get<String>()) return EscapeString(*v);
            break;
        case TypeId::Token:
            if (auto* v = val.Get<String>()) return EscapeString(*v);
            if (auto* t = val.Get<Token>()) return EscapeString(t->GetString());
            break;
        case TypeId::Asset:
            if (auto* v = val.Get<String>()) return "@" + *v + "@";
            break;
        case TypeId::Float2:
            if (auto* v = val.Get<GfVec2f>())
                return "(" + FormatFloat(v->data[0]) + ", " + FormatFloat(v->data[1]) + ")";
            break;
        case TypeId::Float3:
            if (auto* v = val.Get<GfVec3f>())
                return "(" + FormatFloat(v->data[0]) + ", " + FormatFloat(v->data[1]) + ", " + FormatFloat(v->data[2]) + ")";
            break;
        case TypeId::Float4:
            if (auto* v = val.Get<GfVec4f>())
                return "(" + FormatFloat(v->data[0]) + ", " + FormatFloat(v->data[1]) + ", " + FormatFloat(v->data[2]) + ", " + FormatFloat(v->data[3]) + ")";
            break;
        case TypeId::Double2:
            if (auto* v = val.Get<GfVec2d>())
                return "(" + FormatDouble(v->data[0]) + ", " + FormatDouble(v->data[1]) + ")";
            break;
        case TypeId::Double3:
            if (auto* v = val.Get<GfVec3d>())
                return "(" + FormatDouble(v->data[0]) + ", " + FormatDouble(v->data[1]) + ", " + FormatDouble(v->data[2]) + ")";
            break;
        case TypeId::Double4:
            if (auto* v = val.Get<GfVec4d>())
                return "(" + FormatDouble(v->data[0]) + ", " + FormatDouble(v->data[1]) + ", " + FormatDouble(v->data[2]) + ", " + FormatDouble(v->data[3]) + ")";
            break;
        case TypeId::Int2:
            if (auto* v = val.Get<GfVec2i>()) {
                char buf[64]; snprintf(buf,sizeof(buf),"(%d, %d)",v->data[0],v->data[1]); return buf;
            }
            break;
        case TypeId::Int3:
            if (auto* v = val.Get<GfVec3i>()) {
                char buf[64]; snprintf(buf,sizeof(buf),"(%d, %d, %d)",v->data[0],v->data[1],v->data[2]); return buf;
            }
            break;
        case TypeId::Int4:
            if (auto* v = val.Get<GfVec4i>()) {
                char buf[64]; snprintf(buf,sizeof(buf),"(%d, %d, %d, %d)",v->data[0],v->data[1],v->data[2],v->data[3]); return buf;
            }
            break;
        case TypeId::Half2:
            if (auto* v = val.Get<GfVec2h>())
                return "(" + FormatFloat(static_cast<float>(v->data[0])) + ", " + FormatFloat(static_cast<float>(v->data[1])) + ")";
            break;
        case TypeId::Half3:
            if (auto* v = val.Get<GfVec3h>())
                return "(" + FormatFloat(static_cast<float>(v->data[0])) + ", " + FormatFloat(static_cast<float>(v->data[1])) + ", " + FormatFloat(static_cast<float>(v->data[2])) + ")";
            break;
        case TypeId::Half4:
            if (auto* v = val.Get<GfVec4h>())
                return "(" + FormatFloat(static_cast<float>(v->data[0])) + ", " + FormatFloat(static_cast<float>(v->data[1])) + ", " + FormatFloat(static_cast<float>(v->data[2])) + ", " + FormatFloat(static_cast<float>(v->data[3])) + ")";
            break;
        // Quat storage is (i, j, k, r) = data[0..3]; USDA writes (r, i, j, k)
        case TypeId::Quatf:
            if (auto* v = val.Get<GfQuatf>())
                return "(" + FormatFloat(v->data[3]) + ", " + FormatFloat(v->data[0]) + ", " + FormatFloat(v->data[1]) + ", " + FormatFloat(v->data[2]) + ")";
            break;
        case TypeId::Quatd:
            if (auto* v = val.Get<GfQuatd>())
                return "(" + FormatDouble(v->data[3]) + ", " + FormatDouble(v->data[0]) + ", " + FormatDouble(v->data[1]) + ", " + FormatDouble(v->data[2]) + ")";
            break;
        case TypeId::Quath:
            if (auto* v = val.Get<GfQuath>())
                return "(" + FormatFloat(static_cast<float>(v->data[3])) + ", " + FormatFloat(static_cast<float>(v->data[0])) + ", " + FormatFloat(static_cast<float>(v->data[1])) + ", " + FormatFloat(static_cast<float>(v->data[2])) + ")";
            break;
        case TypeId::Matrix2d:
            if (auto* v = val.Get<GfMatrix2d>()) {
                return "((" + FormatDouble(v->data[0]) + ", " + FormatDouble(v->data[1]) + "), "
                     + "(" + FormatDouble(v->data[2]) + ", " + FormatDouble(v->data[3]) + "))";
            }
            break;
        case TypeId::Matrix3d:
            if (auto* v = val.Get<GfMatrix3d>()) {
                return "((" + FormatDouble(v->data[0]) + ", " + FormatDouble(v->data[1]) + ", " + FormatDouble(v->data[2]) + "), "
                     + "(" + FormatDouble(v->data[3]) + ", " + FormatDouble(v->data[4]) + ", " + FormatDouble(v->data[5]) + "), "
                     + "(" + FormatDouble(v->data[6]) + ", " + FormatDouble(v->data[7]) + ", " + FormatDouble(v->data[8]) + "))";
            }
            break;
        case TypeId::Matrix4d:
            if (auto* v = val.Get<GfMatrix4d>()) {
                return "((" + FormatDouble(v->data[0])  + ", " + FormatDouble(v->data[1])  + ", " + FormatDouble(v->data[2])  + ", " + FormatDouble(v->data[3])  + "), "
                     + "(" + FormatDouble(v->data[4])  + ", " + FormatDouble(v->data[5])  + ", " + FormatDouble(v->data[6])  + ", " + FormatDouble(v->data[7])  + "), "
                     + "(" + FormatDouble(v->data[8])  + ", " + FormatDouble(v->data[9])  + ", " + FormatDouble(v->data[10]) + ", " + FormatDouble(v->data[11]) + "), "
                     + "(" + FormatDouble(v->data[12]) + ", " + FormatDouble(v->data[13]) + ", " + FormatDouble(v->data[14]) + ", " + FormatDouble(v->data[15]) + "))";
            }
            break;
        case TypeId::Dictionary:
            if (auto* dict = val.Get<Dictionary>()) {
                return FormatDictionary(*dict, "    ");
            }
            break;
        default:
            break;
    }
    return "# unsupported value";
}

static std::string FormatArray(const Value& val) {
    // Arrays are stored as std::vector<T> inside the Value
    std::string out = "[";
    bool first = true;
    auto append = [&](const std::string& s) {
        if (!first) out += ", ";
        out += s;
        first = false;
    };

    switch (val.GetTypeId()) {
        case TypeId::Bool:
            if (auto* v = val.Get<std::vector<Bool>>()) { for (auto x : *v) append(x ? "1" : "0"); }
            break;
        case TypeId::UChar:
            if (auto* v = val.Get<std::vector<UChar>>()) { char buf[16]; for (auto x : *v) { snprintf(buf,sizeof(buf),"%u",static_cast<unsigned>(x)); append(buf); } }
            break;
        case TypeId::Int:
            if (auto* v = val.Get<std::vector<Int>>()) { char buf[32]; for (auto x : *v) { snprintf(buf,sizeof(buf),"%d",x); append(buf); } }
            break;
        case TypeId::UInt:
            if (auto* v = val.Get<std::vector<UInt>>()) { char buf[32]; for (auto x : *v) { snprintf(buf,sizeof(buf),"%u",x); append(buf); } }
            break;
        case TypeId::Int64:
            if (auto* v = val.Get<std::vector<Int64>>()) { char buf[32]; for (auto x : *v) { snprintf(buf,sizeof(buf),"%lld",(long long)x); append(buf); } }
            break;
        case TypeId::UInt64:
            if (auto* v = val.Get<std::vector<UInt64>>()) { char buf[32]; for (auto x : *v) { snprintf(buf,sizeof(buf),"%llu",(unsigned long long)x); append(buf); } }
            break;
        case TypeId::Half:
            if (auto* v = val.Get<std::vector<Half>>()) { for (auto x : *v) append(FormatFloat(static_cast<float>(x))); }
            break;
        case TypeId::Float:
            if (auto* v = val.Get<std::vector<Float>>()) { for (auto x : *v) append(FormatFloat(x)); }
            break;
        case TypeId::Double:
        case TypeId::TimeCode:
            if (auto* v = val.Get<std::vector<Double>>()) { for (auto x : *v) append(FormatDouble(x)); }
            break;
        case TypeId::String:
            if (auto* v = val.Get<std::vector<String>>()) { for (const auto& x : *v) append(EscapeString(x)); }
            break;
        case TypeId::Token:
            if (auto* v = val.Get<std::vector<String>>()) { for (const auto& x : *v) append(EscapeString(x)); }
            else if (auto* v = val.Get<std::vector<Token>>()) { for (const auto& x : *v) append(EscapeString(x.GetString())); }
            break;
        case TypeId::Asset:
            if (auto* v = val.Get<std::vector<String>>()) { for (const auto& x : *v) append("@" + x + "@"); }
            break;
        case TypeId::Float2:
            if (auto* v = val.Get<std::vector<GfVec2f>>()) { for (const auto& x : *v) append("(" + FormatFloat(x.data[0]) + ", " + FormatFloat(x.data[1]) + ")"); }
            break;
        case TypeId::Float3:
            if (auto* v = val.Get<std::vector<GfVec3f>>()) { for (const auto& x : *v) append("(" + FormatFloat(x.data[0]) + ", " + FormatFloat(x.data[1]) + ", " + FormatFloat(x.data[2]) + ")"); }
            break;
        case TypeId::Float4:
            if (auto* v = val.Get<std::vector<GfVec4f>>()) { for (const auto& x : *v) append("(" + FormatFloat(x.data[0]) + ", " + FormatFloat(x.data[1]) + ", " + FormatFloat(x.data[2]) + ", " + FormatFloat(x.data[3]) + ")"); }
            break;
        case TypeId::Double2:
            if (auto* v = val.Get<std::vector<GfVec2d>>()) { for (const auto& x : *v) append("(" + FormatDouble(x.data[0]) + ", " + FormatDouble(x.data[1]) + ")"); }
            break;
        case TypeId::Double3:
            if (auto* v = val.Get<std::vector<GfVec3d>>()) { for (const auto& x : *v) append("(" + FormatDouble(x.data[0]) + ", " + FormatDouble(x.data[1]) + ", " + FormatDouble(x.data[2]) + ")"); }
            break;
        case TypeId::Double4:
            if (auto* v = val.Get<std::vector<GfVec4d>>()) { for (const auto& x : *v) append("(" + FormatDouble(x.data[0]) + ", " + FormatDouble(x.data[1]) + ", " + FormatDouble(x.data[2]) + ", " + FormatDouble(x.data[3]) + ")"); }
            break;
        case TypeId::Int2:
            if (auto* v = val.Get<std::vector<GfVec2i>>()) { char buf[64]; for (const auto& x : *v) { snprintf(buf,sizeof(buf),"(%d, %d)",x.data[0],x.data[1]); append(buf); } }
            break;
        case TypeId::Int3:
            if (auto* v = val.Get<std::vector<GfVec3i>>()) { char buf[64]; for (const auto& x : *v) { snprintf(buf,sizeof(buf),"(%d, %d, %d)",x.data[0],x.data[1],x.data[2]); append(buf); } }
            break;
        case TypeId::Int4:
            if (auto* v = val.Get<std::vector<GfVec4i>>()) { char buf[64]; for (const auto& x : *v) { snprintf(buf,sizeof(buf),"(%d, %d, %d, %d)",x.data[0],x.data[1],x.data[2],x.data[3]); append(buf); } }
            break;
        case TypeId::Half2:
            if (auto* v = val.Get<std::vector<GfVec2h>>()) { for (const auto& x : *v) append("(" + FormatFloat(static_cast<float>(x.data[0])) + ", " + FormatFloat(static_cast<float>(x.data[1])) + ")"); }
            break;
        case TypeId::Half3:
            if (auto* v = val.Get<std::vector<GfVec3h>>()) { for (const auto& x : *v) append("(" + FormatFloat(static_cast<float>(x.data[0])) + ", " + FormatFloat(static_cast<float>(x.data[1])) + ", " + FormatFloat(static_cast<float>(x.data[2])) + ")"); }
            break;
        case TypeId::Half4:
            if (auto* v = val.Get<std::vector<GfVec4h>>()) { for (const auto& x : *v) append("(" + FormatFloat(static_cast<float>(x.data[0])) + ", " + FormatFloat(static_cast<float>(x.data[1])) + ", " + FormatFloat(static_cast<float>(x.data[2])) + ", " + FormatFloat(static_cast<float>(x.data[3])) + ")"); }
            break;
        // Quat storage is (i, j, k, r) = data[0..3]; USDA writes (r, i, j, k)
        case TypeId::Quath:
            if (auto* v = val.Get<std::vector<GfQuath>>()) { for (const auto& x : *v) append("(" + FormatFloat(static_cast<float>(x.data[3])) + ", " + FormatFloat(static_cast<float>(x.data[0])) + ", " + FormatFloat(static_cast<float>(x.data[1])) + ", " + FormatFloat(static_cast<float>(x.data[2])) + ")"); }
            break;
        case TypeId::Quatf:
            if (auto* v = val.Get<std::vector<GfQuatf>>()) { for (const auto& x : *v) append("(" + FormatFloat(x.data[3]) + ", " + FormatFloat(x.data[0]) + ", " + FormatFloat(x.data[1]) + ", " + FormatFloat(x.data[2]) + ")"); }
            break;
        case TypeId::Quatd:
            if (auto* v = val.Get<std::vector<GfQuatd>>()) { for (const auto& x : *v) append("(" + FormatDouble(x.data[3]) + ", " + FormatDouble(x.data[0]) + ", " + FormatDouble(x.data[1]) + ", " + FormatDouble(x.data[2]) + ")"); }
            break;
        case TypeId::Matrix2d:
            if (auto* v = val.Get<std::vector<GfMatrix2d>>()) {
                for (const auto& m : *v) {
                    append("((" + FormatDouble(m.data[0]) + ", " + FormatDouble(m.data[1]) + "), "
                         + "(" + FormatDouble(m.data[2]) + ", " + FormatDouble(m.data[3]) + "))");
                }
            }
            break;
        case TypeId::Matrix3d:
            if (auto* v = val.Get<std::vector<GfMatrix3d>>()) {
                for (const auto& m : *v) {
                    append("((" + FormatDouble(m.data[0]) + ", " + FormatDouble(m.data[1]) + ", " + FormatDouble(m.data[2]) + "), "
                         + "(" + FormatDouble(m.data[3]) + ", " + FormatDouble(m.data[4]) + ", " + FormatDouble(m.data[5]) + "), "
                         + "(" + FormatDouble(m.data[6]) + ", " + FormatDouble(m.data[7]) + ", " + FormatDouble(m.data[8]) + "))");
                }
            }
            break;
        case TypeId::Matrix4d:
            if (auto* v = val.Get<std::vector<GfMatrix4d>>()) {
                for (const auto& m : *v) {
                    append("((" + FormatDouble(m.data[0]) + ", " + FormatDouble(m.data[1]) + ", " + FormatDouble(m.data[2]) + ", " + FormatDouble(m.data[3]) + "), "
                         + "(" + FormatDouble(m.data[4]) + ", " + FormatDouble(m.data[5]) + ", " + FormatDouble(m.data[6]) + ", " + FormatDouble(m.data[7]) + "), "
                         + "(" + FormatDouble(m.data[8]) + ", " + FormatDouble(m.data[9]) + ", " + FormatDouble(m.data[10]) + ", " + FormatDouble(m.data[11]) + "), "
                         + "(" + FormatDouble(m.data[12]) + ", " + FormatDouble(m.data[13]) + ", " + FormatDouble(m.data[14]) + ", " + FormatDouble(m.data[15]) + "))");
                }
            }
            break;
        default:
            out += "# unsupported array element type";
            break;
    }
    out += "]";
    return out;
}

// Format a dictionary in proper USDA syntax:
//   {
//       type key = value
//       dictionary nested = { ... }
//   }
static std::string FormatDictionary(const Dictionary& dict, const std::string& indent) {
    if (dict.empty()) return "{}";
    std::string out = "{\n";
    for (const auto& [k, v] : dict) {
        const char* typeName = TypeIdToUsda(v.GetTypeId());
        // Check if it's an array value — arrays need [] suffix
        bool isArr = v.IsArray();
        if (v.GetTypeId() == TypeId::Dictionary) {
            if (auto* nested = v.Get<Dictionary>()) {
                out += indent + "    dictionary " + FormatDictKey(k) + " = "
                     + FormatDictionary(*nested, indent + "    ") + "\n";
            }
        } else if (typeName) {
            std::string typeStr = typeName;
            if (isArr) typeStr += "[]";
            out += indent + "    " + typeStr + " " + FormatDictKey(k) + " = "
                 + FormatValue(v) + "\n";
        } else {
            // Fallback: no type prefix (shouldn't happen for well-formed data)
            out += indent + "    " + FormatDictKey(k) + " = " + FormatValue(v) + "\n";
        }
    }
    out += indent + "}";
    return out;
}

static std::string FormatValue(const Value& val, const std::string& /*typeName*/) {
    if (val.IsBlock()) return "None";
    if (val.IsArray()) return FormatArray(val);
    return FormatScalar(val);
}

static void WriteRelocatesMetadata(std::ostringstream& out,
                                    const std::vector<Relocate>& relocates) {
    out << "    relocates = {\n";
    for (const auto& r : relocates) {
        out << "        <" << r.sourcePath.GetText() << ">: ";
        if (r.targetPath) {
            out << "<" << r.targetPath->GetText() << ">";
        } else {
            out << "None";
        }
        out << ",\n";
    }
    out << "    }\n";
}

// ============================================================
// Reference / ListOp formatting
// ============================================================

static std::string FormatReference(const Reference& ref) {
    std::string out;
    if (ref.assetPath) out += "@" + *ref.assetPath + "@";
    if (ref.primPath)  out += "<" + ref.primPath->GetText() + ">";
    if (ref.offset != 0.0 || ref.scale != 1.0) {
        out += " (offset = " + FormatDouble(ref.offset) + "; scale = " + FormatDouble(ref.scale) + ")";
    }
    return out;
}

template <typename T, typename Fmt>
static std::string FormatListOpItems(const std::vector<T>& items, Fmt fmt) {
    if (items.empty()) return "[]";
    std::string out = "[";
    bool first = true;
    for (const auto& item : items) {
        if (!first) out += ", ";
        out += fmt(item);
        first = false;
    }
    out += "]";
    return out;
}

static std::string FormatStringRef(const std::string& s) { return "<" + s + ">"; }

// ============================================================
// Spline serialization (spec §7.4.2.4 / §16.2.13)
// ============================================================

static const char* ExtrapolationModeName(ExtrapolationMode m) {
    switch (m) {
    case ExtrapolationMode::None:          return "none";
    case ExtrapolationMode::Held:          return "held";
    case ExtrapolationMode::Linear:        return "linear";
    case ExtrapolationMode::Sloped:        return "sloped";
    case ExtrapolationMode::LoopRepeat:    return "looprepeat";
    case ExtrapolationMode::LoopReset:     return "loopreset";
    case ExtrapolationMode::LoopOscillate: return "looposcillate";
    }
    return "held";
}

static const char* InterpolationModeName(InterpolationMode m) {
    switch (m) {
    case InterpolationMode::None:   return "none";
    case InterpolationMode::Held:   return "held";
    case InterpolationMode::Linear: return "linear";
    case InterpolationMode::Curve:  return "curve";
    }
    return "held";
}

// Emit a Spline value in USDA syntax starting at the '{' on its own
// line at the given indent. Only non-default fields are written,
// keeping round-tripped output minimal and deterministic.
static void WriteSplineBlock(std::ostringstream& out, const Spline& s,
                              const std::string& indent) {
    const std::string ki = indent + "    ";
    out << "{\n";

    // Curve type — always emit for clarity.
    out << ki << (s.curveType == CurveType::Hermite ? "hermite" : "bezier")
        << ",\n";

    // Extrapolation clauses — emit only when non-default (Held, slope 0).
    auto emitExtrap = [&](const char* side, ExtrapolationMode m, double slope) {
        if (m == ExtrapolationMode::Held && slope == 0.0) return;
        out << ki << side << ": " << ExtrapolationModeName(m);
        if (m == ExtrapolationMode::Sloped) {
            out << "(" << FormatDouble(slope) << ")";
        }
        out << ",\n";
    };
    emitExtrap("pre",  s.preExtrapolationMode,  s.preExtrapolationSlope);
    emitExtrap("post", s.postExtrapolationMode, s.postExtrapolationSlope);

    // Loop parameters — only if any field diverges from defaults.
    const auto& lp = s.loopParameters;
    if (lp.protoStart != 0.0 || lp.protoEnd != 0.0 ||
        lp.numPreLoops != 0  || lp.numPostLoops != 0 ||
        lp.valueOffset != 0.0) {
        out << ki << "loop: ("
            << FormatDouble(lp.protoStart)   << ", "
            << FormatDouble(lp.protoEnd)     << ", "
            << lp.numPreLoops                << ", "
            << lp.numPostLoops               << ", "
            << FormatDouble(lp.valueOffset)  << "),\n";
    }

    // Knots — sorted ascending by time for deterministic output.
    std::vector<SplineKnot> sorted = s.knots;
    std::sort(sorted.begin(), sorted.end(),
              [](const SplineKnot& a, const SplineKnot& b) {
                  return a.time < b.time;
              });
    for (const auto& k : sorted) {
        out << ki << FormatDouble(k.time) << ": " << FormatDouble(k.value);
        if (k.preValue != k.value) {
            out << " & " << FormatDouble(k.preValue);
        }
        if (k.preTangentSlope != 0.0 || k.preTangentWidth != 0.0) {
            out << "; pre (" << FormatDouble(k.preTangentSlope) << ", "
                << FormatDouble(k.preTangentWidth) << ")";
        }
        bool hasPostTangent = (k.postTangentSlope != 0.0 ||
                                k.postTangentWidth != 0.0);
        bool hasInterp = (k.nextInterpolationMode != InterpolationMode::Held);
        if (hasPostTangent || hasInterp) {
            out << "; post";
            if (hasInterp) out << " " << InterpolationModeName(k.nextInterpolationMode);
            if (hasPostTangent) {
                out << " (" << FormatDouble(k.postTangentSlope) << ", "
                    << FormatDouble(k.postTangentWidth) << ")";
            }
        }
        out << ",\n";
    }

    out << indent << "}\n";
}

// ============================================================
// Layer metadata block
// ============================================================

// Fields that are written inline in the prim/layer header and not in metadata blocks
static bool IsStructuralField(const std::string& name) {
    return name == "specifier" || name == "typeName" || name == "variability"
        || name == "custom" || name == "default" || name == "timeSamples"
        || name == "spline"
        || name == "targetPaths" || name == "connectionPaths"
        || name == "primChildren" || name == "propertyChildren"
        || name == "variantSetChildren" || name == "variantChildren";
}

static void WriteMetadataBlock(std::ostringstream& out, const Spec& spec,
                                const std::string& indent) {
    // Collect metadata fields (non-structural)
    std::vector<std::pair<std::string, const Value*>> meta;
    for (const auto& [name, val] : spec.GetFields()) {
        if (!IsStructuralField(name)) {
            meta.push_back({name, &val});
        }
    }
    if (meta.empty()) return;

    out << " (\n";
    for (const auto& [name, val] : meta) {
        // Handle composition arcs
        if (name == "references") {
            if (auto* lop = val->Get<ListOp<Reference>>()) {
                if (lop->IsExplicit()) {
                    out << indent << "    references = " << FormatListOpItems(lop->GetExplicitItems(), FormatReference) << "\n";
                } else {
                    if (!lop->GetPrependedItems().empty())
                        out << indent << "    prepend references = " << FormatListOpItems(lop->GetPrependedItems(), FormatReference) << "\n";
                    if (!lop->GetAppendedItems().empty())
                        out << indent << "    append references = " << FormatListOpItems(lop->GetAppendedItems(), FormatReference) << "\n";
                    if (!lop->GetDeletedItems().empty())
                        out << indent << "    delete references = " << FormatListOpItems(lop->GetDeletedItems(), FormatReference) << "\n";
                }
            }
        } else if (name == "payload") {
            if (auto* lop = val->Get<ListOp<Reference>>()) {
                if (lop->IsExplicit()) {
                    out << indent << "    payload = " << FormatListOpItems(lop->GetExplicitItems(), FormatReference) << "\n";
                } else {
                    if (!lop->GetPrependedItems().empty())
                        out << indent << "    prepend payload = " << FormatListOpItems(lop->GetPrependedItems(), FormatReference) << "\n";
                    if (!lop->GetAppendedItems().empty())
                        out << indent << "    append payload = " << FormatListOpItems(lop->GetAppendedItems(), FormatReference) << "\n";
                }
            }
        } else if (name == "inheritPaths" || name == "specializes") {
            if (auto* lop = val->Get<ListOp<std::string>>()) {
                if (lop->IsExplicit()) {
                    out << indent << "    " << name << " = " << FormatListOpItems(lop->GetExplicitItems(), FormatStringRef) << "\n";
                } else {
                    if (!lop->GetPrependedItems().empty())
                        out << indent << "    prepend " << name << " = " << FormatListOpItems(lop->GetPrependedItems(), FormatStringRef) << "\n";
                    if (!lop->GetAppendedItems().empty())
                        out << indent << "    append " << name << " = " << FormatListOpItems(lop->GetAppendedItems(), FormatStringRef) << "\n";
                }
            }
        } else if (name == "apiSchemas") {
            // apiSchemas is spec-declared listop<token> (§13.2.1.2),
            // stored as ListOp<Token>. A legacy ListOp<std::string>
            // fallback handles any in-memory code still producing the
            // old type.
            auto emitStrings = [&](const auto& items, const char* keyword) {
                auto fmt = [](const std::string& s) { return EscapeString(s); };
                if (items.empty()) return;
                out << indent << "    " << keyword << "apiSchemas = "
                    << FormatListOpItems(items, fmt) << "\n";
            };
            if (auto* lop = val->Get<ListOp<Token>>()) {
                std::vector<std::string> explicitStrs;
                for (const auto& t : lop->GetExplicitItems()) explicitStrs.push_back(t.GetString());
                std::vector<std::string> prependStrs;
                for (const auto& t : lop->GetPrependedItems()) prependStrs.push_back(t.GetString());
                if (lop->IsExplicit()) {
                    emitStrings(explicitStrs, "");
                } else {
                    emitStrings(prependStrs, "prepend ");
                }
            } else if (auto* lop = val->Get<ListOp<std::string>>()) {
                if (lop->IsExplicit()) {
                    emitStrings(lop->GetExplicitItems(), "");
                } else {
                    emitStrings(lop->GetPrependedItems(), "prepend ");
                }
            }
        } else if (name == "customData" || name == "assetInfo") {
            if (auto* dict = val->Get<Dictionary>()) {
                out << indent << "    " << name << " = "
                    << FormatDictionary(*dict, indent + "    ") << "\n";
            }
        } else if (name == "fallbackPrimTypes") {
            if (auto* dict = val->Get<Dictionary>()) {
                out << indent << "    dictionary " << name << " = "
                    << FormatDictionary(*dict, indent + "    ") << "\n";
            }
        } else if (name == "subLayers") {
            if (auto* slp = val->Get<SubLayerPaths>()) {
                out << indent << "    subLayers = [\n";
                for (size_t i = 0; i < slp->paths.size(); ++i) {
                    out << indent << "        @" << slp->paths[i] << "@";
                    if (i < slp->offsets.size() && (slp->offsets[i].offset != 0.0 || slp->offsets[i].scale != 1.0)) {
                        out << " (offset = " << FormatDouble(slp->offsets[i].offset)
                            << "; scale = " << FormatDouble(slp->offsets[i].scale) << ")";
                    }
                    out << ",\n";
                }
                out << indent << "    ]\n";
            }
        } else {
            // Generic scalar/string metadata
            out << indent << "    " << name << " = " << FormatValue(*val) << "\n";
        }
    }
    out << indent << ")";
}

// ============================================================
// Spec writers
// ============================================================

static void WritePrimByPath(std::ostringstream& out, const Layer& layer,
                            const Path& primPath, const std::string& primName,
                            const Spec& spec, const std::string& indent);
static void WriteAttributeByName(std::ostringstream& out,
                                 const std::string& propName,
                                 const Spec& spec, const std::string& indent);
static void WriteRelationshipByName(std::ostringstream& out,
                                    const std::string& propName,
                                    const Spec& spec, const std::string& indent);

static void WritePrimByPath(std::ostringstream& out, const Layer& layer,
                            const Path& primPath, const std::string& primName,
                            const Spec& spec, const std::string& indent) {
    // Specifier keyword
    const char* specKw = "def";
    if (auto* sv = spec.GetField(FieldNames::specifier)) {
        if (auto* iv = sv->Get<Int>()) {
            if (*iv == static_cast<Int>(Specifier::Over))   specKw = "over";
            else if (*iv == static_cast<Int>(Specifier::Class)) specKw = "class";
        }
    }

    // Type name (optional)
    std::string typePart;
    if (auto* tv = spec.GetField(FieldNames::typeName)) {
        if (auto* tok = tv->Get<Token>()) {
            if (!tok->IsEmpty()) typePart = " " + tok->GetString();
        } else if (auto* sv = tv->Get<String>()) {
            if (!sv->empty()) typePart = " " + *sv;
        }
    }

    out << indent << specKw << typePart << " \"" << primName << "\"";

    // Metadata block
    WriteMetadataBlock(out, spec, indent);

    out << "\n" << indent << "{\n";
    const std::string childIndent = indent + "    ";

    // Properties from Layer index — O(1) lookup
    auto propNames = layer.GetPropertyNames(primPath);
    std::sort(propNames.begin(), propNames.end(), PathElementTokenLess);
    if (!propNames.empty()) {
        for (const auto& propName : propNames) {
            Path propPath = primPath.AppendProperty(propName);
            const Spec* propSpec = layer.GetSpec(propPath);
            if (!propSpec) continue;
            switch (propSpec->GetType()) {
                case SpecType::Attribute:
                    WriteAttributeByName(out, propName, *propSpec, childIndent);
                    break;
                case SpecType::Relationship:
                    WriteRelationshipByName(out, propName, *propSpec, childIndent);
                    break;
                default:
                    break;
            }
        }
        out << "\n";
    }

    // Child prims from Layer index — O(1) lookup
    auto childNames = layer.GetChildNames(primPath);
    for (const auto& childName : childNames) {
        Path childPath = primPath.AppendChild(childName);
        const Spec* childSpec = layer.GetSpec(childPath);
        if (childSpec && childSpec->GetType() == SpecType::Prim) {
            WritePrimByPath(out, layer, childPath, childName.GetString(), *childSpec, childIndent);
        }
    }

    out << indent << "}\n";
}

static void WritePathListValue(std::ostringstream& out, const Value& field) {
    auto writeItems = [&](const auto& items) {
        if (items.size() == 1) {
            out << "<" << items[0].GetText() << ">";
            return;
        }
        out << "[";
        for (size_t i = 0; i < items.size(); ++i) {
            if (i) out << ", ";
            out << "<" << items[i].GetText() << ">";
        }
        out << "]";
    };

    if (field.IsBlock()) {
        out << "None";
    } else if (auto* lop = field.Get<ListOp<Path>>()) {
        writeItems(lop->GetItems());
    } else if (auto* legacy = field.Get<ListOp<std::string>>()) {
        std::vector<Path> paths;
        for (const auto& text : legacy->GetItems()) {
            Path p = Path::Parse(text);
            if (!p.IsEmpty()) paths.push_back(std::move(p));
        }
        writeItems(paths);
    } else {
        out << "[]";
    }
}

static void WriteAttributeByName(std::ostringstream& out,
                                 const std::string& propName,
                                 const Spec& spec, const std::string& indent) {
    // Type name
    std::string typeName;
    if (auto* tv = spec.GetField(FieldNames::typeName)) {
        if (auto* tok = tv->Get<Token>()) typeName = tok->GetString();
        else if (auto* sv = tv->Get<String>()) typeName = *sv;
    }

    // Variability prefix
    const char* varPfx = "";
    if (auto* vv = spec.GetField(FieldNames::variability)) {
        if (auto* iv = vv->Get<Int>()) {
            if (*iv == static_cast<Int>(Variability::Uniform)) varPfx = "uniform ";
        }
    }

    // Custom prefix
    const char* custPfx = "";
    if (auto* cv = spec.GetField(FieldNames::custom)) {
        if (auto* bv = cv->Get<Bool>()) {
            if (*bv) custPfx = "custom ";
        }
    }

    // Collect attribute metadata (for metadata block)
    std::vector<std::pair<std::string, const Value*>> attrMeta;
    for (const auto& [name, val] : spec.GetFields()) {
        if (!IsStructuralField(name)) {
            attrMeta.push_back({name, &val});
        }
    }

    // timeSamples-only (no default)
    const Value* tsField = spec.GetField(FieldNames::timeSamples);
    const Value* defField = spec.GetField(FieldNames::defaultValue);
    const Value* splineField = spec.GetField(FieldNames::spline);
    const Value* connField = spec.GetField(FieldNames::connectionPaths);

    auto writeAttributePrefix = [&]() {
        out << indent << custPfx << varPfx;
        if (!typeName.empty()) out << typeName << " ";
        out << propName;
    };

    auto writeConnection = [&]() {
        if (!connField) return;
        writeAttributePrefix();
        out << ".connect = ";
        WritePathListValue(out, *connField);
        out << "\n";
    };

    if (connField && !defField && !tsField && !splineField) {
        writeConnection();
        return;
    }

    if (tsField && !defField) {
        // Write timeSamples form
        out << indent << custPfx << varPfx;
        if (!typeName.empty()) out << typeName << " ";
        out << propName;
        if (!attrMeta.empty()) {
            WriteMetadataBlock(out, spec, indent);
        }
        out << ".timeSamples = {\n";
        if (auto* dict = tsField->Get<Dictionary>()) {
            // Sort by numeric time value
            std::vector<std::pair<double, const Value*>> sorted;
            for (const auto& [k, v] : *dict) {
                try { sorted.push_back({std::stod(k), &v}); } catch (...) {}
            }
            std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
            for (const auto& [t, v] : sorted) {
                out << indent << "    " << FormatDouble(t) << ": " << FormatValue(*v) << ",\n";
            }
        }
        out << indent << "}\n";
        writeConnection();
        return;
    }

    // spline-only (no default, no timeSamples) — emit `type name.spline = {...}`
    // directly. Parallel to the timeSamples-only path.
    if (splineField && !defField && !tsField) {
        out << indent << custPfx << varPfx;
        if (!typeName.empty()) out << typeName << " ";
        out << propName;
        if (!attrMeta.empty()) {
            WriteMetadataBlock(out, spec, indent);
        }
        out << ".spline = ";
        if (auto* s = splineField->Get<Spline>()) {
            WriteSplineBlock(out, *s, indent);
        } else {
            out << "{}\n";
        }
        writeConnection();
        return;
    }

    // Default value (with optional timeSamples)
    out << indent << custPfx << varPfx;
    if (!typeName.empty()) {
        // Arrays get [] suffix on type (unless typeName already has it)
        bool hasArraySuffix = typeName.size() >= 2 &&
            typeName[typeName.size()-2] == '[' && typeName[typeName.size()-1] == ']';
        if (defField && defField->IsArray() && !hasArraySuffix)
            out << typeName << "[] ";
        else
            out << typeName << " ";
    }
    out << propName;

    if (defField && !defField->IsBlock()) {
        out << " = " << FormatValue(*defField);
    } else if (!defField) {
        // No value — bare declaration
    }

    // Attribute metadata block
    if (!attrMeta.empty()) {
        WriteMetadataBlock(out, spec, indent);
    }
    out << "\n";

    // Append timeSamples block if both default and samples exist
    if (tsField && defField) {
        out << indent;
        if (!typeName.empty()) {
            bool hasArraySuffix = typeName.size() >= 2 &&
                typeName[typeName.size()-2] == '[' && typeName[typeName.size()-1] == ']';
            if (defField->IsArray() && !hasArraySuffix)
                out << typeName << "[] ";
            else
                out << typeName << " ";
        }
        out << propName << ".timeSamples = {\n";
        if (auto* dict = tsField->Get<Dictionary>()) {
            std::vector<std::pair<double, const Value*>> sorted;
            for (const auto& [k, v] : *dict) {
                try { sorted.push_back({std::stod(k), &v}); } catch (...) {}
            }
            std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
            for (const auto& [t, v] : sorted) {
                out << indent << "    " << FormatDouble(t) << ": " << FormatValue(*v) << ",\n";
            }
        }
        out << indent << "}\n";
    }

    // Append spline block when authored alongside a default (or
    // alongside timeSamples). Splines authored without a default or
    // timeSamples are emitted through the spline-only path above.
    if (splineField && (defField || tsField)) {
        out << indent;
        if (!typeName.empty()) out << typeName << " ";
        out << propName << ".spline = ";
        if (auto* s = splineField->Get<Spline>()) {
            WriteSplineBlock(out, *s, indent);
        } else {
            out << "{}\n";
        }
    }

    writeConnection();
}

static void WriteRelationshipByName(std::ostringstream& out,
                                    const std::string& propName,
                                    const Spec& spec, const std::string& indent) {
    out << indent << "rel " << propName;

    if (auto* tf = spec.GetField(FieldNames::targetPaths)) {
        // Emit a path-typed listop. After the parser change targetPaths
        // is stored as ListOp<Path>; a string-typed fallback is kept so
        // any in-memory writer caller still constructing the legacy
        // string variant continues to serialize correctly.
        auto emit = [&](const auto& items) {
            if (items.size() == 1) {
                out << " = <" << items[0] << ">";
            } else if (!items.empty()) {
                out << " = [";
                bool first = true;
                for (const auto& t : items) {
                    if (!first) out << ", ";
                    out << "<" << t << ">";
                    first = false;
                }
                out << "]";
            }
        };
        if (auto* plop = tf->Get<ListOp<Path>>()) {
            const auto& items = plop->IsExplicit() ? plop->GetExplicitItems()
                                                    : plop->GetAppendedItems();
            std::vector<std::string> texts;
            texts.reserve(items.size());
            for (const auto& p : items) texts.push_back(p.GetText());
            emit(texts);
        } else if (auto* lop = tf->Get<ListOp<std::string>>()) {
            const auto& items = lop->IsExplicit() ? lop->GetExplicitItems()
                                                   : lop->GetAppendedItems();
            emit(items);
        }
    }

    // Relationship metadata
    std::vector<std::pair<std::string, const Value*>> meta;
    for (const auto& [name, val] : spec.GetFields()) {
        if (!IsStructuralField(name)) meta.push_back({name, &val});
    }
    if (!meta.empty()) WriteMetadataBlock(out, spec, indent);
    out << "\n";
}

// ============================================================
// Layer metadata
// ============================================================

static void WriteLayerMetadata(std::ostringstream& out, const Spec& layerSpec) {
    bool hasAny = false;
    for (const auto& [name, val] : layerSpec.GetFields()) {
        if (name != "subLayers") hasAny = true;
    }
    // subLayers counts too
    if (layerSpec.GetField(FieldNames::subLayers)) hasAny = true;

    if (!hasAny) return;

    out << "(\n";
    for (const auto& [name, val] : layerSpec.GetFields()) {
        if (name == "subLayers") {
            if (auto* slp = val.Get<SubLayerPaths>()) {
                out << "    subLayers = [\n";
                for (size_t i = 0; i < slp->paths.size(); ++i) {
                    out << "        @" << slp->paths[i] << "@";
                    if (i < slp->offsets.size() && (slp->offsets[i].offset != 0.0 || slp->offsets[i].scale != 1.0)) {
                        out << " (offset = " << FormatDouble(slp->offsets[i].offset)
                            << "; scale = " << FormatDouble(slp->offsets[i].scale) << ")";
                    }
                    out << ",\n";
                }
                out << "    ]\n";
            }
        } else if (name == FieldNames::layerRelocates) {
            if (auto* relocates = val.Get<std::vector<Relocate>>()) {
                WriteRelocatesMetadata(out, *relocates);
            }
        } else {
            out << "    " << name.GetString() << " = " << FormatValue(val) << "\n";
        }
    }
    out << ")\n";
}

} // anonymous namespace

// ============================================================
// Public API
// ============================================================

std::string WriteUsda(const Layer& layer) {
    std::ostringstream out;
    out << "#usda 1.0\n";

    // Layer metadata
    WriteLayerMetadata(out, layer.GetLayerSpec());
    out << "\n";

    // Write root-level prims using Layer's child index
    Path rootPath = Path::AbsoluteRoot();
    auto rootChildren = layer.GetChildNames(rootPath);
    for (const auto& childName : rootChildren) {
        Path childPath = Path::Parse("/" + childName.GetString());
        const Spec* spec = layer.GetSpec(childPath);
        if (spec && spec->GetType() == SpecType::Prim) {
            WritePrimByPath(out, layer, childPath, childName.GetString(), *spec, "");
        }
    }

    return out.str();
}

bool WriteUsdaFile(const Layer& layer, const ResolvedLocation& location) {
    std::string usda = WriteUsda(layer);
    if (usda.empty()) return false;
    auto result = WriteResource(location,
                                reinterpret_cast<const uint8_t*>(usda.data()),
                                usda.size());
    return result.success;
}

bool WriteUsdaFile(const Layer& layer, const std::string& path) {
    return WriteUsdaFile(layer, ResolvedLocation::FromResolvedString(path));
}

} // namespace nanousd
