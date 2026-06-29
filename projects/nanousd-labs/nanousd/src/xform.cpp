// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanousd/xform.h"

#include <cmath>
#include <string>
#include <vector>

namespace nanousd {

// ============================================================
// Internal helpers
// ============================================================

namespace {

constexpr double kDegToRad = 3.14159265358979323846 / 180.0;

// 4x4 identity
GfMatrix4d Identity() { return GfMatrix4d::Identity(); }

// Row-major 4x4 multiply: out = a * b
GfMatrix4d Mul(const GfMatrix4d& a, const GfMatrix4d& b) {
    GfMatrix4d out;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            double s = 0;
            for (int k = 0; k < 4; ++k)
                s += a(r, k) * b(k, c);
            out(r, c) = s;
        }
    return out;
}

// --- Op type identification ---

enum class OpType {
    Translate, TranslateX, TranslateY, TranslateZ,
    Scale, ScaleX, ScaleY, ScaleZ,
    RotateX, RotateY, RotateZ,
    RotateXYZ, RotateXZY, RotateYXZ, RotateYZX, RotateZXY, RotateZYX,
    Orient,
    Transform,
    Invalid
};

struct ParsedOp {
    OpType type = OpType::Invalid;
    std::string suffix;
    bool isInverse = false;
    std::string attrName;  // full attribute name without !invert! prefix
};

OpType TokenToOpType(const std::string& tok) {
    if (tok == "translate")  return OpType::Translate;
    if (tok == "translateX") return OpType::TranslateX;
    if (tok == "translateY") return OpType::TranslateY;
    if (tok == "translateZ") return OpType::TranslateZ;
    if (tok == "scale")      return OpType::Scale;
    if (tok == "scaleX")     return OpType::ScaleX;
    if (tok == "scaleY")     return OpType::ScaleY;
    if (tok == "scaleZ")     return OpType::ScaleZ;
    if (tok == "rotateX")    return OpType::RotateX;
    if (tok == "rotateY")    return OpType::RotateY;
    if (tok == "rotateZ")    return OpType::RotateZ;
    if (tok == "rotateXYZ")  return OpType::RotateXYZ;
    if (tok == "rotateXZY")  return OpType::RotateXZY;
    if (tok == "rotateYXZ")  return OpType::RotateYXZ;
    if (tok == "rotateYZX")  return OpType::RotateYZX;
    if (tok == "rotateZXY")  return OpType::RotateZXY;
    if (tok == "rotateZYX")  return OpType::RotateZYX;
    if (tok == "orient")     return OpType::Orient;
    if (tok == "transform")  return OpType::Transform;
    return OpType::Invalid;
}

// Parse a token from xformOpOrder into its components.
// Input: "xformOp:translate:pivot" or "!invert!xformOp:rotateXYZ"
ParsedOp ParseOpToken(const std::string& token) {
    ParsedOp result;

    std::string tok = token;

    // Check for !invert! prefix
    const std::string invertPrefix = "!invert!";
    if (tok.size() > invertPrefix.size() &&
        tok.compare(0, invertPrefix.size(), invertPrefix) == 0) {
        result.isInverse = true;
        tok = tok.substr(invertPrefix.size());
    }

    // Must start with "xformOp:"
    const std::string prefix = "xformOp:";
    if (tok.size() <= prefix.size() ||
        tok.compare(0, prefix.size(), prefix) != 0) {
        return result;  // Invalid
    }

    // The attribute name is the token without !invert!
    result.attrName = tok;

    // Extract op type and optional suffix after "xformOp:"
    std::string rest = tok.substr(prefix.size());

    // Find first colon for suffix separation.
    // But op types like "rotateXYZ" don't contain colons, so the first colon
    // after the op type name is the suffix delimiter.
    // Strategy: try longest matching op type first.
    // Op type names: translate, translateX/Y/Z, scale, scaleX/Y/Z,
    //   rotateX/Y/Z, rotateXYZ/XZY/YXZ/YZX/ZXY/ZYX, orient, transform
    // Try to match the beginning of `rest` against known op types.
    static const char* opNames[] = {
        // Longer names first to avoid prefix matches
        "translateX", "translateY", "translateZ", "translate",
        "rotateXYZ", "rotateXZY", "rotateYXZ", "rotateYZX",
        "rotateZXY", "rotateZYX",
        "rotateX", "rotateY", "rotateZ",
        "scaleX", "scaleY", "scaleZ", "scale",
        "orient", "transform",
        nullptr
    };

    for (const char** p = opNames; *p; ++p) {
        std::string name = *p;
        if (rest.size() >= name.size() &&
            rest.compare(0, name.size(), name) == 0) {
            // Check that the next char is either end-of-string or ':'
            if (rest.size() == name.size()) {
                result.type = TokenToOpType(name);
                return result;
            }
            if (rest[name.size()] == ':') {
                result.type = TokenToOpType(name);
                result.suffix = rest.substr(name.size() + 1);
                return result;
            }
        }
    }

    return result;  // Invalid
}

// --- Matrix builders for each op type ---

GfMatrix4d MakeTranslate(double tx, double ty, double tz) {
    auto m = Identity();
    m(3, 0) = tx; m(3, 1) = ty; m(3, 2) = tz;
    return m;
}

GfMatrix4d MakeScale(double sx, double sy, double sz) {
    auto m = Identity();
    m(0, 0) = sx; m(1, 1) = sy; m(2, 2) = sz;
    return m;
}

GfMatrix4d MakeRotateX(double degrees) {
    double r = degrees * kDegToRad;
    double c = std::cos(r), s = std::sin(r);
    auto m = Identity();
    m(1, 1) = c;  m(1, 2) = s;
    m(2, 1) = -s; m(2, 2) = c;
    return m;
}

GfMatrix4d MakeRotateY(double degrees) {
    double r = degrees * kDegToRad;
    double c = std::cos(r), s = std::sin(r);
    auto m = Identity();
    m(0, 0) = c;  m(0, 2) = -s;
    m(2, 0) = s;  m(2, 2) = c;
    return m;
}

GfMatrix4d MakeRotateZ(double degrees) {
    double r = degrees * kDegToRad;
    double c = std::cos(r), s = std::sin(r);
    auto m = Identity();
    m(0, 0) = c;  m(0, 1) = s;
    m(1, 0) = -s; m(1, 1) = c;
    return m;
}

// Orient quaternion to matrix.
// Internal quat storage: {i, j, k, r} at indices [0,1,2,3]
// Spec formula uses (x,y,z,r) = (i,j,k,r)
GfMatrix4d MakeOrient(double i, double j, double k, double r) {
    auto m = Identity();
    m(0, 0) = 1.0 - 2.0*(j*j + k*k);
    m(0, 1) = 2.0*(i*j + k*r);
    m(0, 2) = 2.0*(i*k - j*r);
    m(1, 0) = 2.0*(i*j - k*r);
    m(1, 1) = 1.0 - 2.0*(k*k + i*i);
    m(1, 2) = 2.0*(j*k + i*r);
    m(2, 0) = 2.0*(i*k + j*r);
    m(2, 1) = 2.0*(j*k - i*r);
    m(2, 2) = 1.0 - 2.0*(j*j + i*i);
    return m;
}

// Read a scalar double from an attribute, handling float/half/double precision
double ReadScalar(const UsdAttribute& attr, UsdTimeCode time) {
    auto resolved = attr.Get(time);
    if (!resolved.found) return 0.0;
    if (auto* d = resolved.value.Get<Double>()) return *d;
    if (auto* f = resolved.value.Get<Float>()) return static_cast<double>(*f);
    if (auto* h = resolved.value.Get<Half>()) return static_cast<double>(static_cast<float>(*h));
    return 0.0;
}

// Read a 3-component vector, handling double3/float3/half3
bool ReadVec3(const UsdAttribute& attr, UsdTimeCode time, double out[3]) {
    auto resolved = attr.Get(time);
    if (!resolved.found) return false;
    if (auto* v = resolved.value.Get<GfVec3d>()) {
        out[0] = (*v)[0]; out[1] = (*v)[1]; out[2] = (*v)[2]; return true;
    }
    if (auto* v = resolved.value.Get<GfVec3f>()) {
        out[0] = (*v)[0]; out[1] = (*v)[1]; out[2] = (*v)[2]; return true;
    }
    if (auto* v = resolved.value.Get<GfVec3h>()) {
        out[0] = static_cast<float>((*v)[0]);
        out[1] = static_cast<float>((*v)[1]);
        out[2] = static_cast<float>((*v)[2]);
        return true;
    }
    return false;
}

// Read a quaternion, handling quatd/quatf/quath
// Returns {i, j, k, r} in internal order
bool ReadQuat(const UsdAttribute& attr, UsdTimeCode time, double out[4]) {
    auto resolved = attr.Get(time);
    if (!resolved.found) return false;
    if (auto* q = resolved.value.Get<GfQuatd>()) {
        out[0] = (*q)[0]; out[1] = (*q)[1]; out[2] = (*q)[2]; out[3] = (*q)[3];
        return true;
    }
    if (auto* q = resolved.value.Get<GfQuatf>()) {
        out[0] = (*q)[0]; out[1] = (*q)[1]; out[2] = (*q)[2]; out[3] = (*q)[3];
        return true;
    }
    if (auto* q = resolved.value.Get<GfQuath>()) {
        out[0] = static_cast<float>((*q)[0]);
        out[1] = static_cast<float>((*q)[1]);
        out[2] = static_cast<float>((*q)[2]);
        out[3] = static_cast<float>((*q)[3]);
        return true;
    }
    return false;
}

// Invert a 4x4 matrix (general case via cofactor expansion).
// Returns identity if singular.
GfMatrix4d Invert(const GfMatrix4d& m) {
    // Use the standard 4x4 adjugate/determinant method
    double inv[16];
    const double* s = m.data.data();

    inv[0]  =  s[5]*s[10]*s[15] - s[5]*s[11]*s[14] - s[9]*s[6]*s[15]
             + s[9]*s[7]*s[14] + s[13]*s[6]*s[11] - s[13]*s[7]*s[10];
    inv[4]  = -s[4]*s[10]*s[15] + s[4]*s[11]*s[14] + s[8]*s[6]*s[15]
             - s[8]*s[7]*s[14] - s[12]*s[6]*s[11] + s[12]*s[7]*s[10];
    inv[8]  =  s[4]*s[9]*s[15] - s[4]*s[11]*s[13] - s[8]*s[5]*s[15]
             + s[8]*s[7]*s[13] + s[12]*s[5]*s[11] - s[12]*s[7]*s[9];
    inv[12] = -s[4]*s[9]*s[14] + s[4]*s[10]*s[13] + s[8]*s[5]*s[14]
             - s[8]*s[6]*s[13] - s[12]*s[5]*s[10] + s[12]*s[6]*s[9];

    double det = s[0]*inv[0] + s[1]*inv[4] + s[2]*inv[8] + s[3]*inv[12];
    if (std::abs(det) < 1e-15) return Identity();

    inv[1]  = -s[1]*s[10]*s[15] + s[1]*s[11]*s[14] + s[9]*s[2]*s[15]
             - s[9]*s[3]*s[14] - s[13]*s[2]*s[11] + s[13]*s[3]*s[10];
    inv[5]  =  s[0]*s[10]*s[15] - s[0]*s[11]*s[14] - s[8]*s[2]*s[15]
             + s[8]*s[3]*s[14] + s[12]*s[2]*s[11] - s[12]*s[3]*s[10];
    inv[9]  = -s[0]*s[9]*s[15] + s[0]*s[11]*s[13] + s[8]*s[1]*s[15]
             - s[8]*s[3]*s[13] - s[12]*s[1]*s[11] + s[12]*s[3]*s[9];
    inv[13] =  s[0]*s[9]*s[14] - s[0]*s[10]*s[13] - s[8]*s[1]*s[14]
             + s[8]*s[2]*s[13] + s[12]*s[1]*s[10] - s[12]*s[2]*s[9];

    inv[2]  =  s[1]*s[6]*s[15] - s[1]*s[7]*s[14] - s[5]*s[2]*s[15]
             + s[5]*s[3]*s[14] + s[13]*s[2]*s[7] - s[13]*s[3]*s[6];
    inv[6]  = -s[0]*s[6]*s[15] + s[0]*s[7]*s[14] + s[4]*s[2]*s[15]
             - s[4]*s[3]*s[14] - s[12]*s[2]*s[7] + s[12]*s[3]*s[6];
    inv[10] =  s[0]*s[5]*s[15] - s[0]*s[7]*s[13] - s[4]*s[1]*s[15]
             + s[4]*s[3]*s[13] + s[12]*s[1]*s[7] - s[12]*s[3]*s[5];
    inv[14] = -s[0]*s[5]*s[14] + s[0]*s[6]*s[13] + s[4]*s[1]*s[14]
             - s[4]*s[2]*s[13] - s[12]*s[1]*s[6] + s[12]*s[2]*s[5];

    inv[3]  = -s[1]*s[6]*s[11] + s[1]*s[7]*s[10] + s[5]*s[2]*s[11]
             - s[5]*s[3]*s[10] - s[9]*s[2]*s[7] + s[9]*s[3]*s[6];
    inv[7]  =  s[0]*s[6]*s[11] - s[0]*s[7]*s[10] - s[4]*s[2]*s[11]
             + s[4]*s[3]*s[10] + s[8]*s[2]*s[7] - s[8]*s[3]*s[6];
    inv[11] = -s[0]*s[5]*s[11] + s[0]*s[7]*s[9] + s[4]*s[1]*s[11]
             - s[4]*s[3]*s[9] - s[8]*s[1]*s[7] + s[8]*s[3]*s[5];
    inv[15] =  s[0]*s[5]*s[10] - s[0]*s[6]*s[9] - s[4]*s[1]*s[10]
             + s[4]*s[2]*s[9] + s[8]*s[1]*s[6] - s[8]*s[2]*s[5];

    double invDet = 1.0 / det;
    GfMatrix4d result;
    for (int i = 0; i < 16; ++i) result.data[i] = inv[i] * invDet;
    return result;
}

// Compute the matrix for a single op.
// For inverse ops, finds the paired non-inverse op on the prim and inverts its matrix.
GfMatrix4d ComputeOpMatrix(const ParsedOp& op, const UsdPrim& prim, UsdTimeCode time) {
    UsdAttribute attr = prim.GetAttribute(op.attrName);
    if (!attr.IsValid() && !op.isInverse) return Identity();

    // For inverse ops, compute the paired op's matrix and invert
    if (op.isInverse) {
        ParsedOp paired = op;
        paired.isInverse = false;
        return Invert(ComputeOpMatrix(paired, prim, time));
    }

    switch (op.type) {
    case OpType::Translate: {
        double v[3] = {};
        if (!ReadVec3(attr, time, v)) return Identity();
        return MakeTranslate(v[0], v[1], v[2]);
    }
    case OpType::TranslateX: return MakeTranslate(ReadScalar(attr, time), 0, 0);
    case OpType::TranslateY: return MakeTranslate(0, ReadScalar(attr, time), 0);
    case OpType::TranslateZ: return MakeTranslate(0, 0, ReadScalar(attr, time));

    case OpType::Scale: {
        double v[3] = {};
        if (!ReadVec3(attr, time, v)) return Identity();
        return MakeScale(v[0], v[1], v[2]);
    }
    case OpType::ScaleX: return MakeScale(ReadScalar(attr, time), 1, 1);
    case OpType::ScaleY: return MakeScale(1, ReadScalar(attr, time), 1);
    case OpType::ScaleZ: return MakeScale(1, 1, ReadScalar(attr, time));

    case OpType::RotateX: return MakeRotateX(ReadScalar(attr, time));
    case OpType::RotateY: return MakeRotateY(ReadScalar(attr, time));
    case OpType::RotateZ: return MakeRotateZ(ReadScalar(attr, time));

    case OpType::RotateXYZ: {
        double v[3] = {};
        if (!ReadVec3(attr, time, v)) return Identity();
        return Mul(MakeRotateX(v[0]), Mul(MakeRotateY(v[1]), MakeRotateZ(v[2])));
    }
    case OpType::RotateXZY: {
        double v[3] = {};
        if (!ReadVec3(attr, time, v)) return Identity();
        return Mul(MakeRotateX(v[0]), Mul(MakeRotateZ(v[2]), MakeRotateY(v[1])));
    }
    case OpType::RotateYXZ: {
        double v[3] = {};
        if (!ReadVec3(attr, time, v)) return Identity();
        return Mul(MakeRotateY(v[1]), Mul(MakeRotateX(v[0]), MakeRotateZ(v[2])));
    }
    case OpType::RotateYZX: {
        double v[3] = {};
        if (!ReadVec3(attr, time, v)) return Identity();
        return Mul(MakeRotateY(v[1]), Mul(MakeRotateZ(v[2]), MakeRotateX(v[0])));
    }
    case OpType::RotateZXY: {
        double v[3] = {};
        if (!ReadVec3(attr, time, v)) return Identity();
        return Mul(MakeRotateZ(v[2]), Mul(MakeRotateX(v[0]), MakeRotateY(v[1])));
    }
    case OpType::RotateZYX: {
        double v[3] = {};
        if (!ReadVec3(attr, time, v)) return Identity();
        return Mul(MakeRotateZ(v[2]), Mul(MakeRotateY(v[1]), MakeRotateX(v[0])));
    }

    case OpType::Orient: {
        double q[4] = {};  // {i, j, k, r}
        if (!ReadQuat(attr, time, q)) return Identity();
        return MakeOrient(q[0], q[1], q[2], q[3]);
    }

    case OpType::Transform: {
        auto resolved = attr.Get(time);
        if (!resolved.found) return Identity();
        if (auto* m = resolved.value.Get<GfMatrix4d>()) return *m;
        return Identity();
    }

    default:
        return Identity();
    }
}

// Read xformOpOrder token array from a prim
std::vector<std::string> GetXformOpOrder(const UsdPrim& prim) {
    auto attr = prim.GetAttribute("xformOpOrder");
    if (!attr.IsValid()) return {};
    auto* val = attr.GetDefault();
    if (!val || !val->IsArray()) return {};
    // Token arrays (spec-correct) or string arrays (legacy)
    if (auto* tokArr = val->Get<std::vector<Token>>()) {
        std::vector<std::string> result;
        result.reserve(tokArr->size());
        for (const auto& t : *tokArr) result.push_back(t.GetString());
        return result;
    }
    if (auto* strArr = val->Get<std::vector<std::string>>()) {
        return *strArr;
    }
    return {};
}

} // anonymous namespace

// ============================================================
// Public API
// ============================================================

GfMatrix4d ComputeLocalTransform(const UsdPrim& prim, UsdTimeCode time,
                                  bool* resetXformStack) {
    if (resetXformStack) *resetXformStack = false;

    auto order = GetXformOpOrder(prim);
    if (order.empty()) return Identity();

    // Check for !resetXformStack! as first token
    size_t start = 0;
    if (order[0] == "!resetXformStack!") {
        if (resetXformStack) *resetXformStack = true;
        start = 1;
    }

    if (start >= order.size()) return Identity();

    // Row-vector post-multiply convention: P' = P * M
    // xformOpOrder = [translate, rotateXYZ, scale] means:
    //   P' = P * scale * rotateXYZ * translate
    //   localMatrix = scale * rotateXYZ * translate
    // Last op in array is leftmost in the product.
    // Iterate forward, left-multiplying each op onto the accumulator.
    GfMatrix4d result = Identity();
    for (size_t i = start; i < order.size(); ++i) {
        auto parsed = ParseOpToken(order[i]);
        if (parsed.type == OpType::Invalid) continue;
        auto opMatrix = ComputeOpMatrix(parsed, prim, time);
        result = Mul(opMatrix, result);
    }

    return result;
}

bool HasResetXformStack(const UsdPrim& prim) {
    auto order = GetXformOpOrder(prim);
    return !order.empty() && order[0] == "!resetXformStack!";
}

} // namespace nanousd
