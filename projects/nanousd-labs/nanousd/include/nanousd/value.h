// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.h"
#include "path.h"

#include <any>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace nanousd {

// --- Reference / Payload per spec Sections 7.6.2.3.1 / 7.6.2.3.2 ---

struct Reference {
    std::optional<Asset> assetPath;
    std::optional<Path> primPath;
    double offset = 0.0;
    double scale  = 1.0;

    bool operator==(const Reference& o) const {
        return assetPath == o.assetPath && primPath == o.primPath &&
               offset == o.offset && scale == o.scale;
    }
    bool operator!=(const Reference& o) const { return !(*this == o); }
};

using Payload = Reference;  // same structure per spec

// --- Retiming per spec Section 7.6.1.2.2 ---

struct Retiming {
    double offset = 0.0;
    double scale  = 1.0;
};

// --- SubLayerPaths per spec Section 7.6.1.2.1 ---
// Combines sublayer asset paths with their layer offsets in a single structure.

struct SubLayerPaths {
    std::vector<std::string> paths;
    std::vector<Retiming> offsets;  // parallel array, same length
};

// --- Relocates per spec Section 7.6.1.2.4 ---

struct Relocate {
    Path sourcePath;
    std::optional<Path> targetPath;
};

} // namespace nanousd

// std::hash specialization for Reference (required by ListOp's internal unordered_set)
template <>
struct std::hash<nanousd::Reference> {
    size_t operator()(const nanousd::Reference& r) const noexcept {
        size_t h = 0;
        if (r.assetPath) h ^= std::hash<std::string>{}(*r.assetPath);
        if (r.primPath)  h ^= std::hash<nanousd::Path>{}(*r.primPath) << 1;
        h ^= std::hash<double>{}(r.offset) << 2;
        h ^= std::hash<double>{}(r.scale) << 3;
        return h;
    }
};

// ListOp must be included after hash specialization so ListOp<Reference> can use unordered_set
#include "listop.h"

namespace nanousd {

using ListOpReference = ListOp<Reference>;

// Forward declaration
class Value;

// --- Dictionary per spec Section 6.6.2 ---
// Unordered map of strings to heterogeneous values.
// Supports scalar, dimensioned, arrays, and nested dictionaries.

using Dictionary = std::map<std::string, Value>;

// Spec §6.6.2.1 — combine two dictionaries under the composition
// strength ordering. The returned dictionary contains:
//   1. Keys in `stronger` not in `weaker`     → stronger's value
//   2. Keys in both where values aren't both dicts → stronger's value
//   3. Keys in both where both values are dicts → recursive combine
//   4. Keys in `weaker` not in `stronger`     → weaker's value
// Pure function; does not mutate its inputs.
NANOUSD_CORE_API Dictionary CombineDicts(const Dictionary& stronger,
                                         const Dictionary& weaker);

// --- UsdTimeCode per spec Section 12.3 ---
// Wraps a double time value. A special "default" sentinel (NaN) represents
// queries for the non-time-varying default value.

class UsdTimeCode {
public:
    // Default time (non-time-varying query)
    UsdTimeCode() : time_(std::numeric_limits<double>::quiet_NaN()) {}
    // Specific time
    UsdTimeCode(double t) : time_(t) {}

    static UsdTimeCode Default() { return UsdTimeCode(); }
    static UsdTimeCode EarliestTime() { return UsdTimeCode(-std::numeric_limits<double>::infinity()); }

    bool IsDefault() const { return std::isnan(time_); }
    double GetValue() const { return time_; }

private:
    double time_;
};

// --- Value Block sentinel per spec Section 7.6.4.2.1 ---

struct ValueBlock {
    bool operator==(const ValueBlock&) const { return true; }
    bool operator!=(const ValueBlock&) const { return false; }
};

// --- Spline types per spec Section 7.4.2.4 ---
// Declared here (ahead of Value) so Value(Spline) can be inline.
// Storage type is double: §7.4.2.4 permits half/float/double, and we
// widen to double at parse / narrow at output — see usda_writer.cpp.

enum class CurveType { Bezier, Hermite };

enum class InterpolationMode { None, Held, Linear, Curve };

enum class ExtrapolationMode {
    None, Held, Linear, Sloped,
    LoopRepeat, LoopReset, LoopOscillate,
};

struct LoopParameters {
    double protoStart = 0.0;
    double protoEnd   = 0.0;
    int numPreLoops   = 0;
    int numPostLoops  = 0;
    double valueOffset = 0.0;
};

struct SplineKnot {
    double time = 0.0;
    double preTangentSlope = 0.0;
    double preTangentWidth = 0.0;
    double postTangentSlope = 0.0;
    double postTangentWidth = 0.0;
    InterpolationMode nextInterpolationMode = InterpolationMode::Held;
    double value = 0.0;
    double preValue = 0.0;
};

struct Spline {
    CurveType curveType = CurveType::Bezier;
    ExtrapolationMode preExtrapolationMode = ExtrapolationMode::Held;
    double preExtrapolationSlope = 0.0;
    ExtrapolationMode postExtrapolationMode = ExtrapolationMode::Held;
    double postExtrapolationSlope = 0.0;
    LoopParameters loopParameters;
    std::vector<SplineKnot> knots;
};

// --- Value: runtime variant type for USD values ---
// Covers all foundational scalar, dimensioned, and array types,
// plus Dictionary and ValueBlock.

class Value {
public:
    Value() = default;

    // Construct from any supported type
    Value(Bool v)    : type_(TypeId::Bool),    data_(v) {}
    Value(UChar v)   : type_(TypeId::UChar),   data_(v) {}
    Value(Int v)     : type_(TypeId::Int),     data_(v) {}
    Value(UInt v)    : type_(TypeId::UInt),    data_(v) {}
    Value(Int64 v)   : type_(TypeId::Int64),   data_(v) {}
    Value(UInt64 v)  : type_(TypeId::UInt64),  data_(v) {}
    Value(Half v)    : type_(TypeId::Half),    data_(v) {}
    Value(Float v)   : type_(TypeId::Float),   data_(v) {}
    Value(Double v)  : type_(TypeId::Double),  data_(v) {}
    Value(const String& v) : type_(TypeId::String), data_(v) {}
    Value(String&& v)      : type_(TypeId::String), data_(std::move(v)) {}
    Value(const char* v)   : type_(TypeId::String), data_(String(v)) {}
    Value(Token v)         : type_(TypeId::Token),  data_(v) {}

    // Tagged constructor for Asset (same underlying type as String).
    struct AssetTag {};
    Value(AssetTag, const std::string& v) : type_(TypeId::Asset), data_(v) {}
    Value(AssetTag, std::string&& v) : type_(TypeId::Asset), data_(std::move(v)) {}

    Value(TimeCode v, std::nullptr_t) : type_(TypeId::TimeCode), data_(v) {}

    // Dimensioned types — half precision
    Value(GfVec2h v)    : type_(TypeId::Half2),    data_(v) {}
    Value(GfVec3h v)    : type_(TypeId::Half3),    data_(v) {}
    Value(GfVec4h v)    : type_(TypeId::Half4),    data_(v) {}
    Value(GfQuath v)    : type_(TypeId::Quath),    data_(v) {}

    // Dimensioned types
    Value(GfVec2f v)    : type_(TypeId::Float2),   data_(v) {}
    Value(GfVec3f v)    : type_(TypeId::Float3),   data_(v) {}
    Value(GfVec4f v)    : type_(TypeId::Float4),   data_(v) {}
    Value(GfVec2d v)    : type_(TypeId::Double2),  data_(v) {}
    Value(GfVec3d v)    : type_(TypeId::Double3),  data_(v) {}
    Value(GfVec4d v)    : type_(TypeId::Double4),  data_(v) {}
    Value(GfVec2i v)    : type_(TypeId::Int2),     data_(v) {}
    Value(GfVec3i v)    : type_(TypeId::Int3),     data_(v) {}
    Value(GfVec4i v)    : type_(TypeId::Int4),     data_(v) {}
    Value(GfMatrix2d v) : type_(TypeId::Matrix2d), data_(v) {}
    Value(GfMatrix3d v) : type_(TypeId::Matrix3d), data_(v) {}
    Value(GfMatrix4d v) : type_(TypeId::Matrix4d), data_(v) {}
    Value(GfQuatf v)    : type_(TypeId::Quatf),    data_(v) {}
    Value(GfQuatd v)    : type_(TypeId::Quatd),    data_(v) {}

    // Dictionary
    Value(Dictionary v) : type_(TypeId::Dictionary), data_(std::move(v)) {}

    // Spline specialized type (§7.4.2.4)
    Value(Spline v) : type_(TypeId::Spline), data_(std::move(v)) {}

    // ValueBlock sentinel
    Value(ValueBlock v) : type_(TypeId::Unknown), isBlock_(true), data_(v) {}

    // Composition types (stored with Unknown type, retrieved via Get<T>())
    Value(SubLayerPaths v) : type_(TypeId::Unknown), data_(std::move(v)) {}
    Value(std::vector<Relocate> v) : type_(TypeId::Unknown), data_(std::move(v)) {}
    Value(ListOp<Reference> v) : type_(TypeId::Unknown), data_(std::move(v)) {}
    Value(ListOp<std::string> v) : type_(TypeId::Unknown), data_(std::move(v)) {}
    Value(ListOp<Path> v) : type_(TypeId::Unknown), data_(std::move(v)) {}
    Value(ListOp<Token> v) : type_(TypeId::Unknown), data_(std::move(v)) {}

    // --- Array constructors ---
    // Use a tagged constructor to disambiguate arrays

    struct ArrayTag {};
    template <typename T>
    Value(ArrayTag, TypeId arrayElemType, std::vector<T> v)
        : type_(arrayElemType), isArray_(true), arraySize_(v.size()), data_(std::move(v)) {}

    // --- Lazy value support ---
    // A lazy thunk populates data_ and arraySize_ on first access.
    // The thunk captures a shared_ptr to the file buffer + lookup tables,
    // keeping them alive as long as any lazy Value references them.
    using LazyThunk = std::function<void(std::any& /*data*/, size_t& /*arraySize*/)>;

    // Factory for lazy values — type and isArray are known from the ValueRep
    // without decoding.  The thunk is called once on first Get/ArraySize access.
    static Value MakeLazy(TypeId type, bool isArray,
                          std::shared_ptr<LazyThunk> thunk) {
        Value v;
        v.type_ = type;
        v.isArray_ = isArray;
        v.lazy_ = std::move(thunk);
        return v;
    }

    // --- Queries ---

    TypeId GetTypeId() const { return type_; }
    bool IsArray() const { return isArray_; }
    size_t ArraySize() const { EnsureDecoded(); return arraySize_; }
    bool IsEmpty() const { return !lazy_ && data_.has_value() == false && !isBlock_; }
    bool IsBlock() const { return isBlock_; }

    // Typed access — returns nullptr if wrong type
    template <typename T>
    const T* Get() const {
        EnsureDecoded();
        return std::any_cast<T>(&data_);
    }

    template <typename T>
    T* Get() {
        EnsureDecoded();
        return std::any_cast<T>(&data_);
    }

    // Semantic role (set externally based on typeName)
    Role GetRole() const { return role_; }
    void SetRole(Role r) { role_ = r; }

private:
    void EnsureDecoded() const {
        if (lazy_) {
            auto thunk = std::move(const_cast<Value*>(this)->lazy_);
            (*thunk)(const_cast<std::any&>(data_),
                     const_cast<size_t&>(arraySize_));
        }
    }

    TypeId type_ = TypeId::Unknown;
    Role role_ = Role::None;
    bool isArray_ = false;
    bool isBlock_ = false;
    mutable size_t arraySize_ = 0;
    mutable std::any data_;
    mutable std::shared_ptr<LazyThunk> lazy_;
};

// --- TimeSamples per spec Section 7.6.4.2.2 ---
// Ordered map from time (double) to Value.

using TimeSamples = std::map<double, Value>;

} // namespace nanousd
