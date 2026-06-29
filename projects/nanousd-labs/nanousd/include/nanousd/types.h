// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <atomic>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>

namespace nanousd {

// --- Half-precision float (IEEE 754-2008 binary16) ---

struct Half {
    uint16_t bits = 0;

    Half() = default;
    explicit Half(float f);
    explicit operator float() const;

    bool operator==(Half o) const { return bits == o.bits; }
    bool operator!=(Half o) const { return bits != o.bits; }
};

namespace detail {

inline uint32_t RoundShiftRightToEven(uint32_t value, int shift) {
    if (shift <= 0) return value;
    if (shift > 31) return 0;

    const uint32_t shifted = value >> shift;
    const uint32_t remainderMask = (uint32_t{1} << shift) - 1u;
    const uint32_t remainder = value & remainderMask;
    const uint32_t halfway = uint32_t{1} << (shift - 1);
    if (remainder > halfway ||
        (remainder == halfway && (shifted & 1u))) {
        return shifted + 1u;
    }
    return shifted;
}

} // namespace detail

// Conversion helpers
inline Half::Half(float f) {
    uint32_t fb;
    std::memcpy(&fb, &f, 4);
    const uint32_t sign = (fb >> 16) & 0x8000u;
    const uint32_t exp = (fb >> 23) & 0xFFu;
    const uint32_t mantissa = fb & 0x7FFFFFu;

    if (exp == 0xFFu) {
        if (mantissa == 0) {
            bits = static_cast<uint16_t>(sign | 0x7C00u);
        } else {
            uint32_t payload = mantissa >> 13;
            payload |= 0x0200u; // keep the result a quiet NaN.
            bits = static_cast<uint16_t>(sign | 0x7C00u | payload);
        }
        return;
    }

    if (exp == 0) {
        bits = static_cast<uint16_t>(sign);
        return;
    }

    int32_t halfExp = static_cast<int32_t>(exp) - 127 + 15;
    if (halfExp <= 0) {
        const uint32_t significand = mantissa | 0x00800000u;
        const uint32_t rounded =
            detail::RoundShiftRightToEven(significand, 14 - halfExp);
        bits = static_cast<uint16_t>(sign | rounded);
        return;
    }

    if (halfExp > 30) {
        bits = static_cast<uint16_t>(sign | 0x7C00u);
        return;
    }

    uint32_t roundedMantissa =
        mantissa + 0x00000FFFu + ((mantissa >> 13) & 1u);
    if (roundedMantissa & 0x00800000u) {
        roundedMantissa = 0;
        ++halfExp;
        if (halfExp > 30) {
            bits = static_cast<uint16_t>(sign | 0x7C00u);
            return;
        }
    }

    bits = static_cast<uint16_t>(
        sign | (static_cast<uint32_t>(halfExp) << 10) |
        (roundedMantissa >> 13));
}

inline Half::operator float() const {
    uint16_t h = bits;
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x03FF;
    uint32_t fb;
    if (exp == 0) {
        if (mantissa == 0) {
            fb = sign;
        } else {
            int32_t unbiasedExp = -14;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --unbiasedExp;
            }
            mantissa &= 0x03FFu;
            fb = sign |
                 (static_cast<uint32_t>(unbiasedExp + 127) << 23) |
                 (mantissa << 13);
        }
    } else if (exp == 31) {
        fb = sign | 0x7F800000u | (mantissa << 13);
        if (mantissa != 0) fb |= 0x00400000u;
    } else {
        fb = sign | ((exp - 15 + 127) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &fb, 4);
    return result;
}

// --- Scalar type aliases per spec Section 6.2 ---

using Bool   = bool;
using UChar  = uint8_t;
using Int    = int32_t;
using UInt   = uint32_t;
using Int64  = int64_t;
using UInt64 = uint64_t;
using Float  = float;
using Double = double;
using String = std::string;
using Asset  = std::string;  // UTF-8, no C0/C1 controls
using TimeCode = double;

// --- Token per spec Section 6.2 ---
// A handle to an interned string. Equality and hashing are O(1) via pointer
// comparison against a process-wide pool. The underlying UTF-8 text is
// accessible via GetString() / GetText().

class NANOUSD_CORE_API Token {
public:
    Token() : ptr_(&EmptyString()) {}
    Token(const char* s)          : ptr_(&Intern(s)) {}
    Token(const std::string& s)   : ptr_(&Intern(s)) {}
    Token(std::string_view s)     : ptr_(&Intern(std::string(s))) {}

    Token& operator=(const char* s)          { ptr_ = &Intern(s); return *this; }
    Token& operator=(const std::string& s)   { ptr_ = &Intern(s); return *this; }
    Token& operator=(std::string_view s)     { ptr_ = &Intern(std::string(s)); return *this; }

    // O(1) equality — pointer compare on interned storage
    bool operator==(const Token& o) const { return ptr_ == o.ptr_; }
    bool operator!=(const Token& o) const { return ptr_ != o.ptr_; }
    bool operator<(const Token& o) const  { return *ptr_ < *o.ptr_; }

    // Convenient comparison with raw strings (interning cost on rhs)
    bool operator==(const std::string& s) const { return *ptr_ == s; }
    bool operator!=(const std::string& s) const { return *ptr_ != s; }
    bool operator==(const char* s) const        { return *ptr_ == s; }
    bool operator!=(const char* s) const        { return *ptr_ != s; }

    const std::string& GetString() const { return *ptr_; }
    const char* GetText() const          { return ptr_->c_str(); }
    bool IsEmpty() const                 { return ptr_->empty(); }

    // Implicit conversion to const std::string& for interop
    operator const std::string&() const { return *ptr_; }

    // O(1) hash — hash the pointer, not the string contents
    struct Hash {
        size_t operator()(const Token& t) const noexcept {
            return std::hash<const std::string*>{}(t.ptr_);
        }
    };

private:
    // Process-wide sharded intern pool.  Strings are never removed once
    // interned.  16 shards reduce mutex contention when multiple threads
    // intern tokens concurrently (e.g. parallel sublayer gathering).
    static constexpr size_t kNumShards = 16;

    struct Shard {
        std::mutex mutex;
        std::unordered_set<std::string> pool;
    };

    // Declared here, defined in types.cpp. Out-of-line because their
    // function-local statics must be a single instance per process — with
    // inline definitions on Windows each DLL/EXE TU would get its own
    // intern pool, breaking Token pointer-identity across DLL boundaries.
    static std::array<Shard, kNumShards>& GetShards();
    static const std::string& EmptyString();

    // NANOUSD_TRACE_TOKEN=1 routes through InternTraced (out-of-line) which
    // times wait-vs-held per intern call. Hot-path overhead when disabled
    // is one relaxed atomic load per intern. Defined out-of-line so the
    // counter is shared across DLL boundaries.
    static std::atomic<bool>& TraceEnabledFlag();
    static const std::string& InternTraced(const std::string& s);

    // Intern remains inline for per-call speed; it calls the exported
    // GetShards() which guarantees a single shared pool.
    static const std::string& Intern(const std::string& s) {
        if (TraceEnabledFlag().load(std::memory_order_relaxed))
            return InternTraced(s);
        size_t h = std::hash<std::string>{}(s);
        auto& shard = GetShards()[h & (kNumShards - 1)];
        std::lock_guard<std::mutex> lock(shard.mutex);
        auto it = shard.pool.find(s);
        if (it != shard.pool.end()) return *it;
        return *shard.pool.insert(s).first;
    }

    const std::string* ptr_;
};

// --- Dimensioned types per spec Section 6.3 ---

template <typename T, size_t N>
struct Vec {
    std::array<T, N> data{};

    T& operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }

    bool operator==(const Vec& o) const { return data == o.data; }
    bool operator!=(const Vec& o) const { return data != o.data; }
};

using GfVec2h = Vec<Half, 2>;
using GfVec3h = Vec<Half, 3>;
using GfVec4h = Vec<Half, 4>;
using GfVec2f = Vec<Float, 2>;
using GfVec3f = Vec<Float, 3>;
using GfVec4f = Vec<Float, 4>;
using GfVec2d = Vec<Double, 2>;
using GfVec3d = Vec<Double, 3>;
using GfVec4d = Vec<Double, 4>;
using GfVec2i = Vec<Int, 2>;
using GfVec3i = Vec<Int, 3>;
using GfVec4i = Vec<Int, 4>;

// Quaternions: (i, j, k, r) - three imaginary followed by real.
// Distinct type from Vec4 to allow separate Value constructors.
template <typename T>
struct Quat {
    std::array<T, 4> data{};  // imaginary i,j,k then real r

    T& operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }

    bool operator==(const Quat& o) const { return data == o.data; }
    bool operator!=(const Quat& o) const { return data != o.data; }
};

using GfQuath = Quat<Half>;
using GfQuatf = Quat<Float>;
using GfQuatd = Quat<Double>;

// Matrices: row-major storage
template <typename T, size_t R, size_t C>
struct Matrix {
    std::array<T, R * C> data{};

    T& operator()(size_t r, size_t c) { return data[r * C + c]; }
    const T& operator()(size_t r, size_t c) const { return data[r * C + c]; }

    bool operator==(const Matrix& o) const { return data == o.data; }
    bool operator!=(const Matrix& o) const { return data != o.data; }

    static Matrix Identity() {
        Matrix m;
        for (size_t i = 0; i < (R < C ? R : C); ++i)
            m(i, i) = T(1);
        return m;
    }
};

using GfMatrix2d = Matrix<Double, 2, 2>;
using GfMatrix3d = Matrix<Double, 3, 3>;
using GfMatrix4d = Matrix<Double, 4, 4>;

// --- Semantic role per spec Section 6.5 ---

enum class Role {
    None,
    Color,
    Normal,
    Point,
    Vector,
    Frame,
    TexCoord,
    Group,
};

// --- Type identifier enum for runtime type dispatch ---

enum class TypeId {
    Unknown,
    // Scalar
    Bool, UChar, Int, UInt, Int64, UInt64,
    Half, Float, Double,
    String, Token, Asset,
    TimeCode,
    // Dimensioned
    Half2, Half3, Half4,
    Float2, Float3, Float4,
    Double2, Double3, Double4,
    Int2, Int3, Int4,
    Quath, Quatf, Quatd,
    Matrix2d, Matrix3d, Matrix4d,
    // Algebraic
    Opaque,
    // Container
    Dictionary,
    // Specialized (§7.4.2.4 — carries a Spline struct via Value::Get<Spline>)
    Spline,
    // Arrays use the scalar/dimensioned TypeId + isArray flag
};

// --- Specifier enum per spec Section 7.6.2.1.1 ---

enum class Specifier {
    Def,   // concrete defining
    Over,  // sparse override
    Class, // abstract defining
};

// --- Variability enum per spec Section 7.6.4.1.2 ---

enum class Variability {
    Varying,
    Uniform,
};

} // namespace nanousd

// std::hash specialization so Token works in std containers
template <>
struct std::hash<nanousd::Token> {
    size_t operator()(const nanousd::Token& t) const noexcept {
        return nanousd::Token::Hash{}(t);
    }
};
