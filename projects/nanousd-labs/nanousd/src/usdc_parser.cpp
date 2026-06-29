// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanousd/usdc_parser.h"
#include "nanousd/decode_backing.h"
#include "nanousd/resource.h"
#include "nanousd/value.h"
#include "nanousd/spec.h"
#include "nanousd/layer.h"
#include "lz4_decode.h"
#include "mmap_handle.h"
#include "path_pool.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#endif

namespace {
// RSS probe used by the MUSD_TRACE_PARSE diagnostic below to attribute
// parser memory growth to specific sub-phases. Always compiled in;
// only invoked when MUSD_TRACE_PARSE is set, so the runtime cost on
// the normal path is one getenv() per parse.
size_t TraceRssBytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return static_cast<size_t>(pmc.WorkingSetSize);
    return 0;
#else
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    long ignored = 0;
    long pages = 0;
    if (std::fscanf(f, "%ld %ld", &ignored, &pages) != 2) pages = 0;
    std::fclose(f);
    return static_cast<size_t>(pages) * 4096u;
#endif
}
} // anonymous

namespace nanousd {

// ============================================================
// Binary reader helpers
// ============================================================

namespace {

// Type-erased file buffer: holds data/size plus a shared_ptr to keep the
// backing store alive (either MmapHandle or vector<uint8_t>).
struct FileBuffer {
    const uint8_t* data_;
    size_t size_;
    std::shared_ptr<void> owner_;  // prevents deallocation of backing store

    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
};

class BinaryReader {
public:
    BinaryReader(const uint8_t* data, size_t size)
        : data_(data), size_(size), pos_(0) {}

    // Position management
    size_t Pos() const { return pos_; }
    size_t Size() const { return size_; }
    void Seek(size_t pos) { pos_ = pos; }
    bool CanRead(size_t n) const { return pos_ + n <= size_; }
    const uint8_t* Data() const { return data_; }

    // Read raw bytes
    bool ReadBytes(void* dst, size_t n) {
        if (!CanRead(n)) return false;
        std::memcpy(dst, data_ + pos_, n);
        pos_ += n;
        return true;
    }

    template <typename T>
    bool Read(T* val) {
        return ReadBytes(val, sizeof(T));
    }

    uint8_t ReadU8() { uint8_t v = 0; Read(&v); return v; }
    uint16_t ReadU16() { uint16_t v = 0; Read(&v); return v; }
    uint32_t ReadU32() { uint32_t v = 0; Read(&v); return v; }
    uint64_t ReadU64() { uint64_t v = 0; Read(&v); return v; }
    int32_t ReadI32() { int32_t v = 0; Read(&v); return v; }
    int64_t ReadI64() { int64_t v = 0; Read(&v); return v; }
    float ReadF32() { float v = 0; Read(&v); return v; }
    double ReadF64() { double v = 0; Read(&v); return v; }

    // Read from an absolute offset without changing current position
    template <typename T>
    T ReadAt(size_t offset) {
        T val{};
        if (offset + sizeof(T) <= size_) {
            std::memcpy(&val, data_ + offset, sizeof(T));
        }
        return val;
    }

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_;
};

// ============================================================
// LZ4 chunked decompression (per spec 16.3.7.1)
// ============================================================

// LZ4 decompression per spec 16.3.7.1.
// The spec says the buffer starts with a uint64 numChunks. In practice, Crate
// files store this as a single byte (0 = single block). We detect both forms:
// if the first 8 bytes interpreted as uint64 LE have non-zero upper bytes,
// treat only byte 0 as the chunk count.
bool Lz4DecompressChunked(const uint8_t* src, size_t srcSize,
                          std::vector<uint8_t>& out, size_t uncompressedSize) {
    out.resize(uncompressedSize);
    if (uncompressedSize == 0) return true;
    if (srcSize < 1) return false;

    // Read the chunk indicator. In observed Crate files the first byte is 0
    // (single-block mode) and the LZ4 data follows immediately at byte 1.
    uint8_t firstByte = src[0];

    if (firstByte == 0) {
        // Single block: the rest of the buffer is one LZ4 block
        int result = detail::Lz4Decompress(
            reinterpret_cast<const char*>(src + 1), static_cast<int>(srcSize - 1),
            reinterpret_cast<char*>(out.data()), static_cast<int>(uncompressedSize));
        return result >= 0;
    }

    // Multi-chunk: firstByte > 0 indicates chunk count.
    // Per spec, each chunk has a uint64 size prefix followed by chunk data.
    size_t numChunks = firstByte;
    const uint8_t* p = src + 1;
    size_t remaining = srcSize - 1;
    size_t outOffset = 0;

    for (size_t i = 0; i < numChunks; ++i) {
        if (remaining < 8) return false;
        uint64_t chunkSize;
        std::memcpy(&chunkSize, p, 8);
        p += 8;
        remaining -= 8;

        if (chunkSize > remaining) return false;
        if (outOffset >= uncompressedSize) return false;

        int outRemaining = static_cast<int>(uncompressedSize - outOffset);
        int result = detail::Lz4Decompress(
            reinterpret_cast<const char*>(p), static_cast<int>(chunkSize),
            reinterpret_cast<char*>(out.data() + outOffset), outRemaining);
        if (result < 0) return false;

        outOffset += result;
        p += chunkSize;
        remaining -= static_cast<size_t>(chunkSize);
    }

    return true;
}

// ============================================================
// Compressed Integer Arrays (spec 16.3.7.2)
// ============================================================

template <typename IntT>
bool DecodeCompressedInts(BinaryReader& reader, std::vector<IntT>& result) {
    // Read compressed size
    uint64_t compressedSize = reader.ReadU64();
    if (!reader.CanRead(static_cast<size_t>(compressedSize))) return false;

    // LZ4-decompress the encoded integer data.
    // Per spec 16.3.7.2.3, the data is LZ4-compressed with the chunked format
    // (first byte is chunk indicator, 0 = single block).
    const uint8_t* compData = reader.Data() + reader.Pos();
    reader.Seek(reader.Pos() + static_cast<size_t>(compressedSize));

    size_t numInts = result.size();
    if (numInts == 0) return true;

    if (compressedSize < 1) return false;

    // Estimate decompressed size — generous upper bound.
    size_t estSize = sizeof(IntT) + (numInts * 2 + 7) / 8 + numInts * sizeof(IntT);
    std::vector<uint8_t> decompressed(estSize);
    int decompSize = -1;

    uint8_t chunkIndicator = compData[0];
    if (chunkIndicator == 0) {
        // Single block: LZ4 data follows directly after the indicator byte
        decompSize = detail::Lz4Decompress(
            reinterpret_cast<const char*>(compData + 1),
            static_cast<int>(compressedSize - 1),
            reinterpret_cast<char*>(decompressed.data()),
            static_cast<int>(estSize));
        if (decompSize < 0) {
            estSize *= 4;
            decompressed.resize(estSize);
            decompSize = detail::Lz4Decompress(
                reinterpret_cast<const char*>(compData + 1),
                static_cast<int>(compressedSize - 1),
                reinterpret_cast<char*>(decompressed.data()),
                static_cast<int>(estSize));
            if (decompSize < 0) return false;
        }
    } else {
        // Multi-chunk: each chunk has a uint64 size prefix
        const uint8_t* p = compData + 1;
        size_t remaining = static_cast<size_t>(compressedSize) - 1;
        size_t outOffset = 0;

        for (int chunk = 0; chunk < chunkIndicator; ++chunk) {
            if (remaining < 8) return false;
            uint64_t chunkSize;
            std::memcpy(&chunkSize, p, 8);
            p += 8;
            remaining -= 8;
            if (chunkSize > remaining) return false;

            size_t outRemaining = estSize - outOffset;
            if (outRemaining == 0) {
                estSize *= 2;
                decompressed.resize(estSize);
                outRemaining = estSize - outOffset;
            }
            int result = detail::Lz4Decompress(
                reinterpret_cast<const char*>(p), static_cast<int>(chunkSize),
                reinterpret_cast<char*>(decompressed.data() + outOffset),
                static_cast<int>(outRemaining));
            if (result < 0) return false;
            outOffset += result;
            p += chunkSize;
            remaining -= static_cast<size_t>(chunkSize);
        }
        decompSize = static_cast<int>(outOffset);
    }

    // Now decode the compressed integer format
    const uint8_t* buf = decompressed.data();
    size_t bufSize = static_cast<size_t>(decompSize);
    size_t bufPos = 0;

    if (bufSize < sizeof(IntT)) return false;

    // Read common value (same width as IntT but signed)
    using SignedT = typename std::make_signed<IntT>::type;
    SignedT commonValue;
    std::memcpy(&commonValue, buf + bufPos, sizeof(IntT));
    bufPos += sizeof(IntT);

    // Read codepoints: 2 bits per value, packed into bytes
    size_t codepointBytes = (numInts * 2 + 7) / 8;
    if (bufPos + codepointBytes > bufSize) return false;
    const uint8_t* codepoints = buf + bufPos;
    bufPos += codepointBytes;

    // Decode values
    SignedT runningValue = 0;
    for (size_t i = 0; i < numInts; ++i) {
        size_t byteIdx = (i * 2) / 8;
        size_t bitIdx = (i * 2) % 8;
        uint8_t code = (codepoints[byteIdx] >> bitIdx) & 0x03;

        SignedT delta;
        if (code == 0) {
            // Common value
            delta = commonValue;
        } else if (code == 1) {
            // Quarter width
            if (bufPos + (sizeof(IntT) <= 4 ? 1 : 2) > bufSize) return false;
            if constexpr (sizeof(IntT) <= 4) {
                int8_t v;
                std::memcpy(&v, buf + bufPos, 1);
                bufPos += 1;
                delta = static_cast<SignedT>(v);
            } else {
                int16_t v;
                std::memcpy(&v, buf + bufPos, 2);
                bufPos += 2;
                delta = static_cast<SignedT>(v);
            }
        } else if (code == 2) {
            // Half width
            if constexpr (sizeof(IntT) <= 4) {
                if (bufPos + 2 > bufSize) return false;
                int16_t v;
                std::memcpy(&v, buf + bufPos, 2);
                bufPos += 2;
                delta = static_cast<SignedT>(v);
            } else {
                if (bufPos + 4 > bufSize) return false;
                int32_t v;
                std::memcpy(&v, buf + bufPos, 4);
                bufPos += 4;
                delta = static_cast<SignedT>(v);
            }
        } else {
            // Full width
            if (bufPos + sizeof(IntT) > bufSize) return false;
            std::memcpy(&delta, buf + bufPos, sizeof(IntT));
            bufPos += sizeof(IntT);
        }

        runningValue += delta;
        result[i] = static_cast<IntT>(runningValue);
    }

    return true;
}

// ============================================================
// Crate Value Type IDs (spec 16.3.10.1)
// ============================================================

enum class CrateTypeId : uint8_t {
    Invalid = 0,
    Bool = 1, UChar = 2, Int = 3, UInt = 4,
    Int64 = 5, UInt64 = 6,
    Half = 7, Float = 8, Double = 9,
    String = 10, Token = 11, Asset = 12,
    Matrix2d = 13, Matrix3d = 14, Matrix4d = 15,
    Quatd = 16, Quatf = 17, Quath = 18,
    Double2 = 19, Float2 = 20, Half2 = 21, Int2 = 22,
    Double3 = 23, Float3 = 24, Half3 = 25, Int3 = 26,
    Double4 = 27, Float4 = 28, Half4 = 29, Int4 = 30,
    Dictionary = 31,
    TokenListOp = 32, StringListOp = 33, PathListOp = 34,
    ReferenceListOp = 35,
    IntListOp = 36, Int64ListOp = 37,
    UIntListOp = 38, UInt64ListOp = 39,
    PathVector = 40, TokenVector = 41,
    Specifier = 42, Permission = 43, Variability = 44,
    VariantSelectionMap = 45,
    TimeSamples = 46, Payload = 47,
    DoubleVector = 48, LayerOffsetVector = 49,
    StringVector = 50,
    ValueBlock = 51, Value = 52,
    UnregisteredValue = 53, UnregisteredValueListOp = 54,
    PayloadListOp = 55,
    TimeCode = 56, PathExpression = 57,
    Relocates = 58, Splines = 59,
};

// ============================================================
// Value Representation (spec 16.3.9)
// ============================================================

struct ValueRep {
    uint64_t data = 0;

    uint64_t GetPayload() const { return data & 0x0000FFFFFFFFFFFF; }
    CrateTypeId GetType() const { return static_cast<CrateTypeId>((data >> 48) & 0xFF); }
    uint8_t GetFlags() const { return static_cast<uint8_t>((data >> 56) & 0xFF); }
    bool IsArray() const { return (GetFlags() & 0x80) != 0; }
    bool IsInlined() const { return (GetFlags() & 0x40) != 0; }
    bool IsCompressed() const { return (GetFlags() & 0x20) != 0; }
};

// ============================================================
// TOC entry
// ============================================================

struct TocSection {
    std::string name;
    uint64_t start = 0;
    uint64_t size = 0;
};

// ============================================================
// Spec form IDs (spec 16.3.8.4.6)
// ============================================================

SpecType SpecFormToSpecType(uint32_t form) {
    switch (form) {
        case 1: return SpecType::Attribute;
        case 6: return SpecType::Prim;
        case 7: return SpecType::Layer;
        case 8: return SpecType::Relationship;
        case 10: return SpecType::Variant;
        case 11: return SpecType::VariantSet;
        default: return SpecType::Prim;  // Unknown forms treated as inert
    }
}

// ============================================================
// CrateTypeId → TypeId mapping (for lazy values)
// ============================================================

TypeId CrateTypeIdToTypeId(CrateTypeId ct) {
    switch (ct) {
        case CrateTypeId::Bool:     return TypeId::Bool;
        case CrateTypeId::UChar:    return TypeId::UChar;
        case CrateTypeId::Int:      return TypeId::Int;
        case CrateTypeId::UInt:     return TypeId::UInt;
        case CrateTypeId::Int64:    return TypeId::Int64;
        case CrateTypeId::UInt64:   return TypeId::UInt64;
        case CrateTypeId::Half:     return TypeId::Half;
        case CrateTypeId::Float:    return TypeId::Float;
        case CrateTypeId::Double:   return TypeId::Double;
        case CrateTypeId::String:   return TypeId::String;
        case CrateTypeId::Token:    return TypeId::Token;
        case CrateTypeId::Asset:    return TypeId::Asset;
        case CrateTypeId::Matrix2d: return TypeId::Matrix2d;
        case CrateTypeId::Matrix3d: return TypeId::Matrix3d;
        case CrateTypeId::Matrix4d: return TypeId::Matrix4d;
        case CrateTypeId::Quatd:    return TypeId::Quatd;
        case CrateTypeId::Quatf:    return TypeId::Quatf;
        case CrateTypeId::Quath:    return TypeId::Quath;
        case CrateTypeId::Double2:  return TypeId::Double2;
        case CrateTypeId::Float2:   return TypeId::Float2;
        case CrateTypeId::Half2:    return TypeId::Half2;
        case CrateTypeId::Int2:     return TypeId::Int2;
        case CrateTypeId::Double3:  return TypeId::Double3;
        case CrateTypeId::Float3:   return TypeId::Float3;
        case CrateTypeId::Half3:    return TypeId::Half3;
        case CrateTypeId::Int3:     return TypeId::Int3;
        case CrateTypeId::Double4:  return TypeId::Double4;
        case CrateTypeId::Float4:   return TypeId::Float4;
        case CrateTypeId::Half4:    return TypeId::Half4;
        case CrateTypeId::Int4:     return TypeId::Int4;
        case CrateTypeId::TimeCode: return TypeId::TimeCode;
        default: return TypeId::Unknown;
    }
}

// ============================================================
// Lazy array decode support
// ============================================================

// POD types whose arrays can be lazily decoded (no token/string table needed).
bool IsLazyEligibleArray(CrateTypeId ct) {
    switch (ct) {
        case CrateTypeId::Bool: case CrateTypeId::UChar:
        case CrateTypeId::Int: case CrateTypeId::UInt:
        case CrateTypeId::Int64: case CrateTypeId::UInt64:
        case CrateTypeId::Half: case CrateTypeId::Float: case CrateTypeId::Double:
        case CrateTypeId::Float2: case CrateTypeId::Float3: case CrateTypeId::Float4:
        case CrateTypeId::Double2: case CrateTypeId::Double3: case CrateTypeId::Double4:
        case CrateTypeId::Int2: case CrateTypeId::Int3: case CrateTypeId::Int4:
        case CrateTypeId::Half2: case CrateTypeId::Half3: case CrateTypeId::Half4:
        case CrateTypeId::Matrix2d: case CrateTypeId::Matrix3d: case CrateTypeId::Matrix4d:
        case CrateTypeId::Quatd: case CrateTypeId::Quatf: case CrateTypeId::Quath:
        case CrateTypeId::TimeCode:
            return true;
        default:
            return false;
    }
}

// Standalone compressed float decode for lazy path.
template <typename FloatT>
bool LazyDecodeCompressedFloat(BinaryReader& reader, size_t n, std::vector<FloatT>& out) {
    char encoding = static_cast<char>(reader.ReadU8());
    if (encoding == 'i') {
        if constexpr (sizeof(FloatT) <= 4) {
            std::vector<int32_t> ints(n);
            if (!DecodeCompressedInts(reader, ints)) return false;
            out.resize(n);
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<FloatT>(ints[i]);
        } else {
            std::vector<int64_t> ints(n);
            if (!DecodeCompressedInts(reader, ints)) return false;
            out.resize(n);
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<FloatT>(ints[i]);
        }
        return true;
    }
    if (encoding == 't') {
        uint32_t lutCount = reader.ReadU32();
        std::vector<FloatT> lut(lutCount);
        reader.ReadBytes(lut.data(), lutCount * sizeof(FloatT));
        std::vector<uint32_t> indices(n);
        if (!DecodeCompressedInts(reader, indices)) return false;
        out.resize(n);
        for (size_t i = 0; i < n; ++i) {
            uint32_t idx = indices[i];
            out[i] = (idx < lutCount) ? lut[idx] : FloatT{};
        }
        return true;
    }
    // Unknown encoding — read uncompressed
    out.resize(n);
    reader.ReadBytes(out.data(), n * sizeof(FloatT));
    return true;
}

// Decode a POD array directly into std::any + size.
// Called by lazy thunks on first access.
uint64_t ReadArrayElementCount(BinaryReader& reader, uint8_t versionMinor) {
    if (versionMinor < 5) {
        // Crate versions before 0.5.0 stored VtArray shape rank first.
        // USD only wrote rank=1 here; discard it and read the element count.
        uint32_t rank = reader.ReadU32();
        if (rank != 1) {
            throw std::runtime_error("Unsupported pre-0.5 USDC array rank");
        }
    }
    if (versionMinor < 7) {
        return reader.ReadU32();
    }
    return reader.ReadU64();
}

void LazyDecodePodArray(const uint8_t* data, size_t dataSize,
                        CrateTypeId type, size_t offset, bool compressed,
                        uint8_t versionMinor,
                        std::any& outData, size_t& outSize) {
    BinaryReader reader(data, dataSize);
    reader.Seek(offset);
    uint64_t n = ReadArrayElementCount(reader, versionMinor);
    outSize = static_cast<size_t>(n);

    // Compressed scalar types (vec/matrix types fall through to uncompressed)
    if (compressed) {
        switch (type) {
            case CrateTypeId::Int: {
                std::vector<Int> arr(n);
                DecodeCompressedInts(reader, arr);
                outData = std::move(arr); return;
            }
            case CrateTypeId::UInt: {
                std::vector<UInt> arr(n);
                DecodeCompressedInts(reader, arr);
                outData = std::move(arr); return;
            }
            case CrateTypeId::Int64: {
                std::vector<Int64> arr(n);
                DecodeCompressedInts(reader, arr);
                outData = std::move(arr); return;
            }
            case CrateTypeId::UInt64: {
                std::vector<UInt64> arr(n);
                DecodeCompressedInts(reader, arr);
                outData = std::move(arr); return;
            }
            case CrateTypeId::Half: {
                std::vector<Half> arr;
                LazyDecodeCompressedFloat(reader, static_cast<size_t>(n), arr);
                outData = std::move(arr); return;
            }
            case CrateTypeId::Float: {
                std::vector<Float> arr;
                LazyDecodeCompressedFloat(reader, static_cast<size_t>(n), arr);
                outData = std::move(arr); return;
            }
            case CrateTypeId::Double: {
                std::vector<Double> arr;
                LazyDecodeCompressedFloat(reader, static_cast<size_t>(n), arr);
                outData = std::move(arr); return;
            }
            default: break;  // fall through to uncompressed read
        }
    }

    // Uncompressed read for all POD types
    switch (type) {
        case CrateTypeId::Bool: {
            std::vector<Bool> arr(n);
            for (size_t i = 0; i < n; ++i) {
                uint8_t b = 0; reader.ReadBytes(&b, 1);
                arr[i] = (b != 0);
            }
            outData = std::move(arr); return;
        }
        #define LAZY_POD(CT, T) case CrateTypeId::CT: { \
            std::vector<T> arr(static_cast<size_t>(n)); \
            reader.ReadBytes(arr.data(), static_cast<size_t>(n) * sizeof(T)); \
            outData = std::move(arr); return; }
        LAZY_POD(UChar, UChar)
        LAZY_POD(Int, Int)
        LAZY_POD(UInt, UInt)
        LAZY_POD(Int64, Int64)
        LAZY_POD(UInt64, UInt64)
        LAZY_POD(Half, Half)
        LAZY_POD(Float, Float)
        LAZY_POD(Double, Double)
        LAZY_POD(Float2, GfVec2f)
        LAZY_POD(Float3, GfVec3f)
        LAZY_POD(Float4, GfVec4f)
        LAZY_POD(Double2, GfVec2d)
        LAZY_POD(Double3, GfVec3d)
        LAZY_POD(Double4, GfVec4d)
        LAZY_POD(Int2, GfVec2i)
        LAZY_POD(Int3, GfVec3i)
        LAZY_POD(Int4, GfVec4i)
        LAZY_POD(Half2, GfVec2h)
        LAZY_POD(Half3, GfVec3h)
        LAZY_POD(Half4, GfVec4h)
        LAZY_POD(Matrix2d, GfMatrix2d)
        LAZY_POD(Matrix3d, GfMatrix3d)
        LAZY_POD(Matrix4d, GfMatrix4d)
        LAZY_POD(Quatd, GfQuatd)
        LAZY_POD(Quatf, GfQuatf)
        LAZY_POD(Quath, GfQuath)
        case CrateTypeId::TimeCode: {
            std::vector<Double> arr(static_cast<size_t>(n));
            reader.ReadBytes(arr.data(), static_cast<size_t>(n) * sizeof(Double));
            outData = std::move(arr); return;
        }
        #undef LAZY_POD
        default: break;
    }
}

// ============================================================
// Persistent reader state for lazy field decoding
// ============================================================
// Captured by per-Spec lazy initializers to decode deferred fields
// after the UsdcReader is destroyed.

// USDC layer's deferred-decode backing.
//
// Holds the parsed crate's section vectors plus the file buffer
// they reference. Implements `DecodeBacking` so `Spec`s carrying a
// fieldset index can ask the Layer to decode a single field by
// name on first access — per-field laziness without per-Spec lambda
// machinery. One of these per loaded USDC layer; specs reference
// it by raw pointer.
struct UsdcReaderState : public DecodeBacking {
    std::shared_ptr<FileBuffer> fileBuffer;
    std::vector<std::string> tokens;
    std::vector<uint32_t> stringIndices;
    std::vector<uint32_t> fieldTokenIndices;
    std::vector<ValueRep> fieldValueReps;
    std::vector<uint32_t> flatFieldSetIndices;
    std::vector<Path> paths;
    uint8_t versionMinor = 0;

    std::string GetToken(uint32_t idx) const {
        return idx < tokens.size() ? tokens[idx] : "";
    }
    std::string GetString(uint32_t idx) const {
        return idx < stringIndices.size() ? GetToken(stringIndices[idx]) : "";
    }
    Path GetPath(uint32_t idx) const {
        return idx < paths.size() ? paths[idx] : Path();
    }

    // DecodeBacking — implementations live below the UsdcReader
    // class definition since they construct one to do the actual
    // value-decode work.
    Value DecodeField(uint32_t fieldsetIdx,
                      const Token& fieldName) const override;
    void DecodeAllFields(uint32_t fieldsetIdx,
                          Spec& target) const override;
};

// Fields that must be eagerly decoded for composition/population.
// Everything else (defaultValue, timeSamples, custom metadata, etc.)
// can be deferred until first access.
static bool IsCompositionCriticalField(const std::string& name) {
    return name == "specifier" || name == "typeName" || name == "active"
        || name == "instanceable" || name == "hidden" || name == "custom"
        || name == "variability" || name == "references" || name == "payload"
        || name == "subLayers" || name == "subLayerOffsets"
        || name == "apiSchemas" || name == "fallbackPrimTypes"
        || name == "primChildren"
        || name == "propertyChildren" || name == "primOrder"
        || name == "propertyOrder" || name == "kind"
        || name == "defaultPrim";
}

// Token-typed variant of IsCompositionCriticalField. Token equality is an
// O(1) pointer compare (both sides interned), so this is ~20x faster than
// the string-per-compare fallback used for legacy call sites.
static bool IsCompositionCriticalFieldTok(const Token& t) {
    return t == FieldNames::specifier || t == FieldNames::typeName
        || t == FieldNames::active || t == FieldNames::instanceable
        || t == FieldNames::hidden || t == FieldNames::custom
        || t == FieldNames::variability || t == FieldNames::references
        || t == FieldNames::payload || t == FieldNames::subLayers
        || t == FieldNames::subLayerOffsets || t == FieldNames::apiSchemas
        || t == FieldNames::fallbackPrimTypes || t == FieldNames::primChildren
        || t == FieldNames::propertyChildren
        || t == FieldNames::primOrder || t == FieldNames::propertyOrder
        || t == FieldNames::kind || t == FieldNames::defaultPrim;
}

// ============================================================
// USDC Reader
// ============================================================

class UsdcReader {
public:
    UsdcReader(const uint8_t* data, size_t size)
        : reader_(data, size) {}

    // Decode-only constructor: initialized from persistent state for lazy field decoding.
    // Creates a fresh BinaryReader pointing to the same file data.
    // Does NOT copy section vectors — references the state directly.
    UsdcReader(const std::shared_ptr<UsdcReaderState>& state)
        : reader_(state->fileBuffer->data(), state->fileBuffer->size()),
          versionMinor_(state->versionMinor),
          fileBuffer_(state->fileBuffer),
          stateRef_(state.get()) {}

    // Raw-pointer overload — used by `UsdcReaderState`'s own
    // `DecodeField` implementation, which needs to hand `this` to a
    // reader without bumping its own refcount.
    explicit UsdcReader(const UsdcReaderState* state)
        : reader_(state->fileBuffer->data(), state->fileBuffer->size()),
          versionMinor_(state->versionMinor),
          fileBuffer_(state->fileBuffer),
          stateRef_(state) {}

    void SetFileBuffer(std::shared_ptr<FileBuffer> buf) {
        fileBuffer_ = std::move(buf);
    }

    // Public decode entry point for lazy spec stores.
    Value DecodeFieldValue(const ValueRep& rep) { return DecodeValue(rep); }

    UsdcParseResult Parse() {
        UsdcParseResult result;
        bool trace = (std::getenv("MUSD_TRACE_PARSE") != nullptr);
        // Capture RSS at every phase boundary when tracing is on so
        // the trace output can attribute memory growth to specific
        // sub-phases. Cheap when not tracing.
        auto rss = [&]() -> size_t { return trace ? TraceRssBytes() : 0; };
        try {
            auto t0 = std::chrono::high_resolution_clock::now();
            size_t r0 = rss();
            ReadHeader();
            ReadBootstrap();
            ReadTableOfContents();
            auto t1 = std::chrono::high_resolution_clock::now();
            size_t r1 = rss();
            ReadTokens();
            auto t2 = std::chrono::high_resolution_clock::now();
            size_t r2 = rss();
            ReadStrings();
            ReadFields();
            auto t3 = std::chrono::high_resolution_clock::now();
            size_t r3 = rss();
            ReadFieldSets();
            auto t4 = std::chrono::high_resolution_clock::now();
            size_t r4 = rss();
            ReadPaths();
            auto t5 = std::chrono::high_resolution_clock::now();
            size_t r5 = rss();
            ReadSpecs();
            auto t6 = std::chrono::high_resolution_clock::now();
            size_t r6 = rss();
            BuildLayer();
            auto t7 = std::chrono::high_resolution_clock::now();
            size_t r7 = rss();
            if (trace) {
                auto ms = [](auto a, auto b) {
                    return std::chrono::duration<double, std::milli>(b - a).count();
                };
                auto fmt = [](size_t bytes) -> std::string {
                    char buf[32];
                    double mb = static_cast<double>(bytes) / (1024.0 * 1024.0);
                    if (mb >= 1024.0) std::snprintf(buf, sizeof(buf), "%.2fGB", mb / 1024.0);
                    else              std::snprintf(buf, sizeof(buf), "%.0fMB", mb);
                    return std::string(buf);
                };
                auto delta = [&](size_t a, size_t b) -> std::string {
                    return fmt(b > a ? b - a : 0);
                };
                fprintf(stderr,
                    "usdc_parse phase    time          rss-delta   rss-current\n"
                    "  toc            : %8.1f ms   %8s   %8s\n"
                    "  ReadTokens     : %8.1f ms   %8s   %8s\n"
                    "  Read{Strings,Fields}: %8.1f ms %8s   %8s\n"
                    "  ReadFieldSets  : %8.1f ms   %8s   %8s\n"
                    "  ReadPaths      : %8.1f ms   %8s   %8s\n"
                    "  ReadSpecs      : %8.1f ms   %8s   %8s\n"
                    "  BuildLayer     : %8.1f ms   %8s   %8s\n"
                    "  TOTAL          : %8.1f ms   %8s   %8s\n",
                    ms(t0,t1), delta(r0,r1).c_str(), fmt(r1).c_str(),
                    ms(t1,t2), delta(r1,r2).c_str(), fmt(r2).c_str(),
                    ms(t2,t3), delta(r2,r3).c_str(), fmt(r3).c_str(),
                    ms(t3,t4), delta(r3,r4).c_str(), fmt(r4).c_str(),
                    ms(t4,t5), delta(r4,r5).c_str(), fmt(r5).c_str(),
                    ms(t5,t6), delta(r5,r6).c_str(), fmt(r6).c_str(),
                    ms(t6,t7), delta(r6,r7).c_str(), fmt(r7).c_str(),
                    ms(t0,t7), delta(r0,r7).c_str(), fmt(r7).c_str());
                fprintf(stderr, "  %zu tokens, %zu fields, %zu paths, %zu specs\n",
                    tokens_.size(), fieldValueReps_.size(), paths_.size(),
                    specPathIndices_.size());
            }
            result.layer = std::move(layer_);
            result.success = true;
        } catch (const std::exception& e) {
            result.error = e.what();
        }
        return result;
    }

private:
    // --- Header ---
    void ReadHeader() {
        char magic[8];
        if (!reader_.ReadBytes(magic, 8)) throw std::runtime_error("Failed to read header");
        if (std::memcmp(magic, "PXR-USDC", 8) != 0)
            throw std::runtime_error("Not a USDC file (bad magic)");

        versionMajor_ = reader_.ReadU8();
        versionMinor_ = reader_.ReadU8();
        versionPatch_ = reader_.ReadU8();

        // Skip 5 unused bytes
        reader_.Seek(reader_.Pos() + 5);

        // AOUSD Core v1.0.1 specifies Crate 0.8.0 through 0.12.0.
        // Reader support for 0.7.x is retained as internal legacy
        // compatibility; the writer still emits only 0.8.0 or newer.
        if (versionMajor_ != 0 || versionMinor_ < 7 || versionMinor_ > 12)
            throw std::runtime_error("Unsupported crate version " +
                std::to_string(versionMajor_) + "." +
                std::to_string(versionMinor_) + "." +
                std::to_string(versionPatch_));
    }

    // --- Bootstrap ---
    void ReadBootstrap() {
        tocOffset_ = reader_.ReadU64();
        reader_.Seek(reader_.Pos() + 8);  // skip reserved
    }

    // --- Table of Contents ---
    void ReadTableOfContents() {
        reader_.Seek(static_cast<size_t>(tocOffset_));
        uint64_t numSections = reader_.ReadU64();

        for (uint64_t i = 0; i < numSections; ++i) {
            TocSection sec;
            char name[16];
            if (!reader_.ReadBytes(name, 16)) throw std::runtime_error("Failed to read TOC entry");
            sec.name = std::string(name, strnlen(name, 16));
            sec.start = reader_.ReadU64();
            sec.size = reader_.ReadU64();
            sections_[sec.name] = sec;
        }
    }

    // --- TOKENS section ---
    void ReadTokens() {
        auto it = sections_.find("TOKENS");
        if (it == sections_.end()) return;

        reader_.Seek(static_cast<size_t>(it->second.start));
        uint64_t numTokens = reader_.ReadU64();
        uint64_t uncompressedSize = reader_.ReadU64();
        uint64_t compressedSize = reader_.ReadU64();

        if (!reader_.CanRead(static_cast<size_t>(compressedSize)))
            throw std::runtime_error("TOKENS section truncated");

        std::vector<uint8_t> decompressed;
        bool ok = Lz4DecompressChunked(
            reader_.Data() + reader_.Pos(),
            static_cast<size_t>(compressedSize),
            decompressed,
            static_cast<size_t>(uncompressedSize));
        if (!ok) throw std::runtime_error("Failed to decompress TOKENS");

        // Split null-terminated strings
        tokens_.reserve(numTokens);
        const char* p = reinterpret_cast<const char*>(decompressed.data());
        const char* end = p + decompressed.size();
        while (p < end && tokens_.size() < numTokens) {
            tokens_.emplace_back(p);
            p += tokens_.back().size() + 1;
        }

        while (tokens_.size() < numTokens) tokens_.emplace_back("");

        // Preintern once up-front so downstream lookups (ReconstructPaths,
        // ApplyField, etc.) can reference pointer-sized Tokens by index
        // without re-interning the same string on every element touch.
        tokenPool_.clear();
        tokenPool_.reserve(tokens_.size());
        for (const auto& s : tokens_) tokenPool_.emplace_back(s);
    }

    // --- STRINGS section ---
    void ReadStrings() {
        auto it = sections_.find("STRINGS");
        if (it == sections_.end()) return;

        reader_.Seek(static_cast<size_t>(it->second.start));
        uint64_t numIndices = reader_.ReadU64();
        stringIndices_.resize(static_cast<size_t>(numIndices));

        for (size_t i = 0; i < stringIndices_.size(); ++i) {
            stringIndices_[i] = reader_.ReadU32();
        }
    }

    // --- FIELDS section ---
    void ReadFields() {
        auto it = sections_.find("FIELDS");
        if (it == sections_.end()) return;

        reader_.Seek(static_cast<size_t>(it->second.start));
        uint64_t numFields = reader_.ReadU64();

        // Read compressed field name indices
        fieldTokenIndices_.resize(static_cast<size_t>(numFields));
        if (!DecodeCompressedInts(reader_, fieldTokenIndices_))
            throw std::runtime_error("Failed to decode field token indices");

        // Read value representations
        uint64_t valueDataSize = reader_.ReadU64();
        if (!reader_.CanRead(static_cast<size_t>(valueDataSize)))
            throw std::runtime_error("FIELDS value data truncated");

        // Decompress value representations
        const uint8_t* compData = reader_.Data() + reader_.Pos();
        // Value reps are 8 bytes each, so uncompressed size = numFields * 8
        size_t uncompSize = static_cast<size_t>(numFields) * 8;
        std::vector<uint8_t> decompressed;
        bool ok = Lz4DecompressChunked(compData, static_cast<size_t>(valueDataSize),
                                       decompressed, uncompSize);
        if (!ok) throw std::runtime_error("Failed to decompress field value reps");

        fieldValueReps_.resize(static_cast<size_t>(numFields));
        for (size_t i = 0; i < fieldValueReps_.size(); ++i) {
            std::memcpy(&fieldValueReps_[i].data, decompressed.data() + i * 8, 8);
        }

        reader_.Seek(reader_.Pos() + static_cast<size_t>(valueDataSize));
    }

    // --- FIELDSETS section ---
    void ReadFieldSets() {
        auto it = sections_.find("FIELDSETS");
        if (it == sections_.end()) return;

        reader_.Seek(static_cast<size_t>(it->second.start));
        uint64_t numIndices = reader_.ReadU64();

        std::vector<uint32_t> flatIndices(static_cast<size_t>(numIndices));
        if (!DecodeCompressedInts(reader_, flatIndices))
            throw std::runtime_error("Failed to decode fieldset indices");

        // The fieldset index from SPECS is an index into this flat array.
        // Each fieldset runs from the given index until the next 0xFFFFFFFF sentinel.
        flatFieldSetIndices_ = std::move(flatIndices);
    }

    // --- PATHS section ---
    void ReadPaths() {
        auto it = sections_.find("PATHS");
        if (it == sections_.end()) return;

        reader_.Seek(static_cast<size_t>(it->second.start));
        uint64_t numPaths = reader_.ReadU64();
        // Second count is the number of entries in the compressed path arrays
        // (jump tree elements). This may differ from numPaths.
        uint64_t numElements = reader_.ReadU64();
        size_t n = static_cast<size_t>(numElements);

        std::vector<uint32_t> pathIndices(n);
        std::vector<int32_t> elementTokenIndices(n);
        std::vector<int32_t> jumps(n);

        if (!DecodeCompressedInts(reader_, pathIndices))
            throw std::runtime_error("Failed to decode path indices");
        if (!DecodeCompressedInts(reader_, elementTokenIndices))
            throw std::runtime_error("Failed to decode element token indices");
        if (!DecodeCompressedInts(reader_, jumps))
            throw std::runtime_error("Failed to decode path jumps");

        // Reconstruct paths using the algorithm from spec 16.3.8.4.5.4
        size_t maxPathIdx = 0;
        for (auto pi : pathIndices) {
            if (pi > maxPathIdx) maxPathIdx = pi;
        }
        paths_.resize(maxPathIdx + 1);

        // Skip the ~log2(n) rehash cycle PathPool's entries map would
        // otherwise hit as it grows under a million-plus interns. The
        // primChildren_/primProperties_ maps are intentionally NOT
        // reserved at `n` — their entry count is bounded by the prim
        // count (one parent per child name, one prim per property), a
        // small fraction of the path count, so reserving at `n` would
        // waste tens of MB of bucket array on a flat scene.
        PathPool::Instance().ReserveAdditional(n);

        ReconstructPaths(pathIndices, elementTokenIndices, jumps, n,
                         prebuiltPrimChildren_, prebuiltPrimChildPaths_,
                         prebuiltPrimProperties_);
    }

    void ReconstructPaths(const std::vector<uint32_t>& pathIndices,
                          const std::vector<int32_t>& elementTokenIndices,
                          const std::vector<int32_t>& jumps,
                          size_t numPaths,
                          PathMap<std::vector<Token>>& primChildren,
                          PathMap<std::vector<Path>>& primChildPaths,
                          PathMap<std::vector<Token>>& primProperties) {
        if (numPaths == 0) return;

        // The DFS below uses tokenPool_ (preinterned by ReadTokens) so every
        // AppendChild / AppendProperty / primChildren push gets a Token by
        // const ref — no per-element Intern() round-trips.
        const auto& tokenPool = tokenPool_;

        // Use iterative DFS with an explicit stack
        struct StackEntry {
            size_t startIdx;
            Path parentPath;
        };

        std::vector<StackEntry> stack;
        stack.push_back({0, Path()});

        while (!stack.empty()) {
            auto [startIdx, parentPath] = stack.back();
            stack.pop_back();

            size_t x = startIdx;

            while (x < numPaths) {
                Path currentPath;
                if (parentPath.IsEmpty()) {
                    // First iteration — set to absolute root
                    currentPath = Path::AbsoluteRoot();
                } else {
                    int32_t tokenIdx = elementTokenIndices[x];
                    int32_t absIdx = std::abs(tokenIdx);
                    if (static_cast<size_t>(absIdx) >= tokenPool.size()) {
                        // Invalid token index, skip
                        break;
                    }
                    const Token& tokenTok = tokenPool[absIdx];
                    if (tokenIdx >= 0) {
                        // Check for variant selection: token starts with '{'
                        const std::string& tokenStr = tokenTok.GetString();
                        if (!tokenStr.empty() && tokenStr[0] == '{') {
                            // Parse {setName=selectionName}
                            auto eqPos = tokenStr.find('=');
                            if (eqPos != std::string::npos && tokenStr.back() == '}') {
                                std::string setName = tokenStr.substr(1, eqPos - 1);
                                std::string selName = tokenStr.substr(eqPos + 1,
                                    tokenStr.size() - eqPos - 2);
                                currentPath = parentPath.AppendVariantSelection(
                                    setName, selName);
                            } else {
                                currentPath = parentPath.AppendChild(tokenTok);
                                primChildren[parentPath].push_back(tokenTok);
                                primChildPaths[parentPath].push_back(currentPath);
                            }
                        } else {
                            // Prim path — record child in index
                            currentPath = parentPath.AppendChild(tokenTok);
                            primChildren[parentPath].push_back(tokenTok);
                            primChildPaths[parentPath].push_back(currentPath);
                        }
                    } else {
                        // Property path — record property in index
                        Path primPath = parentPath;
                        currentPath = primPath.AppendProperty(tokenTok);
                        primProperties[primPath].push_back(tokenTok);
                    }
                }

                // Store the path
                uint32_t pathIdx = pathIndices[x];
                if (pathIdx < paths_.size()) {
                    paths_[pathIdx] = currentPath;
                }

                int32_t jump = jumps[x];

                if (jump == -2) {
                    // Leaf node — stop this chain
                    break;
                }

                if (jump == -1) {
                    // Only has a child — next entry at x+1 is a child
                    parentPath = currentPath;
                    x++;
                    continue;
                }

                if (jump == 0) {
                    // Only has siblings — next entry at x+1 is a sibling (same parent)
                    x++;
                    continue;
                }

                if (jump > 0) {
                    // Has both: sibling at x+jump (same parent), child at x+1
                    stack.push_back({x + static_cast<size_t>(jump), parentPath});
                    parentPath = currentPath;
                    x++;
                    continue;
                }

                break;
            }
        }
    }

    // --- SPECS section ---
    void ReadSpecs() {
        auto it = sections_.find("SPECS");
        if (it == sections_.end()) return;

        reader_.Seek(static_cast<size_t>(it->second.start));
        uint64_t numSpecs = reader_.ReadU64();
        size_t n = static_cast<size_t>(numSpecs);

        specPathIndices_.resize(n);
        specFieldSetIndices_.resize(n);
        specForms_.resize(n);

        if (!DecodeCompressedInts(reader_, specPathIndices_))
            throw std::runtime_error("Failed to decode spec path indices");
        if (!DecodeCompressedInts(reader_, specFieldSetIndices_))
            throw std::runtime_error("Failed to decode spec fieldset indices");
        if (!DecodeCompressedInts(reader_, specForms_))
            throw std::runtime_error("Failed to decode spec forms");
    }

    // ============================================================
    // Value decoding
    // ============================================================

    std::string GetToken(uint32_t idx) const {
        if (stateRef_) return stateRef_->GetToken(idx);
        if (idx < tokens_.size()) return tokens_[idx];
        return "";
    }

    std::string GetString(uint32_t idx) const {
        if (stateRef_) return stateRef_->GetString(idx);
        if (idx < stringIndices_.size()) return GetToken(stringIndices_[idx]);
        return "";
    }

    Path GetPath(uint32_t idx) const {
        if (stateRef_) return stateRef_->GetPath(idx);
        if (idx < paths_.size()) return paths_[idx];
        return Path();
    }

    Value DecodeValue(const ValueRep& rep) {
        CrateTypeId type = rep.GetType();
        bool isArray = rep.IsArray();
        bool isInlined = rep.IsInlined();
        uint64_t payload = rep.GetPayload();

        if (isInlined) {
            return DecodeInlinedValue(type, payload);
        }

        if (isArray) {
            return DecodeArrayValue(type, static_cast<size_t>(payload), rep.IsCompressed());
        }

        return DecodeOffsetValue(type, static_cast<size_t>(payload));
    }

    Value DecodeInlinedValue(CrateTypeId type, uint64_t payload) {
        uint32_t lo = static_cast<uint32_t>(payload & 0xFFFFFFFF);

        switch (type) {
            case CrateTypeId::Bool:
                return Value(static_cast<Bool>(lo != 0));
            case CrateTypeId::UChar:
                return Value(static_cast<UChar>(lo & 0xFF));
            case CrateTypeId::Int:
                return Value(static_cast<Int>(static_cast<int32_t>(lo)));
            case CrateTypeId::UInt:
                return Value(static_cast<UInt>(lo));
            case CrateTypeId::Half: {
                Half h;
                h.bits = static_cast<uint16_t>(lo & 0xFFFF);
                return Value(h);
            }
            case CrateTypeId::Float: {
                float f;
                std::memcpy(&f, &lo, 4);
                return Value(f);
            }
            case CrateTypeId::Double: {
                // Inlined double is stored as float
                float f;
                std::memcpy(&f, &lo, 4);
                return Value(static_cast<Double>(f));
            }
            case CrateTypeId::Int64:
                // Inlined as int32
                return Value(static_cast<Int64>(static_cast<int32_t>(lo)));
            case CrateTypeId::UInt64:
                // Inlined as uint32
                return Value(static_cast<UInt64>(lo));
            case CrateTypeId::String:
                return Value(GetString(lo));
            case CrateTypeId::Token:
                return Value(Token(GetToken(lo)));
            case CrateTypeId::Asset:
                return Value(GetToken(lo));
            case CrateTypeId::Specifier: {
                // Store as specifier enum value
                switch (lo) {
                    case 0: return Value(Int(0));  // Def
                    case 1: return Value(Int(1));  // Over
                    case 2: return Value(Int(2));  // Class
                    default: return Value(Int(static_cast<int32_t>(lo)));
                }
            }
            case CrateTypeId::Variability:
                return Value(Int(static_cast<int32_t>(lo)));
            case CrateTypeId::ValueBlock:
                return Value(ValueBlock{});
            case CrateTypeId::Dictionary:
                // Inlined dictionary is always empty
                return Value(Dictionary{});
            case CrateTypeId::TimeCode: {
                // Inlined as float
                float f;
                std::memcpy(&f, &lo, 4);
                return Value(static_cast<TimeCode>(f), nullptr);
            }
            case CrateTypeId::PathExpression:
                return Value(GetToken(lo));  // Same as asset path
            case CrateTypeId::Half2:
                return DecodeInlinedVec<Half, 2>(payload);
            case CrateTypeId::Half3:
                return DecodeInlinedVec<Half, 3>(payload);
            case CrateTypeId::Half4:
                return DecodeInlinedVec<Half, 4>(payload);
            case CrateTypeId::Float2:
                return DecodeInlinedVec<float, 2>(payload);
            case CrateTypeId::Float3:
                return DecodeInlinedVec<float, 3>(payload);
            case CrateTypeId::Float4:
                return DecodeInlinedVec<float, 4>(payload);
            case CrateTypeId::Double2:
                return DecodeInlinedVec<double, 2>(payload);
            case CrateTypeId::Double3:
                return DecodeInlinedVec<double, 3>(payload);
            case CrateTypeId::Double4:
                return DecodeInlinedVec<double, 4>(payload);
            case CrateTypeId::Int2:
                return DecodeInlinedVec<int32_t, 2>(payload);
            case CrateTypeId::Int3:
                return DecodeInlinedVec<int32_t, 3>(payload);
            case CrateTypeId::Int4:
                return DecodeInlinedVec<int32_t, 4>(payload);
            case CrateTypeId::Quatf:
                return DecodeInlinedQuat<float>(payload);
            case CrateTypeId::Quatd:
                return DecodeInlinedQuat<double>(payload);
            case CrateTypeId::Quath:
                return DecodeInlinedQuat<Half>(payload);
            case CrateTypeId::Matrix2d:
                return DecodeInlinedDiagonalMatrix<2>(payload);
            case CrateTypeId::Matrix3d:
                return DecodeInlinedDiagonalMatrix<3>(payload);
            case CrateTypeId::Matrix4d:
                return DecodeInlinedDiagonalMatrix<4>(payload);
            default:
                return Value();
        }
    }

    static int8_t InlinedByte(uint64_t payload, size_t index) {
        return static_cast<int8_t>((payload >> (index * 8u)) & 0xFFu);
    }

    template <typename T>
    static T InlinedComponent(uint64_t payload, size_t index) {
        const int8_t v = InlinedByte(payload, index);
        if constexpr (std::is_same_v<T, Half>) {
            return Half(static_cast<float>(v));
        } else {
            return static_cast<T>(v);
        }
    }

    template <typename T, size_t N>
    Value DecodeInlinedVec(uint64_t payload) {
        Vec<T, N> v;
        for (size_t i = 0; i < N; ++i)
            v[i] = InlinedComponent<T>(payload, i);
        return Value(v);
    }

    template <typename T>
    Value DecodeInlinedQuat(uint64_t payload) {
        Quat<T> q;
        for (size_t i = 0; i < 4; ++i)
            q[i] = InlinedComponent<T>(payload, i);
        return Value(q);
    }

    template <size_t N>
    Value DecodeInlinedDiagonalMatrix(uint64_t payload) {
        Matrix<double, N, N> m;
        for (size_t i = 0; i < N; ++i)
            m(i, i) = static_cast<double>(InlinedByte(payload, i));
        return Value(m);
    }

    Value DecodeOffsetValue(CrateTypeId type, size_t offset) {
        size_t savedPos = reader_.Pos();
        reader_.Seek(offset);
        Value val = DecodeValueAtCurrentPos(type);
        reader_.Seek(savedPos);
        return val;
    }

    Value DecodeValueAtCurrentPos(CrateTypeId type) {
        switch (type) {
            case CrateTypeId::Int: return Value(reader_.ReadI32());
            case CrateTypeId::UInt: return Value(reader_.ReadU32());
            case CrateTypeId::Int64: return Value(static_cast<Int64>(reader_.ReadI64()));
            case CrateTypeId::UInt64: return Value(static_cast<UInt64>(reader_.ReadU64()));
            case CrateTypeId::Half: {
                Half h;
                reader_.Read(&h.bits);
                return Value(h);
            }
            case CrateTypeId::Float: return Value(reader_.ReadF32());
            case CrateTypeId::Double: return Value(reader_.ReadF64());
            case CrateTypeId::TimeCode: return Value(reader_.ReadF64(), nullptr);
            case CrateTypeId::String: return Value(GetString(reader_.ReadU32()));
            case CrateTypeId::Token: return Value(Token(GetToken(reader_.ReadU32())));
            case CrateTypeId::Asset: return Value(GetString(reader_.ReadU32()));

            // Vec types
            case CrateTypeId::Half2: return ReadVec<Half, 2>();
            case CrateTypeId::Half3: return ReadVec<Half, 3>();
            case CrateTypeId::Half4: return ReadVec<Half, 4>();
            case CrateTypeId::Float2: return ReadVec<float, 2>();
            case CrateTypeId::Float3: return ReadVec<float, 3>();
            case CrateTypeId::Float4: return ReadVec<float, 4>();
            case CrateTypeId::Double2: return ReadVec<double, 2>();
            case CrateTypeId::Double3: return ReadVec<double, 3>();
            case CrateTypeId::Double4: return ReadVec<double, 4>();
            case CrateTypeId::Int2: return ReadVec<int32_t, 2>();
            case CrateTypeId::Int3: return ReadVec<int32_t, 3>();
            case CrateTypeId::Int4: return ReadVec<int32_t, 4>();

            // Quaternions
            case CrateTypeId::Quath: return ReadQuat<Half>();
            case CrateTypeId::Quatf: return ReadQuat<float>();
            case CrateTypeId::Quatd: return ReadQuat<double>();

            // Matrices
            case CrateTypeId::Matrix2d: return ReadMatrix<2>();
            case CrateTypeId::Matrix3d: return ReadMatrix<3>();
            case CrateTypeId::Matrix4d: return ReadMatrix<4>();

            case CrateTypeId::Dictionary: return Value(ReadDictionary());
            case CrateTypeId::TimeSamples: return ReadTimeSamples();

            // ListOps
            case CrateTypeId::TokenListOp:
                return ReadTokenListOp();
            case CrateTypeId::StringListOp:
                return ReadStringListOp();
            case CrateTypeId::PathListOp:
                return ReadPathListOp();
            case CrateTypeId::ReferenceListOp:
                return ReadReferenceListOp();
            case CrateTypeId::PayloadListOp:
                return ReadPayloadListOp();

            // Vectors (variable-length arrays)
            case CrateTypeId::PathVector: return ReadPathVector();
            case CrateTypeId::TokenVector: return ReadTokenVector();
            case CrateTypeId::DoubleVector: return ReadDoubleVector();
            case CrateTypeId::LayerOffsetVector: return ReadLayerOffsetVector();
            case CrateTypeId::StringVector: return ReadStringVector();

            // Special types
            case CrateTypeId::Specifier: return Value(Int(reader_.ReadI32()));
            case CrateTypeId::Variability: return Value(Int(reader_.ReadI32()));
            case CrateTypeId::Permission: return Value(Int(reader_.ReadI32()));
            case CrateTypeId::VariantSelectionMap: return ReadVariantSelectionMap();
            case CrateTypeId::ValueBlock: return Value(ValueBlock{});

            case CrateTypeId::Value: {
                // Indirect value — read a ValueRep at the offset
                ValueRep rep;
                reader_.Read(&rep.data);
                return DecodeValue(rep);
            }

            case CrateTypeId::UnregisteredValue: {
                int64_t offset = reader_.ReadI64();
                size_t targetPos = reader_.Pos() - 8 + static_cast<size_t>(offset);
                size_t saved = reader_.Pos();
                reader_.Seek(targetPos);
                ValueRep rep;
                reader_.Read(&rep.data);
                Value v = DecodeValue(rep);
                reader_.Seek(saved);
                return v;
            }

            case CrateTypeId::Relocates: return ReadRelocates();

            case CrateTypeId::Splines: return ReadSpline();

            default:
                return Value();
        }
    }

    // Spline byte-stream helpers for Crate v0.12.0.
    enum class SplineCrateDataType : uint8_t {
        Unspecified = 0,
        Double = 1,
        Float = 2,
        Half = 3,
    };

    static void RequireSplineBytes(const BinaryReader& reader, size_t n) {
        if (!reader.CanRead(n)) {
            throw std::runtime_error("USDC spline data truncated");
        }
    }

    static ExtrapolationMode DecodeSplineExtrapolation(uint8_t value) {
        switch (value) {
            case 0: return ExtrapolationMode::None;
            case 1: return ExtrapolationMode::Held;
            case 2: return ExtrapolationMode::Linear;
            case 3: return ExtrapolationMode::Sloped;
            case 4: return ExtrapolationMode::LoopRepeat;
            case 5: return ExtrapolationMode::LoopReset;
            case 6: return ExtrapolationMode::LoopOscillate;
            default: return ExtrapolationMode::Held;
        }
    }

    static InterpolationMode DecodeSplineInterpolation(uint8_t value) {
        switch (value) {
            case 0: return InterpolationMode::None;
            case 1: return InterpolationMode::Held;
            case 2: return InterpolationMode::Linear;
            case 3: return InterpolationMode::Curve;
            default: return InterpolationMode::Held;
        }
    }

    static CurveType DecodeSplineCurveType(bool value) {
        return value ? CurveType::Hermite : CurveType::Bezier;
    }

    static size_t SplineTypedValueSize(SplineCrateDataType dataType) {
        switch (dataType) {
            case SplineCrateDataType::Float: return 4;
            case SplineCrateDataType::Half: return 2;
            case SplineCrateDataType::Unspecified:
            case SplineCrateDataType::Double:
            default: return 8;
        }
    }

    static double ReadSplineTypedValue(BinaryReader& reader,
                                       SplineCrateDataType dataType) {
        RequireSplineBytes(reader, SplineTypedValueSize(dataType));
        switch (dataType) {
            case SplineCrateDataType::Float:
                return static_cast<double>(reader.ReadF32());
            case SplineCrateDataType::Half: {
                Half h;
                reader.Read(&h.bits);
                return static_cast<double>(static_cast<float>(h));
            }
            case SplineCrateDataType::Unspecified:
            case SplineCrateDataType::Double:
            default:
                return reader.ReadF64();
        }
    }

    Value ReadSpline() {
        uint64_t byteCount = reader_.ReadU64();
        if (byteCount > static_cast<uint64_t>((std::numeric_limits<size_t>::max)())) {
            throw std::runtime_error("USDC spline data too large");
        }
        std::vector<uint8_t> bytes(static_cast<size_t>(byteCount));
        if (!bytes.empty() && !reader_.ReadBytes(bytes.data(), bytes.size())) {
            throw std::runtime_error("USDC spline data truncated");
        }

        Spline s;
        BinaryReader splineReader(bytes.data(), bytes.size());
        RequireSplineBytes(splineReader, 2);

        const uint8_t firstFlags = splineReader.ReadU8();
        const uint8_t splineVersion = firstFlags & 0x0F;
        if (splineVersion != 1) {
            throw std::runtime_error("Unsupported USDC spline version " +
                                     std::to_string(splineVersion));
        }

        SplineCrateDataType dataType =
            static_cast<SplineCrateDataType>((firstFlags & 0x30) >> 4);
        s.curveType = DecodeSplineCurveType((firstFlags & 0x80) != 0);

        const uint8_t secondFlags = splineReader.ReadU8();
        s.preExtrapolationMode =
            DecodeSplineExtrapolation(secondFlags & 0x07);
        s.postExtrapolationMode =
            DecodeSplineExtrapolation((secondFlags >> 3) & 0x07);
        const bool hasLoop = (secondFlags & 0x40) != 0;

        if (s.preExtrapolationMode == ExtrapolationMode::Sloped) {
            RequireSplineBytes(splineReader, 8);
            s.preExtrapolationSlope = splineReader.ReadF64();
        }
        if (s.postExtrapolationMode == ExtrapolationMode::Sloped) {
            RequireSplineBytes(splineReader, 8);
            s.postExtrapolationSlope = splineReader.ReadF64();
        }
        if (hasLoop) {
            RequireSplineBytes(splineReader, 32);
            s.loopParameters.protoStart = splineReader.ReadF64();
            s.loopParameters.protoEnd = splineReader.ReadF64();
            uint32_t numPreLoops = splineReader.ReadU32();
            uint32_t numPostLoops = splineReader.ReadU32();
            const uint32_t kMaxLoopCount =
                static_cast<uint32_t>((std::numeric_limits<int32_t>::max)());
            s.loopParameters.numPreLoops =
                static_cast<int>(numPreLoops < kMaxLoopCount
                    ? numPreLoops : kMaxLoopCount);
            s.loopParameters.numPostLoops =
                static_cast<int>(numPostLoops < kMaxLoopCount
                    ? numPostLoops : kMaxLoopCount);
            s.loopParameters.valueOffset = splineReader.ReadF64();
        }

        if (splineReader.Pos() < splineReader.Size()) {
            if (dataType == SplineCrateDataType::Unspecified) {
                dataType = SplineCrateDataType::Double;
            }
            RequireSplineBytes(splineReader, 4);
            uint32_t knotCount = splineReader.ReadU32();
            s.knots.reserve(knotCount);
            for (uint32_t i = 0; i < knotCount; ++i) {
                RequireSplineBytes(splineReader, 1 + 8);
                const uint8_t knotFlags = splineReader.ReadU8();
                const bool dualValued = (knotFlags & 0x01) != 0;
                const auto interpolation = DecodeSplineInterpolation(
                    static_cast<uint8_t>((knotFlags & 0x06) >> 1));
                const CurveType knotCurveType =
                    DecodeSplineCurveType((knotFlags & 0x08) != 0);

                SplineKnot k;
                k.nextInterpolationMode = interpolation;
                k.time = splineReader.ReadF64();
                k.value = ReadSplineTypedValue(splineReader, dataType);
                k.preValue = dualValued
                    ? ReadSplineTypedValue(splineReader, dataType)
                    : k.value;
                if (knotCurveType != CurveType::Hermite) {
                    RequireSplineBytes(splineReader, 16);
                    k.preTangentWidth = splineReader.ReadF64();
                    k.postTangentWidth = splineReader.ReadF64();
                }
                k.preTangentSlope = ReadSplineTypedValue(splineReader, dataType);
                k.postTangentSlope = ReadSplineTypedValue(splineReader, dataType);
                s.knots.push_back(k);
            }
        }

        uint64_t customDataCount = reader_.ReadU64();
        for (uint64_t i = 0; i < customDataCount; ++i) {
            (void)reader_.ReadF64();
            (void)ReadDictionary();
        }

        return Value(std::move(s));
    }

    // --- Type-specific readers ---

    template <typename T, size_t N>
    Value ReadVec() {
        Vec<T, N> v;
        for (size_t i = 0; i < N; ++i) {
            if constexpr (std::is_same_v<T, Half>) {
                reader_.Read(&v[i].bits);
            } else {
                reader_.Read(&v[i]);
            }
        }
        return Value(v);
    }

    template <typename T>
    Value ReadQuat() {
        // USDC binary stores {i, j, k, w} — matches internal {i, j, k, r}
        Quat<T> q;
        for (int i = 0; i < 4; ++i) {
            if constexpr (std::is_same_v<T, Half>) {
                reader_.Read(&q[i].bits);
            } else {
                reader_.Read(&q[i]);
            }
        }
        return Value(q);
    }

    template <size_t N>
    Value ReadMatrix() {
        Matrix<double, N, N> m;
        for (size_t i = 0; i < N * N; ++i) {
            m.data[i] = reader_.ReadF64();
        }
        return Value(m);
    }

    Dictionary ReadDictionary() {
        Dictionary dict;
        uint64_t numElements = reader_.ReadU64();
        for (uint64_t i = 0; i < numElements; ++i) {
            uint32_t keyIdx = reader_.ReadU32();
            int64_t valueOffset = reader_.ReadI64();
            std::string key = GetToken(keyIdx);

            // Value is at current position + valueOffset
            size_t valuePos = reader_.Pos() - 8 + static_cast<size_t>(valueOffset);
            size_t saved = reader_.Pos();
            reader_.Seek(valuePos);
            ValueRep rep;
            reader_.Read(&rep.data);
            Value val = DecodeValue(rep);
            reader_.Seek(saved);

            dict[key] = std::move(val);
        }
        return dict;
    }

    Value ReadTimeSamples() {
        // Offset to timecodes
        int64_t timecodesOffset = reader_.ReadI64();
        size_t timecodesPos = reader_.Pos() - 8 + static_cast<size_t>(timecodesOffset);

        size_t savedPos = reader_.Pos();

        // Read timecodes (DoubleVector)
        reader_.Seek(timecodesPos);
        ValueRep timecodesRep;
        reader_.Read(&timecodesRep.data);
        Value timecodesVal = DecodeValue(timecodesRep);

        // After the timecodes ValueRep, read the offset to values
        int64_t valuesOffset = reader_.ReadI64();
        size_t valuesPos = reader_.Pos() - 8 + static_cast<size_t>(valuesOffset);

        // Get the timecodes
        std::vector<double> times;
        if (auto* dv = timecodesVal.Get<std::vector<double>>()) {
            times = *dv;
        }

        // Read values
        reader_.Seek(valuesPos);
        uint64_t numValues = reader_.ReadU64();

        Dictionary tsDict;
        for (uint64_t i = 0; i < numValues && i < times.size(); ++i) {
            ValueRep valRep;
            reader_.Read(&valRep.data);
            Value val = DecodeValue(valRep);

            // Format time key
            double t = times[i];
            std::string key;
            if (t == static_cast<int64_t>(t)) {
                key = std::to_string(static_cast<int64_t>(t));
            } else {
                key = std::to_string(t);
            }
            tsDict[key] = std::move(val);
        }

        reader_.Seek(savedPos);
        return Value(std::move(tsDict));
    }

    // --- ListOp readers ---

    template <typename ReadItemFunc>
    auto ReadListOpItems(ReadItemFunc readItem) {
        using ItemT = decltype(readItem());
        uint64_t count = reader_.ReadU64();
        std::vector<ItemT> items;
        items.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i) {
            items.push_back(readItem());
        }
        return items;
    }

    Value ReadTokenListOp() {
        uint8_t header = reader_.ReadU8();
        // Spec §13.2.1.2: apiSchemas (the primary TokenListOp user) is
        // listop<token>, so decode natively into ListOp<Token> rather
        // than stringifying — preserves type through queries and
        // avoids a string→Token conversion at every read.
        ListOp<Token> listOp;
        auto readToken = [this]() -> Token {
            return Token(GetToken(reader_.ReadU32()));
        };

        bool makeExplicit = (header & 0x01) != 0;
        bool addExplicit = (header & 0x02) != 0;
        bool addItems = (header & 0x04) != 0;
        // bool deleteItems = (header & 0x08) != 0;  // unused directly
        // bool reorderItems = (header & 0x10) != 0;
        bool prependItems = (header & 0x20) != 0;
        bool appendItems = (header & 0x40) != 0;

        if (addExplicit) {
            auto items = ReadListOpItems(readToken);
            listOp.SetExplicitItems(std::move(items));
        }
        if (addItems) { ReadListOpItems(readToken); }  // Legacy, consume but ignore
        if (prependItems) {
            auto items = ReadListOpItems(readToken);
            if (!addExplicit) listOp.SetPrependedItems(std::move(items));
        }
        if (appendItems) {
            auto items = ReadListOpItems(readToken);
            if (!addExplicit) listOp.SetAppendedItems(std::move(items));
        }
        if (header & 0x08) {
            auto items = ReadListOpItems(readToken);
            if (!addExplicit) listOp.SetDeletedItems(std::move(items));
        }
        if (header & 0x10) { ReadListOpItems(readToken); }  // Reorder: legacy, consume

        if (makeExplicit && !addExplicit) {
            listOp.SetExplicitItems({});  // Cleared explicit
        }

        return Value(std::move(listOp));
    }

    Value ReadStringListOp() {
        // Same structure as token list op but uses string indices
        uint8_t header = reader_.ReadU8();
        ListOp<std::string> listOp;
        auto readStr = [this]() -> std::string { return GetString(reader_.ReadU32()); };

        bool addExplicit = (header & 0x02) != 0;
        bool addItems = (header & 0x04) != 0;
        bool prependItems = (header & 0x20) != 0;
        bool appendItems = (header & 0x40) != 0;
        bool makeExplicit = (header & 0x01) != 0;

        if (addExplicit) { auto items = ReadListOpItems(readStr); listOp.SetExplicitItems(std::move(items)); }
        if (addItems) { ReadListOpItems(readStr); }
        if (prependItems) { auto items = ReadListOpItems(readStr); if (!addExplicit) listOp.SetPrependedItems(std::move(items)); }
        if (appendItems) { auto items = ReadListOpItems(readStr); if (!addExplicit) listOp.SetAppendedItems(std::move(items)); }
        if (header & 0x08) { auto items = ReadListOpItems(readStr); if (!addExplicit) listOp.SetDeletedItems(std::move(items)); }
        if (header & 0x10) { ReadListOpItems(readStr); }

        if (makeExplicit && !addExplicit) { listOp.SetExplicitItems({}); }
        return Value(std::move(listOp));
    }

    Value ReadPathListOp() {
        uint8_t header = reader_.ReadU8();
        // Spec §6.6.3 list ops are typed by element. PathListOp's elements
        // are scene-object paths (§6.6.3 "may be extended to … scene
        // object paths"); decode into a typed ListOp<Path> rather than
        // downconverting to ListOp<std::string> so the type round-trips
        // through queries and downstream encoding unambiguously.
        ListOp<Path> listOp;
        auto readPath = [this]() -> Path {
            uint32_t idx = reader_.ReadU32();
            return GetPath(idx);
        };

        bool addExplicit = (header & 0x02) != 0;
        bool addItems = (header & 0x04) != 0;
        bool prependItems = (header & 0x20) != 0;
        bool appendItems = (header & 0x40) != 0;
        bool makeExplicit = (header & 0x01) != 0;

        if (addExplicit) { auto items = ReadListOpItems(readPath); listOp.SetExplicitItems(std::move(items)); }
        if (addItems) { ReadListOpItems(readPath); }
        if (prependItems) { auto items = ReadListOpItems(readPath); if (!addExplicit) listOp.SetPrependedItems(std::move(items)); }
        if (appendItems) { auto items = ReadListOpItems(readPath); if (!addExplicit) listOp.SetAppendedItems(std::move(items)); }
        if (header & 0x08) { auto items = ReadListOpItems(readPath); if (!addExplicit) listOp.SetDeletedItems(std::move(items)); }
        if (header & 0x10) { ReadListOpItems(readPath); }

        if (makeExplicit && !addExplicit) { listOp.SetExplicitItems({}); }
        return Value(std::move(listOp));
    }

    Reference ReadSingleReference() {
        Reference ref;
        uint32_t assetIdx = reader_.ReadU32();
        uint32_t primIdx = reader_.ReadU32();
        double offset = reader_.ReadF64();
        double scale = reader_.ReadF64();

        std::string asset = GetString(assetIdx);
        if (!asset.empty()) ref.assetPath = asset;

        Path primPath = GetPath(primIdx);
        if (!primPath.IsEmpty() && !primPath.IsAbsoluteRoot())
            ref.primPath = primPath;

        ref.offset = offset;
        ref.scale = scale;

        // Custom data (dictionary) for references
        // Read as a dictionary
        Dictionary customData = ReadDictionary();
        // We don't store customData on references for now

        return ref;
    }

    Value ReadReferenceListOp() {
        uint8_t header = reader_.ReadU8();
        ListOp<Reference> listOp;
        auto readRef = [this]() -> Reference { return ReadSingleReference(); };

        bool addExplicit = (header & 0x02) != 0;
        bool addItems = (header & 0x04) != 0;
        bool prependItems = (header & 0x20) != 0;
        bool appendItems = (header & 0x40) != 0;
        bool makeExplicit = (header & 0x01) != 0;

        if (addExplicit) { auto items = ReadListOpItems(readRef); listOp.SetExplicitItems(std::move(items)); }
        if (addItems) { ReadListOpItems(readRef); }
        if (prependItems) { auto items = ReadListOpItems(readRef); if (!addExplicit) listOp.SetPrependedItems(std::move(items)); }
        if (appendItems) { auto items = ReadListOpItems(readRef); if (!addExplicit) listOp.SetAppendedItems(std::move(items)); }
        if (header & 0x08) { auto items = ReadListOpItems(readRef); if (!addExplicit) listOp.SetDeletedItems(std::move(items)); }
        if (header & 0x10) { ReadListOpItems(readRef); }

        if (makeExplicit && !addExplicit) { listOp.SetExplicitItems({}); }
        return Value(std::move(listOp));
    }

    // Payloads differ from References: no customData dictionary, and
    // layerOffset only exists in crate version >= 0.10.
    Reference ReadSinglePayload() {
        Reference ref;
        uint32_t assetIdx = reader_.ReadU32();
        uint32_t primIdx = reader_.ReadU32();

        std::string asset = GetString(assetIdx);
        if (!asset.empty()) ref.assetPath = asset;

        Path primPath = GetPath(primIdx);
        if (!primPath.IsEmpty() && !primPath.IsAbsoluteRoot())
            ref.primPath = primPath;

        if (versionMinor_ >= 10) {
            ref.offset = reader_.ReadF64();
            ref.scale = reader_.ReadF64();
        }
        // No customData dictionary for payloads
        return ref;
    }

    Value ReadPayloadListOp() {
        uint8_t header = reader_.ReadU8();
        ListOp<Reference> listOp;
        auto readPayload = [this]() -> Reference { return ReadSinglePayload(); };

        bool addExplicit = (header & 0x02) != 0;
        bool addItems = (header & 0x04) != 0;
        bool prependItems = (header & 0x20) != 0;
        bool appendItems = (header & 0x40) != 0;
        bool makeExplicit = (header & 0x01) != 0;

        if (addExplicit) { auto items = ReadListOpItems(readPayload); listOp.SetExplicitItems(std::move(items)); }
        if (addItems) { ReadListOpItems(readPayload); }
        if (prependItems) { auto items = ReadListOpItems(readPayload); if (!addExplicit) listOp.SetPrependedItems(std::move(items)); }
        if (appendItems) { auto items = ReadListOpItems(readPayload); if (!addExplicit) listOp.SetAppendedItems(std::move(items)); }
        if (header & 0x08) { auto items = ReadListOpItems(readPayload); if (!addExplicit) listOp.SetDeletedItems(std::move(items)); }
        if (header & 0x10) { ReadListOpItems(readPayload); }

        if (makeExplicit && !addExplicit) { listOp.SetExplicitItems({}); }
        return Value(std::move(listOp));
    }

    // --- Vector readers ---

    Value ReadPathVector() {
        uint64_t count = reader_.ReadU64();
        std::vector<std::string> paths;
        for (uint64_t i = 0; i < count; ++i) {
            uint32_t idx = reader_.ReadU32();
            paths.push_back(GetPath(idx).GetText());
        }
        // Store as vector<string>
        return Value(Value::ArrayTag{}, TypeId::String, std::move(paths));
    }

    Value ReadTokenVector() {
        uint64_t count = reader_.ReadU64();
        std::vector<std::string> tokens;
        for (uint64_t i = 0; i < count; ++i) {
            tokens.push_back(GetToken(reader_.ReadU32()));
        }
        return Value(Value::ArrayTag{}, TypeId::String, std::move(tokens));
    }

    Value ReadDoubleVector() {
        uint64_t count = reader_.ReadU64();
        std::vector<double> values(static_cast<size_t>(count));
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = reader_.ReadF64();
        }
        return Value(Value::ArrayTag{}, TypeId::Double, std::move(values));
    }

    Value ReadLayerOffsetVector() {
        uint64_t count = reader_.ReadU64();
        SubLayerPaths result;
        // Layer offsets come as pairs of doubles
        for (uint64_t i = 0; i < count; ++i) {
            Retiming r;
            r.offset = reader_.ReadF64();
            r.scale = reader_.ReadF64();
            result.offsets.push_back(r);
        }
        // This is just the offsets — paths come from a separate field
        // Store as vector<double> with offset/scale interleaved
        std::vector<double> flat;
        for (const auto& r : result.offsets) {
            flat.push_back(r.offset);
            flat.push_back(r.scale);
        }
        return Value(Value::ArrayTag{}, TypeId::Double, std::move(flat));
    }

    Value ReadStringVector() {
        uint64_t count = reader_.ReadU64();
        std::vector<std::string> strs;
        for (uint64_t i = 0; i < count; ++i) {
            strs.push_back(GetString(reader_.ReadU32()));
        }
        return Value(Value::ArrayTag{}, TypeId::String, std::move(strs));
    }

    Value ReadVariantSelectionMap() {
        uint64_t count = reader_.ReadU64();
        Dictionary dict;
        for (uint64_t i = 0; i < count; ++i) {
            std::string key = GetString(reader_.ReadU32());
            std::string val = GetString(reader_.ReadU32());
            dict[key] = Value(val);
        }
        return Value(std::move(dict));
    }

    Value ReadRelocates() {
        uint64_t count = reader_.ReadU64();
        std::vector<Relocate> relocates;
        for (uint64_t i = 0; i < count; ++i) {
            Relocate r;
            r.sourcePath = GetPath(reader_.ReadU32());
            uint32_t targetIdx = reader_.ReadU32();
            if (targetIdx == 0) {
                r.targetPath = std::nullopt;
            } else {
                r.targetPath = GetPath(targetIdx);
            }
            relocates.push_back(std::move(r));
        }
        return Value(std::move(relocates));
    }

    // --- Array value decoding ---
    Value DecodeArrayValue(CrateTypeId type, size_t offset, bool compressed) {
        if (offset == 0) {
            // Empty array
            return MakeEmptyArray(type);
        }

        size_t saved = reader_.Pos();
        reader_.Seek(offset);

        uint64_t numElements = ReadArrayElementCount(reader_, versionMinor_);
        Value result;

        if (compressed) {
            result = DecodeCompressedArray(type, static_cast<size_t>(numElements));
        } else {
            result = DecodeUncompressedArray(type, static_cast<size_t>(numElements));
        }

        reader_.Seek(saved);
        return result;
    }

    Value MakeEmptyArray(CrateTypeId type) {
        switch (type) {
            case CrateTypeId::Bool: return Value(Value::ArrayTag{}, TypeId::Bool, std::vector<Bool>{});
            case CrateTypeId::UChar: return Value(Value::ArrayTag{}, TypeId::UChar, std::vector<UChar>{});
            case CrateTypeId::Int: return Value(Value::ArrayTag{}, TypeId::Int, std::vector<Int>{});
            case CrateTypeId::UInt: return Value(Value::ArrayTag{}, TypeId::UInt, std::vector<UInt>{});
            case CrateTypeId::Int64: return Value(Value::ArrayTag{}, TypeId::Int64, std::vector<Int64>{});
            case CrateTypeId::UInt64: return Value(Value::ArrayTag{}, TypeId::UInt64, std::vector<UInt64>{});
            case CrateTypeId::Half: return Value(Value::ArrayTag{}, TypeId::Half, std::vector<Half>{});
            case CrateTypeId::Float: return Value(Value::ArrayTag{}, TypeId::Float, std::vector<Float>{});
            case CrateTypeId::Double: return Value(Value::ArrayTag{}, TypeId::Double, std::vector<Double>{});
            case CrateTypeId::String: return Value(Value::ArrayTag{}, TypeId::String, std::vector<String>{});
            case CrateTypeId::Token: return Value(Value::ArrayTag{}, TypeId::Token, std::vector<String>{});
            case CrateTypeId::Asset: return Value(Value::ArrayTag{}, TypeId::Asset, std::vector<String>{});
            case CrateTypeId::Float2: return Value(Value::ArrayTag{}, TypeId::Float2, std::vector<GfVec2f>{});
            case CrateTypeId::Float3: return Value(Value::ArrayTag{}, TypeId::Float3, std::vector<GfVec3f>{});
            case CrateTypeId::Float4: return Value(Value::ArrayTag{}, TypeId::Float4, std::vector<GfVec4f>{});
            case CrateTypeId::Double2: return Value(Value::ArrayTag{}, TypeId::Double2, std::vector<GfVec2d>{});
            case CrateTypeId::Double3: return Value(Value::ArrayTag{}, TypeId::Double3, std::vector<GfVec3d>{});
            case CrateTypeId::Double4: return Value(Value::ArrayTag{}, TypeId::Double4, std::vector<GfVec4d>{});
            case CrateTypeId::Int2: return Value(Value::ArrayTag{}, TypeId::Int2, std::vector<GfVec2i>{});
            case CrateTypeId::Int3: return Value(Value::ArrayTag{}, TypeId::Int3, std::vector<GfVec3i>{});
            case CrateTypeId::Int4: return Value(Value::ArrayTag{}, TypeId::Int4, std::vector<GfVec4i>{});
            case CrateTypeId::Half2: return Value(Value::ArrayTag{}, TypeId::Half2, std::vector<GfVec2h>{});
            case CrateTypeId::Half3: return Value(Value::ArrayTag{}, TypeId::Half3, std::vector<GfVec3h>{});
            case CrateTypeId::Half4: return Value(Value::ArrayTag{}, TypeId::Half4, std::vector<GfVec4h>{});
            case CrateTypeId::Matrix2d: return Value(Value::ArrayTag{}, TypeId::Matrix2d, std::vector<GfMatrix2d>{});
            case CrateTypeId::Matrix3d: return Value(Value::ArrayTag{}, TypeId::Matrix3d, std::vector<GfMatrix3d>{});
            case CrateTypeId::Matrix4d: return Value(Value::ArrayTag{}, TypeId::Matrix4d, std::vector<GfMatrix4d>{});
            case CrateTypeId::Quatd: return Value(Value::ArrayTag{}, TypeId::Quatd, std::vector<GfQuatd>{});
            case CrateTypeId::Quatf: return Value(Value::ArrayTag{}, TypeId::Quatf, std::vector<GfQuatf>{});
            case CrateTypeId::Quath: return Value(Value::ArrayTag{}, TypeId::Quath, std::vector<GfQuath>{});
            default: return Value();
        }
    }

    template <typename T, TypeId TID>
    Value ReadUncompressedPodArray(size_t n) {
        std::vector<T> arr(n);
        reader_.ReadBytes(arr.data(), n * sizeof(T));
        return Value(Value::ArrayTag{}, TID, std::move(arr));
    }

    Value ReadUncompressedBoolArray(size_t n) {
        // std::vector<bool> is special in C++ (bitfield) — can't use .data()
        std::vector<Bool> arr(n);
        for (size_t i = 0; i < n; ++i) {
            uint8_t b = 0;
            reader_.ReadBytes(&b, 1);
            arr[i] = (b != 0);
        }
        return Value(Value::ArrayTag{}, TypeId::Bool, std::move(arr));
    }

    Value DecodeUncompressedArray(CrateTypeId type, size_t n) {
        switch (type) {
            case CrateTypeId::Bool: return ReadUncompressedBoolArray(n);
            case CrateTypeId::UChar: return ReadUncompressedPodArray<UChar, TypeId::UChar>(n);
            case CrateTypeId::Int: return ReadUncompressedPodArray<Int, TypeId::Int>(n);
            case CrateTypeId::UInt: return ReadUncompressedPodArray<UInt, TypeId::UInt>(n);
            case CrateTypeId::Int64: return ReadUncompressedPodArray<Int64, TypeId::Int64>(n);
            case CrateTypeId::UInt64: return ReadUncompressedPodArray<UInt64, TypeId::UInt64>(n);
            case CrateTypeId::Half: return ReadUncompressedPodArray<Half, TypeId::Half>(n);
            case CrateTypeId::Float: return ReadUncompressedPodArray<Float, TypeId::Float>(n);
            case CrateTypeId::Double: return ReadUncompressedPodArray<Double, TypeId::Double>(n);
            case CrateTypeId::String: {
                std::vector<String> arr;
                for (size_t i = 0; i < n; ++i) arr.push_back(GetString(reader_.ReadU32()));
                return Value(Value::ArrayTag{}, TypeId::String, std::move(arr));
            }
            case CrateTypeId::Token: {
                std::vector<String> arr;
                for (size_t i = 0; i < n; ++i) arr.push_back(GetToken(reader_.ReadU32()));
                return Value(Value::ArrayTag{}, TypeId::Token, std::move(arr));
            }
            case CrateTypeId::Asset: {
                std::vector<String> arr;
                for (size_t i = 0; i < n; ++i) arr.push_back(GetString(reader_.ReadU32()));
                return Value(Value::ArrayTag{}, TypeId::Asset, std::move(arr));
            }
            case CrateTypeId::Float2: return ReadUncompressedPodArray<GfVec2f, TypeId::Float2>(n);
            case CrateTypeId::Float3: return ReadUncompressedPodArray<GfVec3f, TypeId::Float3>(n);
            case CrateTypeId::Float4: return ReadUncompressedPodArray<GfVec4f, TypeId::Float4>(n);
            case CrateTypeId::Double2: return ReadUncompressedPodArray<GfVec2d, TypeId::Double2>(n);
            case CrateTypeId::Double3: return ReadUncompressedPodArray<GfVec3d, TypeId::Double3>(n);
            case CrateTypeId::Double4: return ReadUncompressedPodArray<GfVec4d, TypeId::Double4>(n);
            case CrateTypeId::Int2: return ReadUncompressedPodArray<GfVec2i, TypeId::Int2>(n);
            case CrateTypeId::Int3: return ReadUncompressedPodArray<GfVec3i, TypeId::Int3>(n);
            case CrateTypeId::Int4: return ReadUncompressedPodArray<GfVec4i, TypeId::Int4>(n);
            case CrateTypeId::Half2: return ReadUncompressedPodArray<GfVec2h, TypeId::Half2>(n);
            case CrateTypeId::Half3: return ReadUncompressedPodArray<GfVec3h, TypeId::Half3>(n);
            case CrateTypeId::Half4: return ReadUncompressedPodArray<GfVec4h, TypeId::Half4>(n);
            case CrateTypeId::Matrix2d: return ReadUncompressedPodArray<GfMatrix2d, TypeId::Matrix2d>(n);
            case CrateTypeId::Matrix3d: return ReadUncompressedPodArray<GfMatrix3d, TypeId::Matrix3d>(n);
            case CrateTypeId::Matrix4d: return ReadUncompressedPodArray<GfMatrix4d, TypeId::Matrix4d>(n);
            case CrateTypeId::Quatd: return ReadUncompressedPodArray<GfQuatd, TypeId::Quatd>(n);
            case CrateTypeId::Quatf: return ReadUncompressedPodArray<GfQuatf, TypeId::Quatf>(n);
            case CrateTypeId::Quath: return ReadUncompressedPodArray<GfQuath, TypeId::Quath>(n);
            case CrateTypeId::TimeCode: return ReadUncompressedPodArray<Double, TypeId::TimeCode>(n);
            default: return Value();
        }
    }

    Value DecodeCompressedArray(CrateTypeId type, size_t n) {
        // Compressed arrays: integral types use compressed integer arrays,
        // floating point types use either 'i' (integer) or 't' (LUT) encoding
        switch (type) {
            case CrateTypeId::Int: return DecodeCompressedIntArray<Int, TypeId::Int>(n);
            case CrateTypeId::UInt: return DecodeCompressedIntArray<UInt, TypeId::UInt>(n);
            case CrateTypeId::Int64: return DecodeCompressedIntArray<Int64, TypeId::Int64>(n);
            case CrateTypeId::UInt64: return DecodeCompressedIntArray<UInt64, TypeId::UInt64>(n);
            case CrateTypeId::Half: return DecodeCompressedFloatArray<Half, TypeId::Half>(n);
            case CrateTypeId::Float: return DecodeCompressedFloatArray<Float, TypeId::Float>(n);
            case CrateTypeId::Double: return DecodeCompressedFloatArray<Double, TypeId::Double>(n);
            case CrateTypeId::String: {
                // Compressed string indices
                std::vector<uint32_t> indices(n);
                if (!DecodeCompressedInts(reader_, indices)) return Value();
                std::vector<String> arr;
                for (auto idx : indices) arr.push_back(GetString(idx));
                return Value(Value::ArrayTag{}, TypeId::String, std::move(arr));
            }
            case CrateTypeId::Token: {
                std::vector<uint32_t> indices(n);
                if (!DecodeCompressedInts(reader_, indices)) return Value();
                std::vector<String> arr;
                for (auto idx : indices) arr.push_back(GetToken(idx));
                return Value(Value::ArrayTag{}, TypeId::Token, std::move(arr));
            }
            default:
                // Types that don't support compression — read uncompressed
                return DecodeUncompressedArray(type, n);
        }
    }

    template <typename IntT, TypeId TID>
    Value DecodeCompressedIntArray(size_t n) {
        std::vector<IntT> arr(n);
        if (!DecodeCompressedInts(reader_, arr)) return Value();
        return Value(Value::ArrayTag{}, TID, std::move(arr));
    }

    template <typename FloatT, TypeId TID>
    Value DecodeCompressedFloatArray(size_t n) {
        // Read encoding character
        char encoding = static_cast<char>(reader_.ReadU8());

        if (encoding == 'i') {
            // Integer-encoded floats
            if constexpr (sizeof(FloatT) <= 4) {
                std::vector<int32_t> ints(n);
                if (!DecodeCompressedInts(reader_, ints)) return Value();
                std::vector<FloatT> arr(n);
                for (size_t i = 0; i < n; ++i) {
                    arr[i] = static_cast<FloatT>(ints[i]);
                }
                return Value(Value::ArrayTag{}, TID, std::move(arr));
            } else {
                std::vector<int64_t> ints(n);
                if (!DecodeCompressedInts(reader_, ints)) return Value();
                std::vector<FloatT> arr(n);
                for (size_t i = 0; i < n; ++i) {
                    arr[i] = static_cast<FloatT>(ints[i]);
                }
                return Value(Value::ArrayTag{}, TID, std::move(arr));
            }
        }

        if (encoding == 't') {
            // LUT-encoded floats
            uint32_t lutCount = reader_.ReadU32();
            std::vector<FloatT> lut(lutCount);
            reader_.ReadBytes(lut.data(), lutCount * sizeof(FloatT));

            // Read compressed indices into the LUT
            std::vector<uint32_t> indices(n);
            if (!DecodeCompressedInts(reader_, indices)) return Value();

            std::vector<FloatT> arr(n);
            for (size_t i = 0; i < n; ++i) {
                uint32_t idx = indices[i];
                arr[i] = (idx < lutCount) ? lut[idx] : FloatT{};
            }
            return Value(Value::ArrayTag{}, TID, std::move(arr));
        }

        // Unknown encoding — fall back to uncompressed
        return DecodeUncompressedArray(static_cast<CrateTypeId>(0), n);
    }

    // ============================================================
    // Build Layer from decoded data
    // ============================================================

    void BuildLayer() {
        // The Layer's decode backing — owned by the Layer below,
        // referenced by every deferred Spec via raw pointer. Built
        // empty up front so we can hand `state.get()` to specs
        // during the per-spec loop; populated with section vectors
        // afterwards (parser still reads them during the loop).
        std::shared_ptr<UsdcReaderState> state;
        if (fileBuffer_) {
            state = std::make_shared<UsdcReaderState>();
        }
        const DecodeBacking* backingPtr = state.get();  // stable for state's lifetime

        auto store = std::make_unique<HashMapSpecStore>();
        // Pre-size the spec store's bucket array so the per-spec
        // SetSpecNoIndex below doesn't pay ~log2(n) rehashes during
        // the bulk load. Most specs are non-Layer; the LayerSpec is
        // pulled out separately, so this slightly over-reserves but
        // never causes premature rehash.
        store->Reserve(specPathIndices_.size());
        Spec layerSpec(SpecType::Layer);

        for (size_t i = 0; i < specPathIndices_.size(); ++i) {
            uint32_t pathIdx = specPathIndices_[i];
            uint32_t fieldSetIdx = specFieldSetIndices_[i];
            uint32_t form = specForms_[i];

            Path path = GetPath(pathIdx);
            SpecType specType = SpecFormToSpecType(form);

            Spec spec(specType);

            if (state) {
                // Per-field laziness: the spec carries only a
                // (backing, fieldsetIdx) pair. First GetField call
                // asks the backing to decode that single field.
                // No per-spec lambda allocation, no per-spec
                // captured vector — just 12 bytes of context.
                spec.SetDeferralContext(backingPtr, fieldSetIdx);
            } else {
                // No file buffer → eager decode (fallback path for
                // in-memory crate buffers that don't outlive parse).
                for (size_t fi = fieldSetIdx;
                     fi < flatFieldSetIndices_.size(); ++fi) {
                    uint32_t fieldIdx = flatFieldSetIndices_[fi];
                    if (fieldIdx == 0xFFFFFFFF) break;
                    if (fieldIdx >= fieldTokenIndices_.size()) continue;
                    uint32_t fieldTokenIdx = fieldTokenIndices_[fieldIdx];
                    if (fieldTokenIdx >= tokenPool_.size()) continue;
                    const Token& fieldTok = tokenPool_[fieldTokenIdx];
                    const ValueRep& valRep = fieldValueReps_[fieldIdx];
                    Value val = DecodeValue(valRep);
                    ApplyField(spec, specType, fieldTok, std::move(val));
                }
            }

            if (specType == SpecType::Layer) {
                layerSpec = std::move(spec);
            } else {
                store->SetSpecNoIndex(path, std::move(spec));
            }
        }

        // Inject pre-built indices from ReconstructPaths (avoids per-spec AddToIndex)
        store->SetIndices(std::move(prebuiltPrimChildren_),
                          std::move(prebuiltPrimChildPaths_),
                          std::move(prebuiltPrimProperties_));
        layer_ = Layer(std::move(store));
        // Restore the layer spec (Layer constructor creates a fresh one)
        layer_.GetLayerSpec() = std::move(layerSpec);

        // Now that the per-spec loop is done with its direct reads,
        // move the section vectors into the backing state so the
        // backing's DecodeField/DecodeAllFields can use them.
        if (state) {
            state->fileBuffer = fileBuffer_;
            state->tokens = std::move(tokens_);
            state->stringIndices = std::move(stringIndices_);
            state->fieldTokenIndices = std::move(fieldTokenIndices_);
            state->fieldValueReps = std::move(fieldValueReps_);
            state->flatFieldSetIndices = std::move(flatFieldSetIndices_);
            state->paths = std::move(paths_);
            state->versionMinor = versionMinor_;
            // Hand the populated backing to the Layer; Specs already
            // hold raw pointers to it.
            layer_.SetDecodeBacking(state);
        }

        PostProcessSubLayers();
    }

public:
    // Static and public so the Layer's `DecodeBacking` impl can call
    // it from outside the parser (after the parser has been
    // destroyed) to apply per-field transforms during deferred
    // decode.
    static void ApplyField(Spec& spec, SpecType specType, const Token& fieldTok, Value val) {
        // Token-typed comparisons — each branch is a pointer compare.
        if (fieldTok == FieldNames::specifier) {
            if (auto* intVal = val.Get<Int>()) {
                switch (*intVal) {
                    case 0: spec.SetSpecifier(Specifier::Def); break;
                    case 1: spec.SetSpecifier(Specifier::Over); break;
                    case 2: spec.SetSpecifier(Specifier::Class); break;
                }
            }
            return;
        }
        if (fieldTok == FieldNames::variability) {
            if (auto* intVal = val.Get<Int>()) {
                spec.SetField(fieldTok, *intVal == 1 ?
                    Value(String("uniform")) : Value(String("varying")));
            }
            return;
        }
        if (fieldTok == FieldNames::typeName) {
            if (auto* t = val.Get<Token>()) {
                spec.SetTypeName(*t);
            } else if (auto* s = val.Get<String>()) {
                spec.SetTypeName(Token(*s));
            } else {
                spec.SetField(fieldTok, std::move(val));
            }
            return;
        }
        if (fieldTok == FieldNames::active) {
            if (auto* b = val.Get<Bool>()) spec.SetActive(*b);
            return;
        }
        if (fieldTok == FieldNames::hidden) {
            if (auto* b = val.Get<Bool>()) spec.SetHidden(*b);
            return;
        }
        if (fieldTok == FieldNames::custom) {
            if (auto* b = val.Get<Bool>()) spec.SetCustom(*b);
            return;
        }
        // The USDC encoding uses the literal field name "default" for
        // attribute default values; the spec/Layer model exposes it as
        // "defaultValue". Cheap one-time intern at static init time:
        static const Token kUsdcDefault("default");
        if (fieldTok == kUsdcDefault) {
            spec.SetField(FieldNames::defaultValue, std::move(val));
            return;
        }
        static const Token kUsdcRelocates("relocates");
        if (specType == SpecType::Layer && fieldTok == kUsdcRelocates) {
            spec.SetField(FieldNames::layerRelocates, std::move(val));
            return;
        }
        // All other fields: store under the already-interned Token.
        spec.SetField(fieldTok, std::move(val));
    }

    void PostProcessSubLayers() {
        auto& ls = layer_.GetLayerSpec();
        auto* slField = ls.GetField(Token("subLayers"));
        auto* slOffsets = ls.GetField(Token("subLayerOffsets"));

        if (slField) {
            SubLayerPaths slp;
            // subLayers is stored as a vector of asset paths
            if (auto* arr = slField->Get<std::vector<String>>()) {
                slp.paths = *arr;
            }

            // subLayerOffsets is stored as interleaved doubles
            if (slOffsets) {
                if (auto* dblArr = slOffsets->Get<std::vector<double>>()) {
                    for (size_t i = 0; i + 1 < dblArr->size(); i += 2) {
                        Retiming r;
                        r.offset = (*dblArr)[i];
                        r.scale = (*dblArr)[i + 1];
                        slp.offsets.push_back(r);
                    }
                }
            }

            // Pad offsets to match paths
            while (slp.offsets.size() < slp.paths.size()) {
                slp.offsets.push_back({0.0, 1.0});
            }

            if (!slp.paths.empty()) {
                ls.SetField(FieldNames::subLayers, Value(std::move(slp)));
            }

            // Remove raw fields
            ls.RemoveField(Token("subLayerOffsets"));
        }
    }

    // --- Data ---
    BinaryReader reader_;
    uint8_t versionMajor_ = 0, versionMinor_ = 0, versionPatch_ = 0;
    uint64_t tocOffset_ = 0;
    std::map<std::string, TocSection> sections_;

    std::vector<std::string> tokens_;
    // Preinterned Tokens paralleling tokens_ — populated once after
    // ReadTokens and consumed by ReconstructPaths, BuildLayer, and
    // ApplyField so each token-by-index lookup is a pointer-sized fetch
    // instead of a string copy + per-use Intern() round-trip.
    std::vector<Token> tokenPool_;
    std::vector<uint32_t> stringIndices_;
    std::vector<uint32_t> fieldTokenIndices_;
    std::vector<ValueRep> fieldValueReps_;
    std::vector<uint32_t> flatFieldSetIndices_;
    std::vector<Path> paths_;

    std::vector<uint32_t> specPathIndices_;
    std::vector<uint32_t> specFieldSetIndices_;
    std::vector<uint32_t> specForms_;

    // Pre-built indices from ReadPaths (built during ReconstructPaths for free)
    PathMap<std::vector<Token>> prebuiltPrimChildren_;
    PathMap<std::vector<Path>> prebuiltPrimChildPaths_;
    PathMap<std::vector<Token>> prebuiltPrimProperties_;

    Layer layer_;

    // When set, enables lazy array decoding — file buffer stays alive
    // via shared_ptr so lazy thunks can decode on first access.
    // Backed by either mmap or heap vector, via FileBuffer type erasure.
    std::shared_ptr<FileBuffer> fileBuffer_;

    // When non-null, decode-only mode: use state's vectors instead of member vectors.
    const UsdcReaderState* stateRef_ = nullptr;
};

// --- UsdcReaderState DecodeBacking implementation ---
//
// Walks the spec's authored fieldset (a sequence of field indices
// terminated by 0xFFFFFFFF), decodes either a specific named field
// (DecodeField) or all of them (DecodeAllFields). Both share the
// same per-field decode logic; the only difference is whether the
// loop breaks on first match.

namespace {

// Decode a single fieldIdx into a Value. Mirrors the eager branch
// of the old BuildLayer per-spec loop: lazy POD-array thunk where
// possible, full decode otherwise.
Value DecodeFieldByIdx(const UsdcReaderState* state, uint32_t fieldIdx) {
    if (fieldIdx >= state->fieldValueReps.size()) return Value();
    const ValueRep& valRep = state->fieldValueReps[fieldIdx];
    if (valRep.IsArray() && !valRep.IsInlined()
        && valRep.GetPayload() != 0
        && IsLazyEligibleArray(valRep.GetType())) {
        TypeId tid = CrateTypeIdToTypeId(valRep.GetType());
        auto buf = state->fileBuffer;
        uint64_t repData = valRep.data;
        uint8_t versionMinor = state->versionMinor;
        auto thunk = std::make_shared<Value::LazyThunk>(
            [buf, repData, versionMinor](std::any& outData, size_t& outSize) {
                ValueRep r;
                r.data = repData;
                LazyDecodePodArray(buf->data(), buf->size(),
                    r.GetType(),
                    static_cast<size_t>(r.GetPayload()),
                    r.IsCompressed(),
                    versionMinor,
                    outData, outSize);
            });
        return Value::MakeLazy(tid, true, std::move(thunk));
    }
    UsdcReader tempReader(state);
    return tempReader.DecodeFieldValue(valRep);
}

} // anonymous

Value UsdcReaderState::DecodeField(uint32_t fieldsetIdx,
                                    const Token& fieldName) const {
    // The crate stores `defaultValue` as the literal token "default".
    // Translate the public name back at the boundary.
    static const Token kUsdcDefault("default");
    const std::string& wantedName =
        fieldName == FieldNames::defaultValue
            ? kUsdcDefault.GetString() : fieldName.GetString();
    static const Token kUsdcRelocates("relocates");
    const std::string& alternateName =
        fieldName == FieldNames::layerRelocates
            ? kUsdcRelocates.GetString() : wantedName;

    for (size_t fi = fieldsetIdx; fi < flatFieldSetIndices.size(); ++fi) {
        uint32_t fieldIdx = flatFieldSetIndices[fi];
        if (fieldIdx == 0xFFFFFFFF) break;
        if (fieldIdx >= fieldTokenIndices.size()) continue;
        uint32_t tokIdx = fieldTokenIndices[fieldIdx];
        if (tokIdx >= tokens.size()) continue;
        if (tokens[tokIdx] != wantedName && tokens[tokIdx] != alternateName)
            continue;

        Value val = DecodeFieldByIdx(this, fieldIdx);
        // Apply per-field transforms that the old eager-path
        // ApplyField did (variability Int → "uniform"/"varying"
        // string is the one that matters for legacy callers; other
        // transforms are no-ops once the value is in fields_).
        if (fieldName == FieldNames::variability) {
            if (auto* iv = val.Get<Int>()) {
                return *iv == 1 ? Value(String("uniform"))
                                : Value(String("varying"));
            }
        }
        return val;
    }
    return Value();
}

void UsdcReaderState::DecodeAllFields(uint32_t fieldsetIdx,
                                       Spec& target) const {
    SpecType specType = target.GetType();
    for (size_t fi = fieldsetIdx; fi < flatFieldSetIndices.size(); ++fi) {
        uint32_t fieldIdx = flatFieldSetIndices[fi];
        if (fieldIdx == 0xFFFFFFFF) break;
        if (fieldIdx >= fieldTokenIndices.size()) continue;
        uint32_t tokIdx = fieldTokenIndices[fieldIdx];
        if (tokIdx >= tokens.size()) continue;
        Token fieldTok(tokens[tokIdx]);

        Value val = DecodeFieldByIdx(this, fieldIdx);
        UsdcReader::ApplyField(target, specType, fieldTok, std::move(val));
    }
}

} // anonymous namespace

// ============================================================
// Public API
// ============================================================

bool IsUsdcFormat(const uint8_t* data, size_t size) {
    if (size < 8) return false;
    return std::memcmp(data, "PXR-USDC", 8) == 0;
}

UsdcParseResult ParseUsdc(const uint8_t* data, size_t size) {
    UsdcReader reader(data, size);
    return reader.Parse();
}

namespace {

UsdcParseResult ParseUsdcOwnedBuffer(std::shared_ptr<std::vector<uint8_t>> vec) {
    auto fb = std::make_shared<FileBuffer>();
    fb->data_ = vec->data();
    fb->size_ = vec->size();
    fb->owner_ = std::move(vec);

    UsdcReader reader(fb->data(), fb->size());
    reader.SetFileBuffer(fb);
    return reader.Parse();
}

} // anonymous namespace

UsdcParseResult ParseUsdcFile(const ResolvedLocation& location) {
    std::string localPath;
    if (IsLocalFileResource(location, &localPath)) {
        // Try mmap first: file-backed pages are reclaimable under memory pressure,
        // critical for composed scenes with hundreds of sublayers.
        auto mmap = std::make_shared<MmapHandle>(localPath.c_str());
        if (mmap->valid()) {
            auto fb = std::make_shared<FileBuffer>();
            fb->data_ = mmap->data();
            fb->size_ = mmap->size();
            fb->owner_ = mmap;

            mmap->advise_sequential();  // linear read during parse
            UsdcReader reader(fb->data(), fb->size());
            reader.SetFileBuffer(fb);
            auto result = reader.Parse();
            mmap->advise_random();      // lazy thunks access at arbitrary offsets
            return result;
        }
    }

    // Fallback: heap-allocated buffer (pipes, unsupported filesystems, Windows,
    // and byte-backed package resources).
    auto resource = ReadResource(location);
    if (!resource.success) {
        UsdcParseResult result;
        result.error = std::move(resource.error);
        return result;
    }

    auto vec = std::make_shared<std::vector<uint8_t>>(std::move(resource.bytes));
    return ParseUsdcOwnedBuffer(std::move(vec));
}

UsdcParseResult ParseUsdcFile(const std::string& filePath) {
    return ParseUsdcFile(ResolvedLocation::FromResolvedString(filePath));
}

} // namespace nanousd
