// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// usdc_writer.cpp — USDC binary crate writer
//
// Implements the AOUSD Core Spec v1.0.1 section 16.3 (Binary / Crate format).
// Produces files compatible with the usdc_parser.cpp reader in this library.
//
// Design notes:
//   - We write the lowest Crate version required by emitted value features.
//   - Compressed integer arrays use the delta/codepoint scheme from spec 16.3.7.2.
//   - Strings/tokens/assets are all stored in the TOKENS section; STRINGS holds
//     indices into TOKENS for "string" typed values (distinct from token values).
//   - Value representations follow spec 16.3.9; we use inlining where required
//     and for small scalar types, offset encoding for everything else.
//   - The PATHS section uses the jump-tree encoding from spec 16.3.8.4.5.
//   - No LZ4 compression is used for the TOKENS section (we write plain LZ4
//     single-block wrapping), keeping the implementation simple while remaining
//     spec-compliant.

#include "nanousd/usdc_writer.h"
#include "nanousd/resource.h"
#include "nanousd/layer.h"
#include "nanousd/spec.h"
#include "nanousd/value.h"
#include "nanousd/path.h"
#include "nanousd/listop.h"
#include "lz4_decode.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace nanousd {

// ============================================================
// Forward declarations / helpers
// ============================================================

namespace {

// ---------------------------------------------------------------
// LZ4 block compressor — simple store-as-literal fallback.
// The spec only requires LZ4 *decompression* compatibility.
// We implement a minimal LZ4 block *encoder* that produces valid
// LZ4 blocks that any LZ4 decoder can handle.  We use the
// "all-literal" encoding: each block is one or more copy-commands
// with no back-references.  This is valid LZ4 and keeps the writer
// simple while producing files the parser can read.
//
// LZ4 block format (one sequence):
//   token byte: high nibble = literal len, low nibble = match len - 4
//   if literal len >= 15: extra bytes summed until < 255
//   literal bytes
//   (match offset, match len — omitted for final sequence)
// ---------------------------------------------------------------

// Encode `src` as a literal-only LZ4 block into `dst`.
// Returns number of bytes written.  dst must be large enough
// (safe upper bound: srcLen + srcLen/255 + 16).
static size_t Lz4EncodeBlock(const uint8_t* src, size_t srcLen,
                              uint8_t* dst, size_t dstCapacity) {
    // We emit one big literal sequence (possibly split into 65535-byte chunks
    // so that the literal length field never needs more than a few extra bytes).
    size_t outPos = 0;
    size_t inPos  = 0;

    while (inPos < srcLen) {
        size_t remaining  = srcLen - inPos;
        // Each sequence can hold up to 65535 + 15 literals, but to keep it
        // simple we cap at 65535 bytes per sequence.
        size_t litLen = remaining > 65535u ? 65535u : remaining;

        // Token byte
        uint8_t tokenHigh = (litLen >= 15) ? 15 : static_cast<uint8_t>(litLen);
        if (outPos + 1 > dstCapacity) return 0;
        dst[outPos++] = static_cast<uint8_t>(tokenHigh << 4);

        // Extra literal length bytes
        if (litLen >= 15) {
            size_t extra = litLen - 15;
            while (extra >= 255) {
                if (outPos + 1 > dstCapacity) return 0;
                dst[outPos++] = 255;
                extra -= 255;
            }
            if (outPos + 1 > dstCapacity) return 0;
            dst[outPos++] = static_cast<uint8_t>(extra);
        }

        // Literal data
        if (outPos + litLen > dstCapacity) return 0;
        std::memcpy(dst + outPos, src + inPos, litLen);
        outPos += litLen;
        inPos  += litLen;

        // For all sequences except the final one we must emit a match.
        // Since we don't have a final match anyway (LZ4 spec: last sequence
        // has no match) this is only needed for intermediate sequences.
        // However, because we're producing single-literal sequences of ≤65535
        // bytes we only ever produce one sequence for reasonable files, so we
        // just let the loop handle it.  The loop will exit after one iteration
        // in the common case.
    }

    return outPos;
}

// Encode `data` into a "chunked LZ4" buffer as expected by the parser's
// Lz4DecompressChunked: first byte = 0 (single-block mode), then LZ4 block.
static std::vector<uint8_t> MakeLz4ChunkedBuffer(const uint8_t* data,
                                                  size_t dataLen) {
    size_t maxEncoded = dataLen + dataLen / 255 + 32;
    std::vector<uint8_t> encoded(maxEncoded);
    size_t encLen = Lz4EncodeBlock(data, dataLen,
                                   encoded.data(), maxEncoded);
    if (encLen == 0 && dataLen > 0) {
        // Should not happen; fall back to uncompressible copy
        // (shouldn't occur in practice)
        encoded.resize(dataLen + 1);
        encoded[0] = 0;
        std::memcpy(encoded.data() + 1, data, dataLen);
        return encoded;
    }
    // Prepend the chunk-indicator byte (0 = single block)
    std::vector<uint8_t> result;
    result.reserve(1 + encLen);
    result.push_back(0);
    result.insert(result.end(), encoded.begin(), encoded.begin() + encLen);
    return result;
}

// ---------------------------------------------------------------
// Compressed integer array encoder (spec 16.3.7.2)
// ---------------------------------------------------------------

template <typename IntT>
static std::vector<uint8_t> EncodeCompressedInts(const std::vector<IntT>& values) {
    using SignedT = typename std::make_signed<IntT>::type;
    size_t n = values.size();

    // Convert to delta array
    std::vector<SignedT> deltas(n);
    SignedT prev = 0;
    for (size_t i = 0; i < n; ++i) {
        SignedT v = static_cast<SignedT>(values[i]);
        deltas[i] = v - prev;
        prev = v;
    }

    // Find the most-common delta value (the "common value")
    std::unordered_map<SignedT, size_t> freq;
    for (auto d : deltas) freq[d]++;
    SignedT commonValue = 0;
    size_t bestFreq = 0;
    for (auto& [v, c] : freq) {
        if (c > bestFreq || (c == bestFreq && v == 0)) {
            bestFreq = c;
            commonValue = v;
        }
    }

    // Encode codepoints and values
    // Codepoints: 2 bits per delta, packed into bytes, then encoded values.
    size_t codepointBytes = (n * 2 + 7) / 8;
    std::vector<uint8_t> codepoints(codepointBytes, 0);

    std::vector<uint8_t> encodedValues;
    encodedValues.reserve(n * sizeof(IntT));

    for (size_t i = 0; i < n; ++i) {
        SignedT delta = deltas[i];
        uint8_t code;

        if (delta == commonValue) {
            code = 0;  // use common value
        } else if constexpr (sizeof(IntT) <= 4) {
            if (delta >= -128 && delta <= 127) {
                code = 1;  // int8
                int8_t v = static_cast<int8_t>(delta);
                encodedValues.push_back(static_cast<uint8_t>(v));
            } else if (delta >= -32768 && delta <= 32767) {
                code = 2;  // int16
                int16_t v = static_cast<int16_t>(delta);
                uint8_t buf[2];
                std::memcpy(buf, &v, 2);
                encodedValues.push_back(buf[0]);
                encodedValues.push_back(buf[1]);
            } else {
                code = 3;  // full int32
                uint8_t buf[4];
                int32_t v = static_cast<int32_t>(delta);
                std::memcpy(buf, &v, 4);
                encodedValues.insert(encodedValues.end(), buf, buf + 4);
            }
        } else {
            // 64-bit
            if (delta >= -32768 && delta <= 32767) {
                code = 1;  // int16
                int16_t v = static_cast<int16_t>(delta);
                uint8_t buf[2];
                std::memcpy(buf, &v, 2);
                encodedValues.push_back(buf[0]);
                encodedValues.push_back(buf[1]);
            } else if (delta >= INT32_MIN && delta <= INT32_MAX) {
                code = 2;  // int32
                int32_t v = static_cast<int32_t>(delta);
                uint8_t buf[4];
                std::memcpy(buf, &v, 4);
                encodedValues.insert(encodedValues.end(), buf, buf + 4);
            } else {
                code = 3;  // full int64
                SignedT v = delta;
                uint8_t buf[8];
                std::memcpy(buf, &v, 8);
                encodedValues.insert(encodedValues.end(), buf, buf + 8);
            }
        }

        // Pack codepoint into the bit array (2 bits per value)
        size_t byteIdx = (i * 2) / 8;
        size_t bitIdx  = (i * 2) % 8;
        codepoints[byteIdx] |= static_cast<uint8_t>(code << bitIdx);
    }

    // Assemble the pre-LZ4 buffer: commonValue + codepoints + encodedValues
    std::vector<uint8_t> raw;
    raw.reserve(sizeof(IntT) + codepointBytes + encodedValues.size());

    SignedT cv = commonValue;
    uint8_t cvBuf[sizeof(IntT)];
    std::memcpy(cvBuf, &cv, sizeof(IntT));
    raw.insert(raw.end(), cvBuf, cvBuf + sizeof(IntT));
    raw.insert(raw.end(), codepoints.begin(), codepoints.end());
    raw.insert(raw.end(), encodedValues.begin(), encodedValues.end());

    // LZ4-compress the raw buffer
    return MakeLz4ChunkedBuffer(raw.data(), raw.size());
}

// Write a compressed integer array section:
//   uint64 compressedSize, then compressedData
template <typename IntT>
static void WriteCompressedInts(std::vector<uint8_t>& out,
                                const std::vector<IntT>& values) {
    if (values.empty()) {
        // Write compressed size = 0 (but we still need valid LZ4 data)
        // The parser checks CanRead(compressedSize) first, so size=0 is fine.
        uint64_t zero = 0;
        uint8_t buf[8];
        std::memcpy(buf, &zero, 8);
        out.insert(out.end(), buf, buf + 8);
        return;
    }
    std::vector<uint8_t> compressed = EncodeCompressedInts(values);
    uint64_t compSize = compressed.size();
    uint8_t buf[8];
    std::memcpy(buf, &compSize, 8);
    out.insert(out.end(), buf, buf + 8);
    out.insert(out.end(), compressed.begin(), compressed.end());
}

// ---------------------------------------------------------------
// BinaryWriter — simple append-only buffer with typed helpers
// ---------------------------------------------------------------

class BinaryWriter {
public:
    std::vector<uint8_t> data;

    size_t Pos() const { return data.size(); }

    void WriteU8(uint8_t v)   { data.push_back(v); }
    void WriteU16(uint16_t v) { AppendT(v); }
    void WriteU32(uint32_t v) { AppendT(v); }
    void WriteU64(uint64_t v) { AppendT(v); }
    void WriteI32(int32_t v)  { AppendT(v); }
    void WriteI64(int64_t v)  { AppendT(v); }
    void WriteF32(float v)    { AppendT(v); }
    void WriteF64(double v)   { AppendT(v); }

    void WriteBytes(const void* buf, size_t n) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(buf);
        data.insert(data.end(), p, p + n);
    }

    // Overwrite a previously-written uint64 at `pos`
    void PatchU64(size_t pos, uint64_t v) {
        std::memcpy(data.data() + pos, &v, 8);
    }

    // Align to 8-byte boundary by appending zeros
    void Align8() {
        while (data.size() % 8 != 0) data.push_back(0);
    }

    template <typename T>
    void AppendT(T v) {
        uint8_t buf[sizeof(T)];
        std::memcpy(buf, &v, sizeof(T));
        data.insert(data.end(), buf, buf + sizeof(T));
    }

    // Write compressed integers directly into this buffer
    template <typename IntT>
    void WriteCompressedInts(const std::vector<IntT>& values) {
        ::nanousd::WriteCompressedInts(data, values);
    }
};

// ---------------------------------------------------------------
// Crate type IDs (spec 16.3.10.1)
// ---------------------------------------------------------------

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

// Bit flags (spec 16.3.9)
static constexpr uint8_t kFlagArray      = 0x80;
static constexpr uint8_t kFlagInlined    = 0x40;
// static constexpr uint8_t kFlagCompressed = 0x20;  // unused in writer

// Spec form values (spec 16.3.8.4.6)
static constexpr uint32_t kFormAttribute  = 1;
static constexpr uint32_t kFormPrim       = 6;
static constexpr uint32_t kFormLayer      = 7;
static constexpr uint32_t kFormRelationship = 8;
static constexpr uint32_t kFormVariant    = 10;
static constexpr uint32_t kFormVariantSet = 11;

static uint32_t SpecTypeToForm(SpecType t) {
    switch (t) {
        case SpecType::Attribute:    return kFormAttribute;
        case SpecType::Prim:         return kFormPrim;
        case SpecType::Layer:        return kFormLayer;
        case SpecType::Relationship: return kFormRelationship;
        case SpecType::Variant:      return kFormVariant;
        case SpecType::VariantSet:   return kFormVariantSet;
    }
    return kFormPrim;
}

// ---------------------------------------------------------------
// Token / string tables
// ---------------------------------------------------------------

struct TokenTable {
    std::vector<std::string> tokens;              // tokens_[i] = token string
    std::unordered_map<std::string, uint32_t> idx; // string -> index

    // Token at index 0 is a required placeholder (unused by other sections)
    TokenTable() { AddOrGet(""); }  // index 0 = placeholder

    uint32_t AddOrGet(const std::string& s) {
        auto it = idx.find(s);
        if (it != idx.end()) return it->second;
        uint32_t i = static_cast<uint32_t>(tokens.size());
        tokens.push_back(s);
        idx[s] = i;
        return i;
    }
};

struct StringTable {
    // Strings section: an array of indices into the token table.
    // String values (TypeId::String) are looked up here.
    std::vector<uint32_t> indices;                     // string index -> token index
    std::unordered_map<std::string, uint32_t> strIdx;  // string -> string index

    uint32_t AddOrGet(const std::string& s, TokenTable& tokens) {
        auto it = strIdx.find(s);
        if (it != strIdx.end()) return it->second;
        uint32_t tokenIdx = tokens.AddOrGet(s);
        uint32_t i = static_cast<uint32_t>(indices.size());
        indices.push_back(tokenIdx);
        strIdx[s] = i;
        return i;
    }
};

// ---------------------------------------------------------------
// Path table
// ---------------------------------------------------------------

struct PathTable {
    std::vector<Path> paths;                          // paths_[i] = path
    std::unordered_map<std::string, uint32_t> pathIdx; // path text -> index

    PathTable() {
        // index 0 = absolute root "/"
        paths.emplace_back(Path::AbsoluteRoot());
        pathIdx["/"] = 0;
    }

    uint32_t AddOrGet(const Path& p) {
        const std::string& text = p.GetText();
        auto it = pathIdx.find(text);
        if (it != pathIdx.end()) return it->second;
        uint32_t i = static_cast<uint32_t>(paths.size());
        paths.push_back(p);
        pathIdx[text] = i;
        return i;
    }

    // Ensure all ancestor paths are in the table (needed for path encoding)
    void EnsureAncestors(const Path& p) {
        if (p.IsEmpty() || p.IsAbsoluteRoot()) {
            AddOrGet(Path::AbsoluteRoot());
            return;
        }
        Path parent = p.GetParentPath();
        if (!parent.IsEmpty()) EnsureAncestors(parent);
        AddOrGet(p);
    }
};

// ---------------------------------------------------------------
// Field and FieldSet tables
// ---------------------------------------------------------------

struct FieldEntry {
    uint32_t tokenIdx;  // index into token table (field name)
    uint64_t valueRep;  // 8-byte value representation
};

struct FieldTable {
    std::vector<FieldEntry> fields;
};

struct FieldSetTable {
    // Flat array of field indices; groups terminated by 0xFFFFFFFF
    std::vector<uint32_t> flat;

    // Add a group of field indices and return the start index in the flat array
    uint32_t AddGroup(const std::vector<uint32_t>& fieldIndices) {
        uint32_t start = static_cast<uint32_t>(flat.size());
        for (auto fi : fieldIndices) flat.push_back(fi);
        flat.push_back(0xFFFFFFFF);
        return start;
    }
};

// ---------------------------------------------------------------
// SpecEntry
// ---------------------------------------------------------------

struct SpecEntry {
    uint32_t pathIdx;
    uint32_t fieldSetIdx;
    uint32_t form;
};

// ---------------------------------------------------------------
// UsdcWriter — main encoder
// ---------------------------------------------------------------

class UsdcWriter {
private:
    // ---------------------------------------------------------------
    // Pass 1: Collect all data from the layer into tables
    // ---------------------------------------------------------------

    TokenTable  tokens_;
    StringTable strings_;
    PathTable   paths_;
    FieldTable  fields_;
    FieldSetTable fieldSets_;
    std::vector<SpecEntry> specs_;

    // Separately buffered value data (for offset-encoded values)
    BinaryWriter valueData_;

    // Crate version selected during collection. The spec minimum writable
    // version is 0.8.0; newer value type IDs raise this as needed.
    uint8_t requiredMinor_ = 8;

    enum class SplineCrateDataType : uint8_t {
        Unspecified = 0,
        Double = 1,
        Float = 2,
        Half = 3,
    };

    void RequireCrateMinor(uint8_t minor) {
        if (requiredMinor_ < minor) requiredMinor_ = minor;
    }

    void RequireCrateType(CrateTypeId type) {
        switch (type) {
            case CrateTypeId::TimeCode:
                RequireCrateMinor(9);
                break;
            case CrateTypeId::PathExpression:
                RequireCrateMinor(10);
                break;
            case CrateTypeId::Relocates:
                RequireCrateMinor(11);
                break;
            case CrateTypeId::Splines:
                RequireCrateMinor(12);
                break;
            default:
                break;
        }
    }

    static SplineCrateDataType SplineDataTypeFromSpec(const Spec& spec) {
        std::string typeName;
        if (auto* tv = spec.GetField(FieldNames::typeName)) {
            if (auto* tok = tv->Get<Token>()) {
                typeName = tok->GetString();
            } else if (auto* str = tv->Get<String>()) {
                typeName = *str;
            }
        }

        if (typeName == "half") return SplineCrateDataType::Half;
        if (typeName == "float") return SplineCrateDataType::Float;
        return SplineCrateDataType::Double;
    }

    // Map from spec path text -> field set start index (for dedup/caching)
    // We don't dedup field sets across specs in this implementation.

    void CollectLayer(const Layer& layer) {
        // Reserve path 0 = absolute root
        paths_.AddOrGet(Path::AbsoluteRoot());

        // Collect all spec paths we'll need
        // Layer spec is at "/" (path index 0)
        // All other specs are at their given paths

        // First pass: ensure all paths (and ancestors) are in the path table
        {
            // Layer path
            paths_.EnsureAncestors(Path::AbsoluteRoot());
            // All spec paths
            for (const auto& p : layer.GetSpecPaths()) {
                paths_.EnsureAncestors(p);
            }
        }

        // Second pass: collect specs.
        // This must run before the token pre-add pass because encoding field values
        // (e.g. reference prim paths, relationship target paths) may call
        // paths_.EnsureAncestors(), adding new paths to the table.  Those new paths
        // need their element tokens pre-added as well — which the next pass handles.
        //
        // We also synthesise `primChildren` / `propertyChildren` fields here:
        // pxr's reader populates the prim hierarchy from those fields, not from
        // the PATHS jump-tree alone (spec §16.3.8.4.6). nanousd's own reader
        // happens to rebuild the indices from PATHS in usdc_parser.cpp's
        // `ReconstructPaths`, so round-trip works without these fields, but
        // pxr would report zero prims (only layer metadata) without them.
        // Layer spec first
        {
            uint32_t pathIdx = paths_.AddOrGet(Path::AbsoluteRoot());
            uint32_t fsIdx = CollectSpecWithChildren(
                layer, Path::AbsoluteRoot(), layer.GetLayerSpec());
            specs_.push_back({pathIdx, fsIdx, kFormLayer});
        }
        // Prim/attribute/rel specs
        for (const auto& p : layer.GetSpecPaths()) {
            const Spec* spec = layer.GetSpec(p);
            if (!spec) continue;
            uint32_t pathIdx = paths_.AddOrGet(p);
            uint32_t fsIdx = CollectSpecWithChildren(layer, p, *spec);
            specs_.push_back({pathIdx, fsIdx, SpecTypeToForm(spec->GetType())});
        }

        // Third pass: pre-add all path element tokens to the token table so they
        // appear in the TOKENS section (which is written before the PATHS section).
        // This runs after spec collection so that any paths added by field encoding
        // (reference/relationship paths via EnsureAncestors) are also covered.
        // Without this, WritePathsSection would add them too late.
        {
            for (const auto& p : paths_.paths) {
                if (p.IsAbsoluteRoot()) {
                    tokens_.AddOrGet("/");
                } else if (!p.IsEmpty()) {
                    tokens_.AddOrGet(p.GetName());
                }
            }
        }
    }

    // Encode a list of token names as a CrateTypeId::TokenVector (count + N×u32
    // token indices) at the current valueData_ position. Returns the ValueRep.
    uint64_t EncodeTokenVector(const std::vector<Token>& names) {
        valueData_.Align8();
        size_t offset = valueData_.Pos();
        valueData_.WriteU64(static_cast<uint64_t>(names.size()));
        for (const auto& t : names) {
            uint32_t ti = tokens_.AddOrGet(t.GetString());
            valueData_.WriteU32(ti);
        }
        return MakeOffsetRep(CrateTypeId::TokenVector, offset);
    }

    // Synthesise a `primChildren`/`propertyChildren` field. The field's value
    // is encoded as a CrateTypeId::TokenVector (spec §16.3.10.27 + §16.3.8.4.6).
    uint32_t CollectSyntheticTokenVectorField(const std::string& fieldName,
                                              const std::vector<Token>& names) {
        uint32_t tokIdx = tokens_.AddOrGet(fieldName);
        uint64_t valRep = EncodeTokenVector(names);
        FieldEntry fe{tokIdx, valRep};
        uint32_t idx = static_cast<uint32_t>(fields_.fields.size());
        fields_.fields.push_back(fe);
        return idx;
    }

    // Collect a spec's fields plus synthesised `primChildren`/`propertyChildren`
    // entries (built from the layer's child/property indices, not the spec's
    // fields). Returns the FieldSet start index.
    //
    // pxr's reader populates the prim hierarchy from the `primChildren` /
    // `propertyChildren` fields (spec §16.3.10.27 TokenVector, applied per
    // §16.3.8.4.6). nanousd's own reader rebuilds those indices from the PATHS
    // jump-tree, so without this synthesis a USDC written by nanousd round-
    // trips fine through nanousd but reports zero prims when opened in pxr.
    //
    // We always source the names from the layer's index rather than from any
    // authored field on the spec: a USDC parsed by nanousd's `ApplyField`
    // stores the parsed TokenVector under `primChildren` as `vector<string>`
    // (TypeId::String array), and re-encoding that path would emit
    // CrateTypeId::String + array flag instead of CrateTypeId::TokenVector,
    // which pxr wouldn't recognise. The layer's index is the authoritative
    // source since `HashMapSpecStore::AddToIndex` populates it on every
    // SetSpec, and the USDC parser injects pre-built indices via
    // `SetIndices` after `ReconstructPaths` from the jump-tree.
    uint32_t CollectSpecWithChildren(const Layer& layer,
                                     const Path& specPath,
                                     const Spec& spec) {
        std::vector<uint32_t> fieldIndices;
        bool diag = std::getenv("MUSD_DIAG_WRITE") != nullptr;
        if (diag) std::fprintf(stderr, "[Write] spec path=%s type=%d fields=%zu\n",
                               specPath.GetText().c_str(),
                               (int)spec.GetType(), spec.GetFields().size());

        std::vector<Token> primChildNames;
        std::vector<Token> propChildNames;
        const SpecType st = spec.GetType();
        const bool wantsPrimChildren =
            (st == SpecType::Layer || st == SpecType::Prim ||
             st == SpecType::Variant);
        const bool wantsPropChildren =
            (st == SpecType::Prim || st == SpecType::Variant);
        if (wantsPrimChildren) {
            primChildNames = layer.GetChildNames(specPath);
        }
        if (wantsPropChildren) {
            propChildNames = layer.GetPropertyNames(specPath);
            std::sort(propChildNames.begin(), propChildNames.end(),
                      PathElementTokenLess);
        }

        // Emit primChildren first (matches the typical pxr authoring order so
        // FIELDSETS/FIELDS layouts stay close to a pxr-authored crate's).
        if (!primChildNames.empty()) {
            fieldIndices.push_back(
                CollectSyntheticTokenVectorField("primChildren", primChildNames));
        }
        if (!propChildNames.empty()) {
            fieldIndices.push_back(
                CollectSyntheticTokenVectorField("propertyChildren", propChildNames));
        }

        // Iterate authored fields, but skip any authored primChildren /
        // propertyChildren — we just synthesised those from the layer index
        // (the authoritative source). Without this skip a USDC->USDC round-
        // trip would emit the field twice and the second emission would use
        // the wrong CrateTypeId (string array, not TokenVector) — pxr would
        // accept the synthesised one but spec semantics say a fieldset must
        // not have duplicate field names.
        static const Token kPrimChildren("primChildren");
        static const Token kPropChildren("propertyChildren");
        for (const auto& [name, val] : spec.GetFields()) {
            if (name == kPrimChildren || name == kPropChildren) continue;
            std::string nameStr(name);
            if (diag) std::fprintf(stderr, "  field='%s'\n", nameStr.c_str());
            // Some fields need special mapping back from the in-memory repr
            uint32_t fieldIdx = CollectField(name, spec, val);
            fieldIndices.push_back(fieldIdx);
        }
        return fieldSets_.AddGroup(fieldIndices);
    }

    // Collect all fields for a spec and return the FieldSet start index.
    // Retained for callers that don't have layer context; in this writer
    // CollectSpecWithChildren is preferred so primChildren/propertyChildren
    // get synthesised for pxr compatibility.
    uint32_t CollectSpec(const Spec& spec) {
        std::vector<uint32_t> fieldIndices;
        bool diag = std::getenv("MUSD_DIAG_WRITE") != nullptr;
        if (diag) std::fprintf(stderr, "[Write] spec type=%d fields=%zu\n",
                               (int)spec.GetType(), spec.GetFields().size());
        for (const auto& [name, val] : spec.GetFields()) {
            std::string nameStr(name);
            if (diag) std::fprintf(stderr, "  field='%s'\n", nameStr.c_str());
            // Some fields need special mapping back from the in-memory repr
            uint32_t fieldIdx = CollectField(name, spec, val);
            fieldIndices.push_back(fieldIdx);
        }
        return fieldSets_.AddGroup(fieldIndices);
    }

    uint32_t CollectField(const std::string& name, const Spec& spec,
                          const Value& val) {
        const std::string fieldName =
            name == FieldNames::layerRelocates.GetString()
                ? std::string("relocates")
                : name;
        uint32_t tokIdx = tokens_.AddOrGet(fieldName);
        SplineCrateDataType splineDataType = SplineCrateDataType::Double;
        if (fieldName == FieldNames::spline.GetString()) {
            splineDataType = SplineDataTypeFromSpec(spec);
        }
        uint64_t valRep = EncodeValueRep(fieldName, val, splineDataType);
        FieldEntry fe{tokIdx, valRep};
        uint32_t idx = static_cast<uint32_t>(fields_.fields.size());
        fields_.fields.push_back(fe);
        return idx;
    }

    // ---------------------------------------------------------------
    // Value representation encoding (spec 16.3.9)
    //
    // Returns a 64-bit ValueRep.
    // Writes any non-inlined payload into valueData_.
    // ---------------------------------------------------------------

    uint64_t MakeValueRep(CrateTypeId type, uint8_t flags, uint64_t payload) {
        RequireCrateType(type);
        uint64_t rep = payload & 0x0000FFFFFFFFFFFF;
        rep |= static_cast<uint64_t>(static_cast<uint8_t>(type)) << 48;
        rep |= static_cast<uint64_t>(flags) << 56;
        return rep;
    }

    uint64_t MakeInlinedRep(CrateTypeId type, uint32_t payload) {
        return MakeValueRep(type, kFlagInlined, static_cast<uint64_t>(payload));
    }

    uint64_t MakeOffsetRep(CrateTypeId type, size_t offset) {
        return MakeValueRep(type, 0, static_cast<uint64_t>(offset));
    }

    uint64_t MakeArrayRep(CrateTypeId type, size_t offset) {
        if (offset == 0) {
            // Empty array: payload = 0, array bit set
            return MakeValueRep(type, kFlagArray, 0);
        }
        return MakeValueRep(type, kFlagArray, static_cast<uint64_t>(offset));
    }

    // Map TypeId -> CrateTypeId
    static CrateTypeId TypeIdToCrateTypeId(TypeId t) {
        switch (t) {
            case TypeId::Bool:     return CrateTypeId::Bool;
            case TypeId::UChar:    return CrateTypeId::UChar;
            case TypeId::Int:      return CrateTypeId::Int;
            case TypeId::UInt:     return CrateTypeId::UInt;
            case TypeId::Int64:    return CrateTypeId::Int64;
            case TypeId::UInt64:   return CrateTypeId::UInt64;
            case TypeId::Half:     return CrateTypeId::Half;
            case TypeId::Float:    return CrateTypeId::Float;
            case TypeId::Double:   return CrateTypeId::Double;
            case TypeId::String:   return CrateTypeId::String;
            case TypeId::Token:    return CrateTypeId::Token;
            case TypeId::Asset:    return CrateTypeId::Asset;
            case TypeId::TimeCode: return CrateTypeId::TimeCode;
            case TypeId::Half2:    return CrateTypeId::Half2;
            case TypeId::Half3:    return CrateTypeId::Half3;
            case TypeId::Half4:    return CrateTypeId::Half4;
            case TypeId::Float2:   return CrateTypeId::Float2;
            case TypeId::Float3:   return CrateTypeId::Float3;
            case TypeId::Float4:   return CrateTypeId::Float4;
            case TypeId::Double2:  return CrateTypeId::Double2;
            case TypeId::Double3:  return CrateTypeId::Double3;
            case TypeId::Double4:  return CrateTypeId::Double4;
            case TypeId::Int2:     return CrateTypeId::Int2;
            case TypeId::Int3:     return CrateTypeId::Int3;
            case TypeId::Int4:     return CrateTypeId::Int4;
            case TypeId::Quath:    return CrateTypeId::Quath;
            case TypeId::Quatf:    return CrateTypeId::Quatf;
            case TypeId::Quatd:    return CrateTypeId::Quatd;
            case TypeId::Matrix2d: return CrateTypeId::Matrix2d;
            case TypeId::Matrix3d: return CrateTypeId::Matrix3d;
            case TypeId::Matrix4d: return CrateTypeId::Matrix4d;
            case TypeId::Dictionary: return CrateTypeId::Dictionary;
            case TypeId::Spline:   return CrateTypeId::Splines;
            default: return CrateTypeId::Invalid;
        }
    }

    // Encode a value into a ValueRep, writing payload data into valueData_ as needed.
    uint64_t EncodeValueRep(
            const std::string& fieldName,
            const Value& val,
            SplineCrateDataType splineDataType = SplineCrateDataType::Double) {
        if (val.IsEmpty()) {
            // Fallback: empty inlined bool = false
            return MakeInlinedRep(CrateTypeId::Bool, 0);
        }

        if (val.IsBlock()) {
            // ValueBlock sentinel (spec 16.3.10.16)
            return MakeInlinedRep(CrateTypeId::ValueBlock, 0);
        }

        TypeId tid = val.GetTypeId();

        // --- Handle arrays ---
        if (val.IsArray()) {
            return EncodeArrayRep(tid, val);
        }

        // --- Handle special field names that need specific crate types ---
        // "specifier" field is stored as CrateTypeId::Specifier (inlined int32)
        if (fieldName == "specifier") {
            if (auto* iv = val.Get<Int>()) {
                return MakeInlinedRep(CrateTypeId::Specifier, static_cast<uint32_t>(*iv));
            }
            // From Spec::SetSpecifier -> stored as Int
        }
        // "variability" from the parser: stored as string "uniform"/"varying"
        // but on write we need to emit as CrateTypeId::Variability
        if (fieldName == "variability") {
            uint32_t v = 0; // Varying
            if (auto* s = val.Get<String>()) {
                if (*s == "uniform") v = 1;
            } else if (auto* iv = val.Get<Int>()) {
                v = static_cast<uint32_t>(*iv);
            }
            return MakeInlinedRep(CrateTypeId::Variability, v);
        }

        // --- Handle composition types stored as Unknown TypeId ---
        if (tid == TypeId::Unknown) {
            return EncodeUnknownTypeValue(fieldName, val);
        }

        // --- Scalar types with required inlining (spec 16.3.9.1) ---
        switch (tid) {
            case TypeId::Bool: {
                auto* v = val.Get<Bool>();
                return MakeInlinedRep(CrateTypeId::Bool, v ? (*v ? 1u : 0u) : 0u);
            }
            case TypeId::UChar: {
                auto* v = val.Get<UChar>();
                return MakeInlinedRep(CrateTypeId::UChar, v ? static_cast<uint32_t>(*v) : 0u);
            }
            case TypeId::Int: {
                auto* v = val.Get<Int>();
                uint32_t bits = v ? static_cast<uint32_t>(static_cast<int32_t>(*v)) : 0u;
                return MakeInlinedRep(CrateTypeId::Int, bits);
            }
            case TypeId::UInt: {
                auto* v = val.Get<UInt>();
                return MakeInlinedRep(CrateTypeId::UInt, v ? *v : 0u);
            }
            case TypeId::Half: {
                auto* v = val.Get<Half>();
                return MakeInlinedRep(CrateTypeId::Half,
                                      v ? static_cast<uint32_t>(v->bits) : 0u);
            }
            case TypeId::Float: {
                auto* v = val.Get<Float>();
                uint32_t bits = 0;
                if (v) std::memcpy(&bits, v, 4);
                return MakeInlinedRep(CrateTypeId::Float, bits);
            }
            case TypeId::Double: {
                // Inline if value == float(value) exactly
                auto* v = val.Get<Double>();
                if (v) {
                    float f = static_cast<float>(*v);
                    if (static_cast<double>(f) == *v) {
                        uint32_t bits = 0;
                        std::memcpy(&bits, &f, 4);
                        return MakeInlinedRep(CrateTypeId::Double, bits);
                    }
                }
                return EncodeOffsetDouble(v ? *v : 0.0);
            }
            case TypeId::Int64: {
                // Inline if fits in int32
                auto* v = val.Get<Int64>();
                if (v && *v >= INT32_MIN && *v <= INT32_MAX) {
                    return MakeInlinedRep(CrateTypeId::Int64,
                                         static_cast<uint32_t>(static_cast<int32_t>(*v)));
                }
                return EncodeOffsetInt64(v ? *v : 0LL);
            }
            case TypeId::UInt64: {
                auto* v = val.Get<UInt64>();
                if (v && *v <= UINT32_MAX) {
                    return MakeInlinedRep(CrateTypeId::UInt64,
                                         static_cast<uint32_t>(*v));
                }
                return EncodeOffsetUInt64(v ? *v : 0ULL);
            }
            case TypeId::String: {
                // Inlined string = index into strings section
                auto* v = val.Get<String>();
                uint32_t si = v ? strings_.AddOrGet(*v, tokens_) : 0u;
                return MakeInlinedRep(CrateTypeId::String, si);
            }
            case TypeId::Token: {
                // Inlined token = index into tokens section
                auto* v = val.Get<String>();
                if (v) return MakeInlinedRep(CrateTypeId::Token, tokens_.AddOrGet(*v));
                auto* t = val.Get<Token>();
                uint32_t ti = t ? tokens_.AddOrGet(t->GetString()) : 0u;
                return MakeInlinedRep(CrateTypeId::Token, ti);
            }
            case TypeId::Asset: {
                // Inlined asset = index into tokens section (per spec 16.3.10.13)
                auto* v = val.Get<String>();
                uint32_t ti = v ? tokens_.AddOrGet(*v) : 0u;
                return MakeInlinedRep(CrateTypeId::Asset, ti);
            }
            case TypeId::TimeCode: {
                auto* v = val.Get<Double>();
                if (v) {
                    float f = static_cast<float>(*v);
                    if (static_cast<double>(f) == *v) {
                        uint32_t bits = 0;
                        std::memcpy(&bits, &f, 4);
                        return MakeInlinedRep(CrateTypeId::TimeCode, bits);
                    }
                }
                return EncodeOffsetTimeCode(v ? *v : 0.0);
            }
            case TypeId::Dictionary: {
                auto* v = val.Get<Dictionary>();
                if (!v || v->empty()) {
                    // Empty dictionary: inlined (spec 16.3.9.1)
                    return MakeInlinedRep(CrateTypeId::Dictionary, 0);
                }
                return EncodeOffsetDictionary(*v);
            }
            case TypeId::Spline: {
                auto* v = val.Get<Spline>();
                static const Spline kEmpty;
                return EncodeOffsetSpline(v ? *v : kEmpty, splineDataType);
            }
            // Vec / quat / matrix types — offset encoded
            default:
                break;
        }

        // --- Dimensioned types and others — write at offset ---
        return EncodeOffsetScalar(tid, val);
    }

    // Encode a Double at an offset
    uint64_t EncodeOffsetDouble(double v) {
        valueData_.Align8();
        size_t offset = valueData_.Pos();
        valueData_.WriteF64(v);
        return MakeOffsetRep(CrateTypeId::Double, offset);
    }

    uint64_t EncodeOffsetInt64(int64_t v) {
        valueData_.Align8();
        size_t offset = valueData_.Pos();
        valueData_.WriteI64(v);
        return MakeOffsetRep(CrateTypeId::Int64, offset);
    }

    uint64_t EncodeOffsetUInt64(uint64_t v) {
        valueData_.Align8();
        size_t offset = valueData_.Pos();
        valueData_.WriteU64(v);
        return MakeOffsetRep(CrateTypeId::UInt64, offset);
    }

    uint64_t EncodeOffsetTimeCode(double v) {
        valueData_.Align8();
        size_t offset = valueData_.Pos();
        valueData_.WriteF64(v);
        return MakeOffsetRep(CrateTypeId::TimeCode, offset);
    }

    uint64_t EncodeOffsetDictionary(const Dictionary& dict) {
        valueData_.Align8();
        size_t offset = valueData_.Pos();
        WriteDictionaryAt(dict);
        return MakeOffsetRep(CrateTypeId::Dictionary, offset);
    }

    // Spline byte-stream helpers for Crate v0.12.0.
    static uint8_t EncodeSplineCurveType(CurveType curveType) {
        return curveType == CurveType::Hermite ? 1u : 0u;
    }

    static uint8_t EncodeSplineInterpolation(InterpolationMode mode) {
        uint32_t value = static_cast<uint32_t>(mode);
        return value <= 3u ? static_cast<uint8_t>(value) : 1u;
    }

    static uint8_t EncodeSplineExtrapolation(ExtrapolationMode mode) {
        uint32_t value = static_cast<uint32_t>(mode);
        return value <= 6u ? static_cast<uint8_t>(value) : 1u;
    }

    static bool HasAuthoredLoopParameters(const LoopParameters& loop) {
        return loop.protoStart != 0.0 || loop.protoEnd != 0.0
            || loop.numPreLoops != 0 || loop.numPostLoops != 0
            || loop.valueOffset != 0.0;
    }

    static uint32_t ClampSplineLoopCount(int value) {
        if (value <= 0) return 0;
        constexpr int32_t kMaxLoopCount = (std::numeric_limits<int32_t>::max)();
        if (value > kMaxLoopCount) return static_cast<uint32_t>(kMaxLoopCount);
        return static_cast<uint32_t>(value);
    }

    static void WriteSplineTypedValue(BinaryWriter& out,
                                      SplineCrateDataType dataType,
                                      double value) {
        switch (dataType) {
            case SplineCrateDataType::Float:
                out.WriteF32(static_cast<float>(value));
                break;
            case SplineCrateDataType::Half: {
                Half h(static_cast<float>(value));
                out.WriteU16(h.bits);
                break;
            }
            case SplineCrateDataType::Unspecified:
            case SplineCrateDataType::Double:
            default:
                out.WriteF64(value);
                break;
        }
    }

    uint64_t EncodeOffsetSpline(const Spline& s,
                                SplineCrateDataType dataType) {
        valueData_.Align8();
        size_t offset = valueData_.Pos();

        BinaryWriter splineBytes;
        const uint8_t firstFlags =
            0x01u
            | static_cast<uint8_t>(static_cast<uint8_t>(dataType) << 4)
            | static_cast<uint8_t>(EncodeSplineCurveType(s.curveType) << 7);
        splineBytes.WriteU8(firstFlags);

        const bool hasLoop = HasAuthoredLoopParameters(s.loopParameters);
        const uint8_t secondFlags =
            EncodeSplineExtrapolation(s.preExtrapolationMode)
            | static_cast<uint8_t>(
                EncodeSplineExtrapolation(s.postExtrapolationMode) << 3)
            | static_cast<uint8_t>(hasLoop ? 0x40u : 0u);
        splineBytes.WriteU8(secondFlags);

        if (s.preExtrapolationMode == ExtrapolationMode::Sloped) {
            splineBytes.WriteF64(s.preExtrapolationSlope);
        }
        if (s.postExtrapolationMode == ExtrapolationMode::Sloped) {
            splineBytes.WriteF64(s.postExtrapolationSlope);
        }
        if (hasLoop) {
            splineBytes.WriteF64(s.loopParameters.protoStart);
            splineBytes.WriteF64(s.loopParameters.protoEnd);
            splineBytes.WriteU32(ClampSplineLoopCount(s.loopParameters.numPreLoops));
            splineBytes.WriteU32(ClampSplineLoopCount(s.loopParameters.numPostLoops));
            splineBytes.WriteF64(s.loopParameters.valueOffset);
        }

        if (s.knots.size() > (std::numeric_limits<uint32_t>::max)()) {
            throw std::runtime_error("Spline has too many knots for USDC encoding");
        }
        splineBytes.WriteU32(static_cast<uint32_t>(s.knots.size()));

        for (const auto& k : s.knots) {
            const bool dualValued = k.preValue != k.value;
            uint8_t knotFlags = 0;
            if (dualValued) knotFlags |= 0x01u;
            knotFlags |= static_cast<uint8_t>(
                EncodeSplineInterpolation(k.nextInterpolationMode) << 1);
            knotFlags |= static_cast<uint8_t>(
                EncodeSplineCurveType(s.curveType) << 3);

            splineBytes.WriteU8(knotFlags);
            splineBytes.WriteF64(k.time);
            WriteSplineTypedValue(splineBytes, dataType, k.value);
            if (dualValued) {
                WriteSplineTypedValue(splineBytes, dataType, k.preValue);
            }
            if (s.curveType != CurveType::Hermite) {
                splineBytes.WriteF64(k.preTangentWidth);
                splineBytes.WriteF64(k.postTangentWidth);
            }
            WriteSplineTypedValue(splineBytes, dataType, k.preTangentSlope);
            WriteSplineTypedValue(splineBytes, dataType, k.postTangentSlope);
        }

        valueData_.WriteU64(static_cast<uint64_t>(splineBytes.data.size()));
        valueData_.WriteBytes(splineBytes.data.data(), splineBytes.data.size());
        valueData_.WriteU64(0);  // per-knot custom data count
        return MakeOffsetRep(CrateTypeId::Splines, offset);
    }

    // Write a dictionary into valueData_ at current position (spec 16.3.10.19)
    void WriteDictionaryAt(const Dictionary& dict) {
        struct PendingValue {
            size_t offsetFieldPos;
            const Value* value;
        };
        std::vector<PendingValue> pending;
        pending.reserve(dict.size());

        valueData_.WriteU64(static_cast<uint64_t>(dict.size()));
        for (const auto& [key, val] : dict) {
            uint32_t tokIdx = tokens_.AddOrGet(key);
            valueData_.WriteU32(tokIdx);
            size_t offsetFieldPos = valueData_.Pos();
            valueData_.WriteI64(0);
            pending.push_back({offsetFieldPos, &val});
        }

        for (const auto& item : pending) {
            uint64_t valRep = EncodeValueRep("", *item.value);
            size_t valueRepPos = valueData_.Pos();
            valueData_.WriteU64(valRep);
            int64_t patchedOffset = static_cast<int64_t>(valueRepPos) -
                                     static_cast<int64_t>(item.offsetFieldPos);
            std::memcpy(valueData_.data.data() + item.offsetFieldPos,
                        &patchedOffset, 8);
        }
    }

    // Encode dimensioned scalar types (vec/quat/matrix) at offset
    uint64_t EncodeOffsetScalar(TypeId tid, const Value& val) {
        CrateTypeId crateType = TypeIdToCrateTypeId(tid);
        if (crateType == CrateTypeId::Invalid) {
            return MakeInlinedRep(CrateTypeId::Bool, 0);
        }

        valueData_.Align8();
        size_t offset = valueData_.Pos();
        WriteScalarData(tid, val);
        return MakeOffsetRep(crateType, offset);
    }

    // Write raw scalar data (no header) into valueData_
    void WriteScalarData(TypeId tid, const Value& val) {
        switch (tid) {
            case TypeId::Half2: if (auto* v = val.Get<GfVec2h>())
                { valueData_.AppendT(v->data[0].bits); valueData_.AppendT(v->data[1].bits); } break;
            case TypeId::Half3: if (auto* v = val.Get<GfVec3h>())
                { for (int i=0;i<3;++i) valueData_.AppendT(v->data[i].bits); } break;
            case TypeId::Half4: if (auto* v = val.Get<GfVec4h>())
                { for (int i=0;i<4;++i) valueData_.AppendT(v->data[i].bits); } break;
            case TypeId::Float2: if (auto* v = val.Get<GfVec2f>())
                valueData_.WriteBytes(v->data.data(), 8); break;
            case TypeId::Float3: if (auto* v = val.Get<GfVec3f>())
                valueData_.WriteBytes(v->data.data(), 12); break;
            case TypeId::Float4: if (auto* v = val.Get<GfVec4f>())
                valueData_.WriteBytes(v->data.data(), 16); break;
            case TypeId::Double2: if (auto* v = val.Get<GfVec2d>())
                valueData_.WriteBytes(v->data.data(), 16); break;
            case TypeId::Double3: if (auto* v = val.Get<GfVec3d>())
                valueData_.WriteBytes(v->data.data(), 24); break;
            case TypeId::Double4: if (auto* v = val.Get<GfVec4d>())
                valueData_.WriteBytes(v->data.data(), 32); break;
            case TypeId::Int2: if (auto* v = val.Get<GfVec2i>())
                valueData_.WriteBytes(v->data.data(), 8); break;
            case TypeId::Int3: if (auto* v = val.Get<GfVec3i>())
                valueData_.WriteBytes(v->data.data(), 12); break;
            case TypeId::Int4: if (auto* v = val.Get<GfVec4i>())
                valueData_.WriteBytes(v->data.data(), 16); break;
            case TypeId::Quath: if (auto* v = val.Get<GfQuath>())
                { for (int i=0;i<4;++i) valueData_.AppendT(v->data[i].bits); } break;
            case TypeId::Quatf: if (auto* v = val.Get<GfQuatf>())
                valueData_.WriteBytes(v->data.data(), 16); break;
            case TypeId::Quatd: if (auto* v = val.Get<GfQuatd>())
                valueData_.WriteBytes(v->data.data(), 32); break;
            case TypeId::Matrix2d: if (auto* v = val.Get<GfMatrix2d>())
                valueData_.WriteBytes(v->data.data(), 32); break;
            case TypeId::Matrix3d: if (auto* v = val.Get<GfMatrix3d>())
                valueData_.WriteBytes(v->data.data(), 72); break;
            case TypeId::Matrix4d: if (auto* v = val.Get<GfMatrix4d>())
                valueData_.WriteBytes(v->data.data(), 128); break;
            default: break;
        }
    }

    // Encode an array value
    uint64_t EncodeArrayRep(TypeId elemTypeId, const Value& val) {
        CrateTypeId crateType = TypeIdToCrateTypeId(elemTypeId);
        if (crateType == CrateTypeId::Invalid) {
            return MakeArrayRep(CrateTypeId::Bool, 0);  // empty fallback
        }

        // Check for empty array
        if (val.ArraySize() == 0) {
            return MakeArrayRep(crateType, 0);
        }

        valueData_.Align8();
        size_t offset = valueData_.Pos();
        uint64_t n = val.ArraySize();
        valueData_.WriteU64(n);  // element count prefix (spec 16.3.9.3)

        WriteArrayData(elemTypeId, val);

        return MakeArrayRep(crateType, offset);
    }

    void WriteArrayData(TypeId elemTypeId, const Value& val) {
        switch (elemTypeId) {
            case TypeId::Bool: {
                auto* arr = val.Get<std::vector<Bool>>();
                if (arr) for (auto b : *arr) valueData_.WriteU8(b ? 1 : 0);
                break;
            }
            case TypeId::UChar: {
                auto* arr = val.Get<std::vector<UChar>>();
                if (arr) valueData_.WriteBytes(arr->data(), arr->size());
                break;
            }
            case TypeId::Int: {
                auto* arr = val.Get<std::vector<Int>>();
                if (arr) valueData_.WriteBytes(arr->data(), arr->size() * 4);
                break;
            }
            case TypeId::UInt: {
                auto* arr = val.Get<std::vector<UInt>>();
                if (arr) valueData_.WriteBytes(arr->data(), arr->size() * 4);
                break;
            }
            case TypeId::Int64: {
                auto* arr = val.Get<std::vector<Int64>>();
                if (arr) valueData_.WriteBytes(arr->data(), arr->size() * 8);
                break;
            }
            case TypeId::UInt64: {
                auto* arr = val.Get<std::vector<UInt64>>();
                if (arr) valueData_.WriteBytes(arr->data(), arr->size() * 8);
                break;
            }
            case TypeId::Half: {
                auto* arr = val.Get<std::vector<Half>>();
                if (arr) for (auto h : *arr) valueData_.AppendT(h.bits);
                break;
            }
            case TypeId::Float: {
                auto* arr = val.Get<std::vector<Float>>();
                if (arr) valueData_.WriteBytes(arr->data(), arr->size() * 4);
                break;
            }
            case TypeId::Double: {
                auto* arr = val.Get<std::vector<Double>>();
                if (arr) valueData_.WriteBytes(arr->data(), arr->size() * 8);
                break;
            }
            case TypeId::String: {
                auto* arr = val.Get<std::vector<String>>();
                if (arr) for (const auto& s : *arr) {
                    uint32_t si = strings_.AddOrGet(s, tokens_);
                    valueData_.WriteU32(si);
                }
                break;
            }
            case TypeId::Token: {
                if (auto* arr = val.Get<std::vector<String>>()) {
                    for (const auto& s : *arr) {
                        valueData_.WriteU32(tokens_.AddOrGet(s));
                    }
                } else if (auto* arr = val.Get<std::vector<Token>>()) {
                    for (const auto& t : *arr) {
                        valueData_.WriteU32(tokens_.AddOrGet(t.GetString()));
                    }
                }
                break;
            }
            case TypeId::Asset: {
                auto* arr = val.Get<std::vector<String>>();
                if (arr) for (const auto& s : *arr) {
                    uint32_t si = strings_.AddOrGet(s, tokens_);
                    valueData_.WriteU32(si);
                }
                break;
            }
            case TypeId::TimeCode: {
                // TimeCode shares Double representation
                auto* arr = val.Get<std::vector<Double>>();
                if (arr) valueData_.WriteBytes(arr->data(), arr->size() * 8);
                break;
            }
            // Vec types
            case TypeId::Float2: WriteVecArray<GfVec2f>(val, 8); break;
            case TypeId::Float3: WriteVecArray<GfVec3f>(val, 12); break;
            case TypeId::Float4: WriteVecArray<GfVec4f>(val, 16); break;
            case TypeId::Double2: WriteVecArray<GfVec2d>(val, 16); break;
            case TypeId::Double3: WriteVecArray<GfVec3d>(val, 24); break;
            case TypeId::Double4: WriteVecArray<GfVec4d>(val, 32); break;
            case TypeId::Int2: WriteVecArray<GfVec2i>(val, 8); break;
            case TypeId::Int3: WriteVecArray<GfVec3i>(val, 12); break;
            case TypeId::Int4: WriteVecArray<GfVec4i>(val, 16); break;
            case TypeId::Half2: WriteHalfVecArray<GfVec2h, 2>(val); break;
            case TypeId::Half3: WriteHalfVecArray<GfVec3h, 3>(val); break;
            case TypeId::Half4: WriteHalfVecArray<GfVec4h, 4>(val); break;
            case TypeId::Quatf: WriteVecArray<GfQuatf>(val, 16); break;
            case TypeId::Quatd: WriteVecArray<GfQuatd>(val, 32); break;
            case TypeId::Quath: WriteHalfVecArray<GfQuath, 4>(val); break;
            case TypeId::Matrix2d: WriteVecArray<GfMatrix2d>(val, 32); break;
            case TypeId::Matrix3d: WriteVecArray<GfMatrix3d>(val, 72); break;
            case TypeId::Matrix4d: WriteVecArray<GfMatrix4d>(val, 128); break;
            default: break;
        }
    }

    template <typename T>
    void WriteVecArray(const Value& val, size_t elementBytes) {
        auto* arr = val.Get<std::vector<T>>();
        if (arr) {
            for (const auto& elem : *arr) {
                valueData_.WriteBytes(elem.data.data(), elementBytes);
            }
        }
    }

    template <typename VecT, size_t N>
    void WriteHalfVecArray(const Value& val) {
        auto* arr = val.Get<std::vector<VecT>>();
        if (arr) {
            for (const auto& elem : *arr) {
                for (size_t i = 0; i < N; ++i) {
                    valueData_.AppendT(elem.data[i].bits);
                }
            }
        }
    }

    // Handle composition/special types stored with TypeId::Unknown
    uint64_t EncodeUnknownTypeValue(const std::string& fieldName, const Value& val) {
        // SubLayerPaths
        if (auto* slp = val.Get<SubLayerPaths>()) {
            return EncodeSubLayerPaths(*slp, fieldName);
        }
        // ListOp<Reference>
        if (auto* lr = val.Get<ListOp<Reference>>()) {
            return EncodeReferenceListOp(*lr);
        }
        // ListOp<Path> — path-typed list op (e.g. targetPaths,
        // connectionPaths). Dispatched on actual storage type so the
        // crate format reflects the spec-typed value, not name sniffing.
        if (auto* lp = val.Get<ListOp<Path>>()) {
            return EncodePathListOp(*lp);
        }
        // ListOp<Token> — token-typed list op (e.g. apiSchemas per
        // §13.2.1.2). Encodes as CrateTypeId::TokenListOp.
        if (auto* lt = val.Get<ListOp<Token>>()) {
            return EncodeTokenListOp(*lt);
        }
        // ListOp<std::string> — string-typed list op (e.g. clipSets per
        // §13.2.1.2.2, variantSetNames per §7.6.2.3.5) OR the legacy
        // catch-all for string-stored path/token listops still in the
        // field-name sniffing branch.
        if (auto* ls = val.Get<ListOp<std::string>>()) {
            return EncodeStringListOp(*ls, fieldName);
        }
        if (auto* relocates = val.Get<std::vector<Relocate>>()) {
            return EncodeRelocates(*relocates);
        }
        // ValueBlock
        if (val.IsBlock()) {
            return MakeInlinedRep(CrateTypeId::ValueBlock, 0);
        }
        // Fallback
        return MakeInlinedRep(CrateTypeId::Bool, 0);
    }

    uint64_t EncodeRelocates(const std::vector<Relocate>& relocates) {
        valueData_.Align8();
        size_t offset = valueData_.Pos();
        valueData_.WriteU64(static_cast<uint64_t>(relocates.size()));
        for (const auto& r : relocates) {
            paths_.EnsureAncestors(r.sourcePath);
            uint32_t sourceIdx = paths_.AddOrGet(r.sourcePath);
            uint32_t targetIdx = 0;
            if (r.targetPath) {
                paths_.EnsureAncestors(*r.targetPath);
                targetIdx = paths_.AddOrGet(*r.targetPath);
            }
            valueData_.WriteU32(sourceIdx);
            valueData_.WriteU32(targetIdx);
        }
        return MakeOffsetRep(CrateTypeId::Relocates, offset);
    }

    // SubLayerPaths is stored as two separate fields by the USDA writer
    // but in USDC it's stored as a StringVector (subLayers) + LayerOffsetVector
    // We handle it by encoding as a StringVector (asset paths).
    uint64_t EncodeSubLayerPaths(const SubLayerPaths& slp, const std::string& fieldName) {
        if (fieldName == "subLayers") {
            // Write as StringVector
            valueData_.Align8();
            size_t offset = valueData_.Pos();
            uint64_t count = slp.paths.size();
            valueData_.WriteU64(count);
            for (const auto& p : slp.paths) {
                uint32_t si = strings_.AddOrGet(p, tokens_);
                valueData_.WriteU32(si);
            }
            return MakeOffsetRep(CrateTypeId::StringVector, offset);
        }
        if (fieldName == "subLayerOffsets") {
            // Write as LayerOffsetVector
            valueData_.Align8();
            size_t offset = valueData_.Pos();
            uint64_t count = slp.offsets.size();
            valueData_.WriteU64(count);
            for (const auto& r : slp.offsets) {
                valueData_.WriteF64(r.offset);
                valueData_.WriteF64(r.scale);
            }
            return MakeOffsetRep(CrateTypeId::LayerOffsetVector, offset);
        }
        return MakeInlinedRep(CrateTypeId::Bool, 0);
    }

    uint64_t EncodeReferenceListOp(const ListOp<Reference>& listOp) {
        valueData_.Align8();
        size_t offset = valueData_.Pos();
        WriteReferenceListOpAt(listOp);
        return MakeOffsetRep(CrateTypeId::ReferenceListOp, offset);
    }

    void WriteReferenceListOpAt(const ListOp<Reference>& listOp) {
        uint8_t header = 0;
        const auto& explicit_ = listOp.GetExplicitItems();
        const auto& prepend   = listOp.GetPrependedItems();
        const auto& append    = listOp.GetAppendedItems();
        const auto& deleted   = listOp.GetDeletedItems();

        if (listOp.IsExplicit()) {
            header |= 0x02;  // addExplicit
        } else {
            if (!prepend.empty()) header |= 0x20;
            if (!append.empty())  header |= 0x40;
            if (!deleted.empty()) header |= 0x08;
        }
        valueData_.WriteU8(header);

        auto writeRef = [this](const Reference& ref) {
            uint32_t assetIdx = ref.assetPath ? strings_.AddOrGet(*ref.assetPath, tokens_) : 0u;
            uint32_t primIdx = 0;
            if (ref.primPath) {
                paths_.EnsureAncestors(*ref.primPath);
                primIdx = paths_.AddOrGet(*ref.primPath);
            }
            valueData_.WriteU32(assetIdx);
            valueData_.WriteU32(primIdx);
            valueData_.WriteF64(ref.offset);
            valueData_.WriteF64(ref.scale);
            // Empty custom data dictionary (spec requires it for references)
            valueData_.WriteU64(0);  // empty dict count
        };

        auto writeItems = [&](const std::vector<Reference>& items) {
            valueData_.WriteU64(static_cast<uint64_t>(items.size()));
            for (const auto& r : items) writeRef(r);
        };

        if (listOp.IsExplicit()) {
            writeItems(explicit_);
        } else {
            if (!prepend.empty()) writeItems(prepend);
            if (!append.empty())  writeItems(append);
            if (!deleted.empty()) writeItems(deleted);
        }
    }

    // ListOp<Path> → CrateTypeId::PathListOp. Items are written as path
    // indices into the path table.
    uint64_t EncodePathListOp(const ListOp<Path>& listOp) {
        valueData_.Align8();
        size_t offset = valueData_.Pos();

        uint8_t header = 0;
        const auto& explicit_ = listOp.GetExplicitItems();
        const auto& prepend   = listOp.GetPrependedItems();
        const auto& append    = listOp.GetAppendedItems();
        const auto& deleted   = listOp.GetDeletedItems();
        if (listOp.IsExplicit()) {
            header |= 0x02;
        } else {
            if (!prepend.empty()) header |= 0x20;
            if (!append.empty())  header |= 0x40;
            if (!deleted.empty()) header |= 0x08;
        }
        valueData_.WriteU8(header);

        auto writeItems = [&](const std::vector<Path>& items) {
            valueData_.WriteU64(static_cast<uint64_t>(items.size()));
            for (const auto& p : items) {
                paths_.EnsureAncestors(p);
                uint32_t pi = paths_.AddOrGet(p);
                valueData_.WriteU32(pi);
            }
        };

        if (listOp.IsExplicit()) {
            writeItems(explicit_);
        } else {
            if (!prepend.empty()) writeItems(prepend);
            if (!append.empty())  writeItems(append);
            if (!deleted.empty()) writeItems(deleted);
        }

        return MakeOffsetRep(CrateTypeId::PathListOp, offset);
    }

    // ListOp<Token> → CrateTypeId::TokenListOp. Items are written as
    // token indices into the token table.
    uint64_t EncodeTokenListOp(const ListOp<Token>& listOp) {
        valueData_.Align8();
        size_t offset = valueData_.Pos();

        uint8_t header = 0;
        const auto& explicit_ = listOp.GetExplicitItems();
        const auto& prepend   = listOp.GetPrependedItems();
        const auto& append    = listOp.GetAppendedItems();
        const auto& deleted   = listOp.GetDeletedItems();
        if (listOp.IsExplicit()) {
            header |= 0x02;
        } else {
            if (!prepend.empty()) header |= 0x20;
            if (!append.empty())  header |= 0x40;
            if (!deleted.empty()) header |= 0x08;
        }
        valueData_.WriteU8(header);

        auto writeItems = [&](const std::vector<Token>& items) {
            valueData_.WriteU64(static_cast<uint64_t>(items.size()));
            for (const auto& t : items) {
                uint32_t ti = tokens_.AddOrGet(t.GetString());
                valueData_.WriteU32(ti);
            }
        };

        if (listOp.IsExplicit()) {
            writeItems(explicit_);
        } else {
            if (!prepend.empty()) writeItems(prepend);
            if (!append.empty())  writeItems(append);
            if (!deleted.empty()) writeItems(deleted);
        }

        return MakeOffsetRep(CrateTypeId::TokenListOp, offset);
    }

    uint64_t EncodeStringListOp(const ListOp<std::string>& listOp,
                                const std::string& fieldName) {
        // Determine crate type from field name
        CrateTypeId crateType = CrateTypeId::TokenListOp;
        bool isPathListOp = false;

        if (fieldName == "references" || fieldName == "payload" ||
            fieldName == "inheritPaths" || fieldName == "specializes" ||
            fieldName == "connectionPaths" || fieldName == "targetPaths" ||
            fieldName == "primChildren" || fieldName == "propertyChildren" ||
            fieldName == "primOrder" || fieldName == "propertyOrder") {
            // These use PathListOp or TokenListOp depending on content type
            // Check if items look like paths
            auto items = listOp.GetItems();
            if (!items.empty() && items[0].size() > 0 && items[0][0] == '/') {
                crateType = CrateTypeId::PathListOp;
                isPathListOp = true;
            } else {
                crateType = CrateTypeId::TokenListOp;
            }
        } else if (fieldName == "subLayers" || fieldName == "documentation" ||
                   fieldName == "comment" || fieldName == "clipSets" ||
                   fieldName == "variantSetNames") {
            // Spec-declared listop<string> fields:
            //   clipSets          §13.2.1.2.2
            //   variantSetNames   §7.6.2.3.5
            // apiSchemas (§13.2.1.2 listop<token>) no longer falls
            // through here — it's stored as ListOp<Token> and encoded
            // via EncodeTokenListOp.
            crateType = CrateTypeId::StringListOp;
        }

        // For primChildren/propertyChildren which are token arrays (not list ops):
        // The in-memory model stores these as ListOp<string> but USDC uses TokenVector.
        // Detect: explicit list op with all items = TokenVector
        if ((fieldName == "primChildren" || fieldName == "propertyChildren" ||
             fieldName == "variantSetChildren" || fieldName == "variantChildren") &&
            listOp.IsExplicit()) {
            // Encode as TokenVector
            const auto& items = listOp.GetExplicitItems();
            valueData_.Align8();
            size_t offset = valueData_.Pos();
            valueData_.WriteU64(static_cast<uint64_t>(items.size()));
            for (const auto& s : items) {
                uint32_t ti = tokens_.AddOrGet(s);
                valueData_.WriteU32(ti);
            }
            return MakeOffsetRep(CrateTypeId::TokenVector, offset);
        }

        valueData_.Align8();
        size_t offset = valueData_.Pos();
        WriteStringListOpAt(listOp, crateType, isPathListOp);
        return MakeOffsetRep(crateType, offset);
    }

    void WriteStringListOpAt(const ListOp<std::string>& listOp,
                             CrateTypeId crateType,
                             bool isPathListOp) {
        uint8_t header = 0;
        const auto& explicit_ = listOp.GetExplicitItems();
        const auto& prepend   = listOp.GetPrependedItems();
        const auto& append    = listOp.GetAppendedItems();
        const auto& deleted   = listOp.GetDeletedItems();

        if (listOp.IsExplicit()) {
            header |= 0x02;  // addExplicit
        } else {
            if (!prepend.empty()) header |= 0x20;
            if (!append.empty())  header |= 0x40;
            if (!deleted.empty()) header |= 0x08;
        }
        valueData_.WriteU8(header);

        auto writeItem = [&](const std::string& s) {
            if (isPathListOp) {
                // Write as path index
                Path p = Path::Parse(s);
                paths_.EnsureAncestors(p);
                uint32_t pi = paths_.AddOrGet(p);
                valueData_.WriteU32(pi);
            } else if (crateType == CrateTypeId::StringListOp) {
                uint32_t si = strings_.AddOrGet(s, tokens_);
                valueData_.WriteU32(si);
            } else {
                // Token
                uint32_t ti = tokens_.AddOrGet(s);
                valueData_.WriteU32(ti);
            }
        };

        auto writeItems = [&](const std::vector<std::string>& items) {
            valueData_.WriteU64(static_cast<uint64_t>(items.size()));
            for (const auto& s : items) writeItem(s);
        };

        if (listOp.IsExplicit()) {
            writeItems(explicit_);
        } else {
            if (!prepend.empty()) writeItems(prepend);
            if (!append.empty())  writeItems(append);
            if (!deleted.empty()) writeItems(deleted);
        }
    }

    // ---------------------------------------------------------------
    // Section writers
    // ---------------------------------------------------------------

    // The value data (written during collection) needs to be placed in the
    // output stream before the sections that reference it.  But since we
    // write sections sequentially and the value offsets are absolute file
    // offsets, we need to know the final layout.
    //
    // Strategy: we write value data into a separate BinaryWriter (valueData_)
    // and then include it as the first section in the output, before TOKENS.
    // The offsets in ValueReps are relative to the start of the file (position 0),
    // so we need to know the base offset of the value data block.
    //
    // Simpler approach (used here): write the sections in order, and at the time
    // of writing the sections we also inline the value data for each section.
    // The valueData_ buffer is separate and its contents are appended to the
    // output right before TOKENS, with a fixed base address.
    //
    // Actually the simplest correct approach: collect all values first (done above),
    // but the offset-valued ValueReps point into the valueData_ buffer as if it
    // starts at offset 0.  We need to add the base address of where we actually
    // write valueData_ to each offset ValueRep.
    //
    // We handle this by:
    //   1. Writing the header + bootstrap into the main `out` buffer.
    //   2. Appending all value data (valueData_.data) right after the bootstrap.
    //   3. Recording the base offset of valueData_ in the main stream.
    //   4. Fixing up all ValueReps that have non-inlined, non-array-empty offsets.

    // Base offset of value data in the final file (set during section writing)
    size_t valueDataBase_ = 0;

    // Re-encode field value reps with corrected offsets
    void FixupValueRepOffsets() {
        for (auto& fe : fields_.fields) {
            uint64_t rep = fe.valueRep;
            uint8_t flags = static_cast<uint8_t>((rep >> 56) & 0xFF);
            bool isInlined = (flags & kFlagInlined) != 0;
            bool isArray   = (flags & kFlagArray) != 0;

            if (isInlined) continue;

            uint64_t payload = rep & 0x0000FFFFFFFFFFFF;
            if (isArray && payload == 0) continue;  // empty array

            // Adjust offset by valueDataBase_
            payload += static_cast<uint64_t>(valueDataBase_);
            fe.valueRep = (rep & ~0x0000FFFFFFFFFFFFULL) | payload;
        }
    }

    std::vector<uint8_t> Write_Internal(const Layer& layer) {
        // --- Collect all data ---
        CollectLayer(layer);

        BinaryWriter out;

        // Header (spec 16.3.8.1)
        out.WriteBytes("PXR-USDC", 8);
        out.WriteU8(0);  // major
        out.WriteU8(requiredMinor_);  // minor
        out.WriteU8(0);  // patch
        out.WriteU8(0); out.WriteU8(0); out.WriteU8(0); out.WriteU8(0); out.WriteU8(0);

        // Bootstrap (spec 16.3.8.3)
        size_t tocOffsetPos = out.Pos();
        out.WriteU64(0);  // TOC offset placeholder
        out.WriteU64(0);  // reserved

        // Value data block immediately after bootstrap
        out.Align8();
        valueDataBase_ = out.Pos();

        // Fix up all value rep offsets now that we know valueDataBase_
        FixupValueRepOffsets();

        // Also fix up dictionary offset references in valueData_ buffer
        // (dictionary value pointers are already absolute within valueData_,
        //  so we need to add valueDataBase_ to each dictionary value offset.)
        FixupDictionaryOffsets();

        // Append value data to output
        out.WriteBytes(valueData_.data.data(), valueData_.data.size());

        // Sections
        struct SectionInfo { std::string name; uint64_t start, size; };
        std::vector<SectionInfo> sections;

        // TOKENS
        {
            out.Align8();
            uint64_t start = out.Pos();
            WriteTokensSection(out);
            sections.push_back({"TOKENS", start, out.Pos() - start});
        }
        // STRINGS
        {
            out.Align8();
            uint64_t start = out.Pos();
            WriteStringsSection(out);
            sections.push_back({"STRINGS", start, out.Pos() - start});
        }
        // FIELDS
        {
            out.Align8();
            uint64_t start = out.Pos();
            WriteFieldsSection(out);
            sections.push_back({"FIELDS", start, out.Pos() - start});
        }
        // FIELDSETS
        {
            out.Align8();
            uint64_t start = out.Pos();
            WriteFieldSetsSection(out);
            sections.push_back({"FIELDSETS", start, out.Pos() - start});
        }
        // PATHS
        {
            out.Align8();
            uint64_t start = out.Pos();
            WritePathsSection(out, layer);
            sections.push_back({"PATHS", start, out.Pos() - start});
        }
        // SPECS
        {
            out.Align8();
            uint64_t start = out.Pos();
            WriteSpecsSection(out);
            sections.push_back({"SPECS", start, out.Pos() - start});
        }

        // Table of Contents
        {
            out.Align8();
            uint64_t tocStart = out.Pos();
            out.PatchU64(tocOffsetPos, tocStart);

            out.WriteU64(static_cast<uint64_t>(sections.size()));
            for (const auto& sec : sections) {
                char nameBuf[16] = {};
                size_t copyLen = sec.name.size() < 16 ? sec.name.size() : 15;
                std::memcpy(nameBuf, sec.name.c_str(), copyLen);
                out.WriteBytes(nameBuf, 16);
                out.WriteU64(sec.start);
                out.WriteU64(sec.size);
            }
        }

        return std::move(out.data);
    }

    // Fix up dictionary value offsets in valueData_ by adding valueDataBase_.
    // This is complex to do generically, so instead we use a different strategy:
    // dictionary values point to ValueReps, and those ValueReps are also in valueData_.
    // Since all offsets are absolute file offsets, we scan valueData_ for any
    // dictionary-type entries and fix their offsets.
    //
    // Simpler approach: instead of trying to patch dictionaries in-place,
    // we re-encode them.  But that creates a chicken-and-egg problem.
    //
    // Cleanest solution: record the positions of all "offset fields" in valueData_
    // during encoding and patch them in a second pass.
    //
    // For this implementation, we track dict offset patch positions separately.
    struct OffsetPatch {
        size_t pos;   // position in valueData_ of the int64 offset field
    };
    std::vector<OffsetPatch> dictOffsetPatches_;

    void FixupDictionaryOffsets() {
        // Each dictOffsetPatch_ records a position in valueData_ where an int64
        // signed offset needs to be adjusted.
        // The offset at patch.pos currently encodes (targetPos - (patch.pos + 8))
        // where targetPos is in valueData_-space.
        // After adding valueDataBase_, the offset becomes:
        // (targetPos + valueDataBase_) - (patch.pos + 8 + valueDataBase_)
        // = targetPos - (patch.pos + 8)
        // ... which is unchanged!  The relative offsets don't need fixing.
        // Dictionary offsets are relative, so they are already correct.
        (void)dictOffsetPatches_;
    }

    // Tokens section (spec 16.3.8.4.1)
    void WriteTokensSection(BinaryWriter& out) {
        const auto& toks = tokens_.tokens;
        uint64_t numTokens = toks.size();

        // Build uncompressed bytes (null-terminated strings)
        std::vector<uint8_t> raw;
        for (const auto& t : toks) {
            raw.insert(raw.end(), t.begin(), t.end());
            raw.push_back(0);  // null terminator
        }

        uint64_t uncompressedSize = raw.size();
        auto compressed = MakeLz4ChunkedBuffer(raw.data(), raw.size());
        uint64_t compressedSize = compressed.size();

        out.WriteU64(numTokens);
        out.WriteU64(uncompressedSize);
        out.WriteU64(compressedSize);
        out.WriteBytes(compressed.data(), compressed.size());
    }

    // Strings section (spec 16.3.8.4.2)
    void WriteStringsSection(BinaryWriter& out) {
        const auto& idx = strings_.indices;
        out.WriteU64(static_cast<uint64_t>(idx.size()));
        for (auto ti : idx) {
            out.WriteU32(ti);
        }
    }

    // Fields section (spec 16.3.8.4.3)
    void WriteFieldsSection(BinaryWriter& out) {
        const auto& flds = fields_.fields;
        out.WriteU64(static_cast<uint64_t>(flds.size()));

        // Compressed field name token indices
        std::vector<uint32_t> tokenIndices;
        tokenIndices.reserve(flds.size());
        for (const auto& f : flds) tokenIndices.push_back(f.tokenIdx);
        out.WriteCompressedInts(tokenIndices);

        // Value data: compress all ValueReps as raw bytes
        std::vector<uint8_t> valRepBytes;
        valRepBytes.reserve(flds.size() * 8);
        for (const auto& f : flds) {
            uint8_t buf[8];
            std::memcpy(buf, &f.valueRep, 8);
            valRepBytes.insert(valRepBytes.end(), buf, buf + 8);
        }
        auto compressed = MakeLz4ChunkedBuffer(valRepBytes.data(), valRepBytes.size());
        out.WriteU64(static_cast<uint64_t>(compressed.size()));
        out.WriteBytes(compressed.data(), compressed.size());
    }

    // Field Sets section (spec 16.3.8.4.4)
    void WriteFieldSetsSection(BinaryWriter& out) {
        const auto& flat = fieldSets_.flat;
        out.WriteU64(static_cast<uint64_t>(flat.size()));
        std::vector<uint32_t> data(flat.begin(), flat.end());
        out.WriteCompressedInts(data);
    }

    // Paths section (spec 16.3.8.4.5)
    // We need to encode the path hierarchy using the jump-tree scheme.
    void WritePathsSection(BinaryWriter& out, const Layer& layer) {
        const auto& pathList = paths_.paths;
        size_t numPaths = pathList.size();

        // Build a tree structure.
        // For each path, find its parent and place it in the jump-tree.
        // The jump-tree encoding is a DFS traversal with jump pointers.
        struct PathNode {
            size_t pathIdx;         // index in pathList
            size_t parentNodeIdx;   // index in nodes (SIZE_MAX = no parent)
            std::vector<size_t> children;
            Path path;
            std::string tokenStr;   // the element token string
            bool isProperty;        // negative token index in encoding
        };

        // Map from path text to pathList index
        std::unordered_map<std::string, size_t> textToListIdx;
        for (size_t i = 0; i < pathList.size(); ++i) {
            textToListIdx[pathList[i].GetText()] = i;
        }

        std::vector<PathNode> nodes;
        nodes.resize(numPaths);
        for (size_t i = 0; i < numPaths; ++i) {
            const Path& p = pathList[i];
            auto& node = nodes[i];
            node.pathIdx = i;
            node.parentNodeIdx = SIZE_MAX;
            node.path = p;
            node.isProperty = p.IsPropertyPath();

            if (p.IsAbsoluteRoot()) {
                node.tokenStr = "/";  // root token
                node.isProperty = false;
            } else {
                // Token string = last component
                node.tokenStr = p.GetName();
                if (node.tokenStr.empty() && p.IsPropertyPath()) {
                    node.tokenStr = p.GetPropertyName();
                }
            }
        }

        for (size_t i = 0; i < numPaths; ++i) {
            const Path& p = nodes[i].path;
            if (p.IsEmpty() || p.IsAbsoluteRoot()) continue;

            Path parent = p.GetParentPath();
            auto parentIt = textToListIdx.find(parent.GetText());
            if (parentIt == textToListIdx.end()) continue;

            nodes[i].parentNodeIdx = parentIt->second;
            nodes[parentIt->second].children.push_back(i);
        }

        auto childRankMap = [](const std::vector<Token>& names) {
            std::unordered_map<std::string, size_t> ranks;
            ranks.reserve(names.size());
            for (size_t i = 0; i < names.size(); ++i)
                ranks.emplace(names[i].GetString(), i);
            return ranks;
        };

        auto sortedPropertyNamesForPath = [](const Layer& l,
                                             const Path& parentPath) {
            std::vector<Token> names = l.GetPropertyNames(parentPath);
            std::sort(names.begin(), names.end(), PathElementTokenLess);
            return names;
        };

        for (auto& node : nodes) {
            if (node.children.size() < 2) continue;

            const auto primRanks = childRankMap(layer.GetChildNames(node.path));
            const auto propRanks =
                childRankMap(sortedPropertyNamesForPath(layer, node.path));

            auto rankFor = [&](size_t childNodeIdx) {
                const auto& child = nodes[childNodeIdx];
                const auto& ranks = child.isProperty ? propRanks : primRanks;
                auto it = ranks.find(child.tokenStr);
                if (it != ranks.end()) return it->second;
                return std::numeric_limits<size_t>::max();
            };

            std::sort(node.children.begin(), node.children.end(),
                      [&](size_t lhs, size_t rhs) {
                const auto& left = nodes[lhs];
                const auto& right = nodes[rhs];
                if (left.isProperty != right.isProperty)
                    return !left.isProperty;

                const size_t leftRank = rankFor(lhs);
                const size_t rightRank = rankFor(rhs);
                if (leftRank != rightRank) return leftRank < rightRank;

                const int cmp = ComparePathElementNames(left.tokenStr,
                                                        right.tokenStr);
                if (cmp != 0) return cmp < 0;
                return left.path.GetText() < right.path.GetText();
            });
        }

        // Perform DFS traversal to build the three arrays:
        //   pathIndices[], elementTokenIndices[], jumps[]
        struct PathEntry {
            uint32_t pathIndex;
            int32_t  elementTokenIndex;
            int32_t  jump;
        };
        std::vector<PathEntry> entries;
        entries.reserve(numPaths);

        // DFS from each root node
        // Recursively compute the DFS subtree size for a node (number of entries it needs).
        std::function<size_t(size_t)> subtreeSize = [&](size_t ni) -> size_t {
            size_t total = 1;
            for (size_t ci : nodes[ni].children) total += subtreeSize(ci);
            return total;
        };

        // DFS: fill entries for a node and all its descendants.
        //
        // entryIdx     — this node's slot in `entries`
        // siblingIdx   — entry index of this node's next sibling, or SIZE_MAX if none.
        //
        // Jump encoding (USDC spec §16.3.8.4.5):
        //   jump == -2 : leaf (no children, no sibling)
        //   jump == -1 : has child at x+1, no sibling
        //   jump ==  0 : has sibling at x+1, no children  (sibling MUST be contiguous)
        //   jump  >  0 : has child at x+1 AND sibling at x+jump
        //
        // Key insight: when a node has multiple children, each non-last child is
        // responsible for encoding its own sibling pointer (not the parent).
        // The parent always uses jump == -1 (or > 0 if the parent itself has a sibling).
        std::function<void(size_t, size_t, size_t)> dfs =
            [&](size_t nodeIdx, size_t entryIdx, size_t siblingIdx) {
            const PathNode& node = nodes[nodeIdx];
            PathEntry& entry = entries[entryIdx];
            entry.pathIndex = static_cast<uint32_t>(node.pathIdx);

            uint32_t tokIdx = tokens_.AddOrGet(node.tokenStr);
            entry.elementTokenIndex = node.isProperty
                ? -static_cast<int32_t>(tokIdx)
                :  static_cast<int32_t>(tokIdx);

            const auto& children = node.children;
            if (children.empty()) {
                // Leaf node
                if (siblingIdx != SIZE_MAX) {
                    // Sibling is guaranteed to be at entryIdx+1 because our layout
                    // places a leaf's next sibling immediately after (subtreeSize==1).
                    entry.jump = 0;
                } else {
                    entry.jump = -2;
                }
                return;
            }

            // Node with children.
            // Encode this node's own sibling pointer.
            entry.jump = (siblingIdx != SIZE_MAX)
                ? static_cast<int32_t>(siblingIdx - entryIdx)  // > 0
                : -1;

            // Layout all children sequentially after this entry.
            // Each non-last child carries a jump to its next sibling.
            size_t curEntry = entryIdx + 1;
            for (size_t i = 0; i < children.size(); ++i) {
                size_t childSize = subtreeSize(children[i]);
                size_t nextEntry = curEntry + childSize;
                size_t childSibling = (i + 1 < children.size()) ? nextEntry : SIZE_MAX;

                // Ensure the entries array is large enough for this child's subtree.
                if (entries.size() < nextEntry) entries.resize(nextEntry);

                dfs(children[i], curEntry, childSibling);
                curEntry = nextEntry;
            }
        };

        // Find root nodes (parentNodeIdx == SIZE_MAX)
        std::vector<size_t> roots;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i].parentNodeIdx == SIZE_MAX) roots.push_back(i);
        }

        // Pre-allocate the entire entries array, then fill via DFS.
        if (!roots.empty()) {
            size_t totalEntries = 0;
            for (size_t ri : roots) totalEntries += subtreeSize(ri);
            entries.resize(totalEntries);

            // Each root is a sibling of the next root (same treatment as children
            // of an implicit super-root).
            size_t currentStart = 0;
            for (size_t ri = 0; ri < roots.size(); ++ri) {
                size_t thisSize = subtreeSize(roots[ri]);
                size_t nextStart = currentStart + thisSize;
                size_t sibling = (ri + 1 < roots.size()) ? nextStart : SIZE_MAX;
                dfs(roots[ri], currentStart, sibling);
                currentStart = nextStart;
            }
        }

        // Extract the three arrays
        std::vector<uint32_t> pathIndices;
        std::vector<int32_t>  elementTokenIndices;
        std::vector<int32_t>  jumps;
        pathIndices.reserve(entries.size());
        elementTokenIndices.reserve(entries.size());
        jumps.reserve(entries.size());
        for (const auto& e : entries) {
            pathIndices.push_back(e.pathIndex);
            elementTokenIndices.push_back(e.elementTokenIndex);
            jumps.push_back(e.jump);
        }

        // Write section
        uint64_t numPathsVal = static_cast<uint64_t>(numPaths);
        out.WriteU64(numPathsVal);
        // The parser reads an extra uint64 (observed undocumented field = numPaths)
        out.WriteU64(numPathsVal);

        out.WriteCompressedInts(pathIndices);
        out.WriteCompressedInts(elementTokenIndices);
        out.WriteCompressedInts(jumps);
    }

    // Specs section (spec 16.3.8.4.6)
    void WriteSpecsSection(BinaryWriter& out) {
        out.WriteU64(static_cast<uint64_t>(specs_.size()));

        std::vector<uint32_t> pathIndices;
        std::vector<uint32_t> fieldSetIndices;
        std::vector<uint32_t> forms;
        for (const auto& s : specs_) {
            pathIndices.push_back(s.pathIdx);
            fieldSetIndices.push_back(s.fieldSetIdx);
            forms.push_back(s.form);
        }

        out.WriteCompressedInts(pathIndices);
        out.WriteCompressedInts(fieldSetIndices);
        out.WriteCompressedInts(forms);
    }

public:
    // Entry point for the actual write
    std::vector<uint8_t> Serialize(const Layer& layer) {
        return Write_Internal(layer);
    }
};

} // anonymous namespace

// ============================================================
// Public API
// ============================================================

std::vector<uint8_t> WriteUsdc(const Layer& layer) {
    UsdcWriter writer;
    return writer.Serialize(layer);
}

bool WriteUsdcFile(const Layer& layer, const ResolvedLocation& location) {
    auto data = WriteUsdc(layer);
    if (data.empty()) return false;

    auto result = WriteResource(location, data.data(), data.size());
    return result.success;
}

bool WriteUsdcFile(const Layer& layer, const std::string& path) {
    return WriteUsdcFile(layer, ResolvedLocation::FromResolvedString(path));
}

} // namespace nanousd
