// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"
#include "external/unordered_dense.h"
#include "listop.h"
#include "relative_path.h"   // for PathElement (used by GetElements())
#include "types.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace nanousd {

// Forward declaration of the internal pool entry. Defined in
// src/path_pool.{h,cpp}. Public API never exposes this type.
struct PathData;

// SdfPath equivalent per spec Chapter 8.
//
// `Path` is a value-semantic 8-byte handle into a process-global
// path-intern pool. Two `Path` objects compare equal iff they refer
// to the same pool entry, and operations that "modify" a path (e.g.
// `AppendChild`) return a new `Path` whose handle points to the
// pool entry representing the resulting absolute path. This is the
// same interning pattern `Token` uses, applied to paths.
//
// Paths in this API are **always absolute**. Relative authorings
// (".", "..", "../foo.attr") are represented by `RelativePath` and
// only exist transiently at the API boundary; calling
// `Path::Anchor(rel, base)` resolves a `RelativePath` against an
// absolute base into an interned `Path`.
//
// The path "/" is the absolute root (pseudo-root / layer spec).
class NANOUSD_CORE_API Path {
public:
    // Default-constructed Path is empty (no pool entry).
    Path();

    // Parse an **absolute** path from text. Relative authorings
    // ("..", ".", "../foo") return an empty Path here — use
    // `RelativePath::Parse` followed by `Path::Anchor` for those.
    // The single text "/" parses to AbsoluteRoot().
    static Path Parse(std::string_view text);

    // Named constructors
    static Path AbsoluteRoot();

    // --- Queries ---

    bool IsEmpty() const         { return data_ == nullptr; }
    bool IsAbsolute() const      { return !IsEmpty(); }       // always true
    bool IsAbsoluteRoot() const;
    bool IsPrimPath() const;
    bool IsPropertyPath() const  { return HasProperty(); }
    bool HasProperty() const;
    bool HasVariantSelections() const;

    // The prim elements (names and variant selections) — the
    // historical flat view. Materialised by walking the chain on
    // each call. Returned by value so callers can safely hold two
    // results simultaneously (e.g. comparing two paths' elements);
    // an in-place cache would force per-PathData mutable state.
    std::vector<PathElement> GetElements() const;

    // The property name (empty if this is a prim path).
    const Token& GetPropertyName() const;

    // The property namespace segments (split by ':').
    std::vector<std::string_view> GetPropertyNamespaces() const;

    // Canonical textual representation. Built on each call from
    // the chain — see GetElements() for the lifetime rationale.
    std::string GetText() const;
    std::string GetString() const { return GetText(); }

    // --- Path operations ---

    // Strip the last step (last element, or property if any).
    Path GetParentPath() const;

    // Strip the property component (no-op for prim paths).
    Path GetPrimPath() const;

    // Append a child prim name.
    Path AppendChild(const Token& childName) const;

    // Append a property name. Fails (returns empty) if this is
    // already a property path.
    Path AppendProperty(const Token& propName) const;

    // Append a variant selection. Fails (returns empty) if this
    // path is the absolute root or a property path.
    Path AppendVariantSelection(const Token& setName,
                                const Token& variantName) const;

    // The last step's name: property name if HasProperty(),
    // otherwise the last prim element's name. Empty for the root
    // and for empty paths.
    Token GetName() const;

    // --- Comparison ---
    // Pointer-compare equality (interning makes this exact).

    bool operator==(const Path& o) const { return data_ == o.data_; }
    bool operator!=(const Path& o) const { return data_ != o.data_; }

    // Lexicographic order on canonical text. Forces text
    // materialisation on the pool entries being compared (cached).
    bool operator<(const Path& o) const;

    // O(1) cached hash — pointer-derived. Inline so it doesn't
    // need a separate DLL-exported member symbol; calls a free
    // NANOUSD_CORE_API helper that does the heavy work.
    struct Hash {
        std::size_t operator()(const Path& p) const noexcept;
    };

    // Internal: opaque handle to the pool entry. Public only so
    // Hash::operator() can be inlined.
    const PathData* GetData_() const { return data_; }

private:
    explicit Path(const PathData* data) : data_(data) {}
    const PathData* data_ = nullptr;
};

// AOUSD path element ordering (Path Grammar, Element Ordering).
// This is distinct from Token's generic lexical ordering.
NANOUSD_CORE_API int ComparePathElementNames(std::string_view lhs,
                                             std::string_view rhs);
NANOUSD_CORE_API int ComparePathsByPathElementOrder(const Path& lhs,
                                                    const Path& rhs);

inline bool PathElementNameLess(std::string_view lhs, std::string_view rhs) {
    return ComparePathElementNames(lhs, rhs) < 0;
}

inline bool PathElementTokenLess(const Token& lhs, const Token& rhs) {
    return ComparePathElementNames(lhs.GetString(), rhs.GetString()) < 0;
}

inline std::size_t Path::Hash::operator()(const Path& p) const noexcept {
    return std::hash<const void*>{}(p.GetData_());
}

// Path-keyed flat-map and flat-set aliases. ankerl::unordered_dense
// uses open addressing + power-of-two capacity, avoiding the O(N)
// _M_rehash cost std::unordered_map pays on bulk inserts.
template <typename V>
using PathMap = ankerl::unordered_dense::segmented_map<Path, V, Path::Hash>;
using PathSet = ankerl::unordered_dense::segmented_set<Path, Path::Hash>;

// ObjectPath: a path that targets prim or property specs (Section 7.4.2.3)
using ObjectPath = Path;

using ListOpPath = ListOp<Path>;

} // namespace nanousd

// std::hash specialization so Path works in std containers (including ListOp)
template <>
struct std::hash<nanousd::Path> {
    size_t operator()(const nanousd::Path& p) const noexcept {
        return nanousd::Path::Hash{}(p);
    }
};
