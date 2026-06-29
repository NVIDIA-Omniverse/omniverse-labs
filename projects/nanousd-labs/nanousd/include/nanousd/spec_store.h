// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "path.h"   // for PathMap (used by specs_ / primChildren_ / etc.)

#include "spec.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace nanousd {

// ============================================================
// SpecStore — pluggable spec storage backend for Layer
// ============================================================
// Layer delegates all spec storage to a SpecStore implementation.
// HashMapSpecStore is the default (eager, in-memory hash map).
// Format-specific stores (e.g. USDC crate-backed) can decode
// specs on demand without Layer or composition knowing.

class SpecStore {
public:
    virtual ~SpecStore() = default;

    // --- Read access ---
    virtual const Spec* GetSpec(const Path& path) const = 0;
    virtual Spec* GetMutableSpec(const Path& path) = 0;
    virtual bool HasSpec(const Path& path) const = 0;

    // --- Write access ---
    virtual void SetSpec(const Path& path, Spec spec) = 0;
    virtual bool RemoveSpec(const Path& path) = 0;

    // --- Index access ---
    virtual const std::vector<Token>& GetChildNames(const Path& primPath) const = 0;
    virtual const std::vector<Token>& GetPropertyNames(const Path& primPath) const = 0;
    virtual const std::vector<Token>& GetAttributeNames(const Path& primPath) const = 0;

    // --- Iteration ---
    virtual std::vector<Path> GetSpecPaths() const = 0;
    virtual void ForEachSpec(
        const std::function<void(const Path&, const Spec&)>& fn) const = 0;

    // --- COW support ---
    virtual std::unique_ptr<SpecStore> Clone() const = 0;

    // --- Read-only check (for write-upgrade pattern) ---
    virtual bool IsReadOnly() const { return false; }

private:
    friend class Layer;

    // Internal fast indexes used by composition/stage population. Keep these
    // off the public SpecStore surface: they are cache shapes, not API contract.
    virtual const std::vector<Path>& GetChildPathsInternal(
        const Path& primPath) const = 0;
    virtual const std::vector<Path>& GetArcOpinionPrimPathsInternal() const = 0;
    virtual void ForEachChildIndexInternal(
        const std::function<void(const Path&,
                                 const std::vector<Token>&,
                                 const std::vector<Path>&)>& fn) const = 0;
};

// ============================================================
// HashMapSpecStore — default eager in-memory store
// ============================================================

class HashMapSpecStore : public SpecStore {
public:
    struct ChildIndexEntry {
        std::vector<Token> names;
        std::vector<Path> paths;
    };

    const Spec* GetSpec(const Path& path) const override {
        auto it = specs_.find(path);
        return it != specs_.end() ? &it->second : nullptr;
    }

    Spec* GetMutableSpec(const Path& path) override {
        MarkArcOpinionPrimPathsDirty();
        auto it = specs_.find(path);
        return it != specs_.end() ? &it->second : nullptr;
    }

    bool HasSpec(const Path& path) const override {
        return specs_.count(path) > 0;
    }

    void SetSpec(const Path& path, Spec spec) override {
        MarkArcOpinionPrimPathsDirty();
        auto it = specs_.find(path);
        if (it == specs_.end()) {
            auto [inserted, _] = specs_.emplace(path, std::move(spec));
            AddToIndex(path, inserted->second);
            return;
        }

        const SpecType oldType = it->second.GetType();
        bool wasAttribute =
            path.IsPropertyPath() &&
            oldType == SpecType::Attribute;
        SpecType newType = spec.GetType();
        it->second = std::move(spec);
        if (!path.IsPropertyPath() &&
            (oldType == SpecType::Prim) != (newType == SpecType::Prim)) {
            RemoveFromIndex(path);
            AddToIndex(path, it->second);
        } else if (path.IsPropertyPath() &&
                   wasAttribute != (newType == SpecType::Attribute)) {
            // Replacing a property spec must not remove/re-add the property
            // name: that changes authored enumeration order. Only refresh
            // the derived attribute-only index when membership changes.
            RebuildAttributeIndexForPrim(path.GetPrimPath());
        }
    }

    bool RemoveSpec(const Path& path) override {
        MarkArcOpinionPrimPathsDirty();
        if (specs_.erase(path) > 0) {
            RemoveFromIndex(path);
            return true;
        }
        return false;
    }

    const std::vector<Token>& GetChildNames(const Path& primPath) const override {
        auto it = primChildren_.find(primPath);
        return it != primChildren_.end() ? it->second.names : emptyNames_;
    }

    const std::vector<Token>& GetPropertyNames(const Path& primPath) const override {
        auto it = primProperties_.find(primPath);
        return it != primProperties_.end() ? it->second : emptyNames_;
    }

    const std::vector<Token>& GetAttributeNames(const Path& primPath) const override {
        auto it = primAttributes_.find(primPath);
        return it != primAttributes_.end() ? it->second : emptyNames_;
    }

    std::vector<Path> GetSpecPaths() const override {
        std::vector<Path> result;
        result.reserve(specs_.size());
        for (const auto& [path, _] : specs_) {
            result.push_back(path);
        }
        return result;
    }

    void ForEachSpec(
        const std::function<void(const Path&, const Spec&)>& fn) const override {
        for (const auto& [path, spec] : specs_) {
            fn(path, spec);
        }
    }

    std::unique_ptr<SpecStore> Clone() const override {
        auto clone = std::make_unique<HashMapSpecStore>();
        clone->specs_ = specs_;
        clone->primChildren_ = primChildren_;
        clone->primProperties_ = primProperties_;
        clone->primAttributes_ = primAttributes_;
        {
            std::lock_guard<std::mutex> lock(arcOpinionPrimPathsMutex_);
            clone->arcOpinionPrimPaths_ = arcOpinionPrimPaths_;
            clone->arcOpinionPrimPathsDirty_ = arcOpinionPrimPathsDirty_;
        }
        return clone;
    }

    // --- Bulk construction helpers (for parsers with pre-built indices) ---

    // Pre-allocate buckets so the per-spec inserts in SetSpecNoIndex
    // don't trigger ~log2(n) rehashes during a bulk load. Caller passes
    // an upper bound on the expected spec count (USDC's spec table size
    // is known up-front).
    void Reserve(size_t expectedSpecs) {
        specs_.reserve(expectedSpecs);
    }

    // Insert spec without building index (caller provides pre-built indices).
    void SetSpecNoIndex(const Path& path, Spec spec) {
        MarkArcOpinionPrimPathsDirty();
        specs_[path] = std::move(spec);
    }

    // Inject pre-built child/property indices (from USDC path reconstruction).
    void SetIndices(
        PathMap<std::vector<Token>> primChildren,
        PathMap<std::vector<Path>> primChildPaths,
        PathMap<std::vector<Token>> primProperties) {
        primChildren_.clear();
        primChildren_.reserve(primChildren.size());
        for (auto& [parent, names] : primChildren) {
            auto& children = primChildren_[parent];
            children.names = std::move(names);
            auto pathsIt = primChildPaths.find(parent);
            if (pathsIt != primChildPaths.end())
                children.paths = std::move(pathsIt->second);
        }
        primProperties_ = std::move(primProperties);
        RebuildAttributeIndex();
    }

private:
    const std::vector<Path>& GetChildPathsInternal(
        const Path& primPath) const override {
        auto it = primChildren_.find(primPath);
        return it != primChildren_.end() ? it->second.paths : emptyPaths_;
    }

    const std::vector<Path>& GetArcOpinionPrimPathsInternal() const override {
        std::lock_guard<std::mutex> lock(arcOpinionPrimPathsMutex_);
        if (arcOpinionPrimPathsDirty_) RebuildArcOpinionPrimPaths();
        return arcOpinionPrimPaths_;
    }

    void ForEachChildIndexInternal(
        const std::function<void(const Path&,
                                 const std::vector<Token>&,
                                 const std::vector<Path>&)>& fn) const override {
        for (const auto& [parent, children] : primChildren_) {
            fn(parent, children.names, children.paths);
        }
    }

    void MarkArcOpinionPrimPathsDirty() {
        std::lock_guard<std::mutex> lock(arcOpinionPrimPathsMutex_);
        arcOpinionPrimPathsDirty_ = true;
    }

    void AddToIndex(const Path& path, const Spec& spec) {
        if (path.IsPropertyPath()) {
            Path parent = path.GetPrimPath();
            Token name = path.GetPropertyName();
            auto& props = primProperties_[parent];
            for (const auto& p : props) {
                if (p == name) return;
            }
            props.push_back(name);
            if (spec.GetType() == SpecType::Attribute) {
                auto& attrs = primAttributes_[parent];
                for (const auto& a : attrs) {
                    if (a == name) return;
                }
                attrs.push_back(name);
            }
            return;
        }
        if (spec.GetType() != SpecType::Prim) return;
        if (path.IsEmpty() || path.IsAbsoluteRoot()) return;
        Path parent = path.GetParentPath();
        Token name = path.GetName();
        if (name.IsEmpty()) return;
        auto& children = primChildren_[parent];
        for (const auto& c : children.names) {
            if (c == name) return;
        }
        children.names.push_back(name);
        children.paths.push_back(path);
    }

    void RemoveFromIndex(const Path& path) {
        if (path.IsPropertyPath()) {
            Path parent = path.GetPrimPath();
            Token name = path.GetPropertyName();
            auto it = primProperties_.find(parent);
            if (it != primProperties_.end()) {
                auto& v = it->second;
                v.erase(std::remove(v.begin(), v.end(), name), v.end());
                if (v.empty()) primProperties_.erase(it);
            }
            auto ait = primAttributes_.find(parent);
            if (ait != primAttributes_.end()) {
                auto& v = ait->second;
                v.erase(std::remove(v.begin(), v.end(), name), v.end());
                if (v.empty()) primAttributes_.erase(ait);
            }
            return;
        }
        if (path.IsEmpty() || path.IsAbsoluteRoot()) return;
        Path parent = path.GetParentPath();
        Token name = path.GetName();
        if (name.IsEmpty()) return;
        auto it = primChildren_.find(parent);
        if (it != primChildren_.end()) {
            auto& names = it->second.names;
            auto& paths = it->second.paths;
            for (size_t i = 0; i < names.size(); ++i) {
                if (names[i] != name) continue;
                names.erase(names.begin() + static_cast<std::ptrdiff_t>(i));
                if (i < paths.size())
                    paths.erase(paths.begin() + static_cast<std::ptrdiff_t>(i));
                break;
            }
            if (names.empty() && paths.empty()) primChildren_.erase(it);
        }
    }

    void RebuildAttributeIndex() {
        primAttributes_.clear();
        for (const auto& [primPath, props] : primProperties_) {
            auto& attrs = primAttributes_[primPath];
            for (const auto& name : props) {
                Path propPath = primPath.AppendProperty(name);
                auto it = specs_.find(propPath);
                if (it != specs_.end() &&
                    it->second.GetType() == SpecType::Attribute) {
                    attrs.push_back(name);
                }
            }
            if (attrs.empty()) primAttributes_.erase(primPath);
        }
    }

    void RebuildAttributeIndexForPrim(const Path& primPath) {
        auto propsIt = primProperties_.find(primPath);
        if (propsIt == primProperties_.end()) {
            primAttributes_.erase(primPath);
            return;
        }

        std::vector<Token> attrs;
        attrs.reserve(propsIt->second.size());
        for (const auto& name : propsIt->second) {
            Path propPath = primPath.AppendProperty(name);
            auto it = specs_.find(propPath);
            if (it != specs_.end() &&
                it->second.GetType() == SpecType::Attribute) {
                attrs.push_back(name);
            }
        }

        if (attrs.empty()) {
            primAttributes_.erase(primPath);
        } else {
            primAttributes_[primPath] = std::move(attrs);
        }
    }

    void RebuildArcOpinionPrimPaths() const {
        arcOpinionPrimPaths_.clear();
        for (const auto& [path, spec] : specs_) {
            if (spec.GetType() == SpecType::Prim && spec.HasArcOpinion()) {
                arcOpinionPrimPaths_.push_back(path);
            }
        }
        arcOpinionPrimPathsDirty_ = false;
    }

    PathMap<Spec> specs_;
    PathMap<ChildIndexEntry> primChildren_;
    PathMap<std::vector<Token>> primProperties_;
    PathMap<std::vector<Token>> primAttributes_;
    mutable std::mutex arcOpinionPrimPathsMutex_;
    mutable std::vector<Path> arcOpinionPrimPaths_;
    mutable bool arcOpinionPrimPathsDirty_ = true;
    static inline const std::vector<Token> emptyNames_;
    static inline const std::vector<Path> emptyPaths_;
};

} // namespace nanousd
