// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"
#include "spec_store.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nanousd {

class DecodeBacking;  // include "decode_backing.h" to use
class Layer;

namespace detail {
const std::vector<Path>& GetLayerChildPaths(const Layer& layer,
                                            const Path& primPath);
const std::vector<Path>& GetLayerArcOpinionPrimPaths(const Layer& layer);
void ForEachLayerChildIndex(
    const Layer& layer,
    const std::function<void(const Path&,
                             const std::vector<Token>&,
                             const std::vector<Path>&)>& fn);
} // namespace detail

// Layer: the fundamental container per spec Section 5.1.13 / 7.2.
// Contains a layer spec (root metadata at "/") and a hierarchy of child specs
// addressable by path.
//
// Spec storage is delegated to a SpecStore implementation. HashMapSpecStore
// is the default (eager, in-memory). Format-specific stores (e.g. USDC
// crate-backed) can decode specs on demand without callers knowing.

class NANOUSD_CORE_API Layer {
public:
    Layer() : layerSpec_(SpecType::Layer),
              store_(std::make_unique<HashMapSpecStore>()) {}

    // Construct with a custom spec store (used by parsers).
    explicit Layer(std::unique_ptr<SpecStore> store)
        : layerSpec_(SpecType::Layer), store_(std::move(store)) {}

    // Explicit copy/move (needed because of unique_ptr<SpecStore>)
    Layer(const Layer& other)
        : layerSpec_(other.layerSpec_),
          store_(other.store_ ? other.store_->Clone()
                              : std::make_unique<HashMapSpecStore>()),
          backing_(other.backing_) {}
    Layer& operator=(const Layer& other) {
        if (this != &other) {
            layerSpec_ = other.layerSpec_;
            store_ = other.store_ ? other.store_->Clone()
                                  : std::make_unique<HashMapSpecStore>();
            backing_ = other.backing_;
        }
        return *this;
    }
    Layer(Layer&&) noexcept = default;
    Layer& operator=(Layer&&) noexcept = default;

    // --- Layer spec (root metadata at "/") ---

    Spec& GetLayerSpec() { return layerSpec_; }
    const Spec& GetLayerSpec() const { return layerSpec_; }

    // --- Spec access by path ---

    const Spec* GetSpec(const Path& path) const { return store_->GetSpec(path); }
    Spec* GetSpec(const Path& path) { return EnsureWritable().GetMutableSpec(path); }
    const Spec* GetSpec(const std::string& pathText) const;
    Spec* GetSpec(const std::string& pathText);

    void SetSpec(const Path& path, Spec spec) { EnsureWritable().SetSpec(path, std::move(spec)); }
    bool RemoveSpec(const Path& path) { return EnsureWritable().RemoveSpec(path); }
    bool HasSpec(const Path& path) const { return store_->HasSpec(path); }

    // All authored spec paths (excluding the layer spec at "/").
    std::vector<Path> GetSpecPaths() const { return store_->GetSpecPaths(); }

    // Iterate all specs. Preferred over GetSpecPaths for lazy stores.
    void ForEachSpec(const std::function<void(const Path&, const Spec&)>& fn) const {
        store_->ForEachSpec(fn);
    }

    // Child prim names under a given prim path (from index, O(1) lookup).
    const std::vector<Token>& GetChildNames(const Path& primPath) const {
        return store_->GetChildNames(primPath);
    }

    // --- Format-specific decode backing (for lazy Spec fields) ---
    //
    // USDC layers attach a backing here at parse time so Specs in
    // the store can lazily decode their authored fields on first
    // access. USDA / in-memory layers leave this null; their Specs
    // are populated eagerly at construction time.
    //
    // The shared_ptr is held by the Layer (owner) and shared with
    // every Spec in the store via raw pointer — Specs assume the
    // Layer outlives them, which is a contract the SpecStore
    // relationship already enforces.
    void SetDecodeBacking(std::shared_ptr<const DecodeBacking> backing) {
        backing_ = std::move(backing);
    }
    const DecodeBacking* GetDecodeBacking() const { return backing_.get(); }

    // Property names on a given prim path (from index, O(1) lookup).
    const std::vector<Token>& GetPropertyNames(const Path& primPath) const {
        return store_->GetPropertyNames(primPath);
    }

    const std::vector<Token>& GetAttributeNames(const Path& primPath) const {
        return store_->GetAttributeNames(primPath);
    }

    // --- Convenience accessors ---

    const Spec* GetPrimSpec(const Path& path) const;
    const Spec* GetAttributeSpec(const Path& path) const;
    const Spec* GetRelationshipSpec(const Path& path) const;

    Spec* GetMutablePrimSpec(const Path& path);
    Spec* GetMutableAttributeSpec(const Path& path);
    Spec* GetMutableRelationshipSpec(const Path& path);

private:
    friend const std::vector<Path>& detail::GetLayerChildPaths(
        const Layer& layer, const Path& primPath);
    friend const std::vector<Path>& detail::GetLayerArcOpinionPrimPaths(
        const Layer& layer);
    friend void detail::ForEachLayerChildIndex(
        const Layer& layer,
        const std::function<void(const Path&,
                                 const std::vector<Token>&,
                                 const std::vector<Path>&)>& fn);

    const Spec* GetSpecOfType(const Path& path, SpecType type) const;
    Spec* GetMutableSpecOfType(const Path& path, SpecType type);

    const std::vector<Path>& GetChildPathsInternal(
        const Path& primPath) const {
        return store_->GetChildPathsInternal(primPath);
    }
    const std::vector<Path>& GetArcOpinionPrimPathsInternal() const {
        return store_->GetArcOpinionPrimPathsInternal();
    }
    void ForEachChildIndexInternal(
        const std::function<void(const Path&,
                                 const std::vector<Token>&,
                                 const std::vector<Path>&)>& fn) const {
        store_->ForEachChildIndexInternal(fn);
    }

    // Upgrade a read-only store to writable (for write-on-crate-backed layers).
    SpecStore& EnsureWritable() {
        if (store_->IsReadOnly()) {
            auto writable = std::make_unique<HashMapSpecStore>();
            store_->ForEachSpec([&](const Path& p, const Spec& s) {
                writable->SetSpec(p, s);
            });
            store_ = std::move(writable);
        }
        return *store_;
    }

    Spec layerSpec_;
    std::unique_ptr<SpecStore> store_;
    std::shared_ptr<const DecodeBacking> backing_;
};

namespace detail {

inline const std::vector<Path>& GetLayerChildPaths(const Layer& layer,
                                                   const Path& primPath) {
    return layer.GetChildPathsInternal(primPath);
}

inline const std::vector<Path>& GetLayerArcOpinionPrimPaths(
    const Layer& layer) {
    return layer.GetArcOpinionPrimPathsInternal();
}

inline void ForEachLayerChildIndex(
    const Layer& layer,
    const std::function<void(const Path&,
                             const std::vector<Token>&,
                             const std::vector<Path>&)>& fn) {
    layer.ForEachChildIndexInternal(fn);
}

} // namespace detail

} // namespace nanousd
