// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace nanousd {

// List operations per spec Section 6.6.3.
// A ListOp is either "explicit" or "composable" (append/prepend/delete).
// Template parameter T must support equality and hashing.

template <typename T>
class ListOp {
public:
    // Construct the default composable list op I' (all sequences empty).
    ListOp() = default;

    // --- Factory methods ---

    static ListOp CreateExplicit(std::vector<T> items) {
        ListOp op;
        op.explicit_ = std::move(items);
        op.isExplicit_ = true;
        return op;
    }

    static ListOp CreateComposable(std::vector<T> prepended = {},
                                   std::vector<T> appended = {},
                                   std::vector<T> deleted = {}) {
        ListOp op;
        op.prepended_ = std::move(prepended);
        op.appended_ = std::move(appended);
        op.deleted_ = std::move(deleted);
        op.isExplicit_ = false;
        return op;
    }

    // --- Queries ---

    bool IsExplicit() const { return isExplicit_; }

    const std::vector<T>& GetExplicitItems() const { return explicit_; }
    const std::vector<T>& GetPrependedItems() const { return prepended_; }
    const std::vector<T>& GetAppendedItems() const { return appended_; }
    const std::vector<T>& GetDeletedItems() const { return deleted_; }

    // --- Mutators ---

    void SetExplicitItems(std::vector<T> v) { explicit_ = std::move(v); isExplicit_ = true; }
    void SetPrependedItems(std::vector<T> v) { prepended_ = std::move(v); isExplicit_ = false; }
    void SetAppendedItems(std::vector<T> v) { appended_ = std::move(v); isExplicit_ = false; }
    void SetDeletedItems(std::vector<T> v) { deleted_ = std::move(v); isExplicit_ = false; }

    // --- Iteration (spec 6.6.3.4) ---
    // For explicit: yields the explicit sequence.
    // For composable: yields prepend items not in append, followed by append items.

    std::vector<T> GetItems() const {
        if (isExplicit_) {
            return explicit_;
        }
        std::vector<T> result;
        std::unordered_set<T> appendSet(appended_.begin(), appended_.end());
        for (const auto& p : prepended_) {
            if (appendSet.find(p) == appendSet.end()) {
                result.push_back(p);
            }
        }
        result.insert(result.end(), appended_.begin(), appended_.end());
        return result;
    }

    // --- Combining (spec 6.6.3.6) ---
    // *this is the stronger operation, weaker is the weaker.
    // Returns the combined result.

    ListOp Combine(const ListOp& weaker) const {
        // S explicit + any W => S
        if (isExplicit_) {
            return *this;
        }

        // S composable + E explicit => new explicit
        if (weaker.isExplicit_) {
            return CombineComposableWithExplicit(weaker);
        }

        // S composable + C composable => new composable
        return CombineComposableWithComposable(weaker);
    }

    // --- Reducing (spec 6.6.3.8) ---
    // Remove elements that appear in multiple sequences to canonical form.

    ListOp Reduced() const {
        if (isExplicit_) return *this;

        ListOp result;
        result.isExplicit_ = false;
        result.appended_ = appended_;

        std::unordered_set<T> appendSet(appended_.begin(), appended_.end());

        for (const auto& p : prepended_) {
            if (appendSet.find(p) == appendSet.end()) {
                result.prepended_.push_back(p);
            }
        }

        std::unordered_set<T> prependSet(result.prepended_.begin(),
                                          result.prepended_.end());
        for (const auto& d : deleted_) {
            if (appendSet.find(d) == appendSet.end() &&
                prependSet.find(d) == prependSet.end()) {
                result.deleted_.push_back(d);
            }
        }

        return result;
    }

    // --- Equality (spec 6.6.3.5) ---

    bool operator==(const ListOp& other) const {
        if (isExplicit_ != other.isExplicit_) return false;
        if (isExplicit_) return explicit_ == other.explicit_;
        return appended_ == other.appended_ &&
               prepended_ == other.prepended_ &&
               deleted_ == other.deleted_;
    }

    bool operator!=(const ListOp& other) const { return !(*this == other); }

private:
    ListOp CombineComposableWithExplicit(const ListOp& E) const {
        // S composable + E explicit:
        // prepend items not in S.append,
        // then E.explicit items not in S.append/S.delete/S.prepend,
        // then S.append items
        ListOp result;
        result.isExplicit_ = true;

        std::unordered_set<T> sAppend(appended_.begin(), appended_.end());
        std::unordered_set<T> sDelete(deleted_.begin(), deleted_.end());
        std::unordered_set<T> sPrepend(prepended_.begin(), prepended_.end());

        // Prepend items not in append
        for (const auto& p : prepended_) {
            if (sAppend.find(p) == sAppend.end()) {
                result.explicit_.push_back(p);
            }
        }

        // Deduplicate against what we've already added
        std::unordered_set<T> seen(result.explicit_.begin(), result.explicit_.end());

        // E.explicit items not in S.append, S.delete, or S.prepend
        for (const auto& e : E.explicit_) {
            if (sAppend.find(e) == sAppend.end() &&
                sDelete.find(e) == sDelete.end() &&
                sPrepend.find(e) == sPrepend.end() &&
                seen.find(e) == seen.end()) {
                result.explicit_.push_back(e);
                seen.insert(e);
            }
        }

        // S.append items
        for (const auto& a : appended_) {
            if (seen.find(a) == seen.end()) {
                result.explicit_.push_back(a);
                seen.insert(a);
            }
        }

        return result;
    }

    ListOp CombineComposableWithComposable(const ListOp& C) const {
        // S composable + C composable => new composable
        ListOp result;
        result.isExplicit_ = false;

        std::unordered_set<T> sAppend(appended_.begin(), appended_.end());
        std::unordered_set<T> sDelete(deleted_.begin(), deleted_.end());
        std::unordered_set<T> sPrepend(prepended_.begin(), prepended_.end());

        // deleted: C.delete not in S.prepend or S.append,
        //          then S.delete not in S.prepend, S.append, or C.delete
        std::unordered_set<T> seen;
        for (const auto& d : C.deleted_) {
            if (sPrepend.find(d) == sPrepend.end() &&
                sAppend.find(d) == sAppend.end()) {
                result.deleted_.push_back(d);
                seen.insert(d);
            }
        }
        for (const auto& d : deleted_) {
            if (sPrepend.find(d) == sPrepend.end() &&
                sAppend.find(d) == sAppend.end() &&
                seen.find(d) == seen.end()) {
                result.deleted_.push_back(d);
                seen.insert(d);
            }
        }

        // prepended: S.prepend not in S.append,
        //            then C.prepend not in S.append, S.delete, or S.prepend
        for (const auto& p : prepended_) {
            if (sAppend.find(p) == sAppend.end()) {
                result.prepended_.push_back(p);
            }
        }
        std::unordered_set<T> resultPrepend(result.prepended_.begin(),
                                             result.prepended_.end());
        for (const auto& p : C.prepended_) {
            if (sAppend.find(p) == sAppend.end() &&
                sDelete.find(p) == sDelete.end() &&
                sPrepend.find(p) == sPrepend.end() &&
                resultPrepend.find(p) == resultPrepend.end()) {
                result.prepended_.push_back(p);
            }
        }

        // appended: C.append not in S.append, S.delete, or S.prepend,
        //           then S.append
        std::unordered_set<T> resultAppend;
        for (const auto& a : C.appended_) {
            if (sAppend.find(a) == sAppend.end() &&
                sDelete.find(a) == sDelete.end() &&
                sPrepend.find(a) == sPrepend.end()) {
                result.appended_.push_back(a);
                resultAppend.insert(a);
            }
        }
        for (const auto& a : appended_) {
            if (resultAppend.find(a) == resultAppend.end()) {
                result.appended_.push_back(a);
            }
        }

        return result;
    }

    bool isExplicit_ = false;
    std::vector<T> explicit_;
    std::vector<T> prepended_;
    std::vector<T> appended_;
    std::vector<T> deleted_;
};

// Convenience aliases per spec (Section 6.6.3 supported types)
using ListOpInt    = ListOp<int32_t>;
using ListOpInt64  = ListOp<int64_t>;
using ListOpUInt   = ListOp<uint32_t>;
using ListOpUInt64 = ListOp<uint64_t>;
using ListOpToken  = ListOp<std::string>;
using ListOpString = ListOp<std::string>;

} // namespace nanousd
