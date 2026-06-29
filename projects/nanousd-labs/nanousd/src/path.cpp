// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanousd/path.h"

#include "nanousd/relative_path.h"
#include "path_pool.h"
#include "unicode_identifier.h"

#include <algorithm>
#include <sstream>
#include <vector>

namespace nanousd {

// ============================================================
// Path
// ============================================================

Path::Path() : data_(nullptr) {}

Path Path::AbsoluteRoot() {
    return Path(PathPool::Instance().Root());
}

bool Path::IsAbsoluteRoot() const {
    return data_ != nullptr && data_->parent == nullptr;
}

bool Path::HasProperty() const {
    return data_ != nullptr && data_->kind == PathStepKind::Property;
}

bool Path::IsPrimPath() const {
    if (!data_) return false;
    return !HasProperty();
}

bool Path::HasVariantSelections() const {
    return data_ && data_->hasVariantSelection;
}

const Token& Path::GetPropertyName() const {
    static const Token empty;
    if (HasProperty()) return data_->step;
    return empty;
}

std::vector<std::string_view> Path::GetPropertyNamespaces() const {
    std::vector<std::string_view> result;
    if (!HasProperty()) return result;
    std::string_view sv = data_->step.GetString();
    std::size_t start = 0;
    while (start < sv.size()) {
        std::size_t colon = sv.find(':', start);
        if (colon == std::string_view::npos) {
            result.push_back(sv.substr(start));
            break;
        }
        result.push_back(sv.substr(start, colon - start));
        start = colon + 1;
    }
    return result;
}

namespace {

// Build the historical PathElement view by walking the chain.
std::vector<PathElement> BuildElementsFromChain(const PathData* data) {
    std::vector<PathElement> result;
    if (!data || data->parent == nullptr) return result;
    result.reserve(data->depth);

    std::vector<const PathData*> chain;
    chain.reserve(data->depth);
    for (const PathData* d = data; d != nullptr; d = d->parent) {
        if (d->parent == nullptr) break;
        if (d->kind == PathStepKind::Property) continue;
        chain.push_back(d);
    }
    std::reverse(chain.begin(), chain.end());

    for (const PathData* d : chain) {
        if (d->kind == PathStepKind::Child) {
            PathElement elem;
            elem.name = d->step;
            result.push_back(std::move(elem));
        } else {
            if (result.empty()) continue;
            result.back().variantSelections.push_back(
                {d->step, d->variantValue});
        }
    }
    return result;
}

std::string BuildTextFromChain(const PathData* data) {
    if (!data) return std::string();
    if (data->parent == nullptr) return "/";

    std::vector<const PathData*> chain;
    chain.reserve(data->depth);
    for (const PathData* d = data; d != nullptr; d = d->parent) {
        if (d->parent == nullptr) break;
        chain.push_back(d);
    }
    std::reverse(chain.begin(), chain.end());

    std::ostringstream ss;
    for (const PathData* d : chain) {
        switch (d->kind) {
        case PathStepKind::Child:
            ss << '/' << d->step.GetText();
            break;
        case PathStepKind::Property:
            ss << '.' << d->step.GetText();
            break;
        case PathStepKind::VariantSelection:
            ss << '{' << d->step.GetText() << '='
               << d->variantValue.GetText() << '}';
            break;
        }
    }
    return ss.str();
}

bool IsAsciiDigit(uint32_t cp) {
    return cp >= '0' && cp <= '9';
}

bool IsAsciiUpper(uint32_t cp) {
    return cp >= 'A' && cp <= 'Z';
}

bool IsAsciiLower(uint32_t cp) {
    return cp >= 'a' && cp <= 'z';
}

bool IsAsciiAlpha(uint32_t cp) {
    return IsAsciiUpper(cp) || IsAsciiLower(cp);
}

uint32_t AsciiFoldLower(uint32_t cp) {
    return IsAsciiUpper(cp) ? cp + ('a' - 'A') : cp;
}

int ElementCodePointRank(uint32_t cp) {
    if (cp == '_') return 0;
    if (IsAsciiDigit(cp)) return 1;
    if (IsAsciiAlpha(cp)) return 2;
    return 3;
}

struct DecodedCodePoint {
    uint32_t codePoint = 0;
    size_t length = 0;
};

DecodedCodePoint DecodeForOrdering(std::string_view text, size_t pos) {
    auto decoded = unicode::DecodeUtf8(text, pos);
    if (decoded.valid && decoded.length > 0) {
        return {decoded.codePoint, decoded.length};
    }
    return {static_cast<unsigned char>(text[pos]), 1};
}

struct DigitGroup {
    size_t begin = 0;
    size_t end = 0;
    size_t significantBegin = 0;
    size_t significantEnd = 0;
    size_t leadingZeros = 0;
};

DigitGroup ReadAsciiDigitGroup(std::string_view text, size_t pos) {
    DigitGroup group;
    group.begin = pos;
    size_t p = pos;
    while (p < text.size() && text[p] >= '0' && text[p] <= '9') {
        ++p;
    }
    group.end = p;

    size_t significant = pos;
    while (significant < p && text[significant] == '0') {
        ++significant;
    }
    group.leadingZeros = significant - pos;
    group.significantBegin = significant;
    group.significantEnd = p;
    return group;
}

int CompareAsciiDigitGroups(std::string_view lhs,
                            const DigitGroup& lhsGroup,
                            std::string_view rhs,
                            const DigitGroup& rhsGroup) {
    const size_t lhsDigits =
        lhsGroup.significantEnd - lhsGroup.significantBegin;
    const size_t rhsDigits =
        rhsGroup.significantEnd - rhsGroup.significantBegin;

    if (lhsDigits != rhsDigits) {
        return lhsDigits < rhsDigits ? -1 : 1;
    }
    for (size_t i = 0; i < lhsDigits; ++i) {
        const char a = lhs[lhsGroup.significantBegin + i];
        const char b = rhs[rhsGroup.significantBegin + i];
        if (a != b) return a < b ? -1 : 1;
    }
    if (lhsGroup.leadingZeros != rhsGroup.leadingZeros) {
        return lhsGroup.leadingZeros < rhsGroup.leadingZeros ? -1 : 1;
    }
    return 0;
}

std::vector<const PathData*> BuildStepChain(const PathData* data) {
    std::vector<const PathData*> chain;
    if (!data || data->parent == nullptr) return chain;
    chain.reserve(data->depth);
    for (const PathData* d = data; d != nullptr; d = d->parent) {
        if (d->parent == nullptr) break;
        chain.push_back(d);
    }
    std::reverse(chain.begin(), chain.end());
    return chain;
}

int PathStepKindOrder(PathStepKind kind) {
    switch (kind) {
    case PathStepKind::Property:
        return 0;
    case PathStepKind::Child:
        return 1;
    case PathStepKind::VariantSelection:
        return 2;
    }
    return 0;
}

} // anonymous

// Walk the chain root-to-leaf and produce the historical PathElement
// view (prim names with their attached variant selections). Property
// step is excluded — callers use GetPropertyName() for that.
//
// Hot paths (RemapPath in compose, GetText for sorting) call this
// repeatedly on the same Path object; each PathData lazily caches
// the materialised vector via a CAS-protected pointer so subsequent
// reads are a pointer chase. Returned by value at the API boundary
// to keep call sites safe even when their elements_caches haven't
// been built yet — but the value is constructed once per unique
// path lifetime.
std::vector<PathElement> Path::GetElements() const {
    return GetCachedPathElements(*this);  // copy out from the cached reference
}

const std::vector<PathElement>& GetCachedPathElements(const Path& p) {
    // Process-singleton empty vector for the root/empty case so the
    // return-by-reference contract holds without per-call allocation.
    static const std::vector<PathElement> kEmpty;
    const PathData* data = p.GetData_();
    if (!data || data->parent == nullptr) return kEmpty;
    auto* cached = data->elementsCache.load(std::memory_order_acquire);
    if (!cached) {
        auto* fresh = new std::vector<PathElement>(BuildElementsFromChain(data));
        const std::vector<PathElement>* expected = nullptr;
        if (data->elementsCache.compare_exchange_strong(
                expected, fresh,
                std::memory_order_release, std::memory_order_acquire)) {
            cached = fresh;
        } else {
            // Lost the race; another thread cached first. Use that.
            delete fresh;
            cached = expected;
        }
    }
    return *cached;
}

std::string Path::GetText() const {
    if (!data_) return std::string();
    auto* cached = data_->textCache.load(std::memory_order_acquire);
    if (!cached) {
        auto* fresh = new std::string(BuildTextFromChain(data_));
        const std::string* expected = nullptr;
        if (data_->textCache.compare_exchange_strong(
                expected, fresh,
                std::memory_order_release, std::memory_order_acquire)) {
            cached = fresh;
        } else {
            delete fresh;
            cached = expected;
        }
    }
    return *cached;
}

int ComparePathElementNames(std::string_view lhs, std::string_view rhs) {
    size_t lhsPos = 0;
    size_t rhsPos = 0;
    int deferredTieBreak = 0;

    while (lhsPos < lhs.size() && rhsPos < rhs.size()) {
        const auto lhsDecoded = DecodeForOrdering(lhs, lhsPos);
        const auto rhsDecoded = DecodeForOrdering(rhs, rhsPos);
        const uint32_t lhsCp = lhsDecoded.codePoint;
        const uint32_t rhsCp = rhsDecoded.codePoint;

        if (IsAsciiDigit(lhsCp) && IsAsciiDigit(rhsCp)) {
            const DigitGroup lhsGroup = ReadAsciiDigitGroup(lhs, lhsPos);
            const DigitGroup rhsGroup = ReadAsciiDigitGroup(rhs, rhsPos);
            const int cmp =
                CompareAsciiDigitGroups(lhs, lhsGroup, rhs, rhsGroup);
            if (cmp != 0) {
                // Leading-zero differences are secondary: keep walking so
                // a later non-equal element can decide the order first.
                const size_t lhsDigits =
                    lhsGroup.significantEnd - lhsGroup.significantBegin;
                const size_t rhsDigits =
                    rhsGroup.significantEnd - rhsGroup.significantBegin;
                const bool sameValue =
                    lhsDigits == rhsDigits &&
                    lhs.compare(lhsGroup.significantBegin, lhsDigits,
                                rhs, rhsGroup.significantBegin,
                                rhsDigits) == 0;
                if (!sameValue) return cmp;
                if (deferredTieBreak == 0) deferredTieBreak = cmp;
            }
            lhsPos = lhsGroup.end;
            rhsPos = rhsGroup.end;
            continue;
        }

        const int lhsRank = ElementCodePointRank(lhsCp);
        const int rhsRank = ElementCodePointRank(rhsCp);
        if (lhsRank != rhsRank) return lhsRank < rhsRank ? -1 : 1;

        if (IsAsciiAlpha(lhsCp) && IsAsciiAlpha(rhsCp)) {
            const uint32_t lhsFolded = AsciiFoldLower(lhsCp);
            const uint32_t rhsFolded = AsciiFoldLower(rhsCp);
            if (lhsFolded != rhsFolded) {
                return lhsFolded < rhsFolded ? -1 : 1;
            }
            if (lhsCp != rhsCp && deferredTieBreak == 0) {
                // Case-only differences are secondary to later primary
                // differences in the same element.
                deferredTieBreak = IsAsciiUpper(lhsCp) ? -1 : 1;
            }
        } else if (lhsCp != rhsCp) {
            return lhsCp < rhsCp ? -1 : 1;
        }

        lhsPos += lhsDecoded.length;
        rhsPos += rhsDecoded.length;
    }

    if (lhsPos != lhs.size()) return 1;
    if (rhsPos != rhs.size()) return -1;
    return deferredTieBreak;
}

int ComparePathsByPathElementOrder(const Path& lhs, const Path& rhs) {
    if (lhs == rhs) return 0;
    if (lhs.IsEmpty()) return rhs.IsEmpty() ? 0 : -1;
    if (rhs.IsEmpty()) return 1;

    const auto lhsSteps = BuildStepChain(lhs.GetData_());
    const auto rhsSteps = BuildStepChain(rhs.GetData_());
    const size_t common = std::min(lhsSteps.size(), rhsSteps.size());
    for (size_t i = 0; i < common; ++i) {
        const PathData* a = lhsSteps[i];
        const PathData* b = rhsSteps[i];
        if (a->kind != b->kind) {
            const int aOrder = PathStepKindOrder(a->kind);
            const int bOrder = PathStepKindOrder(b->kind);
            if (aOrder != bOrder) return aOrder < bOrder ? -1 : 1;
        }

        const int nameCmp =
            ComparePathElementNames(a->step.GetString(), b->step.GetString());
        if (nameCmp != 0) return nameCmp;

        if (a->kind == PathStepKind::VariantSelection) {
            const int valueCmp = ComparePathElementNames(
                a->variantValue.GetString(), b->variantValue.GetString());
            if (valueCmp != 0) return valueCmp;
        }
    }

    if (lhsSteps.size() == rhsSteps.size()) return 0;
    return lhsSteps.size() < rhsSteps.size() ? -1 : 1;
}

Path Path::GetParentPath() const {
    if (!data_) return Path();
    if (data_->parent == nullptr) return Path();      // root has no parent

    // For Property and Child steps, the parent is just data_->parent.
    // For a VariantSelection step, the variants are attached to a
    // preceding Child prim (per the historical PathElement view), so
    // parent walks past all trailing variant selections AND the prim
    // they're attached to, mirroring the old API where
    // GetParentPath(/Model{x=y}) returned `/` rather than `/Model`.
    if (data_->kind != PathStepKind::VariantSelection) {
        return Path(data_->parent);
    }
    const PathData* d = data_;
    while (d != nullptr && d->kind == PathStepKind::VariantSelection) {
        d = d->parent;
    }
    if (d == nullptr || d->parent == nullptr) return Path();
    return Path(d->parent);
}

Path Path::GetPrimPath() const {
    if (!HasProperty()) return *this;
    return Path(data_->parent);
}

Path Path::AppendChild(const Token& childName) const {
    if (!data_) return Path();
    if (HasProperty()) return Path();
    auto* d = PathPool::Instance().Intern(
        data_, childName, Token(), PathStepKind::Child);
    return Path(d);
}

Path Path::AppendProperty(const Token& propName) const {
    if (!data_) return Path();
    if (HasProperty()) return Path();
    if (data_->parent == nullptr) return Path();      // root has no property
    auto* d = PathPool::Instance().Intern(
        data_, propName, Token(), PathStepKind::Property);
    return Path(d);
}

Path Path::AppendVariantSelection(const Token& setName,
                                   const Token& variantName) const {
    if (!data_) return Path();
    if (HasProperty()) return Path();
    if (data_->parent == nullptr) return Path();      // not on root
    auto* d = PathPool::Instance().Intern(
        data_, setName, variantName, PathStepKind::VariantSelection);
    return Path(d);
}

Token Path::GetName() const {
    if (!data_) return Token();
    if (data_->parent == nullptr) return Token();        // root
    // Property: the step *is* the property name.
    if (data_->kind == PathStepKind::Property) return data_->step;
    // Otherwise: walk past trailing variant selections to the most
    // recent Child step — variant selections aren't "names" in the
    // historical API sense; they're annotations on a prim element.
    for (const PathData* d = data_; d != nullptr; d = d->parent) {
        if (d->kind == PathStepKind::Child && d->parent != nullptr) {
            return d->step;
        }
    }
    return Token();
}

Path Path::Parse(std::string_view text) {
    if (text.empty()) return Path();
    // Reuse RelativePath's parser (it handles all textual shapes —
    // absolute, parent-relative, reflexive, plain). Reject anything
    // that didn't parse as absolute.
    RelativePath rel = RelativePath::Parse(text);
    if (!rel.IsValid() || !rel.IsAlreadyAbsolute()) return Path();
    return rel.Anchor(AbsoluteRoot());
}

bool Path::operator<(const Path& o) const {
    return ComparePathsByPathElementOrder(*this, o) < 0;
}

} // namespace nanousd
