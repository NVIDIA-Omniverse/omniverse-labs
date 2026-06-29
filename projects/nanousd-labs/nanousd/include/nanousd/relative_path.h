// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"
#include "types.h"

#include <string>
#include <string_view>
#include <vector>

namespace nanousd {

class Path;

// Variant selection {setName=variantName} as it appears inside a
// path expression. Variant tokens may also encode this as a single
// canonical "{set=variant}" form — see Token usage in PathStep.
struct VariantSelection {
    Token setName;
    Token variantName;

    bool operator==(const VariantSelection& o) const {
        return setName == o.setName && variantName == o.variantName;
    }
    bool operator!=(const VariantSelection& o) const { return !(*this == o); }
};

// One element in the historical (flat) path view: a prim name plus
// any variant selections attached to that prim.
struct PathElement {
    Token name;
    std::vector<VariantSelection> variantSelections;
};

// A relative path authoring — the form that appears in USDA text
// and in some external inputs (e.g. relative reference targets).
// Always transient: parsed at the API boundary, anchored against an
// absolute base to produce an interned absolute Path, and dropped.
//
// Not interned, not stored in long-lived containers. If you find
// yourself wanting to hold a RelativePath as scene state, anchor it
// first.
class NANOUSD_CORE_API RelativePath {
public:
    RelativePath() = default;

    // Parse text that may use ".", "..", "../foo", ".prop",
    // "../foo/bar.attr", "foo/bar", etc. Absolute text ("/foo")
    // also parses successfully and produces a RelativePath whose
    // Anchor() result is independent of the supplied base. Returns
    // an invalid RelativePath on parse failure.
    static RelativePath Parse(std::string_view text);

    // Anchor this relative path against an absolute `base` to
    // produce an interned absolute `Path`. Returns an empty Path on:
    //   - this RelativePath being invalid
    //   - the relative form going above the absolute root (more
    //     ".." steps than `base` has elements)
    //   - `base` not being a valid absolute Path
    Path Anchor(const Path& base) const;

    // --- Queries --------------------------------------------------

    bool IsValid() const          { return valid_; }
    bool IsReflexive() const      { return reflexive_; }    // "."
    bool IsAlreadyAbsolute() const { return absolute_; }    // text starts with '/'
    int  GetParentCount() const   { return parentCount_; }
    const std::vector<PathElement>& GetElements() const { return elements_; }
    const Token& GetPropertyName() const { return propertyName_; }
    const std::string& GetText() const   { return text_; }

private:
    friend class Path;

    bool valid_ = false;
    bool absolute_ = false;
    bool reflexive_ = false;
    int  parentCount_ = 0;
    std::vector<PathElement> elements_;
    Token propertyName_;
    std::string text_;       // canonical text form, kept for diagnostics
};

} // namespace nanousd
