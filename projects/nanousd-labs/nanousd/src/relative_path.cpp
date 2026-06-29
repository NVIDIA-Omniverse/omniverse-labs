// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanousd/relative_path.h"

#include "nanousd/path.h"
#include "unicode_identifier.h"

#include <sstream>

namespace nanousd {

// ============================================================
// Text helpers (ported from path.cpp's original Parser)
// ============================================================

namespace {

struct Parser {
    std::string_view src;
    std::size_t pos = 0;

    bool AtEnd() const  { return pos >= src.size(); }
    char Peek() const   { return AtEnd() ? '\0' : src[pos]; }
    char Advance()      { return src[pos++]; }
    bool Match(char c)  { if (Peek() == c) { ++pos; return true; } return false; }
    void SkipSpaces()   { while (!AtEnd() && (Peek() == ' ' || Peek() == '\t')) ++pos; }

    bool AtIdentifierStart(std::size_t* len = nullptr) const {
        return unicode::IsIdentifierStartAt(src, pos, len);
    }

    bool AtIdentifierContinue(std::size_t* len = nullptr) const {
        return unicode::IsIdentifierContinueAt(src, pos, len);
    }

    bool AtVariantContinue(std::size_t* len = nullptr) const {
        if (!AtEnd() && (Peek() == '|' || Peek() == '-')) {
            if (len) *len = 1;
            return true;
        }
        return AtIdentifierContinue(len);
    }

    bool ParseIdentifier(std::string& out) {
        std::size_t start = pos;
        std::size_t len = 0;
        if (AtEnd() || !AtIdentifierStart(&len)) return false;
        pos += len;
        while (!AtEnd() && AtIdentifierContinue(&len)) {
            pos += len;
        }
        out = std::string(src.substr(start, pos - start));
        return true;
    }

    bool ParsePrimName(std::string& out) { return ParseIdentifier(out); }

    bool ParsePropertyName(std::string& out) {
        std::string first;
        if (!ParseIdentifier(first)) return false;
        out = first;
        while (!AtEnd() && Peek() == ':') {
            std::size_t save = pos;
            ++pos;
            std::string next;
            if (!ParseIdentifier(next)) { pos = save; break; }
            out += ':';
            out += next;
        }
        return true;
    }

    bool ParsePropertyToken(Token& out) {
        std::string s;
        if (!ParsePropertyName(s)) return false;
        out = Token(s);
        return true;
    }

    bool ParseVariantName(std::string& out) {
        std::size_t start = pos;
        if (!AtEnd() && Peek() == '.') {
            ++pos;
            std::size_t len = 0;
            while (!AtEnd() && AtVariantContinue(&len)) pos += len;
            out = std::string(src.substr(start, pos - start));
            return true;
        }
        std::size_t len = 0;
        if (AtEnd() || !AtVariantContinue(&len))
            return false;
        while (!AtEnd() && AtVariantContinue(&len)) pos += len;
        out = std::string(src.substr(start, pos - start));
        return !out.empty();
    }

    bool ParseVariantSelection(VariantSelection& vs) {
        if (!Match('{')) return false;
        SkipSpaces();
        std::string setName;
        if (!ParseIdentifier(setName)) return false;
        SkipSpaces();
        if (!Match('=')) return false;
        SkipSpaces();
        std::string varName;
        std::size_t save = pos;
        if (!ParseVariantName(varName)) { varName.clear(); pos = save; }
        SkipSpaces();
        if (!Match('}')) return false;
        vs.setName = Token(setName);
        vs.variantName = Token(varName);
        return true;
    }

    void ParseVariantSelections(std::vector<VariantSelection>& sels) {
        while (!AtEnd() && Peek() == '{') {
            std::size_t save = pos;
            VariantSelection vs;
            if (!ParseVariantSelection(vs)) { pos = save; break; }
            sels.push_back(std::move(vs));
        }
    }
};

void RebuildText(RelativePath& p) {
    // Friend access via field set is awkward; build text inline.
    // (kept for diagnostics; no production code paths rely on it.)
}

} // anonymous

// ============================================================
// RelativePath
// ============================================================

RelativePath RelativePath::Parse(std::string_view text) {
    RelativePath result;
    if (text.empty()) return result;

    Parser parser{text, 0};

    // Absolute: starts with '/'.
    if (parser.Peek() == '/') {
        result.absolute_ = true;
        parser.Advance();
        if (parser.AtEnd()) {
            // "/" — absolute root. No elements, no property.
            result.valid_ = true;
            result.text_ = "/";
            return result;
        }

        std::string name;
        if (!parser.ParsePrimName(name)) return result;       // invalid
        PathElement elem;
        elem.name = Token(std::move(name));
        parser.ParseVariantSelections(elem.variantSelections);
        result.elements_.push_back(std::move(elem));

        for (;;) {
            if (!parser.AtEnd() && parser.Peek() == '{') {
                parser.ParseVariantSelections(result.elements_.back().variantSelections);
            }
            if (!parser.AtEnd() && parser.AtIdentifierStart()) {
                std::string childName;
                if (parser.ParsePrimName(childName)) {
                    PathElement childElem;
                    childElem.name = Token(std::move(childName));
                    result.elements_.push_back(std::move(childElem));
                    continue;
                }
            }
            if (!parser.AtEnd() && parser.Peek() == '/') {
                std::size_t save = parser.pos;
                parser.Advance();
                std::string childName;
                if (!parser.ParsePrimName(childName)) { parser.pos = save; break; }
                PathElement childElem;
                childElem.name = Token(std::move(childName));
                result.elements_.push_back(std::move(childElem));
                continue;
            }
            break;
        }

        if (!parser.AtEnd() && parser.Peek() == '.') {
            parser.Advance();
            if (!parser.ParsePropertyToken(result.propertyName_)) return result;
        }
        if (!parser.AtEnd()) return result;                   // trailing garbage
        result.valid_ = true;
        result.text_ = std::string(text);
        return result;
    }

    // Relative starting with '.' — could be "..", ".", or ".prop"
    if (parser.Peek() == '.') {
        parser.Advance();

        // ".." (parent relative)
        if (!parser.AtEnd() && parser.Peek() == '.') {
            parser.Advance();
            result.parentCount_ = 1;
            while (!parser.AtEnd() && parser.Peek() == '/') {
                std::size_t save = parser.pos;
                parser.Advance();
                if (!parser.AtEnd() && parser.Peek() == '.') {
                    std::size_t save2 = parser.pos;
                    parser.Advance();
                    if (!parser.AtEnd() && parser.Peek() == '.') {
                        parser.Advance();
                        result.parentCount_++;
                    } else {
                        parser.pos = save2;
                        break;
                    }
                } else {
                    parser.pos = save;
                    break;
                }
            }
            if (!parser.AtEnd() && parser.Peek() == '/') {
                parser.Advance();
                if (!parser.AtEnd() && parser.Peek() == '.') {
                    parser.Advance();
                    if (!parser.ParsePropertyToken(result.propertyName_)) return result;
                } else {
                    std::string name;
                    if (!parser.ParsePrimName(name)) return result;
                    PathElement elem;
                    elem.name = Token(std::move(name));
                    parser.ParseVariantSelections(elem.variantSelections);
                    result.elements_.push_back(std::move(elem));
                    while (!parser.AtEnd() && parser.Peek() == '/') {
                        std::size_t save = parser.pos;
                        parser.Advance();
                        std::string childName;
                        if (!parser.ParsePrimName(childName)) { parser.pos = save; break; }
                        PathElement childElem;
                        childElem.name = Token(std::move(childName));
                        parser.ParseVariantSelections(childElem.variantSelections);
                        result.elements_.push_back(std::move(childElem));
                    }
                    if (!parser.AtEnd() && parser.Peek() == '.') {
                        parser.Advance();
                        if (!parser.ParsePropertyToken(result.propertyName_)) return result;
                    }
                }
            }
            if (!parser.AtEnd()) return result;
            result.valid_ = true;
            result.text_ = std::string(text);
            return result;
        }

        // "." alone — reflexive
        if (parser.AtEnd()) {
            result.reflexive_ = true;
            result.valid_ = true;
            result.text_ = ".";
            return result;
        }

        // ".prop"
        if (!parser.ParsePropertyToken(result.propertyName_)) return result;
        if (!parser.AtEnd()) return result;
        result.valid_ = true;
        result.text_ = std::string(text);
        return result;
    }

    // Plain relative: starts with a prim name.
    std::string name;
    if (!parser.ParsePrimName(name)) return result;
    PathElement elem;
    elem.name = Token(std::move(name));
    parser.ParseVariantSelections(elem.variantSelections);
    result.elements_.push_back(std::move(elem));

    while (!parser.AtEnd() && parser.Peek() == '/') {
        std::size_t save = parser.pos;
        parser.Advance();
        std::string childName;
        if (!parser.ParsePrimName(childName)) { parser.pos = save; break; }
        PathElement childElem;
        childElem.name = Token(std::move(childName));
        parser.ParseVariantSelections(childElem.variantSelections);
        result.elements_.push_back(std::move(childElem));
    }
    if (!parser.AtEnd() && parser.Peek() == '.') {
        parser.Advance();
        if (!parser.ParsePropertyToken(result.propertyName_)) return result;
    }
    if (!parser.AtEnd()) return result;
    result.valid_ = true;
    result.text_ = std::string(text);
    return result;
}

Path RelativePath::Anchor(const Path& base) const {
    if (!valid_) return Path();
    if (reflexive_) return base;

    // If `this` is already absolute, the base is irrelevant — just
    // walk the prefixes through the pool starting from root.
    if (absolute_) {
        Path p = Path::AbsoluteRoot();
        for (const auto& e : elements_) {
            p = p.AppendChild(e.name);
            if (p.IsEmpty()) return Path();
            for (const auto& v : e.variantSelections) {
                p = p.AppendVariantSelection(v.setName, v.variantName);
                if (p.IsEmpty()) return Path();
            }
        }
        if (!propertyName_.IsEmpty()) {
            p = p.AppendProperty(propertyName_);
        }
        return p;
    }

    // Relative: walk up parentCount_ from base, then descend.
    if (!base.IsAbsolute()) return Path();
    Path cur = base;
    for (int i = 0; i < parentCount_; ++i) {
        cur = cur.GetParentPath();
        if (cur.IsEmpty()) return Path();
    }
    for (const auto& e : elements_) {
        cur = cur.AppendChild(e.name);
        if (cur.IsEmpty()) return Path();
        for (const auto& v : e.variantSelections) {
            cur = cur.AppendVariantSelection(v.setName, v.variantName);
            if (cur.IsEmpty()) return Path();
        }
    }
    if (!propertyName_.IsEmpty()) {
        cur = cur.AppendProperty(propertyName_);
    }
    return cur;
}

} // namespace nanousd
