// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanousd/usda_parser.h"
#include "nanousd/resource.h"
#include "unicode_identifier.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <stdexcept>

namespace nanousd {

// ============================================================
// Lexer / Scanner
// ============================================================

class UsdaParser {
public:
    explicit UsdaParser(std::string_view text)
        : text_(text), pos_(0), line_(1), col_(1) {}

    UsdaParseResult Parse() {
        UsdaParseResult result;
        try {
            ParseLayer();
            result.success = true;
            result.layer = std::move(layer_);
        } catch (const ParseError& e) {
            result.success = false;
            result.error = e.msg;
            result.line = e.line;
            result.column = e.col;
        }
        return result;
    }

private:
    struct ParseError {
        std::string msg;
        int line;
        int col;
    };

    [[noreturn]] void Error(const std::string& msg) {
        throw ParseError{msg, line_, col_};
    }

    // --- Character access ---

    bool AtEnd() const { return pos_ >= text_.size(); }

    char Peek() const {
        return AtEnd() ? '\0' : text_[pos_];
    }

    char PeekAt(size_t offset) const {
        size_t p = pos_ + offset;
        return p < text_.size() ? text_[p] : '\0';
    }

    char Advance() {
        char c = text_[pos_++];
        if (c == '\n') { line_++; col_ = 1; }
        else { col_++; }
        return c;
    }

    void AdvanceBytes(size_t count) {
        for (size_t i = 0; i < count; ++i) Advance();
    }

    bool Match(char c) {
        if (Peek() == c) { Advance(); return true; }
        return false;
    }

    void Expect(char c) {
        if (!Match(c)) {
            Error(std::string("Expected '") + c + "', got '" + (AtEnd() ? "EOF" : std::string(1, Peek())) + "'");
        }
    }

    bool MatchStr(std::string_view s) {
        if (pos_ + s.size() > text_.size()) return false;
        if (text_.substr(pos_, s.size()) == s) {
            for (size_t i = 0; i < s.size(); i++) Advance();
            return true;
        }
        return false;
    }

    // --- Whitespace and comments ---

    void SkipWhitespaceAndComments() {
        while (!AtEnd()) {
            char c = Peek();
            // Semicolons act as statement separators (treated like whitespace)
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == ';') {
                Advance();
            } else if (c == '#') {
                SkipLineComment();
            } else if (c == '/' && PeekAt(1) == '/') {
                SkipLineComment();
            } else if (c == '/' && PeekAt(1) == '*') {
                SkipBlockComment();
            } else {
                break;
            }
        }
    }

    // Consume only same-line padding: spaces, tabs, line comments
    // (Python '#' and C++ '//'), and C++ /* ... */ block comments that
    // contain no newline. Stops at newlines, semicolons, or any other
    // character. Spec: matches (SinglelinePadding)* in §16.2. Used to
    // check a prim-body item's terminator without accidentally
    // swallowing the StatementSeparator (Crlf or Semicolon).
    void SkipSinglelinePadding() {
        while (!AtEnd()) {
            char c = Peek();
            if (c == ' ' || c == '\t') {
                Advance();
            } else if (c == '#') {
                SkipLineComment();  // stops at '\n'
            } else if (c == '/' && PeekAt(1) == '/') {
                SkipLineComment();
            } else if (c == '/' && PeekAt(1) == '*') {
                // A /* ... */ block comment MAY span lines. If so the
                // newline inside counts as "not single-line", but the
                // spec's SinglelinePadding allows comments to be part
                // of single-line padding (the definition of Comment
                // includes the line-comment's internal newline).
                // Conservative: consume if it's a same-line block
                // comment (no internal '\n'). Otherwise leave it so
                // the terminator check can see the implicit newline.
                size_t lookahead = 2;
                bool containsNewline = false;
                while (pos_ + lookahead + 1 < text_.size() &&
                       !(text_[pos_ + lookahead] == '*' &&
                         text_[pos_ + lookahead + 1] == '/')) {
                    if (text_[pos_ + lookahead] == '\n') {
                        containsNewline = true;
                        break;
                    }
                    ++lookahead;
                }
                if (containsNewline) return;
                SkipBlockComment();
            } else {
                break;
            }
        }
    }

    // Spec §16.2.17 (PrimItem production): each item in a prim body
    // must be followed by a StatementSeparator — Crlf for PrimSpec /
    // VariantSetStatement, Crlf or Semicolon for PropertySpec /
    // ChildOrPropertyOrderStatement. Allowing `def Foo "x" { int v = 0 }`
    // on one line (historical nanousd laxity) drops an item's mandatory
    // terminator and lets spec-nonconforming content round-trip.
    //
    // Called in ParsePrimBody with `startLine` captured immediately
    // before the PrimItem's parse entrypoint. Because the parser's
    // Peek/Match/Expect helpers all call SkipWhitespaceAndComments
    // internally — a longstanding laxity that would be invasive to
    // undo here — the trailing StatementSeparator of an item often
    // gets swallowed mid-parse as the parser advances through the
    // next item's opening padding. If `line_` has advanced past
    // `startLine` during the item's parse, a newline was crossed at
    // some point, which is what the spec's StatementSeparator
    // requires. That implicit crossing satisfies the terminator check.
    //
    // Accepts end-of-body (`}`) and EOF without requiring a terminator,
    // since those aren't positions where a separator is productive;
    // the PrimContents production terminates on the outer `}` and the
    // grammar only mandates a separator BETWEEN items. In practice this
    // means an empty `{}` needs no terminator, and the parser tolerates
    // any number of blank/comment lines following a valid separator.
    void RequirePrimItemTerminator(bool allowSemicolon,
                                    const char* itemKindForError,
                                    int startLine) {
        if (line_ > startLine) return;  // newline already crossed implicitly
        SkipSinglelinePadding();
        if (AtEnd()) return;  // EOF after last item is tolerated
        char c = Peek();
        if (c == '\n' || c == '\r') {
            SkipWhitespaceAndComments();  // fold any further padding/separators
            return;
        }
        if (allowSemicolon && c == ';') {
            Advance();
            SkipWhitespaceAndComments();
            return;
        }
        // Note: no special-case for '}' here — the PrimItem's terminator
        // is mandatory per spec. An item directly abutting the closing
        // `}` of its containing body is exactly the `{ int v = 0 }`
        // on-one-line form the spec disallows.
        std::string msg = "Expected newline";
        if (allowSemicolon) msg += " or semicolon";
        msg += " after ";
        msg += itemKindForError;
        msg += " in prim body";
        Error(msg);
    }

    static bool IsCrLf(char c) {
        return c == '\r' || c == '\n';
    }

    static bool IsUsdaUtf8Character(uint32_t cp) {
        if (cp >= 0x0020u && cp <= 0x007Eu) return true;
        if (cp >= 0x00A0u && cp <= 0xD7FFu) return true;
        if (cp >= 0xF900u && cp <= 0xFDCFu) return true;
        if (cp >= 0xFDF0u && cp <= 0xFFEFu) return true;
        if (cp >= 0x10000u && cp <= 0xEFFFDu &&
            (cp & 0xFFFFu) <= 0xFFFDu) {
            return true;
        }
        return false;
    }

    void ValidateAndAdvanceUtf8Character(bool allowCrLf,
                                         const char* context) {
        if (AtEnd()) Error(std::string("Unexpected end in ") + context);
        if (IsCrLf(Peek())) {
            if (!allowCrLf) {
                Error(std::string("CR/LF not allowed in ") + context);
            }
            Advance();
            return;
        }

        auto decoded = unicode::DecodeUtf8(text_, pos_);
        if (!decoded.valid || decoded.length == 0) {
            Error(std::string("Invalid UTF-8 in ") + context);
        }
        if (!IsUsdaUtf8Character(decoded.codePoint)) {
            Error(std::string("Invalid USDA character in ") + context);
        }
        AdvanceBytes(decoded.length);
    }

    void AppendUtf8Character(std::string& out,
                             bool allowCrLf,
                             const char* context) {
        size_t start = pos_;
        ValidateAndAdvanceUtf8Character(allowCrLf, context);
        out.append(text_.substr(start, pos_ - start));
    }

    void SkipLineComment() {
        while (!AtEnd() && !IsCrLf(Peek())) {
            ValidateAndAdvanceUtf8Character(false, "line comment");
        }
    }

    void SkipBlockComment() {
        Advance(); Advance(); // skip /*
        while (!AtEnd()) {
            if (Peek() == '*' && PeekAt(1) == '/') {
                Advance(); Advance();
                return;
            }
            ValidateAndAdvanceUtf8Character(true, "block comment");
        }
        Error("Unterminated block comment");
    }

    // --- Identifiers and keywords ---

    // USDA grammar keywords (spec Section 16.2).
    // A KeywordlessIdentifier is any BaseIdentifier that is not a Keyword.
    static bool IsKeyword(const std::string& id) {
        static const std::unordered_set<std::string> kKeywords = {
            "add", "append", "class", "config", "connect", "custom",
            "customData", "def", "delete", "dictionary", "displayUnit",
            "doc", "inherits", "kind", "nameChildren", "None", "offset",
            "over", "payload", "permission", "prefixSubstitutions",
            "prepend", "properties", "references", "reorder", "relocates",
            "rel", "scale", "specializes", "subLayers",
            "suffixSubstitutions", "symmetryArguments", "symmetryFunction",
            "timeSamples", "uniform", "variantSet", "variantSets", "variants",
        };
        return kKeywords.count(id) > 0;
    }

    bool IsIdentStartAt(size_t pos, size_t* length = nullptr) const {
        return unicode::IsIdentifierStartAt(text_, pos, length);
    }

    bool IsIdentContinueAt(size_t pos, size_t* length = nullptr) const {
        return unicode::IsIdentifierContinueAt(text_, pos, length);
    }

    std::string ParseIdentifier() {
        SkipWhitespaceAndComments();
        size_t len = 0;
        if (AtEnd() || !IsIdentStartAt(pos_, &len)) {
            Error("Expected identifier");
        }
        std::string result;
        result.append(text_.substr(pos_, len));
        AdvanceBytes(len);
        while (!AtEnd() && IsIdentContinueAt(pos_, &len)) {
            result.append(text_.substr(pos_, len));
            AdvanceBytes(len);
        }
        return result;
    }

    // Parse a KeywordlessIdentifier per spec: a BaseIdentifier that is not a Keyword.
    std::string ParseKeywordlessIdentifier() {
        auto id = ParseIdentifier();
        if (IsKeyword(id)) {
            Error("Expected identifier but got keyword '" + id + "'");
        }
        return id;
    }

    // Parse identifier allowing namespaces (colons)
    std::string ParseNamespacedIdentifier() {
        std::string result = ParseIdentifier();
        while (!AtEnd() && Peek() == ':') {
            size_t save = pos_;
            int saveLine = line_, saveCol = col_;
            Advance();
            size_t len = 0;
            if (AtEnd() || !IsIdentStartAt(pos_, &len)) {
                pos_ = save;
                line_ = saveLine;
                col_ = saveCol;
                break;
            }
            result.push_back(':');
            result.append(text_.substr(pos_, len));
            AdvanceBytes(len);
            while (!AtEnd() && IsIdentContinueAt(pos_, &len)) {
                result.append(text_.substr(pos_, len));
                AdvanceBytes(len);
            }
        }
        return result;
    }

    bool TryIdentifier(std::string& out) {
        SkipWhitespaceAndComments();
        size_t len = 0;
        if (AtEnd() || !IsIdentStartAt(pos_, &len)) return false;
        out.clear();
        out.append(text_.substr(pos_, len));
        AdvanceBytes(len);
        while (!AtEnd() && IsIdentContinueAt(pos_, &len)) {
            out.append(text_.substr(pos_, len));
            AdvanceBytes(len);
        }
        return true;
    }

    bool PeekIdentifier(std::string_view expected) {
        SkipWhitespaceAndComments();
        if (pos_ + expected.size() > text_.size()) return false;
        if (text_.substr(pos_, expected.size()) != expected) return false;
        // Must not be followed by an ident-continue char
        size_t after = pos_ + expected.size();
        if (after < text_.size() && IsIdentContinueAt(after)) return false;
        return true;
    }

    bool MatchIdentifier(std::string_view expected) {
        if (PeekIdentifier(expected)) {
            for (size_t i = 0; i < expected.size(); i++) Advance();
            return true;
        }
        return false;
    }

    // --- Numbers ---

    double ParseNumber() {
        SkipWhitespaceAndComments();
        size_t start = pos_;
        bool hasSign = false;
        if (Peek() == '+') {
            Error("Expected number");
        }
        if (Peek() == '-') {
            Advance();
            hasSign = true;
        }

        // Handle inf and nan
        if (PeekIdentifier("inf")) {
            MatchIdentifier("inf");
            return hasSign && text_[start] == '-' ? -std::numeric_limits<double>::infinity()
                                                   : std::numeric_limits<double>::infinity();
        }
        if (!hasSign && PeekIdentifier("nan")) {
            MatchIdentifier("nan");
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (AtEnd() || (!std::isdigit(static_cast<unsigned char>(Peek())) &&
                        Peek() != '.')) {
            Error("Expected number");
        }

        bool digitsBeforeDot = false;
        while (!AtEnd() &&
               std::isdigit(static_cast<unsigned char>(Peek()))) {
            Advance();
            digitsBeforeDot = true;
        }
        bool sawDot = false;
        bool digitsAfterDot = false;
        if (!AtEnd() && Peek() == '.') {
            Advance();
            sawDot = true;
            while (!AtEnd() &&
                   std::isdigit(static_cast<unsigned char>(Peek()))) {
                Advance();
                digitsAfterDot = true;
            }
        }
        if (!digitsBeforeDot && !(sawDot && digitsAfterDot)) {
            Error("Expected number");
        }
        if (!AtEnd() && (Peek() == 'e' || Peek() == 'E')) {
            Advance();
            if (!AtEnd() && (Peek() == '+' || Peek() == '-')) Advance();
            if (AtEnd() ||
                !std::isdigit(static_cast<unsigned char>(Peek()))) {
                Error("Expected exponent digit");
            }
            while (!AtEnd() &&
                   std::isdigit(static_cast<unsigned char>(Peek()))) {
                Advance();
            }
        }

        std::string numStr(text_.substr(start, pos_ - start));
        try {
            return std::stod(numStr);
        } catch (const std::out_of_range&) {
            // Overflow → ±inf; underflow → 0
            return (numStr[0] == '-') ? -std::numeric_limits<double>::infinity()
                                      : std::numeric_limits<double>::infinity();
        }
    }

    int ParseInt() {
        SkipWhitespaceAndComments();
        size_t start = pos_;
        if (Peek() == '+') Error("Expected integer");
        if (Peek() == '-') Advance();
        if (!std::isdigit(static_cast<unsigned char>(Peek()))) Error("Expected integer");
        while (!AtEnd() && std::isdigit(static_cast<unsigned char>(Peek()))) Advance();
        std::string numStr(text_.substr(start, pos_ - start));
        try {
            size_t end;
            long long val = std::stoll(numStr, &end);
            if (val < static_cast<long long>(std::numeric_limits<int32_t>::min()) ||
                val > static_cast<long long>(std::numeric_limits<int32_t>::max())) {
                Error("Value " + numStr + " out of range for int (must be " +
                      std::to_string(std::numeric_limits<int32_t>::min()) + " to " +
                      std::to_string(std::numeric_limits<int32_t>::max()) + ")");
            }
            return static_cast<int>(val);
        } catch (const std::out_of_range&) {
            Error("Value " + numStr + " out of range for int");
        }
    }

    int64_t ParseInt64() {
        SkipWhitespaceAndComments();
        size_t start = pos_;
        if (Peek() == '+') Error("Expected integer");
        if (Peek() == '-') Advance();
        if (!std::isdigit(static_cast<unsigned char>(Peek()))) Error("Expected integer");
        while (!AtEnd() && std::isdigit(static_cast<unsigned char>(Peek()))) Advance();
        std::string numStr(text_.substr(start, pos_ - start));
        try {
            return std::stoll(numStr);
        } catch (const std::out_of_range&) {
            Error("Value " + numStr + " out of range for int64");
        }
    }

    uint64_t ParseUInt64() {
        SkipWhitespaceAndComments();
        size_t start = pos_;
        if (Peek() == '-') {
            Error("Negative value not allowed for unsigned integer type");
        }
        if (Peek() == '+') Error("Expected unsigned integer");
        if (!std::isdigit(static_cast<unsigned char>(Peek()))) Error("Expected unsigned integer");
        while (!AtEnd() && std::isdigit(static_cast<unsigned char>(Peek()))) Advance();
        std::string numStr(text_.substr(start, pos_ - start));
        try {
            return std::stoull(numStr);
        } catch (const std::out_of_range&) {
            Error("Value " + numStr + " out of range for uint64");
        }
    }

    UChar ParseUChar() {
        uint64_t val = ParseUInt64();
        if (val > std::numeric_limits<UChar>::max()) {
            Error("Value " + std::to_string(val) +
                  " out of range for uchar (must be 0 to 255)");
        }
        return static_cast<UChar>(val);
    }

    UInt ParseUInt() {
        uint64_t val = ParseUInt64();
        if (val > std::numeric_limits<UInt>::max()) {
            Error("Value " + std::to_string(val) + " out of range for uint (must be 0 to " +
                  std::to_string(std::numeric_limits<UInt>::max()) + ")");
        }
        return static_cast<UInt>(val);
    }

    // --- Strings ---

    std::string ParseQuotedString() {
        SkipWhitespaceAndComments();
        if (AtEnd()) Error("Expected string");

        char quote = Peek();
        if (quote != '"' && quote != '\'') {
            Error("Expected quoted string");
        }

        // Check for triple-quoted string
        if (PeekAt(1) == quote && PeekAt(2) == quote) {
            return ParseTripleQuotedString(quote);
        }

        Advance(); // opening quote
        std::string result;
        while (!AtEnd() && Peek() != quote) {
            if (Peek() == '\\') {
                Advance();
                AppendEscape(result);
            } else if (IsCrLf(Peek())) {
                Error("Unterminated string literal");
            } else {
                AppendUtf8Character(result, false, "string literal");
            }
        }
        Expect(quote);
        return result;
    }

    std::string ParseTripleQuotedString(char quote) {
        Advance(); Advance(); Advance(); // opening triple
        std::string result;
        while (!AtEnd()) {
            if (Peek() == quote && PeekAt(1) == quote && PeekAt(2) == quote) {
                Advance(); Advance(); Advance();
                return result;
            }
            if (Peek() == '\\') {
                Advance();
                AppendEscape(result);
            } else {
                AppendUtf8Character(result, true, "triple-quoted string");
            }
        }
        Error("Unterminated triple-quoted string");
    }

    static bool IsHexDigit(char c) {
        return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F');
    }

    static int HexDigitValue(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        return 10 + (c - 'A');
    }

    static bool IsOctDigit(char c) {
        return c >= '0' && c <= '7';
    }

    void AppendEscape(std::string& out) {
        if (AtEnd()) Error("Unexpected end in escape");
        char c = Advance();
        switch (c) {
            case 'a': out.push_back('\a'); return;
            case 'b': out.push_back('\b'); return;
            case 'f': out.push_back('\f'); return;
            case 'n': out.push_back('\n'); return;
            case 'r': out.push_back('\r'); return;
            case 't': out.push_back('\t'); return;
            case 'v': out.push_back('\v'); return;
            case '\\': out.push_back('\\'); return;
            case '\'': out.push_back('\''); return;
            case '"': out.push_back('"'); return;
            case 'x': {
                if (AtEnd() || !IsHexDigit(Peek())) {
                    Error("Expected hex digit in escape");
                }
                int value = HexDigitValue(Advance());
                if (!AtEnd() && IsHexDigit(Peek())) {
                    value = (value << 4) + HexDigitValue(Advance());
                }
                out.push_back(static_cast<char>(value));
                return;
            }
            default:
                if (IsOctDigit(c)) {
                    int value = c - '0';
                    for (int i = 0; i < 2 && !AtEnd() && IsOctDigit(Peek()); ++i) {
                        value = (value << 3) + (Advance() - '0');
                    }
                    out.push_back(static_cast<char>(value & 0xFF));
                    return;
                }
                Error(std::string("Unknown escape sequence: \\") + c);
        }
    }

    bool PeekQuotedString() {
        SkipWhitespaceAndComments();
        return !AtEnd() && (Peek() == '"' || Peek() == '\'');
    }

    // --- Asset references (@...@) ---

    std::string ParseAssetRef() {
        SkipWhitespaceAndComments();
        Expect('@');
        // Triple-at: @@@...@@@
        if (Peek() == '@' && PeekAt(1) == '@') {
            Advance(); Advance(); // consume @@
            std::string result;
            while (!AtEnd()) {
                if (Peek() == '@' && PeekAt(1) == '@' && PeekAt(2) == '@') {
                    Advance(); Advance(); Advance();
                    return result;
                }
                AppendUtf8Character(result, false, "asset reference");
            }
            Error("Unterminated triple-at asset reference");
        }
        // Single-at: @...@
        std::string result;
        while (!AtEnd() && Peek() != '@') {
            AppendUtf8Character(result, false, "asset reference");
        }
        Expect('@');
        return result;
    }

    bool PeekAssetRef() {
        SkipWhitespaceAndComments();
        return !AtEnd() && Peek() == '@';
    }

    // --- Path references (<...>) ---

    Path ParsePathRef() {
        SkipWhitespaceAndComments();
        Expect('<');
        std::string pathStr;
        while (!AtEnd() && Peek() != '>') {
            pathStr += Advance();
        }
        Expect('>');
        return Path::Parse(pathStr);
    }

    bool PeekPathRef() {
        SkipWhitespaceAndComments();
        return !AtEnd() && Peek() == '<';
    }

    // --- Peek helpers ---

    bool PeekChar(char c) {
        SkipWhitespaceAndComments();
        return !AtEnd() && Peek() == c;
    }

    bool MatchChar(char c) {
        SkipWhitespaceAndComments();
        if (!AtEnd() && Peek() == c) { Advance(); return true; }
        return false;
    }

    void ExpectChar(char c) {
        SkipWhitespaceAndComments();
        Expect(c);
    }

    // ============================================================
    // USDA Grammar Productions
    // ============================================================

    // layer = header metadata_block? prim_spec*
    void ParseLayer() {
        ParseHeader();
        SkipWhitespaceAndComments();

        // Optional layer metadata block
        if (PeekChar('(')) {
            ParseLayerMetadata();
        }

        // Root prim specs and reorder statements
        SkipWhitespaceAndComments();
        while (!AtEnd()) {
            SkipWhitespaceAndComments();
            if (AtEnd()) break;

            if (PeekIdentifier("reorder")) {
                ParseReorderRootPrims();
            } else {
                auto primPath = Path::AbsoluteRoot();
                ParsePrimSpec(primPath);
            }
        }
    }

    // header = "#usda 1.0"
    void ParseHeader() {
        // Skip only whitespace (not comments!) before the header
        while (!AtEnd() && (Peek() == ' ' || Peek() == '\t' || Peek() == '\r' || Peek() == '\n')) {
            Advance();
        }
        // The magic line
        if (!MatchStr("#usda")) {
            Error("Expected '#usda 1.0' header");
        }
        // Skip whitespace (not newline-based comments)
        while (!AtEnd() && (Peek() == ' ' || Peek() == '\t')) Advance();
        if (!MatchStr("1.0")) {
            Error("Expected version '1.0' after #usda");
        }
        // Skip rest of version line (may have extra version components like 1.0.32)
        while (!AtEnd() && Peek() != '\n') Advance();
    }

    // Layer metadata block: ( key = value ... )
    void ParseLayerMetadata() {
        ExpectChar('(');
        ParseMetadataBlockContents(layer_.GetLayerSpec(), true);
        ExpectChar(')');
    }

    // Shared metadata block parser for layer and prim metadata
    void ParseMetadataBlockContents(Spec& spec, bool isLayer) {
        while (!PeekChar(')')) {
            SkipWhitespaceAndComments();
            if (PeekChar(')')) break;

            // Standalone quoted string → doc/comment
            if (PeekQuotedString()) {
                auto s = ParseQuotedString();
                // Treat standalone string as documentation if not already set
                if (spec.GetDocumentation().empty()) {
                    spec.SetDocumentation(s);
                }
                continue;
            }

            // ListOp prefix keywords
            if (PeekIdentifier("prepend") || PeekIdentifier("append") ||
                PeekIdentifier("delete")) {
                ParseListOpMetadata(spec);
                continue;
            }

            // Must be an identifier
            std::string key = ParseIdentifier();
            SkipWhitespaceAndComments();

            // If no '=', it might be a declaration (skip)
            if (!PeekChar('=')) {
                // Could be a value-less key, skip
                continue;
            }
            ExpectChar('=');

            // Dispatch on known keys
            if (key == "doc") {
                spec.SetDocumentation(ParseQuotedString());
            } else if (key == "subLayers" && isLayer) {
                ParseSubLayersList();
            } else if (key == "relocates" && isLayer) {
                ParseRelocatesMap();
            } else if (key == "timeCodesPerSecond" && isLayer) {
                spec.SetTimeCodesPerSecond(ParseNumber());
            } else if (key == "framesPerSecond" && isLayer) {
                spec.SetFramesPerSecond(ParseNumber());
            } else if (key == "startTimeCode" && isLayer) {
                spec.SetStartTimeCode(ParseNumber());
            } else if (key == "endTimeCode" && isLayer) {
                spec.SetEndTimeCode(ParseNumber());
            } else if (key == "defaultPrim" && isLayer) {
                spec.SetDefaultPrim(Token(ParseQuotedString()));
            } else if (key == "customLayerData" && isLayer) {
                spec.SetField(FieldNames::customLayerData, Value(ParseDictionary()));
            } else if (key == "fallbackPrimTypes" && isLayer) {
                spec.SetField(FieldNames::fallbackPrimTypes, Value(ParseDictionary()));
            } else if (key == "colorSpace" && spec.GetType() == SpecType::Attribute) {
                spec.SetField(FieldNames::colorSpace, Value(Token(ParseQuotedString())));
            } else if (key == "kind") {
                spec.SetKind(Token(ParseQuotedString()));
            } else if (key == "active") {
                spec.SetActive(ParseBoolValue());
            } else if (key == "hidden") {
                spec.SetHidden(ParseBoolValue());
            } else if (key == "instanceable") {
                spec.SetInstanceable(ParseBoolValue());
            } else if (key == "displayName") {
                spec.SetDisplayName(ParseQuotedString());
            } else if (key == "displayGroup") {
                spec.SetField(FieldNames::displayGroup, Value(ParseQuotedString()));
            } else if (key == "customData" || key == "assetInfo") {
                spec.SetField(Token(key), Value(ParseDictionary()));
            } else if (key == "variantSets") {
                // Authored as `variantSets = "name"` or `variantSets = [..]`;
                // stored internally in the `variantSetNames` field as an
                // explicit ListOp<string> (spec §11.2). Prepend / append /
                // delete forms of `variantSets` are handled by
                // ParseListOpMetadata below, which delegates here via the
                // same key name.
                SkipWhitespaceAndComments();
                std::vector<std::string> names;
                if (PeekChar('[')) {
                    names = ParseStringList();
                } else if (PeekQuotedString()) {
                    names.push_back(ParseQuotedString());
                } else {
                    ParseValue();  // unexpected form, swallow gracefully
                }
                spec.SetField(FieldNames::variantSetNames,
                              Value(ListOp<std::string>::CreateExplicit(std::move(names))));
            } else if (key == "variants") {
                // `variants = {"setName": "variantName", ...}` maps to the
                // variantSelection Dictionary field (spec §11.2).
                spec.SetField(FieldNames::variantSelection, Value(ParseDictionary()));
            } else if (key == "apiSchemas") {
                ParseTokenListOp(spec, "apiSchemas", "");
            } else if (key == "clipSets") {
                // Spec §13.2.1.2.2 — listop<string> naming the search
                // order of clip sets. Parsed via ParseStringListOp so
                // the spec's element-type distinction (vs apiSchemas's
                // listop<token>) is preserved at the call site.
                ParseStringListOp(spec, "clipSets", "");
            } else if (key == "references" || key == "payload") {
                ParseReferenceOrNone(spec, key, "");
            } else if (key == "inherits" || key == "specializes") {
                ParsePathListOrNone(spec, key, "");
            } else {
                // Generic metadata: key must be a KeywordlessIdentifier
                if (IsKeyword(key)) {
                    Error("Unexpected keyword '" + key + "' in metadata");
                }
                Value val = ParseValue();
                spec.SetField(Token(key), std::move(val));
            }
        }
    }

    // Parse listop-prefixed metadata: prepend/append/delete key = value
    void ParseListOpMetadata(Spec& spec) {
        std::string prefix = ParseIdentifier(); // prepend/append/delete
        SkipWhitespaceAndComments();
        std::string key = ParseIdentifier();
        ExpectChar('=');

        if (key == "references" || key == "payload") {
            ParseReferenceOrNone(spec, key, prefix);
        } else if (key == "inherits" || key == "specializes") {
            ParsePathListOrNone(spec, key, prefix);
        } else if (key == "apiSchemas") {
            ParseTokenListOp(spec, "apiSchemas", prefix);
        } else if (key == "clipSets") {
            // Spec §13.2.1.2.2 — listop<string> composable form
            // (prepend/append/delete clipSets = ["..."]).
            ParseStringListOp(spec, "clipSets", prefix);
        } else if (key == "variantSets") {
            // variantSets prepend/append/delete maps onto variantSetNames
            // which is spec-declared listop<string> (§7.6.2.3.5).
            ParseStringListOp(spec, "variantSetNames", prefix);
        } else {
            // Generic list op value — just parse and store
            ParseValue();
        }
    }

    void ParseReferenceOrNone(Spec& spec, const std::string& key, const std::string& prefix) {
        SkipWhitespaceAndComments();
        if (PeekIdentifier("None")) {
            MatchIdentifier("None");
            spec.SetField(Token(key), Value(ListOp<Reference>::CreateExplicit({})));
            return;
        }
        if (PeekChar('[')) {
            ParseReferenceList(spec, key, prefix);
        } else {
            auto ref = ParseSingleReference();
            StoreReferences(spec, key, prefix, {ref});
        }
    }

    void ParsePathListOrNone(Spec& spec, const std::string& key, const std::string& prefix) {
        // USDA-syntax `inherits` and `specializes` are stored under the
        // spec field names from FieldNames (`inheritPaths` per spec
        // §10.5.1, `specializes` per §10.5.5). Map the syntax keyword
        // to the canonical field name so consumers can read either via
        // FieldNames::* or via the spec name verbatim.
        Token storeKey;
        if (key == "inherits")        storeKey = FieldNames::inheritPaths;
        else if (key == "specializes") storeKey = FieldNames::specializes;
        else                           storeKey = Token(key);

        SkipWhitespaceAndComments();
        if (PeekIdentifier("None")) {
            MatchIdentifier("None");
            // Explicit empty list-op; record so consumers can distinguish
            // "field present but empty" from "field absent".
            ApplyPathListOp(spec, storeKey, prefix, /*items=*/{});
            return;
        }

        std::vector<Path> items;
        if (PeekChar('[')) {
            ExpectChar('[');
            while (!PeekChar(']')) {
                SkipWhitespaceAndComments();
                if (PeekChar(']')) break;
                if (PeekPathRef()) items.push_back(ParsePathRef());
                SkipWhitespaceAndComments();
                if (PeekChar(',')) Advance();
            }
            ExpectChar(']');
        } else if (PeekPathRef()) {
            // Single-path form: `inherits = </Foo>`.
            items.push_back(ParsePathRef());
        } else {
            ParseValue(); // fallback for unrecognised syntax
            return;
        }
        ApplyPathListOp(spec, storeKey, prefix, std::move(items));
    }

    // Merge a ListOp<Path> opinion onto `spec`. With an empty `prefix`
    // string the opinion is the explicit form (`inherits = [...]`); with
    // "prepend" / "append" / "delete" the opinion is composed onto any
    // existing list-op opinion under the same field name, matching the
    // listop-merge shape used elsewhere in the USDA parser.
    void ApplyPathListOp(Spec& spec, const Token& fieldName,
                          const std::string& prefix,
                          std::vector<Path> items) {
        if (prefix.empty()) {
            ListOp<Path> op;
            op.SetExplicitItems(std::move(items));
            spec.SetField(fieldName, Value(std::move(op)));
            return;
        }
        ListOp<Path> op;
        if (auto* existing = spec.GetField(fieldName)) {
            if (auto* prev = existing->Get<ListOp<Path>>()) op = *prev;
        }
        if (prefix == "prepend") op.SetPrependedItems(std::move(items));
        else if (prefix == "append") op.SetAppendedItems(std::move(items));
        else if (prefix == "delete") op.SetDeletedItems(std::move(items));
        spec.SetField(fieldName, Value(std::move(op)));
    }

    // Parse a single reference: @asset@</path> (offset=...; scale=...)
    Reference ParseSingleReference() {
        Reference ref;
        if (PeekAssetRef()) {
            ref.assetPath = ParseAssetRef();
        }
        SkipWhitespaceAndComments();
        if (PeekPathRef()) {
            ref.primPath = ParsePathRef();
        }
        SkipWhitespaceAndComments();
        if (PeekChar('(')) {
            ParseLayerOffsetBlock(ref);
        }
        return ref;
    }

    void ParseLayerOffsetBlock(Reference& ref) {
        ExpectChar('(');
        while (!PeekChar(')')) {
            SkipWhitespaceAndComments();
            if (PeekChar(')')) break;
            std::string key = ParseIdentifier();
            ExpectChar('=');
            if (key == "offset") {
                ref.offset = ParseNumber();
            } else if (key == "scale") {
                ref.scale = ParseNumber();
            } else {
                // Unknown key in layer offset — could be customData etc.
                ParseValue();
            }
        }
        ExpectChar(')');
    }

    // Parse subLayers list with optional layer offset metadata on each entry
    void ParseSubLayersList() {
        SubLayerPaths subLayers;
        ExpectChar('[');
        while (!PeekChar(']')) {
            SkipWhitespaceAndComments();
            if (PeekChar(']')) break;
            if (PeekAssetRef()) {
                auto asset = ParseAssetRef();
                Retiming retiming;
                SkipWhitespaceAndComments();
                if (PeekChar('(')) {
                    Reference tmp;
                    ParseLayerOffsetBlock(tmp);
                    retiming.offset = tmp.offset;
                    retiming.scale = tmp.scale;
                }
                subLayers.paths.push_back(std::move(asset));
                subLayers.offsets.push_back(retiming);
            } else {
                break;
            }
            SkipWhitespaceAndComments();
            if (PeekChar(',')) Advance();
        }
        ExpectChar(']');
        layer_.GetLayerSpec().SetField(FieldNames::subLayers,
            Value(std::move(subLayers)));
    }

    // Parse a list of quoted strings: ["a", "b", ...]
    std::vector<std::string> ParseStringList() {
        std::vector<std::string> result;
        ExpectChar('[');
        while (!PeekChar(']')) {
            if (!result.empty()) {
                if (!MatchChar(',')) break;
            }
            SkipWhitespaceAndComments();
            if (PeekChar(']')) break;
            result.push_back(ParseQuotedString());
        }
        ExpectChar(']');
        return result;
    }

    // relocates = { <path>: <path>, ... }
    void ParseRelocatesMap() {
        std::vector<Relocate> relocates;
        ExpectChar('{');
        while (!PeekChar('}')) {
            SkipWhitespaceAndComments();
            if (PeekChar('}')) break;
            Relocate r;
            r.sourcePath = ParsePathRef();
            ExpectChar(':');
            SkipWhitespaceAndComments();
            if (PeekIdentifier("None")) {
                MatchIdentifier("None");
                r.targetPath = std::nullopt;
            } else if (PeekPathRef()) {
                r.targetPath = ParsePathRef();
            } else {
                ParseValue();
            }
            relocates.push_back(std::move(r));
            SkipWhitespaceAndComments();
            if (PeekChar(',')) Advance();
        }
        ExpectChar('}');
        layer_.GetLayerSpec().SetField(FieldNames::layerRelocates,
            Value(std::move(relocates)));
    }

    // reorder rootPrims = ["name1", "name2", ...]
    void ParseReorderRootPrims() {
        MatchIdentifier("reorder");
        SkipWhitespaceAndComments();
        if (!MatchIdentifier("rootPrims")) {
            Error("Expected 'rootPrims' after 'reorder'");
        }
        ExpectChar('=');
        auto names = ParseStringList();
        layer_.GetLayerSpec().SetField(FieldNames::primOrder,
            Value(Value::ArrayTag{}, TypeId::Unknown, std::move(names)));
    }

    // ============================================================
    // Prim Spec
    // ============================================================

    // prim_spec = specifier type_name? prim_name metadata_block? { contents }
    void ParsePrimSpec(const Path& parentPath) {
        SkipWhitespaceAndComments();

        // Parse specifier
        Specifier specifier;
        if (MatchIdentifier("def")) {
            specifier = Specifier::Def;
        } else if (MatchIdentifier("over")) {
            specifier = Specifier::Over;
        } else if (MatchIdentifier("class")) {
            specifier = Specifier::Class;
        } else {
            Error("Expected prim specifier (def, over, class)");
        }

        SkipWhitespaceAndComments();

        // Parse optional type name and required prim name
        // Could be: def "PrimName" or def TypeName "PrimName"
        std::string typeName;
        std::string primName;

        if (PeekQuotedString()) {
            // No type name, just prim name
            primName = ParseQuotedString();
        } else {
            // Type name then prim name
            typeName = ParseIdentifier();
            SkipWhitespaceAndComments();
            if (PeekQuotedString()) {
                primName = ParseQuotedString();
            } else {
                // The identifier was actually the prim name (for 'over' without type)
                // Actually, prim names in USDA are always quoted
                Error("Expected quoted prim name");
            }
        }

        // Build path for this prim
        Path primPath = parentPath.AppendChild(primName);

        // Create the spec
        Spec primSpec(SpecType::Prim);
        primSpec.SetSpecifier(specifier);
        if (!typeName.empty()) {
            primSpec.SetTypeName(Token(typeName));
        }

        // Optional metadata block
        SkipWhitespaceAndComments();
        if (PeekChar('(')) {
            ParsePrimMetadata(primSpec);
        }

        // Register prim children on parent
        // (We track this for completeness but the layer stores specs by path)

        // Prim body
        SkipWhitespaceAndComments();
        if (PeekChar('{')) {
            ExpectChar('{');
            ParsePrimBody(primPath, primSpec);
            ExpectChar('}');
        }

        // Store the spec in the layer
        layer_.SetSpec(primPath, std::move(primSpec));
    }

    // Prim metadata: ( key = value, ... )
    void ParsePrimMetadata(Spec& spec) {
        ExpectChar('(');
        ParseMetadataBlockContents(spec, false);
        ExpectChar(')');
    }

    // Store parsed references into a spec's field using ListOp semantics
    void StoreReferences(Spec& spec, const std::string& fieldName,
                         const std::string& listOpMode, std::vector<Reference> refs) {
        Token fieldToken(fieldName);
        ListOp<Reference> listOp;
        if (auto* existing = spec.GetField(fieldToken)) {
            if (auto* p = existing->Get<ListOp<Reference>>()) {
                listOp = *p;
            }
        }

        if (listOpMode == "prepend") {
            auto current = listOp.GetPrependedItems();
            current.insert(current.end(), refs.begin(), refs.end());
            listOp.SetPrependedItems(std::move(current));
        } else if (listOpMode == "append") {
            auto current = listOp.GetAppendedItems();
            current.insert(current.end(), refs.begin(), refs.end());
            listOp.SetAppendedItems(std::move(current));
        } else if (listOpMode == "delete") {
            auto current = listOp.GetDeletedItems();
            current.insert(current.end(), refs.begin(), refs.end());
            listOp.SetDeletedItems(std::move(current));
        } else {
            listOp.SetExplicitItems(std::move(refs));
        }

        spec.SetField(fieldToken, Value(std::move(listOp)));
    }

    // Parse a list of references: [ @asset@</path> (offset; scale), ... ]
    void ParseReferenceList(Spec& spec, const std::string& fieldName,
                            const std::string& listOpMode) {
        std::vector<Reference> refs;
        ExpectChar('[');
        while (!PeekChar(']')) {
            SkipWhitespaceAndComments();
            if (PeekChar(']')) break;
            refs.push_back(ParseSingleReference());
            SkipWhitespaceAndComments();
            if (PeekChar(',')) Advance();
        }
        ExpectChar(']');
        StoreReferences(spec, fieldName, listOpMode, std::move(refs));
    }

    // Parse a list of paths: [ </path1>, </path2>, ... ]
    void ParsePathList(Spec& spec, const std::string& fieldName,
                       const std::string& listOpMode) {
        ExpectChar('[');
        while (!PeekChar(']')) {
            SkipWhitespaceAndComments();
            if (PeekChar(']')) break;
            ParsePathRef();
            SkipWhitespaceAndComments();
            if (PeekChar(',')) Advance();
        }
        ExpectChar(']');
    }

    // Parse a list op of quoted-string items. The USDA syntax for
    // listop<string> (e.g. clipSets per §13.2.1.2.2, variantSetNames
    // per §7.6.2.3.5) and listop<token> (e.g. apiSchemas per
    // §13.2.1.2) is identical — both read quoted-string items — so
    // the parse logic is shared via this template. `Elem` is the
    // storage element type (std::string or Token), and `convert` maps
    // the parsed std::string into that element type.
    //
    // Accepts either a list form (["a", "b"]) or a single-string
    // form ("a") — the single-string form appears for
    // `variantSets = "name"` and its prepend/append equivalents.
    template <typename Elem, typename Convert>
    void ParseQuotedStringListOp(Spec& spec, const std::string& fieldName,
                                  const std::string& listOpMode,
                                  Convert convert) {
        SkipWhitespaceAndComments();

        // Handle None (explicit empty)
        if (PeekIdentifier("None")) {
            MatchIdentifier("None");
            spec.SetField(Token(fieldName),
                          Value(ListOp<Elem>::CreateExplicit({})));
            return;
        }

        // Single-string shorthand: `add variantSets = "which"`.
        if (PeekQuotedString()) {
            std::vector<Elem> items;
            items.push_back(convert(ParseQuotedString()));
            ListOp<Elem> listOp;
            if (listOpMode == "prepend") {
                listOp.SetPrependedItems(std::move(items));
            } else if (listOpMode == "append") {
                listOp.SetAppendedItems(std::move(items));
            } else if (listOpMode == "delete") {
                listOp.SetDeletedItems(std::move(items));
            } else {
                listOp.SetExplicitItems(std::move(items));
            }
            // Merge with any existing listop on the spec.
            if (const auto* existing = spec.GetField(Token(fieldName))) {
                if (const auto* p = existing->Get<ListOp<Elem>>()) {
                    if (listOpMode == "prepend") {
                        auto merged = p->GetPrependedItems();
                        auto added = listOp.GetPrependedItems();
                        merged.insert(merged.end(), added.begin(), added.end());
                        listOp = *p;
                        listOp.SetPrependedItems(std::move(merged));
                    } else if (listOpMode == "append") {
                        auto merged = p->GetAppendedItems();
                        auto added = listOp.GetAppendedItems();
                        merged.insert(merged.end(), added.begin(), added.end());
                        listOp = *p;
                        listOp.SetAppendedItems(std::move(merged));
                    } else if (listOpMode == "delete") {
                        auto merged = p->GetDeletedItems();
                        auto added = listOp.GetDeletedItems();
                        merged.insert(merged.end(), added.begin(), added.end());
                        listOp = *p;
                        listOp.SetDeletedItems(std::move(merged));
                    }
                    // explicit: fully replace — no merge.
                }
            }
            spec.SetField(Token(fieldName), Value(std::move(listOp)));
            return;
        }

        // Parse list form: ["Foo", "Bar"]
        std::vector<Elem> items;
        ExpectChar('[');
        while (!PeekChar(']')) {
            SkipWhitespaceAndComments();
            if (PeekChar(']')) break;
            items.push_back(convert(ParseQuotedString()));
            SkipWhitespaceAndComments();
            if (PeekChar(',')) Advance();
        }
        ExpectChar(']');

        Token fieldToken(fieldName);
        ListOp<Elem> listOp;
        if (auto* existing = spec.GetField(fieldToken)) {
            if (auto* p = existing->Get<ListOp<Elem>>()) {
                listOp = *p;
            }
        }

        if (listOpMode == "prepend") {
            auto current = listOp.GetPrependedItems();
            current.insert(current.end(), items.begin(), items.end());
            listOp.SetPrependedItems(std::move(current));
        } else if (listOpMode == "append") {
            auto current = listOp.GetAppendedItems();
            current.insert(current.end(), items.begin(), items.end());
            listOp.SetAppendedItems(std::move(current));
        } else if (listOpMode == "delete") {
            auto current = listOp.GetDeletedItems();
            current.insert(current.end(), items.begin(), items.end());
            listOp.SetDeletedItems(std::move(current));
        } else {
            listOp.SetExplicitItems(std::move(items));
        }

        spec.SetField(fieldToken, Value(std::move(listOp)));
    }

    // Parse a list op of tokens. Used for spec-declared listop<token>
    // fields (apiSchemas per §13.2.1.2). Produces ListOp<Token>.
    void ParseTokenListOp(Spec& spec, const std::string& fieldName,
                          const std::string& listOpMode) {
        ParseQuotedStringListOp<Token>(spec, fieldName, listOpMode,
            [](std::string s) { return Token(std::move(s)); });
    }

    // Parse a list op of strings. Used for spec-declared listop<string>
    // fields (clipSets per §13.2.1.2.2, variantSetNames per §7.6.2.3.5).
    // Produces ListOp<std::string>.
    void ParseStringListOp(Spec& spec, const std::string& fieldName,
                           const std::string& listOpMode) {
        ParseQuotedStringListOp<std::string>(spec, fieldName, listOpMode,
            [](std::string s) { return s; });
    }

    // Prim body: contains properties, child prims, variant sets
    void ParsePrimBody(const Path& primPath, Spec& primSpec) {
        while (!PeekChar('}')) {
            SkipWhitespaceAndComments();
            if (PeekChar('}')) break;

            // Each PrimItem is parsed with the starting line captured
            // so RequirePrimItemTerminator can detect implicit newline
            // crossings in the item's parse (see the helper's comment).
            int startLine = line_;
            // Variant set block — grammar: VariantSetStatement (SinglelinePadding)* Crlf
            if (PeekIdentifier("variantSet")) {
                ParseVariantSetBlock(primPath);
                RequirePrimItemTerminator(/*allowSemicolon=*/false,
                                          "variantSet block", startLine);
            }
            // Reorder children — ChildOrPropertyOrderStatement — Crlf or Semicolon
            else if (PeekIdentifier("reorder")) {
                MatchIdentifier("reorder");
                SkipWhitespaceAndComments();
                if (MatchIdentifier("nameChildren")) {
                    ExpectChar('=');
                    auto order = ParseStringList();
                    primSpec.SetField(FieldNames::primOrder,
                        Value(Value::ArrayTag{}, TypeId::Unknown, std::move(order)));
                } else if (MatchIdentifier("properties")) {
                    ExpectChar('=');
                    auto order = ParseStringList();
                    primSpec.SetField(FieldNames::propertyOrder,
                        Value(Value::ArrayTag{}, TypeId::Unknown, std::move(order)));
                }
                RequirePrimItemTerminator(/*allowSemicolon=*/true,
                                          "reorder statement", startLine);
            }
            // Child prim (def/over/class) — PrimSpec — Crlf only
            else if (PeekIdentifier("def") || PeekIdentifier("over") || PeekIdentifier("class")) {
                ParsePrimSpec(primPath);
                RequirePrimItemTerminator(/*allowSemicolon=*/false,
                                          "nested prim spec", startLine);
            }
            // Property: attribute or relationship — PropertySpec — Crlf or Semicolon
            else {
                ParseProperty(primPath);
                RequirePrimItemTerminator(/*allowSemicolon=*/true,
                                          "property spec", startLine);
            }
        }
    }

    // ============================================================
    // Properties (Attributes and Relationships)
    // ============================================================

    void ParseProperty(const Path& primPath) {
        SkipWhitespaceAndComments();

        // Handle listop prefixes on properties: delete/append/prepend
        std::string listOpPrefix;
        if (PeekIdentifier("delete") || PeekIdentifier("append") || PeekIdentifier("prepend")) {
            // Save position — could be a listop prefix or a type name
            size_t save = pos_;
            int saveLine = line_, saveCol = col_;
            listOpPrefix = ParseIdentifier();
            SkipWhitespaceAndComments();
            // If followed by a known property pattern, treat as prefix
            // Otherwise restore and treat as type name
            if (!PeekIdentifier("rel") && !PeekIdentifier("custom") && !PeekIdentifier("uniform") &&
                !IsIdentStartAt(pos_)) {
                // Not a property — restore
                pos_ = save; line_ = saveLine; col_ = saveCol;
                listOpPrefix.clear();
            }
        }

        bool isCustom = false;
        if (MatchIdentifier("custom")) {
            isCustom = true;
            SkipWhitespaceAndComments();
        }

        // Check for relationship
        if (PeekIdentifier("rel")) {
            ParseRelationship(primPath, isCustom, listOpPrefix);
            return;
        }

        // Check for uniform variability
        Variability variability = Variability::Varying;
        if (MatchIdentifier("uniform")) {
            variability = Variability::Uniform;
            SkipWhitespaceAndComments();
        }

        // Type name
        std::string typeName = ParseTypeName();
        SkipWhitespaceAndComments();

        // Check for relationship declared as "rel"
        if (typeName == "rel") {
            ParseRelationshipAfterRel(primPath, isCustom, listOpPrefix);
            return;
        }

        // Property name (possibly namespaced with colons)
        std::string propName = ParseNamespacedIdentifier();

        Path attrPath = primPath.AppendProperty(propName);

        Spec attrSpec(SpecType::Attribute);
        attrSpec.SetTypeName(Token(typeName));
        attrSpec.SetVariability(variability);
        attrSpec.SetCustom(isCustom);

        SkipWhitespaceAndComments();

        // Check for dot-suffixed forms: .timeSamples, .connect, .spline
        // The dot may have whitespace around it
        if (PeekChar('.')) {
            Advance(); // skip '.'
            SkipWhitespaceAndComments();
            std::string suffix = ParseIdentifier();
            SkipWhitespaceAndComments();
            ExpectChar('=');
            ParseAttributeSuffix(attrSpec, typeName, suffix);
        }
        // Assignment
        else if (MatchChar('=')) {
            SkipWhitespaceAndComments();
            if (PeekIdentifier("None")) {
                MatchIdentifier("None");
                attrSpec.SetField(FieldNames::defaultValue, Value(ValueBlock{}));
            } else {
                Value val = ParseTypedValue(typeName);
                attrSpec.SetField(FieldNames::defaultValue, std::move(val));
            }
        }
        // else: just a declaration with no value

        // Optional metadata block for attribute. Spec §16.2.16 puts the
        // AttributeMetadata after InlinePadding only — it must live on
        // the same line as the assignment. Using SkipWhitespaceAndComments
        // here would swallow the trailing StatementSeparator (Crlf or
        // Semicolon) that ParsePrimBody's RequirePrimItemTerminator
        // check relies on. Raw Peek() for the same reason — PeekChar
        // internally calls SkipWhitespaceAndComments.
        SkipSinglelinePadding();
        if (Peek() == '(') {
            ParsePropertyMetadata(attrSpec);
        }

        // Merge into existing spec if this attribute was already declared
        // (e.g., default value and .timeSamples as separate statements)
        auto* existing = layer_.GetSpec(attrPath);
        if (existing) {
            for (const auto& [key, val] : attrSpec.GetFields()) {
                existing->SetField(Token(key), val);
            }
        } else {
            layer_.SetSpec(attrPath, std::move(attrSpec));
        }
    }

    void ParseAttributeSuffix(Spec& attrSpec, const std::string& typeName,
                              const std::string& suffix) {
        if (suffix == "timeSamples") {
            auto ts = ParseTimeSamplesMap(typeName);
            Dictionary tsDict;
            for (auto& [time, val] : ts) {
                tsDict[std::to_string(time)] = std::move(val);
            }
            attrSpec.SetField(FieldNames::timeSamples, Value(std::move(tsDict)));
        }
        else if (suffix == "connect") {
            SkipWhitespaceAndComments();
            if (PeekIdentifier("None")) {
                MatchIdentifier("None");
            } else {
                // Spec §6.6.3 + §12.2.6: connectionPaths is a path-typed
                // list op. Store as ListOp<Path> so the type round-trips
                // and combining can fold opinions without re-parsing.
                std::vector<Path> targets;
                if (PeekPathRef()) {
                    targets.push_back(ParsePathRef());
                } else if (PeekChar('[')) {
                    ExpectChar('[');
                    while (!PeekChar(']')) {
                        SkipWhitespaceAndComments();
                        if (PeekChar(']')) break;
                        targets.push_back(ParsePathRef());
                        SkipWhitespaceAndComments();
                        if (PeekChar(',')) Advance();
                    }
                    ExpectChar(']');
                }
                if (!targets.empty()) {
                    ListOp<Path> listOp;
                    listOp.SetExplicitItems(std::move(targets));
                    attrSpec.SetField(FieldNames::connectionPaths,
                                      Value(std::move(listOp)));
                }
            }
        }
        else if (suffix == "spline") {
            ParseSplineBlock(attrSpec);
        }
        else {
            // Unknown suffix, parse as generic value
            ParseValue();
        }
    }

    // Parse spline block per spec §7.4.2.4 + §16.2.13:
    //   '{'
    //     curve_type?          // 'bezier' | 'hermite'
    //     ( extrapolation | loop )*
    //     ( knot )*
    //   '}'
    //
    // Clauses are comma-separated. curve_type is authored first by
    // convention but the parser accepts it in any order (last wins).
    // Extrapolation directives take the form `pre: <mode>` / `post:
    // <mode>` where mode is one of:
    //   none | held | linear | sloped(<slope>) | looprepeat |
    //   loopreset | looposcillate
    // The optional `(<slope>)` is only meaningful for `sloped`; other
    // modes accept and ignore it for tolerance with real-world files.
    //
    // Knot grammar:
    //   time ':' value ('&' preValue)?
    //     (';' ( 'pre' '(' slope ',' width ')'
    //          | 'post' interp_mode? ('(' slope ',' width ')')?
    //          | custom_data_dict                    /* skipped */
    //          ))*
    // interp_mode: none | held | linear | curve.
    // Spline-local whitespace skipper: consumes spaces, tabs, CR, LF,
    // and comments — but NOT semicolons. The general-purpose
    // SkipWhitespaceAndComments treats ';' as a statement separator
    // and eats it, which would destroy the knot grammar's `;`-
    // delimited tangent clauses.
    void SkipSplineWS() {
        while (!AtEnd()) {
            char c = Peek();
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                Advance();
            } else if (c == '#') {
                SkipLineComment();
            } else if (c == '/' && PeekAt(1) == '/') {
                SkipLineComment();
            } else if (c == '/' && PeekAt(1) == '*') {
                SkipBlockComment();
            } else {
                break;
            }
        }
    }

    bool PeekSplineChar(char c) {
        SkipSplineWS();
        return !AtEnd() && Peek() == c;
    }

    bool MatchSplineChar(char c) {
        SkipSplineWS();
        if (!AtEnd() && Peek() == c) { Advance(); return true; }
        return false;
    }

    void ExpectSplineChar(char c) {
        SkipSplineWS();
        if (!Match(c)) {
            Error(std::string("Expected '") + c + "', got '" +
                  (AtEnd() ? "EOF" : std::string(1, Peek())) + "'");
        }
    }

    bool PeekSplineIdent(std::string_view expected) {
        SkipSplineWS();
        if (pos_ + expected.size() > text_.size()) return false;
        if (text_.substr(pos_, expected.size()) != expected) return false;
        size_t after = pos_ + expected.size();
        if (after < text_.size() && IsIdentContinueAt(after)) return false;
        return true;
    }

    bool MatchSplineIdent(std::string_view expected) {
        if (PeekSplineIdent(expected)) {
            for (size_t i = 0; i < expected.size(); ++i) Advance();
            return true;
        }
        return false;
    }

    std::string ParseSplineIdent() {
        SkipSplineWS();
        size_t len = 0;
        if (AtEnd() || !IsIdentStartAt(pos_, &len)) Error("Expected identifier");
        std::string result;
        result.append(text_.substr(pos_, len));
        AdvanceBytes(len);
        while (!AtEnd() && IsIdentContinueAt(pos_, &len)) {
            result.append(text_.substr(pos_, len));
            AdvanceBytes(len);
        }
        return result;
    }

    double ParseSplineNumber() {
        SkipSplineWS();
        return ParseNumber();  // ParseNumber itself only calls SkipWS
                                // at entry, which is harmless here.
    }

    int ParseSplineInt() {
        SkipSplineWS();
        return ParseInt();
    }

    void ParseSplineBlock(Spec& attrSpec) {
        ExpectSplineChar('{');
        Spline spline;

        while (true) {
            SkipSplineWS();
            if (AtEnd()) Error("Unterminated spline block");
            if (PeekSplineChar('}')) break;

            if (PeekSplineIdent("bezier")) {
                MatchSplineIdent("bezier");
                spline.curveType = CurveType::Bezier;
            } else if (PeekSplineIdent("hermite")) {
                MatchSplineIdent("hermite");
                spline.curveType = CurveType::Hermite;
            } else if (PeekSplineIdent("pre") || PeekSplineIdent("post") ||
                       PeekSplineIdent("loop")) {
                ParseSplineDirective(spline);
            } else {
                ParseSplineKnot(spline);
            }

            if (!MatchSplineChar(',')) {
                // Trailing comma before '}' is optional; so is no comma
                // when it's the last clause.
                break;
            }
        }
        SkipSplineWS();
        ExpectSplineChar('}');

        attrSpec.SetField(FieldNames::spline, Value(std::move(spline)));
    }

    // Parse one `pre: …`, `post: …`, or `loop: (…)` clause.
    void ParseSplineDirective(Spline& spline) {
        std::string keyword = ParseSplineIdent();
        ExpectSplineChar(':');

        if (keyword == "loop") {
            ExpectSplineChar('(');
            spline.loopParameters.protoStart   = ParseSplineNumber();
            ExpectSplineChar(',');
            spline.loopParameters.protoEnd     = ParseSplineNumber();
            ExpectSplineChar(',');
            spline.loopParameters.numPreLoops  = ParseSplineInt();
            ExpectSplineChar(',');
            spline.loopParameters.numPostLoops = ParseSplineInt();
            ExpectSplineChar(',');
            spline.loopParameters.valueOffset  = ParseSplineNumber();
            ExpectSplineChar(')');
            return;
        }

        double slope = 0.0;
        ExtrapolationMode mode = ParseExtrapolationMode(slope);
        if (keyword == "pre") {
            spline.preExtrapolationMode  = mode;
            spline.preExtrapolationSlope = slope;
        } else if (keyword == "post") {
            spline.postExtrapolationMode  = mode;
            spline.postExtrapolationSlope = slope;
        } else {
            Error("Unknown spline directive: " + keyword);
        }
    }

    ExtrapolationMode ParseExtrapolationMode(double& slopeOut) {
        std::string ident = ParseSplineIdent();
        // Optional (<slope>) suffix. Only meaningful for `sloped`;
        // tolerated on other modes for resilience against real files
        // that author e.g. `held(0)`.
        if (PeekSplineChar('(')) {
            MatchSplineChar('(');
            slopeOut = ParseSplineNumber();
            ExpectSplineChar(')');
        }
        if (ident == "none")           return ExtrapolationMode::None;
        if (ident == "held")           return ExtrapolationMode::Held;
        if (ident == "linear")         return ExtrapolationMode::Linear;
        if (ident == "sloped")         return ExtrapolationMode::Sloped;
        if (ident == "looprepeat")     return ExtrapolationMode::LoopRepeat;
        if (ident == "loopreset")      return ExtrapolationMode::LoopReset;
        if (ident == "looposcillate")  return ExtrapolationMode::LoopOscillate;
        Error("Unknown extrapolation mode: " + ident);
        return ExtrapolationMode::Held;  // unreachable
    }

    InterpolationMode ParseInterpolationMode() {
        std::string ident = ParseSplineIdent();
        if (ident == "none")   return InterpolationMode::None;
        if (ident == "held")   return InterpolationMode::Held;
        if (ident == "linear") return InterpolationMode::Linear;
        if (ident == "curve")  return InterpolationMode::Curve;
        Error("Unknown interpolation mode: " + ident);
        return InterpolationMode::Held;  // unreachable
    }

    // Parse a single knot entry.
    void ParseSplineKnot(Spline& spline) {
        SplineKnot knot;
        knot.time = ParseSplineNumber();
        ExpectSplineChar(':');
        knot.value = ParseSplineNumber();
        knot.preValue = knot.value;  // single-valued unless `&` authored
        if (MatchSplineChar('&')) {
            knot.preValue = ParseSplineNumber();
        }

        // Zero or more ';'-prefixed tangent / custom-data clauses.
        while (PeekSplineChar(';')) {
            MatchSplineChar(';');
            SkipSplineWS();
            if (PeekSplineChar('{')) {
                // §12.5.3.6 per-knot custom data. Not surfaced on the
                // SplineKnot struct yet — skip the dictionary with
                // balanced-brace matching and move on.
                MatchSplineChar('{');
                int depth = 1;
                while (!AtEnd() && depth > 0) {
                    if (Peek() == '{') ++depth;
                    else if (Peek() == '}') --depth;
                    if (depth > 0) Advance();
                }
                ExpectSplineChar('}');
            } else if (PeekSplineIdent("pre")) {
                MatchSplineIdent("pre");
                ExpectSplineChar('(');
                knot.preTangentSlope = ParseSplineNumber();
                ExpectSplineChar(',');
                knot.preTangentWidth = ParseSplineNumber();
                ExpectSplineChar(')');
            } else if (PeekSplineIdent("post")) {
                MatchSplineIdent("post");
                // Optional interp mode before the tangent parens.
                if (!PeekSplineChar('(') && !PeekSplineChar(';') &&
                    !PeekSplineChar(',') && !PeekSplineChar('}')) {
                    knot.nextInterpolationMode = ParseInterpolationMode();
                }
                if (PeekSplineChar('(')) {
                    MatchSplineChar('(');
                    knot.postTangentSlope = ParseSplineNumber();
                    ExpectSplineChar(',');
                    knot.postTangentWidth = ParseSplineNumber();
                    ExpectSplineChar(')');
                }
            } else {
                Error("Expected 'pre', 'post', or '{' after ';' in knot");
            }
        }

        spline.knots.push_back(knot);
    }

    // Parse type name which may include [] for arrays
    std::string ParseTypeName() {
        std::string name = ParseIdentifier();
        SkipWhitespaceAndComments();
        if (PeekChar('[') && PeekAt(1) == ']') {
            Advance(); Advance();
            name += "[]";
        }
        return name;
    }

    void ParseRelationship(const Path& primPath, bool isCustom,
                           const std::string& listOpPrefix = "") {
        MatchIdentifier("rel");
        SkipWhitespaceAndComments();
        ParseRelationshipAfterRel(primPath, isCustom, listOpPrefix);
    }

    void ParseRelationshipAfterRel(const Path& primPath, bool isCustom,
                                   const std::string& listOpPrefix = "") {
        std::string propName = ParseNamespacedIdentifier();
        Path relPath = primPath.AppendProperty(propName);

        // For listop operations on existing relationships, check if spec already exists
        Spec* existingSpec = layer_.GetSpec(relPath);
        Spec relSpec(SpecType::Relationship);
        if (existingSpec && existingSpec->GetType() == SpecType::Relationship) {
            relSpec = *existingSpec;
        }
        relSpec.SetCustom(isCustom);

        SkipWhitespaceAndComments();

        // Optional assignment or dot-suffix
        if (PeekChar('.')) {
            Advance();
            SkipWhitespaceAndComments();
            // Could be .connect or other suffix — skip
            std::string suffix = ParseIdentifier();
            SkipWhitespaceAndComments();
            if (MatchChar('=')) {
                ParseValue(); // just consume
            }
        }
        else if (MatchChar('=')) {
            SkipWhitespaceAndComments();
            // Spec §6.6.3 + §12.2.6: targetPaths is a path-typed list op.
            // Store as ListOp<Path> so the type is preserved through
            // queries, encoding, and cross-opinion combining.
            std::vector<Path> targets;
            bool isBlocked = false;

            if (PeekIdentifier("None")) {
                MatchIdentifier("None");
                isBlocked = true;
            } else if (PeekPathRef()) {
                auto p = ParsePathRef();
                if (!p.IsEmpty()) targets.push_back(std::move(p));
            } else if (PeekChar('[')) {
                ExpectChar('[');
                while (!PeekChar(']')) {
                    SkipWhitespaceAndComments();
                    if (PeekChar(']')) break;
                    auto p = ParsePathRef();
                    if (!p.IsEmpty()) targets.push_back(std::move(p));
                    SkipWhitespaceAndComments();
                    if (PeekChar(',')) Advance();
                }
                ExpectChar(']');
            }

            if (isBlocked) {
                relSpec.SetField(FieldNames::targetPaths, Value(ValueBlock{}));
            } else if (!targets.empty()) {
                ListOp<Path> listOp;
                if (listOpPrefix.empty()) {
                    listOp.SetExplicitItems(targets);
                } else if (listOpPrefix == "prepend") {
                    listOp.SetPrependedItems(targets);
                } else if (listOpPrefix == "append") {
                    listOp.SetAppendedItems(targets);
                } else if (listOpPrefix == "delete") {
                    listOp.SetDeletedItems(targets);
                }

                // Combine with existing listop if present (multiple
                // listop-prefix lines on the same spec accumulate).
                auto* existing = relSpec.GetField(FieldNames::targetPaths);
                if (existing && !existing->IsBlock()) {
                    auto* existingOp = existing->Get<ListOp<Path>>();
                    if (existingOp) {
                        listOp = listOp.Combine(*existingOp);
                    }
                }
                relSpec.SetField(FieldNames::targetPaths, Value(std::move(listOp)));
            }
        }

        // Optional metadata — same rationale as the attribute branch:
        // single-line padding only, to preserve the trailing
        // StatementSeparator for the RequirePrimItemTerminator check.
        SkipSinglelinePadding();
        if (Peek() == '(') {
            ParsePropertyMetadata(relSpec);
        }

        layer_.SetSpec(relPath, std::move(relSpec));
    }

    void ParsePropertyMetadata(Spec& spec) {
        ExpectChar('(');
        ParseMetadataBlockContents(spec, false);
        ExpectChar(')');
    }

    // ============================================================
    // Variant Sets
    // ============================================================

    void ParseVariantSetBlock(const Path& primPath) {
        MatchIdentifier("variantSet");
        SkipWhitespaceAndComments();
        std::string setName = ParseQuotedString();
        ExpectChar('=');
        ExpectChar('{');

        while (!PeekChar('}')) {
            SkipWhitespaceAndComments();
            if (PeekChar('}')) break;

            // Each variant: "variantName" ( metadata )? { body }
            std::string variantName = ParseQuotedString();

            // Build variant path: /Prim{setName=variantName}
            Path variantSetPath = primPath.AppendVariantSelection(setName, variantName);

            Spec variantSpec(SpecType::Variant);
            variantSpec.SetSpecifier(Specifier::Def);

            SkipWhitespaceAndComments();
            if (PeekChar('(')) {
                ParsePrimMetadata(variantSpec);
            }

            SkipWhitespaceAndComments();
            if (PeekChar('{')) {
                ExpectChar('{');
                ParsePrimBody(variantSetPath, variantSpec);
                ExpectChar('}');
            }

            layer_.SetSpec(variantSetPath, std::move(variantSpec));
        }
        ExpectChar('}');
    }

    // ============================================================
    // Value parsing
    // ============================================================

    bool ParseBoolValue() {
        SkipWhitespaceAndComments();
        if (MatchIdentifier("true") || MatchIdentifier("True")) return true;
        if (MatchIdentifier("false") || MatchIdentifier("False")) return false;
        // Also handle 1/0
        if (PeekChar('1')) { Advance(); return true; }
        if (PeekChar('0')) { Advance(); return false; }
        Error("Expected boolean value");
    }

    // Parse a generic value
    Value ParseValue() {
        SkipWhitespaceAndComments();

        if (PeekIdentifier("None")) {
            MatchIdentifier("None");
            return Value(ValueBlock{});
        }
        if (PeekIdentifier("true") || PeekIdentifier("True")) {
            MatchIdentifier(PeekIdentifier("true") ? "true" : "True");
            return Value(true);
        }
        if (PeekIdentifier("false") || PeekIdentifier("False")) {
            MatchIdentifier(PeekIdentifier("false") ? "false" : "False");
            return Value(false);
        }
        if (PeekQuotedString()) {
            return Value(ParseQuotedString());
        }
        if (PeekAssetRef()) {
            return Value(Value::AssetTag{}, ParseAssetRef());
        }
        if (PeekPathRef()) {
            // Parse as string for now
            auto p = ParsePathRef();
            return Value(p.GetText());
        }
        if (PeekChar('(')) {
            return ParseTupleValue();
        }
        if (PeekChar('[')) {
            return ParseListValue();
        }
        if (PeekChar('{')) {
            auto dict = ParseDictionary();
            return Value(std::move(dict));
        }

        // Number
        if (!AtEnd() &&
            (std::isdigit(static_cast<unsigned char>(Peek())) ||
             Peek() == '-' || Peek() == '.' ||
             PeekIdentifier("inf") || PeekIdentifier("nan"))) {
            return ParseNumberValue();
        }

        // Try identifier as token value
        std::string ident;
        if (TryIdentifier(ident)) {
            return Value(ident);
        }

        Error("Unexpected value");
    }

    Value ParseNumberValue() {
        double d = ParseNumber();
        // If it fits as int and had no decimal point, store as Int
        if (std::isfinite(d) &&
            d == static_cast<int>(d) && std::abs(d) < 2e9) {
            return Value(static_cast<Int>(d));
        }
        return Value(d);
    }

    Value ParseTupleValue() {
        ExpectChar('(');
        std::vector<double> components;
        while (!PeekChar(')')) {
            if (!components.empty()) {
                if (!MatchChar(',')) break;
            }
            SkipWhitespaceAndComments();
            if (PeekChar(')')) break;
            components.push_back(ParseNumber());
        }
        ExpectChar(')');

        // Map to appropriate Vec type based on count
        switch (components.size()) {
            case 2: return Value(GfVec2d{{components[0], components[1]}});
            case 3: return Value(GfVec3d{{components[0], components[1], components[2]}});
            case 4: return Value(GfVec4d{{components[0], components[1], components[2], components[3]}});
            default:
                // Return as the first component or a generic value
                if (components.size() == 1) return Value(components[0]);
                Error("Unsupported tuple size: " + std::to_string(components.size()));
        }
    }

    Value ParseListValue() {
        ExpectChar('[');
        SkipWhitespaceAndComments();
        if (PeekChar(']')) {
            Advance();
            // Empty list
            return Value(Value::ArrayTag{}, TypeId::Unknown, std::vector<Value>{});
        }

        // Parse first element to determine list type
        std::vector<Value> values;
        values.push_back(ParseValue());
        while (MatchChar(',')) {
            SkipWhitespaceAndComments();
            if (PeekChar(']')) break;
            values.push_back(ParseValue());
        }
        ExpectChar(']');

        return Value(Value::ArrayTag{}, TypeId::Unknown, std::move(values));
    }

    Role RoleForTypeName(std::string typeName) const {
        if (typeName.size() >= 2 && typeName.substr(typeName.size() - 2) == "[]") {
            typeName.resize(typeName.size() - 2);
        }
        if (typeName == "color3d" || typeName == "color3f" ||
            typeName == "color3h" || typeName == "color4d" ||
            typeName == "color4f" || typeName == "color4h") {
            return Role::Color;
        }
        if (typeName == "normal3d" || typeName == "normal3f") {
            return Role::Normal;
        }
        if (typeName == "point3d" || typeName == "point3f") {
            return Role::Point;
        }
        if (typeName == "vector3d" || typeName == "vector3f") {
            return Role::Vector;
        }
        if (typeName == "frame4d") {
            return Role::Frame;
        }
        if (typeName == "texCoord2d" || typeName == "texCoord2f") {
            return Role::TexCoord;
        }
        return Role::None;
    }

    void ApplyRoleFromTypeName(Value& value, const std::string& typeName) const {
        if (value.IsBlock()) return;
        Role role = RoleForTypeName(typeName);
        if (role != Role::None) value.SetRole(role);
    }

    // Parse a typed value based on the type name
    Value ParseTypedValue(const std::string& typeName) {
        SkipWhitespaceAndComments();

        // Value block
        if (PeekIdentifier("None")) {
            MatchIdentifier("None");
            return Value(ValueBlock{});
        }

        // Check for array types
        bool isArray = false;
        std::string baseType = typeName;
        if (baseType.size() >= 2 && baseType.substr(baseType.size() - 2) == "[]") {
            isArray = true;
            baseType = baseType.substr(0, baseType.size() - 2);
        }

        if (isArray) {
            Value value = ParseTypedArray(baseType);
            ApplyRoleFromTypeName(value, typeName);
            return value;
        }

        // Scalar/dimensioned value
        Value value = ParseTypedScalar(baseType);
        ApplyRoleFromTypeName(value, typeName);
        return value;
    }

    Value ParseTypedScalar(const std::string& typeName) {
        SkipWhitespaceAndComments();

        // Value block
        if (PeekIdentifier("None")) {
            MatchIdentifier("None");
            return Value(ValueBlock{});
        }

        if (typeName == "bool") {
            return Value(ParseBoolValue());
        }
        if (typeName == "int") {
            return Value(static_cast<Int>(ParseInt()));
        }
        if (typeName == "uchar") {
            return Value(ParseUChar());
        }
        if (typeName == "uint") {
            return Value(ParseUInt());
        }
        if (typeName == "int64") {
            return Value(static_cast<Int64>(ParseInt64()));
        }
        if (typeName == "uint64") {
            return Value(static_cast<UInt64>(ParseUInt64()));
        }
        if (typeName == "half") {
            return Value(Half(static_cast<float>(ParseNumber())));
        }
        if (typeName == "float") {
            return Value(static_cast<Float>(ParseNumber()));
        }
        if (typeName == "double") {
            return Value(static_cast<Double>(ParseNumber()));
        }
        if (typeName == "string") {
            return Value(ParseQuotedString());
        }
        if (typeName == "token") {
            return Value(Token(ParseQuotedString()));
        }
        if (typeName == "asset") {
            return Value(Value::AssetTag{}, ParseAssetRef());
        }
        if (typeName == "timecode") {
            return Value(static_cast<TimeCode>(ParseNumber()), nullptr);
        }

        // Vec types
        if (typeName == "float2" || typeName == "texCoord2f") {
            return Value(ParseVec<Float, 2>());
        }
        if (typeName == "float3" || typeName == "point3f" || typeName == "normal3f" ||
            typeName == "vector3f" || typeName == "color3f") {
            return Value(ParseVec<Float, 3>());
        }
        if (typeName == "float4" || typeName == "color4f") {
            return Value(ParseVec<Float, 4>());
        }
        if (typeName == "double2" || typeName == "texCoord2d") {
            return Value(ParseVec<Double, 2>());
        }
        if (typeName == "double3" || typeName == "point3d" || typeName == "normal3d" ||
            typeName == "vector3d" || typeName == "color3d") {
            return Value(ParseVec<Double, 3>());
        }
        if (typeName == "double4" || typeName == "color4d") {
            return Value(ParseVec<Double, 4>());
        }
        if (typeName == "int2") {
            return Value(ParseVecInt<2>());
        }
        if (typeName == "int3") {
            return Value(ParseVecInt<3>());
        }
        if (typeName == "int4") {
            return Value(ParseVecInt<4>());
        }
        if (typeName == "half2") {
            auto v = ParseVec<Float, 2>();
            GfVec2h result;
            result[0] = Half(v[0]); result[1] = Half(v[1]);
            return Value(result);
        }
        if (typeName == "half3" || typeName == "color3h") {
            auto v = ParseVec<Float, 3>();
            GfVec3h result;
            for (int i = 0; i < 3; i++) result[i] = Half(v[i]);
            return Value(result);
        }
        if (typeName == "half4" || typeName == "color4h") {
            auto v = ParseVec<Float, 4>();
            GfVec4h result;
            for (int i = 0; i < 4; i++) result[i] = Half(v[i]);
            return Value(result);
        }

        // Quaternion types
        if (typeName == "quatf") {
            return Value(ParseQuat<Float>());
        }
        if (typeName == "quatd") {
            return Value(ParseQuat<Double>());
        }
        if (typeName == "quath") {
            auto q = ParseQuat<Float>();
            GfQuath result;
            for (int i = 0; i < 4; i++) result[i] = Half(q[i]);
            return Value(result);
        }

        // Matrix types
        if (typeName == "matrix2d") {
            return Value(ParseMatrix<2>());
        }
        if (typeName == "matrix3d") {
            return Value(ParseMatrix<3>());
        }
        if (typeName == "matrix4d" || typeName == "frame4d") {
            return Value(ParseMatrix<4>());
        }

        if (typeName == "dictionary") {
            return Value(ParseDictionary());
        }

        // Fallback: parse as generic value
        return ParseValue();
    }

    template <typename T, size_t N>
    Vec<T, N> ParseVec() {
        ExpectChar('(');
        Vec<T, N> result;
        for (size_t i = 0; i < N; i++) {
            if (i > 0) {
                SkipWhitespaceAndComments();
                MatchChar(',');
            }
            result[i] = static_cast<T>(ParseNumber());
        }
        ExpectChar(')');
        return result;
    }

    template <size_t N>
    Vec<Int, N> ParseVecInt() {
        ExpectChar('(');
        Vec<Int, N> result;
        for (size_t i = 0; i < N; i++) {
            if (i > 0) {
                SkipWhitespaceAndComments();
                MatchChar(',');
            }
            result[i] = ParseInt();
        }
        ExpectChar(')');
        return result;
    }

    template <typename T>
    Quat<T> ParseQuat() {
        // USDA text format: (w, i, j, k) — real part first
        // Internal storage: {i, j, k, r}
        ExpectChar('(');
        T vals[4];
        for (size_t i = 0; i < 4; i++) {
            if (i > 0) {
                SkipWhitespaceAndComments();
                MatchChar(',');
            }
            vals[i] = static_cast<T>(ParseNumber());
        }
        ExpectChar(')');
        Quat<T> result;
        result[0] = vals[1]; // i
        result[1] = vals[2]; // j
        result[2] = vals[3]; // k
        result[3] = vals[0]; // r = w
        return result;
    }

    template <size_t N>
    Matrix<Double, N, N> ParseMatrix() {
        // Matrices are written as nested tuples: ((r0c0, r0c1, ...), (r1c0, ...), ...)
        ExpectChar('(');
        Matrix<Double, N, N> result;
        for (size_t r = 0; r < N; r++) {
            if (r > 0) {
                SkipWhitespaceAndComments();
                MatchChar(',');
            }
            ExpectChar('(');
            for (size_t c = 0; c < N; c++) {
                if (c > 0) {
                    SkipWhitespaceAndComments();
                    MatchChar(',');
                }
                result(r, c) = ParseNumber();
            }
            ExpectChar(')');
        }
        ExpectChar(')');
        return result;
    }

    // Helper: parse array elements into a typed vector
    template <typename T, typename ParseFn>
    Value ParseTypedArrayOf(TypeId typeId, ParseFn parseFn) {
        if (PeekChar(']')) { Advance(); return Value(Value::ArrayTag{}, typeId, std::vector<T>{}); }
        std::vector<T> arr;
        arr.push_back(parseFn());
        while (MatchChar(',')) {
            SkipWhitespaceAndComments();
            if (PeekChar(']')) break;
            arr.push_back(parseFn());
        }
        ExpectChar(']');
        return Value(Value::ArrayTag{}, typeId, std::move(arr));
    }

    // Parse typed array: [value, value, ...] → contiguous typed vector
    Value ParseTypedArray(const std::string& elemType) {
        ExpectChar('[');
        SkipWhitespaceAndComments();

        // Scalar types
        if (elemType == "bool")
            return ParseTypedArrayOf<Bool>(TypeId::Bool, [this]() { return ParseBoolValue(); });
        if (elemType == "int")
            return ParseTypedArrayOf<Int>(TypeId::Int, [this]() { return static_cast<Int>(ParseInt()); });
        if (elemType == "uchar")
            return ParseTypedArrayOf<UChar>(TypeId::UChar, [this]() { return ParseUChar(); });
        if (elemType == "uint")
            return ParseTypedArrayOf<UInt>(TypeId::UInt, [this]() { return ParseUInt(); });
        if (elemType == "int64")
            return ParseTypedArrayOf<Int64>(TypeId::Int64, [this]() { return static_cast<Int64>(ParseInt64()); });
        if (elemType == "uint64")
            return ParseTypedArrayOf<UInt64>(TypeId::UInt64, [this]() { return static_cast<UInt64>(ParseUInt64()); });
        if (elemType == "half")
            return ParseTypedArrayOf<Half>(TypeId::Half, [this]() { return Half(static_cast<float>(ParseNumber())); });
        if (elemType == "float")
            return ParseTypedArrayOf<Float>(TypeId::Float, [this]() { return static_cast<Float>(ParseNumber()); });
        if (elemType == "double")
            return ParseTypedArrayOf<Double>(TypeId::Double, [this]() { return static_cast<Double>(ParseNumber()); });
        if (elemType == "string")
            return ParseTypedArrayOf<String>(TypeId::String, [this]() { return ParseQuotedString(); });
        if (elemType == "token")
            return ParseTypedArrayOf<Token>(TypeId::Token, [this]() { return Token(ParseQuotedString()); });
        if (elemType == "asset")
            return ParseTypedArrayOf<String>(TypeId::Asset, [this]() { return ParseAssetRef(); });
        if (elemType == "timecode")
            return ParseTypedArrayOf<TimeCode>(TypeId::TimeCode, [this]() {
                return static_cast<TimeCode>(ParseNumber());
            });

        // Vec types — float
        if (elemType == "float2" || elemType == "texCoord2f")
            return ParseTypedArrayOf<GfVec2f>(TypeId::Float2, [this]() { return ParseVec<Float, 2>(); });
        if (elemType == "float3" || elemType == "point3f" || elemType == "normal3f" ||
            elemType == "vector3f" || elemType == "color3f")
            return ParseTypedArrayOf<GfVec3f>(TypeId::Float3, [this]() { return ParseVec<Float, 3>(); });
        if (elemType == "float4" || elemType == "color4f")
            return ParseTypedArrayOf<GfVec4f>(TypeId::Float4, [this]() { return ParseVec<Float, 4>(); });

        // Vec types — double
        if (elemType == "double2" || elemType == "texCoord2d")
            return ParseTypedArrayOf<GfVec2d>(TypeId::Double2, [this]() { return ParseVec<Double, 2>(); });
        if (elemType == "double3" || elemType == "point3d" || elemType == "normal3d" ||
            elemType == "vector3d" || elemType == "color3d")
            return ParseTypedArrayOf<GfVec3d>(TypeId::Double3, [this]() { return ParseVec<Double, 3>(); });
        if (elemType == "double4" || elemType == "color4d")
            return ParseTypedArrayOf<GfVec4d>(TypeId::Double4, [this]() { return ParseVec<Double, 4>(); });

        // Vec types — int
        if (elemType == "int2")
            return ParseTypedArrayOf<GfVec2i>(TypeId::Int2, [this]() { return ParseVecInt<2>(); });
        if (elemType == "int3")
            return ParseTypedArrayOf<GfVec3i>(TypeId::Int3, [this]() { return ParseVecInt<3>(); });
        if (elemType == "int4")
            return ParseTypedArrayOf<GfVec4i>(TypeId::Int4, [this]() { return ParseVecInt<4>(); });

        // Vec types — half
        if (elemType == "half2")
            return ParseTypedArrayOf<GfVec2h>(TypeId::Half2, [this]() {
                auto v = ParseVec<Float, 2>();
                GfVec2h r; r[0] = Half(v[0]); r[1] = Half(v[1]); return r;
            });
        if (elemType == "half3" || elemType == "color3h")
            return ParseTypedArrayOf<GfVec3h>(TypeId::Half3, [this]() {
                auto v = ParseVec<Float, 3>();
                GfVec3h r; for (int i = 0; i < 3; i++) r[i] = Half(v[i]); return r;
            });
        if (elemType == "half4" || elemType == "color4h")
            return ParseTypedArrayOf<GfVec4h>(TypeId::Half4, [this]() {
                auto v = ParseVec<Float, 4>();
                GfVec4h r; for (int i = 0; i < 4; i++) r[i] = Half(v[i]); return r;
            });

        // Quaternion types
        if (elemType == "quatf")
            return ParseTypedArrayOf<GfQuatf>(TypeId::Quatf, [this]() { return ParseQuat<Float>(); });
        if (elemType == "quatd")
            return ParseTypedArrayOf<GfQuatd>(TypeId::Quatd, [this]() { return ParseQuat<Double>(); });
        if (elemType == "quath")
            return ParseTypedArrayOf<GfQuath>(TypeId::Quath, [this]() {
                auto q = ParseQuat<Float>();
                GfQuath r; for (int i = 0; i < 4; i++) r[i] = Half(q[i]); return r;
            });

        // Matrix types
        if (elemType == "matrix2d")
            return ParseTypedArrayOf<GfMatrix2d>(TypeId::Matrix2d, [this]() { return ParseMatrix<2>(); });
        if (elemType == "matrix3d")
            return ParseTypedArrayOf<GfMatrix3d>(TypeId::Matrix3d, [this]() { return ParseMatrix<3>(); });
        if (elemType == "matrix4d" || elemType == "frame4d")
            return ParseTypedArrayOf<GfMatrix4d>(TypeId::Matrix4d, [this]() { return ParseMatrix<4>(); });

        // Fallback for unknown element types — keep as vector<Value>
        if (PeekChar(']')) {
            Advance();
            return Value(Value::ArrayTag{}, TypeId::Unknown, std::vector<Value>{});
        }
        std::vector<Value> values;
        values.push_back(ParseTypedScalar(elemType));
        while (MatchChar(',')) {
            SkipWhitespaceAndComments();
            if (PeekChar(']')) break;
            values.push_back(ParseTypedScalar(elemType));
        }
        ExpectChar(']');
        return Value(Value::ArrayTag{}, TypeId::Unknown, std::move(values));
    }

    // Parse timeSamples map: { time: value, ... }
    TimeSamples ParseTimeSamplesMap(const std::string& typeName) {
        TimeSamples ts;
        ExpectChar('{');
        while (!PeekChar('}')) {
            SkipWhitespaceAndComments();
            if (PeekChar('}')) break;
            double time = ParseNumber();
            ExpectChar(':');
            SkipWhitespaceAndComments();
            Value val;
            if (PeekIdentifier("None")) {
                MatchIdentifier("None");
                val = Value(ValueBlock{});
            } else {
                // Try typed parsing; fall back to generic if it would fail
                // (e.g., scalar value for a vec type)
                size_t save = pos_;
                int saveLine = line_, saveCol = col_;
                try {
                    val = ParseTypedValue(typeName);
                } catch (const ParseError&) {
                    pos_ = save; line_ = saveLine; col_ = saveCol;
                    val = ParseValue();
                }
            }
            ts[time] = std::move(val);
            SkipWhitespaceAndComments();
            if (PeekChar(',')) Advance();
        }
        ExpectChar('}');
        return ts;
    }

    // Parse a dictionary: { type key = value, ... }
    Dictionary ParseDictionary() {
        Dictionary dict;
        ExpectChar('{');
        while (!PeekChar('}')) {
            SkipWhitespaceAndComments();
            if (PeekChar('}')) break;

            // Each entry: [type] key = value
            // Type is optional in some contexts
            std::string firstWord = ParseIdentifier();
            SkipWhitespaceAndComments();

            if (PeekChar('=')) {
                // firstWord is the key, no type prefix
                ExpectChar('=');
                Value val = ParseValue();
                dict[firstWord] = std::move(val);
            } else {
                // firstWord is the type, next is the key
                std::string typeName = firstWord;
                // Check for array type suffix
                if (PeekChar('[') && PeekAt(1) == ']') {
                    Advance(); Advance();
                    typeName += "[]";
                }
                // Key can be a bare identifier or a quoted string
                std::string key;
                SkipWhitespaceAndComments();
                if (PeekChar('"')) {
                    key = ParseQuotedString();
                } else {
                    key = ParseNamespacedIdentifier();
                }
                ExpectChar('=');
                Value val = ParseTypedValue(typeName);
                dict[key] = std::move(val);
            }

            SkipWhitespaceAndComments();
        }
        ExpectChar('}');
        return dict;
    }

    // --- Data ---
    std::string_view text_;
    size_t pos_;
    int line_;
    int col_;
    Layer layer_;
};

// ============================================================
// Public API
// ============================================================

UsdaParseResult ParseUsda(std::string_view text) {
    UsdaParser parser(text);
    return parser.Parse();
}

UsdaParseResult ParseUsdaFile(const ResolvedLocation& location) {
    auto resource = ReadResource(location);
    if (!resource.success) return {false, resource.error, 0, 0, {}};
    const char* data = resource.bytes.empty()
        ? ""
        : reinterpret_cast<const char*>(resource.bytes.data());
    std::string_view text(data, resource.bytes.size());
    return ParseUsda(text);
}

UsdaParseResult ParseUsdaFile(const std::string& filePath) {
    return ParseUsdaFile(ResolvedLocation::FromResolvedString(filePath));
}

} // namespace nanousd
