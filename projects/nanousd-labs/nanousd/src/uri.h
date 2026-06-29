// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <string_view>

// Internal RFC 3986 URI-reference support used by asset resolution and parser
// dispatch. This is intentionally not part of the public nanousd interface:
// callers should pass asset identifiers through AssetResolver instead.
#if defined(_WIN32)
#define NANOUSD_DETAIL_API
#else
#define NANOUSD_DETAIL_API __attribute__((visibility("hidden")))
#endif

namespace nanousd {
namespace detail {

// Parsed URI-reference components. Optional fields distinguish an absent
// component from an explicitly empty one, e.g. file:///tmp has an empty
// authority while file:/tmp has no authority component.
struct NANOUSD_DETAIL_API UriReference {
    std::optional<std::string> scheme;
    std::optional<std::string> authority;
    std::string path;
    std::optional<std::string> query;
    std::optional<std::string> fragment;
};

// Result wrapper for URI parsing/resolution. On failure, success is false and
// error contains a diagnostic suitable for internal/debug reporting.
struct NANOUSD_DETAIL_API UriParseResult {
    bool success = false;
    std::string error;
    UriReference uri;
};

// Small shared predicates used by URI-aware call sites. HasUriSchemePrefix
// follows RFC 3986 scheme syntax but intentionally treats Windows drive
// prefixes such as C: as non-URI path syntax.
NANOUSD_DETAIL_API bool StartsWith(std::string_view text, std::string_view prefix);
NANOUSD_DETAIL_API bool HasWindowsDrivePrefix(std::string_view path);
NANOUSD_DETAIL_API bool HasUriSchemePrefix(std::string_view text);

// Parse an RFC 3986 URI-reference. Absolute URI and relative-reference forms
// are accepted. ASCII URI syntax is validated; IRI handling is intentionally
// delegated to schemes/protocols rather than normalized here.
NANOUSD_DETAIL_API UriParseResult ParseUriReference(std::string_view text);

// Recompose parsed components without additional normalization or escaping.
NANOUSD_DETAIL_API std::string FormatUriReference(const UriReference& uri);

// RFC 3986 section 5.2.4 dot-segment removal for URI paths.
NANOUSD_DETAIL_API std::string RemoveUriDotSegments(std::string_view path);

// Resolve a URI-reference against an absolute base URI using RFC 3986 section
// 5.2. The file scheme is additionally constrained to the RFC 8089 shape this
// resolver supports: absolute path, optional host authority, no query/fragment.
NANOUSD_DETAIL_API UriParseResult ResolveUriReference(std::string_view baseUri,
                                                      std::string_view reference);

} // namespace detail
} // namespace nanousd

#undef NANOUSD_DETAIL_API
