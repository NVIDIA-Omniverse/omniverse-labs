// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "uri.h"

#include <algorithm>
#include <cctype>
#include <utility>
#include <vector>

namespace nanousd {
namespace detail {

bool StartsWith(std::string_view text, std::string_view prefix) {
    return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

bool HasWindowsDrivePrefix(std::string_view path) {
    return path.size() >= 2 &&
           std::isalpha(static_cast<unsigned char>(path[0])) &&
           path[1] == ':';
}

bool HasUriSchemePrefix(std::string_view text) {
    if (HasWindowsDrivePrefix(text)) return false;

    size_t colon = text.find(':');
    if (colon == std::string_view::npos) return false;

    size_t firstDelimiter = text.find_first_of("/?#");
    if (firstDelimiter != std::string_view::npos && firstDelimiter < colon) {
        return false;
    }

    if (colon == 0 || !std::isalpha(static_cast<unsigned char>(text.front()))) {
        return false;
    }
    for (size_t i = 1; i < colon; ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (std::isalnum(c) == 0 && text[i] != '+' && text[i] != '-' && text[i] != '.') {
            return false;
        }
    }
    return true;
}

namespace {

// RFC 3986 syntax is defined over ASCII character classes. Keep these helpers
// narrow so callers do not accidentally treat locale-dependent characters as
// valid URI grammar.
bool IsAlpha(char c) {
    unsigned char uc = static_cast<unsigned char>(c);
    return std::isalpha(uc) != 0;
}

bool IsDigit(char c) {
    unsigned char uc = static_cast<unsigned char>(c);
    return std::isdigit(uc) != 0;
}

bool IsHex(char c) {
    unsigned char uc = static_cast<unsigned char>(c);
    return std::isxdigit(uc) != 0;
}

bool IsSchemeChar(char c) {
    return IsAlpha(c) || IsDigit(c) || c == '+' || c == '-' || c == '.';
}

bool IsUnreserved(char c) {
    return IsAlpha(c) || IsDigit(c) || c == '-' || c == '.' || c == '_' || c == '~';
}

bool IsSubDelim(char c) {
    switch (c) {
        case '!': case '$': case '&': case '\'': case '(':
        case ')': case '*': case '+': case ',': case ';': case '=':
            return true;
        default:
            return false;
    }
}

bool IsPChar(char c) {
    return IsUnreserved(c) || IsSubDelim(c) || c == ':' || c == '@';
}

bool ValidateScheme(std::string_view text) {
    if (text.empty() || !IsAlpha(text.front())) return false;
    return std::all_of(text.begin() + 1, text.end(), IsSchemeChar);
}

std::string ToLowerAscii(std::string_view text) {
    std::string out(text);
    std::transform(out.begin(), out.end(), out.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

bool IsFileScheme(std::string_view scheme) {
    return ToLowerAscii(scheme) == "file";
}

// Validate one URI component after it has been syntactically separated. This
// accepts percent-encoded octets but does not decode them; decoding is a
// scheme-specific step performed by the asset resolver for local file URIs.
bool ValidatePercentEncoded(std::string_view text,
                            bool (*allowed)(char),
                            const char* component,
                            std::string& error) {
    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        if (c == '%') {
            if (i + 2 >= text.size() || !IsHex(text[i + 1]) || !IsHex(text[i + 2])) {
                error = std::string("Invalid percent encoding in ") + component;
                return false;
            }
            i += 2;
            continue;
        }
        if (static_cast<unsigned char>(c) >= 0x80 || !allowed(c)) {
            error = std::string("Invalid character in ") + component;
            return false;
        }
    }
    return true;
}

bool IsUserInfoChar(char c) {
    return IsUnreserved(c) || IsSubDelim(c) || c == ':';
}

bool IsRegNameChar(char c) {
    return IsUnreserved(c) || IsSubDelim(c);
}

bool IsQueryOrFragmentChar(char c) {
    return IsPChar(c) || c == '/' || c == '?';
}

bool ValidatePort(std::string_view port, std::string& error) {
    for (char c : port) {
        if (!IsDigit(c)) {
            error = "Invalid port in authority";
            return false;
        }
    }
    return true;
}

std::vector<std::string_view> Split(std::string_view text, char delimiter) {
    std::vector<std::string_view> parts;
    size_t start = 0;
    while (start <= text.size()) {
        size_t pos = text.find(delimiter, start);
        if (pos == std::string_view::npos) {
            parts.push_back(text.substr(start));
            break;
        }
        parts.push_back(text.substr(start, pos - start));
        start = pos + 1;
    }
    return parts;
}

bool ValidateH16(std::string_view text) {
    if (text.empty() || text.size() > 4) return false;
    return std::all_of(text.begin(), text.end(), IsHex);
}

bool ValidateDecOctet(std::string_view text) {
    if (text.empty() || text.size() > 3) return false;
    if (!std::all_of(text.begin(), text.end(), IsDigit)) return false;
    if (text.size() > 1 && text.front() == '0') return false;

    int value = 0;
    for (char c : text) value = value * 10 + (c - '0');
    return value <= 255;
}

bool ValidateIpv4Address(std::string_view text) {
    auto parts = Split(text, '.');
    if (parts.size() != 4) return false;
    return std::all_of(parts.begin(), parts.end(), ValidateDecOctet);
}

int CountIpv6Groups(std::string_view text, bool allowIpv4Last) {
    if (text.empty()) return 0;

    auto parts = Split(text, ':');
    int groups = 0;
    for (size_t i = 0; i < parts.size(); ++i) {
        std::string_view part = parts[i];
        if (part.empty()) return -1;

        if (part.find('.') != std::string_view::npos) {
            if (!allowIpv4Last || i + 1 != parts.size() || !ValidateIpv4Address(part)) {
                return -1;
            }
            groups += 2;
        } else {
            if (!ValidateH16(part)) return -1;
            groups += 1;
        }
    }
    return groups;
}

bool ValidateIpv6Address(std::string_view text) {
    size_t compression = text.find("::");
    if (compression == std::string_view::npos) {
        return CountIpv6Groups(text, true) == 8;
    }
    if (text.find("::", compression + 2) != std::string_view::npos) return false;

    int leftGroups = CountIpv6Groups(text.substr(0, compression), false);
    int rightGroups = CountIpv6Groups(text.substr(compression + 2), true);
    if (leftGroups < 0 || rightGroups < 0) return false;

    return leftGroups + rightGroups < 8;
}

bool IsIpFutureTailChar(char c) {
    return IsUnreserved(c) || IsSubDelim(c) || c == ':';
}

bool ValidateIpvFuture(std::string_view text) {
    if (text.size() < 4 || (text.front() != 'v' && text.front() != 'V')) return false;

    size_t dot = text.find('.');
    if (dot == std::string_view::npos || dot == 1 || dot + 1 == text.size()) return false;
    std::string_view version = text.substr(1, dot - 1);
    std::string_view tail = text.substr(dot + 1);
    if (!std::all_of(version.begin(), version.end(), IsHex)) return false;
    return std::all_of(tail.begin(), tail.end(), IsIpFutureTailChar);
}

// Host validation covers the RFC 3986 host forms needed to avoid confusing
// authorities with paths: bracketed IP literals, IPv4 addresses, and reg-name.
bool ValidateIpLiteral(std::string_view literal) {
    return ValidateIpv6Address(literal) || ValidateIpvFuture(literal);
}

bool ValidateHost(std::string_view host, std::string& error) {
    if (StartsWith(host, "[")) {
        size_t close = host.find(']');
        if (close == std::string_view::npos) {
            error = "Unterminated IP-literal in authority";
            return false;
        }
        if (close + 1 != host.size()) {
            error = "Invalid authority after IP-literal";
            return false;
        }
        std::string_view literal = host.substr(1, close - 1);
        if (!ValidateIpLiteral(literal)) {
            error = "Invalid IP-literal in authority";
            return false;
        }
        return true;
    }

    if (host.find(':') != std::string_view::npos) {
        error = "Invalid authority host";
        return false;
    }
    return ValidatePercentEncoded(host, IsRegNameChar, "authority host", error);
}

bool ValidateAuthority(std::string_view authority, std::string& error) {
    size_t at = authority.rfind('@');
    std::string_view hostPort = authority;
    if (at != std::string_view::npos) {
        std::string_view userInfo = authority.substr(0, at);
        if (!ValidatePercentEncoded(userInfo, IsUserInfoChar, "authority userinfo", error)) {
            return false;
        }
        hostPort = authority.substr(at + 1);
    }

    if (StartsWith(hostPort, "[")) {
        size_t close = hostPort.find(']');
        if (close == std::string_view::npos) {
            error = "Unterminated IP-literal in authority";
            return false;
        }
        std::string_view literal = hostPort.substr(1, close - 1);
        if (!ValidateIpLiteral(literal)) {
            error = "Invalid IP-literal in authority";
            return false;
        }
        std::string_view suffix = hostPort.substr(close + 1);
        if (!suffix.empty()) {
            if (suffix.front() != ':') {
                error = "Invalid authority after IP-literal";
                return false;
            }
            return ValidatePort(suffix.substr(1), error);
        }
        return true;
    }

    size_t colon = hostPort.find(':');
    if (colon != std::string_view::npos) {
        if (hostPort.find(':', colon + 1) != std::string_view::npos) {
            error = "Invalid authority host";
            return false;
        }
        if (!ValidatePercentEncoded(hostPort.substr(0, colon),
                                    IsRegNameChar,
                                    "authority host",
                                    error)) {
            return false;
        }
        return ValidatePort(hostPort.substr(colon + 1), error);
    }

    return ValidatePercentEncoded(hostPort, IsRegNameChar, "authority host", error);
}

// nanousd accepts the common RFC 8089 file URI forms that can be mapped by the
// default resolver: file:/absolute/path, file:///absolute/path, and
// file://host/absolute/path. Queries and fragments are rejected because they
// have no filesystem-location meaning in DefaultResolve.
bool IsPathAbsolute(std::string_view path) {
    return !path.empty() && path.front() == '/' &&
           (path.size() == 1 || path[1] != '/');
}

bool ValidateFileUri(const UriReference& uri, std::string& error) {
    if (uri.query) {
        error = "file URI must not include a query component";
        return false;
    }
    if (uri.fragment) {
        error = "file URI must not include a fragment component";
        return false;
    }
    if (!IsPathAbsolute(uri.path)) {
        error = "file URI path must be absolute";
        return false;
    }
    if (uri.authority && !ValidateHost(*uri.authority, error)) {
        return false;
    }
    return true;
}

bool ValidatePath(std::string_view path, std::string& error) {
    return ValidatePercentEncoded(path,
        [](char c) { return IsPChar(c) || c == '/'; },
        "path",
        error);
}

bool ValidatePathNoScheme(std::string_view path, std::string& error) {
    // RFC 3986 path-noscheme prevents the first segment of a relative
    // reference from containing ':', which avoids ambiguity with a scheme.
    size_t slash = path.find('/');
    std::string_view firstSegment =
        slash == std::string_view::npos ? path : path.substr(0, slash);
    if (firstSegment.find(':') != std::string_view::npos) {
        error = "Relative path first segment must not contain ':'";
        return false;
    }
    return ValidatePath(path, error);
}

void RemoveLastOutputSegment(std::string& output) {
    if (output.empty()) return;
    size_t slash = output.rfind('/');
    if (slash == std::string::npos) {
        output.clear();
    } else {
        output.erase(slash);
    }
}

std::string MergePaths(const UriReference& base, std::string_view referencePath) {
    // RFC 3986 section 5.2.3: an authority with an empty path contributes a
    // leading slash before the relative path is merged.
    if (base.authority && base.path.empty()) {
        std::string merged = "/";
        merged.append(referencePath);
        return merged;
    }

    size_t slash = base.path.rfind('/');
    if (slash == std::string::npos) return std::string(referencePath);

    std::string merged = base.path.substr(0, slash + 1);
    merged.append(referencePath);
    return merged;
}

} // namespace

// Parse URI components in dependency order: fragment, query, scheme,
// authority, then path. A colon only starts a scheme when the preceding text is
// a valid scheme; otherwise path-noscheme validation reports the ambiguity for
// relative references such as "C:foo".
UriParseResult ParseUriReference(std::string_view text) {
    UriParseResult result;

    std::string_view withoutFragment = text;
    size_t fragmentPos = withoutFragment.find('#');
    if (fragmentPos != std::string_view::npos) {
        result.uri.fragment = std::string(withoutFragment.substr(fragmentPos + 1));
        withoutFragment = withoutFragment.substr(0, fragmentPos);
    }

    std::string_view withoutQuery = withoutFragment;
    size_t queryPos = withoutQuery.find('?');
    if (queryPos != std::string_view::npos) {
        result.uri.query = std::string(withoutQuery.substr(queryPos + 1));
        withoutQuery = withoutQuery.substr(0, queryPos);
    }

    std::string_view rest = withoutQuery;
    size_t firstDelim = rest.find_first_of("/:?#");
    if (firstDelim != std::string_view::npos && rest[firstDelim] == ':') {
        std::string_view scheme = rest.substr(0, firstDelim);
        if (ValidateScheme(scheme)) {
            result.uri.scheme = std::string(scheme);
            rest.remove_prefix(firstDelim + 1);
        }
    }

    if (StartsWith(rest, "//")) {
        rest.remove_prefix(2);
        size_t pathStart = rest.find('/');
        if (pathStart == std::string_view::npos) {
            result.uri.authority = std::string(rest);
            result.uri.path.clear();
            rest = {};
        } else {
            result.uri.authority = std::string(rest.substr(0, pathStart));
            result.uri.path = std::string(rest.substr(pathStart));
            rest = {};
        }
    } else {
        result.uri.path = std::string(rest);
    }

    if (result.uri.authority) {
        if (!ValidateAuthority(*result.uri.authority, result.error)) return result;
        if (!result.uri.path.empty() && result.uri.path.front() != '/') {
            result.error = "Path with authority must be empty or begin with '/'";
            return result;
        }
        if (!ValidatePath(result.uri.path, result.error)) return result;
    } else if (result.uri.scheme) {
        if (!ValidatePath(result.uri.path, result.error)) return result;
    } else if (!result.uri.path.empty() && result.uri.path.front() != '/') {
        if (!ValidatePathNoScheme(result.uri.path, result.error)) return result;
    } else {
        if (!ValidatePath(result.uri.path, result.error)) return result;
    }

    if (result.uri.query &&
            !ValidatePercentEncoded(*result.uri.query,
                                    IsQueryOrFragmentChar,
                                    "query",
                                    result.error)) {
        return result;
    }
    if (result.uri.fragment &&
            !ValidatePercentEncoded(*result.uri.fragment,
                                    IsQueryOrFragmentChar,
                                    "fragment",
                                    result.error)) {
        return result;
    }
    if (result.uri.scheme && IsFileScheme(*result.uri.scheme) &&
            !ValidateFileUri(result.uri, result.error)) {
        return result;
    }

    result.success = true;
    return result;
}

// Formatter is deliberately lossless for parsed component values. It does not
// lowercase schemes/hosts, percent-encode, or normalize dot segments.
std::string FormatUriReference(const UriReference& uri) {
    std::string result;
    if (uri.scheme) {
        result.append(*uri.scheme);
        result.push_back(':');
    }
    if (uri.authority) {
        result.append("//");
        result.append(*uri.authority);
    }
    result.append(uri.path);
    if (uri.query) {
        result.push_back('?');
        result.append(*uri.query);
    }
    if (uri.fragment) {
        result.push_back('#');
        result.append(*uri.fragment);
    }
    return result;
}

// RFC 3986 section 5.2.4 dot-segment removal. This operates on the path
// component only; callers are responsible for preserving query and fragment.
std::string RemoveUriDotSegments(std::string_view path) {
    std::string input(path);
    std::string output;

    while (!input.empty()) {
        if (StartsWith(input, "../")) {
            input.erase(0, 3);
        } else if (StartsWith(input, "./")) {
            input.erase(0, 2);
        } else if (StartsWith(input, "/./")) {
            input.replace(0, 3, "/");
        } else if (input == "/.") {
            input.replace(0, 2, "/");
        } else if (StartsWith(input, "/../")) {
            input.replace(0, 4, "/");
            RemoveLastOutputSegment(output);
        } else if (input == "/..") {
            input.replace(0, 3, "/");
            RemoveLastOutputSegment(output);
        } else if (input == "." || input == "..") {
            input.clear();
        } else {
            size_t segmentEnd = std::string::npos;
            if (input.front() == '/') {
                segmentEnd = input.find('/', 1);
            } else {
                segmentEnd = input.find('/');
            }
            if (segmentEnd == std::string::npos) {
                output.append(input);
                input.clear();
            } else {
                output.append(input.substr(0, segmentEnd));
                input.erase(0, segmentEnd);
            }
        }
    }

    return output;
}

// RFC 3986 section 5.2 reference resolution. AOUSD's asset resolver decides
// when this applies; this function only implements URI base/reference merging.
UriParseResult ResolveUriReference(std::string_view baseUri, std::string_view reference) {
    UriParseResult base = ParseUriReference(baseUri);
    if (!base.success) {
        base.error = "Invalid base URI: " + base.error;
        return base;
    }
    if (!base.uri.scheme) {
        base.success = false;
        base.error = "Base URI must be absolute";
        return base;
    }

    UriParseResult ref = ParseUriReference(reference);
    if (!ref.success) {
        ref.error = "Invalid URI reference: " + ref.error;
        return ref;
    }

    UriReference target;
    if (ref.uri.scheme) {
        target.scheme = ref.uri.scheme;
        target.authority = ref.uri.authority;
        target.path = RemoveUriDotSegments(ref.uri.path);
        target.query = ref.uri.query;
    } else {
        if (ref.uri.authority) {
            target.authority = ref.uri.authority;
            target.path = RemoveUriDotSegments(ref.uri.path);
            target.query = ref.uri.query;
        } else {
            if (ref.uri.path.empty()) {
                target.path = base.uri.path;
                target.query = ref.uri.query ? ref.uri.query : base.uri.query;
            } else {
                if (ref.uri.path.front() == '/') {
                    target.path = RemoveUriDotSegments(ref.uri.path);
                } else {
                    target.path = RemoveUriDotSegments(MergePaths(base.uri, ref.uri.path));
                }
                target.query = ref.uri.query;
            }
            target.authority = base.uri.authority;
        }
        target.scheme = base.uri.scheme;
    }
    target.fragment = ref.uri.fragment;

    UriParseResult result;
    if (target.scheme && IsFileScheme(*target.scheme) &&
            !ValidateFileUri(target, result.error)) {
        return result;
    }

    result.success = true;
    result.uri = std::move(target);
    return result;
}

} // namespace detail
} // namespace nanousd
