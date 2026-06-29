// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanousd/usd_parser.h"
#include "nanousd/resource.h"
#include "nanousd/usda_parser.h"
#include "nanousd/usdc_parser.h"
#include "nanousd/usdz_package.h"
#include "uri.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

namespace nanousd {

namespace {

// Return the lowercase file extension (including the dot), e.g. ".usda".
std::string GetLowerExtension(const std::string& path) {
    std::string pathComponent = path;
    auto parsed = detail::ParseUriReference(path);
    if (parsed.success) pathComponent = parsed.uri.path;

    auto dot = pathComponent.rfind('.');
    if (dot == std::string::npos) return "";
    std::string ext = pathComponent.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

UsdParseResult FromUsda(const ResolvedLocation& location) {
    auto r = ParseUsdaFile(location);
    UsdParseResult result;
    result.success = r.success;
    result.error   = std::move(r.error);
    result.line    = r.line;
    result.column  = r.column;
    result.layer   = std::move(r.layer);
    return result;
}

UsdParseResult FromUsdc(const ResolvedLocation& location) {
    auto r = ParseUsdcFile(location);
    UsdParseResult result;
    result.success = r.success;
    result.error   = std::move(r.error);
    result.layer   = std::move(r.layer);
    return result;
}

UsdParseResult FromUsdcBuffer(const uint8_t* data, size_t size) {
    auto r = ParseUsdc(data, size);
    UsdParseResult result;
    result.success = r.success;
    result.error   = std::move(r.error);
    result.layer   = std::move(r.layer);
    return result;
}

UsdParseResult FromUsdaBuffer(const uint8_t* data, size_t size) {
    const char* text = size == 0 ? "" : reinterpret_cast<const char*>(data);
    auto r = ParseUsda(std::string_view(text, size));
    UsdParseResult result;
    result.success = r.success;
    result.error   = std::move(r.error);
    result.line    = r.line;
    result.column  = r.column;
    result.layer   = std::move(r.layer);
    return result;
}

UsdParseResult FromUsdz(const ResolvedLocation& location) {
    auto package = ReadUsdzFile(location.DisplayString());
    UsdParseResult result;
    if (!package.success) {
        result.error = package.error;
        return result;
    }
    if (IsUsdcFormat(package.data.data(), package.data.size())) {
        return FromUsdcBuffer(package.data.data(), package.data.size());
    }
    return FromUsdaBuffer(package.data.data(), package.data.size());
}

} // anonymous namespace

UsdParseResult ParseUsdFile(const ResolvedLocation& location) {
    std::string ext = GetLowerExtension(location.DisplayString());

    // Fast path: known extensions dispatch directly without reading the
    // whole file just to inspect the header.
    if (location.IsPackage() || HasUsdzExtension(location.DisplayString()))
        return FromUsdz(location);
    if (ext == ".usda") return FromUsda(location);
    if (ext == ".usdc") return FromUsdc(location);

    // For .usd (or unknown extensions), detect format from resource content
    // per spec Section 16.1: check for the "PXR-USDC" magic bytes.
    auto resource = ReadResource(location);
    if (!resource.success) {
        UsdParseResult result;
        result.error = std::move(resource.error);
        return result;
    }

    if (IsUsdcFormat(resource.bytes.data(), resource.bytes.size())) {
        if (IsLocalFileResource(location)) return FromUsdc(location);
        return FromUsdcBuffer(resource.bytes.data(), resource.bytes.size());
    }

    return FromUsdaBuffer(resource.bytes.data(), resource.bytes.size());
}

UsdParseResult ParseUsdFile(const std::string& filePath) {
    return ParseUsdFile(ResolvedLocation::FromResolvedString(filePath));
}

} // namespace nanousd
