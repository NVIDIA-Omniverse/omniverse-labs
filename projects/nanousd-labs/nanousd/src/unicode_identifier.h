// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Unicode 15.1 XID_Start / XID_Continue support for AOUSD identifier grammar.

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace nanousd::unicode {

struct Utf8DecodeResult {
    uint32_t codePoint = 0;
    size_t length = 0;
    bool valid = false;
};

Utf8DecodeResult DecodeUtf8(std::string_view text, size_t pos);
bool IsXidStart(uint32_t codePoint);
bool IsXidContinue(uint32_t codePoint);
bool IsIdentifierStart(uint32_t codePoint);
bool IsIdentifierContinue(uint32_t codePoint);
bool IsIdentifierStartAt(std::string_view text, size_t pos, size_t* length = nullptr);
bool IsIdentifierContinueAt(std::string_view text, size_t pos, size_t* length = nullptr);

} // namespace nanousd::unicode
