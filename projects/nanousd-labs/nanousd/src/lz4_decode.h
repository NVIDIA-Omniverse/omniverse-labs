// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace nanousd {
namespace detail {

// Minimal LZ4 block-level decompressor.
// Returns the number of bytes written to dst, or a negative value on error.
// srcSize is the size of the compressed block.
// dstCapacity is the maximum number of bytes that can be written to dst.
int Lz4Decompress(const char* src, int srcSize, char* dst, int dstCapacity);

} // namespace detail
} // namespace nanousd
