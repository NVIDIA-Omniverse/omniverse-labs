// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Minimal LZ4 block-level decompressor.
// Implements the LZ4 block format as specified by https://lz4.org
// Only decompression is needed (no compression).

#include "lz4_decode.h"

#include <cstring>

namespace nanousd {
namespace detail {

int Lz4Decompress(const char* src, int srcSize, char* dst, int dstCapacity) {
    const uint8_t* ip = reinterpret_cast<const uint8_t*>(src);
    const uint8_t* const iend = ip + srcSize;
    uint8_t* op = reinterpret_cast<uint8_t*>(dst);
    uint8_t* const oend = op + dstCapacity;

    while (ip < iend) {
        // Read the sequence byte
        uint8_t token = *ip++;

        // Literal length
        int literalLen = token >> 4;
        if (literalLen == 15) {
            while (ip < iend) {
                uint8_t extra = *ip++;
                literalLen += extra;
                if (extra != 255) break;
            }
        }

        // Copy literals
        if (literalLen > 0) {
            if (ip + literalLen > iend) return -1;  // source overflow
            if (op + literalLen > oend) return -1;   // destination overflow
            std::memcpy(op, ip, literalLen);
            ip += literalLen;
            op += literalLen;
        }

        // Check if this was the last sequence (no match section)
        if (ip >= iend) break;

        // Match offset (2 bytes, little-endian)
        if (ip + 2 > iend) return -1;
        int offset = ip[0] | (ip[1] << 8);
        ip += 2;
        if (offset == 0) return -1;  // invalid offset

        uint8_t* match = op - offset;
        if (match < reinterpret_cast<uint8_t*>(dst)) return -1;  // offset before start

        // Match length (minimum 4)
        int matchLen = (token & 0x0F) + 4;
        if (matchLen == 19) {  // 15 + 4
            while (ip < iend) {
                uint8_t extra = *ip++;
                matchLen += extra;
                if (extra != 255) break;
            }
        }

        if (op + matchLen > oend) return -1;  // destination overflow

        // Copy match (may overlap — byte-by-byte for overlapping copies)
        if (offset >= matchLen) {
            // Non-overlapping: safe to memcpy
            std::memcpy(op, match, matchLen);
            op += matchLen;
        } else {
            // Overlapping: must copy byte by byte
            for (int i = 0; i < matchLen; ++i) {
                *op++ = *match++;
            }
        }
    }

    return static_cast<int>(op - reinterpret_cast<uint8_t*>(dst));
}

} // namespace detail
} // namespace nanousd
