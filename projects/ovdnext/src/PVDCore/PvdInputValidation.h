/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <string>

#include <cstddef>
#include <cstdint>

namespace pvdapp
{
namespace input
{

// A PVD event group is buffered before parsing. Keep a single peer-controlled
// group comfortably below the 32-bit buffer boundary and bound queued data
// independently so a fast producer is back-pressured instead of exhausting RAM.
static const uint32_t kMaxEventGroupDataSize = 64u * 1024u * 1024u;
static const uint32_t kSerializedEventGroupHeaderSize = 24u;
static const uint32_t kMaxBufferedEventGroupBytes =
    kMaxEventGroupDataSize + kSerializedEventGroupHeaderSize;
static const uint32_t kMaxQueuedEventBytes = 256u * 1024u * 1024u;
static const uint32_t kMaxQueuedEventGroups = 1024u;

// Object-model arrays are indexed with 32-bit byte offsets throughout PVD3.
static const uint32_t kMaxPropertyArrayBytes = 512u * 1024u * 1024u;
static const uint32_t kMaxAccumulatedPropertyBytes = kMaxPropertyArrayBytes;
static const uint32_t kMaxPropertyItems = 16u * 1024u * 1024u;

// Parser-session limits prevent multiplexed captures from multiplying the
// existing per-property allowance with fresh stream IDs. The 64 MiB retained
// SPV budget matches the largest accepted event group and exceeds the total
// size of every PXD2 regression capture in this repository (largest: 42.8 MB).
static const uint32_t kMaxParserStreams = 1024u;
static const uint32_t kMaxIncompletePropertyStates = 256u;
static const uint64_t kMaxRetainedPropertyBytes = 64ull * 1024ull * 1024ull;

bool isEventGroupDataSizeAllowed(uint32_t dataSize);
bool canQueueEventGroup(uint64_t queuedBytes, size_t queuedGroups, uint32_t incomingBytes);
bool canAccumulatePropertyData(uint32_t accumulatedBytes, uint32_t accumulatedItems,
                               uint32_t incomingBytes, uint32_t incomingItems);
bool checkedMultiplyToLimit(uint32_t lhs, uint32_t rhs, uint32_t limit, uint32_t& product);

// Returns a capacity at least desiredByteSize, or false if the request is
// malformed or exceeds the supported per-property resource limit.
bool computeArrayAllocationSize(uint32_t objectSize, uint32_t desiredByteSize, uint32_t& capacity);

// Reads up to four available bytes and initializes the missing high bytes to
// zero. This is safe for legacy captures that expose a short StringHandle.
uint32_t readPaddedU32(const void* data, size_t availableBytes);

// Leaves a small amount of free space after extracting an entry. There is no
// absolute capture-size limit: large captures remain valid when the target
// volume has enough room.
bool hasArchiveExtractionSpace(uint64_t uncompressedBytes, uint64_t availableBytes);

// Generates an unpredictable candidate name without a fixed-size output
// buffer. Paths are UTF-8; the caller must still create it atomically.
std::string temporaryBackingFileCandidate(const std::string& tempRoot);

} // namespace input
} // namespace pvdapp
