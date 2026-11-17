/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "PvdInputValidation.h"

#include <filesystem>
#include <random>

#include <algorithm>
#include <cstring>
#include <limits>

namespace pvdapp
{
namespace input
{

namespace
{

static const uint64_t kArchiveExtractionReserveBytes = 64ull * 1024ull * 1024ull;

} // namespace

bool isEventGroupDataSizeAllowed(uint32_t dataSize)
{
    return dataSize <= kMaxEventGroupDataSize;
}

bool canQueueEventGroup(uint64_t queuedBytes, size_t queuedGroups, uint32_t incomingBytes)
{
    if(incomingBytes > kMaxBufferedEventGroupBytes || queuedGroups >= kMaxQueuedEventGroups)
        return false;
    if(queuedBytes > kMaxQueuedEventBytes)
        return false;
    return static_cast<uint64_t>(incomingBytes) <=
           static_cast<uint64_t>(kMaxQueuedEventBytes) - queuedBytes;
}

bool canAccumulatePropertyData(uint32_t accumulatedBytes, uint32_t accumulatedItems,
                               uint32_t incomingBytes, uint32_t incomingItems)
{
    if(accumulatedBytes > kMaxAccumulatedPropertyBytes)
        return false;
    if(incomingBytes > kMaxAccumulatedPropertyBytes - accumulatedBytes)
        return false;
    if(accumulatedItems > kMaxPropertyItems)
        return false;
    return incomingItems <= kMaxPropertyItems - accumulatedItems;
}

bool checkedMultiplyToLimit(uint32_t lhs, uint32_t rhs, uint32_t limit, uint32_t& product)
{
    const uint64_t wideProduct = static_cast<uint64_t>(lhs) * rhs;
    if(wideProduct > limit || wideProduct > std::numeric_limits<uint32_t>::max())
    {
        product = 0;
        return false;
    }
    product = static_cast<uint32_t>(wideProduct);
    return true;
}

bool computeArrayAllocationSize(uint32_t objectSize, uint32_t desiredByteSize, uint32_t& capacity)
{
    capacity = 0;
    if(objectSize == 0 || desiredByteSize % objectSize != 0 ||
       desiredByteSize > kMaxPropertyArrayBytes)
        return false;
    if(desiredByteSize == 0)
        return true;

    const uint64_t objectCount = desiredByteSize / objectSize;
    const uint64_t grownObjectCount = std::max<uint64_t>(4, objectCount);
    const uint64_t grownBytes = grownObjectCount * static_cast<uint64_t>(objectSize) * 2u;

    // Growth is an optimization, not a reason to reject a valid boundary
    // request. At the ceiling, allocate exactly what was requested.
    const uint64_t candidate = grownBytes <= kMaxPropertyArrayBytes
        ? grownBytes
        : static_cast<uint64_t>(desiredByteSize);
    if(candidate < desiredByteSize || candidate > std::numeric_limits<uint32_t>::max())
        return false;

    capacity = static_cast<uint32_t>(candidate);
    return true;
}

uint32_t readPaddedU32(const void* data, size_t availableBytes)
{
    uint32_t value = 0;
    if(data && availableBytes)
        std::memcpy(&value, data, std::min(availableBytes, sizeof(value)));
    return value;
}

bool hasArchiveExtractionSpace(uint64_t uncompressedBytes, uint64_t availableBytes)
{
    // Preserve 5% of a small volume, capped at 64 MiB for larger volumes. The
    // guard is relative to actual free space rather than an arbitrary maximum
    // capture size.
    const uint64_t reserve = std::min(kArchiveExtractionReserveBytes, availableBytes / 20u);
    return uncompressedBytes <= availableBytes - reserve;
}

std::string temporaryBackingFileCandidate(const std::string& tempRoot)
{
    std::random_device random;
    std::string suffix;
    static const char digits[] = "0123456789abcdef";
    for(int word = 0; word < 4; ++word) {
        const auto value = random();
        for(int shift = 0; shift < 32; shift += 4)
            suffix += digits[(value >> shift) & 15u];
    }
    return std::filesystem::absolute(std::filesystem::u8path(tempRoot) /
        ("backingFile-" + suffix + ".pxd3")).lexically_normal().u8string();
}

} // namespace input
} // namespace pvdapp
