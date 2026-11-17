/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "OvdCaptureControl.h"
#include <algorithm>
#include <stdexcept>

namespace pvdapp
{
bool canApplyCapture(const OvdResourceUsage& used, const OvdResourceUsage& previous, const OvdResourceLimits& limits)
{
    const auto fits = [](uint64_t current, uint64_t last, uint64_t limit, uint64_t floor) {
        // ceil(3*limit/4), without multiplying an untrusted/injected limit.
        return current < limit - limit / 4 && std::max(last, floor) <= limit - current;
    };
    return fits(used.streamBytes, previous.streamBytes, limits.maxStreamBytes, 16ull << 20) &&
        fits(used.commands, previous.commands, limits.maxCommands, 65536) &&
        fits(used.attributeMutations, previous.attributeMutations, limits.maxAttributeMutations, 65536) &&
        fits(used.payloadBytes, previous.payloadBytes, limits.maxCumulativeAttributePayloadBytes, 0) &&
        fits(used.classes, previous.classes, limits.maxClasses, 72) &&
        fits(used.attributes, previous.attributes, limits.maxAttributes, 450) &&
        fits(used.objectCreates, previous.objectCreates, limits.maxObjectCreates, 4096) &&
        fits(used.messages, previous.messages, limits.maxMessages, 0) &&
        fits(used.retainedDomBytes, previous.retainedDomBytes, limits.maxRetainedDomBytes, 32ull << 20);
}

bool canRefreshCapture(const OvdResourceUsage& used, uint64_t previousCatalogBytes, const OvdResourceLimits& limits)
{
    return used.streamBytes < limits.maxStreamBytes && previousCatalogBytes <= limits.maxStreamBytes - used.streamBytes;
}

std::vector<uint8_t> encodeCaptureRequest(const CaptureRequest& r, uint16_t capabilities)
{
    const bool hasSelectors = !r.scenes.empty() || !r.objects.empty();
    if (!r.requestId || !(capabilities & (r.catalogOnly ? 8 : 4)))
        throw std::invalid_argument("Peer does not support this request or request ID is zero");
    if (r.scenes.size() > 65536 || r.objects.size() > 65536 - r.scenes.size())
        throw std::invalid_argument("Capture selection exceeds 65536 selectors");
    if (r.catalogOnly ? (hasSelectors || r.secondary || r.all) : (r.all == hasSelectors))
        throw std::invalid_argument("Choose explicit All permitted or a nonempty selection");
    if (std::find(r.scenes.begin(), r.scenes.end(), 0) != r.scenes.end() ||
        std::find(r.objects.begin(), r.objects.end(), 0) != r.objects.end())
        throw std::invalid_argument("Zero is not a capture selector");
    const auto length = static_cast<uint32_t>(28 + 8 * (r.scenes.size() + r.objects.size()));
    std::vector<uint8_t> bytes;
    bytes.reserve(length);
    const auto append = [&bytes](uint64_t value, unsigned width) {
        for (unsigned i = 0; i < width; ++i) bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
    };
    append(r.catalogOnly ? 0x3344564f : 0x3343564f, 4);
    append(length, 4); append(r.requestId, 8); append(r.secondary ? 1 : 0, 4);
    append(r.scenes.size(), 4); append(r.objects.size(), 4);
    for (const auto handle : r.scenes) append(handle, 8);
    for (const auto handle : r.objects) append(handle, 8);
    return bytes;
}
}
