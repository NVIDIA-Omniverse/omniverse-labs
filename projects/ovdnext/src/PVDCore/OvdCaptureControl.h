/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PVD_OVD_CAPTURE_CONTROL_H
#define PVD_OVD_CAPTURE_CONTROL_H

#include <cstdint>
#include <chrono>
#include <vector>
#include "OvdInputValidation.h"

namespace pvdapp
{
struct CaptureRequest
{
    uint64_t requestId = 0;
    bool catalogOnly = false, secondary = false, all = false;
    std::vector<uint64_t> scenes, objects;
};

// Explicit little endian encoding; never serialize a host-layout struct.
std::vector<uint8_t> encodeCaptureRequest(const CaptureRequest& request, uint16_t capabilities);

// Conservative admission, never a guarantee that an unseen snapshot will fit.
bool canApplyCapture(const OvdResourceUsage& used, const OvdResourceUsage& previousSegment,
                     const OvdResourceLimits& limits);
bool canRefreshCapture(const OvdResourceUsage& used, uint64_t previousCatalogBytes,
                       const OvdResourceLimits& limits);
struct CaptureBudgetSnapshot
{
    OvdResourceUsage used{}, previousSegment{};
    OvdResourceLimits limits = OvdResourceLimits::defaults();
    uint64_t previousCatalogBytes = 0;
};

enum class CaptureTransportState { Unsupported, Idle, Sending, AwaitingAck, Unavailable };
struct CaptureTransportSnapshot
{
    CaptureTransportState state = CaptureTransportState::Unsupported;
    uint64_t requestId = 0, bytesSent = 0, totalBytes = 0;
    std::chrono::steady_clock::time_point lastProgress{}, sentAt{};
};
}
#endif
