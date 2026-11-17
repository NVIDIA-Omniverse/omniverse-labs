/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PVD_OVD_CANCELABLE_SOCKET_READ_STREAM_H
#define PVD_OVD_CANCELABLE_SOCKET_READ_STREAM_H

#ifdef PVD_OVD_SUPPORT

#include <cstdint>
#include <memory>
#include "OmniPvdSocketReadStream.h"
#include "OvdCaptureControl.h"

namespace pvdapp
{

// Viewer-owned live transport whose read/accept loops observe cancellation
// without relying on a cross-thread closesocket() to wake blocking recv().
class ControlSocketReadStream : public OmniPvdSocketReadStream
{
public:
    virtual uint16_t controlCapabilities() const = 0;
    virtual void queueCaptureRequest(const CaptureRequest& request) = 0;
    virtual void queueCaptureFrame(uint64_t id, std::shared_ptr<const std::vector<uint8_t>> frame) = 0;
    virtual CaptureTransportSnapshot controlTransportState() const = 0;
};

ControlSocketReadStream* createCancelableOvdSocketReadStream(uint16_t port);
void destroyCancelableOvdSocketReadStream(OmniPvdSocketReadStream& stream);

} // namespace pvdapp

#endif // PVD_OVD_SUPPORT
#endif // PVD_OVD_CANCELABLE_SOCKET_READ_STREAM_H
