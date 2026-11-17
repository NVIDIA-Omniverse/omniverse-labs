/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifdef PVD_OVD_SUPPORT

#include "OvdCancelableSocketReadStream.h"

#include "OmniPvdSocketReadStream.h"

#include "TcpSocket.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace
{
const uint32_t kHandshakeMagic = 0x4F56444Cu; // 'O''V''D''L'
const uint16_t kLiveProtocol = 1;
const uint64_t kMaxReadChunk = 64ull * 1024;

#pragma pack(push, 1)
struct LiveHandshake
{
    uint32_t magic;
    uint16_t protocol;
    uint16_t flags;
};
#pragma pack(pop)

static_assert(sizeof(LiveHandshake) == 8, "unexpected live handshake layout");

class CancelableSocketReadStream final : public pvdapp::ControlSocketReadStream
{
public:
    explicit CancelableSocketReadStream(uint16_t port)
        : mPort(port), mCancelled(false), mOpen(false)
    {
    }

    ~CancelableSocketReadStream() override
    {
        mCancelled.store(true);
        mSocket.close();
        mAcceptor.close();
    }

    uint16_t controlCapabilities() const override { return mCapabilities.load(); }

    pvdapp::CaptureTransportSnapshot controlTransportState() const override
    {
        std::lock_guard<std::mutex> guard(mControlMutex);
        return mControl;
    }

    void queueCaptureRequest(const pvdapp::CaptureRequest& request) override
    {
        auto frame = std::make_shared<const std::vector<uint8_t>>(
            pvdapp::encodeCaptureRequest(request, controlCapabilities()));
        queueCaptureFrame(request.requestId, std::move(frame));
    }

    void queueCaptureFrame(uint64_t id, std::shared_ptr<const std::vector<uint8_t>> frame) override
    {
        std::lock_guard<std::mutex> guard(mControlMutex);
        if (id <= mControl.requestId)
            throw std::invalid_argument("Capture request IDs must increase on this connection");
        if (mCancelled.load() || mControl.state == pvdapp::CaptureTransportState::Unavailable ||
            mControl.state == pvdapp::CaptureTransportState::Sending)
            throw std::runtime_error("Capture control is unavailable or a transfer is still in progress");
        mFrame = std::move(frame);
        mControl = {pvdapp::CaptureTransportState::Sending, id, 0, mFrame->size(),
                    std::chrono::steady_clock::now(), {}};
    }

    uint64_t OMNI_PVD_CALL readBytes(uint8_t* bytes, uint64_t count) override
    {
        if (!bytes || count == 0 || !mOpen)
            return 0;
        uint64_t total = 0;
        while (total < count && !mCancelled.load())
        {
            pumpControl();
            const size_t chunk = static_cast<size_t>(
                std::min<uint64_t>(count - total, kMaxReadChunk));
            std::error_code error;
            const size_t received = mSocket.readSome(bytes + total, chunk, error);
            pumpControl();
            if (received != 0)
            {
                total += received;
                continue;
            }
            if (pvdapp::tcpRetryable(error))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            disableControl();
            break;
        }
        return total;
    }

    uint64_t OMNI_PVD_CALL skipBytes(uint64_t count) override
    {
        uint8_t scratch[4096];
        uint64_t total = 0;
        while (total < count)
        {
            const uint64_t chunk = std::min<uint64_t>(count - total, sizeof(scratch));
            const uint64_t received = readBytes(scratch, chunk);
            total += received;
            if (received != chunk)
                break;
        }
        return total;
    }

    bool OMNI_PVD_CALL openStream() override
    {
        if (mOpen)
            return true;
        if (mCancelled.load())
            return false;

        std::error_code error;
        // Anonymous remote/data-center streaming is intentional: bind IPv4 ANY.
        if (!mAcceptor.listen(mPort, false, error)) return false;

        while (!mCancelled.load())
        {
            if (mAcceptor.accept(mSocket, error))
                break;
            if (!pvdapp::tcpRetryable(error))
                return false;
            error.clear();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        if (mCancelled.load() || !mSocket.isOpen())
            return false;

        if (!mSocket.nonBlocking(true, error)) return false;
        // Bound kernel buffering too, so a paused producer cannot hide an
        // arbitrarily large control transfer behind a successful send.
        if (!mSocket.sendBufferSize(65536, error)) return false;
        mOpen = true;

        LiveHandshake handshake{};
        if (readBytes(reinterpret_cast<uint8_t*>(&handshake), sizeof(handshake))
                != sizeof(handshake)
            || handshake.magic != kHandshakeMagic
            || handshake.protocol > kLiveProtocol)
        {
            mOpen = false;
            return false;
        }
        // Protocol zero remains telemetry-only. Obsolete capability bits never
        // authorize a reverse message, and schema versions are unrelated.
        {
            std::lock_guard<std::mutex> guard(mControlMutex);
            if (mCancelled.load()) return false;
            const uint16_t capabilities = handshake.protocol == kLiveProtocol ? handshake.flags : 0;
            mControl.state = (capabilities & 12) ? pvdapp::CaptureTransportState::Idle
                                               : pvdapp::CaptureTransportState::Unsupported;
            // Publish only after queue state is ready; a polling caller may
            // enqueue immediately when it first observes these capabilities.
            mCapabilities.store(capabilities);
        }
        return true;
    }

    bool OMNI_PVD_CALL closeStream() override
    {
        // The worker's nonblocking loops observe this within two milliseconds.
        // Actual socket destruction happens after the worker is joined.
        mCancelled.store(true);
        disableControl();
        return true;
    }

private:
    void disableControl()
    {
        std::lock_guard<std::mutex> guard(mControlMutex);
        mControl.state = pvdapp::CaptureTransportState::Unavailable;
    }

    void pumpControl()
    {
        using pvdapp::CaptureTransportState;
        using Clock = std::chrono::steady_clock;
        std::shared_ptr<const std::vector<uint8_t>> frame;
        size_t offset = 0;
        {
            std::lock_guard<std::mutex> guard(mControlMutex);
            if (mControl.state != CaptureTransportState::Sending) return;
            if (Clock::now() - mControl.lastProgress >= std::chrono::seconds(30))
            {
                mControl.state = CaptureTransportState::Unavailable;
                return;
            }
            frame = mFrame;
            offset = static_cast<size_t>(mControl.bytesSent);
        }
        std::error_code error;
        const size_t sent = mSocket.writeSome(frame->data() + offset,
            std::min<size_t>(frame->size() - offset, 65536), error);
        {
            std::lock_guard<std::mutex> guard(mControlMutex);
            if (mControl.state != CaptureTransportState::Sending) return;
            if (sent)
            {
                mControl.bytesSent += sent;
                mControl.lastProgress = Clock::now();
            }
            if (error && !pvdapp::tcpRetryable(error)) mControl.state = CaptureTransportState::Unavailable;
            else if (mControl.bytesSent == mControl.totalBytes)
            {
                mControl.state = CaptureTransportState::AwaitingAck;
                mControl.sentAt = Clock::now();
                mFrame.reset(); // frame keeps storage alive until after unlocking.
            }
        }
    }

    std::atomic<uint16_t> mCapabilities{0};
    mutable std::mutex mControlMutex;
    pvdapp::CaptureTransportSnapshot mControl;
    std::shared_ptr<const std::vector<uint8_t>> mFrame;
    uint16_t mPort;
    pvdapp::TcpSocket mAcceptor;
    pvdapp::TcpSocket mSocket;
    std::atomic<bool> mCancelled;
    bool mOpen;
};
} // namespace

pvdapp::ControlSocketReadStream* pvdapp::createCancelableOvdSocketReadStream(uint16_t port)
{
    return new CancelableSocketReadStream(port);
}

void pvdapp::destroyCancelableOvdSocketReadStream(OmniPvdSocketReadStream& stream)
{
    delete &stream;
}

#endif // PVD_OVD_SUPPORT
