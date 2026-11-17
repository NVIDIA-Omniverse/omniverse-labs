/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "TestFiles.h"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <thread>
#include "TcpTestSupport.h"
#include <gtest/gtest.h>
#include <PvdEngine.h>
#include <PvdSimpleAllocator.h>
#include <PxErrorCallback.h>
#include <PxFoundation.h>
#include <PxFoundationVersion.h>
#include "OmniPvdSocketWriteStream.h"

extern "C" {
    OmniPvdSocketWriteStream* OMNI_PVD_CALL createOmniPvdSocketWriteStream(
        const char* address, uint16_t port, uint32_t sendTimeout);
    void OMNI_PVD_CALL destroyOmniPvdSocketWriteStream(OmniPvdSocketWriteStream& stream);
}

namespace live_test_support {
using namespace physx;
using namespace pvdapp;
using test_files::readFile;

class LiveCaptureTest : public ::testing::Test {
protected:
    class ErrorCallback : public PxErrorCallback {
        void reportError(PxErrorCode::Enum, const char*, const char*, int) override {}
    };
    void SetUp() override {
        m_foundation = PxCreateFoundation(PX_FOUNDATION_VERSION, m_allocator, m_errorCallback);
        ASSERT_NE(m_foundation, nullptr);
    }
    void TearDown() override { if (m_foundation) m_foundation->release(); }
    CheckingAllocator m_allocator;
    ErrorCallback m_errorCallback;
    PxFoundation* m_foundation = nullptr;
};

template <typename Predicate>
bool waitFor(Predicate predicate, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
        if (predicate()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);
    return predicate();
}

inline uint16_t testPort() {
    if (const char* overrideValue = std::getenv("OVDNEXT_TEST_LIVE_PORT")) {
        char* end = nullptr;
        const auto port = std::strtoul(overrideValue, &end, 10);
        if (end != overrideValue && *end == '\0' && port > 1024 && port <= 65535)
            return static_cast<uint16_t>(port);
    }
    return tcp_test::reservePort();
}

inline std::string readFixture(const char* path) { return readFile(path); }

inline bool replayBytes(uint16_t port, const std::string& bytes) {
    OmniPvdSocketWriteStream* stream = createOmniPvdSocketWriteStream("127.0.0.1", port, 3000);
    if (!stream) return false;
    bool ok = stream->openStream();
    uint64_t offset = 0;
    while (ok && offset < bytes.size()) {
        const uint64_t written = stream->writeBytes(
            reinterpret_cast<const uint8_t*>(bytes.data()) + offset, bytes.size() - offset);
        if (written == 0) { ok = false; break; }
        offset += written;
    }
    ok = ok && stream->flush();
    ok = stream->closeStream() && ok;
    destroyOmniPvdSocketWriteStream(*stream);
    return ok && offset == bytes.size();
}

template <typename Producer>
bool sendCapture(PvdEngine* engine, uint16_t port, Producer producer) {
    std::atomic<bool> producerOk{false};
    std::thread worker([&] { producerOk.store(producer(port)); });
    const bool ended = waitFor([&] { return engine->isLiveEnded(); }, std::chrono::seconds(10));
    if (!ended) engine->stopLive();
    worker.join();
    return ended && producerOk.load();
}
} // namespace live_test_support
