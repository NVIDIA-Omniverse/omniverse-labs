/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "OvdCaptureControl.h"
#include "OvdCancelableSocketReadStream.h"
#include "OvdLiveSource.h"
#include "../src/PVDSession/CaptureRequestState.h"
#include "TcpTestSupport.h"
#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <thread>

using namespace pvdapp;

TEST(CaptureRequests, UnknownRequiresFullSendAndExplicitSupersession)
{
    ovdnext::CaptureRequestState state;
    CaptureRequest request; request.requestId=1; request.all=true;
    const auto now=std::chrono::steady_clock::now();
    state.submitted(request, now);
    CaptureTransportSnapshot transport{CaptureTransportState::Sending,1,10,28,now,{}};
    state.observe(transport, {}, now+std::chrono::seconds(40), true);
    EXPECT_EQ(state.current().state,"sending");
    EXPECT_THROW(state.nextId(true),std::runtime_error);
    transport.state=CaptureTransportState::AwaitingAck; transport.sentAt=now+std::chrono::seconds(40);
    state.observe(transport, {}, now+std::chrono::seconds(49), true);
    EXPECT_EQ(state.current().state,"awaiting_ack");
    EXPECT_THROW(state.nextId(true),std::runtime_error);
    state.observe(transport, {}, now+std::chrono::seconds(50), true);
    EXPECT_EQ(state.current().state,"unknown");
    EXPECT_THROW(state.nextId(false),std::runtime_error);
    EXPECT_EQ(state.nextId(true),2u);
    request.requestId=2; state.submitted(request,now+std::chrono::seconds(50));
    CaptureMetadataSnapshot ack; ack.ackRevision=1; ack.ackId=1; ack.ackStatus=1; ack.ackEpoch=8;
    transport.requestId=2; transport.sentAt=now+std::chrono::seconds(50);
    state.observe(transport,ack,now+std::chrono::seconds(51),true);
    EXPECT_EQ(state.current().id,2u); EXPECT_EQ(state.current().state,"awaiting_ack");
    EXPECT_EQ(state.ackClassification(),"late");
    EXPECT_EQ(state.history()[0].state,"superseded");
    EXPECT_EQ(state.history()[0].ackStatus,1u);
    ack.ackRevision=2; ack.ackId=2;
    state.observe(transport,ack,now+std::chrono::seconds(52),true);
    EXPECT_EQ(state.current().state,"applied");
    EXPECT_EQ(state.nextId(false),3u);
}

TEST(CaptureRequests, LateAckSettlesUnknownButEvictedOrUnissuedIdsCannotSettleCurrent)
{
    ovdnext::CaptureRequestState state;
    const auto now=std::chrono::steady_clock::now();
    CaptureTransportSnapshot transport{CaptureTransportState::AwaitingAck,1,28,28,now,now};
    CaptureMetadataSnapshot ack;
    for(uint64_t id=1; id<=70; ++id) {
        CaptureRequest request; request.requestId=id; request.catalogOnly=true;
        state.submitted(request,now); transport.requestId=id;
        state.observe(transport,ack,now+std::chrono::seconds(11),true);
        EXPECT_EQ(state.current().state,"unknown");
        ack.ackRevision=id; ack.ackId=id; ack.ackStatus=1; ack.ackEpoch=0;
        state.observe(transport,ack,now+std::chrono::seconds(12),true);
        EXPECT_EQ(state.current().state,"applied");
    }
    EXPECT_EQ(state.highWater(),70u); ASSERT_EQ(state.history().size(),64u);
    EXPECT_EQ(state.history().front().id,7u);
    CaptureRequest request; request.requestId=71; request.all=true;
    state.submitted(request,now); transport.requestId=71;
    ack.ackRevision=71; ack.ackId=1;
    state.observe(transport,ack,now,true);
    EXPECT_EQ(state.ackClassification(),"late"); EXPECT_EQ(state.current().state,"awaiting_ack");
    ack.ackRevision=72; ack.ackId=900;
    state.observe(transport,ack,now,true);
    EXPECT_EQ(state.ackClassification(),"unknown"); EXPECT_EQ(state.current().state,"awaiting_ack");
    state.observe(transport,ack,now,false);
    EXPECT_EQ(state.current().state,"unknown");
    EXPECT_THROW(ovdnext::nextCaptureRequestId(UINT64_MAX),std::overflow_error);
    EXPECT_EQ(ovdnext::nextCaptureRequestId(UINT64_MAX-1),UINT64_MAX);
}

TEST(CaptureControl, ExactSelectionBytesPreserve64BitHandles)
{
    CaptureRequest request;
    request.requestId = 0x0102030405060708ull;
    request.secondary = true;
    request.scenes = {0x1122334455667788ull};
    request.objects = {0xfedcba9876543210ull};
    const std::vector<uint8_t> expected = {
        0x4f,0x56,0x43,0x33, 44,0,0,0, 8,7,6,5,4,3,2,1,
        1,0,0,0, 1,0,0,0, 1,0,0,0,
        0x88,0x77,0x66,0x55,0x44,0x33,0x22,0x11,
        0x10,0x32,0x54,0x76,0x98,0xba,0xdc,0xfe};
    EXPECT_EQ(encodeCaptureRequest(request, 4), expected);
}

TEST(CaptureControl, IndependentCapabilitiesAndObsoleteBits)
{
    CaptureRequest selection; selection.requestId = 1; selection.all = true;
    CaptureRequest catalog; catalog.requestId = 1; catalog.catalogOnly = true;
    for (uint16_t flags : {0,1,2,3,4,8,12,0x8000,0x800c})
    {
        if (flags & 4) EXPECT_EQ(encodeCaptureRequest(selection, flags).size(), 28u);
        else EXPECT_THROW(encodeCaptureRequest(selection, flags), std::invalid_argument);
        if (flags & 8) EXPECT_EQ(encodeCaptureRequest(catalog, flags).size(), 28u);
        else EXPECT_THROW(encodeCaptureRequest(catalog, flags), std::invalid_argument);
    }
    const std::vector<uint8_t> expected = {0x4f,0x56,0x44,0x33,28,0,0,0,1,0,0,0,0,0,0,0,
                                         0,0,0,0,0,0,0,0,0,0,0,0};
    EXPECT_EQ(encodeCaptureRequest(catalog, 8), expected);
}

TEST(CaptureControl, RejectsAmbiguousEmptyOrOversizedSelection)
{
    CaptureRequest request; request.requestId = 1;
    EXPECT_THROW(encodeCaptureRequest(request, 12), std::invalid_argument);
    request.all = true;
    EXPECT_EQ(encodeCaptureRequest(request, 12).size(), 28u);
    request.requestId = 0;
    EXPECT_THROW(encodeCaptureRequest(request, 12), std::invalid_argument);
    request.requestId = 1; request.objects = {1};
    EXPECT_THROW(encodeCaptureRequest(request, 12), std::invalid_argument);
    request.all = false; request.objects.assign(65536, 0x20000000000001ull);
    EXPECT_EQ(encodeCaptureRequest(request, 12).size(), 524316u);
    request.scenes = {1};
    EXPECT_THROW(encodeCaptureRequest(request, 12), std::invalid_argument);
    request.objects.clear(); request.scenes = {0};
    EXPECT_THROW(encodeCaptureRequest(request, 12), std::invalid_argument);
    request.scenes.clear(); request.catalogOnly = true; request.secondary = true;
    EXPECT_THROW(encodeCaptureRequest(request, 12), std::invalid_argument);
}

namespace
{
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

template<typename Predicate> bool until(Predicate predicate, Clock::duration timeout = 5s)
{
    const auto end = Clock::now() + timeout;
    do { if (predicate()) return true; std::this_thread::sleep_for(2ms); } while (Clock::now() < end);
    return predicate();
}

struct Peer
{
    tcp_test::Socket socket;
    std::unique_ptr<ControlSocketReadStream> stream;
    std::thread worker;
    std::atomic<bool> opened{false};
    std::atomic<uint64_t> forwardBytes{0}, forwardSum{0};

    explicit Peer(uint16_t flags, bool fragmented = false)
    {
        const auto port = tcp_test::reservePort();
        stream.reset(createCancelableOvdSocketReadStream(port));
        worker = std::thread([this] {
            if (!stream->openStream()) return;
            opened.store(true);
            std::array<uint8_t, 4096> data{};
            uint64_t count;
            while ((count = stream->readBytes(data.data(), data.size())) != 0)
            {
                uint64_t sum = 0;
                for (size_t i = 0; i < count; ++i) sum += data[i];
                forwardSum.fetch_add(sum); forwardBytes.fetch_add(count);
            }
        });
        const bool connected = until([&] { return tcp_test::connect(socket, port, 32768); });
        if (!connected) { stop(); throw std::runtime_error("Loopback connection failed"); }
        const std::array<uint8_t, 8> handshake{0x4c,0x44,0x56,0x4f,1,0,uint8_t(flags),uint8_t(flags >> 8)};
        if (fragmented)
            for (auto byte : handshake) { tcp_test::send(socket, &byte, 1); std::this_thread::sleep_for(2ms); }
        else tcp_test::send(socket, handshake.data(), handshake.size());
        if (!until([&] { return opened.load(); })) { stop(); throw std::runtime_error("Handshake failed"); }
        tcp_test::nonBlocking(socket, true);
    }
    ~Peer() { stop(); }
    void stop() { if (stream) stream->closeStream(); if (worker.joinable()) worker.join(); }
    std::vector<uint8_t> receive(size_t count, bool slow = false)
    {
        std::vector<uint8_t> data(count);
        size_t received = 0;
        const auto end = Clock::now() + 20s;
        while (received < count && Clock::now() < end)
        {
            const auto chunkEnd = std::min(count, received + 65536);
            while (received < chunkEnd && Clock::now() < end)
            {
                std::error_code error;
                const auto count = socket.readSome(data.data() + received, chunkEnd - received, error);
                received += count;
                if (tcpRetryable(error))
                    std::this_thread::sleep_for(2ms);
                else tcp_test::require(!error && count != 0, error);
            }
            if (slow && received < count) std::this_thread::sleep_for(500ms);
        }
        data.resize(received);
        return data;
    }
    void forward()
    {
        std::array<uint8_t, 4096> data; data.fill(0x5a);
        tcp_test::nonBlocking(socket, false);
        tcp_test::send(socket, data.data(), data.size());
        tcp_test::nonBlocking(socket, true);
    }
};

CaptureRequest maximumRequest()
{
    CaptureRequest request; request.requestId = 1;
    request.objects.assign(65536, 0x0807060504030201ull);
    return request;
}
}

TEST(CaptureControl, MaxRequestSlowDrainerCompletes)
{
    Peer peer(12, true);
    const auto start = Clock::now();
    peer.stream->queueCaptureRequest(maximumRequest());
    const auto data = peer.receive(524316, true);
    std::vector<uint8_t> expected{0x4f,0x56,0x43,0x33,0x1c,0,8,0,1,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,1,0};
    for (size_t i = 0; i < 65536; ++i) expected.insert(expected.end(), {1,2,3,4,5,6,7,8});
    EXPECT_EQ(data, expected);
    EXPECT_GT(Clock::now() - start, 3s);
    EXPECT_TRUE(until([&] { return peer.stream->controlTransportState().state == CaptureTransportState::AwaitingAck; }));
}

TEST(CaptureControl, QueueRejectsDuplicateAndBusyIds)
{
    Peer peer(12);
    peer.stream->queueCaptureRequest(maximumRequest());
    CaptureRequest next; next.requestId = 2; next.all = true;
    EXPECT_THROW(peer.stream->queueCaptureRequest(next), std::runtime_error);
    ASSERT_EQ(peer.receive(524316).size(), 524316u);
    ASSERT_TRUE(until([&] { return peer.stream->controlTransportState().state == CaptureTransportState::AwaitingAck; }));
    next.requestId = 1;
    EXPECT_THROW(peer.stream->queueCaptureRequest(next), std::invalid_argument);
    next.requestId = 0;
    EXPECT_THROW(peer.stream->queueCaptureRequest(next), std::invalid_argument);
    next.requestId = 2;
    EXPECT_NO_THROW(peer.stream->queueCaptureRequest(next));
    EXPECT_EQ(peer.receive(28).size(), 28u);
    ASSERT_TRUE(until([&] { return peer.stream->controlTransportState().state == CaptureTransportState::AwaitingAck; }));
    next.requestId = 1;
    EXPECT_THROW(peer.stream->queueCaptureRequest(next), std::invalid_argument);
    next.requestId = UINT64_MAX;
    EXPECT_NO_THROW(peer.stream->queueCaptureRequest(next));
    EXPECT_EQ(peer.receive(28).size(), 28u);
    ASSERT_TRUE(until([&] { return peer.stream->controlTransportState().state == CaptureTransportState::AwaitingAck; }));
    next.requestId = 2;
    EXPECT_THROW(peer.stream->queueCaptureRequest(next), std::invalid_argument);
}

TEST(CaptureControl, LegacyPeersReceiveZeroReverseBytes)
{
    for (uint16_t flags : {0,1,2,3,0x8000})
    {
        Peer peer(flags);
        CaptureRequest request; request.requestId = 1; request.all = true;
        EXPECT_THROW(peer.stream->queueCaptureRequest(request), std::invalid_argument);
        peer.forward();
        ASSERT_TRUE(until([&] { return peer.forwardBytes.load() == 4096; }));
        EXPECT_EQ(tcp_test::available(peer.socket), 0u);
    }
}

TEST(CaptureControl, ContinuousForwardMakesControlProgress)
{
    Peer peer(12);
    peer.stream->queueCaptureRequest(maximumRequest());
    for (unsigned i = 0; i < 100; ++i) peer.forward();
    ASSERT_EQ(peer.receive(524316).size(), 524316u);
    ASSERT_TRUE(until([&] { return peer.forwardBytes.load() == 409600; }));
    EXPECT_EQ(peer.forwardSum.load(), 409600ull * 0x5a);
}

TEST(CaptureControl, CancelPartialSendJoins)
{
    Peer peer(12);
    peer.stream->queueCaptureRequest(maximumRequest());
    ASSERT_TRUE(until([&] { return peer.stream->controlTransportState().bytesSent > 0; }));
    ASSERT_LT(peer.stream->controlTransportState().bytesSent, 524316u);
    const auto start = Clock::now();
    peer.stop();
    EXPECT_LE(Clock::now() - start, 250ms);
}

TEST(CaptureControl, StoppedReaderStallsWithoutStoppingTelemetry)
{
    Peer peer(12);
    peer.stream->queueCaptureRequest(maximumRequest());
    ASSERT_TRUE(until([&] { return peer.stream->controlTransportState().bytesSent > 0; }));
    ASSERT_LT(peer.stream->controlTransportState().bytesSent, 524316u);
    const auto deadline = Clock::now() + 35s;
    while (peer.stream->controlTransportState().state != CaptureTransportState::Unavailable && Clock::now() < deadline)
    {
        peer.forward(); std::this_thread::sleep_for(10ms);
    }
    const auto state = peer.stream->controlTransportState();
    EXPECT_EQ(state.state, CaptureTransportState::Unavailable);
    EXPECT_GE(Clock::now() - state.lastProgress, 30s);
    EXPECT_LE(Clock::now() - state.lastProgress, 31s);
    const auto before = peer.forwardBytes.load();
    peer.forward();
    EXPECT_TRUE(until([&] { return peer.forwardBytes.load() > before; }));
    CaptureRequest next; next.requestId = 2; next.all = true;
    EXPECT_THROW(peer.stream->queueCaptureRequest(next), std::runtime_error);
    EXPECT_GT(peer.forwardBytes.load(), 4096u);
}

class CaptureControlRecording : public ::testing::TestWithParam<unsigned> {};

TEST_P(CaptureControlRecording, ReverseFramesNeverEnterRecording)
{
    std::ifstream input(std::filesystem::path(TEST_DATA_DIR) / "test_scene.ovd", std::ios::binary);
    const std::vector<uint8_t> expected{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    ASSERT_FALSE(expected.empty());
    const auto port = tcp_test::reservePort();
    const auto directory = std::filesystem::temp_directory_path() /
        ("ovd-control-" + std::to_string(port) + "-" + std::to_string(Clock::now().time_since_epoch().count()));
    ASSERT_TRUE(std::filesystem::create_directory(directory));
    const auto path = directory / "capture.ovd";
    struct Cleanup {
        std::filesystem::path path, directory;
        ~Cleanup() { std::error_code error; std::filesystem::remove(path, error); std::filesystem::remove(directory, error); }
    } cleanup{path, directory};
    OmniPvdDOMState dom;
    std::recursive_mutex mutex;
    OvdLiveSource source(&dom, &mutex);
    ASSERT_TRUE(source.start(port, path.string().c_str()));
    tcp_test::Socket producer;
    ASSERT_TRUE(until([&] { return tcp_test::connect(producer, port, 32768); }));
    const std::array<uint8_t, 8> handshake{0x4c,0x44,0x56,0x4f,1,0,12,0};
    tcp_test::send(producer, handshake.data(), handshake.size());
    tcp_test::send(producer, expected.data(), expected.size());
    ASSERT_TRUE(until([&] { return source.controlCapabilities() == 12 && source.isConnected(); }));
    tcp_test::nonBlocking(producer, true);
    if (GetParam() != 0)
    {
        auto large = maximumRequest(); large.requestId = 3;
        source.queueCaptureRequest(large);
        if (GetParam() == 1)
        {
            std::array<uint8_t, 65536> buffer{};
            size_t received = 0;
            while (received < 524316)
            {
                const auto chunk = std::min<size_t>(65536, 524316 - received);
                size_t count = 0;
                ASSERT_TRUE(until([&] {
                    std::error_code error;
                    count += producer.readSome(buffer.data() + count, chunk - count, error);
                    return count == chunk;
                }));
                received += count;
                std::this_thread::sleep_for(500ms);
            }
        }
        else
        {
            ASSERT_TRUE(until([&] { return source.controlTransportState().state == CaptureTransportState::Unavailable; }, 35s));
            large.requestId = 4;
            EXPECT_THROW(source.queueCaptureRequest(large), std::runtime_error);
        }
    }
    else
    for (unsigned id = 1; id <= 2; ++id)
    {
        CaptureRequest request; request.requestId = id; request.catalogOnly = id == 2; request.all = id == 1;
        source.queueCaptureRequest(request);
        std::array<uint8_t, 28> reverse{};
        size_t count = 0;
        ASSERT_TRUE(until([&] {
            std::error_code error;
            count += producer.readSome(reverse.data() + count, reverse.size() - count, error);
            return count == reverse.size();
        }));
        EXPECT_EQ(reverse[2], id == 1 ? 0x43 : 0x44);
        EXPECT_EQ(reverse[8], id);
        ASSERT_TRUE(until([&] { return source.controlTransportState().state == CaptureTransportState::AwaitingAck; }));
    }
    producer.close();
    ASSERT_TRUE(until([&] { return source.hasEnded(); }));
    source.stop();
    EXPECT_FALSE(source.hasFailed());
    std::ifstream saved(path, std::ios::binary);
    const std::vector<uint8_t> actual{std::istreambuf_iterator<char>(saved), std::istreambuf_iterator<char>()};
    EXPECT_EQ(actual, expected);
    EXPECT_THROW(source.queueCaptureRequest(maximumRequest()), std::runtime_error);
}

INSTANTIATE_TEST_SUITE_P(CaptureControl, CaptureControlRecording, ::testing::Values(0u, 1u, 2u));
