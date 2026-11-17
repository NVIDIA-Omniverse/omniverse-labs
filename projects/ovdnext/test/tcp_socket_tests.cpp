/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "TcpSocket.h"
#include "OvdCancelableSocketReadStream.h"
#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

namespace {
using pvdapp::TcpSocket;
using namespace std::chrono_literals;
template<class Predicate> bool waitSocket(Predicate predicate) {
    const auto end = std::chrono::steady_clock::now() + 2s;
    do { if (predicate()) return true; std::this_thread::sleep_for(2ms); }
    while (std::chrono::steady_clock::now() < end);
    return predicate();
}
struct Pair {
    TcpSocket listener, client, server;
    Pair() {
        std::error_code error;
        if (!listener.listen(0, true, error) ||
            !client.connect("127.0.0.1", listener.localPort(error), error) ||
            !listener.accept(server, error) || !server.nonBlocking(true, error))
            throw std::system_error(error);
    }
};
}

TEST(NativeTcp, NonBlockingAcceptAndPartialRoundTrip) {
    TcpSocket listener, client, server;
    std::error_code error;
    ASSERT_TRUE(listener.listen(0, false, error));
    EXPECT_FALSE(listener.accept(server, error));
    EXPECT_TRUE(pvdapp::tcpRetryable(error));
    EXPECT_FALSE(server.isOpen());
    const auto port = listener.localPort(error);
    ASSERT_NE(port, 0);
    ASSERT_TRUE(client.connect("127.0.0.1", port, error));
    ASSERT_TRUE(listener.accept(server, error));
    ASSERT_TRUE(server.nonBlocking(true, error));
    ASSERT_TRUE(server.sendBufferSize(65536, error));
    ASSERT_TRUE(client.receiveBufferSize(32768, error));
    std::array<unsigned char, 6> received{};
    EXPECT_EQ(server.readSome(received.data(), received.size(), error), 0u);
    EXPECT_TRUE(pvdapp::tcpRetryable(error));
    const std::array<unsigned char, 6> sent{0, 1, 127, 128, 254, 255};
    ASSERT_EQ(client.writeSome(sent.data(), sent.size(), error), sent.size());
    size_t count = 0;
    ASSERT_TRUE(waitSocket([&] {
        count += server.readSome(received.data() + count, std::min<size_t>(3, received.size() - count), error);
        return count == received.size();
    }));
    EXPECT_EQ(received, sent);
    ASSERT_EQ(server.writeSome(received.data(), received.size(), error), received.size());
    ASSERT_TRUE(waitSocket([&] { return client.available(error) == sent.size(); }));
    received.fill(0);
    ASSERT_EQ(client.readSome(received.data(), received.size(), error), received.size());
    EXPECT_EQ(received, sent);
}

TEST(NativeTcp, MoveAndCloseTransferOwnershipAndExposeEof) {
    Pair pair;
    TcpSocket moved(std::move(pair.server));
    EXPECT_FALSE(pair.server.isOpen());
    ASSERT_TRUE(moved.isOpen());
    pair.server = std::move(moved);
    EXPECT_FALSE(moved.isOpen());
    ASSERT_TRUE(pair.server.isOpen());
    pair.server.close();
    pair.server.close();
    std::error_code error;
    unsigned char byte = 0;
    EXPECT_EQ(pair.client.readSome(&byte, 1, error), 0u);
    EXPECT_FALSE(error);
}

TEST(NativeTcp, InvalidAddressAndClosedSocketReportErrors) {
    TcpSocket socket;
    std::error_code error;
    unsigned char byte = 0;
    EXPECT_EQ(socket.readSome(&byte, 1, error), 0u);
    EXPECT_TRUE(error);
    EXPECT_EQ(socket.writeSome(&byte, 1, error), 0u);
    EXPECT_TRUE(error);
    EXPECT_FALSE(socket.connect("not-an-ip-address", 5425, error));
    EXPECT_TRUE(error);
    EXPECT_FALSE(socket.isOpen());
}

TEST(NativeTcp, DisconnectedPeerSendDoesNotTerminateProcess) {
    Pair pair;
    pair.server.close();
    std::error_code error;
    ASSERT_TRUE(pair.client.nonBlocking(true, error));
    unsigned char byte = 42;
    ASSERT_TRUE(waitSocket([&] {
        return pair.client.readSome(&byte, 1, error) == 0 && !error;
    }));
    EXPECT_TRUE(waitSocket([&] {
        pair.client.writeSome(&byte, 1, error);
        return error && !pvdapp::tcpRetryable(error);
    }));
}

TEST(NativeTcp, CancelAcceptAndPartialHandshakeJoinsPromptly) {
    for (bool partialHandshake : {false, true}) {
        TcpSocket reservation, producer;
        std::error_code error;
        ASSERT_TRUE(reservation.listen(0, true, error));
        const auto port = reservation.localPort(error);
        reservation.close();
        std::unique_ptr<pvdapp::ControlSocketReadStream> stream(pvdapp::createCancelableOvdSocketReadStream(port));
        std::atomic<bool> done{false}, opened{false};
        std::thread worker([&] { opened = stream->openStream(); done = true; });
        bool connected = true;
        if (partialHandshake) {
            connected = waitSocket([&] { return producer.connect("127.0.0.1", port, error); });
            const unsigned char firstByte = 0x4c;
            if (connected) connected = producer.writeSome(&firstByte, 1, error) == 1;
        }
        const auto start = std::chrono::steady_clock::now();
        stream->closeStream();
        worker.join();
        EXPECT_TRUE(connected);
        EXPECT_TRUE(done);
        EXPECT_FALSE(opened);
        EXPECT_LT(std::chrono::steady_clock::now() - start, 250ms);
    }
}
