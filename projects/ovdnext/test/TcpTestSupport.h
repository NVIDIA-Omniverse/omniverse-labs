/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "TcpSocket.h"
#include <chrono>
#include <thread>

namespace tcp_test {
using Socket = pvdapp::TcpSocket;
inline void require(bool success, const std::error_code& error) {
    if (!success) throw std::system_error(error ? error : std::make_error_code(std::errc::io_error));
}
inline uint16_t reservePort() {
    Socket probe;
    std::error_code error;
    require(probe.listen(0, true, error), error);
    const auto port = probe.localPort(error);
    require(!error && port != 0, error);
    return port; // Probe-to-bind race remains; PvdEngine cannot adopt this socket.
}
inline bool connect(Socket& socket, uint16_t port, int receiveBuffer = 0) {
    socket.close();
    std::error_code error;
    return socket.open(error) && (!receiveBuffer || socket.receiveBufferSize(receiveBuffer, error)) &&
        socket.connect("127.0.0.1", port, error);
}
inline void nonBlocking(Socket& socket, bool enabled) {
    std::error_code error;
    require(socket.nonBlocking(enabled, error), error);
}
inline size_t available(Socket& socket) {
    std::error_code error;
    const auto bytes = socket.available(error);
    require(!error, error);
    return bytes;
}
inline void send(Socket& socket, const void* data, size_t bytes) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    size_t offset = 0;
    while (offset < bytes) {
        std::error_code error;
        const auto count = socket.writeSome(static_cast<const char*>(data) + offset, bytes - offset, error);
        offset += count;
        if (pvdapp::tcpRetryable(error)) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        else require(!error && count != 0, error);
        if (std::chrono::steady_clock::now() >= end) throw std::runtime_error("TCP test send timed out");
    }
}
}
