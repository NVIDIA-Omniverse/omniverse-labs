/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <system_error>

namespace pvdapp {
// Move-only IPv4 socket. Callers serialize access and join workers before close.
// A zero-byte read with no error is EOF; retryable errors are not EOF.
class TcpSocket {
public:
    TcpSocket() = default;
    ~TcpSocket();
    TcpSocket(const TcpSocket&) = delete;
    TcpSocket& operator=(const TcpSocket&) = delete;
    TcpSocket(TcpSocket&& other) noexcept;
    TcpSocket& operator=(TcpSocket&& other) noexcept;
    bool isOpen() const { return mHandle != invalid; }
    void close() noexcept;
    bool open(std::error_code& error);
    bool listen(uint16_t port, bool loopback, std::error_code& error);
    bool accept(TcpSocket& peer, std::error_code& error);
    bool connect(const char* address, uint16_t port, std::error_code& error);
    bool nonBlocking(bool enabled, std::error_code& error);
    bool sendBufferSize(int bytes, std::error_code& error);
    bool receiveBufferSize(int bytes, std::error_code& error);
    size_t readSome(void* data, size_t bytes, std::error_code& error);
    size_t writeSome(const void* data, size_t bytes, std::error_code& error);
    size_t available(std::error_code& error) const;
    uint16_t localPort(std::error_code& error) const;
private:
    bool option(int name, int value, std::error_code& error);
    static constexpr uintptr_t invalid = UINTPTR_MAX;
    uintptr_t mHandle = invalid;
};
bool tcpRetryable(const std::error_code& error);
}
