/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "TcpSocket.h"
#include <algorithm>
#include <climits>
#include <utility>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace pvdapp {
namespace {
#ifdef _WIN32
using Handle = SOCKET;
using AddressSize = int;
struct Winsock {
    WSADATA data{};
    int error = WSAStartup(MAKEWORD(2, 2), &data);
    ~Winsock() { if (!error) WSACleanup(); }
};
int lastError() { return WSAGetLastError(); }
#else
using Handle = int;
using AddressSize = socklen_t;
int lastError() { return errno; }
#endif
bool result(bool success, std::error_code& error) {
    error = success ? std::error_code{} : std::error_code(lastError(), std::system_category());
    return success;
}
}

bool tcpRetryable(const std::error_code& error) {
    if (!error) return false;
#ifdef _WIN32
    return error.value() == WSAEWOULDBLOCK || error.value() == WSAEINTR;
#else
    return error.value() == EAGAIN || error.value() == EWOULDBLOCK || error.value() == EINTR;
#endif
}

TcpSocket::~TcpSocket() { close(); }
TcpSocket::TcpSocket(TcpSocket&& other) noexcept : mHandle(std::exchange(other.mHandle, invalid)) {}
TcpSocket& TcpSocket::operator=(TcpSocket&& other) noexcept {
    if (this != &other) { close(); mHandle = std::exchange(other.mHandle, invalid); }
    return *this;
}
void TcpSocket::close() noexcept {
    if (!isOpen()) return;
    const auto handle = static_cast<Handle>(std::exchange(mHandle, invalid));
#ifdef _WIN32
    ::shutdown(handle, SD_BOTH);
    ::closesocket(handle);
#else
    ::shutdown(handle, SHUT_RDWR);
    ::close(handle);
#endif
}
bool TcpSocket::open(std::error_code& error) {
    close();
#ifdef _WIN32
    static Winsock runtime;
    if (runtime.error) { error = {runtime.error, std::system_category()}; return false; }
#endif
    const Handle handle = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (!result(handle != static_cast<Handle>(-1), error)) return false;
    mHandle = static_cast<uintptr_t>(handle);
#if defined(SO_NOSIGPIPE) && !defined(MSG_NOSIGNAL)
    if (!option(SO_NOSIGPIPE, 1, error)) { close(); return false; }
#endif
    return true;
}
bool TcpSocket::option(int name, int value, std::error_code& error) {
    return result(::setsockopt(static_cast<Handle>(mHandle), SOL_SOCKET, name,
        reinterpret_cast<const char*>(&value), sizeof(value)) == 0, error);
}
bool TcpSocket::sendBufferSize(int bytes, std::error_code& error) { return option(SO_SNDBUF, bytes, error); }
bool TcpSocket::receiveBufferSize(int bytes, std::error_code& error) { return option(SO_RCVBUF, bytes, error); }
bool TcpSocket::nonBlocking(bool enabled, std::error_code& error) {
#ifdef _WIN32
    u_long value = enabled ? 1 : 0;
    return result(::ioctlsocket(static_cast<Handle>(mHandle), FIONBIO, &value) == 0, error);
#else
    const int flags = ::fcntl(static_cast<Handle>(mHandle), F_GETFL, 0);
    if (!result(flags != -1, error)) return false;
    return result(::fcntl(static_cast<Handle>(mHandle), F_SETFL,
        enabled ? flags | O_NONBLOCK : flags & ~O_NONBLOCK) != -1, error);
#endif
}
bool TcpSocket::listen(uint16_t port, bool loopback, std::error_code& error) {
    if (!open(error)) return false;
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(loopback ? INADDR_LOOPBACK : INADDR_ANY);
    if (!option(SO_REUSEADDR, 1, error) ||
        !result(::bind(static_cast<Handle>(mHandle), reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0, error) ||
        !result(::listen(static_cast<Handle>(mHandle), 1) == 0, error) || !nonBlocking(true, error)) {
        close(); return false;
    }
    return true;
}
bool TcpSocket::accept(TcpSocket& peer, std::error_code& error) {
    if (peer.isOpen()) { error = std::make_error_code(std::errc::already_connected); return false; }
    const Handle handle = ::accept(static_cast<Handle>(mHandle), nullptr, nullptr);
    if (!result(handle != static_cast<Handle>(-1), error)) return false;
    peer.mHandle = static_cast<uintptr_t>(handle);
#if defined(SO_NOSIGPIPE) && !defined(MSG_NOSIGNAL)
    if (!peer.option(SO_NOSIGPIPE, 1, error)) { peer.close(); return false; }
#endif
    return true;
}
bool TcpSocket::connect(const char* ip, uint16_t port, std::error_code& error) {
    if (!isOpen() && !open(error)) return false;
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    if (!ip || ::inet_pton(AF_INET, ip, &address.sin_addr) != 1) {
        error = std::make_error_code(std::errc::invalid_argument); close(); return false;
    }
    if (!result(::connect(static_cast<Handle>(mHandle), reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0, error)) {
        close(); return false;
    }
    return true;
}
size_t TcpSocket::readSome(void* data, size_t bytes, std::error_code& error) {
    error.clear();
    if (!isOpen()) { error = std::make_error_code(std::errc::bad_file_descriptor); return 0; }
    if (!bytes) return 0;
    const auto count = ::recv(static_cast<Handle>(mHandle), static_cast<char*>(data),
        static_cast<int>(std::min<size_t>(bytes, INT_MAX)), 0);
    return result(count >= 0, error) ? static_cast<size_t>(count) : 0;
}
size_t TcpSocket::writeSome(const void* data, size_t bytes, std::error_code& error) {
    error.clear();
    if (!isOpen()) { error = std::make_error_code(std::errc::bad_file_descriptor); return 0; }
    if (!bytes) return 0;
#ifdef MSG_NOSIGNAL
    const int flags = MSG_NOSIGNAL;
#else
    const int flags = 0;
#endif
    const auto count = ::send(static_cast<Handle>(mHandle), static_cast<const char*>(data),
        static_cast<int>(std::min<size_t>(bytes, INT_MAX)), flags);
    return result(count >= 0, error) ? static_cast<size_t>(count) : 0;
}
size_t TcpSocket::available(std::error_code& error) const {
#ifdef _WIN32
    u_long bytes = 0;
    const bool ok = result(::ioctlsocket(static_cast<Handle>(mHandle), FIONREAD, &bytes) == 0, error);
#else
    int bytes = 0;
    const bool ok = result(::ioctl(static_cast<Handle>(mHandle), FIONREAD, &bytes) == 0, error);
#endif
    return ok ? static_cast<size_t>(bytes) : 0;
}
uint16_t TcpSocket::localPort(std::error_code& error) const {
    sockaddr_in address{};
    AddressSize length = sizeof(address);
    return result(::getsockname(static_cast<Handle>(mHandle), reinterpret_cast<sockaddr*>(&address), &length) == 0, error)
        ? ntohs(address.sin_port) : 0;
}
}
