/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "PvdCheckedOutput.h"
#include <algorithm>
#include <array>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ovdnext {
struct OutputFile::State {
    std::filesystem::path destination, temporary;
    bool overwrite = false;
    bool ownsTemporary = false;
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
    bool isOpen() const noexcept { return handle != INVALID_HANDLE_VALUE; }
    bool close() noexcept {
        const auto value = handle;
        handle = INVALID_HANDLE_VALUE;
        return value == INVALID_HANDLE_VALUE || CloseHandle(value) != 0;
    }
#else
    int handle = -1;
    bool isOpen() const noexcept { return handle >= 0; }
    bool close() noexcept {
        const int value = handle;
        handle = -1;
        // A failed close (including EINTR) must never be retried: the descriptor
        // may already have been released and reused by another thread.
        return value < 0 || ::close(value) == 0;
    }
#endif
    ~State() noexcept {
        close();
        if(ownsTemporary) {
#ifdef _WIN32
            DeleteFileW(temporary.c_str());
#else
            ::unlink(temporary.c_str());
#endif
        }
    }
};

OutputFile::OutputFile(const std::string& path, bool overwrite) : state(new State) {
    if(path.empty()) throw std::invalid_argument("Output path is empty");
    if(path.find('\0') != std::string::npos)
        throw std::invalid_argument("Output path contains a null byte");
    state->destination = std::filesystem::absolute(std::filesystem::u8path(path));
    state->overwrite = overwrite;
    const auto status = std::filesystem::symlink_status(state->destination);
    if(!overwrite && std::filesystem::exists(status))
        throw std::runtime_error("Output already exists: " + path);
#ifndef _WIN32
    struct stat previous{};
    const bool preserveMode = overwrite && ::stat(state->destination.c_str(), &previous) == 0;
    if(overwrite && !preserveMode && errno != ENOENT)
        throw std::system_error(errno, std::generic_category(), "Cannot inspect output destination");
#endif
    std::random_device random;
    for(int attempt = 0; attempt < 64; ++attempt) {
        state->temporary = state->destination;
        state->temporary += ".ovui-" + std::to_string(random()) + "-" + std::to_string(random());
#ifdef _WIN32
        state->handle = CreateFileW(state->temporary.c_str(), GENERIC_WRITE, 0,
            nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if(state->isOpen()) { state->ownsTemporary = true; return; }
        const auto code = GetLastError();
        if(code != ERROR_FILE_EXISTS && code != ERROR_ALREADY_EXISTS)
            throw std::system_error(code, std::system_category(), "Cannot create output temporary");
#else
        state->handle = ::open(state->temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if(state->isOpen()) {
            state->ownsTemporary = true;
            if(preserveMode && ::fchmod(state->handle, previous.st_mode & 07777) != 0)
                throw std::system_error(errno, std::generic_category(), "Cannot preserve output mode");
            return;
        }
        if(errno != EEXIST)
            throw std::system_error(errno, std::generic_category(), "Cannot create output temporary");
#endif
    }
    throw std::runtime_error("Cannot create exclusive output temporary");
}

OutputFile::~OutputFile() noexcept = default;

bool OutputFile::fail(const char* message) noexcept {
    if(!error) error = message;
    return false;
}

bool OutputFile::write(const uint8_t* bytes, uint32_t count) {
    if(error) return false;
    if(!state->isOpen()) return fail("Native output is closed");
    if((count && !bytes) || count > std::numeric_limits<uint64_t>::max() - length)
        return fail("Invalid native output write");
    while(count) {
        const auto bounded = std::min(count, uint32_t(64 * 1024));
#ifdef _WIN32
        DWORD actual = 0;
        if(!WriteFile(state->handle, bytes, bounded, &actual, nullptr))
            return fail("Native output write failed");
#else
        ssize_t actual;
        do { actual = ::write(state->handle, bytes, bounded); } while(actual < 0 && errno == EINTR);
        if(actual < 0) return fail("Native output write failed");
#endif
        if(!actual) return fail("Native output short write");
        length += actual;
        bytes += actual;
        count -= static_cast<uint32_t>(actual);
    }
    return true;
}

bool OutputFile::directCopy(physx::pvdsdk::PvdInputStream& input, uint32_t count) {
    return directCopy(input, uint64_t(count));
}
bool OutputFile::directCopy(physx::pvdsdk::PvdInputStream& input, uint64_t count) {
    if(error) return false;
    std::array<uint8_t, 64 * 1024> buffer;
    try {
        while(count) {
            const auto n = uint32_t(std::min<uint64_t>(count, buffer.size()));
            uint32_t actual = n;
            if(!input.read(buffer.data(), actual) || actual != n)
                return fail("Native output source read failed");
            if(!write(buffer.data(), n)) return false;
            count -= n;
        }
    } catch(...) { return fail("Native output source read failed"); }
    return true;
}

void OutputFile::zeroPadToPageSize(uint32_t page) {
    if(error) return;
    if(!page) page = 4096;
    uint32_t count = uint32_t(page - length % page);
    const std::array<uint8_t, 4096> zeros{};
    while(count) {
        const auto n = std::min(count, uint32_t(zeros.size()));
        if(!write(zeros.data(), n)) return;
        count -= n;
    }
}

pvdapp::Option<pvdapp::ErrorCode> OutputFile::flush() {
    if(error) return pvdapp::ErrorCode(1);
    if(!state->isOpen()) { fail("Native output is closed"); return pvdapp::ErrorCode(1); }
#ifdef _WIN32
    const bool ok = FlushFileBuffers(state->handle) != 0;
#else
    int result;
    do { result = ::fsync(state->handle); } while(result < 0 && errno == EINTR);
    const bool ok = result == 0;
#endif
    if(!ok) { fail("Native output flush failed"); return pvdapp::ErrorCode(1); }
    return pvdapp::None();
}

uint64_t OutputFile::commit() {
    flush();
    if(error) throw std::runtime_error(error);
    if(!length) throw std::runtime_error("Native exporter produced an empty file");
    if(!state->close()) {
        fail("Native output close failed");
        throw std::runtime_error(error);
    }
#ifdef _WIN32
    if(!MoveFileExW(state->temporary.c_str(), state->destination.c_str(),
                   state->overwrite ? MOVEFILE_REPLACE_EXISTING : 0))
        throw std::system_error(GetLastError(), std::system_category(), "Cannot publish output");
    state->ownsTemporary = false;
#else
    if(state->overwrite) {
        if(::rename(state->temporary.c_str(), state->destination.c_str()) != 0)
            throw std::system_error(errno, std::generic_category(), "Cannot publish output");
        state->ownsTemporary = false;
    } else {
        if(::link(state->temporary.c_str(), state->destination.c_str()) != 0)
            throw std::system_error(errno, std::generic_category(), "Cannot publish output");
        // Publication succeeded. A cleanup failure cannot undo that commit;
        // retain ownership so destruction retries removing only our temporary.
        if(::unlink(state->temporary.c_str()) == 0) state->ownsTemporary = false;
    }
#endif
    return length;
}
}
