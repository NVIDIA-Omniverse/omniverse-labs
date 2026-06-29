// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// mmap_handle.h — RAII mmap wrapper for memory-mapped file I/O.
// Replaces heap-allocated file buffers with OS-managed page cache.
// Falls back gracefully: caller checks valid() before use.

#ifndef NANOUSD_MMAP_HANDLE_H
#define NANOUSD_MMAP_HANDLE_H

#include <cstddef>
#include <cstdint>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace nanousd {

class MmapHandle {
public:
    MmapHandle() = default;

    explicit MmapHandle(const char* path) {
#ifdef _WIN32
        file_ = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
                             NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (file_ == INVALID_HANDLE_VALUE) return;
        LARGE_INTEGER fileSize;
        if (!GetFileSizeEx(file_, &fileSize) || fileSize.QuadPart == 0) {
            CloseHandle(file_); file_ = INVALID_HANDLE_VALUE; return;
        }
        size_ = static_cast<size_t>(fileSize.QuadPart);
        mapping_ = CreateFileMappingA(file_, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!mapping_) {
            size_ = 0; CloseHandle(file_); file_ = INVALID_HANDLE_VALUE; return;
        }
        data_ = static_cast<const uint8_t*>(
            MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
        if (!data_) {
            size_ = 0;
            CloseHandle(mapping_); mapping_ = NULL;
            CloseHandle(file_); file_ = INVALID_HANDLE_VALUE;
        }
#else
        fd_ = ::open(path, O_RDONLY);
        if (fd_ < 0) return;
        struct stat st;
        if (::fstat(fd_, &st) < 0 || st.st_size == 0) { ::close(fd_); fd_ = -1; return; }
        size_ = static_cast<size_t>(st.st_size);
        data_ = static_cast<const uint8_t*>(
            ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (data_ == MAP_FAILED) { data_ = nullptr; size_ = 0; ::close(fd_); fd_ = -1; }
        else {
            // The mapping remains valid after close; do not pin one fd per layer.
            ::close(fd_);
            fd_ = -1;
        }
#endif
    }

    ~MmapHandle() {
#ifdef _WIN32
        if (data_) UnmapViewOfFile(data_);
        if (mapping_) CloseHandle(mapping_);
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
#else
        if (data_) ::munmap(const_cast<uint8_t*>(data_), size_);
        if (fd_ >= 0) ::close(fd_);
#endif
    }

    // Non-copyable, movable
    MmapHandle(const MmapHandle&) = delete;
    MmapHandle& operator=(const MmapHandle&) = delete;
    MmapHandle(MmapHandle&& o) noexcept
        : data_(o.data_), size_(o.size_)
#ifdef _WIN32
        , file_(o.file_), mapping_(o.mapping_)
#else
        , fd_(o.fd_)
#endif
    {
        o.data_ = nullptr; o.size_ = 0;
#ifdef _WIN32
        o.file_ = INVALID_HANDLE_VALUE; o.mapping_ = NULL;
#else
        o.fd_ = -1;
#endif
    }
    MmapHandle& operator=(MmapHandle&& o) noexcept {
        if (this != &o) {
#ifdef _WIN32
            if (data_) UnmapViewOfFile(data_);
            if (mapping_) CloseHandle(mapping_);
            if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
            file_ = o.file_; mapping_ = o.mapping_;
            o.file_ = INVALID_HANDLE_VALUE; o.mapping_ = NULL;
#else
            if (data_) ::munmap(const_cast<uint8_t*>(data_), size_);
            if (fd_ >= 0) ::close(fd_);
            fd_ = o.fd_;
            o.fd_ = -1;
#endif
            data_ = o.data_; size_ = o.size_;
            o.data_ = nullptr; o.size_ = 0;
        }
        return *this;
    }

    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
    bool valid() const { return data_ != nullptr; }

    void advise_sequential() const {
#ifndef _WIN32
        if (data_) ::madvise(const_cast<uint8_t*>(data_), size_, MADV_SEQUENTIAL);
#endif
    }

    void advise_random() const {
#ifndef _WIN32
        if (data_) ::madvise(const_cast<uint8_t*>(data_), size_, MADV_RANDOM);
#endif
    }

private:
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
#ifdef _WIN32
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = NULL;
#else
    int fd_ = -1;
#endif
};

} // namespace nanousd

#endif // NANOUSD_MMAP_HANDLE_H
