/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

namespace test_files {
namespace fs = std::filesystem;

// Own only the directory created by this instance; never reuse an existing path.
class Directory {
    fs::path owned;
public:
    explicit Directory(const std::string& prefix = "ovdnext-test-") {
        std::error_code error;
        const auto parent = fs::temp_directory_path(error);
        if (error) return;
        std::random_device random;
        for (int attempt = 0; attempt < 64; ++attempt) {
            const auto candidate = parent / fs::u8path(prefix + std::to_string(random()) + "-" + std::to_string(random()));
            if (fs::create_directory(candidate, error)) { owned = candidate; return; }
            if (error && error != std::errc::file_exists) return;
        }
    }
    ~Directory() { if (valid()) { std::error_code error; fs::remove_all(owned, error); } }
    Directory(const Directory&) = delete;
    Directory& operator=(const Directory&) = delete;
    bool valid() const { return !owned.empty(); }
    std::string path() const { return owned.u8string(); }
    std::string file(const std::string& name) const { return (owned / fs::u8path(name)).u8string(); }
};

class File {
    fs::path path;
    std::fstream stream;
public:
    explicit File(const std::string& name) : path(fs::u8path(name)) {}
    bool open(std::ios::openmode mode) { stream.open(path, mode | std::ios::binary); return stream.is_open(); }
    void close() { stream.close(); }
    bool flush() { stream.flush(); return bool(stream); }
    std::streamsize write(const char* data, std::streamsize size) {
        stream.write(data, size);
        return stream ? size : -1;
    }
    std::streamsize write(const std::string& data) { return write(data.data(), data.size()); }
    std::string read(std::streamsize size) {
        std::string bytes(static_cast<size_t>(size), '\0');
        stream.read(bytes.data(), size);
        bytes.resize(static_cast<size_t>(stream.gcount()));
        return bytes;
    }
    std::string readAll() { return {std::istreambuf_iterator<char>(stream), {}}; }
    bool seek(std::streamoff offset) { stream.seekg(offset); return bool(stream); }
};

inline std::string readFile(const std::string& path) {
    File file(path);
    return file.open(std::ios::in) ? file.readAll() : std::string();
}
inline std::vector<std::string> fileNames(const std::string& path) {
    std::vector<std::string> names;
    for (const auto& entry : fs::directory_iterator(fs::u8path(path)))
        if (entry.is_regular_file()) names.push_back(entry.path().filename().u8string());
    std::sort(names.begin(), names.end());
    return names;
}
} // namespace test_files
