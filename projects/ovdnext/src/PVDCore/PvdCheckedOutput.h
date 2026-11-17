/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "PvdRandomAccessIOStream.h"
#include <memory>
#include <cstdint>
#include <string>

namespace ovdnext {
// Scope-owned checked append-only stream; only commit publishes the output.
class OutputFile: public pvdapp::PvdRandomAccessIOStream {
    struct State;
    std::unique_ptr<State> state;
    uint64_t length = 0;
    const char* error = nullptr;
    bool fail(const char* message) noexcept;
public:
    OutputFile(const std::string& path, bool overwrite);
    ~OutputFile() noexcept override;
    void addRef() override {} // Exporters do not retain this scope-owned bridge.
    void release() override {}
    pvdapp::Option<pvdapp::ErrorCode> read(uint8_t*, uint32_t, uint64_t) override { return pvdapp::ErrorCode(1); }
    bool write(const uint8_t* bytes, uint32_t count) override;
    bool directCopy(physx::pvdsdk::PvdInputStream& input, uint32_t count) override;
    bool directCopy(physx::pvdsdk::PvdInputStream& input, uint64_t count) override;
    void zeroPadToPageSize(uint32_t page = 4096) override;
    uint64_t getLength() const override { return length; }
    pvdapp::Option<pvdapp::ErrorCode> flush() override;
    pvdapp::Option<pvdapp::ErrorCode> asyncFlush() override { return flush(); }
    double getTimeCost() override { return 0; }
    uint64_t commit();
};
}
