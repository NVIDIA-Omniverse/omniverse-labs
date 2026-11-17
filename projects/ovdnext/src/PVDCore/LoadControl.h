/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

namespace pvdapp {

struct LoadProgress {
    std::string stage;
    uint64_t processed = 0;
    uint64_t total = 0;
    // Indexed legacy loading has no safe interruption point inside its reader.
    // Cancellation is honored immediately before and after that stage.
    bool interruptible = true;
};

// Shared native control; no Python callback or interpreter access during parsing.
class LoadControl {
public:
    void cancel() { mCancelled.store(true, std::memory_order_relaxed); }
    bool cancelled() const { return mCancelled.load(std::memory_order_relaxed); }
    void update(const char* stage, uint64_t processed, uint64_t total, bool interruptible = true) {
        std::lock_guard<std::mutex> guard(mMutex);
        mProgress = {stage, processed, total, interruptible};
    }
    LoadProgress progress() const {
        std::lock_guard<std::mutex> guard(mMutex);
        return mProgress;
    }
private:
    std::atomic<bool> mCancelled{false};
    mutable std::mutex mMutex;
    LoadProgress mProgress;
};

} // namespace pvdapp
