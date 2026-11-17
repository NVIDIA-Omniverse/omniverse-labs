/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstdint>
#include <memory>
#include <string>
namespace pybind11 { class module_; }
namespace pvdapp { class LoadControl; }
namespace ovdnext {
class Session;
struct AnalysisPresence {
    bool known=true,available=false;
    std::string reason;
};
struct AnalysisAvailability {
    AnalysisPresence profile,memory,errors,frameTiming;
    uint32_t profileZones=0,memoryEntries=0,scannedInstances=0;
    uint64_t memoryBytes=0,memoryEvents=0;
    int32_t errorLog=-1;
};
// Presence only: no scope aggregation, memory rows, error text or geometry.
// Caller uses a private file Session at the desired frame. All fields owned.
AnalysisAvailability analysisAvailability(Session&,uint64_t sceneId,
    std::shared_ptr<pvdapp::LoadControl> control,uint32_t maxInstances,
    uint32_t maxEvents,uint32_t maxBytes);
void bindAnalysisAvailability(pybind11::module_&);
}
