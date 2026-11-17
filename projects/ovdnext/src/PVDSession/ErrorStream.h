/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace pybind11 { class module_; }
namespace ovdnext {
class Session;
struct ErrorStreamRecord {
    uint32_t index=0,code=0,line=0;
    std::string message,file;
};
struct ErrorStreamSnapshot {
    bool supported=false,hasLog=false,truncated=false;
    std::string reason;
    uint64_t sceneId=0;
    uint32_t currentFrame=0,total=0;
    int32_t logInstanceId=-1;
    std::vector<ErrorStreamRecord> records;
};
// Legacy Error Stream: first UserCustomedEvent instance's native Events array,
// following PVDModel::getErrorLog(0), not OVD's capture-message stream. No seek.
// All strings owned. maxRecords bounds copies (1..100000); total remains exact.
std::shared_ptr<ErrorStreamSnapshot> errorStreamSnapshot(Session&,uint64_t sceneId,uint32_t maxRecords);
void bindErrorStream(pybind11::module_&);
}
