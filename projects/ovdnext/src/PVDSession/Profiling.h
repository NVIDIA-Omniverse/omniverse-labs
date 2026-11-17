/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstdint>
#include <memory>
#include <map>
#include <string>
#include <vector>

namespace pybind11 { class module_; }
namespace ovdnext {
class Session;

struct ProfileEventName { uint16_t id=0; std::string name; bool enabled=false; };
struct ProfileZoneInfo {
    int32_t id=0; uint64_t streamId=0; std::string name;
    std::vector<ProfileEventName> events;
};
struct ProfileScope {
    int32_t zoneId=0; uint16_t eventId=0; uint32_t threadId=0,depth=0;
    uint64_t contextId=0,startTick=0,endTick=0;
    std::string name;
    // Relative to snapshot.startTick, avoiding loss of precision at large ticks.
    double startMs=0,endMs=0,durationMs=0;
};
struct ProfileSummary {
    int32_t zoneId=0; uint16_t eventId=0; uint32_t threadId=0,count=0;
    uint64_t contextId=0,totalTicks=0;
    std::string name;
    double totalMs=0,averageMs=0,minMs=0,maxMs=0,percent=0;
};
struct ProfileBoundary {
    int32_t zoneId=0; uint16_t eventId=0; uint32_t threadId=0,starts=0,stops=0;
    uint64_t contextId=0;
};
struct ProfileCounter {
    int32_t zoneId=0; uint16_t eventId=0; uint32_t threadId=0;
    uint64_t contextId=0; int64_t value=0; std::string name;
    // Native EventValue records have no timestamp: never place them on a time axis.
};
struct ProfileSnapshot {
    bool supported=false;
    std::string reason;
    uint64_t sceneId=0,startTick=0,endTick=0,tickNumerator=0,tickDenominator=0;
    uint32_t currentFrame=0,rawEventCount=0,unmatchedStarts=0,unmatchedStops=0,ignoredEvents=0;
    int64_t startEvent=0,endEvent=0;
    double durationMs=0;
    std::vector<ProfileZoneInfo> zones;
    std::map<uint64_t,std::string> contextNames;
    std::vector<ProfileScope> scopes;
    std::vector<ProfileSummary> summaries;
    std::vector<ProfileCounter> counters;
    std::vector<ProfileBoundary> unmatched;
    std::vector<std::string> diagnostics;
};
struct MemoryRow {
    std::string file,type; uint32_t line=0;
    uint64_t allocatedBytes=0,deallocatedBytes=0,allocationCount=0,deallocationCount=0;
};
struct MemoryOutstanding {
    std::string file,type; uint32_t line=0,index=0;
    bool valid=false; int32_t bytes=0; // preserve native signed32 totals, not guessed uint64.
};
struct MemorySnapshot {
    bool supported=false; std::string reason;
    uint64_t sceneId=0; uint32_t currentFrame=0,rawEventCount=0,unmatchedDeallocations=0,zeroAllocations=0;
    int64_t startEvent=0,endEvent=0;
    MemoryRow total;
    std::vector<MemoryRow> rows;
    std::vector<MemoryOutstanding> outstanding;
    std::vector<std::string> diagnostics;
};
// Read-only, atomic current-frame snapshots; no seek, borrowed strings, or partial
// results. eventLimit bounds parser callbacks and copied metadata (1..1,000,000).
// OVD is explicitly unsupported: its absence of profile/memory measurements is
// never represented as a valid zero-valued legacy series.
std::shared_ptr<ProfileSnapshot> profileSnapshot(Session&,uint64_t sceneId,uint32_t eventLimit,bool includeContextNames=true);
std::shared_ptr<MemorySnapshot> memorySnapshot(Session&,uint64_t sceneId,uint32_t eventLimit);
void bindProfiling(pybind11::module_&);
}
