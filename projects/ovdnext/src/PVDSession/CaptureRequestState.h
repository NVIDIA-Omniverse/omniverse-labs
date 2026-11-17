/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "../PVDCore/OvdCaptureCatalog.h"
#include "../PVDCore/OvdCaptureControl.h"
#include <array>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ovdnext {
inline uint64_t nextCaptureRequestId(uint64_t previous) {
    if(previous==std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("Capture request IDs exhausted; reconnect the producer");
    return previous+1;
}
struct CaptureRequestResult {
    uint64_t id=0,epoch=0;
    uint32_t ackStatus=0,sceneCount=0,objectCount=0;
    bool catalogOnly=false,secondary=false,all=false;
    std::string state="idle";
    double queuedAt=0,sentAt=0,completedAt=0;
};
// Session-mutex owned. Only scalar summaries survive an outbound request.
class CaptureRequestState {
    using Clock=std::chrono::steady_clock;
    std::array<CaptureRequestResult,64> results{};
    size_t count=0;
    uint64_t issued=0,ackRevision=0;
    std::string classification="none";
    static double seconds(Clock::time_point time) {
        return std::chrono::duration<double>(time.time_since_epoch()).count();
    }
public:
    uint64_t highWater() const { return issued; }
    const std::string& ackClassification() const { return classification; }
    const CaptureRequestResult& current() const { return results[issued ? (issued-1)%64 : 0]; }
    uint64_t nextId(bool supersede) const {
        const auto& state=current().state;
        if(state=="sending" || state=="awaiting_ack" || (state=="unknown" && !supersede))
            throw std::runtime_error("A capture request is pending; only Unknown permits explicit supersession");
        return nextCaptureRequestId(issued);
    }
    void submitted(const pvdapp::CaptureRequest& request, Clock::time_point now) {
        if(issued && current().state=="unknown") results[(issued-1)%64].state="superseded";
        issued=request.requestId; count=std::min<size_t>(count+1,64);
        auto& result=results[(issued-1)%64]; result={};
        result.id=issued; result.catalogOnly=request.catalogOnly;
        result.secondary=request.secondary; result.all=request.all;
        result.sceneCount=static_cast<uint32_t>(request.scenes.size());
        result.objectCount=static_cast<uint32_t>(request.objects.size());
        result.queuedAt=seconds(now); result.state="sending";
    }
    void observe(const pvdapp::CaptureTransportSnapshot& transport,
                 const pvdapp::CaptureMetadataSnapshot& metadata, Clock::time_point now, bool connected) {
        auto& active=results[issued ? (issued-1)%64 : 0];
        if(issued && (active.state=="sending" || active.state=="awaiting_ack" || active.state=="unknown")) {
            if(!connected) active.state="unknown";
            else if(transport.state==pvdapp::CaptureTransportState::Unavailable) active.state="unavailable";
            else if(transport.requestId==issued && transport.state==pvdapp::CaptureTransportState::AwaitingAck) {
                active.sentAt=seconds(transport.sentAt);
                active.state=now-transport.sentAt>=std::chrono::seconds(10)?"unknown":"awaiting_ack";
            }
        }
        if(metadata.ackRevision==ackRevision) return;
        ackRevision=metadata.ackRevision;
        classification=metadata.ackId>issued || !metadata.ackId ? "unknown" : metadata.ackId==issued ? "current" : "late";
        if(metadata.ackId && metadata.ackId<=issued && issued-metadata.ackId<count) {
            auto& result=results[(metadata.ackId-1)%64];
            result.ackStatus=metadata.ackStatus; result.epoch=metadata.ackEpoch;
            result.completedAt=seconds(now);
            if(result.id==issued && (result.state=="awaiting_ack" || result.state=="unknown")) {
                static const char* states[]={"unknown","applied","invalid","busy","denied"};
                result.state=metadata.ackStatus<=4?states[metadata.ackStatus]:"unknown";
            }
        }
    }
    std::vector<CaptureRequestResult> history() const {
        std::vector<CaptureRequestResult> copy; copy.reserve(count);
        for(size_t n=0;n<count;++n) copy.push_back(results[(issued-count+n)%64]);
        return copy;
    }
};
struct CaptureControlInfo {
    uint16_t capabilities=0;
    bool connected=false,canApply=false,canRefresh=false,hasCompletedFrame=false;
    uint64_t issuedHighWater=0,forwardBytes=0,completedBoundarySerial=0;
    double forwardIdleSeconds=0;
    std::string reason,ackClassification;
    pvdapp::CaptureCatalogSnapshot catalog{};
    pvdapp::CaptureMetadataSnapshot metadata{};
    pvdapp::CaptureBudgetSnapshot budget{};
    pvdapp::CaptureTransportSnapshot transport{};
    CaptureRequestResult request;
    std::vector<CaptureRequestResult> results;
};
}
