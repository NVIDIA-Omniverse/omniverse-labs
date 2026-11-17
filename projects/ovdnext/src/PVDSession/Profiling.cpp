/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "Profiling.h"
#include "DiagnosticAccess.h"
#include "PvdEngine.h"
#include "PxProfileEventHandler.h"
#include "PxProfileEventParser.h"
#include "PxProfileEvents.h"
#include "PxProfileMemoryEventTypes.h"
#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>

namespace ovdnext {
namespace {
using namespace physx::profile;
using Key=std::tuple<int32_t,uint32_t,uint64_t,uint16_t>;
std::string text(const char* s) { return s?s:""; }
void checkLimit(uint32_t limit) {
    if(!limit || limit>1000000) throw std::invalid_argument("Diagnostic event limit must be between 1 and 1000000");
}
void consume(uint32_t& count,uint32_t n,uint32_t limit) {
    if(n>limit-count) throw std::runtime_error("Diagnostic event budget exceeded; narrow the frame or increase max_events");
    count+=n;
}
uint64_t scene(pvdapp::PvdEngine& e,uint64_t requested) {
    const auto count=e.getNbScenes();
    if(!count) throw std::invalid_argument("Capture has no scene timeline");
    if(count>1000000) throw std::runtime_error("Scene metadata budget exceeded");
    std::vector<pvdapp::SectionListInfo> scenes(count);
    scenes.resize(e.getScenes(scenes.data(),count));
    if(scenes.empty()) throw std::invalid_argument("Capture has no scene timeline");
    if(!requested) return scenes.front().mStreamId;
    for(const auto& s:scenes) if(s.mStreamId==requested) return requested;
    throw std::out_of_range("Unknown native scene ID");
}
double milliseconds(uint64_t ticks,uint64_t num,uint64_t den) {
    if(!den || !num) throw std::runtime_error("Capture has no valid timestamp conversion");
    return double(static_cast<long double>(ticks)*num/den/1000000.0L);
}
std::map<uint64_t,std::string> contextNames(pvdapp::PvdEngine& engine,uint32_t limit) {
    // PVDModel::loadSceneNames uses the native physx3::PxScene stream ID and
    // instance ID. Copy those captions under the same publication guard as
    // the timings; never retain engine strings or consult a later display.
    std::map<uint64_t,std::string> names;
    std::map<int32_t,bool> sceneClasses;
    std::vector<pvdapp::InstanceDescription> page(1024);
    const auto count=engine.getNbInstances();
    for(uint32_t offset=0;offset<count;) {
        const auto copied=engine.getInstances(page.data(),std::min<uint32_t>(page.size(),count-offset),offset);
        if(!copied)throw std::runtime_error("Incomplete profile context metadata");
        for(uint32_t i=0;i<copied;++i) {
            const auto& instance=page[i];
            auto found=sceneClasses.find(instance.mClassId);
            if(found==sceneClasses.end()) {
                const auto cls=engine.getClass(instance.mClassId);
                const bool isScene=cls.hasValue() && text(cls.getValue().mName.mNamespace)=="physx3" && text(cls.getValue().mName.mName)=="PxScene";
                found=sceneClasses.emplace(instance.mClassId,isScene).first;
                if(sceneClasses.size()>limit)throw std::runtime_error("Profile context class metadata budget exceeded");
            }
            if(found->second) {
                names[engine.getStreamId(instance.mId)]="PxScene id: "+std::to_string(instance.mId);
                if(names.size()>limit)throw std::runtime_error("Profile context metadata budget exceeded");
            }
        }
        offset+=copied;
    }
    return names;
}
struct ProfileCollector:PxProfileBulkEventHandler {
    ProfileSnapshot& output; uint32_t limit; int32_t zoneId=0;
    std::map<uint16_t,std::string> names;
    std::map<Key,std::vector<uint64_t>> opened;
    std::map<Key,uint32_t> unmatchedStops;
    bool exceeded=false;
    ProfileCollector(ProfileSnapshot& out,uint32_t cap):output(out),limit(cap) { output.startTick=std::numeric_limits<uint64_t>::max(); }
    std::string name(uint16_t id) const {
        const auto it=names.find(id);
        std::string s=it==names.end()?"Event "+std::to_string(id):it->second;
        if(!s.empty() && s.front()=='.') s.erase(0,1);
        return s;
    }
    void handleEvents(const Event* events,uint32_t count) override {
        if(count>limit-output.rawEventCount) { exceeded=true;return; }
        if(exceeded)return;
        output.rawEventCount+=count;
        for(uint32_t i=0;i<count;++i) {
            const auto& event=events[i]; const auto id=event.getHeader().mEventId;
            switch(event.getHeader().mEventType) {
            case EventTypes::StartEvent: {
                const auto& value=event.getData().toType(Type2Type<StartEvent>());
                output.startTick=std::min(output.startTick,value.getTimestamp());
                opened[{zoneId,value.mContextInformation.mThreadId,value.mContextInformation.mContextId,id}].push_back(value.getTimestamp());
                break;
            }
            case EventTypes::StopEvent: {
                const auto& value=event.getData().toType(Type2Type<StopEvent>());
                output.endTick=std::max(output.endTick,value.getTimestamp());
                const Key key{zoneId,value.mContextInformation.mThreadId,value.mContextInformation.mContextId,id};
                auto it=opened.find(key);
                if(it==opened.end() || it->second.empty()) { ++output.unmatchedStops;++unmatchedStops[key];break; }
                const auto start=it->second.back();it->second.pop_back();
                if(value.getTimestamp()<start) { ++output.ignoredEvents;break; }
                ProfileScope s;
                s.zoneId=zoneId;s.eventId=id;s.threadId=value.mContextInformation.mThreadId;
                s.contextId=value.mContextInformation.mContextId;s.startTick=start;s.endTick=value.getTimestamp();s.name=name(id);
                output.scopes.push_back(std::move(s));break;
            }
            case EventTypes::EventValue: {
                const auto& value=event.getData().toType(Type2Type<EventValue>());
                ProfileCounter v;v.zoneId=zoneId;v.eventId=id;v.threadId=value.mThreadId;v.contextId=value.mContextId;
                // The old EventValue::getValue truncates to int16. Copy the actual
                // native signed64 payload, which contains no timestamp.
                v.value=static_cast<int64_t>(value.mValue);v.name=name(id);output.counters.push_back(std::move(v));break;
            }
            default: ++output.ignoredEvents;break;
            }
        }
    }
};
void finish(ProfileSnapshot& out,ProfileCollector& collector) {
    std::map<Key,ProfileBoundary> incomplete;
    for(const auto& item:collector.opened) if(!item.second.empty()) {
        out.unmatchedStarts+=item.second.size();incomplete[item.first].starts=item.second.size();
    }
    for(const auto& item:collector.unmatchedStops) incomplete[item.first].stops=item.second;
    for(auto& item:incomplete) {
        auto& b=item.second;b.zoneId=std::get<0>(item.first);b.threadId=std::get<1>(item.first);
        b.contextId=std::get<2>(item.first);b.eventId=std::get<3>(item.first);out.unmatched.push_back(b);
    }
    if(out.unmatchedStarts || out.unmatchedStops) out.diagnostics.push_back("Unmatched profile boundaries are omitted; durations are never guessed across frame boundaries");
    if(out.ignoredEvents) out.diagnostics.push_back("Unsupported or invalid profile records were not converted into timed scopes");
    if(out.scopes.empty()) { out.startTick=out.endTick=0;return; }
    // Original Qt's elapsed-range denominator includes observed unmatched
    // boundaries; only matched pairs contribute invocation durations.
    out.durationMs=milliseconds(out.endTick-out.startTick,out.tickNumerator,out.tickDenominator);
    // Match original event+context+thread grouping, retaining zone identity to
    // avoid collisions when multiple zones use the same short event IDs.
    std::map<Key,ProfileSummary> summaries;
    std::sort(out.scopes.begin(),out.scopes.end(),[](const ProfileScope& a,const ProfileScope& b){
        const auto ag=std::tie(a.zoneId,a.contextId,a.threadId),bg=std::tie(b.zoneId,b.contextId,b.threadId);
        if(ag!=bg)return ag<bg;
        if(a.startTick!=b.startTick)return a.startTick<b.startTick;
        if(a.endTick!=b.endTick)return a.endTick>b.endTick;
        return a.eventId<b.eventId;
    });
    std::tuple<int32_t,uint64_t,uint32_t> group{};bool first=true;
    std::vector<uint64_t> enclosing;
    for(auto& s:out.scopes) {
        const auto current=std::make_tuple(s.zoneId,s.contextId,s.threadId);
        if(first || current!=group) { enclosing.clear();group=current;first=false; }
        while(!enclosing.empty() && (s.startTick>=enclosing.back() || s.endTick>enclosing.back())) enclosing.pop_back();
        s.depth=enclosing.size();enclosing.push_back(s.endTick);
        s.startMs=milliseconds(s.startTick-out.startTick,out.tickNumerator,out.tickDenominator);
        s.endMs=milliseconds(s.endTick-out.startTick,out.tickNumerator,out.tickDenominator);
        const auto duration=s.endTick-s.startTick;
        s.durationMs=milliseconds(duration,out.tickNumerator,out.tickDenominator);
        auto& row=summaries[{s.zoneId,s.threadId,s.contextId,s.eventId}];
        if(!row.count) {
            row.zoneId=s.zoneId;row.threadId=s.threadId;row.contextId=s.contextId;row.eventId=s.eventId;row.name=s.name;row.minMs=s.durationMs;
        }
        if(duration>std::numeric_limits<uint64_t>::max()-row.totalTicks) throw std::overflow_error("Profile duration sum exceeds uint64");
        row.totalTicks+=duration;++row.count;row.minMs=std::min(row.minMs,s.durationMs);row.maxMs=std::max(row.maxMs,s.durationMs);
    }
    for(auto& item:summaries) {
        auto& row=item.second;row.totalMs=milliseconds(row.totalTicks,out.tickNumerator,out.tickDenominator);
        row.averageMs=row.totalMs/row.count;row.percent=out.durationMs>0?100*row.totalMs/out.durationMs:0;
        out.summaries.push_back(std::move(row));
    }
}
struct MemoryCollector:PxProfileBulkMemoryEventHandler {
    std::vector<PxProfileBulkMemoryEvent> allocations,deallocations;
    uint32_t count=0,limit; bool exceeded=false;
    explicit MemoryCollector(uint32_t cap):limit(cap) {}
    void handleEvents(const PxProfileBulkMemoryEvent* events,uint32_t n) override {
        if(n>limit-count) { exceeded=true;return; }
        if(exceeded)return;
        count+=n;
        for(uint32_t i=0;i<n;++i) {
            if(events[i].mType==PxProfileMemoryEventType::Allocation) allocations.push_back(events[i]);
            else if(events[i].mType==PxProfileMemoryEventType::Deallocation) deallocations.push_back(events[i]);
        }
    }
};
}
std::shared_ptr<ProfileSnapshot> profileSnapshot(Session& session,uint64_t sceneId,uint32_t eventLimit,bool includeContextNames) {
    checkLimit(eventLimit);auto result=std::make_shared<ProfileSnapshot>();
    ProfilingAccess::read(session,[&](pvdapp::PvdEngine& e) {
        if(e.isOvdFile()) { result->reason="OVD captures do not contain legacy profile event streams";return; }
        result->sceneId=scene(e,sceneId);result->currentFrame=e.getCurrentFrame(result->sceneId);
        result->endEvent=e.getCurrentEvent();result->startEvent=e.findPreviousFrameEvent(result->endEvent,result->sceneId);
        const auto count=e.getNbProfileZones();
        if(!count) { result->reason="This capture contains no profile zones";return; }
        if(count>eventLimit) throw std::runtime_error("Profile zone metadata budget exceeded");
        e.getTimestampNanosecondConversion(result->tickNumerator,result->tickDenominator);
        if(!result->tickNumerator || !result->tickDenominator) { result->reason="Capture has no valid profile timestamp conversion";return; }
        std::vector<pvdapp::PvdProfileZone> zones(count);zones.resize(e.getProfileZones(zones.data(),count));
        ProfileCollector collector(*result,eventLimit);uint32_t metadataCount=0;
        for(const auto& zone:zones) {
            consume(metadataCount,zone.mEvents.size(),eventLimit);
            ProfileZoneInfo owned;owned.id=zone.mInstanceId;owned.streamId=zone.mStreamId;owned.name=text(zone.mName);
            collector.zoneId=owned.id;collector.names.clear();
            for(const auto& event:zone.mEvents) {
                owned.events.push_back({event.mEventId,text(event.mName),event.mIsCompiledTimeEnabled});
                collector.names[event.mEventId]=text(event.mName);
            }
            result->zones.push_back(std::move(owned));
            if(result->endEvent>result->startEvent) e.parseProfileEvents(result->startEvent,result->endEvent,zone.mInstanceId,collector);
            // Do not throw from original parser callbacks: its own cleanup is not
            // exception-safe. Discard bounded output and fail after it returns.
            if(collector.exceeded) throw std::runtime_error("Diagnostic event budget exceeded; increase max_events");
        }
        finish(*result,collector);
        if(includeContextNames)result->contextNames=contextNames(e,eventLimit);
        result->supported=true;
    });return result;
}
std::shared_ptr<MemorySnapshot> memorySnapshot(Session& session,uint64_t sceneId,uint32_t eventLimit) {
    checkLimit(eventLimit);auto result=std::make_shared<MemorySnapshot>();
    ProfilingAccess::read(session,[&](pvdapp::PvdEngine& e) {
        if(e.isOvdFile()) { result->reason="OVD captures do not contain legacy memory event streams";return; }
        result->sceneId=scene(e,sceneId);result->currentFrame=e.getCurrentFrame(result->sceneId);
        result->endEvent=e.getCurrentEvent();result->startEvent=e.findPreviousFrameEvent(result->endEvent,result->sceneId);
        const auto entriesCount=e.getNbMemoryEventEntries();
        if(entriesCount>eventLimit) throw std::runtime_error("Memory metadata budget exceeded");
        MemoryCollector collector(eventLimit);
        if(result->endEvent>result->startEvent) e.parseMemoryEvents(result->startEvent,result->endEvent,collector);
        if(collector.exceeded) throw std::runtime_error("Diagnostic event budget exceeded; increase max_events");
        result->rawEventCount=collector.count;
        if(!entriesCount && !collector.count) { result->reason="This capture contains no memory event measurements";return; }
        using RowKey=std::tuple<std::string,uint32_t,std::string>;
        std::map<RowKey,MemoryRow> rows;
        std::map<uint64_t,std::deque<size_t>> byAddress;
        auto rowFor=[&](const PxProfileBulkMemoryEvent& event)->MemoryRow& {
            const auto file=text(e.handleToString(event.mFile)),type=text(e.handleToString(event.mDatatype));
            auto& row=rows[{file,event.mLine,type}];row.file=file;row.type=type;row.line=event.mLine;return row;
        };
        for(size_t i=0;i<collector.allocations.size();++i) {
            const auto& event=collector.allocations[i];
            if(!event.mSize) { ++result->zeroAllocations;continue; }
            byAddress[event.mAddress].push_back(i);
            auto& row=rowFor(event);row.allocatedBytes+=event.mSize;++row.allocationCount;
            result->total.allocatedBytes+=event.mSize;++result->total.allocationCount;
        }
        // Preserve Qt's documented same-frame address matching. Outstanding data
        // below comes independently from the native whole-capture summarizer.
        for(const auto& event:collector.deallocations) {
            auto found=byAddress.find(event.mAddress);
            if(found==byAddress.end() || found->second.empty()) { ++result->unmatchedDeallocations;continue; }
            const auto& allocation=collector.allocations[found->second.front()];found->second.pop_front();
            auto& row=rowFor(allocation);row.deallocatedBytes+=allocation.mSize;++row.deallocationCount;
            result->total.deallocatedBytes+=allocation.mSize;++result->total.deallocationCount;
        }
        for(auto& row:rows) result->rows.push_back(std::move(row.second));
        std::vector<pvdapp::PvdMemoryEventEntry> entries(entriesCount);
        if(entriesCount) entries.resize(e.getMemoryEventEntries(entries.data(),entriesCount,0));
        const auto count=e.getNbInstances()?e.getNbMemoryTotals():0;
        if(count>eventLimit) throw std::runtime_error("Memory totals budget exceeded");
        std::vector<int32_t> totals(count);
        // Core returns a BYTE count, unlike the metadata entry reader.
        if(count) totals.resize(std::min<size_t>(count,e.getMemotyEventTotals(totals.data(),count)/sizeof(int32_t)));
        bool negative=false;
        for(const auto& entry:entries) {
            MemoryOutstanding owned;owned.file=text(entry.mFile);owned.type=text(entry.mType);owned.line=entry.mLine;owned.index=entry.mTotalsArrayIndex;
            owned.valid=owned.index<totals.size();
            if(owned.valid) { owned.bytes=totals[owned.index];negative|=owned.bytes<0; }
            result->outstanding.push_back(std::move(owned));
        }
        if(result->unmatchedDeallocations) result->diagnostics.push_back("Frame deallocation totals exclude frees without a matching allocation in this frame");
        if(negative) result->diagnostics.push_back("Native outstanding totals contain negative signed32 values; they are preserved, not reinterpreted or clamped");
        result->supported=true;
    });return result;
}
}
