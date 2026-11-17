/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "AnalysisAvailability.h"
#include "DiagnosticAccess.h"
#include "PvdEngine.h"
#include "LoadControl.h"
#include "PvdObjectModelEventStream.h"
#include "PvdSimpleAllocator.h"
#include "PxProfileMemoryEventTypes.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>
namespace ovdnext {
namespace {
using namespace pvdapp;
void cancelled(const std::shared_ptr<LoadControl>& control) {
    if(control && control->cancelled()) throw std::runtime_error("Analysis availability check cancelled");
}
AnalysisPresence absent(const char* reason) {return {true,false,reason};}
AnalysisPresence unknown(const char* reason) {return {false,false,reason};}
struct MemoryPresence: PvdObjectModelEventStreamPropertyHandler,
                       physx::profile::PxProfileBulkMemoryEventHandler {
    std::shared_ptr<LoadControl> control;
    MallocAllocator allocator;
    uint64_t bytes=0,events=0;
    uint32_t maxBytes,maxEvents;
    bool exceeded=false,found=false;
    MemoryPresence(std::shared_ptr<LoadControl> c,uint32_t b,uint32_t e):control(c),maxBytes(b),maxEvents(e){}
    void handleEvents(const physx::profile::PxProfileBulkMemoryEvent* records,uint32_t count) override {
        // Never throw through the original parser's non-RAII callback path.
        if(count>maxEvents-events) {exceeded=true;return;}
        events+=count;
        if(count) found=true;
    }
    void handleProperty(PxU64,PxI32,PxI32,const PxU8* data,PxU32 len) override {
        if(found || exceeded || (control && control->cancelled())) return;
        // Bound the serialized input before invoking the authoritative parser.
        if(len>maxBytes-bytes) {exceeded=true;return;}
        bytes+=len;
        physx::profile::PxProfileBulkMemoryEventHandler::parseEventBuffer(data,len,*this,false,&allocator);
    }
};
}
AnalysisAvailability analysisAvailability(Session& session,uint64_t sceneId,
    std::shared_ptr<LoadControl> control,uint32_t maxInstances,uint32_t maxEvents,uint32_t maxBytes) {
    if(!maxInstances || maxInstances>1000000 || !maxEvents || maxEvents>65536 || !maxBytes || maxBytes>8*1024*1024)
        throw std::invalid_argument("Availability budgets exceed supported instance/event/byte limits");
    cancelled(control);
    AnalysisAvailability out;
    ProfilingAccess::read(session,[&](PvdEngine& engine) {
        if(engine.isOvdFile()) {
            out.profile=absent("OVD does not contain legacy profile event streams");
            out.memory=absent("OVD does not contain legacy memory event streams");
            out.errors=absent("OVD uses Capture messages; it has no legacy Error Stream");
            out.frameTiming=absent("OVD does not record legacy frame timing measurements");
            return;
        }
        const auto sceneCount=engine.getNbScenes();
        if(sceneCount>1000000) throw std::runtime_error("Scene metadata budget exceeded");
        std::vector<SectionListInfo> scenes(sceneCount);
        if(sceneCount) scenes.resize(engine.getScenes(scenes.data(),sceneCount));
        if(!sceneId && !scenes.empty()) sceneId=scenes.front().mStreamId;
        if(!sceneId || std::none_of(scenes.begin(),scenes.end(),[&](const SectionListInfo& s){return s.mStreamId==sceneId;}))
            throw std::out_of_range("Capture has no requested scene timeline");
        PxU64 numerator=0,denominator=0;
        engine.getTimestampNanosecondConversion(numerator,denominator);
        out.profileZones=engine.getNbProfileZones();
        out.profile=out.profileZones && numerator && denominator?AnalysisPresence{true,true,{}}:
            absent(!out.profileZones?"This capture contains no profile zones":"Capture has no valid profile timestamp conversion");
        out.frameTiming=denominator?AnalysisPresence{true,true,{}}:absent("Capture has no valid frame timestamp conversion");
        out.memoryEntries=engine.getNbMemoryEventEntries();
        if(out.memoryEntries) out.memory={true,true,{}};
        else {
            out.memory=absent("This frame contains no memory event measurements or recorded memory metadata");
            const auto buffer=engine.getMemoryEventBufferInstance();
            const auto end=engine.getCurrentEvent(),start=engine.findPreviousFrameEvent(end,sceneId);
            if(buffer>=0 && end>start) {
                if(start<0 || uint64_t(end-start)>maxEvents)
                    out.memory=unknown("Memory availability exceeds the event scan budget");
                else {
                    // Match the native buffer class; ambiguity is uncertainty.
                    int32_t property=-1;bool ambiguous=false;
                    const auto classCount=engine.getNbClasses();
                    if(classCount>10000) out.memory=unknown("Memory availability exceeds the class metadata budget");
                    else {
                        std::vector<ClassDescription> classes(classCount);
                        if(classCount) classes.resize(engine.getClasses(classes.data(),classCount));
                        for(const auto& cls:classes) {
                            cancelled(control);
                            if(!cls.mName.mName || !cls.mName.mNamespace ||
                               std::strcmp(cls.mName.mName,"PxProfileMemoryEventBuffer") ||
                               std::strcmp(cls.mName.mNamespace,"_debugger_"))continue;
                            const auto prop=engine.findProperty(cls.mClassId,"events");
                            if(prop.hasValue()) {if(property>=0)ambiguous=true;property=prop->mPropertyId;}
                        }
                        if(property<0 || ambiguous || !engine.getEventStream())
                            out.memory=unknown("Memory buffer metadata cannot be resolved unambiguously");
                        else {
                            cancelled(control);
                            MemoryPresence presence(control,maxBytes,maxEvents);
                            const bool read=engine.getEventStream()->getPropertyHistory(InstancePropertyId(buffer,property),start,end,presence);
                            cancelled(control);
                            out.memoryBytes=presence.bytes;out.memoryEvents=presence.events;
                            if(presence.found) out.memory={true,true,{}};
                            else if(!read || presence.exceeded) out.memory=unknown("Memory availability exceeds the byte/event budget or history is unavailable");
                        }
                    }
                }
            }
        }
        cancelled(control);
        out.errors=absent("This frame has no legacy Error Stream log");
        const auto count=engine.getNbInstances(),limit=std::min(count,maxInstances);
        std::vector<InstanceDescription> instances(std::min(limit,uint32_t(1024)));
        std::unordered_map<int32_t,bool> isLog;
        for(uint32_t offset=0;offset<limit;) {
            cancelled(control);
            const auto read=engine.getInstances(instances.data(),std::min(uint32_t(instances.size()),limit-offset),offset);
            if(!read) {out.errors=unknown("Error log enumeration did not complete");break;}
            offset+=read;out.scannedInstances=offset;
            for(uint32_t i=0;i<read;++i) {
                const auto& instance=instances[i];
                auto known=isLog.find(instance.mClassId);
                if(known==isLog.end()) {
                    const auto cls=engine.getClass(instance.mClassId);
                    known=isLog.emplace(instance.mClassId,cls.hasValue() && cls->mName.mName &&
                        !std::strcmp(cls->mName.mName,"UserCustomedEvent")).first;
                }
                if(known->second) {out.errors={true,true,{}};out.errorLog=instance.mId;return;}
            }
        }
        if(!out.errors.available && count>limit) out.errors=unknown("Error log availability exceeds the instance scan budget");
    });
    cancelled(control);
    return out;
}
}
