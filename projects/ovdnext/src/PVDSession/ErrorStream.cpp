/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "ErrorStream.h"
#include "DiagnosticAccess.h"
#include "PvdEngine.h"
#include "fakeinstanceheader.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
namespace ovdnext {
namespace {
uint32_t field(pvdapp::PvdEngine& engine,FakeInstanceHeader& header,const char* name) {
    const auto value=engine.getPropertyValue<uint32_t>(&header,name);
    if(!value.hasValue()) throw std::runtime_error(std::string("Legacy error record is missing field: ")+name);
    return value.getValue();
}
std::string string(pvdapp::PvdEngine& engine,uint32_t handle) {
    const char* value=engine.handleToString(handle);
    if(!value) return {};
    constexpr size_t limit=1024*1024;
    size_t size=0;
    while(size<=limit && value[size]) ++size;
    if(size>limit) throw std::runtime_error("Legacy error text exceeds 1 MiB transfer limit");
    return std::string(value,size);
}
}
std::shared_ptr<ErrorStreamSnapshot> errorStreamSnapshot(Session& session,uint64_t sceneId,uint32_t maxRecords) {
    if(!maxRecords || maxRecords>100000) throw std::invalid_argument("Error record limit must be between 1 and 100000");
    auto result=std::make_shared<ErrorStreamSnapshot>();
    ProfilingAccess::read(session,[&](pvdapp::PvdEngine& engine) {
        if(engine.isOvdFile()) {result->reason="OVD uses Capture messages; it has no legacy Error Stream";return;}
        const auto sceneCount=engine.getNbScenes();
        if(sceneCount>1000000) throw std::runtime_error("Scene metadata budget exceeded");
        std::vector<pvdapp::SectionListInfo> scenes(sceneCount);
        if(sceneCount) scenes.resize(engine.getScenes(scenes.data(),sceneCount));
        if(!sceneId && !scenes.empty()) sceneId=scenes.front().mStreamId;
        if(sceneId && std::none_of(scenes.begin(),scenes.end(),[&](const pvdapp::SectionListInfo& s){return s.mStreamId==sceneId;})) throw std::out_of_range("Unknown native scene ID");
        result->sceneId=sceneId;result->currentFrame=sceneId?engine.getCurrentFrame(sceneId):0;result->supported=true;
        // Preserve original enumeration order without allocating the full object graph.
        const auto count=engine.getNbInstances();
        if(count>1000000) throw std::runtime_error("Error log instance scan exceeds 1000000 object budget");
        std::vector<pvdapp::InstanceDescription> instances(std::min(count,uint32_t(1024)));
        size_t copiedBytes=0;
        for(uint32_t offset=0;offset<count;offset+=uint32_t(instances.size())) {
            const auto read=engine.getInstances(instances.data(),std::min(uint32_t(instances.size()),count-offset),offset);
            if(!read) break;
            for(uint32_t i=0;i<read;++i) {
                const auto& instance=instances[i];
                const auto cls=engine.getClass(instance.mClassId);
                if(!cls.hasValue() || !cls.getValue().mName.mName || std::strcmp(cls.getValue().mName.mName,"UserCustomedEvent")) continue;
                result->hasLog=true;result->logInstanceId=instance.mId;
                const auto prop=engine.findProperty(instance.mClassId,"Events");
                if(!prop.hasValue() || prop.getValue().mPropertyType!=pvdapp::PropertyType::Array) throw std::runtime_error("Legacy error log has no Events array");
                const auto& descriptor=prop.getValue();
                const auto elementClass=engine.getClass(descriptor.mDatatype);
                if(!elementClass.hasValue() || !elementClass.getValue().get32BitSize()) throw std::runtime_error("Legacy error array has no native structure layout");
                const auto stride=elementClass.getValue().get32BitSize();
                const auto raw=engine.getRawPropertyValue(instance.mInstPtr,descriptor.mPropertyId);
                if(raw.size()%stride) throw std::runtime_error("Malformed legacy error array");
                result->total=raw.size()/stride;result->truncated=result->total>maxRecords;
                const auto limit=std::min(result->total,maxRecords);result->records.reserve(limit);
                for(uint32_t j=0;j<limit;++j) {
                    FakeInstanceHeader header(1,descriptor.mDatatype,descriptor.mDatatypeName,stride);
                    header.mDataSectionStart=reinterpret_cast<uint64_t>(raw.begin()+size_t(j)*stride);
                    ErrorStreamRecord record;record.index=j;
                    record.code=field(engine,header,"Code");record.line=field(engine,header,"Line");
                    record.message=string(engine,field(engine,header,"Message"));record.file=string(engine,field(engine,header,"File"));
                    copiedBytes+=record.message.size()+record.file.size();
                    if(copiedBytes>16*1024*1024) throw std::runtime_error("Legacy error text exceeds 16 MiB snapshot budget");
                    result->records.push_back(std::move(record));
                }
                return; // Historical first-log rule; do not merge unrelated logs.
            }
        }
    });
    return result;
}
}
