/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "Presentation.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <set>
namespace ovdnext {
// OVD's pinned schema stores [length,speed] as exactly two F32 fields;
// legacy PVD records separate scalar properties. Both components share one
// lazy ownership resolution, using only this publication's owned metadata.
void resolveCapturedTolerances(PresentationData& data,int32_t sceneInstanceId,bool capturedSpeed) {
    auto instance=[&](int32_t id)->const InstanceInfo* {
        const auto found=data.instanceIndex.find(id);
        if(found==data.instanceIndex.end())return nullptr;
        const auto& result=data.instances[found->second];return result.alive?&result:nullptr;
    };
    std::unordered_set<int32_t> physicsClasses;
    for(const auto& cls:data.schema->classes)if(cls.name=="PxPhysics")physicsClasses.insert(cls.id);
    auto isPhysics=[&](const InstanceInfo& i) {return physicsClasses.count(i.classId)!=0;};
    auto value=[&](int32_t id,int32_t property)->const PresentationValue* {
        const auto found=data.values.find(Presentation::valueKey(id,property));
        return found==data.values.end()?nullptr:&found->second;
    };
    const auto* scene=instance(sceneInstanceId);
    bool ownershipResolved=false;
    const InstanceInfo* physics=nullptr;
    const char* ownerResolution="physics-owner";
    const char* ownerError=nullptr;
    auto resolveOwner=[&] {
        if(ownershipResolved)return;
        ownershipResolved=true;
        std::set<int32_t> owners;
        // OVD initially parents scenes to the last physics even without a
        // membership. Exact references avoid treating that default as proof.
        for(const auto& prop:data.schema->properties.at(scene->classId)) {
            if(prop.type!="ObjectRef" || (prop.name!="Physics" && prop.name!="physics"))continue;
            const auto* refs=value(scene->id,prop.id);
            if(refs && refs->hasReferences)for(const auto id:refs->references)if(id>0)owners.insert(id);
        }
        for(const auto& i:data.instances)if(i.alive && isPhysics(i)) {
            for(const auto& prop:data.schema->properties.at(i.classId)) {
                if(prop.type!="ObjectRef" || (prop.name!="Scenes" && prop.name!="scenes"))continue;
                const auto* refs=value(i.id,prop.id);
                if(refs && refs->hasReferences && std::find(refs->references.begin(),refs->references.end(),scene->id)!=refs->references.end())
                    owners.insert(i.id);
            }
        }
        if(owners.empty()) {
            for(const auto& i:data.instances)if(i.alive && isPhysics(i))owners.insert(i.id);
            ownerResolution="single-physics";
            if(scene->physicsId>0 && !owners.count(scene->physicsId)){ownerError="physics-unavailable";return;}
        }
        if(owners.size()>1){ownerError="ambiguous-physics";return;}
        if(owners.empty())return;
        const auto* candidate=instance(*owners.begin());
        if(candidate && isPhysics(*candidate))physics=candidate;
    };
    auto resolve=[&](bool speed,float& result,int32_t& sourceId,int32_t& propertyId,std::string& resolution) {
        result=1.0f;sourceId=0;propertyId=-1;resolution="missing";
        auto read=[&](const InstanceInfo& i,const char* provenance) {
            for(const auto& prop:data.schema->properties.at(i.classId)) {
                const bool compound=data.info.ovd && prop.name=="tolerancesScale";
                if(!compound && prop.name!=(speed?"TolerancesScale.Speed":"TolerancesScale.Length"))continue;
                const uint32_t bytes=compound?8:4;
                const auto* owned=value(i.id,prop.id);
                if(!owned || !owned->size)continue;
                const bool type=compound ? prop.type=="PxVec2" && prop.fieldCount==2
                    : (prop.type=="PvdF32" || prop.type=="PxF32") && (prop.fieldCount==1 || prop.fieldCount==-1);
                if(!type || prop.array || owned->size!=bytes){resolution="invalid";continue;}
                const uint8_t* raw=nullptr;
                for(const auto& window:owned->windows)if(!window.offset && window.bytes.size()>=bytes){raw=window.bytes.data();break;}
                if(!raw){resolution="invalid";continue;}
                float fields[2]={1.0f,1.0f};std::memcpy(fields,raw,bytes);
                const float number=fields[compound && speed?1:0];
                // Preserve the speed contract's validation of both compound
                // fields. Valid length itself does not require valid speed.
                if(!std::isfinite(number) || number<=0 || (compound && speed && (!std::isfinite(fields[0]) || fields[0]<=0))) {
                    resolution="invalid";continue;
                }
                result=number;sourceId=i.id;propertyId=prop.id;resolution=provenance;return true;
            }
            return false;
        };
        if(!scene){resolution="scene-unavailable";return;}
        if(read(*scene,"scene"))return;
        resolveOwner();
        if(ownerError){resolution=ownerError;return;}
        if(physics)read(*physics,ownerResolution);
    };
    resolve(false,data.toleranceLength,data.toleranceLengthSourceId,data.toleranceLengthPropertyId,data.toleranceLengthResolution);
    if(capturedSpeed)resolve(true,data.toleranceSpeed,data.toleranceSpeedSourceId,data.toleranceSpeedPropertyId,data.toleranceSpeedResolution);
}

CaptureIdentity::CaptureIdentity():key([]{static std::atomic<uint64_t> next{1};return next.fetch_add(1);}()) {}
const std::vector<ClassInfo>& Presentation::classes() const {
    static const std::vector<ClassInfo> empty;
    return data->schema?data->schema->classes:empty;
}
std::vector<InstanceInfo> Presentation::instances(uint32_t offset,uint32_t limit) const {
    if(limit>10000)throw std::invalid_argument("Instance page limit exceeds 10000");
    const size_t end=std::min(data->instances.size(),size_t(offset)+limit);
    if(offset>=end)return {};
    return {data->instances.begin()+offset,data->instances.begin()+end};
}
const std::vector<PropertyInfo>& Presentation::properties(int32_t classId) const {
    if(data->schema) {const auto found=data->schema->properties.find(classId);if(found!=data->schema->properties.end())return found->second;}
    throw std::out_of_range("Unknown class ID in presentation");
}
const PresentationValue& Presentation::value(int32_t instanceId,int32_t propertyId) const {
    const auto instance=data->instanceIndex.find(instanceId);
    if(instance==data->instanceIndex.end())throw std::out_of_range("Unknown instance ID in presentation");
    if(!data->instances[instance->second].alive)throw std::out_of_range("Instance is not alive in this presentation");
    const auto found=data->values.find(valueKey(instanceId,propertyId));
    if(found==data->values.end())throw SnapshotDataUnavailable("Property was not requested in this presentation; request a new coherent view");
    return found->second;
}
std::string Presentation::propertyAvailability(int32_t instanceId,int32_t propertyId) const {
    const auto instance=data->instanceIndex.find(instanceId);
    if(instance==data->instanceIndex.end())throw std::out_of_range("Unknown instance ID in presentation");
    const auto& props=properties(data->instances.at(instance->second).classId);
    const auto property=std::find_if(props.begin(),props.end(),[&](const PropertyInfo& p){return p.id==propertyId;});
    if(property==props.end())throw std::out_of_range("Property does not belong to this instance");
    const auto found=data->values.find(valueKey(instanceId,propertyId));
    if(found==data->values.end())return "not_projected";
    // Empty arrays/references are complete values. Scalar sample presence is
    // kept separately from byte size so a zero-byte SET remains distinguishable.
    return found->second.samplePresent || property->array || property->type=="ObjectRef"
        ? "present" : "absent_at_frame";
}
std::vector<uint8_t> Presentation::read(int32_t instanceId,int32_t propertyId,uint32_t offset,uint32_t limit) const {
    if(limit>8*1024*1024)throw std::invalid_argument("Property page limit exceeds 8 MiB");
    const auto& v=value(instanceId,propertyId);
    if(offset>=v.size || !limit)return {};
    const uint64_t end=std::min(uint64_t(v.size),uint64_t(offset)+limit);
    for(const auto& window:v.windows)if(offset>=window.offset && end<=uint64_t(window.offset)+window.bytes.size())
        return {window.bytes.begin()+offset-window.offset,window.bytes.begin()+end-window.offset};
    throw SnapshotDataUnavailable("Array window was not requested in this presentation; request a new coherent view");
}
uint32_t Presentation::propertySize(int32_t instanceId,int32_t propertyId) const {return value(instanceId,propertyId).size;}
const std::vector<int32_t>& Presentation::references(int32_t instanceId,int32_t propertyId) const {
    const auto& v=value(instanceId,propertyId);
    if(!v.hasReferences)throw SnapshotDataUnavailable("Reference values are absent from this presentation");
    return v.references;
}
const std::string& Presentation::stringValue(int32_t instanceId,int32_t propertyId) const {
    const auto& v=value(instanceId,propertyId);
    if(!v.hasString)throw SnapshotDataUnavailable("String value was not requested in this presentation");
    return v.string;
}
const RecordInfo& Presentation::arrayRecord(int32_t instanceId,int32_t propertyId,uint32_t index) const {
    const auto found=data->records.find(std::make_tuple(instanceId,propertyId,index));
    if(found==data->records.end())throw SnapshotDataUnavailable("Structure record was not requested in this presentation");
    return found->second;
}
const std::vector<std::pair<std::string,uint32_t>>& Presentation::namedValues(int32_t propertyId) const {
    if(data->schema){const auto found=data->schema->namedValues.find(propertyId);if(found!=data->schema->namedValues.end())return found->second;}
    throw std::out_of_range("Unknown property ID in presentation");
}
HierarchyChanges Presentation::hierarchyChanges(uint64_t previousRevision) const {
    if(previousRevision==data->hierarchy.revision) {HierarchyChanges result;result.revision=previousRevision;return result;}
    auto result=data->hierarchy;
    if(previousRevision+1!=result.revision)result.schemaChanged=true;
    return result;
}
const pvdapp::OvdEvidenceIdentity* Presentation::evidenceIdentity(int32_t id) const {
    const auto found=data->evidenceIdentities.find(id);
    return found==data->evidenceIdentities.end()?nullptr:&found->second;
}
const pvdapp::OvdEvidenceSampleStamp* Presentation::evidenceSampleStamp(int32_t id,int32_t property) const {
    const auto found=data->evidenceStamps.find(valueKey(id,property));
    return found==data->evidenceStamps.end()?nullptr:&found->second;
}
bool Presentation::hasRead(int32_t instanceId,int32_t propertyId,uint32_t offset,uint32_t limit) const {
    if(limit>8*1024*1024)return false;
    try {
        const auto& v=value(instanceId,propertyId);
        if(offset>=v.size || !limit)return true;
        const uint64_t end=std::min(uint64_t(v.size),uint64_t(offset)+limit);
        for(const auto& window:v.windows)
            if(offset>=window.offset && end<=uint64_t(window.offset)+window.bytes.size())return true;
        return false;
    } catch(const std::exception&) {return false;}
}
}
