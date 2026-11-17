/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "Session.h"
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace pybind11 { class module_; }
namespace ovdnext {
struct CaptureIdentity {
    CaptureIdentity();
    const uint64_t key;
};
struct ReadRequest { int32_t instanceId,propertyId;uint32_t offset,limit; };
struct RecordRequest { int32_t instanceId,propertyId;uint32_t index; };
class SnapshotDataUnavailable:public std::runtime_error {
public: using std::runtime_error::runtime_error;
};
struct PresentationSchema {
    std::vector<ClassInfo> classes;
    std::unordered_map<int32_t,std::vector<PropertyInfo>> properties;
    std::unordered_map<int32_t,std::vector<std::pair<std::string,uint32_t>>> namedValues;
};
struct PresentationValue {
    uint32_t size=0;
    bool hasString=false,hasReferences=false;
    bool samplePresent=true;
    std::string string;
    std::vector<int32_t> references;
    struct Window {uint32_t offset;std::vector<uint8_t> bytes;};
    std::vector<Window> windows;
};
struct PresentationData {
    std::shared_ptr<CaptureIdentity> owner;
    bool ready=false;
    uint64_t generation=0,sceneId=0,serial=0,copiedBytes=0;
    uint32_t frame=0;
    double extractionMs=0;
    float toleranceLength=1.0f;
    int32_t toleranceLengthSourceId=0,toleranceLengthPropertyId=-1;
    std::string toleranceLengthResolution="missing";
    float toleranceSpeed=10.0f;
    int32_t toleranceSpeedSourceId=0,toleranceSpeedPropertyId=-1;
    std::string toleranceSpeedResolution="explicit";
    SessionInfo info{};
    std::vector<SceneInfo> scenes;
    std::shared_ptr<PresentationSchema> schema;
    std::vector<InstanceInfo> instances;
    std::unordered_map<int32_t,size_t> instanceIndex;
    std::unordered_map<uint64_t,PresentationValue> values;
    std::map<std::tuple<int32_t,int32_t,uint32_t>,RecordInfo> records;
    std::vector<int32_t> inspectedIds,unavailableIds;
    std::vector<ReadRequest> unavailableReads;
    std::vector<RecordRequest> unavailableRecords;
    std::vector<QueryRecord> queries;
    std::shared_ptr<GeometrySnapshot> geometry;
    std::shared_ptr<OverlaySnapshot> overlays;
    HierarchyChanges hierarchy;
    JointScaleRequest jointScaling;
    pvdapp::OvdEvidenceDomain evidenceDomain;
    std::unordered_map<int32_t,pvdapp::OvdEvidenceIdentity> evidenceIdentities;
    std::unordered_map<uint64_t,pvdapp::OvdEvidenceSampleStamp> evidenceStamps;
};
// Resolves only from this publication's owned schema, values and identities.
void resolveCapturedTolerances(PresentationData& data,int32_t sceneInstanceId,bool capturedSpeed);

// Immutable once published. Every read uses owned data, including delayed GUI
// array callbacks. The source engine and producer mutex are never touched.
class Presentation {
public:
    explicit Presentation(std::shared_ptr<PresentationData> data,bool applied=false):data(std::move(data)),requestApplied(applied) {}
    std::shared_ptr<PresentationData> data;
    bool requestApplied;
    const pvdapp::OvdEvidenceIdentity* evidenceIdentity(int32_t id) const;
    const pvdapp::OvdEvidenceSampleStamp* evidenceSampleStamp(int32_t id,int32_t property) const;
    SessionInfo info() const {return data->info;}
    const std::vector<SceneInfo>& scenes() const {return data->scenes;}
    const std::vector<ClassInfo>& classes() const;
    std::vector<InstanceInfo> instances(uint32_t offset,uint32_t limit) const;
    const std::vector<PropertyInfo>& properties(int32_t classId) const;
    std::vector<uint8_t> read(int32_t instanceId,int32_t propertyId,uint32_t offset,uint32_t limit) const;
    uint32_t propertySize(int32_t instanceId,int32_t propertyId) const;
    const std::vector<int32_t>& references(int32_t instanceId,int32_t propertyId) const;
    const std::string& stringValue(int32_t instanceId,int32_t propertyId) const;
    const RecordInfo& arrayRecord(int32_t instanceId,int32_t propertyId,uint32_t index) const;
    const std::vector<std::pair<std::string,uint32_t>>& namedValues(int32_t propertyId) const;
    HierarchyChanges hierarchyChanges(uint64_t previousRevision) const;
    bool hasRead(int32_t instanceId,int32_t propertyId,uint32_t offset,uint32_t limit) const;
    std::string propertyAvailability(int32_t instanceId,int32_t propertyId) const;
    static uint64_t valueKey(int32_t instanceId,int32_t propertyId) {return uint64_t(uint32_t(instanceId))<<32|uint32_t(propertyId);}
private:
    const PresentationValue& value(int32_t instanceId,int32_t propertyId) const;
};
void bindPresentation(pybind11::module_& module);
}
