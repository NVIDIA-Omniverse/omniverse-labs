/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "Geometry.h"
#include "Analysis.h"
#include "Overlay.h"
#include "CaptureRequestState.h"
#include "../PVDCore/OvdEvidenceMetadata.h"
#include <cstdint>
#include <memory>
#include <functional>
#include <string>
#include <vector>

namespace pvdapp { class LoadControl; }
namespace pybind11 { class module_; }

namespace ovdnext {
class Presentation;
struct ReadRequest;
struct RecordRequest;
struct EvidenceMetadataPage {
    uint64_t captureKey=0,segmentEpoch=0,total=0,nextOffset=0;
    pvdapp::OvdEvidenceDomain domain;
    std::vector<pvdapp::OvdEvidenceIdentity> rows;
};
void bindEvidence(pybind11::module_& module);

struct SceneInfo {
    uint64_t id;
    std::string name;
    uint32_t frameCount;
    uint32_t currentFrame;
    int32_t instanceId;
    uint64_t streamId;
    bool alive;
    uint64_t recordingSegment = 0;
};
struct ClassInfo {
    int32_t id;
    int32_t baseId;
    uint32_t byteSize;
    std::string name;
};
struct InstanceInfo {
    int32_t id;
    int32_t classId;
    int32_t sceneId;
    int32_t physicsId;
    bool alive;
    bool sceneJoint;
};
struct PropertyInfo {
    int32_t id;
    std::string name;
    std::string type;
    std::string nativeType;
    std::string semantic;
    bool array;
    int64_t fieldCount;
};
struct HierarchyChanges {
    uint64_t revision=0;
    bool schemaChanged=false,queryRecordsChanged=false;
    std::vector<int32_t> changedIds,removedIds;
};
struct ExportCapabilities {
    std::string format,saveReason,clipReason,repxReason;
    bool save=false,clip=false,repx=false;
};
struct RecordField {
    int32_t id;
    std::string name,type,nativeType,semantic,stringValue;
    bool array;
    std::vector<uint8_t> data;
};
struct RecordInfo {
    int32_t classId;
    std::vector<RecordField> fields;
};
struct SessionInfo {
    bool ovd;
    uint32_t instanceCount;
    uint64_t maxFrame;
    bool live;
    bool connected;
    bool ended;
    uint64_t completedSerial;
    uint64_t segmentEpoch;
    std::string error;
    bool unpublishedChanges;
    uint64_t liveMaxFrame;
    uint64_t liveCurrentFrame;
    uint64_t recordingSize;
};

struct LiveUpdate {
    bool ready = false;
    bool changed = false;
    uint64_t serial = 0, segmentEpoch = 0, currentFrame = 0, maxFrame = 0;
    uint64_t segmentTransitionSerial = 0;
    std::vector<SceneInfo> scenes;
};

// A toolkit-independent owning boundary. No Qt/Python types or borrowed engine
// pointers cross this interface. Bulk reads are bounded copies so a seek/close or
// live index publication cannot invalidate a Python view.
class Session {
public:
    Session(const std::string& path, const std::string& temporaryDirectory,
            std::shared_ptr<pvdapp::LoadControl> control = nullptr);
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    static std::shared_ptr<Session> listen(uint16_t port, const std::string& recordingPath);
    uint64_t captureKey();
    CaptureControlInfo captureControlInfo();
    pvdapp::CaptureCatalogPage captureCatalog(uint64_t generation, uint64_t offset, uint32_t limit,
                                             uint64_t parent, const std::string& prefix);
    uint64_t requestCapture(const std::vector<uint64_t>& scenes, const std::vector<uint64_t>& objects,
                            bool secondary, bool all, uint64_t generation, bool supersede);
    uint64_t requestCaptureCatalog(bool supersede);
    std::shared_ptr<Presentation> presentation(uint64_t sceneId,int64_t frame,const std::vector<int32_t>& inspectIds,const std::vector<ReadRequest>& readRequests,const std::vector<RecordRequest>& recordRequests,uint32_t overlayMask,float gizmoScale,float toleranceSpeed,bool allowUnavailableRequests,bool capturedToleranceSpeed=false,const JointScaleRequest& jointScaling={},const std::map<uint32_t,float>& familyScales={},float masterScale=1.0f,bool includeEvidence=false);
    EvidenceMetadataPage evidenceMetadata(uint32_t offset,uint32_t limit);
    SessionInfo info();
    std::shared_ptr<GeometrySnapshot> geometrySnapshot();
    std::shared_ptr<OverlaySnapshot> overlaySnapshot(uint32_t mask,float gizmoScale,float toleranceSpeed,const JointScaleRequest& jointScaling={},const std::map<uint32_t,float>& familyScales={},float masterScale=1.0f);
    uint64_t hierarchyRevision();
    HierarchyChanges hierarchyChanges(uint64_t previousRevision);
    std::vector<QueryRecord> queryRecords();
    RecordInfo arrayRecord(int32_t instanceId,int32_t propertyId,uint32_t elementIndex);
    std::vector<std::pair<std::string,uint32_t>> namedValues(int32_t propertyId);
    ExportCapabilities exportCapabilities();
    uint64_t save(const std::string& path,bool overwrite);
    uint64_t saveClip(const std::string& path,uint64_t sceneId,uint32_t start,uint32_t stop,bool overwrite);
    uint64_t exportRepX(const std::string& path,const std::vector<int32_t>& ids,bool all,bool overwrite);
    uint64_t messageCount();
    std::vector<MessageInfo> messages(uint32_t offset, uint32_t limit);
    FrameTiming frameTiming();
    std::shared_ptr<ScalarHistory> scalarHistory(uint64_t sceneId, int32_t instanceId, int32_t propertyId, uint32_t start, uint32_t stop);
    std::vector<SceneInfo> scenes();
    std::vector<ClassInfo> classes();
    std::vector<InstanceInfo> instances(uint32_t offset, uint32_t limit);
    std::vector<PropertyInfo> properties(int32_t classId);
    std::vector<uint8_t> read(int32_t instanceId, int32_t propertyId, uint32_t offset, uint32_t limit);
    uint32_t propertySize(int32_t instanceId, int32_t propertyId);
    std::vector<int32_t> references(int32_t instanceId, int32_t propertyId);
    std::string stringValue(int32_t instanceId, int32_t propertyId);
    bool seek(uint64_t sceneId, uint32_t frame);
    void refresh();
    LiveUpdate refreshLive();
    void stopLive();
    void flushRecording();
    std::string recordingPath();

private:
    friend struct ProfilingAccess;
    void withEngineLocked(const std::function<void(pvdapp::PvdEngine&)>& callback);
    Session(uint16_t port, const std::string& recordingPath);
    struct Impl;
    std::unique_ptr<Impl> m;
};

} // namespace ovdnext
