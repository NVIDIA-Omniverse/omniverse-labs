/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "Geometry.h"
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace pvdapp { class PvdEngine; }
namespace ovdnext {
enum class OverlayKind:uint8_t { Point=1,Line=2,Triangle=3,Arrow=4 };
// Explicit semantic roles for generated primitives. Captured/query colors have
// role zero and must remain unchanged by an editable frontend palette.
enum class OverlayColorRole:uint8_t {
    Captured=0,Bounds=1,CenterOfMass=2,Velocity=3,AxisX=4,AxisY=5,AxisZ=6,
    JointParent=7,JointChild=8,ContactPoint=9,ContactNormal=10
};
enum OverlayCategory:uint32_t {
    OverlayBounds=1,OverlayAxes=2,OverlayVelocity=4,OverlayAngularVelocity=8,
    OverlayJoints=16,OverlayContacts=32,OverlayQueries=64,OverlayCustom=128,OverlayCenterOfMass=256,
    OverlayAll=511
};
struct QueryRecord {
    int32_t queryId,sceneInstanceId,propertyId,classId;
    uint32_t elementIndex,elementSize;
};
// Explicit joint scale is source-specific: structured OVD joints use base
// length and the Qt 0.5 floor; captured SDK events use the raw value directly.
// Omission retains the historical general-gizmo-coupled API behavior.
struct JointScaleRequest {
    bool independent=false;
    float scale=1.0f,baseLength=1.0f;
};
void validateJointScaleRequest(const JointScaleRequest& request,bool structuredOvd);
void validateFamilyScales(const std::map<uint32_t,float>& scales);
void validateMasterScale(float masterScale,float gizmoScale,const JointScaleRequest& jointScaling,bool structuredOvd,const std::map<uint32_t,float>& familyScales);
struct OverlaySnapshot {
    uint64_t segmentEpoch=0;
    uint32_t currentFrame=0, mask=0;
    bool ready=false;
    std::vector<float> positions,colors;
    std::vector<uint8_t> kinds,colorRoles;
    std::vector<uint32_t> categories;
    std::vector<int32_t> objectIds,relatedIds,sceneIds;
    std::vector<QueryRecord> queryRecords;
    std::vector<GeometryRequest> queryRequests;
    std::shared_ptr<GeometrySnapshot> queryGeometry;
    std::vector<std::string> diagnostics;
    JointScaleRequest jointScaling;
    std::map<uint32_t,float> familyScales;
    float masterScale=1.0f;
};
std::vector<QueryRecord> extractQueryRecords(pvdapp::PvdEngine& engine);
std::shared_ptr<OverlaySnapshot> extractOverlays(pvdapp::PvdEngine& engine,uint32_t mask,float gizmoScale,float toleranceSpeed,const JointScaleRequest& jointScaling={},const std::map<uint32_t,float>& familyScales={},float masterScale=1.0f);
}
