/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>

namespace pvdapp {
// Owned source namespace, deliberately separate from synthetic instance IDs and
// the viewer's publication epoch. No DOM pointer survives this boundary.
struct OvdEvidenceLifeSpan { uint64_t first=0,stop=0; };
struct OvdEvidenceFrameScope {
    uint64_t segment=0,context=0,sceneUid=0,first=0,last=0,count=0;
    bool sceneMapped=false,monotonic=true,consecutive=true,sceneRecreated=false;
};
struct OvdEvidenceFrameScopes {
    bool complete=false;
    std::vector<OvdEvidenceFrameScope> scopes;
};
struct OvdEvidenceDomain {
    bool supported=false,sawSegmentMetadata=false,frameScopesComplete=false;
    uint64_t minFrame=0,maxFrame=0;
    std::vector<uint64_t> segmentMaxima;
    std::vector<OvdEvidenceFrameScope> scopes;
};
struct OvdEvidenceIdentity {
    bool supported=false;
    int32_t instanceId=0,sceneInstanceId=-1;
    uint64_t objectUid=0,apiHandle=0,recordingSegment=0,sceneContext=0,sceneObjectUid=0;
    std::string className;
    std::vector<OvdEvidenceLifeSpan> lifespans;
};
struct OvdEvidenceSampleStamp { bool supported=false,hasSample=false;uint64_t frame=0; };
inline uint64_t evidenceBytes(const OvdEvidenceIdentity& v) {
    return sizeof(v)+v.className.size()+v.lifespans.size()*sizeof(OvdEvidenceLifeSpan);
}
inline uint64_t evidenceBytes(const OvdEvidenceDomain& v) {
    return sizeof(v)+v.segmentMaxima.size()*sizeof(uint64_t)+v.scopes.size()*sizeof(OvdEvidenceFrameScope);
}
constexpr uint64_t OvdEvidencePageBytes=8*1024*1024;
}
