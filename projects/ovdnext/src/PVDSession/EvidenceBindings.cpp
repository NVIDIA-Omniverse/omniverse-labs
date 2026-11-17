/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "Session.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
namespace ovdnext {
void bindEvidence(py::module_& m) {
    using namespace pvdapp;
    py::class_<OvdEvidenceLifeSpan>(m,"EvidenceLifeSpan")
        .def_readonly("first",&OvdEvidenceLifeSpan::first).def_readonly("stop",&OvdEvidenceLifeSpan::stop);
    py::class_<OvdEvidenceIdentity>(m,"EvidenceIdentity")
        .def_readonly("supported",&OvdEvidenceIdentity::supported)
        .def_readonly("instance_id",&OvdEvidenceIdentity::instanceId)
        .def_readonly("scene_instance_id",&OvdEvidenceIdentity::sceneInstanceId)
        .def_readonly("object_uid",&OvdEvidenceIdentity::objectUid)
        .def_readonly("api_handle",&OvdEvidenceIdentity::apiHandle)
        .def_readonly("recording_segment",&OvdEvidenceIdentity::recordingSegment)
        .def_readonly("scene_context",&OvdEvidenceIdentity::sceneContext)
        .def_readonly("scene_object_uid",&OvdEvidenceIdentity::sceneObjectUid)
        .def_readonly("class_name",&OvdEvidenceIdentity::className)
        .def_readonly("lifespans",&OvdEvidenceIdentity::lifespans);
    py::class_<OvdEvidenceFrameScope>(m,"EvidenceFrameScope")
        .def_readonly("segment",&OvdEvidenceFrameScope::segment).def_readonly("context",&OvdEvidenceFrameScope::context)
        .def_readonly("scene_uid",&OvdEvidenceFrameScope::sceneUid).def_readonly("first",&OvdEvidenceFrameScope::first)
        .def_readonly("last",&OvdEvidenceFrameScope::last).def_readonly("count",&OvdEvidenceFrameScope::count)
        .def_readonly("scene_mapped",&OvdEvidenceFrameScope::sceneMapped)
        .def_readonly("monotonic",&OvdEvidenceFrameScope::monotonic).def_readonly("consecutive",&OvdEvidenceFrameScope::consecutive)
        .def_readonly("scene_recreated",&OvdEvidenceFrameScope::sceneRecreated);
    py::class_<OvdEvidenceDomain>(m,"EvidenceDomain")
        .def_readonly("supported",&OvdEvidenceDomain::supported)
        .def_readonly("saw_segment_metadata",&OvdEvidenceDomain::sawSegmentMetadata)
        .def_readonly("frame_scopes_complete",&OvdEvidenceDomain::frameScopesComplete)
        .def_readonly("min_frame",&OvdEvidenceDomain::minFrame).def_readonly("max_frame",&OvdEvidenceDomain::maxFrame)
        .def_readonly("segment_maxima",&OvdEvidenceDomain::segmentMaxima).def_readonly("scopes",&OvdEvidenceDomain::scopes);
    py::class_<OvdEvidenceSampleStamp>(m,"EvidenceSampleStamp")
        .def_readonly("supported",&OvdEvidenceSampleStamp::supported)
        .def_readonly("has_sample",&OvdEvidenceSampleStamp::hasSample).def_readonly("frame",&OvdEvidenceSampleStamp::frame);
    py::class_<EvidenceMetadataPage>(m,"EvidenceMetadataPage")
        .def_readonly("capture_key",&EvidenceMetadataPage::captureKey).def_readonly("segment_epoch",&EvidenceMetadataPage::segmentEpoch)
        .def_readonly("domain",&EvidenceMetadataPage::domain).def_readonly("total",&EvidenceMetadataPage::total)
        .def_readonly("rows",&EvidenceMetadataPage::rows)
        .def_property_readonly("next_offset",[](const EvidenceMetadataPage& p)->py::object {
            return p.nextOffset>=p.total?py::none():py::cast(p.nextOffset);
        });
}
}
