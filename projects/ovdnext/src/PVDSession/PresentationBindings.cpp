/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "Presentation.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
namespace ovdnext {
void bindPresentation(py::module_& m) {
    py::register_exception<SnapshotDataUnavailable>(m,"SnapshotDataUnavailable",PyExc_RuntimeError);
    py::class_<CaptureIdentity,std::shared_ptr<CaptureIdentity>>(m,"CaptureIdentity").def_readonly("key",&CaptureIdentity::key);
    py::class_<Presentation,std::shared_ptr<Presentation>>(m,"Presentation")
        .def("evidence_identity",&Presentation::evidenceIdentity,py::return_value_policy::copy)
        .def("evidence_sample_stamp",&Presentation::evidenceSampleStamp,py::return_value_policy::copy)
        .def_property_readonly("evidence_domain",[](const Presentation& p){return p.data->evidenceDomain;})
        .def_readonly("request_applied",&Presentation::requestApplied)
        .def_property_readonly("data_backend",[](const Presentation& p){return p.data->info.ovd?"pvddom":"pvd";})
        .def("property_availability",&Presentation::propertyAvailability)
        .def_property_readonly("ready",[](const Presentation& p){return p.data->ready;})
        .def_property_readonly("capture_key",[](const Presentation& p){return p.data->owner->key;})
        .def_property_readonly("owner_token",[](const Presentation& p){return p.data->owner;})
        .def_property_readonly("generation",[](const Presentation& p){return p.data->generation;})
        .def_property_readonly("serial",[](const Presentation& p){return p.data->serial;})
        .def_property_readonly("scene_id",[](const Presentation& p){return p.data->sceneId;})
        .def_property_readonly("frame",[](const Presentation& p){return p.data->frame;})
        .def_property_readonly("segment_epoch",[](const Presentation& p){return p.data->info.segmentEpoch;})
        .def_property_readonly("tolerance_length",[](const Presentation& p){return p.data->toleranceLength;})
        .def_property_readonly("tolerance_length_source_id",[](const Presentation& p){return p.data->toleranceLengthSourceId;})
        .def_property_readonly("tolerance_length_property_id",[](const Presentation& p){return p.data->toleranceLengthPropertyId;})
        .def_property_readonly("tolerance_length_resolution",[](const Presentation& p){return p.data->toleranceLengthResolution;})
        .def_property_readonly("joint_scale",[](const Presentation& p)->py::object{return p.data->jointScaling.independent?py::cast(p.data->jointScaling.scale):py::none();})
        .def_property_readonly("joint_base_length",[](const Presentation& p){return p.data->jointScaling.baseLength;})
        .def_property_readonly("tolerance_speed",[](const Presentation& p){return p.data->toleranceSpeed;})
        .def_property_readonly("tolerance_speed_source_id",[](const Presentation& p){return p.data->toleranceSpeedSourceId;})
        .def_property_readonly("tolerance_speed_property_id",[](const Presentation& p){return p.data->toleranceSpeedPropertyId;})
        .def_property_readonly("tolerance_speed_resolution",[](const Presentation& p){return p.data->toleranceSpeedResolution;})
        .def_property_readonly("copied_bytes",[](const Presentation& p){return p.data->copiedBytes;})
        .def_property_readonly("extraction_ms",[](const Presentation& p){return p.data->extractionMs;})
        .def_property_readonly("inspected_ids",[](const Presentation& p){return p.data->inspectedIds;})
        .def_property_readonly("unavailable_ids",[](const Presentation& p){return p.data->unavailableIds;})
        .def_property_readonly("unavailable_reads",[](const Presentation& p){
            py::list result;for(const auto& r:p.data->unavailableReads)result.append(py::make_tuple(r.instanceId,r.propertyId,r.offset,r.limit));return result;
        })
        .def_property_readonly("unavailable_records",[](const Presentation& p){
            py::list result;for(const auto& r:p.data->unavailableRecords)result.append(py::make_tuple(r.instanceId,r.propertyId,r.index));return result;
        })
        .def("info",&Presentation::info)
        .def("scenes",&Presentation::scenes,py::return_value_policy::copy)
        .def("classes",&Presentation::classes,py::return_value_policy::copy)
        .def("instances",&Presentation::instances,py::arg("offset")=0,py::arg("limit")=10000)
        .def("properties",&Presentation::properties,py::return_value_policy::copy)
        .def("read",[](const Presentation& p,int32_t instance,int32_t property,uint32_t offset,uint32_t limit){
            const auto bytes=p.read(instance,property,offset,limit);return py::bytes(reinterpret_cast<const char*>(bytes.data()),bytes.size());
        },py::arg("instance_id"),py::arg("property_id"),py::arg("offset")=0,py::arg("limit")=65536)
        .def("has_read",&Presentation::hasRead,py::arg("instance_id"),py::arg("property_id"),py::arg("offset")=0,py::arg("limit")=65536)
        .def("property_size",&Presentation::propertySize)
        .def("references",&Presentation::references,py::return_value_policy::copy)
        .def("string_value",&Presentation::stringValue,py::return_value_policy::copy)
        .def("array_record",&Presentation::arrayRecord,py::arg("instance_id"),py::arg("property_id"),py::arg("element_index"),py::return_value_policy::copy)
        .def("named_values",&Presentation::namedValues,py::return_value_policy::copy)
        .def("query_records",[](const Presentation& p){return p.data->queries;})
        .def("hierarchy_revision",[](const Presentation& p){return p.data->hierarchy.revision;})
        .def("hierarchy_changes",&Presentation::hierarchyChanges)
        .def("geometry_snapshot",[](const Presentation& p){return p.data->geometry;})
        .def("overlay_snapshot",[](const Presentation& p){return p.data->overlays;});
}
}
