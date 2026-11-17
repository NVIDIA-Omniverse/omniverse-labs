/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "Session.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
using namespace pybind11::literals;
namespace ovdnext {
namespace {
uint64_t integer(py::handle value) {
    if(!PyLong_Check(value.ptr()) || PyBool_Check(value.ptr())) throw py::type_error("Expected an unsigned 64-bit integer");
    return py::cast<uint64_t>(value);
}
std::vector<uint64_t> selectors(const py::sequence& values) {
    if(values.size()>65536) throw py::value_error("Capture selection exceeds 65536 selectors");
    std::vector<uint64_t> result; result.reserve(values.size());
    for(auto value:values) result.push_back(integer(value));
    return result;
}
py::dict usage(const OvdResourceUsage& u) {
    return py::dict("stream_bytes"_a=u.streamBytes,"commands"_a=u.commands,"attribute_mutations"_a=u.attributeMutations,
        "payload_bytes"_a=u.payloadBytes,"classes"_a=u.classes,"attributes"_a=u.attributes,
        "object_creates"_a=u.objectCreates,"messages"_a=u.messages,"retained_dom_bytes"_a=u.retainedDomBytes);
}
py::dict result(const CaptureRequestResult& r) {
    return py::dict("id"_a=r.id,"state"_a=r.state,"catalog_only"_a=r.catalogOnly,"secondary"_a=r.secondary,"all"_a=r.all,
        "scene_count"_a=r.sceneCount,"object_count"_a=r.objectCount,"ack_status"_a=r.ackStatus,"epoch"_a=r.epoch,
        "queued_at"_a=r.queuedAt,"sent_at"_a=r.sentAt,"completed_at"_a=r.completedAt);
}
}
void bindCaptureControl(py::class_<Session,std::shared_ptr<Session>>& session) {
    session.def("capture_control_info",[](Session& s) {
        CaptureControlInfo i;
        { py::gil_scoped_release release; i=s.captureControlInfo(); }
        py::list history; for(const auto& r:i.results) history.append(result(r));
        const auto& l=i.budget.limits;
        return py::dict("connected"_a=i.connected,"capabilities"_a=i.capabilities,"can_apply"_a=i.canApply,
            "can_refresh"_a=i.canRefresh,"reason"_a=i.reason,"state"_a=i.request.state,"request_id"_a=i.request.id,
            "request"_a=result(i.request),"results"_a=history,"issued_high_water"_a=i.issuedHighWater,
            "ack_classification"_a=i.ackClassification,"ack_id"_a=i.metadata.ackId,"ack_status"_a=i.metadata.ackStatus,
            "ack_epoch"_a=i.metadata.ackEpoch,"capture_epoch"_a=i.metadata.captureEpoch,"policy_known"_a=i.metadata.policyKnown,
            "actual_all"_a=i.metadata.all,"actual_secondary"_a=i.metadata.secondary,
            "catalog_ready"_a=i.catalog.ready,"generation"_a=i.catalog.generation,"catalog_rows"_a=i.catalog.rows,
            "catalog_revision"_a=i.catalog.revision,"metadata_revision"_a=i.metadata.revision,
            "metadata_boundary_serial"_a=i.metadata.boundarySerial,"segment_transition_serial"_a=i.completedBoundarySerial,
            "has_completed_frame"_a=i.hasCompletedFrame,
            "bytes_sent"_a=i.transport.bytesSent,"request_bytes"_a=i.transport.totalBytes,
            "forward_bytes"_a=i.forwardBytes,"forward_idle_seconds"_a=i.forwardIdleSeconds,
            "used"_a=usage(i.budget.used),"previous_segment"_a=usage(i.budget.previousSegment),
            "limits"_a=usage({l.maxStreamBytes,l.maxCommands,l.maxAttributeMutations,l.maxCumulativeAttributePayloadBytes,
                l.maxClasses,l.maxAttributes,l.maxObjectCreates,l.maxMessages,l.maxRetainedDomBytes}),
            "previous_catalog_bytes"_a=i.budget.previousCatalogBytes);
    });
    session.def("capture_catalog",[](Session& s,py::object gen,py::object off,py::object lim,py::object parent,const std::string& prefix) {
        const auto generation=integer(gen),offset=integer(off),limit=integer(lim),owner=integer(parent);
        if(limit>1000) throw py::value_error("Catalog page exceeds 1000 rows");
        pvdapp::CaptureCatalogPage page;
        { py::gil_scoped_release release; page=s.captureCatalog(generation,offset,static_cast<uint32_t>(limit),owner,prefix); }
        py::list entries;
        for(const auto& r:page.entries) entries.append(py::dict("target_handle"_a=r.targetHandle,"owner_handle"_a=r.ownerHandle,
            "scene_handle"_a=r.sceneHandle,"type_name"_a=r.typeName,"name"_a=r.name,"can_select"_a=r.canSelect));
        return py::dict("state"_a=(page.state==pvdapp::CaptureCatalogPageState::Ready?"ready":
            page.state==pvdapp::CaptureCatalogPageState::Stale?"stale":"busy"),"generation"_a=page.generation,
            "total"_a=page.total,"copied_bytes"_a=page.copiedBytes,"entries"_a=entries);
    },py::arg("generation"),py::arg("offset")=0,py::arg("limit")=1000,py::arg("parent")=0,py::arg("name_prefix")="");
    session.def("request_capture",[](Session& s,const py::sequence& scenes,const py::sequence& objects,bool secondary,bool all,
                                     py::object gen,bool supersede) {
        if(scenes.size()+objects.size()>65536) throw py::value_error("Capture selection exceeds 65536 selectors");
        const auto selectedScenes=selectors(scenes),selectedObjects=selectors(objects); const auto generation=integer(gen);
        py::gil_scoped_release release; return s.requestCapture(selectedScenes,selectedObjects,secondary,all,generation,supersede);
    },py::arg("scenes"),py::arg("objects"),py::arg("secondary"),py::arg("all"),py::arg("generation"),py::arg("supersede_unknown")=false);
    session.def("request_capture_catalog",&Session::requestCaptureCatalog,py::arg("supersede_unknown")=false,
                py::call_guard<py::gil_scoped_release>());
}
}
