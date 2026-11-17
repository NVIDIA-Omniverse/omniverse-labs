/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "Profiling.h"
#include "Session.h"
#include "PxProfileEventSender.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
namespace ovdnext {
void bindProfiling(py::module_& m) {
    m.attr("CROSS_THREAD_ID")=static_cast<uint32_t>(physx::profile::PxProfileEventSender::CrossThreadId);
    py::class_<ProfileEventName>(m,"ProfileEventName")
        .def_readonly("id",&ProfileEventName::id).def_readonly("name",&ProfileEventName::name).def_readonly("enabled",&ProfileEventName::enabled);
    py::class_<ProfileZoneInfo>(m,"ProfileZoneInfo")
        .def_readonly("id",&ProfileZoneInfo::id).def_readonly("stream_id",&ProfileZoneInfo::streamId)
        .def_readonly("name",&ProfileZoneInfo::name).def_readonly("events",&ProfileZoneInfo::events);
    py::class_<ProfileScope>(m,"ProfileScope")
        .def_readonly("zone_id",&ProfileScope::zoneId).def_readonly("event_id",&ProfileScope::eventId)
        .def_readonly("thread_id",&ProfileScope::threadId).def_readonly("context_id",&ProfileScope::contextId)
        .def_readonly("name",&ProfileScope::name).def_readonly("depth",&ProfileScope::depth)
        .def_readonly("start_tick",&ProfileScope::startTick).def_readonly("end_tick",&ProfileScope::endTick)
        .def_readonly("start_ms",&ProfileScope::startMs).def_readonly("end_ms",&ProfileScope::endMs).def_readonly("duration_ms",&ProfileScope::durationMs);
    py::class_<ProfileSummary>(m,"ProfileSummary")
        .def_readonly("zone_id",&ProfileSummary::zoneId).def_readonly("event_id",&ProfileSummary::eventId)
        .def_readonly("thread_id",&ProfileSummary::threadId).def_readonly("context_id",&ProfileSummary::contextId)
        .def_readonly("name",&ProfileSummary::name).def_readonly("count",&ProfileSummary::count).def_readonly("total_ticks",&ProfileSummary::totalTicks)
        .def_readonly("total_ms",&ProfileSummary::totalMs).def_readonly("average_ms",&ProfileSummary::averageMs)
        .def_readonly("min_ms",&ProfileSummary::minMs).def_readonly("max_ms",&ProfileSummary::maxMs).def_readonly("percent",&ProfileSummary::percent);
    py::class_<ProfileBoundary>(m,"ProfileBoundary")
        .def_readonly("zone_id",&ProfileBoundary::zoneId).def_readonly("event_id",&ProfileBoundary::eventId)
        .def_readonly("thread_id",&ProfileBoundary::threadId).def_readonly("context_id",&ProfileBoundary::contextId)
        .def_readonly("starts",&ProfileBoundary::starts).def_readonly("stops",&ProfileBoundary::stops);
    py::class_<ProfileCounter>(m,"ProfileCounter")
        .def_readonly("zone_id",&ProfileCounter::zoneId).def_readonly("event_id",&ProfileCounter::eventId)
        .def_readonly("thread_id",&ProfileCounter::threadId).def_readonly("context_id",&ProfileCounter::contextId)
        .def_readonly("name",&ProfileCounter::name).def_readonly("value",&ProfileCounter::value);
    py::class_<ProfileSnapshot,std::shared_ptr<ProfileSnapshot>>(m,"ProfileSnapshot")
        .def_readonly("supported",&ProfileSnapshot::supported).def_readonly("reason",&ProfileSnapshot::reason)
        .def_readonly("scene_id",&ProfileSnapshot::sceneId).def_readonly("current_frame",&ProfileSnapshot::currentFrame)
        .def_readonly("start_event",&ProfileSnapshot::startEvent).def_readonly("end_event",&ProfileSnapshot::endEvent)
        .def_readonly("start_tick",&ProfileSnapshot::startTick).def_readonly("end_tick",&ProfileSnapshot::endTick)
        .def_readonly("tick_numerator",&ProfileSnapshot::tickNumerator).def_readonly("tick_denominator",&ProfileSnapshot::tickDenominator)
        .def_readonly("duration_ms",&ProfileSnapshot::durationMs).def_readonly("raw_event_count",&ProfileSnapshot::rawEventCount)
        .def_readonly("unmatched_starts",&ProfileSnapshot::unmatchedStarts).def_readonly("unmatched_stops",&ProfileSnapshot::unmatchedStops)
        .def_readonly("ignored_events",&ProfileSnapshot::ignoredEvents).def_readonly("zones",&ProfileSnapshot::zones)
        .def_readonly("context_names",&ProfileSnapshot::contextNames)
        .def_readonly("scopes",&ProfileSnapshot::scopes).def_readonly("summaries",&ProfileSnapshot::summaries)
        .def_readonly("unmatched",&ProfileSnapshot::unmatched).def_readonly("counters",&ProfileSnapshot::counters).def_readonly("diagnostics",&ProfileSnapshot::diagnostics);
    py::class_<MemoryRow>(m,"MemoryRow")
        .def_readonly("file",&MemoryRow::file).def_readonly("type",&MemoryRow::type).def_readonly("line",&MemoryRow::line)
        .def_readonly("allocated_bytes",&MemoryRow::allocatedBytes).def_readonly("deallocated_bytes",&MemoryRow::deallocatedBytes)
        .def_readonly("allocation_count",&MemoryRow::allocationCount).def_readonly("deallocation_count",&MemoryRow::deallocationCount)
        .def_property_readonly("churn_bytes",[](const MemoryRow& r){return int64_t(r.allocatedBytes)-int64_t(r.deallocatedBytes);});
    py::class_<MemoryOutstanding>(m,"MemoryOutstanding")
        .def_readonly("file",&MemoryOutstanding::file).def_readonly("type",&MemoryOutstanding::type).def_readonly("line",&MemoryOutstanding::line)
        .def_readonly("index",&MemoryOutstanding::index).def_readonly("valid",&MemoryOutstanding::valid)
        .def_property_readonly("bytes",[](const MemoryOutstanding& v)->py::object {return v.valid?py::cast(v.bytes):py::none();});
    py::class_<MemorySnapshot,std::shared_ptr<MemorySnapshot>>(m,"MemorySnapshot")
        .def_readonly("supported",&MemorySnapshot::supported).def_readonly("reason",&MemorySnapshot::reason)
        .def_readonly("scene_id",&MemorySnapshot::sceneId).def_readonly("current_frame",&MemorySnapshot::currentFrame)
        .def_readonly("start_event",&MemorySnapshot::startEvent).def_readonly("end_event",&MemorySnapshot::endEvent)
        .def_readonly("raw_event_count",&MemorySnapshot::rawEventCount).def_readonly("unmatched_deallocations",&MemorySnapshot::unmatchedDeallocations)
        .def_readonly("zero_allocations",&MemorySnapshot::zeroAllocations).def_readonly("total",&MemorySnapshot::total)
        .def_readonly("rows",&MemorySnapshot::rows).def_readonly("outstanding",&MemorySnapshot::outstanding)
        .def_readonly("diagnostics",&MemorySnapshot::diagnostics);
    m.def("profile_snapshot",&profileSnapshot,py::arg("session"),py::arg("scene_id")=0,py::arg("max_events")=200000,py::arg("include_context_names")=true,
        py::call_guard<py::gil_scoped_release>(),"Owned current-frame profile scopes; unsupported captures carry a reason, never fake timings.");
    m.def("memory_snapshot",&memorySnapshot,py::arg("session"),py::arg("scene_id")=0,py::arg("max_events")=200000,
        py::call_guard<py::gil_scoped_release>(),"Owned native frame allocation summaries and separately validated native outstanding totals.");
}
}
