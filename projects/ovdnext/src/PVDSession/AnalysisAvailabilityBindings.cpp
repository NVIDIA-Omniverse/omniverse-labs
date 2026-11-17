/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "AnalysisAvailability.h"
#include "Session.h"
#include "LoadControl.h"
#include <pybind11/pybind11.h>
namespace py=pybind11;
namespace ovdnext {
void bindAnalysisAvailability(py::module_& m) {
    py::class_<AnalysisPresence>(m,"AnalysisPresence")
        .def_readonly("known",&AnalysisPresence::known).def_readonly("available",&AnalysisPresence::available)
        .def_readonly("reason",&AnalysisPresence::reason);
    py::class_<AnalysisAvailability>(m,"AnalysisAvailability")
        .def_readonly("profile",&AnalysisAvailability::profile).def_readonly("memory",&AnalysisAvailability::memory)
        .def_readonly("errors",&AnalysisAvailability::errors).def_readonly("frame_timing",&AnalysisAvailability::frameTiming)
        .def_readonly("profile_zones",&AnalysisAvailability::profileZones).def_readonly("memory_entries",&AnalysisAvailability::memoryEntries)
        .def_readonly("scanned_instances",&AnalysisAvailability::scannedInstances).def_readonly("error_log",&AnalysisAvailability::errorLog)
        .def_readonly("memory_bytes",&AnalysisAvailability::memoryBytes).def_readonly("memory_events",&AnalysisAvailability::memoryEvents);
    m.def("analysis_availability",&analysisAvailability,py::arg("session"),py::arg("scene_id")=0,
        py::arg("control")=nullptr,py::arg("max_instances")=100000,py::arg("max_events")=8192,
        py::arg("max_bytes")=1024*1024,py::call_guard<py::gil_scoped_release>());
}
}
