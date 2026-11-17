/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "ErrorStream.h"
#include "Session.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
namespace ovdnext {
void bindErrorStream(py::module_& m) {
    py::class_<ErrorStreamRecord>(m,"ErrorStreamRecord")
        .def_readonly("index",&ErrorStreamRecord::index).def_readonly("code",&ErrorStreamRecord::code)
        .def_readonly("line",&ErrorStreamRecord::line).def_readonly("message",&ErrorStreamRecord::message).def_readonly("file",&ErrorStreamRecord::file);
    py::class_<ErrorStreamSnapshot,std::shared_ptr<ErrorStreamSnapshot>>(m,"ErrorStreamSnapshot")
        .def_readonly("supported",&ErrorStreamSnapshot::supported).def_readonly("reason",&ErrorStreamSnapshot::reason)
        .def_readonly("has_log",&ErrorStreamSnapshot::hasLog).def_readonly("truncated",&ErrorStreamSnapshot::truncated)
        .def_readonly("scene_id",&ErrorStreamSnapshot::sceneId).def_readonly("current_frame",&ErrorStreamSnapshot::currentFrame)
        .def_readonly("log_instance_id",&ErrorStreamSnapshot::logInstanceId).def_readonly("total",&ErrorStreamSnapshot::total)
        .def_readonly("records",&ErrorStreamSnapshot::records);
    m.def("error_snapshot",&errorStreamSnapshot,py::arg("session"),py::arg("scene_id")=0,py::arg("max_records")=10000,py::call_guard<py::gil_scoped_release>());
}
}
