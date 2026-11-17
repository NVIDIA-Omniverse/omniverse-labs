# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
if(NOT OVD_SUPPORT)
    message(FATAL_ERROR "PVD3_BUILD_PYTHON requires the pinned PvdDom/PVDRuntime roots")
endif()
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

add_library(PVDSession STATIC Session.cpp Geometry.cpp Overlay.cpp Profiling.cpp)
target_include_directories(PVDSession PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(PVDSession PUBLIC PVDCore)

pybind11_add_module(_pvd_native bindings.cpp ProfilingBindings.cpp CaptureControlBindings.cpp)
target_link_libraries(_pvd_native PRIVATE PVDSession)
set_target_properties(_pvd_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)

# Native writer fixture for Python geometry integration tests; not packaged.
add_executable(pvd_geometry_fixture EXCLUDE_FROM_ALL ../../python/tests/native_geometry_fixture.cpp)
target_link_libraries(pvd_geometry_fixture PRIVATE PVDSession)

# Owned atomic frontend data; no GUI lock crosses this boundary.
target_sources(PVDSession PRIVATE Presentation.cpp)

target_sources(PVDSession PRIVATE ErrorStream.cpp)
target_sources(_pvd_native PRIVATE ErrorStreamBindings.cpp)

target_sources(_pvd_native PRIVATE PresentationBindings.cpp)
target_sources(_pvd_native PRIVATE EvidenceBindings.cpp)

# Exercise the original raw comm-stream loader independently of indexed fixtures.
add_executable(pvd_ingestion_fixture EXCLUDE_FROM_ALL ../../python/tests/native_ingestion_fixture.cpp)
target_link_libraries(pvd_ingestion_fixture PRIVATE PVDSession)

# Captured tolerance/ownership cases use the same native writer as production.
add_executable(pvd_tolerance_fixture EXCLUDE_FROM_ALL ../../python/tests/native_tolerance_fixture.cpp)
target_link_libraries(pvd_tolerance_fixture PRIVATE PVDSession)

# Bounded presence checks for analysis action availability.
target_sources(PVDSession PRIVATE AnalysisAvailability.cpp)
target_sources(_pvd_native PRIVATE AnalysisAvailabilityBindings.cpp)

add_executable(pvd_availability_fixture EXCLUDE_FROM_ALL ../../python/tests/native_availability_fixture.cpp)
target_link_libraries(pvd_availability_fixture PRIVATE PVDSession)

if(PVD3_BUILD_TESTS)
    add_executable(pvd_joint_calibration_fixture EXCLUDE_FROM_ALL ../../python/tests/native_joint_calibration_fixture.cpp)
    target_link_libraries(pvd_joint_calibration_fixture PRIVATE PVDSession)
endif()
