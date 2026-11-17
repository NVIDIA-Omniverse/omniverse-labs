# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.18)

set(_ovddataomnipvd_tests_default OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(OvdDataOmniPvd LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    enable_testing()

    if(NOT PVDRUNTIME_ROOT)
        message(FATAL_ERROR "PVDRUNTIME_ROOT must name the pvdruntime source directory")
    endif()
    get_filename_component(PVDRUNTIME_ROOT "${PVDRUNTIME_ROOT}" ABSOLUTE)
    get_filename_component(PHYSX_ROOT_DIR "${PVDRUNTIME_ROOT}" DIRECTORY)
    if(WIN32)
        set(TARGET_BUILD_PLATFORM windows)
    elseif(UNIX)
        set(TARGET_BUILD_PLATFORM linux)
    else()
        message(FATAL_ERROR "Standalone PVDRuntime is unsupported on this platform")
    endif()
    set(PROJECT_CMAKE_FILES_DIR compiler/cmake)
    set(PX_ENABLE_INSTALL OFF)
    include("${PVDRUNTIME_ROOT}/compiler/cmake/PVDRuntime.cmake")
    set(OVDDATA_BUILD_TESTS ON CACHE BOOL "Build the standalone OvdData tests")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../OvdData" "${CMAKE_BINARY_DIR}/OvdData")
    set(_ovddataomnipvd_tests_default ON)
elseif(DEFINED PVD3_BUILD_TESTS)
    set(_ovddataomnipvd_tests_default ${PVD3_BUILD_TESTS})
endif()

option(OVDDATAOMNIPVD_BUILD_TESTS "Build OvdData OmniPVD adapter tests"
       ${_ovddataomnipvd_tests_default})

add_library(OvdDataOmniPvd STATIC
    OvdFileIngest.cpp
    "${CMAKE_CURRENT_SOURCE_DIR}/../PVDCore/OvdInputValidation.cpp")
target_compile_definitions(OvdDataOmniPvd PRIVATE PVD_OVD_SUPPORT)
target_include_directories(OvdDataOmniPvd
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../PVDCore")
target_link_libraries(OvdDataOmniPvd PUBLIC OvdData PRIVATE PVDRuntime)
target_compile_features(OvdDataOmniPvd PUBLIC cxx_std_14)

add_executable(ovdnext_data_probe ovdnext_data_probe.cpp OvdDataProbe.cpp)
target_link_libraries(ovdnext_data_probe PRIVATE OvdDataOmniPvd)
target_compile_features(ovdnext_data_probe PRIVATE cxx_std_14)

if(OVDDATAOMNIPVD_BUILD_TESTS)
    add_executable(ovddata_omnipvd_tests tests/OvdFileIngestTests.cpp)
    target_link_libraries(ovddata_omnipvd_tests PRIVATE OvdDataOmniPvd PVDRuntime)
    target_compile_definitions(ovddata_omnipvd_tests PRIVATE
        OVDDATAOMNIPVD_REPOSITORY_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../..")
    add_test(NAME ovddata_omnipvd_tests COMMAND ovddata_omnipvd_tests)

    add_executable(ovdnext_data_probe_tests tests/OvdDataProbeTests.cpp OvdDataProbe.cpp)
    target_link_libraries(ovdnext_data_probe_tests PRIVATE OvdDataOmniPvd)
    target_compile_features(ovdnext_data_probe_tests PRIVATE cxx_std_14)
    target_compile_definitions(ovdnext_data_probe_tests PRIVATE
        OVDNEXT_PROBE_TEST_SCENE="${CMAKE_CURRENT_SOURCE_DIR}/../../test/data/test_scene.ovd")
    add_test(NAME ovdnext_data_probe_tests COMMAND ovdnext_data_probe_tests)
endif()
