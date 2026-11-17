# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.18)

set(_ovddata_tests_default OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(OvdData LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    enable_testing()
    set(_ovddata_tests_default ON)
elseif(DEFINED PVD3_BUILD_TESTS)
    set(_ovddata_tests_default ${PVD3_BUILD_TESTS})
endif()

option(OVDDATA_BUILD_TESTS "Build the standalone OvdData tests" ${_ovddata_tests_default})

add_library(OvdData STATIC
    OvdData.cpp
    OvdIngest.cpp
    OvdRecordedModel.cpp
    OvdRecordedModelBuilder.cpp)
target_include_directories(OvdData PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(OvdData PUBLIC cxx_std_14)

if(OVDDATA_BUILD_TESTS)
    add_executable(ovddata_tests tests/OvdDataTests.cpp)
    target_link_libraries(ovddata_tests PRIVATE OvdData)
    add_test(NAME ovddata_tests COMMAND ovddata_tests)

    add_executable(ovddata_ingest_tests tests/OvdIngestTests.cpp)
    target_link_libraries(ovddata_ingest_tests PRIVATE OvdData)
    add_test(NAME ovddata_ingest_tests COMMAND ovddata_ingest_tests)
endif()
