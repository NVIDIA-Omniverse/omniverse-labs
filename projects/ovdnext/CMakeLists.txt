# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.18)
project(PVD3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(PVD3_BUILD_PYTHON "Build the native Python session binding" OFF)
if(PVD3_BUILD_PYTHON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# OVD support (optional) - set PVDRUNTIME_ROOT and PVDDOM_ROOT to enable
set(PVDRUNTIME_ROOT "" CACHE PATH "Path to pvdruntime directory")
set(PVDDOM_ROOT "" CACHE PATH "Path to pvddom directory")
option(PVD3_REQUIRE_OVD_SUPPORT "Fail configure unless OmniPVD support is available" OFF)

# MSVC otherwise expands __FILE__ and CodeView records to developer-local
# paths. Remap every non-Debug configuration before adding any in-tree or
# external OmniPVD targets; RelWithDebInfo is also used for distributable builds.
if(MSVC)
    file(TO_NATIVE_PATH "${CMAKE_SOURCE_DIR}" PVD3_SOURCE_NATIVE)
    add_compile_options(
        "$<$<NOT:$<CONFIG:Debug>>:/experimental:deterministic>"
        "$<$<NOT:$<CONFIG:Debug>>:/pathmap:${PVD3_SOURCE_NATIVE}=pvd3>"
    )
    add_link_options(
        "$<$<NOT:$<CONFIG:Debug>>:/PDBALTPATH:%_PDB%>"
    )
    if(PVDRUNTIME_ROOT)
        file(TO_NATIVE_PATH "${PVDRUNTIME_ROOT}" PVDRUNTIME_ROOT_NATIVE)
        add_compile_options(
            "$<$<NOT:$<CONFIG:Debug>>:/pathmap:${PVDRUNTIME_ROOT_NATIVE}=pvdruntime>"
        )
    endif()
    if(PVDDOM_ROOT)
        file(TO_NATIVE_PATH "${PVDDOM_ROOT}" PVDDOM_ROOT_NATIVE)
        add_compile_options(
            "$<$<NOT:$<CONFIG:Debug>>:/pathmap:${PVDDOM_ROOT_NATIVE}=pvddom>"
        )
    endif()
endif()

set(_pvd3_ovd_roots_valid OFF)
if(PVDRUNTIME_ROOT AND PVDDOM_ROOT
   AND EXISTS "${PVDRUNTIME_ROOT}/compiler/cmake/PVDRuntime.cmake"
   AND EXISTS "${PVDDOM_ROOT}/CMakeLists.txt")
    set(_pvd3_ovd_roots_valid ON)
endif()

if(_pvd3_ovd_roots_valid)
    add_subdirectory("${PVDDOM_ROOT}" "${CMAKE_BINARY_DIR}/pvddom")
    if(TARGET PvdDom AND TARGET PVDRuntime)
        set(OVD_SUPPORT ON)
        message(STATUS "OVD support enabled")
    else()
        message(FATAL_ERROR "OmniPVD roots did not create PvdDom and PVDRuntime")
    endif()
else()
    set(OVD_SUPPORT OFF)
    if(PVD3_REQUIRE_OVD_SUPPORT)
        message(FATAL_ERROR
            "PVD3_REQUIRE_OVD_SUPPORT=ON requires valid PVDRUNTIME_ROOT and PVDDOM_ROOT")
    endif()
    message(STATUS "OVD support disabled (set PVDRUNTIME_ROOT and PVDDOM_ROOT to enable)")
endif()

option(PVD3_BUILD_TESTS "Build OvdNext project tests" ON)
if(PVD3_BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(src/OvdData)
if(TARGET PVDRuntime)
    add_subdirectory(src/OvdDataOmniPvd)
endif()
add_subdirectory(src/PxShared)
add_subdirectory(src/Profiler)
add_subdirectory(src/PVDCore)
if(PVD3_BUILD_PYTHON)
    add_subdirectory(src/PVDSession)
endif()

if(PVD3_BUILD_TESTS)
    add_subdirectory(test)
endif()
