# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.21)
project(OvdTcpSelectionSender LANGUAGES C CXX)
# The SDK declares checked/profile in its child scope; initialize executable
# flags in this parent scope too, where the standalone fixture target lives.
foreach(config CHECKED PROFILE)
    set(CMAKE_EXE_LINKER_FLAGS_${config} "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO}" CACHE STRING "Fixture link flags")
    set(CMAKE_CXX_FLAGS_${config} "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}" CACHE STRING "Fixture compile flags")
endforeach()
option(OVDNEXT_SENDER_GPU "Build the separate genuine GPU-family fixture" OFF)
if(NOT PX_BUILDPVDRUNTIME OR NOT PX_GENERATE_STATIC_LIBRARIES OR
   NOT "${PX_GENERATE_GPU_PROJECTS}" STREQUAL "${OVDNEXT_SENDER_GPU}")
    message(FATAL_ERROR "Fixture GPU mode must match the checked static SDK with PVDRuntime")
endif()
add_subdirectory("${PHYSX_ROOT_DIR}/compiler/public" "${CMAKE_BINARY_DIR}/physx")
add_executable(ovd_tcp_selection_sender main.cpp)
target_compile_features(ovd_tcp_selection_sender PRIVATE cxx_std_17)
target_compile_definitions(ovd_tcp_selection_sender PRIVATE NDEBUG PX_CHECKED=1 PX_SUPPORT_OMNI_PVD=1
    OVDNEXT_SENDER_GPU=$<BOOL:${OVDNEXT_SENDER_GPU}>)
target_include_directories(ovd_tcp_selection_sender PRIVATE "${PHYSX_ROOT_DIR}/include" "${PHYSX_ROOT_DIR}/pvdruntime/include")
target_link_libraries(ovd_tcp_selection_sender PRIVATE PhysX PhysXCommon PhysXFoundation PhysXExtensions PVDRuntime)
set_target_properties(ovd_tcp_selection_sender PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PX_OUTPUT_BIN_DIR}/$<0:unused>"
    MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
if(OVDNEXT_SENDER_GPU)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(ovd_tcp_selection_sender PRIVATE CUDA::cuda_driver)
    add_dependencies(ovd_tcp_selection_sender PhysXGpu)
    add_custom_command(TARGET ovd_tcp_selection_sender POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:PhysXGpu> $<TARGET_FILE_DIR:ovd_tcp_selection_sender>)
endif()
