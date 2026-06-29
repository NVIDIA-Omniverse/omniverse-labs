# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(openusd_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to the prebuilt OpenUSD installation
if(NOT DEFINED OPENUSD_ROOT)
    set(OPENUSD_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../OpenUSD"
        CACHE PATH "Root of prebuilt OpenUSD installation")
endif()

if(NOT EXISTS "${OPENUSD_ROOT}/include/pxr/pxr.h")
    message(FATAL_ERROR
        "OpenUSD not found at ${OPENUSD_ROOT}. "
        "Set OPENUSD_ROOT to the prebuilt OpenUSD installation directory.")
endif()

# The monolithic import/shared lib (platform-specific name)
if(WIN32)
    set(OPENUSD_LIB "${OPENUSD_ROOT}/lib/ov_25.11usd_ms.lib")
else()
    set(OPENUSD_LIB "${OPENUSD_ROOT}/lib/libov_25.11usd_ms.so")
endif()
if(NOT EXISTS "${OPENUSD_LIB}")
    message(FATAL_ERROR "OpenUSD monolithic lib not found: ${OPENUSD_LIB}")
endif()

add_library(openusd_backend SHARED openusd_backend.cpp)

target_include_directories(openusd_backend PRIVATE
    # Our project headers (for nanousd_backend.h)
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
    # OpenUSD headers
    "${OPENUSD_ROOT}/include"
)

target_link_directories(openusd_backend PRIVATE "${OPENUSD_ROOT}/lib")
target_link_libraries(openusd_backend PRIVATE "${OPENUSD_LIB}")

target_compile_definitions(openusd_backend PRIVATE
    NANOUSD_BACKEND_BUILDING
)

if(MSVC)
    target_compile_definitions(openusd_backend PRIVATE NOMINMAX)
endif()
