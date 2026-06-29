# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

cmake_minimum_required(VERSION 3.15)
project(radar-composite-tensor LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FETCHCONTENT_QUIET FALSE)

# Fetch ovrtx library.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake")
include(ovrtx)
ovrtx_fetch()

add_executable(radar-composite-tensor main.cpp)
target_link_libraries(radar-composite-tensor PRIVATE ovrtx::ovrtx)

set(RADAR_EXAMPLE_USDA "${CMAKE_CURRENT_BINARY_DIR}/radar_example.usda")
configure_file("${CMAKE_CURRENT_LIST_DIR}/radar_example.usda" "${RADAR_EXAMPLE_USDA}" COPYONLY)
target_compile_definitions(radar-composite-tensor
    PRIVATE "RADAR_EXAMPLE_USDA_PATH=\"${RADAR_EXAMPLE_USDA}\"")

# Setup runtime dependencies (DLL copying, junctions, rpath).
ovrtx_setup_runtime(radar-composite-tensor)
