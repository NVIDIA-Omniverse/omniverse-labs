# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mtlx2glsl — tiny MaterialX → GLSL driver for Phase 1 PoC.
#
# Stand-alone build that links against a pre-installed MaterialX. Set
# MATERIALX_INSTALL to the prefix (default /tmp/mtlx-install).

cmake_minimum_required(VERSION 3.20)
project(mtlx2glsl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED MATERIALX_INSTALL)
    set(MATERIALX_INSTALL "/tmp/mtlx-install")
endif()

add_executable(mtlx2glsl
    main.cpp
    RtSurfaceNodeGlsl.cpp)

target_include_directories(mtlx2glsl PRIVATE "${MATERIALX_INSTALL}/include")
target_link_directories(mtlx2glsl PRIVATE "${MATERIALX_INSTALL}/lib")

# Link order matters for static archives — generators depend on GenShader,
# which depends on Format + Core.
target_link_libraries(mtlx2glsl PRIVATE
    MaterialXGenGlsl
    MaterialXGenShader
    MaterialXFormat
    MaterialXCore
    pthread
    dl)
