# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mtlx2msl — tiny MaterialX → MSL driver for Phase 1 PoC.
#
# Stand-alone build that links against a pre-installed MaterialX. Set
# MATERIALX_INSTALL to the prefix (default $HOME/OpenUSD-install which
# is the layout the chess-comparison harness already depends on).
#
# Modeled on tools/mtlx2glsl/CMakeLists.txt on the Vulkan side; this one
# links GenMsl instead of GenGlsl, defaults MATERIALX_INSTALL to
# $HOME/OpenUSD-install, drops the glsl-only RtSurfaceNodeGlsl.cpp source
# and the pthread/dl links, and adds a runtime RPATH.

cmake_minimum_required(VERSION 3.20)
project(mtlx2msl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED MATERIALX_INSTALL)
    set(MATERIALX_INSTALL "$ENV{HOME}/OpenUSD-install")
endif()

add_executable(mtlx2msl main.cpp)

target_include_directories(mtlx2msl PRIVATE "${MATERIALX_INSTALL}/include")
target_link_directories(mtlx2msl PRIVATE "${MATERIALX_INSTALL}/lib")

# Link order: generators → GenShader → Format → Core. Same shape as the
# Vulkan-side tools/mtlx2glsl tool.
target_link_libraries(mtlx2msl PRIVATE
    MaterialXGenMsl
    MaterialXGenShader
    MaterialXFormat
    MaterialXCore)

# Make the binary find the dylibs at runtime when linked against the
# OpenUSD install.
set_target_properties(mtlx2msl PROPERTIES
    BUILD_RPATH   "${MATERIALX_INSTALL}/lib"
    INSTALL_RPATH "${MATERIALX_INSTALL}/lib")
