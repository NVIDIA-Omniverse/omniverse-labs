# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(nanousd-metal-renderer LANGUAGES C CXX OBJC OBJCXX)

if(NOT APPLE)
    message(FATAL_ERROR "nanousd-metal-renderer is macOS only. For Linux/Windows, use nanousd-vulkan-renderer.")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJC_STANDARD 11)
set(CMAKE_OBJCXX_STANDARD 17)

# Metal 3 hardware ray tracing requires macOS 13+.
if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 13.0)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ARC for the Objective-C++ Metal backend.
set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -fobjc-arc")

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------

option(NUSD_ENABLE_MATERIALS "Build with MaterialX MSL backend (Phase 7)" ON)
option(NUSD_ENABLE_MDL_SDK "Enable optional NVIDIA MDL SDK integration hooks." OFF)
set(MDL_SDK_ROOT "" CACHE PATH "Optional install prefix or build prefix for NVIDIA MDL SDK.")

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

include(FetchContent)

# Note: no GLFW — the Metal backend is fully headless via offscreen MTLTexture
# and renderer.c no longer creates a window.

# nanousd
if(NOT TARGET nanousdapi)
    if(NANOUSD_DIR)
        find_path(NANOUSD_INCLUDE_DIR nanousd/nanousdapi.h
            PATHS ${NANOUSD_DIR}/include
            NO_DEFAULT_PATH)
        find_library(NANOUSD_STATIC nanousdapi
            PATHS ${NANOUSD_DIR}/build/Release
                  ${NANOUSD_DIR}/build
                  ${NANOUSD_DIR}/lib
            NO_DEFAULT_PATH)
        find_library(NANOUSD_SHARED nanousd
            PATHS ${NANOUSD_DIR}/build/Release
                  ${NANOUSD_DIR}/build
                  ${NANOUSD_DIR}/lib
            NO_DEFAULT_PATH)
    else()
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../nanousd/CMakeLists.txt)
            set(NANOUSD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../nanousd)
            find_path(NANOUSD_INCLUDE_DIR nanousd/nanousdapi.h
                PATHS ${NANOUSD_DIR}/include
                NO_DEFAULT_PATH)
            find_library(NANOUSD_STATIC nanousdapi
                PATHS ${NANOUSD_DIR}/build/Release
                      ${NANOUSD_DIR}/build
                      ${NANOUSD_DIR}/lib
                NO_DEFAULT_PATH)
            find_library(NANOUSD_SHARED nanousd
                PATHS ${NANOUSD_DIR}/build/Release
                      ${NANOUSD_DIR}/build
                      ${NANOUSD_DIR}/lib
                NO_DEFAULT_PATH)
        else()
            find_path(NANOUSD_INCLUDE_DIR nanousd/nanousdapi.h)
            find_library(NANOUSD_STATIC nanousdapi)
            find_library(NANOUSD_SHARED nanousd)
        endif()
    endif()

    if(NANOUSD_INCLUDE_DIR AND NANOUSD_STATIC AND NANOUSD_SHARED)
        add_library(nanousdapi STATIC IMPORTED)
        set_target_properties(nanousdapi PROPERTIES
            IMPORTED_LOCATION ${NANOUSD_STATIC}
            INTERFACE_INCLUDE_DIRECTORIES ${NANOUSD_INCLUDE_DIR}
            INTERFACE_LINK_LIBRARIES "${NANOUSD_SHARED};stdc++")
        message(STATUS "nanousd: ${NANOUSD_STATIC} + ${NANOUSD_SHARED}")
    elseif(NANOUSD_INCLUDE_DIR AND NANOUSD_STATIC)
        add_library(nanousdapi STATIC IMPORTED)
        set_target_properties(nanousdapi PROPERTIES
            IMPORTED_LOCATION ${NANOUSD_STATIC}
            INTERFACE_INCLUDE_DIRECTORIES ${NANOUSD_INCLUDE_DIR}
            INTERFACE_LINK_LIBRARIES "stdc++")
        message(STATUS "nanousd: ${NANOUSD_STATIC} (static only)")
    elseif(NOT TARGET nanousdapi)
        message(FATAL_ERROR "nanousd not found. Set -DNANOUSD_DIR=<path>.")
    endif()
endif()

# MaterialX (MSL backend) — only when materials are enabled.
# Strategy: prefer a pre-installed MaterialX (find_package via
# CMAKE_PREFIX_PATH) — picks up the system OpenUSD-bundled MaterialX
# at ~/OpenUSD-install in development setups. Falls back to a
# FetchContent build if find_package returns empty. The fallback path
# is slow (~10 min on first configure to clone + build) but lets the
# build still work in fresh environments without an external install.
if(NUSD_ENABLE_MATERIALS)
    if(NOT MATERIALX_DIR AND DEFINED ENV{MATERIALX_DIR})
        set(MATERIALX_DIR $ENV{MATERIALX_DIR})
    endif()
    if(NOT MATERIALX_DIR AND EXISTS "$ENV{HOME}/OpenUSD-install/lib/cmake/MaterialX/MaterialXConfig.cmake")
        set(MATERIALX_DIR "$ENV{HOME}/OpenUSD-install")
    endif()

    set(_materialx_found FALSE)
    if(MATERIALX_DIR)
        list(PREPEND CMAKE_PREFIX_PATH ${MATERIALX_DIR})
        find_package(MaterialX 1.39.4 CONFIG QUIET)
        if(MaterialX_FOUND)
            set(_materialx_found TRUE)
            set(MATERIALX_SEARCH_PATH_DIR "${MATERIALX_DIR}/libraries")
            message(STATUS "MaterialX: using pre-installed at ${MATERIALX_DIR} (libraries=${MATERIALX_SEARCH_PATH_DIR})")
        endif()
    endif()

    if(NOT _materialx_found)
        message(STATUS "MaterialX: pre-installed not found; falling back to FetchContent (slow first build)")
        set(MATERIALX_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_VIEWER OFF CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_RENDER OFF CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_GEN_GLSL OFF CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_GEN_OSL OFF CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_GEN_MDL OFF CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_GEN_MSL ON CACHE BOOL "" FORCE)
        set(MATERIALX_INSTALL_LIB_PATH lib CACHE STRING "" FORCE)
        set(MATERIALX_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            MaterialX
            GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/MaterialX.git
            GIT_TAG        v1.39.4
            GIT_SHALLOW    TRUE
            GIT_SUBMODULES ""
        )
        FetchContent_MakeAvailable(MaterialX)
        set(MATERIALX_SEARCH_PATH_DIR ${MaterialX_SOURCE_DIR}/libraries)
    endif()
endif()

# meshoptimizer — vertex cache optimization at nu_add_mesh time.
FetchContent_Declare(meshoptimizer
    GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
    GIT_TAG        v0.24
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(meshoptimizer)
set_target_properties(meshoptimizer PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Apple frameworks
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
find_library(COREGRAPHICS_FRAMEWORK CoreGraphics REQUIRED)
# MetalPerformanceShaders: bilinear upscale for the reduced-res orbit path
# (NUSD_ORBIT_RENDER_SCALE). MetalFX (temporal upscale) is added on top later.
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)

# -----------------------------------------------------------------------------
# Shader compilation (Phase 2+)
#
# .metal source files are compiled to .air (intermediate) and then archived
# into a .metallib via xcrun. The renderer loads the metallib at runtime via
# [MTLDevice newLibraryWithURL:].
# -----------------------------------------------------------------------------

set(SHADER_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_DIR})

# Phase 1 placeholder: renderer.c looks for foo.vert.spv etc. and only calls
# gpu_create_pipeline if the load succeeds. Touch empty files so the stubs
# wire up. Phase 2 replaces this with a real .metallib + named functions.
foreach(name mesh.vert mesh.frag mesh_shadow.frag overlay.vert overlay.frag
             raytrace.rgen raytrace.rmiss raytrace.rchit
             raytrace_tiled.rgen raycast.comp)
    if(NOT EXISTS ${SHADER_DIR}/${name}.spv)
        file(WRITE ${SHADER_DIR}/${name}.spv "")
    endif()
endforeach()

file(GLOB METAL_SHADER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/*.metal")

# Strategy: copy .metal sources to SHADER_DIR; gpu_metal.mm compiles them at
# startup via [MTLDevice newLibraryWithSource:options:error:]. This avoids a
# hard dependency on Xcode's `metal`/`metallib` AOT toolchain (which is not
# included in Command Line Tools). A future optimization can switch to
# precompiled .metallib when Xcode is detected.
set(METAL_SHADERS_COPIED)
foreach(metal_src ${METAL_SHADER_SOURCES})
    get_filename_component(metal_name ${metal_src} NAME)
    set(dst ${SHADER_DIR}/${metal_name})
    add_custom_command(
        OUTPUT ${dst}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${metal_src} ${dst}
        DEPENDS ${metal_src}
        COMMENT "Staging Metal shader ${metal_name}"
    )
    list(APPEND METAL_SHADERS_COPIED ${dst})
endforeach()

if(METAL_SHADERS_COPIED)
    add_custom_target(renderer_shaders ALL DEPENDS ${METAL_SHADERS_COPIED})
else()
    add_custom_target(renderer_shaders ALL)
endif()

# -----------------------------------------------------------------------------
# Shared library: libnusd_renderer.dylib
# -----------------------------------------------------------------------------

if(NUSD_ENABLE_MATERIALS)
    set(MATERIAL_SOURCE src/material.cpp)
else()
    set(MATERIAL_SOURCE src/material_stub.cpp)
endif()

add_library(nusd_renderer SHARED
    src/renderer.c
    src/gpu_metal.mm
    src/scene.c
    src/gs_scene.c
    src/primitive_geom.c
    src/camera.c
    ${MATERIAL_SOURCE}
    src/mdl_bridge.cpp
    src/dlss_stub.c
)

target_include_directories(nusd_renderer
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(nusd_renderer PRIVATE
    SHADER_DIR="${SHADER_DIR}"
    NUSD_RENDERER_HEADLESS=1
    NUSD_BACKEND_METAL=1
)

if(NUSD_ENABLE_MATERIALS)
    target_compile_definitions(nusd_renderer PRIVATE
        MATERIALX_SEARCH_PATH="${MATERIALX_SEARCH_PATH_DIR}"
        NUSD_ENABLE_MATERIALS=1
    )
endif()

target_link_libraries(nusd_renderer PRIVATE
    nanousdapi
    meshoptimizer
    ${METAL_FRAMEWORK}
    ${METALKIT_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${QUARTZCORE_FRAMEWORK}
    ${COREGRAPHICS_FRAMEWORK}
    ${MPS_FRAMEWORK}
    ${CMAKE_DL_LIBS}
)

if(NUSD_ENABLE_MATERIALS)
    target_link_libraries(nusd_renderer PRIVATE
        MaterialXGenMsl
        MaterialXGenShader
        MaterialXFormat
        MaterialXCore
    )
endif()

if(NUSD_ENABLE_MDL_SDK)
    if(MDL_SDK_ROOT)
        list(PREPEND CMAKE_PREFIX_PATH
            "${MDL_SDK_ROOT}"
            "${MDL_SDK_ROOT}/share"
            "${MDL_SDK_ROOT}/lib/cmake")
    endif()
    find_package(mdl CONFIG QUIET)
    if(mdl_FOUND AND TARGET mdl::mdl_sdk)
        target_link_libraries(nusd_renderer PRIVATE mdl::mdl_sdk)
        target_compile_definitions(nusd_renderer PRIVATE NUSD_HAVE_MDL_SDK=1)
        message(STATUS "NVIDIA MDL SDK enabled for nusd_renderer")
    else()
        message(FATAL_ERROR
            "NUSD_ENABLE_MDL_SDK=ON but CMake package 'mdl' was not found. "
            "Build/install NVIDIA/MDL-SDK and set -DMDL_SDK_ROOT=<prefix>.")
    endif()
endif()

add_dependencies(nusd_renderer renderer_shaders)

set_target_properties(nusd_renderer PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    SOVERSION 1
)

# -----------------------------------------------------------------------------
# Test executable
# -----------------------------------------------------------------------------

add_executable(test_headless_render test/test_headless_render.c)
target_link_libraries(test_headless_render PRIVATE nusd_renderer)
set_target_properties(test_headless_render PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_executable(test_scene_curves test/test_scene_curves.c)
target_link_libraries(test_scene_curves PRIVATE nusd_renderer)
target_include_directories(test_scene_curves PRIVATE src)
set_target_properties(test_scene_curves PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_executable(test_scene_lights test/test_scene_lights.c)
target_link_libraries(test_scene_lights PRIVATE nusd_renderer)
target_include_directories(test_scene_lights PRIVATE src)
set_target_properties(test_scene_lights PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_executable(test_gaussian_render test/test_gaussian_render.c)
target_link_libraries(test_gaussian_render PRIVATE nusd_renderer)
set_target_properties(test_gaussian_render PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

enable_testing()
find_package(Python3 COMPONENTS Interpreter)
add_test(NAME headless_render COMMAND test_headless_render)
add_test(NAME headless_render_meshlets COMMAND test_headless_render)
add_test(NAME scene_curves COMMAND test_scene_curves
         ${CMAKE_CURRENT_SOURCE_DIR}/test/basicCurves.usda)
add_test(NAME scene_lights COMMAND test_scene_lights
         ${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures/authored_lights.usda)
add_test(NAME gaussian_render COMMAND test_gaussian_render)
if(Python3_Interpreter_FOUND)
    add_test(NAME ovrtx_python_metal
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python
            NUSD_RENDERER_LIB=$<TARGET_FILE:nusd_renderer>
            NANOUSD_OVRTX_BACKEND=metal
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_ovrtx_metal_api.py)
    # Automated Metal-vs-OVRTX FLIP regression gate over the committed comparison
    # frames. Skips (77) when flip-evaluator/numpy/Pillow/scipy/scikit-image are absent
    # from the test interpreter (scipy + scikit-image are required — they power the gate's
    # GMSD structure and CIEDE2000 colour legs); else runs and fails on any regression.
    add_test(NAME flip_metal_vs_ovrtx
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/comparisons/flip_pipeline.py --check)
endif()
set_tests_properties(headless_render headless_render_meshlets scene_curves scene_lights gaussian_render PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
if(Python3_Interpreter_FOUND)
    set_tests_properties(ovrtx_python_metal PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        SKIP_RETURN_CODE 77
    )
    set_tests_properties(flip_metal_vs_ovrtx PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        SKIP_RETURN_CODE 77
    )
endif()
set_tests_properties(headless_render_meshlets PROPERTIES
    ENVIRONMENT "NUSD_MESHLET_RASTER=1"
)
