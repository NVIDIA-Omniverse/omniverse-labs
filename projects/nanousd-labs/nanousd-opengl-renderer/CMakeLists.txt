# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(nanousd-opengl-renderer LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

option(NUSD_ENABLE_MDL_SDK "Enable optional NVIDIA MDL SDK integration hooks." OFF)
option(NUSD_ENABLE_PTEX "Enable Disney Ptex material loading for authored .ptx surface maps." ON)
set(MDL_SDK_ROOT "" CACHE PATH "Optional install prefix or build prefix for NVIDIA MDL SDK.")
set(NUSD_PTEX_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../nanousd-vulkan-renderer/third_party/ptex"
    CACHE PATH "Path to vendored Ptex source tree.")
set(NUSD_PTEX_HELPER
    "${CMAKE_CURRENT_SOURCE_DIR}/../nanousd-vulkan-renderer/src/ptex_material.cpp"
    CACHE FILEPATH "Path to the shared nanousd Ptex material helper.")

include(FetchContent)

# -----------------------------------------------------------------------------
# MaterialX — needed for parsing .mtlx files (chess set + other open assets
# that ship MaterialX shaders alongside their USD geometry). Mirrors
# nanousd-vulkan-renderer's setup: prefer pre-built artifacts from this build
# dir or the sibling nanousd-vulkan-renderer build, fall back to FetchContent.
# -----------------------------------------------------------------------------
# Look for pre-built MaterialX in (in priority order):
#   1. this repo's build/_deps/materialx-* (if already fetched)
#   2. nanousd-vulkan-renderer/build/_deps/materialx-* (if vulkan-renderer was built)
set(_MTX_SEARCH_BASES
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../nanousd-vulkan-renderer/build)
set(MATERIALX_SRC_DIR  "")
set(MATERIALX_LIB_DIR  "")
set(MATERIALX_BUILD_GEN_DIR "")
foreach(_base ${_MTX_SEARCH_BASES})
    if(EXISTS ${_base}/_deps/materialx-src/source/MaterialXCore)
        # vulkan-renderer layout: lib under _deps/materialx-build/lib
        if(EXISTS ${_base}/_deps/materialx-build/lib/libMaterialXCore.a)
            set(MATERIALX_SRC_DIR  ${_base}/_deps/materialx-src)
            set(MATERIALX_LIB_DIR  ${_base}/_deps/materialx-build/lib)
            set(MATERIALX_BUILD_GEN_DIR ${_base}/_deps/materialx-build/source)
            break()
        endif()
        # vulkan-viewer layout: lib at build/lib
        if(EXISTS ${_base}/lib/libMaterialXCore.a)
            set(MATERIALX_SRC_DIR  ${_base}/_deps/materialx-src)
            set(MATERIALX_LIB_DIR  ${_base}/lib)
            set(MATERIALX_BUILD_GEN_DIR ${_base}/_deps/materialx-build/source)
            break()
        endif()
    endif()
endforeach()
if(MATERIALX_LIB_DIR AND EXISTS ${MATERIALX_LIB_DIR}/libMaterialXCore.a)
    message(STATUS "MaterialX: using pre-built libs from ${MATERIALX_LIB_DIR}")
    set(MATERIALX_INCLUDE_DIR ${MATERIALX_SRC_DIR}/source)
    set(MATERIALX_LIBRARIES_DIR ${MATERIALX_LIB_DIR})
    set(MATERIALX_SEARCH_PATH_DIR ${MATERIALX_SRC_DIR}/libraries)
    foreach(lib MaterialXCore MaterialXFormat MaterialXGenShader MaterialXGenGlsl)
        if(NOT TARGET ${lib})
            add_library(${lib} STATIC IMPORTED)
            set_target_properties(${lib} PROPERTIES
                IMPORTED_LOCATION ${MATERIALX_LIBRARIES_DIR}/lib${lib}.a
                INTERFACE_INCLUDE_DIRECTORIES "${MATERIALX_INCLUDE_DIR};${MATERIALX_BUILD_GEN_DIR}")
        endif()
    endforeach()
else()
    message(STATUS "MaterialX: fetching from source (this may take a while)")
    set(MATERIALX_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(MATERIALX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(MATERIALX_BUILD_VIEWER OFF CACHE BOOL "" FORCE)
    set(MATERIALX_BUILD_RENDER OFF CACHE BOOL "" FORCE)
    set(MATERIALX_BUILD_GEN_GLSL ON CACHE BOOL "" FORCE)
    set(MATERIALX_BUILD_GEN_OSL OFF CACHE BOOL "" FORCE)
    set(MATERIALX_BUILD_GEN_MDL OFF CACHE BOOL "" FORCE)
    set(MATERIALX_BUILD_GEN_MSL OFF CACHE BOOL "" FORCE)
    set(MATERIALX_INSTALL_LIB_PATH lib CACHE STRING "" FORCE)
    set(MATERIALX_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(MaterialX
        GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/MaterialX.git
        GIT_TAG        v1.39.4
        GIT_SHALLOW    TRUE
        GIT_SUBMODULES "")
    FetchContent_MakeAvailable(MaterialX)
    set(MATERIALX_SEARCH_PATH_DIR ${MaterialX_SOURCE_DIR}/libraries)
endif()

# -----------------------------------------------------------------------------
# meshoptimizer — vertex cache optimization at mesh-merge time.
# MIT licensed, header + .cpp, zero external deps. Static lib must be PIC
# because it gets linked into libnusd_gles.so.
# -----------------------------------------------------------------------------
FetchContent_Declare(meshoptimizer
    GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
    GIT_TAG        v0.24
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(meshoptimizer)
set_target_properties(meshoptimizer PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(NUSD_ENABLE_PTEX AND EXISTS "${NUSD_PTEX_SOURCE_DIR}/src/ptex/PtexReader.cpp"
   AND EXISTS "${NUSD_PTEX_HELPER}")
    find_package(Threads REQUIRED)
    find_package(libdeflate CONFIG QUIET)
    if(TARGET libdeflate::libdeflate_shared)
        set(NUSD_LIBDEFLATE_TARGET libdeflate::libdeflate_shared)
    elseif(TARGET libdeflate::libdeflate)
        set(NUSD_LIBDEFLATE_TARGET libdeflate::libdeflate)
    elseif(TARGET libdeflate::libdeflate_static)
        set(NUSD_LIBDEFLATE_TARGET libdeflate::libdeflate_static)
    else()
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(LIBDEFLATE REQUIRED IMPORTED_TARGET libdeflate)
        set(NUSD_LIBDEFLATE_TARGET PkgConfig::LIBDEFLATE)
    endif()

    add_library(nusd_ptex STATIC
        ${NUSD_PTEX_SOURCE_DIR}/src/ptex/PtexCache.cpp
        ${NUSD_PTEX_SOURCE_DIR}/src/ptex/PtexFilters.cpp
        ${NUSD_PTEX_SOURCE_DIR}/src/ptex/PtexHalf.cpp
        ${NUSD_PTEX_SOURCE_DIR}/src/ptex/PtexReader.cpp
        ${NUSD_PTEX_SOURCE_DIR}/src/ptex/PtexSeparableFilter.cpp
        ${NUSD_PTEX_SOURCE_DIR}/src/ptex/PtexSeparableKernel.cpp
        ${NUSD_PTEX_SOURCE_DIR}/src/ptex/PtexTriangleFilter.cpp
        ${NUSD_PTEX_SOURCE_DIR}/src/ptex/PtexTriangleKernel.cpp
        ${NUSD_PTEX_SOURCE_DIR}/src/ptex/PtexUtils.cpp
        ${NUSD_PTEX_SOURCE_DIR}/src/ptex/PtexWriter.cpp
    )
    target_compile_definitions(nusd_ptex PUBLIC PTEX_STATIC)
    target_include_directories(nusd_ptex PUBLIC
        ${NUSD_PTEX_SOURCE_DIR}/src/ptex)
    target_link_libraries(nusd_ptex PUBLIC Threads::Threads PRIVATE ${NUSD_LIBDEFLATE_TARGET})
    set_target_properties(nusd_ptex PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set(NUSD_OPENGL_HAVE_PTEX ON)
elseif(NUSD_ENABLE_PTEX)
    message(WARNING
        "NUSD_ENABLE_PTEX=ON but Ptex sources/helper were not found; "
        "OpenGL Moana Ptex colors will be disabled.")
endif()

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
#
# This project builds two shared libraries from the same render core, and
# consumes a third from nanousd:
#
#   1. libnusd_gles — embeddable headless renderer, loaded from Python via
#      ctypes / ovgear. No native interactive binary is built here.
#
#   2. libnusd_renderer_opengl — the same object code under a different
#      OUTPUT_NAME, exposing the canonical nu_* C API (nusd_renderer.h)
#      via nu_renderer_shim.c. nanousdview dlopens this for `--backend
#      opengl`; interactive viewing goes through `./run.sh scene.usd`.
#
#   3. libnanousdapi — built from a sibling/workspace nanousd checkout or
#      loaded from an install prefix, then opened from Python via ctypes
#      alongside the renderer libs.
#
# GLFW is no longer a dependency of any of them: the library carries no
# GLFW windowing code, and the dylibs must stay GLFW-free so they can
# coexist with ovgear's own libglfw inside a single Python process
# without duplicate Cocoa-class registrations.

# nanousd — use a source checkout or a pre-built library.
if(NOT TARGET nanousdapi)
    set(_NANOUSD_SOURCE_DIR "")
    if(NANOUSD_DIR AND EXISTS ${NANOUSD_DIR}/CMakeLists.txt)
        set(_NANOUSD_SOURCE_DIR ${NANOUSD_DIR})
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/nanousd/CMakeLists.txt)
        set(_NANOUSD_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nanousd)
    elseif(NOT NANOUSD_DIR AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../nanousd/CMakeLists.txt)
        set(_NANOUSD_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../nanousd)
    endif()

    if(_NANOUSD_SOURCE_DIR)
        set(NANOUSD_SHARED_API ON CACHE BOOL "" FORCE)
        set(NANOUSD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        add_subdirectory(${_NANOUSD_SOURCE_DIR} ${CMAKE_BINARY_DIR}/nanousd)
        foreach(_nanousd_target nanousd nanousdapi)
            if(TARGET ${_nanousd_target})
                target_compile_options(${_nanousd_target} PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:-UNANOUSD_BUILDING>
                    $<$<COMPILE_LANGUAGE:CXX>:-Wno-error>)
            endif()
        endforeach()
    else()
        if(NANOUSD_DIR)
            find_path(NANOUSD_INCLUDE_DIR nanousd/nanousdapi.h
                PATHS ${NANOUSD_DIR}/include
                NO_DEFAULT_PATH)
            find_library(NANOUSD_LIB nanousdapi
                PATHS ${NANOUSD_DIR}/build/Release
                      ${NANOUSD_DIR}/build
                      ${NANOUSD_DIR}/lib
                NO_DEFAULT_PATH)
        else()
            find_path(NANOUSD_INCLUDE_DIR nanousd/nanousdapi.h)
            find_library(NANOUSD_LIB nanousdapi)
        endif()
    endif()

    if(NANOUSD_INCLUDE_DIR AND NANOUSD_LIB)
        add_library(nanousdapi SHARED IMPORTED)
        set_target_properties(nanousdapi PROPERTIES
            IMPORTED_LOCATION ${NANOUSD_LIB}
            INTERFACE_INCLUDE_DIRECTORIES ${NANOUSD_INCLUDE_DIR})
        message(STATUS "nanousd: using library at ${NANOUSD_LIB}")
    elseif(NOT TARGET nanousdapi)
        message(FATAL_ERROR "nanousd not found. Set -DNANOUSD_DIR=<prefix>, install it via the workspace build, or place nanousd alongside this repo.")
    endif()
endif()

# OpenGL ES 3.1 + EGL (or desktop GL on macOS)
if(APPLE)
    find_package(OpenGL REQUIRED)
    set(GL_LIBS OpenGL::GL)
    set(GL_INCLUDE_DIRS "")
else()
    find_library(GLESV2_LIB GLESv2)
    find_library(EGL_LIB EGL)
    find_path(GLES3_INCLUDE_DIR GLES3/gl31.h)
    find_path(EGL_INCLUDE_DIR EGL/egl.h)

    if(NOT GLESV2_LIB OR NOT EGL_LIB)
        message(WARNING "GLESv2/EGL not found. Trying OpenGL desktop (GLES emulation via ANGLE or mesa).")
        find_package(OpenGL)
        if(OpenGL_FOUND)
            set(GLESV2_LIB OpenGL::GL)
            set(EGL_LIB OpenGL::EGL)
        else()
            message(FATAL_ERROR "Neither GLESv2/EGL nor desktop OpenGL found. "
                    "Install libgles-dev and libegl-dev, or the NVIDIA driver.")
        endif()
    endif()
    set(GL_LIBS ${GLESV2_LIB} ${EGL_LIB})
    set(GL_INCLUDE_DIRS ${GLES3_INCLUDE_DIR} ${EGL_INCLUDE_DIR})
endif()

# -----------------------------------------------------------------------------
# Sources
# -----------------------------------------------------------------------------
#
# Pure render-core: scene parsing, GPU pipeline, materials, GL thread,
# headless context. No interactive code, no GLFW. The whole library
# exposes its public API through src/viewer.h and is consumed by Python
# via ctypes (gl_thread.c is the entry-point shim).

set(CORE_SOURCES
    src/viewer.c
    src/scene.c
    src/material.c
    src/material_mtlx.cpp
    src/mdl_bridge.cpp
    src/gpu_opengles.c
    src/gl_thread.c
    src/nu_renderer_shim.c
    src/geo_cache.c
    src/nu_parallel.cpp
)
if(NUSD_OPENGL_HAVE_PTEX)
    list(APPEND CORE_SOURCES ${NUSD_PTEX_HELPER})
endif()

if(APPLE)
    list(APPEND CORE_SOURCES src/cgl_headless.c)
else()
    list(APPEND CORE_SOURCES src/egl_headless.c)
endif()

add_library(nusd_gles_core OBJECT ${CORE_SOURCES})
set_property(TARGET nusd_gles_core PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(nusd_gles_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(GL_INCLUDE_DIRS)
    target_include_directories(nusd_gles_core PUBLIC ${GL_INCLUDE_DIRS})
endif()

target_compile_definitions(nusd_gles_core PUBLIC
    NUSD_GLES=1
    MATERIALX_SEARCH_PATH="${MATERIALX_SEARCH_PATH_DIR}"
)
if(APPLE)
    target_compile_definitions(nusd_gles_core PUBLIC NUSD_DESKTOP_GL=1)
endif()
if(NUSD_OPENGL_HAVE_PTEX)
    target_compile_definitions(nusd_gles_core PUBLIC NUSD_HAVE_PTEX=1)
    target_include_directories(nusd_gles_core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../nanousd-vulkan-renderer/src)
endif()

target_link_libraries(nusd_gles_core PUBLIC
    nanousdapi
    ${GL_LIBS}
    pthread
    MaterialXGenGlsl
    MaterialXGenShader
    MaterialXFormat
    MaterialXCore
    meshoptimizer
    ${CMAKE_DL_LIBS}
)
if(NUSD_OPENGL_HAVE_PTEX)
    target_link_libraries(nusd_gles_core PUBLIC nusd_ptex)
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
        target_link_libraries(nusd_gles_core PUBLIC mdl::mdl_sdk)
        target_compile_definitions(nusd_gles_core PUBLIC NUSD_HAVE_MDL_SDK=1)
        message(STATUS "NVIDIA MDL SDK enabled for nusd_gles_core")
    else()
        message(WARNING
            "NUSD_ENABLE_MDL_SDK=ON but CMake package 'mdl' was not found. "
            "Build/install NVIDIA/MDL-SDK and set -DMDL_SDK_ROOT=<prefix>.")
    endif()
endif()

# nanousdapi is C++ internally; ensure stdc++ follows it on the link line
# regardless of how downstream targets order their libs.
if(NOT APPLE)
    target_link_libraries(nusd_gles_core PUBLIC stdc++)
endif()

# OpenMP — parallelises the geo cache's per-prototype meshopt decode (iter 37).
# PUBLIC on the OBJECT library so the compile flag reaches geo_cache.c and the
# runtime link propagates to every binary that links nusd_gles_core.
find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(nusd_gles_core PUBLIC OpenMP::OpenMP_C)
    message(STATUS "OpenMP enabled for nusd_gles_core (geo cache parallel decode)")
else()
    message(STATUS "OpenMP not found — geo cache decode will run serial")
endif()

# -----------------------------------------------------------------------------
# Shared library: embeddable headless renderer (Python ctypes / ovgear).
# -----------------------------------------------------------------------------

add_library(nusd_gles SHARED $<TARGET_OBJECTS:nusd_gles_core>)
target_link_libraries(nusd_gles PRIVATE nusd_gles_core)

if(APPLE)
    target_link_libraries(nusd_gles PRIVATE c++ m ${CMAKE_DL_LIBS})
else()
    target_link_libraries(nusd_gles PRIVATE stdc++ m ${CMAKE_DL_LIBS})
endif()

set_target_properties(nusd_gles PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    OUTPUT_NAME nusd_gles
)

# -----------------------------------------------------------------------------
# nusd_renderer_opengl — same .so, different OUTPUT_NAME, exposes the
# canonical nu_* C API (nusd_renderer.h) on top of the existing nusd_gles
# core via nu_renderer_shim.c. nanousdview dlopens this when --backend
# opengl is selected.
#
# The shim is already compiled into nusd_gles_core (above), so the new
# target is just an additional .so view that exports the nu_* symbols
# under the canonical name.
# -----------------------------------------------------------------------------

add_library(nusd_renderer_opengl SHARED $<TARGET_OBJECTS:nusd_gles_core>)
target_link_libraries(nusd_renderer_opengl PRIVATE nusd_gles_core)
if(APPLE)
    target_link_libraries(nusd_renderer_opengl PRIVATE c++ m ${CMAKE_DL_LIBS})
else()
    target_link_libraries(nusd_renderer_opengl PRIVATE stdc++ m ${CMAKE_DL_LIBS})
endif()
set_target_properties(nusd_renderer_opengl PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    OUTPUT_NAME nusd_renderer_opengl
    SOVERSION 1
    VERSION 1.0.0
)

# -----------------------------------------------------------------------------
# Tests (CTest)
# -----------------------------------------------------------------------------

option(NUSD_BUILD_TESTS "Build the test suite" ON)
if(NUSD_BUILD_TESTS)
    enable_testing()

    add_executable(scene_test tests/scene_test.c $<TARGET_OBJECTS:nusd_gles_core>)
    target_link_libraries(scene_test PRIVATE nusd_gles_core)
    if(APPLE)
        target_link_libraries(scene_test PRIVATE c++ m ${CMAKE_DL_LIBS})
    else()
        target_link_libraries(scene_test PRIVATE stdc++ m ${CMAKE_DL_LIBS})
    endif()
    add_test(NAME scene_test
             COMMAND scene_test
                     ${CMAKE_CURRENT_SOURCE_DIR}/test_cube.usda
                     ${CMAKE_CURRENT_SOURCE_DIR}/test_materials.usda
                     ${CMAKE_CURRENT_SOURCE_DIR}/test_pbr_materials.usda
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/authored_lights.usda
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/reference_wrapper.usda)

    # Skinned-normal inverse-transpose path through the real scene loader (task #14).
    add_executable(scene_skin_test tests/scene_skin_test.c $<TARGET_OBJECTS:nusd_gles_core>)
    target_link_libraries(scene_skin_test PRIVATE nusd_gles_core)
    if(APPLE)
        target_link_libraries(scene_skin_test PRIVATE c++ m ${CMAKE_DL_LIBS})
    else()
        target_link_libraries(scene_skin_test PRIVATE stdc++ m ${CMAKE_DL_LIBS})
    endif()
    add_test(NAME scene_skin_test
             COMMAND scene_skin_test
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/skinned_nonuniform.usda)

    # UV tiling on MaterialX authored directly in USD (task #15).
    add_executable(mtlx_usd_uvtiling_test tests/mtlx_usd_uvtiling_test.c
                                          $<TARGET_OBJECTS:nusd_gles_core>)
    target_link_libraries(mtlx_usd_uvtiling_test PRIVATE nusd_gles_core)
    if(APPLE)
        target_link_libraries(mtlx_usd_uvtiling_test PRIVATE c++ m ${CMAKE_DL_LIBS})
    else()
        target_link_libraries(mtlx_usd_uvtiling_test PRIVATE stdc++ m ${CMAKE_DL_LIBS})
    endif()
    add_test(NAME mtlx_usd_uvtiling_test
             COMMAND mtlx_usd_uvtiling_test
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/mtlx_usd_uvtiling.usda
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)

    # UsdGeomCapsule true-scale tessellation (Newton robot limbs).
    add_executable(capsule_test tests/capsule_test.c $<TARGET_OBJECTS:nusd_gles_core>)
    target_link_libraries(capsule_test PRIVATE nusd_gles_core)
    if(APPLE)
        target_link_libraries(capsule_test PRIVATE c++ m ${CMAKE_DL_LIBS})
    else()
        target_link_libraries(capsule_test PRIVATE stdc++ m ${CMAKE_DL_LIBS})
    endif()
    add_test(NAME capsule_test
             COMMAND capsule_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/capsule_z.usda)

    add_executable(gprim_test tests/gprim_test.c $<TARGET_OBJECTS:nusd_gles_core>)
    target_link_libraries(gprim_test PRIVATE nusd_gles_core)
    if(APPLE)
        target_link_libraries(gprim_test PRIVATE c++ m ${CMAKE_DL_LIBS})
    else()
        target_link_libraries(gprim_test PRIVATE stdc++ m ${CMAKE_DL_LIBS})
    endif()
    add_test(NAME gprim_test
             COMMAND gprim_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/gprim_cube_cone.usda)

    add_executable(mdl_material_test tests/mdl_material_test.c
                                     $<TARGET_OBJECTS:nusd_gles_core>)
    target_link_libraries(mdl_material_test PRIVATE nusd_gles_core)
    if(APPLE)
        target_link_libraries(mdl_material_test PRIVATE c++ m ${CMAKE_DL_LIBS})
    else()
        target_link_libraries(mdl_material_test PRIVATE stdc++ m ${CMAKE_DL_LIBS})
    endif()
    add_test(NAME mdl_material_test
             COMMAND mdl_material_test
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/mdl_surface_outputs.usda
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/no_material_extra_point.usda
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/materialx_proc_graph/scene.usda
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/materialx_resource_graph/scene.usda)

    add_executable(xform_correctness_test tests/xform_correctness_test.c
                                          $<TARGET_OBJECTS:nusd_gles_core>)
    target_link_libraries(xform_correctness_test PRIVATE nusd_gles_core)
    if(APPLE)
        target_link_libraries(xform_correctness_test PRIVATE c++ m ${CMAKE_DL_LIBS})
    else()
        target_link_libraries(xform_correctness_test PRIVATE stdc++ m ${CMAKE_DL_LIBS})
    endif()
    add_test(NAME xform_correctness_test
             COMMAND xform_correctness_test
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/reset_xform_stack.usda
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/metrics_zup_centimeters.usda
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/face_varying_uv_seam.usda)
    set_tests_properties(xform_correctness_test PROPERTIES
        ENVIRONMENT "NUSD_BAKE_ZUP_TO_YUP=1;NUSD_APPLY_METERS_PER_UNIT=1")

    add_executable(determinism_test tests/determinism_test.c
                                    $<TARGET_OBJECTS:nusd_gles_core>)
    target_link_libraries(determinism_test PRIVATE nusd_gles_core)
    if(APPLE)
        target_link_libraries(determinism_test PRIVATE
            "-framework Cocoa" "-framework IOKit" c++ m ${CMAKE_DL_LIBS})
    else()
        target_link_libraries(determinism_test PRIVATE stdc++ m ${CMAKE_DL_LIBS})
    endif()

    add_executable(curves_smoketest tests/curves_smoketest.c
                                    $<TARGET_OBJECTS:nusd_gles_core>)
    target_link_libraries(curves_smoketest PRIVATE nusd_gles_core)
    if(APPLE)
        target_link_libraries(curves_smoketest PRIVATE
            "-framework Cocoa" "-framework IOKit" c++ m ${CMAKE_DL_LIBS})
    else()
        target_link_libraries(curves_smoketest PRIVATE stdc++ m ${CMAKE_DL_LIBS})
    endif()

    add_executable(geo_cache_test tests/geo_cache_test.c
                                  $<TARGET_OBJECTS:nusd_gles_core>)
    target_link_libraries(geo_cache_test PRIVATE nusd_gles_core)
    if(APPLE)
        target_link_libraries(geo_cache_test PRIVATE c++ m ${CMAKE_DL_LIBS})
    else()
        target_link_libraries(geo_cache_test PRIVATE stdc++ m ${CMAKE_DL_LIBS})
    endif()
    add_test(NAME geo_cache_test
             COMMAND geo_cache_test ${CMAKE_CURRENT_SOURCE_DIR}/test_cube.usda)

    add_executable(geo_cache_render_smoke tests/geo_cache_render_smoke.c
                                          $<TARGET_OBJECTS:nusd_gles_core>)
    target_link_libraries(geo_cache_render_smoke PRIVATE nusd_gles_core)
    if(APPLE)
        target_link_libraries(geo_cache_render_smoke PRIVATE
            "-framework Cocoa" "-framework IOKit" c++ m ${CMAKE_DL_LIBS})
    else()
        target_link_libraries(geo_cache_render_smoke PRIVATE stdc++ m ${CMAKE_DL_LIBS})
    endif()

    add_executable(dedup_render_gate tests/dedup_render_gate.c
                                     $<TARGET_OBJECTS:nusd_gles_core>)
    target_link_libraries(dedup_render_gate PRIVATE nusd_gles_core)
    if(APPLE)
        target_link_libraries(dedup_render_gate PRIVATE
            "-framework Cocoa" "-framework IOKit" c++ m ${CMAKE_DL_LIBS})
    else()
        target_link_libraries(dedup_render_gate PRIVATE stdc++ m ${CMAKE_DL_LIBS})
    endif()
    # Asserts cross-material dedup (default-on) renders bit-identically to the
    # per-mesh path. Real pixel gate on EGL; trivially passes where headless GL
    # only clears. SKIP (77) when no GL context is available.
    add_test(NAME dedup_render_gate
             COMMAND dedup_render_gate ${CMAKE_CURRENT_SOURCE_DIR}/test_cube.usda)
    set_tests_properties(dedup_render_gate PROPERTIES SKIP_RETURN_CODE 77)

    # cull_stats — CPU-only measurement tool (frustum cull / straddle stats).
    add_executable(cull_stats tests/cull_stats.c
                              $<TARGET_OBJECTS:nusd_gles_core>)
    target_link_libraries(cull_stats PRIVATE nusd_gles_core)
    if(APPLE)
        target_link_libraries(cull_stats PRIVATE c++ m ${CMAKE_DL_LIBS})
    else()
        target_link_libraries(cull_stats PRIVATE stdc++ m ${CMAKE_DL_LIBS})
    endif()

    # meshlet_draw_test — render-compare gate for the meshlet draw path.
    add_executable(meshlet_draw_test tests/meshlet_draw_test.c
                                     $<TARGET_OBJECTS:nusd_gles_core>)
    target_link_libraries(meshlet_draw_test PRIVATE nusd_gles_core)
    if(APPLE)
        target_link_libraries(meshlet_draw_test PRIVATE
            "-framework Cocoa" "-framework IOKit" c++ m ${CMAKE_DL_LIBS})
    else()
        target_link_libraries(meshlet_draw_test PRIVATE stdc++ m ${CMAKE_DL_LIBS})
    endif()

    add_test(NAME determinism_test
             COMMAND determinism_test
                     ${CMAKE_CURRENT_SOURCE_DIR}/test_pbr_materials.usda)
    set_tests_properties(determinism_test PROPERTIES SKIP_RETURN_CODE 77)

    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME python_bindings_test
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_bindings_test.py)
        set_tests_properties(python_bindings_test PROPERTIES
            SKIP_RETURN_CODE 77
            ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/nanousd:$ENV{LD_LIBRARY_PATH}")

        add_test(NAME compare_smoke
                 COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_smoke.sh)
        set_tests_properties(compare_smoke PROPERTIES
            SKIP_RETURN_CODE 77
            ENVIRONMENT "REPO=${CMAKE_CURRENT_SOURCE_DIR};PYTHON=${Python3_EXECUTABLE}")

        # Three-way BasisCurves diff (ours vs. Hydra/Storm GL vs. Metal).
        # EXPECTED to fail its RMSE pass threshold until phase 2 of the
        # BasisCurves port lands; the script returns 0 as long as the
        # infrastructure ran end-to-end. Returns 77 (SKIP) when neither
        # libnusd_gles nor any usdrecord install is available.
        add_test(NAME curves_compare
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/tests/curves_compare.py)
        set_tests_properties(curves_compare PROPERTIES
            SKIP_RETURN_CODE 77
            ENVIRONMENT "DYLD_LIBRARY_PATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/nanousd:$ENV{DYLD_LIBRARY_PATH};LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/nanousd:$ENV{LD_LIBRARY_PATH}")

        # Automated OpenGL-vs-OVRTX FLIP regression gate over the committed
        # comparison frames. Skips (77) when flip-evaluator/numpy/Pillow/scipy/
        # scikit-image are absent (scipy + scikit-image are required — they power
        # the gate's GMSD structure and CIEDE2000 colour legs); else fails on any regression.
        add_test(NAME flip_opengl_vs_ovrtx
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/comparisons/flip_pipeline.py --check)
        set_tests_properties(flip_opengl_vs_ovrtx PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            SKIP_RETURN_CODE 77)
    endif()
endif()
