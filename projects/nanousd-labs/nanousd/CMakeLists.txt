# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(nanousd VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/sanitizers/sanitize.cmake")

# Options
option(NANOUSD_BUILD_TESTS "Build tests" ON)
option(NANOUSD_SHARED_API "Build nanousdapi as a shared library (OFF = static)" OFF)
option(NANOUSD_BUILD_OPENUSD_BACKEND "Build OpenUSD backend (requires OPENUSD_ROOT)" OFF)
set(OPENUSD_ROOT "" CACHE PATH "Root of prebuilt OpenUSD installation")

add_compile_options(
    $<$<COMPILE_LANG_AND_ID:C,GNU,Clang,AppleClang>:-Werror>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Werror>
)

function(nanousd_install)
    if(NOT NANOUSD_ENABLE_SANITIZERS)
        install(${ARGV})
    endif()
endfunction()

# --- nanousd: shared library combining core implementation + built-in backend ---
#
# This is the default backend. An alternative backend (e.g. openusd_backend)
# can be loaded at runtime by setting NANOUSD_BACKEND=/path/to/libopenusd_backend.so

add_library(nanousd SHARED
    src/types.cpp
    src/unicode_identifier.cpp
    src/path.cpp
    src/path_pool.cpp
    src/relative_path.cpp
    src/uri.cpp
    src/value.cpp
    src/asset_resolver.cpp
    src/resource.cpp
    src/usda_parser.cpp
    src/usdc_parser.cpp
    src/usdc_writer.cpp
    src/usda_writer.cpp
    src/usd_parser.cpp
    src/usdz_package.cpp
    src/lz4_decode.cpp
    src/compose.cpp
    src/stage.cpp
    src/schema.cpp
    src/clips.cpp
    src/diagnostic.cpp
    src/spline.cpp
    src/xform.cpp
    src/nanousd_backend.cpp
)

target_include_directories(nanousd PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(nanousd PRIVATE NANOUSD_BACKEND_BUILDING)

# psapi: used by the MUSD_TRACE_PARSE diagnostic in usdc_parser.cpp
# (and the load-phase benchmarks) to query process working-set size
# for per-phase RSS attribution. Linked unconditionally on Windows
# since the diagnostic is compiled in on every build and gated only
# at runtime.
if(WIN32)
    target_link_libraries(nanousd PRIVATE psapi)
endif()
nanousd_sanitize_target(nanousd)

set_target_properties(nanousd PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# --- nanousdapi: C API dispatch layer (static or shared, consumer's choice) ---
#
# Thin layer that dlopen's a backend (nanousd by default) and dispatches
# all public C API calls through the backend vtable.

if(NANOUSD_SHARED_API)
    add_library(nanousdapi SHARED src/nanousdapi.cpp)
    target_compile_definitions(nanousdapi PRIVATE NANOUSD_BUILDING)
else()
    add_library(nanousdapi STATIC src/nanousdapi.cpp)
    target_compile_definitions(nanousdapi PUBLIC NANOUSD_STATIC)
    set_target_properties(nanousdapi PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(nanousdapi PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link dl on Unix for dlopen/dlsym/dladdr
if(NOT WIN32)
    target_link_libraries(nanousdapi PRIVATE ${CMAKE_DL_LIBS})
endif()
nanousd_sanitize_target(nanousdapi)

# When nanousdapi is static, consumers must also load libnanousd.so so the
# RTLD_DEFAULT probe in LoadBackend() finds nanousd_create_backend_v1.
# --no-as-needed prevents the linker from dropping it (no compile-time refs).
if(NOT NANOUSD_SHARED_API)
    if(APPLE)
        target_link_libraries(nanousdapi PUBLIC nanousd)
    elseif(NOT WIN32)
        target_link_libraries(nanousdapi
            PUBLIC -Wl,--no-as-needed nanousd -Wl,--as-needed)
    else()
        target_link_libraries(nanousdapi PUBLIC nanousd)
    endif()
else()
    add_dependencies(nanousdapi nanousd)
endif()

# Place shared libs in the same output directory so nanousdapi can find the backend
set_target_properties(nanousd nanousdapi PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
)

# --- Install rules ---

nanousd_install(TARGETS nanousd nanousdapi
    EXPORT nanousdTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

nanousd_install(DIRECTORY include/nanousd
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

nanousd_install(EXPORT nanousdTargets
    FILE nanousdTargets.cmake
    NAMESPACE nanousd::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nanousd
)

configure_package_config_file(
    cmake/nanousdConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/nanousdConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nanousd
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/nanousdConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

nanousd_install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/nanousdConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/nanousdConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nanousd
)

# --- OpenUSD backend (optional) ---
#
# An alternative backend that delegates to a prebuilt OpenUSD (pxr) library.
# Enable with: cmake -DNANOUSD_BUILD_OPENUSD_BACKEND=ON -DOPENUSD_ROOT=/path/to/openusd ..
# Test backend swap with: NANOUSD_BACKEND=./libopenusd_backend.so ./compliance_test

if(NANOUSD_BUILD_OPENUSD_BACKEND)
    if(NOT OPENUSD_ROOT OR NOT EXISTS "${OPENUSD_ROOT}/include/pxr/pxr.h")
        message(FATAL_ERROR
            "NANOUSD_BUILD_OPENUSD_BACKEND=ON requires OPENUSD_ROOT set to a "
            "prebuilt OpenUSD installation (must contain include/pxr/pxr.h).")
    endif()

    # Find the monolithic OpenUSD shared library
    if(WIN32)
        file(GLOB _openusd_libs "${OPENUSD_ROOT}/lib/*usd_ms.lib")
    else()
        file(GLOB _openusd_libs "${OPENUSD_ROOT}/lib/*usd_ms.so")
    endif()
    list(LENGTH _openusd_libs _openusd_libs_count)
    if(_openusd_libs_count EQUAL 0)
        message(FATAL_ERROR
            "No monolithic OpenUSD library (*usd_ms.so) found in ${OPENUSD_ROOT}/lib")
    endif()
    list(GET _openusd_libs 0 OPENUSD_LIB)
    message(STATUS "OpenUSD monolithic lib: ${OPENUSD_LIB}")

    add_library(openusd_backend SHARED
        backends/openusd_backend/openusd_backend.cpp
    )

    target_include_directories(openusd_backend PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        "${OPENUSD_ROOT}/include"
    )

    target_link_directories(openusd_backend PRIVATE "${OPENUSD_ROOT}/lib")
    target_link_libraries(openusd_backend PRIVATE "${OPENUSD_LIB}")

    target_compile_definitions(openusd_backend PRIVATE NANOUSD_BACKEND_BUILDING)
    nanousd_sanitize_target(openusd_backend)

    if(MSVC)
        target_compile_definitions(openusd_backend PRIVATE NOMINMAX)
    endif()

    set_target_properties(openusd_backend PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
    )

    nanousd_install(TARGETS openusd_backend
        EXPORT nanousdTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# --- Tests ---

if(NANOUSD_BUILD_TESTS)
    enable_testing()

    add_executable(nanousd_tests
        tests/test_main.cpp
        tests/test_traverse_provenance.cpp
    )
    target_link_libraries(nanousd_tests PRIVATE nanousd)
    nanousd_sanitize_target(nanousd_tests)
    # Keep asserts active in tests even for Release builds (-DNDEBUG)
    target_compile_options(nanousd_tests PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>
    )
    add_test(NAME nanousd_tests COMMAND nanousd_tests
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # Compliance test — pure C11, links only nanousdapi (no core dependency)
    add_executable(compliance_test tests/compliance/compliance_test.c)
    target_link_libraries(compliance_test PRIVATE nanousdapi)
    nanousd_sanitize_target(compliance_test)
    set_property(TARGET compliance_test PROPERTY C_STANDARD 11)
    add_test(NAME compliance_test COMMAND compliance_test
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(compliance_test PROPERTIES LABELS "compliance;release")

    nanousd_sanitize_configure_tests(nanousd_tests compliance_test)

    # Benchmark — pure C11, links only nanousdapi. Not a ctest target
    # because it requires external stage data passed as a CLI argument.
    #   ./benchmark_stage_load <path/to/root.usd>
    add_executable(benchmark_stage_load tests/benchmark_stage_load.c)
    target_link_libraries(benchmark_stage_load PRIVATE nanousdapi)
    nanousd_sanitize_target(benchmark_stage_load)
    set_property(TARGET benchmark_stage_load PROPERTY C_STANDARD 11)
    if(WIN32)
        target_link_libraries(benchmark_stage_load PRIVATE psapi)
    endif()

    # Phase-by-phase memory probe — links nanousd directly so it can
    # call ParseUsdFile / Compose / Stage construction individually
    # and report RSS deltas at each phase boundary. Useful for
    # diagnosing where memory growth originates.
    #   ./benchmark_load_phases <path/to/root.usd>
    add_executable(benchmark_load_phases tests/benchmark_load_phases.cpp)
    target_link_libraries(benchmark_load_phases PRIVATE nanousd)
    nanousd_sanitize_target(benchmark_load_phases)
    if(WIN32)
        target_link_libraries(benchmark_load_phases PRIVATE psapi)
    endif()

    # Place test executables alongside the shared libs
    set_target_properties(nanousd_tests compliance_test
                          benchmark_stage_load benchmark_load_phases
                          PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
    )
endif()
