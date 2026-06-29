# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(nanousdview LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The supported viewer is the Python nanousdview entry point, which talks only
# to ovrtx.Renderer and selects implementations with NANOUSD_VIEW_BACKEND /
# NANOUSD_OVRTX_BACKEND. Generate a build-tree launcher so tests and humans
# can use the same executable path.
set(NANOUSDVIEW_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
get_filename_component(NANOUSDVIEW_WORKSPACE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/nanousdview.in"
    "${CMAKE_CURRENT_BINARY_DIR}/nanousdview"
    @ONLY
)
execute_process(
    COMMAND chmod +x "${CMAKE_CURRENT_BINARY_DIR}/nanousdview"
    RESULT_VARIABLE NANOUSDVIEW_CHMOD_RESULT
)
if(NOT NANOUSDVIEW_CHMOD_RESULT EQUAL 0)
    message(FATAL_ERROR "nanousdview: failed to mark launcher executable")
endif()

include(CTest)

if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    add_test(
        NAME nanousdview_help_smoke
        COMMAND "${CMAKE_CURRENT_BINARY_DIR}/nanousdview" --help
    )
    set_tests_properties(
        nanousdview_help_smoke
        PROPERTIES LABELS "smoke;viewer"
    )

    option(NANOUSDVIEW_ENABLE_GOLDEN_TESTS
           "Register nanousdview golden image comparison with CTest" OFF)
    set(NANOUSDVIEW_GOLDEN_BACKENDS "opengl" CACHE STRING
        "Comma-separated backends for the nanousdview golden CTest")
    set(NANOUSDVIEW_GOLDEN_CTEST_ARGS "" CACHE STRING
        "Extra semicolon-separated arguments for tests/render_goldens.py")

    if(NANOUSDVIEW_ENABLE_GOLDEN_TESTS)
        add_test(
            NAME nanousdview_goldens
            COMMAND
                "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/render_goldens.py"
                --backend "${NANOUSDVIEW_GOLDEN_BACKENDS}"
                --launcher "${CMAKE_CURRENT_BINARY_DIR}/nanousdview"
                ${NANOUSDVIEW_GOLDEN_CTEST_ARGS}
        )
        set_tests_properties(
            nanousdview_goldens
            PROPERTIES
                LABELS "golden;renderer;viewer;gpu"
                SKIP_RETURN_CODE 77
        )
    endif()
endif()

message(STATUS "nanousdview: generated Python launcher at ${CMAKE_CURRENT_BINARY_DIR}/nanousdview")
