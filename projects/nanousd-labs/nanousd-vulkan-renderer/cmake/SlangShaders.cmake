# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SlangShaders.cmake — Slang shader integration for VK3DGRT
#
# Fetches a prebuilt slangc release and provides nu_compile_slang() to
# compile a .slang shader to SPIR-V. Coexists with the existing GLSL
# pipeline (glslc/glslangValidator); Slang outputs land under
# ${SHADER_DIR}/slang/ to avoid name collisions.

include_guard(GLOBAL)
include(FetchContent)

set(NU_SLANG_VERSION "2026.8" CACHE STRING "Slang release version")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(_NU_SLANG_PLATFORM "linux-x86_64")
elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(_NU_SLANG_PLATFORM "macos-aarch64")
elseif(APPLE)
    set(_NU_SLANG_PLATFORM "macos-x86_64")
else()
    message(FATAL_ERROR
        "Slang fetcher: unsupported platform "
        "${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(_NU_SLANG_URL
    "https://github.com/shader-slang/slang/releases/download/v${NU_SLANG_VERSION}/slang-${NU_SLANG_VERSION}-${_NU_SLANG_PLATFORM}.tar.gz")

FetchContent_Declare(slang_prebuilt URL ${_NU_SLANG_URL})
FetchContent_MakeAvailable(slang_prebuilt)

find_program(SLANGC slangc
    HINTS ${slang_prebuilt_SOURCE_DIR}/bin
    NO_DEFAULT_PATH
    REQUIRED)
message(STATUS "Slang: ${SLANGC} (v${NU_SLANG_VERSION})")

# nu_compile_slang(OUT_VAR FILE STAGE ENTRY)
#
# Append a build rule that compiles ${FILE} (.slang) to a .spv under
# ${SHADER_DIR}/slang/. STAGE is one of: raygeneration, closesthit, miss,
# anyhit, intersection, compute, vertex, fragment. ENTRY is the entry-point
# function name in the .slang source. The output path is appended to the
# variable named by OUT_VAR (in parent scope).
function(nu_compile_slang OUT_VAR FILE STAGE ENTRY)
    cmake_path(ABSOLUTE_PATH FILE BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} NORMALIZE)
    get_filename_component(_name ${FILE} NAME_WE)
    set(_out "${SHADER_DIR}/slang/${_name}.${STAGE}.spv")
    set(_dep "${SHADER_DIR}/slang/${_name}.${STAGE}.d")
    file(MAKE_DIRECTORY "${SHADER_DIR}/slang")

    add_custom_command(
        OUTPUT ${_out}
        COMMAND ${SLANGC}
            -target spirv
            -profile sm_6_6
            -emit-spirv-directly
            -fvk-use-entrypoint-name
            -matrix-layout-column-major
            -O3
            -entry ${ENTRY}
            -stage ${STAGE}
            -depfile ${_dep}
            -o ${_out}
            ${FILE}
        DEPENDS ${FILE}
        DEPFILE ${_dep}
        COMMENT "Slang ${_name}.slang [${STAGE}/${ENTRY}] -> ${_name}.${STAGE}.spv"
        VERBATIM
    )

    set(${OUT_VAR} ${${OUT_VAR}} ${_out} PARENT_SCOPE)
endfunction()
