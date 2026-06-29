# CMakeLists.txt — fmi_usd_helper
#
# Builds a standalone C++ executable that reads FMI USD schema from a .usda
# file and prints JSON to stdout (same format as parse_fmi_schema.py).
#
# Links against the USD .so files bundled in the ovphysx SDK instead of a
# separately-installed OpenUSD or the usd-core pip package.
#
# Required variables (pass at configure time or via env):
#
#   OVPHYSX_SDK_DIR   — root of the extracted ovphysx SDK tar.gz.
#                       USD .so files live in ${OVPHYSX_SDK_DIR}/plugins/.
#
#   OPENUSD_INCLUDE_DIR — root of an OpenUSD source tree whose pxr/ directory
#                         contains the C++ headers.  Only the headers are
#                         needed — the USD itself is not built here.
#                         Minimum version: OpenUSD 25.11 (match ovphysx ABI).
#
# Quick setup (Linux):
#
#   # 1. Get just the pxr/ headers from OpenUSD v25.11:
#   git clone --depth 1 --branch v25.11 --filter=blob:none \
#     https://github.com/PixarAnimationStudios/OpenUSD.git openusd_src
#
#   # 2. Build:
#   cmake -B build \
#     -DOVPHYSX_SDK_DIR=/path/to/ovphysx-sdk \
#     -DOPENUSD_INCLUDE_DIR=/path/to/openusd_src \
#     apps/ov-fmi/fmi_usd_helper
#   cmake --build build
#
# The resulting binary is at build/fmi_usd_helper.
# main.py looks for it at:
#   apps/ov-fmi/fmi_usd_helper/build/fmi_usd_helper
#   apps/ov-fmi/fmi_usd_helper/fmi_usd_helper

cmake_minimum_required(VERSION 3.18)
project(fmi_usd_helper CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Validate required inputs ----

if(NOT DEFINED OVPHYSX_SDK_DIR)
    message(FATAL_ERROR
        "OVPHYSX_SDK_DIR is required.\n"
        "Point it to the root of the extracted ovphysx SDK tar.gz:\n"
        "  cmake -DOVPHYSX_SDK_DIR=/path/to/ovphysx-sdk ...")
endif()

if(NOT DEFINED OPENUSD_INCLUDE_DIR)
    message(FATAL_ERROR
        "OPENUSD_INCLUDE_DIR is required.\n"
        "Point it to an OpenUSD 25.11 source tree that contains pxr/ headers:\n"
        "  git clone --depth 1 --branch v25.11 \\\n"
        "    https://github.com/PixarAnimationStudios/OpenUSD.git openusd_src\n"
        "  cmake -DOPENUSD_INCLUDE_DIR=/path/to/openusd_src ...")
endif()

set(USD_LIB_DIR "${OVPHYSX_SDK_DIR}/plugins")

# ---- Find USD shared libraries ----

foreach(USD_LIB ar sdf usd arch tf vt gf)
    find_library(USD_LIB_${USD_LIB}
        NAMES libusd_${USD_LIB}.so usd_${USD_LIB}
        PATHS "${USD_LIB_DIR}"
        NO_DEFAULT_PATH
        REQUIRED
    )
    message(STATUS "Found USD lib: ${USD_LIB_${USD_LIB}}")
endforeach()

# ---- Target ----

add_executable(fmi_usd_helper main.cpp)

target_include_directories(fmi_usd_helper PRIVATE "${OPENUSD_INCLUDE_DIR}")

# Match the CXX11 ABI used by ovphysx (default on modern GCC, but be explicit)
target_compile_definitions(fmi_usd_helper PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(fmi_usd_helper PRIVATE
    ${USD_LIB_ar}
    ${USD_LIB_sdf}
    ${USD_LIB_usd}
    ${USD_LIB_arch}
    ${USD_LIB_tf}
    ${USD_LIB_vt}
    ${USD_LIB_gf}
)

# Bake the ovphysx plugins/ directory into the RPATH so the binary finds the
# USD .so files at runtime without LD_LIBRARY_PATH.
set_target_properties(fmi_usd_helper PROPERTIES
    BUILD_RPATH   "${USD_LIB_DIR}"
    INSTALL_RPATH "${USD_LIB_DIR}"
    BUILD_RPATH_USE_ORIGIN TRUE
)
