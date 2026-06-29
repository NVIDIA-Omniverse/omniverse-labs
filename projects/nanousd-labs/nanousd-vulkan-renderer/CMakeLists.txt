# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)  # 3.20 for DEPFILE in add_custom_command (Slang dep tracking)
project(nanousd-vulkan-renderer LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(_NU_BUILD_GS_RT_DEFAULT ON)
set(_NUSD_ENABLE_PTEX_DEFAULT ON)
set(_NUSD_ENABLE_OPENMP_DEFAULT ON)
if(WIN32)
    set(_NU_BUILD_GS_RT_DEFAULT OFF)
    set(_NUSD_ENABLE_PTEX_DEFAULT OFF)
endif()
if(MSVC)
    set(_NUSD_ENABLE_OPENMP_DEFAULT OFF)
endif()

option(NU_BUILD_GS_RT "Build VK3DGRT (3D Gaussian Ray Tracing) path. Requires Slang." ${_NU_BUILD_GS_RT_DEFAULT})
option(NUSD_ENABLE_MDL_SDK "Enable optional NVIDIA MDL SDK integration hooks." OFF)
option(NUSD_ENABLE_PTEX "Enable Disney Ptex material loading for authored .ptx surface maps." ${_NUSD_ENABLE_PTEX_DEFAULT})
option(NUSD_ENABLE_OPENMP "Enable OpenMP acceleration for curve loading and material baking." ${_NUSD_ENABLE_OPENMP_DEFAULT})
option(NUSD_WINDOWS_COMPAT "Enable Windows compatibility shims for native MSVC builds." ON)
set(MDL_SDK_ROOT "" CACHE PATH "Optional install prefix or build prefix for NVIDIA MDL SDK.")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build shared library by default
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

include(FetchContent)

find_package(Vulkan)

# RTX Mega Geometry (VK_NV_cluster_acceleration_structure / partitioned AS / LSS)
# needs Vulkan headers >= ~v304; the system headers may be older (e.g. v275) and
# lack the struct/enum definitions. Prefer a newer header set when available so
# the cluster-AS device setup compiles. Applies dir-wide (incl. volk) via BEFORE.
# Overridable: -DNUSD_VULKAN_HEADERS=/path/to/include.
if(NOT NUSD_VULKAN_HEADERS)
    if(DEFINED ENV{NUSD_VULKAN_HEADERS})
        set(NUSD_VULKAN_HEADERS "$ENV{NUSD_VULKAN_HEADERS}")
    elseif(EXISTS "$ENV{HOME}/blender/lib/linux_x64/vulkan_headers/include/vulkan/vulkan_core.h")
        set(NUSD_VULKAN_HEADERS "$ENV{HOME}/blender/lib/linux_x64/vulkan_headers/include")
    elseif(DEFINED ENV{VULKAN_SDK} AND EXISTS "$ENV{VULKAN_SDK}/include/vulkan/vulkan_core.h")
        set(NUSD_VULKAN_HEADERS "$ENV{VULKAN_SDK}/include")
    endif()
endif()
if(NUSD_VULKAN_HEADERS AND EXISTS "${NUSD_VULKAN_HEADERS}/vulkan/vulkan_core.h")
    include_directories(BEFORE SYSTEM "${NUSD_VULKAN_HEADERS}")
    message(STATUS "nusd: Vulkan headers (RT mega-geometry) from ${NUSD_VULKAN_HEADERS}")
else()
    message(WARNING "nusd: no newer Vulkan headers found; cluster-AS extensions may not compile")
endif()

# GLFW (hidden window for Vulkan surface — true headless in v2)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG 1.4.304
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(volk)

# nanousd
if(NOT TARGET nanousdapi)
    if(NANOUSD_DIR)
        find_path(NANOUSD_INCLUDE_DIR nanousd/nanousdapi.h
            PATHS ${NANOUSD_DIR}/include
                  ${NANOUSD_DIR}/_install/release/include
                  ${NANOUSD_DIR}/_install/debug/include
            NO_DEFAULT_PATH)
        find_library(NANOUSD_STATIC nanousdapi
            PATHS ${NANOUSD_DIR}/build/Release
                  ${NANOUSD_DIR}/build
                  ${NANOUSD_DIR}/build-win/Release
                  ${NANOUSD_DIR}/build-win
                  ${NANOUSD_DIR}/lib
                  ${NANOUSD_DIR}/_build/release/Release
                  ${NANOUSD_DIR}/_build/debug/Debug
                  ${NANOUSD_DIR}/_install/release/lib
                  ${NANOUSD_DIR}/_install/debug/lib
            NO_DEFAULT_PATH)
        find_library(NANOUSD_SHARED nanousd
            PATHS ${NANOUSD_DIR}/build/Release
                  ${NANOUSD_DIR}/build
                  ${NANOUSD_DIR}/build-win/Release
                  ${NANOUSD_DIR}/build-win
                  ${NANOUSD_DIR}/lib
                  ${NANOUSD_DIR}/_build/release/Release
                  ${NANOUSD_DIR}/_build/debug/Debug
                  ${NANOUSD_DIR}/_install/release/lib
                  ${NANOUSD_DIR}/_install/debug/lib
            NO_DEFAULT_PATH)
    else()
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../nanousd/CMakeLists.txt)
            set(NANOUSD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../nanousd)
            find_path(NANOUSD_INCLUDE_DIR nanousd/nanousdapi.h
                PATHS ${NANOUSD_DIR}/include
                      ${NANOUSD_DIR}/_install/release/include
                      ${NANOUSD_DIR}/_install/debug/include
                NO_DEFAULT_PATH)
            find_library(NANOUSD_STATIC nanousdapi
                PATHS ${NANOUSD_DIR}/build/Release
                      ${NANOUSD_DIR}/build
                      ${NANOUSD_DIR}/build-win/Release
                      ${NANOUSD_DIR}/build-win
                      ${NANOUSD_DIR}/lib
                      ${NANOUSD_DIR}/_build/release/Release
                      ${NANOUSD_DIR}/_build/debug/Debug
                      ${NANOUSD_DIR}/_install/release/lib
                      ${NANOUSD_DIR}/_install/debug/lib
                NO_DEFAULT_PATH)
            find_library(NANOUSD_SHARED nanousd
                PATHS ${NANOUSD_DIR}/build/Release
                      ${NANOUSD_DIR}/build
                      ${NANOUSD_DIR}/build-win/Release
                      ${NANOUSD_DIR}/build-win
                      ${NANOUSD_DIR}/lib
                      ${NANOUSD_DIR}/_build/release/Release
                      ${NANOUSD_DIR}/_build/debug/Debug
                      ${NANOUSD_DIR}/_install/release/lib
                      ${NANOUSD_DIR}/_install/debug/lib
                NO_DEFAULT_PATH)
        else()
            find_path(NANOUSD_INCLUDE_DIR nanousd/nanousdapi.h)
            find_library(NANOUSD_STATIC nanousdapi)
            find_library(NANOUSD_SHARED nanousd)
        endif()
    endif()

    set(_NANOUSD_RUNTIME_HINTS)
    if(NANOUSD_DIR)
        list(APPEND _NANOUSD_RUNTIME_HINTS
            ${NANOUSD_DIR}/build/Release
            ${NANOUSD_DIR}/build
            ${NANOUSD_DIR}/build-win/Release
            ${NANOUSD_DIR}/build-win
            ${NANOUSD_DIR}/lib
            ${NANOUSD_DIR}/bin
            ${NANOUSD_DIR}/_build/release/Release
            ${NANOUSD_DIR}/_build/debug/Debug
            ${NANOUSD_DIR}/_build/release
            ${NANOUSD_DIR}/_build/debug
            ${NANOUSD_DIR}/_install/release/lib
            ${NANOUSD_DIR}/_install/debug/lib
            ${NANOUSD_DIR}/_install/release/bin
            ${NANOUSD_DIR}/_install/debug/bin)
    endif()
    if(WIN32 AND _NANOUSD_RUNTIME_HINTS)
        find_file(NANOUSD_API_DLL nanousdapi.dll
            PATHS ${_NANOUSD_RUNTIME_HINTS}
            NO_DEFAULT_PATH)
        find_file(NANOUSD_BACKEND_DLL nanousd.dll
            PATHS ${_NANOUSD_RUNTIME_HINTS}
            NO_DEFAULT_PATH)
    endif()

    # Detect whether NANOUSD_STATIC is actually a static archive or a shared
    # import library. On Windows, .lib is ambiguous, so a sibling DLL wins.
    set(_NANOUSD_API_KIND "")
    if(NANOUSD_STATIC)
        get_filename_component(_NANOUSD_API_EXT ${NANOUSD_STATIC} EXT)
        if(WIN32 AND _NANOUSD_API_EXT STREQUAL ".lib" AND NANOUSD_API_DLL)
            set(_NANOUSD_API_KIND "SHARED")
        elseif(_NANOUSD_API_EXT STREQUAL ".a" OR _NANOUSD_API_EXT STREQUAL ".lib")
            set(_NANOUSD_API_KIND "STATIC")
        else()
            set(_NANOUSD_API_KIND "SHARED")
        endif()
    endif()

    set(_NANOUSD_API_EXTRA_LIBS)
    if(NOT WIN32)
        list(APPEND _NANOUSD_API_EXTRA_LIBS stdc++)
    endif()

    if(NANOUSD_INCLUDE_DIR AND NANOUSD_STATIC AND NANOUSD_SHARED AND _NANOUSD_API_KIND STREQUAL "STATIC")
        add_library(nanousdapi STATIC IMPORTED)
        set_target_properties(nanousdapi PROPERTIES
            IMPORTED_LOCATION ${NANOUSD_STATIC}
            INTERFACE_INCLUDE_DIRECTORIES ${NANOUSD_INCLUDE_DIR}
            INTERFACE_COMPILE_DEFINITIONS NANOUSD_STATIC
            INTERFACE_LINK_LIBRARIES "${NANOUSD_SHARED};${_NANOUSD_API_EXTRA_LIBS}")
        message(STATUS "nanousd: ${NANOUSD_STATIC} (static) + ${NANOUSD_SHARED}")
    elseif(NANOUSD_INCLUDE_DIR AND NANOUSD_STATIC AND _NANOUSD_API_KIND STREQUAL "SHARED")
        # When the API library is shared, link dynamically — this lets the
        # renderer share libnanousd state with other consumers (e.g. pxr_compat
        # in nuView) instead of holding a private static copy that diverges.
        add_library(nanousdapi SHARED IMPORTED)
        if(WIN32)
            set_target_properties(nanousdapi PROPERTIES
                IMPORTED_IMPLIB ${NANOUSD_STATIC}
                IMPORTED_LOCATION ${NANOUSD_API_DLL}
                INTERFACE_INCLUDE_DIRECTORIES ${NANOUSD_INCLUDE_DIR})
        else()
            set_target_properties(nanousdapi PROPERTIES
                IMPORTED_LOCATION ${NANOUSD_STATIC}
                INTERFACE_INCLUDE_DIRECTORIES ${NANOUSD_INCLUDE_DIR}
                INTERFACE_LINK_LIBRARIES "${_NANOUSD_API_EXTRA_LIBS}")
        endif()
        message(STATUS "nanousd: ${NANOUSD_STATIC} (shared)")
    elseif(NANOUSD_INCLUDE_DIR AND NANOUSD_STATIC)
        add_library(nanousdapi STATIC IMPORTED)
        set_target_properties(nanousdapi PROPERTIES
            IMPORTED_LOCATION ${NANOUSD_STATIC}
            INTERFACE_INCLUDE_DIRECTORIES ${NANOUSD_INCLUDE_DIR}
            INTERFACE_COMPILE_DEFINITIONS NANOUSD_STATIC
            INTERFACE_LINK_LIBRARIES "${_NANOUSD_API_EXTRA_LIBS}")
        message(STATUS "nanousd: ${NANOUSD_STATIC} (static only)")
    elseif(NOT TARGET nanousdapi)
        message(FATAL_ERROR "nanousd not found. Set -DNANOUSD_DIR=<path>.")
    endif()
endif()

# MaterialX — reuse this build dir's FetchContent artifacts if available,
# otherwise fetch MaterialX below.
if(NOT DEFINED MATERIALX_BUILD_DIR)
    set(MATERIALX_BUILD_DIR ${CMAKE_BINARY_DIR})
endif()
set(MATERIALX_SRC_DIR ${MATERIALX_BUILD_DIR}/_deps/materialx-src)
set(MATERIALX_LIB_DIR ${MATERIALX_BUILD_DIR}/_deps/materialx-build/lib)

set(MATERIALX_BUILD_GEN_DIR ${MATERIALX_BUILD_DIR}/_deps/materialx-build/source)
set(_MATERIALX_REQUIRED_LIBS MaterialXCore MaterialXFormat MaterialXGenShader MaterialXGenGlsl)
set(_MATERIALX_PREBUILT TRUE)
if(NOT EXISTS ${MATERIALX_SRC_DIR}/source/MaterialXCore)
    set(_MATERIALX_PREBUILT FALSE)
endif()
foreach(lib ${_MATERIALX_REQUIRED_LIBS})
    if(NOT EXISTS ${MATERIALX_LIB_DIR}/lib${lib}.a)
        set(_MATERIALX_PREBUILT FALSE)
    endif()
endforeach()
if(_MATERIALX_PREBUILT)
    message(STATUS "Using pre-built MaterialX from ${MATERIALX_BUILD_DIR}")
    set(MATERIALX_INCLUDE_DIR ${MATERIALX_SRC_DIR}/source)
    set(MATERIALX_LIBRARIES_DIR ${MATERIALX_LIB_DIR})
    set(MATERIALX_SEARCH_PATH_DIR ${MATERIALX_SRC_DIR}/libraries)

    foreach(lib ${_MATERIALX_REQUIRED_LIBS})
        add_library(${lib} STATIC IMPORTED)
        set_target_properties(${lib} PROPERTIES
            IMPORTED_LOCATION ${MATERIALX_LIBRARIES_DIR}/lib${lib}.a
            INTERFACE_INCLUDE_DIRECTORIES "${MATERIALX_INCLUDE_DIR};${MATERIALX_BUILD_GEN_DIR}")
    endforeach()
else()
    message(STATUS "Fetching MaterialX from source (this may take a while)")
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

# -----------------------------------------------------------------------------
# meshoptimizer — vertex cache optimization at nu_add_mesh time.
# MIT licensed, header + .cpp, zero external deps. PIC because the static
# lib gets linked into the shared nusd_renderer.
# -----------------------------------------------------------------------------
FetchContent_Declare(meshoptimizer
    GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
    GIT_TAG        v0.24
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(meshoptimizer)
set_target_properties(meshoptimizer PROPERTIES POSITION_INDEPENDENT_CODE ON)

# shaderc
set(_NUSD_SHADERC_HINTS)
if(DEFINED ENV{VULKAN_SDK})
    list(APPEND _NUSD_SHADERC_HINTS
        "$ENV{VULKAN_SDK}/Lib"
        "$ENV{VULKAN_SDK}/lib")
endif()
list(APPEND _NUSD_SHADERC_HINTS /usr/lib/x86_64-linux-gnu)
find_library(SHADERC_LIB shaderc_shared shaderc HINTS ${_NUSD_SHADERC_HINTS})
if(NOT SHADERC_LIB)
    find_library(SHADERC_LIB shaderc_combined HINTS ${_NUSD_SHADERC_HINTS})
endif()
if(NOT SHADERC_LIB)
    message(FATAL_ERROR "shaderc not found. Install Vulkan SDK or shaderc.")
endif()

# CUDA (for Vulkan-CUDA interop)
find_package(CUDAToolkit)

if(NUSD_ENABLE_PTEX)
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
        third_party/ptex/src/ptex/PtexCache.cpp
        third_party/ptex/src/ptex/PtexFilters.cpp
        third_party/ptex/src/ptex/PtexHalf.cpp
        third_party/ptex/src/ptex/PtexReader.cpp
        third_party/ptex/src/ptex/PtexSeparableFilter.cpp
        third_party/ptex/src/ptex/PtexSeparableKernel.cpp
        third_party/ptex/src/ptex/PtexTriangleFilter.cpp
        third_party/ptex/src/ptex/PtexTriangleKernel.cpp
        third_party/ptex/src/ptex/PtexUtils.cpp
        third_party/ptex/src/ptex/PtexWriter.cpp
    )
    target_compile_definitions(nusd_ptex PUBLIC PTEX_STATIC)
    target_include_directories(nusd_ptex PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ptex/src/ptex)
    target_link_libraries(nusd_ptex PUBLIC Threads::Threads PRIVATE ${NUSD_LIBDEFLATE_TARGET})
    set_target_properties(nusd_ptex PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# -----------------------------------------------------------------------------
# Shader compilation
# -----------------------------------------------------------------------------

find_program(GLSLC glslc HINTS $ENV{VULKAN_SDK}/bin ${Vulkan_INCLUDE_DIR}/../bin)
find_program(GLSLANG glslangValidator
    HINTS $ENV{VULKAN_SDK}/bin
          ${Vulkan_INCLUDE_DIR}/../bin
          $ENV{HOME}/blender/lib/linux_x64/shaderc/bin
          $ENV{HOME}/build_linux/deps_x64/Release/shaderc/bin)
if(NOT GLSLC)
    find_program(GLSLC glslc
        HINTS $ENV{HOME}/blender/lib/linux_x64/shaderc/bin
              $ENV{HOME}/build_linux/deps_x64/Release/shaderc/bin)
endif()
if(NOT GLSLC)
    message(FATAL_ERROR
        "glslc not found. Install the glslc package/Vulkan SDK or configure with "
        "-DGLSLC=/path/to/glslc.")
endif()

# Workgroup size for the curve AABB-gen compute shader. Tunable at
# build time so the perf branch can A/B 64 / 128 / 256 without forking
# the shader. Both the GLSL `local_size_x` and the host-side dispatch
# divisor read this same constant via -DNUSD_AABB_GEN_LOCAL_X.
if(NOT DEFINED NUSD_AABB_GEN_LOCAL_X)
    set(NUSD_AABB_GEN_LOCAL_X 128 CACHE STRING "Workgroup size for curve_aabb_gen.comp")
endif()
message(STATUS "NUSD_AABB_GEN_LOCAL_X = ${NUSD_AABB_GEN_LOCAL_X}")

set(SHADER_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_DIR})

# Raster shaders
set(RASTER_SHADERS
    mesh.vert mesh_instanced.vert mesh.frag mesh_shadow.frag
    curve.vert curve.tesc curve.tese curve.frag
    overlay.vert overlay.frag)
set(RASTER_SHADER_OUTPUTS)
foreach(shader ${RASTER_SHADERS})
    set(src ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${shader}.glsl)
    set(out ${SHADER_DIR}/${shader}.spv)
    string(REGEX REPLACE ".*\\." "" stage ${shader})
    add_custom_command(
        OUTPUT ${out}
        COMMAND ${GLSLC} --target-env=vulkan1.2 -fshader-stage=${stage} -o ${out} ${src}
        DEPENDS ${src}
        COMMENT "Compiling ${shader}.glsl -> ${shader}.spv"
    )
    list(APPEND RASTER_SHADER_OUTPUTS ${out})
endforeach()

# Barycentric variant of mesh.frag (same source, -DUSE_BARYCENTRIC). Selected at
# runtime when VK_KHR_fragment_shader_barycentric is available so raster Ptex is
# interpolated per pixel instead of flat-averaged.
add_custom_command(
    OUTPUT ${SHADER_DIR}/mesh_bary.frag.spv
    COMMAND ${GLSLC} --target-env=vulkan1.2 -fshader-stage=frag -DUSE_BARYCENTRIC
            -o ${SHADER_DIR}/mesh_bary.frag.spv
            ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/mesh.frag.glsl
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/mesh.frag.glsl
    COMMENT "Compiling mesh.frag.glsl -> mesh_bary.frag.spv (USE_BARYCENTRIC)"
)
list(APPEND RASTER_SHADER_OUTPUTS ${SHADER_DIR}/mesh_bary.frag.spv)

# RT shaders
set(RT_SHADER_OUTPUTS)
set(RT_SHADER_COMPILER "")
set(RT_SHADER_COMPILER_KIND "")
if(GLSLANG)
    set(RT_SHADER_COMPILER ${GLSLANG})
    set(RT_SHADER_COMPILER_KIND glslang)
elseif(GLSLC)
    set(RT_SHADER_COMPILER ${GLSLC})
    set(RT_SHADER_COMPILER_KIND glslc)
endif()

if(RT_SHADER_COMPILER)
    set(RT_SHADERS raytrace.rgen raytrace.rmiss raytrace.rchit raytrace_tiled.rgen raycast.comp raytrace_curve.rint raytrace_curve.rchit curve_aabb_gen.comp tlas_translate.comp tlas_translate_partitioned.comp deferred_shade.comp)
    foreach(shader ${RT_SHADERS})
        set(src ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${shader}.glsl)
        set(out ${SHADER_DIR}/${shader}.spv)
        string(REGEX REPLACE ".*\\." "" stage ${shader})
        # curve_aabb_gen.comp picks up the configurable workgroup size.
        set(extra_flags)
        if(shader STREQUAL "curve_aabb_gen.comp")
            list(APPEND extra_flags
                 -DNUSD_AABB_GEN_LOCAL_X=${NUSD_AABB_GEN_LOCAL_X})
        endif()
        if(RT_SHADER_COMPILER_KIND STREQUAL glslc)
            add_custom_command(
                OUTPUT ${out}
                COMMAND ${RT_SHADER_COMPILER} --target-env=vulkan1.2
                        -fshader-stage=${stage} ${extra_flags} -o ${out} ${src}
                DEPENDS ${src}
                COMMENT "Compiling ${shader}.glsl -> ${shader}.spv"
            )
        else()
            add_custom_command(
                OUTPUT ${out}
                COMMAND ${RT_SHADER_COMPILER} --target-env vulkan1.2 -V
                        -S ${stage} ${extra_flags} -o ${out} ${src}
                DEPENDS ${src}
                COMMENT "Compiling ${shader}.glsl -> ${shader}.spv"
            )
        endif()
        list(APPEND RT_SHADER_OUTPUTS ${out})
    # SER (Shader Execution Reordering) variant of the tiled raygen.
    # Compiled with -DSER_ENABLED=1 from the same source. Loaded at runtime
    # only when VK_NV_ray_tracing_invocation_reorder is supported by the
    # device, otherwise the non-SER raytrace_tiled.rgen.spv is used.
    if(shader STREQUAL "raytrace_tiled.rgen")
        set(ser_out ${SHADER_DIR}/raytrace_tiled_ser.rgen.spv)
        if(RT_SHADER_COMPILER_KIND STREQUAL glslc)
            add_custom_command(
                OUTPUT ${ser_out}
                COMMAND ${RT_SHADER_COMPILER} --target-env=vulkan1.2
                        -fshader-stage=rgen -DSER_ENABLED=1 -o ${ser_out} ${src}
                DEPENDS ${src}
                COMMENT "Compiling raytrace_tiled.rgen.glsl (SER) -> raytrace_tiled_ser.rgen.spv"
            )
        else()
            add_custom_command(
                OUTPUT ${ser_out}
                COMMAND ${RT_SHADER_COMPILER} --target-env vulkan1.2 -V -S rgen
                        -DSER_ENABLED=1 -o ${ser_out} ${src}
                DEPENDS ${src}
                COMMENT "Compiling raytrace_tiled.rgen.glsl (SER) -> raytrace_tiled_ser.rgen.spv"
            )
        endif()
        list(APPEND RT_SHADER_OUTPUTS ${ser_out})
    endif()
    endforeach()
endif()

# VK3DGRT Slang shaders (gs_3dgrt.*) — see docs/plans/VK3DGRT_PLAN.md.
# Helper + slangc fetch live in cmake/SlangShaders.cmake. Outputs go under
# ${SHADER_DIR}/slang/ to avoid collisions with the GLSL track.
set(GS_SHADER_OUTPUTS)
if(NU_BUILD_GS_RT)
    include(SlangShaders)
    set(GS_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/slang)
    if(EXISTS ${GS_SHADER_DIR}/gs_3dgrt.rgen.slang)
        nu_compile_slang(GS_SHADER_OUTPUTS ${GS_SHADER_DIR}/gs_3dgrt.rgen.slang  raygeneration main)
        nu_compile_slang(GS_SHADER_OUTPUTS ${GS_SHADER_DIR}/gs_3dgrt.rahit.slang anyhit        main)
        nu_compile_slang(GS_SHADER_OUTPUTS ${GS_SHADER_DIR}/gs_3dgrt.rchit.slang closesthit    main)
        nu_compile_slang(GS_SHADER_OUTPUTS ${GS_SHADER_DIR}/gs_3dgrt.rint.slang  intersection  main)
        nu_compile_slang(GS_SHADER_OUTPUTS ${GS_SHADER_DIR}/gs_3dgrt.rmiss.slang miss          main)
        nu_compile_slang(GS_SHADER_OUTPUTS ${GS_SHADER_DIR}/gs_as_build.comp.slang compute     main)
    endif()
endif()

add_custom_target(renderer_shaders ALL DEPENDS ${RASTER_SHADER_OUTPUTS} ${RT_SHADER_OUTPUTS} ${GS_SHADER_OUTPUTS})

# -----------------------------------------------------------------------------
# Shared library: libnusd_renderer.so
# -----------------------------------------------------------------------------

if(WIN32 AND NUSD_WINDOWS_COMPAT)
    set(NUSD_WIN32_COMPAT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/win32_compat")
    set(NUSD_WIN32_COMPAT_HEADER "${NUSD_WIN32_COMPAT_DIR}/win32_compat.h")
    include_directories(BEFORE "${NUSD_WIN32_COMPAT_DIR}")
    add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX)
    if(MSVC)
        add_compile_options("/FI${NUSD_WIN32_COMPAT_HEADER}")
    else()
        add_compile_options("-include" "${NUSD_WIN32_COMPAT_HEADER}")
    endif()
    if(NOT TARGET m)
        add_library(m INTERFACE)
    endif()
endif()

add_library(nusd_renderer SHARED
    src/renderer.c
    src/gpu_vulkan.c
    src/scene.c
    src/geo_cache.c
    src/camera.c
    src/material.cpp
    src/ptex_material.cpp
    src/mdl_bridge.cpp
    src/dlss_stub.c
    src/gs_scene.c
)

target_include_directories(nusd_renderer
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${volk_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            /usr/include
)

target_compile_definitions(nusd_renderer PRIVATE
    SHADER_DIR="${SHADER_DIR}"
    VK_NO_PROTOTYPES
    MATERIALX_SEARCH_PATH="${MATERIALX_SEARCH_PATH_DIR}"
    NUSD_RENDERER_HEADLESS=1
    NUSD_AABB_GEN_LOCAL_X=${NUSD_AABB_GEN_LOCAL_X}u
)

target_link_libraries(nusd_renderer PRIVATE
    glfw
    nanousdapi
    volk_headers
    MaterialXGenGlsl
    MaterialXGenShader
    meshoptimizer
    MaterialXFormat
    MaterialXCore
    ${SHADERC_LIB}
    ${CMAKE_DL_LIBS}
)

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
        message(WARNING
            "NUSD_ENABLE_MDL_SDK=ON but CMake package 'mdl' was not found. "
            "Build/install NVIDIA/MDL-SDK and set -DMDL_SDK_ROOT=<prefix>.")
    endif()
endif()

if(CUDAToolkit_FOUND)
    target_link_libraries(nusd_renderer PRIVATE CUDA::cuda_driver)
    target_compile_definitions(nusd_renderer PRIVATE NUSD_CUDA_INTEROP=1)
endif()

if(NUSD_ENABLE_PTEX)
    target_link_libraries(nusd_renderer PRIVATE nusd_ptex)
    target_compile_definitions(nusd_renderer PRIVATE NUSD_HAVE_PTEX=1)
    message(STATUS "Disney Ptex enabled for authored .ptx material maps")
endif()

# OpenMP — used to parallelise per-prim curve loading + per-curve segment
# extraction. Falls back to the serial path automatically when not found
# (the source guards every #pragma omp under #ifdef _OPENMP).
if(NUSD_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        target_link_libraries(nusd_renderer PRIVATE OpenMP::OpenMP_C)
        message(STATUS "OpenMP enabled for nusd_renderer C (curves loader parallelisation)")
    endif()
    if(OpenMP_CXX_FOUND)
        # CXX too: material.cpp parallelises the MDL masked-albedo bake pixel loops
        # (measured ~8.5s of cold load; race-free, output-identical).
        target_link_libraries(nusd_renderer PRIVATE OpenMP::OpenMP_CXX)
        message(STATUS "OpenMP enabled for nusd_renderer C++ (material bake parallelisation)")
    endif()
    if(NOT OpenMP_C_FOUND AND NOT OpenMP_CXX_FOUND)
        message(STATUS "OpenMP not found — curves loader + material bakes use serial path")
    endif()
else()
    message(STATUS "OpenMP disabled - curves loader + material bakes use serial path")
endif()

add_dependencies(nusd_renderer renderer_shaders)

set_target_properties(nusd_renderer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    WINDOWS_EXPORT_ALL_SYMBOLS TRUE
    SOVERSION 1
    VERSION 1.0.0
)

# Install rules — make this repo the canonical source of nusd_renderer.h
# for sibling renderers (opengl, metal) and downstream consumers
# (nanousdview, IsaacLab). See UNIFIED_VIEWER_PLAN.md Phase 0.5.
# NOTE: OUTPUT_NAME stays "nusd_renderer" (no _vulkan suffix) until Phase 3
# lands the soname split with a back-compat symlink — flipping today would
# break sibling repos that link ${BUILD}/libnusd_renderer.so directly.
include(GNUInstallDirs)
install(TARGETS nusd_renderer
    EXPORT nusd_rendererTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES include/nusd_renderer.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT nusd_rendererTargets
    FILE nusd_rendererTargets.cmake
    NAMESPACE nusd::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nusd_renderer)

# Bind every internal call within libnusd_renderer.so to the symbol
# defined inside this library — even if the same name is also exported by
# another loaded library (e.g., libnanousdapi.so loaded by pxr_compat).
# Without this, the renderer's internal `nanousd_open` (and other
# statically-linked nanousd_* helpers) get rebound by ELF interposition
# to the API library's copies, which have separate static state, leading
# to the renderer parsing 0 prims when libnanousdapi is loaded first.
if(UNIX AND NOT APPLE)
    target_link_options(nusd_renderer PRIVATE "-Wl,-Bsymbolic-functions")
endif()

# -----------------------------------------------------------------------------
# Test executable
# -----------------------------------------------------------------------------

option(NUSD_BUILD_TESTS "Build and register renderer tests" ON)
if(NUSD_BUILD_TESTS)
    enable_testing()
endif()

add_executable(test_headless_render test/test_headless_render.c)
target_link_libraries(test_headless_render PRIVATE nusd_renderer)
set_target_properties(test_headless_render PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_executable(test_scene_curves test/test_scene_curves.c)
target_link_libraries(test_scene_curves PRIVATE nusd_renderer m)
target_include_directories(test_scene_curves PRIVATE src)

add_executable(test_scene_curve_basis test/test_scene_curve_basis.c)
target_link_libraries(test_scene_curve_basis PRIVATE nusd_renderer m)
target_include_directories(test_scene_curve_basis PRIVATE src)
set_target_properties(test_scene_curve_basis PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
if(NUSD_BUILD_TESTS)
    add_test(NAME test_scene_curve_basis
             COMMAND test_scene_curve_basis)
endif()

add_executable(test_scene_xform test/test_scene_xform.c)
target_link_libraries(test_scene_xform PRIVATE nusd_renderer m)
target_include_directories(test_scene_xform PRIVATE src)
set_target_properties(test_scene_xform PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
if(NUSD_BUILD_TESTS)
    add_test(NAME test_scene_xform
             COMMAND test_scene_xform
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/reset_xform_stack.usda
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/metrics_zup_centimeters.usda
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/face_varying_uv_seam.usda
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/point_instancer_basic.usda)
    set_tests_properties(test_scene_xform PROPERTIES
        ENVIRONMENT "NUSD_BAKE_ZUP_TO_YUP=1;NUSD_APPLY_METERS_PER_UNIT=1")
endif()

add_executable(test_native_instance_curves test/test_native_instance_curves.c)
target_link_libraries(test_native_instance_curves PRIVATE nusd_renderer m)
target_include_directories(test_native_instance_curves PRIVATE src)
set_target_properties(test_native_instance_curves PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
if(NUSD_BUILD_TESTS)
    add_test(NAME test_native_instance_curves
             COMMAND test_native_instance_curves
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/native_instance_curve_root.usda)
endif()

add_executable(test_ironwood_native_curves test/test_ironwood_native_curves.c)
target_link_libraries(test_ironwood_native_curves PRIVATE nusd_renderer m)
target_include_directories(test_ironwood_native_curves PRIVATE src)
set_target_properties(test_ironwood_native_curves PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
if(NUSD_BUILD_TESTS)
    add_test(NAME test_ironwood_native_curves
             COMMAND test_ironwood_native_curves
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/ironwood_compare_ref.usda
                     56
                     28853520
                     $HOME/moana-island-scene-usd/island/usd/elements/isIronwoodA1/element.usda
                     $HOME/moana-island-scene-usd/island/usd/elements/isIronwoodA1/xgenInstances/xgBonsai_curves.usd)
    set_tests_properties(test_ironwood_native_curves PROPERTIES
        SKIP_RETURN_CODE 77)
endif()

add_executable(test_moana_curve_assets test/test_moana_curve_assets.c)
target_link_libraries(test_moana_curve_assets PRIVATE nusd_renderer m)
target_include_directories(test_moana_curve_assets PRIVATE src)
set_target_properties(test_moana_curve_assets PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
if(NUSD_BUILD_TESTS)
    add_test(NAME test_moana_curve_assets
             COMMAND test_moana_curve_assets)
    set_tests_properties(test_moana_curve_assets PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 300)
endif()

add_executable(test_scene_instance_transform test/test_scene_instance_transform.c)
target_include_directories(test_scene_instance_transform PRIVATE src)
set_target_properties(test_scene_instance_transform PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
if(NUSD_BUILD_TESTS)
    add_test(NAME test_scene_instance_transform
             COMMAND test_scene_instance_transform)
endif()

# Skinning normal-matrix (inverse-transpose) math — pure math, no renderer link.
add_executable(test_normal_matrix test/test_normal_matrix.c)
target_link_libraries(test_normal_matrix PRIVATE m)
set_target_properties(test_normal_matrix PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
if(NUSD_BUILD_TESTS)
    add_test(NAME test_normal_matrix COMMAND test_normal_matrix)
endif()

# End-to-end skinned-normal path through the real scene loader.
add_executable(test_scene_skin test/test_scene_skin.c)
target_link_libraries(test_scene_skin PRIVATE nusd_renderer m)
target_include_directories(test_scene_skin PRIVATE src)
set_target_properties(test_scene_skin PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
if(NUSD_BUILD_TESTS)
    add_test(NAME test_scene_skin
             COMMAND test_scene_skin
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/skinned_nonuniform.usda)
endif()

# UV tiling on MaterialX authored directly in USD (task #15).
add_executable(test_mtlx_usd_uvtiling test/test_mtlx_usd_uvtiling.c)
target_link_libraries(test_mtlx_usd_uvtiling PRIVATE nusd_renderer nanousdapi m)
target_include_directories(test_mtlx_usd_uvtiling PRIVATE src)
set_target_properties(test_mtlx_usd_uvtiling PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
if(NUSD_BUILD_TESTS)
    add_test(NAME test_mtlx_usd_uvtiling
             COMMAND test_mtlx_usd_uvtiling
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/mtlx_usd_uvtiling.usda
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/assets)
endif()

# UsdGeomCapsule true-scale tessellation (Newton robot limbs).
add_executable(test_capsule test/test_capsule.c)
target_link_libraries(test_capsule PRIVATE nusd_renderer m)
target_include_directories(test_capsule PRIVATE src)
set_target_properties(test_capsule PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
if(NUSD_BUILD_TESTS)
    add_test(NAME test_capsule
             COMMAND test_capsule ${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/capsule_z.usda)
endif()

add_executable(test_gprim test/test_gprim.c)
target_link_libraries(test_gprim PRIVATE nusd_renderer m)
target_include_directories(test_gprim PRIVATE src)
set_target_properties(test_gprim PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
if(NUSD_BUILD_TESTS)
    add_test(NAME test_gprim
             COMMAND test_gprim ${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/gprim_cube_cone.usda)
endif()

add_executable(mdl_material_test test/mdl_material_test.c)
target_link_libraries(mdl_material_test PRIVATE nusd_renderer nanousdapi m)
target_include_directories(mdl_material_test PRIVATE src)
set_target_properties(mdl_material_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
if(NUSD_BUILD_TESTS)
    add_test(NAME mdl_material_test
             COMMAND mdl_material_test
                     ${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures/mdl_surface_outputs.usda
                     ${CMAKE_CURRENT_SOURCE_DIR}/test/materialx_proc_graph/scene.usda
                     ${CMAKE_CURRENT_SOURCE_DIR}/test/materialx_resource_graph/scene.usda
                     ${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures/usd_preview_interface_fallback.usda
                     ${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures/usd_preview_ptex_placeholder_fallback.usda)

    # Automated Vulkan-vs-OVRTX FLIP regression gate over the committed comparison
    # frames. Skips (77) when flip-evaluator/numpy/Pillow/scipy/scikit-image are absent
    # from the test interpreter (scipy + scikit-image are required — they power the gate's
    # GMSD structure and CIEDE2000 colour legs); else fails on any regression.
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME flip_vulkan_vs_ovrtx
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/comparisons/flip_pipeline.py --check)
        set_tests_properties(flip_vulkan_vs_ovrtx PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            SKIP_RETURN_CODE 77)
    endif()
endif()

# geo_cook — CPU-only geometry-cache pre-warm tool (no GPU/display).
add_executable(geo_cook test/geo_cook.c)
target_link_libraries(geo_cook PRIVATE nusd_renderer)
target_include_directories(geo_cook PRIVATE src)
set_target_properties(test_scene_curves PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# geo_cache_meshlet_test — CPU-only meshlet-cache round-trip gate (no GPU).
add_executable(geo_cache_meshlet_test test/geo_cache_meshlet_test.c)
target_link_libraries(geo_cache_meshlet_test PRIVATE nusd_renderer m)
target_include_directories(geo_cache_meshlet_test PRIVATE src)
set_target_properties(geo_cache_meshlet_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# nanousd loads its backend at runtime via dlopen('${binary_dir}/libnanousd.so').
# Symlink the canonical build into the renderer's build dir so test_scene_curves
# (and any other binary running from build/) can find the backend without an
# LD_LIBRARY_PATH dance.
if(WIN32)
    set(_NUSD_RUNTIME_DLLS)
    foreach(_dll ${NANOUSD_API_DLL} ${NANOUSD_BACKEND_DLL})
        if(_dll AND EXISTS "${_dll}")
            list(APPEND _NUSD_RUNTIME_DLLS "${_dll}")
        endif()
    endforeach()
    if(_NUSD_RUNTIME_DLLS)
        add_custom_command(TARGET nusd_renderer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${_NUSD_RUNTIME_DLLS} $<TARGET_FILE_DIR:nusd_renderer>
            COMMENT "Copy nanousd runtime DLLs into the renderer build directory")
    endif()
elseif(NANOUSD_SHARED AND EXISTS ${NANOUSD_SHARED})
    add_custom_command(TARGET nusd_renderer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${NANOUSD_SHARED} ${CMAKE_BINARY_DIR}/libnanousd.so
        COMMENT "Symlink libnanousd.so backend into build/ for test binaries")
endif()
