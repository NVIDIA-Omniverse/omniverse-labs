# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.22)
project(RockingCapture LANGUAGES CXX)
find_package(PhysX CONFIG REQUIRED)
find_package(Threads REQUIRED)
if(NOT CMAKE_BUILD_TYPE STREQUAL "checked")
  message(FATAL_ERROR "Use the checked SDK configuration; Release disables OmniPVD sampling")
endif()
if(NOT SHOWCASE_SDK_REVISION MATCHES "^[0-9a-f]+$" OR NOT SHOWCASE_SDK_REVISION STREQUAL "b97365c6ddb32fda8469bc788ec18024862b0368")
  message(FATAL_ERROR "Provide the exact documented CPU-control SDK revision")
endif()
get_target_property(sdk_library PhysX::PhysX IMPORTED_LOCATION_CHECKED)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS rocking_capture.cpp)
file(SHA256 "${sdk_library}" sdk_sha256)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/rocking_capture.cpp" client_sha256)
add_executable(rocking-capture rocking_capture.cpp)
target_compile_features(rocking-capture PRIVATE cxx_std_17)
target_compile_definitions(rocking-capture PRIVATE PX_PHYSX_STATIC_LIB PX_SUPPORT_OMNI_PVD=1
  SHOWCASE_SDK_REVISION="${SHOWCASE_SDK_REVISION}" SHOWCASE_SDK_SHA256="${sdk_sha256}"
  SHOWCASE_CLIENT_SHA256="${client_sha256}" SHOWCASE_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}")
target_link_libraries(rocking-capture PRIVATE PhysX::PhysXExtensions PhysX::PhysX PhysX::PVDRuntime Threads::Threads)
