# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG 6910c9d9165801d8827d628cb72eb7ea9dd538c5 # v1.16.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

if(OVD_SUPPORT)
    add_executable(ovd_capture_catalog_tests ovd_capture_catalog_tests.cpp)
    target_link_libraries(ovd_capture_catalog_tests PRIVATE PVDCore GTest::gtest_main)
    if(WIN32)
        target_compile_definitions(ovd_capture_catalog_tests PRIVATE _WIN32_WINNT=0x0601)
    endif()
    gtest_discover_tests(ovd_capture_catalog_tests DISCOVERY_MODE PRE_TEST
        PROPERTIES RUN_SERIAL TRUE TIMEOUT 180)
    add_executable(ovd_capture_control_tests ovd_capture_control_tests.cpp tcp_socket_tests.cpp)
    target_link_libraries(ovd_capture_control_tests PRIVATE PVDCore GTest::gtest_main)
    target_compile_definitions(ovd_capture_control_tests PRIVATE
        TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
    if(WIN32)
        target_compile_definitions(ovd_capture_control_tests PRIVATE _WIN32_WINNT=0x0601)
    endif()
    gtest_discover_tests(ovd_capture_control_tests DISCOVERY_MODE PRE_TEST
        PROPERTIES RUN_SERIAL TRUE TIMEOUT 60)
endif()

add_executable(pvdcore_tests main.cpp)
target_compile_features(pvdcore_tests PRIVATE cxx_std_17)
target_include_directories(pvdcore_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pvdcore_tests PRIVATE
    PVDCore PxFoundation PxPvdSDK Profiler GTest::gtest GTest::gmock)
target_compile_definitions(pvdcore_tests PRIVATE
    PX_FOUNDATION_DLL=0 PX_PVDSDK_DLL=0 PX_SUPPORT_PVD=1
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
if(WIN32)
    target_link_libraries(pvdcore_tests PRIVATE ws2_32)
    target_compile_definitions(pvdcore_tests PRIVATE _WIN32_WINNT=0x0601)
endif()
gtest_discover_tests(pvdcore_tests
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DISCOVERY_MODE PRE_TEST
    PROPERTIES RUN_SERIAL TRUE TIMEOUT 30)

add_executable(checked_output_tests checked_output_main.cpp)
target_link_libraries(checked_output_tests PRIVATE PVDCore GTest::gtest)
gtest_discover_tests(checked_output_tests
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DISCOVERY_MODE PRE_TEST
    PROPERTIES RUN_SERIAL TRUE TIMEOUT 30)
