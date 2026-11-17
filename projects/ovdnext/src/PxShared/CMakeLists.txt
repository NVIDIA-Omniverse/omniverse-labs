# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ===========================================================================
# PxFoundation - PhysX platform abstraction static library
# ===========================================================================
add_library(PxFoundation STATIC
    # Common sources
    src/foundation/src/PsAllocator.cpp
    src/foundation/src/PsAssert.cpp
    src/foundation/src/PsFoundation.cpp
    src/foundation/src/PsMathUtils.cpp
    src/foundation/src/PsString.cpp
    src/foundation/src/PsTempAllocator.cpp
    src/foundation/src/PsUtilities.cpp
)

if(WIN32)
    target_sources(PxFoundation PRIVATE
        src/foundation/src/windows/PsWindowsAtomic.cpp
        src/foundation/src/windows/PsWindowsCpu.cpp
        src/foundation/src/windows/PsWindowsFPU.cpp
        src/foundation/src/windows/PsWindowsMutex.cpp
        src/foundation/src/windows/PsWindowsPrintString.cpp
        src/foundation/src/windows/PsWindowsSList.cpp
        src/foundation/src/windows/PsWindowsSocket.cpp
        src/foundation/src/windows/PsWindowsSync.cpp
        src/foundation/src/windows/PsWindowsThread.cpp
        src/foundation/src/windows/PsWindowsTime.cpp
    )
elseif(UNIX)
    # POSIX implementations of Mutex/Thread/Time/SList/etc. for macOS + Linux.
    target_sources(PxFoundation PRIVATE
        src/foundation/src/unix/PsUnixAtomic.cpp
        src/foundation/src/unix/PsUnixCpu.cpp
        src/foundation/src/unix/PsUnixFPU.cpp
        src/foundation/src/unix/PsUnixMutex.cpp
        src/foundation/src/unix/PsUnixPrintString.cpp
        src/foundation/src/unix/PsUnixSList.cpp
        src/foundation/src/unix/PsUnixSocket.cpp
        src/foundation/src/unix/PsUnixSync.cpp
        src/foundation/src/unix/PsUnixThread.cpp
        src/foundation/src/unix/PsUnixTime.cpp
    )
endif()

target_include_directories(PxFoundation PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/foundation
    ${CMAKE_CURRENT_SOURCE_DIR}/src/foundation/include
)

if(WIN32)
    target_include_directories(PxFoundation PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/foundation/include/windows
    )
elseif(UNIX)
    target_include_directories(PxFoundation PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/foundation/include/unix
    )
endif()

target_compile_definitions(PxFoundation PUBLIC
    PX_FOUNDATION_DLL=0
    PX_PVDSDK_DLL=0
    PX_SUPPORT_PVD=1
    PX_ENABLE_ASSERTS
)

target_compile_definitions(PxFoundation PUBLIC
    $<$<CONFIG:Debug>:_DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

# ===========================================================================
# PxPvdSDK - PhysX Visual Debugger protocol static library
# ===========================================================================
add_library(PxPvdSDK STATIC
    src/pvd/src/PxProfileEventImpl.cpp
    src/pvd/src/PxPvd.cpp
    src/pvd/src/PxPvdDataStream.cpp
    src/pvd/src/PxPvdDefaultFileTransport.cpp
    src/pvd/src/PxPvdDefaultSocketTransport.cpp
    src/pvd/src/PxPvdImpl.cpp
    src/pvd/src/PxPvdMemClient.cpp
    src/pvd/src/PxPvdObjectModelMetaData.cpp
    src/pvd/src/PxPvdObjectRegistrar.cpp
    src/pvd/src/PxPvdProfileZoneClient.cpp
    src/pvd/src/PxPvdUserRenderer.cpp
)

target_include_directories(PxPvdSDK PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pvd/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pvd/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/filebuf/include
)

target_link_libraries(PxPvdSDK PUBLIC PxFoundation)
