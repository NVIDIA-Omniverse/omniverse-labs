# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set(PVDCORE_SOURCES
    PVDClient.cpp
    TcpSocket.cpp
    PvdEngine.cpp
    PvdInputValidation.cpp
    PvdCheckedOutput.cpp
    PvdRandomAccessIOStream.cpp
    PvdCommStreamMetaData.cpp
    pvdfileloader.cpp
    PvdCommStreamEventSerializer.cpp
    PvdCommStreamParsing.cpp
    PvdCommStreamParsingConverters.cpp
    PvdObjectModel.cpp
    PvdObjectModelEventStream.cpp
    PvdObjectModelEventStreamFactory.cpp
    PvdRepXExport.cpp
)

if(OVD_SUPPORT)
    list(APPEND PVDCORE_SOURCES OvdFileLoader.cpp OvdInputValidation.cpp OvdMessageCollector.cpp OvdCancelableSocketReadStream.cpp OvdCaptureControl.cpp OvdCaptureCatalog.cpp OvdDomAdapter.cpp OvdLiveSource.cpp)
endif()

add_library(PVDCore STATIC ${PVDCORE_SOURCES})
target_compile_features(PVDCore PUBLIC cxx_std_17)

target_include_directories(PVDCore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(PVDCore PUBLIC
    PxFoundation
    PxPvdSDK
    Profiler
)

if(OVD_SUPPORT)
    target_link_libraries(PVDCore PUBLIC PvdDom)
    target_compile_definitions(PVDCore PUBLIC PVD_OVD_SUPPORT)
endif()

# PVDCore removes UNICODE (uses MBCS on Windows)
if(WIN32)
    target_compile_definitions(PVDCore PRIVATE _MBCS)
    # Remove UNICODE if it's set by default
    get_target_property(_defs PVDCore COMPILE_DEFINITIONS)
    if(_defs)
        list(REMOVE_ITEM _defs UNICODE _UNICODE)
        set_target_properties(PVDCore PROPERTIES COMPILE_DEFINITIONS "${_defs}")
    endif()
    # The native TCP helper uses Winsock.
    target_link_libraries(PVDCore PRIVATE ws2_32)
    target_compile_definitions(PVDCore PRIVATE _WIN32_WINNT=0x0601)
endif()

if(UNIX)
    target_link_libraries(PVDCore PRIVATE pthread)
endif()
