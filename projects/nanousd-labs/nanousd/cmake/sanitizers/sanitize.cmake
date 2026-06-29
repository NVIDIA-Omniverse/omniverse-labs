# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

option(NANOUSD_ENABLE_SANITIZERS "Enable compiler sanitizers for dev/CI test builds" OFF)
set(NANOUSD_SANITIZERS "address;undefined" CACHE STRING
    "Semicolon-separated sanitizer list. Supported: address;undefined;thread")
set_property(CACHE NANOUSD_SANITIZERS PROPERTY STRINGS
    "address;undefined" "address" "undefined" "thread" "thread;undefined")

if(NANOUSD_ENABLE_SANITIZERS)
    message(STATUS
        "nanousd install rules disabled for sanitizer builds; "
        "sanitized builds are intended for CI/dev test runs only.")

    set(_nanousd_supported_sanitizers address undefined thread)
    set(_nanousd_sanitizers)
    foreach(_nanousd_sanitizer IN LISTS NANOUSD_SANITIZERS)
        string(STRIP "${_nanousd_sanitizer}" _nanousd_sanitizer)
        if(_nanousd_sanitizer STREQUAL "")
            continue()
        endif()
        if(NOT _nanousd_sanitizer IN_LIST _nanousd_supported_sanitizers)
            message(FATAL_ERROR
                "Unsupported sanitizer '${_nanousd_sanitizer}'. "
                "Supported sanitizers: address;undefined;thread")
        endif()
        list(APPEND _nanousd_sanitizers "${_nanousd_sanitizer}")
    endforeach()

    list(REMOVE_DUPLICATES _nanousd_sanitizers)
    if(NOT _nanousd_sanitizers)
        message(FATAL_ERROR
            "NANOUSD_ENABLE_SANITIZERS=ON requires at least one sanitizer "
            "in NANOUSD_SANITIZERS")
    endif()
    if("address" IN_LIST _nanousd_sanitizers AND "thread" IN_LIST _nanousd_sanitizers)
        message(FATAL_ERROR
            "AddressSanitizer and ThreadSanitizer are mutually exclusive. "
            "Choose address[;undefined] or thread[;undefined]; do not list "
            "address and thread together.")
    endif()
    if("thread" IN_LIST _nanousd_sanitizers
            AND "undefined" IN_LIST _nanousd_sanitizers
            AND (CMAKE_C_COMPILER_ID STREQUAL "GNU"
                OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
        message(FATAL_ERROR
            "ThreadSanitizer and UndefinedBehaviorSanitizer are not supported "
            "together with GCC. Use NANOUSD_SANITIZERS=thread for GCC "
            "ThreadSanitizer builds.")
    endif()
    list(JOIN _nanousd_sanitizers "," _nanousd_sanitize_arg)

    if((CMAKE_C_COMPILER_ID MATCHES "^(Clang|AppleClang|GNU)$")
            AND (CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|AppleClang|GNU)$")
            AND (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR APPLE)
            AND NOT WIN32)
        add_library(nanousd_sanitizers INTERFACE)
        target_compile_options(nanousd_sanitizers INTERFACE
            -fsanitize=${_nanousd_sanitize_arg}
            -fno-omit-frame-pointer
            -fno-sanitize-recover=all
        )
        target_link_options(nanousd_sanitizers INTERFACE
            -fsanitize=${_nanousd_sanitize_arg}
            -fno-sanitize-recover=all
        )
        target_compile_definitions(nanousd_sanitizers INTERFACE
            NANOUSD_SANITIZERS_ENABLED=1
        )
        message(STATUS "nanousd sanitizers enabled: ${_nanousd_sanitize_arg}")
    else()
        message(FATAL_ERROR
            "NANOUSD_ENABLE_SANITIZERS currently supports GCC/Clang-style "
            "sanitizers on Linux/macOS only. Windows support is intentionally "
            "left for a future MSVC/clang-cl-specific setup.")
    endif()

    set(_nanousd_ubsan_options "halt_on_error=1:print_stacktrace=1")
    if(DEFINED ENV{UBSAN_OPTIONS} AND NOT "$ENV{UBSAN_OPTIONS}" STREQUAL "")
        string(APPEND _nanousd_ubsan_options ":$ENV{UBSAN_OPTIONS}")
    endif()

    set(NANOUSD_SANITIZER_TEST_ENVIRONMENT
        "UBSAN_OPTIONS=${_nanousd_ubsan_options}")
    if("address" IN_LIST _nanousd_sanitizers)
        set(_nanousd_asan_options "halt_on_error=1:detect_leaks=1")
        if(DEFINED ENV{ASAN_OPTIONS} AND NOT "$ENV{ASAN_OPTIONS}" STREQUAL "")
            string(APPEND _nanousd_asan_options ":$ENV{ASAN_OPTIONS}")
        endif()

        set(_nanousd_lsan_options
            "suppressions=${CMAKE_CURRENT_LIST_DIR}/lsan.supp:print_suppressions=0")
        if(DEFINED ENV{LSAN_OPTIONS} AND NOT "$ENV{LSAN_OPTIONS}" STREQUAL "")
            string(APPEND _nanousd_lsan_options ":$ENV{LSAN_OPTIONS}")
        endif()

        list(APPEND NANOUSD_SANITIZER_TEST_ENVIRONMENT
            "ASAN_OPTIONS=${_nanousd_asan_options}"
            "LSAN_OPTIONS=${_nanousd_lsan_options}")
    endif()
    if("thread" IN_LIST _nanousd_sanitizers)
        set(_nanousd_tsan_options "halt_on_error=1")
        if(DEFINED ENV{TSAN_OPTIONS} AND NOT "$ENV{TSAN_OPTIONS}" STREQUAL "")
            string(APPEND _nanousd_tsan_options ":$ENV{TSAN_OPTIONS}")
        endif()

        list(APPEND NANOUSD_SANITIZER_TEST_ENVIRONMENT
            "TSAN_OPTIONS=${_nanousd_tsan_options}")
    endif()
endif()

function(nanousd_sanitize_target target)
    if(NANOUSD_ENABLE_SANITIZERS)
        if(NOT TARGET "${target}")
            message(FATAL_ERROR
                "nanousd_target_sanitize called for unknown target '${target}'")
        endif()
        target_link_libraries("${target}" PRIVATE nanousd_sanitizers)
    endif()
endfunction()

function(nanousd_sanitize_configure_tests)
    if(NANOUSD_ENABLE_SANITIZERS)
        set_tests_properties(${ARGV} PROPERTIES
            ENVIRONMENT "${NANOUSD_SANITIZER_TEST_ENVIRONMENT}")
    endif()
endfunction()
