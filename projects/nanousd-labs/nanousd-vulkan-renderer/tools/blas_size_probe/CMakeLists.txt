# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(blas_size_probe LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

add_executable(blas_size_probe main.c)
target_link_libraries(blas_size_probe PRIVATE Vulkan::Vulkan)

if(MSVC)
  target_compile_options(blas_size_probe PRIVATE /W3)
else()
  target_compile_options(blas_size_probe PRIVATE -Wall -Wextra)
endif()
