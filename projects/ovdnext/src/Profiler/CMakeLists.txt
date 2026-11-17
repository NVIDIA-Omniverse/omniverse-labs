# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_library(Profiler STATIC profiler.cpp)
target_include_directories(Profiler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
