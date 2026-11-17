/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef OVD_LOG_H
#define OVD_LOG_H

#include <cstdio>
#include <cstdarg>

// Set to 1 to enable debug logging to C:/temp/ovd_debug.log
#define OVD_LOG_ENABLED 0

inline void ovdLog(const char* fmt, ...)
{
#if OVD_LOG_ENABLED
    FILE* f = fopen("C:/temp/ovd_debug.log", "a");
    if (!f) return;
    va_list args;
    va_start(args, fmt);
    vfprintf(f, fmt, args);
    va_end(args);
    fflush(f);
    fclose(f);
#else
    (void)fmt;
#endif
}

#endif
