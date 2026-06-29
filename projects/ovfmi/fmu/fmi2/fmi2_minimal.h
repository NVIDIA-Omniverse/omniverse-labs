// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// Minimal FMI 2.0 Co-Simulation type declarations for self-contained demo FMUs.

#pragma once

#include <cstddef>

#ifdef _WIN32
#define FMI_EXPORT extern "C" __declspec(dllexport)
#else
#define FMI_EXPORT extern "C" __attribute__((visibility("default")))
#endif

using fmi2Component            = void*;
using fmi2ComponentEnvironment = void*;
using fmi2Status               = int;
using fmi2ValueReference       = unsigned int;
using fmi2Real                 = double;
using fmi2Integer              = int;
using fmi2Boolean              = int;
using fmi2Char                 = char;
using fmi2String               = const fmi2Char*;
using fmi2Byte                 = char;

static constexpr fmi2Status fmi2OK    = 0;
static constexpr fmi2Status fmi2Error = 3;

struct fmi2CallbackFunctions {
    void* logger;
    void* allocateMemory;
    void* freeMemory;
    void* stepFinished;
    fmi2ComponentEnvironment componentEnvironment;
};

inline double clamp_real(double value, double lo, double hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}
