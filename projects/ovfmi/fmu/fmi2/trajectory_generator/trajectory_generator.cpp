// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// FMI 2.0 CoSimulation Trajectory Generator.
//
// Same math as trajectory_generator_fmu.cpp (FMI 3.0) but using the FMI 2.0
// C API.  Created for SSP embedding where fmpy only supports FMI 1.0/2.0.
//
// Outputs a time-varying target position tracing a circular orbit:
//   target_x = center_x + radius * cos(omega * t + phase)
//   target_y = height (constant)
//   target_z = center_z + radius * sin(omega * t + phase)

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

#ifdef _WIN32
#define FMI_EXPORT extern "C" __declspec(dllexport)
#else
#define FMI_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// ---------------------------------------------------------------------------
// FMI 2.0 types
// ---------------------------------------------------------------------------

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

// FMI 2.0 type constant for co-simulation
static constexpr int fmi2CoSimulation = 1;

// Callback function types (we accept but ignore them)
struct fmi2CallbackFunctions {
    void* logger;
    void* allocateMemory;
    void* freeMemory;
    void* stepFinished;
    fmi2ComponentEnvironment componentEnvironment;
};

// ---------------------------------------------------------------------------
// State (same VR layout as FMI 3.0 version)
// ---------------------------------------------------------------------------

struct TrajectoryGenerator2 {
    // Outputs (read by host after step)
    double target_x = 0.0; // VR 0
    double target_y = 2.0; // VR 1
    double target_z = 0.0; // VR 2

    // Parameters (tunable)
    double radius   = 1.5; // VR 3
    double height   = 2.0; // VR 4
    double omega    = 1.0; // VR 5
    double phase    = 0.0; // VR 6
    double center_x = 0.0; // VR 7
    double center_z = 0.0; // VR 8

    // Independent variable
    double time     = 0.0; // VR 9
};

static TrajectoryGenerator2* as_tg(fmi2Component c) {
    return static_cast<TrajectoryGenerator2*>(c);
}

// ---------------------------------------------------------------------------
// FMI 2.0 inquiry
// ---------------------------------------------------------------------------

FMI_EXPORT const char* fmi2GetTypesPlatform() { return "default"; }
FMI_EXPORT const char* fmi2GetVersion() { return "2.0"; }

// ---------------------------------------------------------------------------
// FMI 2.0 lifecycle
// ---------------------------------------------------------------------------

FMI_EXPORT fmi2Component fmi2Instantiate(
    fmi2String /*instanceName*/,
    int /*fmuType*/,
    fmi2String /*guid*/,
    fmi2String /*resourceLocation*/,
    const fmi2CallbackFunctions* /*functions*/,
    fmi2Boolean /*visible*/,
    fmi2Boolean /*loggingOn*/) {
    return new (std::nothrow) TrajectoryGenerator2();
}

FMI_EXPORT void fmi2FreeInstance(fmi2Component c) {
    delete as_tg(c);
}

FMI_EXPORT fmi2Status fmi2SetDebugLogging(
    fmi2Component, fmi2Boolean, size_t, const fmi2String*) {
    return fmi2OK;
}

FMI_EXPORT fmi2Status fmi2SetupExperiment(
    fmi2Component c,
    fmi2Boolean /*toleranceDefined*/,
    fmi2Real /*tolerance*/,
    fmi2Real /*startTime*/,
    fmi2Boolean /*stopTimeDefined*/,
    fmi2Real /*stopTime*/) {
    return c ? fmi2OK : fmi2Error;
}

FMI_EXPORT fmi2Status fmi2EnterInitializationMode(fmi2Component c) {
    return c ? fmi2OK : fmi2Error;
}

FMI_EXPORT fmi2Status fmi2ExitInitializationMode(fmi2Component c) {
    auto* tg = as_tg(c);
    if (!tg) return fmi2Error;
    // Compute initial outputs
    double angle = tg->omega * tg->time + tg->phase;
    tg->target_x = tg->center_x + tg->radius * std::cos(angle);
    tg->target_y = tg->height;
    tg->target_z = tg->center_z + tg->radius * std::sin(angle);
    return fmi2OK;
}

FMI_EXPORT fmi2Status fmi2Terminate(fmi2Component c) {
    return c ? fmi2OK : fmi2Error;
}

FMI_EXPORT fmi2Status fmi2Reset(fmi2Component c) {
    auto* tg = as_tg(c);
    if (!tg) return fmi2Error;
    *tg = TrajectoryGenerator2();
    return fmi2OK;
}

// ---------------------------------------------------------------------------
// DoStep
// ---------------------------------------------------------------------------

FMI_EXPORT fmi2Status fmi2DoStep(
    fmi2Component c,
    fmi2Real currentCommunicationPoint,
    fmi2Real communicationStepSize,
    fmi2Boolean /*noSetFMUStatePriorToCurrentPoint*/) {
    auto* tg = as_tg(c);
    if (!tg || communicationStepSize < 0.0) return fmi2Error;

    tg->time = currentCommunicationPoint + communicationStepSize;
    double angle = tg->omega * tg->time + tg->phase;
    tg->target_x = tg->center_x + tg->radius * std::cos(angle);
    tg->target_y = tg->height;
    tg->target_z = tg->center_z + tg->radius * std::sin(angle);

    return fmi2OK;
}

// ---------------------------------------------------------------------------
// Get / Set Real
// ---------------------------------------------------------------------------

FMI_EXPORT fmi2Status fmi2SetReal(
    fmi2Component c,
    const fmi2ValueReference* vr,
    size_t nvr,
    const fmi2Real* values) {
    auto* tg = as_tg(c);
    if (!tg || !vr || !values) return fmi2Error;

    for (size_t i = 0; i < nvr; ++i) {
        switch (vr[i]) {
        case 3: tg->radius   = values[i]; break;
        case 4: tg->height   = values[i]; break;
        case 5: tg->omega    = values[i]; break;
        case 6: tg->phase    = values[i]; break;
        case 7: tg->center_x = values[i]; break;
        case 8: tg->center_z = values[i]; break;
        default: break;
        }
    }
    return fmi2OK;
}

FMI_EXPORT fmi2Status fmi2GetReal(
    fmi2Component c,
    const fmi2ValueReference* vr,
    size_t nvr,
    fmi2Real* values) {
    auto* tg = as_tg(c);
    if (!tg || !vr || !values) return fmi2Error;

    for (size_t i = 0; i < nvr; ++i) {
        switch (vr[i]) {
        case 0: values[i] = tg->target_x; break;
        case 1: values[i] = tg->target_y; break;
        case 2: values[i] = tg->target_z; break;
        case 3: values[i] = tg->radius;   break;
        case 4: values[i] = tg->height;   break;
        case 5: values[i] = tg->omega;    break;
        case 6: values[i] = tg->phase;    break;
        case 7: values[i] = tg->center_x; break;
        case 8: values[i] = tg->center_z; break;
        case 9: values[i] = tg->time;     break;
        default: return fmi2Error;
        }
    }
    return fmi2OK;
}

// ---------------------------------------------------------------------------
// Stubs
// ---------------------------------------------------------------------------

FMI_EXPORT fmi2Status fmi2GetInteger(fmi2Component, const fmi2ValueReference*, size_t, fmi2Integer*) { return fmi2Error; }
FMI_EXPORT fmi2Status fmi2SetInteger(fmi2Component, const fmi2ValueReference*, size_t, const fmi2Integer*) { return fmi2OK; }
FMI_EXPORT fmi2Status fmi2GetBoolean(fmi2Component, const fmi2ValueReference*, size_t, fmi2Boolean*) { return fmi2Error; }
FMI_EXPORT fmi2Status fmi2SetBoolean(fmi2Component, const fmi2ValueReference*, size_t, const fmi2Boolean*) { return fmi2OK; }
FMI_EXPORT fmi2Status fmi2GetString(fmi2Component, const fmi2ValueReference*, size_t, fmi2String*) { return fmi2Error; }
FMI_EXPORT fmi2Status fmi2SetString(fmi2Component, const fmi2ValueReference*, size_t, const fmi2String*) { return fmi2OK; }

FMI_EXPORT fmi2Status fmi2GetFMUstate(fmi2Component, void**) { return fmi2Error; }
FMI_EXPORT fmi2Status fmi2SetFMUstate(fmi2Component, void*) { return fmi2Error; }
FMI_EXPORT fmi2Status fmi2FreeFMUstate(fmi2Component, void**) { return fmi2OK; }
FMI_EXPORT fmi2Status fmi2SerializedFMUstateSize(fmi2Component, void*, size_t*) { return fmi2Error; }
FMI_EXPORT fmi2Status fmi2SerializeFMUstate(fmi2Component, void*, fmi2Byte*, size_t) { return fmi2Error; }
FMI_EXPORT fmi2Status fmi2DeSerializeFMUstate(fmi2Component, const fmi2Byte*, size_t, void**) { return fmi2Error; }
FMI_EXPORT fmi2Status fmi2GetDirectionalDerivative(fmi2Component, const fmi2ValueReference*, size_t, const fmi2ValueReference*, size_t, const fmi2Real*, fmi2Real*) { return fmi2Error; }
FMI_EXPORT fmi2Status fmi2SetRealInputDerivatives(fmi2Component, const fmi2ValueReference*, size_t, const fmi2Integer*, const fmi2Real*) { return fmi2Error; }
FMI_EXPORT fmi2Status fmi2GetRealOutputDerivatives(fmi2Component, const fmi2ValueReference*, size_t, const fmi2Integer*, fmi2Real*) { return fmi2Error; }
FMI_EXPORT fmi2Status fmi2CancelStep(fmi2Component) { return fmi2OK; }
FMI_EXPORT fmi2Status fmi2GetStatus(fmi2Component, int, fmi2Status*) { return fmi2Error; }
FMI_EXPORT fmi2Status fmi2GetRealStatus(fmi2Component, int, fmi2Real*) { return fmi2Error; }
FMI_EXPORT fmi2Status fmi2GetIntegerStatus(fmi2Component, int, fmi2Integer*) { return fmi2Error; }
FMI_EXPORT fmi2Status fmi2GetBooleanStatus(fmi2Component, int, fmi2Boolean*) { return fmi2Error; }
FMI_EXPORT fmi2Status fmi2GetStringStatus(fmi2Component, int, fmi2String*) { return fmi2Error; }
