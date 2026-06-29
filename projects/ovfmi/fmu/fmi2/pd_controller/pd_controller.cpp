// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// FMI 2.0 CoSimulation PD Controller.
//
// Same math as pd_controller_fmu.cpp (FMI 3.0) but using the FMI 2.0 C API.
// Created for SSP embedding where fmpy only supports FMI 1.0/2.0.
//
// Reads body position and velocity, computes PD restoring force towards a
// configurable target with gravity compensation.

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

static constexpr int fmi2CoSimulation = 1;

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

struct PDController2 {
    // Inputs (set by host each step)
    double pos_x = 0.0;   // VR 0
    double pos_y = 0.0;   // VR 1
    double pos_z = 0.0;   // VR 2
    double vel_x = 0.0;   // VR 3
    double vel_y = 0.0;   // VR 4
    double vel_z = 0.0;   // VR 5

    // Outputs (read by host after step)
    double force_x = 0.0; // VR 6
    double force_y = 0.0; // VR 7
    double force_z = 0.0; // VR 8

    // Parameters (tunable)
    double target_x = 0.0;   // VR 9
    double target_y = 2.0;   // VR 10
    double target_z = 0.0;   // VR 11
    double kp       = 50.0;  // VR 12
    double kd       = 10.0;  // VR 13
    double mass     = 1.0;   // VR 14
    double gravity   = -9.81; // VR 15
    double time      = 0.0;   // VR 16
};

static PDController2* as_pd(fmi2Component c) {
    return static_cast<PDController2*>(c);
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
    return new (std::nothrow) PDController2();
}

FMI_EXPORT void fmi2FreeInstance(fmi2Component c) {
    delete as_pd(c);
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
    return c ? fmi2OK : fmi2Error;
}

FMI_EXPORT fmi2Status fmi2Terminate(fmi2Component c) {
    return c ? fmi2OK : fmi2Error;
}

FMI_EXPORT fmi2Status fmi2Reset(fmi2Component c) {
    auto* pd = as_pd(c);
    if (!pd) return fmi2Error;
    *pd = PDController2();
    return fmi2OK;
}

// ---------------------------------------------------------------------------
// DoStep — PD controller with gravity compensation
// ---------------------------------------------------------------------------

FMI_EXPORT fmi2Status fmi2DoStep(
    fmi2Component c,
    fmi2Real currentCommunicationPoint,
    fmi2Real communicationStepSize,
    fmi2Boolean /*noSetFMUStatePriorToCurrentPoint*/) {
    auto* pd = as_pd(c);
    if (!pd || communicationStepSize < 0.0) return fmi2Error;

    // PD control: force = kp * (target - pos) + kd * (0 - vel)
    double err_x = pd->target_x - pd->pos_x;
    double err_y = pd->target_y - pd->pos_y;
    double err_z = pd->target_z - pd->pos_z;

    pd->force_x = pd->kp * err_x + pd->kd * (0.0 - pd->vel_x);
    pd->force_y = pd->kp * err_y + pd->kd * (0.0 - pd->vel_y);
    pd->force_z = pd->kp * err_z + pd->kd * (0.0 - pd->vel_z);

    // Gravity compensation
    pd->force_y += pd->mass * (-pd->gravity);

    pd->time = currentCommunicationPoint + communicationStepSize;

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
    auto* pd = as_pd(c);
    if (!pd || !vr || !values) return fmi2Error;

    for (size_t i = 0; i < nvr; ++i) {
        switch (vr[i]) {
        case  0: pd->pos_x    = values[i]; break;
        case  1: pd->pos_y    = values[i]; break;
        case  2: pd->pos_z    = values[i]; break;
        case  3: pd->vel_x    = values[i]; break;
        case  4: pd->vel_y    = values[i]; break;
        case  5: pd->vel_z    = values[i]; break;
        case  9: pd->target_x = values[i]; break;
        case 10: pd->target_y = values[i]; break;
        case 11: pd->target_z = values[i]; break;
        case 12: pd->kp       = values[i]; break;
        case 13: pd->kd       = values[i]; break;
        case 14: pd->mass     = values[i]; break;
        case 15: pd->gravity  = values[i]; break;
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
    auto* pd = as_pd(c);
    if (!pd || !vr || !values) return fmi2Error;

    for (size_t i = 0; i < nvr; ++i) {
        switch (vr[i]) {
        case  0: values[i] = pd->pos_x;    break;
        case  1: values[i] = pd->pos_y;    break;
        case  2: values[i] = pd->pos_z;    break;
        case  3: values[i] = pd->vel_x;    break;
        case  4: values[i] = pd->vel_y;    break;
        case  5: values[i] = pd->vel_z;    break;
        case  6: values[i] = pd->force_x;  break;
        case  7: values[i] = pd->force_y;  break;
        case  8: values[i] = pd->force_z;  break;
        case  9: values[i] = pd->target_x; break;
        case 10: values[i] = pd->target_y; break;
        case 11: values[i] = pd->target_z; break;
        case 12: values[i] = pd->kp;       break;
        case 13: values[i] = pd->kd;       break;
        case 14: values[i] = pd->mass;     break;
        case 15: values[i] = pd->gravity;  break;
        case 16: values[i] = pd->time;     break;
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
