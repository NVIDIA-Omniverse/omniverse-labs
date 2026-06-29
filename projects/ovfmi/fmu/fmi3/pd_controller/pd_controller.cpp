// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// FMI 3.0 CoSimulation PD (proportional-derivative) controller.
//
// Reads body position and velocity each step, computes a restoring force
// towards a configurable target with gravity compensation, and outputs
// force_x / force_y / force_z for the host to inject into a physics solver.
//
// Modelled after bouncing_ball_fmu.cpp for structure and FMI lifecycle.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>

#ifdef _WIN32
#define FMI_EXPORT extern "C" __declspec(dllexport)
#else
#define FMI_EXPORT extern "C" __attribute__((visibility("default")))
#endif

using fmi3Instance = void*;
using fmi3InstanceEnvironment = void*;
using fmi3Status = int;
using fmi3ValueReference = uint32_t;
using fmi3Float64 = double;

static constexpr fmi3Status fmi3OK = 0;
static constexpr fmi3Status fmi3Error = 3;

using fmi3LogMessageCallback =
    void (*)(fmi3InstanceEnvironment, fmi3Instance, fmi3Status, const char*,
             const char*);
using fmi3IntermediateUpdateCallback =
    void (*)(fmi3InstanceEnvironment, fmi3Float64, bool, bool, bool, bool, bool,
             bool*, fmi3Float64*);

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

struct PDController {
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
    double target_y = 2.0;   // VR 10 — levitate at 2 m
    double target_z = 0.0;   // VR 11
    double kp       = 50.0;  // VR 12 — proportional gain
    double kd       = 10.0;  // VR 13 — derivative gain
    double mass     = 1.0;   // VR 14 — body mass for gravity compensation
    double gravity   = -9.81; // VR 15
    double time      = 0.0;    // VR 16 — independent variable

    bool terminated = false;
};

static PDController* as_pd(fmi3Instance instance) {
    return static_cast<PDController*>(instance);
}

// ---------------------------------------------------------------------------
// FMI 3.0 lifecycle
// ---------------------------------------------------------------------------

FMI_EXPORT const char* fmi3GetVersion() {
    return "3.0";
}

FMI_EXPORT fmi3Status fmi3SetDebugLogging(
    fmi3Instance /*instance*/,
    bool /*loggingOn*/,
    size_t /*nCategories*/,
    const char** /*categories*/) {
    return fmi3OK;
}

FMI_EXPORT fmi3Instance fmi3InstantiateModelExchange(
    const char*, const char*, const char*, bool, bool,
    fmi3InstanceEnvironment, fmi3LogMessageCallback) {
    return nullptr;
}

FMI_EXPORT fmi3Instance fmi3InstantiateCoSimulation(
    const char* /*instanceName*/,
    const char* /*instantiationToken*/,
    const char* /*resourcePath*/,
    bool /*visible*/,
    bool /*loggingOn*/,
    bool /*eventModeUsed*/,
    bool /*earlyReturnAllowed*/,
    const fmi3ValueReference* /*requiredIntermediateVariables*/,
    size_t /*nRequiredIntermediateVariables*/,
    fmi3InstanceEnvironment /*instanceEnvironment*/,
    fmi3LogMessageCallback /*logMessage*/,
    fmi3IntermediateUpdateCallback /*intermediateUpdate*/) {
    return new (std::nothrow) PDController();
}

FMI_EXPORT fmi3Instance fmi3InstantiateScheduledExecution(...) {
    return nullptr;
}

FMI_EXPORT fmi3Status fmi3EnterInitializationMode(
    fmi3Instance instance,
    bool /*toleranceDefined*/,
    fmi3Float64 /*tolerance*/,
    fmi3Float64 /*startTime*/,
    bool /*stopTimeDefined*/,
    fmi3Float64 /*stopTime*/) {
    auto* pd = as_pd(instance);
    if (!pd) return fmi3Error;
    pd->terminated = false;
    return fmi3OK;
}

FMI_EXPORT fmi3Status fmi3ExitInitializationMode(fmi3Instance instance) {
    return as_pd(instance) ? fmi3OK : fmi3Error;
}

FMI_EXPORT fmi3Status fmi3EnterEventMode(fmi3Instance instance) {
    return as_pd(instance) ? fmi3OK : fmi3Error;
}

FMI_EXPORT fmi3Status fmi3EnterStepMode(fmi3Instance instance) {
    return as_pd(instance) ? fmi3OK : fmi3Error;
}

FMI_EXPORT fmi3Status fmi3Terminate(fmi3Instance instance) {
    auto* pd = as_pd(instance);
    if (!pd) return fmi3Error;
    pd->terminated = true;
    return fmi3OK;
}

FMI_EXPORT fmi3Status fmi3Reset(fmi3Instance instance) {
    auto* pd = as_pd(instance);
    if (!pd) return fmi3Error;
    *pd = PDController();
    return fmi3OK;
}

FMI_EXPORT void fmi3FreeInstance(fmi3Instance instance) {
    delete as_pd(instance);
}

// ---------------------------------------------------------------------------
// DoStep — PD controller with gravity compensation
// ---------------------------------------------------------------------------

FMI_EXPORT fmi3Status fmi3DoStep(
    fmi3Instance instance,
    fmi3Float64 currentCommunicationPoint,
    fmi3Float64 communicationStepSize,
    bool /*noSetFMUStatePriorToCurrentPoint*/,
    bool* eventHandlingNeeded,
    bool* terminateSimulation,
    bool* earlyReturn,
    fmi3Float64* lastSuccessfulTime) {
    auto* pd = as_pd(instance);
    if (!pd || communicationStepSize < 0.0) return fmi3Error;

    if (eventHandlingNeeded) *eventHandlingNeeded = false;
    if (terminateSimulation) *terminateSimulation = false;
    if (earlyReturn) *earlyReturn = false;

    // PD control: force = kp * (target - pos) + kd * (0 - vel)
    double err_x = pd->target_x - pd->pos_x;
    double err_y = pd->target_y - pd->pos_y;
    double err_z = pd->target_z - pd->pos_z;

    pd->force_x = pd->kp * err_x + pd->kd * (0.0 - pd->vel_x);
    pd->force_y = pd->kp * err_y + pd->kd * (0.0 - pd->vel_y);
    pd->force_z = pd->kp * err_z + pd->kd * (0.0 - pd->vel_z);

    // Gravity compensation: add upward force to counteract gravity
    pd->force_y += pd->mass * (-pd->gravity);

    pd->time = currentCommunicationPoint + communicationStepSize;

    if (lastSuccessfulTime) *lastSuccessfulTime = pd->time;
    return fmi3OK;
}

// ---------------------------------------------------------------------------
// Get / Set
// ---------------------------------------------------------------------------

FMI_EXPORT fmi3Status fmi3SetFloat64(
    fmi3Instance instance,
    const fmi3ValueReference* valueReferences,
    size_t nValueReferences,
    const fmi3Float64* values,
    size_t nValues) {
    auto* pd = as_pd(instance);
    if (!pd || !valueReferences || !values || nValues < nValueReferences) {
        return fmi3Error;
    }

    for (size_t i = 0; i < nValueReferences; ++i) {
        switch (valueReferences[i]) {
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
    return fmi3OK;
}

FMI_EXPORT fmi3Status fmi3GetFloat64(
    fmi3Instance instance,
    const fmi3ValueReference* valueReferences,
    size_t nValueReferences,
    fmi3Float64* values,
    size_t nValues) {
    auto* pd = as_pd(instance);
    if (!pd || !valueReferences || !values || nValues < nValueReferences) {
        return fmi3Error;
    }

    for (size_t i = 0; i < nValueReferences; ++i) {
        switch (valueReferences[i]) {
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
        default: return fmi3Error;
        }
    }
    return fmi3OK;
}

// ---------------------------------------------------------------------------
// Stubs for unused API functions (same pattern as bouncing_ball_fmu.cpp)
// ---------------------------------------------------------------------------

#define FMI_OK_STUB(name) FMI_EXPORT fmi3Status name(...) { return fmi3OK; }
#define FMI_ERROR_STUB(name) FMI_EXPORT fmi3Status name(...) { return fmi3Error; }

FMI_ERROR_STUB(fmi3GetFloat32)
FMI_OK_STUB(fmi3SetFloat32)
FMI_ERROR_STUB(fmi3GetInt8)
FMI_OK_STUB(fmi3SetInt8)
FMI_ERROR_STUB(fmi3GetUInt8)
FMI_OK_STUB(fmi3SetUInt8)
FMI_ERROR_STUB(fmi3GetInt16)
FMI_OK_STUB(fmi3SetInt16)
FMI_ERROR_STUB(fmi3GetUInt16)
FMI_OK_STUB(fmi3SetUInt16)
FMI_ERROR_STUB(fmi3GetInt32)
FMI_OK_STUB(fmi3SetInt32)
FMI_ERROR_STUB(fmi3GetUInt32)
FMI_OK_STUB(fmi3SetUInt32)
FMI_ERROR_STUB(fmi3GetInt64)
FMI_OK_STUB(fmi3SetInt64)
FMI_ERROR_STUB(fmi3GetUInt64)
FMI_OK_STUB(fmi3SetUInt64)
FMI_ERROR_STUB(fmi3GetBoolean)
FMI_OK_STUB(fmi3SetBoolean)
FMI_ERROR_STUB(fmi3GetString)
FMI_OK_STUB(fmi3SetString)
FMI_ERROR_STUB(fmi3GetBinary)
FMI_OK_STUB(fmi3SetBinary)
FMI_ERROR_STUB(fmi3GetClock)
FMI_OK_STUB(fmi3SetClock)
FMI_ERROR_STUB(fmi3GetNumberOfVariableDependencies)
FMI_ERROR_STUB(fmi3GetVariableDependencies)
FMI_ERROR_STUB(fmi3GetFMUState)
FMI_ERROR_STUB(fmi3SetFMUState)
FMI_OK_STUB(fmi3FreeFMUState)
FMI_ERROR_STUB(fmi3SerializedFMUStateSize)
FMI_ERROR_STUB(fmi3SerializeFMUState)
FMI_ERROR_STUB(fmi3DeserializeFMUState)
FMI_ERROR_STUB(fmi3GetDirectionalDerivative)
FMI_ERROR_STUB(fmi3GetAdjointDerivative)
FMI_OK_STUB(fmi3EnterConfigurationMode)
FMI_OK_STUB(fmi3ExitConfigurationMode)
FMI_ERROR_STUB(fmi3GetIntervalDecimal)
FMI_ERROR_STUB(fmi3GetIntervalFraction)
FMI_ERROR_STUB(fmi3GetShiftDecimal)
FMI_ERROR_STUB(fmi3GetShiftFraction)
FMI_OK_STUB(fmi3SetIntervalDecimal)
FMI_OK_STUB(fmi3SetIntervalFraction)
FMI_OK_STUB(fmi3SetShiftDecimal)
FMI_OK_STUB(fmi3SetShiftFraction)
FMI_OK_STUB(fmi3EvaluateDiscreteStates)
FMI_OK_STUB(fmi3UpdateDiscreteStates)
FMI_OK_STUB(fmi3EnterContinuousTimeMode)
FMI_OK_STUB(fmi3CompletedIntegratorStep)
FMI_OK_STUB(fmi3SetTime)
FMI_OK_STUB(fmi3SetContinuousStates)
FMI_ERROR_STUB(fmi3GetContinuousStateDerivatives)
FMI_ERROR_STUB(fmi3GetEventIndicators)
FMI_ERROR_STUB(fmi3GetContinuousStates)
FMI_ERROR_STUB(fmi3GetNominalsOfContinuousStates)
FMI_ERROR_STUB(fmi3GetNumberOfEventIndicators)
FMI_ERROR_STUB(fmi3GetNumberOfContinuousStates)
FMI_ERROR_STUB(fmi3GetOutputDerivatives)
FMI_ERROR_STUB(fmi3ActivateModelPartition)

#undef FMI_OK_STUB
#undef FMI_ERROR_STUB
