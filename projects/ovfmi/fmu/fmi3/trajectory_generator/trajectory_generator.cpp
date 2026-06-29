// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// FMI 3.0 CoSimulation Trajectory Generator.
//
// Outputs a time-varying target position tracing a circular orbit:
//   target_x = center_x + radius * cos(omega * t + phase)
//   target_y = center_y (constant height)
//   target_z = center_z + radius * sin(omega * t + phase)
//
// Parameters: radius, height (center_y), omega (angular velocity), phase,
//             center_x, center_z.

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

struct TrajectoryGenerator {
    // Outputs (read by host after step)
    double target_x = 0.0; // VR 0
    double target_y = 2.0; // VR 1
    double target_z = 0.0; // VR 2

    // Parameters (tunable)
    double radius   = 1.5; // VR 3 — orbit radius
    double height   = 2.0; // VR 4 — orbit height (center_y)
    double omega    = 1.0; // VR 5 — angular velocity (rad/s)
    double phase    = 0.0; // VR 6 — initial phase offset (rad)
    double center_x = 0.0; // VR 7 — orbit center X
    double center_z = 0.0; // VR 8 — orbit center Z

    // Independent variable
    double time     = 0.0; // VR 9

    bool terminated = false;
};

static TrajectoryGenerator* as_tg(fmi3Instance instance) {
    return static_cast<TrajectoryGenerator*>(instance);
}

// ---------------------------------------------------------------------------
// FMI 3.0 lifecycle
// ---------------------------------------------------------------------------

FMI_EXPORT const char* fmi3GetVersion() {
    return "3.0";
}

FMI_EXPORT fmi3Status fmi3SetDebugLogging(
    fmi3Instance, bool, size_t, const char**) {
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
    return new (std::nothrow) TrajectoryGenerator();
}

FMI_EXPORT fmi3Instance fmi3InstantiateScheduledExecution(...) {
    return nullptr;
}

FMI_EXPORT fmi3Status fmi3EnterInitializationMode(
    fmi3Instance instance, bool, fmi3Float64, fmi3Float64, bool, fmi3Float64) {
    auto* tg = as_tg(instance);
    if (!tg) return fmi3Error;
    tg->terminated = false;
    return fmi3OK;
}

FMI_EXPORT fmi3Status fmi3ExitInitializationMode(fmi3Instance instance) {
    auto* tg = as_tg(instance);
    if (!tg) return fmi3Error;
    // Compute initial outputs
    double angle = tg->omega * tg->time + tg->phase;
    tg->target_x = tg->center_x + tg->radius * std::cos(angle);
    tg->target_y = tg->height;
    tg->target_z = tg->center_z + tg->radius * std::sin(angle);
    return fmi3OK;
}

FMI_EXPORT fmi3Status fmi3EnterEventMode(fmi3Instance instance) {
    return as_tg(instance) ? fmi3OK : fmi3Error;
}

FMI_EXPORT fmi3Status fmi3EnterStepMode(fmi3Instance instance) {
    return as_tg(instance) ? fmi3OK : fmi3Error;
}

FMI_EXPORT fmi3Status fmi3Terminate(fmi3Instance instance) {
    auto* tg = as_tg(instance);
    if (!tg) return fmi3Error;
    tg->terminated = true;
    return fmi3OK;
}

FMI_EXPORT fmi3Status fmi3Reset(fmi3Instance instance) {
    auto* tg = as_tg(instance);
    if (!tg) return fmi3Error;
    *tg = TrajectoryGenerator();
    return fmi3OK;
}

FMI_EXPORT void fmi3FreeInstance(fmi3Instance instance) {
    delete as_tg(instance);
}

// ---------------------------------------------------------------------------
// DoStep — compute circular orbit position
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
    auto* tg = as_tg(instance);
    if (!tg || communicationStepSize < 0.0) return fmi3Error;

    if (eventHandlingNeeded) *eventHandlingNeeded = false;
    if (terminateSimulation) *terminateSimulation = false;
    if (earlyReturn) *earlyReturn = false;

    tg->time = currentCommunicationPoint + communicationStepSize;

    double angle = tg->omega * tg->time + tg->phase;
    tg->target_x = tg->center_x + tg->radius * std::cos(angle);
    tg->target_y = tg->height;
    tg->target_z = tg->center_z + tg->radius * std::sin(angle);

    if (lastSuccessfulTime) *lastSuccessfulTime = tg->time;
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
    auto* tg = as_tg(instance);
    if (!tg || !valueReferences || !values || nValues < nValueReferences)
        return fmi3Error;

    for (size_t i = 0; i < nValueReferences; ++i) {
        switch (valueReferences[i]) {
        case 3: tg->radius   = values[i]; break;
        case 4: tg->height   = values[i]; break;
        case 5: tg->omega    = values[i]; break;
        case 6: tg->phase    = values[i]; break;
        case 7: tg->center_x = values[i]; break;
        case 8: tg->center_z = values[i]; break;
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
    auto* tg = as_tg(instance);
    if (!tg || !valueReferences || !values || nValues < nValueReferences)
        return fmi3Error;

    for (size_t i = 0; i < nValueReferences; ++i) {
        switch (valueReferences[i]) {
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
        default: return fmi3Error;
        }
    }
    return fmi3OK;
}

// ---------------------------------------------------------------------------
// Stubs
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
