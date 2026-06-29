// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// Minimal FMI 3.0 CoSimulation implementation for the bundled BouncingBall
// sample stage. The repository already contains a Linux binary for this FMU;
// this target supplies the matching Windows DLL during local builds.

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

struct BouncingBall {
    double time = 0.0;
    double h = 1.0;
    double v = 0.0;
    double g = -9.81;
    double e = 0.7;
    double v_min = 0.1;
    bool terminated = false;
};

static BouncingBall* as_ball(fmi3Instance instance) {
    return static_cast<BouncingBall*>(instance);
}

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
    const char* /*instanceName*/,
    const char* /*instantiationToken*/,
    const char* /*resourcePath*/,
    bool /*visible*/,
    bool /*loggingOn*/,
    fmi3InstanceEnvironment /*instanceEnvironment*/,
    fmi3LogMessageCallback /*logMessage*/) {
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
    return new (std::nothrow) BouncingBall();
}

FMI_EXPORT fmi3Instance fmi3InstantiateScheduledExecution(...) {
    return nullptr;
}

FMI_EXPORT fmi3Status fmi3EnterInitializationMode(
    fmi3Instance instance,
    bool /*toleranceDefined*/,
    fmi3Float64 /*tolerance*/,
    fmi3Float64 startTime,
    bool /*stopTimeDefined*/,
    fmi3Float64 /*stopTime*/) {
    auto* ball = as_ball(instance);
    if (!ball) return fmi3Error;
    ball->time = startTime;
    ball->h = 1.0;
    ball->v = 0.0;
    ball->terminated = false;
    return fmi3OK;
}

FMI_EXPORT fmi3Status fmi3ExitInitializationMode(fmi3Instance instance) {
    return as_ball(instance) ? fmi3OK : fmi3Error;
}

FMI_EXPORT fmi3Status fmi3EnterEventMode(fmi3Instance instance) {
    return as_ball(instance) ? fmi3OK : fmi3Error;
}

FMI_EXPORT fmi3Status fmi3EnterStepMode(fmi3Instance instance) {
    return as_ball(instance) ? fmi3OK : fmi3Error;
}

FMI_EXPORT fmi3Status fmi3Terminate(fmi3Instance instance) {
    auto* ball = as_ball(instance);
    if (!ball) return fmi3Error;
    ball->terminated = true;
    return fmi3OK;
}

FMI_EXPORT fmi3Status fmi3Reset(fmi3Instance instance) {
    auto* ball = as_ball(instance);
    if (!ball) return fmi3Error;
    *ball = BouncingBall();
    return fmi3OK;
}

FMI_EXPORT void fmi3FreeInstance(fmi3Instance instance) {
    delete as_ball(instance);
}

FMI_EXPORT fmi3Status fmi3DoStep(
    fmi3Instance instance,
    fmi3Float64 currentCommunicationPoint,
    fmi3Float64 communicationStepSize,
    bool /*noSetFMUStatePriorToCurrentPoint*/,
    bool* eventHandlingNeeded,
    bool* terminateSimulation,
    bool* earlyReturn,
    fmi3Float64* lastSuccessfulTime) {
    auto* ball = as_ball(instance);
    if (!ball || communicationStepSize < 0.0) return fmi3Error;

    if (eventHandlingNeeded) *eventHandlingNeeded = false;
    if (terminateSimulation) *terminateSimulation = false;
    if (earlyReturn) *earlyReturn = false;

    ball->time = currentCommunicationPoint;
    double target_time = currentCommunicationPoint + communicationStepSize;

    while (ball->time < target_time) {
        double dt = std::min(1.0e-3, target_time - ball->time);
        ball->v += ball->g * dt;
        ball->h += ball->v * dt;

        if (ball->h <= 0.0) {
            ball->h = 0.0;
            if (ball->v < 0.0) {
                ball->v = -ball->e * ball->v;
                if (std::abs(ball->v) < ball->v_min) ball->v = 0.0;
            }
        }
        ball->time += dt;
    }

    if (lastSuccessfulTime) *lastSuccessfulTime = ball->time;
    return fmi3OK;
}

FMI_EXPORT fmi3Status fmi3SetFloat64(
    fmi3Instance instance,
    const fmi3ValueReference* valueReferences,
    size_t nValueReferences,
    const fmi3Float64* values,
    size_t nValues) {
    auto* ball = as_ball(instance);
    if (!ball || !valueReferences || !values || nValues < nValueReferences) {
        return fmi3Error;
    }

    for (size_t i = 0; i < nValueReferences; ++i) {
        switch (valueReferences[i]) {
        case 1: ball->h = values[i]; break;
        case 3: ball->v = values[i]; break;
        case 5: ball->g = values[i]; break;
        case 6: ball->e = values[i]; break;
        case 7: ball->v_min = values[i]; break;
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
    auto* ball = as_ball(instance);
    if (!ball || !valueReferences || !values || nValues < nValueReferences) {
        return fmi3Error;
    }

    for (size_t i = 0; i < nValueReferences; ++i) {
        switch (valueReferences[i]) {
        case 0: values[i] = ball->time; break;
        case 1: values[i] = ball->h; break;
        case 2: values[i] = ball->v; break;
        case 3: values[i] = ball->v; break;
        case 4: values[i] = ball->g; break;
        case 5: values[i] = ball->g; break;
        case 6: values[i] = ball->e; break;
        case 7: values[i] = ball->v_min; break;
        default: return fmi3Error;
        }
    }
    return fmi3OK;
}

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
