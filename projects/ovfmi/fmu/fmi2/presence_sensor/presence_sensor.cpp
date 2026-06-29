// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// FMI 2.0 CoSimulation presence sensor conditioner.

#include "../fmi2_minimal.h"

#include <new>

struct PresenceSensor {
    double raw_presence = 0.0;  // VR 0, input
    double presence = 0.0;      // VR 2, output
    double rising_edge = 0.0;   // VR 3, output
    double falling_edge = 0.0;  // VR 4, output
    double dwell_time = 0.0;    // VR 5, output
    double on_threshold = 0.5;  // VR 7, parameter
    double off_threshold = 0.5; // VR 8, parameter
    double time = 0.0;          // VR 9, independent
};

static PresenceSensor* as_sensor(fmi2Component c) {
    return static_cast<PresenceSensor*>(c);
}

FMI_EXPORT const char* fmi2GetTypesPlatform() { return "default"; }
FMI_EXPORT const char* fmi2GetVersion() { return "2.0"; }

FMI_EXPORT fmi2Component fmi2Instantiate(
    fmi2String,
    int,
    fmi2String,
    fmi2String,
    const fmi2CallbackFunctions*,
    fmi2Boolean,
    fmi2Boolean) {
    return new (std::nothrow) PresenceSensor();
}

FMI_EXPORT void fmi2FreeInstance(fmi2Component c) {
    delete as_sensor(c);
}

FMI_EXPORT fmi2Status fmi2SetDebugLogging(fmi2Component, fmi2Boolean, size_t, const fmi2String*) {
    return fmi2OK;
}

FMI_EXPORT fmi2Status fmi2SetupExperiment(fmi2Component c, fmi2Boolean, fmi2Real, fmi2Real, fmi2Boolean, fmi2Real) {
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
    auto* sensor = as_sensor(c);
    if (!sensor) return fmi2Error;
    *sensor = PresenceSensor();
    return fmi2OK;
}

FMI_EXPORT fmi2Status fmi2DoStep(
    fmi2Component c,
    fmi2Real currentCommunicationPoint,
    fmi2Real communicationStepSize,
    fmi2Boolean) {
    auto* sensor = as_sensor(c);
    if (!sensor || communicationStepSize < 0.0) return fmi2Error;

    const double raw = sensor->raw_presence > 0.5 ? 1.0 : 0.0;
    const double previous = sensor->presence;
    const double threshold = previous > 0.5 ? sensor->off_threshold : sensor->on_threshold;
    sensor->presence = raw >= threshold ? 1.0 : 0.0;
    sensor->rising_edge = (sensor->presence > 0.5 && previous <= 0.5) ? 1.0 : 0.0;
    sensor->falling_edge = (sensor->presence <= 0.5 && previous > 0.5) ? 1.0 : 0.0;
    sensor->dwell_time = sensor->presence > 0.5
                             ? sensor->dwell_time + communicationStepSize
                             : 0.0;
    sensor->time = currentCommunicationPoint + communicationStepSize;
    return fmi2OK;
}

FMI_EXPORT fmi2Status fmi2SetReal(
    fmi2Component c,
    const fmi2ValueReference* vr,
    size_t nvr,
    const fmi2Real* values) {
    auto* sensor = as_sensor(c);
    if (!sensor || !vr || !values) return fmi2Error;
    for (size_t i = 0; i < nvr; ++i) {
        switch (vr[i]) {
        case 0: sensor->raw_presence = values[i]; break;
        case 7: sensor->on_threshold = values[i]; break;
        case 8: sensor->off_threshold = values[i]; break;
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
    auto* sensor = as_sensor(c);
    if (!sensor || !vr || !values) return fmi2Error;
    for (size_t i = 0; i < nvr; ++i) {
        switch (vr[i]) {
        case 0: values[i] = sensor->raw_presence; break;
        case 2: values[i] = sensor->presence; break;
        case 3: values[i] = sensor->rising_edge; break;
        case 4: values[i] = sensor->falling_edge; break;
        case 5: values[i] = sensor->dwell_time; break;
        case 7: values[i] = sensor->on_threshold; break;
        case 8: values[i] = sensor->off_threshold; break;
        case 9: values[i] = sensor->time; break;
        default: return fmi2Error;
        }
    }
    return fmi2OK;
}

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
