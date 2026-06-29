// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// FMI 2.0 CoSimulation roller motor drive.

#include "../fmi2_minimal.h"

#include <algorithm>
#include <cmath>
#include <new>

struct MotorDrive {
    double speed_command = 8.0;        // VR 0, input
    double enable = 1.0;               // VR 1, input
    double e_stop = 0.0;               // VR 2, input
    double target_velocity = 0.0;      // VR 3, output
    double actual_velocity = 0.0;      // VR 4, output
    double fault = 0.0;                // VR 5, output
    double max_speed = 20.0;           // VR 6, parameter
    double acceleration = 8.0;         // VR 7, parameter
    double deceleration = 12.0;        // VR 8, parameter
    double brake_deceleration = 50.0;  // VR 9, parameter
    double initial_velocity = 0.0;     // VR 10, parameter
    double time = 0.0;                 // VR 11, independent
};

static MotorDrive* as_drive(fmi2Component c) {
    return static_cast<MotorDrive*>(c);
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
    return new (std::nothrow) MotorDrive();
}

FMI_EXPORT void fmi2FreeInstance(fmi2Component c) {
    delete as_drive(c);
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
    auto* drive = as_drive(c);
    if (!drive) return fmi2Error;
    drive->actual_velocity = clamp_real(drive->initial_velocity, -drive->max_speed, drive->max_speed);
    drive->target_velocity = drive->actual_velocity;
    return fmi2OK;
}

FMI_EXPORT fmi2Status fmi2Terminate(fmi2Component c) {
    return c ? fmi2OK : fmi2Error;
}

FMI_EXPORT fmi2Status fmi2Reset(fmi2Component c) {
    auto* drive = as_drive(c);
    if (!drive) return fmi2Error;
    *drive = MotorDrive();
    return fmi2OK;
}

FMI_EXPORT fmi2Status fmi2DoStep(
    fmi2Component c,
    fmi2Real currentCommunicationPoint,
    fmi2Real communicationStepSize,
    fmi2Boolean) {
    auto* drive = as_drive(c);
    if (!drive || communicationStepSize < 0.0) return fmi2Error;

    const bool enabled = drive->enable > 0.5 && drive->e_stop <= 0.5;
    double command = enabled ? drive->speed_command : 0.0;
    command = clamp_real(command, -std::abs(drive->max_speed), std::abs(drive->max_speed));

    const double error = command - drive->actual_velocity;
    const bool braking = drive->e_stop > 0.5 || !enabled;
    const bool slowing = std::abs(command) < std::abs(drive->actual_velocity);
    double rate = braking ? drive->brake_deceleration
                          : (slowing ? drive->deceleration : drive->acceleration);
    rate = std::max(0.0, rate);

    const double max_delta = rate * communicationStepSize;
    drive->actual_velocity += clamp_real(error, -max_delta, max_delta);
    drive->target_velocity = drive->actual_velocity;
    drive->fault = 0.0;
    drive->time = currentCommunicationPoint + communicationStepSize;
    return fmi2OK;
}

FMI_EXPORT fmi2Status fmi2SetReal(
    fmi2Component c,
    const fmi2ValueReference* vr,
    size_t nvr,
    const fmi2Real* values) {
    auto* drive = as_drive(c);
    if (!drive || !vr || !values) return fmi2Error;
    for (size_t i = 0; i < nvr; ++i) {
        switch (vr[i]) {
        case 0: drive->speed_command = values[i]; break;
        case 1: drive->enable = values[i]; break;
        case 2: drive->e_stop = values[i]; break;
        case 6: drive->max_speed = values[i]; break;
        case 7: drive->acceleration = values[i]; break;
        case 8: drive->deceleration = values[i]; break;
        case 9: drive->brake_deceleration = values[i]; break;
        case 10: drive->initial_velocity = values[i]; break;
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
    auto* drive = as_drive(c);
    if (!drive || !vr || !values) return fmi2Error;
    for (size_t i = 0; i < nvr; ++i) {
        switch (vr[i]) {
        case 0: values[i] = drive->speed_command; break;
        case 1: values[i] = drive->enable; break;
        case 2: values[i] = drive->e_stop; break;
        case 3: values[i] = drive->target_velocity; break;
        case 4: values[i] = drive->actual_velocity; break;
        case 5: values[i] = drive->fault; break;
        case 6: values[i] = drive->max_speed; break;
        case 7: values[i] = drive->acceleration; break;
        case 8: values[i] = drive->deceleration; break;
        case 9: values[i] = drive->brake_deceleration; break;
        case 10: values[i] = drive->initial_velocity; break;
        case 11: values[i] = drive->time; break;
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
