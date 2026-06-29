// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// FMI 2.0 CoSimulation conveyor line controller.

#include "../fmi2_minimal.h"

#include <algorithm>
#include <cmath>
#include <new>

struct ConveyorController {
    double operator_speed = 8.0;   // VR 0, input
    double enable = 1.0;           // VR 1, input
    double e_stop = 0.0;           // VR 2, input
    double sensor_presence = 0.0;  // VR 3, input
    double reject_speed = 8.0;     // VR 4, input

    double zone_speed[5] = {0.0, 0.0, 0.0, 0.0, 0.0};  // VR 10..14, outputs
    double reject_active = 0.0;    // VR 15, output

    double default_speed = 8.0;    // VR 20, parameter
    double reject_duration = 5.0;  // VR 21, parameter
    double time = 0.0;             // VR 22, independent

    double reject_until = 0.0;
    double previous_presence = 0.0;
};

static ConveyorController* as_controller(fmi2Component c) {
    return static_cast<ConveyorController*>(c);
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
    return new (std::nothrow) ConveyorController();
}

FMI_EXPORT void fmi2FreeInstance(fmi2Component c) {
    delete as_controller(c);
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
    auto* ctrl = as_controller(c);
    if (!ctrl) return fmi2Error;
    if (std::abs(ctrl->operator_speed) < 1e-9) {
        ctrl->operator_speed = ctrl->default_speed;
    }
    return fmi2OK;
}

FMI_EXPORT fmi2Status fmi2Terminate(fmi2Component c) {
    return c ? fmi2OK : fmi2Error;
}

FMI_EXPORT fmi2Status fmi2Reset(fmi2Component c) {
    auto* ctrl = as_controller(c);
    if (!ctrl) return fmi2Error;
    *ctrl = ConveyorController();
    return fmi2OK;
}

FMI_EXPORT fmi2Status fmi2DoStep(
    fmi2Component c,
    fmi2Real currentCommunicationPoint,
    fmi2Real communicationStepSize,
    fmi2Boolean) {
    auto* ctrl = as_controller(c);
    if (!ctrl || communicationStepSize < 0.0) return fmi2Error;

    const double next_time = currentCommunicationPoint + communicationStepSize;
    const double presence = ctrl->sensor_presence > 0.5 ? 1.0 : 0.0;
    if (presence > 0.5 && ctrl->previous_presence <= 0.5) {
        ctrl->reject_until = next_time + std::max(0.0, ctrl->reject_duration);
    }
    ctrl->previous_presence = presence;

    const bool enabled = ctrl->enable > 0.5 && ctrl->e_stop <= 0.5;
    if (!enabled) {
        if (ctrl->e_stop > 0.5) {
            ctrl->reject_until = next_time;
        }
        ctrl->reject_active = 0.0;
        for (double& speed : ctrl->zone_speed) {
            speed = 0.0;
        }
        ctrl->time = next_time;
        return fmi2OK;
    }

    const bool rejecting = next_time < ctrl->reject_until;
    ctrl->reject_active = rejecting ? 1.0 : 0.0;
    double forward_speed = std::abs(ctrl->operator_speed) > 1e-9
                               ? ctrl->operator_speed
                               : ctrl->default_speed;
    double reject_speed = std::abs(ctrl->reject_speed) > 1e-9
                              ? std::abs(ctrl->reject_speed)
                              : std::abs(forward_speed);
    const double commanded = rejecting ? -reject_speed : forward_speed;
    for (double& speed : ctrl->zone_speed) {
        speed = commanded;
    }
    ctrl->time = next_time;
    return fmi2OK;
}

FMI_EXPORT fmi2Status fmi2SetReal(
    fmi2Component c,
    const fmi2ValueReference* vr,
    size_t nvr,
    const fmi2Real* values) {
    auto* ctrl = as_controller(c);
    if (!ctrl || !vr || !values) return fmi2Error;
    for (size_t i = 0; i < nvr; ++i) {
        switch (vr[i]) {
        case 0: ctrl->operator_speed = values[i]; break;
        case 1: ctrl->enable = values[i]; break;
        case 2: ctrl->e_stop = values[i]; break;
        case 3: ctrl->sensor_presence = values[i]; break;
        case 4: ctrl->reject_speed = values[i]; break;
        case 20: ctrl->default_speed = values[i]; break;
        case 21: ctrl->reject_duration = values[i]; break;
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
    auto* ctrl = as_controller(c);
    if (!ctrl || !vr || !values) return fmi2Error;
    for (size_t i = 0; i < nvr; ++i) {
        switch (vr[i]) {
        case 0: values[i] = ctrl->operator_speed; break;
        case 1: values[i] = ctrl->enable; break;
        case 2: values[i] = ctrl->e_stop; break;
        case 3: values[i] = ctrl->sensor_presence; break;
        case 4: values[i] = ctrl->reject_speed; break;
        case 10: values[i] = ctrl->zone_speed[0]; break;
        case 11: values[i] = ctrl->zone_speed[1]; break;
        case 12: values[i] = ctrl->zone_speed[2]; break;
        case 13: values[i] = ctrl->zone_speed[3]; break;
        case 14: values[i] = ctrl->zone_speed[4]; break;
        case 15: values[i] = ctrl->reject_active; break;
        case 20: values[i] = ctrl->default_speed; break;
        case 21: values[i] = ctrl->reject_duration; break;
        case 22: values[i] = ctrl->time; break;
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
