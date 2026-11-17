/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef OVDFILELOADER_H
#define OVDFILELOADER_H

#ifdef PVD_OVD_SUPPORT

#include "OvdInputValidation.h"
#include "LoadControl.h"

#include <memory>

namespace pvdapp
{

class OvdDomAdapter;

// Check if a filename has .ovd extension
bool hasOvdExtension(const char* filename);

// Parse an OVD file and return an adapter that serves queries directly
// from the DOM. Returns nullptr on failure.
std::unique_ptr<OvdDomAdapter> loadOvdFile(const char* ovdFilePath);

// Injectable limits are used by focused tests and embedders that need a
// tighter policy. errorCode receives a stable reason on rejection.
std::unique_ptr<OvdDomAdapter> loadOvdFile(const char* ovdFilePath,
    const OvdResourceLimits& limits, OvdInputErrorCode* errorCode, LoadControl* control = nullptr);

} // namespace pvdapp

#endif // PVD_OVD_SUPPORT
#endif // OVDFILELOADER_H
