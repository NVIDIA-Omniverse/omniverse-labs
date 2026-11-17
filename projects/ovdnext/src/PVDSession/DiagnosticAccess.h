/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "Session.h"
namespace ovdnext {
// The engine reference is scoped to this mutex + live publication guard visitor.
// Diagnostic translation units may copy owned data, never retain the engine.
struct ProfilingAccess {
    static void read(Session& session,const std::function<void(pvdapp::PvdEngine&)>& f) { session.withEngineLocked(f); }
};
}
