// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Export/import macro for nanousd's C++ core symbols.
//
// The core is always built as a SHARED library (nanousd.dll on Windows,
// libnanousd.so on POSIX). On Windows, DLLs export nothing unless annotated
// with __declspec(dllexport); consumers that link against the DLL need the
// same symbols annotated with __declspec(dllimport). On POSIX,
// __attribute__((visibility("default"))) is used so the symbol is visible in
// the shared object's export table.
//
// Scope: place NANOUSD_CORE_API on public-facing classes and free functions
// exposed to consumers of nanousd's C++ API — e.g. UsdStage, UsdPrim,
// WriteUsdc. Internal helpers and implementation-detail types do not need
// the annotation.
//
// NANOUSD_BACKEND_BUILDING is set by the nanousd CMake target's private
// compile definitions; consumers of the headers never see it, so they get
// the import-side of the macro.

#ifdef _WIN32
#   ifdef NANOUSD_BACKEND_BUILDING
#       define NANOUSD_CORE_API __declspec(dllexport)
#   else
#       define NANOUSD_CORE_API __declspec(dllimport)
#   endif
#else
#   define NANOUSD_CORE_API __attribute__((visibility("default")))
#endif
