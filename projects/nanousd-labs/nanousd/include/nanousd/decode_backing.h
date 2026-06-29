// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"
#include "types.h"

#include <cstdint>

namespace nanousd {

class Spec;
class Value;

// Format-specific decode-on-demand support for a Layer.
//
// `DecodeBacking` is the abstraction that lets a `Spec` defer its
// fields until first access without knowing what format the layer
// came from. The Layer owns the backing (typically as a shared_ptr);
// individual `Spec`s reference it by raw pointer and ask it to
// decode a specific field by name when queried.
//
// USDC layers populate a USDC-specific implementation backed by the
// crate file's section data. USDA layers and in-memory layers don't
// need this and leave the Layer's backing pointer null.
class NANOUSD_CORE_API DecodeBacking {
public:
    virtual ~DecodeBacking() = default;

    // Decode a single field from the fieldset at `fieldsetIdx`,
    // matching by name. Returns an empty `Value` when the fieldset
    // doesn't author a field by that name.
    virtual Value DecodeField(std::uint32_t fieldsetIdx,
                              const Token& fieldName) const = 0;

    // Decode every field in the fieldset and apply them to `target`.
    // Used when the caller wants the spec fully materialised at once
    // (e.g. `Spec::GetFields()` for iteration).
    virtual void DecodeAllFields(std::uint32_t fieldsetIdx,
                                  Spec& target) const = 0;
};

} // namespace nanousd
