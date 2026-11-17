/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <iosfwd>

namespace ovdnext
{
namespace data
{
class FrameView;
}

namespace omnipvd
{

void writeExactClassAliveSummary(const data::FrameView& view,
                                 std::ostream& output);
int runProbe(int argc, const char* const argv[], std::ostream& output, std::ostream& error);

} // namespace omnipvd
} // namespace ovdnext
