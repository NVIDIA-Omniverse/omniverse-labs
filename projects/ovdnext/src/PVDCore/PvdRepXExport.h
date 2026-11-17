/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <PsArray.h>

namespace pvdapp
{
class PvdClient;
class PvdRandomAccessIOStream;
} // namespace pvdapp

namespace pvdapp
{
namespace repx
{
class PvdRepxExporter
{
  public:
	static bool canExport(pvdapp::PvdClient& client);
	static void exportToRepX(physx::shdfnd::Array<uint64_t>& objects, pvdapp::PvdClient& client, pvdapp::PvdRandomAccessIOStream& stream, physx::PxAllocatorCallback& alloc);
};
} // namespace repx
} // namespace pvdapp
