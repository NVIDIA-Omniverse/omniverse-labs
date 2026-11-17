/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PVDFILELOADER_H
#define PVDFILELOADER_H

#include "PxPvdObjectModelBaseTypes.h"

namespace pvdapp
{

class PvdClientMutator;
class PvdRandomAccessIOStream;

using namespace physx;

struct PvdFileStreamLoadResult
{
	enum Status
	{
		Loaded,
		Completed,
		Failed
	};

	Status mStatus;
	PxU32 mBytesProcessed;

	PvdFileStreamLoadResult(Status status, PxU32 bytesProcessed)
	: mStatus(status), mBytesProcessed(bytesProcessed)
	{
	}
};

class PvdFileStreamLoader
{
  protected:
	virtual ~PvdFileStreamLoader() {}

  public:
	// Legacy API: zero means either completion or failure. New code should use
	// loadNextChunkResult() to preserve that distinction.
	virtual PxU32 loadNextChunk(PxU32 inRequestedAmountInBytes) = 0;
	virtual pvdapp::PvdClientMutator& getClient() = 0;
	virtual void release() = 0;

	// Appended after the original virtual sequence to preserve old slot order
	// and source overrides. Subclasses must be recompiled for this interface.
	virtual PvdFileStreamLoadResult loadNextChunkResult(PxU32 inRequestedAmountInBytes)
	{
		const PxU32 bytesProcessed = loadNextChunk(inRequestedAmountInBytes);
		return PvdFileStreamLoadResult(
			bytesProcessed ? PvdFileStreamLoadResult::Loaded
			               : PvdFileStreamLoadResult::Completed,
			bytesProcessed);
	}
};

struct PvdFileTypes
{
	enum Enum
	{
		Unknown,
		Stream,
		Database,
	};
};

class PvdFileLoaderFactory
{
  protected:
	virtual ~PvdFileLoaderFactory() {}

  public:
	virtual PvdFileStreamLoader* createFileStreamLoader(pvdapp::PvdRandomAccessIOStream& stream, pvdapp::PvdRandomAccessIOStream& backingStream) = 0;
	virtual PvdFileTypes::Enum getFileType(PxU32 inFirstFileWord) = 0;
	virtual void release() = 0;

	static PvdFileLoaderFactory& createFileLoaderFactory(PxAllocatorCallback& alloc);
};
} // namespace pvdapp

#endif // PVDFILELOADER_H
