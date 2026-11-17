/*
 * SPDX-FileCopyrightText: Copyright (c) 2008-2014 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Copyright (c) 2008-2014 NVIDIA Corporation. All rights reserved.
#ifndef PVD_NETWORK_TO_BYTE_STREAMS_H
#define PVD_NETWORK_TO_BYTE_STREAMS_H
#include "PvdNetworkStreams.h"
#include "PxPvdByteStreams.h"
#include "PxPvdErrorCodes.h"
#include "PxPvdFoundation.h"

namespace pvdapp
{
struct PvdNetworkToInStream : public PvdInputStream
{
	PxAllocatorCallback& mAlloc;
	PvdNetworkInStream& mStream;
	PvdNetworkToInStream(PxAllocatorCallback& alloc, PvdNetworkInStream& str) : mAlloc(alloc), mStream(str) {}
	virtual ~PvdNetworkToInStream() { mStream.release(); }
	virtual bool read(PxU8* buffer, PxU32& len)
	{
		PvdError error = mStream.readBytes(buffer, len);
		if(error != PvdErrorType::Success)
			len = 0;
		return error == PvdErrorType::Success;
	}
	virtual void release() { PVD_DELETE(this); }

	virtual PxU64 getLoadedDataSize() { return mStream.getLoadedDataSize(); }
};

struct PvdNetworkToOutStream : public PvdOutputStream
{
	PxAllocatorCallback& mAlloc;
	PvdNetworkOutStream& mStream;
	PvdNetworkToOutStream(PxAllocatorCallback& alloc, PvdNetworkOutStream& str) : mAlloc(alloc), mStream(str) {}
	virtual ~PvdNetworkToOutStream() { mStream.release(); }
	virtual bool write(const PxU8* buffer, PxU32 len)
	{
		PvdError error = mStream.write(buffer, len);
		return error == PvdErrorType::Success;
	}
	virtual bool directCopy(PvdInputStream& inStream, PxU32 len)
	{
		PxU8 buf[2048];
		while(len)
		{
			PxU32 request = 2048;
			inStream.read(buf, request);
			bool success = write(buf, request);
			len -= request;
			if(success == false || (len > 0 && request != 2048))
				return false;
		}
		return true;
	}
	virtual void release() { PVD_DELETE(this); }
};
} // namespace pvdapp

#endif
