/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "PvdSimpleFoundation.h"
#include "PxPvdByteStreams.h"
#include "PxPvdObjectModelBaseTypes.h"
#include "PxPvdObjectModelMetaData.h"
#include "foundation/PxMath.h"

namespace pvdapp
{

struct ErrorCode
{
	PxU32 mErrorCode;
	ErrorCode(PxU32 code = 0) : mErrorCode(code) {}
	operator PxU32() const { return mErrorCode; }
};
// Wraps a possibly OS-specific error code.
template <typename TDatatype>
class ErrorCodeOption : public physx::pvdsdk::Option<TDatatype>
{
	ErrorCode mErrorCode;

  public:
	ErrorCodeOption(ErrorCode code) : mErrorCode(code) {}
	ErrorCodeOption(const TDatatype& inType) : Option<TDatatype>(inType) {}
	ErrorCodeOption(const ErrorCodeOption<TDatatype>& inOther) : Option<TDatatype>(inOther), mErrorCode(inOther.mErrorCode) {}
	ErrorCodeOption& operator=(const ErrorCodeOption<TDatatype>& inOther)
	{
		Option<TDatatype>::operator=(inOther);
		mErrorCode = inOther.mErrorCode;
		return *this;
	}
};

// Database stream can read from anywhere but writes only to the
// end of the stream.
// File supports multiple readers without serializing access but writers
// are serialized.  Writes always write to the end of the file.
class PvdRandomAccessIOStream : public PvdOutputStream
{
  protected:
	virtual ~PvdRandomAccessIOStream() {}

  public:
	// Refcounted object.
	virtual void addRef() = 0;
	virtual void release() = 0;

	virtual Option<ErrorCode> read(PxU8* buffer, PxU32 inLen, PxU64 inOffset) = 0;

	// Pulled in from output stream
	// virtual bool write( const PxU8* buffer, PxU32 inLen ) = 0;

	// Direct copy another stream.
	virtual bool directCopy(PvdInputStream& inStream, PxU64 inLen) = 0;
	// Zero pad the file length till we end on a page boundary.
	virtual void zeroPadToPageSize(PxU32 inPageSize = 0x1000) = 0;
	virtual PxU64 getLength() const = 0;
	virtual Option<ErrorCode> flush() = 0;
	virtual Option<ErrorCode> asyncFlush() = 0;
	virtual double getTimeCost() = 0;

	static ErrorCodeOption<PvdRandomAccessIOStream*> open(PxAllocatorCallback& allocator, const char* inFilename, bool inCanWrite, bool inIsTemporary = false); // throws
	                                                                                                                                                            // runtime error
	                                                                                                                                                            // on failure.
};

struct IOStreamToInputStream : public physx::pvdsdk::PvdInputStream
{
	PxU64 mOffset;
	PxU64 mLen;
	PvdRandomAccessIOStream& mStream;

	IOStreamToInputStream(PxU64 offset, PvdRandomAccessIOStream& stream) : mOffset(offset), mLen(stream.getLength()), mStream(stream) {}

	virtual bool read(PxU8* buffer, PxU32& len)
	{
		const PxU64 remaining = mOffset < mLen ? mLen - mOffset : 0;
		const PxU32 available = remaining < static_cast<PxU64>(len)
			? static_cast<PxU32>(remaining)
			: len;
		if(available == 0)
		{
			if(len != 0)
				physx::intrinsics::memZero(buffer, len);
			len = 0;
			return true;
		}
		physx::pvdsdk::Option<ErrorCode> error = mStream.read(buffer, available, mOffset);
		PX_ASSERT(error.hasValue() == false);
		if(len > available)
			physx::intrinsics::memZero(buffer + available, len - available);
		PX_ASSERT(available == 0 || len == available || (mLen == (mOffset + available)));
		len = available;
		mOffset += available;
		return !error.hasValue();
	}
};
} // namespace pvdapp
