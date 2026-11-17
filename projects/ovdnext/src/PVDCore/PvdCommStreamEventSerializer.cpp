/*
 * SPDX-FileCopyrightText: Copyright (c) 2008-2014 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Copyright (c) 2008-2014 NVIDIA Corporation. All rights reserved.

#include "PxPvdBits.h"
#include "PxPvdCommStreamTypes.h"
#include "PxPvdFoundation.h"
#include "PxSimpleTypes.h"
#include "PvdInputValidation.h"

#include <cstring>

using namespace physx::pvdsdk;
using namespace physx;
using namespace physx::shdfnd;

namespace
{
// The non-swapping version is the default
template <bool SwapBytes>
struct EventSerializerStream
{
	uint8_t* mBegin;
	uint8_t* mEnd;
	bool mGood;
	EventSerializerStream() : mBegin(NULL), mEnd(NULL), mGood(false) {}
	void setData(DataRef<const uint8_t> eventData)
	{
		mBegin = const_cast<uint8_t*>(eventData.begin());
		mEnd = const_cast<uint8_t*>(eventData.end());
		mGood = (mBegin == mEnd) || (mBegin && mEnd && mBegin < mEnd);
	}
	uint32_t size()
	{
		return mBegin && mEnd ? static_cast<uint32_t>(mEnd - mBegin) : 0;
	}
	void fail() { mGood = false; }
	bool checkSize(uint32_t desired)
	{
		if(!mGood)
			return false;
		const bool retval = desired <= size();
		if(!retval)
			mGood = false;
		return retval;
	}
	template <typename TDataType>
	void operator>>(TDataType& val)
	{
		if(checkSize(sizeof(val)))
		{
			physx::intrinsics::memCopy(&val, mBegin, sizeof(val));
			mBegin += sizeof(val);
		}
		else
			val = TDataType(0);
	}
};

template <>
struct EventSerializerStream<true>
{
	uint8_t* mBegin;
	uint8_t* mEnd;
	bool mGood;
	EventSerializerStream() : mBegin(NULL), mEnd(NULL), mGood(false) {}
	void setData(DataRef<const uint8_t> eventData)
	{
		mBegin = const_cast<uint8_t*>(eventData.begin());
		mEnd = const_cast<uint8_t*>(eventData.end());
		mGood = (mBegin == mEnd) || (mBegin && mEnd && mBegin < mEnd);
	}

	uint32_t size()
	{
		return mBegin && mEnd ? static_cast<uint32_t>(mEnd - mBegin) : 0;
	}
	void fail() { mGood = false; }
	bool checkSize(uint32_t desired)
	{
		if(!mGood)
			return false;
		const bool retval = desired <= size();
		if(!retval)
			mGood = false;
		return retval;
	}
	template <typename TDataType>
	void basicSwapBytes(TDataType& val)
	{
		if(checkSize(sizeof(val)))
		{
			physx::intrinsics::memCopy(&val, mBegin, sizeof(val));
			mBegin += sizeof(val);
			doSwapBytes<sizeof(TDataType)>(reinterpret_cast<PxU8*>(&val));
		}
		else
			val = TDataType(0);
	}

	void operator>>(uint8_t& val)
	{
		if(checkSize(sizeof(val)))
		{
			physx::intrinsics::memCopy(&val, mBegin, sizeof(val));
			mBegin += sizeof(val);
		}
		else
			val = uint8_t(0);
	}
	void operator>>(uint16_t& val) { basicSwapBytes(val); }
	void operator>>(uint32_t& val) { basicSwapBytes(val); }
	void operator>>(float& val) { basicSwapBytes(val); }
	void operator>>(uint64_t& val) { basicSwapBytes(val); }
};

void serialize(StringHandle& hdl, PvdEventSerializer& s) { s.streamify(hdl.mHandle); }

uint32_t serializedItemSize(const NameHandleValue*)
{
	return 2u * sizeof(uint32_t);
}

uint32_t serializedItemSize(const StreamPropMessageArg*)
{
	return 5u * sizeof(uint32_t);
}

uint32_t serializedItemSize(const StringHandle*)
{
	return sizeof(uint32_t);
}

template <typename TDataType>
void serialize(TDataType& dtype, PvdEventSerializer& s)
{
	dtype.serialize(s);
}

template <bool SwapBytes>
struct EventSerializerImpl : public ExtendedEventSerializer
{
	EventSerializerStream<SwapBytes> mStream;
	Array<PxU8> mTempData;

	EventSerializerImpl() : mTempData("EventSerializerImpl::mTempData") {}

	virtual void setData(DataRef<const PxU8> eventData) { mStream.setData(eventData); }
	virtual bool isGood() { return mStream.mGood; }
	// True if this serializer performs byte swapping
	virtual bool performsSwap() { return SwapBytes; }

	virtual void streamify(uint8_t& val) { mStream >> val; }
	virtual void streamify(uint32_t& val) { mStream >> val; }
	virtual void streamify(uint16_t& val) { mStream >> val; }
	virtual void streamify(float& val) { mStream >> val; }
	virtual void streamify(uint64_t& val) { mStream >> val; }
	virtual void streamify(String& val)
	{
		val = "";
		uint32_t numBytes = 0;
		mStream >> numBytes;
		if(!mStream.mGood || !numBytes)
			return;
		if(numBytes > pvdapp::input::kMaxAccumulatedPropertyBytes ||
		   !mStream.checkSize(numBytes))
		{
			mStream.fail();
			return;
		}
		if(!std::memchr(mStream.mBegin, 0, numBytes))
		{
			mStream.fail();
			return;
		}
		val = reinterpret_cast<const char*>(mStream.mBegin);
		mStream.mBegin += numBytes;
	}

	template <typename TDataType>
	bool streamifyBlock(DataRef<TDataType>& data)
	{
		data = DataRef<TDataType>();
		uint32_t numItems = 0;
		mStream >> numItems;
		if(!mStream.mGood)
			return false;
		if(!numItems)
			return true;

		uint32_t numBytes = 0;
		if(!pvdapp::input::checkedMultiplyToLimit(
			numItems, static_cast<uint32_t>(sizeof(TDataType)),
			pvdapp::input::kMaxAccumulatedPropertyBytes, numBytes))
		{
			mStream.fail();
			return false;
		}
		if(!mStream.checkSize(numBytes))
			return false;

		data = DataRef<TDataType>(
			reinterpret_cast<TDataType*>(mStream.mBegin), numItems);
		mStream.mBegin += numBytes;
		return true;
	}

	virtual void streamify(DataRef<const PxU8>& data) { streamifyBlock(data); }

	template <typename TDataType>
	TDataType* allocTemp(uint32_t numItems, uint32_t numBytes)
	{
		if(mTempData.size() < numBytes)
			mTempData.resize(numBytes);
		if(mTempData.size() < numBytes || !mTempData.begin())
		{
			mStream.fail();
			return NULL;
		}
		TDataType* retval = reinterpret_cast<TDataType*>(mTempData.begin());
		PVD_FOREACH(idx, numItems) new(retval + idx) TDataType();
		return retval;
	}

	template <typename TDataType>
	void streamifyTemp(DataRef<TDataType>& data)
	{
		data = DataRef<TDataType>();
		uint32_t numItems = 0;
		mStream >> numItems;
		if(!mStream.mGood || !numItems)
			return;
		if(numItems > pvdapp::input::kMaxPropertyItems)
		{
			mStream.fail();
			return;
		}

		uint32_t serializedBytes = 0;
		uint32_t allocationBytes = 0;
		if(!pvdapp::input::checkedMultiplyToLimit(
			numItems, serializedItemSize(static_cast<TDataType*>(NULL)),
			pvdapp::input::kMaxAccumulatedPropertyBytes, serializedBytes) ||
		   !pvdapp::input::checkedMultiplyToLimit(
			numItems, static_cast<uint32_t>(sizeof(TDataType)),
			pvdapp::input::kMaxAccumulatedPropertyBytes, allocationBytes))
		{
			mStream.fail();
			return;
		}
		if(!mStream.checkSize(serializedBytes))
			return;

		TDataType* dataPtr = allocTemp<TDataType>(numItems, allocationBytes);
		if(!dataPtr)
			return;

		const uint32_t bytesBefore = mStream.size();
		PVD_FOREACH(idx, numItems)
		{
			serialize(dataPtr[idx], *this);
			if(!mStream.mGood)
				return;
		}
		if(bytesBefore - mStream.size() != serializedBytes)
		{
			mStream.fail();
			return;
		}
		data = DataRef<TDataType>(dataPtr, numItems);
	}

	virtual void streamify(DataRef<NameHandleValue>& data) { streamifyTemp(data); }
	virtual void streamify(DataRef<StreamPropMessageArg>& data) { streamifyTemp(data); }
	// don't send the count, just send the handles if any.
	virtual void streamify(DataRef<StringHandle>& data) { streamifyTemp(data); }

	virtual void release() { PVD_DELETE(this); }
};
} // namespace

ExtendedEventSerializer& ExtendedEventSerializer::createInputSerializer(bool swapBytes)
{
	if(swapBytes)
		return *PVD_NEW(EventSerializerImpl<true>)();
	return *PVD_NEW(EventSerializerImpl<false>)();
}
