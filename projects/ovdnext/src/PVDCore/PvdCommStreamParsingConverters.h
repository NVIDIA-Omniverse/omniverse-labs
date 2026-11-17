/*
 * SPDX-FileCopyrightText: Copyright (c) 2008-2014 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Copyright (c) 2008-2014 NVIDIA Corporation. All rights reserved.

#ifndef PVD_COMM_STREAM_PARSING_CONVERTERS_H
#define PVD_COMM_STREAM_PARSING_CONVERTERS_H
#include "PxPvdObjectModelBaseTypes.h"

#include <cstring>

namespace physx
{
namespace pvdsdk
{
class StringTable;
}
} // namespace physx

namespace pvdapp
{
namespace comm
{

using namespace physx::pvdsdk;

struct ParserConverterData
{
	uint32_t incomingOffset;
	uint32_t outgoingOffset;
	uint32_t numItems;
	ParserConverterData() : incomingOffset(0), outgoingOffset(0), numItems(0) {}
};

struct ConverterSrcData
{
	uint8_t* src;            // Ptr to incoming data
	uint8_t* srcStringStart; // Ptr to next string data, if any
	uint8_t* srcEnd;         // Ptr to end of the src data.
	ConverterSrcData(uint8_t* _src, uint8_t* _srcEnd, uint32_t dataSectionSize) : src(_src), srcStringStart(_src + dataSectionSize), srcEnd(_srcEnd)
	{
		if(src == srcEnd)
			srcStringStart = srcEnd;
		verify();
	}
	ConverterSrcData(const ConverterSrcData& other) : src(other.src), srcStringStart(other.srcStringStart), srcEnd(other.srcEnd) {}
	inline void setStringStart(uint8_t* newStart)
	{
		srcStringStart = newStart;
		verify();
	}
	inline void verify()
	{
		PX_ASSERT(src <= srcEnd);
		PX_ASSERT(srcStringStart <= srcEnd);
		PX_ASSERT(src <= srcStringStart);
	}

	inline const char* nextStr()
	{
		const char* retval = "";
		if(srcStringStart < srcEnd)
		{
			const void* terminator = ::memchr(
				srcStringStart, 0, static_cast<size_t>(srcEnd - srcStringStart));
			if(terminator)
			{
				retval = reinterpret_cast<const char*>(srcStringStart);
				srcStringStart = static_cast<uint8_t*>(const_cast<void*>(terminator)) + 1;
			}
			else
				srcStringStart = srcEnd;
		}
		return retval;
	}
};

// Converters need to do 3 things
// 1. Endian convert
// 2. Stream convert (void* to 32 bit db id)
// 3. String convert (const char* to 32 bit StringHandle)
// 4. Marshal between one type of data item and another (u8 -> u32, or u8 -> float)
// Furthermore, there are 3 conditions this happens under.
// 1. Convert this data, store result back in the src location.
// 2. Convert this data, store result into dest location.
// 2. Convert a large block of this data, as quickly as you can.
// The first three conditions are handled by the underlying datatype of the converter.
// the last three conditions need to be handled by the converters themselves.
// Strings are stored, null terminated, *after* the initial block of data.  Thus
// anything sending string data will send it *after* the main message.
// For arrays, this means after each entry in the array any strings contains in the entry
// are embedded in the data stream.  For property messages, this means the string is sent
// after the property message.
//
// srcEnd marks the absolute end of the message, string offset
// is always <= srcEnd - src;
//
// Finally the converter returns the location of the next string in the dataset.
// which must lie directly after the last string that was read.
class ParserStreamDataConverter
{
  protected:
	virtual ~ParserStreamDataConverter() {}

  public:
	virtual uint8_t* Convert(ParserConverterData data, ConverterSrcData srcData) = 0;
	// Convert, write results to dest
	virtual uint8_t* Convert(ParserConverterData data, ConverterSrcData srcData, uint8_t* dest) = 0;
	// Convert, write results to src
	virtual void ConvertBlock(ParserConverterData data, ConverterSrcData srcData) = 0;
	// Convert, write results to dest
	virtual void ConvertBlock(ParserConverterData data, ConverterSrcData srcData, uint8_t* dest) = 0;
};

class ParserIdToDBHandle
{
  protected:
	virtual ~ParserIdToDBHandle() {}

  public:
	virtual int32_t getDBHandle(uint32_t streamId) = 0;
	virtual int32_t getDBHandle(uint64_t streamId) = 0;
};

class ParserStreamDataConverterFactory
{
  protected:
	virtual ~ParserStreamDataConverterFactory() {}

  public:
	virtual ParserStreamDataConverter* getConverter(int32_t incomingDatatype, int32_t outgoingDatatype, uint32_t incomingDatatypePackedWidth) = 0;
	virtual void release() = 0;

	static ParserStreamDataConverterFactory& createConverterFactory(physx::PxAllocatorCallback& alloc, bool swapBytes, bool PtrsAre64Bit, ParserIdToDBHandle& handleConversion, StringTable& stringTable);
};

struct ConverterAndData
{
	ParserConverterData mData;
	ParserStreamDataConverter* mConverter;
	ConverterAndData(ParserStreamDataConverter* converter) : mData(ParserConverterData()), mConverter(converter) {}

	void Convert(ConverterSrcData& srcData)
	{
		PX_ASSERT(mData.incomingOffset >= mData.outgoingOffset);
		srcData.setStringStart(mConverter->Convert(mData, srcData));
	}
	void Convert(ConverterSrcData& srcData, uint8_t* dest)
	{
		PX_ASSERT(mData.incomingOffset >= mData.outgoingOffset);
		srcData.setStringStart(mConverter->Convert(mData, srcData, dest));
	}
	void ConvertBlock(ConverterSrcData& srcData)
	{
		PX_ASSERT(mData.incomingOffset == 0);
		PX_ASSERT(mData.outgoingOffset == 0);
		mConverter->ConvertBlock(mData, srcData);
	}
	void ConvertBlock(ConverterSrcData& srcData, uint8_t* dest)
	{
		PX_ASSERT(mData.incomingOffset == 0);
		PX_ASSERT(mData.outgoingOffset == 0);
		mConverter->ConvertBlock(mData, srcData, dest);
	}
};
} // namespace comm
} // namespace pvdapp

#endif
