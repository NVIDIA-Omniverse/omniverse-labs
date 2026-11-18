/*
 * SPDX-FileCopyrightText: Copyright (c) 2008-2014 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Copyright (c) 2008-2014 NVIDIA Corporation. All rights reserved.

#include "PvdCommStreamParsingConverters.h"
#include "PsHashMap.h"
#include "PxPvdBits.h"
#include "PxPvdFoundation.h"
#include "PxPvdMarshalling.h"
#include "PxPvdObjectModelMetaData.h"
#include "PxSimpleTypes.h"

using namespace physx;
using namespace physx::pvdsdk;
using namespace pvdapp::comm;
using namespace physx::shdfnd;

namespace
{
class ParserStreamDataConverterInternal : public ParserStreamDataConverter
{
  public:
	virtual void release() = 0;
};

template <PxU8 ByteCount, bool ShouldSwap, typename TDataConverter>
struct ByteSwapDataConverter : public ParserStreamDataConverterInternal
{
	// Converters perform conversion *and* increment the src and dest pointers
	TDataConverter mConverter;
	PvdByteSwapper<ByteCount, ShouldSwap> mSwapper;

	ByteSwapDataConverter(ParserIdToDBHandle& hc, StringTable& st) : mConverter(hc, st) {}

	virtual PxU8* Convert(ParserConverterData data, ConverterSrcData srcData)
	{
		srcData.src += data.incomingOffset;
		srcData.verify();
		for(PxU32 idx = 0; idx < data.numItems; ++idx)
		{
			mSwapper.swapBytes(srcData.src);
			mConverter(srcData);
		}
		return srcData.srcStringStart;
	}
	virtual PxU8* Convert(ParserConverterData data, ConverterSrcData srcData, PxU8* dest)
	{
		srcData.src += data.incomingOffset;
		dest += data.outgoingOffset;
		srcData.verify();
		for(PxU32 idx = 0; idx < data.numItems; ++idx)
		{
			mSwapper.swapBytes(srcData.src);
			mConverter(srcData, dest);
		}
		return srcData.srcStringStart;
	}
	virtual void ConvertBlock(ParserConverterData data, ConverterSrcData srcData)
	{
		PX_ASSERT(data.incomingOffset == 0);
		PX_ASSERT(data.outgoingOffset == 0);
		while(srcData.src < srcData.srcEnd)
		{
			PVD_FOREACH(idx, data.numItems)
			{
				mSwapper.swapBytes(srcData.src);
				mConverter(srcData);
			}
			srcData.src = srcData.srcStringStart;
			srcData.srcStringStart = srcData.src + ByteCount * data.numItems;
		}
	}
	virtual void ConvertBlock(ParserConverterData data, ConverterSrcData srcData, PxU8* dest)
	{
		PX_ASSERT(data.incomingOffset == 0);
		PX_ASSERT(data.outgoingOffset == 0);
		while(srcData.src < srcData.srcEnd)
		{
			PVD_FOREACH(idx, data.numItems)
			{
				mSwapper.swapBytes(srcData.src);
				mConverter(srcData, dest);
			}
			srcData.src = srcData.srcStringStart;
			srcData.srcStringStart = srcData.src + ByteCount * data.numItems;
		}
	}

	virtual void release() { PVD_DELETE(this); }
};

template <PxU8 ByteCount>
struct IdentityConverter
{
	IdentityConverter(ParserIdToDBHandle&, StringTable&) {}
	void operator()(ConverterSrcData& src) { src.src += ByteCount; }
	void operator()(ConverterSrcData& src, PxU8*& dest)
	{
		physx::intrinsics::memCopy(dest, src.src, ByteCount);
		src.src += ByteCount;
		dest += ByteCount;
	}
};

// Specialize this class because this case shouldn't do anything.
template <PxU8 ByteCount>
struct ByteSwapDataConverter<ByteCount, false, IdentityConverter<ByteCount> > : public ParserStreamDataConverter
{
	ByteSwapDataConverter(ParserIdToDBHandle&, StringTable&) {}
	virtual PxU8* Convert(ParserConverterData, ConverterSrcData srcData) { return srcData.srcStringStart; }
	// Convert, write results to dest
	virtual PxU8* Convert(ParserConverterData data, ConverterSrcData srcData, PxU8* dest)
	{
		srcData.src += data.incomingOffset;
		dest += data.outgoingOffset;
		PxU32 numBytes = ByteCount * data.numItems;
		physx::intrinsics::memCopy(dest, srcData.src, numBytes);
		return srcData.srcStringStart;
	}
	// Convert, write results to src
	virtual void ConvertBlock(ParserConverterData, ConverterSrcData)
	{
		// Hopefully happens all the time.
	}
	// Convert, write results to dest
	virtual void ConvertBlock(ParserConverterData data, ConverterSrcData srcData, PxU8* dest)
	{
		PX_UNUSED(data);
		PX_ASSERT(false);
		PxU32 numBytes = static_cast<PxU32>(srcData.srcEnd - srcData.src);
		physx::intrinsics::memCopy(dest, srcData.src, numBytes);
	}
};

template <typename StreamIdType>
struct StreamIdConverter
{
	ParserIdToDBHandle& mConverter;
	StreamIdConverter(ParserIdToDBHandle& c, StringTable&) : mConverter(c) {}
	void operator()(ConverterSrcData& src)
	{
		StreamIdType streamId;
		physx::intrinsics::memCopy(&streamId, src.src, sizeof(StreamIdType));
		PxI32 hdlVal = streamId ? mConverter.getDBHandle(streamId) : 0;
		physx::intrinsics::memCopy(src.src, &hdlVal, 4);
		src.src += sizeof(StreamIdType);
	}

	void operator()(ConverterSrcData& src, PxU8*& dest)
	{
		StreamIdType streamId;
		physx::intrinsics::memCopy(&streamId, src.src, sizeof(StreamIdType));
		PxI32 hdlVal = streamId ? mConverter.getDBHandle(streamId) : 0;
		physx::intrinsics::memCopy(dest, &hdlVal, 4);
		src.src += sizeof(StreamIdType);
		dest += 4;
	}
};

template <PxU8 ByteCount>
struct StringConverter
{
	StringTable& mStringTable;
	StringConverter(ParserIdToDBHandle&, StringTable& table) : mStringTable(table) {}
	void operator()(ConverterSrcData& src)
	{
		const char* strData = src.nextStr();
		StringHandle strHandle = mStringTable.strToHandle(strData);
		physx::intrinsics::memCopy(src.src, &strHandle, sizeof(StringHandle));
		src.src += ByteCount;
	}

	void operator()(ConverterSrcData& src, PxU8*& dest)
	{
		const char* strData = src.nextStr();
		StringHandle strHandle = mStringTable.strToHandle(strData);
		physx::intrinsics::memCopy(dest, &strHandle, sizeof(StringHandle));
		src.src += ByteCount;
		dest += sizeof(StringHandle);
	}
};

template <typename smtype, typename lgtype>
struct MarshallingConverter
{
	MarshallingConverter(ParserIdToDBHandle&, StringTable&) {}
	void operator()(ConverterSrcData& src)
	{
		PX_ASSERT(sizeof(smtype) == sizeof(lgtype));
		marshalSingleT<smtype, lgtype>(src.src, src.src);
		src.src += sizeof(smtype);
	}
	void operator()(ConverterSrcData& src, PxU8*& dest)
	{
		marshalSingleT<smtype, lgtype>(src.src, dest);
		src.src += sizeof(smtype);
		dest += sizeof(lgtype);
	}
};

struct MarshalConverterKey
{
	PxI32 smtype;
	PxI32 lgtype;
	MarshalConverterKey(PxI32 _smtype, PxI32 _lgtype) : smtype(_smtype), lgtype(_lgtype) {}
};

struct MarshalConverterKeyHash
{
	PxU32 operator()(const MarshalConverterKey& key) { return hash(key.smtype) ^ hash(key.lgtype); }

	bool operator()(const MarshalConverterKey& lhs, const MarshalConverterKey& rhs) { return lhs.smtype == rhs.smtype && lhs.lgtype == rhs.lgtype; }

	bool equal(const MarshalConverterKey& lhs, const MarshalConverterKey& rhs) { return lhs.smtype == rhs.smtype && lhs.lgtype == rhs.lgtype; }
};

template <bool ShouldSwap, PxU8 TSizeofStreamIdType, typename TStreamIdType>
class ConverterFactory : public ParserStreamDataConverterFactory
{
	typedef HashMap<MarshalConverterKey, ParserStreamDataConverterInternal*, MarshalConverterKeyHash> TMarshalMapType;
	PxAllocatorCallback& mAllocator;
	ParserIdToDBHandle& mIdToHandle;
	StringTable& mStringTable;

	ByteSwapDataConverter<1, ShouldSwap, IdentityConverter<1> > mConverter1Identity;
	ByteSwapDataConverter<2, ShouldSwap, IdentityConverter<2> > mConverter2Identity;
	ByteSwapDataConverter<4, ShouldSwap, IdentityConverter<4> > mConverter4Identity;
	ByteSwapDataConverter<8, ShouldSwap, IdentityConverter<8> > mConverter8Identity;

	ByteSwapDataConverter<TSizeofStreamIdType, ShouldSwap, StreamIdConverter<TStreamIdType> > mIdConverter;
	ByteSwapDataConverter<TSizeofStreamIdType, ShouldSwap, StringConverter<TSizeofStreamIdType> > mStringConverter;

	TMarshalMapType mMarshalMap;

  public:
	ConverterFactory(PxAllocatorCallback& alloc, ParserIdToDBHandle& hc, StringTable& st)
	: mAllocator(alloc)
	, mIdToHandle(hc)
	, mStringTable(st)
	, mConverter1Identity(hc, st)
	, mConverter2Identity(hc, st)
	, mConverter4Identity(hc, st)
	, mConverter8Identity(hc, st)
	, mIdConverter(hc, st)
	, mStringConverter(hc, st)
	, mMarshalMap("ConverterFactory::mMarshalMap")
	{
	}
	virtual ~ConverterFactory()
	{
		for(TMarshalMapType::Iterator iter = mMarshalMap.getIterator(); iter.done() == false; ++iter)
			iter->second->release();
		mMarshalMap.clear();
	}
	virtual void release() { PVD_DELETE(this); }
	template <typename smtype>
	ParserStreamDataConverterInternal* getMarshallingConverter(PxI32 outgoingDatatype)
	{
		switch(outgoingDatatype)
		{
#define CREATE_CONVERTER_TYPE(type)                                                                                                                                                                    \
	case PvdBaseType::type:                                                                                                                                                                            \
	{                                                                                                                                                                                                  \
		typedef ByteSwapDataConverter<sizeof(smtype), ShouldSwap, MarshallingConverter<smtype, type> > TDataType;                                                                                      \
		return PVD_NEW(TDataType)(mIdToHandle, mStringTable);                                                                                                                                          \
	}
			CREATE_CONVERTER_TYPE(PvdU16);
			CREATE_CONVERTER_TYPE(PvdI16);

			CREATE_CONVERTER_TYPE(PvdU32);
			CREATE_CONVERTER_TYPE(PvdI32);
			CREATE_CONVERTER_TYPE(PvdF32);

			CREATE_CONVERTER_TYPE(PvdU64);
			CREATE_CONVERTER_TYPE(PvdI64);
			CREATE_CONVERTER_TYPE(PvdF64);

			CREATE_CONVERTER_TYPE(PvdI8);
		// fallthrough intentional, at least create something no matter what, however.
		default:
			PX_ASSERT(false);
			CREATE_CONVERTER_TYPE(PvdU8);
		}
	}

	ParserStreamDataConverterInternal* getMarshallingConverter(PxI32 incomingDatatype, PxI32 outgoingDatatype)
	{
		switch(incomingDatatype)
		{
		case PvdBaseType::PvdU16:
			return getMarshallingConverter<PxU16>(outgoingDatatype);
		case PvdBaseType::PvdI16:
			return getMarshallingConverter<PxI16>(outgoingDatatype);

		case PvdBaseType::PvdU32:
			return getMarshallingConverter<PxU32>(outgoingDatatype);
		case PvdBaseType::PvdI32:
			return getMarshallingConverter<PxI32>(outgoingDatatype);
		case PvdBaseType::PvdF32:
			return getMarshallingConverter<PxF32>(outgoingDatatype);

		case PvdBaseType::PvdU64:
			return getMarshallingConverter<PxU64>(outgoingDatatype);
		case PvdBaseType::PvdI64:
			return getMarshallingConverter<PxI64>(outgoingDatatype);
		case PvdBaseType::PvdF64:
			return getMarshallingConverter<PxF64>(outgoingDatatype);

		case PvdBaseType::PvdI8:
			return getMarshallingConverter<PxI8>(outgoingDatatype);
		// fallthrough intentional, return *something*.
		default:
			PX_ASSERT(false);
		case PvdBaseType::PvdU8:
			return getMarshallingConverter<PxU8>(outgoingDatatype);
		}
	}

	virtual ParserStreamDataConverter* getConverter(PxI32 incomingDatatype, PxI32 outgoingDatatype, PxU32 incomingDatatypePackedWidth)
	{
		if(incomingDatatype == getPvdTypeForType<ObjectRef>() || incomingDatatype == getPvdTypeForType<VoidPtr>())
			return &mIdConverter;

		if(incomingDatatype == getPvdTypeForType<String>() || incomingDatatype == getPvdTypeForType<StringHandle>())
			return &mStringConverter;

		if(incomingDatatype == outgoingDatatype)
		{
			switch(incomingDatatypePackedWidth)
			{
			case 2:
				return &mConverter2Identity;
			case 4:
				return &mConverter4Identity;
			case 8:
				return &mConverter8Identity;
			// fallthrough intentional, at least in release it shouldn't crash...
			default:
				PX_ASSERT(false);
			case 1:
				return &mConverter1Identity;
			}
		}
		else
		{
			MarshalConverterKey converterKey(incomingDatatype, outgoingDatatype);
			const TMarshalMapType::Entry* entry = mMarshalMap.find(converterKey);
			if(entry != NULL)
				return entry->second;
			ParserStreamDataConverterInternal* converter = getMarshallingConverter(incomingDatatype, outgoingDatatype);
			mMarshalMap.insert(converterKey, converter);
			return converter;
		}
	}
};
} // namespace

ParserStreamDataConverterFactory& ParserStreamDataConverterFactory::createConverterFactory(PxAllocatorCallback& alloc, bool swapBytes, bool ptrsAre64Bit, ParserIdToDBHandle& handleConversion,
                                                                                           StringTable& stringTable)
{
	if(swapBytes)
	{
		if(ptrsAre64Bit)
		{
			typedef ConverterFactory<true, 8, PxU64> TDataType;
			return *PVD_NEW(TDataType)(alloc, handleConversion, stringTable);
		}
		else
		{
			typedef ConverterFactory<true, 4, PxU32> TDataType;
			return *PVD_NEW(TDataType)(alloc, handleConversion, stringTable);
		}
	}
	else
	{
		if(ptrsAre64Bit)
		{
			typedef ConverterFactory<false, 8, PxU64> TDataType;
			return *PVD_NEW(TDataType)(alloc, handleConversion, stringTable);
		}
		else
		{
			typedef ConverterFactory<false, 4, PxU32> TDataType;
			return *PVD_NEW(TDataType)(alloc, handleConversion, stringTable);
		}
	}
}
