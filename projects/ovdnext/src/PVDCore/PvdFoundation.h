/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PVD_FOUNDATION_H
#define PVD_FOUNDATION_H

#include "PxAllocatorCallback.h"
#include <math.h>
#include "PxMemory.h"
#include "Ps.h"
#include "NsArray.h"
#include "NsHashMap.h"
#include "NsHashSet.h"
#include "NsPool.h"
#include "PvdObjectModelBaseTypes.h"
#include "PvdByteStreams.h"
#include "PxMath.h"
#include "NvVec3.h"
#include "NvTransform.h"
#include "NvBounds3.h"
#define PX_MAX_BOUNDS_EXTENTS NV_MAX_BOUNDS_EXTENTS
#include "PxFiltering.h"
#include "NsString.h"

namespace physx
{
	namespace pvdsdk
	{
        using namespace physx::shdfnd;

		struct ForwardingAllocator
		{
			PxAllocatorCallback& mAllocator;
			const char* mTypeName;

			ForwardingAllocator(PxAllocatorCallback& alloc, const char* typeName) : mAllocator(alloc), mTypeName(typeName)
			{
			}
			ForwardingAllocator(PxAllocatorCallback* alloc = NULL) : mAllocator(*alloc), mTypeName("__error__")
			{
                PX_ASSERT(false);
            }
			ForwardingAllocator(const ForwardingAllocator& other) : mAllocator(other.mAllocator), mTypeName(other.mTypeName)
			{
			}
			ForwardingAllocator& operator=(const ForwardingAllocator&)
			{
				return *this;
			}
			PxAllocatorCallback& getAllocator()
			{
				return mAllocator;
			}
			void* allocate(size_t size, const char* filename, int line)
			{
				return getAllocator().allocate(size, mTypeName, filename, line);
			}
			void deallocate(void* ptr)
			{
				getAllocator().deallocate(ptr);
			}
			virtual ~ForwardingAllocator()
			{
			}
		};

		template <class T>
        struct ForwardingArray : public nvidia::shdfnd::Array<T, ForwardingAllocator>
		{
			typedef ForwardingAllocator TAllocatorType;

			ForwardingArray(PxAllocatorCallback& alloc, const char* typeName)
                : nvidia::shdfnd::Array<T, TAllocatorType>(TAllocatorType(alloc, typeName))
			{
			}

            ForwardingArray(const ForwardingArray<T>& inOther) : nvidia::shdfnd::Array<T, TAllocatorType>(inOther, inOther)
			{
			}
		};

        template <typename TKeyType, typename TValueType, typename THashType = nvidia::shdfnd::Hash<TKeyType> >
        struct ForwardingHashMap : public nvidia::shdfnd::HashMap<TKeyType, TValueType, THashType, ForwardingAllocator>
		{
            typedef nvidia::shdfnd::HashMap<TKeyType, TValueType, THashType, ForwardingAllocator> THashMapType;
			typedef ForwardingAllocator TAllocatorType;
			ForwardingHashMap(PxAllocatorCallback& alloc, const char* typeName) : THashMapType(TAllocatorType(alloc, typeName))
			{
			}
			ForwardingHashMap(const ForwardingHashMap<TKeyType, TValueType, THashType>& inOther) : THashMapType(inOther)
			{
				for(typename ForwardingHashMap<TKeyType, TValueType, THashType>::Iterator iter =
					const_cast<ForwardingHashMap<TKeyType, TValueType, THashType>&>(inOther).getIterator();
					iter.done() == false; ++iter)
					insert(iter->first, iter->second);
			}
			ForwardingHashMap<TKeyType, TValueType, THashType>&
				operator=(const ForwardingHashMap<TKeyType, TValueType, THashType>& inOther)
			{
				if(this != &inOther)
				{
                    nvidia::shdfnd::HashMap<TKeyType, TValueType, THashType, ForwardingAllocator>::clear();
					for(typename ForwardingHashMap<TKeyType, TValueType, THashType>::Iterator iter =
						const_cast<ForwardingHashMap<TKeyType, TValueType, THashType>&>(inOther).getIterator();
						iter.done() == false; ++iter)
                    {
                        this->insert(iter->first, iter->second);
                    }
				}
				return *this;
			}
		};

        template <typename TValueType, typename THashType = nvidia::shdfnd::Hash<TValueType> >
        struct ForwardingHashSet : public nvidia::shdfnd::HashSet<TValueType, THashType, ForwardingAllocator>
		{
            typedef nvidia::shdfnd::HashSet<TValueType, THashType, ForwardingAllocator> THashSetType;
			typedef ForwardingAllocator TAllocatorType;
			ForwardingHashSet(PxAllocatorCallback& alloc, const char* typeName) : THashSetType(TAllocatorType(alloc, typeName))
			{
			}
		};

		template <typename TDataType>
        struct ForwardingPool : public nvidia::shdfnd::Pool<TDataType, ForwardingAllocator>
		{
            ForwardingPool(const ForwardingAllocator& allocator) : nvidia::shdfnd::Pool<TDataType, ForwardingAllocator>(allocator)
			{
			}
		};

        template <typename TAllocator = typename nvidia::shdfnd::AllocatorTraits<PxU8>::Type>
		class RawMemoryBuffer : public TAllocator
		{
			PxU8* mBegin;
			PxU8* mEnd;
			PxU8* mCapacityEnd;

		public:
			RawMemoryBuffer(const TAllocator& inAlloc = TAllocator()) : TAllocator(inAlloc), mBegin(0), mEnd(0), mCapacityEnd(0)
			{
			}
			~RawMemoryBuffer()
			{
				if(mBegin)
					TAllocator::deallocate(mBegin);
			}
			PxU32 size() const
			{
				return mBegin ? static_cast<PxU32>(mEnd - mBegin) : 0;
			}
			PxU32 capacity() const
			{
				return mBegin ? static_cast<PxU32>(mCapacityEnd - mBegin) : 0;
			}
			PxU8* begin()
			{
				return mBegin;
			}
			PxU8* end()
			{
				return mEnd;
			}
			const PxU8* begin() const
			{
				return mBegin;
			}
			const PxU8* end() const
			{
				return mEnd;
			}
			void clear()
			{
				mEnd = mBegin;
			}
			const char* cStr()
			{
				if(mEnd && (*mEnd != 0))
					write((PxU8)0);
				return reinterpret_cast<const char*>(mBegin);
			}
			PxU32 write(PxU8 inValue)
			{
				PxU8* target = growBuf(1);
				if(!target)
					return 0;
				*target = inValue;
				return 1;
			}

			template <typename TDataType>
			PxU32 write(const TDataType& inValue)
			{
				const PxU8* __restrict readPtr = reinterpret_cast<const PxU8*>(&inValue);
				PxU8* __restrict writePtr = growBuf(sizeof(TDataType));
				if(!writePtr)
					return 0;
				for(PxU32 idx = 0; idx < sizeof(TDataType); ++idx)
					writePtr[idx] = readPtr[idx];
				return sizeof(TDataType);
			}

			template <typename TDataType>
            PxU32 write(const TDataType* inValue, PxU64 inLength)
			{
				if(inLength > static_cast<PxU64>(0xffffffffu) / sizeof(TDataType))
					return 0;
				const PxU64 requestedSize = inLength * sizeof(TDataType);
				PxU32 writeSize = static_cast<PxU32>(requestedSize);
				if(inValue && inLength)
				{
					PxU8* target = growBuf(writeSize);
					if(!target)
						return 0;
					PxMemCopy(target, inValue, writeSize);
				}
				if(inLength && !inValue)
				{
					PX_ASSERT(false);
					// You can't not write something, because that will cause
					// the receiving end to crash.
					PxU8* target = growBuf(writeSize);
					if(!target)
						return 0;
					PxMemZero(target, writeSize);
				}
				return writeSize;
			}

            PxU8* growBuf(PxU64 inAmount)
			{
				if(inAmount == 0)
					return mEnd;
				PxU32 offset = size();
				if(inAmount > static_cast<PxU64>(0xffffffffu) - offset)
					return NULL;
				PxU32 newSize = offset + static_cast<PxU32>(inAmount);
				reserve(newSize);
				if(newSize > capacity())
					return NULL;
				mEnd += static_cast<PxU32>(inAmount);
				return mBegin + offset;
			}
			void writeZeros(PxU32 inAmount)
			{
				PxU32 offset = size();
				if(!growBuf(inAmount))
					return;
				physx::PxMemZero(begin() + offset, inAmount);
			}
			void reserve(PxU32 newSize)
			{
				PxU32 currentSize = size();
				if(newSize && newSize >= capacity())
				{
					PxU64 grownSize = newSize > 8192
						? static_cast<PxU64>(newSize) + newSize / 5
						: static_cast<PxU64>(newSize) * 2;
					PxU32 newDataSize = grownSize <= static_cast<PxU64>(0xffffffffu)
						? static_cast<PxU32>(grownSize) : newSize;
					PxU8* newData = static_cast<PxU8*>(TAllocator::allocate(newDataSize, __FILE__, __LINE__));
					if(!newData)
						return;
					if(mBegin)
					{
						PxMemCopy(newData, mBegin, currentSize);
						TAllocator::deallocate(mBegin);
					}
					mBegin = newData;
					mEnd = mBegin + currentSize;
					mCapacityEnd = mBegin + newDataSize;
				}
			}
		};

		struct ForwardingMemoryBuffer : public RawMemoryBuffer<ForwardingAllocator>
		{
			ForwardingMemoryBuffer(PxAllocatorCallback& callback, const char* bufDataName)
				: RawMemoryBuffer<ForwardingAllocator>(ForwardingAllocator(callback, bufDataName))
			{
			}

			ForwardingMemoryBuffer& operator<<(const char* inString)
			{
				if(inString && *inString)
				{
					PxU32 len(strLen(inString));
					write(inString, len);
				}
				return *this;
			}

			template <typename TDataType>
			inline ForwardingMemoryBuffer& toStream(const char* inFormat, const TDataType inData)
			{
				char buffer[128] = { 0 };
                nvidia::shdfnd::snprintf(buffer, 128, inFormat, inData);
				*this << buffer;
				return *this;
			}

			inline ForwardingMemoryBuffer& operator<<(bool inData)
			{
				*this << (inData ? "true" : "false");
				return *this;
			}
			inline ForwardingMemoryBuffer& operator<<(PxI32 inData)
			{
				return toStream("%d", inData);
			}
			inline ForwardingMemoryBuffer& operator<<(PxU16 inData)
			{
				return toStream("%u", (PxU32)inData);
			}
			inline ForwardingMemoryBuffer& operator<<(PxU8 inData)
			{
				return toStream("%u", (PxU32)inData);
			}
			inline ForwardingMemoryBuffer& operator<<(char inData)
			{
				return toStream("%c", inData);
			}
			inline ForwardingMemoryBuffer& operator<<(PxU32 inData)
			{
				return toStream("%u", inData);
			}
			inline ForwardingMemoryBuffer& operator<<(PxU64 inData)
			{
				return toStream("%I64u", inData);
			}
			inline ForwardingMemoryBuffer& operator<<(PxI64 inData)
			{
				return toStream("%I64d", inData);
			}
			inline ForwardingMemoryBuffer& operator<<(const void* inData)
			{
				return *this << static_cast<PxU64>(reinterpret_cast<size_t>(inData));
			}
			inline ForwardingMemoryBuffer& operator<<(PxF32 inData)
			{
				return toStream("%g", (PxF64)inData);
			}
			inline ForwardingMemoryBuffer& operator<<(PxF64 inData)
			{
				return toStream("%g", inData);
			}
			inline ForwardingMemoryBuffer& operator<<(const PxVec3& inData)
			{
				*this << inData[0];
				*this << " ";
				*this << inData[1];
				*this << " ";
				*this << inData[2];
				return *this;
			}

			inline ForwardingMemoryBuffer& operator<<(const PxQuat& inData)
			{
				*this << inData.x;
				*this << " ";
				*this << inData.y;
				*this << " ";
				*this << inData.z;
				*this << " ";
				*this << inData.w;
				return *this;
			}

			inline ForwardingMemoryBuffer& operator<<(const PxTransform& inData)
			{
				*this << inData.q;
				*this << " ";
				*this << inData.p;
				return *this;
			}

			inline ForwardingMemoryBuffer& operator<<(const PxBounds3& inData)
			{
				*this << inData.minimum;
				*this << " ";
				*this << inData.maximum;
				return *this;
			}

			inline ForwardingMemoryBuffer& operator<<(const PxFilterData& inData)
			{
				*this << inData.word0 << " " << inData.word1 << " " << inData.word2 << " " << inData.word3;
				return *this;
			}
		};

		template <typename TObjectType, typename TGetSetIndexOp, typename TSetSetIndexOp>
		class InvasiveSet
		{
			ForwardingArray<TObjectType*> mSet;

			InvasiveSet(const InvasiveSet& other);
			InvasiveSet& operator=(const InvasiveSet& other);

		public:
			InvasiveSet(PxAllocatorCallback& callback, const char* allocName) : mSet(callback, allocName)
			{
			}

			bool insert(TObjectType& inObject)
			{
				PxU32 currentIdx = TGetSetIndexOp()(inObject);
				if(currentIdx == PX_MAX_U32)
				{
					TSetSetIndexOp()(inObject, mSet.size());
					mSet.pushBack(&inObject);
					return true;
				}
				return false;
			}

			bool remove(TObjectType& inObject)
			{
				PxU32 currentIdx = TGetSetIndexOp()(inObject);
				if(currentIdx != PX_MAX_U32)
				{
					TObjectType* theEnd = mSet.back();
					TObjectType* theObj = &inObject;
					if(theEnd != theObj)
					{
						TSetSetIndexOp()(*theEnd, currentIdx);
						mSet[currentIdx] = theEnd;
					}
					mSet.popBack();
					TSetSetIndexOp()(inObject, PX_MAX_U32);
					return true;
				}
				return false;
			}

			bool contains(TObjectType& inObject)
			{
				return TGetSetIndexOp()(inObject) != PX_MAX_U32;
			}

			void clear()
			{
				for(PxU32 idx = 0; idx < mSet.size(); ++idx)
					TSetSetIndexOp()(*(mSet[idx]), PX_MAX_U32);
				mSet.clear();
			}

			TObjectType* operator[](PxU32 idx)
			{
				return mSet[idx];
			}
			const TObjectType* operator[](PxU32 idx) const
			{
				return mSet[idx];
			}
			PxU32 size() const
			{
				return mSet.size();
			}
			TObjectType** begin()
			{
				return mSet.begin();
			}
			TObjectType** end()
			{
				return mSet.end();
			}
			const TObjectType** begin() const
			{
				return mSet.begin();
			}
			const TObjectType** end() const
			{
				return mSet.end();
			}
			const TObjectType* back() const
			{
				return mSet.back();
			}
			TObjectType* back()
			{
				return mSet.back();
			}
		};

		template <typename TDataType>
		inline TDataType* PvdAllocate(PxAllocatorCallback& allocator, const char* typeName, const char* file, int line)
		{
			return reinterpret_cast<TDataType*>(allocator.allocate(sizeof(TDataType), typeName, file, line));
		}

		template <typename TDataType>
		inline void PvdDeleteAndDeallocate(PxAllocatorCallback& allocator, TDataType* inDType)
		{
			if(inDType)
			{
				inDType->~TDataType();
				allocator.deallocate(inDType);
			}
		}
	}
}

#define PVD_NEW(allocator, dtype) new (PvdAllocate<dtype>(allocator, #dtype, __FILE__, __LINE__)) dtype
#define PVD_DELETE(allocator, obj) PvdDeleteAndDeallocate(allocator, obj);
// Construct containers that tell us what they contain.
#define PVD_CONSTRUCT_CONTAINER(name, allocator, type) name(allocator, #type)
#define PVD_FOREACH(varname, stop) for(PxU32 varname = 0; varname < stop; ++varname)

namespace physx
{
	namespace pvdsdk
	{

		template <typename TKeyType, typename TValueType, typename THashType, typename TBufType, typename TOperator>
		PxU32 getMapKeysOp(ForwardingHashMap<TKeyType, TValueType, THashType>& map, TBufType* buffer, PxU32 bufSize,
			PxU32 startIdx, TOperator op)
		{
			PxU32 numItems = static_cast<PxU32>(map.size());
			if(numItems == 0 || bufSize == 0)
				return 0;

			startIdx = PxMin(numItems - 1, startIdx);
			PxU32 retval = 0;
			for(typename ForwardingHashMap<TKeyType, TValueType, THashType>::Iterator iter = map.getIterator();
				iter.done() == false && bufSize; ++iter)
			{
				if(startIdx)
					--startIdx;
				else
				{
					buffer[retval] = op(iter->first);
					--bufSize;
					++retval;
				}
			}
			return retval;
		}

		struct IdOp
		{
			template <typename TDataType>
			TDataType operator()(const TDataType& item)
			{
				return item;
			}
		};

		template <typename TKeyType, typename TValueType, typename THashType>
		PxU32 getMapKeys(ForwardingHashMap<TKeyType, TValueType, THashType>& map, TKeyType* buffer, PxU32 bufSize, PxU32 startIdx)
		{
			return getMapKeysOp(map, buffer, bufSize, startIdx, IdOp());
		}

		struct DerefOp
		{
			template <typename TDataType>
			TDataType operator()(const TDataType* item)
			{
				return *item;
			}
		};

		template <typename TKeyType, typename TValueType, typename TBufType, typename TOp>
		PxU32 getMapValues(ForwardingHashMap<TKeyType, TValueType>& map, TBufType* buffer, PxU32 bufSize, PxU32 startIdx, TOp op)
		{
			PxU32 numItems = static_cast<PxU32>(map.size());
			if(numItems == 0 || bufSize == 0)
				return 0;

			startIdx = PxMin(numItems - 1, startIdx);
			PxU32 retval = 0;
			for(typename ForwardingHashMap<TKeyType, TValueType>::Iterator iter = map.getIterator();
				iter.done() == false && bufSize; ++iter)
			{
				if(startIdx)
					--startIdx;
				else
				{
					buffer[retval] = op(iter->second);
					--bufSize;
					++retval;
				}
			}
			return retval;
		}

		template <typename TValueType, typename TBufType>
		PxU32 getArrayEntries(ForwardingArray<TValueType>& data, TBufType* buffer, PxU32 bufSize, PxU32 startIdx)
		{
			PxU32 numItems = static_cast<PxU32>(data.size());
			if(numItems == 0 || bufSize == 0)
				return 0;

			startIdx = PxMin(numItems - 1, startIdx);
			PxU32 available = PxMin(numItems - startIdx, bufSize);
			PVD_FOREACH(idx, available)
				buffer[idx] = data[idx + startIdx];
			return available;
		}
	}
}

#endif
