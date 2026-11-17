/*
 * SPDX-FileCopyrightText: Copyright (c) 2008-2014 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Copyright (c) 2008-2014 NVIDIA Corporation. All rights reserved.
#ifndef PVD_SIMPLE_ALLOCATOR_H
#define PVD_SIMPLE_ALLOCATOR_H
#include "PsMutex.h"
#include "PsString.h"
#include "PvdSimpleFoundation.h"
#include "PxPvdFoundation.h"
#include "foundation/PxAllocatorCallback.h"

namespace pvdapp
{
using namespace physx::shdfnd;

// The simplest allocator.  Performs no alignment nor memory checking.
struct MallocAllocator : public PxAllocatorCallback
{
	virtual void* allocate(size_t size, const char*, int) { return malloc(size); }
	virtual void* allocate(size_t size, const char*, const char*, int) { return malloc(size); }
	virtual void deallocate(void* ptr) { free(ptr); }
};

struct MemInfo
{
	MemInfo* prev;
	MemInfo* next;
	void* allocator;
	size_t size;
	void* originalAddress;
	const char* name;
	const char* file;
	int line;
	MemInfo(void* _allocator, size_t _size = 0, void* addr = 0, const char* _name = "", const char* _file = "", int _line = 0)
	: prev(0), next(0), allocator(_allocator), size(_size), originalAddress(addr), name(_name), file(_file), line(_line)
	{
	}
};

PX_COMPILE_TIME_ASSERT(0 == (15 & sizeof(MemInfo)));

template <typename TBaseAllocater>
class CheckingAllocatorT : public PxAllocatorCallback
{
  public:
  private:
	MemInfo* mHead;
	MemInfo* mTail;
	TBaseAllocater mAlloc;
	MutexT<TBaseAllocater> mMutex;

  public:
	CheckingAllocatorT() : mHead(NULL), mTail(NULL)
	//, mMutex( "MutexT" )
	{
	}
	virtual ~CheckingAllocatorT()
	{
		if(NULL != mHead || NULL != mTail)
		{
			printMemInfo("CheckingAllocator: Memory leak found\n");
			// PX_ASSERT(false);
		}
	}

	size_t getTotalBytesAllocated() const
	{
		size_t totalBytesAllocated = 0;
		for(MemInfo* info = mTail; info != NULL; info = info->prev)
			totalBytesAllocated += info->size;
		return totalBytesAllocated;
	}

	void printMemInfo(const char* prompt)
	{
		printString(prompt);
#if PX_DEBUG
		char buf[1024];
		size_t totalBytesAllocated = 0;
		PxU32 i = 0;
		for(MemInfo* info = mTail; info != NULL; info = info->prev, ++i)
		{
			if(this != info->allocator)
			{
				printString("CheckingAllocator: Internal data structure corrupted\n");
				PX_ASSERT(false);
				return;
			}

			totalBytesAllocated += info->size;

			if(i < 100)
			{
				physx::shdfnd::snprintf(buf, sizeof(buf) - 1, "[%2u] %8u bytes, tag \"%s\", \"%s:%d\"\n", i, info->size, info->name, info->file, info->line);
				printString(buf);
			}
		}
		physx::shdfnd::snprintf(buf, sizeof(buf) - 1, "Totally %u bytes allocated in %u allocation\n", totalBytesAllocated, i);
		printString(buf);
#endif
	}

	virtual void* allocate(size_t size, const char* tn, const char* fl, int ln)
	{
		PX_ASSERT(size);
		if(size)
		{
			size_t actualSize = 15 + sizeof(MemInfo) + size;
			typename MutexT<TBaseAllocater>::ScopedLock locker(mMutex);
			void* original = mAlloc.allocate(actualSize, tn, fl, ln);
			if(!original)
			{
				printMemInfo("CheckingAllocator: No enough memory\n");
			}
			else
			{
				size_t temp = (size_t)original;
				temp = (temp + 15) & (~15);
				void* retval = (void*)temp;
				MemInfo* info = new(retval) MemInfo(this, size, original, tn, fl, ln);

				if(NULL == mTail)
					mHead = info;
				else
					mTail->next = info;
				info->prev = mTail;
				mTail = info;

				return info + 1;
			}
		}
		return NULL;
	}

	virtual void deallocate(void* ptr)
	{
		if(ptr)
		{
			typename MutexT<TBaseAllocater>::ScopedLock locker(mMutex);

			MemInfo* info = reinterpret_cast<MemInfo*>(ptr) - 1;
			if(this != info->allocator)
			{
				printMemInfo("CheckingAllocator: Suspicious pointer deallocation\n");
			}
			else
			{
				MemInfo* prev = info->prev;
				MemInfo* next = info->next;
				if(prev)
					prev->next = next;
				else
					mHead = next;
				if(next)
					next->prev = prev;
				else
					mTail = prev;

				mAlloc.deallocate(info->originalAddress);
			}
		}
	}
};

typedef CheckingAllocatorT<MallocAllocator> CheckingAllocator;
} // namespace pvdapp
#endif
