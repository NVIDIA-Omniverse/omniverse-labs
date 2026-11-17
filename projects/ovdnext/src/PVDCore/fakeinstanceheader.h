/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef FAKEINSTANCEHEADER_H
#define FAKEINSTANCEHEADER_H

#include "PxPvdFoundation.h"
#include <PxSimpleTypes.h>

struct FakeInstanceHeader
{
	int32_t mId;
	int32_t mClassId;
	pvdapp::NamespacedName mClassName;
	uint32_t mDataSize;
	uint32_t mAliveSetIndex;
	uint32_t mUpdateSetIndex;
	uint32_t mPinnedSetIndex;
	// The offset in the stream where this object was last saved
	// This value is set to an invalid number (PxU64)-1 every time
	// is changed.
	uint64_t mDataSectionStart;

	// Unfortunately, because classes can be modified after they are created
	// we may have less data in our data section than the classes published size.
	FakeInstanceHeader(int32_t id, int32_t classId, const pvdapp::NamespacedName& nm, uint32_t dataSize)
	: mId(id), mClassId(classId), mClassName(nm), mDataSize(dataSize), mAliveSetIndex(PX_MAX_U32), mUpdateSetIndex(PX_MAX_U32), mPinnedSetIndex(PX_MAX_U32)
	{
		mDataSectionStart = PVD_POINTER_TO_U64(pointerAfterThis());
	}

	uint8_t* pointerAfterThis() { return reinterpret_cast<uint8_t*>(this) + sizeof(*this); }

	bool isAlive() const { return mAliveSetIndex != PX_MAX_U32; }
	bool isUpdated() const { return mUpdateSetIndex != PX_MAX_U32; }
	bool isPinned() const { return mPinnedSetIndex != PX_MAX_U32; }

	bool isReferenced() const { return isAlive() || isUpdated() || isPinned(); }

	uint8_t* getDataSection() { return reinterpret_cast<uint8_t*>(static_cast<size_t>(mDataSectionStart)); }

	bool isPartOf(uint8_t* ptr) { return ptr == pointerAfterThis(); }

	//                            operator FakeInstanceHeader() const
	//                            {
	//                                return FakeInstanceHeader(mId, mClassId,
	// const_cast<void*>(reinterpret_cast<const void*>(this)), isAlive());
	//                            }
};

#endif // FAKEINSTANCEHEADER_H
