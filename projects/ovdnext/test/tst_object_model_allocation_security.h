/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "PxPvdObjectModel.h"
#include "PxPvdObjectModelMetaData.h"

#include <cstring>
#include <limits>

namespace object_model_allocation_security
{

class FailNamedAllocation final : public physx::PxAllocatorCallback
{
	physx::PxAllocatorCallback& mDelegate;
	const char* mAllocationName;
	bool mArmed;

  public:
	explicit FailNamedAllocation(physx::PxAllocatorCallback& delegate)
	: mDelegate(delegate), mAllocationName(nullptr), mArmed(false)
	{
	}

	void failNext(const char* allocationName)
	{
		mAllocationName = allocationName;
		mArmed = true;
	}

	virtual void* allocate(size_t size, const char* typeName, const char* filename, int line)
	{
		if(mArmed && typeName && mAllocationName && std::strcmp(typeName, mAllocationName) == 0)
		{
			mArmed = false;
			return nullptr;
		}
		return mDelegate.allocate(size, typeName, filename, line);
	}

	virtual void deallocate(void* ptr) { mDelegate.deallocate(ptr); }
};

} // namespace object_model_allocation_security

TEST_F(PvdEngineTest, ObjectModelRejectsOversizedInstanceData)
{
	using namespace physx::pvdsdk;
	PvdObjectModelMetaData& metadata = PvdObjectModelMetaData::create();
	const ClassDescription holder = metadata.getOrCreateClass(
		NamespacedName("security", "OversizedInstance"));
	ClassDescription* mutableHolder = metadata.getClassPtr(holder.mClassId);
	ASSERT_NE(mutableHolder, nullptr);
	mutableHolder->get32BitSize() = 64u * 1024u * 1024u + 1u;

	PvdObjectModel& model = PvdObjectModel::create(*m_alloc, metadata);
	const InstanceDescription instance = model.createInstance(holder.mClassId);
	EXPECT_EQ(instance.mInstPtr, nullptr);
	model.release();
}

TEST_F(PvdEngineTest, ObjectModelPreservesInstanceWhenResizeAllocationFails)
{
	using namespace physx::pvdsdk;
	object_model_allocation_security::FailNamedAllocation allocator(*m_alloc);
	PvdObjectModelMetaData& metadata = PvdObjectModelMetaData::create();
	const ClassDescription holder = metadata.getOrCreateClass(
		NamespacedName("security", "ResizeFailureInstance"));
	const Option<ClassDescription> u32Type = metadata.findClass<PxU32>();
	ASSERT_TRUE(u32Type.hasValue());
	const Option<PropertyDescription> first = metadata.createProperty(
		holder.mClassId, "first", "", u32Type->mClassId);
	ASSERT_TRUE(first.hasValue());
	const Option<PropertyDescription> second = metadata.createProperty(
		holder.mClassId, "second", "", u32Type->mClassId);
	ASSERT_TRUE(second.hasValue());
	ClassDescription* mutableHolder = metadata.getClassPtr(holder.mClassId);
	ASSERT_NE(mutableHolder, nullptr);
	mutableHolder->get32BitSize() = sizeof(PxU32);

	PvdObjectModel& model = PvdObjectModel::create(allocator, metadata);
	const InstanceDescription instance = model.createInstance(holder.mClassId);
	ASSERT_NE(instance.mInstPtr, nullptr);
	const PxU32 sentinel = 0x4d3c2b1au;
	ASSERT_TRUE(model.setPropertyValue(instance.mInstPtr, first->mPropertyId,
		reinterpret_cast<const PxU8*>(&sentinel), sizeof(sentinel), -1));

	mutableHolder->get32BitSize() = sizeof(PxU32) * 2u;
	allocator.failNext("PvdObjectModelImpl::InstanceData");
	const PxU32 replacement = 0xabcdef01u;
	EXPECT_FALSE(model.setPropertyValue(instance.mInstPtr, second->mPropertyId,
		reinterpret_cast<const PxU8*>(&replacement), sizeof(replacement), -1));

	PxU32 preserved = 0;
	EXPECT_EQ(sizeof(preserved), model.getPropertyValue(instance.mInstPtr,
		first->mPropertyId, reinterpret_cast<PxU8*>(&preserved), sizeof(preserved)));
	EXPECT_EQ(sentinel, preserved);
	model.release();
}

TEST_F(PvdEngineTest, ObjectModelLocksClassAfterInstanceCreation)
{
	using namespace physx::pvdsdk;
	PvdObjectModelMetaData& metadata = PvdObjectModelMetaData::create();
	const ClassDescription holder = metadata.getOrCreateClass(
		NamespacedName("security", "LockedInstanceLayout"));
	const Option<ClassDescription> u32Type = metadata.findClass<PxU32>();
	ASSERT_TRUE(u32Type.hasValue());

	PvdObjectModel& model = PvdObjectModel::create(*m_alloc, metadata);
	const InstanceDescription instance = model.createInstance(holder.mClassId);
	ASSERT_NE(instance.mInstPtr, nullptr);
	const ClassDescription* lockedHolder = metadata.getClassPtr(holder.mClassId);
	ASSERT_NE(lockedHolder, nullptr);
	EXPECT_TRUE(lockedHolder->mLocked);
	model.release();
}

TEST_F(PvdEngineTest, ObjectModelHandlesInstanceAllocationFailure)
{
	using namespace physx::pvdsdk;
	object_model_allocation_security::FailNamedAllocation allocator(*m_alloc);
	PvdObjectModelMetaData& metadata = PvdObjectModelMetaData::create();
	const ClassDescription holder = metadata.getOrCreateClass(
		NamespacedName("security", "AllocationFailureInstance"));
	PvdObjectModel& model = PvdObjectModel::create(allocator, metadata);

	allocator.failNext("PvdObjectModel::Instance");
	const InstanceDescription instance = model.createInstance(holder.mClassId);
	EXPECT_EQ(instance.mInstPtr, nullptr);
	model.release();
}

TEST_F(PvdEngineTest, ObjectModelRejectsCrossClassInstanceReuse)
{
	using namespace physx::pvdsdk;
	PvdObjectModelMetaData& metadata = PvdObjectModelMetaData::create();
	const ClassDescription firstClass = metadata.getOrCreateClass(
		NamespacedName("security", "ReuseClassA"));
	const ClassDescription secondClass = metadata.getOrCreateClass(
		NamespacedName("security", "ReuseClassB"));
	PvdObjectModel& model = PvdObjectModel::create(*m_alloc, metadata);
	const InstanceDescription original = model.createInstance(firstClass.mClassId, 17);
	ASSERT_NE(original.mInstPtr, nullptr);

	const InstanceDescription replacement = model.createInstance(secondClass.mClassId, 17);
	EXPECT_EQ(replacement.mInstPtr, nullptr);
	EXPECT_EQ(model.idToDescriptor(17).mClassId, firstClass.mClassId);
	EXPECT_EQ(model.idToPtr(17), original.mInstPtr);
	model.release();
}

TEST_F(PvdEngineTest, ObjectModelRejectsExhaustedInstanceHandles)
{
	using namespace physx::pvdsdk;
	PvdObjectModelMetaData& metadata = PvdObjectModelMetaData::create();
	const ClassDescription holder = metadata.getOrCreateClass(
		NamespacedName("security", "HandleOverflowInstance"));
	PvdObjectModel& model = PvdObjectModel::create(*m_alloc, metadata);
	const PxI32 maximumHandle = (std::numeric_limits<PxI32>::max)();

	EXPECT_EQ(model.createInstance(holder.mClassId, maximumHandle).mInstPtr, nullptr);
	model.setNextInstanceHandleValue(maximumHandle - 1);
	EXPECT_EQ(model.createInstance(holder.mClassId).mInstPtr, nullptr);
	EXPECT_EQ(model.idToPtr(maximumHandle), nullptr);
	model.release();
}

TEST_F(PvdEngineTest, ObjectModelRejectsOutOfRangeArrayRemoval)
{
	using namespace physx::pvdsdk;
	PvdObjectModelMetaData& metadata = PvdObjectModelMetaData::create();
	const ClassDescription holder = metadata.getOrCreateClass(
		NamespacedName("security", "ArrayRemovalInstance"));
	const Option<ClassDescription> u32Type = metadata.findClass<PxU32>();
	ASSERT_TRUE(u32Type.hasValue());
	const Option<PropertyDescription> values = metadata.createProperty(
		holder.mClassId, "values", "", u32Type->mClassId, PropertyType::Array);
	ASSERT_TRUE(values.hasValue());

	PvdObjectModel& model = PvdObjectModel::create(*m_alloc, metadata);
	const InstanceDescription instance = model.createInstance(holder.mClassId);
	ASSERT_NE(instance.mInstPtr, nullptr);
	const PxU32 initialValues[] = { 0x12345678u, 0xabcdef01u };
	ASSERT_TRUE(model.setPropertyValue(instance.mInstPtr, values->mPropertyId,
		reinterpret_cast<const PxU8*>(initialValues), sizeof(initialValues), -1));

	EXPECT_FALSE(model.removeArrayElement(instance.mInstPtr, values->mPropertyId,
		(std::numeric_limits<PxI32>::max)()));
	EXPECT_EQ(model.getNbArrayElements(instance.mInstPtr, values->mPropertyId), 2u);
	PxU32 preserved[2] = {};
	EXPECT_EQ(model.getPropertyValue(instance.mInstPtr, values->mPropertyId,
		reinterpret_cast<PxU8*>(preserved), sizeof(preserved)), sizeof(preserved));
	EXPECT_EQ(preserved[0], initialValues[0]);
	EXPECT_EQ(preserved[1], initialValues[1]);
	model.release();
}

TEST_F(PvdEngineTest, ObjectModelOriginShiftAcceptsNullInstance)
{
	using namespace physx::pvdsdk;
	PvdObjectModelMetaData& metadata = PvdObjectModelMetaData::create();
	PvdObjectModel& model = PvdObjectModel::create(*m_alloc, metadata);
	model.originShift(nullptr, PxVec3(1.0f, 2.0f, 3.0f));
	model.release();
}
