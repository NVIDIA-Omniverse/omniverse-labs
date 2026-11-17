/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "TestFiles.h"
#pragma once

#include "PvdInputValidation.h"
#include "PvdRandomAccessIOStream.h"
#include "PxPvdFoundation.h"
#include "PxPvdObjectModel.h"
#include "PxPvdObjectModelMetaData.h"


#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace
{

class VirtualLengthRandomAccessStream : public pvdapp::PvdRandomAccessIOStream
{
  public:
    explicit VirtualLengthRandomAccessStream(physx::PxU64 length)
    : mLength(length), mLastOffset(0), mLastLength(0), mReadCount(0)
    {
    }

    bool write(const physx::PxU8*, physx::PxU32) override { return false; }
    bool directCopy(physx::pvdsdk::PvdInputStream&, physx::PxU32) override
    {
        return false;
    }
    bool directCopy(physx::pvdsdk::PvdInputStream&, physx::PxU64) override
    {
        return false;
    }
    void addRef() override {}
    void release() override {}
    physx::pvdsdk::Option<pvdapp::ErrorCode> read(
        physx::PxU8* buffer, physx::PxU32 length, physx::PxU64 offset) override
    {
        mLastOffset = offset;
        mLastLength = length;
        ++mReadCount;
        std::fill(buffer, buffer + length, static_cast<physx::PxU8>(0x5a));
        return physx::pvdsdk::None();
    }
    void zeroPadToPageSize(physx::PxU32) override {}
    physx::PxU64 getLength() const override { return mLength; }
    physx::pvdsdk::Option<pvdapp::ErrorCode> flush() override
    {
        return physx::pvdsdk::None();
    }
    physx::pvdsdk::Option<pvdapp::ErrorCode> asyncFlush() override
    {
        return physx::pvdsdk::None();
    }
    double getTimeCost() override { return 0.0; }

    physx::PxU64 mLength;
    physx::PxU64 mLastOffset;
    physx::PxU32 mLastLength;
    physx::PxU32 mReadCount;
};

} // namespace

TEST(PvdRandomAccessIOStreamTest, PreservesRemainingLengthAboveFourGiB)
{
    VirtualLengthRandomAccessStream source((physx::PxU64(1) << 32) + 3);
    pvdapp::IOStreamToInputStream input(0, source);
    physx::PxU8 buffer[8] = {};
    physx::PxU32 length = sizeof(buffer);

    ASSERT_TRUE(input.read(buffer, length));
    EXPECT_EQ(sizeof(buffer), length);
    EXPECT_EQ(sizeof(buffer), source.mLastLength);
    EXPECT_EQ(0u, source.mLastOffset);
    EXPECT_EQ(1u, source.mReadCount);
    EXPECT_EQ(physx::PxU64(sizeof(buffer)), input.mOffset);
}

TEST(PvdRandomAccessIOStreamTest, TreatsOffsetPastCachedLengthAsEof)
{
    VirtualLengthRandomAccessStream source(5);
    pvdapp::IOStreamToInputStream input(6, source);
    physx::PxU8 buffer[4] = { 1, 2, 3, 4 };
    physx::PxU32 length = sizeof(buffer);

    ASSERT_TRUE(input.read(buffer, length));
    EXPECT_EQ(0u, length);
    EXPECT_EQ(0u, source.mReadCount);
    for(physx::PxU8 value : buffer)
        EXPECT_EQ(0u, value);
}

TEST(PvdInputValidationTest, RejectsOversizedEventGroupsAndQueueGrowth)
{
    EXPECT_TRUE(pvdapp::input::isEventGroupDataSizeAllowed(0));
    EXPECT_TRUE(pvdapp::input::isEventGroupDataSizeAllowed(pvdapp::input::kMaxEventGroupDataSize));
    EXPECT_FALSE(pvdapp::input::isEventGroupDataSizeAllowed(pvdapp::input::kMaxEventGroupDataSize + 1));

    EXPECT_TRUE(pvdapp::input::canQueueEventGroup(0, 0, pvdapp::input::kMaxEventGroupDataSize));
    EXPECT_TRUE(pvdapp::input::canQueueEventGroup(
        0, 0, pvdapp::input::kMaxBufferedEventGroupBytes));
    EXPECT_FALSE(pvdapp::input::canQueueEventGroup(
        0, 0, pvdapp::input::kMaxBufferedEventGroupBytes + 1));
    EXPECT_FALSE(pvdapp::input::canQueueEventGroup(
        pvdapp::input::kMaxQueuedEventBytes, 1, 1));
    EXPECT_FALSE(pvdapp::input::canQueueEventGroup(
        0, pvdapp::input::kMaxQueuedEventGroups, 1));
}

TEST(PvdInputValidationTest, ComputesArrayCapacityWithoutOverflow)
{
    uint32_t capacity = 0;
    EXPECT_TRUE(pvdapp::input::computeArrayAllocationSize(16, 64, capacity));
    EXPECT_GE(capacity, 64u);
    EXPECT_EQ(0u, capacity % 16u);

    EXPECT_TRUE(pvdapp::input::computeArrayAllocationSize(
        16, pvdapp::input::kMaxPropertyArrayBytes, capacity));
    EXPECT_EQ(pvdapp::input::kMaxPropertyArrayBytes, capacity);

    EXPECT_FALSE(pvdapp::input::computeArrayAllocationSize(0, 64, capacity));
    EXPECT_FALSE(pvdapp::input::computeArrayAllocationSize(16, 63, capacity));
    EXPECT_FALSE(pvdapp::input::computeArrayAllocationSize(
        1, pvdapp::input::kMaxPropertyArrayBytes + 1, capacity));
}

TEST(PvdInputValidationTest, RejectsPropertyCountAndAccumulatorOverflow)
{
	uint32_t product = 0;
	EXPECT_TRUE(pvdapp::input::checkedMultiplyToLimit(1024, 64,
		pvdapp::input::kMaxPropertyArrayBytes, product));
	EXPECT_EQ(65536u, product);
	EXPECT_FALSE(pvdapp::input::checkedMultiplyToLimit(
		std::numeric_limits<uint32_t>::max(), 2,
		pvdapp::input::kMaxPropertyArrayBytes, product));

	EXPECT_TRUE(pvdapp::input::canAccumulatePropertyData(0, 0, 64, 1));
	EXPECT_FALSE(pvdapp::input::canAccumulatePropertyData(
		pvdapp::input::kMaxAccumulatedPropertyBytes, 0, 1, 0));
	EXPECT_FALSE(pvdapp::input::canAccumulatePropertyData(
		0, pvdapp::input::kMaxPropertyItems, 0, 1));
}

TEST(PvdInputValidationTest, ZeroPadsShortStringHandles)
{
    const uint8_t one[] = { 0x78 };
    const uint8_t two[] = { 0x78, 0x56 };
    const uint8_t three[] = { 0x78, 0x56, 0x34 };
    const uint8_t four[] = { 0x78, 0x56, 0x34, 0x12 };

    EXPECT_EQ(0x00000078u, pvdapp::input::readPaddedU32(one, sizeof(one)));
    EXPECT_EQ(0x00005678u, pvdapp::input::readPaddedU32(two, sizeof(two)));
    EXPECT_EQ(0x00345678u, pvdapp::input::readPaddedU32(three, sizeof(three)));
    EXPECT_EQ(0x12345678u, pvdapp::input::readPaddedU32(four, sizeof(four)));
    EXPECT_EQ(0u, pvdapp::input::readPaddedU32(nullptr, 0));
}

TEST(PvdInputValidationTest, GeneratesUniqueBackingPathsWithoutFixedBuffers)
{
    const auto rootPath = test_files::fs::temp_directory_path() / std::string(1200, 'a');
    const std::string root = rootPath.u8string();
    ASSERT_TRUE(rootPath.is_absolute());
    const std::string first = pvdapp::input::temporaryBackingFileCandidate(root.c_str());
    const std::string second = pvdapp::input::temporaryBackingFileCandidate(root.c_str());

    EXPECT_NE(first, second);
    EXPECT_EQ(test_files::fs::u8path(first).parent_path(), rootPath);
    EXPECT_EQ(test_files::fs::u8path(first).extension(), ".pxd3");
    EXPECT_GT(first.size(), 1024);
}

TEST(PvdArchiveSecurityTest, KeepsFreeSpaceInReserveWithoutCappingLargeCaptures)
{
    const uint64_t gib = 1024ull * 1024ull * 1024ull;
    EXPECT_TRUE(pvdapp::input::hasArchiveExtractionSpace(3 * gib, 4 * gib));
    EXPECT_FALSE(pvdapp::input::hasArchiveExtractionSpace(4 * gib, 4 * gib));
    EXPECT_FALSE(pvdapp::input::hasArchiveExtractionSpace(4 * gib, 3 * gib));
    EXPECT_TRUE(pvdapp::input::hasArchiveExtractionSpace(0, gib));
}

TEST_F(PvdEngineTest, RawMemoryBufferRejectsWraparoundGrowth)
{
	physx::pvdsdk::ForwardingMemoryBuffer buffer("SecurityRegressionBuffer");
	EXPECT_EQ(0u, buffer.size());
	EXPECT_EQ(0u, buffer.capacity());
	EXPECT_EQ(1u, buffer.write(static_cast<uint8_t>(0x7f)));

    EXPECT_EQ(nullptr, buffer.growBuf(std::numeric_limits<uint32_t>::max()));
    EXPECT_EQ(1u, buffer.size());
}

TEST_F(PvdEngineTest, TemporaryStreamNeverTruncatesAnExistingPath)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    const std::string path = root.file("existing.pxd3");

    test_files::File sentinel(path);
    ASSERT_TRUE(sentinel.open(std::ios::out));
	ASSERT_EQ(15, sentinel.write("do not replace!"));
    sentinel.close();

    pvdapp::ErrorCodeOption<pvdapp::PvdRandomAccessIOStream*> result =
        pvdapp::PvdRandomAccessIOStream::open(*m_alloc, path.c_str(), true, true);
    EXPECT_FALSE(result.hasValue());
    if(result.hasValue())
        result.getValue()->release();

    ASSERT_TRUE(sentinel.open(std::ios::in));
	EXPECT_EQ(std::string("do not replace!"), sentinel.readAll());
}

TEST_F(PvdEngineTest, StreamLoadReleasesSourceAndTemporaryBackingOnDestroy)
{
	test_files::Directory root;
	ASSERT_TRUE(root.valid());
	const std::string source = root.file("capture.pxd2");
	ASSERT_TRUE(test_files::fs::copy_file(TEST_DATA_DIR "/apex_destruction.pxd2", test_files::fs::u8path(source)));
	pvdapp::CheckingAllocator* checkingAllocator =
		static_cast<pvdapp::CheckingAllocator*>(m_alloc);
	const size_t allocatedBeforeLoad = checkingAllocator->getTotalBytesAllocated();

	pvdapp::PvdEngine* engine = pvdapp::PvdEngine::load(
		*m_alloc, root.path().c_str(),
		source.c_str());
	ASSERT_NE(engine, nullptr);
	delete engine;

	EXPECT_TRUE(test_files::fs::remove(test_files::fs::u8path(source)));
	EXPECT_TRUE(test_files::fileNames(root.path()).empty());
	EXPECT_EQ(allocatedBeforeLoad, checkingAllocator->getTotalBytesAllocated());
}

TEST_F(PvdEngineTest, ObjectModelRejectsOversizedScalarAndArrayElementWrites)
{
	using namespace physx::pvdsdk;
	PvdObjectModelMetaData& metadata = PvdObjectModelMetaData::create();
	const ClassDescription holder = metadata.getOrCreateClass(
		NamespacedName("security", "ScalarWriteHolder"));
	const Option<ClassDescription> u32Type = metadata.findClass<PxU32>();
	ASSERT_TRUE(u32Type.hasValue());
	const Option<PropertyDescription> first =
		metadata.createProperty(holder.mClassId, "first", "", u32Type->mClassId);
	const Option<PropertyDescription> second =
		metadata.createProperty(holder.mClassId, "second", "", u32Type->mClassId);
	const Option<PropertyDescription> values =
		metadata.createProperty(holder.mClassId, "values", "", u32Type->mClassId,
			PropertyType::Array);
	ASSERT_TRUE(first.hasValue());
	ASSERT_TRUE(second.hasValue());
	ASSERT_TRUE(values.hasValue());
	metadata.lockClass(holder.mClassId);

	PvdObjectModel& model = PvdObjectModel::create(*m_alloc, metadata);
	const InstanceDescription instance = model.createInstance(holder.mClassId);
	ASSERT_NE(instance.mInstPtr, nullptr);

	const PxU32 sentinel = 0x13579bdfu;
	ASSERT_TRUE(model.setPropertyValue(instance.mInstPtr, second->mPropertyId,
		reinterpret_cast<const PxU8*>(&sentinel), sizeof(sentinel), -1));

	const PxU32 oversized[2] = { 0xaaaaaaaaU, 0xbbbbbbbbU };
	EXPECT_FALSE(model.setPropertyValue(instance.mInstPtr, first->mPropertyId,
		reinterpret_cast<const PxU8*>(oversized), sizeof(oversized), -1));
	EXPECT_FALSE(model.insertArrayElement(instance.mInstPtr, values->mPropertyId, -1,
		reinterpret_cast<const PxU8*>(oversized), sizeof(oversized), -1));

	PxU32 preserved = 0;
	EXPECT_EQ(sizeof(preserved), model.getPropertyValue(instance.mInstPtr,
		second->mPropertyId, reinterpret_cast<PxU8*>(&preserved), sizeof(preserved)));
	EXPECT_EQ(sentinel, preserved);
	EXPECT_EQ(0u, model.getNbArrayElements(instance.mInstPtr, values->mPropertyId));

	model.release();
}

TEST_F(PvdEngineTest, PropertyMessageMetadataRejectsWrappedEntryRange)
{
	using namespace physx::pvdsdk;
	PvdObjectModelMetaData& metadata = PvdObjectModelMetaData::create();
	const ClassDescription holder = metadata.getOrCreateClass(
		NamespacedName("security", "MessageWriteHolder"));
	const Option<ClassDescription> u32Type = metadata.findClass<PxU32>();
	ASSERT_TRUE(u32Type.hasValue());
	ASSERT_TRUE(metadata.createProperty(
		holder.mClassId, "value", "", u32Type->mClassId).hasValue());

	PropertyMessageArg argument("value", u32Type->mName,
		std::numeric_limits<uint32_t>::max() - 3u, sizeof(PxU32));
	const Option<PropertyMessageDescription> message = metadata.createPropertyMessage(
		holder.mName, NamespacedName("security", "wrappedMessage"),
		DataRef<PropertyMessageArg>(&argument, 1), 0);
	EXPECT_FALSE(message.hasValue());

	metadata.release();
}

TEST_F(PvdEngineTest, PropertyMessagesRejectMismatchedAndArrayTargets)
{
	using namespace physx::pvdsdk;
	PvdObjectModelMetaData& metadata = PvdObjectModelMetaData::create();
	const Option<ClassDescription> u32Type = metadata.findClass<PxU32>();
	ASSERT_TRUE(u32Type.hasValue());

	const ClassDescription wide = metadata.getOrCreateClass(
		NamespacedName("security", "WideMessageValue"));
	ASSERT_TRUE(metadata.createProperty(
		wide.mClassId, "low", "", u32Type->mClassId).hasValue());
	ASSERT_TRUE(metadata.createProperty(
		wide.mClassId, "high", "", u32Type->mClassId).hasValue());
	const Option<ClassDescription> wideType = metadata.findClass(wide.mName);
	ASSERT_TRUE(wideType.hasValue());

	const ClassDescription holder = metadata.getOrCreateClass(
		NamespacedName("security", "StrictMessageHolder"));
	ASSERT_TRUE(metadata.createProperty(
		holder.mClassId, "scalar", "", u32Type->mClassId).hasValue());
	ASSERT_TRUE(metadata.createProperty(
		holder.mClassId, "array", "", u32Type->mClassId,
		PropertyType::Array).hasValue());

	PropertyMessageArg mismatched("scalar", wideType->mName, 0,
		wideType->get32BitSize());
	EXPECT_FALSE(metadata.createPropertyMessage(
		holder.mName, NamespacedName("security", "mismatchedMessage"),
		DataRef<PropertyMessageArg>(&mismatched, 1),
		wideType->get32BitSize()).hasValue());

	PropertyMessageArg arrayTarget("array", u32Type->mName, 0, sizeof(PxU32));
	EXPECT_FALSE(metadata.createPropertyMessage(
		holder.mName, NamespacedName("security", "arrayTargetMessage"),
		DataRef<PropertyMessageArg>(&arrayTarget, 1), sizeof(PxU32)).hasValue());

	metadata.release();
}

TEST_F(PvdEngineTest, MetadataRejectsSelfAndCyclicDerivation)
{
	using namespace physx::pvdsdk;
	PvdObjectModelMetaData& metadata = PvdObjectModelMetaData::create();
	const NamespacedName classA("security", "CycleA");
	const NamespacedName classB("security", "CycleB");

	EXPECT_FALSE(metadata.deriveClass(classA, classA));
	EXPECT_TRUE(metadata.deriveClass(classA, classB));
	EXPECT_FALSE(metadata.deriveClass(classB, classA));

	metadata.release();
}

TEST_F(PvdEngineTest, MetadataRejectsDerivingAClassWithExistingLayout)
{
	using namespace physx::pvdsdk;
	PvdObjectModelMetaData& metadata = PvdObjectModelMetaData::create();
	const ClassDescription parent = metadata.getOrCreateClass(
		NamespacedName("security", "LateParent"));
	const ClassDescription child = metadata.getOrCreateClass(
		NamespacedName("security", "ConfiguredChild"));
	const Option<ClassDescription> u32Type = metadata.findClass<PxU32>();
	ASSERT_TRUE(u32Type.hasValue());
	ASSERT_TRUE(metadata.createProperty(
		child.mClassId, "existing", "", u32Type->mClassId).hasValue());

	EXPECT_FALSE(metadata.deriveClass(parent.mName, child.mName));
	metadata.release();
}

TEST_F(PvdEngineTest, InstantiationLocksClassLayout)
{
	using namespace physx::pvdsdk;
	PvdObjectModelMetaData& metadata = PvdObjectModelMetaData::create();
	const ClassDescription holder = metadata.getOrCreateClass(
		NamespacedName("security", "InstantiatedLayout"));
	const Option<ClassDescription> u32Type = metadata.findClass<PxU32>();
	ASSERT_TRUE(u32Type.hasValue());
	PvdObjectModel& model = PvdObjectModel::create(*m_alloc, metadata);
	const InstanceDescription instance = model.createInstance(holder.mClassId);
	ASSERT_NE(nullptr, instance.mInstPtr);

	EXPECT_FALSE(metadata.createProperty(
		holder.mClassId, "lateProperty", "", u32Type->mClassId).hasValue());
	model.release();
}

TEST_F(PvdEngineTest, ArrayPropertiesLockElementLayouts)
{
	using namespace physx::pvdsdk;
	PvdObjectModelMetaData& metadata = PvdObjectModelMetaData::create();
	const Option<ClassDescription> u32Type = metadata.findClass<PxU32>();
	ASSERT_TRUE(u32Type.hasValue());
	const ClassDescription element = metadata.getOrCreateClass(
		NamespacedName("security", "ArrayElementLayout"));
	ASSERT_TRUE(metadata.createProperty(
		element.mClassId, "first", "", u32Type->mClassId).hasValue());
	const ClassDescription holder = metadata.getOrCreateClass(
		NamespacedName("security", "ArrayElementHolder"));
	ASSERT_TRUE(metadata.createProperty(
		holder.mClassId, "elements", "", element.mClassId,
		PropertyType::Array).hasValue());

	EXPECT_FALSE(metadata.createProperty(
		element.mClassId, "late", "", u32Type->mClassId).hasValue());
	metadata.release();
}

TEST_F(PvdEngineTest, MetadataLimitsClassAncestryDepth)
{
	using namespace physx::pvdsdk;
	PvdObjectModelMetaData& metadata = PvdObjectModelMetaData::create();
	std::vector<std::string> classNames;
	classNames.reserve(65);
	classNames.push_back("Depth0");
	NamespacedName parent("security", classNames.back().c_str());
	metadata.getOrCreateClass(parent);

	for(PxU32 depth = 1; depth < 64; ++depth)
	{
		classNames.push_back(std::string("Depth") + std::to_string(depth));
		const NamespacedName child("security", classNames.back().c_str());
		ASSERT_TRUE(metadata.deriveClass(parent, child)) << "depth " << depth;
		parent = child;
	}

	const NamespacedName tooDeep("security", "Depth64");
	EXPECT_FALSE(metadata.deriveClass(parent, tooDeep));
	metadata.release();
}

TEST_F(PvdEngineTest, MetadataConvenienceOverloadsRejectUnknownClasses)
{
	using namespace physx::pvdsdk;
	PvdObjectModelMetaData& metadata = PvdObjectModelMetaData::create();
	const NamespacedName owner("security", "KnownOwner");
	metadata.getOrCreateClass(owner);

	EXPECT_FALSE(metadata.createProperty(
		NamespacedName("security", "MissingOwner"), "value", "",
		getPvdNamespacedNameForType<PxU32>()).hasValue());
	EXPECT_FALSE(metadata.createProperty(
		owner, "value", "", NamespacedName("security", "MissingType")).hasValue());

	metadata.release();
}

TEST_F(PvdEngineTest, ObjectModelRejectsUnknownClassWithoutDereference)
{
	using namespace physx::pvdsdk;
	PvdObjectModelMetaData& metadata = PvdObjectModelMetaData::create();
	PvdObjectModel& model = PvdObjectModel::create(*m_alloc, metadata);

	const InstanceDescription instance = model.createInstance(
		NamespacedName("security", "MissingInstanceClass"));
	EXPECT_EQ(nullptr, instance.mInstPtr);
	EXPECT_LT(instance.mId, 0);

	model.release();
}

#ifdef PVD_OVD_SUPPORT
TEST_F(OvdLoaderTest, OvdStringPropertiesExposeACompleteHandle)
{
	ASSERT_NE(sEngine, nullptr);
	const PxU32 instanceCount = sEngine->getNbInstances();
	std::vector<InstanceDescription> instances(instanceCount);
	sEngine->getInstances(instances.data(), instanceCount);

	PxU32 checkedNames = 0;
	for(const InstanceDescription& instance : instances)
	{
		Option<PropertyDescription> nameProperty =
			sEngine->findProperty(instance.mClassId, "name");
		if(!nameProperty.hasValue())
			continue;

		const PxU32 byteSize = sEngine->getPropertyByteSize(
			instance.mInstPtr, nameProperty->mPropertyId);
		if(byteSize == 0)
			continue;

		EXPECT_EQ(sizeof(PxU32), byteSize);
		PxU32 handle = 0;
		EXPECT_EQ(sizeof(PxU32), sEngine->getPropertyValue(
			instance.mInstPtr, nameProperty->mPropertyId, &handle, sizeof(handle)));
		EXPECT_NE(nullptr, sEngine->handleToString(handle));
		++checkedNames;
	}

	EXPECT_GT(checkedNames, 0u);
}
#endif
