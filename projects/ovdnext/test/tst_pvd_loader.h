/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "TestFiles.h"
#pragma once

#include "pvdfileloader.h"
#include "pvdcommstreamparsing.h"
#include "PvdInputValidation.h"
#include "PvdRandomAccessIOStream.h"
#include "PxPvdCommStreamEventSink.h"
#include "PxPvdCommStreamEvents.h"
#include "PxPvdInternalByteStreams.h"


#include <cstdint>
#include <vector>

namespace pvd_loader_test
{

using namespace physx;
using namespace physx::pvdsdk;

// These declarations compile only while the original subclass contract stays
// virtual and keeps its exact return and argument types.
class LegacyParserOverride : public pvdapp::PvdCommStreamParser
{
  public:
    bool parseNextEventGroup(DataRef<const PxU8>) override;
    void release() override;
};

class LegacyEventReaderOverride : public pvdapp::PvdCommStreamEventReader
{
  public:
    bool readNextEventGroup(PvdInputStream&, PvdOutputStream&) override;
    bool isSwappingBytes() override;
    void release() override;
    pvdapp::PvdCommStreamParser& createParser(
        PvdObjectModelMetaData&, PvdObjectModelMutator&,
        pvdapp::PvdCommStreamMetaDataMutator&) override;
};

class LegacyFileStreamLoaderOverride : public pvdapp::PvdFileStreamLoader
{
  public:
    PxU32 loadNextChunk(PxU32) override;
    pvdapp::PvdClientMutator& getClient() override;
    void release() override;
};

TEST(PvdLoaderApiCompatibilityTest, RetainsLegacyVirtualOverrideSignatures)
{
    SUCCEED();
}

class PvdLoaderTest : public PvdEngineTest
{
  protected:
    template <typename T>
    static bool writeValue(test_files::File& file, const T& value)
    {
        return file.write(reinterpret_cast<const char*>(&value), sizeof(value)) ==
            sizeof(value);
    }

    static bool writeStreamHeader(
        test_files::File& file, PxU32 streamVersion, PxU64 timestampNumerator,
        PxU64 timestampDenominator, PxU32 streamFlags)
    {
        const PxU32 streamId = StreamInitialization::getStreamId();
        return writeValue(file, streamId) && writeValue(file, streamVersion) &&
            writeValue(file, timestampNumerator) &&
            writeValue(file, timestampDenominator) &&
            writeValue(file, streamFlags);
    }

    static bool writeStreamHeader(test_files::File& file)
    {
        return writeStreamHeader(
            file, StreamInitialization::getStreamVersion(), 1, 1,
            sizeof(void*) == 4 ? 0u : 1u);
    }

    template <typename TEvent>
    static std::string serializeEvent(const TEvent& event)
    {
        MemPvdOutputStream stream("PvdLoaderTest::event");
        PvdCommStreamEventSink::writeStreamEvent(
            event, getCommStreamEventType<TEvent>(), stream);
        const DataRef<const PxU8> bytes = stream.toRef();
        return std::string(reinterpret_cast<const char*>(bytes.begin()), bytes.size());
    }

    static bool writeEventGroup(
        test_files::File& file, PxU64 streamId, const std::string& eventData,
        PxU32 eventCount = 1)
    {
        const PxU32 dataSize = static_cast<PxU32>(eventData.size());
        const PxU64 timestamp = 0;
        return writeValue(file, dataSize) && writeValue(file, eventCount) &&
            writeValue(file, streamId) && writeValue(file, timestamp) &&
            file.write(eventData) == eventData.size();
    }

    static std::string capturePath(const test_files::Directory& root)
    {
        return root.file("capture.pxd2");
    }

    static pvdapp::PvdEngine* loadCapture(const test_files::Directory& root)
    {
        const std::string path = capturePath(root);
        return pvdapp::PvdEngine::load(
            *m_alloc, root.path().c_str(), path.c_str());
    }

    static bool writePointerConversionCapture(
        const test_files::Directory& root, PxU32 streamCount, PxU32 itemCount)
    {
        test_files::File file(capturePath(root));
        if(!file.open(std::ios::out) || !writeStreamHeader(file))
            return false;

        const PxU32 testNamespaceHandle = 1;
        const PxU32 holderNameHandle = 2;
        const PxU32 propertyNameHandle = 3;
        const PxU32 builtinNamespaceHandle = 4;
        const PxU32 objectRefNameHandle = 5;
        const PxU32 voidPtrNameHandle = 6;
        const PxU64 instanceId = 1;
        const PxU64 setupStreamId = 1;

        if(!writeEventGroup(file, setupStreamId,
               serializeEvent(StringHandleEvent("scratch", testNamespaceHandle))) ||
           !writeEventGroup(file, setupStreamId,
               serializeEvent(StringHandleEvent("Holder", holderNameHandle))) ||
           !writeEventGroup(file, setupStreamId,
               serializeEvent(StringHandleEvent("refs", propertyNameHandle))) ||
           !writeEventGroup(file, setupStreamId,
               serializeEvent(StringHandleEvent("physx3", builtinNamespaceHandle))) ||
           !writeEventGroup(file, setupStreamId,
               serializeEvent(StringHandleEvent("ObjectRef", objectRefNameHandle))) ||
           !writeEventGroup(file, setupStreamId,
               serializeEvent(StringHandleEvent("VoidPtr", voidPtrNameHandle))))
            return false;

        const StreamNamespacedName holder = StreamNamespacedName(
            StringHandle(testNamespaceHandle), StringHandle(holderNameHandle));
        const StreamNamespacedName objectRefType = StreamNamespacedName(
            StringHandle(builtinNamespaceHandle), StringHandle(objectRefNameHandle));
        const StreamNamespacedName voidPtrType = StreamNamespacedName(
            StringHandle(builtinNamespaceHandle), StringHandle(voidPtrNameHandle));
        if(!writeEventGroup(file, setupStreamId, serializeEvent(CreateClass(holder))) ||
           !writeEventGroup(file, setupStreamId, serializeEvent(CreateProperty(
               holder, StringHandle(propertyNameHandle), StringHandle(0), objectRefType,
               PropertyType::Array, DataRef<NameHandleValue>()))) ||
           !writeEventGroup(file, setupStreamId,
               serializeEvent(CreateInstance(holder, instanceId))))
            return false;

        const std::vector<PxU64> pointers(itemCount, 0);
        const DataRef<const PxU8> pointerBytes(
            reinterpret_cast<const PxU8*>(pointers.data()),
            static_cast<PxU32>(pointers.size() * sizeof(PxU64)));
        const std::string setEvent = serializeEvent(SetPropertyValue(
            instanceId, StringHandle(propertyNameHandle), pointerBytes,
            voidPtrType, itemCount));
        for(PxU64 streamId = 1; streamId <= streamCount; ++streamId)
        {
            if(!writeEventGroup(file, streamId, setEvent))
                return false;
        }
        return true;
    }
};

TEST_F(PvdLoaderTest, OrdinaryEofCompletesAHeaderOnlyCapture)
{
    test_files::Directory root(u8"pvd-loader-\u00fc-");
    ASSERT_TRUE(root.valid());
    test_files::File file(capturePath(root));
    ASSERT_TRUE(file.open(std::ios::out));
    ASSERT_TRUE(writeStreamHeader(file));
    file.close();

    pvdapp::PvdEngine* engine = loadCapture(root);
    ASSERT_NE(nullptr, engine);
    delete engine;
}

TEST_F(PvdLoaderTest, RejectsTruncatedStreamInitializationHeader)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    test_files::File file(capturePath(root));
    ASSERT_TRUE(file.open(std::ios::out));
    ASSERT_TRUE(writeValue(file, StreamInitialization::getStreamId()));
    file.close();

    pvdapp::PvdEngine* engine = loadCapture(root);
    EXPECT_EQ(nullptr, engine);
    delete engine;
}

TEST_F(PvdLoaderTest, RejectsZeroTimestampDenominator)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    test_files::File file(capturePath(root));
    ASSERT_TRUE(file.open(std::ios::out));
    ASSERT_TRUE(writeStreamHeader(
        file, StreamInitialization::getStreamVersion(), 1, 0,
        sizeof(void*) == 4 ? 0u : 1u));
    file.close();

    pvdapp::PvdEngine* engine = loadCapture(root);
    EXPECT_EQ(nullptr, engine);
    delete engine;
}

TEST_F(PvdLoaderTest, RejectsUnsupportedStreamVersion)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    test_files::File file(capturePath(root));
    ASSERT_TRUE(file.open(std::ios::out));
    ASSERT_TRUE(writeStreamHeader(
        file, StreamInitialization::getStreamVersion() + 1u, 1, 1,
        sizeof(void*) == 4 ? 0u : 1u));
    file.close();

    pvdapp::PvdEngine* engine = loadCapture(root);
    EXPECT_EQ(nullptr, engine);
    delete engine;
}

TEST_F(PvdLoaderTest, RejectsUnknownStreamFlags)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    test_files::File file(capturePath(root));
    ASSERT_TRUE(file.open(std::ios::out));
    ASSERT_TRUE(writeStreamHeader(
        file, StreamInitialization::getStreamVersion(), 1, 1, 2u));
    file.close();

    pvdapp::PvdEngine* engine = loadCapture(root);
    EXPECT_EQ(nullptr, engine);
    delete engine;
}

TEST_F(PvdLoaderTest, RejectsPhysicalEofDuringSetPropertyValue)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    test_files::File file(capturePath(root));
    ASSERT_TRUE(file.open(std::ios::out));
    ASSERT_TRUE(writeStreamHeader(file));
    ASSERT_TRUE(writeEventGroup(file, 1, serializeEvent(BeginSetPropertyValue(
        0, StringHandle(0), StreamNamespacedName()))));
    file.close();

    pvdapp::PvdEngine* engine = loadCapture(root);
    EXPECT_EQ(nullptr, engine);
    delete engine;
}

TEST_F(PvdLoaderTest, RejectsUnrelatedEventInsideSetPropertyValueGroupImmediately)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    test_files::File file(capturePath(root));
    ASSERT_TRUE(file.open(std::ios::out));
    ASSERT_TRUE(writeStreamHeader(file));
    ASSERT_TRUE(writeEventGroup(file, 1, serializeEvent(BeginSetPropertyValue(
        0, StringHandle(0), StreamNamespacedName()))));
    ASSERT_TRUE(writeEventGroup(file, 1, serializeEvent(SetCamera(
        "camera", PxVec3(0.0f), PxVec3(0.0f, 1.0f, 0.0f),
        PxVec3(0.0f, 0.0f, 1.0f)))));
    ASSERT_TRUE(writeEventGroup(file, 1, serializeEvent(EndSetPropertyValue())));
    file.close();

    const std::string sourcePath = capturePath(root);
    const std::string backingPath =
        root.file("backing.pxd3");
    pvdapp::ErrorCodeOption<pvdapp::PvdRandomAccessIOStream*> source =
        pvdapp::PvdRandomAccessIOStream::open(
            *m_alloc, sourcePath.c_str(), false, false);
    pvdapp::ErrorCodeOption<pvdapp::PvdRandomAccessIOStream*> backing =
        pvdapp::PvdRandomAccessIOStream::open(
            *m_alloc, backingPath.c_str(), true, true);
    ASSERT_TRUE(source.hasValue());
    ASSERT_TRUE(backing.hasValue());

    pvdapp::PvdFileLoaderFactory& factory =
        pvdapp::PvdFileLoaderFactory::createFileLoaderFactory(*m_alloc);
    pvdapp::PvdFileStreamLoader* loader = factory.createFileStreamLoader(
        *source.getValue(), *backing.getValue());
    factory.release();
    ASSERT_NE(nullptr, loader);

    EXPECT_EQ(pvdapp::PvdFileStreamLoadResult::Loaded,
        loader->loadNextChunkResult(1).mStatus);
    EXPECT_EQ(pvdapp::PvdFileStreamLoadResult::Failed,
        loader->loadNextChunkResult(1).mStatus);
    loader->release();
}

TEST_F(PvdLoaderTest, RejectsStreamEndDuringSetPropertyValue)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    test_files::File file(capturePath(root));
    ASSERT_TRUE(file.open(std::ios::out));
    ASSERT_TRUE(writeStreamHeader(file));
    ASSERT_TRUE(writeEventGroup(file, 1, serializeEvent(BeginSetPropertyValue(
        0, StringHandle(0), StreamNamespacedName()))));
    ASSERT_TRUE(writeEventGroup(file, 1, serializeEvent(StreamEndEvent())));
    file.close();

    pvdapp::PvdEngine* engine = loadCapture(root);
    EXPECT_EQ(nullptr, engine);
    delete engine;
}

TEST_F(PvdLoaderTest, RejectedEventGroupFailsTheWholeLoad)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    test_files::File file(capturePath(root));
    ASSERT_TRUE(file.open(std::ios::out));
    ASSERT_TRUE(writeStreamHeader(file));

    const PxU32 rejectedSize = pvdapp::input::kMaxEventGroupDataSize + 1u;
    const PxU32 eventCount = 1;
    const PxU64 streamId = 1;
    const PxU64 timestamp = 0;
    ASSERT_TRUE(writeValue(file, rejectedSize));
    ASSERT_TRUE(writeValue(file, eventCount));
    ASSERT_TRUE(writeValue(file, streamId));
    ASSERT_TRUE(writeValue(file, timestamp));
    file.close();

    pvdapp::PvdEngine* engine = loadCapture(root);
    EXPECT_EQ(nullptr, engine);
    delete engine;
}

TEST_F(PvdLoaderTest, StreamEndDoesNotHideALaterMalformedGroup)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    test_files::File file(capturePath(root));
    ASSERT_TRUE(file.open(std::ios::out));
    ASSERT_TRUE(writeStreamHeader(file));
    ASSERT_TRUE(writeEventGroup(file, 1, serializeEvent(StreamEndEvent())));

    const PxU32 rejectedSize = pvdapp::input::kMaxEventGroupDataSize + 1u;
    const PxU32 eventCount = 1;
    const PxU64 streamId = 2;
    const PxU64 timestamp = 0;
    ASSERT_TRUE(writeValue(file, rejectedSize));
    ASSERT_TRUE(writeValue(file, eventCount));
    ASSERT_TRUE(writeValue(file, streamId));
    ASSERT_TRUE(writeValue(file, timestamp));
    file.close();

    pvdapp::PvdEngine* engine = loadCapture(root);
    EXPECT_EQ(nullptr, engine);
    delete engine;
}

TEST_F(PvdLoaderTest, RejectsEventsAfterStreamEndForTheSameStream)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    test_files::File file(capturePath(root));
    ASSERT_TRUE(file.open(std::ios::out));
    ASSERT_TRUE(writeStreamHeader(file));
    ASSERT_TRUE(writeEventGroup(file, 1, serializeEvent(StreamEndEvent())));
    ASSERT_TRUE(writeEventGroup(file, 1, serializeEvent(SetCamera(
        "camera", PxVec3(0.0f), PxVec3(0.0f, 1.0f, 0.0f),
        PxVec3(0.0f, 0.0f, 1.0f)))));
    file.close();

    pvdapp::PvdEngine* engine = loadCapture(root);
    EXPECT_EQ(nullptr, engine);
    delete engine;
}

TEST_F(PvdLoaderTest, AllowsAnotherStreamAfterStreamEnd)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    test_files::File file(capturePath(root));
    ASSERT_TRUE(file.open(std::ios::out));
    ASSERT_TRUE(writeStreamHeader(file));
    ASSERT_TRUE(writeEventGroup(file, 1, serializeEvent(StreamEndEvent())));
    ASSERT_TRUE(writeEventGroup(file, 2, serializeEvent(SetCamera(
        "camera", PxVec3(0.0f), PxVec3(0.0f, 1.0f, 0.0f),
        PxVec3(0.0f, 0.0f, 1.0f)))));
    file.close();

    pvdapp::PvdEngine* engine = loadCapture(root);
    ASSERT_NE(nullptr, engine);
    delete engine;
}

TEST_F(PvdLoaderTest, StreamEndMustBeTheFinalEventInItsGroup)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    test_files::File file(capturePath(root));
    ASSERT_TRUE(file.open(std::ios::out));
    ASSERT_TRUE(writeStreamHeader(file));

    std::string events = serializeEvent(StreamEndEvent());
    events.push_back(static_cast<char>(PvdCommStreamEventTypes::Unknown));
    ASSERT_TRUE(writeEventGroup(file, 1, events, 2));
    file.close();

    pvdapp::PvdEngine* engine = loadCapture(root);
    EXPECT_EQ(nullptr, engine);
    delete engine;
}

TEST_F(PvdLoaderTest, RejectsUndeclaredTrailingEventBytes)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    test_files::File file(capturePath(root));
    ASSERT_TRUE(file.open(std::ios::out));
    ASSERT_TRUE(writeStreamHeader(file));

    std::string eventData = serializeEvent(StreamEndEvent());
    eventData.push_back(static_cast<char>(PvdCommStreamEventTypes::Unknown));
    ASSERT_TRUE(writeEventGroup(file, 1, eventData, 1));
    file.close();

    pvdapp::PvdEngine* engine = loadCapture(root);
    EXPECT_EQ(nullptr, engine);
    delete engine;
}

TEST_F(PvdLoaderTest, ZeroByteChunkRequestStillProcessesAvailableData)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    test_files::File file(capturePath(root));
    ASSERT_TRUE(file.open(std::ios::out));
    ASSERT_TRUE(writeStreamHeader(file));
    ASSERT_TRUE(writeEventGroup(file, 1, serializeEvent(SetCamera(
        "camera", PxVec3(0.0f), PxVec3(0.0f, 1.0f, 0.0f),
        PxVec3(0.0f, 0.0f, 1.0f)))));
    file.close();

    const std::string sourcePath = capturePath(root);
    const std::string backingPath =
        root.file("backing.pxd3");
    pvdapp::ErrorCodeOption<pvdapp::PvdRandomAccessIOStream*> source =
        pvdapp::PvdRandomAccessIOStream::open(
            *m_alloc, sourcePath.c_str(), false, false);
    pvdapp::ErrorCodeOption<pvdapp::PvdRandomAccessIOStream*> backing =
        pvdapp::PvdRandomAccessIOStream::open(
            *m_alloc, backingPath.c_str(), true, true);
    ASSERT_TRUE(source.hasValue());
    ASSERT_TRUE(backing.hasValue());

    pvdapp::PvdFileLoaderFactory& factory =
        pvdapp::PvdFileLoaderFactory::createFileLoaderFactory(*m_alloc);
    pvdapp::PvdFileStreamLoader* loader = factory.createFileStreamLoader(
        *source.getValue(), *backing.getValue());
    factory.release();
    ASSERT_NE(nullptr, loader);

    const pvdapp::PvdFileStreamLoadResult result = loader->loadNextChunkResult(0);
    EXPECT_GT(result.mBytesProcessed, 0u);
    EXPECT_EQ(pvdapp::PvdFileStreamLoadResult::Completed, result.mStatus);
    loader->release();
}

TEST_F(PvdLoaderTest, RejectedStreamEndReleasesIncompleteSpvState)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    pvdapp::CheckingAllocator* checkingAllocator =
        static_cast<pvdapp::CheckingAllocator*>(m_alloc);
    const size_t allocatedBeforeLoad =
        checkingAllocator->getTotalBytesAllocated();
    test_files::File file(capturePath(root));
    ASSERT_TRUE(file.open(std::ios::out));
    ASSERT_TRUE(writeStreamHeader(file));

    const std::string beginEvent = serializeEvent(BeginSetPropertyValue(
        0, StringHandle(0), StreamNamespacedName()));
    const std::vector<PxU8> payload(1024u * 1024u, 0x5a);
    const std::string appendEvent = serializeEvent(AppendPropertyValueData(
        DataRef<const PxU8>(payload.data(), static_cast<PxU32>(payload.size())),
        static_cast<PxU32>(payload.size())));
    const std::string endStreamEvent = serializeEvent(StreamEndEvent());
    ASSERT_TRUE(writeEventGroup(file, 1, beginEvent));
    ASSERT_TRUE(writeEventGroup(file, 1, appendEvent));
    ASSERT_TRUE(writeEventGroup(file, 1, endStreamEvent));
    file.close();

    pvdapp::PvdEngine* engine = loadCapture(root);
    EXPECT_EQ(nullptr, engine);
    delete engine;
    EXPECT_EQ(allocatedBeforeLoad, checkingAllocator->getTotalBytesAllocated());
}

TEST_F(PvdLoaderTest, RejectsTooManyParserStreamsAcrossFreshStreamIds)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    test_files::File file(capturePath(root));
    ASSERT_TRUE(file.open(std::ios::out));
    ASSERT_TRUE(writeStreamHeader(file));

    const std::string event = serializeEvent(SetCamera(
        "camera", PxVec3(0.0f), PxVec3(0.0f, 1.0f, 0.0f),
        PxVec3(0.0f, 0.0f, 1.0f)));
    for(PxU64 streamId = 1; streamId <= 1025; ++streamId)
        ASSERT_TRUE(writeEventGroup(file, streamId, event));
    file.close();

    pvdapp::PvdEngine* engine = loadCapture(root);
    EXPECT_EQ(nullptr, engine);
    delete engine;
}

TEST_F(PvdLoaderTest, RejectsTooManyIncompleteSpvStatesAcrossFreshStreamIds)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    test_files::File file(capturePath(root));
    ASSERT_TRUE(file.open(std::ios::out));
    ASSERT_TRUE(writeStreamHeader(file));

    const std::string beginEvent = serializeEvent(BeginSetPropertyValue(
        0, StringHandle(0), StreamNamespacedName()));
    for(PxU64 streamId = 1; streamId <= 257; ++streamId)
        ASSERT_TRUE(writeEventGroup(file, streamId, beginEvent));
    file.close();

    pvdapp::PvdEngine* engine = loadCapture(root);
    EXPECT_EQ(nullptr, engine);
    delete engine;
}

TEST_F(PvdLoaderTest, RejectsAggregateRetainedSpvBytesAcrossFreshStreamIds)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    test_files::File file(capturePath(root));
    ASSERT_TRUE(file.open(std::ios::out));
    ASSERT_TRUE(writeStreamHeader(file));

    const std::string beginEvent = serializeEvent(BeginSetPropertyValue(
        0, StringHandle(0), StreamNamespacedName()));
    const std::vector<PxU8> payload(1024u * 1024u, 0x5a);
    const std::string appendEvent = serializeEvent(AppendPropertyValueData(
        DataRef<const PxU8>(payload.data(), static_cast<PxU32>(payload.size())),
        static_cast<PxU32>(payload.size())));

    for(PxU64 streamId = 1; streamId <= 65; ++streamId)
    {
        ASSERT_TRUE(writeEventGroup(file, streamId, beginEvent));
        ASSERT_TRUE(writeEventGroup(file, streamId, appendEvent));
    }
    file.close();

    pvdapp::PvdEngine* engine = loadCapture(root);
    EXPECT_EQ(nullptr, engine);
    delete engine;
}

TEST_F(PvdLoaderTest, ReusesConversionScratchAcrossParserStreams)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    pvdapp::CheckingAllocator* checkingAllocator =
        static_cast<pvdapp::CheckingAllocator*>(m_alloc);
    const size_t baseline = checkingAllocator->getTotalBytesAllocated();
    const PxU32 itemCount = 256u * 1024u;

    ASSERT_TRUE(writePointerConversionCapture(root, 1, itemCount));
    pvdapp::PvdEngine* oneStreamEngine = loadCapture(root);
    ASSERT_NE(nullptr, oneStreamEngine);
    const size_t oneStreamBytes =
        checkingAllocator->getTotalBytesAllocated() - baseline;
    delete oneStreamEngine;
    ASSERT_EQ(baseline, checkingAllocator->getTotalBytesAllocated());

    ASSERT_TRUE(writePointerConversionCapture(root, 16, itemCount));
    pvdapp::PvdEngine* manyStreamEngine = loadCapture(root);
    ASSERT_NE(nullptr, manyStreamEngine);
    const size_t manyStreamBytes =
        checkingAllocator->getTotalBytesAllocated() - baseline;

    // The capture/event history grows with the command count. Conversion
    // scratch must not add another one-megabyte allocation per parser stream.
    EXPECT_LT(manyStreamBytes, oneStreamBytes + 32u * 1024u * 1024u);
    delete manyStreamEngine;
    EXPECT_EQ(baseline, checkingAllocator->getTotalBytesAllocated());
}

} // namespace pvd_loader_test
