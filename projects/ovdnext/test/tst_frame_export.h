/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "TestFiles.h"
#include "PvdCheckedOutput.h"
#include "PvdObjectModelEventStream.h"
#include "PxTransform.h"
#include "PxBounds3.h"
#include "tst_pvd_loader.h"
#include <functional>
#include <memory>

// The previous engine returned null for every supported legacy crop.
TEST_F(PvdEngineTest, LegacyCropCreatesAnOwnedSnapshot)
{
    test_files::Directory temporary;
    ASSERT_TRUE(temporary.valid());
    const std::string folder = temporary.path();
    std::unique_ptr<pvdapp::PvdEngine> source(pvdapp::PvdEngine::load(
        *m_alloc, folder.c_str(), TEST_DATA_DIR "/apex_destruction.pxd2"));
    ASSERT_NE(source, nullptr);
    const auto before = source->getCurrentEvent();
    auto release = [](pvdapp::PvdClient* client) { if(client) client->release(); };
    std::unique_ptr<pvdapp::PvdClient, decltype(release)> clip(
        source->crop(1, 2, folder.c_str()), release);
    ASSERT_NE(clip, nullptr);
    EXPECT_EQ(source->getCurrentEvent(), before);
}

TEST_F(PvdEngineTest, NativeCropRejectsInvalidInclusiveRanges)
{
    test_files::Directory temporary;
    ASSERT_TRUE(temporary.valid());
    const std::string folder = temporary.path();
    std::unique_ptr<pvdapp::PvdEngine> source(pvdapp::PvdEngine::load(
        *m_alloc, folder.c_str(), TEST_DATA_DIR "/apex_destruction.pxd2"));
    ASSERT_NE(source, nullptr);
    const auto allocated = [&] { return static_cast<pvdapp::CheckingAllocator*>(m_alloc)->getTotalBytesAllocated(); };
    const auto before = allocated();
    EXPECT_THROW(source->crop(74133808, -1, 2, folder.c_str()), std::out_of_range);
    EXPECT_THROW(source->crop(74133808, 3, 2, folder.c_str()), std::out_of_range);
    EXPECT_THROW(source->crop(74133808, 0, 260, folder.c_str()), std::out_of_range);
    EXPECT_EQ(allocated(), before);
}

namespace frame_export_test {
struct Release {
    template<class T> void operator()(T* object) const { if(object) object->release(); }
};
}

class FrameExportTest : public pvd_loader_test::PvdLoaderTest
{
protected:
    test_files::Directory temporary;

    void createCapture(bool alignedHeader = false,
        std::function<void(pvdapp::PvdClientMutator&)> beforeFlush = {})
    {
        using namespace pvdapp;
        using namespace physx::pvdsdk;
        ASSERT_TRUE(temporary.valid());
        const std::string backing = temporary.file("capture.backing");
        const auto opened = PvdRandomAccessIOStream::open(*m_alloc, backing.c_str(), true, true);
        ASSERT_TRUE(opened.hasValue());
        opened.getValue()->addRef();
        std::unique_ptr<PvdRandomAccessIOStream, frame_export_test::Release> stream(opened.getValue());
        std::unique_ptr<PvdClientMutator, frame_export_test::Release> mutator(
            &PvdClientMutator::create(*m_alloc, *stream, false));
        auto& client = mutator->getClient();
        auto& metadata = client.getMetaData();
        auto& model = mutator->getObjectModelMutator();
        auto& frames = mutator->getCommStreamMetaDataMutator();
        const auto sceneClass = metadata.getOrCreateClass(NamespacedName("physx3", "PxScene"));
        const auto actorClass = metadata.getOrCreateClass(NamespacedName("physx3", "PxRigidDynamic"));
        const auto dynamics = metadata.createProperty(sceneClass.mClassId, "RigidDynamics", "children",
            getPvdTypeForType<ObjectRef>(), PropertyType::Array).getValue();
        metadata.createProperty(actorClass.mClassId, "GlobalPose", "",
            getPvdTypeForType<PxTransform>(), PropertyType::Scalar);
        metadata.createProperty(actorClass.mClassId, "WorldBounds", "",
            getPvdTypeForType<PxBounds3>(), PropertyType::Scalar);
        const auto scene = model.createInstance(sceneClass.mClassId, 1);
        const auto actor = model.createInstance(actorClass.mClassId, 2);
        frames.setInstanceId(100, 1);
        frames.setInstanceId(200, 2);
        const int32_t reference = 2;
        model.setPropertyValue(scene.mInstPtr, dynamics.mPropertyId,
            reinterpret_cast<const uint8_t*>(&reference), sizeof(reference), getPvdTypeForType<ObjectRef>());
        model.setPropertyValue(actor.mInstPtr, "GlobalPose", PxTransform(PxVec3(100, 200, 300)));
        model.setPropertyValue(actor.mInstPtr, "WorldBounds", PxBounds3(PxVec3(99, 199, 299), PxVec3(101, 201, 301)));
        const auto marker = metadata.getStringTable().strToHandle("frame");
        for(uint32_t frame = 1; frame <= 3; ++frame) {
            frames.addSectionEvent(100, marker, true, frame * 100);
            if(frame > 1) model.originShift(scene.mInstPtr, PxVec3(10, 20, 30));
            frames.addSectionEvent(100, marker, false, frame * 100 + 50);
        }
        if(beforeFlush) {
            // Intern serialization's lazy semantic strings before measuring
            // branch ownership; events remain in the unflushed write page.
            MemPvdOutputStream meta("unflushed metadata");
            metadata.save(meta);
            // Warm the backing stream's OS read buffers, without flushing
            // the factory's pending event page.
            uint8_t snapshotHeader[4];
            ASSERT_FALSE(stream->read(snapshotHeader, 4, 0).hasValue());
            beforeFlush(*mutator);
        }
        mutator->getEventStreamFactory()->flush();
        if(alignedHeader) {
            MemPvdOutputStream meta("aligned metadata"), comm("aligned timeline");
            metadata.save(meta);
            client.getStreamMetaData().save(comm);
            // Serialization interns class/property strings on its first pass.
            meta.clear();
            metadata.save(meta);
            // Each string record adds a 4-byte handle, 4-byte length, and NUL.
            const uint32_t length = 4096 - ((16 + meta.size() + comm.size() + 9) % 4096);
            metadata.getStringTable().strToHandle(std::string(length, 'x').c_str());
            meta.clear();
            metadata.save(meta);
            ASSERT_EQ((16 + meta.size() + comm.size()) % 4096, 0u);
        }
        ovdnext::OutputFile output(temporary.file("source.pxd2"), false);
        client.save(output);
        output.commit();
    }

    void expectActor(pvdapp::PvdEngine& engine, const PxVec3& position)
    {
        pvdapp::InstanceDescription objects[2];
        ASSERT_EQ(engine.getInstances(objects, 2), 2u);
        const auto& actor = objects[0].mId == 2 ? objects[0] : objects[1];
        ASSERT_EQ(actor.mId, 2);
        const auto pose = engine.getPropertyValue<PxTransform>(actor.mInstPtr, "GlobalPose");
        const auto bounds = engine.getPropertyValue<PxBounds3>(actor.mInstPtr, "WorldBounds");
        ASSERT_TRUE(pose.hasValue());
        ASSERT_TRUE(bounds.hasValue());
        EXPECT_EQ(pose.getValue().p, position);
        EXPECT_EQ(bounds.getValue().minimum, position - PxVec3(1));
        EXPECT_EQ(bounds.getValue().maximum, position + PxVec3(1));
    }

    void verifyCrop(PxI64 start, PxI64 end)
    {
        createCapture();
        const std::string folder = temporary.path();
        std::unique_ptr<pvdapp::PvdEngine> source(pvdapp::PvdEngine::load(*m_alloc, folder.c_str(),
            temporary.file("source.pxd2").c_str()));
        ASSERT_NE(source, nullptr);
        source->setCurrentFrame(2, 100);
        const auto before = source->getCurrentEvent();
        expectActor(*source, PxVec3(90, 180, 270));
        std::unique_ptr<pvdapp::PvdClient, frame_export_test::Release> clip(
            source->crop(100, start, end, folder.c_str()));
        ASSERT_NE(clip, nullptr);
        const std::string path = temporary.file("clip.pxd2");
        { ovdnext::OutputFile output(path, false); clip->save(output); output.commit(); }
        EXPECT_EQ(source->getCurrentEvent(), before);
        expectActor(*source, PxVec3(90, 180, 270));
        std::unique_ptr<pvdapp::PvdEngine> reopened(pvdapp::PvdEngine::load(
            *m_alloc, folder.c_str(), path.c_str()));
        ASSERT_NE(reopened, nullptr);
        ASSERT_EQ(reopened->getNbScenes(), 1u);
        ASSERT_EQ(reopened->getFrameCount(100), PxU32(end - start + 1));
        const PxVec3 positions[] = {PxVec3(100, 200, 300), PxVec3(90, 180, 270), PxVec3(80, 160, 240)};
        for(PxI64 local = 0; local <= end - start; ++local) {
            reopened->setCurrentFrame(PxU32(local), 100);
            expectActor(*reopened, positions[start + local - 1]);
        }
    }
};

TEST_F(FrameExportTest, SingleFrameRetainsTimelineAndStartingState) { verifyCrop(2, 2); }
TEST_F(FrameExportTest, InclusiveCropReplaysOriginShiftsWithoutSeekingSource) { verifyCrop(1, 3); }

TEST_F(FrameExportTest, AlignedSerializedHeaderReopensFinalActorState)
{
    ASSERT_NO_FATAL_FAILURE(createCapture(true));
    const std::string path = temporary.file("source.pxd2");
    test_files::File saved(path);
    ASSERT_TRUE(saved.open(std::ios::in));
    const std::string header = saved.read(16);
    ASSERT_EQ(header.size(), 16);
    uint32_t sizes[2];
    memcpy(sizes, header.c_str() + 8, sizeof(sizes));
    const uint64_t headerSize = 16u + sizes[0] + sizes[1];
    ASSERT_EQ(headerSize % 4096, 0u);
    ASSERT_TRUE(saved.seek(headerSize));
    ASSERT_EQ(saved.read(4096), std::string(4096, '\0'));
    saved.close();
    std::unique_ptr<pvdapp::PvdEngine> reopened(pvdapp::PvdEngine::load(*m_alloc,
        temporary.path().c_str(), path.c_str()));
    ASSERT_NE(reopened, nullptr);
    ASSERT_EQ(reopened->getNbScenes(), 1u);
    ASSERT_EQ(reopened->getFrameCount(100), 4u);
    reopened->setCurrentFrame(3, 100);
    expectActor(*reopened, PxVec3(80, 160, 240));
}

TEST_F(FrameExportTest, UnflushedBranchTransfersFinalActorStateWithoutChangingSource)
{
    ASSERT_NO_FATAL_FAILURE(createCapture(false, [&](pvdapp::PvdClientMutator& mutator) {
        auto& source = mutator.getClient();
        pvdapp::InstanceDescription instances[2];
        ASSERT_EQ(source.getReader().getInstances(instances, 2), 2u);
        const auto count = source.getEventCount();
        const auto current = source.getCurrentEvent();
        const auto before = static_cast<pvdapp::CheckingAllocator*>(m_alloc)->getTotalBytesAllocated();
        {
            std::unique_ptr<pvdapp::PvdClient, frame_export_test::Release> branch(source.branch());
            ASSERT_NE(branch, nullptr);
            ASSERT_EQ(branch->getEventCount(), count);
            branch->setCurrentEvent(0);
            const auto backing = pvdapp::PvdRandomAccessIOStream::open(*m_alloc,
                temporary.file("branch.backing").c_str(), true, true);
            ASSERT_TRUE(backing.hasValue());
            backing.getValue()->addRef();
            std::unique_ptr<pvdapp::PvdRandomAccessIOStream, frame_export_test::Release> stream(backing.getValue());
            std::unique_ptr<pvdapp::PvdClient, frame_export_test::Release> transferred(
                branch->transferEvents(count, *stream));
            ASSERT_NE(transferred, nullptr);
            ovdnext::OutputFile output(temporary.file("branch.pxd2"), false);
            transferred->save(output);
            output.commit();
        }
        EXPECT_EQ(source.getEventCount(), count);
        EXPECT_EQ(source.getCurrentEvent(), current);
        EXPECT_EQ(static_cast<pvdapp::CheckingAllocator*>(m_alloc)->getTotalBytesAllocated(), before);
        const auto pose = source.getReader().getPropertyValue<PxTransform>(
            source.getReader().idToPtr(2), "GlobalPose");
        ASSERT_TRUE(pose.hasValue());
        EXPECT_EQ(pose->p, PxVec3(80, 160, 240));
    }));
    std::unique_ptr<pvdapp::PvdEngine> reopened(pvdapp::PvdEngine::load(*m_alloc,
        temporary.path().c_str(),
        temporary.file("branch.pxd2").c_str()));
    ASSERT_NE(reopened, nullptr);
    ASSERT_EQ(reopened->getNbScenes(), 1u);
    ASSERT_EQ(reopened->getFrameCount(100), 4u);
    reopened->setCurrentFrame(3, 100);
    expectActor(*reopened, PxVec3(80, 160, 240));
}

TEST_F(FrameExportTest, RawShortCaptureLastFrameReopensWithoutManualFlush)
{
    using namespace physx::pvdsdk;
    const std::string raw = temporary.file("raw.pxd2");
    test_files::File file(raw);
    ASSERT_TRUE(file.open(std::ios::out));
    ASSERT_TRUE(writeStreamHeader(file));
    // Include the built-in semantic string so cloning need not intern it
    // lazily while we measure retained allocation after the crop.
    const char* strings[] = {"physx3", "PxScene", "PxRigidDynamic", "GlobalPose", "PxTransform", "frame", "user message"};
    for(uint32_t i = 0; i < 7; ++i)
        ASSERT_TRUE(writeEventGroup(file, 1, serializeEvent(StringHandleEvent(strings[i], i + 1))));
    const StreamNamespacedName scene(1, 2), actor(1, 3), transform(1, 5);
    ASSERT_TRUE(writeEventGroup(file, 1, serializeEvent(CreateClass(scene))));
    ASSERT_TRUE(writeEventGroup(file, 1, serializeEvent(CreateClass(actor))));
    ASSERT_TRUE(writeEventGroup(file, 1, serializeEvent(CreateProperty(actor, 4, 0,
        transform, PropertyType::Scalar, DataRef<NameHandleValue>()))));
    ASSERT_TRUE(writeEventGroup(file, 1, serializeEvent(CreateInstance(scene, 100))));
    ASSERT_TRUE(writeEventGroup(file, 1, serializeEvent(CreateInstance(actor, 200))));
    for(uint32_t frame = 1; frame <= 3; ++frame) {
        const PxTransform pose(PxVec3(float(frame), float(frame * 2), float(frame * 3)));
        ASSERT_TRUE(writeEventGroup(file, 1, serializeEvent(BeginSection(100, 6, frame * 100))));
        ASSERT_TRUE(writeEventGroup(file, 1, serializeEvent(SetPropertyValue(200, 4,
            DataRef<const uint8_t>(reinterpret_cast<const uint8_t*>(&pose), sizeof(pose)), transform, 1))));
        ASSERT_TRUE(writeEventGroup(file, 1, serializeEvent(EndSection(100, 6, frame * 100 + 50))));
    }
    file.close();
    const std::string folder = temporary.path();
    std::unique_ptr<pvdapp::PvdEngine> source(pvdapp::PvdEngine::load(
        *m_alloc, folder.c_str(), raw.c_str()));
    ASSERT_NE(source, nullptr);
    ASSERT_EQ(source->getNbScenes(), 1u);
    ASSERT_EQ(source->getFrameCount(100), 4u);
    source->setCurrentFrame(2, 100);
    auto expectPose = [](pvdapp::PvdEngine& engine, const PxVec3& expected) {
        pvdapp::InstanceDescription objects[2];
        ASSERT_EQ(engine.getInstances(objects, 2), 2u);
        for(const auto& object : objects) {
            if(object.mId != 2) continue;
            const auto pose = engine.getPropertyValue<PxTransform>(object.mInstPtr, "GlobalPose");
            if(pose.hasValue()) { EXPECT_EQ(pose->p, expected); return; }
        }
        FAIL() << "Missing actor pose";
    };
    expectPose(*source, PxVec3(2, 4, 6));
    const auto beforeEvent = source->getCurrentEvent();
    const auto beforeCount = source->getEventCount();
    const auto before = static_cast<pvdapp::CheckingAllocator*>(m_alloc)->getTotalBytesAllocated();
    const std::string outputPath = temporary.file("raw-clip.pxd2");
    {
        std::unique_ptr<pvdapp::PvdClient, frame_export_test::Release> clip(
            source->crop(100, 3, 3, folder.c_str()));
        ASSERT_NE(clip, nullptr);
        ovdnext::OutputFile output(outputPath, false);
        clip->save(output);
        output.commit();
    }
    EXPECT_EQ(source->getCurrentEvent(), beforeEvent);
    EXPECT_EQ(source->getEventCount(), beforeCount);
    EXPECT_EQ(static_cast<pvdapp::CheckingAllocator*>(m_alloc)->getTotalBytesAllocated(), before);
    expectPose(*source, PxVec3(2, 4, 6));
    std::unique_ptr<pvdapp::PvdEngine> reopened(pvdapp::PvdEngine::load(
        *m_alloc, folder.c_str(), outputPath.c_str()));
    ASSERT_NE(reopened, nullptr);
    ASSERT_EQ(reopened->getNbScenes(), 1u);
    ASSERT_EQ(reopened->getFrameCount(100), 1u);
    reopened->setCurrentFrame(0, 100);
    expectPose(*reopened, PxVec3(3, 6, 9));
}

TEST_F(FrameExportTest, FailedEventTransferReturnsNullAndReleasesOwnedState)
{
    ASSERT_NO_FATAL_FAILURE(createCapture());
    const auto opened = pvdapp::PvdRandomAccessIOStream::open(*m_alloc,
        temporary.file("source.pxd2").c_str(), false);
    ASSERT_TRUE(opened.hasValue());
    opened.getValue()->addRef();
    std::unique_ptr<pvdapp::PvdRandomAccessIOStream, frame_export_test::Release> file(opened.getValue());
    // Fail only the disk-read boundary, after loading the real client and its
    // initial snapshot. All metadata, transfer, and cleanup are production code.
    struct FailingReads : pvdapp::PvdRandomAccessIOStream {
        pvdapp::PvdRandomAccessIOStream& file;
        bool failReads = false;
        int rejectedReads = 0;
        explicit FailingReads(pvdapp::PvdRandomAccessIOStream& stream) : file(stream) {}
        void addRef() override { file.addRef(); }
        void release() override { file.release(); }
        pvdapp::Option<pvdapp::ErrorCode> read(uint8_t* data, uint32_t size, uint64_t offset) override {
            if(failReads) { ++rejectedReads; return pvdapp::ErrorCode(1); }
            return file.read(data, size, offset);
        }
        bool write(const uint8_t* data, uint32_t size) override { return file.write(data, size); }
        bool directCopy(physx::pvdsdk::PvdInputStream& input, uint32_t size) override { return file.directCopy(input, size); }
        bool directCopy(physx::pvdsdk::PvdInputStream& input, uint64_t size) override { return file.directCopy(input, size); }
        void zeroPadToPageSize(uint32_t page = 4096) override { file.zeroPadToPageSize(page); }
        uint64_t getLength() const override { return file.getLength(); }
        pvdapp::Option<pvdapp::ErrorCode> flush() override { return file.flush(); }
        pvdapp::Option<pvdapp::ErrorCode> asyncFlush() override { return file.asyncFlush(); }
        double getTimeCost() override { return file.getTimeCost(); }
    } input(*file);
    std::unique_ptr<pvdapp::PvdClient, frame_export_test::Release> source(pvdapp::PvdClient::load(*m_alloc, input));
    ASSERT_NE(source, nullptr);
    const auto before = static_cast<pvdapp::CheckingAllocator*>(m_alloc)->getTotalBytesAllocated();
    const auto backing = pvdapp::PvdRandomAccessIOStream::open(*m_alloc,
        temporary.file("failed-transfer.backing").c_str(), true, true);
    ASSERT_TRUE(backing.hasValue());
    backing.getValue()->addRef();
    std::unique_ptr<pvdapp::PvdRandomAccessIOStream, frame_export_test::Release> output(backing.getValue());
    const auto current = source->getCurrentEvent();
    const auto count = source->getEventCount();
    input.failReads = true;
    struct ExpectedReadAssertions : PxAssertHandler {
        PxAssertHandler& previous = PxGetAssertHandler();
        ExpectedReadAssertions() { PxSetAssertHandler(*this); }
        ~ExpectedReadAssertions() { PxSetAssertHandler(previous); }
        void operator()(const char* expression, const char* file, int, bool&) override {
            EXPECT_STREQ(expression, "false");
            EXPECT_EQ(test_files::fs::path(file).filename(), "PvdObjectModelEventStream.cpp");
        }
    } assertions;
    {
        std::unique_ptr<pvdapp::PvdClient, frame_export_test::Release> result(
            source->transferEvents(count, *output));
        EXPECT_EQ(result, nullptr);
    }
    EXPECT_GT(input.rejectedReads, 0);
    output.reset();
    EXPECT_EQ(source->getCurrentEvent(), current);
    EXPECT_EQ(source->getEventCount(), count);
    EXPECT_EQ(static_cast<pvdapp::CheckingAllocator*>(m_alloc)->getTotalBytesAllocated(), before);
}

TEST_F(FrameExportTest, ReopenedSingleFrameCanBeCroppedAgainWithoutLosingItsTimeline)
{
    ASSERT_NO_FATAL_FAILURE(verifyCrop(2, 2));
    const std::string folder = temporary.path();
    std::unique_ptr<pvdapp::PvdEngine> source(pvdapp::PvdEngine::load(*m_alloc, folder.c_str(),
        temporary.file("clip.pxd2").c_str()));
    ASSERT_NE(source, nullptr);
    const auto before = source->getCurrentEvent();
    std::unique_ptr<pvdapp::PvdClient, frame_export_test::Release> clip(
        source->crop(100, 0, 0, folder.c_str()));
    ASSERT_NE(clip, nullptr);
    const std::string path = temporary.file("recropped.pxd2");
    { ovdnext::OutputFile output(path, false); clip->save(output); output.commit(); }
    EXPECT_EQ(source->getCurrentEvent(), before);
    expectActor(*source, PxVec3(90, 180, 270));
    std::unique_ptr<pvdapp::PvdEngine> reopened(pvdapp::PvdEngine::load(
        *m_alloc, folder.c_str(), path.c_str()));
    ASSERT_NE(reopened, nullptr);
    ASSERT_EQ(reopened->getNbScenes(), 1u);
    ASSERT_EQ(reopened->getFrameCount(100), 1u);
    reopened->setCurrentFrame(0, 100);
    expectActor(*reopened, PxVec3(90, 180, 270));
}
