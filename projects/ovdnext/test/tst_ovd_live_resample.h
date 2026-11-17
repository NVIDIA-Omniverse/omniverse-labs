/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "TestFiles.h"
#ifndef TST_OVD_LIVE_RESAMPLE_H
#define TST_OVD_LIVE_RESAMPLE_H

#ifdef PVD_OVD_SUPPORT

// LIVE-path regression test for multi-segment OVD ingestion: a producer doing
// stopSampling/startSampling MID-CONNECTION re-registers the whole schema and
// re-creates every object while the viewer is already ingesting and the UI may
// re-index (rebuildIndex) between the segments. The file-load tests cannot
// exercise that: buildIndex there runs once over the finished DOM. Here a
// producer thread writes a synthetic two-segment stream (same shape as the
// decoded SnippetOmniPvd --resample capture: schema re-registered under new
// handles, objects re-created under their external handles before the scene,
// per-scene frames restarting at 1) over a real TCP socket into
// PvdEngine::connectLive, with a mid-stream refreshLive() between the
// segments, then asserts id stability and the same segment-consistent
// reference resolution the file tests assert.

#include <atomic>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>


#include <gtest/gtest.h>
#include <PvdEngine.h>
#include <PvdSimpleAllocator.h>
#include <PxErrorCallback.h>
#include <PxFoundation.h>
#include <PxFoundationVersion.h>
#include <PxPvdObjectModelBaseTypes.h>

#include "OmniPvdWriter.h"
#include "OmniPvdSocketWriteStream.h"
#include "OmniPvdDefines.h"

// pvdruntime factory functions (compiled into pvddom in standalone mode)
extern "C" {
    OmniPvdWriter* OMNI_PVD_CALL createOmniPvdWriter();
    void OMNI_PVD_CALL destroyOmniPvdWriter(OmniPvdWriter& writer);
    OmniPvdSocketWriteStream* OMNI_PVD_CALL createOmniPvdSocketWriteStream(const char* address, uint16_t port, uint32_t sendTimeout);
    void OMNI_PVD_CALL destroyOmniPvdSocketWriteStream(OmniPvdSocketWriteStream& stream);
}

namespace tst_live_resample {

using namespace physx;
using namespace physx::pvdsdk;
using namespace pvdapp;

// Keep the fixed test block below the default Windows and Linux dynamic client
// port ranges; otherwise an unrelated outbound connection can prevent bind().
static const uint16_t kPort = 24873;
static const uint16_t kExistingRecordingPort = 24878;
static const uint16_t kCandidateRecordingPort = 24879;
static const uint16_t kIndexPublicationPort = 24880;
static const uint16_t kCleanEofPublicationPort = 24881;
static const uint16_t kSequentialCapturePort = 24882;

class OvdLiveResampleTest : public ::testing::Test
{
protected:
    class ErrorCb : public PxErrorCallback
    {
        void reportError(PxErrorCode::Enum, const char*, const char*, int) override {}
    };

    static bool isClass(PvdEngine* e, const InstanceDescription& inst, const char* name)
    {
        Option<ClassDescription> cls = e->getClass(inst.mClassId);
        return cls.hasValue() && ::strcmp(cls.getValue().mName.mName, name) == 0;
    }

    static std::vector<InstanceDescription> allInstances(PvdEngine* e)
    {
        PxU32 n = e->getNbInstances();
        std::vector<InstanceDescription> insts(n);
        e->getInstances(insts.data(), n);
        return insts;
    }

    static std::vector<int32_t> readRefList(PvdEngine* e, const InstanceDescription& inst, const char* propName)
    {
        int32_t buf[64] = { 0 };
        PxU32 n = e->getPropertyValue(inst.mInstPtr, propName, buf, 64);
        return std::vector<int32_t>(buf, buf + n);
    }

    // Poll pred() up to timeoutMs; true if it became true.
    template <typename Pred>
    static bool waitFor(Pred pred, int timeoutMs)
    {
        for (int waited = 0; waited < timeoutMs; waited += 25)
        {
            if (pred())
                return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
        return pred();
    }
};

// One producer "segment": register the schema (fresh handles each call since
// the writer's counters keep counting), create meta/scene/actor/shape under
// the FIXED external handles, wire the lists, and run frames 1..3 with the
// scene handle as the frame context.
static void writeSegment(OmniPvdWriter* w, bool firstSegment)
{
    const OmniPvdContextHandle ctx = 1;
    const OmniPvdObjectHandle hMeta = 0xCC0, hScene = 0xCC1, hActor = 0xCC2, hShape = 0xCC3;

    OmniPvdClassHandle metaCls = w->registerClass("PxOmniPvdMetaData");
    OmniPvdAttributeHandle verMaj = w->registerAttribute(metaCls, "ovdIntegrationVersionMajor", OmniPvdDataType::eUINT32, 1);
    OmniPvdAttributeHandle verMin = w->registerAttribute(metaCls, "ovdIntegrationVersionMinor", OmniPvdDataType::eUINT32, 1);
    OmniPvdClassHandle sceneCls = w->registerClass("PxScene");
    OmniPvdAttributeHandle actorsAttr = w->registerUniqueListAttribute(sceneCls, "actors", OmniPvdDataType::eOBJECT_HANDLE);
    OmniPvdClassHandle actorCls = w->registerClass("PxRigidDynamic");
    OmniPvdAttributeHandle shapesAttr = w->registerUniqueListAttribute(actorCls, "shapes", OmniPvdDataType::eOBJECT_HANDLE);
    OmniPvdClassHandle shapeCls = w->registerClass("PxShape");
    OmniPvdClassHandle actorTypeCls = w->registerClass("PxActorType");
    w->registerEnumValue(actorTypeCls, "eRIGID_DYNAMIC", 1);
    OmniPvdAttributeHandle typeAttr = w->registerFlagsAttribute(actorCls, "type", actorTypeCls);

    w->createObject(ctx, metaCls, hMeta, "");
    uint32_t vMaj = 3, vMin = 1;
    w->setAttribute(ctx, hMeta, verMaj, reinterpret_cast<const uint8_t*>(&vMaj), sizeof(vMaj));
    w->setAttribute(ctx, hMeta, verMin, reinterpret_cast<const uint8_t*>(&vMin), sizeof(vMin));

    // The snapshot re-creates shared objects BEFORE the scene (decoded PhysX
    // order); the first (live) segment creates the scene first.
    if (firstSegment)
        w->createObject(ctx, sceneCls, hScene, "");
    w->createObject(ctx, shapeCls, hShape, "");
    w->createObject(ctx, actorCls, hActor, "");
    uint32_t rigidDynamic = 1;
    w->setAttribute(ctx, hActor, typeAttr, reinterpret_cast<const uint8_t*>(&rigidDynamic), sizeof(rigidDynamic));
    w->addToUniqueListAttribute(ctx, hActor, shapesAttr, reinterpret_cast<const uint8_t*>(&hShape), sizeof(hShape));
    if (!firstSegment)
        w->createObject(ctx, sceneCls, hScene, "");
    w->startFrame(hScene, 1);
    w->addToUniqueListAttribute(ctx, hScene, actorsAttr, reinterpret_cast<const uint8_t*>(&hActor), sizeof(hActor));
    w->stopFrame(hScene, 1);
    w->startFrame(hScene, 2); w->stopFrame(hScene, 2);
    w->startFrame(hScene, 3); w->stopFrame(hScene, 3);
}

TEST_F(OvdLiveResampleTest, CandidateRaceIsRejectedWithoutOverwrite)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    const std::string path = root.file("capture.ovd");
    ASSERT_FALSE(path.empty());
    ASSERT_FALSE(test_files::fs::exists(test_files::fs::u8path(path)));

    // Simulate another process winning the path after candidate generation but
    // before connectLive. Exclusive create must fail closed and preserve it.
    const std::string sentinel("do not replace!");
    test_files::File file(path);
    ASSERT_TRUE(file.open(std::ios::out));
    ASSERT_EQ(file.write(sentinel), sentinel.size());
    file.close();

    pvdapp::CheckingAllocator alloc;
    ErrorCb errorCb;
    PxFoundation* foundation = PxCreateFoundation(PX_FOUNDATION_VERSION, alloc, errorCb);
    ASSERT_NE(foundation, nullptr);

    PvdEngine* engine = PvdEngine::connectLive(
        alloc, kExistingRecordingPort, path.c_str());
    EXPECT_EQ(engine, nullptr)
        << "a public live-recording path must be exclusively created, never reopened";
    delete engine;

    ASSERT_TRUE(file.open(std::ios::in));
    EXPECT_EQ(file.readAll(), sentinel);
    file.close();
    foundation->release();
}

TEST_F(OvdLiveResampleTest, CandidateIsExclusivelyCreatedAndRecordsLoadableStream)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    const std::string path = root.file("capture.ovd");
    ASSERT_FALSE(path.empty());
    ASSERT_FALSE(test_files::fs::exists(test_files::fs::u8path(path))) << "candidate helper must not leave a placeholder";

    pvdapp::CheckingAllocator alloc;
    ErrorCb errorCb;
    PxFoundation* foundation = PxCreateFoundation(PX_FOUNDATION_VERSION, alloc, errorCb);
    ASSERT_NE(foundation, nullptr);

    PvdEngine* engine = PvdEngine::connectLive(
        alloc, kCandidateRecordingPort, path.c_str());
    ASSERT_NE(engine, nullptr) << "exclusive creation of a new candidate failed";
    EXPECT_TRUE(test_files::fs::exists(test_files::fs::u8path(path)));
    EXPECT_EQ(engine->getLiveRecordingPath(), path);

    std::atomic<bool> producerOk{ false };
    std::thread producer([&]
    {
        OmniPvdWriter* writer = createOmniPvdWriter();
        OmniPvdSocketWriteStream* stream = createOmniPvdSocketWriteStream(
            "127.0.0.1", kCandidateRecordingPort, 3000);
        if (writer && stream && stream->openStream())
        {
            writer->setWriteStream(*stream);
            writeSegment(writer, true);
            stream->closeStream();
            producerOk = true;
        }
        if (stream)
            destroyOmniPvdSocketWriteStream(*stream);
        if (writer)
            destroyOmniPvdWriter(*writer);
    });

    const bool ended = waitFor([&] { return engine->isLiveEnded(); }, 10000);
    if (!ended)
        engine->stopLive();
    producer.join();
    EXPECT_TRUE(ended) << "recorded producer stream did not finish";
    EXPECT_TRUE(producerOk.load()) << "recorded producer failed";
    engine->stopLive(); // finalizes and closes the owned recording handle

    test_files::File recorded(path);
    ASSERT_TRUE(recorded.open(std::ios::in));
    EXPECT_GT(test_files::fs::file_size(test_files::fs::u8path(path)), 12u) << "recording contains no commands";
    recorded.close();

    const std::string tempBytes = root.path();
    const std::string pathBytes = path;
    PvdEngine* reloaded = PvdEngine::load(
        alloc, tempBytes.c_str(), pathBytes.c_str());
    EXPECT_NE(reloaded, nullptr) << "recording is not a loadable .ovd stream";
    delete reloaded;
    delete engine;
    foundation->release();
}

TEST_F(OvdLiveResampleTest, CleanEofPublishesIndexChangesAfterTheLastFrame)
{
    pvdapp::CheckingAllocator alloc;
    ErrorCb errorCb;
    PxFoundation* foundation = PxCreateFoundation(PX_FOUNDATION_VERSION, alloc, errorCb);
    ASSERT_NE(foundation, nullptr);

    PvdEngine* engine = PvdEngine::connectLive(alloc, kCleanEofPublicationPort);
    ASSERT_NE(engine, nullptr) << "could not open the live listener";

    std::atomic<bool> producerOk{ false };
    std::thread producer([&]
    {
        OmniPvdWriter* writer = createOmniPvdWriter();
        OmniPvdSocketWriteStream* stream = createOmniPvdSocketWriteStream(
            "127.0.0.1", kCleanEofPublicationPort, 3000);
        if (writer && stream && stream->openStream())
        {
            writer->setWriteStream(*stream);
            writeSegment(writer, true);
            writer->registerClass("TrailingClass");
            stream->closeStream();
            producerOk = true;
        }
        if (stream)
            destroyOmniPvdSocketWriteStream(*stream);
        if (writer)
            destroyOmniPvdWriter(*writer);
    });

    const bool ended = waitFor([&] { return engine->isLiveEnded(); }, 10000);
    if (!ended)
        engine->stopLive();
    producer.join();
    ASSERT_TRUE(ended) << "producer stream did not finish";
    ASSERT_TRUE(producerOk.load()) << "producer failed";
    ASSERT_EQ(engine->getLiveErrorCode(), OvdInputErrorCode::None);

    EXPECT_FALSE(engine->hasLiveUnpublishedIndexChanges())
        << "clean EOF must publish index changes after the final frame";
    engine->refreshLive();

    const PxU32 classCount = engine->getNbClasses();
    std::vector<ClassDescription> classes(classCount);
    engine->getClasses(classes.data(), classCount, 0);
    bool foundTrailingClass = false;
    for (const ClassDescription& cls : classes)
        foundTrailingClass |= ::strcmp(cls.mName.mName, "TrailingClass") == 0;
    EXPECT_TRUE(foundTrailingClass);

    delete engine;
    foundation->release();
}

TEST_F(OvdLiveResampleTest, CompletedCaptureRemainsReadableWhileNextListenerUsesSamePort)
{
    pvdapp::CheckingAllocator alloc;
    ErrorCb errorCb;
    PxFoundation* foundation = PxCreateFoundation(PX_FOUNDATION_VERSION, alloc, errorCb);
    ASSERT_NE(foundation, nullptr);

    const auto runProducer = [](std::atomic<bool>& producerOk)
    {
        OmniPvdWriter* writer = createOmniPvdWriter();
        OmniPvdSocketWriteStream* stream = createOmniPvdSocketWriteStream(
            "127.0.0.1", kSequentialCapturePort, 3000);
        if(writer && stream && stream->openStream())
        {
            writer->setWriteStream(*stream);
            writeSegment(writer, true);
            stream->closeStream();
            producerOk = true;
        }
        if(stream)
            destroyOmniPvdSocketWriteStream(*stream);
        if(writer)
            destroyOmniPvdWriter(*writer);
    };

    PvdEngine* current = PvdEngine::connectLive(alloc, kSequentialCapturePort);
    ASSERT_NE(current, nullptr) << "could not open the first live listener";
    std::atomic<bool> firstProducerOk{ false };
    std::thread firstProducer(runProducer, std::ref(firstProducerOk));
    const bool firstEnded = waitFor([&] { return current->isLiveEnded(); }, 10000);
    if(!firstEnded)
        current->stopLive();
    firstProducer.join();
    ASSERT_TRUE(firstEnded) << "first producer stream did not finish";
    ASSERT_TRUE(firstProducerOk.load()) << "first producer failed";
    ASSERT_EQ(current->getLiveErrorCode(), OvdInputErrorCode::None);

    current->stopLive(); // releases the port but intentionally retains the DOM
    current->refreshLive();
    const PxU32 currentInstances = current->getNbInstances();
    ASSERT_GT(currentInstances, 0u);

    PvdEngine* standby = PvdEngine::connectLive(alloc, kSequentialCapturePort);
    ASSERT_NE(standby, nullptr) << "could not rearm the same listen port";
    EXPECT_EQ(current->getNbInstances(), currentInstances)
        << "arming the next listener must not clear the displayed capture";

    std::atomic<bool> secondProducerOk{ false };
    std::thread secondProducer(runProducer, std::ref(secondProducerOk));
    const bool secondReady = waitFor([&]
    {
        return standby->getLiveMaxFrame() > 0 && !standby->hasLiveUnpublishedIndexChanges();
    }, 10000);
    if(!secondReady)
        standby->stopLive();
    secondProducer.join();
    EXPECT_TRUE(secondReady) << "second producer did not publish a complete frame";
    EXPECT_TRUE(secondProducerOk.load()) << "second producer failed";
    EXPECT_EQ(current->getNbInstances(), currentInstances)
        << "the previous capture must remain readable until UI promotion";

    delete standby;
    delete current;
    foundation->release();
}

TEST_F(OvdLiveResampleTest, MidConnectionResampleIngestsSegmentConsistently)
{
    pvdapp::CheckingAllocator alloc;
    ErrorCb errorCb;
    PxFoundation* foundation = PxCreateFoundation(PX_FOUNDATION_VERSION, alloc, errorCb);
    ASSERT_NE(foundation, nullptr);

    PvdEngine* engine = PvdEngine::connectLive(alloc, kPort);
    ASSERT_NE(engine, nullptr) << "could not open the live listener";

    std::atomic<bool> segmentAWritten{ false };
    std::atomic<bool> proceedToB{ false };
    std::atomic<bool> producerOk{ false };

    std::thread producer([&]
    {
        OmniPvdWriter* w = createOmniPvdWriter();
        OmniPvdSocketWriteStream* s = createOmniPvdSocketWriteStream("127.0.0.1", kPort, 3000);
        if (!w || !s || !s->openStream())
        {
            if (s) destroyOmniPvdSocketWriteStream(*s);
            if (w) destroyOmniPvdWriter(*w);
            segmentAWritten = true; // unblock the main thread; producerOk stays false
            return;
        }
        w->setWriteStream(*s);
        writeSegment(w, true);
        segmentAWritten = true;
        // Hold the connection open until the main thread has re-indexed segment A,
        // so the mid-stream rebuildIndex genuinely runs BETWEEN the segments.
        for (int waited = 0; !proceedToB.load() && waited < 10000; waited += 25)
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        writeSegment(w, false); // stop/startSampling: schema re-registered, world re-created
        s->closeStream();       // EOF -> the viewer sees the stream end
        destroyOmniPvdSocketWriteStream(*s);
        destroyOmniPvdWriter(*w);
        producerOk = true;
    });

    // Wait for segment A to be fully ingested, then re-index mid-stream.
    ASSERT_TRUE(waitFor([&] { return segmentAWritten.load() && engine->getLiveMaxFrame() >= 3; }, 10000))
        << "segment A did not ingest (connected=" << engine->isLiveConnected()
        << " ended=" << engine->isLiveEnded() << " maxFrame=" << engine->getLiveMaxFrame() << ")";
    engine->refreshLive();
    // A live adapter starts on an empty DOM at frame 0; move to the newest
    // ingested frame like the UI does after each refresh, so frame-filtered
    // reads (accumulated ref lists, aliveness) see the segment's data.
    engine->setCurrentFrame((PxU32)engine->getLiveMaxFrame(), 0);

    PxI32 sceneAId = -1;
    for (const InstanceDescription& inst : allInstances(engine))
        if (isClass(engine, inst, "PxScene"))
            sceneAId = (PxI32)inst.mId;
    ASSERT_GT(sceneAId, 0) << "segment A scene not indexed mid-stream";
    EXPECT_TRUE(engine->isTopLevel(sceneAId))
        << "a capture without PxPhysics must expose PxScene as an inspector root";

    // Let segment B flow, wait for the stream end, and re-index again.
    proceedToB = true;
    ASSERT_TRUE(waitFor([&] { return engine->isLiveEnded(); }, 10000)) << "stream did not end";
    producer.join();
    ASSERT_TRUE(producerOk.load()) << "producer failed to connect/stream";
    engine->refreshLive();
    engine->setCurrentFrame((PxU32)engine->getLiveMaxFrame(), 0);

    // Both segments' scenes exist; the segment-A scene kept its id (append-only
    // re-index across a schema re-registration).
    std::vector<PxI32> sceneIds;
    for (const InstanceDescription& inst : allInstances(engine))
        if (isClass(engine, inst, "PxScene"))
            sceneIds.push_back((PxI32)inst.mId);
    EXPECT_EQ(sceneIds.size(), 2u) << "expected one PxScene per recording segment";
    EXPECT_TRUE(std::find(sceneIds.begin(), sceneIds.end(), sceneAId) != sceneIds.end())
        << "segment A scene id changed across the mid-stream re-index";

    // Segment-consistent resolution, same invariant as the file-load tests.
    int refsChecked = 0;
    for (const InstanceDescription& scene : allInstances(engine))
    {
        if (!isClass(engine, scene, "PxScene"))
            continue;
        for (int32_t actorId : readRefList(engine, scene, "actors"))
        {
            if (actorId <= 0)
                continue;
            ++refsChecked;
            EXPECT_EQ(engine->getOwningSceneId(actorId), (PxI32)scene.mId)
                << "live: scene id=" << scene.mId << " 'actors' ref resolved to actor id="
                << actorId << " owned by a DIFFERENT segment's scene";
        }
    }
    EXPECT_GT(refsChecked, 0) << "no actor references resolved on the live path";

    delete engine;
    foundation->release();
}

TEST_F(OvdLiveResampleTest, RefreshDefersWhileNextFrameHasPartialSchema)
{
    pvdapp::CheckingAllocator alloc;
    ErrorCb errorCb;
    PxFoundation* foundation = PxCreateFoundation(PX_FOUNDATION_VERSION, alloc, errorCb);
    ASSERT_NE(foundation, nullptr);

    PvdEngine* engine = PvdEngine::connectLive(alloc, kIndexPublicationPort);
    ASSERT_NE(engine, nullptr) << "could not open the live listener";

    std::atomic<bool> initialFrameWritten{ false };
    std::atomic<bool> allowPartialSchema{ false };
    std::atomic<bool> partialSchemaWritten{ false };
    std::atomic<bool> allowSchemaCompletion{ false };
    std::atomic<bool> producerOk{ false };

    std::thread producer([&]
    {
        OmniPvdWriter* w = createOmniPvdWriter();
        OmniPvdSocketWriteStream* s = createOmniPvdSocketWriteStream(
            "127.0.0.1", kIndexPublicationPort, 3000);
        if (!w || !s || !s->openStream())
        {
            if (s) destroyOmniPvdSocketWriteStream(*s);
            if (w) destroyOmniPvdWriter(*w);
            initialFrameWritten = true;
            partialSchemaWritten = true;
            return;
        }

        w->setWriteStream(*s);
        const OmniPvdContextHandle ctx = 1;
        const OmniPvdObjectHandle scene = 0xD01;
        const OmniPvdObjectHandle lateObject = 0xD02;
        OmniPvdClassHandle sceneClass = w->registerClass("PxScene");
        w->createObject(ctx, sceneClass, scene, "");
        w->startFrame(scene, 1);
        w->stopFrame(scene, 1);
        initialFrameWritten = true;

        for (int waited = 0; !allowPartialSchema.load() && waited < 10000; waited += 25)
            std::this_thread::sleep_for(std::chrono::milliseconds(25));

        // Deliberately pause between the class and its attribute. Refreshing
        // here used to cache LateClass with an empty, append-only property list.
        OmniPvdClassHandle lateClass = w->registerClass("LateClass");
        partialSchemaWritten = true;

        for (int waited = 0; !allowSchemaCompletion.load() && waited < 10000; waited += 25)
            std::this_thread::sleep_for(std::chrono::milliseconds(25));

        OmniPvdAttributeHandle lateValue = w->registerAttribute(
            lateClass, "lateValue", OmniPvdDataType::eUINT32, 1);
        w->createObject(ctx, lateClass, lateObject, "");
        uint32_t value = 42;
        w->startFrame(scene, 2);
        w->setAttribute(ctx, lateObject, lateValue,
                        reinterpret_cast<const uint8_t*>(&value), sizeof(value));
        w->stopFrame(scene, 2);
        s->closeStream();
        destroyOmniPvdSocketWriteStream(*s);
        destroyOmniPvdWriter(*w);
        producerOk = true;
    });

    const bool firstFrameReady = waitFor([&]
    {
        return initialFrameWritten.load() && engine->getLiveCompletedFrameSerial() >= 1;
    }, 10000);
    EXPECT_TRUE(firstFrameReady) << "initial frame did not ingest";
    allowPartialSchema = true;

    const bool partialSchemaReady = waitFor([&]
    {
        return partialSchemaWritten.load() && engine->hasLiveUnpublishedIndexChanges();
    }, 10000);
    EXPECT_TRUE(partialSchemaReady) << "partial schema did not reach the live DOM";

    if (firstFrameReady && partialSchemaReady)
    {
        engine->refreshLive();
        EXPECT_EQ(engine->getNbClasses(), 0u)
            << "refresh published a completed-frame serial against later partial schema";
        EXPECT_EQ(engine->getNbInstances(), 0u);
        EXPECT_EQ(engine->getLiveErrorCode(), OvdInputErrorCode::None);
    }

    allowSchemaCompletion = true;
    const bool completed = waitFor([&]
    {
        return engine->isLiveEnded() && engine->getLiveCompletedFrameSerial() >= 2;
    }, 10000);
    if (!completed)
        engine->stopLive();
    producer.join();
    EXPECT_TRUE(completed) << "completed schema/frame did not ingest";
    EXPECT_TRUE(producerOk.load()) << "producer failed";

    if (completed)
    {
        EXPECT_FALSE(engine->hasLiveUnpublishedIndexChanges());
        engine->refreshLive();
        engine->setCurrentFrame((PxU32)engine->getLiveMaxFrame(), 0);

        PxI32 lateClassId = -1;
        for (const InstanceDescription& inst : allInstances(engine))
            if (isClass(engine, inst, "LateClass"))
                lateClassId = inst.mClassId;
        EXPECT_GE(lateClassId, 0) << "completed LateClass object was not indexed";
        if (lateClassId >= 0)
        {
            EXPECT_EQ(engine->getNbProperties(lateClassId), 1u);
            EXPECT_TRUE(engine->findProperty(lateClassId, "lateValue").hasValue())
                << "partial refresh permanently cached an empty LateClass schema";
        }
    }

    delete engine;
    foundation->release();
}

} // namespace tst_live_resample

#endif // PVD_OVD_SUPPORT
#endif // TST_OVD_LIVE_RESAMPLE_H
