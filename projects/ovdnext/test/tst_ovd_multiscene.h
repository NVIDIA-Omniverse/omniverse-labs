/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TST_OVD_MULTISCENE_H
#define TST_OVD_MULTISCENE_H

#ifdef PVD_OVD_SUPPORT

// Regression tests for multi-scene OVD captures where the writer reuses a
// PhysX pointer across scenes (a unit test that releases + recreates its
// scene records two PxScene objects, same api handle / distinct internal
// handles, with re-registered actors into one .ovd). The viewer must:
//   * keep the re-registered objects as DISTINCT renderable entities, and
//   * attribute each to its correct owning PxScene via the DOM tree
//     (mAncestor), NOT via the many-to-one external-handle actors map, so
//     hiding one scene isolates the other.
// Fixture: test/data/multiscene_rigid.ovd (~35 KB), generated from
//   SdkUnitTests_64.exe --gtest_filter=ActorTests.WorldBounds --omnipvdfile=...
// (WorldBounds calls releaseDefaultScene + createDefaultScene, so the same
// rigid-actor pointers are re-registered under a second PxScene). Rigid poses
// are tiny vs deformable per-frame verts, so this stays well under the 5 MiB
// GitLab no-LFS blob cap and is committed alongside the other small fixtures.

#include <cstring>
#include <fstream>
#include <memory>
#include <set>
#include <vector>

#include <gtest/gtest.h>
#include <PvdEngine.h>
#include <PvdSimpleAllocator.h>
#include <PxErrorCallback.h>
#include <PxFoundation.h>
#include <PxFoundationVersion.h>
#include <PxPvdObjectModelBaseTypes.h>

#include "OvdDomAdapter.h"
#include "OvdLiveSource.h"
#include "PvdDomParser.h"
#include "OmniPvdLibraryFunctions.h"
#include "OmniPvdMemoryStream.h"
#include "OmniPvdReader.h"
#include "OmniPvdWriter.h"

using namespace physx;
using namespace physx::pvdsdk;
using namespace pvdapp;

class OvdMultiSceneTest : public ::testing::Test
{
protected:
    class ErrorCb : public PxErrorCallback
    {
        void reportError(PxErrorCode::Enum, const char*, const char*, int) override {}
    };

    static PxAllocatorCallback* sAlloc;
    static ErrorCb* sErrorCb;
    static PxFoundation* sFoundation;
    static pvdapp::PvdEngine* sEngine;

    static void SetUpTestSuite()
    {
        sAlloc = new pvdapp::CheckingAllocator();
        sErrorCb = new ErrorCb();
        sFoundation = PxCreateFoundation(PX_FOUNDATION_VERSION, *sAlloc, *sErrorCb);
        sEngine = pvdapp::PvdEngine::load(*sAlloc, "./", TEST_DATA_DIR "/multiscene_rigid.ovd");
    }

    static void TearDownTestSuite()
    {
        delete sEngine;
        sFoundation->release();
        delete sErrorCb;
        delete sAlloc;
    }

    // True if the instance's class name (no namespace) equals `name`.
    static bool isClass(const InstanceDescription& inst, const char* name)
    {
        Option<ClassDescription> cls = sEngine->getClass(inst.mClassId);
        return cls.hasValue() && ::strcmp(cls.getValue().mName.mName, name) == 0;
    }

    static std::vector<InstanceDescription> allInstances()
    {
        PxU32 n = sEngine->getNbInstances();
        std::vector<InstanceDescription> insts(n);
        sEngine->getInstances(insts.data(), n);
        return insts;
    }
};

PxAllocatorCallback* OvdMultiSceneTest::sAlloc = nullptr;
OvdMultiSceneTest::ErrorCb* OvdMultiSceneTest::sErrorCb = nullptr;
PxFoundation* OvdMultiSceneTest::sFoundation = nullptr;
pvdapp::PvdEngine* OvdMultiSceneTest::sEngine = nullptr;

TEST(OvdDomAdapterIdentityTest, PublishedInternalKeyDoesNotAliasCollidingExternalHandle)
{
    OmniPvdWriter* writer = createOmniPvdWriter();
    OmniPvdMemoryStream* stream = createOmniPvdMemoryStream();
    ASSERT_NE(writer, nullptr);
    ASSERT_NE(stream, nullptr);
    ASSERT_GE(stream->setBufferSize(64 * 1024), 64 * 1024u);
    writer->setWriteStream(*stream->getWriteStream());

    const OmniPvdContextHandle ctx = 1;
    const OmniPvdObjectHandle firstExternalHandle = 0xA001;
    const OmniPvdObjectHandle collidingExternalHandle = 1; // first object's internal key
    const OmniPvdClassHandle firstClass = writer->registerClass("FirstObject");
    const OmniPvdAttributeHandle scalarTarget = writer->registerAttribute(
        firstClass, "scalarTarget", OmniPvdDataType::eOBJECT_HANDLE, 1);
    const OmniPvdAttributeHandle listTargets = writer->registerUniqueListAttribute(
        firstClass, "listTargets", OmniPvdDataType::eOBJECT_HANDLE);
    const OmniPvdClassHandle secondClass = writer->registerClass("SecondObject");
    writer->createObject(ctx, firstClass, firstExternalHandle, "first");
    writer->createObject(ctx, secondClass, collidingExternalHandle, "second");
    writer->setAttribute(ctx, firstExternalHandle, scalarTarget,
        reinterpret_cast<const uint8_t*>(&collidingExternalHandle), sizeof(collidingExternalHandle));
    writer->addToUniqueListAttribute(ctx, firstExternalHandle, listTargets,
        reinterpret_cast<const uint8_t*>(&collidingExternalHandle), sizeof(collidingExternalHandle));
    ASSERT_TRUE(stream->getWriteStream()->closeStream());

    OmniPvdReader* reader = createOmniPvdReader();
    ASSERT_NE(reader, nullptr);
    reader->setReadStream(*stream->getReadStream());
    OmniPvdVersionType major = 0, minor = 0, patch = 0;
    ASSERT_TRUE(reader->startReading(major, minor, patch));
    std::unique_ptr<OmniPvdDOMState> dom(new OmniPvdDOMState());
    ASSERT_TRUE(buildPvdDomState(reader, *dom));
    destroyOmniPvdReader(*reader);

    OvdDomAdapter adapter(std::move(dom), {});
    std::vector<InstanceDescription> instances(adapter.getNbInstances());
    ASSERT_EQ(adapter.getInstances(instances.data(), static_cast<PxU32>(instances.size())), 2u);

    const InstanceDescription* first = nullptr;
    const InstanceDescription* second = nullptr;
    for (const InstanceDescription& instance : instances)
    {
        const char* className = adapter.getClass(instance.mClassId).getValue().mName.mName;
        if (::strcmp(className, "FirstObject") == 0) first = &instance;
        if (::strcmp(className, "SecondObject") == 0) second = &instance;
    }
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    ASSERT_EQ(reinterpret_cast<PxU64>(first->mInstPtr), collidingExternalHandle);

    EXPECT_STREQ(adapter.getClassOf(first->mInstPtr).mName.mName, "FirstObject");
    EXPECT_EQ(adapter.resolveInstanceId(collidingExternalHandle), second->mId)
        << "external references must retain their own namespace";

    Option<PropertyDescription> scalarProp = adapter.findProperty(first->mClassId, "scalarTarget");
    Option<PropertyDescription> listProp = adapter.findProperty(first->mClassId, "listTargets");
    ASSERT_TRUE(scalarProp.hasValue());
    ASSERT_TRUE(listProp.hasValue());
    EXPECT_EQ(adapter.getPropertyByteSize(first->mInstPtr, scalarProp.getValue().mPropertyId),
              sizeof(PxI32))
        << "scalar eOBJECT_HANDLE size must match the converted instance ID width";
    PxI32 scalarId = -1;
    ASSERT_EQ(adapter.getPropertyValue(first->mInstPtr, scalarProp.getValue().mPropertyId,
                                       &scalarId, sizeof(scalarId)), sizeof(scalarId));
    EXPECT_EQ(scalarId, second->mId)
        << "scalar eOBJECT_HANDLE must resolve in the external-reference namespace";
    PxI32 listId = -1;
    ASSERT_EQ(adapter.getNbArrayElements(first->mInstPtr, listProp.getValue().mPropertyId), 1u);
    ASSERT_EQ(adapter.getPropertyValue(first->mInstPtr, listProp.getValue().mPropertyId,
                                       &listId, sizeof(listId)), sizeof(listId));
    EXPECT_EQ(listId, second->mId)
        << "list eOBJECT_HANDLE must resolve in the external-reference namespace";

    destroyOmniPvdWriter(*writer);
    stream->getReadStream()->closeStream();
    destroyOmniPvdMemoryStream(*stream);
}

TEST(OvdDomAdapterIdentityTest, KnownSegmentDoesNotFallBackToPriorSegmentCandidates)
{
    OmniPvdWriter* writer = createOmniPvdWriter();
    OmniPvdMemoryStream* stream = createOmniPvdMemoryStream();
    ASSERT_NE(writer, nullptr);
    ASSERT_NE(stream, nullptr);
    ASSERT_GE(stream->setBufferSize(64 * 1024), 64 * 1024u);
    writer->setWriteStream(*stream->getWriteStream());

    struct Schema
    {
        OmniPvdClassHandle targetClass;
        OmniPvdClassHandle referrerClass;
        OmniPvdAttributeHandle soleTarget;
        OmniPvdAttributeHandle multiTarget;
        OmniPvdAttributeHandle listTargets;
    };
    const auto registerSchema = [writer]()
    {
        Schema schema;
        schema.targetClass = writer->registerClass("Target");
        schema.referrerClass = writer->registerClass("Referrer");
        schema.soleTarget = writer->registerAttribute(
            schema.referrerClass, "soleTarget", OmniPvdDataType::eOBJECT_HANDLE, 1);
        schema.multiTarget = writer->registerAttribute(
            schema.referrerClass, "multiTarget", OmniPvdDataType::eOBJECT_HANDLE, 1);
        schema.listTargets = writer->registerUniqueListAttribute(
            schema.referrerClass, "listTargets", OmniPvdDataType::eOBJECT_HANDLE);
        return schema;
    };

    const OmniPvdContextHandle ctx = 1;
    const OmniPvdObjectHandle soleTargetHandle = 0xD001;
    const OmniPvdObjectHandle multiTargetHandle = 0xD002;
    const OmniPvdObjectHandle referrerHandle = 0xD003;

    const Schema first = registerSchema();
    writer->createObject(ctx, first.targetClass, soleTargetHandle, "sole-first-segment");
    writer->createObject(ctx, first.targetClass, multiTargetHandle, "multi-first-segment");

    const Schema second = registerSchema();
    writer->createObject(ctx, second.targetClass, multiTargetHandle, "multi-second-segment");

    const Schema third = registerSchema();
    writer->createObject(ctx, third.referrerClass, referrerHandle, "third-segment-referrer");
    writer->setAttribute(ctx, referrerHandle, third.soleTarget,
        reinterpret_cast<const uint8_t*>(&soleTargetHandle), sizeof(soleTargetHandle));
    writer->setAttribute(ctx, referrerHandle, third.multiTarget,
        reinterpret_cast<const uint8_t*>(&multiTargetHandle), sizeof(multiTargetHandle));
    writer->addToUniqueListAttribute(ctx, referrerHandle, third.listTargets,
        reinterpret_cast<const uint8_t*>(&soleTargetHandle), sizeof(soleTargetHandle));
    writer->addToUniqueListAttribute(ctx, referrerHandle, third.listTargets,
        reinterpret_cast<const uint8_t*>(&multiTargetHandle), sizeof(multiTargetHandle));
    ASSERT_TRUE(stream->getWriteStream()->closeStream());

    OmniPvdReader* reader = createOmniPvdReader();
    ASSERT_NE(reader, nullptr);
    reader->setReadStream(*stream->getReadStream());
    OmniPvdVersionType major = 0, minor = 0, patch = 0;
    ASSERT_TRUE(reader->startReading(major, minor, patch));
    std::unique_ptr<OmniPvdDOMState> dom(new OmniPvdDOMState());
    ASSERT_TRUE(buildPvdDomState(reader, *dom));
    destroyOmniPvdReader(*reader);

    OvdDomAdapter adapter(std::move(dom), {});
    std::vector<InstanceDescription> instances(adapter.getNbInstances());
    adapter.getInstances(instances.data(), static_cast<PxU32>(instances.size()));
    const InstanceDescription* referrer = nullptr;
    for (const InstanceDescription& instance : instances)
    {
        const char* className = adapter.getClass(instance.mClassId).getValue().mName.mName;
        if (::strcmp(className, "Referrer") == 0)
            referrer = &instance;
    }
    ASSERT_NE(referrer, nullptr);

    EXPECT_EQ(adapter.resolveInstanceId(soleTargetHandle, referrer->mInstPtr), -1)
        << "a sole candidate from an older segment must remain unresolved";
    EXPECT_EQ(adapter.resolveInstanceId(multiTargetHandle, referrer->mInstPtr), -1)
        << "multiple candidates from older segments must remain unresolved";

    Option<PropertyDescription> soleProp = adapter.findProperty(referrer->mClassId, "soleTarget");
    Option<PropertyDescription> multiProp = adapter.findProperty(referrer->mClassId, "multiTarget");
    Option<PropertyDescription> listProp = adapter.findProperty(referrer->mClassId, "listTargets");
    ASSERT_TRUE(soleProp.hasValue());
    ASSERT_TRUE(multiProp.hasValue());
    ASSERT_TRUE(listProp.hasValue());

    PxI32 soleId = 0;
    PxI32 multiId = 0;
    ASSERT_EQ(adapter.getPropertyValue(referrer->mInstPtr, soleProp.getValue().mPropertyId,
                                       &soleId, sizeof(soleId)), sizeof(soleId));
    ASSERT_EQ(adapter.getPropertyValue(referrer->mInstPtr, multiProp.getValue().mPropertyId,
                                       &multiId, sizeof(multiId)), sizeof(multiId));
    EXPECT_EQ(soleId, -1);
    EXPECT_EQ(multiId, -1);

    PxI32 listIds[2] = { 0, 0 };
    ASSERT_EQ(adapter.getNbArrayElements(referrer->mInstPtr,
                                         listProp.getValue().mPropertyId), 2u);
    ASSERT_EQ(adapter.getPropertyValue(referrer->mInstPtr, listProp.getValue().mPropertyId,
                                       listIds, sizeof(listIds)), sizeof(listIds));
    EXPECT_EQ(listIds[0], -1);
    EXPECT_EQ(listIds[1], -1);

    destroyOmniPvdWriter(*writer);
    stream->getReadStream()->closeStream();
    destroyOmniPvdMemoryStream(*stream);
}

TEST(OvdDomAdapterIdentityTest, DomSegmentMetadataOverridesChangedSchemaOrder)
{
    OmniPvdWriter* writer = createOmniPvdWriter();
    OmniPvdMemoryStream* stream = createOmniPvdMemoryStream();
    ASSERT_NE(writer, nullptr);
    ASSERT_NE(stream, nullptr);
    ASSERT_GE(stream->setBufferSize(64 * 1024), 64 * 1024u);
    writer->setWriteStream(*stream->getWriteStream());

    const OmniPvdContextHandle ctx = 1;
    const OmniPvdObjectHandle firstMetadataHandle = 0xE001;
    const OmniPvdObjectHandle secondMetadataHandle = 0xE002;
    const OmniPvdObjectHandle targetHandle = 0xE003;
    const OmniPvdObjectHandle addedReferrerHandle = 0xE004;

    const OmniPvdClassHandle firstMetadataClass = writer->registerClass("PxOmniPvdMetaData");
    const OmniPvdClassHandle firstTargetClass = writer->registerClass("Target");
    writer->createObject(ctx, firstMetadataClass, firstMetadataHandle, "first-segment-metadata");
    writer->createObject(ctx, firstTargetClass, targetHandle, "first-segment-target");

    // A newly introduced class precedes the first repeated class in segment B.
    // Class-wave inference labels AddedReferrer as segment A, while pvddom's
    // object metadata records its exact segment after the metadata object.
    const OmniPvdClassHandle addedReferrerClass = writer->registerClass("AddedReferrer");
    const OmniPvdAttributeHandle targetAttribute = writer->registerAttribute(
        addedReferrerClass, "target", OmniPvdDataType::eOBJECT_HANDLE, 1);
    const OmniPvdClassHandle secondMetadataClass = writer->registerClass("PxOmniPvdMetaData");
    const OmniPvdClassHandle secondTargetClass = writer->registerClass("Target");
    writer->createObject(ctx, secondMetadataClass, secondMetadataHandle, "second-segment-metadata");
    writer->createObject(ctx, secondTargetClass, targetHandle, "second-segment-target");
    writer->createObject(ctx, addedReferrerClass, addedReferrerHandle, "second-segment-referrer");
    writer->setAttribute(ctx, addedReferrerHandle, targetAttribute,
        reinterpret_cast<const uint8_t*>(&targetHandle), sizeof(targetHandle));
    ASSERT_TRUE(stream->getWriteStream()->closeStream());

    OmniPvdReader* reader = createOmniPvdReader();
    ASSERT_NE(reader, nullptr);
    reader->setReadStream(*stream->getReadStream());
    OmniPvdVersionType major = 0, minor = 0, patch = 0;
    ASSERT_TRUE(reader->startReading(major, minor, patch));
    std::unique_ptr<OmniPvdDOMState> dom(new OmniPvdDOMState());
    ASSERT_TRUE(buildPvdDomState(reader, *dom));
    destroyOmniPvdReader(*reader);
    ASSERT_TRUE(dom->mSawRecordingSegmentMetadata);

    OvdDomAdapter adapter(std::move(dom), {});
    std::vector<InstanceDescription> instances(adapter.getNbInstances());
    adapter.getInstances(instances.data(), static_cast<PxU32>(instances.size()));
    const InstanceDescription* referrer = nullptr;
    PxI32 secondSegmentTargetId = -1;
    for (const InstanceDescription& instance : instances)
    {
        const char* className = adapter.getClass(instance.mClassId).getValue().mName.mName;
        const OmniPvdObjectHandle internalKey = reinterpret_cast<OmniPvdObjectHandle>(instance.mInstPtr);
        OmniPvdObject* object = adapter.dom()->mObjectHandleToObjectMap[internalKey];
        ASSERT_NE(object, nullptr);
        if (::strcmp(className, "AddedReferrer") == 0)
        {
            referrer = &instance;
            EXPECT_EQ(object->mRecordingSegmentId, 1u);
        }
        else if (::strcmp(className, "Target") == 0 && object->mRecordingSegmentId == 1)
        {
            secondSegmentTargetId = instance.mId;
        }
    }
    ASSERT_NE(referrer, nullptr);
    ASSERT_GT(secondSegmentTargetId, 0);

    EXPECT_EQ(adapter.resolveInstanceId(targetHandle, referrer->mInstPtr), secondSegmentTargetId);
    Option<PropertyDescription> targetProp = adapter.findProperty(referrer->mClassId, "target");
    ASSERT_TRUE(targetProp.hasValue());
    PxI32 resolvedTargetId = -1;
    ASSERT_EQ(adapter.getPropertyValue(referrer->mInstPtr, targetProp.getValue().mPropertyId,
                                       &resolvedTargetId, sizeof(resolvedTargetId)),
              sizeof(resolvedTargetId));
    EXPECT_EQ(resolvedTargetId, secondSegmentTargetId);

    destroyOmniPvdWriter(*writer);
    stream->getReadStream()->closeStream();
    destroyOmniPvdMemoryStream(*stream);
}

TEST(OvdDomAdapterEpochTest, SchemaReregistrationAdvancesSegmentEpoch)
{
    OmniPvdWriter* writer = createOmniPvdWriter();
    OmniPvdMemoryStream* stream = createOmniPvdMemoryStream();
    ASSERT_NE(writer, nullptr);
    ASSERT_NE(stream, nullptr);
    ASSERT_GE(stream->setBufferSize(64 * 1024), 64 * 1024u);
    writer->setWriteStream(*stream->getWriteStream());

    const OmniPvdContextHandle ctx = 1;
    const OmniPvdClassHandle firstClass = writer->registerClass("RepeatedSchema");
    writer->createObject(ctx, firstClass, 0xC001, "first");
    const OmniPvdClassHandle secondClass = writer->registerClass("RepeatedSchema");
    writer->createObject(ctx, secondClass, 0xC002, "second");
    ASSERT_TRUE(stream->getWriteStream()->closeStream());

    OmniPvdReader* reader = createOmniPvdReader();
    ASSERT_NE(reader, nullptr);
    reader->setReadStream(*stream->getReadStream());
    OmniPvdVersionType major = 0, minor = 0, patch = 0;
    ASSERT_TRUE(reader->startReading(major, minor, patch));
    std::unique_ptr<OmniPvdDOMState> dom(new OmniPvdDOMState());
    ASSERT_TRUE(buildPvdDomState(reader, *dom));
    destroyOmniPvdReader(*reader);

    OvdDomAdapter adapter(std::move(dom), {});
    EXPECT_EQ(adapter.getSegmentEpoch(), 1u);

    destroyOmniPvdWriter(*writer);
    stream->getReadStream()->closeStream();
    destroyOmniPvdMemoryStream(*stream);
}

TEST(OvdLiveSegmentEpochStateTest, PublishesSchemaEpochOnlyAtCompletedFrameBoundary)
{
    OvdLiveSegmentEpochState state;

    state.observeClassRegistration("RepeatedSchema");
    EXPECT_EQ(state.completeFrame(), 0u);
    // An old-segment frame may complete without a UI poll.
    EXPECT_EQ(state.completeFrame(), 0u);
    // The next schema begins, but its epoch remains unpublished until its own
    // STOP_FRAME instead of being paired with the unobserved old boundary.
    state.observeClassRegistration("RepeatedSchema");
    EXPECT_EQ(state.completedEpoch(), 0u);
    EXPECT_EQ(state.completeFrame(), 1u);
}

TEST(OvdLivePollDecisionTest, EqualRawFrameInNewSegmentInvalidatesCacheAndRerenders)
{
    const OvdLivePollDecision decision = decideOvdLivePollUpdate(0, 7, 1, 7);
    EXPECT_TRUE(decision.invalidateFrameCache);
    EXPECT_TRUE(decision.followTargetChanged);
}

TEST(OvdLiveSegmentEpochStateTest, ExactMetadataDetectsDisjointSchemaAtEqualRawFrame)
{
    OvdLiveSegmentEpochState state;

    // A class-name wave cannot detect a re-attach whose schema is disjoint.
    state.observeClassRegistration("FirstSegmentOnly");
    const uint64_t firstEpoch = state.completeFrame(true, 0);
    state.observeClassRegistration("SecondSegmentOnly");
    const uint64_t secondEpoch = state.completeFrame(true, 1);

    EXPECT_EQ(firstEpoch, 0u);
    EXPECT_EQ(secondEpoch, 1u);
    const OvdLivePollDecision decision = decideOvdLivePollUpdate(
        firstEpoch, 7, secondEpoch, 7);
    EXPECT_TRUE(decision.invalidateFrameCache);
    EXPECT_TRUE(decision.followTargetChanged);
}

TEST(OvdDomAdapterRootTest, MixedPhysicsAndRootlessScenesStayReachableAfterRebuild)
{
    OmniPvdWriter* writer = createOmniPvdWriter();
    OmniPvdMemoryStream* stream = createOmniPvdMemoryStream();
    ASSERT_NE(writer, nullptr);
    ASSERT_NE(stream, nullptr);
    ASSERT_GE(stream->setBufferSize(64 * 1024), 64 * 1024u);
    writer->setWriteStream(*stream->getWriteStream());

    const OmniPvdContextHandle ctx = 1;
    const OmniPvdObjectHandle physicsHandle = 0xB000;
    const OmniPvdObjectHandle nestedSceneHandle = 0xB001;
    const OmniPvdObjectHandle rootlessSceneHandle = 0xB002;
    const OmniPvdClassHandle physicsClass = writer->registerClass("PxPhysics");
    const OmniPvdAttributeHandle physicsScenes = writer->registerUniqueListAttribute(
        physicsClass, "scenes", OmniPvdDataType::eOBJECT_HANDLE);
    const OmniPvdClassHandle sceneClass = writer->registerClass("PxScene");
    writer->createObject(ctx, physicsClass, physicsHandle, "physics");
    writer->createObject(ctx, sceneClass, nestedSceneHandle, "nested");
    writer->createObject(ctx, sceneClass, rootlessSceneHandle, "rootless");
    writer->addToUniqueListAttribute(ctx, physicsHandle, physicsScenes,
        reinterpret_cast<const uint8_t*>(&nestedSceneHandle), sizeof(nestedSceneHandle));
    ASSERT_TRUE(stream->getWriteStream()->closeStream());

    OmniPvdReader* reader = createOmniPvdReader();
    ASSERT_NE(reader, nullptr);
    reader->setReadStream(*stream->getReadStream());
    OmniPvdVersionType major = 0, minor = 0, patch = 0;
    ASSERT_TRUE(reader->startReading(major, minor, patch));
    std::unique_ptr<OmniPvdDOMState> dom(new OmniPvdDOMState());
    ASSERT_TRUE(buildPvdDomState(reader, *dom));
    destroyOmniPvdReader(*reader);
    destroyOmniPvdWriter(*writer);
    stream->getReadStream()->closeStream();
    destroyOmniPvdMemoryStream(*stream);

    const OmniPvdObjectHandle physicsKey = dom->mExternalToInternalHandleMap[physicsHandle];
    const OmniPvdObjectHandle nestedSceneKey = dom->mExternalToInternalHandleMap[nestedSceneHandle];
    const OmniPvdObjectHandle rootlessSceneKey = dom->mExternalToInternalHandleMap[rootlessSceneHandle];
    OmniPvdObject* physics = dom->mObjectHandleToObjectMap[physicsKey];
    OmniPvdObject* nestedScene = dom->mObjectHandleToObjectMap[nestedSceneKey];
    OmniPvdObject* rootlessScene = dom->mObjectHandleToObjectMap[rootlessSceneKey];
    ASSERT_NE(physics, nullptr);
    ASSERT_NE(nestedScene, nullptr);
    ASSERT_NE(rootlessScene, nullptr);
    ASSERT_EQ(nestedScene->mAncestor, physics);
    ASSERT_NE(rootlessScene->mAncestor, nullptr);
    rootlessScene->mAncestor->removeChild(rootlessScene);
    dom->mSceneRoot->appendChild(rootlessScene);

    // Add the rootless scene through the live/incremental indexing path.
    dom->mObjectHandleToObjectMap.erase(rootlessSceneKey);
    OvdDomAdapter adapter(std::move(dom), {});
    adapter.dom()->mObjectHandleToObjectMap[rootlessSceneKey] = rootlessScene;
    adapter.rebuildIndex();

    std::vector<InstanceDescription> instances(adapter.getNbInstances());
    adapter.getInstances(instances.data(), static_cast<PxU32>(instances.size()));
    PxI32 physicsId = -1, nestedSceneId = -1, rootlessSceneId = -1;
    for (const InstanceDescription& instance : instances)
    {
        const PxU64 key = reinterpret_cast<PxU64>(instance.mInstPtr);
        if (key == physicsKey) physicsId = instance.mId;
        if (key == nestedSceneKey) nestedSceneId = instance.mId;
        if (key == rootlessSceneKey) rootlessSceneId = instance.mId;
    }
    ASSERT_GT(physicsId, 0);
    ASSERT_GT(nestedSceneId, 0);
    ASSERT_GT(rootlessSceneId, 0);
    EXPECT_TRUE(adapter.isTopLevel(physicsId));
    EXPECT_FALSE(adapter.isTopLevel(nestedSceneId));
    EXPECT_TRUE(adapter.isTopLevel(rootlessSceneId));
}

TEST(OvdDomAdapterTopologyTest, RejectsSceneActorParentCycleWithoutPublishingInstances)
{
    OmniPvdWriter* writer = createOmniPvdWriter();
    OmniPvdMemoryStream* stream = createOmniPvdMemoryStream();
    ASSERT_NE(writer, nullptr);
    ASSERT_NE(stream, nullptr);
    ASSERT_GE(stream->setBufferSize(64 * 1024), 64 * 1024u);
    writer->setWriteStream(*stream->getWriteStream());

    const OmniPvdContextHandle ctx = 1;
    const OmniPvdObjectHandle physicsHandle = 0xBA00;
    const OmniPvdObjectHandle sceneHandle = 0xBA01;
    const OmniPvdObjectHandle actorHandle = 0xBA02;
    const OmniPvdClassHandle physicsClass = writer->registerClass("PxPhysics");
    const OmniPvdAttributeHandle physicsScenes = writer->registerUniqueListAttribute(
        physicsClass, "scenes", OmniPvdDataType::eOBJECT_HANDLE);
    const OmniPvdClassHandle sceneClass = writer->registerClass("PxScene");
    const OmniPvdAttributeHandle sceneActors = writer->registerUniqueListAttribute(
        sceneClass, "actors", OmniPvdDataType::eOBJECT_HANDLE);
    const OmniPvdClassHandle actorClass = writer->registerClass("PxActor");
    const OmniPvdAttributeHandle actorShapes = writer->registerUniqueListAttribute(
        actorClass, "shapes", OmniPvdDataType::eOBJECT_HANDLE);

    writer->createObject(ctx, physicsClass, physicsHandle, "physics");
    writer->createObject(ctx, sceneClass, sceneHandle, "scene");
    writer->createObject(ctx, actorClass, actorHandle, "actor");
    writer->addToUniqueListAttribute(ctx, physicsHandle, physicsScenes,
        reinterpret_cast<const uint8_t*>(&sceneHandle), sizeof(sceneHandle));
    writer->addToUniqueListAttribute(ctx, sceneHandle, sceneActors,
        reinterpret_cast<const uint8_t*>(&actorHandle), sizeof(actorHandle));
    // Malicious but schema-valid: the parser trusts the referenced handle's
    // class and reparents the PxScene under its own PxActor descendant.
    writer->addToUniqueListAttribute(ctx, actorHandle, actorShapes,
        reinterpret_cast<const uint8_t*>(&sceneHandle), sizeof(sceneHandle));
    ASSERT_TRUE(stream->getWriteStream()->closeStream());

    OmniPvdReader* reader = createOmniPvdReader();
    ASSERT_NE(reader, nullptr);
    reader->setReadStream(*stream->getReadStream());
    OmniPvdVersionType major = 0, minor = 0, patch = 0;
    ASSERT_TRUE(reader->startReading(major, minor, patch));
    std::unique_ptr<OmniPvdDOMState> dom(new OmniPvdDOMState());
    ASSERT_TRUE(buildPvdDomState(reader, *dom));
    destroyOmniPvdReader(*reader);
    destroyOmniPvdWriter(*writer);
    stream->getReadStream()->closeStream();
    destroyOmniPvdMemoryStream(*stream);

    OmniPvdObject* scene = dom->mObjectHandleToObjectMap[
        dom->mExternalToInternalHandleMap[sceneHandle]];
    OmniPvdObject* actor = dom->mObjectHandleToObjectMap[
        dom->mExternalToInternalHandleMap[actorHandle]];
    ASSERT_NE(scene, nullptr);
    ASSERT_NE(actor, nullptr);
    ASSERT_EQ(scene->mAncestor, actor);
    ASSERT_EQ(actor->mAncestor, scene);

    OvdDomAdapter adapter(std::move(dom), {});
    EXPECT_FALSE(adapter.isValid());
    EXPECT_EQ(adapter.errorCode(), OvdInputErrorCode::InvalidDomTopology);
    EXPECT_EQ(adapter.getNbInstances(), 0u);
}

TEST_F(OvdMultiSceneTest, FixtureLoads)
{
    ASSERT_NE(sEngine, nullptr);
    EXPECT_GT(sEngine->getNbInstances(), 0u);
}

// The scene-recreate produces at least two PxScene objects in the DOM.
TEST_F(OvdMultiSceneTest, HasAtLeastTwoSceneInstances)
{
    ASSERT_NE(sEngine, nullptr);
    int sceneCount = 0;
    for (const InstanceDescription& inst : allInstances())
        if (isClass(inst, "PxScene"))
            ++sceneCount;
    EXPECT_GE(sceneCount, 2) << "expected the scene-recreate capture to hold >=2 PxScene objects";
}

// A PxScene has no owning PxScene of its own.
TEST_F(OvdMultiSceneTest, SceneHasNoOwningScene)
{
    ASSERT_NE(sEngine, nullptr);
    for (const InstanceDescription& inst : allInstances())
        if (isClass(inst, "PxScene"))
            EXPECT_EQ(sEngine->getOwningSceneId((PxI32)inst.mId), -1)
                << "PxScene id=" << inst.mId << " should report no owning scene";
}

// Every rigid actor resolves to an owning PxScene, and that id is a real
// PxScene instance in the capture.
TEST_F(OvdMultiSceneTest, EveryRigidActorHasValidOwningScene)
{
    ASSERT_NE(sEngine, nullptr);
    std::set<int> sceneIds;
    for (const InstanceDescription& inst : allInstances())
        if (isClass(inst, "PxScene"))
            sceneIds.insert((int)inst.mId);
    ASSERT_FALSE(sceneIds.empty());

    int actorsChecked = 0;
    for (const InstanceDescription& inst : allInstances())
    {
        if (!isClass(inst, "PxRigidDynamic") && !isClass(inst, "PxRigidStatic"))
            continue;
        ++actorsChecked;
        const PxI32 owner = sEngine->getOwningSceneId((PxI32)inst.mId);
        EXPECT_GT(owner, 0) << "rigid actor id=" << inst.mId << " has no owning scene";
        EXPECT_TRUE(sceneIds.count(owner) > 0)
            << "rigid actor id=" << inst.mId << " owning scene " << owner
            << " is not a PxScene instance";
    }
    EXPECT_GT(actorsChecked, 0) << "fixture should contain rigid actors";
}

// The crux: a capture that puts actors into TWO scenes while reusing the same
// PhysX pointers (the writer re-registers each object under each scene) must
// have its actors partition across BOTH owning scenes. The collapsing
// external-handle actors map would funnel them all onto the newest scene; the
// mAncestor (DOM-tree) resolution behind getOwningSceneId keeps them split.
//
// This needs a capture with TWO *populated* scenes, which among our fixtures
// only the deformable VolumeVolumeCollision capture provides (its unit test
// runs collision-on then collision-off into one OVD). That file is ~18 MB —
// over the GitLab no-LFS blob cap — so it is gitignored, not committed. When
// it's present (locally, or regenerated in CI via
//   SdkUnitTests_64.exe --gtest_filter=*VolumeVolumeCollision* --omnipvdfile=<...>)
// this test runs; otherwise it skips. The small committed multiscene_rigid.ovd
// only has one populated scene (ActorTests.WorldBounds populates just the
// recreated scene), so it can't exercise the partition.
TEST(OvdMultiSceneDeformableTest, DeformablesPartitionAcrossScenes)
{
    const char* candidates[] = {
        TEST_DATA_DIR "/VolumeVolumeCollision_regen.ovd",
        TEST_DATA_DIR "/VolumeVolumeCollision.ovd",
    };
    const char* path = nullptr;
    for (const char* c : candidates)
    {
        std::ifstream f(c, std::ios::binary);
        if (f.good()) { path = c; break; }
    }
    if (!path)
        GTEST_SKIP() << "VolumeVolumeCollision[_regen].ovd not present (gitignored, "
                        "regenerate via SdkUnitTests --gtest_filter=*VolumeVolumeCollision* --omnipvdfile=)";

    class ErrCb : public PxErrorCallback
    { void reportError(PxErrorCode::Enum, const char*, const char*, int) override {} };
    PxAllocatorCallback* alloc = new pvdapp::CheckingAllocator();
    ErrCb errCb;
    PxFoundation* foundation = PxCreateFoundation(PX_FOUNDATION_VERSION, *alloc, errCb);
    pvdapp::PvdEngine* eng = pvdapp::PvdEngine::load(*alloc, "./", path);
    ASSERT_NE(eng, nullptr);

    PxU32 n = eng->getNbInstances();
    std::vector<InstanceDescription> insts(n);
    eng->getInstances(insts.data(), n);

    std::set<int> distinctOwningScenes;
    int deformables = 0;
    for (const InstanceDescription& inst : insts)
    {
        Option<ClassDescription> cls = eng->getClass(inst.mClassId);
        if (!cls.hasValue()) continue;
        const char* cn = cls.getValue().mName.mName;
        if (::strcmp(cn, "PxDeformableVolume") != 0 && ::strcmp(cn, "PxDeformableSurface") != 0)
            continue;
        ++deformables;
        const PxI32 owner = eng->getOwningSceneId((PxI32)inst.mId);
        if (owner > 0)
            distinctOwningScenes.insert(owner);
    }

    EXPECT_GT(deformables, 0);
    EXPECT_GE((int)distinctOwningScenes.size(), 2)
        << "deformables should be attributed to >=2 distinct owning scenes via "
           "the DOM tree (the pointer-reuse split the external-handle map collapses)";

    delete eng;
    foundation->release();
    delete alloc;
}

// Factory children that live under PxPhysics, not a scene, report no owning
// scene (so per-scene Vis never spuriously hides shared materials/meshes).
TEST_F(OvdMultiSceneTest, FactoryChildrenHaveNoOwningScene)
{
    ASSERT_NE(sEngine, nullptr);
    int checked = 0;
    for (const InstanceDescription& inst : allInstances())
    {
        if (!isClass(inst, "PxMaterial"))
            continue;
        ++checked;
        EXPECT_EQ(sEngine->getOwningSceneId((PxI32)inst.mId), -1)
            << "PxMaterial id=" << inst.mId << " should have no owning scene";
    }
    // Not all captures expose a PxMaterial; only assert when present.
    if (checked == 0)
        GTEST_SKIP() << "no PxMaterial in fixture";
}

#endif // PVD_OVD_SUPPORT
#endif // TST_OVD_MULTISCENE_H
