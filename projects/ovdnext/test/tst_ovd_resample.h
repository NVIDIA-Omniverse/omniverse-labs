/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TST_OVD_RESAMPLE_H
#define TST_OVD_RESAMPLE_H

#ifdef PVD_OVD_SUPPORT

// Regression tests for multi-segment OVD captures produced by repeated
// stopSampling()/startSampling() on ONE stream (the decoupled recording
// lifecycle). Each startSampling() re-registers the schema under new class
// handles, re-creates every live object under its existing external handle,
// and restarts frame numbering at 1 -- so the segments OVERLAY the shared
// timeline and BOTH copies of every re-created object are alive at every
// frame. Reference resolution (PxScene.actors, PxRigidActor.shapes, ...)
// must therefore be SEGMENT-CONSISTENT: a reference read from a segment-A
// object resolves to the segment-A object with that handle, never to another
// segment's copy (the collapsing last-wins external-handle map funneled every
// segment's references onto one arbitrary copy).
// Fixture: test/data/resample_two_segment.ovd (~100 KB), generated with
//   SnippetOmniPvd --omnipvdfile=... --resample
// (60 sampled frames, stopSampling, 30 unsampled frames incl. one actor
// created mid-gap, startSampling -> full snapshot, 60 more frames, teardown).
// NOTE: needs a pvddom with the multi-segment lifespan fixes (branch
// dev/pdeheras/pvddom-multi-segment-lifespans); on an older pvddom the
// re-created objects carry empty lifespans and these assertions misfire.

#include <cstring>
#include <map>
#include <set>
#include <vector>

#include <gtest/gtest.h>
#include <PvdEngine.h>
#include <PvdSimpleAllocator.h>
#include <PxErrorCallback.h>
#include <PxFoundation.h>
#include <PxFoundationVersion.h>
#include <PxPvdObjectModelBaseTypes.h>

namespace tst_resample {

using namespace physx;
using namespace physx::pvdsdk;
using namespace pvdapp;

class OvdResampleTest : public ::testing::Test
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
        sEngine = pvdapp::PvdEngine::load(*sAlloc, "./", TEST_DATA_DIR "/resample_two_segment.ovd");
    }

    static void TearDownTestSuite()
    {
        delete sEngine;
        sFoundation->release();
        delete sErrorCb;
        delete sAlloc;
    }

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

    // Read an object-ref list property (resolved to 1-based instance ids, the
    // same id space getOwningSceneId takes). 64 entries bounds every list in
    // this fixture (max 3 refs) with a wide margin.
    static std::vector<int32_t> readRefList(const InstanceDescription& inst, const char* propName)
    {
        int32_t buf[64] = { 0 };
        PxU32 n = sEngine->getPropertyValue(inst.mInstPtr, propName, buf, 64);
        return std::vector<int32_t>(buf, buf + n);
    }
};

PxAllocatorCallback* OvdResampleTest::sAlloc = nullptr;
OvdResampleTest::ErrorCb* OvdResampleTest::sErrorCb = nullptr;
PxFoundation* OvdResampleTest::sFoundation = nullptr;
pvdapp::PvdEngine* OvdResampleTest::sEngine = nullptr;

TEST_F(OvdResampleTest, FixtureLoads)
{
    ASSERT_NE(sEngine, nullptr);
    EXPECT_GT(sEngine->getNbInstances(), 0u);
}

// The two recording segments re-create the scene under one PhysX pointer:
// two PxScene instances must exist (the overlay design, one per segment).
TEST_F(OvdResampleTest, HasTwoSceneInstances)
{
    ASSERT_NE(sEngine, nullptr);
    int sceneCount = 0;
    for (const InstanceDescription& inst : allInstances())
        if (isClass(inst, "PxScene"))
            ++sceneCount;
    EXPECT_EQ(sceneCount, 2) << "expected one PxScene object per recording segment";
}

// THE CRUX -- segment-consistent reference resolution. Each PxScene's
// "actors" references must resolve to actor objects whose owning scene (the
// mAncestor-based getOwningSceneId, which is per-segment correct) is that
// same scene. With the last-wins external-handle map, both scenes' actor
// lists resolve to ONE arbitrary segment's actor copies, so one scene's list
// contains actors owned by the OTHER scene.
TEST_F(OvdResampleTest, SceneActorRefsResolveToOwnSegment)
{
    ASSERT_NE(sEngine, nullptr);
    int scenesChecked = 0, refsChecked = 0;
    for (const InstanceDescription& scene : allInstances())
    {
        if (!isClass(scene, "PxScene"))
            continue;
        ++scenesChecked;
        for (int32_t actorId : readRefList(scene, "actors"))
        {
            if (actorId <= 0)
                continue;
            ++refsChecked;
            EXPECT_EQ(sEngine->getOwningSceneId(actorId), (PxI32)scene.mId)
                << "scene id=" << scene.mId << " 'actors' ref resolved to actor id="
                << actorId << " owned by a DIFFERENT segment's scene";
        }
    }
    EXPECT_EQ(scenesChecked, 2);
    EXPECT_GT(refsChecked, 0) << "no actor references resolved at all";
}

// Same consistency one level down: each actor's "shapes" references resolve
// to shape objects of the SAME segment. A shape's segment is identified via
// its owning scene (walking mAncestor through the actor).
TEST_F(OvdResampleTest, ActorShapeRefsResolveToOwnSegment)
{
    ASSERT_NE(sEngine, nullptr);
    int refsChecked = 0;
    for (const InstanceDescription& actor : allInstances())
    {
        if (!isClass(actor, "PxRigidDynamic") && !isClass(actor, "PxRigidStatic"))
            continue;
        const PxI32 actorScene = sEngine->getOwningSceneId((PxI32)actor.mId);
        if (actorScene <= 0)
            continue; // actor not scene-parented (nothing to compare against)
        for (int32_t shapeId : readRefList(actor, "shapes"))
        {
            if (shapeId <= 0)
                continue;
            const PxI32 shapeScene = sEngine->getOwningSceneId(shapeId);
            if (shapeScene <= 0)
                continue; // shape not reachable through a scene ancestor
            ++refsChecked;
            EXPECT_EQ(shapeScene, actorScene)
                << "actor id=" << actor.mId << " 'shapes' ref resolved to shape id="
                << shapeId << " of a DIFFERENT segment";
        }
    }
    EXPECT_GT(refsChecked, 0) << "no shape references were scene-comparable";
}

TEST_F(OvdResampleTest, ExternalActorHandlesResolveWithinReferringScene)
{
    ASSERT_NE(sEngine, nullptr);
    int refsChecked = 0;
    for (const InstanceDescription& scene : allInstances())
    {
        if (!isClass(scene, "PxScene"))
            continue;
        for (PxI32 actorId : readRefList(scene, "actors"))
        {
            if (actorId <= 0)
                continue;
            ++refsChecked;
            const PxU64 externalHandle = sEngine->getStreamId(actorId);
            EXPECT_EQ(sEngine->resolveInstanceId(externalHandle, scene.mInstPtr), actorId)
                << "external actor handle must resolve within scene id=" << scene.mId;
        }
    }
    EXPECT_GT(refsChecked, 0);
}

} // namespace tst_resample

#endif // PVD_OVD_SUPPORT
#endif // TST_OVD_RESAMPLE_H
