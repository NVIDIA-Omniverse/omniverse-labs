/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TST_OVD_LOADER_H
#define TST_OVD_LOADER_H

#ifdef PVD_OVD_SUPPORT

#include <cstring>

#include <gtest/gtest.h>
#include <PvdEngine.h>
#include <PvdSimpleAllocator.h>
#include <PxErrorCallback.h>
#include <PxFoundation.h>
#include <PxFoundationVersion.h>
#include <PxPvdObjectModelBaseTypes.h>
#include <PvdCommStreamMetaData.h>

#include "ovd_snapshot.h"

using namespace physx;
using namespace physx::pvdsdk;
using namespace pvdapp;

class OvdLoaderTest : public ::testing::Test
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
        sEngine = pvdapp::PvdEngine::load(*sAlloc, "./", TEST_DATA_DIR "/test_scene.ovd");
    }

    static void TearDownTestSuite()
    {
        delete sEngine;
        sFoundation->release();
        delete sErrorCb;
        delete sAlloc;
    }
};

PxAllocatorCallback* OvdLoaderTest::sAlloc = nullptr;
OvdLoaderTest::ErrorCb* OvdLoaderTest::sErrorCb = nullptr;
PxFoundation* OvdLoaderTest::sFoundation = nullptr;
pvdapp::PvdEngine* OvdLoaderTest::sEngine = nullptr;

// ---------------------------------------------------------------------------
// Basic loading
// ---------------------------------------------------------------------------

TEST_F(OvdLoaderTest, LoadOvdFile)
{
    ASSERT_NE(sEngine, nullptr);
}

TEST_F(OvdLoaderTest, LoadNonExistentOvdReturnsEmptyEngine)
{
    // Non-existent OVD file returns an engine with 0 instances (DOM parse fails gracefully)
    pvdapp::PvdEngine* eng = pvdapp::PvdEngine::load(*sAlloc, "./", "nonexistent.ovd");
    if (eng)
    {
        EXPECT_EQ(eng->getNbInstances(), 0u);
        delete eng;
    }
}

// ---------------------------------------------------------------------------
// Scenes
// ---------------------------------------------------------------------------

TEST_F(OvdLoaderTest, SceneCount)
{
    ASSERT_NE(sEngine, nullptr);
    EXPECT_EQ(sEngine->getNbScenes(), 1u);
}

TEST_F(OvdLoaderTest, SceneHasStreamId)
{
    ASSERT_NE(sEngine, nullptr);
    SectionListInfo scenes[4];
    PxU32 n = sEngine->getScenes(scenes, 4);
    ASSERT_EQ(n, 1u);
    EXPECT_GT(scenes[0].mStreamId, 0ull);
}

// ---------------------------------------------------------------------------
// Frames
// ---------------------------------------------------------------------------

TEST_F(OvdLoaderTest, FrameCountPositive)
{
    ASSERT_NE(sEngine, nullptr);
    SectionListInfo scenes[1];
    sEngine->getScenes(scenes, 1);
    PxU32 fc = sEngine->getFrameCount(scenes[0].mStreamId);
    EXPECT_GE(fc, 1u);
}

// ---------------------------------------------------------------------------
// Classes
// ---------------------------------------------------------------------------

TEST_F(OvdLoaderTest, ClassCountNonZero)
{
    ASSERT_NE(sEngine, nullptr);
    EXPECT_GT(sEngine->getNbClasses(), 0u);
}

TEST_F(OvdLoaderTest, ClassCountAtLeast64)
{
    ASSERT_NE(sEngine, nullptr);
    // test_scene.ovd has 96 DOM classes, 64 non-enum
    EXPECT_GE(sEngine->getNbClasses(), 64u);
}

TEST_F(OvdLoaderTest, FindPxSceneClass)
{
    ASSERT_NE(sEngine, nullptr);
    PxU32 nCls = sEngine->getNbClasses();
    std::vector<ClassDescription> classes(nCls);
    sEngine->getClasses(classes.data(), nCls);
    bool found = false;
    for (auto& c : classes)
    {
        if (c.mName == NamespacedName("physx3", "PxScene"))
        {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(OvdLoaderTest, FindPxRigidDynamicClass)
{
    ASSERT_NE(sEngine, nullptr);
    PxU32 nCls = sEngine->getNbClasses();
    std::vector<ClassDescription> classes(nCls);
    sEngine->getClasses(classes.data(), nCls);
    bool found = false;
    for (auto& c : classes)
    {
        if (c.mName == NamespacedName("physx3", "PxRigidDynamic"))
        {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(OvdLoaderTest, FindPxMaterialClass)
{
    ASSERT_NE(sEngine, nullptr);
    PxU32 nCls = sEngine->getNbClasses();
    std::vector<ClassDescription> classes(nCls);
    sEngine->getClasses(classes.data(), nCls);
    bool found = false;
    for (auto& c : classes)
    {
        if (c.mName == NamespacedName("physx3", "PxMaterial"))
        {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(OvdLoaderTest, FindPxShapeClass)
{
    ASSERT_NE(sEngine, nullptr);
    PxU32 nCls = sEngine->getNbClasses();
    std::vector<ClassDescription> classes(nCls);
    sEngine->getClasses(classes.data(), nCls);
    bool found = false;
    for (auto& c : classes)
    {
        if (c.mName == NamespacedName("physx3", "PxShape"))
        {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(OvdLoaderTest, FindPxSphereGeometryClass)
{
    ASSERT_NE(sEngine, nullptr);
    PxU32 nCls = sEngine->getNbClasses();
    std::vector<ClassDescription> classes(nCls);
    sEngine->getClasses(classes.data(), nCls);
    bool found = false;
    for (auto& c : classes)
    {
        if (c.mName == NamespacedName("physx3", "PxSphereGeometry"))
        {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

TEST_F(OvdLoaderTest, PxSceneHasGravityProperty)
{
    ASSERT_NE(sEngine, nullptr);
    Option<ClassDescription> cls = sEngine->getClass(0); // search all
    PxU32 nCls = sEngine->getNbClasses();
    std::vector<ClassDescription> classes(nCls);
    sEngine->getClasses(classes.data(), nCls);
    int32_t sceneClsId = -1;
    for (auto& c : classes)
    {
        if (c.mName == NamespacedName("physx3", "PxScene"))
        {
            sceneClsId = c.mClassId;
            break;
        }
    }
    ASSERT_GE(sceneClsId, 0);
    Option<PropertyDescription> prop = sEngine->findProperty(sceneClsId, "gravity");
    EXPECT_TRUE(prop.hasValue());
}

TEST_F(OvdLoaderTest, PxRigidDynamicHasMassProperty)
{
    ASSERT_NE(sEngine, nullptr);
    PxU32 nCls = sEngine->getNbClasses();
    std::vector<ClassDescription> classes(nCls);
    sEngine->getClasses(classes.data(), nCls);
    int32_t clsId = -1;
    for (auto& c : classes)
    {
        if (c.mName == NamespacedName("physx3", "PxRigidDynamic"))
        {
            clsId = c.mClassId;
            break;
        }
    }
    ASSERT_GE(clsId, 0);
    Option<PropertyDescription> prop = sEngine->findProperty(clsId, "mass");
    EXPECT_TRUE(prop.hasValue());
}

TEST_F(OvdLoaderTest, PxRigidDynamicHasGlobalPoseProperty)
{
    ASSERT_NE(sEngine, nullptr);
    PxU32 nCls = sEngine->getNbClasses();
    std::vector<ClassDescription> classes(nCls);
    sEngine->getClasses(classes.data(), nCls);
    int32_t clsId = -1;
    for (auto& c : classes)
    {
        if (c.mName == NamespacedName("physx3", "PxRigidDynamic"))
        {
            clsId = c.mClassId;
            break;
        }
    }
    ASSERT_GE(clsId, 0);
    Option<PropertyDescription> prop = sEngine->findProperty(clsId, "globalPose");
    EXPECT_TRUE(prop.hasValue());
}

TEST_F(OvdLoaderTest, PxRigidDynamicHasLinearVelocityProperty)
{
    ASSERT_NE(sEngine, nullptr);
    PxU32 nCls = sEngine->getNbClasses();
    std::vector<ClassDescription> classes(nCls);
    sEngine->getClasses(classes.data(), nCls);
    int32_t clsId = -1;
    for (auto& c : classes)
    {
        if (c.mName == NamespacedName("physx3", "PxRigidDynamic"))
        {
            clsId = c.mClassId;
            break;
        }
    }
    ASSERT_GE(clsId, 0);
    Option<PropertyDescription> prop = sEngine->findProperty(clsId, "linearVelocity");
    EXPECT_TRUE(prop.hasValue());
}

TEST_F(OvdLoaderTest, PxMaterialHasStaticFrictionProperty)
{
    ASSERT_NE(sEngine, nullptr);
    PxU32 nCls = sEngine->getNbClasses();
    std::vector<ClassDescription> classes(nCls);
    sEngine->getClasses(classes.data(), nCls);
    int32_t clsId = -1;
    for (auto& c : classes)
    {
        if (c.mName == NamespacedName("physx3", "PxMaterial"))
        {
            clsId = c.mClassId;
            break;
        }
    }
    ASSERT_GE(clsId, 0);
    Option<PropertyDescription> prop = sEngine->findProperty(clsId, "staticFriction");
    EXPECT_TRUE(prop.hasValue());
}

TEST_F(OvdLoaderTest, PxShapeHasLocalPoseProperty)
{
    ASSERT_NE(sEngine, nullptr);
    PxU32 nCls = sEngine->getNbClasses();
    std::vector<ClassDescription> classes(nCls);
    sEngine->getClasses(classes.data(), nCls);
    int32_t clsId = -1;
    for (auto& c : classes)
    {
        if (c.mName == NamespacedName("physx3", "PxShape"))
        {
            clsId = c.mClassId;
            break;
        }
    }
    ASSERT_GE(clsId, 0);
    Option<PropertyDescription> prop = sEngine->findProperty(clsId, "localPose");
    EXPECT_TRUE(prop.hasValue());
}

TEST_F(OvdLoaderTest, PropertyCountOnPxScene)
{
    ASSERT_NE(sEngine, nullptr);
    PxU32 nCls = sEngine->getNbClasses();
    std::vector<ClassDescription> classes(nCls);
    sEngine->getClasses(classes.data(), nCls);
    int32_t clsId = -1;
    for (auto& c : classes)
    {
        if (c.mName == NamespacedName("physx3", "PxScene"))
        {
            clsId = c.mClassId;
            break;
        }
    }
    ASSERT_GE(clsId, 0);
    EXPECT_GT(sEngine->getNbProperties(clsId), 30u); // PxScene has many properties
}

// ---------------------------------------------------------------------------
// Inheritance-chain coverage — the property panel populates itself by calling
// getNbProperties() / getProperties() on the instance's class. Both must
// return the union of own + ancestor properties, otherwise users see only
// the leaf-class attributes (e.g. a PxRigidStatic would appear empty since
// it has zero own attributes — every entry comes from PxRigidActor / PxActor).
// ---------------------------------------------------------------------------

namespace ovd_chain_test_helpers
{
    inline int32_t findClassId(pvdapp::PvdEngine* eng, const char* name)
    {
        PxU32 nCls = eng->getNbClasses();
        std::vector<ClassDescription> classes(nCls);
        eng->getClasses(classes.data(), nCls);
        for (auto& c : classes)
            if (c.mName == NamespacedName("physx3", name))
                return c.mClassId;
        return -1;
    }

    inline bool propertyListContains(pvdapp::PvdEngine* eng, int32_t clsId, const char* name)
    {
        PxU32 n = eng->getNbProperties(clsId);
        std::vector<PropertyDescription> props(n);
        eng->getProperties(clsId, props.data(), n, 0);
        for (auto& p : props)
            if (p.mName && std::strcmp(p.mName, name) == 0)
                return true;
        return false;
    }
}

TEST_F(OvdLoaderTest, PxRigidDynamicPropertyListCoversFullChain)
{
    // Inheritance chain: PxRigidDynamic → PxRigidBody → PxRigidActor → PxActor.
    // getProperties() must return at least one attribute defined at each level.
    using namespace ovd_chain_test_helpers;
    ASSERT_NE(sEngine, nullptr);
    int32_t clsId = findClassId(sEngine, "PxRigidDynamic");
    ASSERT_GE(clsId, 0);

    // Defined on PxRigidDynamic itself
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "isSleeping"));
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "wakeCounter"));
    // Inherited from PxRigidBody
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "mass"));
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "cMassLocalPose"));
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "linearVelocity"));
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "rigidBodyFlags"));
    // Inherited from PxRigidActor
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "globalPose"));
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "shapes"));
    // Inherited from PxActor (root)
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "type"));
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "name"));
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "worldBounds"));
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "dominance"));
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "ownerClient"));
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "environmentID"));
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "flags"));
}

TEST_F(OvdLoaderTest, PxRigidStaticHasOnlyInheritedProperties)
{
    // PxRigidStatic defines NO own attributes (see OmniPvdTypes.h:465-466).
    // If getProperties() returns 0 the inheritance walk is broken — the
    // property panel would render the actor as empty.
    using namespace ovd_chain_test_helpers;
    ASSERT_NE(sEngine, nullptr);
    int32_t clsId = findClassId(sEngine, "PxRigidStatic");
    ASSERT_GE(clsId, 0);

    PxU32 nbProps = sEngine->getNbProperties(clsId);
    EXPECT_GT(nbProps, 0u) << "PxRigidStatic returned no properties — inheritance walk broken";

    // From PxRigidActor
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "globalPose"));
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "shapes"));
    // From PxActor
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "type"));
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "name"));
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "worldBounds"));
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "dominance"));
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "ownerClient"));
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "environmentID"));
    EXPECT_TRUE(propertyListContains(sEngine, clsId, "flags"));
}

TEST_F(OvdLoaderTest, PropertyOwnerClassNamesTraceToAncestors)
{
    // Each PropertyDescription records the class that originally declared it
    // (PropertyDescription::mOwnerClassName). For PxRigidDynamic the union of
    // owners returned by getProperties() must span all four levels — confirms
    // the adapter is not collapsing every entry under the leaf class.
    using namespace ovd_chain_test_helpers;
    ASSERT_NE(sEngine, nullptr);
    int32_t clsId = findClassId(sEngine, "PxRigidDynamic");
    ASSERT_GE(clsId, 0);

    PxU32 n = sEngine->getNbProperties(clsId);
    std::vector<PropertyDescription> props(n);
    sEngine->getProperties(clsId, props.data(), n, 0);

    bool sawDynamic = false, sawBody = false, sawActor = false, sawBase = false;
    for (auto& p : props)
    {
        const char* owner = p.mOwnerClassName.mName;
        if (!owner) continue;
        if (std::strcmp(owner, "PxRigidDynamic") == 0) sawDynamic = true;
        else if (std::strcmp(owner, "PxRigidBody") == 0) sawBody = true;
        else if (std::strcmp(owner, "PxRigidActor") == 0) sawActor = true;
        else if (std::strcmp(owner, "PxActor") == 0) sawBase = true;
    }
    EXPECT_TRUE(sawDynamic) << "no property attributed to PxRigidDynamic";
    EXPECT_TRUE(sawBody) << "no property attributed to PxRigidBody";
    EXPECT_TRUE(sawActor) << "no property attributed to PxRigidActor";
    EXPECT_TRUE(sawBase) << "no property attributed to PxActor";
}

TEST_F(OvdLoaderTest, EnumAttributesGetEnumerationSemantic)
{
    // PhysX's OMNI_PVD_ATTRIBUTE_FLAG macro emits both true bitflags
    // (PxActorFlags: 1, 2, 4, 8, …) and sequential enums (PxActorType:
    // 0, 1, 2, 3, …) as eFLAGS_WORD at the schema level. The adapter must
    // disambiguate via the named-value table — otherwise enums render as
    // empty bitflag rows in the property panel. PxActor.type must be tagged
    // "Enumeration Value" so eRIGID_STATIC (=0) shows its name in the Value
    // column instead of nothing.
    using namespace ovd_chain_test_helpers;
    ASSERT_NE(sEngine, nullptr);
    int32_t clsId = findClassId(sEngine, "PxRigidDynamic");
    ASSERT_GE(clsId, 0);

    Option<PropertyDescription> typeProp = sEngine->findProperty(clsId, "type");
    ASSERT_TRUE(typeProp.hasValue());
    EXPECT_STREQ(typeProp.getValue().mSemantic, "Enumeration Value")
        << "PxActor.type must be enum (PxActorType is sequential 0..N), not bitflag";

    Option<PropertyDescription> flagsProp = sEngine->findProperty(clsId, "flags");
    ASSERT_TRUE(flagsProp.hasValue());
    EXPECT_STREQ(flagsProp.getValue().mSemantic, "Bitflag")
        << "PxActor.flags must stay bitflag (PxActorFlags is power-of-two)";
}

TEST_F(OvdLoaderTest, PxRigidDynamicPropertyOwnersAreOnlyAncestors)
{
    // Stronger guarantee: every property returned by getProperties(PxRigidDynamic)
    // must have an owner class name in the schema chain {PxRigidDynamic,
    // PxRigidBody, PxRigidActor, PxActor}. Anything else means the adapter
    // is leaking properties from sibling/unrelated classes — the property
    // panel would show stray attributes the SDK never declared on this type.
    // The chain itself is read from what the OVD file recorded (pvddom's
    // OmniPvdClass::mInheritanceChain), which the writer populated from
    // OmniPvdTypes.h:499+ at capture time. So a passing test here means
    // the on-disk schema → DOM → engine → property-panel pipeline preserves
    // the inheritance edges declared in OmniPvdTypes.h.
    using namespace ovd_chain_test_helpers;
    ASSERT_NE(sEngine, nullptr);
    int32_t clsId = findClassId(sEngine, "PxRigidDynamic");
    ASSERT_GE(clsId, 0);

    PxU32 n = sEngine->getNbProperties(clsId);
    std::vector<PropertyDescription> props(n);
    sEngine->getProperties(clsId, props.data(), n, 0);

    for (auto& p : props)
    {
        const char* owner = p.mOwnerClassName.mName ? p.mOwnerClassName.mName : "<null>";
        const bool ok =
            std::strcmp(owner, "PxRigidDynamic") == 0 ||
            std::strcmp(owner, "PxRigidBody")    == 0 ||
            std::strcmp(owner, "PxRigidActor")   == 0 ||
            std::strcmp(owner, "PxActor")        == 0;
        EXPECT_TRUE(ok) << "property '" << (p.mName ? p.mName : "<null>")
                        << "' has unexpected owner class '" << owner
                        << "' — not in PxRigidDynamic's inheritance chain";
    }
}

// ---------------------------------------------------------------------------
// Instances
// ---------------------------------------------------------------------------

TEST_F(OvdLoaderTest, InstanceCount)
{
    ASSERT_NE(sEngine, nullptr);
    PxU32 n = sEngine->getNbInstances();
    EXPECT_EQ(n, 8u);
}

TEST_F(OvdLoaderTest, InstancesAreAlive)
{
    ASSERT_NE(sEngine, nullptr);
    // Every instance must have a valid (non-null) handle regardless of frame.
    PxU32 n = sEngine->getNbInstances();
    std::vector<InstanceDescription> insts(n);
    sEngine->getInstances(insts.data(), n);
    for (PxU32 i = 0; i < n; ++i)
        EXPECT_NE(insts[i].mInstPtr, nullptr) << "Instance " << i << " has null ptr";

    // Aliveness is frame-dependent under the half-open [mFrameStart, mFrameStop)
    // lifespan rule: at the default pre-section frame (the frame before
    // mMinFrame) objects born at mMinFrame aren't alive yet, so "all alive"
    // only holds once the scene is populated. Seek to a populated frame, assert
    // aliveness tracking actually reports objects alive, then restore the
    // shared engine's frame for the other tests in this suite.
    SectionListInfo scenes[4];
    PxU32 ns = sEngine->getScenes(scenes, 4);
    ASSERT_GT(ns, 0u);
    const PxU64 sid    = scenes[0].mStreamId;
    const PxU32 saved  = sEngine->getCurrentFrame(sid);
    const PxU32 fc     = sEngine->getFrameCount(sid);
    const PxU32 probe  = fc > 1 ? fc - 1 : 0;
    sEngine->setCurrentFrame(probe, sid);

    sEngine->getInstances(insts.data(), n);
    int aliveCount = 0;
    for (PxU32 i = 0; i < n; ++i)
        if (insts[i].mAlive) ++aliveCount;
    EXPECT_GT(aliveCount, 0) << "no instances alive at frame " << probe;

    sEngine->setCurrentFrame(saved, sid);
}

TEST_F(OvdLoaderTest, InstanceClassesValid)
{
    ASSERT_NE(sEngine, nullptr);
    PxU32 n = sEngine->getNbInstances();
    std::vector<InstanceDescription> insts(n);
    sEngine->getInstances(insts.data(), n);
    for (PxU32 i = 0; i < n; ++i)
    {
        Option<ClassDescription> cls = sEngine->getClass(insts[i].mClassId);
        EXPECT_TRUE(cls.hasValue()) << "Instance " << i << " has invalid classId " << insts[i].mClassId;
    }
}

TEST_F(OvdLoaderTest, HasPxSceneInstance)
{
    ASSERT_NE(sEngine, nullptr);
    PxU32 n = sEngine->getNbInstances();
    std::vector<InstanceDescription> insts(n);
    sEngine->getInstances(insts.data(), n);
    bool found = false;
    for (PxU32 i = 0; i < n; ++i)
    {
        Option<ClassDescription> cls = sEngine->getClass(insts[i].mClassId);
        if (cls.hasValue() && cls.getValue().mName == NamespacedName("physx3", "PxScene"))
        {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(OvdLoaderTest, HasPxRigidDynamicInstance)
{
    ASSERT_NE(sEngine, nullptr);
    PxU32 n = sEngine->getNbInstances();
    std::vector<InstanceDescription> insts(n);
    sEngine->getInstances(insts.data(), n);
    bool found = false;
    for (PxU32 i = 0; i < n; ++i)
    {
        Option<ClassDescription> cls = sEngine->getClass(insts[i].mClassId);
        if (cls.hasValue() && cls.getValue().mName == NamespacedName("physx3", "PxRigidDynamic"))
        {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(OvdLoaderTest, HasPxMaterialInstance)
{
    ASSERT_NE(sEngine, nullptr);
    PxU32 n = sEngine->getNbInstances();
    std::vector<InstanceDescription> insts(n);
    sEngine->getInstances(insts.data(), n);
    bool found = false;
    for (PxU32 i = 0; i < n; ++i)
    {
        Option<ClassDescription> cls = sEngine->getClass(insts[i].mClassId);
        if (cls.hasValue() && cls.getValue().mName == NamespacedName("physx3", "PxMaterial"))
        {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(OvdLoaderTest, HasPxShapeInstance)
{
    ASSERT_NE(sEngine, nullptr);
    PxU32 n = sEngine->getNbInstances();
    std::vector<InstanceDescription> insts(n);
    sEngine->getInstances(insts.data(), n);
    bool found = false;
    for (PxU32 i = 0; i < n; ++i)
    {
        Option<ClassDescription> cls = sEngine->getClass(insts[i].mClassId);
        if (cls.hasValue() && cls.getValue().mName == NamespacedName("physx3", "PxShape"))
        {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// Property values on instances
// ---------------------------------------------------------------------------

TEST_F(OvdLoaderTest, PxSceneGravityValue)
{
    ASSERT_NE(sEngine, nullptr);
    PxU32 n = sEngine->getNbInstances();
    std::vector<InstanceDescription> insts(n);
    sEngine->getInstances(insts.data(), n);
    for (PxU32 i = 0; i < n; ++i)
    {
        Option<ClassDescription> cls = sEngine->getClass(insts[i].mClassId);
        if (cls.hasValue() && cls.getValue().mName == NamespacedName("physx3", "PxScene"))
        {
            Option<PxVec3> gravity = sEngine->getPropertyValue<PxVec3>(insts[i].mInstPtr, "gravity");
            ASSERT_TRUE(gravity.hasValue());
            // Gravity should point downward (negative y typically)
            EXPECT_NE(gravity.getValue().magnitudeSquared(), 0.0f);
            return;
        }
    }
    FAIL() << "No PxScene instance found";
}

TEST_F(OvdLoaderTest, PxRigidDynamicMassPositive)
{
    ASSERT_NE(sEngine, nullptr);
    PxU32 n = sEngine->getNbInstances();
    std::vector<InstanceDescription> insts(n);
    sEngine->getInstances(insts.data(), n);
    for (PxU32 i = 0; i < n; ++i)
    {
        Option<ClassDescription> cls = sEngine->getClass(insts[i].mClassId);
        if (cls.hasValue() && cls.getValue().mName == NamespacedName("physx3", "PxRigidDynamic"))
        {
            Option<float> mass = sEngine->getPropertyValue<float>(insts[i].mInstPtr, "mass");
            ASSERT_TRUE(mass.hasValue());
            EXPECT_GT(mass.getValue(), 0.0f);
            return;
        }
    }
    FAIL() << "No PxRigidDynamic instance found";
}

TEST_F(OvdLoaderTest, PxSphereGeometryRadiusPositive)
{
    ASSERT_NE(sEngine, nullptr);
    PxU32 n = sEngine->getNbInstances();
    std::vector<InstanceDescription> insts(n);
    sEngine->getInstances(insts.data(), n);
    for (PxU32 i = 0; i < n; ++i)
    {
        Option<ClassDescription> cls = sEngine->getClass(insts[i].mClassId);
        if (cls.hasValue() && cls.getValue().mName == NamespacedName("physx3", "PxSphereGeometry"))
        {
            Option<float> radius = sEngine->getPropertyValue<float>(insts[i].mInstPtr, "radius");
            ASSERT_TRUE(radius.hasValue());
            EXPECT_GT(radius.getValue(), 0.0f);
            return;
        }
    }
    FAIL() << "No PxSphereGeometry instance found";
}

TEST_F(OvdLoaderTest, PxMaterialRestitutionInRange)
{
    ASSERT_NE(sEngine, nullptr);
    PxU32 n = sEngine->getNbInstances();
    std::vector<InstanceDescription> insts(n);
    sEngine->getInstances(insts.data(), n);
    for (PxU32 i = 0; i < n; ++i)
    {
        Option<ClassDescription> cls = sEngine->getClass(insts[i].mClassId);
        if (cls.hasValue() && cls.getValue().mName == NamespacedName("physx3", "PxMaterial"))
        {
            Option<float> restitution = sEngine->getPropertyValue<float>(insts[i].mInstPtr, "restitution");
            ASSERT_TRUE(restitution.hasValue());
            EXPECT_GE(restitution.getValue(), 0.0f);
            EXPECT_LE(restitution.getValue(), 1.0f);
            return;
        }
    }
    FAIL() << "No PxMaterial instance found";
}

// ---------------------------------------------------------------------------
// TopLevel and stream ID
// ---------------------------------------------------------------------------

TEST_F(OvdLoaderTest, SomeInstancesAreTopLevel)
{
    ASSERT_NE(sEngine, nullptr);
    PxU32 n = sEngine->getNbInstances();
    std::vector<InstanceDescription> insts(n);
    sEngine->getInstances(insts.data(), n);
    int topCount = 0;
    for (PxU32 i = 0; i < n; ++i)
    {
        if (sEngine->isTopLevel(insts[i].mId))
            topCount++;
    }
    EXPECT_GT(topCount, 0);
}

TEST_F(OvdLoaderTest, SceneRemainsNestedWhenPhysicsRootExists)
{
    ASSERT_NE(sEngine, nullptr);
    PxU32 n = sEngine->getNbInstances();
    std::vector<InstanceDescription> insts(n);
    sEngine->getInstances(insts.data(), n);

    int physicsRoots = 0;
    int scenes = 0;
    for (const InstanceDescription& inst : insts)
    {
        Option<ClassDescription> cls = sEngine->getClass(inst.mClassId);
        ASSERT_TRUE(cls.hasValue());
        if (::strcmp(cls.getValue().mName.mName, "PxPhysics") == 0)
        {
            ++physicsRoots;
            EXPECT_TRUE(sEngine->isTopLevel(inst.mId));
        }
        else if (::strcmp(cls.getValue().mName.mName, "PxScene") == 0)
        {
            ++scenes;
            EXPECT_FALSE(sEngine->isTopLevel(inst.mId));
        }
    }
    EXPECT_GT(physicsRoots, 0);
    EXPECT_GT(scenes, 0);
}

// ---------------------------------------------------------------------------
// Event count
// ---------------------------------------------------------------------------

TEST_F(OvdLoaderTest, EventCountZeroForDomAdapter)
{
    ASSERT_NE(sEngine, nullptr);
    // OVD DOM adapter has no event stream — event count is 0
    EXPECT_EQ(sEngine->getEventCount(), 0ull);
}

// ---------------------------------------------------------------------------
// PXD2 backward compatibility (existing tests still pass)
// ---------------------------------------------------------------------------

TEST_F(OvdLoaderTest, LoadPxd2StillWorks)
{
    pvdapp::PvdEngine* eng = pvdapp::PvdEngine::load(*sAlloc, "./", TEST_DATA_DIR "/apex_destruction.pxd2");
    ASSERT_NE(eng, nullptr);
    EXPECT_GT(eng->getNbClasses(), 0u);
    delete eng;
}

// ---------------------------------------------------------------------------
// Smoke tests over a set of real-world OVD captures.
//
// These captures are dropped into test/data/ as recorded by PhysX apps; we
// don't know their exact contents, but we do know loading them must succeed
// and every one of the DOM-adapter read paths the UI relies on (instance/
// class/property enumeration, frame scrubbing) must not crash or leak.
// ---------------------------------------------------------------------------

class OvdCaptureSmokeTest : public ::testing::TestWithParam<const char*>
{
protected:
    class ErrorCb : public PxErrorCallback
    {
        void reportError(PxErrorCode::Enum, const char*, const char*, int) override {}
    };

    static PxAllocatorCallback* sAlloc;
    static ErrorCb* sErrorCb;
    static PxFoundation* sFoundation;

    static void SetUpTestSuite()
    {
        sAlloc = new pvdapp::CheckingAllocator();
        sErrorCb = new ErrorCb();
        sFoundation = PxCreateFoundation(PX_FOUNDATION_VERSION, *sAlloc, *sErrorCb);
    }

    static void TearDownTestSuite()
    {
        sFoundation->release();
        delete sErrorCb;
        delete sAlloc;
    }
};

PxAllocatorCallback* OvdCaptureSmokeTest::sAlloc = nullptr;
OvdCaptureSmokeTest::ErrorCb* OvdCaptureSmokeTest::sErrorCb = nullptr;
PxFoundation* OvdCaptureSmokeTest::sFoundation = nullptr;

TEST_P(OvdCaptureSmokeTest, LoadsAndExposesNonZeroCounts)
{
    const std::string path = std::string(TEST_DATA_DIR "/") + GetParam();
    pvdapp::PvdEngine* eng = pvdapp::PvdEngine::load(*sAlloc, "./", path.c_str());
    ASSERT_NE(eng, nullptr) << "Failed to load " << path;

    EXPECT_GT(eng->getNbClasses(), 0u);
    EXPECT_GT(eng->getNbInstances(), 0u);
    EXPECT_GT(eng->getNbScenes(), 0u);

    delete eng;
}

TEST_P(OvdCaptureSmokeTest, FrameScrubDoesNotCrash)
{
    const std::string path = std::string(TEST_DATA_DIR "/") + GetParam();
    pvdapp::PvdEngine* eng = pvdapp::PvdEngine::load(*sAlloc, "./", path.c_str());
    ASSERT_NE(eng, nullptr);

    const PxU32 nScenes = eng->getNbScenes();
    ASSERT_GT(nScenes, 0u);
    std::vector<SectionListInfo> scenes(nScenes);
    eng->getScenes(scenes.data(), nScenes);

    for (const SectionListInfo& s : scenes)
    {
        const PxU32 fc = eng->getFrameCount(s.mStreamId);
        ASSERT_GT(fc, 0u);

        // Sample frame 0, midpoint, and last frame.  The DOM adapter is O(1)
        // per setCurrentFrame, so three calls is enough to cover seek logic
        // without making the test slow.
        const PxU32 mid = fc / 2;
        eng->setCurrentFrame(0, s.mStreamId);
        eng->setCurrentFrame(mid, s.mStreamId);
        eng->setCurrentFrame(fc - 1, s.mStreamId);
    }

    delete eng;
}

// Opt-in: dumps frame 1 snapshot (first ~25 lines) for each real-world capture
// so we can eyeball shape localPose + mesh-geometry scale output against actual
// data.  Disabled; run with --gtest_also_run_disabled_tests.
TEST_P(OvdCaptureSmokeTest, DISABLED_DumpSnapshotToStdout)
{
    const std::string path = std::string(TEST_DATA_DIR "/") + GetParam();
    pvdapp::PvdEngine* eng = pvdapp::PvdEngine::load(*sAlloc, "./", path.c_str());
    ASSERT_NE(eng, nullptr);

    SectionListInfo scn;
    ASSERT_GT(eng->getScenes(&scn, 1), 0u);
    eng->setCurrentFrame(1, scn.mStreamId);

    const std::string text = ovdtest::formatFrameSnapshot(ovdtest::captureFrameSnapshot(*eng, 1));
    std::cout << "====== " << GetParam() << " frame 1 (first 40 lines) ======\n";
    std::istringstream iss(text);
    std::string line;
    for (int i = 0; i < 40 && std::getline(iss, line); ++i)
        std::cout << line << "\n";

    delete eng;
}

TEST_P(OvdCaptureSmokeTest, InstanceEnumerationIsConsistent)
{
    const std::string path = std::string(TEST_DATA_DIR "/") + GetParam();
    pvdapp::PvdEngine* eng = pvdapp::PvdEngine::load(*sAlloc, "./", path.c_str());
    ASSERT_NE(eng, nullptr);

    const PxU32 n = eng->getNbInstances();
    ASSERT_GT(n, 0u);
    std::vector<InstanceDescription> insts(n);
    eng->getInstances(insts.data(), n);

    for (PxU32 i = 0; i < n; ++i)
    {
        EXPECT_NE(insts[i].mInstPtr, nullptr) << "instance " << i << " has null ptr";
        Option<ClassDescription> cls = eng->getClass(insts[i].mClassId);
        EXPECT_TRUE(cls.hasValue()) << "instance " << i << " classId " << insts[i].mClassId << " not found";
    }

    delete eng;
}

INSTANTIATE_TEST_SUITE_P(
    RealWorldCaptures,
    OvdCaptureSmokeTest,
    ::testing::Values(
        "2026_04_01_15_21_33_0000019f4952a142.ovd",
        "2026_04_01_16_42_23_00000062b3443b1b.ovd",
        "snippet_meshes.ovd",
        "multi_geom.ovd"
    )
);

// ---------------------------------------------------------------------------
// Structural render snapshot.  The snapshot is GPU-free: it captures the
// records the renderer consumes (actor class/id/pose/shape list, mesh
// resources) in a deterministic, diff-friendly form.  See ovd_snapshot.h
// for the format.
// ---------------------------------------------------------------------------

// Opt-in: dumps the structural snapshot for test_scene.ovd at frame 1
// and the last frame to stdout.  Disabled by default; run with
// --gtest_also_run_disabled_tests to see the output.
TEST_F(OvdLoaderTest, DISABLED_DumpSnapshotToStdout)
{
    ASSERT_NE(sEngine, nullptr);
    SectionListInfo scn;
    ASSERT_GT(sEngine->getScenes(&scn, 1), 0u);
    const PxU64 sceneId = scn.mStreamId;
    const PxU32 fc = sEngine->getFrameCount(sceneId);

    sEngine->setCurrentFrame(1, sceneId);
    std::cout << "====== Frame 1 snapshot ======\n"
              << ovdtest::formatFrameSnapshot(ovdtest::captureFrameSnapshot(*sEngine, 1));

    if (fc > 1)
    {
        sEngine->setCurrentFrame(fc - 1, sceneId);
        std::cout << "====== Frame " << (fc - 1) << " snapshot ======\n"
                  << ovdtest::formatFrameSnapshot(ovdtest::captureFrameSnapshot(*sEngine, fc - 1));
    }
}

TEST_F(OvdLoaderTest, SnapshotFrame1IsNonEmptyAndPosesFinite)
{
    ASSERT_NE(sEngine, nullptr);
    SectionListInfo scn;
    ASSERT_GT(sEngine->getScenes(&scn, 1), 0u);
    const PxU64 sceneId = scn.mStreamId;
    sEngine->setCurrentFrame(1, sceneId);

    ovdtest::FrameSnapshot snap = ovdtest::captureFrameSnapshot(*sEngine, 1);
    ASSERT_FALSE(snap.mActors.empty()) << "No actors at frame 1";
    for (const ovdtest::ActorRecord& a : snap.mActors)
    {
        EXPECT_TRUE(PxIsFinite(a.mGlobalPose.p.x));
        EXPECT_TRUE(PxIsFinite(a.mGlobalPose.p.y));
        EXPECT_TRUE(PxIsFinite(a.mGlobalPose.p.z));
    }
}

TEST_F(OvdLoaderTest, SnapshotIsDeterministic)
{
    ASSERT_NE(sEngine, nullptr);
    SectionListInfo scn;
    ASSERT_GT(sEngine->getScenes(&scn, 1), 0u);
    const PxU64 sceneId = scn.mStreamId;
    sEngine->setCurrentFrame(1, sceneId);
    const std::string a = ovdtest::formatFrameSnapshot(ovdtest::captureFrameSnapshot(*sEngine, 1));
    const std::string b = ovdtest::formatFrameSnapshot(ovdtest::captureFrameSnapshot(*sEngine, 1));
    EXPECT_EQ(a, b);
}

TEST_F(OvdLoaderTest, SnapshotDiffersAcrossFramesWhenSimulating)
{
    ASSERT_NE(sEngine, nullptr);
    SectionListInfo scn;
    ASSERT_GT(sEngine->getScenes(&scn, 1), 0u);
    const PxU64 sceneId = scn.mStreamId;
    const PxU32 fc = sEngine->getFrameCount(sceneId);
    if (fc < 4)
    {
        GTEST_SKIP() << "test_scene.ovd has too few frames to see motion";
    }
    sEngine->setCurrentFrame(1, sceneId);
    const std::string first = ovdtest::formatFrameSnapshot(ovdtest::captureFrameSnapshot(*sEngine, 1));
    sEngine->setCurrentFrame(fc - 1, sceneId);
    const std::string last  = ovdtest::formatFrameSnapshot(ovdtest::captureFrameSnapshot(*sEngine, fc - 1));
    EXPECT_NE(first, last) << "Simulation produced no visible change between frame 1 and " << (fc - 1);
}

TEST_P(OvdCaptureSmokeTest, SnapshotMeshRecordsHaveConsistentVertsAndTris)
{
    const std::string path = std::string(TEST_DATA_DIR "/") + GetParam();
    pvdapp::PvdEngine* eng = pvdapp::PvdEngine::load(*sAlloc, "./", path.c_str());
    ASSERT_NE(eng, nullptr);

    const PxU32 nScenes = eng->getNbScenes();
    ASSERT_GT(nScenes, 0u);
    std::vector<SectionListInfo> scenes(nScenes);
    eng->getScenes(scenes.data(), nScenes);
    const PxU64 sceneId = scenes[0].mStreamId;

    const PxU32 fc = eng->getFrameCount(sceneId);
    const PxU32 f = fc > 1 ? fc - 1 : 0;
    eng->setCurrentFrame(f, sceneId);

    ovdtest::FrameSnapshot snap = ovdtest::captureFrameSnapshot(*eng, f);
    for (const ovdtest::MeshRecord& m : snap.mMeshes)
    {
        // Zero is allowed (mesh with no data at this frame yet); otherwise
        // the index count must be a multiple of 3 (triangles).
        EXPECT_EQ(m.mNbTriIndices % 3u, 0u) << "mesh " << m.mInstId << " (" << m.mClass << ")";
    }
    delete eng;
}

#endif // PVD_OVD_SUPPORT

#endif // TST_OVD_LOADER_H
