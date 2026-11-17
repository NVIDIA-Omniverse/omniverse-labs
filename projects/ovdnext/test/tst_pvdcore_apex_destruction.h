/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TST_PVDCORE_APEX_DESTRUCTION_H
#define TST_PVDCORE_APEX_DESTRUCTION_H

#include <iostream>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <PvdEngine.h>
#include <PvdSimpleAllocator.h>
#include <PxErrorCallback.h>
#include <PxFoundation.h>
#include <PxFoundationVersion.h>

using namespace testing;

using namespace physx;

class PvdEngineApexDestructionTest : public Test
{
  protected:
	class PvdErrorCallback : public physx::PxErrorCallback
	{
		virtual void reportError(physx::PxErrorCode::Enum code, const char* message, const char* file, int line)
		{
			std::cerr << "Error code: " << code << "File: " << file << ": " << line << " Message: " << message << '\n';
		}
	};

	static physx::pvdsdk::PxAllocatorCallback* m_alloc;
	static PvdErrorCallback* m_errorCallback;
	static physx::PxFoundation* sFoundation;

	pvdapp::PvdEngine* mEngine = nullptr;

	static void SetUpTestCase()
	{
		m_alloc = new pvdapp::CheckingAllocator();
		m_errorCallback = new PvdErrorCallback();
		sFoundation = PxCreateFoundation(PX_FOUNDATION_VERSION, *m_alloc, *m_errorCallback);
	}

	static void TearDownTestCase()
	{
		sFoundation->release();
		delete m_errorCallback;
		delete m_alloc;
	}

	virtual void SetUp()
	{
		const char* CLIPFILENAME = TEST_DATA_DIR "/apex_destruction.pxd2";
		mEngine = pvdapp::PvdEngine::load(*m_alloc, "./", CLIPFILENAME);
		ASSERT_TRUE(mEngine != nullptr);
	}
	virtual void TearDown()
	{
		delete mEngine;
	}
};

physx::pvdsdk::PxAllocatorCallback* PvdEngineApexDestructionTest::m_alloc = nullptr;
PvdEngineApexDestructionTest::PvdErrorCallback* PvdEngineApexDestructionTest::m_errorCallback = nullptr;
physx::PxFoundation* PvdEngineApexDestructionTest::sFoundation = nullptr;

TEST_F(PvdEngineApexDestructionTest, SceneNumber)
{
	const PxU32 sceneNumber = mEngine->getNbScenes();
	EXPECT_EQ(sceneNumber, 2);
	ASSERT_TRUE(mEngine != nullptr);
}

TEST_F(PvdEngineApexDestructionTest, FrameCount)
{
	const PxU32 frameCount0 = mEngine->getFrameCount(74133808);
	const PxU32 frameCount1 = mEngine->getFrameCount(87068096);
	EXPECT_EQ(frameCount0, 260);
	EXPECT_EQ(frameCount1, 260);
	ASSERT_TRUE(mEngine != nullptr);
}

TEST_F(PvdEngineApexDestructionTest, EventCount)
{
	const PxU64 eventCount = mEngine->getEventCount();
	EXPECT_EQ(eventCount, 329912);
	ASSERT_TRUE(mEngine != nullptr);
}

TEST_F(PvdEngineApexDestructionTest, ClassCount)
{
	const PxU64 classCount = mEngine->getNbClasses();
	EXPECT_EQ(classCount, 119);
	ASSERT_TRUE(mEngine != nullptr);
}

TEST_F(PvdEngineApexDestructionTest, ProfileZonesCount)
{
	const PxU32 profileZonesCount = mEngine->getNbProfileZones();
	EXPECT_EQ(profileZonesCount, 3);
	ASSERT_TRUE(mEngine != nullptr);
}

TEST_F(PvdEngineApexDestructionTest, InstancesCount)
{
	const PxU32 instancesCount = mEngine->getNbInstances();
	EXPECT_EQ(instancesCount, 0);
	ASSERT_TRUE(mEngine != nullptr);
}

TEST_F(PvdEngineApexDestructionTest, InstancesCountAtFrame1)
{
	const int sceneId = 74133808;
	mEngine->setCurrentFrame(1, sceneId);
	const PxU32 instancesCount = mEngine->getNbInstances();
	EXPECT_EQ(instancesCount, 29);
	ASSERT_TRUE(mEngine != nullptr);
}

TEST_F(PvdEngineApexDestructionTest, InstancesCountAtFrame128)
{
	const int sceneId = 74133808;
	mEngine->setCurrentFrame(128, sceneId);
	const PxU32 instancesCount = mEngine->getNbInstances();
	EXPECT_EQ(instancesCount, 4578);
	ASSERT_TRUE(mEngine != nullptr);
}

TEST_F(PvdEngineApexDestructionTest, InstancesCountAtFrame256)
{
	const int sceneId = 74133808;
	mEngine->setCurrentFrame(256, sceneId);
	const PxU32 instancesCount = mEngine->getNbInstances();
	EXPECT_EQ(instancesCount, 4784);
	ASSERT_TRUE(mEngine != nullptr);
}

TEST_F(PvdEngineApexDestructionTest, MemoryEventEntriesCount)
{
	const PxU32 memoryEventEntriesCount = mEngine->getNbMemoryEventEntries();
	EXPECT_EQ(memoryEventEntriesCount, 376);
	ASSERT_TRUE(mEngine != nullptr);
}

TEST_F(PvdEngineApexDestructionTest, MemoryTotalsCount)
{
	const PxU32 memoryTotalsCount = mEngine->getNbMemoryTotals();
	EXPECT_EQ(memoryTotalsCount, 0);
	ASSERT_TRUE(mEngine != nullptr);
}

TEST_F(PvdEngineApexDestructionTest, MemoryTotalsCountAtFrame1)
{
	const int sceneId = 74133808;
	mEngine->setCurrentFrame(1, sceneId);
	const PxU32 memoryTotalsCount = mEngine->getNbMemoryTotals();
	EXPECT_EQ(memoryTotalsCount, 324);
	ASSERT_TRUE(mEngine != nullptr);
}

TEST_F(PvdEngineApexDestructionTest, MemoryTotalsCountAtFrame128)
{
	const int sceneId = 74133808;
	mEngine->setCurrentFrame(128, sceneId);
	const PxU32 memoryTotalsCount = mEngine->getNbMemoryTotals();
	EXPECT_EQ(memoryTotalsCount, 375);
	ASSERT_TRUE(mEngine != nullptr);
}

TEST_F(PvdEngineApexDestructionTest, ScrollFrame)
{
	const int sceneId = 74133808;
	const PxU32 frameCount = mEngine->getFrameCount(sceneId);
	for(PxU32 frame = 0; frame < frameCount; ++frame)
	{
		mEngine->setCurrentFrame(frame, sceneId);
	}
	const PxI64 currentEvent = mEngine->getCurrentEvent();
	EXPECT_EQ(currentEvent, 328961);
	ASSERT_TRUE(mEngine != nullptr);
}

#endif // TST_PVDCORE_APEX_DESTRUCTION_H
