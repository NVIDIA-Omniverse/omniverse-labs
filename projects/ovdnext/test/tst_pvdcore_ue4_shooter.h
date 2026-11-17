/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TST_PVDCORE_UE4_SHOOTER_H
#define TST_PVDCORE_UE4_SHOOTER_H

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

class PvdEngineUE4ShooterTest : public Test
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
		const char* CLIPFILENAME = TEST_DATA_DIR "/ue4_shooter.pxd2";
		mEngine = pvdapp::PvdEngine::load(*m_alloc, "./", CLIPFILENAME);
		ASSERT_TRUE(mEngine != nullptr);
	}
	virtual void TearDown()
	{
		delete mEngine;
	}
};

physx::pvdsdk::PxAllocatorCallback* PvdEngineUE4ShooterTest::m_alloc = nullptr;
PvdEngineUE4ShooterTest::PvdErrorCallback* PvdEngineUE4ShooterTest::m_errorCallback = nullptr;
physx::PxFoundation* PvdEngineUE4ShooterTest::sFoundation = nullptr;

TEST_F(PvdEngineUE4ShooterTest, SceneNumber)
{
	const PxU32 sceneNumber = mEngine->getNbScenes();
	EXPECT_EQ(sceneNumber, 4);
}

#endif // TST_PVDCORE_UE4_SHOOTER_H
