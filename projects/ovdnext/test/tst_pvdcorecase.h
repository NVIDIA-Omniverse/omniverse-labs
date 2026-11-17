/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
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

class PvdErrorCallback : public physx::PxErrorCallback
{
	virtual void reportError(physx::PxErrorCode::Enum code, const char* message, const char* file, int line)
	{
		std::cerr << "Error code: " << code << " File: " << file << ": " << line << " Message: " << message << '\n';
	}
};

class PvdEngineTest : public Test
{
  protected:
	static physx::pvdsdk::PxAllocatorCallback* m_alloc;
	static PvdErrorCallback* m_errorCallback;
	static physx::PxFoundation* sFoundation;

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

};

physx::pvdsdk::PxAllocatorCallback* PvdEngineTest::m_alloc = nullptr;
PvdErrorCallback* PvdEngineTest::m_errorCallback = nullptr;
physx::PxFoundation* PvdEngineTest::sFoundation = nullptr;

TEST_F(PvdEngineTest, LoadFile_apex_destruction)
{
	const char* CLIPFILENAME = TEST_DATA_DIR "/apex_destruction.pxd2";
	pvdapp::PvdEngine* engine = pvdapp::PvdEngine::load(*m_alloc, "./", CLIPFILENAME);
	ASSERT_TRUE(engine != nullptr);
	delete engine;
}

TEST_F(PvdEngineTest, LoadFile_cloth_physx34)
{
	const char* CLIPFILENAME = TEST_DATA_DIR "/cloth_physx34.pxd2";
	pvdapp::PvdEngine* engine = pvdapp::PvdEngine::load(*m_alloc, "./", CLIPFILENAME);
	ASSERT_TRUE(engine != nullptr);
	delete engine;
}

TEST_F(PvdEngineTest, LoadFile_ue4_shooter)
{
	const char* CLIPFILENAME = TEST_DATA_DIR "/ue4_shooter.pxd2";
	pvdapp::PvdEngine* engine = pvdapp::PvdEngine::load(*m_alloc, "./", CLIPFILENAME);
	ASSERT_TRUE(engine != nullptr);
	delete engine;
}
