/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "PvdSimpleAllocator.h"
#include "PxPvdUserRenderImpl.h"

#include <PxErrorCallback.h>
#include <PxFoundation.h>
#include <PxFoundationVersion.h>
#include <foundation/PxAssert.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace renderer_security_test
{

class SilentErrorCallback : public physx::PxErrorCallback
{
    void reportError(
        physx::PxErrorCode::Enum, const char*, const char*, int) override
    {
    }
};

class IgnoringAssertHandler : public physx::PxAssertHandler
{
    void operator()(const char*, const char*, int, bool& ignore) override
    {
        ignore = true;
    }
};

class CountingRenderHandler : public physx::pvdsdk::PvdUserRenderHandler
{
  public:
    uint32_t callCount = 0;

#define DECLARE_PVD_IMMEDIATE_RENDER_TYPE(type)                                      \
    void handleRenderEvent(                                                         \
        const physx::pvdsdk::type##RenderEvent&) override                           \
    {                                                                                \
        ++callCount;                                                                 \
    }
#include "PxPvdUserRenderTypes.h"
#undef DECLARE_PVD_IMMEDIATE_RENDER_TYPE
};

class RendererParserSecurityTest : public testing::Test
{
  protected:
    static pvdapp::CheckingAllocator* sAllocator;
    static SilentErrorCallback* sErrorCallback;
    static physx::PxFoundation* sFoundation;
    static IgnoringAssertHandler* sIgnoringAssertHandler;
    static physx::PxAssertHandler* sOriginalAssertHandler;

    static void SetUpTestSuite()
    {
        sAllocator = new pvdapp::CheckingAllocator();
        sErrorCallback = new SilentErrorCallback();
        sFoundation = PxCreateFoundation(
            PX_FOUNDATION_VERSION, *sAllocator, *sErrorCallback);
        ASSERT_NE(nullptr, sFoundation);
        sOriginalAssertHandler = &physx::PxGetAssertHandler();
        sIgnoringAssertHandler = new IgnoringAssertHandler();
        physx::PxSetAssertHandler(*sIgnoringAssertHandler);
    }

    static void TearDownTestSuite()
    {
        physx::PxSetAssertHandler(*sOriginalAssertHandler);
        delete sIgnoringAssertHandler;
        sFoundation->release();
        delete sErrorCallback;
        delete sAllocator;
    }

    static std::vector<uint8_t> pointsHeader(uint32_t count, bool byteSwapped)
    {
        std::vector<uint8_t> bytes;
        bytes.push_back(static_cast<uint8_t>(
            physx::pvdsdk::PvdUserRenderTypes::Points));
        appendU32(bytes, count, byteSwapped);
        return bytes;
    }

    static std::vector<uint8_t> textHeader(uint32_t length, bool byteSwapped)
    {
        std::vector<uint8_t> bytes;
        bytes.push_back(static_cast<uint8_t>(
            physx::pvdsdk::PvdUserRenderTypes::Text));
        bytes.resize(bytes.size() + 20u, 0);
        appendU32(bytes, length, byteSwapped);
        return bytes;
    }

    static void appendU32(
        std::vector<uint8_t>& bytes, uint32_t value, bool byteSwapped)
    {
        for(uint32_t index = 0; index < sizeof(value); ++index)
        {
            const uint32_t shift = byteSwapped ? 24u - index * 8u : index * 8u;
            bytes.push_back(static_cast<uint8_t>(value >> shift));
        }
    }

    static void expectRejectedWithoutAllocation(
        const std::vector<uint8_t>& bytes, bool byteSwapped)
    {
        physx::pvdsdk::PvdUserRenderParser& parser =
            physx::pvdsdk::PvdUserRenderParser::create(byteSwapped);
        CountingRenderHandler handler;
        const size_t allocatedBeforeParse = sAllocator->getTotalBytesAllocated();

        parser.parseData(
            physx::pvdsdk::DataRef<const uint8_t>(
                bytes.data(), static_cast<uint32_t>(bytes.size())),
            handler);

        EXPECT_EQ(0u, handler.callCount);
        EXPECT_EQ(allocatedBeforeParse, sAllocator->getTotalBytesAllocated());
        parser.release();
    }

    static void expectAccepted(
        const std::vector<uint8_t>& bytes, bool byteSwapped)
    {
        physx::pvdsdk::PvdUserRenderParser& parser =
            physx::pvdsdk::PvdUserRenderParser::create(byteSwapped);
        CountingRenderHandler handler;
        parser.parseData(
            physx::pvdsdk::DataRef<const uint8_t>(
                bytes.data(), static_cast<uint32_t>(bytes.size())),
            handler);
        EXPECT_EQ(1u, handler.callCount);
        parser.release();
    }
};

pvdapp::CheckingAllocator* RendererParserSecurityTest::sAllocator = nullptr;
SilentErrorCallback* RendererParserSecurityTest::sErrorCallback = nullptr;
physx::PxFoundation* RendererParserSecurityTest::sFoundation = nullptr;
IgnoringAssertHandler* RendererParserSecurityTest::sIgnoringAssertHandler = nullptr;
physx::PxAssertHandler* RendererParserSecurityTest::sOriginalAssertHandler = nullptr;

TEST_F(RendererParserSecurityTest, RejectsTruncatedNativeArrayBeforeAllocation)
{
    expectRejectedWithoutAllocation(pointsHeader(1, false), false);
}

TEST_F(RendererParserSecurityTest, RejectsOverflowingNativeArrayCount)
{
    expectRejectedWithoutAllocation(pointsHeader(0x40000000u, false), false);
}

TEST_F(RendererParserSecurityTest, RejectsTruncatedByteSwappedArrayBeforeAllocation)
{
    expectRejectedWithoutAllocation(pointsHeader(1, true), true);
}

TEST_F(RendererParserSecurityTest, ParsesValidNativeAndByteSwappedArrays)
{
    std::vector<uint8_t> native = pointsHeader(1, false);
    native.resize(native.size() + sizeof(physx::pvdsdk::PvdDebugPoint), 0);
    expectAccepted(native, false);

    std::vector<uint8_t> byteSwapped = pointsHeader(1, true);
    byteSwapped.resize(byteSwapped.size() + sizeof(physx::pvdsdk::PvdDebugPoint), 0);
    expectAccepted(byteSwapped, true);
}

TEST_F(RendererParserSecurityTest, RejectsTruncatedNativeTextBeforeAllocation)
{
    expectRejectedWithoutAllocation(textHeader(1, false), false);
}

TEST_F(RendererParserSecurityTest, RejectsTruncatedByteSwappedTextBeforeAllocation)
{
    expectRejectedWithoutAllocation(textHeader(1, true), true);
}

TEST_F(RendererParserSecurityTest, RejectsUnterminatedNativeTextBeforeAllocation)
{
    std::vector<uint8_t> bytes = textHeader(1, false);
    bytes.push_back('x');
    expectRejectedWithoutAllocation(bytes, false);
}

TEST_F(RendererParserSecurityTest, ParsesValidNativeAndByteSwappedText)
{
    std::vector<uint8_t> native = textHeader(2, false);
    native.push_back('x');
    native.push_back(0);
    expectAccepted(native, false);

    std::vector<uint8_t> byteSwapped = textHeader(2, true);
    byteSwapped.resize(byteSwapped.size() + 2u, 'x');
    byteSwapped.back() = 0;
    expectAccepted(byteSwapped, true);
}

} // namespace renderer_security_test
