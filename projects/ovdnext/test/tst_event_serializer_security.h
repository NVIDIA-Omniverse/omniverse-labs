/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "PvdInputValidation.h"
#include "PvdSimpleAllocator.h"
#include "PxPvdCommStreamTypes.h"

#include <PxErrorCallback.h>
#include <PxFoundation.h>
#include <PxFoundationVersion.h>
#include <foundation/PxAssert.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace event_serializer_security_test
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

class EventSerializerSecurityTest : public testing::Test
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

    static void appendU32(
        std::vector<uint8_t>& bytes, uint32_t value, bool byteSwapped = false)
    {
        for(uint32_t index = 0; index < sizeof(value); ++index)
        {
            const uint32_t shift = byteSwapped ? 24u - index * 8u : index * 8u;
            bytes.push_back(static_cast<uint8_t>(value >> shift));
        }
    }

    static physx::pvdsdk::ExtendedEventSerializer& makeSerializer(
        const std::vector<uint8_t>& bytes, bool byteSwapped = false)
    {
        physx::pvdsdk::ExtendedEventSerializer& serializer =
            physx::pvdsdk::ExtendedEventSerializer::createInputSerializer(
                byteSwapped);
        serializer.setData(physx::pvdsdk::DataRef<const uint8_t>(
            bytes.data(), static_cast<uint32_t>(bytes.size())));
        return serializer;
    }
};

pvdapp::CheckingAllocator* EventSerializerSecurityTest::sAllocator = nullptr;
SilentErrorCallback* EventSerializerSecurityTest::sErrorCallback = nullptr;
physx::PxFoundation* EventSerializerSecurityTest::sFoundation = nullptr;
IgnoringAssertHandler* EventSerializerSecurityTest::sIgnoringAssertHandler = nullptr;
physx::PxAssertHandler* EventSerializerSecurityTest::sOriginalAssertHandler = nullptr;

TEST_F(EventSerializerSecurityTest, RejectsTruncatedFixedSizeBlockBeforeAllocation)
{
    std::vector<uint8_t> bytes;
    appendU32(bytes, 1u);
    appendU32(bytes, 0x12345678u); // Half of one NameHandleValue.
    physx::pvdsdk::ExtendedEventSerializer& serializer = makeSerializer(bytes);
    const size_t allocatedBeforeParse = sAllocator->getTotalBytesAllocated();

    physx::pvdsdk::DataRef<physx::pvdsdk::NameHandleValue> values;
    serializer.streamify(values);

    EXPECT_FALSE(serializer.isGood());
    EXPECT_EQ(nullptr, values.begin());
    EXPECT_EQ(allocatedBeforeParse, sAllocator->getTotalBytesAllocated());
    serializer.release();
}

TEST_F(EventSerializerSecurityTest, RejectsTruncatedByteSwappedBlockBeforeAllocation)
{
    std::vector<uint8_t> bytes;
    appendU32(bytes, 1u, true);
    appendU32(bytes, 0x12345678u, true);
    physx::pvdsdk::ExtendedEventSerializer& serializer =
        makeSerializer(bytes, true);
    const size_t allocatedBeforeParse = sAllocator->getTotalBytesAllocated();

    physx::pvdsdk::DataRef<physx::pvdsdk::StreamPropMessageArg> values;
    serializer.streamify(values);

    EXPECT_FALSE(serializer.isGood());
    EXPECT_EQ(nullptr, values.begin());
    EXPECT_EQ(allocatedBeforeParse, sAllocator->getTotalBytesAllocated());
    serializer.release();
}

TEST_F(EventSerializerSecurityTest, RejectsOverflowingAndExcessiveBlockCounts)
{
    const uint32_t rejectedCounts[] = {
        (std::numeric_limits<uint32_t>::max)(),
        pvdapp::input::kMaxPropertyItems + 1u
    };

    for(uint32_t count : rejectedCounts)
    {
        std::vector<uint8_t> bytes;
        appendU32(bytes, count);
        physx::pvdsdk::ExtendedEventSerializer& serializer =
            makeSerializer(bytes);
        const size_t allocatedBeforeParse = sAllocator->getTotalBytesAllocated();

        physx::pvdsdk::DataRef<physx::pvdsdk::StringHandle> values;
        serializer.streamify(values);

        EXPECT_FALSE(serializer.isGood());
        EXPECT_EQ(nullptr, values.begin());
        EXPECT_EQ(allocatedBeforeParse, sAllocator->getTotalBytesAllocated());
        serializer.release();
    }
}

TEST_F(EventSerializerSecurityTest, RejectsUnterminatedLengthDelimitedString)
{
    std::vector<uint8_t> bytes;
    appendU32(bytes, 4u);
    bytes.insert(bytes.end(), {'t', 'e', 's', 't'});
    physx::pvdsdk::ExtendedEventSerializer& serializer = makeSerializer(bytes);

    physx::pvdsdk::String value = "unchanged";
    serializer.streamify(value);

    EXPECT_FALSE(serializer.isGood());
    EXPECT_STREQ("", value);
    serializer.release();
}

TEST_F(EventSerializerSecurityTest, AcceptsTerminatorWithinDeclaredStringBytes)
{
    std::vector<uint8_t> bytes;
    appendU32(bytes, 4u);
    bytes.insert(bytes.end(), {'x', 0, 'y', 'z'});
    appendU32(bytes, 0x12345678u);
    physx::pvdsdk::ExtendedEventSerializer& serializer = makeSerializer(bytes);

    physx::pvdsdk::String value = nullptr;
    uint32_t followingValue = 0;
    serializer.streamify(value);
    serializer.streamify(followingValue);

    EXPECT_TRUE(serializer.isGood());
    EXPECT_STREQ("x", value);
    EXPECT_EQ(0x12345678u, followingValue);
    serializer.release();
}

TEST_F(EventSerializerSecurityTest, DoesNotResumeAfterMalformedInput)
{
    std::vector<uint8_t> bytes;
    appendU32(bytes, 1u);
    bytes.push_back('x'); // No string terminator.
    appendU32(bytes, 0u); // A valid empty block that must not be consumed.
    physx::pvdsdk::ExtendedEventSerializer& serializer = makeSerializer(bytes);

    physx::pvdsdk::String value = nullptr;
    serializer.streamify(value);
    physx::pvdsdk::DataRef<physx::pvdsdk::StringHandle> values;
    serializer.streamify(values);

    EXPECT_FALSE(serializer.isGood());
    EXPECT_EQ(nullptr, values.begin());
    serializer.release();
}

TEST_F(EventSerializerSecurityTest, ParsesValidFixedSizeBlock)
{
    std::vector<uint8_t> bytes;
    appendU32(bytes, 1u);
    appendU32(bytes, 0x12345678u);
    appendU32(bytes, 0x9abcdef0u);
    physx::pvdsdk::ExtendedEventSerializer& serializer = makeSerializer(bytes);

    physx::pvdsdk::DataRef<physx::pvdsdk::NameHandleValue> values;
    serializer.streamify(values);

    EXPECT_TRUE(serializer.isGood());
    if(values.begin())
    {
        EXPECT_EQ(1u, values.size());
        EXPECT_EQ(0x12345678u, values[0].mName.mHandle);
        EXPECT_EQ(0x9abcdef0u, values[0].mValue);
    }
    else
    {
        ADD_FAILURE() << "valid block produced no values";
    }
    serializer.release();
}

} // namespace event_serializer_security_test
