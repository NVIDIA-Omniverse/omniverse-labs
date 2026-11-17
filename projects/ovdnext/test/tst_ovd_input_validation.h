/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "TestFiles.h"
#ifndef TST_OVD_INPUT_VALIDATION_H
#define TST_OVD_INPUT_VALIDATION_H

#ifdef PVD_OVD_SUPPORT

#include <gtest/gtest.h>

#include "OvdFileLoader.h"
#include "OvdInputValidation.h"
#include "OvdLiveSource.h"
#include "OmniPvdReader.h"
#include "OmniPvdSocketWriteStream.h"
#include "OmniPvdWriter.h"
#include "PvdDomParser.h"


#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <mutex>
#include <thread>
#include <vector>

extern "C" {
    OmniPvdReader* OMNI_PVD_CALL createOmniPvdReader();
    void OMNI_PVD_CALL destroyOmniPvdReader(OmniPvdReader& reader);
    OmniPvdWriter* OMNI_PVD_CALL createOmniPvdWriter();
    void OMNI_PVD_CALL destroyOmniPvdWriter(OmniPvdWriter& writer);
    OmniPvdSocketWriteStream* OMNI_PVD_CALL createOmniPvdSocketWriteStream(
        const char* address, uint16_t port, uint32_t sendTimeout);
    void OMNI_PVD_CALL destroyOmniPvdSocketWriteStream(OmniPvdSocketWriteStream& stream);
}

namespace
{
template <typename T>
void appendOvdValue(std::vector<uint8_t>& bytes, const T& value)
{
    const uint8_t* first = reinterpret_cast<const uint8_t*>(&value);
    bytes.insert(bytes.end(), first, first + sizeof(T));
}

void appendOvdHeader(std::vector<uint8_t>& bytes, uint32_t minor = 4, uint32_t major = 0)
{
    appendOvdValue<uint32_t>(bytes, major);
    appendOvdValue<uint32_t>(bytes, minor);
    appendOvdValue<uint32_t>(bytes, 0);
}

void appendOvdString(std::vector<uint8_t>& bytes, const char* value)
{
    const uint16_t length = static_cast<uint16_t>(std::strlen(value));
    appendOvdValue<uint16_t>(bytes, length);
    bytes.insert(bytes.end(), value, value + length);
}

void appendRegisterClass(std::vector<uint8_t>& bytes, uint32_t handle,
                         uint32_t base, const char* name, bool includeBase = true)
{
    appendOvdValue<uint8_t>(bytes, static_cast<uint8_t>(OmniPvdCommand::eREGISTER_CLASS));
    appendOvdValue<uint32_t>(bytes, handle);
    if (includeBase) appendOvdValue<uint32_t>(bytes, base);
    appendOvdString(bytes, name);
}

void appendRegisterAttribute(std::vector<uint8_t>& bytes, uint32_t classHandle,
                             uint32_t attributeHandle, const char* name,
                             OmniPvdDataType::Enum dataType = OmniPvdDataType::eUINT32,
                             uint32_t elementCount = 1)
{
    appendOvdValue<uint8_t>(bytes, static_cast<uint8_t>(OmniPvdCommand::eREGISTER_ATTRIBUTE));
    appendOvdValue<uint32_t>(bytes, classHandle);
    appendOvdValue<uint32_t>(bytes, attributeHandle);
    appendOvdValue<uint16_t>(bytes, static_cast<uint16_t>(dataType));
    appendOvdValue<uint32_t>(bytes, elementCount);
    appendOvdString(bytes, name);
}

void appendRegisterUniqueListAttribute(std::vector<uint8_t>& bytes, uint32_t classHandle,
                                       uint32_t attributeHandle, const char* name,
                                       OmniPvdDataType::Enum dataType = OmniPvdDataType::eOBJECT_HANDLE)
{
    appendOvdValue<uint8_t>(bytes,
        static_cast<uint8_t>(OmniPvdCommand::eREGISTER_UNIQUE_LIST_ATTRIBUTE));
    appendOvdValue<uint32_t>(bytes, classHandle);
    appendOvdValue<uint32_t>(bytes, attributeHandle);
    appendOvdValue<uint16_t>(bytes, static_cast<uint16_t>(dataType));
    appendOvdString(bytes, name);
}

void appendCreateObject(std::vector<uint8_t>& bytes, uint32_t classHandle,
                        uint64_t objectHandle, const char* name)
{
    appendOvdValue<uint8_t>(bytes, static_cast<uint8_t>(OmniPvdCommand::eCREATE_OBJECT));
    appendOvdValue<uint64_t>(bytes, 1);
    appendOvdValue<uint32_t>(bytes, classHandle);
    appendOvdValue<uint64_t>(bytes, objectHandle);
    appendOvdString(bytes, name);
}

void appendRecordMessage(std::vector<uint8_t>& bytes, const char* message)
{
    appendOvdValue<uint8_t>(bytes, static_cast<uint8_t>(OmniPvdCommand::eRECORD_MESSAGE));
    appendOvdValue<uint64_t>(bytes, 1);
    appendOvdString(bytes, message);
    appendOvdString(bytes, "file.cpp");
    appendOvdValue<uint32_t>(bytes, 10);
    appendOvdValue<uint32_t>(bytes, 1);
    appendOvdValue<uint32_t>(bytes, 0);
}

void appendSetAttribute(std::vector<uint8_t>& bytes, uint8_t depth,
                        uint32_t declaredLength, bool includePayload = true)
{
    appendOvdValue<uint8_t>(bytes, static_cast<uint8_t>(OmniPvdCommand::eSET_ATTRIBUTE));
    appendOvdValue<uint64_t>(bytes, 1);
    appendOvdValue<uint64_t>(bytes, 2);
    appendOvdValue<uint8_t>(bytes, depth);
    for (uint8_t i = 0; i < depth; ++i)
        appendOvdValue<uint32_t>(bytes, static_cast<uint32_t>(i + 1));
    appendOvdValue<uint32_t>(bytes, declaredLength);
    if (includePayload)
        bytes.insert(bytes.end(), declaredLength, uint8_t(0x5a));
}

void appendSetAttributeHandles(std::vector<uint8_t>& bytes, uint32_t attributeHandle,
                               const std::vector<uint64_t>& handles)
{
    appendOvdValue<uint8_t>(bytes, static_cast<uint8_t>(OmniPvdCommand::eSET_ATTRIBUTE));
    appendOvdValue<uint64_t>(bytes, 1);
    appendOvdValue<uint64_t>(bytes, 9);
    appendOvdValue<uint8_t>(bytes, 1);
    appendOvdValue<uint32_t>(bytes, attributeHandle);
    appendOvdValue<uint32_t>(bytes, static_cast<uint32_t>(handles.size() * sizeof(uint64_t)));
    for (uint64_t handle : handles) appendOvdValue<uint64_t>(bytes, handle);
}

void appendSetScalarAttribute(std::vector<uint8_t>& bytes, uint64_t objectHandle,
                              uint32_t attributeHandle, uint8_t value)
{
    appendOvdValue<uint8_t>(bytes, static_cast<uint8_t>(OmniPvdCommand::eSET_ATTRIBUTE));
    appendOvdValue<uint64_t>(bytes, 1);
    appendOvdValue<uint64_t>(bytes, objectHandle);
    appendOvdValue<uint8_t>(bytes, 1);
    appendOvdValue<uint32_t>(bytes, attributeHandle);
    appendOvdValue<uint32_t>(bytes, 1);
    appendOvdValue<uint8_t>(bytes, value);
}

void appendAttributeMutation(std::vector<uint8_t>& bytes, OmniPvdCommand::Enum command,
                             uint64_t objectHandle, uint32_t attributeHandle,
                             const std::vector<uint8_t>& payload)
{
    appendOvdValue<uint8_t>(bytes, static_cast<uint8_t>(command));
    appendOvdValue<uint64_t>(bytes, 1);
    appendOvdValue<uint64_t>(bytes, objectHandle);
    appendOvdValue<uint8_t>(bytes, 1);
    appendOvdValue<uint32_t>(bytes, attributeHandle);
    appendOvdValue<uint32_t>(bytes, static_cast<uint32_t>(payload.size()));
    bytes.insert(bytes.end(), payload.begin(), payload.end());
}

class VectorOvdReadStream : public OmniPvdReadStream
{
public:
    explicit VectorOvdReadStream(std::vector<uint8_t> bytes)
        : mBytes(std::move(bytes)), mOffset(0), mMaxRequestedRead(0) {}

    uint64_t OMNI_PVD_CALL readBytes(uint8_t* dest, uint64_t count) override
    {
        mMaxRequestedRead = std::max(mMaxRequestedRead, count);
        const uint64_t available = static_cast<uint64_t>(mBytes.size() - mOffset);
        const uint64_t copied = std::min(count, available);
        if (copied)
        {
            std::memcpy(dest, mBytes.data() + mOffset, static_cast<size_t>(copied));
            mOffset += static_cast<size_t>(copied);
        }
        return copied;
    }

    uint64_t OMNI_PVD_CALL skipBytes(uint64_t count) override
    {
        const uint64_t skipped = std::min<uint64_t>(count, mBytes.size() - mOffset);
        mOffset += static_cast<size_t>(skipped);
        return skipped;
    }

    bool OMNI_PVD_CALL openStream() override { return true; }
    bool OMNI_PVD_CALL closeStream() override { return true; }

    uint64_t maxRequestedRead() const { return mMaxRequestedRead; }

private:
    std::vector<uint8_t> mBytes;
    size_t mOffset;
    uint64_t mMaxRequestedRead;
};

class OvdReaderScope
{
public:
    OvdReaderScope() : reader(createOmniPvdReader()) {}
    ~OvdReaderScope() { if (reader) destroyOmniPvdReader(*reader); }
    OmniPvdReader* reader;
};

OvdResourceLimits tinyOvdLimits()
{
    OvdResourceLimits limits = OvdResourceLimits::defaults();
    limits.maxSingleAttributePayloadBytes = 8;
    limits.maxCumulativeAttributePayloadBytes = 8;
    limits.maxStreamBytes = 1024 * 1024;
    limits.maxCommands = 100;
    limits.maxAttributeMutations = 100;
    limits.maxRetainedDomBytes = 1024 * 1024;
    return limits;
}
} // namespace

TEST(OvdInputValidationTest, RejectsOversizeDeclaredPayloadBeforePayloadSizedInnerRead)
{
    const uint32_t rejectedPayloadLength = 1024;
    std::vector<uint8_t> bytes;
    appendOvdHeader(bytes);
    appendSetAttribute(bytes, 1, rejectedPayloadLength, false);
    VectorOvdReadStream inner(std::move(bytes));
    OvdValidatedReadStream validated(inner, tinyOvdLimits());
    OvdReaderScope scope;
    ASSERT_NE(scope.reader, nullptr);
    scope.reader->setReadStream(validated);

    EXPECT_EQ(scope.reader->getNextCommand(), OmniPvdCommand::eINVALID);
    EXPECT_EQ(validated.errorCode(), OvdInputErrorCode::AttributePayloadLimit);
    EXPECT_LT(inner.maxRequestedRead(), rejectedPayloadLength);
}

TEST(OvdInputValidationTest, RejectsAttributeHandleDepthAboveRuntimeStack)
{
    std::vector<uint8_t> bytes;
    appendOvdHeader(bytes);
    appendSetAttribute(bytes, 33, 0, false);
    VectorOvdReadStream inner(std::move(bytes));
    OvdValidatedReadStream validated(inner, tinyOvdLimits());
    OvdReaderScope scope;
    scope.reader->setReadStream(validated);

    EXPECT_EQ(scope.reader->getNextCommand(), OmniPvdCommand::eINVALID);
    EXPECT_EQ(validated.errorCode(), OvdInputErrorCode::AttributeHandleDepthLimit);
}

TEST(OvdInputValidationTest, RejectsCumulativeAttributePayloadBeforeThirdDomApply)
{
    std::vector<uint8_t> bytes;
    appendOvdHeader(bytes);
    appendRegisterClass(bytes, 1, 0, "C");
    appendRegisterAttribute(bytes, 1, 1, "a");
    appendCreateObject(bytes, 1, 2, "o");
    appendSetAttribute(bytes, 1, 4);
    appendSetAttribute(bytes, 1, 4);
    appendSetAttribute(bytes, 1, 4);
    VectorOvdReadStream inner(std::move(bytes));
    OvdValidatedReadStream validated(inner, tinyOvdLimits());
    OvdReaderScope scope;
    scope.reader->setReadStream(validated);

    for (int i = 0; i < 3; ++i)
    {
        const OmniPvdCommand::Enum schema = scope.reader->getNextCommand();
        ASSERT_NE(schema, OmniPvdCommand::eINVALID);
        ASSERT_TRUE(validated.validateCommand(schema, *scope.reader));
    }

    for (int i = 0; i < 2; ++i)
    {
        const OmniPvdCommand::Enum cmd = scope.reader->getNextCommand();
        ASSERT_EQ(cmd, OmniPvdCommand::eSET_ATTRIBUTE);
        ASSERT_TRUE(validated.validateCommand(cmd, *scope.reader));
    }
    const OmniPvdCommand::Enum third = scope.reader->getNextCommand();
    ASSERT_EQ(third, OmniPvdCommand::eSET_ATTRIBUTE);
    EXPECT_FALSE(validated.validateCommand(third, *scope.reader));
    EXPECT_EQ(validated.errorCode(), OvdInputErrorCode::CumulativeAttributePayloadLimit);
}

TEST(OvdInputValidationTest, RejectsFixedWidthPayloadThatDoesNotMatchRegisteredShape)
{
    std::vector<uint8_t> bytes;
    appendOvdHeader(bytes);
    appendRegisterClass(bytes, 1, 0, "C");
    appendRegisterAttribute(bytes, 1, 1, "value", OmniPvdDataType::eUINT32, 1);
    appendCreateObject(bytes, 1, 9, "o");
    appendAttributeMutation(bytes, OmniPvdCommand::eSET_ATTRIBUTE, 9, 1,
                            std::vector<uint8_t>{0x5a});

    VectorOvdReadStream inner(std::move(bytes));
    OvdValidatedReadStream validated(inner, OvdResourceLimits::defaults());
    OvdReaderScope scope;
    scope.reader->setReadStream(validated);

    for (int i = 0; i < 3; ++i)
    {
        const OmniPvdCommand::Enum command = scope.reader->getNextCommand();
        ASSERT_NE(command, OmniPvdCommand::eINVALID);
        ASSERT_TRUE(validated.validateCommand(command, *scope.reader));
    }
    const OmniPvdCommand::Enum mutation = scope.reader->getNextCommand();
    ASSERT_EQ(mutation, OmniPvdCommand::eSET_ATTRIBUTE);
    EXPECT_FALSE(validated.validateCommand(mutation, *scope.reader));
    EXPECT_EQ(validated.errorCode(), OvdInputErrorCode::InvalidAttributePayload);
}

TEST(OvdInputValidationTest, VariableWidthArrayRequiresWholeRegisteredElements)
{
    std::vector<uint8_t> bytes;
    appendOvdHeader(bytes);
    appendRegisterClass(bytes, 1, 0, "C");
    appendRegisterAttribute(bytes, 1, 1, "values", OmniPvdDataType::eUINT32, 0);
    appendCreateObject(bytes, 1, 9, "o");
    appendAttributeMutation(bytes, OmniPvdCommand::eSET_ATTRIBUTE, 9, 1,
                            std::vector<uint8_t>(8, 0x11));
    appendAttributeMutation(bytes, OmniPvdCommand::eSET_ATTRIBUTE, 9, 1,
                            std::vector<uint8_t>(6, 0x22));

    VectorOvdReadStream inner(std::move(bytes));
    OvdValidatedReadStream validated(inner, OvdResourceLimits::defaults());
    OvdReaderScope scope;
    scope.reader->setReadStream(validated);

    for (int i = 0; i < 4; ++i)
    {
        const OmniPvdCommand::Enum command = scope.reader->getNextCommand();
        ASSERT_NE(command, OmniPvdCommand::eINVALID);
        ASSERT_TRUE(validated.validateCommand(command, *scope.reader));
    }
    const OmniPvdCommand::Enum mutation = scope.reader->getNextCommand();
    ASSERT_EQ(mutation, OmniPvdCommand::eSET_ATTRIBUTE);
    EXPECT_FALSE(validated.validateCommand(mutation, *scope.reader));
    EXPECT_EQ(validated.errorCode(), OvdInputErrorCode::InvalidAttributePayload);
}

TEST(OvdInputValidationTest, RejectsMutationCommandThatDoesNotMatchAttributeKind)
{
    std::vector<uint8_t> bytes;
    appendOvdHeader(bytes);
    appendRegisterClass(bytes, 1, 0, "C");
    appendRegisterAttribute(bytes, 1, 1, "value", OmniPvdDataType::eUINT32, 1);
    appendCreateObject(bytes, 1, 9, "o");
    appendAttributeMutation(bytes, OmniPvdCommand::eADD_TO_UNIQUE_LIST_ATTRIBUTE,
                            9, 1, std::vector<uint8_t>(4, 0x33));

    VectorOvdReadStream inner(std::move(bytes));
    OvdValidatedReadStream validated(inner, OvdResourceLimits::defaults());
    OvdReaderScope scope;
    scope.reader->setReadStream(validated);

    for (int i = 0; i < 3; ++i)
    {
        const OmniPvdCommand::Enum command = scope.reader->getNextCommand();
        ASSERT_NE(command, OmniPvdCommand::eINVALID);
        ASSERT_TRUE(validated.validateCommand(command, *scope.reader));
    }
    const OmniPvdCommand::Enum mutation = scope.reader->getNextCommand();
    ASSERT_EQ(mutation, OmniPvdCommand::eADD_TO_UNIQUE_LIST_ATTRIBUTE);
    EXPECT_FALSE(validated.validateCommand(mutation, *scope.reader));
    EXPECT_EQ(validated.errorCode(), OvdInputErrorCode::InvalidAttributePayload);
}

TEST(OvdInputValidationTest, ReportsOnlyReferenceAttributeMutationsAsTopologyRelevant)
{
    std::vector<uint8_t> bytes;
    appendOvdHeader(bytes);
    appendRegisterClass(bytes, 1, 0, "C");
    appendRegisterAttribute(bytes, 1, 1, "transform", OmniPvdDataType::eFLOAT32, 3);
    appendRegisterUniqueListAttribute(bytes, 1, 2, "children");
    appendCreateObject(bytes, 1, 9, "o");
    appendAttributeMutation(bytes, OmniPvdCommand::eSET_ATTRIBUTE,
                            9, 1, std::vector<uint8_t>(3 * sizeof(float), 0));
    appendAttributeMutation(bytes, OmniPvdCommand::eADD_TO_UNIQUE_LIST_ATTRIBUTE,
                            9, 2, std::vector<uint8_t>(sizeof(uint64_t), 0));
    appendRecordMessage(bytes, "after-reference");

    VectorOvdReadStream inner(std::move(bytes));
    OvdValidatedReadStream validated(inner, OvdResourceLimits::defaults());
    OvdReaderScope scope;
    scope.reader->setReadStream(validated);

    for (int i = 0; i < 4; ++i)
    {
        const OmniPvdCommand::Enum command = scope.reader->getNextCommand();
        ASSERT_NE(command, OmniPvdCommand::eINVALID);
        ASSERT_TRUE(validated.validateCommand(command, *scope.reader));
        EXPECT_FALSE(validated.mayMutateTopology());
    }

    OmniPvdCommand::Enum command = scope.reader->getNextCommand();
    ASSERT_EQ(command, OmniPvdCommand::eSET_ATTRIBUTE);
    ASSERT_TRUE(validated.validateCommand(command, *scope.reader));
    EXPECT_FALSE(validated.mayMutateTopology());

    command = scope.reader->getNextCommand();
    ASSERT_EQ(command, OmniPvdCommand::eADD_TO_UNIQUE_LIST_ATTRIBUTE);
    ASSERT_TRUE(validated.validateCommand(command, *scope.reader));
    EXPECT_TRUE(validated.mayMutateTopology());

    command = scope.reader->getNextCommand();
    ASSERT_EQ(command, OmniPvdCommand::eRECORD_MESSAGE);
    ASSERT_TRUE(validated.validateCommand(command, *scope.reader));
    EXPECT_FALSE(validated.mayMutateTopology())
        << "the signal must describe only the most recently validated command";
}

TEST(OvdInputValidationTest, RejectsNarrowSchemaForFixedWidthPhysxParserHook)
{
    const char* actorClasses[] = {
        "PxActor", "PxActorType", "PxRigidActor", "PxRigidStatic",
        "PxRigidDynamic", "PxArticulationLink", "PxPBDParticleSystem", "PxDeformableBody"
    };
    for (const char* actorClass : actorClasses)
    for (const char* attributeName : {"type", "TYPE", "Type",
                                      "articulation", "ARTICULATION", "1rticulation"})
    for (bool validWidth : {false, true})
    {
        SCOPED_TRACE(actorClass);
        SCOPED_TRACE(attributeName);
        SCOPED_TRACE(validWidth);
        const bool actorType = attributeName[0] == 't' || attributeName[0] == 'T';
        const auto validDataType = actorType ? OmniPvdDataType::eUINT32
                                            : OmniPvdDataType::eOBJECT_HANDLE;
        const size_t validBytes = actorType ? sizeof(uint32_t) : sizeof(OmniPvdObjectHandle);
        std::vector<uint8_t> bytes;
        appendOvdHeader(bytes);
        appendRegisterClass(bytes, 1, 0, actorClass);
        appendRegisterAttribute(bytes, 1, 1, attributeName,
                               validWidth ? validDataType : OmniPvdDataType::eUINT8, 1);
        appendCreateObject(bytes, 1, 9, "actor");
        appendAttributeMutation(bytes, OmniPvdCommand::eSET_ATTRIBUTE, 9, 1,
                                std::vector<uint8_t>(validWidth ? validBytes : 1, 0));

        VectorOvdReadStream inner(std::move(bytes));
        OvdValidatedReadStream validated(inner, OvdResourceLimits::defaults());
        OvdReaderScope scope;
        scope.reader->setReadStream(validated);

        for (int i = 0; i < 3; ++i)
        {
            const OmniPvdCommand::Enum command = scope.reader->getNextCommand();
            ASSERT_NE(command, OmniPvdCommand::eINVALID);
            ASSERT_TRUE(validated.validateCommand(command, *scope.reader));
        }
        const OmniPvdCommand::Enum mutation = scope.reader->getNextCommand();
        ASSERT_EQ(mutation, OmniPvdCommand::eSET_ATTRIBUTE);
        EXPECT_EQ(validated.validateCommand(mutation, *scope.reader), validWidth);
        EXPECT_EQ(validated.errorCode(), validWidth ? OvdInputErrorCode::None
                                                   : OvdInputErrorCode::InvalidAttributePayload);
    }
}

TEST(OvdInputValidationTest, RejectsNarrowSchemaForGeometryReferenceHook)
{
    std::vector<uint8_t> bytes;
    appendOvdHeader(bytes);
    appendRegisterClass(bytes, 1, 0, "PxConvexMeshGeometry");
    appendRegisterAttribute(bytes, 1, 1, "convexmesh", OmniPvdDataType::eUINT8, 1);
    appendCreateObject(bytes, 1, 9, "geometry");
    appendAttributeMutation(bytes, OmniPvdCommand::eSET_ATTRIBUTE, 9, 1,
                            std::vector<uint8_t>{1});

    VectorOvdReadStream inner(std::move(bytes));
    OvdValidatedReadStream validated(inner, OvdResourceLimits::defaults());
    OvdReaderScope scope;
    scope.reader->setReadStream(validated);

    for (int i = 0; i < 3; ++i)
    {
        const OmniPvdCommand::Enum command = scope.reader->getNextCommand();
        ASSERT_NE(command, OmniPvdCommand::eINVALID);
        ASSERT_TRUE(validated.validateCommand(command, *scope.reader));
    }
    const OmniPvdCommand::Enum mutation = scope.reader->getNextCommand();
    ASSERT_EQ(mutation, OmniPvdCommand::eSET_ATTRIBUTE);
    EXPECT_FALSE(validated.validateCommand(mutation, *scope.reader));
    EXPECT_EQ(validated.errorCode(), OvdInputErrorCode::InvalidAttributePayload);
}

TEST(OvdInputValidationTest, RejectsNarrowRetainedParentLinkBeforeChildLinkUsesIt)
{
    for (const char* className : {"PxArticulationJointReducedCoordinate", "CustomJoint"})
    for (const char* attributeName : {"parentLink", "PARENTLINK", "ParentLink"})
    for (bool validWidth : {false, true})
    {
        SCOPED_TRACE(className);
        SCOPED_TRACE(attributeName);
        SCOPED_TRACE(validWidth);
        std::vector<uint8_t> bytes;
        appendOvdHeader(bytes);
        appendRegisterClass(bytes, 1, 0, className);
        appendRegisterAttribute(bytes, 1, 1, attributeName,
                                validWidth ? OmniPvdDataType::eOBJECT_HANDLE : OmniPvdDataType::eUINT8, 1);
        appendRegisterAttribute(bytes, 1, 2, "childLink", OmniPvdDataType::eOBJECT_HANDLE, 1);
        appendCreateObject(bytes, 1, 9, "joint");
        appendAttributeMutation(bytes, OmniPvdCommand::eSET_ATTRIBUTE, 9, 1,
                                std::vector<uint8_t>(validWidth ? sizeof(OmniPvdObjectHandle) : 1, 0));
        appendAttributeMutation(bytes, OmniPvdCommand::eSET_ATTRIBUTE, 9, 2,
                                std::vector<uint8_t>(sizeof(uint64_t), 0));

        VectorOvdReadStream inner(std::move(bytes));
        OvdValidatedReadStream validated(inner, OvdResourceLimits::defaults());
        OvdReaderScope scope;
        scope.reader->setReadStream(validated);

        for (int i = 0; i < 4; ++i)
        {
            const OmniPvdCommand::Enum command = scope.reader->getNextCommand();
            ASSERT_NE(command, OmniPvdCommand::eINVALID);
            ASSERT_TRUE(validated.validateCommand(command, *scope.reader));
        }
        const OmniPvdCommand::Enum parent = scope.reader->getNextCommand();
        ASSERT_EQ(parent, OmniPvdCommand::eSET_ATTRIBUTE);
        EXPECT_EQ(validated.validateCommand(parent, *scope.reader), validWidth);
        EXPECT_EQ(validated.errorCode(), validWidth ? OvdInputErrorCode::None
                                                   : OvdInputErrorCode::InvalidAttributePayload);
    }
}

TEST(OvdInputValidationTest, RejectsNarrowIntegrationVersionBeforeNextObjectUsesIt)
{
    for (const char* attributeName : {"ovdIntegrationVersionMajor", "ovdIntegrationVersionMinor",
                                      "OVDINTEGRATIONVERSIONMAJOR", "OVDINTEGRATIONVERSIONMINOR"})
    for (bool validWidth : {false, true})
    {
        SCOPED_TRACE(attributeName);
        SCOPED_TRACE(validWidth);
        std::vector<uint8_t> bytes;
        appendOvdHeader(bytes);
        appendRegisterClass(bytes, 1, 0, "Meta");
        appendRegisterAttribute(bytes, 1, 1, attributeName,
                                validWidth ? OmniPvdDataType::eUINT32 : OmniPvdDataType::eUINT8, 1);
        appendCreateObject(bytes, 1, 9, "metadata");
        appendAttributeMutation(bytes, OmniPvdCommand::eSET_ATTRIBUTE, 9, 1,
                                std::vector<uint8_t>(validWidth ? sizeof(uint32_t) : 1, 0));
        appendCreateObject(bytes, 1, 10, "trigger");

        VectorOvdReadStream inner(std::move(bytes));
        OvdValidatedReadStream validated(inner, OvdResourceLimits::defaults());
        OvdReaderScope scope;
        scope.reader->setReadStream(validated);

        for (int i = 0; i < 3; ++i)
        {
            const OmniPvdCommand::Enum command = scope.reader->getNextCommand();
            ASSERT_NE(command, OmniPvdCommand::eINVALID);
            ASSERT_TRUE(validated.validateCommand(command, *scope.reader));
        }
        const OmniPvdCommand::Enum version = scope.reader->getNextCommand();
        ASSERT_EQ(version, OmniPvdCommand::eSET_ATTRIBUTE);
        EXPECT_EQ(validated.validateCommand(version, *scope.reader), validWidth);
        EXPECT_EQ(validated.errorCode(), validWidth ? OvdInputErrorCode::None
                                                   : OvdInputErrorCode::InvalidAttributePayload);
    }
}

TEST(OvdInputValidationTest, DistinguishesCleanEofFromTruncatedAndUnknownCommands)
{
    {
        std::vector<uint8_t> bytes;
        appendOvdHeader(bytes);
        VectorOvdReadStream inner(std::move(bytes));
        OvdValidatedReadStream validated(inner, tinyOvdLimits());
        OvdReaderScope scope;
        scope.reader->setReadStream(validated);
        EXPECT_EQ(scope.reader->getNextCommand(), OmniPvdCommand::eINVALID);
        EXPECT_TRUE(validated.cleanEof());
        EXPECT_EQ(validated.errorCode(), OvdInputErrorCode::None);
    }
    {
        std::vector<uint8_t> bytes;
        appendOvdHeader(bytes);
        appendOvdValue<uint8_t>(bytes, static_cast<uint8_t>(OmniPvdCommand::eCREATE_OBJECT));
        appendOvdValue<uint64_t>(bytes, 1); // truncated before class/object/name
        VectorOvdReadStream inner(std::move(bytes));
        OvdValidatedReadStream validated(inner, tinyOvdLimits());
        OvdReaderScope scope;
        scope.reader->setReadStream(validated);
        EXPECT_EQ(scope.reader->getNextCommand(), OmniPvdCommand::eINVALID);
        EXPECT_FALSE(validated.cleanEof());
        EXPECT_EQ(validated.errorCode(), OvdInputErrorCode::TruncatedInput);
    }
    {
        std::vector<uint8_t> bytes;
        appendOvdHeader(bytes);
        appendOvdValue<uint8_t>(bytes, uint8_t(0xff));
        VectorOvdReadStream inner(std::move(bytes));
        OvdValidatedReadStream validated(inner, tinyOvdLimits());
        OvdReaderScope scope;
        scope.reader->setReadStream(validated);
        EXPECT_EQ(scope.reader->getNextCommand(), OmniPvdCommand::eINVALID);
        EXPECT_EQ(validated.errorCode(), OvdInputErrorCode::UnknownCommand);
    }
}

TEST(OvdInputValidationTest, AcceptsPre03RegisterClassLayoutWithoutBaseHandle)
{
    std::vector<uint8_t> bytes;
    appendOvdHeader(bytes, 2);
    appendOvdValue<uint8_t>(bytes, static_cast<uint8_t>(OmniPvdCommand::eREGISTER_CLASS));
    appendOvdValue<uint32_t>(bytes, 7);
    appendOvdValue<uint16_t>(bytes, 3);
    bytes.insert(bytes.end(), {'F', 'o', 'o'});
    VectorOvdReadStream inner(std::move(bytes));
    OvdValidatedReadStream validated(inner, tinyOvdLimits());
    OvdReaderScope scope;
    scope.reader->setReadStream(validated);

    const OmniPvdCommand::Enum cmd = scope.reader->getNextCommand();
    ASSERT_EQ(cmd, OmniPvdCommand::eREGISTER_CLASS);
    EXPECT_TRUE(validated.validateCommand(cmd, *scope.reader));
    EXPECT_STREQ(scope.reader->getClassName(), "Foo");
}

TEST(OvdInputValidationTest, UsesBaseClassFieldForNonzeroMajorEvenWhenMinorIsBelow3)
{
    std::vector<uint8_t> bytes;
    appendOvdHeader(bytes, 2, 1);
    appendRegisterClass(bytes, 7, 5, "Foo", true);
    VectorOvdReadStream inner(std::move(bytes));
    OvdValidatedReadStream validated(inner, tinyOvdLimits());

    uint8_t header[12] = {};
    ASSERT_EQ(validated.readBytes(header, sizeof(header)), sizeof(header));
    uint8_t command = 0;
    ASSERT_EQ(validated.readBytes(&command, sizeof(command)), sizeof(command));
    EXPECT_EQ(command, static_cast<uint8_t>(OmniPvdCommand::eREGISTER_CLASS));
    uint32_t classHandle = 0;
    uint32_t baseClassHandle = 0;
    ASSERT_EQ(validated.readBytes(reinterpret_cast<uint8_t*>(&classHandle), sizeof(classHandle)), sizeof(classHandle));
    ASSERT_EQ(validated.readBytes(reinterpret_cast<uint8_t*>(&baseClassHandle), sizeof(baseClassHandle)), sizeof(baseClassHandle));
    EXPECT_EQ(classHandle, 7u);
    EXPECT_EQ(baseClassHandle, 5u);
    EXPECT_FALSE(validated.failed());
}

TEST(OvdInputValidationTest, DuplicateClassHandleCannotEraseRetainedAttributeSlots)
{
    std::vector<uint8_t> bytes;
    appendOvdHeader(bytes);
    appendRegisterClass(bytes, 1, 0, "C");
    appendRegisterAttribute(bytes, 1, 1, "a");
    appendRegisterClass(bytes, 1, 0, "C");
    appendCreateObject(bytes, 1, 9, "o");
    VectorOvdReadStream inner(std::move(bytes));
    OvdResourceLimits limits = OvdResourceLimits::defaults();
    limits.maxRetainedDomBytes = 4400; // object fits only if duplicate registration erased its slot
    OvdValidatedReadStream validated(inner, limits);
    OvdReaderScope scope;
    scope.reader->setReadStream(validated);

    for (int i = 0; i < 3; ++i)
    {
        const OmniPvdCommand::Enum command = scope.reader->getNextCommand();
        ASSERT_NE(command, OmniPvdCommand::eINVALID);
        ASSERT_TRUE(validated.validateCommand(command, *scope.reader));
    }
    const OmniPvdCommand::Enum create = scope.reader->getNextCommand();
    ASSERT_EQ(create, OmniPvdCommand::eCREATE_OBJECT);
    EXPECT_FALSE(validated.validateCommand(create, *scope.reader));
    EXPECT_EQ(validated.errorCode(), OvdInputErrorCode::RetainedDomLimit);
}

TEST(OvdInputValidationTest, RejectsAttributeBeforeItsClassCanPoisonInheritanceAccounting)
{
    std::vector<uint8_t> bytes;
    appendOvdHeader(bytes);
    appendRegisterAttribute(bytes, 77, 1, "premature");
    VectorOvdReadStream inner(std::move(bytes));
    OvdValidatedReadStream validated(inner, OvdResourceLimits::defaults());
    OvdReaderScope scope;
    scope.reader->setReadStream(validated);

    const OmniPvdCommand::Enum command = scope.reader->getNextCommand();
    ASSERT_EQ(command, OmniPvdCommand::eREGISTER_ATTRIBUTE);
    EXPECT_FALSE(validated.validateCommand(command, *scope.reader));
    EXPECT_EQ(validated.errorCode(), OvdInputErrorCode::UnknownCommand);
}

TEST(OvdInputValidationTest, RejectsClassWithUnregisteredBase)
{
    std::vector<uint8_t> bytes;
    appendOvdHeader(bytes);
    appendRegisterClass(bytes, 2, 99, "Derived");
    VectorOvdReadStream inner(std::move(bytes));
    OvdValidatedReadStream validated(inner, OvdResourceLimits::defaults());
    OvdReaderScope scope;
    scope.reader->setReadStream(validated);

    const OmniPvdCommand::Enum command = scope.reader->getNextCommand();
    ASSERT_EQ(command, OmniPvdCommand::eREGISTER_CLASS);
    EXPECT_FALSE(validated.validateCommand(command, *scope.reader));
    EXPECT_EQ(validated.errorCode(), OvdInputErrorCode::UnknownCommand);
}

TEST(OvdInputValidationTest, ChargesQuadraticInheritedClassState)
{
    std::vector<uint8_t> bytes;
    appendOvdHeader(bytes);
    for (uint32_t handle = 1; handle <= 16; ++handle)
        appendRegisterClass(bytes, handle, handle - 1, "C");

    VectorOvdReadStream inner(std::move(bytes));
    OvdResourceLimits limits = OvdResourceLimits::defaults();
    limits.maxRetainedDomBytes = 25000;
    OvdValidatedReadStream validated(inner, limits);
    OvdReaderScope scope;
    scope.reader->setReadStream(validated);

    bool rejected = false;
    for (uint32_t handle = 1; handle <= 16; ++handle)
    {
        const OmniPvdCommand::Enum command = scope.reader->getNextCommand();
        ASSERT_EQ(command, OmniPvdCommand::eREGISTER_CLASS);
        if (!validated.validateCommand(command, *scope.reader))
        {
            rejected = true;
            break;
        }
    }
    EXPECT_TRUE(rejected);
    EXPECT_EQ(validated.errorCode(), OvdInputErrorCode::RetainedDomLimit);
}

TEST(OvdInputValidationTest, ChargesSyntheticDomNodesForObjectHandlePayloads)
{
    std::vector<uint8_t> bytes;
    appendOvdHeader(bytes);
    appendRegisterClass(bytes, 1, 0, "C");
    appendRegisterAttribute(bytes, 1, 1, "refs", OmniPvdDataType::eOBJECT_HANDLE);
    appendCreateObject(bytes, 1, 9, "o");
    appendSetAttributeHandles(bytes, 1, std::vector<uint64_t>{9});
    VectorOvdReadStream inner(std::move(bytes));
    OvdResourceLimits limits = OvdResourceLimits::defaults();
    limits.maxRetainedDomBytes = 4000;
    OvdValidatedReadStream validated(inner, limits);
    OvdReaderScope scope;
    scope.reader->setReadStream(validated);

    for (int i = 0; i < 3; ++i)
    {
        const OmniPvdCommand::Enum command = scope.reader->getNextCommand();
        ASSERT_NE(command, OmniPvdCommand::eINVALID);
        ASSERT_TRUE(validated.validateCommand(command, *scope.reader));
    }
    const OmniPvdCommand::Enum mutation = scope.reader->getNextCommand();
    ASSERT_EQ(mutation, OmniPvdCommand::eSET_ATTRIBUTE);
    EXPECT_FALSE(validated.validateCommand(mutation, *scope.reader));
    EXPECT_EQ(validated.errorCode(), OvdInputErrorCode::RetainedDomLimit);
}

TEST(OvdInputValidationTest, RecreatedExternalOwnerGetsFreshSyntheticNodeCharges)
{
    std::vector<uint8_t> bytes;
    appendOvdHeader(bytes);
    appendRegisterClass(bytes, 1, 0, "C");
    appendRegisterAttribute(bytes, 1, 1, "refs", OmniPvdDataType::eOBJECT_HANDLE);
    appendCreateObject(bytes, 1, 10, "target");
    appendCreateObject(bytes, 1, 9, "owner");
    appendSetAttributeHandles(bytes, 1, std::vector<uint64_t>{10});
    appendCreateObject(bytes, 1, 9, "owner"); // same external handle, new DOM object generation
    appendSetAttributeHandles(bytes, 1, std::vector<uint64_t>{10});
    VectorOvdReadStream inner(std::move(bytes));
    OvdResourceLimits limits = OvdResourceLimits::defaults();
    limits.maxRetainedDomBytes = 9400;
    OvdValidatedReadStream validated(inner, limits);
    OvdReaderScope scope;
    scope.reader->setReadStream(validated);

    for (int i = 0; i < 6; ++i)
    {
        const OmniPvdCommand::Enum command = scope.reader->getNextCommand();
        ASSERT_NE(command, OmniPvdCommand::eINVALID) << i;
        ASSERT_TRUE(validated.validateCommand(command, *scope.reader)) << i;
    }
    const OmniPvdCommand::Enum secondMutation = scope.reader->getNextCommand();
    ASSERT_EQ(secondMutation, OmniPvdCommand::eSET_ATTRIBUTE);
    EXPECT_FALSE(validated.validateCommand(secondMutation, *scope.reader));
    EXPECT_EQ(validated.errorCode(), OvdInputErrorCode::RetainedDomLimit);
}

TEST(OvdInputValidationTest, ChargesLateAttributeVectorGrowthForExistingObjects)
{
    std::vector<uint8_t> bytes;
    appendOvdHeader(bytes);
    appendRegisterClass(bytes, 1, 0, "C");
    appendCreateObject(bytes, 1, 9, "o");
    for (uint32_t attribute = 1; attribute <= 16; ++attribute)
        appendRegisterAttribute(bytes, 1, attribute, "a", OmniPvdDataType::eUINT8);
    appendSetScalarAttribute(bytes, 9, 16, 1);

    VectorOvdReadStream inner(std::move(bytes));
    OvdResourceLimits limits = OvdResourceLimits::defaults();
    limits.maxRetainedDomBytes = 8000;
    OvdValidatedReadStream validated(inner, limits);
    OvdReaderScope scope;
    scope.reader->setReadStream(validated);

    for (int i = 0; i < 18; ++i)
    {
        const OmniPvdCommand::Enum command = scope.reader->getNextCommand();
        ASSERT_NE(command, OmniPvdCommand::eINVALID) << i;
        ASSERT_TRUE(validated.validateCommand(command, *scope.reader)) << i;
    }
    const OmniPvdCommand::Enum mutation = scope.reader->getNextCommand();
    ASSERT_EQ(mutation, OmniPvdCommand::eSET_ATTRIBUTE);
    EXPECT_FALSE(validated.validateCommand(mutation, *scope.reader));
    EXPECT_EQ(validated.errorCode(), OvdInputErrorCode::RetainedDomLimit);
}

TEST(OvdInputValidationTest, StreamByteBudgetAcceptsExactAndRejectsOneOver)
{
    std::vector<uint8_t> header;
    appendOvdHeader(header);
    {
        VectorOvdReadStream inner(header);
        OvdResourceLimits limits = OvdResourceLimits::defaults();
        limits.maxStreamBytes = header.size();
        OvdValidatedReadStream validated(inner, limits);
        OvdReaderScope scope;
        scope.reader->setReadStream(validated);
        EXPECT_EQ(scope.reader->getNextCommand(), OmniPvdCommand::eINVALID);
        EXPECT_TRUE(validated.cleanEof());
    }
    {
        VectorOvdReadStream inner(header);
        OvdResourceLimits limits = OvdResourceLimits::defaults();
        limits.maxStreamBytes = header.size() - 1;
        OvdValidatedReadStream validated(inner, limits);
        OvdReaderScope scope;
        scope.reader->setReadStream(validated);
        EXPECT_EQ(scope.reader->getNextCommand(), OmniPvdCommand::eINVALID);
        EXPECT_EQ(validated.errorCode(), OvdInputErrorCode::StreamBytesLimit);
    }
}

TEST(OvdInputValidationTest, ClassBudgetRejectsOneOver)
{
    std::vector<uint8_t> bytes;
    appendOvdHeader(bytes);
    appendRegisterClass(bytes, 1, 0, "A");
    appendRegisterClass(bytes, 2, 0, "B");
    VectorOvdReadStream inner(std::move(bytes));
    OvdResourceLimits limits = OvdResourceLimits::defaults();
    limits.maxClasses = 1;
    OvdValidatedReadStream validated(inner, limits);
    OvdReaderScope scope;
    scope.reader->setReadStream(validated);
    OmniPvdCommand::Enum command = scope.reader->getNextCommand();
    ASSERT_TRUE(validated.validateCommand(command, *scope.reader));
    command = scope.reader->getNextCommand();
    EXPECT_FALSE(validated.validateCommand(command, *scope.reader));
    EXPECT_EQ(validated.errorCode(), OvdInputErrorCode::ClassCountLimit);
}

TEST(OvdInputValidationTest, TotalAndPerClassAttributeBudgetsRejectOneOver)
{
    auto run = [](bool perClass)
    {
        std::vector<uint8_t> bytes;
        appendOvdHeader(bytes);
        appendRegisterClass(bytes, 1, 0, "A");
        appendRegisterAttribute(bytes, 1, 1, "a");
        appendRegisterAttribute(bytes, 1, 2, "b");
        VectorOvdReadStream inner(std::move(bytes));
        OvdResourceLimits limits = OvdResourceLimits::defaults();
        if (perClass) limits.maxAttributesPerClass = 1;
        else limits.maxAttributes = 1;
        OvdValidatedReadStream validated(inner, limits);
        OvdReaderScope scope;
        scope.reader->setReadStream(validated);
        for (int i = 0; i < 2; ++i)
        {
            const OmniPvdCommand::Enum command = scope.reader->getNextCommand();
            EXPECT_TRUE(validated.validateCommand(command, *scope.reader));
        }
        const OmniPvdCommand::Enum secondAttribute = scope.reader->getNextCommand();
        EXPECT_FALSE(validated.validateCommand(secondAttribute, *scope.reader));
        EXPECT_EQ(validated.errorCode(), perClass
            ? OvdInputErrorCode::AttributesPerClassLimit
            : OvdInputErrorCode::AttributeCountLimit);
    };
    run(false);
    run(true);
}

TEST(OvdInputValidationTest, ObjectCreateBudgetRejectsOneOver)
{
    std::vector<uint8_t> bytes;
    appendOvdHeader(bytes);
    appendRegisterClass(bytes, 1, 0, "A");
    appendCreateObject(bytes, 1, 1, "one");
    appendCreateObject(bytes, 1, 2, "two");
    VectorOvdReadStream inner(std::move(bytes));
    OvdResourceLimits limits = OvdResourceLimits::defaults();
    limits.maxObjectCreates = 1;
    OvdValidatedReadStream validated(inner, limits);
    OvdReaderScope scope;
    scope.reader->setReadStream(validated);
    for (int i = 0; i < 2; ++i)
    {
        const OmniPvdCommand::Enum command = scope.reader->getNextCommand();
        ASSERT_TRUE(validated.validateCommand(command, *scope.reader));
    }
    const OmniPvdCommand::Enum secondObject = scope.reader->getNextCommand();
    EXPECT_FALSE(validated.validateCommand(secondObject, *scope.reader));
    EXPECT_EQ(validated.errorCode(), OvdInputErrorCode::ObjectCountLimit);
}

TEST(OvdInputValidationTest, MessageBudgetRejectsOneOver)
{
    std::vector<uint8_t> bytes;
    appendOvdHeader(bytes);
    appendRecordMessage(bytes, "one");
    appendRecordMessage(bytes, "two");
    VectorOvdReadStream inner(std::move(bytes));
    OvdResourceLimits limits = OvdResourceLimits::defaults();
    limits.maxMessages = 1;
    OvdValidatedReadStream validated(inner, limits);
    OvdReaderScope scope;
    scope.reader->setReadStream(validated);
    OmniPvdCommand::Enum command = scope.reader->getNextCommand();
    ASSERT_TRUE(validated.validateCommand(command, *scope.reader));
    command = scope.reader->getNextCommand();
    EXPECT_FALSE(validated.validateCommand(command, *scope.reader));
    EXPECT_EQ(validated.errorCode(), OvdInputErrorCode::MessageCountLimit);
}

TEST(OvdInputValidationTest, LiveBudgetFailureStopsBeforeOverBudgetDomApplyAndJoins)
{
    const uint16_t port = 24874;
    const uint64_t externalObject = 0xabc;
    OmniPvdDOMState dom;
    std::recursive_mutex mutex;
    OvdResourceLimits limits = OvdResourceLimits::defaults();
    limits.maxAttributeMutations = 1;
    pvdapp::OvdLiveSource source(&dom, &mutex, limits);
    ASSERT_TRUE(source.start(port));

    std::atomic<bool> producerOk{false};
    std::thread producer([&]
    {
        OmniPvdWriter* writer = createOmniPvdWriter();
        OmniPvdSocketWriteStream* stream = createOmniPvdSocketWriteStream("127.0.0.1", port, 3000);
        if (writer && stream && stream->openStream())
        {
            writer->setWriteStream(*stream);
            const OmniPvdClassHandle cls = writer->registerClass("BudgetObject");
            const OmniPvdAttributeHandle attr = writer->registerAttribute(
                cls, "value", OmniPvdDataType::eUINT32, 1);
            writer->createObject(1, cls, externalObject, "object");
            uint32_t value = 1;
            writer->setAttribute(1, externalObject, attr,
                reinterpret_cast<const uint8_t*>(&value), sizeof(value));
            value = 2;
            writer->setAttribute(1, externalObject, attr,
                reinterpret_cast<const uint8_t*>(&value), sizeof(value));
            stream->closeStream();
            producerOk = true;
        }
        if (stream) destroyOmniPvdSocketWriteStream(*stream);
        if (writer) destroyOmniPvdWriter(*writer);
    });

    for (int waited = 0; !source.hasEnded() && waited < 5000; waited += 10)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    producer.join();
    ASSERT_TRUE(producerOk.load());
    ASSERT_TRUE(source.hasEnded());
    EXPECT_TRUE(source.hasFailed());
    EXPECT_EQ(source.errorCode(), OvdInputErrorCode::AttributeMutationCountLimit);

    {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        const auto external = dom.mExternalToInternalHandleMap.find(externalObject);
        ASSERT_NE(external, dom.mExternalToInternalHandleMap.end());
        const auto object = dom.mObjectHandleToObjectMap.find(external->second);
        ASSERT_NE(object, dom.mObjectHandleToObjectMap.end());
        ASSERT_NE(object->second, nullptr);
        ASSERT_EQ(object->second->mInheritedClassInstances.size(), 1u);
        ASSERT_EQ(object->second->mInheritedClassInstances[0].mClassAttributeLists.size(), 1u);
        OmniPvdAttributeInstList* samples =
            object->second->mInheritedClassInstances[0].mClassAttributeLists[0];
        ASSERT_NE(samples, nullptr);
        ASSERT_NE(samples->mFirst, nullptr);
        EXPECT_EQ(samples->mFirst->mNextAttribute, nullptr)
            << "the rejected second mutation must not reach the append-only DOM";
    }

    source.stop(); // regression: closes any blocked read and joins cleanly
    EXPECT_TRUE(source.hasEnded());
}

TEST(OvdInputValidationTest, LiveMalformedPayloadIsRejectedAndWorkerStopsCleanly)
{
    const uint16_t port = 24875;
    const uint64_t externalObject = 0xdef;
    OmniPvdDOMState dom;
    std::recursive_mutex mutex;
    pvdapp::OvdLiveSource source(&dom, &mutex);
    ASSERT_TRUE(source.start(port));

    std::atomic<bool> producerConnected{false};
    std::thread producer([&]
    {
        OmniPvdWriter* writer = createOmniPvdWriter();
        OmniPvdSocketWriteStream* stream = createOmniPvdSocketWriteStream("127.0.0.1", port, 3000);
        if (writer && stream && stream->openStream())
        {
            producerConnected = true;
            writer->setWriteStream(*stream);
            const OmniPvdClassHandle cls = writer->registerClass("MalformedObject");
            const OmniPvdAttributeHandle attr = writer->registerAttribute(
                cls, "value", OmniPvdDataType::eUINT32, 1);
            writer->createObject(1, cls, externalObject, "object");
            const uint8_t shortValue = 1;
            writer->setAttribute(1, externalObject, attr, &shortValue, sizeof(shortValue));
            for (int waited = 0; !source.hasEnded() && waited < 3000; waited += 10)
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            stream->closeStream();
        }
        if (stream) destroyOmniPvdSocketWriteStream(*stream);
        if (writer) destroyOmniPvdWriter(*writer);
    });

    for (int waited = 0; !source.hasEnded() && waited < 5000; waited += 10)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    producer.join();
    ASSERT_TRUE(producerConnected.load());
    ASSERT_TRUE(source.hasEnded());
    EXPECT_EQ(source.errorCode(), OvdInputErrorCode::InvalidAttributePayload);

    {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        const auto external = dom.mExternalToInternalHandleMap.find(externalObject);
        ASSERT_NE(external, dom.mExternalToInternalHandleMap.end());
        const auto object = dom.mObjectHandleToObjectMap.find(external->second);
        ASSERT_NE(object, dom.mObjectHandleToObjectMap.end());
        ASSERT_NE(object->second, nullptr);
        ASSERT_EQ(object->second->mInheritedClassInstances.size(), 1u);
        const auto& attributes =
            object->second->mInheritedClassInstances[0].mClassAttributeLists;
        ASSERT_EQ(attributes.size(), 1u);
        EXPECT_EQ(attributes[0], nullptr) << "the malformed mutation must not reach the DOM";
    }

    source.stop();
    EXPECT_TRUE(source.hasEnded());
}

TEST(OvdInputValidationTest, LiveCyclicTopologyIsRejectedBeforeStoppedFramePublication)
{
    const uint16_t port = 24877;
    const OmniPvdContextHandle ctx = 1;
    const OmniPvdObjectHandle physicsHandle = 0xDA00;
    const OmniPvdObjectHandle sceneHandle = 0xDA01;
    const OmniPvdObjectHandle actorHandle = 0xDA02;
    OmniPvdDOMState dom;
    std::recursive_mutex mutex;
    pvdapp::OvdLiveSource source(&dom, &mutex);
    ASSERT_TRUE(source.start(port));

    std::atomic<bool> producerConnected{false};
    std::thread producer([&]
    {
        OmniPvdWriter* writer = createOmniPvdWriter();
        OmniPvdSocketWriteStream* stream = createOmniPvdSocketWriteStream(
            "127.0.0.1", port, 3000);
        if (writer && stream && stream->openStream())
        {
            producerConnected = true;
            writer->setWriteStream(*stream);
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
            writer->startFrame(sceneHandle, 9);
            writer->addToUniqueListAttribute(ctx, sceneHandle, sceneActors,
                reinterpret_cast<const uint8_t*>(&actorHandle), sizeof(actorHandle));
            writer->addToUniqueListAttribute(ctx, actorHandle, actorShapes,
                reinterpret_cast<const uint8_t*>(&sceneHandle), sizeof(sceneHandle));
            writer->stopFrame(sceneHandle, 9);

            for (int waited = 0; !source.hasEnded() && waited < 3000; waited += 10)
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            stream->closeStream();
        }
        if (stream) destroyOmniPvdSocketWriteStream(*stream);
        if (writer) destroyOmniPvdWriter(*writer);
    });

    for (int waited = 0; !source.hasEnded() && waited < 5000; waited += 10)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    producer.join();
    ASSERT_TRUE(producerConnected.load());
    ASSERT_TRUE(source.hasEnded());
    EXPECT_EQ(source.errorCode(), OvdInputErrorCode::InvalidDomTopology);
    EXPECT_EQ(source.completedFrameSerial(), 0u);
    EXPECT_EQ(source.maxFrame(), 0u);
    EXPECT_EQ(source.currentFrame(), 0u);

    source.stop();
    EXPECT_TRUE(source.hasEnded());
}

TEST(OvdInputValidationTest, StopCancelsConnectedIdleProducerPromptly)
{
    const uint16_t port = 24876;
    OmniPvdDOMState dom;
    std::recursive_mutex mutex;
    pvdapp::OvdLiveSource source(&dom, &mutex);
    ASSERT_TRUE(source.start(port));

    std::atomic<bool> producerConnected{false};
    std::atomic<bool> releaseProducer{false};
    std::thread producer([&]
    {
        OmniPvdWriter* writer = createOmniPvdWriter();
        OmniPvdSocketWriteStream* stream = createOmniPvdSocketWriteStream("127.0.0.1", port, 3000);
        if (writer && stream && stream->openStream())
        {
            writer->setWriteStream(*stream);
            writer->registerClass("IdleProducer"); // sends the version header, then idles
            producerConnected = true;
            while (!releaseProducer.load())
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            stream->closeStream();
        }
        if (stream) destroyOmniPvdSocketWriteStream(*stream);
        if (writer) destroyOmniPvdWriter(*writer);
    });

    for (int waited = 0; !source.isConnected() && waited < 5000; waited += 10)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const bool connected = producerConnected.load() && source.isConnected();
    if (!connected)
    {
        source.stop();
        releaseProducer = true;
        producer.join();
    }
    ASSERT_TRUE(connected);

    const auto started = std::chrono::steady_clock::now();
    source.stop();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started);
    EXPECT_LT(elapsed.count(), 1000);
    EXPECT_TRUE(source.hasEnded());

    releaseProducer = true;
    producer.join();
}

TEST(OvdInputValidationTest, FileLoaderRejectsMalformedFixedWidthPayload)
{
    std::vector<uint8_t> bytes;
    appendOvdHeader(bytes);
    appendRegisterClass(bytes, 1, 0, "C");
    appendRegisterAttribute(bytes, 1, 1, "value", OmniPvdDataType::eUINT32, 1);
    appendCreateObject(bytes, 1, 9, "o");
    appendAttributeMutation(bytes, OmniPvdCommand::eSET_ATTRIBUTE, 9, 1,
                            std::vector<uint8_t>{0x5a});

    test_files::Directory directory;
    ASSERT_TRUE(directory.valid());
    const std::string captureName = directory.file("ovd_invalid_payload.ovd");
    test_files::File capture(captureName);
    ASSERT_TRUE(capture.open(std::ios::out));
    ASSERT_EQ(capture.write(reinterpret_cast<const char*>(bytes.data()),
                            static_cast<std::streamsize>(bytes.size())),
              static_cast<std::streamsize>(bytes.size()));
    ASSERT_TRUE(capture.flush());
    capture.close();
    const std::string capturePath = captureName;

    OvdInputErrorCode error = OvdInputErrorCode::None;
    std::unique_ptr<pvdapp::OvdDomAdapter> adapter = pvdapp::loadOvdFile(
        capturePath.c_str(), OvdResourceLimits::defaults(), &error);
    test_files::fs::remove(test_files::fs::u8path(captureName));
    EXPECT_EQ(adapter, nullptr);
    EXPECT_EQ(error, OvdInputErrorCode::InvalidAttributePayload);
}

TEST(OvdInputValidationTest, FileLoaderRejectsCyclicDomBeforeAdapterPublication)
{
    std::vector<uint8_t> bytes;
    appendOvdHeader(bytes);
    appendRegisterClass(bytes, 1, 0, "PxPhysics");
    appendRegisterUniqueListAttribute(bytes, 1, 1, "scenes");
    appendRegisterClass(bytes, 2, 0, "PxScene");
    appendRegisterUniqueListAttribute(bytes, 2, 2, "actors");
    appendRegisterClass(bytes, 3, 0, "PxActor");
    appendRegisterUniqueListAttribute(bytes, 3, 3, "shapes");
    appendCreateObject(bytes, 1, 0xCA00, "physics");
    appendCreateObject(bytes, 2, 0xCA01, "scene");
    appendCreateObject(bytes, 3, 0xCA02, "actor");

    const auto objectHandlePayload = [](uint64_t handle)
    {
        std::vector<uint8_t> payload(sizeof(handle));
        std::memcpy(payload.data(), &handle, sizeof(handle));
        return payload;
    };
    appendAttributeMutation(bytes, OmniPvdCommand::eADD_TO_UNIQUE_LIST_ATTRIBUTE,
                            0xCA00, 1, objectHandlePayload(0xCA01));
    appendAttributeMutation(bytes, OmniPvdCommand::eADD_TO_UNIQUE_LIST_ATTRIBUTE,
                            0xCA01, 2, objectHandlePayload(0xCA02));
    appendAttributeMutation(bytes, OmniPvdCommand::eADD_TO_UNIQUE_LIST_ATTRIBUTE,
                            0xCA02, 3, objectHandlePayload(0xCA01));

    test_files::Directory directory;
    ASSERT_TRUE(directory.valid());
    const std::string captureName = directory.file("ovd_cyclic_topology.ovd");
    test_files::File capture(captureName);
    ASSERT_TRUE(capture.open(std::ios::out));
    ASSERT_EQ(capture.write(reinterpret_cast<const char*>(bytes.data()),
                            static_cast<std::streamsize>(bytes.size())),
              static_cast<std::streamsize>(bytes.size()));
    ASSERT_TRUE(capture.flush());
    capture.close();
    const std::string capturePath = captureName;

    OvdInputErrorCode error = OvdInputErrorCode::None;
    std::unique_ptr<pvdapp::OvdDomAdapter> adapter = pvdapp::loadOvdFile(
        capturePath.c_str(), OvdResourceLimits::defaults(), &error);
    test_files::fs::remove(test_files::fs::u8path(captureName));
    EXPECT_EQ(adapter, nullptr);
    EXPECT_EQ(error, OvdInputErrorCode::InvalidDomTopology);
}

TEST(OvdInputValidationTest, FileLoaderRejectsTinyCommandBudgetAndClearsGlobalMessages)
{
    OvdResourceLimits limits = OvdResourceLimits::defaults();
    limits.maxCommands = 1;
    gOmniPvdMessages.emplace_back();
    OvdInputErrorCode error = OvdInputErrorCode::None;

    std::unique_ptr<pvdapp::OvdDomAdapter> adapter =
        pvdapp::loadOvdFile(TEST_DATA_DIR "/test_scene.ovd", limits, &error);

    EXPECT_EQ(adapter, nullptr);
    EXPECT_EQ(error, OvdInputErrorCode::CommandCountLimit);
    EXPECT_TRUE(gOmniPvdMessages.empty());
}

TEST(OvdInputValidationTest, FileLoaderReportsStableOpenFailure)
{
    OvdInputErrorCode error = OvdInputErrorCode::None;
    std::unique_ptr<pvdapp::OvdDomAdapter> adapter = pvdapp::loadOvdFile(
        TEST_DATA_DIR "/definitely_missing_budget_test.ovd",
        OvdResourceLimits::defaults(), &error);
    EXPECT_EQ(adapter, nullptr);
    EXPECT_EQ(error, OvdInputErrorCode::ReaderRejectedCommand);
}

TEST(OvdInputValidationTest, FileLoaderAcceptsRepresentativeCaptureWithDefaultBudgets)
{
    OvdInputErrorCode error = OvdInputErrorCode::None;
    std::unique_ptr<pvdapp::OvdDomAdapter> adapter =
        pvdapp::loadOvdFile(TEST_DATA_DIR "/test_scene.ovd", OvdResourceLimits::defaults(), &error);
    EXPECT_NE(adapter, nullptr);
    EXPECT_EQ(error, OvdInputErrorCode::None);
}

TEST(OvdInputValidationTest, FileLoaderAcceptsLargestAvailableCorpusCapture)
{
    const char* path = TEST_DATA_DIR "/2026_04_23_10_39_03_01_rec.ovd";
    std::ifstream probe(path, std::ios::binary);
    if (!probe.good())
        GTEST_SKIP() << "largest local corpus capture is not installed";
    probe.close();

    OvdInputErrorCode error = OvdInputErrorCode::None;
    std::unique_ptr<pvdapp::OvdDomAdapter> adapter =
        pvdapp::loadOvdFile(path, OvdResourceLimits::defaults(), &error);
    EXPECT_NE(adapter, nullptr) << ovdInputErrorMessage(error);
    EXPECT_EQ(error, OvdInputErrorCode::None);
}

#endif // PVD_OVD_SUPPORT
#endif // TST_OVD_INPUT_VALIDATION_H
