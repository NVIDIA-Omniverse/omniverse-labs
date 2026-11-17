/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PVD_OVD_INPUT_VALIDATION_H
#define PVD_OVD_INPUT_VALIDATION_H

#ifdef PVD_OVD_SUPPORT

#include "OmniPvdCommands.h"
#include "OmniPvdDefines.h"
#include "OmniPvdReadStream.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class OmniPvdReader;

enum class OvdInputErrorCode : uint32_t
{
    None = 0,
    TruncatedInput,
    UnknownCommand,
    InvalidStringLength,
    InvalidAttributePayload,
    AttributePayloadLimit,
    AttributeHandleDepthLimit,
    CumulativeAttributePayloadLimit,
    StreamBytesLimit,
    CommandCountLimit,
    AttributeMutationCountLimit,
    ClassCountLimit,
    AttributeCountLimit,
    AttributesPerClassLimit,
    ObjectCountLimit,
    MessageCountLimit,
    RetainedDomLimit,
    ReaderRejectedCommand,
    UnsupportedIntegrationVersion,
    EmptyDom,
    InvalidDomTopology,
    InvalidCaptureCatalog,
    CaptureCatalogRowLimit,
    CaptureCatalogStringLimit,
    CaptureCatalogAllocationLimit,
    CaptureCatalogCommandLimit,
    CaptureCatalogMutationLimit
};

enum class CatalogValidationMode { Ordinary, SeparateNonRetained };
enum class OvdCommandDestination { Dom, Catalog };
struct OvdValidatedCaptureCommand
{
    enum Kind { Create, Destroy, Field, Commit, MetadataCreate, MetadataDestroy, MetadataField } kind;
    uint64_t row = 0;
    uint32_t schema = 0;
    uint8_t field = 0;
    uint64_t value = 0;
    const char* text = nullptr;
    size_t textSize = 0;
};
// C++14 boundary: ordinary OvdData users neither link the catalog store nor
// receive exemptions. The non-retaining sink owns bounded row lifetimes.
class OvdCaptureValidationSink
{
public:
    virtual ~OvdCaptureValidationSink() = default;
    virtual uint32_t catalogRowSchema(uint64_t row) const = 0;
    virtual OvdInputErrorCode observeValidatedCapture(const OvdValidatedCaptureCommand& command) = 0;
};
struct OvdResourceUsage
{
    uint64_t streamBytes, commands, attributeMutations, payloadBytes, classes, attributes,
             objectCreates, messages, retainedDomBytes;
};

const char* ovdInputErrorMessage(OvdInputErrorCode code);

struct OvdResourceLimits
{
    uint64_t maxSingleAttributePayloadBytes;
    uint64_t maxCumulativeAttributePayloadBytes;
    uint64_t maxStreamBytes;
    uint64_t maxCommands;
    uint64_t maxAttributeMutations;
    uint64_t maxClasses;
    uint64_t maxAttributes;
    uint64_t maxAttributesPerClass;
    uint64_t maxObjectCreates;
    uint64_t maxMessages;
    uint64_t maxRetainedDomBytes;

    static OvdResourceLimits defaults();
};

// Grammar-aware stream gate placed in front of OmniPvdReader. It stages only
// command metadata, validates lengths/depth, and lets accepted attribute data
// pass through directly. Rejected payload bytes are never requested by the
// reader, so OmniPvdReaderImpl cannot allocate for them.
class OvdValidatedReadStream : public OmniPvdReadStream
{
public:
    OvdValidatedReadStream(OmniPvdReadStream& inner, const OvdResourceLimits& limits,
        CatalogValidationMode mode = CatalogValidationMode::Ordinary, OvdCaptureValidationSink* catalog = nullptr);

    uint64_t OMNI_PVD_CALL readBytes(uint8_t* bytes, uint64_t nbrBytes) override;
    uint64_t OMNI_PVD_CALL skipBytes(uint64_t nbrBytes) override;
    bool OMNI_PVD_CALL openStream() override;
    bool OMNI_PVD_CALL closeStream() override;

    // Must be called after each non-invalid getNextCommand and before applying
    // that command to the DOM.
    bool validateCommand(OmniPvdCommand::Enum command, OmniPvdReader& reader);

    bool failed() const { return mError != OvdInputErrorCode::None; }
    bool cleanEof() const { return mCleanEof && !failed(); }
    OvdInputErrorCode errorCode() const { return mError; }
    uint64_t streamBytes() const { return mStreamBytes; }
    uint64_t retainedDomBytes() const { return mRetainedDomBytes; }
    OvdCommandDestination commandDestination() const { return mDestination; }
    const OvdValidatedCaptureCommand& captureCommand() const { return mCaptureCommand; }
    OvdResourceUsage usage() const; // ingest-thread snapshot, not a concurrent view
    // True only when the most recently accepted command mutates an attribute
    // whose registered payload can drive pvddom parent/reference hooks.
    bool mayMutateTopology() const { return mCurrentCommandMayMutateTopology; }

private:
    enum class AttributeKind : uint8_t
    {
        Scalar,
        UniqueList,
        ClassLink,
        EnumValue
    };

    struct ClassAccounting
    {
        uint32_t baseClass;
        std::string name;
        uint64_t inheritanceDepth;
        uint64_t ownAttributes;
        std::vector<uint64_t> attributeRegistrationSequences;
        uint8_t captureFields = 0;
        uint16_t metadataFields = 0;
    };
    struct AttributeAccounting
    {
        uint32_t ownerClass;
        std::string name;
        OmniPvdDataType::Enum dataType;
        uint32_t elementCount;
        AttributeKind kind;
        uint64_t classOffset;
        uint64_t registrationSequence;
        bool createsReferences;
        int8_t captureField = -1;
        int8_t metadataField = -1;
    };
    struct ObjectAccounting
    {
        uint64_t generation;
        uint64_t attributeSequenceAtCreation;
        uint32_t classHandle;
        uint64_t context;
        bool alive;
    };

    struct AttributeNodeKey
    {
        uint64_t owner;
        uint32_t attribute;
        bool operator==(const AttributeNodeKey& other) const
        { return owner == other.owner && attribute == other.attribute; }
    };
    struct ReferenceNodeKey
    {
        uint64_t owner;
        uint64_t referenced;
        uint32_t attribute;
        bool operator==(const ReferenceNodeKey& other) const
        { return owner == other.owner && referenced == other.referenced && attribute == other.attribute; }
    };
    struct ObjectClassKey
    {
        uint64_t objectGeneration;
        uint32_t ownerClass;
        bool operator==(const ObjectClassKey& other) const
        { return objectGeneration == other.objectGeneration && ownerClass == other.ownerClass; }
    };
    struct AttributeNodeKeyHash
    {
        size_t operator()(const AttributeNodeKey& key) const;
    };
    struct ReferenceNodeKeyHash
    {
        size_t operator()(const ReferenceNodeKey& key) const;
    };
    struct ObjectClassKeyHash
    {
        size_t operator()(const ObjectClassKey& key) const;
    };

    bool stageHeader();
    bool stageNextCommand();
    bool stageString();
    bool stageBytes(uint64_t count);
    bool readInner(uint8_t* dest, uint64_t count, bool eofIsClean = false);
    bool setError(OvdInputErrorCode error);
    bool consume(uint64_t& current, uint64_t amount, uint64_t limit, OvdInputErrorCode error);
    bool chargeRetained(uint64_t amount);
    bool validateCaptureCommand(OmniPvdCommand::Enum command, OmniPvdReader& reader, bool& handled);
    bool acceptCapture(const OvdValidatedCaptureCommand& command);
    bool validateDecodedCommand(OmniPvdCommand::Enum command, OmniPvdReader& reader);
    bool observeMetadataCommand(OmniPvdCommand::Enum command, OmniPvdReader& reader);
    uint64_t inheritedAttributeSlots(uint32_t classHandle) const;
    bool validateAttributePayload(OmniPvdCommand::Enum command, OmniPvdReader& reader,
                                  const AttributeAccounting& attribute);
    bool validatePhysxParserContract(OmniPvdCommand::Enum command, OmniPvdReader& reader,
                                     const AttributeAccounting& attribute) const;

    OmniPvdReadStream& mInner;
    OvdResourceLimits mLimits;
    OvdCaptureValidationSink* mCatalog;
    OvdCommandDestination mDestination = OvdCommandDestination::Dom;
    OvdValidatedCaptureCommand mCaptureCommand{OvdValidatedCaptureCommand::Create};
    OvdInputErrorCode mError;
    bool mCleanEof;
    bool mHeaderStaged;
    uint32_t mMajorVersion;
    uint32_t mMinorVersion;
    std::vector<uint8_t> mStaged;
    size_t mStagedOffset;
    uint64_t mDirectPayloadRemaining;
    OmniPvdCommand::Enum mStagedCommand;
    uint32_t mStagedPayloadLength;
    uint8_t mStagedAttributeDepth = 0;
    bool mCurrentCommandMayMutateTopology;

    uint64_t mStreamBytes;
    uint64_t mCommands;
    uint64_t mAttributeMutations;
    uint64_t mCumulativeAttributePayloadBytes;
    uint64_t mClasses;
    uint64_t mAttributes;
    uint64_t mObjectCreates;
    uint64_t mMessages;
    uint64_t mRetainedDomBytes;
    uint64_t mNextObjectGeneration;
    std::unordered_map<uint32_t, ClassAccounting> mClassAccounting;
    std::unordered_map<uint32_t, AttributeAccounting> mAttributeAccounting;
    std::unordered_map<uint64_t, ObjectAccounting> mObjectAccounting;
    std::unordered_map<ObjectClassKey, uint64_t, ObjectClassKeyHash> mObjectClassAllocatedSlots;
    std::unordered_set<AttributeNodeKey, AttributeNodeKeyHash> mSyntheticAttributeNodes;
    std::unordered_set<ReferenceNodeKey, ReferenceNodeKeyHash> mSyntheticReferenceNodes;
};

#endif // PVD_OVD_SUPPORT
#endif // PVD_OVD_INPUT_VALIDATION_H
