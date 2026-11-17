/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifdef PVD_OVD_SUPPORT

#include "OvdInputValidation.h"

#include "OmniPvdDefines.h"
#include "OmniPvdReader.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace
{
const uint64_t kClassRetainedCharge = 1024;
// pvddom stores one inherited-class pointer and one hash-map entry for every
// ancestor of every registered class. Charging the full chain prevents a
// deep hierarchy from turning a linear class budget into quadratic storage.
const uint64_t kInheritedClassRetainedCharge = 128;
const uint64_t kAttributeRetainedCharge = 256;
const uint64_t kObjectRetainedCharge = 2048;
const uint64_t kInheritedAttributeSlotCharge = 64;
const uint64_t kMutationRetainedCharge = 128;
// Each object-handle element can create an auxiliary reference node in the
// DOM. Account its pointers/container linkage in addition to the separately
// charged payload/sample overhead.
const uint64_t kSyntheticNodeRetainedCharge = 512;
const uint64_t kMessageRetainedCharge = 8192;
const uint16_t kMaxRuntimeStringLength = 2047;

enum class PhysxParserHookClass
{
    None,
    Actor,
    DeformableVolume,
    DeformableSurface,
    ArticulationJoint,
    Geometry,
    Joint,
    Physics,
    Scene,
    Shape
};

PhysxParserHookClass classifyPhysxParserHookClass(const std::string& name)
{
    if (name == "PxDeformableVolume") return PhysxParserHookClass::DeformableVolume;
    if (name == "PxDeformableSurface") return PhysxParserHookClass::DeformableSurface;
    if (name == "PxArticulationJointReducedCoordinate") return PhysxParserHookClass::ArticulationJoint;
    if (name == "PxConvexMeshGeometry" || name == "PxHeightFieldGeometry"
        || name == "PxTriangleMeshGeometry" || name == "PxTetrahedronMeshGeometry"
        || name == "PxConvexCoreGeometry") return PhysxParserHookClass::Geometry;
    if (name == "PxPhysics") return PhysxParserHookClass::Physics;
    if (name == "PxScene") return PhysxParserHookClass::Scene;
    if (name == "PxShape") return PhysxParserHookClass::Shape;
    if (name == "PxFixedJoint" || name == "PxPrismaticJoint"
        || name == "PxRevoluteJoint" || name == "PxSphericalJoint"
        || name == "PxDistanceJoint" || name == "PxGearJoint"
        || name == "PxRackAndPinionJoint" || name == "PxD6Joint")
        return PhysxParserHookClass::Joint;
    if (name == "PxActor" || name == "PxActorType" || name == "PxRigidActor" || name == "PxRigidStatic"
        || name == "PxRigidDynamic" || name == "PxArticulationLink"
        || name == "PxPBDParticleSystem" || name == "PxDeformableBody")
        return PhysxParserHookClass::Actor;
    return PhysxParserHookClass::None;
}

bool matchesPhysxParserAttribute(const std::string& name, const char* expected)
{
    // PvdDom sanitizes registered names, then compares hook names without case.
    // Its hooks are short ASCII names, so a longer name (including one truncated
    // to 511 characters by PvdDom) cannot match. Keep this matcher independent of
    // PvdDom: OvdDataOmniPvd also builds this validator with only PVDRuntime.
    if (name.size() != std::strlen(expected)) return false;
    const auto isAlpha = [](char value)
    {
        return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z');
    };
    const auto lowerAscii = [](char value)
    {
        return value >= 'A' && value <= 'Z' ? static_cast<char>(value + ('a' - 'A')) : value;
    };
    for (size_t i = 0; i < name.size(); ++i)
    {
        char value = name[i];
        if (i == 0 && !isAlpha(value)) value = 'a';
        else if (i != 0 && !isAlpha(value) && !(value >= '0' && value <= '9')) value = '_';
        if (lowerAscii(value) != lowerAscii(expected[i])) return false;
    }
    return true;
}

template <typename T>
T stagedValue(const std::vector<uint8_t>& bytes, size_t offset)
{
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

}

size_t OvdValidatedReadStream::AttributeNodeKeyHash::operator()(const AttributeNodeKey& key) const
{
    return std::hash<uint64_t>()(key.owner) ^ (std::hash<uint32_t>()(key.attribute) << 1);
}

size_t OvdValidatedReadStream::ReferenceNodeKeyHash::operator()(const ReferenceNodeKey& key) const
{
    return std::hash<uint64_t>()(key.owner)
        ^ (std::hash<uint64_t>()(key.referenced) << 1)
        ^ (std::hash<uint32_t>()(key.attribute) << 2);
}

size_t OvdValidatedReadStream::ObjectClassKeyHash::operator()(const ObjectClassKey& key) const
{
    return std::hash<uint64_t>()(key.objectGeneration)
        ^ (std::hash<uint32_t>()(key.ownerClass) << 1);
}

OvdResourceLimits OvdResourceLimits::defaults()
{
    OvdResourceLimits limits{};
    limits.maxSingleAttributePayloadBytes = 16ull * 1024 * 1024;
    limits.maxCumulativeAttributePayloadBytes = 256ull * 1024 * 1024;
    limits.maxStreamBytes = 512ull * 1024 * 1024;
    limits.maxCommands = 2000000;
    limits.maxAttributeMutations = 2000000;
    limits.maxClasses = 4096;
    limits.maxAttributes = 16384;
    limits.maxAttributesPerClass = 4096;
    limits.maxObjectCreates = 100000;
    limits.maxMessages = 16384;
    limits.maxRetainedDomBytes = 512ull * 1024 * 1024;
    return limits;
}

const char* ovdInputErrorMessage(OvdInputErrorCode code)
{
    switch (code)
    {
    case OvdInputErrorCode::None: return "no error";
    case OvdInputErrorCode::TruncatedInput: return "truncated OVD input";
    case OvdInputErrorCode::UnknownCommand: return "unknown OVD command";
    case OvdInputErrorCode::InvalidStringLength: return "OVD string exceeds the runtime limit";
    case OvdInputErrorCode::InvalidAttributePayload: return "OVD attribute payload does not match its registered schema";
    case OvdInputErrorCode::AttributePayloadLimit: return "OVD attribute payload exceeds the per-command limit";
    case OvdInputErrorCode::AttributeHandleDepthLimit: return "OVD attribute handle depth exceeds the runtime stack";
    case OvdInputErrorCode::CumulativeAttributePayloadLimit: return "OVD cumulative attribute payload budget exhausted";
    case OvdInputErrorCode::StreamBytesLimit: return "OVD stream byte budget exhausted";
    case OvdInputErrorCode::CommandCountLimit: return "OVD command budget exhausted";
    case OvdInputErrorCode::AttributeMutationCountLimit: return "OVD attribute mutation budget exhausted";
    case OvdInputErrorCode::ClassCountLimit: return "OVD class budget exhausted";
    case OvdInputErrorCode::AttributeCountLimit: return "OVD attribute budget exhausted";
    case OvdInputErrorCode::AttributesPerClassLimit: return "OVD per-class attribute budget exhausted";
    case OvdInputErrorCode::ObjectCountLimit: return "OVD object creation budget exhausted";
    case OvdInputErrorCode::MessageCountLimit: return "OVD message budget exhausted";
    case OvdInputErrorCode::RetainedDomLimit: return "OVD retained DOM budget exhausted";
    case OvdInputErrorCode::ReaderRejectedCommand: return "OVD reader rejected a command";
    case OvdInputErrorCode::UnsupportedIntegrationVersion: return "unsupported OVD integration version";
    case OvdInputErrorCode::EmptyDom: return "OVD stream produced no objects";
    case OvdInputErrorCode::InvalidDomTopology: return "OVD object topology contains an invalid parent chain";
    case OvdInputErrorCode::InvalidCaptureCatalog: return "OVD capture catalog schema or lifetime is invalid";
    case OvdInputErrorCode::CaptureCatalogRowLimit: return "OVD capture catalog row budget exhausted";
    case OvdInputErrorCode::CaptureCatalogStringLimit: return "OVD capture catalog string budget exhausted";
    case OvdInputErrorCode::CaptureCatalogAllocationLimit: return "OVD capture catalog allocation budget exhausted";
    case OvdInputErrorCode::CaptureCatalogCommandLimit: return "OVD capture catalog command budget exhausted";
    case OvdInputErrorCode::CaptureCatalogMutationLimit: return "OVD capture catalog mutation budget exhausted";
    }
    return "unknown OVD validation error";
}

OvdValidatedReadStream::OvdValidatedReadStream(OmniPvdReadStream& inner, const OvdResourceLimits& limits,
    CatalogValidationMode mode, OvdCaptureValidationSink* catalog)
    : mInner(inner), mLimits(limits), mCatalog(mode == CatalogValidationMode::SeparateNonRetained ? catalog : nullptr)
    , mError(OvdInputErrorCode::None), mCleanEof(false)
    , mHeaderStaged(false), mMajorVersion(0), mMinorVersion(0), mStagedOffset(0)
    , mDirectPayloadRemaining(0), mStagedCommand(OmniPvdCommand::eINVALID), mStagedPayloadLength(0)
    , mCurrentCommandMayMutateTopology(false)
    , mStreamBytes(0), mCommands(0), mAttributeMutations(0)
    , mCumulativeAttributePayloadBytes(0), mClasses(0), mAttributes(0)
    , mObjectCreates(0), mMessages(0), mRetainedDomBytes(0), mNextObjectGeneration(0)
{
    if (mode == CatalogValidationMode::SeparateNonRetained && !catalog)
        setError(OvdInputErrorCode::InvalidCaptureCatalog);
}

OvdResourceUsage OvdValidatedReadStream::usage() const
{
    return {mStreamBytes, mCommands, mAttributeMutations, mCumulativeAttributePayloadBytes,
        mClasses, mAttributes, mObjectCreates, mMessages, mRetainedDomBytes};
}

bool OvdValidatedReadStream::setError(OvdInputErrorCode error)
{
    if (mError == OvdInputErrorCode::None)
        mError = error;
    mCleanEof = false;
    return false;
}

bool OvdValidatedReadStream::readInner(uint8_t* dest, uint64_t count, bool eofIsClean)
{
    const uint64_t got = mInner.readBytes(dest, count);
    if (got > std::numeric_limits<uint64_t>::max() - mStreamBytes)
        return setError(OvdInputErrorCode::StreamBytesLimit);
    mStreamBytes += got;
    if (mStreamBytes > mLimits.maxStreamBytes)
        return setError(OvdInputErrorCode::StreamBytesLimit);
    if (got != count)
    {
        if (got == 0 && eofIsClean)
        {
            mCleanEof = true;
            return false;
        }
        return setError(OvdInputErrorCode::TruncatedInput);
    }
    return true;
}

bool OvdValidatedReadStream::stageBytes(uint64_t count)
{
    const size_t oldSize = mStaged.size();
    if (count > std::numeric_limits<size_t>::max() - oldSize)
        return setError(OvdInputErrorCode::StreamBytesLimit);
    mStaged.resize(oldSize + static_cast<size_t>(count));
    if (!readInner(mStaged.data() + oldSize, count))
    {
        mStaged.resize(oldSize);
        return false;
    }
    return true;
}

bool OvdValidatedReadStream::stageString()
{
    const size_t lengthOffset = mStaged.size();
    if (!stageBytes(sizeof(uint16_t)))
        return false;
    const uint16_t length = stagedValue<uint16_t>(mStaged, lengthOffset);
    if (length > kMaxRuntimeStringLength)
        return setError(OvdInputErrorCode::InvalidStringLength);
    return stageBytes(length);
}

bool OvdValidatedReadStream::stageHeader()
{
    mStaged.clear();
    mStagedOffset = 0;
    if (!stageBytes(3 * sizeof(uint32_t)))
        return false;
    mMajorVersion = stagedValue<uint32_t>(mStaged, 0);
    mMinorVersion = stagedValue<uint32_t>(mStaged, sizeof(uint32_t));
    mHeaderStaged = true;
    return true;
}

bool OvdValidatedReadStream::stageNextCommand()
{
    mStaged.clear();
    mStagedOffset = 0;
    mStagedPayloadLength = 0;
    uint8_t commandByte = 0;
    if (!readInner(&commandByte, sizeof(commandByte), true))
        return false;
    mStaged.push_back(commandByte);
    mStagedCommand = static_cast<OmniPvdCommand::Enum>(commandByte);

    switch (mStagedCommand)
    {
    case OmniPvdCommand::eREGISTER_CLASS:
        if (!stageBytes(sizeof(uint32_t))) return false;
        if (!((mMajorVersion == 0) && (mMinorVersion < 3))
            && !stageBytes(sizeof(uint32_t))) return false;
        return stageString();
    case OmniPvdCommand::eREGISTER_ATTRIBUTE:
    {
        if (!stageBytes(sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint16_t))) return false;
        const uint16_t dataType = stagedValue<uint16_t>(mStaged, 1 + sizeof(uint32_t) + sizeof(uint32_t));
        if (dataType == OmniPvdDataType::eENUM_VALUE || dataType == OmniPvdDataType::eFLAGS_WORD)
        {
            if (!stageBytes(sizeof(uint32_t))) return false;
        }
        else if (!stageBytes(sizeof(uint32_t))) return false;
        return stageString();
    }
    case OmniPvdCommand::eREGISTER_CLASS_ATTRIBUTE:
        return stageBytes(3 * sizeof(uint32_t)) && stageString();
    case OmniPvdCommand::eREGISTER_UNIQUE_LIST_ATTRIBUTE:
        return stageBytes(2 * sizeof(uint32_t) + sizeof(uint16_t)) && stageString();
    case OmniPvdCommand::eSET_ATTRIBUTE:
    case OmniPvdCommand::eADD_TO_UNIQUE_LIST_ATTRIBUTE:
    case OmniPvdCommand::eREMOVE_FROM_UNIQUE_LIST_ATTRIBUTE:
    {
        if (!stageBytes(2 * sizeof(uint64_t) + sizeof(uint8_t))) return false;
        const uint8_t depth = mStaged.back();
        mStagedAttributeDepth = depth;
        if (depth > 32)
            return setError(OvdInputErrorCode::AttributeHandleDepthLimit);
        if (!stageBytes(static_cast<uint64_t>(depth) * sizeof(uint32_t))) return false;
        const size_t lengthOffset = mStaged.size();
        if (!stageBytes(sizeof(uint32_t))) return false;
        mStagedPayloadLength = stagedValue<uint32_t>(mStaged, lengthOffset);
        if (mStagedPayloadLength > mLimits.maxSingleAttributePayloadBytes)
            return setError(OvdInputErrorCode::AttributePayloadLimit);
        if (mStagedPayloadLength > mLimits.maxStreamBytes - std::min(mStreamBytes, mLimits.maxStreamBytes))
            return setError(OvdInputErrorCode::StreamBytesLimit);
        mDirectPayloadRemaining = mStagedPayloadLength;
        return true;
    }
    case OmniPvdCommand::eCREATE_OBJECT:
        return stageBytes(sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint64_t)) && stageString();
    case OmniPvdCommand::eDESTROY_OBJECT:
        return stageBytes(2 * sizeof(uint64_t));
    case OmniPvdCommand::eSTART_FRAME:
    case OmniPvdCommand::eSTOP_FRAME:
        return stageBytes(2 * sizeof(uint64_t));
    case OmniPvdCommand::eRECORD_MESSAGE:
        return stageBytes(sizeof(uint64_t)) && stageString() && stageString()
            && stageBytes(3 * sizeof(uint32_t));
    default:
        return setError(OvdInputErrorCode::UnknownCommand);
    }
}

uint64_t OvdValidatedReadStream::readBytes(uint8_t* bytes, uint64_t nbrBytes)
{
    if (failed() || nbrBytes == 0)
        return 0;
    uint64_t done = 0;
    while (done < nbrBytes && !failed())
    {
        if (mStagedOffset < mStaged.size())
        {
            const uint64_t copied = std::min<uint64_t>(nbrBytes - done, mStaged.size() - mStagedOffset);
            std::memcpy(bytes + done, mStaged.data() + mStagedOffset, static_cast<size_t>(copied));
            mStagedOffset += static_cast<size_t>(copied);
            done += copied;
            continue;
        }
        if (mDirectPayloadRemaining)
        {
            const uint64_t requested = std::min(nbrBytes - done, mDirectPayloadRemaining);
            if (!readInner(bytes + done, requested))
                break;
            mDirectPayloadRemaining -= requested;
            done += requested;
            continue;
        }
        if (!mHeaderStaged)
        {
            if (!stageHeader()) break;
        }
        else
        {
            if (!stageNextCommand()) break;
        }
    }
    return done;
}

uint64_t OvdValidatedReadStream::skipBytes(uint64_t nbrBytes)
{
    uint8_t scratch[4096];
    uint64_t done = 0;
    while (done < nbrBytes)
    {
        const uint64_t chunk = std::min<uint64_t>(nbrBytes - done, sizeof(scratch));
        const uint64_t got = readBytes(scratch, chunk);
        done += got;
        if (got != chunk) break;
    }
    return done;
}

bool OvdValidatedReadStream::openStream() { return mInner.openStream(); }
bool OvdValidatedReadStream::closeStream() { return mInner.closeStream(); }

bool OvdValidatedReadStream::consume(uint64_t& current, uint64_t amount, uint64_t limit,
                                     OvdInputErrorCode error)
{
    if (amount > limit || current > limit - amount)
        return setError(error);
    current += amount;
    return true;
}

bool OvdValidatedReadStream::chargeRetained(uint64_t amount)
{
    return consume(mRetainedDomBytes, amount, mLimits.maxRetainedDomBytes,
                   OvdInputErrorCode::RetainedDomLimit);
}

uint64_t OvdValidatedReadStream::inheritedAttributeSlots(uint32_t classHandle) const
{
    uint64_t total = 0;
    uint64_t hops = 0;
    while (classHandle && hops++ <= mLimits.maxClasses)
    {
        const auto found = mClassAccounting.find(classHandle);
        if (found == mClassAccounting.end()) break;
        if (found->second.ownAttributes > std::numeric_limits<uint64_t>::max() - total)
            return std::numeric_limits<uint64_t>::max();
        total += found->second.ownAttributes;
        classHandle = found->second.baseClass;
    }
    return total;
}

bool OvdValidatedReadStream::validatePhysxParserContract(
    OmniPvdCommand::Enum command, OmniPvdReader& reader,
    const AttributeAccounting& attribute) const
{
    const auto object = mObjectAccounting.find(reader.getObjectHandle());
    if (object == mObjectAccounting.end())
        return true; // pvddom ignores mutations for unknown objects
    const auto objectClass = mClassAccounting.find(object->second.classHandle);
    if (objectClass == mClassAccounting.end())
        return true;

    const PhysxParserHookClass hookClass =
        classifyPhysxParserHookClass(objectClass->second.name);
    const uint32_t payloadLength = reader.getAttributeDataLength();
    const auto requireScalar = [&](OmniPvdDataType::Enum dataType, uint32_t bytes)
    {
        return attribute.kind == AttributeKind::Scalar
            && attribute.dataType == dataType
            && command == OmniPvdCommand::eSET_ATTRIBUTE
            && payloadLength == bytes;
    };
    const auto requireHandleList = [&]()
    {
        return attribute.kind == AttributeKind::UniqueList
            && attribute.dataType == OmniPvdDataType::eOBJECT_HANDLE
            && (command == OmniPvdCommand::eADD_TO_UNIQUE_LIST_ATTRIBUTE
                || command == OmniPvdCommand::eREMOVE_FROM_UNIQUE_LIST_ATTRIBUTE)
            && payloadLength == sizeof(OmniPvdObjectHandle);
    };
    const auto requireHandleScalar = [&]()
    {
        return requireScalar(OmniPvdDataType::eOBJECT_HANDLE,
                             sizeof(OmniPvdObjectHandle));
    };

    const auto nameMatches = [&](const char* expected)
    {
        return matchesPhysxParserAttribute(attribute.name, expected);
    };
    const std::string& className = objectClass->second.name;
    const auto requireActorType = [&]()
    {
        return attribute.kind == AttributeKind::Scalar
            && (attribute.dataType == OmniPvdDataType::eFLAGS_WORD
                || attribute.dataType == OmniPvdDataType::eUINT32)
            && command == OmniPvdCommand::eSET_ATTRIBUTE
            && payloadLength == sizeof(uint32_t);
    };

    // pvddom later reads these retained fields as uint32_t when the next
    // stream object is created, independent of the declaring class.
    if (nameMatches("ovdIntegrationVersionMajor") || nameMatches("ovdIntegrationVersionMinor"))
        return requireScalar(OmniPvdDataType::eUINT32, sizeof(uint32_t));

    // An actor's inboundJoint hook reads this retained field from its referenced
    // object without requiring that object to have an articulation-joint class.
    if (nameMatches("parentLink")) return requireHandleScalar();

    switch (hookClass)
    {
    case PhysxParserHookClass::DeformableVolume:
    case PhysxParserHookClass::DeformableSurface:
        if (nameMatches("type")) return requireActorType();
        if (nameMatches("shapes")) return requireHandleList();
        if (hookClass == PhysxParserHookClass::DeformableVolume
            && nameMatches("simulationMeshShapes")) return requireHandleList();
        break;
    case PhysxParserHookClass::ArticulationJoint:
        if (nameMatches("childLink")) return requireHandleScalar();
        break;
    case PhysxParserHookClass::Geometry:
        if ((className == "PxConvexMeshGeometry" && nameMatches("convexmesh"))
            || (className == "PxHeightFieldGeometry" && nameMatches("heightfield"))
            || (className == "PxTriangleMeshGeometry" && nameMatches("trianglemesh"))
            || (className == "PxTetrahedronMeshGeometry" && nameMatches("tetrahedronMesh"))
            || (className == "PxConvexCoreGeometry" && nameMatches("core")))
            return requireHandleScalar();
        break;
    case PhysxParserHookClass::Joint:
        if (nameMatches("constraint")) return requireHandleScalar();
        break;
    case PhysxParserHookClass::Physics:
        if (nameMatches("scenes")) return requireHandleList();
        break;
    case PhysxParserHookClass::Scene:
        if (nameMatches("actors") || nameMatches("articulations") || nameMatches("constraints"))
            return requireHandleList();
        break;
    case PhysxParserHookClass::Actor:
        if (nameMatches("type")) return requireActorType();
        if (nameMatches("shapes") || nameMatches("particleBuffers")) return requireHandleList();
        if (nameMatches("articulation") || nameMatches("inboundJoint")) return requireHandleScalar();
        break;
    case PhysxParserHookClass::Shape:
        if (nameMatches("isExclusive")) return requireScalar(OmniPvdDataType::eUINT8, 1);
        if (nameMatches("geom")) return requireHandleScalar();
        break;
    case PhysxParserHookClass::None:
        break;
    }
    return true;
}

bool OvdValidatedReadStream::validateAttributePayload(
    OmniPvdCommand::Enum command, OmniPvdReader& reader,
    const AttributeAccounting& attribute)
{
    const uint64_t payloadLength = reader.getAttributeDataLength();
    const auto reject = [&]()
    {
        return setError(OvdInputErrorCode::InvalidAttributePayload);
    };
    switch (attribute.kind)
    {
    case AttributeKind::ClassLink:
    case AttributeKind::EnumValue:
        return reject();
    case AttributeKind::Scalar:
        if (command != OmniPvdCommand::eSET_ATTRIBUTE)
            return reject();
        break;
    case AttributeKind::UniqueList:
        if (command != OmniPvdCommand::eADD_TO_UNIQUE_LIST_ATTRIBUTE
            && command != OmniPvdCommand::eREMOVE_FROM_UNIQUE_LIST_ATTRIBUTE)
            return reject();
        break;
    }

    if (attribute.dataType == OmniPvdDataType::eFLAGS_WORD)
    {
        // FLAGS_WORD carries the enum-class handle in the registration, not
        // the PxFlags storage width. Valid producers use 1-, 2-, or 4-byte
        // flag words (and an 8-byte storage type remains representable).
        if (payloadLength != 1 && payloadLength != 2
            && payloadLength != 4 && payloadLength != 8)
            return reject();
    }
    else if (attribute.dataType != OmniPvdDataType::eSTRING)
    {
        const uint64_t elementSize = getOmniPvdDataTypeSizeFromEnum(attribute.dataType);
        if (elementSize == 0)
            return reject();
        if (attribute.kind == AttributeKind::UniqueList)
        {
            if (payloadLength != elementSize)
                return reject();
        }
        else if (attribute.elementCount != 0)
        {
            if (attribute.elementCount > std::numeric_limits<uint64_t>::max() / elementSize
                || payloadLength != elementSize * attribute.elementCount)
                return reject();
        }
        else if ((payloadLength % elementSize) != 0)
        {
            return reject();
        }
    }

    if (!validatePhysxParserContract(command, reader, attribute))
        return reject();
    return true;
}

bool OvdValidatedReadStream::acceptCapture(const OvdValidatedCaptureCommand& command)
{
    mCaptureCommand = command;
    const auto error = mCatalog->observeValidatedCapture(command);
    if (error != OvdInputErrorCode::None) return setError(error);
    mDestination = OvdCommandDestination::Catalog;
    return true;
}

bool OvdValidatedReadStream::validateCaptureCommand(OmniPvdCommand::Enum command, OmniPvdReader& reader, bool& handled)
{
    handled = false;
    if (!mCatalog) return true;
    const auto invalid = [&] { return setError(OvdInputErrorCode::InvalidCaptureCatalog); };
    const uint64_t object = reader.getObjectHandle();
    const uint32_t rowSchema = mCatalog->catalogRowSchema(object);
    if (command == OmniPvdCommand::eCREATE_OBJECT)
    {
        if (rowSchema) return invalid();
        const auto cls = mClassAccounting.find(reader.getClassHandle());
        if (cls == mClassAccounting.end() || cls->second.name != "PxOmniPvdCaptureEntry") return true;
        const auto existing = mObjectAccounting.find(object);
        if (reader.getContextHandle() != 2 || cls->second.baseClass || cls->second.captureFields != 63 ||
            (existing != mObjectAccounting.end() && existing->second.alive)) return invalid();
        handled = true;
        return acceptCapture({OvdValidatedCaptureCommand::Create, object, reader.getClassHandle()});
    }
    if (command == OmniPvdCommand::eDESTROY_OBJECT && rowSchema)
    {
        if (reader.getContextHandle() != 2) return invalid();
        handled = true;
        return acceptCapture({OvdValidatedCaptureCommand::Destroy, object, rowSchema});
    }
    if (command != OmniPvdCommand::eSET_ATTRIBUTE && command != OmniPvdCommand::eADD_TO_UNIQUE_LIST_ATTRIBUTE &&
        command != OmniPvdCommand::eREMOVE_FROM_UNIQUE_LIST_ATTRIBUTE) return true;
    const auto attr = mAttributeAccounting.find(reader.getAttributeHandle());
    if (!rowSchema && (attr == mAttributeAccounting.end() || attr->second.captureField < 0)) return true;
    if (!rowSchema || reader.getContextHandle() != 2 || attr == mAttributeAccounting.end() ||
        attr->second.ownerClass != rowSchema || attr->second.captureField < 0 ||
        command != OmniPvdCommand::eSET_ATTRIBUTE || mStagedAttributeDepth != 1) return invalid();
    if (!validateAttributePayload(command, reader, attr->second)) return false;
    if (!consume(mCumulativeAttributePayloadBytes, reader.getAttributeDataLength(),
        mLimits.maxCumulativeAttributePayloadBytes, OvdInputErrorCode::CumulativeAttributePayloadLimit)) return false;
    OvdValidatedCaptureCommand view{OvdValidatedCaptureCommand::Field, object, rowSchema,
        static_cast<uint8_t>(attr->second.captureField)};
    if (attr->second.dataType == OmniPvdDataType::eSTRING)
    {
        view.text = reinterpret_cast<const char*>(reader.getAttributeDataPointer());
        view.textSize = reader.getAttributeDataLength();
    }
    else std::memcpy(&view.value, reader.getAttributeDataPointer(), reader.getAttributeDataLength());
    handled = true;
    return acceptCapture(view);
}

bool OvdValidatedReadStream::validateCommand(OmniPvdCommand::Enum command, OmniPvdReader& reader)
{
    return validateDecodedCommand(command, reader) && observeMetadataCommand(command, reader);
}

bool OvdValidatedReadStream::observeMetadataCommand(OmniPvdCommand::Enum command, OmniPvdReader& reader)
{
    if (!mCatalog || mDestination == OvdCommandDestination::Catalog || reader.getContextHandle() != 1) return true;
    const auto object = mObjectAccounting.find(reader.getObjectHandle());
    if (object == mObjectAccounting.end() || object->second.context != 1) return true;
    const auto cls = mClassAccounting.find(object->second.classHandle);
    if (cls == mClassAccounting.end() || cls->second.name != "PxOmniPvdMetaData" || cls->second.baseClass ||
        !(cls->second.metadataFields & (1 << 3))) return true;
    OvdValidatedCaptureCommand view{OvdValidatedCaptureCommand::MetadataCreate, reader.getObjectHandle(), object->second.classHandle};
    if (command == OmniPvdCommand::eDESTROY_OBJECT) view.kind = OvdValidatedCaptureCommand::MetadataDestroy;
    else if (command == OmniPvdCommand::eSET_ATTRIBUTE && object->second.alive)
    {
        const auto attr = mAttributeAccounting.find(reader.getAttributeHandle());
        if (attr == mAttributeAccounting.end() || attr->second.ownerClass != object->second.classHandle ||
            attr->second.metadataField < 0 || mStagedAttributeDepth != 1) return true;
        if (attr->second.metadataField >= 5 && (cls->second.metadataFields != 2047 || cls->second.ownAttributes != 11)) return true;
        view.kind = OvdValidatedCaptureCommand::MetadataField;
        view.field = static_cast<uint8_t>(attr->second.metadataField);
        std::memcpy(&view.value, reader.getAttributeDataPointer(), reader.getAttributeDataLength());
    }
    else if (command != OmniPvdCommand::eCREATE_OBJECT) return true;
    const auto error = mCatalog->observeValidatedCapture(view);
    return error == OvdInputErrorCode::None || setError(error);
}

bool OvdValidatedReadStream::validateDecodedCommand(OmniPvdCommand::Enum command, OmniPvdReader& reader)
{
    mCurrentCommandMayMutateTopology = false;
    mDestination = OvdCommandDestination::Dom;
    if (failed()) return false;
    if (command != mStagedCommand)
        return setError(OvdInputErrorCode::ReaderRejectedCommand);
    const bool hasReservedHandle =
        (command == OmniPvdCommand::eREGISTER_CLASS
            && reader.getClassHandle() == OMNI_PVD_INVALID_HANDLE)
        || ((command == OmniPvdCommand::eREGISTER_ATTRIBUTE
                || command == OmniPvdCommand::eREGISTER_CLASS_ATTRIBUTE
                || command == OmniPvdCommand::eREGISTER_UNIQUE_LIST_ATTRIBUTE)
            && reader.getAttributeHandle() == OMNI_PVD_INVALID_HANDLE)
        || (command == OmniPvdCommand::eREGISTER_CLASS_ATTRIBUTE
            && reader.getAttributeClassHandle() == OMNI_PVD_INVALID_HANDLE)
        || (command == OmniPvdCommand::eREGISTER_ATTRIBUTE
            && reader.getAttributeDataType() == OmniPvdDataType::eFLAGS_WORD
            && reader.getEnumClassHandle() == OMNI_PVD_INVALID_HANDLE)
        || (command == OmniPvdCommand::eCREATE_OBJECT
            && (reader.getClassHandle() == OMNI_PVD_INVALID_HANDLE
                || reader.getObjectHandle() == OMNI_PVD_INVALID_HANDLE));
    if (hasReservedHandle)
        return setError(OvdInputErrorCode::UnknownCommand);
    bool captureHandled = false;
    if (!validateCaptureCommand(command, reader, captureHandled)) return false;
    if (captureHandled) return true;
    if (!consume(mCommands, 1, mLimits.maxCommands, OvdInputErrorCode::CommandCountLimit))
        return false;

    switch (command)
    {
    case OmniPvdCommand::eREGISTER_CLASS:
    {
        if (!consume(mClasses, 1, mLimits.maxClasses, OvdInputErrorCode::ClassCountLimit)) return false;
        const uint32_t handle = reader.getClassHandle();
        const uint32_t baseHandle = reader.getBaseClassHandle();
        const std::string className = reader.getClassName() ? reader.getClassName() : "";
        if (mCatalog && className == "PxOmniPvdCaptureEntry" && baseHandle)
            return setError(OvdInputErrorCode::InvalidCaptureCatalog);
        const auto duplicate = mClassAccounting.find(handle);
        if (duplicate != mClassAccounting.end())
        {
            if (duplicate->second.baseClass != baseHandle || duplicate->second.name != className)
                return setError(OvdInputErrorCode::UnknownCommand);
            return chargeRetained(kClassRetainedCharge + className.size() + 1);
        }

        uint64_t inheritanceDepth = 1;
        if (baseHandle != 0)
        {
            const auto base = mClassAccounting.find(baseHandle);
            if (base == mClassAccounting.end()
                || base->second.inheritanceDepth >= mLimits.maxClasses)
                return setError(OvdInputErrorCode::UnknownCommand);
            inheritanceDepth = base->second.inheritanceDepth + 1;
        }
        if (inheritanceDepth > std::numeric_limits<uint64_t>::max()
                                   / kInheritedClassRetainedCharge)
            return setError(OvdInputErrorCode::RetainedDomLimit);
        const uint64_t inheritanceCharge =
            inheritanceDepth * kInheritedClassRetainedCharge;
        const uint64_t baseCharge = kClassRetainedCharge + className.size() + 1;
        if (baseCharge > std::numeric_limits<uint64_t>::max() - inheritanceCharge)
            return setError(OvdInputErrorCode::RetainedDomLimit);
        // A malformed duplicate handle must not erase attributes/base already
        // charged for this class and make later object-slot estimates smaller.
        mClassAccounting.emplace(handle, ClassAccounting{
            baseHandle, className, inheritanceDepth, 0, {}});
        return chargeRetained(baseCharge + inheritanceCharge);
    }
    case OmniPvdCommand::eREGISTER_ATTRIBUTE:
    case OmniPvdCommand::eREGISTER_CLASS_ATTRIBUTE:
    case OmniPvdCommand::eREGISTER_UNIQUE_LIST_ATTRIBUTE:
    {
        const auto classAccounting = mClassAccounting.find(reader.getClassHandle());
        if (classAccounting == mClassAccounting.end())
            return setError(OvdInputErrorCode::UnknownCommand);
        if (command == OmniPvdCommand::eREGISTER_CLASS_ATTRIBUTE
            && mClassAccounting.find(reader.getAttributeClassHandle()) == mClassAccounting.end())
            return setError(OvdInputErrorCode::UnknownCommand);
        if (command == OmniPvdCommand::eREGISTER_ATTRIBUTE
            && reader.getAttributeDataType() == OmniPvdDataType::eFLAGS_WORD
            && mClassAccounting.find(reader.getEnumClassHandle()) == mClassAccounting.end())
            return setError(OvdInputErrorCode::UnknownCommand);
        const uint32_t attributeHandle = reader.getAttributeHandle();
        if (mAttributeAccounting.find(attributeHandle) != mAttributeAccounting.end())
            return setError(OvdInputErrorCode::UnknownCommand);
        if (!consume(mAttributes, 1, mLimits.maxAttributes, OvdInputErrorCode::AttributeCountLimit)) return false;
        ClassAccounting& accounting = classAccounting->second;
        if (!consume(accounting.ownAttributes, 1, mLimits.maxAttributesPerClass,
                     OvdInputErrorCode::AttributesPerClassLimit)) return false;
        OmniPvdDataType::Enum dataType = OmniPvdDataType::eOBJECT_HANDLE;
        uint32_t elementCount = 0;
        AttributeKind kind = AttributeKind::ClassLink;
        if (command != OmniPvdCommand::eREGISTER_CLASS_ATTRIBUTE)
        {
            dataType = reader.getAttributeDataType();
            if (static_cast<uint32_t>(dataType)
                > static_cast<uint32_t>(OmniPvdDataType::eFLAGS_WORD))
                return setError(OvdInputErrorCode::InvalidAttributePayload);
            if (command == OmniPvdCommand::eREGISTER_UNIQUE_LIST_ATTRIBUTE)
            {
                kind = AttributeKind::UniqueList;
                elementCount = 1;
            }
            else if (dataType == OmniPvdDataType::eENUM_VALUE)
            {
                kind = AttributeKind::EnumValue;
                elementCount = 1;
            }
            else
            {
                kind = AttributeKind::Scalar;
                elementCount = (dataType == OmniPvdDataType::eFLAGS_WORD)
                    ? 1 : reader.getAttributeNumberElements();
            }
        }
        const bool createsReferences = kind == AttributeKind::ClassLink
            || dataType == OmniPvdDataType::eOBJECT_HANDLE;
        int8_t captureField = -1;
        int8_t metadataField = -1;
        if (mCatalog && accounting.name == "PxOmniPvdMetaData" && !accounting.baseClass)
        {
            const char* names[] = {"physxVersionMajor", "physxVersionMinor", "physxVersionBugfix", "ovdIntegrationVersionMajor",
                "ovdIntegrationVersionMinor", "recordSecondaryData", "captureAllObjects", "captureRequestId",
                "captureRequestStatus", "captureEpoch", "captureCatalogGeneration"};
            for (int8_t i = 0; i < 11; ++i)
                if (std::strcmp(reader.getAttributeName(), names[i]) == 0) metadataField = i;
            if (metadataField >= 0)
            {
                const auto expected = metadataField == 5 || metadataField == 6 ? OmniPvdDataType::eUINT8 :
                    (metadataField == 7 || metadataField >= 9 ? OmniPvdDataType::eUINT64 : OmniPvdDataType::eUINT32);
                if (kind != AttributeKind::Scalar || dataType != expected || elementCount != 1 ||
                    (accounting.metadataFields & (1 << metadataField)))
                    return setError(OvdInputErrorCode::InvalidAttributePayload);
                accounting.metadataFields |= static_cast<uint16_t>(1 << metadataField);
            }
        }
        if (mCatalog && accounting.name == "PxOmniPvdCaptureEntry")
        {
            const char* names[] = {"targetHandle", "typeName", "name", "ownerHandle", "sceneHandle", "canSelect"};
            for (int8_t i = 0; i < 6; ++i)
                if (std::strcmp(reader.getAttributeName(), names[i]) == 0) captureField = i;
            if (captureField < 0 || kind != AttributeKind::Scalar || (accounting.captureFields & (1 << captureField)))
                return setError(OvdInputErrorCode::InvalidCaptureCatalog);
            const bool string = captureField == 1 || captureField == 2;
            const auto expectedType = string ? OmniPvdDataType::eSTRING :
                (captureField == 5 ? OmniPvdDataType::eUINT8 : OmniPvdDataType::eUINT64);
            if (dataType != expectedType || elementCount != 1)
                return setError(OvdInputErrorCode::InvalidCaptureCatalog);
            accounting.captureFields |= static_cast<uint8_t>(1 << captureField);
        }
        const uint64_t classOffset = accounting.ownAttributes - 1;
        accounting.attributeRegistrationSequences.push_back(mAttributes);
        mAttributeAccounting.emplace(attributeHandle, AttributeAccounting{
            reader.getClassHandle(), reader.getAttributeName() ? reader.getAttributeName() : "",
            dataType, elementCount, kind, classOffset, mAttributes, createsReferences, captureField, metadataField});
        return chargeRetained(kAttributeRetainedCharge + std::strlen(reader.getAttributeName()) + 1);
    }
    case OmniPvdCommand::eSET_ATTRIBUTE:
    case OmniPvdCommand::eADD_TO_UNIQUE_LIST_ATTRIBUTE:
    case OmniPvdCommand::eREMOVE_FROM_UNIQUE_LIST_ATTRIBUTE:
    {
        const auto attribute = mAttributeAccounting.find(reader.getAttributeHandle());
        if (attribute == mAttributeAccounting.end())
            return setError(OvdInputErrorCode::UnknownCommand);
        if (!validateAttributePayload(command, reader, attribute->second))
            return false;
        if (!consume(mAttributeMutations, 1, mLimits.maxAttributeMutations,
                     OvdInputErrorCode::AttributeMutationCountLimit)) return false;
        if (!consume(mCumulativeAttributePayloadBytes, reader.getAttributeDataLength(),
                     mLimits.maxCumulativeAttributePayloadBytes,
                     OvdInputErrorCode::CumulativeAttributePayloadLimit)) return false;
        uint64_t retainedCharge = kMutationRetainedCharge + reader.getAttributeDataLength();
        const auto ownerObject = mObjectAccounting.find(reader.getObjectHandle());
        if (ownerObject != mObjectAccounting.end())
        {
            const auto ownerClass = mClassAccounting.find(attribute->second.ownerClass);
            uint64_t initiallyAllocated = 0;
            if (ownerClass != mClassAccounting.end())
            {
                const std::vector<uint64_t>& registrations =
                    ownerClass->second.attributeRegistrationSequences;
                initiallyAllocated = static_cast<uint64_t>(std::upper_bound(
                    registrations.begin(), registrations.end(),
                    ownerObject->second.attributeSequenceAtCreation) - registrations.begin());
            }
            const ObjectClassKey slotKey{
                ownerObject->second.generation, attribute->second.ownerClass};
            auto slotState = mObjectClassAllocatedSlots.emplace(slotKey, initiallyAllocated).first;
            const uint64_t requiredSlots = attribute->second.classOffset + 1;
            if (requiredSlots > slotState->second)
            {
                const uint64_t addedSlots = requiredSlots - slotState->second;
                if (addedSlots > (std::numeric_limits<uint64_t>::max() - retainedCharge)
                    / kInheritedAttributeSlotCharge)
                    return setError(OvdInputErrorCode::RetainedDomLimit);
                retainedCharge += addedSlots * kInheritedAttributeSlotCharge;
                slotState->second = requiredSlots;
            }
        }
        if (attribute->second.createsReferences
            && command != OmniPvdCommand::eREMOVE_FROM_UNIQUE_LIST_ATTRIBUTE)
        {
            const uint64_t owner = reader.getObjectHandle();
            const uint32_t attributeHandle = reader.getAttributeHandle();
            const uint8_t* data = reader.getAttributeDataPointer();
            const uint64_t references = reader.getAttributeDataLength() / sizeof(OmniPvdObjectHandle);
            const auto ownerGeneration = mObjectAccounting.find(owner);
            for (uint64_t i = 0; i < references; ++i)
            {
                uint64_t referenced = 0;
                std::memcpy(&referenced, data + i * sizeof(referenced), sizeof(referenced));
                const auto referencedGeneration = mObjectAccounting.find(referenced);
                if (ownerGeneration == mObjectAccounting.end()
                    || referencedGeneration == mObjectAccounting.end())
                    continue;
                if (mSyntheticAttributeNodes.emplace(
                    AttributeNodeKey{ownerGeneration->second.generation, attributeHandle}).second)
                {
                    if (retainedCharge > std::numeric_limits<uint64_t>::max() - kSyntheticNodeRetainedCharge)
                        return setError(OvdInputErrorCode::RetainedDomLimit);
                    retainedCharge += kSyntheticNodeRetainedCharge;
                }
                if (mSyntheticReferenceNodes.emplace(ReferenceNodeKey{
                    ownerGeneration->second.generation,
                    referencedGeneration->second.generation, attributeHandle}).second)
                {
                    if (retainedCharge > std::numeric_limits<uint64_t>::max() - kSyntheticNodeRetainedCharge)
                        return setError(OvdInputErrorCode::RetainedDomLimit);
                    retainedCharge += kSyntheticNodeRetainedCharge;
                }
            }
        }
        if (!chargeRetained(retainedCharge))
            return false;
        mCurrentCommandMayMutateTopology = attribute->second.createsReferences;
        return true;
    }
    case OmniPvdCommand::eCREATE_OBJECT:
    {
        if (!consume(mObjectCreates, 1, mLimits.maxObjectCreates, OvdInputErrorCode::ObjectCountLimit)) return false;
        if (mNextObjectGeneration == std::numeric_limits<uint64_t>::max())
            return setError(OvdInputErrorCode::ObjectCountLimit);
        mObjectAccounting[reader.getObjectHandle()] = ObjectAccounting{
            ++mNextObjectGeneration, mAttributes, reader.getClassHandle(), reader.getContextHandle(), true};
        const uint64_t slots = inheritedAttributeSlots(reader.getClassHandle());
        if (slots > (std::numeric_limits<uint64_t>::max() - kObjectRetainedCharge) / kInheritedAttributeSlotCharge)
            return setError(OvdInputErrorCode::RetainedDomLimit);
        return chargeRetained(kObjectRetainedCharge + slots * kInheritedAttributeSlotCharge
                              + std::strlen(reader.getObjectName()) + 1);
    }
    case OmniPvdCommand::eDESTROY_OBJECT:
    {
        const auto found = mObjectAccounting.find(reader.getObjectHandle());
        if (found != mObjectAccounting.end() && found->second.context == reader.getContextHandle())
            found->second.alive = false;
        return true;
    }
    case OmniPvdCommand::eRECORD_MESSAGE:
        if (!consume(mMessages, 1, mLimits.maxMessages, OvdInputErrorCode::MessageCountLimit)) return false;
        return chargeRetained(kMessageRetainedCharge);
    default:
        return true;
    }
}

#endif // PVD_OVD_SUPPORT
