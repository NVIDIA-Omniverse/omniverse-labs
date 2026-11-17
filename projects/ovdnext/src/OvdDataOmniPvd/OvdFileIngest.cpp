/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "OvdFileIngest.h"

#include "OvdInputValidation.h"

#include "OmniPvdFileReadStream.h"
#include "OmniPvdLibraryFunctions.h"
#include "OmniPvdReader.h"

#include <cstring>
#include <map>
#include <set>
#include <utility>

namespace ovdnext
{
namespace omnipvd
{
namespace
{

const std::uint32_t kAcceptedIntegrationMajor = 4;

struct ReaderDeleter
{
    void operator()(OmniPvdReader* reader) const
    {
        if (reader)
            destroyOmniPvdReader(*reader);
    }
};

struct FileStreamDeleter
{
    void operator()(OmniPvdFileReadStream* stream) const
    {
        if (stream)
            destroyOmniPvdFileReadStream(*stream);
    }
};

OvdResourceLimits validationLimits(const FileIngestLimits& limits)
{
    OvdResourceLimits converted;
    converted.maxSingleAttributePayloadBytes = limits.maxSingleAttributePayloadBytes;
    converted.maxCumulativeAttributePayloadBytes = limits.maxCumulativeAttributePayloadBytes;
    converted.maxStreamBytes = limits.maxStreamBytes;
    converted.maxCommands = limits.maxCommands;
    converted.maxAttributeMutations = limits.maxAttributeMutations;
    converted.maxClasses = limits.maxClasses;
    converted.maxAttributes = limits.maxAttributes;
    converted.maxAttributesPerClass = limits.maxAttributesPerClass;
    converted.maxObjectCreates = limits.maxObjectCreates;
    converted.maxMessages = limits.maxMessages;
    converted.maxRetainedDomBytes = limits.maxRetainedValidationBytes;
    return converted;
}

FileIngestResult failure(FileIngestError error, const char* diagnostic)
{
    FileIngestResult result;
    result.error = error;
    result.diagnostic = diagnostic;
    return result;
}

FileIngestResult failure(FileIngestError error, const std::string& diagnostic)
{
    FileIngestResult result;
    result.error = error;
    result.diagnostic = diagnostic;
    return result;
}

FileIngestResult validationFailure(OvdInputErrorCode error)
{
    if (error == OvdInputErrorCode::TruncatedInput)
        return failure(FileIngestError::TruncatedInput, ovdInputErrorMessage(error));
    return failure(FileIngestError::ValidationFailed, ovdInputErrorMessage(error));
}

const char* commandName(OmniPvdCommand::Enum command)
{
    switch (command)
    {
    case OmniPvdCommand::eINVALID: return "eINVALID";
    case OmniPvdCommand::eREGISTER_CLASS: return "eREGISTER_CLASS";
    case OmniPvdCommand::eREGISTER_ENUM: return "eREGISTER_ENUM";
    case OmniPvdCommand::eREGISTER_ATTRIBUTE: return "eREGISTER_ATTRIBUTE";
    case OmniPvdCommand::eREGISTER_CLASS_ATTRIBUTE: return "eREGISTER_CLASS_ATTRIBUTE";
    case OmniPvdCommand::eREGISTER_UNIQUE_LIST_ATTRIBUTE:
        return "eREGISTER_UNIQUE_LIST_ATTRIBUTE";
    case OmniPvdCommand::eSET_ATTRIBUTE: return "eSET_ATTRIBUTE";
    case OmniPvdCommand::eADD_TO_UNIQUE_LIST_ATTRIBUTE:
        return "eADD_TO_UNIQUE_LIST_ATTRIBUTE";
    case OmniPvdCommand::eREMOVE_FROM_UNIQUE_LIST_ATTRIBUTE:
        return "eREMOVE_FROM_UNIQUE_LIST_ATTRIBUTE";
    case OmniPvdCommand::eCREATE_OBJECT: return "eCREATE_OBJECT";
    case OmniPvdCommand::eDESTROY_OBJECT: return "eDESTROY_OBJECT";
    case OmniPvdCommand::eSTART_FRAME: return "eSTART_FRAME";
    case OmniPvdCommand::eSTOP_FRAME: return "eSTOP_FRAME";
    case OmniPvdCommand::eRECORD_MESSAGE: return "eRECORD_MESSAGE";
    }
    return "eUNKNOWN";
}

const char* ingestErrorName(data::IngestError error)
{
    switch (error)
    {
    case data::IngestError::None: return "None";
    case data::IngestError::AlreadyFinished: return "AlreadyFinished";
    case data::IngestError::UnknownBaseClass: return "UnknownBaseClass";
    case data::IngestError::InheritanceCycle: return "InheritanceCycle";
    case data::IngestError::ConflictingClassRegistration:
        return "ConflictingClassRegistration";
    case data::IngestError::InvalidSourceHandle: return "InvalidSourceHandle";
    case data::IngestError::UnknownAttributeOwner: return "UnknownAttributeOwner";
    case data::IngestError::DuplicateAttributeRegistration:
        return "DuplicateAttributeRegistration";
    case data::IngestError::UnknownRelatedClass: return "UnknownRelatedClass";
    case data::IngestError::InvalidAttributeDeclaration:
        return "InvalidAttributeDeclaration";
    case data::IngestError::UnknownClass: return "UnknownClass";
    case data::IngestError::UnknownProperty: return "UnknownProperty";
    case data::IngestError::UnknownObject: return "UnknownObject";
    case data::IngestError::PropertyKindMismatch: return "PropertyKindMismatch";
    case data::IngestError::NestedStart: return "NestedStart";
    case data::IngestError::StopWithoutStart: return "StopWithoutStart";
    case data::IngestError::MismatchedStop: return "MismatchedStop";
    case data::IngestError::SegmentTransitionDuringFrame:
        return "SegmentTransitionDuringFrame";
    case data::IngestError::ActiveFrameAtEof: return "ActiveFrameAtEof";
    case data::IngestError::StoreFailure: return "StoreFailure";
    }
    return "UnknownIngestError";
}

bool mapDataType(OmniPvdDataType::Enum source,
                 data::RecordedDataType& destination)
{
    switch (source)
    {
    case OmniPvdDataType::eINT8:
        destination = data::RecordedDataType::Int8;
        return true;
    case OmniPvdDataType::eINT16:
        destination = data::RecordedDataType::Int16;
        return true;
    case OmniPvdDataType::eINT32:
        destination = data::RecordedDataType::Int32;
        return true;
    case OmniPvdDataType::eINT64:
        destination = data::RecordedDataType::Int64;
        return true;
    case OmniPvdDataType::eUINT8:
        destination = data::RecordedDataType::UInt8;
        return true;
    case OmniPvdDataType::eUINT16:
        destination = data::RecordedDataType::UInt16;
        return true;
    case OmniPvdDataType::eUINT32:
        destination = data::RecordedDataType::UInt32;
        return true;
    case OmniPvdDataType::eUINT64:
        destination = data::RecordedDataType::UInt64;
        return true;
    case OmniPvdDataType::eFLOAT32:
        destination = data::RecordedDataType::Float32;
        return true;
    case OmniPvdDataType::eFLOAT64:
        destination = data::RecordedDataType::Float64;
        return true;
    case OmniPvdDataType::eSTRING:
        destination = data::RecordedDataType::String;
        return true;
    case OmniPvdDataType::eOBJECT_HANDLE:
        destination = data::RecordedDataType::ObjectHandle;
        return true;
    case OmniPvdDataType::eENUM_VALUE:
        destination = data::RecordedDataType::EnumValue;
        return true;
    case OmniPvdDataType::eFLAGS_WORD:
        destination = data::RecordedDataType::FlagsWord;
        return true;
    default:
        return false;
    }
}

struct AttributeSchema
{
    std::uint64_t classHandle;
    std::string name;
};

struct ClassSchema
{
    std::uint64_t baseClassHandle;
    std::string name;
};

using SourceObjectKey = std::pair<std::uint64_t, std::uint64_t>;

struct SourceSchemaState
{
    std::map<std::uint64_t, ClassSchema> classes;
    std::map<std::uint64_t, AttributeSchema> attributes;
    std::map<SourceObjectKey, std::uint64_t> liveObjectClasses;
    std::set<std::string> legacyClassNames;
    bool metadataObjectSeen = false;
};

bool classChainContainsHandle(const SourceSchemaState& state, std::uint64_t classHandle,
                              std::uint64_t expectedHandle)
{
    std::set<std::uint64_t> visited;
    while (classHandle != OMNI_PVD_INVALID_HANDLE && visited.insert(classHandle).second)
    {
        if (classHandle == expectedHandle)
            return true;
        const auto found = state.classes.find(classHandle);
        if (found == state.classes.end())
            return false;
        classHandle = found->second.baseClassHandle;
    }
    return false;
}

bool classChainContainsName(const SourceSchemaState& state, std::uint64_t classHandle,
                            const char* expectedName)
{
    std::set<std::uint64_t> visited;
    while (classHandle != OMNI_PVD_INVALID_HANDLE && visited.insert(classHandle).second)
    {
        const auto found = state.classes.find(classHandle);
        if (found == state.classes.end())
            return false;
        if (found->second.name == expectedName)
            return true;
        classHandle = found->second.baseClassHandle;
    }
    return false;
}

bool isIntegrationMajor(const SourceSchemaState& state, std::uint64_t context,
                        std::uint64_t object, std::uint64_t attribute)
{
    const auto found = state.attributes.find(attribute);
    if (found == state.attributes.end()
        || found->second.name != "ovdIntegrationVersionMajor")
        return false;
    const auto objectFound = state.liveObjectClasses.find(SourceObjectKey(context, object));
    if (objectFound == state.liveObjectClasses.end())
        return false;
    return classChainContainsHandle(state, objectFound->second, found->second.classHandle)
        && classChainContainsName(state, objectFound->second, "PxOmniPvdMetaData");
}

bool mapCommand(OmniPvdCommand::Enum raw, OmniPvdReader& reader,
                data::CommandIngestor& ingestor,
                SourceSchemaState& sourceState,
                FileIngestResult& result)
{
    data::Command command;
    switch (raw)
    {
    case OmniPvdCommand::eREGISTER_CLASS:
    {
        command.kind = data::CommandKind::RegisterClass;
        command.classHandle = reader.getClassHandle();
        command.baseClassHandle = reader.getBaseClassHandle();
        command.name = reader.getClassName();
        const ClassSchema schema{
            command.baseClassHandle, command.name == nullptr ? "" : command.name};
        const bool inserted = sourceState.classes.emplace(command.classHandle, schema).second;
        if (inserted && !sourceState.metadataObjectSeen
            && command.name != nullptr && *command.name != '\0')
        {
            if (sourceState.legacyClassNames.count(command.name) != 0)
            {
                sourceState.liveObjectClasses.clear();
                sourceState.legacyClassNames.clear();
            }
            sourceState.legacyClassNames.insert(command.name);
        }
        break;
    }
    case OmniPvdCommand::eREGISTER_ATTRIBUTE:
    {
        sourceState.attributes[reader.getAttributeHandle()] = {reader.getClassHandle(),
            reader.getAttributeName() == nullptr ? "" : reader.getAttributeName()};
        command.kind = data::CommandKind::RegisterAttribute;
        command.ownerClassHandle = reader.getClassHandle();
        command.attributeHandle = reader.getAttributeHandle();
        command.name = reader.getAttributeName();
        const OmniPvdDataType::Enum sourceType = reader.getAttributeDataType();
        if (!mapDataType(sourceType, command.dataType))
        {
            result = failure(FileIngestError::TranslationFailed,
                             "unsupported OmniPVD attribute data type");
            return false;
        }
        if (sourceType == OmniPvdDataType::eENUM_VALUE)
        {
            command.attributeKind = data::RecordedAttributeKind::EnumConstant;
            command.enumValue = reader.getEnumValue();
        }
        else if (sourceType == OmniPvdDataType::eFLAGS_WORD)
        {
            command.attributeKind = data::RecordedAttributeKind::Value;
            command.flagsEnumClassHandle = reader.getEnumClassHandle();
        }
        else
        {
            command.attributeKind = data::RecordedAttributeKind::Value;
            command.hasElementCount = true;
            command.elementCount = reader.getAttributeNumberElements();
        }
        break;
    }
    case OmniPvdCommand::eREGISTER_CLASS_ATTRIBUTE:
        sourceState.attributes[reader.getAttributeHandle()] = {reader.getClassHandle(),
            reader.getAttributeName() == nullptr ? "" : reader.getAttributeName()};
        command.kind = data::CommandKind::RegisterAttribute;
        command.ownerClassHandle = reader.getClassHandle();
        command.attributeHandle = reader.getAttributeHandle();
        command.attributeKind = data::RecordedAttributeKind::ClassAttribute;
        command.dataType = data::RecordedDataType::None;
        command.attributeClassHandle = reader.getAttributeClassHandle();
        command.name = reader.getAttributeName();
        break;
    case OmniPvdCommand::eREGISTER_UNIQUE_LIST_ATTRIBUTE:
        sourceState.attributes[reader.getAttributeHandle()] = {reader.getClassHandle(),
            reader.getAttributeName() == nullptr ? "" : reader.getAttributeName()};
        command.kind = data::CommandKind::RegisterAttribute;
        command.ownerClassHandle = reader.getClassHandle();
        command.attributeHandle = reader.getAttributeHandle();
        command.attributeKind = data::RecordedAttributeKind::UniqueList;
        command.name = reader.getAttributeName();
        if (!mapDataType(reader.getAttributeDataType(), command.dataType))
        {
            result = failure(FileIngestError::TranslationFailed,
                             "unsupported OmniPVD attribute data type");
            return false;
        }
        break;
    case OmniPvdCommand::eCREATE_OBJECT:
    {
        command.kind = data::CommandKind::CreateObject;
        command.context = reader.getContextHandle();
        command.classHandle = reader.getClassHandle();
        command.objectHandle = reader.getObjectHandle();
        command.name = reader.getObjectName();
        const auto createdClass = sourceState.classes.find(command.classHandle);
        if (createdClass != sourceState.classes.end()
            && createdClass->second.name == "PxOmniPvdMetaData")
        {
            if (sourceState.metadataObjectSeen)
                sourceState.liveObjectClasses.clear();
            sourceState.metadataObjectSeen = true;
        }
        sourceState.liveObjectClasses[SourceObjectKey(command.context, command.objectHandle)] =
            command.classHandle;
        break;
    }
    case OmniPvdCommand::eDESTROY_OBJECT:
        command.kind = data::CommandKind::DestroyObject;
        command.context = reader.getContextHandle();
        command.objectHandle = reader.getObjectHandle();
        sourceState.liveObjectClasses.erase(SourceObjectKey(command.context, command.objectHandle));
        break;
    case OmniPvdCommand::eSET_ATTRIBUTE:
    case OmniPvdCommand::eADD_TO_UNIQUE_LIST_ATTRIBUTE:
    case OmniPvdCommand::eREMOVE_FROM_UNIQUE_LIST_ATTRIBUTE:
        if (raw == OmniPvdCommand::eSET_ATTRIBUTE)
            command.kind = data::CommandKind::SetProperty;
        else if (raw == OmniPvdCommand::eADD_TO_UNIQUE_LIST_ATTRIBUTE)
            command.kind = data::CommandKind::AddToList;
        else
            command.kind = data::CommandKind::RemoveFromList;
        command.context = reader.getContextHandle();
        command.objectHandle = reader.getObjectHandle();
        command.attributeHandle = reader.getAttributeHandle();
        command.payload = reader.getAttributeDataPointer();
        command.payloadSize = reader.getAttributeDataLength();
        if (isIntegrationMajor(sourceState, command.context, command.objectHandle,
                               command.attributeHandle))
        {
            if (command.payloadSize != sizeof(std::uint32_t) || command.payload == nullptr)
            {
                result = failure(FileIngestError::TranslationFailed,
                                 "invalid ovdIntegrationVersionMajor payload");
                return false;
            }
            std::uint32_t major = 0;
            std::memcpy(&major, command.payload, sizeof(major));
            if (major > kAcceptedIntegrationMajor)
            {
                result = failure(FileIngestError::UnsupportedIntegrationVersion,
                                 "unsupported OmniPVD integration major version");
                return false;
            }
        }
        break;
    case OmniPvdCommand::eSTART_FRAME:
        command.kind = data::CommandKind::StartFrame;
        command.context = reader.getContextHandle();
        command.frame = reader.getFrameTimeStart();
        break;
    case OmniPvdCommand::eSTOP_FRAME:
        command.kind = data::CommandKind::StopFrame;
        command.context = reader.getContextHandle();
        command.frame = reader.getFrameTimeStop();
        break;
    case OmniPvdCommand::eRECORD_MESSAGE:
        return true; // Validation accounts for messages, but OvdData has no message model.
    default:
        result = failure(FileIngestError::TranslationFailed, "unsupported OmniPVD command");
        return false;
    }

    if (!ingestor.consume(command))
    {
        result = failure(FileIngestError::TranslationFailed,
            std::string("OvdData command translation failed (command ")
            + commandName(raw) + ", error " + ingestErrorName(ingestor.error()) + ", context "
            + std::to_string(command.context) + ", object "
            + std::to_string(command.objectHandle) + ")");
        return false;
    }
    return true;
}

} // namespace

FileIngestLimits::FileIngestLimits()
{
    const OvdResourceLimits resourceDefaults = OvdResourceLimits::defaults();
    maxSingleAttributePayloadBytes = resourceDefaults.maxSingleAttributePayloadBytes;
    maxCumulativeAttributePayloadBytes = resourceDefaults.maxCumulativeAttributePayloadBytes;
    maxStreamBytes = resourceDefaults.maxStreamBytes;
    maxCommands = resourceDefaults.maxCommands;
    maxAttributeMutations = resourceDefaults.maxAttributeMutations;
    maxClasses = resourceDefaults.maxClasses;
    maxAttributes = resourceDefaults.maxAttributes;
    maxAttributesPerClass = resourceDefaults.maxAttributesPerClass;
    maxObjectCreates = resourceDefaults.maxObjectCreates;
    maxMessages = resourceDefaults.maxMessages;
    maxRetainedValidationBytes = resourceDefaults.maxRetainedDomBytes;
}

FileIngestLimits FileIngestLimits::defaults()
{
    return FileIngestLimits();
}

FileIngestResult ingestOvdFile(const std::string& path, const data::LoadPlan& loadPlan,
                               const FileIngestLimits& limits)
{
    std::unique_ptr<OmniPvdFileReadStream, FileStreamDeleter> stream(createOmniPvdFileReadStream());
    if (!stream)
        return failure(FileIngestError::ReaderRejected, "could not create OmniPVD file stream");
    stream->setFileName(path.c_str());
    if (!stream->openStream())
        return failure(FileIngestError::FileOpenFailed, "could not open OmniPVD file");

    OvdValidatedReadStream validated(*stream, validationLimits(limits));
    std::unique_ptr<OmniPvdReader, ReaderDeleter> reader(createOmniPvdReader());
    if (!reader)
        return failure(FileIngestError::ReaderRejected, "could not create OmniPVD reader");
    reader->setReadStream(validated);

    OmniPvdVersionType major = 0;
    OmniPvdVersionType minor = 0;
    OmniPvdVersionType patch = 0;
    if (!reader->startReading(major, minor, patch))
    {
        if (validated.failed())
            return validationFailure(validated.errorCode());
        return failure(FileIngestError::ReaderRejected, "OmniPVD reader rejected the stream header");
    }

    data::CommandIngestor ingestor(loadPlan);
    SourceSchemaState sourceState;
    for (;;)
    {
        const OmniPvdCommand::Enum command = reader->getNextCommand();
        if (command == OmniPvdCommand::eINVALID)
        {
            if (validated.failed())
                return validationFailure(validated.errorCode());
            if (!validated.cleanEof())
                return failure(FileIngestError::IncompleteInput, "incomplete OmniPVD input");
            break;
        }
        if (command == OmniPvdCommand::eREGISTER_ATTRIBUTE
            || command == OmniPvdCommand::eREGISTER_UNIQUE_LIST_ATTRIBUTE)
        {
            data::RecordedDataType mappedType = data::RecordedDataType::None;
            if (!mapDataType(reader->getAttributeDataType(), mappedType))
            {
                return failure(FileIngestError::TranslationFailed,
                               "unsupported OmniPVD attribute data type");
            }
        }
        if (!validated.validateCommand(command, *reader))
            return validationFailure(validated.errorCode());
        FileIngestResult mappingFailure;
        if (!mapCommand(command, *reader, ingestor, sourceState, mappingFailure))
            return mappingFailure;
    }

    if (!ingestor.finish())
        return failure(FileIngestError::IncompleteInput, "OmniPVD input ended with an active frame");
    FileIngestResult success;
    success.capture = ingestor.takeCapture();
    if (!success.capture)
        return failure(FileIngestError::TranslationFailed, "could not finalize OvdData capture");
    return success;
}

} // namespace omnipvd
} // namespace ovdnext
