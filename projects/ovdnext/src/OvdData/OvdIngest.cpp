/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "OvdIngest.h"
#include "OvdRecordedModelBuilder.h"

#include <map>
#include <set>
#include <string>
#include <utility>

namespace ovdnext
{
namespace data
{

struct Capture::Impl
{
    Impl(TemporalStore&& temporalStore, std::vector<FrameStamp> frameDirectory,
        const IngestStats& ingestStats,
        std::shared_ptr<const RecordedClassModel> immutableClassModel)
        : store(std::move(temporalStore))
        , frames(std::move(frameDirectory))
        , stats(ingestStats)
        , classModel(std::move(immutableClassModel))
    {
    }

    TemporalStore store;
    std::vector<FrameStamp> frames;
    IngestStats stats;
    std::shared_ptr<const RecordedClassModel> classModel;
};

Capture::Capture(std::unique_ptr<Impl> impl)
    : mImpl(std::move(impl))
{
}

Capture::~Capture() = default;
Capture::Capture(Capture&&) noexcept = default;
Capture& Capture::operator=(Capture&&) noexcept = default;

const std::vector<FrameStamp>& Capture::frames() const
{
    return mImpl->frames;
}

const IngestStats& Capture::stats() const
{
    return mImpl->stats;
}

std::shared_ptr<const RecordedClassModel> Capture::recordedClassModel() const
{
    return mImpl->classModel;
}

std::shared_ptr<const FrameView> Capture::view(std::size_t directoryIndex) const
{
    if (directoryIndex >= mImpl->frames.size())
    {
        return nullptr;
    }
    return view(FrameViewRequest(mImpl->frames[directoryIndex].address));
}

std::shared_ptr<const FrameView> Capture::view(const FrameAddress& address) const
{
    return view(FrameViewRequest(address));
}

std::shared_ptr<const FrameView> Capture::view(const FrameViewRequest& request) const
{
    FrameViewRequest resolvedRequest(request);
    resolvedRequest.mRecordedClassModel = mImpl->classModel;
    return mImpl->store.view(resolvedRequest);
}

struct CommandIngestor::Impl
{
    struct ObjectKey
    {
        std::uint64_t segment;
        std::uint64_t context;
        std::uint64_t external;

        bool operator<(const ObjectKey& rhs) const
        {
            if (segment != rhs.segment)
            {
                return segment < rhs.segment;
            }
            if (context != rhs.context)
            {
                return context < rhs.context;
            }
            return external < rhs.external;
        }
    };

    struct DeferredCommand
    {
        CommandKind kind;
        ObjectId object;
        PropertyId property;
        std::unique_ptr<Bytes> payload;
    };

    struct PendingObjectDescriptor
    {
        std::uint64_t sourceClassHandle;
        std::string recordedName;
    };

    explicit Impl(const LoadPlan& selectedPlan)
        : loadPlan(selectedPlan)
        , resolvedRetention(detail::ResolvedPropertyRetention::unresolved())
        , store(std::shared_ptr<const detail::ResolvedPropertyRetention>(resolvedRetention))
    {
    }

    LoadPlan loadPlan;
    std::shared_ptr<detail::ResolvedPropertyRetention> resolvedRetention;
    TemporalStore store;
    detail::RecordedClassModelBuilder classModelBuilder;
    std::shared_ptr<const RecordedClassModel> recordedClassModel;
    std::set<std::string> legacyClassNames;
    std::set<ObjectKey> deadObjectKeys;
    std::map<ObjectKey, ObjectId> logicalLiveObjects;
    std::map<ObjectKey, ObjectId> committedLiveObjects;
    std::map<ObjectId, PendingObjectDescriptor> pendingObjectDescriptors;
    std::set<ObjectId> activeCreatedObjects;
    std::vector<DeferredCommand> deferred;
    std::vector<FrameStamp> frames;
    IngestStats stats;
    IngestError error = IngestError::None;
    std::uint64_t segment = 0;
    std::uint64_t activeContext = 0;
    std::uint64_t activeFrame = 0;
    ObjectId nextObject = 1;
    bool active = false;
    bool finished = false;
    bool taken = false;
    bool metadataEstablished = false;

    bool fail(IngestError value)
    {
        if (error == IngestError::None)
        {
            if (active)
            {
                store.cancelFrame();
            }
            active = false;
            error = value;
        }
        return false;
    }

    bool failModelBuild()
    {
        switch (classModelBuilder.error())
        {
        case detail::ModelBuildError::InvalidSourceHandle:
            return fail(IngestError::InvalidSourceHandle);
        case detail::ModelBuildError::UnknownBaseClass:
            return fail(IngestError::UnknownBaseClass);
        case detail::ModelBuildError::ConflictingClassRegistration:
            return fail(IngestError::ConflictingClassRegistration);
        case detail::ModelBuildError::InheritanceCycle:
            return fail(IngestError::InheritanceCycle);
        case detail::ModelBuildError::UnknownAttributeOwner:
            return fail(IngestError::UnknownAttributeOwner);
        case detail::ModelBuildError::DuplicateAttributeRegistration:
            return fail(IngestError::DuplicateAttributeRegistration);
        case detail::ModelBuildError::UnknownRelatedClass:
            return fail(IngestError::UnknownRelatedClass);
        case detail::ModelBuildError::InvalidAttributeDeclaration:
            return fail(IngestError::InvalidAttributeDeclaration);
        default:
            return fail(IngestError::StoreFailure);
        }
    }

    bool apply(const DeferredCommand& command)
    {
        static const Bytes emptyPayload;
        const Bytes& payload = command.payload ? *command.payload : emptyPayload;
        StoreResult result = StoreResult::Ok;
        switch (command.kind)
        {
        case CommandKind::CreateObject:
            result = store.createObject(command.object);
            break;
        case CommandKind::DestroyObject:
            result = store.destroyObject(command.object);
            break;
        case CommandKind::SetProperty:
            result = store.setProperty(command.object, command.property, payload);
            break;
        case CommandKind::AddToList:
            result = store.addToList(command.object, command.property, payload);
            break;
        case CommandKind::RemoveFromList:
            result = store.removeFromList(command.object, command.property, payload);
            break;
        default:
            return fail(IngestError::StoreFailure);
        }
        if (result != StoreResult::Ok)
        {
            return fail(IngestError::StoreFailure);
        }
        if (command.kind == CommandKind::CreateObject)
        {
            activeCreatedObjects.insert(command.object);
        }
        return true;
    }

    void discardActiveObjectDescriptors()
    {
        for (ObjectId object : activeCreatedObjects)
        {
            pendingObjectDescriptors.erase(object);
        }
        activeCreatedObjects.clear();
    }

    bool publishActiveObjectDescriptors()
    {
        for (ObjectId object : activeCreatedObjects)
        {
            const auto pending = pendingObjectDescriptors.find(object);
            if (pending == pendingObjectDescriptors.end()
                || !classModelBuilder.registerObject(object,
                    pending->second.sourceClassHandle, pending->second.recordedName))
            {
                return classModelBuilder.error() == detail::ModelBuildError::None
                    ? fail(IngestError::UnknownClass) : failModelBuild();
            }
            pendingObjectDescriptors.erase(pending);
        }
        activeCreatedObjects.clear();
        return true;
    }

    bool queueOrApply(DeferredCommand&& command)
    {
        if (!active)
        {
            deferred.push_back(std::move(command));
            return true;
        }
        return apply(command);
    }

    bool transitionSegment()
    {
        if (active)
        {
            if (store.cancelFrame() != StoreResult::Ok)
            {
                return fail(IngestError::StoreFailure);
            }
            active = false;
            deferred.clear();
            discardActiveObjectDescriptors();
            logicalLiveObjects = committedLiveObjects;
        }
        for (auto object = logicalLiveObjects.begin(); object != logicalLiveObjects.end(); )
        {
            deferred.push_back({CommandKind::DestroyObject, object->second, 0, nullptr});
            object = logicalLiveObjects.erase(object);
        }
        deadObjectKeys.clear();
        ++segment;
        return true;
    }

    bool consumeCreate(const Command& command)
    {
        const ObjectKey key{segment, command.context, command.objectHandle};
        deadObjectKeys.erase(key);
        const auto previous = logicalLiveObjects.find(key);
        if (previous != logicalLiveObjects.end())
        {
            if (!queueOrApply({CommandKind::DestroyObject, previous->second, 0, nullptr}))
            {
                return false;
            }
            logicalLiveObjects.erase(previous);
        }
        const ObjectId generation = nextObject++;
        pendingObjectDescriptors.emplace(generation, PendingObjectDescriptor{
            command.classHandle, command.name == nullptr ? "" : command.name});
        logicalLiveObjects.insert(std::make_pair(key, generation));
        return queueOrApply({CommandKind::CreateObject, generation, 0, nullptr});
    }

    bool consumeDestroy(const Command& command)
    {
        const ObjectKey key{segment, command.context, command.objectHandle};
        const auto found = logicalLiveObjects.find(key);
        if (found == logicalLiveObjects.end())
        {
            return true;
        }
        const ObjectId generation = found->second;
        logicalLiveObjects.erase(found);
        deadObjectKeys.insert(key);
        return queueOrApply({CommandKind::DestroyObject, generation, 0, nullptr});
    }

    bool consumeMutation(const Command& command)
    {
        if (command.attributeHandle == 0 || command.objectHandle == 0)
        {
            return fail(IngestError::InvalidSourceHandle);
        }
        const RecordedAttribute* attribute =
            classModelBuilder.findAttributeBySource(command.attributeHandle);
        if (attribute == nullptr)
        {
            return fail(IngestError::UnknownProperty);
        }
        const bool kindMatches = command.kind == CommandKind::SetProperty
            ? attribute->kind == RecordedAttributeKind::Value
            : attribute->kind == RecordedAttributeKind::UniqueList;
        if (!kindMatches)
        {
            return fail(IngestError::PropertyKindMismatch);
        }
        const PropertyId property =
            classModelBuilder.propertyIdForSource(command.attributeHandle);
        const ObjectKey key{segment, command.context, command.objectHandle};
        const auto object = logicalLiveObjects.find(key);
        if (object == logicalLiveObjects.end())
        {
            if (command.kind == CommandKind::RemoveFromList && deadObjectKeys.count(key) != 0)
            {
                return true;
            }
            return fail(IngestError::UnknownObject);
        }

        if (command.payloadSize != 0 && command.payload == nullptr)
        {
            return fail(IngestError::StoreFailure);
        }

        std::unique_ptr<Bytes> payload;
        if (resolvedRetention->retains(property))
        {
            if (command.payloadSize != 0)
            {
                payload.reset(new Bytes(command.payload, command.payload + command.payloadSize));
            }
            stats.retainedPayloadBytes += command.payloadSize;
        }
        else
        {
            stats.filteredPayloadBytes += command.payloadSize;
        }
        return queueOrApply({command.kind, object->second, property, std::move(payload)});
    }
};

CommandIngestor::CommandIngestor(const LoadPlan& loadPlan)
    : mImpl(new Impl(loadPlan))
{
}

CommandIngestor::~CommandIngestor() = default;
CommandIngestor::CommandIngestor(CommandIngestor&&) noexcept = default;
CommandIngestor& CommandIngestor::operator=(CommandIngestor&&) noexcept = default;

bool CommandIngestor::consume(const Command& command)
{
    if (mImpl->error != IngestError::None)
    {
        return false;
    }
    if (mImpl->finished)
    {
        return mImpl->fail(IngestError::AlreadyFinished);
    }
    ++mImpl->stats.commandCount;

    switch (command.kind)
    {
    case CommandKind::RegisterClass:
    {
        const std::string name = command.name == nullptr ? "" : command.name;
        const bool handleWasRegistered =
            mImpl->classModelBuilder.findClassBySource(command.classHandle) != nullptr;
        if (!mImpl->classModelBuilder.registerClass(
                command.classHandle, command.baseClassHandle, name))
        {
            return mImpl->failModelBuild();
        }
        if (handleWasRegistered)
        {
            return true;
        }
        const RecordedClass* recordedClass =
            mImpl->classModelBuilder.findClassBySource(command.classHandle);
        const std::string& recordedName = recordedClass->name;
        if (!mImpl->metadataEstablished && !recordedName.empty())
        {
            if (mImpl->legacyClassNames.count(recordedName) != 0)
            {
                if (!mImpl->transitionSegment())
                {
                    return false;
                }
                mImpl->legacyClassNames.clear();
            }
            mImpl->legacyClassNames.insert(recordedName);
        }
        return true;
    }
    case CommandKind::RegisterAttribute:
    {
        AttributeId recordedAttribute = 0;
        PropertyId valueProperty = 0;
        if (!mImpl->classModelBuilder.registerAttribute(command.attributeHandle,
                command.ownerClassHandle, command.attributeKind, command.dataType,
                command.hasElementCount, command.elementCount,
                command.attributeClassHandle, command.flagsEnumClassHandle,
                command.enumValue, command.name == nullptr ? "" : command.name,
                recordedAttribute, valueProperty))
        {
            return mImpl->failModelBuild();
        }
        if (valueProperty != 0)
        {
            const RecordedClass* owner =
                mImpl->classModelBuilder.findClassBySource(command.ownerClassHandle);
            const RecordedAttributeSelector selector{
                owner->name,
                command.name == nullptr ? "" : command.name
            };
            mImpl->resolvedRetention->resolve(
                valueProperty, mImpl->loadPlan.retains(selector));
        }
        return true;
    }
    case CommandKind::CreateObject:
    {
        if (command.objectHandle == 0)
        {
            return mImpl->fail(IngestError::InvalidSourceHandle);
        }
        const RecordedClass* recordedClass = command.classHandle == 0 ? nullptr
            : mImpl->classModelBuilder.findClassBySource(command.classHandle);
        if (recordedClass == nullptr)
        {
            return mImpl->fail(IngestError::UnknownClass);
        }
        if (recordedClass != nullptr && recordedClass->name == "PxOmniPvdMetaData")
        {
            if (mImpl->metadataEstablished)
            {
                if (!mImpl->transitionSegment())
                {
                    return false;
                }
            }
            else
            {
                if (mImpl->active)
                {
                    return mImpl->fail(IngestError::SegmentTransitionDuringFrame);
                }
                mImpl->metadataEstablished = true;
            }
        }
        return mImpl->consumeCreate(command);
    }
    case CommandKind::DestroyObject:
        if (command.objectHandle == 0)
        {
            return mImpl->fail(IngestError::InvalidSourceHandle);
        }
        return mImpl->consumeDestroy(command);
    case CommandKind::SetProperty:
    case CommandKind::AddToList:
    case CommandKind::RemoveFromList:
        return mImpl->consumeMutation(command);
    case CommandKind::StartFrame:
    {
        if (mImpl->active)
        {
            return mImpl->fail(IngestError::NestedStart);
        }
        if (mImpl->store.beginFrame({mImpl->segment, command.context, command.frame}) != StoreResult::Ok)
        {
            return mImpl->fail(IngestError::StoreFailure);
        }
        mImpl->active = true;
        mImpl->activeContext = command.context;
        mImpl->activeFrame = command.frame;
        for (const Impl::DeferredCommand& deferred : mImpl->deferred)
        {
            if (!mImpl->apply(deferred))
            {
                return false;
            }
        }
        mImpl->deferred.clear();
        return true;
    }
    case CommandKind::StopFrame:
    {
        if (!mImpl->active)
        {
            return mImpl->fail(IngestError::StopWithoutStart);
        }
        if (mImpl->activeContext != command.context || mImpl->activeFrame != command.frame)
        {
            return mImpl->fail(IngestError::MismatchedStop);
        }
        FrameStamp published;
        if (mImpl->store.finishFrame(published) != StoreResult::Ok)
        {
            return mImpl->fail(IngestError::StoreFailure);
        }
        if (!mImpl->publishActiveObjectDescriptors())
        {
            return false;
        }
        mImpl->frames.push_back(published);
        ++mImpl->stats.publicationCount;
        mImpl->active = false;
        mImpl->committedLiveObjects = mImpl->logicalLiveObjects;
        return true;
    }
    }
    return mImpl->fail(IngestError::StoreFailure);
}

bool CommandIngestor::finish()
{
    if (mImpl->error != IngestError::None)
    {
        return false;
    }
    if (mImpl->finished)
    {
        return mImpl->fail(IngestError::AlreadyFinished);
    }
    if (mImpl->active)
    {
        return mImpl->fail(IngestError::ActiveFrameAtEof);
    }
    mImpl->pendingObjectDescriptors.clear();
    mImpl->recordedClassModel = mImpl->classModelBuilder.freeze();
    if (!mImpl->recordedClassModel)
    {
        return mImpl->failModelBuild();
    }
    mImpl->resolvedRetention->freeze();
    mImpl->finished = true;
    return true;
}

IngestError CommandIngestor::error() const
{
    return mImpl->error;
}

const IngestStats& CommandIngestor::stats() const
{
    return mImpl->stats;
}

const std::vector<FrameStamp>& CommandIngestor::frames() const
{
    return mImpl->frames;
}

std::unique_ptr<Capture> CommandIngestor::takeCapture()
{
    if (!mImpl->finished || mImpl->error != IngestError::None || mImpl->taken)
    {
        return nullptr;
    }
    mImpl->taken = true;
    mImpl->resolvedRetention.reset();
    std::unique_ptr<Capture::Impl> capture(new Capture::Impl(
        std::move(mImpl->store), std::move(mImpl->frames), mImpl->stats,
        std::move(mImpl->recordedClassModel)));
    return std::unique_ptr<Capture>(new Capture(std::move(capture)));
}

} // namespace data
} // namespace ovdnext
