/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "OvdData.h"
#include "OvdRecordedModel.h"

#include <algorithm>
#include <list>
#include <map>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ovdnext
{
namespace data
{

bool operator==(const FrameAddress& lhs, const FrameAddress& rhs)
{
    return lhs.segment == rhs.segment && lhs.context == rhs.context && lhs.frame == rhs.frame;
}

bool operator<(const FrameAddress& lhs, const FrameAddress& rhs)
{
    if (lhs.segment != rhs.segment)
    {
        return lhs.segment < rhs.segment;
    }
    if (lhs.context != rhs.context)
    {
        return lhs.context < rhs.context;
    }
    return lhs.frame < rhs.frame;
}

bool operator<(const RecordedAttributeSelector& lhs,
               const RecordedAttributeSelector& rhs)
{
    return lhs.ownerClassName < rhs.ownerClassName ||
        (lhs.ownerClassName == rhs.ownerClassName &&
         lhs.attributeName < rhs.attributeName);
}

LoadPlan::LoadPlan(bool retainsAllProperties,
                   std::set<RecordedAttributeSelector> selectors)
    : mRetainsAllProperties(retainsAllProperties)
    , mSelectors(std::move(selectors))
{
}

LoadPlan LoadPlan::allProperties()
{
    return LoadPlan(true, {});
}

LoadPlan LoadPlan::onlyRecordedAttributes(
    std::set<RecordedAttributeSelector> selectors)
{
    return LoadPlan(false, std::move(selectors));
}

bool LoadPlan::retains(const RecordedAttributeSelector& selector) const
{
    return mRetainsAllProperties || mSelectors.count(selector) != 0;
}

namespace detail
{
ResolvedPropertyRetention::ResolvedPropertyRetention(
    bool retainsAll, std::set<PropertyId> properties, bool frozen)
    : mRetainsAll(retainsAll)
    , mFrozen(frozen)
    , mProperties(std::move(properties))
{
}

std::shared_ptr<ResolvedPropertyRetention> ResolvedPropertyRetention::allProperties()
{
    return std::shared_ptr<ResolvedPropertyRetention>(
        new ResolvedPropertyRetention(true, {}, true));
}

std::shared_ptr<ResolvedPropertyRetention> ResolvedPropertyRetention::onlyProperties(
    std::set<PropertyId> properties)
{
    return std::shared_ptr<ResolvedPropertyRetention>(
        new ResolvedPropertyRetention(false, std::move(properties), true));
}

std::shared_ptr<ResolvedPropertyRetention> ResolvedPropertyRetention::unresolved()
{
    return std::shared_ptr<ResolvedPropertyRetention>(
        new ResolvedPropertyRetention(false, {}, false));
}

bool ResolvedPropertyRetention::retains(PropertyId property) const
{
    return mRetainsAll || mProperties.count(property) != 0;
}

void ResolvedPropertyRetention::resolve(PropertyId property, bool retained)
{
    if (mFrozen || property == 0 || mResolved.count(property) != 0)
    {
        throw std::logic_error("invalid property retention resolution");
    }
    mResolved.insert(property);
    if (retained)
    {
        mProperties.insert(property);
    }
}

void ResolvedPropertyRetention::freeze()
{
    mFrozen = true;
}
} // namespace detail

FrameViewRequest::FrameViewRequest(FrameAddress address)
    : mAddress(address)
{
}

struct FrameView::Data
{
    explicit Data(std::shared_ptr<const detail::ResolvedPropertyRetention> selectedRetention)
        : retention(std::move(selectedRetention))
    {
    }

    FrameStamp stamp;
    std::shared_ptr<const detail::ResolvedPropertyRetention> retention;
    std::shared_ptr<const RecordedClassModel> recordedClassModel;
    std::set<ObjectId> retainedAliveObjects;
    std::set<PropertyId> propertyAllowList;
    std::vector<ObjectId> aliveObjects;
    std::map<ObjectId, std::map<PropertyId, Bytes>> persistentProperties;
    std::map<ObjectId, std::map<PropertyId, Bytes>> exactProperties;
    std::map<ObjectId, std::map<PropertyId, std::vector<Bytes>>> lists;
};

FrameView::FrameView(std::shared_ptr<const Data> data)
    : mData(std::move(data))
{
}

const FrameStamp& FrameView::stamp() const
{
    return mData->stamp;
}

const std::vector<ObjectId>& FrameView::aliveObjects() const
{
    return mData->aliveObjects;
}

bool FrameView::isAlive(ObjectId object) const
{
    return std::binary_search(mData->aliveObjects.begin(), mData->aliveObjects.end(), object);
}

PropertyResult FrameView::persistentProperty(ObjectId object, PropertyId property) const
{
    if (!mData->retention->retains(property))
    {
        return {Availability::NotLoaded, {}};
    }
    if ((!mData->propertyAllowList.empty() && mData->propertyAllowList.count(property) == 0) ||
        (mData->retainedAliveObjects.count(object) != 0 && !isAlive(object)))
    {
        return {Availability::NotProjected, {}};
    }
    if (mData->retainedAliveObjects.count(object) == 0)
    {
        return {Availability::AbsentAtFrame, {}};
    }

    const auto objectProperties = mData->persistentProperties.find(object);
    if (objectProperties == mData->persistentProperties.end())
    {
        return {Availability::AbsentAtFrame, {}};
    }
    const auto value = objectProperties->second.find(property);
    return value == objectProperties->second.end()
        ? PropertyResult{Availability::AbsentAtFrame, {}}
        : PropertyResult{Availability::Present, value->second};
}

PropertyResult FrameView::exactProperty(ObjectId object, PropertyId property) const
{
    if (!mData->retention->retains(property))
    {
        return {Availability::NotLoaded, {}};
    }
    if ((!mData->propertyAllowList.empty() && mData->propertyAllowList.count(property) == 0) ||
        (mData->retainedAliveObjects.count(object) != 0 && !isAlive(object)))
    {
        return {Availability::NotProjected, {}};
    }
    if (mData->retainedAliveObjects.count(object) == 0)
    {
        return {Availability::AbsentAtFrame, {}};
    }

    const auto objectProperties = mData->exactProperties.find(object);
    if (objectProperties == mData->exactProperties.end())
    {
        return {Availability::AbsentAtFrame, {}};
    }
    const auto value = objectProperties->second.find(property);
    return value == objectProperties->second.end()
        ? PropertyResult{Availability::AbsentAtFrame, {}}
        : PropertyResult{Availability::Present, value->second};
}

ListResult FrameView::list(ObjectId object, PropertyId property) const
{
    if (!mData->retention->retains(property))
    {
        return {Availability::NotLoaded, {}};
    }
    if ((!mData->propertyAllowList.empty() && mData->propertyAllowList.count(property) == 0) ||
        (mData->retainedAliveObjects.count(object) != 0 && !isAlive(object)))
    {
        return {Availability::NotProjected, {}};
    }
    if (mData->retainedAliveObjects.count(object) == 0)
    {
        return {Availability::AbsentAtFrame, {}};
    }

    const auto objectLists = mData->lists.find(object);
    if (objectLists == mData->lists.end())
    {
        return {Availability::AbsentAtFrame, {}};
    }
    const auto value = objectLists->second.find(property);
    return value == objectLists->second.end()
        ? ListResult{Availability::AbsentAtFrame, {}}
        : ListResult{Availability::Present, value->second};
}

std::shared_ptr<const RecordedClassModel> FrameView::recordedClassModel() const
{
    return mData->recordedClassModel;
}

const RecordedObject* FrameView::recordedObject(ObjectId object) const
{
    return mData->recordedClassModel ? mData->recordedClassModel->findObject(object) : nullptr;
}

ClassObjectResult FrameView::aliveObjectsOfExactClass(ClassId recordedClass) const
{
    ClassObjectResult result;
    if (!mData->recordedClassModel || !mData->recordedClassModel->findClass(recordedClass))
    {
        return result;
    }
    result.classKnown = true;
    for (const ObjectId object : mData->aliveObjects)
    {
        const RecordedObject* descriptor = mData->recordedClassModel->findObject(object);
        if (descriptor && descriptor->recordedClass == recordedClass)
        {
            result.objects.push_back(object);
        }
    }
    return result;
}

ClassObjectResult FrameView::aliveObjectsOfClassOrDerived(ClassId recordedClass) const
{
    ClassObjectResult result;
    if (!mData->recordedClassModel || !mData->recordedClassModel->findClass(recordedClass))
    {
        return result;
    }
    result.classKnown = true;
    for (const ObjectId object : mData->aliveObjects)
    {
        const RecordedObject* descriptor = mData->recordedClassModel->findObject(object);
        if (descriptor && mData->recordedClassModel->isSameOrDerived(
                descriptor->recordedClass, recordedClass))
        {
            result.objects.push_back(object);
        }
    }
    return result;
}

struct TemporalStore::Impl
{
    enum class ListOperationKind
    {
        Add,
        Remove
    };

    struct PropertyWrite
    {
        ObjectId object;
        PropertyId property;
        Bytes value;
    };

    struct ListWrite
    {
        ObjectId object;
        PropertyId property;
        std::uint64_t sequence;
        ListOperationKind kind;
        Bytes value;
    };

    struct PendingFrame
    {
        FrameAddress address;
        std::set<ObjectId> createdObjects;
        std::set<ObjectId> destroyedObjects;
        std::vector<PropertyWrite> propertyWrites;
        std::vector<ListWrite> listWrites;
        std::uint64_t nextSequence = 0;
    };

    struct FrameRecord final
    {
        FrameRecord(std::uint64_t publication, const PendingFrame& pending)
            : stamp{pending.address, publication}
            , createdObjects(pending.createdObjects)
            , destroyedObjects(pending.destroyedObjects)
            , propertyWrites(pending.propertyWrites)
            , listWrites(pending.listWrites)
        {
        }

        FrameRecord(const FrameRecord&) = delete;
        FrameRecord& operator=(const FrameRecord&) = delete;
        FrameRecord(FrameRecord&&) = delete;
        FrameRecord& operator=(FrameRecord&&) = delete;

        const FrameStamp stamp;
        const std::set<ObjectId> createdObjects;
        const std::set<ObjectId> destroyedObjects;
        const std::vector<PropertyWrite> propertyWrites;
        const std::vector<ListWrite> listWrites;
    };

    static_assert(!std::is_copy_constructible<FrameRecord>::value,
        "Published frame records must never copy retained history");
    static_assert(!std::is_copy_assignable<FrameRecord>::value,
        "Published frame records must remain immutable");

    using ObjectSet = std::set<ObjectId>;

    explicit Impl(std::shared_ptr<const detail::ResolvedPropertyRetention> selectedRetention)
        : retention(std::move(selectedRetention))
    {
    }

    std::shared_ptr<const detail::ResolvedPropertyRetention> retention;
    std::unique_ptr<PendingFrame> pendingFrame;
    std::list<FrameRecord> frameRecords;
    std::map<FrameAddress, const FrameRecord*> publishedAddresses;
    ObjectSet knownObjects;
    ObjectSet liveObjects;
    std::uint64_t nextPublication = 1;

    StoreResult validateLiveObject(ObjectId object) const
    {
        if (pendingFrame->destroyedObjects.count(object) != 0)
        {
            return StoreResult::ObjectNotLive;
        }
        if (pendingFrame->createdObjects.count(object) != 0)
        {
            return StoreResult::Ok;
        }

        if (knownObjects.count(object) == 0)
        {
            return StoreResult::ObjectUnknown;
        }
        return liveObjects.count(object) != 0
            ? StoreResult::Ok
            : StoreResult::ObjectNotLive;
    }

    std::shared_ptr<FrameView::Data> materialize(
        const FrameRecord& target,
        const std::set<ObjectId>& objectAllowList,
        const std::set<PropertyId>& propertyAllowList,
        const std::shared_ptr<const RecordedClassModel>& recordedClassModel) const
    {
        std::shared_ptr<FrameView::Data> data(new FrameView::Data(retention));
        data->stamp = target.stamp;
        data->propertyAllowList = propertyAllowList;
        data->recordedClassModel = recordedClassModel;

        for (const FrameRecord& frame : frameRecords)
        {
            for (const ObjectId object : frame.createdObjects)
            {
                data->retainedAliveObjects.insert(object);
            }
            for (const ObjectId object : frame.destroyedObjects)
            {
                data->retainedAliveObjects.erase(object);
            }
            if (frame.stamp.publication == target.stamp.publication)
            {
                break;
            }
        }

        for (const ObjectId object : data->retainedAliveObjects)
        {
            if (objectAllowList.empty() || objectAllowList.count(object) != 0)
            {
                data->aliveObjects.push_back(object);
            }
        }

        for (const FrameRecord& frame : frameRecords)
        {
            const bool exactFrame = frame.stamp.publication == target.stamp.publication;
            for (const PropertyWrite& write : frame.propertyWrites)
            {
                if (data->retainedAliveObjects.count(write.object) == 0 ||
                    (!objectAllowList.empty() && objectAllowList.count(write.object) == 0) ||
                    (!propertyAllowList.empty() && propertyAllowList.count(write.property) == 0))
                {
                    continue;
                }

                data->persistentProperties[write.object][write.property] = write.value;
                if (exactFrame)
                {
                    data->exactProperties[write.object][write.property] = write.value;
                }
            }

            for (const ListWrite& write : frame.listWrites)
            {
                if (data->retainedAliveObjects.count(write.object) == 0 ||
                    (!objectAllowList.empty() && objectAllowList.count(write.object) == 0) ||
                    (!propertyAllowList.empty() && propertyAllowList.count(write.property) == 0))
                {
                    continue;
                }

                std::vector<Bytes>& values = data->lists[write.object][write.property];
                const auto found = std::find(values.begin(), values.end(), write.value);
                if (write.kind == ListOperationKind::Add)
                {
                    if (found == values.end())
                    {
                        values.push_back(write.value);
                    }
                }
                else if (found != values.end())
                {
                    values.erase(found);
                }
            }

            if (exactFrame)
            {
                break;
            }
        }

        return data;
    }
};

TemporalStore::TemporalStore()
    : TemporalStore(detail::ResolvedPropertyRetention::allProperties())
{
}

TemporalStore::TemporalStore(
    std::shared_ptr<const detail::ResolvedPropertyRetention> retention)
    : mImpl(new Impl(std::move(retention)))
{
}

TemporalStore::~TemporalStore() = default;
TemporalStore::TemporalStore(TemporalStore&&) noexcept = default;
TemporalStore& TemporalStore::operator=(TemporalStore&&) noexcept = default;

StoreResult TemporalStore::beginFrame(const FrameAddress& address)
{
    if (mImpl->pendingFrame)
    {
        return StoreResult::FrameAlreadyPending;
    }
    if (mImpl->publishedAddresses.count(address) != 0)
    {
        return StoreResult::FrameAddressAlreadyPublished;
    }

    mImpl->pendingFrame.reset(new Impl::PendingFrame{address, {}});
    return StoreResult::Ok;
}

StoreResult TemporalStore::createObject(ObjectId object)
{
    if (!mImpl->pendingFrame)
    {
        return StoreResult::NoPendingFrame;
    }
    if (mImpl->knownObjects.count(object) != 0 ||
        mImpl->pendingFrame->createdObjects.count(object) != 0)
    {
        return StoreResult::ObjectAlreadyKnown;
    }

    mImpl->pendingFrame->createdObjects.insert(object);
    return StoreResult::Ok;
}

StoreResult TemporalStore::destroyObject(ObjectId object)
{
    if (!mImpl->pendingFrame)
    {
        return StoreResult::NoPendingFrame;
    }
    const StoreResult validation = mImpl->validateLiveObject(object);
    if (validation != StoreResult::Ok)
    {
        return validation;
    }

    mImpl->pendingFrame->destroyedObjects.insert(object);
    return StoreResult::Ok;
}

StoreResult TemporalStore::setProperty(ObjectId object, PropertyId property, const Bytes& bytes)
{
    if (!mImpl->pendingFrame)
    {
        return StoreResult::NoPendingFrame;
    }
    const StoreResult validation = mImpl->validateLiveObject(object);
    if (validation != StoreResult::Ok)
    {
        return validation;
    }
    if (!mImpl->retention->retains(property))
    {
        return StoreResult::Ok;
    }

    mImpl->pendingFrame->propertyWrites.push_back({object, property, bytes});
    return StoreResult::Ok;
}

StoreResult TemporalStore::addToList(ObjectId object, PropertyId property, const Bytes& bytes)
{
    if (!mImpl->pendingFrame)
    {
        return StoreResult::NoPendingFrame;
    }
    const StoreResult validation = mImpl->validateLiveObject(object);
    if (validation != StoreResult::Ok)
    {
        return validation;
    }
    if (!mImpl->retention->retains(property))
    {
        return StoreResult::Ok;
    }

    const std::uint64_t sequence = mImpl->pendingFrame->nextSequence;
    mImpl->pendingFrame->listWrites.push_back(
        {object, property, sequence, Impl::ListOperationKind::Add, bytes});
    ++mImpl->pendingFrame->nextSequence;
    return StoreResult::Ok;
}

StoreResult TemporalStore::removeFromList(ObjectId object, PropertyId property, const Bytes& bytes)
{
    if (!mImpl->pendingFrame)
    {
        return StoreResult::NoPendingFrame;
    }
    const StoreResult validation = mImpl->validateLiveObject(object);
    if (validation != StoreResult::Ok)
    {
        return validation;
    }
    if (!mImpl->retention->retains(property))
    {
        return StoreResult::Ok;
    }

    const std::uint64_t sequence = mImpl->pendingFrame->nextSequence;
    mImpl->pendingFrame->listWrites.push_back(
        {object, property, sequence, Impl::ListOperationKind::Remove, bytes});
    ++mImpl->pendingFrame->nextSequence;
    return StoreResult::Ok;
}

StoreResult TemporalStore::finishFrame(FrameStamp& publishedStamp)
{
    if (!mImpl->pendingFrame)
    {
        return StoreResult::NoPendingFrame;
    }

    std::vector<Impl::ObjectSet::iterator> insertedKnownObjects;
    std::vector<Impl::ObjectSet::iterator> insertedLiveObjects;
    std::vector<Impl::ObjectSet::iterator> destroyedLiveObjects;
    insertedKnownObjects.reserve(mImpl->pendingFrame->createdObjects.size());
    insertedLiveObjects.reserve(mImpl->pendingFrame->createdObjects.size());
    destroyedLiveObjects.reserve(mImpl->pendingFrame->destroyedObjects.size());

    const std::uint64_t publication = mImpl->nextPublication;
    mImpl->frameRecords.emplace_back(publication, *mImpl->pendingFrame);

    const auto rollbackPreparation = [&]() noexcept {
        for (auto iterator = insertedLiveObjects.rbegin();
             iterator != insertedLiveObjects.rend(); ++iterator)
        {
            mImpl->liveObjects.erase(*iterator);
        }
        for (auto iterator = insertedKnownObjects.rbegin();
             iterator != insertedKnownObjects.rend(); ++iterator)
        {
            mImpl->knownObjects.erase(*iterator);
        }
        mImpl->frameRecords.pop_back();
    };

    try
    {
        for (const ObjectId object : mImpl->pendingFrame->createdObjects)
        {
            const auto knownInsertion = mImpl->knownObjects.insert(object);
            if (!knownInsertion.second)
            {
                throw std::logic_error("pending object unexpectedly became known");
            }
            insertedKnownObjects.push_back(knownInsertion.first);

            const auto liveInsertion = mImpl->liveObjects.insert(object);
            if (!liveInsertion.second)
            {
                throw std::logic_error("newly known object unexpectedly became live");
            }
            insertedLiveObjects.push_back(liveInsertion.first);
        }

        for (const ObjectId object : mImpl->pendingFrame->destroyedObjects)
        {
            const auto liveObject = mImpl->liveObjects.find(object);
            if (liveObject == mImpl->liveObjects.end())
            {
                throw std::logic_error("pending destruction unexpectedly lost its live object");
            }
            destroyedLiveObjects.push_back(liveObject);
        }

        const Impl::FrameRecord* const stagedFrame = &mImpl->frameRecords.back();
        const auto addressInsertion =
            mImpl->publishedAddresses.emplace(stagedFrame->stamp.address, stagedFrame);
        if (!addressInsertion.second)
        {
            rollbackPreparation();
            return StoreResult::FrameAddressAlreadyPublished;
        }
    }
    catch (...)
    {
        rollbackPreparation();
        throw;
    }

    // The address insertion above is the visibility commit. Everything below is no-throw.
    const Impl::FrameRecord& publishedFrame = mImpl->frameRecords.back();
    for (const Impl::ObjectSet::iterator liveObject : destroyedLiveObjects)
    {
        mImpl->liveObjects.erase(liveObject);
    }
    ++mImpl->nextPublication;
    publishedStamp = publishedFrame.stamp;
    mImpl->pendingFrame.reset();
    return StoreResult::Ok;
}

StoreResult TemporalStore::cancelFrame()
{
    if (!mImpl->pendingFrame)
    {
        return StoreResult::NoPendingFrame;
    }

    mImpl->pendingFrame.reset();
    return StoreResult::Ok;
}

std::shared_ptr<const FrameView> TemporalStore::latestView() const
{
    if (mImpl->frameRecords.empty())
    {
        return nullptr;
    }

    const std::shared_ptr<FrameView::Data> data =
        mImpl->materialize(mImpl->frameRecords.back(), {}, {}, {});
    return std::shared_ptr<const FrameView>(
        new FrameView(std::shared_ptr<const FrameView::Data>(data)));
}

std::shared_ptr<const FrameView> TemporalStore::view(const FrameViewRequest& request) const
{
    const auto found = mImpl->publishedAddresses.find(request.mAddress);
    if (found == mImpl->publishedAddresses.end())
    {
        return nullptr;
    }

    const std::shared_ptr<FrameView::Data> data = mImpl->materialize(
        *found->second,
        request.objectAllowList,
        request.propertyAllowList,
        request.mRecordedClassModel);
    return std::shared_ptr<const FrameView>(
        new FrameView(std::shared_ptr<const FrameView::Data>(data)));
}

} // namespace data
} // namespace ovdnext
