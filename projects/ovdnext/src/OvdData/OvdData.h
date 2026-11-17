/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace ovdnext
{
namespace data
{

using ObjectId = std::uint64_t;
using PropertyId = std::uint64_t;
using ClassId = std::uint64_t;
using AttributeId = std::uint64_t;
using Bytes = std::vector<std::uint8_t>;

struct FrameAddress
{
    std::uint64_t segment = 0;
    std::uint64_t context = 0;
    std::uint64_t frame = 0;
};

bool operator==(const FrameAddress& lhs, const FrameAddress& rhs);
bool operator<(const FrameAddress& lhs, const FrameAddress& rhs);

struct FrameStamp
{
    FrameAddress address;
    std::uint64_t publication = 0;
};

enum class StoreResult
{
    Ok,
    FrameAlreadyPending,
    NoPendingFrame,
    FrameAddressAlreadyPublished,
    ObjectAlreadyKnown,
    ObjectUnknown,
    ObjectNotLive
};

enum class Availability { Present, AbsentAtFrame, NotLoaded, NotProjected };

struct PropertyResult { Availability availability; Bytes bytes; };
struct ListResult { Availability availability; std::vector<Bytes> values; };

class RecordedClassModel;
struct RecordedObject;
class Capture;

struct RecordedAttributeSelector
{
    std::string ownerClassName;
    std::string attributeName;
};

bool operator<(const RecordedAttributeSelector& lhs,
               const RecordedAttributeSelector& rhs);

class LoadPlan
{
public:
    static LoadPlan allProperties();
    static LoadPlan onlyRecordedAttributes(
        std::set<RecordedAttributeSelector> selectors);
    bool retains(const RecordedAttributeSelector& selector) const;

private:
    LoadPlan(bool retainsAllProperties,
             std::set<RecordedAttributeSelector> selectors);

    bool mRetainsAllProperties;
    std::set<RecordedAttributeSelector> mSelectors;
};

namespace detail
{
class ResolvedPropertyRetention
{
public:
    static std::shared_ptr<ResolvedPropertyRetention> allProperties();
    static std::shared_ptr<ResolvedPropertyRetention> onlyProperties(
        std::set<PropertyId> properties);
    static std::shared_ptr<ResolvedPropertyRetention> unresolved();
    bool retains(PropertyId property) const;
    void resolve(PropertyId property, bool retained);
    void freeze();

private:
    ResolvedPropertyRetention(bool retainsAll,
                              std::set<PropertyId> properties,
                              bool frozen);
    bool mRetainsAll;
    bool mFrozen;
    std::set<PropertyId> mProperties;
    std::set<PropertyId> mResolved;
};
}

class FrameViewRequest
{
public:
    explicit FrameViewRequest(FrameAddress address);

    std::set<ObjectId> objectAllowList;
    std::set<PropertyId> propertyAllowList;

private:
    FrameAddress mAddress;
    std::shared_ptr<const RecordedClassModel> mRecordedClassModel;

    friend class Capture;
    friend class TemporalStore;
};

struct ClassObjectResult
{
    bool classKnown = false;
    std::vector<ObjectId> objects;
};

class FrameView
{
public:
    FrameView(const FrameView&) = default;
    FrameView& operator=(const FrameView&) = default;
    const FrameStamp& stamp() const;
    const std::vector<ObjectId>& aliveObjects() const;
    bool isAlive(ObjectId object) const;
    PropertyResult persistentProperty(ObjectId object, PropertyId property) const;
    PropertyResult exactProperty(ObjectId object, PropertyId property) const;
    ListResult list(ObjectId object, PropertyId property) const;
    std::shared_ptr<const RecordedClassModel> recordedClassModel() const;
    const RecordedObject* recordedObject(ObjectId object) const;
    ClassObjectResult aliveObjectsOfExactClass(ClassId recordedClass) const;
    ClassObjectResult aliveObjectsOfClassOrDerived(ClassId recordedClass) const;

private:
    struct Data;
    explicit FrameView(std::shared_ptr<const Data> data);
    std::shared_ptr<const Data> mData;

    friend class TemporalStore;
};

class TemporalStore
{
public:
    TemporalStore();
    explicit TemporalStore(
        std::shared_ptr<const detail::ResolvedPropertyRetention> retention);
    ~TemporalStore();
    TemporalStore(TemporalStore&&) noexcept;
    TemporalStore& operator=(TemporalStore&&) noexcept;
    TemporalStore(const TemporalStore&) = delete;
    TemporalStore& operator=(const TemporalStore&) = delete;

    StoreResult beginFrame(const FrameAddress& address);
    StoreResult createObject(ObjectId object);
    StoreResult destroyObject(ObjectId object);
    StoreResult setProperty(ObjectId object, PropertyId property, const Bytes& bytes);
    StoreResult addToList(ObjectId object, PropertyId property, const Bytes& bytes);
    StoreResult removeFromList(ObjectId object, PropertyId property, const Bytes& bytes);
    StoreResult finishFrame(FrameStamp& publishedStamp);
    StoreResult cancelFrame();

    std::shared_ptr<const FrameView> latestView() const;
    std::shared_ptr<const FrameView> view(const FrameViewRequest& request) const;

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace data
} // namespace ovdnext
