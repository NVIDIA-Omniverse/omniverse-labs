/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "OvdData.h"

#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ovdnext::data;

namespace
{

void require(bool condition)
{
    if (!condition)
    {
        throw std::runtime_error("requirement failed");
    }
}

void requireStoreResult(StoreResult actual, StoreResult expected, const char* operation)
{
    if (actual != expected)
    {
        throw std::runtime_error(
            std::string(operation) + " returned StoreResult " +
            std::to_string(static_cast<int>(actual)) + ", expected " +
            std::to_string(static_cast<int>(expected)));
    }
}

Bytes bytes(std::initializer_list<std::uint8_t> values)
{
    return Bytes(values);
}

template <typename Action>
void finish(TemporalStore& store, const FrameAddress& address, Action action)
{
    requireStoreResult(store.beginFrame(address), StoreResult::Ok, "beginFrame");
    action();
    FrameStamp stamp;
    requireStoreResult(store.finishFrame(stamp), StoreResult::Ok, "finishFrame");
}

void pendingFramesRemainInvisibleUntilFinish()
{
    TemporalStore store;
    const FrameAddress address{1, 10, 7};
    require(store.beginFrame(address) == StoreResult::Ok);
    require(store.createObject(100) == StoreResult::Ok);
    require(!store.latestView());
    require(!store.view(FrameViewRequest(address)));

    FrameStamp stamp;
    require(store.finishFrame(stamp) == StoreResult::Ok);
    require(stamp.address == address);
    require(stamp.publication == 1);
    require(store.latestView()->isAlive(100));
}

void equalLocalFramesInDifferentScopesRemainDistinct()
{
    TemporalStore store;
    FrameStamp first;
    FrameStamp second;
    FrameStamp third;
    require(store.beginFrame(FrameAddress{1, 10, 7}) == StoreResult::Ok);
    require(store.finishFrame(first) == StoreResult::Ok);
    require(store.beginFrame(FrameAddress{2, 10, 7}) == StoreResult::Ok);
    require(store.finishFrame(second) == StoreResult::Ok);
    require(store.beginFrame(FrameAddress{2, 11, 7}) == StoreResult::Ok);
    require(store.finishFrame(third) == StoreResult::Ok);
    require(first.publication == 1);
    require(second.publication == 2);
    require(third.publication == 3);
    require(store.view(FrameViewRequest(first.address))->stamp().publication == 1);
    require(store.view(FrameViewRequest(second.address))->stamp().publication == 2);
    require(store.view(FrameViewRequest(third.address))->stamp().publication == 3);
}

void lifetimesAreHalfOpen()
{
    TemporalStore store;
    finish(store, FrameAddress{1, 1, 1}, [&] { require(store.createObject(10) == StoreResult::Ok); });
    finish(store, FrameAddress{1, 1, 2}, [] {});
    const auto beforeDestroy = store.view(FrameViewRequest(FrameAddress{1, 1, 2}));
    finish(store, FrameAddress{1, 1, 3}, [&] { require(store.destroyObject(10) == StoreResult::Ok); });
    require(beforeDestroy->isAlive(10));
    require(!store.view(FrameViewRequest(FrameAddress{1, 1, 3}))->isAlive(10));
}

void persistentAndExactPropertiesDiffer()
{
    TemporalStore store;
    finish(store, FrameAddress{1, 1, 1}, [&] {
        require(store.createObject(10) == StoreResult::Ok);
        require(store.setProperty(10, 5, bytes({1})) == StoreResult::Ok);
    });
    finish(store, FrameAddress{1, 1, 2}, [] {});
    const auto silent = store.view(FrameViewRequest(FrameAddress{1, 1, 2}));
    const PropertyResult persistent = silent->persistentProperty(10, 5);
    require(persistent.availability == Availability::Present);
    require(persistent.bytes == bytes({1}));
    require(silent->exactProperty(10, 5).availability == Availability::AbsentAtFrame);
}

void publishedViewsRemainImmutable()
{
    std::shared_ptr<const FrameView> surviving;
    {
        TemporalStore store;
        finish(store, FrameAddress{1, 1, 1}, [&] {
            require(store.createObject(10) == StoreResult::Ok);
            require(store.setProperty(10, 5, bytes({1})) == StoreResult::Ok);
        });
        const auto first = store.view(FrameViewRequest(FrameAddress{1, 1, 1}));
        finish(store, FrameAddress{1, 1, 2}, [&] {
            require(store.setProperty(10, 5, bytes({2})) == StoreResult::Ok);
        });
        finish(store, FrameAddress{1, 1, 3}, [&] { require(store.destroyObject(10) == StoreResult::Ok); });
        require(first->isAlive(10));
        require(first->persistentProperty(10, 5).bytes == bytes({1}));
        surviving = first;
    }
    require(surviving->isAlive(10));
    require(surviving->persistentProperty(10, 5).availability == Availability::Present);
    require(surviving->persistentProperty(10, 5).bytes == bytes({1}));
}

void resolvedRetentionRejectsInvalidAndLateResolution()
{
    // Catches unresolved retention accepting invalid IDs, duplicate decisions, or post-freeze mutation.
    const auto retention = detail::ResolvedPropertyRetention::unresolved();
    bool rejectedZero = false;
    try
    {
        retention->resolve(PropertyId{0}, true);
    }
    catch (const std::logic_error&)
    {
        rejectedZero = true;
    }
    require(rejectedZero);

    retention->resolve(PropertyId{1}, true);
    retention->resolve(PropertyId{2}, false);
    require(retention->retains(PropertyId{1}));
    require(!retention->retains(PropertyId{2}));

    bool rejectedDuplicate = false;
    try
    {
        retention->resolve(PropertyId{1}, false);
    }
    catch (const std::logic_error&)
    {
        rejectedDuplicate = true;
    }
    require(rejectedDuplicate);

    retention->freeze();
    bool rejectedFrozen = false;
    try
    {
        retention->resolve(PropertyId{3}, true);
    }
    catch (const std::logic_error&)
    {
        rejectedFrozen = true;
    }
    require(rejectedFrozen);
}

void availabilityStatesRemainDistinct()
{
    TemporalStore store(detail::ResolvedPropertyRetention::onlyProperties({1, 2, 3}));
    finish(store, FrameAddress{1, 1, 1}, [&] {
        require(store.createObject(10) == StoreResult::Ok);
        require(store.setProperty(10, 1, bytes({11})) == StoreResult::Ok);
        require(store.setProperty(10, 99, bytes({99})) == StoreResult::Ok);
    });

    FrameViewRequest request(FrameAddress{1, 1, 1});
    request.propertyAllowList.insert(1);
    request.propertyAllowList.insert(2);
    request.propertyAllowList.insert(99);
    const auto view = store.view(request);
    require(view->persistentProperty(10, 1).availability == Availability::Present);
    require(view->persistentProperty(10, 1).bytes == bytes({11}));
    require(view->persistentProperty(10, 2).availability == Availability::AbsentAtFrame);
    require(view->persistentProperty(10, 3).availability == Availability::NotProjected);
    require(view->persistentProperty(10, 99).availability == Availability::NotLoaded);

    TemporalStore noPayloadStore(detail::ResolvedPropertyRetention::onlyProperties({}));
    finish(noPayloadStore, FrameAddress{2, 1, 1}, [&] {
        require(noPayloadStore.createObject(20) == StoreResult::Ok);
        require(noPayloadStore.setProperty(20, 1, bytes({1})) == StoreResult::Ok);
        require(noPayloadStore.addToList(20, 2, bytes({2})) == StoreResult::Ok);
    });
    const auto noPayloadView = noPayloadStore.view(FrameViewRequest(FrameAddress{2, 1, 1}));
    require(noPayloadView->persistentProperty(20, 1).availability == Availability::NotLoaded);
    require(noPayloadView->exactProperty(20, 1).availability == Availability::NotLoaded);
    require(noPayloadView->list(20, 2).availability == Availability::NotLoaded);
}

void notLoadedWinsOverObjectAndPropertyProjection()
{
    TemporalStore store(detail::ResolvedPropertyRetention::onlyProperties({1}));
    finish(store, FrameAddress{1, 6, 1}, [&] {
        require(store.createObject(10) == StoreResult::Ok);
        require(store.setProperty(10, 99, bytes({9})) == StoreResult::Ok);
        require(store.addToList(10, 99, bytes({8})) == StoreResult::Ok);
    });

    FrameViewRequest request(FrameAddress{1, 6, 1});
    request.objectAllowList.insert(20);
    request.propertyAllowList.insert(1);
    const auto view = store.view(request);
    require(view->persistentProperty(10, 99).availability == Availability::NotLoaded);
    require(view->exactProperty(10, 99).availability == Availability::NotLoaded);
    require(view->list(10, 99).availability == Availability::NotLoaded);
}

void projectionDoesNotMutateRetainedState()
{
    TemporalStore store(detail::ResolvedPropertyRetention::onlyProperties({1, 2}));
    finish(store, FrameAddress{1, 2, 1}, [&] {
        require(store.createObject(20) == StoreResult::Ok);
        require(store.createObject(10) == StoreResult::Ok);
        require(store.setProperty(20, 1, bytes({21})) == StoreResult::Ok);
        require(store.setProperty(10, 1, bytes({11})) == StoreResult::Ok);
        require(store.addToList(20, 2, bytes({22})) == StoreResult::Ok);
        require(store.addToList(10, 2, bytes({12})) == StoreResult::Ok);
    });

    FrameViewRequest restrictedRequest(FrameAddress{1, 2, 1});
    restrictedRequest.objectAllowList.insert(20);
    restrictedRequest.propertyAllowList.insert(1);
    const auto restricted = store.view(restrictedRequest);
    require(restricted->aliveObjects() == std::vector<ObjectId>{20});
    require(restricted->isAlive(20));
    require(!restricted->isAlive(10));
    require(restricted->persistentProperty(10, 1).availability == Availability::NotProjected);
    require(restricted->exactProperty(20, 2).availability == Availability::NotProjected);
    require(restricted->list(20, 2).availability == Availability::NotProjected);

    const auto unrestricted = store.view(FrameViewRequest(FrameAddress{1, 2, 1}));
    require(unrestricted->aliveObjects() == std::vector<ObjectId>({10, 20}));
    require(unrestricted->isAlive(10));
    require(unrestricted->isAlive(20));
    require(unrestricted->persistentProperty(10, 1).bytes == bytes({11}));
    require(unrestricted->persistentProperty(20, 1).bytes == bytes({21}));
    require(unrestricted->list(10, 2).values == std::vector<Bytes>{bytes({12})});
    require(unrestricted->list(20, 2).values == std::vector<Bytes>{bytes({22})});
}

void orderedUniqueListsReconstructHistory()
{
    TemporalStore store;
    const Bytes valueA = bytes({0, 255});
    const Bytes valueB = bytes({0});
    const Bytes valueC = bytes({255});
    finish(store, FrameAddress{1, 3, 1}, [&] {
        require(store.createObject(10) == StoreResult::Ok);
        require(store.addToList(10, 7, valueA) == StoreResult::Ok);
        require(store.addToList(10, 7, valueB) == StoreResult::Ok);
        require(store.addToList(10, 7, valueA) == StoreResult::Ok);
    });
    finish(store, FrameAddress{1, 3, 2}, [&] {
        require(store.removeFromList(10, 7, valueA) == StoreResult::Ok);
        require(store.removeFromList(10, 7, valueC) == StoreResult::Ok);
    });
    finish(store, FrameAddress{1, 3, 3}, [&] {
        require(store.removeFromList(10, 7, valueB) == StoreResult::Ok);
    });

    const auto first = store.view(FrameViewRequest(FrameAddress{1, 3, 1}))->list(10, 7);
    require(first.availability == Availability::Present);
    require(first.values == std::vector<Bytes>({valueA, valueB}));
    const auto second = store.view(FrameViewRequest(FrameAddress{1, 3, 2}))->list(10, 7);
    require(second.availability == Availability::Present);
    require(second.values == std::vector<Bytes>{valueB});
    const auto third = store.view(FrameViewRequest(FrameAddress{1, 3, 3}))->list(10, 7);
    require(third.availability == Availability::Present);
    require(third.values.empty());
}

void removeThenReAddMovesListValueToTheEnd()
{
    TemporalStore store;
    const Bytes valueA = bytes({1});
    const Bytes valueB = bytes({2});
    finish(store, FrameAddress{1, 7, 1}, [&] {
        require(store.createObject(10) == StoreResult::Ok);
        require(store.addToList(10, 7, valueA) == StoreResult::Ok);
        require(store.addToList(10, 7, valueB) == StoreResult::Ok);
    });
    finish(store, FrameAddress{1, 7, 2}, [&] {
        require(store.removeFromList(10, 7, valueA) == StoreResult::Ok);
        require(store.addToList(10, 7, valueA) == StoreResult::Ok);
    });

    const auto result = store.view(FrameViewRequest(FrameAddress{1, 7, 2}))->list(10, 7);
    require(result.availability == Availability::Present);
    require(result.values == std::vector<Bytes>({valueB, valueA}));
}

void absentRemovalAsFirstOperationEstablishesPresentEmptyList()
{
    TemporalStore store;
    finish(store, FrameAddress{1, 8, 1}, [&] {
        require(store.createObject(10) == StoreResult::Ok);
        require(store.removeFromList(10, 7, bytes({1})) == StoreResult::Ok);
    });

    const auto result = store.view(FrameViewRequest(FrameAddress{1, 8, 1}))->list(10, 7);
    require(result.availability == Availability::Present);
    require(result.values.empty());
}

void rejectedCommandsDoNotChangePendingOrPublishedState()
{
    TemporalStore store(detail::ResolvedPropertyRetention::onlyProperties({1, 2}));
    FrameStamp stamp;
    require(store.finishFrame(stamp) == StoreResult::NoPendingFrame);
    require(store.cancelFrame() == StoreResult::NoPendingFrame);
    require(store.createObject(10) == StoreResult::NoPendingFrame);
    require(store.destroyObject(10) == StoreResult::NoPendingFrame);
    require(store.setProperty(10, 1, bytes({1})) == StoreResult::NoPendingFrame);
    require(store.addToList(10, 2, bytes({2})) == StoreResult::NoPendingFrame);
    require(store.removeFromList(10, 2, bytes({2})) == StoreResult::NoPendingFrame);
    require(!store.latestView());

    const FrameAddress firstAddress{1, 4, 1};
    require(store.beginFrame(firstAddress) == StoreResult::Ok);
    require(store.beginFrame(FrameAddress{9, 9, 9}) == StoreResult::FrameAlreadyPending);
    require(store.destroyObject(99) == StoreResult::ObjectUnknown);
    require(store.setProperty(99, 99, bytes({99})) == StoreResult::ObjectUnknown);
    require(store.addToList(99, 99, bytes({99})) == StoreResult::ObjectUnknown);
    require(store.removeFromList(99, 99, bytes({99})) == StoreResult::ObjectUnknown);
    require(store.createObject(10) == StoreResult::Ok);
    require(store.createObject(10) == StoreResult::ObjectAlreadyKnown);
    require(store.createObject(20) == StoreResult::Ok);
    require(store.setProperty(20, 1, bytes({21})) == StoreResult::Ok);
    require(store.addToList(20, 2, bytes({22})) == StoreResult::Ok);
    require(store.destroyObject(10) == StoreResult::Ok);
    require(store.destroyObject(10) == StoreResult::ObjectNotLive);
    require(store.setProperty(10, 99, bytes({101})) == StoreResult::ObjectNotLive);
    require(store.addToList(10, 99, bytes({102})) == StoreResult::ObjectNotLive);
    require(store.removeFromList(10, 99, bytes({103})) == StoreResult::ObjectNotLive);
    require(store.setProperty(20, 1, bytes({23})) == StoreResult::Ok);
    require(store.addToList(20, 2, bytes({24})) == StoreResult::Ok);
    require(store.finishFrame(stamp) == StoreResult::Ok);
    require(stamp.address == firstAddress);
    require(stamp.publication == 1);

    const auto first = store.view(FrameViewRequest(firstAddress));
    require(first->aliveObjects() == std::vector<ObjectId>{20});
    require(first->persistentProperty(20, 1).bytes == bytes({23}));
    require(first->list(20, 2).values == std::vector<Bytes>({bytes({22}), bytes({24})}));
    require(first->persistentProperty(20, 99).availability == Availability::NotLoaded);
    require(!first->isAlive(10));
    require(!first->isAlive(99));

    require(store.beginFrame(firstAddress) == StoreResult::FrameAddressAlreadyPublished);
    const FrameAddress secondAddress{1, 4, 2};
    require(store.beginFrame(secondAddress) == StoreResult::Ok);
    require(store.createObject(10) == StoreResult::ObjectAlreadyKnown);
    require(store.setProperty(20, 1, bytes({25})) == StoreResult::Ok);
    require(store.finishFrame(stamp) == StoreResult::Ok);
    require(stamp.address == secondAddress);
    require(stamp.publication == 2);
    require(store.view(FrameViewRequest(firstAddress))->persistentProperty(20, 1).bytes == bytes({23}));
    require(store.view(FrameViewRequest(secondAddress))->persistentProperty(20, 1).bytes == bytes({25}));
    require(store.cancelFrame() == StoreResult::NoPendingFrame);
}

void canceledFrameCanBeRetriedWithoutConsumingPublication()
{
    TemporalStore store;
    const FrameAddress address{1, 5, 1};
    require(store.beginFrame(address) == StoreResult::Ok);
    require(store.createObject(1) == StoreResult::Ok);
    require(store.cancelFrame() == StoreResult::Ok);
    require(!store.latestView());
    require(!store.view(FrameViewRequest(address)));

    require(store.beginFrame(address) == StoreResult::Ok);
    require(store.createObject(2) == StoreResult::Ok);
    FrameStamp stamp;
    require(store.finishFrame(stamp) == StoreResult::Ok);
    require(stamp.address == address);
    require(stamp.publication == 1);
    const auto view = store.view(FrameViewRequest(address));
    require(view->isAlive(2));
    require(!view->isAlive(1));
}

} // namespace

int main()
{
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"pendingFramesRemainInvisibleUntilFinish", pendingFramesRemainInvisibleUntilFinish},
        {"equalLocalFramesInDifferentScopesRemainDistinct", equalLocalFramesInDifferentScopesRemainDistinct},
        {"lifetimesAreHalfOpen", lifetimesAreHalfOpen},
        {"persistentAndExactPropertiesDiffer", persistentAndExactPropertiesDiffer},
        {"publishedViewsRemainImmutable", publishedViewsRemainImmutable},
        {"resolvedRetentionRejectsInvalidAndLateResolution", resolvedRetentionRejectsInvalidAndLateResolution},
        {"availabilityStatesRemainDistinct", availabilityStatesRemainDistinct},
        {"notLoadedWinsOverObjectAndPropertyProjection", notLoadedWinsOverObjectAndPropertyProjection},
        {"projectionDoesNotMutateRetainedState", projectionDoesNotMutateRetainedState},
        {"orderedUniqueListsReconstructHistory", orderedUniqueListsReconstructHistory},
        {"removeThenReAddMovesListValueToTheEnd", removeThenReAddMovesListValueToTheEnd},
        {"absentRemovalAsFirstOperationEstablishesPresentEmptyList", absentRemovalAsFirstOperationEstablishesPresentEmptyList},
        {"rejectedCommandsDoNotChangePendingOrPublishedState", rejectedCommandsDoNotChangePendingOrPublishedState},
        {"canceledFrameCanBeRetriedWithoutConsumingPublication", canceledFrameCanBeRetriedWithoutConsumingPublication},
    };

    int failures = 0;
    for (const auto& test : tests)
    {
        try
        {
            test.second();
            std::cout << "PASS " << test.first << '\n';
        }
        catch (const std::exception& error)
        {
            ++failures;
            std::cout << "FAIL " << test.first << ": " << error.what() << '\n';
        }
    }

    return failures == 0 ? 0 : 1;
}
