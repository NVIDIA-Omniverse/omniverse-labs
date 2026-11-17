/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "OvdIngest.h"
#include "OvdRecordedModel.h"

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

constexpr std::uint64_t kNeutralClassHandle = UINT64_C(0xfffffffffffffff0);

void require(bool condition)
{
    if (!condition)
    {
        throw std::runtime_error("requirement failed");
    }
}

void require(const std::shared_ptr<const RecordedClassModel>& condition)
{
    if (!condition)
    {
        throw std::runtime_error("requirement failed");
    }
}

Bytes bytes(std::initializer_list<std::uint8_t> values)
{
    return Bytes(values);
}

Command command(CommandKind kind)
{
    Command result{};
    result.kind = kind;
    return result;
}

void consume(CommandIngestor& ingestor, const Command& value)
{
    require(ingestor.consume(value));
}

void registerScalar(CommandIngestor& ingestor, std::uint64_t handle)
{
    Command value = command(CommandKind::RegisterAttribute);
    value.ownerClassHandle = kNeutralClassHandle;
    value.attributeHandle = handle;
    value.attributeKind = RecordedAttributeKind::Value;
    value.dataType = RecordedDataType::UInt8;
    value.hasElementCount = true;
    consume(ingestor, value);
}

void registerList(CommandIngestor& ingestor, std::uint64_t handle)
{
    Command value = command(CommandKind::RegisterAttribute);
    value.ownerClassHandle = kNeutralClassHandle;
    value.attributeHandle = handle;
    value.attributeKind = RecordedAttributeKind::UniqueList;
    value.dataType = RecordedDataType::UInt8;
    consume(ingestor, value);
}

void create(CommandIngestor& ingestor, std::uint64_t object, std::uint64_t context)
{
    Command value = command(CommandKind::CreateObject);
    value.objectHandle = object;
    value.classHandle = kNeutralClassHandle;
    value.context = context;
    consume(ingestor, value);
}

void registerClass(CommandIngestor& ingestor, std::uint64_t handle, const char* name)
{
    Command value = command(CommandKind::RegisterClass);
    value.classHandle = handle;
    value.name = name;
    consume(ingestor, value);
}

void registerClass(CommandIngestor& ingestor, std::uint64_t handle,
    std::uint64_t baseHandle, const char* name)
{
    Command value = command(CommandKind::RegisterClass);
    value.classHandle = handle;
    value.baseClassHandle = baseHandle;
    value.name = name;
    consume(ingestor, value);
}

void registerNeutralClass(CommandIngestor& ingestor)
{
    registerClass(ingestor, kNeutralClassHandle, "NeutralObject");
}

void registerAttribute(CommandIngestor& ingestor, std::uint64_t attributeHandle,
    std::uint64_t ownerClassHandle, RecordedAttributeKind kind, RecordedDataType dataType,
    bool hasElementCount, std::uint32_t elementCount,
    std::uint64_t attributeClassHandle, std::uint64_t flagsEnumClassHandle,
    std::uint32_t enumValue, const char* name)
{
    Command value = command(CommandKind::RegisterAttribute);
    value.attributeHandle = attributeHandle;
    value.ownerClassHandle = ownerClassHandle;
    value.attributeKind = kind;
    value.dataType = dataType;
    value.hasElementCount = hasElementCount;
    value.elementCount = elementCount;
    value.attributeClassHandle = attributeClassHandle;
    value.flagsEnumClassHandle = flagsEnumClassHandle;
    value.enumValue = enumValue;
    value.name = name;
    consume(ingestor, value);
}

void createWithClass(CommandIngestor& ingestor, std::uint64_t object,
    std::uint64_t classHandle, std::uint64_t context)
{
    Command value = command(CommandKind::CreateObject);
    value.objectHandle = object;
    value.classHandle = classHandle;
    value.context = context;
    consume(ingestor, value);
}

void start(CommandIngestor& ingestor, std::uint64_t context, std::uint64_t frame)
{
    Command value = command(CommandKind::StartFrame);
    value.context = context;
    value.frame = frame;
    consume(ingestor, value);
}

void stop(CommandIngestor& ingestor, std::uint64_t context, std::uint64_t frame)
{
    Command value = command(CommandKind::StopFrame);
    value.context = context;
    value.frame = frame;
    consume(ingestor, value);
}

void set(CommandIngestor& ingestor, std::uint64_t object, std::uint64_t attribute,
    std::initializer_list<std::uint8_t> payload, std::uint64_t context)
{
    const Bytes value(payload);
    Command commandValue = command(CommandKind::SetProperty);
    commandValue.objectHandle = object;
    commandValue.attributeHandle = attribute;
    commandValue.context = context;
    commandValue.payload = value.data();
    commandValue.payloadSize = value.size();
    consume(ingestor, commandValue);
}

void list(CommandIngestor& ingestor, CommandKind kind, std::uint64_t object,
    std::uint64_t attribute, std::initializer_list<std::uint8_t> payload, std::uint64_t context)
{
    const Bytes value(payload);
    Command commandValue = command(kind);
    commandValue.objectHandle = object;
    commandValue.attributeHandle = attribute;
    commandValue.context = context;
    commandValue.payload = value.data();
    commandValue.payloadSize = value.size();
    consume(ingestor, commandValue);
}

Capture finish(CommandIngestor& ingestor)
{
    require(ingestor.finish());
    std::unique_ptr<Capture> capture = ingestor.takeCapture();
    require(capture != nullptr);
    return std::move(*capture);
}

void preFrameCommandsAreDeferredIntoFirstPublication()
{
    // Catches a regression that drops deferred lifecycle or mutations before beginFrame().
    CommandIngestor ingestor(LoadPlan::allProperties());
    registerNeutralClass(ingestor);
    registerScalar(ingestor, 90);
    create(ingestor, 77, 3);
    set(ingestor, 77, 90, {0x10, 0x20}, 3);
    start(ingestor, 3, 12);
    stop(ingestor, 3, 12);

    Capture capture = finish(ingestor);
    require(capture.frames().size() == 1);
    const auto view = capture.view(0);
    require(view != nullptr);
    require(view->stamp().address == FrameAddress{0, 3, 12});
    require(view->isAlive(1));
    require(view->persistentProperty(1, 1).bytes == bytes({0x10, 0x20}));
}

void frameIsNotPublishedUntilMatchingStop()
{
    // Catches a regression that publishes a directory entry at START_FRAME.
    CommandIngestor ingestor(LoadPlan::allProperties());
    start(ingestor, 7, 11);
    require(ingestor.frames().empty());
    stop(ingestor, 7, 11);
    require(ingestor.frames().size() == 1);
    Capture capture = finish(ingestor);
    require(capture.frames().size() == 1);
}

void scalarAndListCommandsRetainOrderedBytes()
{
    // Catches a regression that aliases borrowed payloads or loses list add/remove ordering.
    CommandIngestor ingestor(LoadPlan::allProperties());
    registerNeutralClass(ingestor);
    registerScalar(ingestor, 10);
    registerList(ingestor, 20);
    create(ingestor, 88, 1);
    start(ingestor, 1, 1);
    set(ingestor, 88, 10, {0x01, 0xff}, 1);
    list(ingestor, CommandKind::AddToList, 88, 20, {0xa0}, 1);
    list(ingestor, CommandKind::AddToList, 88, 20, {0xb0, 0x00}, 1);
    list(ingestor, CommandKind::RemoveFromList, 88, 20, {0xa0}, 1);
    stop(ingestor, 1, 1);

    Capture capture = finish(ingestor);
    const auto view = capture.view(0);
    require(view->persistentProperty(1, 1).bytes == bytes({0x01, 0xff}));
    require(view->list(1, 2).values == std::vector<Bytes>{bytes({0xb0, 0x00})});
}

void excludedPayloadsPreserveLifecycleAndCountFilteredBytes()
{
    // Catches a regression that copies excluded payloads or loses lifecycle when no property is retained.
    CommandIngestor ingestor(LoadPlan::onlyRecordedAttributes({}));
    registerNeutralClass(ingestor);
    registerScalar(ingestor, 50);
    create(ingestor, 9, 2);
    start(ingestor, 2, 3);
    set(ingestor, 9, 50, {0x01, 0x02, 0x03}, 2);
    stop(ingestor, 2, 3);

    Capture capture = finish(ingestor);
    require(capture.stats().retainedPayloadBytes == 0);
    require(capture.stats().filteredPayloadBytes == 3);
    const auto view = capture.view(0);
    require(view->isAlive(1));
    require(view->persistentProperty(1, 1).availability == Availability::NotLoaded);
    require(capture.recordedClassModel()->classes().size() == 1);
    require(capture.recordedClassModel()->attributes().size() == 1);
    require(capture.recordedClassModel()->objects().size() == 1);
    require(view->recordedObject(ObjectId{1}) != nullptr);
}

void excludedMutationStillRejectsMissingPayloadStorage()
{
    // Catches filtering allowing malformed non-empty payloads to bypass input validation.
    CommandIngestor ingestor(LoadPlan::onlyRecordedAttributes({}));
    registerNeutralClass(ingestor);
    registerScalar(ingestor, 50);
    create(ingestor, 9, 2);
    start(ingestor, 2, 3);

    Command malformed = command(CommandKind::SetProperty);
    malformed.objectHandle = 9;
    malformed.attributeHandle = 50;
    malformed.context = 2;
    malformed.payload = nullptr;
    malformed.payloadSize = 3;
    require(!ingestor.consume(malformed));
    require(ingestor.error() == IngestError::StoreFailure);
    require(!ingestor.finish());
    require(!ingestor.takeCapture());
}

void recordedSelectorsAreExactPairsWithoutInheritanceInference()
{
    // Catches matching attribute names alone or inheriting selector retention into derived owners.
    CommandIngestor ingestor(LoadPlan::onlyRecordedAttributes({
        RecordedAttributeSelector{"ProducerBody", "position"}
    }));
    registerClass(ingestor, 10, 0, "ProducerBody");
    registerClass(ingestor, 20, 10, "DerivedBody");
    registerAttribute(ingestor, 100, 10, RecordedAttributeKind::Value,
        RecordedDataType::UInt8, true, 1, 0, 0, 0, "position");
    registerAttribute(ingestor, 200, 20, RecordedAttributeKind::Value,
        RecordedDataType::UInt8, true, 1, 0, 0, 0, "position");
    createWithClass(ingestor, 1, 10, 1);
    createWithClass(ingestor, 2, 20, 1);
    start(ingestor, 1, 1);
    set(ingestor, 1, 100, {0x11}, 1);
    set(ingestor, 2, 200, {0x22, 0x23}, 1);
    stop(ingestor, 1, 1);

    Capture capture = finish(ingestor);
    require(capture.stats().retainedPayloadBytes == 1);
    require(capture.stats().filteredPayloadBytes == 2);
    require(capture.view(0)->persistentProperty(ObjectId{1}, PropertyId{1}).availability ==
        Availability::Present);
    require(capture.view(0)->persistentProperty(ObjectId{2}, PropertyId{2}).availability ==
        Availability::NotLoaded);
}

void recordedNameFilteringResolvesEveryDeclarationCycle()
{
    // Catches filtering by capture PropertyId or source handle instead of the exact declaration-name pair.
    const LoadPlan plan = LoadPlan::onlyRecordedAttributes({
        RecordedAttributeSelector{"ProducerBody", "position"}
    });
    CommandIngestor ingestor(plan);

    registerClass(ingestor, 100, "ProducerBody");
    registerAttribute(ingestor, 1000, 100, RecordedAttributeKind::Value,
        RecordedDataType::UInt8, true, 1, 0, 0, 0, "position");
    registerAttribute(ingestor, 1001, 100, RecordedAttributeKind::Value,
        RecordedDataType::UInt8, true, 1, 0, 0, 0, "velocity");
    createWithClass(ingestor, 10, 100, 7);
    start(ingestor, 7, 1);
    set(ingestor, 10, 1000, {0x11, 0x12}, 7);
    set(ingestor, 10, 1001, {0x21, 0x22, 0x23}, 7);
    stop(ingestor, 7, 1);

    registerClass(ingestor, 200, "ProducerBody");
    registerAttribute(ingestor, 2000, 200, RecordedAttributeKind::Value,
        RecordedDataType::UInt8, true, 1, 0, 0, 0, "position");
    createWithClass(ingestor, 20, 200, 7);
    start(ingestor, 7, 2);
    set(ingestor, 20, 2000, {0x31}, 7);
    stop(ingestor, 7, 2);

    Capture capture = finish(ingestor);
    require(capture.recordedClassModel()->classes().size() == 2);
    require(capture.recordedClassModel()->attributes().size() == 3);
    require(capture.stats().retainedPayloadBytes == 3);
    require(capture.stats().filteredPayloadBytes == 3);
    require(capture.view(0)->persistentProperty(ObjectId{1}, PropertyId{1}).availability ==
        Availability::Present);
    require(capture.view(0)->persistentProperty(ObjectId{1}, PropertyId{1}).bytes ==
        bytes({0x11, 0x12}));
    require(capture.view(0)->persistentProperty(ObjectId{1}, PropertyId{2}).availability ==
        Availability::NotLoaded);
    require(capture.view(1)->persistentProperty(ObjectId{2}, PropertyId{3}).availability ==
        Availability::Present);
    require(capture.view(1)->persistentProperty(ObjectId{2}, PropertyId{3}).bytes == bytes({0x31}));

    FrameViewRequest projectedRequest(capture.frames()[1].address);
    projectedRequest.propertyAllowList.insert(PropertyId{1});
    projectedRequest.propertyAllowList.insert(PropertyId{2});
    const auto projected = capture.view(projectedRequest);
    require(projected->persistentProperty(ObjectId{2}, PropertyId{3}).availability ==
        Availability::NotProjected);
    require(capture.view(1)->persistentProperty(ObjectId{2}, PropertyId{3}).availability ==
        Availability::Present);
}

void frameViewsQueryProjectedAliveObjectsByRecordedClass()
{
    // Catches class queries walking the capture-wide catalog instead of the alive projected subset.
    std::shared_ptr<const FrameView> surviving;
    {
        CommandIngestor ingestor(LoadPlan::allProperties());
        registerClass(ingestor, 10, 0, "Base");
        registerClass(ingestor, 20, 10, "Child");
        registerClass(ingestor, 30, 20, "Grandchild");
        registerClass(ingestor, 40, 0, "Unrelated");
        createWithClass(ingestor, 101, 10, 1);
        createWithClass(ingestor, 102, 20, 1);
        createWithClass(ingestor, 103, 30, 1);
        createWithClass(ingestor, 104, 40, 1);
        start(ingestor, 1, 1);
        stop(ingestor, 1, 1);

        Command destroyChild = command(CommandKind::DestroyObject);
        destroyChild.objectHandle = 102;
        destroyChild.context = 1;
        consume(ingestor, destroyChild);
        start(ingestor, 1, 2);
        stop(ingestor, 1, 2);

        Capture capture = finish(ingestor);
        const auto full = capture.view(0);
        require(full->recordedClassModel() == capture.recordedClassModel());
        require(full->aliveObjectsOfExactClass(ClassId{2}).classKnown);
        require(full->aliveObjectsOfExactClass(ClassId{2}).objects ==
            std::vector<ObjectId>({ObjectId{2}}));
        require(full->aliveObjectsOfClassOrDerived(ClassId{1}).objects ==
            std::vector<ObjectId>({ObjectId{1}, ObjectId{2}, ObjectId{3}}));
        const ClassObjectResult unknown = full->aliveObjectsOfExactClass(ClassId{999});
        require(!unknown.classKnown && unknown.objects.empty());
        require(full->recordedObject(ObjectId{2})->recordedClass == ClassId{2});
        require(full->recordedObject(ObjectId{999}) == nullptr);
        require(capture.view(1)->aliveObjectsOfExactClass(ClassId{2}).objects.empty());
        require(capture.view(1)->recordedObject(ObjectId{2})->recordedClass == ClassId{2});

        FrameViewRequest projectedRequest(capture.frames()[0].address);
        projectedRequest.objectAllowList.insert(ObjectId{3});
        projectedRequest.objectAllowList.insert(ObjectId{4});
        surviving = capture.view(projectedRequest);
        require(surviving->aliveObjectsOfExactClass(ClassId{2}).objects.empty());
        require(surviving->aliveObjectsOfClassOrDerived(ClassId{1}).objects ==
            std::vector<ObjectId>({ObjectId{3}}));
        require(surviving->recordedObject(ObjectId{1})->recordedClass == ClassId{1});
    }

    require(surviving->recordedClassModel()->findClass(ClassId{3}) != nullptr);
    require(surviving->recordedObject(ObjectId{1})->recordedClass == ClassId{1});
    require(surviving->aliveObjectsOfClassOrDerived(ClassId{1}).objects ==
        std::vector<ObjectId>({ObjectId{3}}));
}

void attributesAllocateMonotonicPropertyIds()
{
    // Catches a regression that leaks sparse producer handles into capture-owned property IDs.
    CommandIngestor ingestor(LoadPlan::allProperties());
    registerNeutralClass(ingestor);
    registerScalar(ingestor, 41);
    create(ingestor, 1, 1);
    start(ingestor, 1, 1);
    set(ingestor, 1, 41, {0x4a}, 1);
    stop(ingestor, 1, 1);

    Capture capture = finish(ingestor);
    require(capture.view(0)->persistentProperty(1, 1).bytes == bytes({0x4a}));
}

void equalLocalFramesAcrossContextsAndSegmentsRemainDistinct()
{
    // Catches a regression that omits context or segment from a published frame address.
    CommandIngestor ingestor(LoadPlan::allProperties());
    registerClass(ingestor, 40, "PxOmniPvdMetaData");
    createWithClass(ingestor, 1, 40, 4);
    start(ingestor, 4, 8);
    stop(ingestor, 4, 8);
    start(ingestor, 5, 8);
    stop(ingestor, 5, 8);
    createWithClass(ingestor, 2, 40, 4);
    start(ingestor, 4, 8);
    stop(ingestor, 4, 8);

    Capture capture = finish(ingestor);
    require(capture.frames().size() == 3);
    require(capture.frames()[0].address == FrameAddress{0, 4, 8});
    require(capture.frames()[1].address == FrameAddress{0, 5, 8});
    require(capture.frames()[2].address == FrameAddress{1, 4, 8});
    require(capture.view(FrameAddress{0, 4, 8})->stamp().publication == 1);
    require(capture.view(FrameAddress{0, 5, 8})->stamp().publication == 2);
    require(capture.view(FrameAddress{1, 4, 8})->stamp().publication == 3);
}

void recreatedExternalHandlesBecomeNewObjectGenerations()
{
    // Catches a regression that reuses the external producer handle as ObjectId.
    CommandIngestor ingestor(LoadPlan::allProperties());
    registerNeutralClass(ingestor);
    create(ingestor, 99, 1);
    start(ingestor, 1, 1);
    stop(ingestor, 1, 1);
    create(ingestor, 99, 1);
    start(ingestor, 1, 2);
    stop(ingestor, 1, 2);

    Capture capture = finish(ingestor);
    require(capture.view(0)->isAlive(1));
    require(!capture.view(1)->isAlive(1));
    require(capture.view(1)->isAlive(2));
}

void metadataSegmentTransitionClosesPriorLiveGenerations()
{
    // Catches a regression that carries old-segment live objects into the next recording segment.
    CommandIngestor ingestor(LoadPlan::allProperties());
    registerClass(ingestor, 40, "PxOmniPvdMetaData");
    registerNeutralClass(ingestor);
    createWithClass(ingestor, 10, 40, 1);
    create(ingestor, 20, 1);
    start(ingestor, 1, 1);
    stop(ingestor, 1, 1);
    createWithClass(ingestor, 11, 40, 1);
    start(ingestor, 1, 1);
    stop(ingestor, 1, 1);

    Capture capture = finish(ingestor);
    require(capture.frames()[1].address.segment == 1);
    require(capture.view(0)->isAlive(2));
    require(!capture.view(1)->isAlive(2));
}

void repeatedClassRegistrationProvidesLegacySegmentFallback()
{
    // Catches a regression that cannot separate legacy recordings without metadata objects.
    CommandIngestor ingestor(LoadPlan::allProperties());
    registerClass(ingestor, 71, "LegacyClass");
    start(ingestor, 3, 1);
    stop(ingestor, 3, 1);
    registerClass(ingestor, 72, "LegacyClass");
    start(ingestor, 3, 1);
    stop(ingestor, 3, 1);

    Capture capture = finish(ingestor);
    require(capture.recordedClassModel()->classes().size() == 2);
    require(capture.frames()[0].address.segment == 0);
    require(capture.frames()[1].address.segment == 1);
}

void recordedClassModelPreservesIdentityHierarchyAndExactNames()
{
    // Catches source handles being collapsed by name or hierarchy links being omitted.
    CommandIngestor ingestor(LoadPlan::allProperties());
    registerClass(ingestor, 100, 0, "ProducerBase");
    registerClass(ingestor, 200, 100, "ProducerDerived");
    registerClass(ingestor, 300, 0, "ProducerBase");

    Capture capture = finish(ingestor);
    const std::shared_ptr<const RecordedClassModel> model = capture.recordedClassModel();
    require(model);
    require(model->classes().size() == 3);
    require(model->classes()[0].id == ClassId{1});
    require(model->classes()[0].name == "ProducerBase");
    require(model->classes()[1].baseClass == ClassId{1});
    require(model->classes()[2].name == "ProducerBase");
    require(model->findClassesByName("ProducerBase") ==
            std::vector<ClassId>({ClassId{1}, ClassId{3}}));
    require(model->isSameOrDerived(ClassId{2}, ClassId{1}));
    require(!model->isSameOrDerived(ClassId{1}, ClassId{2}));
    require(model->findClass(ClassId{999}) == nullptr);
}

void recordedDeclarationsPreserveAttributesObjectsAndInheritance()
{
    // Catches descriptor loss, source-ID leakage, unstable object generations, and wrong inherited order.
    CommandIngestor ingestor(LoadPlan::allProperties());
    registerClass(ingestor, 100, 0, "ProducerBase");
    registerClass(ingestor, 200, 100, "ProducerChild");
    registerClass(ingestor, 300, 0, "ProducerEnum");

    registerAttribute(ingestor, 1000, 100, RecordedAttributeKind::Value,
        RecordedDataType::Float32, true, 3, 0, 0, 0, "shared");
    registerAttribute(ingestor, 1001, 100, RecordedAttributeKind::ClassAttribute,
        RecordedDataType::None, false, 0, 200, 0, 0, "nested");
    registerAttribute(ingestor, 1002, 200, RecordedAttributeKind::UniqueList,
        RecordedDataType::ObjectHandle, false, 0, 0, 0, 0, "list");
    registerAttribute(ingestor, 1003, 300, RecordedAttributeKind::EnumConstant,
        RecordedDataType::EnumValue, false, 0, 0, 0, 8, "Eight");
    registerAttribute(ingestor, 1004, 100, RecordedAttributeKind::Value,
        RecordedDataType::FlagsWord, false, 0, 0, 300, 0, "shared");

    char firstName[] = "FirstGeneration";
    Command firstCreate = command(CommandKind::CreateObject);
    firstCreate.classHandle = 100;
    firstCreate.objectHandle = 77;
    firstCreate.context = 0;
    firstCreate.name = firstName;
    consume(ingestor, firstCreate);
    firstName[0] = 'X';
    start(ingestor, 0, 1);
    stop(ingestor, 0, 1);

    Command destroyFirst = command(CommandKind::DestroyObject);
    destroyFirst.objectHandle = 77;
    destroyFirst.context = 0;
    consume(ingestor, destroyFirst);
    Command secondCreate = command(CommandKind::CreateObject);
    secondCreate.classHandle = 200;
    secondCreate.objectHandle = 77;
    secondCreate.context = 0;
    secondCreate.name = "SecondGeneration";
    consume(ingestor, secondCreate);
    set(ingestor, 77, 1000, {0x2a}, 0);
    start(ingestor, 0, 2);
    stop(ingestor, 0, 2);

    Capture capture = finish(ingestor);
    const std::shared_ptr<const RecordedClassModel> model = capture.recordedClassModel();
    require(model);
    require(model->attributes().size() == 5);

    const RecordedAttribute& value = model->attributes()[0];
    require(value.kind == RecordedAttributeKind::Value);
    require(value.dataType == RecordedDataType::Float32);
    require(value.hasElementCount && value.elementCount == 3);
    require(value.valueProperty == PropertyId{1});

    const RecordedAttribute& nested = model->attributes()[1];
    require(nested.kind == RecordedAttributeKind::ClassAttribute);
    require(nested.dataType == RecordedDataType::None);
    require(nested.attributeClass == ClassId{2});
    require(nested.valueProperty == PropertyId{0});

    const RecordedAttribute& listAttribute = model->attributes()[2];
    require(listAttribute.kind == RecordedAttributeKind::UniqueList);
    require(listAttribute.dataType == RecordedDataType::ObjectHandle);
    require(!listAttribute.hasElementCount);
    require(listAttribute.valueProperty == PropertyId{2});

    const RecordedAttribute& constant = model->attributes()[3];
    require(constant.kind == RecordedAttributeKind::EnumConstant);
    require(constant.dataType == RecordedDataType::EnumValue);
    require(constant.enumValue == 8);
    require(constant.valueProperty == PropertyId{0});

    const RecordedAttribute& flags = model->attributes()[4];
    require(flags.kind == RecordedAttributeKind::Value);
    require(flags.dataType == RecordedDataType::FlagsWord);
    require(flags.flagsEnumClass == ClassId{3});
    require(!flags.hasElementCount);
    require(flags.valueProperty == PropertyId{3});

    require(model->findAttribute(AttributeId{3}) == &model->attributes()[2]);
    require(model->findAttribute(AttributeId{99}) == nullptr);
    require(model->findDeclaredAttributesByName(ClassId{1}, "shared") ==
        std::vector<AttributeId>({AttributeId{1}, AttributeId{5}}));
    require(model->findClass(ClassId{1})->declaredAttributes ==
        std::vector<AttributeId>({AttributeId{1}, AttributeId{2}, AttributeId{5}}));
    require(model->visibleAttributes(ClassId{2}) ==
        std::vector<AttributeId>({AttributeId{1}, AttributeId{2}, AttributeId{5}, AttributeId{3}}));
    require(model->visibleAttributes(ClassId{99}).empty());

    require(model->objects().size() == 2);
    require(model->objects()[0].id == ObjectId{1});
    require(model->objects()[0].recordedClass == ClassId{1});
    require(model->objects()[0].recordedName == "FirstGeneration");
    require(model->objects()[1].id == ObjectId{2});
    require(model->objects()[1].recordedClass == ClassId{2});
    require(model->objects()[1].recordedName == "SecondGeneration");
    require(model->findObject(ObjectId{2}) == &model->objects()[1]);
    require(model->findObject(ObjectId{99}) == nullptr);
    require(capture.view(1)->persistentProperty(ObjectId{2}, PropertyId{1}).bytes == bytes({0x2a}));
}

void declarationValidationFailsClosedWithExactErrors()
{
    // Each row catches one declaration-matrix branch accepting malformed or unresolved input.
    struct FailureCase
    {
        Command declaration;
        IngestError expected;
        bool duplicate;
    };
    std::vector<FailureCase> cases;

    Command unknownOwner = command(CommandKind::RegisterAttribute);
    unknownOwner.attributeHandle = 1;
    unknownOwner.ownerClassHandle = 999;
    unknownOwner.attributeKind = RecordedAttributeKind::Value;
    unknownOwner.dataType = RecordedDataType::Int32;
    unknownOwner.hasElementCount = true;
    cases.push_back({unknownOwner, IngestError::UnknownAttributeOwner, false});

    Command duplicate = unknownOwner;
    duplicate.ownerClassHandle = 10;
    cases.push_back({duplicate, IngestError::DuplicateAttributeRegistration, true});

    Command unknownNested = command(CommandKind::RegisterAttribute);
    unknownNested.attributeHandle = 2;
    unknownNested.ownerClassHandle = 10;
    unknownNested.attributeKind = RecordedAttributeKind::ClassAttribute;
    unknownNested.dataType = RecordedDataType::None;
    unknownNested.attributeClassHandle = 999;
    cases.push_back({unknownNested, IngestError::UnknownRelatedClass, false});

    Command unknownFlags = command(CommandKind::RegisterAttribute);
    unknownFlags.attributeHandle = 3;
    unknownFlags.ownerClassHandle = 10;
    unknownFlags.attributeKind = RecordedAttributeKind::Value;
    unknownFlags.dataType = RecordedDataType::FlagsWord;
    unknownFlags.flagsEnumClassHandle = 999;
    cases.push_back({unknownFlags, IngestError::UnknownRelatedClass, false});

    Command valueWithoutCount = command(CommandKind::RegisterAttribute);
    valueWithoutCount.attributeHandle = 4;
    valueWithoutCount.ownerClassHandle = 10;
    valueWithoutCount.attributeKind = RecordedAttributeKind::Value;
    valueWithoutCount.dataType = RecordedDataType::Float32;
    cases.push_back({valueWithoutCount, IngestError::InvalidAttributeDeclaration, false});

    Command valueNone = valueWithoutCount;
    valueNone.attributeHandle = 5;
    valueNone.dataType = RecordedDataType::None;
    valueNone.hasElementCount = true;
    cases.push_back({valueNone, IngestError::InvalidAttributeDeclaration, false});

    Command listWithCount = command(CommandKind::RegisterAttribute);
    listWithCount.attributeHandle = 6;
    listWithCount.ownerClassHandle = 10;
    listWithCount.attributeKind = RecordedAttributeKind::UniqueList;
    listWithCount.dataType = RecordedDataType::UInt64;
    listWithCount.hasElementCount = true;
    listWithCount.elementCount = 1;
    cases.push_back({listWithCount, IngestError::InvalidAttributeDeclaration, false});

    Command enumWithRelationship = command(CommandKind::RegisterAttribute);
    enumWithRelationship.attributeHandle = 7;
    enumWithRelationship.ownerClassHandle = 10;
    enumWithRelationship.attributeKind = RecordedAttributeKind::EnumConstant;
    enumWithRelationship.dataType = RecordedDataType::EnumValue;
    enumWithRelationship.attributeClassHandle = 20;
    cases.push_back({enumWithRelationship, IngestError::InvalidAttributeDeclaration, false});

    Command valueWithUnknownDataType = command(CommandKind::RegisterAttribute);
    valueWithUnknownDataType.attributeHandle = 8;
    valueWithUnknownDataType.ownerClassHandle = 10;
    valueWithUnknownDataType.attributeKind = RecordedAttributeKind::Value;
    valueWithUnknownDataType.dataType = static_cast<RecordedDataType>(999);
    valueWithUnknownDataType.hasElementCount = true;
    cases.push_back({valueWithUnknownDataType, IngestError::InvalidAttributeDeclaration, false});

    Command listWithUnknownDataType = command(CommandKind::RegisterAttribute);
    listWithUnknownDataType.attributeHandle = 9;
    listWithUnknownDataType.ownerClassHandle = 10;
    listWithUnknownDataType.attributeKind = RecordedAttributeKind::UniqueList;
    listWithUnknownDataType.dataType = static_cast<RecordedDataType>(999);
    cases.push_back({listWithUnknownDataType, IngestError::InvalidAttributeDeclaration, false});

    Command unknownKind = command(CommandKind::RegisterAttribute);
    unknownKind.attributeHandle = 10;
    unknownKind.ownerClassHandle = 10;
    unknownKind.attributeKind = static_cast<RecordedAttributeKind>(999);
    unknownKind.dataType = RecordedDataType::Float32;
    unknownKind.hasElementCount = true;
    cases.push_back({unknownKind, IngestError::InvalidAttributeDeclaration, false});

    Command unknownClassCreate = command(CommandKind::CreateObject);
    unknownClassCreate.classHandle = 999;
    unknownClassCreate.objectHandle = 1;
    cases.push_back({unknownClassCreate, IngestError::UnknownClass, false});

    for (const FailureCase& failure : cases)
    {
        CommandIngestor ingestor(LoadPlan::allProperties());
        registerClass(ingestor, 10, "Owner");
        registerClass(ingestor, 20, "Related");
        if (failure.duplicate)
        {
            consume(ingestor, failure.declaration);
        }
        require(!ingestor.consume(failure.declaration));
        require(ingestor.error() == failure.expected);
        require(!ingestor.takeCapture());
    }
}

void nonScalarDeclarationsRejectSetMutation()
{
    // Catches SetProperty accepting list storage or terminal metadata declarations.
    CommandIngestor listMutation(LoadPlan::allProperties());
    registerNeutralClass(listMutation);
    registerList(listMutation, 1);
    create(listMutation, 1, 0);
    Command setList = command(CommandKind::SetProperty);
    setList.objectHandle = 1;
    setList.attributeHandle = 1;
    setList.context = 0;
    require(!listMutation.consume(setList));
    require(listMutation.error() == IngestError::PropertyKindMismatch);

    CommandIngestor classMutation(LoadPlan::allProperties());
    registerClass(classMutation, 10, "Owner");
    registerClass(classMutation, 20, "Nested");
    registerAttribute(classMutation, 2, 10, RecordedAttributeKind::ClassAttribute,
        RecordedDataType::None, false, 0, 20, 0, 0, "nested");
    createWithClass(classMutation, 1, 10, 0);
    Command setClassAttribute = command(CommandKind::SetProperty);
    setClassAttribute.objectHandle = 1;
    setClassAttribute.attributeHandle = 2;
    setClassAttribute.context = 0;
    require(!classMutation.consume(setClassAttribute));
    require(classMutation.error() == IngestError::PropertyKindMismatch);

    CommandIngestor enumMutation(LoadPlan::allProperties());
    registerClass(enumMutation, 10, "ProducerEnum");
    registerAttribute(enumMutation, 3, 10, RecordedAttributeKind::EnumConstant,
        RecordedDataType::EnumValue, false, 0, 0, 0, 7, "Seven");
    createWithClass(enumMutation, 1, 10, 0);
    Command setEnumConstant = command(CommandKind::SetProperty);
    setEnumConstant.objectHandle = 1;
    setEnumConstant.attributeHandle = 3;
    setEnumConstant.context = 0;
    require(!enumMutation.consume(setEnumConstant));
    require(enumMutation.error() == IngestError::PropertyKindMismatch);
}

void zeroSourceAttributeAndObjectHandlesFailButZeroContextSucceeds()
{
    // Catches zero identities being confused with the valid producer-context zero value.
    CommandIngestor zeroAttribute(LoadPlan::allProperties());
    registerClass(zeroAttribute, 10, "Owner");
    Command invalidAttribute = command(CommandKind::RegisterAttribute);
    invalidAttribute.ownerClassHandle = 10;
    invalidAttribute.attributeKind = RecordedAttributeKind::Value;
    invalidAttribute.dataType = RecordedDataType::Int8;
    invalidAttribute.hasElementCount = true;
    require(!zeroAttribute.consume(invalidAttribute));
    require(zeroAttribute.error() == IngestError::InvalidSourceHandle);
    require(!zeroAttribute.takeCapture());

    CommandIngestor zeroObject(LoadPlan::allProperties());
    registerClass(zeroObject, 10, "Owner");
    Command invalidObject = command(CommandKind::CreateObject);
    invalidObject.classHandle = 10;
    invalidObject.objectHandle = 0;
    require(!zeroObject.consume(invalidObject));
    require(zeroObject.error() == IngestError::InvalidSourceHandle);
    require(!zeroObject.takeCapture());

    CommandIngestor zeroContext(LoadPlan::allProperties());
    registerClass(zeroContext, 10, "Owner");
    createWithClass(zeroContext, 1, 10, 0);
    start(zeroContext, 0, 1);
    stop(zeroContext, 0, 1);
    Capture capture = finish(zeroContext);
    require(capture.frames()[0].address.context == 0);
    require(capture.view(0)->isAlive(ObjectId{1}));
}

void identicalClassRegistrationIsIdempotent()
{
    // Catches an identical source declaration advancing the legacy segment or killing live objects.
    CommandIngestor ingestor(LoadPlan::allProperties());
    registerClass(ingestor, 100, 0, "ProducerBase");
    createWithClass(ingestor, 1, 100, 7);
    start(ingestor, 7, 1);
    stop(ingestor, 7, 1);
    registerClass(ingestor, 100, 0, "ProducerBase");
    start(ingestor, 7, 2);
    stop(ingestor, 7, 2);

    Capture capture = finish(ingestor);
    require(capture.recordedClassModel()->classes().size() == 1);
    require(capture.recordedClassModel()->classes()[0].id == ClassId{1});
    require(capture.recordedClassModel()->objects().size() == 1);
    require(capture.frames().size() == 2);
    require(capture.frames()[0].address == FrameAddress{0, 7, 1});
    require(capture.frames()[1].address == FrameAddress{0, 7, 2});
    require(capture.view(1)->isAlive(ObjectId{1}));
}

void conflictingClassReuseFailsClosed()
{
    // Catches a source handle being silently rebound to a different declaration.
    CommandIngestor conflicting(LoadPlan::allProperties());
    registerClass(conflicting, 100, 0, "ProducerBase");
    Command conflictingReuse = command(CommandKind::RegisterClass);
    conflictingReuse.classHandle = 100;
    conflictingReuse.name = "DifferentName";
    require(!conflicting.consume(conflictingReuse));
    require(conflicting.error() == IngestError::ConflictingClassRegistration);
    require(!conflicting.takeCapture());

    CommandIngestor changedBase(LoadPlan::allProperties());
    registerClass(changedBase, 10, 0, "FirstBase");
    registerClass(changedBase, 20, 0, "SecondBase");
    registerClass(changedBase, 30, 10, "Derived");
    Command reboundBase = command(CommandKind::RegisterClass);
    reboundBase.classHandle = 30;
    reboundBase.baseClassHandle = 20;
    reboundBase.name = "Derived";
    require(!changedBase.consume(reboundBase));
    require(changedBase.error() == IngestError::ConflictingClassRegistration);
    require(!changedBase.takeCapture());
}

void unknownBaseClassFailsClosed()
{
    // Catches a derived declaration retaining an unresolved source base handle.
    CommandIngestor unknownBase(LoadPlan::allProperties());
    Command orphan = command(CommandKind::RegisterClass);
    orphan.classHandle = 200;
    orphan.baseClassHandle = 999;
    orphan.name = "Orphan";
    require(!unknownBase.consume(orphan));
    require(unknownBase.error() == IngestError::UnknownBaseClass);
    require(!unknownBase.takeCapture());
}

void selfInheritanceCycleFailsClosed()
{
    // Catches a self-referential declaration entering the immutable hierarchy.
    CommandIngestor selfCycle(LoadPlan::allProperties());
    Command cyclic = command(CommandKind::RegisterClass);
    cyclic.classHandle = 200;
    cyclic.baseClassHandle = 200;
    cyclic.name = "Cyclic";
    require(!selfCycle.consume(cyclic));
    require(selfCycle.error() == IngestError::InheritanceCycle);
    require(!selfCycle.takeCapture());
}

void zeroSourceClassHandleFailsClosed()
{
    // Catches the no-base sentinel being accepted as a source class identity.
    CommandIngestor zeroHandle(LoadPlan::allProperties());
    Command invalid = command(CommandKind::RegisterClass);
    invalid.classHandle = 0;
    invalid.baseClassHandle = 0;
    invalid.name = "Invalid";
    require(!zeroHandle.consume(invalid));
    require(zeroHandle.error() == IngestError::InvalidSourceHandle);
    require(!zeroHandle.takeCapture());
}

void recordedClassModelOutlivesCapture()
{
    // Catches the immutable model borrowing storage owned by Capture.
    std::shared_ptr<const RecordedClassModel> model;
    {
        CommandIngestor ingestor(LoadPlan::allProperties());
        registerClass(ingestor, 100, 0, "PersistentClass");
        Capture capture = finish(ingestor);
        model = capture.recordedClassModel();
    }
    require(model);
    require(model->findClass(ClassId{1}) != nullptr);
    require(model->findClass(ClassId{1})->name == "PersistentClass");
}

void invalidCommandsFailClosedWithoutReturningPartialCapture()
{
    // Catches a regression that returns already-published frames after any rejected command.
    CommandIngestor unknownObject(LoadPlan::allProperties());
    start(unknownObject, 1, 1);
    Command unknownSet = command(CommandKind::SetProperty);
    unknownSet.objectHandle = 4;
    unknownSet.attributeHandle = 9;
    unknownSet.context = 1;
    require(!unknownObject.consume(unknownSet));
    require(unknownObject.error() == IngestError::UnknownProperty);
    require(!unknownObject.finish());
    require(!unknownObject.takeCapture());

    CommandIngestor missingObject(LoadPlan::allProperties());
    registerNeutralClass(missingObject);
    registerScalar(missingObject, 90);
    start(missingObject, 1, 1);
    Command missingObjectSet = command(CommandKind::SetProperty);
    missingObjectSet.objectHandle = 4;
    missingObjectSet.attributeHandle = 90;
    missingObjectSet.context = 1;
    require(!missingObject.consume(missingObjectSet));
    require(missingObject.error() == IngestError::UnknownObject);
    require(!missingObject.takeCapture());

    CommandIngestor kindMismatch(LoadPlan::allProperties());
    registerNeutralClass(kindMismatch);
    registerScalar(kindMismatch, 90);
    create(kindMismatch, 4, 1);
    start(kindMismatch, 1, 1);
    const Bytes mismatchPayload{0x01};
    Command mismatchedKind = command(CommandKind::AddToList);
    mismatchedKind.objectHandle = 4;
    mismatchedKind.attributeHandle = 90;
    mismatchedKind.context = 1;
    mismatchedKind.payload = mismatchPayload.data();
    mismatchedKind.payloadSize = mismatchPayload.size();
    require(!kindMismatch.consume(mismatchedKind));
    require(kindMismatch.error() == IngestError::PropertyKindMismatch);
    require(!kindMismatch.takeCapture());

    CommandIngestor nestedStart(LoadPlan::allProperties());
    start(nestedStart, 1, 1);
    Command secondStart = command(CommandKind::StartFrame);
    secondStart.context = 2;
    secondStart.frame = 2;
    require(!nestedStart.consume(secondStart));
    require(nestedStart.error() == IngestError::NestedStart);
    require(!nestedStart.takeCapture());

    CommandIngestor stopWithoutStart(LoadPlan::allProperties());
    Command orphanStop = command(CommandKind::StopFrame);
    orphanStop.context = 1;
    orphanStop.frame = 1;
    require(!stopWithoutStart.consume(orphanStop));
    require(stopWithoutStart.error() == IngestError::StopWithoutStart);
    require(!stopWithoutStart.takeCapture());

    CommandIngestor mismatchedStop(LoadPlan::allProperties());
    start(mismatchedStop, 1, 1);
    Command wrongStop = command(CommandKind::StopFrame);
    wrongStop.context = 2;
    wrongStop.frame = 1;
    require(!mismatchedStop.consume(wrongStop));
    require(mismatchedStop.error() == IngestError::MismatchedStop);
    require(!mismatchedStop.takeCapture());

    CommandIngestor wrongFrameStop(LoadPlan::allProperties());
    start(wrongFrameStop, 1, 1);
    Command sameContextWrongFrame = command(CommandKind::StopFrame);
    sameContextWrongFrame.context = 1;
    sameContextWrongFrame.frame = 2;
    require(!wrongFrameStop.consume(sameContextWrongFrame));
    require(wrongFrameStop.error() == IngestError::MismatchedStop);
    require(!wrongFrameStop.takeCapture());
}

void eofRulesDoNotPublishSyntheticFrames()
{
    // Catches a regression that accepts an incomplete frame or invents a frame for trailing deferred commands.
    CommandIngestor incomplete(LoadPlan::allProperties());
    start(incomplete, 1, 9);
    require(!incomplete.finish());
    require(incomplete.error() == IngestError::ActiveFrameAtEof);
    require(!incomplete.takeCapture());

    CommandIngestor trailing(LoadPlan::allProperties());
    registerNeutralClass(trailing);
    create(trailing, 1, 1);
    start(trailing, 1, 1);
    stop(trailing, 1, 1);
    create(trailing, 2, 1);
    Capture capture = finish(trailing);
    require(capture.frames().size() == 1);
    require(capture.view(0)->isAlive(1));
    require(!capture.view(0)->isAlive(2));
    require(capture.recordedClassModel()->objects().size() == 1);
    require(capture.recordedClassModel()->findObject(ObjectId{1}) != nullptr);
    require(capture.recordedClassModel()->findObject(ObjectId{2}) == nullptr);
}

void commandLocalProducerContextsPopulateADistinctFrameContext()
{
    // Catches a regression that treats the command-local producer context as a frame context.
    CommandIngestor ingestor(LoadPlan::allProperties());
    registerNeutralClass(ingestor);
    registerScalar(ingestor, 10);
    registerList(ingestor, 20);

    Command frameStart = command(CommandKind::StartFrame);
    frameStart.context = 0xABC;
    frameStart.frame = 7;
    consume(ingestor, frameStart);

    Command firstCreate = command(CommandKind::CreateObject);
    firstCreate.classHandle = kNeutralClassHandle;
    firstCreate.objectHandle = 100;
    firstCreate.context = 1;
    consume(ingestor, firstCreate);

    const Bytes scalarPayload{0x10, 0x20};
    Command scalarSet = command(CommandKind::SetProperty);
    scalarSet.objectHandle = 100;
    scalarSet.attributeHandle = 10;
    scalarSet.context = 1;
    scalarSet.payload = scalarPayload.data();
    scalarSet.payloadSize = scalarPayload.size();
    consume(ingestor, scalarSet);

    const Bytes listPayload{0x30};
    Command listAdd = command(CommandKind::AddToList);
    listAdd.objectHandle = 100;
    listAdd.attributeHandle = 20;
    listAdd.context = 1;
    listAdd.payload = listPayload.data();
    listAdd.payloadSize = listPayload.size();
    consume(ingestor, listAdd);

    Command secondCreate = command(CommandKind::CreateObject);
    secondCreate.classHandle = kNeutralClassHandle;
    secondCreate.objectHandle = 200;
    secondCreate.context = 1;
    consume(ingestor, secondCreate);
    Command secondDestroy = command(CommandKind::DestroyObject);
    secondDestroy.objectHandle = 200;
    secondDestroy.context = 1;
    consume(ingestor, secondDestroy);

    Command frameStop = command(CommandKind::StopFrame);
    frameStop.context = 0xABC;
    frameStop.frame = 7;
    consume(ingestor, frameStop);

    Capture capture = finish(ingestor);
    require(capture.frames().size() == 1);
    require(capture.frames()[0].address == FrameAddress{0, 0xABC, 7});
    const auto view = capture.view(0);
    require(view->isAlive(1));
    require(!view->isAlive(2));
    require(view->persistentProperty(1, 1).bytes == bytes({0x10, 0x20}));
    require(view->list(1, 2).values == std::vector<Bytes>{bytes({0x30})});
    require(capture.recordedClassModel()->findObject(ObjectId{2}) != nullptr);
}

void legacyClassNamesDefineOneFallbackTransitionPerSchemaCycle()
{
    // Catches a regression that uses class handles instead of names or advances repeatedly within one cycle.
    CommandIngestor ingestor(LoadPlan::allProperties());
    registerClass(ingestor, 1, "LegacyA");
    registerClass(ingestor, 2, "LegacyB");
    start(ingestor, 1, 1);
    stop(ingestor, 1, 1);
    registerClass(ingestor, 11, "LegacyA");
    registerClass(ingestor, 12, "LegacyB");
    registerClass(ingestor, 13, "LegacyC");
    start(ingestor, 1, 1);
    stop(ingestor, 1, 1);

    Capture capture = finish(ingestor);
    require(capture.frames().size() == 2);
    require(capture.frames()[0].address.segment == 0);
    require(capture.frames()[1].address.segment == 1);
}

void trailingInvalidLifecycleCommandsFailBeforeEof()
{
    // Catches a regression that postpones lifecycle/property validation until a never-created synthetic frame.
    CommandIngestor unknownSet(LoadPlan::allProperties());
    registerNeutralClass(unknownSet);
    registerScalar(unknownSet, 10);
    Command setValue = command(CommandKind::SetProperty);
    setValue.objectHandle = 7;
    setValue.attributeHandle = 10;
    setValue.context = 1;
    require(!unknownSet.consume(setValue));
    require(unknownSet.error() == IngestError::UnknownObject);
    require(!unknownSet.finish());

    CommandIngestor unknownAdd(LoadPlan::allProperties());
    registerNeutralClass(unknownAdd);
    registerList(unknownAdd, 20);
    Command addValue = command(CommandKind::AddToList);
    addValue.objectHandle = 7;
    addValue.attributeHandle = 20;
    addValue.context = 1;
    require(!unknownAdd.consume(addValue));
    require(unknownAdd.error() == IngestError::UnknownObject);
    require(!unknownAdd.finish());

    CommandIngestor unknownRemove(LoadPlan::allProperties());
    registerNeutralClass(unknownRemove);
    registerList(unknownRemove, 20);
    Command removeValue = command(CommandKind::RemoveFromList);
    removeValue.objectHandle = 7;
    removeValue.attributeHandle = 20;
    removeValue.context = 1;
    require(!unknownRemove.consume(removeValue));
    require(unknownRemove.error() == IngestError::UnknownObject);
    require(!unknownRemove.finish());

    CommandIngestor trailingCreate(LoadPlan::allProperties());
    registerNeutralClass(trailingCreate);
    start(trailingCreate, 1, 1);
    stop(trailingCreate, 1, 1);
    create(trailingCreate, 7, 1);
    Capture capture = finish(trailingCreate);
    require(capture.frames().size() == 1);
}

void unknownDestroysAreIdempotentLifecycleNoOps()
{
    // Catches a regression that rejects a never-seen or already-destroyed OmniPVD object handle.
    CommandIngestor trailing(LoadPlan::allProperties());
    registerNeutralClass(trailing);
    Command trailingDestroy = command(CommandKind::DestroyObject);
    trailingDestroy.objectHandle = 0x138ab176aa8;
    trailingDestroy.context = 1;
    consume(trailing, trailingDestroy);
    create(trailing, 10, 1);
    start(trailing, 0xABC, 7);
    stop(trailing, 0xABC, 7);
    Capture trailingCapture = finish(trailing);
    require(trailingCapture.frames().size() == 1);
    require(trailingCapture.view(0)->isAlive(1));

    CommandIngestor inFrame(LoadPlan::allProperties());
    registerNeutralClass(inFrame);
    create(inFrame, 10, 1);
    start(inFrame, 0xABC, 7);
    Command firstDestroy = command(CommandKind::DestroyObject);
    firstDestroy.objectHandle = 10;
    firstDestroy.context = 1;
    consume(inFrame, firstDestroy);
    consume(inFrame, firstDestroy);
    Command neverSeenDestroy = command(CommandKind::DestroyObject);
    neverSeenDestroy.objectHandle = 0x138ab176aa8;
    neverSeenDestroy.context = 1;
    consume(inFrame, neverSeenDestroy);
    stop(inFrame, 0xABC, 7);

    Capture capture = finish(inFrame);
    require(capture.frames().size() == 1);
    require(!capture.view(0)->isAlive(1));
}

void deadObjectListCleanupIsAnExactNoOp()
{
    // Catches a regression that rejects cleanup REMOVE commands after the source object was STOP-published dead.
    const std::uint64_t scene = 0x138ab174e00;
    const Bytes cleanupPayload{0x42};

    CommandIngestor cleanup(LoadPlan::allProperties());
    registerNeutralClass(cleanup);
    registerScalar(cleanup, 10);
    registerList(cleanup, 20);
    create(cleanup, scene, 1);
    start(cleanup, 0xABC, 3);
    Command destroyScene = command(CommandKind::DestroyObject);
    destroyScene.objectHandle = scene;
    destroyScene.context = 1;
    consume(cleanup, destroyScene);
    stop(cleanup, 0xABC, 3);
    Command cleanupRemove = command(CommandKind::RemoveFromList);
    cleanupRemove.objectHandle = scene;
    cleanupRemove.attributeHandle = 20;
    cleanupRemove.context = 1;
    cleanupRemove.payload = cleanupPayload.data();
    cleanupRemove.payloadSize = cleanupPayload.size();
    consume(cleanup, cleanupRemove);

    Capture capture = finish(cleanup);
    require(capture.frames().size() == 1);
    require(!capture.view(0)->isAlive(1));
    require(capture.stats().retainedPayloadBytes == 0);

    CommandIngestor differentContextRemove(LoadPlan::allProperties());
    registerNeutralClass(differentContextRemove);
    registerList(differentContextRemove, 20);
    create(differentContextRemove, scene, 1);
    start(differentContextRemove, 0xABC, 3);
    consume(differentContextRemove, destroyScene);
    stop(differentContextRemove, 0xABC, 3);
    Command wrongContextRemove = cleanupRemove;
    wrongContextRemove.context = 2;
    require(!differentContextRemove.consume(wrongContextRemove));
    require(differentContextRemove.error() == IngestError::UnknownObject);

    CommandIngestor neverSeenRemove(LoadPlan::allProperties());
    registerNeutralClass(neverSeenRemove);
    registerList(neverSeenRemove, 20);
    require(!neverSeenRemove.consume(cleanupRemove));
    require(neverSeenRemove.error() == IngestError::UnknownObject);

    CommandIngestor unknownDestroyDoesNotTombstone(LoadPlan::allProperties());
    registerNeutralClass(unknownDestroyDoesNotTombstone);
    registerList(unknownDestroyDoesNotTombstone, 20);
    Command unknownDestroy = command(CommandKind::DestroyObject);
    unknownDestroy.objectHandle = scene;
    unknownDestroy.context = 1;
    consume(unknownDestroyDoesNotTombstone, unknownDestroy);
    require(!unknownDestroyDoesNotTombstone.consume(cleanupRemove));
    require(unknownDestroyDoesNotTombstone.error() == IngestError::UnknownObject);

    CommandIngestor deadUnknownProperty(LoadPlan::allProperties());
    registerNeutralClass(deadUnknownProperty);
    registerList(deadUnknownProperty, 20);
    create(deadUnknownProperty, scene, 1);
    start(deadUnknownProperty, 0xABC, 3);
    consume(deadUnknownProperty, destroyScene);
    stop(deadUnknownProperty, 0xABC, 3);
    Command unknownPropertyRemove = cleanupRemove;
    unknownPropertyRemove.attributeHandle = 99;
    require(!deadUnknownProperty.consume(unknownPropertyRemove));
    require(deadUnknownProperty.error() == IngestError::UnknownProperty);

    CommandIngestor deadScalarProperty(LoadPlan::allProperties());
    registerNeutralClass(deadScalarProperty);
    registerScalar(deadScalarProperty, 10);
    registerList(deadScalarProperty, 20);
    create(deadScalarProperty, scene, 1);
    start(deadScalarProperty, 0xABC, 3);
    consume(deadScalarProperty, destroyScene);
    stop(deadScalarProperty, 0xABC, 3);
    Command scalarPropertyRemove = cleanupRemove;
    scalarPropertyRemove.attributeHandle = 10;
    require(!deadScalarProperty.consume(scalarPropertyRemove));
    require(deadScalarProperty.error() == IngestError::PropertyKindMismatch);

    CommandIngestor deadSet(LoadPlan::allProperties());
    registerNeutralClass(deadSet);
    registerScalar(deadSet, 10);
    create(deadSet, scene, 1);
    start(deadSet, 0xABC, 3);
    consume(deadSet, destroyScene);
    stop(deadSet, 0xABC, 3);
    Command deadSetValue = command(CommandKind::SetProperty);
    deadSetValue.objectHandle = scene;
    deadSetValue.attributeHandle = 10;
    deadSetValue.context = 1;
    require(!deadSet.consume(deadSetValue));
    require(deadSet.error() == IngestError::UnknownObject);

    CommandIngestor deadAdd(LoadPlan::allProperties());
    registerNeutralClass(deadAdd);
    registerList(deadAdd, 20);
    create(deadAdd, scene, 1);
    start(deadAdd, 0xABC, 3);
    consume(deadAdd, destroyScene);
    stop(deadAdd, 0xABC, 3);
    Command deadAddValue = command(CommandKind::AddToList);
    deadAddValue.objectHandle = scene;
    deadAddValue.attributeHandle = 20;
    deadAddValue.context = 1;
    require(!deadAdd.consume(deadAddValue));
    require(deadAdd.error() == IngestError::UnknownObject);
}

void producerContextsAllocateDistinctObjectGenerations()
{
    // Catches a regression that aliases identical external handles from distinct producer contexts.
    CommandIngestor ingestor(LoadPlan::allProperties());
    registerNeutralClass(ingestor);
    create(ingestor, 9, 1);
    create(ingestor, 9, 2);
    start(ingestor, 1, 1);
    stop(ingestor, 1, 1);
    start(ingestor, 1, 2);
    Command destroyFirstContext = command(CommandKind::DestroyObject);
    destroyFirstContext.objectHandle = 9;
    destroyFirstContext.context = 1;
    consume(ingestor, destroyFirstContext);
    stop(ingestor, 1, 2);

    Capture capture = finish(ingestor);
    require(capture.view(0)->isAlive(1));
    require(capture.view(0)->isAlive(2));
    require(!capture.view(1)->isAlive(1));
    require(capture.view(1)->isAlive(2));
}

void firstMetadataMarkerDuringFrameFailsClosed()
{
    // Catches a regression that lets the first segment-zero metadata marker alter an active frame.
    CommandIngestor metadataDuringFrame(LoadPlan::allProperties());
    registerClass(metadataDuringFrame, 40, "PxOmniPvdMetaData");
    start(metadataDuringFrame, 1, 1);
    Command metadata = command(CommandKind::CreateObject);
    metadata.classHandle = 40;
    metadata.objectHandle = 1;
    metadata.context = 1;
    require(!metadataDuringFrame.consume(metadata));
    require(metadataDuringFrame.error() == IngestError::SegmentTransitionDuringFrame);
    require(!metadataDuringFrame.takeCapture());
}

void metadataBoundarySupersedesAnActiveOldSegmentFrame()
{
    // Catches a regression that publishes or leaks mutations from a frame superseded by a later metadata marker.
    CommandIngestor ingestor(LoadPlan::allProperties());
    registerNeutralClass(ingestor);
    registerScalar(ingestor, 10);
    registerClass(ingestor, 40, "PxOmniPvdMetaData");
    createWithClass(ingestor, 1, 40, 1);
    create(ingestor, 10, 1);
    start(ingestor, 0x100, 120);
    set(ingestor, 10, 10, {0xaa}, 1);
    stop(ingestor, 0x100, 120);

    start(ingestor, 0x100, 121);
    set(ingestor, 10, 10, {0xbb}, 1);
    Command oldDestroy = command(CommandKind::DestroyObject);
    oldDestroy.objectHandle = 10;
    oldDestroy.context = 1;
    consume(ingestor, oldDestroy);
    create(ingestor, 20, 1);
    createWithClass(ingestor, 2, 40, 1);
    start(ingestor, 0x200, 1);
    createWithClass(ingestor, 3, 40, 1);
    start(ingestor, 0x300, 1);
    stop(ingestor, 0x300, 1);

    Capture capture = finish(ingestor);
    require(capture.frames().size() == 2);
    require(capture.frames()[0].address == FrameAddress{0, 0x100, 120});
    require(capture.frames()[1].address == FrameAddress{2, 0x300, 1});
    require(capture.view(0)->isAlive(2));
    require(capture.view(0)->persistentProperty(2, 1).bytes == bytes({0xaa}));
    require(!capture.view(1)->isAlive(2));
    require(!capture.view(1)->isAlive(3));
    require(!capture.view(1)->isAlive(4));
    require(capture.view(1)->isAlive(5));
    const std::vector<RecordedObject>& objects = capture.recordedClassModel()->objects();
    require(objects.size() == 3);
    require(objects[0].id == ObjectId{1});
    require(objects[1].id == ObjectId{2});
    require(objects[2].id == ObjectId{5});
    require(capture.recordedClassModel()->findObject(ObjectId{3}) == nullptr);
    require(capture.recordedClassModel()->findObject(ObjectId{4}) == nullptr);
}

void legacyBoundarySupersedesAnActiveOldSegmentFrame()
{
    // Catches a regression that fails to roll back an active dangling frame at a legacy schema boundary.
    CommandIngestor ingestor(LoadPlan::allProperties());
    registerClass(ingestor, 1, "LegacyA");
    registerNeutralClass(ingestor);
    create(ingestor, 10, 1);
    start(ingestor, 0x100, 120);
    stop(ingestor, 0x100, 120);

    start(ingestor, 0x100, 121);
    Command oldDestroy = command(CommandKind::DestroyObject);
    oldDestroy.objectHandle = 10;
    oldDestroy.context = 1;
    consume(ingestor, oldDestroy);
    create(ingestor, 20, 1);
    registerClass(ingestor, 2, "LegacyA");
    create(ingestor, 30, 1);
    start(ingestor, 0x200, 1);
    stop(ingestor, 0x200, 1);

    Capture capture = finish(ingestor);
    require(capture.frames().size() == 2);
    require(capture.frames()[1].address == FrameAddress{1, 0x200, 1});
    require(!capture.view(1)->isAlive(1));
    require(!capture.view(1)->isAlive(2));
    require(capture.view(1)->isAlive(3));
}

} // namespace

int main()
{
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"preFrameCommandsAreDeferredIntoFirstPublication", preFrameCommandsAreDeferredIntoFirstPublication},
        {"frameIsNotPublishedUntilMatchingStop", frameIsNotPublishedUntilMatchingStop},
        {"scalarAndListCommandsRetainOrderedBytes", scalarAndListCommandsRetainOrderedBytes},
        {"excludedPayloadsPreserveLifecycleAndCountFilteredBytes", excludedPayloadsPreserveLifecycleAndCountFilteredBytes},
        {"excludedMutationStillRejectsMissingPayloadStorage", excludedMutationStillRejectsMissingPayloadStorage},
        {"recordedSelectorsAreExactPairsWithoutInheritanceInference", recordedSelectorsAreExactPairsWithoutInheritanceInference},
        {"recordedNameFilteringResolvesEveryDeclarationCycle", recordedNameFilteringResolvesEveryDeclarationCycle},
        {"frameViewsQueryProjectedAliveObjectsByRecordedClass", frameViewsQueryProjectedAliveObjectsByRecordedClass},
        {"attributesAllocateMonotonicPropertyIds", attributesAllocateMonotonicPropertyIds},
        {"equalLocalFramesAcrossContextsAndSegmentsRemainDistinct", equalLocalFramesAcrossContextsAndSegmentsRemainDistinct},
        {"recreatedExternalHandlesBecomeNewObjectGenerations", recreatedExternalHandlesBecomeNewObjectGenerations},
        {"metadataSegmentTransitionClosesPriorLiveGenerations", metadataSegmentTransitionClosesPriorLiveGenerations},
        {"repeatedClassRegistrationProvidesLegacySegmentFallback", repeatedClassRegistrationProvidesLegacySegmentFallback},
        {"recordedClassModelPreservesIdentityHierarchyAndExactNames", recordedClassModelPreservesIdentityHierarchyAndExactNames},
        {"recordedDeclarationsPreserveAttributesObjectsAndInheritance", recordedDeclarationsPreserveAttributesObjectsAndInheritance},
        {"declarationValidationFailsClosedWithExactErrors", declarationValidationFailsClosedWithExactErrors},
        {"nonScalarDeclarationsRejectSetMutation", nonScalarDeclarationsRejectSetMutation},
        {"zeroSourceAttributeAndObjectHandlesFailButZeroContextSucceeds", zeroSourceAttributeAndObjectHandlesFailButZeroContextSucceeds},
        {"identicalClassRegistrationIsIdempotent", identicalClassRegistrationIsIdempotent},
        {"conflictingClassReuseFailsClosed", conflictingClassReuseFailsClosed},
        {"unknownBaseClassFailsClosed", unknownBaseClassFailsClosed},
        {"selfInheritanceCycleFailsClosed", selfInheritanceCycleFailsClosed},
        {"zeroSourceClassHandleFailsClosed", zeroSourceClassHandleFailsClosed},
        {"recordedClassModelOutlivesCapture", recordedClassModelOutlivesCapture},
        {"invalidCommandsFailClosedWithoutReturningPartialCapture", invalidCommandsFailClosedWithoutReturningPartialCapture},
        {"eofRulesDoNotPublishSyntheticFrames", eofRulesDoNotPublishSyntheticFrames},
        {"commandLocalProducerContextsPopulateADistinctFrameContext", commandLocalProducerContextsPopulateADistinctFrameContext},
        {"legacyClassNamesDefineOneFallbackTransitionPerSchemaCycle", legacyClassNamesDefineOneFallbackTransitionPerSchemaCycle},
        {"trailingInvalidLifecycleCommandsFailBeforeEof", trailingInvalidLifecycleCommandsFailBeforeEof},
        {"unknownDestroysAreIdempotentLifecycleNoOps", unknownDestroysAreIdempotentLifecycleNoOps},
        {"deadObjectListCleanupIsAnExactNoOp", deadObjectListCleanupIsAnExactNoOp},
        {"producerContextsAllocateDistinctObjectGenerations", producerContextsAllocateDistinctObjectGenerations},
        {"firstMetadataMarkerDuringFrameFailsClosed", firstMetadataMarkerDuringFrameFailsClosed},
        {"metadataBoundarySupersedesAnActiveOldSegmentFrame", metadataBoundarySupersedesAnActiveOldSegmentFrame},
        {"legacyBoundarySupersedesAnActiveOldSegmentFrame", legacyBoundarySupersedesAnActiveOldSegmentFrame},
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
