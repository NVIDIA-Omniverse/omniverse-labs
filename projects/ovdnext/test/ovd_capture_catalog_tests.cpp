/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "OvdCaptureCatalog.h"
#include <gtest/gtest.h>
#include "OvdInputValidation.h"
#include "OmniPvdReader.h"
#include "OmniPvdWriter.h"
#include "OvdLiveSource.h"
#include "OvdFileLoader.h"
#include "OvdDomAdapter.h"
#include "OvdMessageCollector.h"
#include "PvdEngine.h"
#include "PvdSimpleAllocator.h"
#include "TcpTestSupport.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <thread>

using namespace pvdapp;

namespace
{
bool row(OvdCaptureCatalog& catalog, uint64_t rowHandle, uint64_t target, const char* name,
         uint64_t owner = 0, bool selectable = true, uint32_t schema = 24)
{
    return catalog.observeValidated({CaptureCatalogCommand::Create, rowHandle, schema}) &&
        catalog.observeValidated({CaptureCatalogCommand::Field, rowHandle, schema, CaptureCatalogField::Target, target}) &&
        catalog.observeValidated({CaptureCatalogCommand::Field, rowHandle, schema, CaptureCatalogField::TypeName, 0, "PxRigidDynamic"}) &&
        catalog.observeValidated({CaptureCatalogCommand::Field, rowHandle, schema, CaptureCatalogField::Name, 0, name}) &&
        catalog.observeValidated({CaptureCatalogCommand::Field, rowHandle, schema, CaptureCatalogField::Owner, owner}) &&
        catalog.observeValidated({CaptureCatalogCommand::Field, rowHandle, schema, CaptureCatalogField::Scene, 99}) &&
        catalog.observeValidated({CaptureCatalogCommand::Field, rowHandle, schema, CaptureCatalogField::Selectable, selectable ? 1u : 0u});
}

bool publish(OvdCaptureCatalog& catalog, uint64_t generation)
{
    return catalog.observeValidated({CaptureCatalogCommand::Commit, 0, 0, CaptureCatalogField::Target, generation});
}
}

TEST(CaptureCatalog, PublishesOnlyCompleteGenerationsAndPreservesHandles)
{
    OvdCaptureCatalog catalog;
    EXPECT_FALSE(catalog.snapshot().ready);
    ASSERT_TRUE(row(catalog, 10, 0xfedcba9876543210ull, "box"));
    EXPECT_FALSE(catalog.snapshot().ready);
    ASSERT_TRUE(publish(catalog, 1));
    const auto page = catalog.catalogPage(1, 0, 128);
    ASSERT_EQ(page.entries.size(), 1u);
    EXPECT_EQ(page.entries[0].targetHandle, 0xfedcba9876543210ull);
    EXPECT_EQ(page.entries[0].name, "box");
    EXPECT_TRUE(catalog.canSelect(1, 0xfedcba9876543210ull, false));
    EXPECT_FALSE(catalog.canSelect(1, 0xfedcba9876543210ull, true));
    EXPECT_EQ(catalog.catalogPage(2, 0, 128).state, CaptureCatalogPageState::Stale);
}

TEST(CaptureCatalog, GenerationChangeInvalidatesBeforeRowsAreReused)
{
    OvdCaptureCatalog catalog;
    ASSERT_TRUE(row(catalog, 10, 100, "old", 0, true, 24));
    ASSERT_TRUE(publish(catalog, 1));
    const auto oldRevision = catalog.snapshot().revision;
    ASSERT_TRUE(catalog.observeValidated({CaptureCatalogCommand::Destroy, 10}));
    EXPECT_FALSE(catalog.snapshot().ready);
    EXPECT_GT(catalog.snapshot().revision, oldRevision);
    EXPECT_FALSE(catalog.canSelect(1, 100, false));
    ASSERT_TRUE(row(catalog, 10, 200, "new", 0, true, 96));
    ASSERT_TRUE(publish(catalog, 2));
    EXPECT_FALSE(catalog.canSelect(1, 100, false));
    EXPECT_TRUE(catalog.canSelect(2, 200, false));
    ASSERT_TRUE(catalog.observeValidated({CaptureCatalogCommand::Destroy, 10}));
    ASSERT_TRUE(publish(catalog, 3));
    EXPECT_TRUE(catalog.snapshot().ready);
    EXPECT_EQ(catalog.catalogPage(3, 0, 128).total, 0u);
}

TEST(CaptureCatalog, RejectsInvalidRowLifetimesAndIncompleteCommits)
{
    OvdCaptureCatalog incomplete;
    ASSERT_TRUE(incomplete.observeValidated({CaptureCatalogCommand::Create, 10, 24}));
    EXPECT_FALSE(publish(incomplete, 1));
    EXPECT_FALSE(incomplete.snapshot().ready);
    OvdCaptureCatalog duplicate;
    ASSERT_TRUE(row(duplicate, 10, 100, "row"));
    EXPECT_FALSE(duplicate.observeValidated({CaptureCatalogCommand::Create, 10, 96}));
    OvdCaptureCatalog staleClass;
    ASSERT_TRUE(row(staleClass, 10, 100, "row"));
    EXPECT_FALSE(staleClass.observeValidated({CaptureCatalogCommand::Field, 10, 96, CaptureCatalogField::Target, 100}));
    OvdCaptureCatalog retired;
    ASSERT_TRUE(row(retired, 10, 100, "row"));
    ASSERT_TRUE(publish(retired, 1));
    ASSERT_TRUE(row(retired, 11, 200, "new"));
    EXPECT_FALSE(publish(retired, 2)); // Old row 10 was never retired.
}

TEST(CaptureCatalog, IndexedPagesFilterAcrossDuplicateNamesAndParents)
{
    OvdCaptureCatalog catalog;
    ASSERT_TRUE(row(catalog, 10, 100, "box A", 8, false));
    ASSERT_TRUE(row(catalog, 11, 101, "box A", 7));
    ASSERT_TRUE(row(catalog, 12, 102, "sphere", 7));
    ASSERT_TRUE(row(catalog, 13, 103, "box B", 7));
    ASSERT_TRUE(publish(catalog, 1));
    const auto page = catalog.catalogPage(1, 1, 1, 7, "box");
    ASSERT_EQ(page.total, 2u);
    ASSERT_EQ(page.entries.size(), 1u);
    EXPECT_EQ(page.entries[0].targetHandle, 103u);
    EXPECT_FALSE(catalog.canSelect(1, 100, false));
    EXPECT_TRUE(catalog.catalogPage(1, 99, 128).entries.empty());
    EXPECT_THROW(catalog.catalogPage(1, 0, 1001), std::invalid_argument);
    EXPECT_EQ(catalog.catalogPage(1, 0, 128, 7, "absent").total, 0u);
}

TEST(CaptureCatalog, EnforcesRowStringAllocationAndMutationBudgets)
{
    auto limits = CaptureCatalogLimits::defaults(); limits.rows = 1;
    OvdCaptureCatalog rows(limits);
    ASSERT_TRUE(row(rows, 10, 100, "row"));
    EXPECT_FALSE(row(rows, 11, 101, "extra"));
    EXPECT_EQ(rows.error(), CaptureCatalogError::Rows);
    // MSVC allocates its empty hash-table sentinel during construction; GCC
    // does not. Leave room for that baseline, but none for a catalog row.
    OvdCaptureCatalog empty;
    const auto emptyBytes = empty.snapshot().allocations;
    limits = CaptureCatalogLimits::defaults(); limits.allocations = emptyBytes;
    OvdCaptureCatalog allocation(limits);
    EXPECT_FALSE(row(allocation, 10, 100, "row"));
    EXPECT_EQ(allocation.error(), CaptureCatalogError::Allocations);
    EXPECT_LE(allocation.snapshot().peakAllocations, emptyBytes);
    limits = CaptureCatalogLimits::defaults(); limits.mutations = 5;
    OvdCaptureCatalog mutations(limits);
    EXPECT_FALSE(row(mutations, 10, 100, "row"));
    EXPECT_EQ(mutations.error(), CaptureCatalogError::Mutations);
    limits = CaptureCatalogLimits::defaults(); limits.stringBytes = 16;
    OvdCaptureCatalog names(limits);
    EXPECT_FALSE(row(names, 10, 100, "long name"));
    EXPECT_EQ(names.error(), CaptureCatalogError::Strings);
    OvdCaptureCatalog longName;
    ASSERT_TRUE(longName.observeValidated({CaptureCatalogCommand::Create, 10, 24}));
    const std::string oversized(2049, 'x');
    EXPECT_FALSE(longName.observeValidated({CaptureCatalogCommand::Field, 10, 24, CaptureCatalogField::Name, 0, oversized}));
    EXPECT_EQ(longName.error(), CaptureCatalogError::Strings);
}

TEST(CaptureCatalog, EightMaximumGenerationsKeepOneBoundedStoreAndPages)
{
    OvdCaptureCatalog catalog;
    for (uint64_t generation = 1; generation <= 8; ++generation)
    {
        if (generation > 1)
            for (uint64_t i = 1; i <= 262144; ++i)
                ASSERT_TRUE(catalog.observeValidated({CaptureCatalogCommand::Destroy, i}));
        for (uint64_t i = 1; i <= 262144; ++i)
            ASSERT_TRUE(row(catalog, i, 0x20000000000000ull + i, "b"));
        ASSERT_TRUE(publish(catalog, generation));
        const auto snapshot = catalog.snapshot();
        ASSERT_EQ(snapshot.rows, 262144u);
        EXPECT_LE(snapshot.peakAllocations, 128ull << 20);
        EXPECT_LE(snapshot.stringBytes, 64ull << 20);
        const auto page = catalog.catalogPage(generation, 262100, 128, 99, "b");
        ASSERT_EQ(page.total, 262144u); ASSERT_EQ(page.entries.size(), 44u);
        EXPECT_EQ(page.entries.front().targetHandle, 0x20000000000000ull + 262101);
        EXPECT_LE(page.copiedBytes, 1ull << 20);
        RecordProperty("catalog_peak_allocations", std::to_string(snapshot.peakAllocations));
    }
    EXPECT_FALSE(row(catalog, 262145, 0x20000000040001ull, "overflow"));
    EXPECT_EQ(catalog.error(), CaptureCatalogError::Rows);
}

TEST(CaptureCatalog, RejectsAggregateStringAndCommandCapPlusOne)
{
    auto limits = CaptureCatalogLimits::defaults(); limits.stringBytes = 15;
    OvdCaptureCatalog names(limits);
    ASSERT_TRUE(row(names, 1, 1, "x")); // 14-byte type + one-byte label
    ASSERT_TRUE(names.observeValidated({CaptureCatalogCommand::Create, 2, 24}));
    EXPECT_FALSE(names.observeValidated({CaptureCatalogCommand::Field, 2, 24, CaptureCatalogField::Name, 0, "x"}));
    EXPECT_EQ(names.error(), CaptureCatalogError::Strings);
    limits = CaptureCatalogLimits::defaults(); limits.commands = 7;
    OvdCaptureCatalog commands(limits);
    ASSERT_TRUE(row(commands, 1, 1, "x"));
    EXPECT_FALSE(commands.observeValidated({CaptureCatalogCommand::Destroy, 1}));
    EXPECT_EQ(commands.error(), CaptureCatalogError::Commands);
}

TEST(CaptureCatalog, PageCopiesStayBoundedWithLongNames)
{
    OvdCaptureCatalog catalog;
    const std::string name(2048, 'x');
    for (uint64_t i = 1; i <= 1000; ++i) ASSERT_TRUE(row(catalog, i, i, name.c_str()));
    ASSERT_TRUE(publish(catalog, 1));
    const auto page = catalog.catalogPage(1, 0, 1000);
    EXPECT_GT(page.entries.size(), 128u); EXPECT_LT(page.entries.size(), 1000u);
    EXPECT_LE(page.copiedBytes, 1ull << 20);
    const auto next = catalog.catalogPage(1, page.entries.size(), 1000);
    ASSERT_FALSE(next.entries.empty()); EXPECT_EQ(next.entries[0].targetHandle, page.entries.size() + 1);
}

extern "C" OmniPvdReader* OMNI_PVD_CALL createOmniPvdReader();
extern "C" OmniPvdWriter* OMNI_PVD_CALL createOmniPvdWriter();
namespace
{
struct Bytes : OmniPvdReadStream, OmniPvdWriteStream
{
    std::vector<uint8_t> data;
    size_t position = 0;
    bool OMNI_PVD_CALL openStream() override { return true; }
    bool OMNI_PVD_CALL closeStream() override { return true; }
    bool OMNI_PVD_CALL flush() override { return true; }
    uint64_t OMNI_PVD_CALL writeBytes(const uint8_t* bytes, uint64_t size) override
    { data.insert(data.end(), bytes, bytes + size); return size; }
    uint64_t OMNI_PVD_CALL readBytes(uint8_t* bytes, uint64_t size) override
    { size = std::min<uint64_t>(size, data.size() - position); std::memcpy(bytes, data.data() + position, size); position += size; return size; }
    uint64_t OMNI_PVD_CALL skipBytes(uint64_t size) override
    { size = std::min<uint64_t>(size, data.size() - position); position += size; return size; }
};
struct WrittenCatalog
{
    Bytes bytes;
    std::unique_ptr<OmniPvdWriter> writer{createOmniPvdWriter()};
    uint32_t cls = 0;
    std::array<uint32_t, 6> fields{};
    std::array<uint32_t, 11> metaFields{};
    WrittenCatalog() { writer->setWriteStream(bytes); }
    void schema(int malformed = 0)
    {
        const auto base = malformed == 1 ? writer->registerClass("Base") : 0;
        cls = writer->registerClass("PxOmniPvdCaptureEntry", base);
        const char* names[] = {"targetHandle", "typeName", "name", "ownerHandle", "sceneHandle", "canSelect"};
        // Register in a different order than the genuine SDK, with exact shapes.
        for (int i : {5, 4, 3, 2, 1, 0})
        {
            if (malformed == 2 && i == 2) continue;
            const auto type = i == 1 || i == 2 ? OmniPvdDataType::eSTRING :
                (i == 5 ? OmniPvdDataType::eUINT8 : OmniPvdDataType::eUINT64);
            fields[i] = writer->registerAttribute(cls, malformed == 3 && i == 2 ? "typeName" : names[i],
                malformed == 4 && i == 0 ? OmniPvdDataType::eUINT32 : type, malformed == 5 && i == 1 ? 0 : 1);
        }
    }
    void field(uint64_t context, uint64_t handle, int index, const void* value, uint32_t size)
    { writer->setAttribute(context, handle, &fields[index], 1, static_cast<const uint8_t*>(value), size); }
    void entry(uint64_t handle = 10, uint64_t context = 2, const std::string& name = "box",
               uint64_t target = 0xfedcba9876543210ull)
    {
        writer->createObject(context, cls, handle, "");
        uint64_t zero = 0, scene = 99;
        uint8_t selectable = 1;
        field(context, handle, 0, &target, 8); field(context, handle, 1, "PxRigidDynamic", 14);
        field(context, handle, 2, name.data(), static_cast<uint32_t>(name.size()));
        field(context, handle, 3, &zero, 8); field(context, handle, 4, &scene, 8);
        field(context, handle, 5, &selectable, 1);
    }
    void metadata(uint64_t object = 1000, uint32_t major = 3)
    {
        const auto meta = writer->registerClass("PxOmniPvdMetaData");
        const char* names[] = {"physxVersionMajor", "physxVersionMinor", "physxVersionBugfix", "ovdIntegrationVersionMajor",
            "ovdIntegrationVersionMinor", "recordSecondaryData", "captureAllObjects", "captureRequestId",
            "captureRequestStatus", "captureEpoch", "captureCatalogGeneration"};
        for (int i = 0; i < 11; ++i)
            metaFields[i] = writer->registerAttribute(meta, names[i], i == 5 || i == 6 ? OmniPvdDataType::eUINT8 :
                (i == 7 || i >= 9 ? OmniPvdDataType::eUINT64 : OmniPvdDataType::eUINT32), 1);
        writer->createObject(1, meta, object, "");
        metaField(3, major, object);
    }
    void metaField(int index, uint64_t value, uint64_t object = 1000)
    {
        writer->setAttribute(1, object, &metaFields[index], 1, reinterpret_cast<const uint8_t*>(&value),
            index == 5 || index == 6 ? 1 : (index == 7 || index >= 9 ? 8 : 4));
    }
    void detailedSnapshot()
    {
        metadata();
        const auto objectType = OmniPvdDataType::eOBJECT_HANDLE;
        const auto scene = writer->registerClass("PxScene");
        const auto actors = writer->registerUniqueListAttribute(scene, "actors", objectType);
        const auto actor = writer->registerClass("PxActor");
        const auto shapes = writer->registerUniqueListAttribute(actor, "shapes", objectType);
        const auto rigid = writer->registerClass("PxRigidDynamic", actor);
        const auto shape = writer->registerClass("PxShape");
        const auto geometry = writer->registerAttribute(shape, "geometry", objectType, 1);
        const auto sphere = writer->registerClass("PxSphereGeometry");
        const auto radius = writer->registerAttribute(sphere, "radius", OmniPvdDataType::eFLOAT32, 1);
        writer->createObject(1, scene, 99, ""); writer->createObject(1, shape, 101, "");
        writer->createObject(1, sphere, 102, "");
        uint64_t geom = 102, sharedShape = 101; float unit = 1;
        writer->setAttribute(1, 101, &geometry, 1, reinterpret_cast<const uint8_t*>(&geom), 8);
        writer->setAttribute(1, 102, &radius, 1, reinterpret_cast<const uint8_t*>(&unit), 4);
        for (uint64_t id = 0x10000; id < 0x10400; ++id)
        {
            writer->createObject(1, rigid, id, "");
            writer->addToUniqueListAttribute(1, 99, actors, reinterpret_cast<const uint8_t*>(&id), 8);
            writer->addToUniqueListAttribute(1, id, shapes, reinterpret_cast<const uint8_t*>(&sharedShape), 8);
        }
        writer->startFrame(99, 1); writer->stopFrame(99, 1);
    }
};
struct ValidationResult
{
    OvdInputErrorCode error;
    uint64_t rows, commands, creates, payload, retained;
    CaptureMetadataSnapshot metadata;
    CaptureCatalogSnapshot catalog;
};
ValidationResult validate(WrittenCatalog& fixture, bool separate = true,
                          OvdResourceLimits limits = OvdResourceLimits::defaults())
{
    fixture.bytes.position = 0;
    OvdCaptureCatalog catalog;
    OvdValidatedReadStream gate(fixture.bytes, limits,
        separate ? CatalogValidationMode::SeparateNonRetained : CatalogValidationMode::Ordinary, &catalog);
    std::unique_ptr<OmniPvdReader> reader(createOmniPvdReader()); reader->setReadStream(gate);
    uint64_t rowCommands = 0;
    while (const auto command = reader->getNextCommand())
    {
        if (!gate.validateCommand(command, *reader)) break;
        if (gate.commandDestination() == OvdCommandDestination::Catalog) ++rowCommands;
    }
    const auto used = gate.usage();
    return {gate.errorCode(), rowCommands, used.commands, used.objectCreates, used.payloadBytes, used.retainedDomBytes,
        catalog.metadataSnapshot(), catalog.snapshot()};
}
}

TEST(CaptureValidation, SeparatesOnlyValidatedCatalogAndPreservesOrdinaryMode)
{
    WrittenCatalog fixture; fixture.schema(); fixture.entry();
    const auto separate = validate(fixture);
    ASSERT_EQ(separate.error, OvdInputErrorCode::None);
    EXPECT_EQ(separate.rows, 7u); EXPECT_EQ(separate.commands, 7u); EXPECT_EQ(separate.creates, 0u);
    EXPECT_EQ(separate.payload, 42u);
    const auto ordinary = validate(fixture, false);
    ASSERT_EQ(ordinary.error, OvdInputErrorCode::None);
    EXPECT_EQ(ordinary.rows, 0u); EXPECT_EQ(ordinary.commands, 14u); EXPECT_EQ(ordinary.creates, 1u);
    EXPECT_GT(ordinary.retained, separate.retained); EXPECT_EQ(ordinary.payload, separate.payload);
    auto limits = OvdResourceLimits::defaults(); limits.maxObjectCreates = 0;
    EXPECT_EQ(validate(fixture, true, limits).error, OvdInputErrorCode::None);
    EXPECT_EQ(validate(fixture, false, limits).error, OvdInputErrorCode::ObjectCountLimit);
    limits.maxCumulativeAttributePayloadBytes = 41;
    EXPECT_EQ(validate(fixture, true, limits).error, OvdInputErrorCode::CumulativeAttributePayloadLimit);
    limits = OvdResourceLimits::defaults(); limits.maxStreamBytes = fixture.bytes.data.size() - 1;
    EXPECT_EQ(validate(fixture, true, limits).error, OvdInputErrorCode::StreamBytesLimit);
    limits = OvdResourceLimits::defaults(); limits.maxSingleAttributePayloadBytes = 7;
    EXPECT_EQ(validate(fixture, true, limits).error, OvdInputErrorCode::AttributePayloadLimit);
}

TEST(CaptureValidation, ValidatesFixedPayloadWidthsAndBooleanRange)
{
    for (bool boolean : {false, true})
    {
        WrittenCatalog fixture; fixture.schema(); fixture.writer->createObject(2, fixture.cls, 10, "");
        uint64_t value = 2;
        fixture.field(2, 10, boolean ? 5 : 0, &value, boolean ? 1 : 4);
        EXPECT_EQ(validate(fixture).error, boolean ? OvdInputErrorCode::InvalidCaptureCatalog : OvdInputErrorCode::InvalidAttributePayload);
    }
}

TEST(CaptureValidation, RejectsMalformedSchemasAndWrongContext)
{
    for (int malformed = 1; malformed <= 5; ++malformed)
    {
        SCOPED_TRACE(malformed);
        WrittenCatalog fixture; fixture.schema(malformed); fixture.entry();
        EXPECT_EQ(validate(fixture).error, OvdInputErrorCode::InvalidCaptureCatalog);
    }
    WrittenCatalog wrongContext; wrongContext.schema(); wrongContext.entry(10, 1);
    EXPECT_EQ(validate(wrongContext).error, OvdInputErrorCode::InvalidCaptureCatalog);
    WrittenCatalog general;
    const auto cls = general.writer->registerClass("GeneralContextTwoObject");
    general.writer->createObject(2, cls, 10, "ordinary");
    const auto result = validate(general);
    EXPECT_EQ(result.error, OvdInputErrorCode::None); EXPECT_EQ(result.rows, 0u); EXPECT_EQ(result.creates, 1u);
}

TEST(CaptureValidation, RejectsNestedPathsAndForeignAttributes)
{
    for (bool nested : {false, true})
    {
        WrittenCatalog fixture; fixture.schema();
        const auto other = fixture.writer->registerClass("Other");
        const auto foreign = fixture.writer->registerAttribute(other, "targetHandle", OmniPvdDataType::eUINT64, 1);
        fixture.writer->createObject(2, fixture.cls, 10, "");
        uint64_t value = 4;
        const uint32_t path[] = {fixture.fields[0], nested ? fixture.fields[0] : foreign};
        fixture.writer->setAttribute(2, 10, nested ? path : &foreign, nested ? 2 : 1,
            reinterpret_cast<const uint8_t*>(&value), 8);
        EXPECT_EQ(validate(fixture).error, OvdInputErrorCode::InvalidCaptureCatalog);
    }
}

TEST(CaptureValidation, ValidatesActiveLifetimeOwnershipAndStringBounds)
{
    for (int bad = 0; bad < 5; ++bad)
    {
        SCOPED_TRACE(bad);
        WrittenCatalog fixture; fixture.schema(); fixture.entry();
        uint64_t value = 7;
        if (bad == 0) fixture.field(1, 10, 0, &value, 8);
        if (bad == 1) fixture.field(2, 10, 0, &value, 8); // repeated field
        if (bad == 2) fixture.writer->createObject(2, fixture.cls, 10, "");
        if (bad == 3) fixture.writer->destroyObject(1, 10);
        if (bad == 4) { fixture.writer->destroyObject(2, 10); fixture.field(2, 10, 0, &value, 8); }
        EXPECT_EQ(validate(fixture).error, OvdInputErrorCode::InvalidCaptureCatalog);
    }
    WrittenCatalog reuse; reuse.schema(); reuse.entry(); reuse.writer->destroyObject(2, 10);
    reuse.schema(); reuse.entry();
    EXPECT_EQ(validate(reuse).error, OvdInputErrorCode::None);
    WrittenCatalog boundary; boundary.schema(); boundary.entry(10, 2, std::string(2048, 'x'));
    EXPECT_EQ(validate(boundary).error, OvdInputErrorCode::None);
    WrittenCatalog oversized; oversized.schema(); oversized.entry(10, 2, std::string(2049, 'x'));
    EXPECT_EQ(validate(oversized).error, OvdInputErrorCode::CaptureCatalogStringLimit);
}

TEST(CaptureValidation, RejectsCrossDomainLiveHandleCollisions)
{
    for (bool catalogFirst : {false, true})
    {
        WrittenCatalog fixture; fixture.schema();
        const auto cls = fixture.writer->registerClass("Ordinary");
        if (catalogFirst) fixture.entry();
        fixture.writer->createObject(1, cls, 10, "ordinary");
        if (!catalogFirst) fixture.entry();
        EXPECT_EQ(validate(fixture).error, OvdInputErrorCode::InvalidCaptureCatalog);
    }
}

TEST(CaptureMetadata, PublishesCatalogWithoutFrameAndKeepsItAcrossMetadataRecreation)
{
    WrittenCatalog fixture; fixture.schema(); fixture.metadata(); fixture.entry(); fixture.metaField(10, 1);
    auto result = validate(fixture);
    ASSERT_EQ(result.error, OvdInputErrorCode::None);
    ASSERT_TRUE(result.catalog.ready); EXPECT_EQ(result.catalog.generation, 1u);
    EXPECT_EQ(result.metadata.boundarySerial, 0u);
    fixture.metadata(); fixture.metaField(5, 1); fixture.metaField(6, 0);
    result = validate(fixture);
    ASSERT_EQ(result.error, OvdInputErrorCode::None);
    EXPECT_TRUE(result.catalog.ready); EXPECT_EQ(result.catalog.generation, 1u);
    EXPECT_EQ(result.metadata.boundarySerial, 1u);
    EXPECT_TRUE(result.metadata.policyKnown); EXPECT_TRUE(result.metadata.secondary); EXPECT_FALSE(result.metadata.all);
    fixture.writer->destroyObject(2, 10); fixture.metaField(10, 2);
    result = validate(fixture);
    EXPECT_EQ(result.error, OvdInputErrorCode::None);
    EXPECT_TRUE(result.catalog.ready); EXPECT_EQ(result.catalog.generation, 2u); EXPECT_EQ(result.catalog.rows, 0u);
}

TEST(CaptureMetadata, PublishesOnlyCompleteOrderedAckOnCurrentMetadataLifetime)
{
    WrittenCatalog fixture; fixture.metadata(); fixture.metaField(7, 17); fixture.metaField(8, 1);
    EXPECT_EQ(validate(fixture).metadata.ackId, 0u);
    fixture.metaField(9, 4);
    auto result = validate(fixture);
    ASSERT_EQ(result.error, OvdInputErrorCode::None);
    EXPECT_EQ(result.metadata.ackId, 17u); EXPECT_EQ(result.metadata.ackStatus, 1u); EXPECT_EQ(result.metadata.captureEpoch, 4u);
    EXPECT_EQ(result.metadata.ackEpoch, 4u); const auto ackRevision = result.metadata.ackRevision;
    fixture.metaField(8, 1); fixture.metaField(7, 18); fixture.metaField(9, 5); // out of order
    result = validate(fixture);
    EXPECT_EQ(result.metadata.ackId, 17u); EXPECT_EQ(result.metadata.ackEpoch, 4u);
    EXPECT_EQ(result.metadata.ackRevision, ackRevision); EXPECT_EQ(result.metadata.captureEpoch, 5u);
    fixture.metaField(7, 19); fixture.metadata(); fixture.metaField(8, 1); fixture.metaField(9, 6);
    EXPECT_EQ(validate(fixture).metadata.ackId, 17u); // recreation cannot finish old ID
    fixture.metaField(7, 20); fixture.metaField(8, 4); fixture.metaField(9, 6);
    result = validate(fixture);
    EXPECT_EQ(result.metadata.ackId, 20u); EXPECT_EQ(result.metadata.ackStatus, 4u);
    EXPECT_EQ(result.metadata.ackEpoch, 6u); EXPECT_GT(result.metadata.ackRevision, ackRevision);
    fixture.metaField(7, 21, 1234); fixture.metaField(8, 1, 1234); fixture.metaField(9, 7, 1234);
    result = validate(fixture);
    EXPECT_EQ(result.metadata.ackId, 20u); EXPECT_EQ(result.metadata.captureEpoch, 6u);
    for (uint32_t status : {2u, 3u})
    {
        fixture.metaField(7, 20 + status); fixture.metaField(8, status); fixture.metaField(9, 6);
        EXPECT_EQ(validate(fixture).metadata.ackStatus, status);
    }
    // The observer reports actual old/unknown IDs; only Session's issued-ID
    // ring may correlate them. Do not hide an actual epoch behind a stale ACK.
    fixture.metaField(7, 17); fixture.metaField(8, 1); fixture.metaField(9, 7);
    EXPECT_EQ(validate(fixture).metadata.ackId, 17u);
    fixture.metaField(10, 1); // explicit empty Refresh does not create a segment
    fixture.metaField(7, 999); fixture.metaField(8, 1); fixture.metaField(9, 7);
    result = validate(fixture);
    EXPECT_EQ(result.metadata.ackId, 999u); EXPECT_EQ(result.metadata.captureEpoch, 7u);
    EXPECT_EQ(result.metadata.boundarySerial, 1u); EXPECT_EQ(result.catalog.generation, 1u);
}

TEST(CaptureMetadata, RejectsIncompleteCommitAndUnsupportedDiscoveryIntegration)
{
    WrittenCatalog fixture; fixture.schema(); fixture.metadata();
    fixture.writer->createObject(2, fixture.cls, 10, ""); fixture.metaField(10, 1);
    EXPECT_EQ(validate(fixture).error, OvdInputErrorCode::InvalidCaptureCatalog);
    WrittenCatalog unsupported; unsupported.metadata(1000, 5);
    EXPECT_EQ(validate(unsupported).error, OvdInputErrorCode::UnsupportedIntegrationVersion);
}

TEST(CaptureMetadata, AcceptsLegacyAndCurrentIntegrationDuringDiscovery)
{
    for (uint32_t major : {3u, 4u})
    {
        WrittenCatalog fixture; fixture.schema(); fixture.metadata(1000, major);
        fixture.entry(); fixture.metaField(10, 1);
        EXPECT_EQ(validate(fixture).error, OvdInputErrorCode::None);
        EXPECT_TRUE(validate(fixture).catalog.ready);
    }
}

TEST(CaptureMetadata, LegacyAndLateVersionFieldsCannotBypassDiscoveryVersionGate)
{
    for (bool late : {false, true})
    {
        WrittenCatalog fixture; fixture.schema();
        const auto cls = fixture.writer->registerClass("PxOmniPvdMetaData");
        if (late) fixture.writer->createObject(1, cls, 1000, "");
        const auto major = fixture.writer->registerAttribute(cls, "ovdIntegrationVersionMajor", OmniPvdDataType::eUINT32, 1);
        if (!late) fixture.writer->createObject(1, cls, 1000, "");
        const uint32_t version = 5;
        fixture.writer->setAttribute(1, 1000, &major, 1, reinterpret_cast<const uint8_t*>(&version), 4);
        fixture.entry();
        EXPECT_EQ(validate(fixture).error, OvdInputErrorCode::UnsupportedIntegrationVersion);
    }
}

TEST(CaptureMetadata, RejectsIncorrectControlFieldRegistration)
{
    for (bool duplicate : {false, true})
    {
        WrittenCatalog fixture;
        const auto cls = fixture.writer->registerClass("PxOmniPvdMetaData");
        fixture.writer->registerAttribute(cls, "captureEpoch", OmniPvdDataType::eUINT64, 1);
        // The current SDK writer refuses duplicate names. Encode the malformed
        // declaration directly so this still exercises the viewer's validator.
        const auto append = [&](const auto& value) {
            fixture.bytes.writeBytes(reinterpret_cast<const uint8_t*>(&value), sizeof(value));
        };
        const char* name = duplicate ? "captureEpoch" : "captureRequestId";
        append(uint8_t(OmniPvdCommand::eREGISTER_ATTRIBUTE));
        append(cls); append(uint32_t(2));
        append(uint16_t(OmniPvdDataType::eUINT32)); append(uint32_t(1));
        append(uint16_t(std::strlen(name)));
        fixture.bytes.writeBytes(reinterpret_cast<const uint8_t*>(name), std::strlen(name));
        EXPECT_EQ(validate(fixture).error, OvdInputErrorCode::InvalidAttributePayload);
    }
}

TEST(CaptureIngestion, DiscoveryOnlyEofIsNotRejectedAndSavedRowsStayOutOfDom)
{
    WrittenCatalog fixture; fixture.schema(); fixture.metadata(); fixture.entry(); fixture.metaField(10, 1);
    using Clock = std::chrono::steady_clock;
    const auto until = [](const std::function<bool()>& predicate) {
        const auto end = Clock::now() + std::chrono::seconds(5);
        do { if (predicate()) return true; std::this_thread::sleep_for(std::chrono::milliseconds(2)); } while (Clock::now() < end);
        return predicate();
    };
    const auto port = tcp_test::reservePort();
    const auto directory = std::filesystem::temp_directory_path() /
        ("ovd-catalog-" + std::to_string(port) + "-" + std::to_string(Clock::now().time_since_epoch().count()));
    ASSERT_TRUE(std::filesystem::create_directory(directory));
    const auto path = directory / "capture.ovd";
    struct Cleanup { std::filesystem::path path, directory;
        ~Cleanup() { std::error_code error; std::filesystem::remove(path, error); std::filesystem::remove(directory, error); }
    } cleanup{path, directory};
    OmniPvdDOMState dom; std::recursive_mutex mutex;
    auto limits = OvdResourceLimits::defaults(); limits.maxObjectCreates = 4096;
    OvdLiveSource source(&dom, &mutex, limits);
    ASSERT_TRUE(source.start(port, path.string().c_str()));
    tcp_test::Socket producer;
    ASSERT_TRUE(until([&] { return tcp_test::connect(producer, port); }));
    const std::array<uint8_t, 8> handshake{0x4c,0x44,0x56,0x4f,1,0,12,0};
    tcp_test::send(producer, handshake.data(), handshake.size());
    tcp_test::send(producer, fixture.bytes.data.data(), fixture.bytes.data.size());
    ASSERT_TRUE(until([&] { return source.catalogSnapshot().ready || source.hasEnded(); }));
    ASSERT_TRUE(source.catalogSnapshot().ready);
    ASSERT_TRUE(until([&] { return source.captureBudget().used.streamBytes == fixture.bytes.data.size(); }));
    const auto budget = source.captureBudget();
    EXPECT_EQ(budget.used.objectCreates, 1u);
    EXPECT_EQ(budget.previousSegment.objectCreates, 0u);
    EXPECT_GT(budget.previousCatalogBytes, 0u);
    EXPECT_FALSE(canApplyCapture(budget.used, budget.previousSegment, budget.limits));
    CaptureRequest apply; apply.requestId = 1; apply.all = true;
    EXPECT_THROW(source.queueCaptureRequest(apply), std::runtime_error);
    CaptureRequest refresh; refresh.requestId = 2; refresh.catalogOnly = true;
    ASSERT_NO_THROW(source.queueCaptureRequest(refresh));
    tcp_test::nonBlocking(producer, true);
    std::array<uint8_t, 28> reverse{}; size_t received = 0;
    ASSERT_TRUE(until([&] {
        std::error_code error;
        received += producer.readSome(reverse.data() + received, reverse.size() - received, error);
        return received == reverse.size();
    }));
    EXPECT_EQ(reverse[2], 0x44); EXPECT_EQ(reverse[8], 2); // only Refresh was sent
    EXPECT_EQ(source.maxFrame(), 0u);
    const auto page = source.catalogPage(1, 0, 128);
    ASSERT_EQ(page.entries.size(), 1u); EXPECT_EQ(page.entries[0].targetHandle, 0xfedcba9876543210ull);
    producer.close(); ASSERT_TRUE(until([&] { return source.hasEnded(); })); source.stop();
    EXPECT_FALSE(source.hasFailed()); EXPECT_EQ(source.maxFrame(), 0u);
    EXPECT_EQ(dom.mExternalToInternalHandleMap.count(10), 0u);
    EXPECT_EQ(dom.mObjectHandleToObjectMap.size(), 1u); // metadata only
    Bytes exported; ASSERT_TRUE(source.writeRecording(exported)); EXPECT_EQ(exported.data, fixture.bytes.data);
    OvdInputErrorCode error;
    auto reopened = loadOvdFile(path.string().c_str(), OvdResourceLimits::defaults(), &error);
    ASSERT_NE(reopened, nullptr) << ovdInputErrorMessage(error);
    EXPECT_EQ(reopened->dom()->mObjectHandleToObjectMap.size(), 1u);
    EXPECT_EQ(reopened->dom()->mExternalToInternalHandleMap.count(10), 0u);
}

namespace
{
void exerciseLiveScale(const std::vector<uint8_t>& bytes, const std::vector<size_t>& catalogs, const std::vector<size_t>& frames)
{
    using Clock = std::chrono::steady_clock;
    const auto port = tcp_test::reservePort();
    MallocAllocator allocator;
    std::unique_ptr<PvdEngine> engine(PvdEngine::connectLive(allocator, port, nullptr));
    ASSERT_NE(engine, nullptr);
    std::atomic<uint64_t> catalogObserved{0}, frameObserved{0};
    std::atomic<bool> cancel{false}; std::exception_ptr peerFailure;
    std::thread peer([&] {
        try
        {
            tcp_test::Socket socket;
            for (int attempt = 0;; ++attempt)
            {
                if (tcp_test::connect(socket, port)) break;
                if (attempt == 2500) throw std::runtime_error("TCP test connection timed out");
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            const std::array<uint8_t, 8> handshake{0x4c,0x44,0x56,0x4f,1,0,12,0};
            tcp_test::send(socket, handshake.data(), handshake.size()); size_t offset = 0;
            for (size_t generation = 0; generation < catalogs.size() && !cancel.load(); ++generation)
            {
                tcp_test::send(socket, bytes.data() + offset, catalogs[generation] - offset);
                while (catalogObserved.load() <= generation && !cancel.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (cancel.load()) break;
                tcp_test::send(socket, bytes.data() + catalogs[generation], frames[generation] - catalogs[generation]);
                while (frameObserved.load() <= generation && !cancel.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                offset = frames[generation];
            }
            socket.close();
        }
        catch (...) { peerFailure = std::current_exception(); }
    });
    struct Join { std::thread& peer; std::atomic<bool>& cancel; PvdEngine& engine;
        ~Join() { cancel.store(true); engine.stopLive(); if (peer.joinable()) peer.join(); }
    } join{peer, cancel, *engine};
    std::vector<double> latencies;
    const auto deadline = Clock::now() + std::chrono::seconds(120);
    while (!engine->isLiveEnded() && Clock::now() < deadline)
    {
        const auto start = Clock::now();
        const auto catalog = engine->captureCatalogSnapshot();
        const auto budget = engine->captureBudget();
        if (catalog.ready) EXPECT_LE(engine->captureCatalogPage(catalog.generation, 0, 128).entries.size(), 128u);
        {
        PvdEngine::LiveReadGuard guard(*engine);
        engine->refreshLive();
        if (catalog.ready && catalog.generation > catalogObserved.load())
        {
            // Producer is paused at the catalog commit: row refresh alone must
            // not publish a frame or rebuild the ordinary scene index.
            EXPECT_EQ(engine->liveIndexRebuildCount(), frameObserved.load());
            EXPECT_EQ(engine->getLiveMetadataBoundarySerial(), frameObserved.load());
            catalogObserved.store(catalog.generation);
        }
        if (!engine->hasLiveUnpublishedIndexChanges() && engine->getLiveCompletedFrameSerial() > frameObserved.load())
        {
            EXPECT_EQ(engine->getLiveMetadataBoundarySerial(), engine->getLiveCompletedFrameSerial());
            EXPECT_GT(budget.previousSegment.classes, 0u);
            frameObserved.store(engine->getLiveCompletedFrameSerial());
        }
        }
        latencies.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(engine->isLiveEnded());
    const auto stop = Clock::now(); cancel.store(true); engine->stopLive(); peer.join();
    EXPECT_LT(std::chrono::duration<double>(Clock::now() - stop).count(), 2.0);
    ASSERT_FALSE(peerFailure);
    EXPECT_EQ(engine->getLiveErrorCode(), OvdInputErrorCode::None);
    EXPECT_EQ(frameObserved.load(), 8u); EXPECT_EQ(engine->liveIndexRebuildCount(), 8u);
    ASSERT_GE(latencies.size(), 20u); std::sort(latencies.begin(), latencies.end());
    const auto p95 = latencies[(latencies.size() - 1) * 95 / 100];
    std::cout << "live polls=" << latencies.size() << " p95_ms=" << p95 << " max_ms=" << latencies.back() << '\n';
    EXPECT_LE(p95, 50.0); EXPECT_LE(latencies.back(), 250.0);
    struct Compare : OmniPvdWriteStream
    {
        const std::vector<uint8_t>& expected; size_t offset = 0;
        explicit Compare(const std::vector<uint8_t>& bytes) : expected(bytes) {}
        bool OMNI_PVD_CALL openStream() override { return true; }
        bool OMNI_PVD_CALL closeStream() override { return true; }
        bool OMNI_PVD_CALL flush() override { return true; }
        uint64_t OMNI_PVD_CALL writeBytes(const uint8_t* data, uint64_t size) override {
            if (size > expected.size() - offset || std::memcmp(data, expected.data() + offset, size)) return 0;
            offset += size; return size;
        }
    } compare(bytes);
    ASSERT_TRUE(engine->writeLiveRecording(compare)); EXPECT_EQ(compare.offset, bytes.size());
}
}

TEST(CaptureIngestion, CatalogEightMaxGenerationsWithEightApplies)
{
    WrittenCatalog fixture; fixture.bytes.data.reserve(512ull << 20);
    fixture.schema(); fixture.metadata();
    std::vector<size_t> catalogs, frames;
    for (uint64_t generation = 1; generation <= 8; ++generation)
    {
        if (generation > 1)
            for (uint64_t i = 0; i < 262144; ++i) fixture.writer->destroyObject(2, 0x100000 + i);
        for (uint64_t i = 0; i < 262144; ++i)
            fixture.entry(0x100000 + i, 2, "b", 0x20000000000000ull + i);
        fixture.metaField(10, generation);
        catalogs.push_back(fixture.bytes.data.size());
        fixture.detailedSnapshot();
        frames.push_back(fixture.bytes.data.size());
    }
    ASSERT_LT(fixture.bytes.data.size(), 512ull << 20);
    OvdCaptureCatalog catalog;
    OvdValidatedReadStream gate(fixture.bytes, OvdResourceLimits::defaults(), CatalogValidationMode::SeparateNonRetained, &catalog);
    std::unique_ptr<OmniPvdReader> reader(createOmniPvdReader()); reader->setReadStream(gate);
    auto dom = std::make_unique<OmniPvdDOMState>(); std::vector<OmniPvdMessage> messages;
    unsigned completedFrames = 0;
    while (const auto command = reader->getNextCommand())
    {
        ASSERT_TRUE(gate.validateCommand(command, *reader)) << ovdInputErrorMessage(gate.errorCode());
        if (gate.commandDestination() == OvdCommandDestination::Dom)
            applyOvdDomCommandAndCollect(*reader, *dom, command, messages);
        if (command == OmniPvdCommand::eSTOP_FRAME)
        {
            ++completedFrames;
            ASSERT_TRUE(validateOvdDomTopology(*dom));
            EXPECT_EQ(catalog.snapshot().generation, completedFrames);
        }
    }
    ASSERT_TRUE(gate.cleanEof()) << ovdInputErrorMessage(gate.errorCode());
    EXPECT_EQ(completedFrames, 8u);
    const auto usage = gate.usage(); const auto peak = catalog.snapshot().peakAllocations;
    EXPECT_LT(usage.objectCreates, 10000u); EXPECT_LT(usage.retainedDomBytes, 96ull << 20);
    EXPECT_EQ(usage.streamBytes, fixture.bytes.data.size()); EXPECT_LE(peak, 128ull << 20);
    EXPECT_LE(catalog.snapshot().stringBytes, 64ull << 20);
    auto adapter = std::make_unique<OvdDomAdapter>(std::move(dom), std::move(messages));
    ASSERT_TRUE(adapter->isValid()); EXPECT_EQ(adapter->dom()->mSceneCreations.size(), 8u);
    uint64_t expectedSegment = 1;
    for (const auto* scene : adapter->dom()->mSceneCreations)
        EXPECT_EQ(scene->mRecordingSegmentId, expectedSegment++);
    EXPECT_EQ(adapter->dom()->mExternalToInternalHandleMap.count(0x100000), 0u);
    std::cout << "scale raw=" << usage.streamBytes << " ordinary_creates=" << usage.objectCreates
              << " retained_charge=" << usage.retainedDomBytes << " catalog_peak=" << peak << '\n';
    const auto path = std::filesystem::temp_directory_path() /
        ("ovd-catalog-scale-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".ovd");
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::error_code error; std::filesystem::remove(path, error); } } cleanup{path};
    { std::ofstream output(path, std::ios::binary); output.write(reinterpret_cast<const char*>(fixture.bytes.data.data()), fixture.bytes.data.size());
      output.close(); ASSERT_TRUE(output); }
    adapter.reset();
    OvdInputErrorCode error;
    auto reopened = loadOvdFile(path.string().c_str(), OvdResourceLimits::defaults(), &error);
    ASSERT_NE(reopened, nullptr) << ovdInputErrorMessage(error);
    EXPECT_EQ(reopened->dom()->mSceneCreations.size(), 8u);
    expectedSegment = 1;
    for (const auto* scene : reopened->dom()->mSceneCreations)
        EXPECT_EQ(scene->mRecordingSegmentId, expectedSegment++);
    EXPECT_EQ(reopened->dom()->mExternalToInternalHandleMap.count(0x100000), 0u);
    reopened.reset();
    exerciseLiveScale(fixture.bytes.data, catalogs, frames);
}

TEST(CaptureAdmission, EnforcesEvery75PercentBoundaryWithoutOverflow)
{
    const auto limits = OvdResourceLimits::defaults();
    const OvdResourceUsage previous{};
    struct Counter { uint64_t OvdResourceUsage::*used; uint64_t OvdResourceLimits::*limit; };
    const Counter counters[] = {
        {&OvdResourceUsage::streamBytes, &OvdResourceLimits::maxStreamBytes},
        {&OvdResourceUsage::commands, &OvdResourceLimits::maxCommands},
        {&OvdResourceUsage::attributeMutations, &OvdResourceLimits::maxAttributeMutations},
        {&OvdResourceUsage::payloadBytes, &OvdResourceLimits::maxCumulativeAttributePayloadBytes},
        {&OvdResourceUsage::classes, &OvdResourceLimits::maxClasses},
        {&OvdResourceUsage::attributes, &OvdResourceLimits::maxAttributes},
        {&OvdResourceUsage::objectCreates, &OvdResourceLimits::maxObjectCreates},
        {&OvdResourceUsage::messages, &OvdResourceLimits::maxMessages},
        {&OvdResourceUsage::retainedDomBytes, &OvdResourceLimits::maxRetainedDomBytes}};
    EXPECT_TRUE(canApplyCapture({}, previous, limits));
    for (const auto& counter : counters)
    {
        OvdResourceUsage used{};
        used.*counter.used = (limits.*counter.limit) * 3 / 4 - 1;
        EXPECT_TRUE(canApplyCapture(used, previous, limits));
        ++(used.*counter.used);
        EXPECT_FALSE(canApplyCapture(used, previous, limits));
        used.*counter.used = UINT64_MAX;
        EXPECT_FALSE(canApplyCapture(used, previous, limits));
    }
}

TEST(CaptureAdmission, ReservesFloorsAndPreviousSegmentButRefreshIgnoresDomHeadroom)
{
    struct Floor { uint64_t OvdResourceUsage::*used; uint64_t OvdResourceLimits::*limit; uint64_t reserve; };
    const Floor floors[] = {
        {&OvdResourceUsage::objectCreates, &OvdResourceLimits::maxObjectCreates, 4096},
        {&OvdResourceUsage::retainedDomBytes, &OvdResourceLimits::maxRetainedDomBytes, 32ull << 20},
        {&OvdResourceUsage::commands, &OvdResourceLimits::maxCommands, 65536},
        {&OvdResourceUsage::attributeMutations, &OvdResourceLimits::maxAttributeMutations, 65536},
        {&OvdResourceUsage::streamBytes, &OvdResourceLimits::maxStreamBytes, 16ull << 20},
        {&OvdResourceUsage::classes, &OvdResourceLimits::maxClasses, 72},
        {&OvdResourceUsage::attributes, &OvdResourceLimits::maxAttributes, 450}};
    for (const auto& floor : floors)
    {
        auto limits = OvdResourceLimits::defaults(); limits.*floor.limit = floor.reserve + 1;
        OvdResourceUsage used{}, previous{}; used.*floor.used = 1;
        EXPECT_TRUE(canApplyCapture(used, previous, limits));
        used.*floor.used = 2;
        EXPECT_FALSE(canApplyCapture(used, previous, limits));
        used.*floor.used = 0; previous.*floor.used = floor.reserve + 2;
        EXPECT_FALSE(canApplyCapture(used, previous, limits));
        EXPECT_TRUE(canRefreshCapture(used, 1, limits));
    }
    auto limits = OvdResourceLimits::defaults(); OvdResourceUsage used{};
    used.retainedDomBytes = limits.maxRetainedDomBytes;
    EXPECT_TRUE(canRefreshCapture(used, 100, limits));
    used.streamBytes = limits.maxStreamBytes - 100;
    EXPECT_TRUE(canRefreshCapture(used, 100, limits));
    EXPECT_FALSE(canRefreshCapture(used, 101, limits));
    used.streamBytes = UINT64_MAX;
    EXPECT_FALSE(canRefreshCapture(used, 1, limits));
}

TEST(CaptureAdmission, UnseenLargerSnapshotStillHitsUnchangedHardLimit)
{
    auto limits = OvdResourceLimits::defaults(); limits.maxObjectCreates = 5000;
    ASSERT_TRUE(canApplyCapture({}, {}, limits));
    WrittenCatalog fixture;
    const auto cls = fixture.writer->registerClass("OrdinarySnapshotObject");
    for (uint64_t i = 1; i <= 5001; ++i) fixture.writer->createObject(1, cls, i, "");
    EXPECT_EQ(validate(fixture, true, limits).error, OvdInputErrorCode::ObjectCountLimit);
}
