/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "OvdFileIngest.h"

#include "OmniPvdCommands.h"
#include "OmniPvdFileWriteStream.h"
#include "OmniPvdLibraryFunctions.h"
#include "OmniPvdWriter.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <memory>
#include <new>
#include <set>
#include <string>
#include <vector>

namespace
{

void check(bool value, const char* expression, const char* file, int line)
{
    if (!value)
    {
        std::fprintf(stderr, "%s:%d: check failed: %s\n", file, line, expression);
        std::abort();
    }
}

#undef assert
#define assert(expression) check(static_cast<bool>(expression), #expression, __FILE__, __LINE__)

using ovdnext::data::Capture;
using ovdnext::data::Availability;
using ovdnext::data::AttributeId;
using ovdnext::data::ClassId;
using ovdnext::data::FrameAddress;
using ovdnext::data::LoadPlan;
using ovdnext::data::ObjectId;
using ovdnext::data::PropertyId;
using ovdnext::data::RecordedAttribute;
using ovdnext::data::RecordedAttributeKind;
using ovdnext::data::RecordedClass;
using ovdnext::data::RecordedClassModel;
using ovdnext::data::RecordedDataType;
using ovdnext::data::RecordedObject;
using ovdnext::omnipvd::FileIngestError;
using ovdnext::omnipvd::FileIngestLimits;
using ovdnext::omnipvd::FileIngestResult;
using ovdnext::omnipvd::ingestOvdFile;

const char* fixture(const char* name)
{
    static const std::string root = OVDDATAOMNIPVD_REPOSITORY_ROOT;
    static const std::string prefix = root + "/test/data/";
    static std::string path;
    path = prefix + name;
    return path.c_str();
}

std::string temporaryPath(const char* stem)
{
    static unsigned counter = 0;
    return std::string(stem) + "_" + std::to_string(static_cast<unsigned long>(std::time(nullptr)))
        + "_" + std::to_string(++counter) + ".ovd";
}

void copyPrefix(const char* source, const std::string& target, std::size_t limit)
{
    std::ifstream input(source, std::ios::binary);
    std::ofstream output(target.c_str(), std::ios::binary | std::ios::trunc);
    assert(input && output);
    std::vector<char> bytes(limit);
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.write(bytes.data(), input.gcount());
    assert(output.good());
}

std::vector<FrameAddress> addresses(const Capture& capture)
{
    std::vector<FrameAddress> result;
    for (const ovdnext::data::FrameStamp& stamp : capture.frames())
        result.push_back(stamp.address);
    return result;
}

class TemporaryFile
{
public:
    explicit TemporaryFile(const char* stem)
        : mPath(temporaryPath(stem))
    {
    }

    ~TemporaryFile()
    {
        std::remove(mPath.c_str());
    }

    const std::string& path() const { return mPath; }

private:
    std::string mPath;
};

void writeCapture(const std::string& path, const std::function<void(OmniPvdWriter&)>& write)
{
    OmniPvdWriter* writer = createOmniPvdWriter();
    OmniPvdFileWriteStream* stream = createOmniPvdFileWriteStream();
    assert(writer && stream);
    stream->setFileName(path.c_str());
    writer->setWriteStream(*stream);
    write(*writer);
    assert(stream->closeStream());
    destroyOmniPvdWriter(*writer);
    destroyOmniPvdFileWriteStream(*stream);
}

template <typename T>
void appendValue(std::vector<std::uint8_t>& bytes, T value)
{
    const std::uint8_t* first = reinterpret_cast<const std::uint8_t*>(&value);
    bytes.insert(bytes.end(), first, first + sizeof(value));
}

void appendString(std::vector<std::uint8_t>& bytes, const char* value)
{
    const std::uint16_t length = static_cast<std::uint16_t>(std::strlen(value));
    appendValue(bytes, length);
    bytes.insert(bytes.end(), value, value + length);
}

void writeRawFile(const std::string& path, const std::vector<std::uint8_t>& bytes)
{
    std::ofstream output(path.c_str(), std::ios::binary | std::ios::trunc);
    assert(output);
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    assert(output.good());
}

std::vector<std::uint8_t> rawHeader()
{
    std::vector<std::uint8_t> bytes;
    appendValue<std::uint32_t>(bytes, 0);
    appendValue<std::uint32_t>(bytes, 4);
    appendValue<std::uint32_t>(bytes, 0);
    return bytes;
}

void appendRegisterClass(std::vector<std::uint8_t>& bytes, std::uint32_t handle,
                         std::uint32_t baseHandle, const char* name)
{
    appendValue<std::uint8_t>(bytes, static_cast<std::uint8_t>(OmniPvdCommand::eREGISTER_CLASS));
    appendValue(bytes, handle);
    appendValue(bytes, baseHandle);
    appendString(bytes, name);
}

void appendRegisterClass(std::vector<std::uint8_t>& bytes, std::uint32_t handle,
                         const char* name)
{
    appendRegisterClass(bytes, handle, 0, name);
}

void appendRegisterUInt32Attribute(std::vector<std::uint8_t>& bytes,
                                   std::uint32_t classHandle,
                                   std::uint32_t attributeHandle,
                                   const char* name)
{
    appendValue<std::uint8_t>(bytes,
        static_cast<std::uint8_t>(OmniPvdCommand::eREGISTER_ATTRIBUTE));
    appendValue(bytes, classHandle);
    appendValue(bytes, attributeHandle);
    appendValue<std::uint16_t>(bytes,
        static_cast<std::uint16_t>(OmniPvdDataType::eUINT32));
    appendValue<std::uint32_t>(bytes, 1);
    appendString(bytes, name);
}

void appendCreateObject(std::vector<std::uint8_t>& bytes, std::uint64_t context,
                        std::uint32_t classHandle, std::uint64_t objectHandle,
                        const char* name)
{
    appendValue<std::uint8_t>(bytes, static_cast<std::uint8_t>(OmniPvdCommand::eCREATE_OBJECT));
    appendValue(bytes, context);
    appendValue(bytes, classHandle);
    appendValue(bytes, objectHandle);
    appendString(bytes, name);
}

void appendSetUInt32(std::vector<std::uint8_t>& bytes, std::uint64_t context,
                     std::uint64_t objectHandle, std::uint32_t attributeHandle,
                     std::uint32_t value)
{
    appendValue<std::uint8_t>(bytes, static_cast<std::uint8_t>(OmniPvdCommand::eSET_ATTRIBUTE));
    appendValue(bytes, context);
    appendValue(bytes, objectHandle);
    appendValue<std::uint8_t>(bytes, 1);
    appendValue(bytes, attributeHandle);
    appendValue<std::uint32_t>(bytes, sizeof(value));
    appendValue(bytes, value);
}

void assertValidationRejected(const std::string& path)
{
    const FileIngestResult result = ingestOvdFile(path, LoadPlan::allProperties());
    assert(result.error == FileIngestError::ValidationFailed);
    assert(!result.capture);
    assert(result.diagnostic == "unknown OVD command");
}

std::set<ovdnext::data::ObjectId> alive(const Capture& capture, std::size_t index)
{
    const std::shared_ptr<const ovdnext::data::FrameView> view = capture.view(index);
    assert(view);
    return std::set<ovdnext::data::ObjectId>(view->aliveObjects().begin(), view->aliveObjects().end());
}

void assertRecordedModelsEqual(const RecordedClassModel& lhs,
                               const RecordedClassModel& rhs)
{
    assert(lhs.classes().size() == rhs.classes().size());
    for (std::size_t index = 0; index != lhs.classes().size(); ++index)
    {
        const RecordedClass& left = lhs.classes()[index];
        const RecordedClass& right = rhs.classes()[index];
        assert(left.id == right.id);
        assert(left.baseClass == right.baseClass);
        assert(left.name == right.name);
        assert(left.declaredAttributes == right.declaredAttributes);
    }

    assert(lhs.attributes().size() == rhs.attributes().size());
    for (std::size_t index = 0; index != lhs.attributes().size(); ++index)
    {
        const RecordedAttribute& left = lhs.attributes()[index];
        const RecordedAttribute& right = rhs.attributes()[index];
        assert(left.id == right.id);
        assert(left.valueProperty == right.valueProperty);
        assert(left.ownerClass == right.ownerClass);
        assert(left.attributeClass == right.attributeClass);
        assert(left.flagsEnumClass == right.flagsEnumClass);
        assert(left.kind == right.kind);
        assert(left.dataType == right.dataType);
        assert(left.hasElementCount == right.hasElementCount);
        assert(left.elementCount == right.elementCount);
        assert(left.enumValue == right.enumValue);
        assert(left.name == right.name);
    }

    assert(lhs.objects().size() == rhs.objects().size());
    for (std::size_t index = 0; index != lhs.objects().size(); ++index)
    {
        const RecordedObject& left = lhs.objects()[index];
        const RecordedObject& right = rhs.objects()[index];
        assert(left.id == right.id);
        assert(left.recordedClass == right.recordedClass);
        assert(left.recordedName == right.recordedName);
    }
}

void assertFrameDirectoriesEqual(const Capture& lhs, const Capture& rhs)
{
    assert(lhs.frames().size() == rhs.frames().size());
    for (std::size_t index = 0; index != lhs.frames().size(); ++index)
    {
        assert(lhs.frames()[index].address == rhs.frames()[index].address);
        assert(lhs.frames()[index].publication == rhs.frames()[index].publication);
    }
}

void assertDefaultLimits(const FileIngestLimits& actual)
{
    assert(actual.maxSingleAttributePayloadBytes == 16ull * 1024 * 1024);
    assert(actual.maxCumulativeAttributePayloadBytes == 256ull * 1024 * 1024);
    assert(actual.maxStreamBytes == 512ull * 1024 * 1024);
    assert(actual.maxCommands == 2000000);
    assert(actual.maxAttributeMutations == 2000000);
    assert(actual.maxClasses == 4096);
    assert(actual.maxAttributes == 16384);
    assert(actual.maxAttributesPerClass == 4096);
    assert(actual.maxObjectCreates == 100000);
    assert(actual.maxMessages == 16384);
    assert(actual.maxRetainedValidationBytes == 512ull * 1024 * 1024);
}

void testOrdinaryDefaultConstructionUsesParserSafetyLimits()
{
    alignas(FileIngestLimits) unsigned char storage[sizeof(FileIngestLimits)];
    std::memset(storage, 0xa5, sizeof(storage));
    FileIngestLimits* actual = ::new (static_cast<void*>(storage)) FileIngestLimits;
    assertDefaultLimits(*actual);
    actual->~FileIngestLimits();
}

void testTrackedFixtureCompatibilityMatrixIsLiteral()
{
    // Every row is a hand-checked compatibility expectation for one committed OVD fixture.
    struct ExpectedFixture
    {
        const char* name;
        bool loads;
        FileIngestError error;
        std::size_t frames;
        const char* diagnosticFragment;
    };
    const ExpectedFixture expectations[] = {
        {"2026_04_01_15_21_33_0000019f4952a142.ovd", false,
            FileIngestError::TranslationFailed, 0,
            "command eSET_ATTRIBUTE, error UnknownObject"},
        {"2026_04_01_16_42_23_00000062b3443b1b.ovd", false,
            FileIngestError::TranslationFailed, 0,
            "command eSET_ATTRIBUTE, error UnknownObject"},
        {"2026_04_28_15_24_52_000000a751246bf6.ovd", false,
            FileIngestError::TranslationFailed, 0,
            "command eSET_ATTRIBUTE, error UnknownObject"},
        {"2026_04_29_10_49_28_0000004c43ddea6e.ovd", false,
            FileIngestError::TranslationFailed, 0,
            "command eSET_ATTRIBUTE, error UnknownObject"},
        {"articulation.ovd", true, FileIngestError::None, 3, ""},
        {"multi_geom.ovd", true, FileIngestError::None, 241, ""},
        {"multiscene_rigid.ovd", false, FileIngestError::TranslationFailed, 0,
            "command eSET_ATTRIBUTE, error UnknownObject"},
        {"particle_system.ovd", true, FileIngestError::None, 3, ""},
        {"resample_two_segment.ovd", true, FileIngestError::None, 241, ""},
        {"snippet_meshes.ovd", true, FileIngestError::None, 121, ""},
        {"test_scene.ovd", true, FileIngestError::None, 3, ""},
        {"testRecordMessage.ovd", true, FileIngestError::None, 0, ""},
    };

    for (const ExpectedFixture& expected : expectations)
    {
        const FileIngestResult result = ingestOvdFile(
            fixture(expected.name), LoadPlan::allProperties());
        assert(result.error == expected.error);
        assert(static_cast<bool>(result.capture) == expected.loads);
        if (result.capture)
        {
            assert(result.capture->frames().size() == expected.frames);
            assert(result.diagnostic.empty());
        }
        else
        {
            assert(expected.frames == 0);
            assert(result.diagnostic.find(expected.diagnosticFragment) != std::string::npos);
        }
    }
}

void testSceneOwnsImmutableViews()
{
    FileIngestResult result = ingestOvdFile(fixture("test_scene.ovd"), LoadPlan::allProperties());
    assert(result.error == FileIngestError::None);
    assert(result.capture && !result.capture->frames().empty());
    const std::shared_ptr<const ovdnext::data::FrameView> first = result.capture->view(0);
    assert(first && !first->aliveObjects().empty());
    assert(!alive(*result.capture, result.capture->frames().size() - 1).empty());
    const std::vector<ovdnext::data::ObjectId> firstAlive = first->aliveObjects();
    result.capture.reset();
    assert(first->aliveObjects() == firstAlive);
}

void testSceneFrameDirectoryIsCanonical()
{
    const FileIngestResult result = ingestOvdFile(
        fixture("test_scene.ovd"), LoadPlan::allProperties());
    assert(result.error == FileIngestError::None && result.capture);

    // Hand-checked against the committed fixture's raw START_FRAME/STOP_FRAME commands.
    const std::vector<FrameAddress> expected = {
        {0, 0x138ab174e00ull, 1},
        {0, 0x138ab174e00ull, 2},
        {0, 0x138ab174e00ull, 3},
    };
    assert(addresses(*result.capture) == expected);
}

void testTwoSegmentFrameDirectoryIsCanonical()
{
    FileIngestResult result = ingestOvdFile(
        fixture("resample_two_segment.ovd"), LoadPlan::allProperties());
    assert(result.error == FileIngestError::None && result.capture);

    // The raw fixture closes frames 1..120, then starts an unclosed frame 121 before
    // repeating its schema. That pending frame is cancelled at the segment boundary.
    // The second segment then closes frames 1..121.
    const std::uint64_t context = 0x2aa193658b0ull;
    const std::vector<ovdnext::data::FrameStamp>& frames = result.capture->frames();
    assert(frames.size() == 241);
    for (std::size_t index = 0; index != 120; ++index)
    {
        const FrameAddress expected{0, context, index + 1};
        assert(frames[index].address == expected);
    }
    for (std::size_t index = 120; index != frames.size(); ++index)
    {
        const FrameAddress expected{1, context, index - 119};
        assert(frames[index].address == expected);
    }
}

void testSceneRecordedDeclarationsAndClassAwareLifetimesAreLiteral()
{
    const FileIngestResult result = ingestOvdFile(
        fixture("test_scene.ovd"), LoadPlan::allProperties());
    assert(result.error == FileIngestError::None && result.capture);

    const std::shared_ptr<const RecordedClassModel> model =
        result.capture->recordedClassModel();
    assert(model);
    assert(model->classes().size() == 96);
    assert(model->attributes().size() == 593);
    assert(model->findClass(ClassId{49}));
    assert(model->findClass(ClassId{49})->name == "PxScene");
    assert(model->findClass(ClassId{61}));
    assert(model->findClass(ClassId{61})->baseClass == ClassId{60});
    assert(model->findClass(ClassId{60}));
    assert(model->findClass(ClassId{60})->baseClass == ClassId{58});
    assert(model->findClass(ClassId{58}));
    assert(model->findClass(ClassId{58})->baseClass == ClassId{57});

    const RecordedAttribute* gravity = model->findAttribute(AttributeId{306});
    assert(gravity);
    assert(gravity->ownerClass == ClassId{49});
    assert(gravity->valueProperty == PropertyId{164});
    assert(gravity->name == "gravity");
    assert(gravity->dataType == RecordedDataType::Float32);
    assert(gravity->hasElementCount && gravity->elementCount == 3);

    const RecordedAttribute* actorFlags = model->findAttribute(AttributeId{395});
    assert(actorFlags);
    assert(actorFlags->dataType == RecordedDataType::FlagsWord);
    assert(actorFlags->flagsEnumClass == ClassId{25});
    assert(!actorFlags->hasElementCount);

    std::size_t valuePropertyCount = 0;
    for (const RecordedAttribute& attribute : model->attributes())
        valuePropertyCount += attribute.valueProperty == PropertyId{0} ? 0 : 1;
    assert(valuePropertyCount == 451);

    const std::vector<ClassId> physicsClasses = model->findClassesByName("PxPhysics");
    assert(physicsClasses.size() == 1);
    const std::vector<AttributeId> scenes =
        model->findDeclaredAttributesByName(physicsClasses[0], "scenes");
    assert(scenes == std::vector<AttributeId>({AttributeId{261}}));
    const RecordedAttribute* sceneList = model->findAttribute(AttributeId{261});
    assert(sceneList);
    assert(sceneList->valueProperty == PropertyId{119});
    assert(sceneList->kind == RecordedAttributeKind::UniqueList);
    assert(sceneList->dataType == RecordedDataType::ObjectHandle);

    const RecordedAttribute* firstConstant = model->findAttribute(AttributeId{1});
    assert(firstConstant);
    assert(firstConstant->ownerClass != ClassId{0});
    assert(model->findClass(firstConstant->ownerClass));
    assert(model->findClass(firstConstant->ownerClass)->name == "PxConstraintFlag");
    assert(firstConstant->kind == RecordedAttributeKind::EnumConstant);
    assert(firstConstant->dataType == RecordedDataType::EnumValue);
    assert(firstConstant->name == "eBROKEN");
    assert(firstConstant->enumValue == 1);

    const ClassId expectedAliveClasses[] = {
        ClassId{46}, ClassId{47}, ClassId{48}, ClassId{49},
        ClassId{51}, ClassId{61}, ClassId{70}, ClassId{72}
    };
    assert(result.capture->frames().size() == 3);
    for (std::size_t publicationIndex = 0; publicationIndex != 3; ++publicationIndex)
    {
        const std::shared_ptr<const ovdnext::data::FrameView> view =
            result.capture->view(publicationIndex);
        assert(view);
        assert(view->stamp().publication == publicationIndex + 1);
        assert(view->recordedClassModel() == model);
        for (ClassId recordedClass : expectedAliveClasses)
        {
            const ovdnext::data::ClassObjectResult objects =
                view->aliveObjectsOfExactClass(recordedClass);
            assert(objects.classKnown);
            const std::size_t expected =
                publicationIndex == 2 && recordedClass == ClassId{49} ? 0 : 1;
            assert(objects.objects.size() == expected);
        }
    }
}

void testTwoSegmentRecordedDeclarationsAndObjectGenerationsAreLiteral()
{
    const FileIngestResult result = ingestOvdFile(
        fixture("resample_two_segment.ovd"), LoadPlan::allProperties());
    assert(result.error == FileIngestError::None && result.capture);

    const std::shared_ptr<const RecordedClassModel> model =
        result.capture->recordedClassModel();
    assert(model);
    assert(model->classes().size() == 148);
    assert(model->attributes().size() == 894);
    std::size_t valuePropertyCount = 0;
    for (const RecordedAttribute& attribute : model->attributes())
        valuePropertyCount += attribute.valueProperty == PropertyId{0} ? 0 : 1;
    assert(valuePropertyCount == 676);

    const std::vector<ClassId> repeatedScenes = model->findClassesByName("PxScene");
    assert(repeatedScenes.size() == 2);
    assert(repeatedScenes[0] != repeatedScenes[1]);
    const std::shared_ptr<const ovdnext::data::FrameView> firstSegment =
        result.capture->view(119);
    const std::shared_ptr<const ovdnext::data::FrameView> secondSegment =
        result.capture->view(120);
    assert(firstSegment && secondSegment);
    const ovdnext::data::ClassObjectResult firstScene =
        firstSegment->aliveObjectsOfExactClass(repeatedScenes[0]);
    const ovdnext::data::ClassObjectResult secondScene =
        secondSegment->aliveObjectsOfExactClass(repeatedScenes[1]);
    assert(firstScene.classKnown && firstScene.objects.size() == 1);
    assert(secondScene.classKnown && secondScene.objects.size() == 1);
    assert(firstScene.objects[0] != secondScene.objects[0]);

    const std::set<ObjectId> firstAlive(firstSegment->aliveObjects().begin(),
                                         firstSegment->aliveObjects().end());
    const std::set<ObjectId> secondAlive(secondSegment->aliveObjects().begin(),
                                          secondSegment->aliveObjects().end());
    std::vector<ObjectId> sharedGenerations;
    std::set_intersection(firstAlive.begin(), firstAlive.end(),
        secondAlive.begin(), secondAlive.end(), std::back_inserter(sharedGenerations));
    assert(sharedGenerations.empty());
}

void testRecordedAttributeSelectorChangesOnlyPayloadAvailability()
{
    const FileIngestResult all = ingestOvdFile(
        fixture("test_scene.ovd"), LoadPlan::allProperties());
    const FileIngestResult selected = ingestOvdFile(fixture("test_scene.ovd"),
        LoadPlan::onlyRecordedAttributes({{"PxScene", "gravity"}}));
    assert(all.error == FileIngestError::None && all.capture);
    assert(selected.error == FileIngestError::None && selected.capture);
    assert(all.capture->recordedClassModel() && selected.capture->recordedClassModel());
    assertRecordedModelsEqual(*all.capture->recordedClassModel(),
                              *selected.capture->recordedClassModel());
    assertFrameDirectoriesEqual(*all.capture, *selected.capture);
    assert(all.capture->stats().retainedPayloadBytes
        > selected.capture->stats().retainedPayloadBytes);
    assert(selected.capture->stats().filteredPayloadBytes > 0);

    for (std::size_t index = 0; index != all.capture->frames().size(); ++index)
        assert(alive(*all.capture, index) == alive(*selected.capture, index));

    const std::shared_ptr<const ovdnext::data::FrameView> allFirst = all.capture->view(0);
    const std::shared_ptr<const ovdnext::data::FrameView> selectedFirst =
        selected.capture->view(0);
    assert(allFirst && selectedFirst);
    const ovdnext::data::ClassObjectResult scenes =
        allFirst->aliveObjectsOfExactClass(ClassId{49});
    assert(scenes.classKnown && scenes.objects.size() == 1);
    const ObjectId scene = scenes.objects[0];
    const ovdnext::data::PropertyResult allGravity =
        allFirst->persistentProperty(scene, PropertyId{164});
    const ovdnext::data::PropertyResult selectedGravity =
        selectedFirst->persistentProperty(scene, PropertyId{164});
    assert(allGravity.availability == Availability::Present);
    assert(selectedGravity.availability == allGravity.availability);
    assert(selectedGravity.bytes == allGravity.bytes);

    const std::vector<ClassId> physicsClasses =
        all.capture->recordedClassModel()->findClassesByName("PxPhysics");
    assert(physicsClasses.size() == 1);
    const ovdnext::data::ClassObjectResult physicsObjects =
        allFirst->aliveObjectsOfExactClass(physicsClasses[0]);
    assert(physicsObjects.classKnown && physicsObjects.objects.size() == 1);
    const ObjectId physics = physicsObjects.objects[0];
    assert(allFirst->list(physics, PropertyId{119}).availability == Availability::Present);
    assert(selectedFirst->list(physics, PropertyId{119}).availability == Availability::NotLoaded);
}

void testEmptyPropertyPlanPreservesFramesAndLifecycleWithoutRetainedPayload()
{
    FileIngestResult all = ingestOvdFile(fixture("test_scene.ovd"), LoadPlan::allProperties());
    FileIngestResult filtered = ingestOvdFile(fixture("test_scene.ovd"),
                                               LoadPlan::onlyRecordedAttributes({}));
    assert(all.error == FileIngestError::None && all.capture);
    assert(filtered.error == FileIngestError::None && filtered.capture);
    assert(addresses(*all.capture) == addresses(*filtered.capture));
    for (std::size_t index = 0; index != all.capture->frames().size(); ++index)
        assert(alive(*all.capture, index) == alive(*filtered.capture, index));
    assert(filtered.capture->stats().retainedPayloadBytes == 0);
    assert(filtered.capture->stats().filteredPayloadBytes > 0);
}

void testMalformedExcludedPayloadReturnsNoCapture()
{
    const TemporaryFile file("ovddata_malformed_excluded_payload");
    writeCapture(file.path(), [](OmniPvdWriter& writer) {
        const OmniPvdClassHandle objectClass = writer.registerClass("Object");
        const OmniPvdAttributeHandle ignored = writer.registerAttribute(
            objectClass, "ignored", OmniPvdDataType::eUINT32, 1);
        writer.createObject(0, objectClass, 1, "object");
        const std::uint8_t malformed = 7;
        writer.setAttribute(0, 1, ignored, &malformed, sizeof(malformed));
    });

    const FileIngestResult result = ingestOvdFile(file.path(),
        LoadPlan::onlyRecordedAttributes({{"Object", "different"}}));
    assert(result.error == FileIngestError::ValidationFailed);
    assert(!result.capture);
    assert(result.diagnostic == "OVD attribute payload does not match its registered schema");
}

void testObjectNameIsCopiedFromTheReader()
{
    const TemporaryFile file("ovddata_object_name");
    writeCapture(file.path(), [](OmniPvdWriter& writer) {
        const OmniPvdClassHandle objectClass = writer.registerClass("NamedObject");
        writer.createObject(7, objectClass, 11, "fixture-object-name");
        writer.startFrame(7, 1);
        writer.stopFrame(7, 1);
    });

    const FileIngestResult result = ingestOvdFile(file.path(), LoadPlan::allProperties());
    assert(result.error == FileIngestError::None && result.capture);
    const std::shared_ptr<const RecordedClassModel> model =
        result.capture->recordedClassModel();
    assert(model && model->objects().size() == 1);
    assert(model->objects()[0].recordedName == "fixture-object-name");
}

void testTruncatedStreamReturnsNoCapture()
{
    const TemporaryFile file("ovddata_truncated");
    copyPrefix(fixture("test_scene.ovd"), file.path(), 16);
    const FileIngestResult result = ingestOvdFile(file.path(), LoadPlan::allProperties());
    assert(result.error == FileIngestError::TruncatedInput);
    assert(!result.capture);
    assert(result.diagnostic == "truncated OVD input");
}

void testCurrentAndLegacyIntegrationVersionsOpen()
{
    for (std::uint32_t version : {3u, 4u})
    {
        const TemporaryFile file("ovddata_supported_integration");
        writeCapture(file.path(), [version](OmniPvdWriter& writer) {
            const auto metadata = writer.registerClass("PxOmniPvdMetaData");
            const auto major = writer.registerAttribute(metadata,
                "ovdIntegrationVersionMajor", OmniPvdDataType::eUINT32, 1);
            writer.createObject(1, metadata, 1, "metadata");
            writer.setAttribute(1, 1, major,
                reinterpret_cast<const std::uint8_t*>(&version), sizeof(version));
            writer.startFrame(1, 1);
            writer.stopFrame(1, 1);
        });
        const auto result = ingestOvdFile(file.path(), LoadPlan::allProperties());
        assert(result.capture);
        assert(result.capture->frames().size() == 1);
    }
}

void testTooNewIntegrationVersionReturnsNoCapture()
{
    const TemporaryFile file("ovddata_too_new");
    writeCapture(file.path(), [](OmniPvdWriter& writer) {
        const OmniPvdClassHandle metadata = writer.registerClass("PxOmniPvdMetaData");
        const OmniPvdAttributeHandle major = writer.registerAttribute(
            metadata, "ovdIntegrationVersionMajor", OmniPvdDataType::eUINT32, 1);
        writer.createObject(1, metadata, 1, "metadata");
        const std::uint32_t tooNew = 5;
        writer.setAttribute(1, 1, major,
            reinterpret_cast<const std::uint8_t*>(&tooNew), sizeof(tooNew));
        writer.startFrame(1, 1);
        writer.stopFrame(1, 1);
    });

    const FileIngestResult result = ingestOvdFile(file.path(), LoadPlan::allProperties());
    assert(result.error == FileIngestError::UnsupportedIntegrationVersion);
    assert(!result.capture);
    assert(result.diagnostic == "unsupported OmniPVD integration major version");
}

// Catches treating only directly-declared metadata properties as the version gate.
void testInheritedTooNewIntegrationVersionReturnsNoCapture()
{
    const TemporaryFile file("ovddata_inherited_too_new");
    writeCapture(file.path(), [](OmniPvdWriter& writer) {
        const OmniPvdClassHandle base = writer.registerClass("IntegrationBase");
        const OmniPvdAttributeHandle major = writer.registerAttribute(
            base, "ovdIntegrationVersionMajor", OmniPvdDataType::eUINT32, 1);
        const OmniPvdClassHandle metadata = writer.registerClass("PxOmniPvdMetaData", base);
        writer.createObject(1, metadata, 1, "metadata");
        const std::uint32_t tooNew = 5;
        writer.setAttribute(1, 1, major,
            reinterpret_cast<const std::uint8_t*>(&tooNew), sizeof(tooNew));
        writer.startFrame(1, 1);
        writer.stopFrame(1, 1);
    });

    const FileIngestResult result = ingestOvdFile(file.path(), LoadPlan::allProperties());
    assert(result.error == FileIngestError::UnsupportedIntegrationVersion);
    assert(!result.capture);
    assert(result.diagnostic == "unsupported OmniPVD integration major version");
}

// Catches clearing source object classes at a derived create that is not an OvdData segment boundary.
void testDerivedMetadataCreateDoesNotForgetEarlierLiveMetadataObject()
{
    const TemporaryFile file("ovddata_derived_metadata_lifetime");
    writeCapture(file.path(), [](OmniPvdWriter& writer) {
        const OmniPvdClassHandle base = writer.registerClass("IntegrationBase");
        const OmniPvdAttributeHandle major = writer.registerAttribute(
            base, "ovdIntegrationVersionMajor", OmniPvdDataType::eUINT32, 1);
        const OmniPvdClassHandle metadata = writer.registerClass("PxOmniPvdMetaData", base);
        const OmniPvdClassHandle derived = writer.registerClass("DerivedMetaData", metadata);
        writer.createObject(1, derived, 1, "first");
        const std::uint32_t accepted = 3;
        writer.setAttribute(1, 1, major,
            reinterpret_cast<const std::uint8_t*>(&accepted), sizeof(accepted));
        writer.createObject(1, derived, 2, "second");
        const std::uint32_t tooNew = 5;
        writer.setAttribute(1, 1, major,
            reinterpret_cast<const std::uint8_t*>(&tooNew), sizeof(tooNew));
    });

    const FileIngestResult result = ingestOvdFile(file.path(), LoadPlan::allProperties());
    assert(result.error == FileIngestError::UnsupportedIntegrationVersion);
    assert(!result.capture);
    assert(result.diagnostic == "unsupported OmniPVD integration major version");
}

// Catches applying metadata version semantics without resolving the target object.
void testIntegrationVersionOnUnknownObjectRemainsACommandTranslationFailure()
{
    const TemporaryFile file("ovddata_version_unknown_object");
    writeCapture(file.path(), [](OmniPvdWriter& writer) {
        const OmniPvdClassHandle metadata = writer.registerClass("PxOmniPvdMetaData");
        const OmniPvdAttributeHandle major = writer.registerAttribute(
            metadata, "ovdIntegrationVersionMajor", OmniPvdDataType::eUINT32, 1);
        const std::uint32_t tooNew = 5;
        writer.setAttribute(1, 99, major,
            reinterpret_cast<const std::uint8_t*>(&tooNew), sizeof(tooNew));
    });

    const FileIngestResult result = ingestOvdFile(file.path(), LoadPlan::allProperties());
    assert(result.error == FileIngestError::TranslationFailed);
    assert(!result.capture);
    assert(result.diagnostic ==
        "OvdData command translation failed (command eSET_ATTRIBUTE, error UnknownObject, "
        "context 1, object 99)");
}

// Catches identical same-handle class registration clearing the adapter's live object catalog.
void testIdenticalClassRegistrationPreservesDerivedMetadataObject()
{
    const TemporaryFile file("ovddata_identical_class_registration");
    std::vector<std::uint8_t> bytes = rawHeader();
    appendRegisterClass(bytes, 1, "IntegrationBase");
    appendRegisterUInt32Attribute(bytes, 1, 1, "ovdIntegrationVersionMajor");
    appendRegisterClass(bytes, 2, 1, "PxOmniPvdMetaData");
    appendRegisterClass(bytes, 3, 2, "DerivedMetaData");
    appendCreateObject(bytes, 1, 3, 1, "metadata");
    appendRegisterClass(bytes, 3, 2, "DerivedMetaData");
    appendSetUInt32(bytes, 1, 1, 1, 5);
    writeRawFile(file.path(), bytes);

    const FileIngestResult result = ingestOvdFile(file.path(), LoadPlan::allProperties());
    assert(result.error == FileIngestError::UnsupportedIntegrationVersion);
    assert(!result.capture);
    assert(result.diagnostic == "unsupported OmniPVD integration major version");
}

// Catches allowing the raw reserved class handle to enter normalized ingestion.
void testZeroClassCreateReturnsValidationFailure()
{
    const TemporaryFile file("ovddata_zero_create_class");
    writeCapture(file.path(), [](OmniPvdWriter& writer) {
        writer.createObject(0, OMNI_PVD_INVALID_HANDLE, 1, "zero-class");
        writer.startFrame(0, 1);
        writer.stopFrame(0, 1);
    });
    assertValidationRejected(file.path());
}

// Catches allowing the raw reserved object handle to become a live generation.
void testZeroObjectCreateReturnsValidationFailure()
{
    const TemporaryFile file("ovddata_zero_create_object");
    writeCapture(file.path(), [](OmniPvdWriter& writer) {
        const OmniPvdClassHandle objectClass = writer.registerClass("Object");
        writer.createObject(0, objectClass, OMNI_PVD_INVALID_HANDLE, "zero-object");
        writer.startFrame(0, 1);
        writer.stopFrame(0, 1);
    });
    assertValidationRejected(file.path());
}

// Catches accidentally rejecting context zero while hardening other reserved handles.
void testZeroContextRemainsValid()
{
    const TemporaryFile file("ovddata_zero_context");
    writeCapture(file.path(), [](OmniPvdWriter& writer) {
        const OmniPvdClassHandle objectClass = writer.registerClass("Object");
        writer.createObject(0, objectClass, 1, "object");
        writer.startFrame(0, 1);
        writer.stopFrame(0, 1);
    });
    const FileIngestResult result = ingestOvdFile(file.path(), LoadPlan::allProperties());
    assert(result.error == FileIngestError::None);
    assert(result.capture && result.capture->frames().size() == 1);
}

// Catches accepting a raw reserved class registration handle.
void testZeroRegisteredClassHandleReturnsValidationFailure()
{
    const TemporaryFile file("ovddata_zero_registered_class");
    std::vector<std::uint8_t> bytes = rawHeader();
    appendRegisterClass(bytes, OMNI_PVD_INVALID_HANDLE, "ZeroClass");
    writeRawFile(file.path(), bytes);
    assertValidationRejected(file.path());
}

// Catches accepting reserved handles from any of the three attribute registrations.
void testZeroRegisteredAttributeHandlesReturnValidationFailure()
{
    for (std::uint8_t rawCommand : {
             static_cast<std::uint8_t>(OmniPvdCommand::eREGISTER_ATTRIBUTE),
             static_cast<std::uint8_t>(OmniPvdCommand::eREGISTER_CLASS_ATTRIBUTE),
             static_cast<std::uint8_t>(OmniPvdCommand::eREGISTER_UNIQUE_LIST_ATTRIBUTE)})
    {
        const TemporaryFile file("ovddata_zero_registered_attribute");
        std::vector<std::uint8_t> bytes = rawHeader();
        appendRegisterClass(bytes, 1, "Owner");
        appendValue(bytes, rawCommand);
        appendValue<std::uint32_t>(bytes, 1);
        appendValue<std::uint32_t>(bytes, OMNI_PVD_INVALID_HANDLE);
        if (rawCommand == static_cast<std::uint8_t>(OmniPvdCommand::eREGISTER_CLASS_ATTRIBUTE))
            appendValue<std::uint32_t>(bytes, 1);
        else
        {
            appendValue<std::uint16_t>(bytes,
                static_cast<std::uint16_t>(OmniPvdDataType::eUINT32));
            if (rawCommand == static_cast<std::uint8_t>(OmniPvdCommand::eREGISTER_ATTRIBUTE))
                appendValue<std::uint32_t>(bytes, 1);
        }
        appendString(bytes, "zero");
        writeRawFile(file.path(), bytes);
        assertValidationRejected(file.path());
    }
}

// The bounded SDK reader rejects unknown types before command translation.
void testUnsupportedNumericDeclarationDataTypeReturnsNoCapture()
{
    const TemporaryFile file("ovddata_unsupported_declaration_type");
    std::vector<std::uint8_t> bytes = rawHeader();
    appendRegisterClass(bytes, 1, "Owner");
    appendValue<std::uint8_t>(bytes,
        static_cast<std::uint8_t>(OmniPvdCommand::eREGISTER_ATTRIBUTE));
    appendValue<std::uint32_t>(bytes, 1);
    appendValue<std::uint32_t>(bytes, 1);
    appendValue<std::uint16_t>(bytes, 0xffffu);
    appendValue<std::uint32_t>(bytes, 1);
    appendString(bytes, "unsupported");
    writeRawFile(file.path(), bytes);

    const FileIngestResult result = ingestOvdFile(file.path(), LoadPlan::allProperties());
    assert(result.error == FileIngestError::IncompleteInput);
    assert(!result.capture);
    assert(result.diagnostic == "incomplete OmniPVD input");
}

// Catches accepting a class-valued attribute whose referenced class is invalid.
void testInvalidClassAttributeTargetsReturnValidationFailure()
{
    for (std::uint32_t targetClass : {std::uint32_t(OMNI_PVD_INVALID_HANDLE),
                                      std::uint32_t(99)})
    {
        const TemporaryFile file("ovddata_invalid_class_attribute_target");
        std::vector<std::uint8_t> bytes = rawHeader();
        appendRegisterClass(bytes, 1, "Owner");
        appendValue<std::uint8_t>(bytes,
            static_cast<std::uint8_t>(OmniPvdCommand::eREGISTER_CLASS_ATTRIBUTE));
        appendValue<std::uint32_t>(bytes, 1);
        appendValue<std::uint32_t>(bytes, 1);
        appendValue(bytes, targetClass);
        appendString(bytes, "target");
        writeRawFile(file.path(), bytes);
        assertValidationRejected(file.path());
    }
}

// Catches accepting a flags attribute whose required enum class is invalid.
void testInvalidFlagsEnumTargetsReturnValidationFailure()
{
    for (std::uint32_t enumClass : {std::uint32_t(OMNI_PVD_INVALID_HANDLE),
                                    std::uint32_t(99)})
    {
        const TemporaryFile file("ovddata_invalid_flags_enum_target");
        std::vector<std::uint8_t> bytes = rawHeader();
        appendRegisterClass(bytes, 1, "Owner");
        appendValue<std::uint8_t>(bytes,
            static_cast<std::uint8_t>(OmniPvdCommand::eREGISTER_ATTRIBUTE));
        appendValue<std::uint32_t>(bytes, 1);
        appendValue<std::uint32_t>(bytes, 1);
        appendValue<std::uint16_t>(bytes,
            static_cast<std::uint16_t>(OmniPvdDataType::eFLAGS_WORD));
        appendValue(bytes, enumClass);
        appendString(bytes, "flags");
        writeRawFile(file.path(), bytes);
        assertValidationRejected(file.path());
    }
}

} // namespace

int main()
{
    testOrdinaryDefaultConstructionUsesParserSafetyLimits();
    testTrackedFixtureCompatibilityMatrixIsLiteral();
    testSceneOwnsImmutableViews();
    testSceneFrameDirectoryIsCanonical();
    testTwoSegmentFrameDirectoryIsCanonical();
    testSceneRecordedDeclarationsAndClassAwareLifetimesAreLiteral();
    testTwoSegmentRecordedDeclarationsAndObjectGenerationsAreLiteral();
    testRecordedAttributeSelectorChangesOnlyPayloadAvailability();
    testEmptyPropertyPlanPreservesFramesAndLifecycleWithoutRetainedPayload();
    testMalformedExcludedPayloadReturnsNoCapture();
    testObjectNameIsCopiedFromTheReader();
    testTruncatedStreamReturnsNoCapture();
    testCurrentAndLegacyIntegrationVersionsOpen();
    testTooNewIntegrationVersionReturnsNoCapture();
    testZeroClassCreateReturnsValidationFailure();
    testZeroObjectCreateReturnsValidationFailure();
    testZeroContextRemainsValid();
    testZeroRegisteredClassHandleReturnsValidationFailure();
    testZeroRegisteredAttributeHandlesReturnValidationFailure();
    testUnsupportedNumericDeclarationDataTypeReturnsNoCapture();
    testInvalidClassAttributeTargetsReturnValidationFailure();
    testInvalidFlagsEnumTargetsReturnValidationFailure();
    testIntegrationVersionOnUnknownObjectRemainsACommandTranslationFailure();
    testIdenticalClassRegistrationPreservesDerivedMetadataObject();
    testInheritedTooNewIntegrationVersionReturnsNoCapture();
    testDerivedMetadataCreateDoesNotForgetEarlierLiveMetadataObject();
    return 0;
}
