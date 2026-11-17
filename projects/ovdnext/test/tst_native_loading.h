/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "TestFiles.h"
#pragma once

#include "LoadControl.h"
#include "PVDClient.h"
#include "PvdEngine.h"
#include "PvdRandomAccessIOStream.h"
#include "PvdSimpleAllocator.h"
#include "pvdfileloader.h"
#include "PxPvdCommStreamEvents.h"

#ifdef PVD_OVD_SUPPORT
#include "OvdDomAdapter.h"
#include "OvdFileLoader.h"
#endif


#include <limits>
#include <memory>
#include <cstring>

namespace native_loading_test
{

using namespace physx;
using namespace physx::pvdsdk;
using namespace pvdapp;

class PointerConversionCaptureWriter : public pvd_loader_test::PvdLoaderTest
{
  public:
    using PvdLoaderTest::writePointerConversionCapture;
};

class NativeLoadingTest : public PvdEngineTest
{
  protected:
    static constexpr uint64_t StreamHeaderBytes = 28;

    struct CancelOnNamedAllocation : PxAllocatorCallback
    {
        PxAllocatorCallback& backing;
        LoadControl& control;
        const char* allocationName;
        unsigned cancellations = 0;

        CancelOnNamedAllocation(
            PxAllocatorCallback& alloc, LoadControl& load, const char* name)
            : backing(alloc), control(load), allocationName(name) {}

        void* allocate(
            size_t size, const char* name, const char* file, int line) override
        {
            void* result = backing.allocate(size, name, file, line);
            if(cancellations == 0 && name &&
               std::strcmp(name, allocationName) == 0)
            {
                ++cancellations;
                control.cancel();
            }
            return result;
        }

        void deallocate(void* ptr) override { backing.deallocate(ptr); }
    };

    template <typename T>
    static bool writeValue(test_files::File& file, const T& value)
    {
        return file.write(reinterpret_cast<const char*>(&value), sizeof(value)) ==
            sizeof(value);
    }

    static std::string rawPath(const test_files::Directory& root)
    {
        return root.file("capture.pxd2");
    }

    static std::string indexedPath(const test_files::Directory& root)
    {
        return root.file("capture.pxd3");
    }

    static bool writeRawCapture(const test_files::Directory& root)
    {
        test_files::File file(rawPath(root));
        if(!file.open(std::ios::out))
            return false;
        const PxU32 streamId = StreamInitialization::getStreamId();
        const PxU32 streamVersion = StreamInitialization::getStreamVersion();
        const PxU64 timestampNumerator = 1;
        const PxU64 timestampDenominator = 1;
        const PxU32 streamFlags = sizeof(void*) == 4 ? 0u : 1u;
        return writeValue(file, streamId) && writeValue(file, streamVersion) &&
            writeValue(file, timestampNumerator) &&
            writeValue(file, timestampDenominator) &&
            writeValue(file, streamFlags);
    }

    static PvdEngine* loadCapture(
        const test_files::Directory& root, const std::string& path, LoadControl* control,
        PxAllocatorCallback* allocator = nullptr)
    {
        const std::string encodedRoot = root.path();
        const std::string encodedPath = path;
        PxAllocatorCallback& callback = allocator ? *allocator : *m_alloc;
#ifdef PVD_OVD_SUPPORT
        return PvdEngine::load(
            callback, encodedRoot.c_str(), encodedPath.c_str(), nullptr,
            control);
#else
        return PvdEngine::load(
            callback, encodedRoot.c_str(), encodedPath.c_str(), control);
#endif
    }

    static bool writeIndexedCaptureFromRawCapture(const test_files::Directory& root)
    {
        std::unique_ptr<PvdEngine> source(
            loadCapture(root, rawPath(root), nullptr));
        if(!source)
            return false;

        const std::string path = indexedPath(root);
        ErrorCodeOption<PvdRandomAccessIOStream*> stream =
            PvdRandomAccessIOStream::open(
                *m_alloc, path.c_str(), true, false);
        if(!stream.hasValue())
            return false;
        source->save(stream.getValue());
        const bool flushed = !stream.getValue()->flush().hasValue();
        stream.getValue()->release();
        return flushed;
    }

    static bool writeIndexedCapture(const test_files::Directory& root)
    {
        return writeRawCapture(root) && writeIndexedCaptureFromRawCapture(root);
    }

    static bool writeIndexedCaptureWithInitialInstance(const test_files::Directory& root)
    {
        const std::string sourcePath = rawPath(root);
        const std::string parserBackingPath =
            root.file("parser-backing.pxd3");
        ErrorCodeOption<PvdRandomAccessIOStream*> source =
            PvdRandomAccessIOStream::open(
                *m_alloc, sourcePath.c_str(), false, false);
        if(!source.hasValue())
            return false;
        ErrorCodeOption<PvdRandomAccessIOStream*> parserBacking =
            PvdRandomAccessIOStream::open(
                *m_alloc, parserBackingPath.c_str(), true, true);
        if(!parserBacking.hasValue())
        {
            source.getValue()->release();
            return false;
        }

        PvdFileLoaderFactory& factory =
            PvdFileLoaderFactory::createFileLoaderFactory(*m_alloc);
        PvdFileStreamLoader* loader = factory.createFileStreamLoader(
            *source.getValue(), *parserBacking.getValue());
        factory.release();
        if(!loader)
        {
            source.getValue()->release();
            parserBacking.getValue()->release();
            return false;
        }
        for(;;)
        {
            const PvdFileStreamLoadResult result =
                loader->loadNextChunkResult(0x80000);
            if(result.mStatus == PvdFileStreamLoadResult::Failed)
            {
                loader->release();
                return false;
            }
            if(result.mStatus == PvdFileStreamLoadResult::Completed)
                break;
        }

        PvdClientMutator& mutator = loader->getClient();
        mutator.disableRecording();
        const std::string initialStateBackingPath = root.file("initial-state-backing.pxd3")
            ;
        ErrorCodeOption<PvdRandomAccessIOStream*> initialStateBacking =
            PvdRandomAccessIOStream::open(
                *m_alloc, initialStateBackingPath.c_str(), true, true);
        if(!initialStateBacking.hasValue())
        {
            loader->release();
            return false;
        }
        mutator.enableRecording(*initialStateBacking.getValue());

        const std::string outputPath = indexedPath(root);
        ErrorCodeOption<PvdRandomAccessIOStream*> output =
            PvdRandomAccessIOStream::open(
                *m_alloc, outputPath.c_str(), true, false);
        if(!output.hasValue())
        {
            loader->release();
            return false;
        }
        mutator.getClient().save(*output.getValue());
        const bool flushed = !output.getValue()->flush().hasValue();
        output.getValue()->release();
        loader->release();
        return flushed;
    }

    static bool writeIndexedHeader(
        const test_files::Directory& root, PxU32 version, PxU32 metaSize,
        PxU32 streamMetaSize, int words = 4)
    {
        test_files::File file(indexedPath(root));
        if(!file.open(std::ios::out))
            return false;
        const PxU32 fileId = PvdClient::getFileId();
        if(words >= 1 && !writeValue(file, fileId)) return false;
        if(words >= 2 && !writeValue(file, version)) return false;
        if(words >= 3 && !writeValue(file, metaSize)) return false;
        if(words >= 4 && !writeValue(file, streamMetaSize)) return false;
        return true;
    }

    static void expectRejectedLoadLeavesNoAllocation(
        const test_files::Directory& root, const std::string& path,
        LoadControl* control = nullptr, PxAllocatorCallback* allocator = nullptr)
    {
        pvdapp::CheckingAllocator* checkingAllocator =
            static_cast<pvdapp::CheckingAllocator*>(m_alloc);
        const size_t baseline = checkingAllocator->getTotalBytesAllocated();
        PvdEngine* engine = loadCapture(root, path, control, allocator);
        EXPECT_EQ(nullptr, engine);
        delete engine;
        EXPECT_EQ(baseline, checkingAllocator->getTotalBytesAllocated());
    }
};

TEST_F(NativeLoadingTest, PreCancelledRawLoadReturnsNullBeforeProgress)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    ASSERT_TRUE(writeRawCapture(root));
    LoadControl control;
    control.cancel();

    expectRejectedLoadLeavesNoAllocation(root, rawPath(root), &control);
    EXPECT_TRUE(control.progress().stage.empty());
}

TEST_F(NativeLoadingTest, RawLoadReportsSuccessfulFinalProgress)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    ASSERT_TRUE(writeRawCapture(root));
    LoadControl control;

    std::unique_ptr<PvdEngine> engine(
        loadCapture(root, rawPath(root), &control));
    ASSERT_NE(nullptr, engine);
    const LoadProgress progress = control.progress();
    EXPECT_EQ("Capture parsed", progress.stage);
    EXPECT_EQ(static_cast<uint64_t>(test_files::fs::file_size(test_files::fs::u8path(rawPath(root)))), progress.total);
    EXPECT_EQ(progress.total, progress.processed);
    EXPECT_TRUE(progress.interruptible);
}

TEST_F(NativeLoadingTest, RawCancellationAfterParsingStopsBeforeEndAndReleasesStreams)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    ASSERT_TRUE(PointerConversionCaptureWriter::writePointerConversionCapture(
        root, 3, 32768));
    const uint64_t captureSize =
        static_cast<uint64_t>(test_files::fs::file_size(test_files::fs::u8path(rawPath(root))));
    ASSERT_GT(captureSize, 0x80000u);
    LoadControl control;
    CancelOnNamedAllocation allocator(
        *m_alloc, control, "PvdObjectModel::Instance");

    expectRejectedLoadLeavesNoAllocation(
        root, rawPath(root), &control, &allocator);
    ASSERT_EQ(1u, allocator.cancellations);
    const LoadProgress progress = control.progress();
    EXPECT_EQ("Reading legacy stream", progress.stage);
    EXPECT_GT(progress.processed, 0u);
    EXPECT_LT(progress.processed, progress.total - StreamHeaderBytes);
    EXPECT_EQ(captureSize, progress.total);
    EXPECT_TRUE(progress.interruptible);
    EXPECT_EQ(
        std::vector<std::string>({test_files::fs::u8path(rawPath(root)).filename().u8string()}),
        test_files::fileNames(root.path()));
    EXPECT_TRUE(test_files::fs::remove(test_files::fs::u8path(rawPath(root))));
    EXPECT_TRUE(test_files::fileNames(root.path()).empty());
}

TEST_F(NativeLoadingTest, RawCancellationAfterParserCompletionDoesNotPublishFinalProgress)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    ASSERT_TRUE(PointerConversionCaptureWriter::writePointerConversionCapture(
        root, 1, 1));
    LoadControl control;
    CancelOnNamedAllocation allocator(
        *m_alloc, control, "PvdObjectModel::Instance");

    expectRejectedLoadLeavesNoAllocation(
        root, rawPath(root), &control, &allocator);
    ASSERT_EQ(1u, allocator.cancellations);
    const LoadProgress progress = control.progress();
    EXPECT_EQ("Reading legacy stream", progress.stage);
    EXPECT_GT(progress.processed, 0u);
    EXPECT_EQ(progress.total - StreamHeaderBytes, progress.processed);
    EXPECT_TRUE(progress.interruptible);
    EXPECT_EQ(
        std::vector<std::string>({test_files::fs::u8path(rawPath(root)).filename().u8string()}),
        test_files::fileNames(root.path()));
    EXPECT_TRUE(test_files::fs::remove(test_files::fs::u8path(rawPath(root))));
    EXPECT_TRUE(test_files::fileNames(root.path()).empty());
}

TEST_F(NativeLoadingTest, PreCancelledIndexedLoadReturnsNullBeforeProgress)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    ASSERT_TRUE(writeIndexedCapture(root));
    LoadControl control;
    control.cancel();

    PvdEngine* engine = loadCapture(root, indexedPath(root), &control);
    EXPECT_EQ(nullptr, engine);
    delete engine;
    EXPECT_TRUE(control.progress().stage.empty());
}

TEST_F(NativeLoadingTest, IndexedCancellationDuringOpenDoesNotEnterReader)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    ASSERT_TRUE(writeIndexedCapture(root));
    LoadControl control;
    CancelOnNamedAllocation allocator(
        *m_alloc, control, "RandomAccessIOStreamImpl");
    const auto baseline = static_cast<CheckingAllocator*>(m_alloc)->getTotalBytesAllocated();

    EXPECT_FALSE(control.cancelled());
    PvdEngine* engine = loadCapture(root, indexedPath(root), &control, &allocator);
    EXPECT_EQ(nullptr, engine);
    delete engine;
    ASSERT_EQ(1u, allocator.cancellations);
    EXPECT_TRUE(control.cancelled());
    // Cancellation while opening the real stream must not enter the expensive,
    // non-interruptible indexed stage and only then discard its result.
    EXPECT_TRUE(control.progress().stage.empty());
    EXPECT_EQ(baseline, static_cast<CheckingAllocator*>(m_alloc)->getTotalBytesAllocated());
    EXPECT_TRUE(test_files::fs::remove(test_files::fs::u8path(indexedPath(root)))); // No open source handle remains.
}

TEST_F(NativeLoadingTest, IndexedCancellationDuringClientParsingReleasesStream)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    ASSERT_TRUE(PointerConversionCaptureWriter::writePointerConversionCapture(
        root, 1, 1));
    ASSERT_TRUE(writeIndexedCaptureWithInitialInstance(root));
    LoadControl control;
    CancelOnNamedAllocation allocator(
        *m_alloc, control, "PvdObjectModel::Instance");

    expectRejectedLoadLeavesNoAllocation(
        root, indexedPath(root), &control, &allocator);
    ASSERT_EQ(1u, allocator.cancellations);
    const LoadProgress progress = control.progress();
    EXPECT_EQ("Reading indexed legacy capture", progress.stage);
    EXPECT_EQ(0u, progress.processed);
    EXPECT_EQ(0u, progress.total);
    EXPECT_FALSE(progress.interruptible);
    EXPECT_TRUE(test_files::fs::remove(test_files::fs::u8path(indexedPath(root))));
}

TEST_F(NativeLoadingTest, IndexedLoadReportsSuccessfulFinalProgress)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    ASSERT_TRUE(writeIndexedCapture(root));
    LoadControl control;

    std::unique_ptr<PvdEngine> engine(
        loadCapture(root, indexedPath(root), &control));
    ASSERT_NE(nullptr, engine);
    const LoadProgress progress = control.progress();
    EXPECT_EQ("Capture parsed", progress.stage);
    EXPECT_EQ(static_cast<uint64_t>(test_files::fs::file_size(test_files::fs::u8path(indexedPath(root)))), progress.total);
    EXPECT_EQ(progress.total, progress.processed);
    EXPECT_TRUE(progress.interruptible);
}

TEST_F(NativeLoadingTest, RejectsTruncatedIndexedHeaderAndReleasesStream)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    ASSERT_TRUE(writeIndexedHeader(root, 1, 0, 0, 3));
    LoadControl control;

    expectRejectedLoadLeavesNoAllocation(root, indexedPath(root), &control);
    const LoadProgress progress = control.progress();
    EXPECT_EQ("Reading indexed legacy capture", progress.stage);
    EXPECT_FALSE(progress.interruptible);
}

TEST_F(NativeLoadingTest, RejectsUnsupportedIndexedVersionAndReleasesStream)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    ASSERT_TRUE(writeIndexedHeader(root, 2, 0, 0));

    expectRejectedLoadLeavesNoAllocation(root, indexedPath(root));
}

TEST_F(NativeLoadingTest, RejectsOverflowingIndexedMetadataBoundsAndReleasesStream)
{
    test_files::Directory root;
    ASSERT_TRUE(root.valid());
    ASSERT_TRUE(writeIndexedHeader(
        root, 1, std::numeric_limits<PxU32>::max(),
        1));

    expectRejectedLoadLeavesNoAllocation(root, indexedPath(root));
}

#ifdef PVD_OVD_SUPPORT
TEST_F(NativeLoadingTest, PreCancelledOvdLoadReturnsNullWithoutAnError)
{
    LoadControl control;
    control.cancel();
    OvdInputErrorCode error = OvdInputErrorCode::InvalidDomTopology;

    std::unique_ptr<OvdDomAdapter> adapter = loadOvdFile(
        TEST_DATA_DIR "/test_scene.ovd", OvdResourceLimits::defaults(), &error,
        &control);

    EXPECT_EQ(nullptr, adapter);
    EXPECT_EQ(OvdInputErrorCode::None, error);
    EXPECT_TRUE(control.progress().stage.empty());
}

TEST_F(NativeLoadingTest, OvdLoadReportsSuccessfulFinalProgress)
{
    LoadControl control;
    OvdInputErrorCode error = OvdInputErrorCode::None;

    std::unique_ptr<OvdDomAdapter> adapter = loadOvdFile(
        TEST_DATA_DIR "/test_scene.ovd", OvdResourceLimits::defaults(), &error,
        &control);

    ASSERT_NE(nullptr, adapter);
    EXPECT_EQ(OvdInputErrorCode::None, error);
    const LoadProgress progress = control.progress();
    EXPECT_EQ("Capture parsed", progress.stage);
    EXPECT_GT(progress.total, 0u);
    EXPECT_EQ(progress.total, progress.processed);
    EXPECT_TRUE(progress.interruptible);
}
#endif

} // namespace native_loading_test
