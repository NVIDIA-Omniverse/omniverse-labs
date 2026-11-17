/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "TestFiles.h"
#pragma once

#ifdef PVD_OVD_SUPPORT

#include "LiveTestSupport.h"
#include "OvdDomAdapter.h"
#include "OvdLiveSource.h"
#include "OvdMessageCollector.h"
#include "OmniPvdReader.h"
#include "OmniPvdReadStream.h"
#include "OmniPvdWriteStream.h"

#include <algorithm>
#include <cstring>
#include <limits>

extern "C" {
    OmniPvdReader* OMNI_PVD_CALL createOmniPvdReader();
    void OMNI_PVD_CALL destroyOmniPvdReader(OmniPvdReader& reader);
}

namespace tst_memory_recording
{

using namespace live_test_support;

class OvdMemoryRecordingTest : public LiveCaptureTest {};

class ByteArrayWriteStream : public OmniPvdWriteStream
{
public:
    explicit ByteArrayWriteStream(uint64_t capacity = std::numeric_limits<int>::max())
        : mCapacity(capacity) {}

    uint64_t OMNI_PVD_CALL writeBytes(const uint8_t* bytes, uint64_t count) override
    {
        const uint64_t written = std::min(count, mCapacity - uint64_t(data.size()));
        data.append(reinterpret_cast<const char*>(bytes), static_cast<int>(written));
        return written;
    }
    bool OMNI_PVD_CALL flush() override { return true; }
    bool OMNI_PVD_CALL openStream() override { return true; }
    bool OMNI_PVD_CALL closeStream() override { return true; }

    std::string data;

private:
    uint64_t mCapacity;
};

class ByteArrayReadStream : public OmniPvdReadStream
{
public:
    explicit ByteArrayReadStream(const std::string& data) : mData(data) {}

    uint64_t OMNI_PVD_CALL readBytes(uint8_t* bytes, uint64_t count) override
    {
        const uint64_t read = std::min(count, uint64_t(mData.size()) - mOffset);
        if (read)
            std::memcpy(bytes, mData.c_str() + mOffset, static_cast<size_t>(read));
        mOffset += read;
        return read;
    }
    uint64_t OMNI_PVD_CALL skipBytes(uint64_t count) override
    {
        const uint64_t skipped = std::min(count, uint64_t(mData.size()) - mOffset);
        mOffset += skipped;
        return skipped;
    }
    bool OMNI_PVD_CALL openStream() override { return true; }
    bool OMNI_PVD_CALL closeStream() override { return true; }

private:
    const std::string& mData;
    uint64_t mOffset = 0;
};

TEST_F(OvdMemoryRecordingTest, RetentionOffExportsByteExactAndReloadsFromMemory)
{
    // This fixture crosses a recording chunk boundary and includes mesh payloads.
    const std::string fixture = readFixture(TEST_DATA_DIR "/snippet_meshes.ovd");
    ASSERT_GT(fixture.size(), 64 * 1024);
    const uint16_t port = testPort();
    ASSERT_NE(port, 0u);
    std::unique_ptr<PvdEngine> engine(PvdEngine::connectLive(m_allocator, port));
    ASSERT_NE(engine, nullptr);
    EXPECT_TRUE(engine->getLiveRecordingPath().empty());
    ASSERT_TRUE(sendCapture(engine.get(), port,
        [&](uint16_t producerPort) { return replayBytes(producerPort, fixture); }));
    engine->stopLive();
    ASSERT_EQ(engine->getLiveErrorCode(), OvdInputErrorCode::None);
    engine->refreshLive();

    ByteArrayWriteStream exported;
    ASSERT_EQ(engine->getLiveRecordingSize(), uint64_t(fixture.size()));
    ASSERT_TRUE(engine->writeLiveRecording(exported));
    ASSERT_EQ(exported.data, fixture);

    ByteArrayReadStream source(exported.data);
    OvdValidatedReadStream validated(source, OvdResourceLimits::defaults());
    const auto destroyReader = [](OmniPvdReader* reader)
    {
        if (reader)
            destroyOmniPvdReader(*reader);
    };
    std::unique_ptr<OmniPvdReader, decltype(destroyReader)> reader(
        createOmniPvdReader(), destroyReader);
    ASSERT_NE(reader, nullptr);
    reader->setReadStream(validated);
    std::unique_ptr<OmniPvdDOMState> dom(new OmniPvdDOMState());
    std::vector<OmniPvdMessage> messages;
    for (OmniPvdCommand::Enum command = reader->getNextCommand();
         command != OmniPvdCommand::eINVALID; command = reader->getNextCommand())
    {
        ASSERT_TRUE(validated.validateCommand(command, *reader));
        applyOvdDomCommandAndCollect(*reader, *dom, command, messages);
    }
    ASSERT_TRUE(validated.cleanEof());
    OvdDomAdapter reloaded(std::move(dom), std::move(messages));
    EXPECT_TRUE(reloaded.isValid());
    EXPECT_GT(reloaded.getNbInstances(), 0u);
    EXPECT_EQ(reloaded.getNbInstances(), engine->getNbInstances());
    EXPECT_EQ(reloaded.getMaxFrame(), engine->getMaxFrame());
}

TEST_F(OvdMemoryRecordingTest, ExportRequiresJoinedWorkerEvenAfterProducerEnded)
{
    const std::string fixture = readFixture(TEST_DATA_DIR "/multiscene_rigid.ovd");
    ASSERT_FALSE(fixture.empty());
    const uint16_t port = testPort();
    ASSERT_NE(port, 0u);
    std::unique_ptr<PvdEngine> engine(PvdEngine::connectLive(m_allocator, port));
    ASSERT_NE(engine, nullptr);
    ASSERT_TRUE(sendCapture(engine.get(), port,
        [&](uint16_t producerPort) { return replayBytes(producerPort, fixture); }));

    ByteArrayWriteStream exported;
    EXPECT_EQ(engine->getLiveRecordingSize(), uint64_t(fixture.size()));
    EXPECT_FALSE(engine->writeLiveRecording(exported));
    EXPECT_TRUE(exported.data.empty());
    engine->stopLive();
    EXPECT_TRUE(engine->writeLiveRecording(exported));
    EXPECT_EQ(exported.data, fixture);
}

TEST_F(OvdMemoryRecordingTest, PartialDestinationWriteFailsWithoutLosingRecording)
{
    const std::string fixture = readFixture(TEST_DATA_DIR "/snippet_meshes.ovd");
    ASSERT_GT(fixture.size(), 64 * 1024);
    const uint16_t port = testPort();
    ASSERT_NE(port, 0u);
    std::unique_ptr<PvdEngine> engine(PvdEngine::connectLive(m_allocator, port));
    ASSERT_NE(engine, nullptr);
    ASSERT_TRUE(sendCapture(engine.get(), port,
        [&](uint16_t producerPort) { return replayBytes(producerPort, fixture); }));
    engine->stopLive();

    const int acceptedBytes = 64 * 1024 + 17;
    ByteArrayWriteStream partial(acceptedBytes);
    EXPECT_FALSE(engine->writeLiveRecording(partial));
    EXPECT_EQ(partial.data, fixture.substr(0, acceptedBytes));
    ByteArrayWriteStream retry;
    EXPECT_TRUE(engine->writeLiveRecording(retry));
    EXPECT_EQ(retry.data, fixture);
}

TEST_F(OvdMemoryRecordingTest, OptionalFileMatchesMemoryAndIsNotNeededForExport)
{
    test_files::Directory directory;
    ASSERT_TRUE(directory.valid());
    const std::string path = directory.file("capture.ovd");
    const std::string encodedPath = path;
    const std::string fixture = readFixture(TEST_DATA_DIR "/snippet_meshes.ovd");
    ASSERT_FALSE(fixture.empty());
    const uint16_t port = testPort();
    ASSERT_NE(port, 0u);
    std::unique_ptr<PvdEngine> engine(PvdEngine::connectLive(
        m_allocator, port, encodedPath.c_str()));
    ASSERT_NE(engine, nullptr);
    ASSERT_TRUE(sendCapture(engine.get(), port,
        [&](uint16_t producerPort) { return replayBytes(producerPort, fixture); }));
    engine->stopLive();

    EXPECT_EQ(readFile(path), fixture);
    ASSERT_TRUE(test_files::fs::remove(test_files::fs::u8path(path)));
    ByteArrayWriteStream exported;
    EXPECT_TRUE(engine->writeLiveRecording(exported));
    EXPECT_EQ(exported.data, fixture);
}

TEST_F(OvdMemoryRecordingTest, TruncatedCaptureRetainsConsumedBytesButCannotExport)
{
    const std::string fixture = readFixture(TEST_DATA_DIR "/multiscene_rigid.ovd");
    ASSERT_FALSE(fixture.empty());
    const std::string truncated = fixture.substr(0, fixture.size() - 1);
    const uint16_t port = testPort();
    ASSERT_NE(port, 0u);
    std::unique_ptr<PvdEngine> engine(PvdEngine::connectLive(m_allocator, port));
    ASSERT_NE(engine, nullptr);
    ASSERT_TRUE(sendCapture(engine.get(), port,
        [&](uint16_t producerPort) { return replayBytes(producerPort, truncated); }));
    engine->stopLive();

    EXPECT_EQ(engine->getLiveErrorCode(), OvdInputErrorCode::TruncatedInput);
    EXPECT_EQ(engine->getLiveRecordingSize(), uint64_t(truncated.size()));
    ByteArrayWriteStream exported;
    EXPECT_FALSE(engine->writeLiveRecording(exported));
    EXPECT_TRUE(exported.data.empty());
}

TEST_F(OvdMemoryRecordingTest, StreamLimitBoundsRetainedBytesAndRejectsExport)
{
    const std::string fixture = readFixture(TEST_DATA_DIR "/multiscene_rigid.ovd");
    ASSERT_FALSE(fixture.empty());
    OvdResourceLimits limits = OvdResourceLimits::defaults();
    limits.maxStreamBytes = 12;
    OmniPvdDOMState dom;
    std::recursive_mutex mutex;
    OvdLiveSource source(&dom, &mutex, limits);
    const uint16_t port = testPort();
    ASSERT_NE(port, 0u);
    ASSERT_TRUE(source.start(port));
    std::thread producer([&] { replayBytes(port, fixture); });
    const bool ended = waitFor([&] { return source.hasEnded(); }, std::chrono::seconds(10));
    source.stop();
    producer.join();
    ASSERT_TRUE(ended);

    EXPECT_EQ(source.errorCode(), OvdInputErrorCode::StreamBytesLimit);
    EXPECT_EQ(source.recordingSize(), limits.maxStreamBytes);
    ByteArrayWriteStream exported;
    EXPECT_FALSE(source.writeRecording(exported));
    EXPECT_TRUE(exported.data.empty());
}

TEST_F(OvdMemoryRecordingTest, DestroyedCaptureDoesNotLeakIntoNextListener)
{
    const std::string fixture = readFixture(TEST_DATA_DIR "/multiscene_rigid.ovd");
    ASSERT_FALSE(fixture.empty());
    const size_t baselineBytes = m_allocator.getTotalBytesAllocated();
    for (int cycle = 0; cycle < 3; ++cycle)
    {
        const uint16_t port = testPort();
        ASSERT_NE(port, 0u);
        std::unique_ptr<PvdEngine> engine(PvdEngine::connectLive(m_allocator, port));
        ASSERT_NE(engine, nullptr);
        EXPECT_EQ(engine->getLiveRecordingSize(), 0u);
        ASSERT_TRUE(sendCapture(engine.get(), port,
            [&](uint16_t producerPort) { return replayBytes(producerPort, fixture); }));
        engine->stopLive();
        EXPECT_EQ(engine->getLiveRecordingSize(), uint64_t(fixture.size()));
        engine.reset();
        EXPECT_EQ(m_allocator.getTotalBytesAllocated(), baselineBytes);
    }
}

} // namespace tst_memory_recording

#endif // PVD_OVD_SUPPORT
