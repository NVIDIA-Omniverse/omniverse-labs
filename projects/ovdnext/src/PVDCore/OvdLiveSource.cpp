/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifdef PVD_OVD_SUPPORT

#include "OvdLiveSource.h"
#include "OvdDomAdapter.h"
#include "OvdInputValidation.h"
#include "OvdLog.h"
#include "OvdMessageCollector.h"
#include "OvdCancelableSocketReadStream.h"

#include "OmniPvdReader.h"
#include "OmniPvdReadStream.h"
#include "OmniPvdWriteStream.h"
#include "OmniPvdSocketReadStream.h"
#include "OmniPvdCommands.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <new>
#include <stdexcept>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// pvdruntime factory functions are compiled directly into the PvdDom static lib
// (standalone mode, no DLL loading); call them through the same extern "C"
// convention PvdDomParser.cpp uses for the file reader/stream factories.
extern "C" {
    OmniPvdReader* OMNI_PVD_CALL createOmniPvdReader();
    void OMNI_PVD_CALL destroyOmniPvdReader(OmniPvdReader& reader);
}

namespace pvdapp
{

// Append-only chunks keep payload addresses stable and bound individual
// allocations. Only the worker mutates storage; readers can poll the published
// byte count, and export accesses the chunks only after stop() joins the worker.
class OvdLiveRecording
{
public:
    explicit OvdLiveRecording(uint64_t maxBytes) : mMaxBytes(maxBytes) {}

    void append(const uint8_t* bytes, uint64_t count)
    {
        if (mError != OvdInputErrorCode::None)
            return;
        uint64_t size = mSize.load();
        const uint64_t retained = std::min(count, mMaxBytes - size);
        uint64_t copied = 0;
        try
        {
            while (copied < retained)
            {
                const uint64_t offset = size % kChunkBytes;
                if (offset == 0)
                {
                    const size_t capacity = static_cast<size_t>(
                        std::min(kChunkBytes, mMaxBytes - size));
                    std::unique_ptr<uint8_t[]> chunk(new uint8_t[capacity]);
                    mChunks.push_back(std::move(chunk));
                }
                const uint64_t amount = std::min(retained - copied, kChunkBytes - offset);
                std::memcpy(mChunks.back().get() + offset, bytes + copied,
                            static_cast<size_t>(amount));
                size += amount;
                copied += amount;
                mSize.store(size);
            }
        }
        catch (const std::bad_alloc&)
        {
            mError = OvdInputErrorCode::ReaderRejectedCommand;
            return;
        }
        catch (const std::length_error&)
        {
            mError = OvdInputErrorCode::ReaderRejectedCommand;
            return;
        }
        if (retained != count)
            mError = OvdInputErrorCode::StreamBytesLimit;
    }

    uint64_t size() const { return mSize.load(); }
    OvdInputErrorCode errorCode() const { return mError; }

    bool writeTo(OmniPvdWriteStream& destination) const
    {
        if (mError != OvdInputErrorCode::None)
            return false;
        uint64_t remaining = size();
        try
        {
            for (const std::unique_ptr<uint8_t[]>& chunk : mChunks)
            {
                const uint64_t amount = std::min(remaining, kChunkBytes);
                if (destination.writeBytes(chunk.get(), amount) != amount)
                    return false;
                remaining -= amount;
            }
        }
        catch (...)
        {
            // Destination implementations can fail by throwing as well as by
            // returning a short write. The retained source stays retryable.
            return false;
        }
        return remaining == 0;
    }

private:
    static constexpr uint64_t kChunkBytes = 64 * 1024;
    const uint64_t mMaxBytes;
    std::vector<std::unique_ptr<uint8_t[]>> mChunks;
    std::atomic<uint64_t> mSize{0};
    OvdInputErrorCode mError = OvdInputErrorCode::None;
};

constexpr uint64_t OvdLiveRecording::kChunkBytes;

} // namespace pvdapp

namespace
{
// Owns one file created with OS-level exclusive-create semantics. It never
// opens, truncates, or appends to an existing path, so both viewer-generated
// candidates and paths supplied through PvdEngine::connectLive have the same
// no-overwrite guarantee. The handle stays open for the whole recording.
class ExclusiveFileWriteStream : public OmniPvdWriteStream
{
public:
    explicit ExclusiveFileWriteStream(const char* path)
        : mPath(std::filesystem::u8path(path ? path : "")), mFile(nullptr)
    {
    }

    ~ExclusiveFileWriteStream() override
    {
        closeStream();
    }

    uint64_t OMNI_PVD_CALL writeBytes(const uint8_t* bytes, uint64_t nbrBytes) override
    {
        if (!mFile || (!bytes && nbrBytes))
            return 0;

        uint64_t written = 0;
        while (written < nbrBytes)
        {
            const size_t chunk = static_cast<size_t>(std::min<uint64_t>(
                nbrBytes - written, std::numeric_limits<size_t>::max()));
            const size_t count = std::fwrite(bytes + written, 1, chunk, mFile);
            written += count;
            if (count != chunk)
                break;
        }
        return written;
    }

    bool OMNI_PVD_CALL flush() override
    {
        return mFile && std::fflush(mFile) == 0;
    }

    bool OMNI_PVD_CALL openStream() override
    {
        if (mFile)
            return true;
        if (mPath.empty())
            return false;

        int fd = -1;
#ifdef _WIN32
        if (_wsopen_s(&fd, mPath.c_str(),
                     _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY | _O_NOINHERIT,
                     _SH_DENYWR, _S_IREAD | _S_IWRITE) != 0)
            return false;
        mFile = _fdopen(fd, "wb");
#else
        fd = ::open(mPath.c_str(), O_WRONLY | O_CREAT | O_EXCL,
                    S_IRUSR | S_IWUSR);
        if (fd < 0)
            return false;
        mFile = fdopen(fd, "wb");
#endif
        if (!mFile)
        {
#ifdef _WIN32
            _close(fd);
#else
            ::close(fd);
#endif
            // Exclusive create proves this process created the path, so cleanup
            // cannot remove a pre-existing file.
#ifdef _WIN32
            _wremove(mPath.c_str());
#else
            std::remove(mPath.c_str());
#endif
            return false;
        }
        return true;
    }

    bool OMNI_PVD_CALL closeStream() override
    {
        if (!mFile)
            return true;
        FILE* file = mFile;
        mFile = nullptr;
        return std::fclose(file) == 0;
    }

private:
    std::filesystem::path mPath;
    FILE* mFile;
};

// Installed below validation after the transport handshake, so the capture
// contains the OVD version header and consumed command bytes, including skips.
class TeeReadStream : public OmniPvdReadStream
{
public:
    TeeReadStream(OmniPvdReadStream* inner, pvdapp::OvdLiveRecording& recording,
                  OmniPvdWriteStream* sink)
        : mInner(inner), mRecording(recording), mSink(sink) {}

    uint64_t OMNI_PVD_CALL readBytes(uint8_t* bytes, uint64_t nbrBytes) override
    {
        if (mRecording.errorCode() != OvdInputErrorCode::None)
            return 0;
        const uint64_t got = mInner->readBytes(bytes, nbrBytes);
        if (got)
        {
            mRecording.append(bytes, got);
            if (mSink && !mFileFailed)
            {
                mFileFailed = mSink->writeBytes(bytes, got) != got;
                if (mFileFailed)
                    ovdLog("[OVD-live] file recording write failed; memory capture retained\n");
            }
        }
        return got;
    }

    uint64_t OMNI_PVD_CALL skipBytes(uint64_t nbrBytes) override
    {
        // Read-and-tee skipped bytes too so the recorded file stays byte-exact.
        uint8_t buf[4096];
        uint64_t done = 0;
        while(done < nbrBytes)
        {
            const uint64_t chunk = std::min<uint64_t>(nbrBytes - done, sizeof(buf));
            const uint64_t g = readBytes(buf, chunk);
            if(!g)
                break;
            done += g;
        }
        return done;
    }

    bool OMNI_PVD_CALL openStream() override { return mInner->openStream(); }
    bool OMNI_PVD_CALL closeStream() override { return mInner->closeStream(); }

private:
    OmniPvdReadStream*  mInner;
    pvdapp::OvdLiveRecording& mRecording;
    OmniPvdWriteStream* mSink;
    bool mFileFailed = false;
};
} // anonymous namespace

namespace pvdapp
{

void OvdLiveSegmentEpochState::reset()
{
    mSeenClassNames.clear();
    mAvailableEpoch = 0;
    mCompletedEpoch = 0;
}

void OvdLiveSegmentEpochState::observeClassRegistration(const char* className)
{
    if (!className)
        return;
    const std::string name(className);
    if (mSeenClassNames.insert(name).second)
        return;

    ++mAvailableEpoch;
    mSeenClassNames.clear();
    mSeenClassNames.insert(name);
}

uint64_t OvdLiveSegmentEpochState::completeFrame(bool hasRecordingSegmentMetadata,
                                                  uint64_t recordingSegmentId)
{
    mCompletedEpoch = hasRecordingSegmentMetadata
        ? recordingSegmentId : mAvailableEpoch;
    return mCompletedEpoch;
}

OvdLivePollDecision decideOvdLivePollUpdate(uint64_t previousEpoch,
                                             uint64_t previousFrame,
                                             uint64_t currentEpoch,
                                             uint64_t currentFrame)
{
    const bool segmentChanged = currentEpoch != previousEpoch;
    return OvdLivePollDecision{
        segmentChanged,
        segmentChanged || currentFrame != previousFrame
    };
}

OvdLiveSource::OvdLiveSource(OmniPvdDOMState* dom, std::recursive_mutex* mutex,
                             const OvdResourceLimits& limits,
                             std::vector<OmniPvdMessage>* messages)
    : mDom(dom)
    , mMutex(mutex)
    , mMessages(messages)
    , mReader(nullptr)
    , mStream(nullptr)
    , mValidator(nullptr)
    , mLimits(limits)
    , mCaptureCatalog(new OvdCaptureCatalog)
    , mTee(nullptr)
    , mRecordStream(nullptr)
    , mStop(false)
    , mConnected(false)
    , mEnded(false)
    , mWorkerStopped(false)
    , mRecordingComplete(false)
    , mMaxFrame(0)
    , mCurrentFrame(0)
    , mCompletedFrameSerial(0)
    , mHasUnpublishedIndexChanges(false)
    , mCompletedSegmentEpoch(0)
    , mErrorCode(static_cast<uint32_t>(OvdInputErrorCode::None))
{
    mCaptureBudget.limits = limits;
}

CaptureCatalogSnapshot OvdLiveSource::catalogSnapshot() const
{
    std::lock_guard<std::mutex> guard(mTransportMutex);
    return mCaptureCatalog->snapshot();
}
CaptureMetadataSnapshot OvdLiveSource::metadataSnapshot() const
{
    std::lock_guard<std::mutex> guard(mTransportMutex);
    return mCaptureCatalog->metadataSnapshot();
}
CaptureBudgetSnapshot OvdLiveSource::captureBudget() const
{
    std::lock_guard<std::mutex> guard(mBudgetMutex);
    return mCaptureBudget;
}
CaptureCatalogPage OvdLiveSource::catalogPage(uint64_t generation, uint64_t offset, uint32_t limit,
                                            uint64_t parent, std::string_view prefix) const
{
    std::lock_guard<std::mutex> guard(mTransportMutex);
    return mCaptureCatalog->catalogPage(generation, offset, limit, parent, prefix);
}

OvdLiveSource::~OvdLiveSource()
{
    stop();
}

bool OvdLiveSource::start(uint16_t port, const char* recordPath)
try
{
    std::unique_ptr<OvdCaptureCatalog> retiredCatalog; // freed after the lifetime lock
    std::lock_guard<std::mutex> transportGuard(mTransportMutex);
    if (mWorker.joinable() || mReader || mStream)
        return false;
    auto freshCatalog = std::make_unique<OvdCaptureCatalog>();
    retiredCatalog = std::move(mCaptureCatalog);
    mCaptureCatalog = std::move(freshCatalog);
    {
        std::lock_guard<std::mutex> guard(mBudgetMutex);
        mCaptureBudget = CaptureBudgetSnapshot{};
        mCaptureBudget.limits = mLimits;
    }
    mRecording.reset(new(std::nothrow) OvdLiveRecording(mLimits.maxStreamBytes));
    if (!mRecording)
        return false;
    mRecordingComplete = false;
    mWorkerStopped.store(false);
    mReader = createOmniPvdReader();
    if (!mReader)
    {
        ovdLog("[OVD-live] createOmniPvdReader failed\n");
        return false;
    }

    // The viewer is the TCP server; the worker's openStream() accepts the
    // producer (the client).
    mStream = createCancelableOvdSocketReadStream(port);
    if (!mStream)
    {
        ovdLog("[OVD-live] createCancelableOvdSocketReadStream failed\n");
        destroyOmniPvdReader(*mReader);
        mReader = nullptr;
        return false;
    }

    mPort = port;
    mRecordPath.clear();
    if (recordPath && recordPath[0])
    {
        ExclusiveFileWriteStream* recordStream =
            new(std::nothrow) ExclusiveFileWriteStream(recordPath);
        if (!recordStream || !recordStream->openStream())
        {
            ovdLog("[OVD-live] recording target already exists or could not be created: %s\n",
                   recordPath);
            delete recordStream;
            destroyCancelableOvdSocketReadStream(*mStream);
            mStream = nullptr;
            destroyOmniPvdReader(*mReader);
            mReader = nullptr;
            return false;
        }
        mRecordStream = recordStream;
        mRecordPath = recordPath;
    }

    // Do NOT open the stream here: openStream() listens+accepts and blocks until a
    // producer connects, which would freeze the calling (UI) thread. The worker opens
    // it, sets up the recording tee, validates the handshake/version, then
    // ingests. The connection is therefore asynchronous (poll isConnected()/hasEnded()).
    mConnected.store(false);
    mEnded.store(false);
    mStop.store(false);
    mMaxFrame.store(0);
    mCurrentFrame.store(0);
    mCompletedFrameSerial.store(0);
    mHasUnpublishedIndexChanges.store(false);
    mSegmentEpochState.reset();
    mCompletedSegmentEpoch.store(0);
    mCompletedMetadataBoundarySerial.store(0);
    mErrorCode.store(static_cast<uint32_t>(OvdInputErrorCode::None));
    mWorker = std::thread(&OvdLiveSource::run, this);
    ovdLog("[OVD-live] listening on port %u (waiting for a producer to connect)\n", (unsigned)port);
    return true;
}
catch (const std::exception&)
{
    stop();
    return false;
}

void OvdLiveSource::run()
try
{
    const auto finish = [&]
    {
        // A validation failure must release the accepted socket immediately;
        // otherwise a producer can remain connected until the engine is later
        // replaced or destroyed on the UI thread.
        if (mStream)
            mStream->closeStream();
        mEnded.store(true);
    };

    // stop() may have been requested before the worker was scheduled; bail without
    // opening the listen socket so stop()/join() cannot hang on a blocking accept.
    if (mStop.load())
    {
        finish();
        return;
    }

    // Blocking: bind+listen on the port, accept the producer (the client), and consume
    // the live handshake. Returns false if listening failed (e.g. the port is in use) or
    // stop() closed the socket before a producer connected.
    if (!mStream->openStream())
    {
        ovdLog("[OVD-live] listen/accept on port %u failed or was cancelled\n", (unsigned)mPort);
        finish();
        return;
    }

    // The handshake was consumed by openStream(), so recording begins exactly
    // at the OVD version header. File output is an optional second destination.
    mTee = new(std::nothrow) TeeReadStream(mStream, *mRecording, mRecordStream);
    if (!mTee)
    {
        mErrorCode.store(static_cast<uint32_t>(OvdInputErrorCode::ReaderRejectedCommand));
        finish();
        return;
    }
    if (mRecordStream)
        ovdLog("[OVD-live] recording session to %s\n", mRecordPath.c_str());

    // Keep the tee below validation: any prefix bytes consumed while deciding
    // to reject a command are still present in the byte-exact recording.
    mValidator = new(std::nothrow) OvdValidatedReadStream(*mTee, mLimits,
        CatalogValidationMode::SeparateNonRetained, mCaptureCatalog.get());
    if (!mValidator)
    {
        mErrorCode.store(static_cast<uint32_t>(OvdInputErrorCode::ReaderRejectedCommand));
        finish();
        return;
    }
    mReader->setReadStream(*mValidator);

    OmniPvdVersionType major = 0, minor = 0, patch = 0;
    if (!mReader->startReading(major, minor, patch)
        || mRecording->errorCode() != OvdInputErrorCode::None)
    {
        ovdLog("[OVD-live] startReading failed (incompatible or truncated stream header)\n");
        const OvdInputErrorCode error = mRecording->errorCode() != OvdInputErrorCode::None
            ? mRecording->errorCode() : (mValidator->failed()
                ? mValidator->errorCode() : OvdInputErrorCode::ReaderRejectedCommand);
        mErrorCode.store(static_cast<uint32_t>(error));
        ovdLog("[OVD-live] rejected stream: %s\n", ovdInputErrorMessage(error));
        clearOvdParserMessages();
        finish(); // reader/tee/record are torn down in stop()
        return;
    }

    ovdLog("[OVD-live] producer connected on port %u (stream version %u.%u.%u)\n",
           (unsigned)mPort, (unsigned)major, (unsigned)minor, (unsigned)patch);
    mConnected.store(true);
    mRecordingComplete = true;
    bool topologyDirty = false;
    OvdResourceUsage segmentStart{};
    uint64_t budgetBoundary = 0, budgetGeneration = 0, catalogStart = 0;
    bool catalogBuilding = false;
    const auto difference = [](const OvdResourceUsage& now, const OvdResourceUsage& start) {
        return OvdResourceUsage{now.streamBytes - start.streamBytes, now.commands - start.commands,
            now.attributeMutations - start.attributeMutations, now.payloadBytes - start.payloadBytes,
            now.classes - start.classes, now.attributes - start.attributes, now.objectCreates - start.objectCreates,
            now.messages - start.messages, now.retainedDomBytes - start.retainedDomBytes};
    };

    for (;;)
    {
        if (mStop.load())
            break;

        // Fetch the next command UNLOCKED: this blocks on recv until bytes arrive (or
        // the peer disconnects). Holding the DOM mutex here would stall the UI thread.
        mRecordingComplete = false;
        const auto beforeCommand = mValidator->usage();
        OmniPvdCommand::Enum cmd = mReader->getNextCommand();
        if (mRecording->errorCode() != OvdInputErrorCode::None)
        {
            mErrorCode.store(static_cast<uint32_t>(mRecording->errorCode()));
            ovdLog("[OVD-live] capture storage failed: %s\n",
                   ovdInputErrorMessage(mRecording->errorCode()));
            clearOvdParserMessages();
            break;
        }
        if (cmd == OmniPvdCommand::eINVALID)
        {
            if (!mStop.load() && mValidator->failed())
            {
                mErrorCode.store(static_cast<uint32_t>(mValidator->errorCode()));
                ovdLog("[OVD-live] rejected stream: %s\n",
                       ovdInputErrorMessage(mValidator->errorCode()));
                clearOvdParserMessages();
            }
            else if (!mStop.load() && !mValidator->cleanEof())
            {
                mErrorCode.store(static_cast<uint32_t>(OvdInputErrorCode::ReaderRejectedCommand));
                ovdLog("[OVD-live] rejected stream: %s\n",
                       ovdInputErrorMessage(OvdInputErrorCode::ReaderRejectedCommand));
                clearOvdParserMessages();
            }
            else if (!mStop.load())
            {
                // A valid stream may end with schema or lifetime commands after
                // its final STOP_FRAME. The worker is now quiescent, so publish
                // those index changes as one final clean-EOF boundary.
                std::lock_guard<std::recursive_mutex> lk(*mMutex);
                if (mHasUnpublishedIndexChanges.load())
                {
                    if (topologyDirty && !validateOvdDomTopology(*mDom))
                    {
                        const OvdInputErrorCode error = OvdInputErrorCode::InvalidDomTopology;
                        mErrorCode.store(static_cast<uint32_t>(error));
                        ovdLog("[OVD-live] rejected stream: %s\n", ovdInputErrorMessage(error));
                        if (mMessages)
                            mMessages->clear();
                        clearOvdParserMessages();
                    }
                    else
                    {
                        topologyDirty = false;
                        mHasUnpublishedIndexChanges.store(false);
                        mCompletedFrameSerial.fetch_add(1);
                    }
                }
            }
            mRecordingComplete = mValidator->cleanEof() && !hasFailed();
            break;
        }

        if (!mValidator->validateCommand(cmd, *mReader))
        {
            mErrorCode.store(static_cast<uint32_t>(mValidator->errorCode()));
            ovdLog("[OVD-live] rejected stream: %s\n",
                   ovdInputErrorMessage(mValidator->errorCode()));
            clearOvdParserMessages();
            break;
        }
        const auto used = mValidator->usage();
        const auto catalog = mCaptureCatalog->snapshot();
        const auto metadata = mCaptureCatalog->metadataSnapshot();
        {
            std::lock_guard<std::mutex> guard(mBudgetMutex);
            mCaptureBudget.used = used;
            if (metadata.boundarySerial != budgetBoundary)
            {
                mCaptureBudget.previousSegment = difference(beforeCommand, segmentStart);
                segmentStart = beforeCommand;
                budgetBoundary = metadata.boundarySerial;
            }
            if (!catalog.ready && !catalogBuilding && mValidator->commandDestination() == OvdCommandDestination::Catalog)
            { catalogBuilding = true; catalogStart = beforeCommand.streamBytes; }
            if (catalog.generation != budgetGeneration)
            {
                mCaptureBudget.previousCatalogBytes = used.streamBytes -
                    (catalogBuilding ? catalogStart : beforeCommand.streamBytes);
                budgetGeneration = catalog.generation;
                catalogBuilding = false;
            }
        }
        if (mValidator->commandDestination() == OvdCommandDestination::Catalog)
        {
            mRecordingComplete = true;
            continue; // validated/stored, never retained or indexed by PvdDom
        }

        {
            // Apply the DOM mutation under the lock so a concurrent UI read never
            // observes a half-applied command.
            std::lock_guard<std::recursive_mutex> lk(*mMutex);
            if (mMessages)
                applyOvdDomCommandAndCollect(*mReader, *mDom, cmd, *mMessages);
            else
            {
                std::vector<OmniPvdMessage> discardedMessages;
                applyOvdDomCommandAndCollect(*mReader, *mDom, cmd, discardedMessages);
            }

            const bool objectLifetimeChanged = cmd == OmniPvdCommand::eCREATE_OBJECT
                || cmd == OmniPvdCommand::eDESTROY_OBJECT;
            const bool topologyMayHaveChanged = mValidator->mayMutateTopology();
            if (cmd == OmniPvdCommand::eREGISTER_CLASS
                || cmd == OmniPvdCommand::eREGISTER_ENUM
                || cmd == OmniPvdCommand::eREGISTER_ATTRIBUTE
                || cmd == OmniPvdCommand::eREGISTER_CLASS_ATTRIBUTE
                || cmd == OmniPvdCommand::eREGISTER_UNIQUE_LIST_ATTRIBUTE
                || objectLifetimeChanged
                || topologyMayHaveChanged)
            {
                mHasUnpublishedIndexChanges.store(true);
            }
            if (objectLifetimeChanged || topologyMayHaveChanged)
                topologyDirty = true;

            if (mDom->mOvdIntegVersionWasChecked && !mDom->mOvdIntegVersionPassed)
            {
                mErrorCode.store(static_cast<uint32_t>(OvdInputErrorCode::UnsupportedIntegrationVersion));
                ovdLog("[OVD-live] rejected stream: %s\n",
                       ovdInputErrorMessage(OvdInputErrorCode::UnsupportedIntegrationVersion));
                clearOvdParserMessages();
                break;
            }
            if (cmd == OmniPvdCommand::eREGISTER_CLASS)
                mSegmentEpochState.observeClassRegistration(mReader->getClassName());

            // Publish frame, schema generation, and serial under the DOM mutex.
            // LiveReadGuard therefore observes one completed boundary, never a
            // partially registered schema paired with an older STOP_FRAME.
            if (cmd == OmniPvdCommand::eSTOP_FRAME)
            {
                if (topologyDirty && !validateOvdDomTopology(*mDom))
                {
                    const OvdInputErrorCode error = OvdInputErrorCode::InvalidDomTopology;
                    mErrorCode.store(static_cast<uint32_t>(error));
                    ovdLog("[OVD-live] rejected stream: %s\n", ovdInputErrorMessage(error));
                    if (mMessages)
                        mMessages->clear();
                    clearOvdParserMessages();
                    break;
                }
                topologyDirty = false;
                // Clear the gate only after all index-affecting commands through
                // this boundary have been applied and topology has validated.
                // This store and the serial publication are both under mMutex.
                mHasUnpublishedIndexChanges.store(false);
                mMaxFrame.store(mDom->mMaxFrame);
                mCurrentFrame.store(mDom->mLatestStartedFrame);
                mCompletedSegmentEpoch.store(mSegmentEpochState.completeFrame(
                    mDom->mSawRecordingSegmentMetadata,
                    mDom->mCurrentRecordingSegmentId));
                mCompletedMetadataBoundarySerial.store(metadata.boundarySerial);
                mCompletedFrameSerial.fetch_add(1);
            }
        }
        mRecordingComplete = true;
    }
    // A manual stop can end on a complete command before the next EOF/frame
    // topology check. Validate that final boundary before allowing export.
    if (mRecordingComplete && topologyDirty)
    {
        std::lock_guard<std::recursive_mutex> lock(*mMutex);
        if (!validateOvdDomTopology(*mDom))
        {
            mRecordingComplete = false;
            mErrorCode.store(static_cast<uint32_t>(OvdInputErrorCode::InvalidDomTopology));
            clearOvdParserMessages();
        }
    }
    finish();
}
catch (const std::exception&)
{
    // Storage/reader allocations must fail the capture without terminating the
    // process or leaving a producer connected to an abandoned worker.
    mRecordingComplete = false;
    mErrorCode.store(static_cast<uint32_t>(OvdInputErrorCode::ReaderRejectedCommand));
    clearOvdParserMessages();
    if (mStream)
        mStream->closeStream();
    mEnded.store(true);
}

void OvdLiveSource::stop()
{
    mStop.store(true);

    // Closing the socket unblocks a worker parked either in openStream()'s blocking
    // accept (no producer connected yet) or in getNextCommand()'s blocking recv (live:
    // readBytes returns short -> getNextCommand returns eINVALID -> the loop exits), so
    // the join below cannot hang.
    {
        std::lock_guard<std::mutex> transportGuard(mTransportMutex);
        if (mStream) mStream->closeStream();
    }

    if (mWorker.joinable())
        mWorker.join();

    // The worker has stopped, so nothing else reads/writes the streams now.
    if (mReader)
    {
        destroyOmniPvdReader(*mReader);
        mReader = nullptr;
    }
    delete mValidator;
    mValidator = nullptr;
    delete mTee; // owns nothing; just the decorator
    mTee = nullptr;
    {
        std::lock_guard<std::mutex> transportGuard(mTransportMutex);
        if (mStream)
        {
            destroyCancelableOvdSocketReadStream(*mStream);
            mStream = nullptr;
        }
    }
    if (mRecordStream)
    {
        const bool flushed = mRecordStream->flush();
        const bool closed = mRecordStream->closeStream();
        if (!flushed || !closed)
            ovdLog("[OVD-live] file recording finalization failed; memory capture retained\n");
        delete mRecordStream;
        mRecordStream = nullptr;
        // mRecordPath is intentionally kept so the UI can still copy the file out.
    }
    mWorkerStopped.store(true);
}

uint16_t OvdLiveSource::controlCapabilities() const
{
    std::lock_guard<std::mutex> guard(mTransportMutex);
    return mStream && !mStop.load() && !hasEnded() ? mStream->controlCapabilities() : 0;
}

CaptureTransportSnapshot OvdLiveSource::controlTransportState() const
{
    std::lock_guard<std::mutex> guard(mTransportMutex);
    return mStream ? mStream->controlTransportState() : CaptureTransportSnapshot{};
}

void OvdLiveSource::queueCaptureRequest(const CaptureRequest& request)
{
    std::lock_guard<std::mutex> guard(mTransportMutex);
    if (!mStream || mStop.load() || hasEnded()) throw std::runtime_error("Live capture has ended");
    const auto budget = captureBudget();
    if (request.catalogOnly ? !canRefreshCapture(budget.used, budget.previousCatalogBytes, budget.limits) :
        !canApplyCapture(budget.used, budget.previousSegment, budget.limits))
        throw std::runtime_error("Recording budget is low; save and reconnect before changing selection");
    mStream->queueCaptureRequest(request);
}

void OvdLiveSource::queueAuthorizedCaptureRequest(const CaptureRequest& request, uint64_t generation)
{
    std::lock_guard<std::mutex> guard(mTransportMutex);
    if (!mStream || mStop.load() || hasEnded()) throw std::runtime_error("Live capture has ended");
    const auto capabilities = mStream->controlCapabilities();
    if (!request.catalogOnly && !request.all && !(capabilities & 8))
        throw std::runtime_error("Peer has no catalog capability; choose All permitted");
    auto frame = std::make_shared<const std::vector<uint8_t>>(encodeCaptureRequest(request, capabilities));
    const auto budget = captureBudget();
    if (request.catalogOnly ? !canRefreshCapture(budget.used, budget.previousCatalogBytes, budget.limits) :
        !canApplyCapture(budget.used, budget.previousSegment, budget.limits))
        throw std::runtime_error("Recording budget is low; save and reconnect before changing selection");
    const auto enqueue = [&] { mStream->queueCaptureFrame(request.requestId, std::move(frame)); };
    if (request.catalogOnly || request.all) enqueue();
    else mCaptureCatalog->withSelection(generation, request.scenes, request.objects, enqueue);
}

void OvdLiveSource::flushRecording()
{
    if (mRecordStream)
        mRecordStream->flush();
}

uint64_t OvdLiveSource::recordingSize() const
{
    return mRecording ? mRecording->size() : 0;
}

bool OvdLiveSource::writeRecording(OmniPvdWriteStream& destination) const
{
    return mWorkerStopped.load() && mRecordingComplete && !hasFailed()
        && mRecording && mRecording->writeTo(destination);
}

} // namespace pvdapp

#endif // PVD_OVD_SUPPORT
