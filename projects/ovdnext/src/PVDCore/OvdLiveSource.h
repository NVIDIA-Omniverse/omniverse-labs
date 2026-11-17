/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PVD_OVD_LIVE_SOURCE_H
#define PVD_OVD_LIVE_SOURCE_H

#ifdef PVD_OVD_SUPPORT

#include "OvdInputValidation.h"
#include "OvdCaptureControl.h"
#include "OvdCaptureCatalog.h"
#include "PvdDom.h"
#include "PvdDomParser.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>

#include <string>
#include <unordered_set>
#include <vector>

class OmniPvdReader;
class OmniPvdReadStream;
class OmniPvdSocketReadStream;
class OmniPvdWriteStream;
class OvdValidatedReadStream;

namespace pvdapp
{

class OvdLiveRecording;
class ControlSocketReadStream;

// Tracks live ingest segment identity. Explicit recording metadata wins when
// available; repeated class registration supplies the legacy-DOM fallback.
// The completed value advances only at completeFrame(), so next-segment schema
// commands cannot be paired with an earlier STOP_FRAME.
class OvdLiveSegmentEpochState
{
public:
    void reset();
    void observeClassRegistration(const char* className);
    uint64_t completeFrame(bool hasRecordingSegmentMetadata = false,
                           uint64_t recordingSegmentId = 0);
    uint64_t completedEpoch() const { return mCompletedEpoch; }

private:
    std::unordered_set<std::string> mSeenClassNames;
    uint64_t mAvailableEpoch = 0;
    uint64_t mCompletedEpoch = 0;
};

struct OvdLivePollDecision
{
    bool invalidateFrameCache;
    bool followTargetChanged;
};

// A segment boundary is a visual change even when both producers use the
// same raw frame number. Keep this decision independently testable from Qt UI.
OvdLivePollDecision decideOvdLivePollUpdate(uint64_t previousEpoch,
                                            uint64_t previousFrame,
                                            uint64_t currentEpoch,
                                            uint64_t currentFrame);

// ---------------------------------------------------------------------------
// OvdLiveSource
//
// Background TCP listener that tails a live OmniPVD command stream from a
// running PhysX producer and grows a shared OmniPvdDOMState in place. The DOM
// and the recursive mutex that guards it are owned elsewhere (the OvdDomAdapter)
// and borrowed here: every command this worker applies to the DOM is applied
// under that mutex, while the (blocking) socket read that fetches the next
// command runs UNLOCKED so the UI thread never stalls waiting on the network.
//
// The UI observes progress by polling maxFrame()/isConnected()/hasEnded()
// (mirrors PVD3's PvdListener + QTimer poll model), then re-indexes the adapter
// and refreshes under the same mutex.
// ---------------------------------------------------------------------------
class OvdLiveSource
{
public:
    // dom + mutex are borrowed (owned by the OvdDomAdapter); they must outlive
    // this source, which the engine guarantees by stopping the worker before the
    // adapter (and thus the DOM) is destroyed.
    OvdLiveSource(OmniPvdDOMState* dom, std::recursive_mutex* mutex,
                  const OvdResourceLimits& limits = OvdResourceLimits::defaults(),
                  std::vector<OmniPvdMessage>* messages = nullptr);
    ~OvdLiveSource();

    OvdLiveSource(const OvdLiveSource&) = delete;
    OvdLiveSource& operator=(const OvdLiveSource&) = delete;

    // Listen on `port` (TCP server; the producer is the client that connects to us)
    // and start the background ingest thread. Returns true once the reader/stream are
    // created and the worker is spawned; accepting the producer + the OmniPVD handshake
    // happen ON the worker (openStream blocks on accept), so the connection is
    // asynchronous and observed via isConnected()/hasEnded(). Returns false if the
    // reader/socket cannot be created or the recording target cannot be exclusively
    // created. If recordPath is non-null/non-empty, the
    // raw post-handshake byte stream is also teed to that file. Every source retains
    // the consumed bytes in memory regardless of recordPath. The path must not exist: it is atomically created
    // and held open before this method succeeds, and an existing/unwritable target fails.
    bool start(uint16_t port, const char* recordPath = nullptr);

    // Stop the worker, close the socket and tear down the reader/stream (and
    // flush+close the recording). Idempotent; also called from the destructor.
    void stop();

    uint16_t controlCapabilities() const;
    CaptureTransportSnapshot controlTransportState() const;
    void queueCaptureRequest(const CaptureRequest& request);
    void queueAuthorizedCaptureRequest(const CaptureRequest& request, uint64_t generation);
    CaptureCatalogSnapshot catalogSnapshot() const;
    CaptureMetadataSnapshot metadataSnapshot() const;
    CaptureBudgetSnapshot captureBudget() const;
    CaptureCatalogPage catalogPage(uint64_t generation, uint64_t offset = 0, uint32_t limit = 1000,
                                   uint64_t parent = 0, std::string_view prefix = {}) const;

    // Path of the .ovd the live stream is being teed to ("" if not recording).
    // Remains valid after stop() so the UI can copy it out on demand.
    const std::string& recordedFilePath() const { return mRecordPath; }
    // Flush buffered recorded bytes to disk so a copy taken now is up to date.
    void flushRecording();

    // Retained raw OVD bytes, safe to poll while the worker is ingesting.
    uint64_t recordingSize() const;
    // Export a validated capture after stop() has joined the worker. The caller
    // owns opening/flushing/closing destination. Returns false before stop(),
    // for incomplete/rejected captures, or if any destination write is short.
    bool writeRecording(OmniPvdWriteStream& destination) const;

    // Highest frame id whose data is fully ingested (0 before the first frame
    // completes). Lock-free; polled by the UI to know when to re-index/refresh.
    uint64_t maxFrame()    const { return mMaxFrame.load(); }
    // Frame id of the newest fully-ingested frame (the current live position). A
    // new recording segment restarts frame numbering, so unlike maxFrame() this
    // drops back down at a re-attach; the UI follows it so the playhead snaps to
    // the new segment instead of parking on the older, higher-numbered segment.
    uint64_t currentFrame() const { return mCurrentFrame.load(); }
    // Monotonic publication serial; advances for every completed frame and
    // once at clean EOF if index changes followed the final frame.
    uint64_t completedFrameSerial() const { return mCompletedFrameSerial.load(); }
    // True while the shared DOM contains schema, object-lifetime, or reference
    // changes newer than the latest validated STOP_FRAME or clean-EOF boundary.
    // The adapter must not rebuild its append-only index while this is true.
    bool hasUnpublishedIndexChanges() const { return mHasUnpublishedIndexChanges.load(); }
    // Exact recording segment (or legacy schema generation) captured at the
    // same STOP_FRAME boundary as the frame values and serial.
    uint64_t completedSegmentEpoch() const { return mCompletedSegmentEpoch.load(); }
    uint64_t completedMetadataBoundarySerial() const { return mCompletedMetadataBoundarySerial.load(); }
    // True once a producer has connected and the stream version header was read.
    bool     isConnected() const { return mConnected.load(); }
    // True once the producer disconnected / the stream ended.
    bool     hasEnded()    const { return mEnded.load(); }
    bool     hasFailed()   const { return errorCode() != OvdInputErrorCode::None; }
    OvdInputErrorCode errorCode() const
    {
        return static_cast<OvdInputErrorCode>(mErrorCode.load());
    }

private:
    void run(); // worker-thread body

    OmniPvdDOMState*       mDom;    // borrowed (owned by OvdDomAdapter)
    std::recursive_mutex*  mMutex;  // borrowed; guards DOM writes vs UI reads
    std::vector<OmniPvdMessage>* mMessages; // borrowed; guarded by mMutex

    OmniPvdReader*           mReader;
    ControlSocketReadStream* mStream;
    mutable std::mutex      mTransportMutex; // pointer lifetime only; never held while joining
    OvdValidatedReadStream*   mValidator;
    OvdResourceLimits         mLimits;
    std::unique_ptr<OvdCaptureCatalog> mCaptureCatalog;
    mutable std::mutex mBudgetMutex;
    CaptureBudgetSnapshot mCaptureBudget;

    // Memory bytes outlive stop(), and are released with this source. The tee
    // always captures to memory and can additionally write an optional file.
    std::unique_ptr<OvdLiveRecording> mRecording;
    OmniPvdReadStream*       mTee;          // tee wrapper handed to the reader (owns nothing)
    OmniPvdWriteStream*      mRecordStream; // exclusively-created raw .ovd sink
    std::string              mRecordPath;   // destination temp path ("" if not recording)
    uint16_t                 mPort;         // TCP port the viewer listens on

    std::thread           mWorker;
    std::atomic<bool>     mStop;
    std::atomic<bool>     mConnected;
    std::atomic<bool>     mEnded;
    std::atomic<bool>     mWorkerStopped;
    bool                  mRecordingComplete; // worker-owned until stop() joins
    std::atomic<uint64_t> mMaxFrame;
    std::atomic<uint64_t> mCurrentFrame; // newest fully-ingested frame id (resets at a re-attach)
    std::atomic<uint64_t> mCompletedFrameSerial;
    std::atomic<bool>     mHasUnpublishedIndexChanges;
    OvdLiveSegmentEpochState mSegmentEpochState;
    std::atomic<uint64_t> mCompletedSegmentEpoch;
    std::atomic<uint64_t> mCompletedMetadataBoundarySerial{0};
    std::atomic<uint32_t> mErrorCode;
};

} // namespace pvdapp

#endif // PVD_OVD_SUPPORT
#endif // PVD_OVD_LIVE_SOURCE_H
