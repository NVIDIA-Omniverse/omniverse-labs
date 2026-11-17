/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "LoadControl.h"
#include "OvdEvidenceMetadata.h"

#include "PVDClient.h"
#include "PvdCommStreamMetaData.h"
#include "PvdRandomAccessIOStream.h"
#include "PxPvdObjectModel.h"
#include "PxPvdObjectModelMetaData.h"
#include "pvdfileloader.h"

#ifdef PVD_OVD_SUPPORT
#include "OvdInputValidation.h"
#include "OvdCaptureCatalog.h"
#include "OvdCaptureControl.h"
#include "PvdDomParser.h"   // OmniPvdMessage
#endif

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

class OmniPvdWriteStream;

namespace pvdapp
{

class OvdDomAdapter;
class OvdLiveSource;

// Publishes a newly discovered schema generation only when OvdLiveSource has
// published another fully ingested frame. This prevents schema commands that
// precede STOP_FRAME from consuming the epoch early.
class PvdEngine
{
  private:
	static const PxU32 STREAM_MAGIC_WORD = 837150850;
	static const PxU32 FILE_MAGIC_WORD = 907025328;

  public:
	PvdEngine();
	~PvdEngine();

	PxU32 getFrameCount(PxU64 sceneId);
	PxU32 getCurrentFrame(PxU64 sceneId);
	void setCurrentFrame(PxU32 frame, PxU64 sceneId);
	// OVD only: highest raw frame stamp seen by the writer. Returns 0 for
	// non-OVD captures (PXD2 has no equivalent — the count via getFrameCount
	// is what callers use there).
	PxU64 getMaxFrame() const;

	PxU64 getEventCount();
	PxI64 getCurrentEvent();

	PxI64 getInnerFrameTime();
	PxI64 getTotalFrameTime();

	PvdObjectModelEventStream* getEventStream();
	// What event does the next frame correspond to
	PxI64 findNextFrameEvent(PxI64 inStartEvent, PxU64 sceneId);
	// What event does the previous frame correspond to
	PxI64 findPreviousFrameEvent(PxI64 inStartEvent, PxU64 sceneId);

	void getTimestampNanosecondConversion(PxU64& numerator, PxU64& denominator);

	void parseProfileEvents(PxI64 startEvent, PxI64 endEvent, int profileZone, physx::profile::PxProfileBulkEventHandler& handler);

	void parseMemoryEvents(PxI64 startEvent, PxI64 endEvent, physx::profile::PxProfileBulkMemoryEventHandler& handler);
	PxU32 getNbMemoryEventEntries();
	// Recorded legacy stream identity; -1 when no memory buffer was recorded.
	PxI32 getMemoryEventBufferInstance() const;
	PxU32 getMemoryEventEntries(PvdMemoryEventEntry* buffer, PxU32 bufSize, PxU32 idx);
	PxU32 getNbMemoryTotals();
	PxU32 getMemotyEventTotals(PxI32* buffer, PxU32 bufSize);

	PxU32 getNbProfileZones();
	PxU32 getProfileZones(PvdProfileZone* buffer, PxU32 bufSize, PxU32 idx = 0);

	PxU32 getNbClasses() const;
	PxU32 getClasses(ClassDescription* outClasses, PxU32 requestCount, PxU32 startIndex = 0);

	PxU32 getNbProperties(const PxI32 classId) const;
	PxU32 getProperties(const PxI32 classId, PropertyDescription* outBuffer, PxU32 bufCount, PxU32 startIdx = 0) const;

	PxU32 getNbInstances() const;
	PxU32 getInstances(InstanceDescription* outBuffer, PxU32 count, PxU32 startIndex = 0) const;

	PxU32 getNbScenes() const;
	PxU32 getScenes(SectionListInfo* outBuffer, PxU32 count, PxU32 startIndex = 0) const;

	PxU32 getPropertyByteSize(void* instPtr, const PxU32 propId);
	bool hasPropertySample(void* instPtr, int32_t propId);
    OvdEvidenceDomain ovdEvidenceDomain() const;
    OvdEvidenceIdentity ovdEvidenceIdentity(int32_t instanceId) const;
    OvdEvidenceSampleStamp ovdEvidenceSampleStamp(void* instance,int32_t property,uint64_t frame) const;

	PxU64 getStreamId(PxI32 instance);

	InstanceDescription createInstance();

	PxU32 getNbArrayElements(void* instPtr, const char* propName);
	PxU32 getNbArrayElements(void* instPtr, int propId);

	template <typename TDataType>
	PxU32 getPropertyValue(void* instPtr, const char* propName, TDataType* outBuffer, PxU32 outNumBufferItems)
	{
		ClassDescription cls = getClassOf(instPtr);
		Option<PropertyDescription> descOpt = findPropertyFlex(cls.mClassId, propName);
		if(!descOpt.hasValue())
		{
			return 0;
		}
		const PropertyDescription& prop(descOpt);
		PxU32 desired = outNumBufferItems * sizeof(TDataType);
		return getPropertyValue(instPtr, prop.mPropertyId, reinterpret_cast<uint8_t*>(outBuffer), desired) / sizeof(TDataType);
	}

	template <typename TDataType>
	Option<TDataType> getPropertyValue(void* instPtr, const char* propName)
	{
#ifdef PVD_OVD_SUPPORT
		if (m_isOvd && m_ovdAdapter)
		{
			ClassDescription cls = getClassOf(instPtr);
			Option<PropertyDescription> descOpt = findPropertyFlex(cls.mClassId, propName);
			if (!descOpt.hasValue()) return Option<TDataType>();
			TDataType val;
			PxU32 read = getPropertyValue(instPtr, descOpt.getValue().mPropertyId, &val, sizeof(TDataType));
			if (read >= sizeof(TDataType)) return Option<TDataType>(val);
			return Option<TDataType>();
		}
#endif
		return m_reader->getPropertyValue<TDataType>(instPtr, propName);
	}
	PxU32 getPropertyValue(void* instPtr, PxI32 propId, void* outData, PxU32 outDataLen);
	DataRef<uint8_t> getRawPropertyValue(void* instPtr, int32_t propId);

	// "Freshest known value" — latest sample with mTimeStamp <= frame, no
	// keyframe-range check. Use for normal per-frame property reads (poses,
	// names, flags …) where the writer's last value persists until overwritten.
	DataRef<uint8_t> getRawPropertyValueAtFrame(void* instPtr, int32_t propId, uint64_t frame);

	// "Was data recorded at this frame?" — sample data iff frame == mTimeStamp
	// (pvddom doesn't dedup, mTimeStamp == mEndTimeStamp). Use ONLY for
	// properties where stale-after-writer-stops would be wrong (collision
	// arrays, etc.).
	DataRef<uint8_t> getRawPropertyValueRecordedAtFrame(void* instPtr, int32_t propId, uint64_t frame);

	DataRef<uint8_t> getRawPropertyValue(void* instPtr, const char* propName)
	{
		const ClassDescription cls(getClassOf(instPtr));
		const Option<PropertyDescription> descOpt(findPropertyFlex(cls.mClassId, propName));
		if(!descOpt.hasValue())
		{
			return DataRef<uint8_t>();
		}
		return getRawPropertyValue(instPtr, descOpt->mPropertyId);
	}

	template <typename TDataType>
	DataRef<TDataType> getTypedRawPropertyValue(void* instPtr, int32_t propId)
	{
		DataRef<uint8_t> propVal = getRawPropertyValue(instPtr, propId);
		return DataRef<TDataType>(reinterpret_cast<const TDataType*>(propVal.begin()), propVal.size() / sizeof(TDataType));
	}

	template <typename TDataType>
	DataRef<TDataType> getTypedRawPropertyValue(void* instPtr, const char* propName)
	{
		DataRef<uint8_t> propVal = getRawPropertyValue(instPtr, propName);
		return DataRef<TDataType>(reinterpret_cast<const TDataType*>(propVal.begin()), propVal.size() / sizeof(TDataType));
	}

	DataRef<NamedValue> getNamedPropertyValues(PxI32 propId) const;

	void save(PvdRandomAccessIOStream* fileStream);
	bool canExportRepX();
	void exportRepX(physx::pvdsdk::PxAllocatorCallback& alloc, PvdRandomAccessIOStream* fileStream, const std::vector<uint64_t>* objs = 0);

	// load/crop filesystem paths are UTF-8 on every platform.
#ifdef PVD_OVD_SUPPORT
	static PvdEngine* load(PxAllocatorCallback& alloc, const char* tempDir, const char* filename,
	                       OvdInputErrorCode* ovdErrorCode = nullptr, LoadControl* control = nullptr);
#else
	static PvdEngine* load(PxAllocatorCallback& alloc, const char* tempDir, const char* filename, LoadControl* control = nullptr);
#endif
	static PvdEngine* create(PvdClient* client);

	// Live OVD: listen on `port` (the viewer is the TCP server) for a running PhysX
	// producer to connect (the producer is the client), and tail its OmniPVD command
	// stream into a growing DOM, served through the same OVD adapter path as a loaded
	// .ovd file. Returns nullptr if the listener/reader could not be created or
	// the requested recording target could not be exclusively created;
	// accepting the producer is asynchronous (the producer may connect later). Sibling
	// of load(). If recordPath is non-null, it must not exist: the target is atomically
	// created before this call succeeds and the stream is teed byte-for-byte to it.
	// Raw OVD bytes are always retained in memory until the engine is destroyed.
	static PvdEngine* connectLive(PxAllocatorCallback& alloc, uint16_t port,
	                              const char* recordPath = nullptr);
	PvdClient* crop(PxI64 startFrame, PxI64 endFrame, const char* tempDir);
	// Inclusive source-frame range. Output frame zero is the starting snapshot.
	PvdClient* crop(PxU64 sceneId, PxI64 startFrame, PxI64 endFrame, const char* tempDir);

	bool isTopLevel(PxI32 instanceId);
	// 1-based synthetic id of the PxScene that owns instanceId (via the OVD
	// DOM tree), or -1 if none / not an OVD capture. See OvdDomAdapter.
	PxI32 getOwningSceneId(PxI32 instanceId);
	// Instance id of the PxPhysics that owns this instance (nearest PxPhysics
	// ancestor in the DOM tree), or -1 if none / not an OVD capture.
	PxI32 getOwningPhysicsId(PxI32 instanceId);
	// True for OVD extension objects processed as PxJoint descendants.
	bool isOvdSceneJoint(PxI32 instanceId);
	ClassDescription getClassOf(void* instPtr);
	Option<ClassDescription> getClass(PxI32 classId);
	Option<PropertyDescription> findProperty(PxI32 classId, const char* propName);
	// Flexible lookup: tries propName as-is, then camelCase variant (for OVD compat)
	Option<PropertyDescription> findPropertyFlex(PxI32 classId, const char* propName);

	Option<Section> getFrameInfo(PxU32 frame, PxU64 sceneId);

	const char* handleToString(const uint32_t handle) const;

	// File format detection
	bool isOvdFile() const { return m_isOvd; }

	// Live OVD connection state (all false / 0 for files and PXD2 streams).
	bool   isLive() const;          // true while a live source is attached
	bool   isLiveConnected() const; // socket connected + version header read
	bool   isLiveEnded() const;     // producer disconnected / stream ended
#ifdef PVD_OVD_SUPPORT
	OvdInputErrorCode getLiveErrorCode() const; // stable validation reason, or None
	CaptureCatalogSnapshot captureCatalogSnapshot() const;
	CaptureBudgetSnapshot captureBudget() const;
	CaptureMetadataSnapshot captureMetadata() const;
	CaptureTransportSnapshot captureTransport() const;
	uint16_t captureCapabilities() const;
	void queueCaptureRequest(const CaptureRequest& request, uint64_t generation);
	CaptureCatalogPage captureCatalogPage(uint64_t generation, uint64_t offset = 0, uint32_t limit = 1000,
	                                     uint64_t parent = 0, std::string_view prefix = {}) const;
	uint64_t getLiveMetadataBoundarySerial() const;
	uint64_t liveIndexRebuildCount() const { return mLiveIndexRebuildCount; } // UI-thread diagnostic
#endif
	PxU64  getLiveMaxFrame() const; // highest fully-ingested frame so far (monotonic timeline range)
	PxU64  getLiveCurrentFrame() const; // newest fully-ingested frame (drops back at a re-attach)
	PxU64  getLiveCompletedFrameSerial() const; // changes at STOP_FRAME and final clean-EOF publication boundaries
	// True while live DOM changes that affect the adapter index have not reached
	// a validated STOP_FRAME or clean-EOF publication boundary.
	bool   hasLiveUnpublishedIndexChanges() const;
	PxU64  getOvdSegmentEpoch() const; // exact recording segment, or legacy schema generation
	PxU64  getOvdRecordingSegment(PxI32 instId) const; // instance's segment, not latest ingestion

	// Re-scan the growing live DOM and re-index the adapter (no-op for files).
	// Takes the DOM lock internally; call when getLiveMaxFrame() has advanced.
	void   refreshLive();

	// Stop the live ingest worker (freeze the capture) without tearing down the
	// DOM/adapter, so the buffered frames stay scrubbable. No-op for files.
	void   stopLive();

	// Path of the .ovd the live session is being recorded to ("" if none / not
	// live). Valid until the engine is destroyed, including after stopLive().
	std::string getLiveRecordingPath() const;
	// Flush the live recording so a copy taken now reflects all received frames.
	void   flushLiveRecording();

	// Raw captured byte count, safe to poll during live ingestion; zero for files.
	uint64_t getLiveRecordingSize() const;
	// Requires stopLive() first, including after isLiveEnded() becomes true.
	// Writes to an already-open destination; the caller flushes/closes it.
	// Returns false for rejected/incomplete captures or a short destination write.
	bool writeLiveRecording(OmniPvdWriteStream& destination) const;

	// RAII serialization of UI/renderer reads of a live OVD DOM against the
	// ingest worker. Locks the adapter's recursive mutex for a live connection;
	// a no-op for file-loaded captures and PXD2 streams. Hold one across a whole
	// render/refresh so the frame observed stays internally consistent.
	class LiveReadGuard
	{
	  public:
		explicit LiveReadGuard(PvdEngine& engine);
		~LiveReadGuard();
		LiveReadGuard(const LiveReadGuard&) = delete;
		LiveReadGuard& operator=(const LiveReadGuard&) = delete;
	  private:
		void* m_mtx; // std::recursive_mutex* (opaque to avoid leaking <mutex> here)
	};

	// OVD-only: resolve an external object handle to a 1-based instance id.
	// A published referrer mInstPtr makes pointer reuse segment-consistent.
	// Returns -1 for non-OVD files or unknown handles.
	PxI32 resolveInstanceId(PxU64 externalHandle, void* referrerInstPtr = nullptr) const;

#ifdef PVD_OVD_SUPPORT
	// OVD diagnostic messages (eRECORD_MESSAGE stream). Returns an empty
	// vector for PXD2 files.
	const std::vector<OmniPvdMessage>& getMessages() const;

	// OVD-native datatype label for a property (e.g. "eFLOAT32x3",
	// "eFLAGS_WORD (PxShapeFlag)"). Returns empty string for PXD2 files or
	// unknown propIds. Used by InstanceView's optional Type column to surface
	// OVD's native type info instead of the PhysX-shaped translation.
	std::string getOvdNativeTypeLabel(PxI32 propId) const;
	// Declared scalar fields: 0 variable/list, 1 scalar; -1 unknown/legacy.
	int64_t getOvdPropertyFieldCount(PxI32 propId) const;
#endif

  private:
	PxAllocatorCallback* mAllocator = nullptr;
	PvdClient* m_client = nullptr;
	PvdObjectModelMetaData* m_meta = nullptr;
	PvdObjectModelReader* m_reader = nullptr;
	PvdCommStreamMetaData* m_streamMeta = nullptr;

	PvdFileStreamLoader* m_loader = nullptr;
	bool m_isOvd = false;

#ifdef PVD_OVD_SUPPORT
	std::unique_ptr<OvdDomAdapter> m_ovdAdapter;
	// Declared AFTER m_ovdAdapter so it is destroyed FIRST: the ingest worker
	// (which writes into the adapter's DOM under the adapter's mutex) must be
	// stopped and joined before the adapter and its DOM are torn down.
	std::unique_ptr<OvdLiveSource>  m_live;
	PxU64 m_lastRefreshedCompletedFrameSerial = 0;
	uint64_t mLiveIndexRebuildCount = 0;
	friend class OvdDomAdapter;
#endif
};
} // namespace pvdapp
