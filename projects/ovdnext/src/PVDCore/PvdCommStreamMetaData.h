/*
 * SPDX-FileCopyrightText: Copyright (c) 2008-2014 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Copyright (c) 2008-2014 NVIDIA Corporation. All rights reserved.
#ifndef PVD_COMM_STREAM_DATA_H
#define PVD_COMM_STREAM_DATA_H

#include "PvdObjectModelEventStream.h"
#include "PvdSimpleFoundation.h"
#include "PxPvdObjectModel.h"
#include "PxPvdObjectModelBaseTypes.h"
#include "foundation/PxFlags.h"
#include "foundation/PxVec3.h"

namespace physx
{
namespace pvdsdk
{
struct PvdErrorMessage
{
	uint32_t mCode;
	uint32_t mMessage; // string handle
	uint32_t mFile;    // string handle
	uint32_t mLine;
	uint32_t mIndex;
	PvdErrorMessage(uint32_t code, uint32_t message, uint32_t file, uint32_t line, uint32_t index) : mCode(code), mMessage(message), mFile(file), mLine(line), mIndex(index) {}
	PvdErrorMessage() {}
};
} // namespace pvdsdk
} // namespace physx

namespace pvdapp
{

using namespace physx::pvdsdk;

struct CameraProps
{
	PxVec3 mPosition;
	PxVec3 mUp;
	PxVec3 mTarget;
	CameraProps(const PxVec3& pos, const PxVec3& up, const PxVec3& target) : mPosition(pos), mUp(up), mTarget(target) {}
	CameraProps() {}
};

struct SectionListInfo
{
	PxU64 mStreamId;
	PxU32 mName; // string handle
	SectionListInfo(PxU64 sid, PxU32 name) : mStreamId(sid), mName(name) {}
	SectionListInfo() : mStreamId(0), mName(0) {}
};

struct SectionBracket
{
	PxU64 mTimestamp;
	PxU64 mEventIndex;

	SectionBracket(PxU64 ts, PxU64 evIdx) : mTimestamp(ts), mEventIndex(evIdx) {}
	SectionBracket() : mTimestamp((PxU64)-1), mEventIndex((PxU64)-1) {}
	SectionBracket(const SectionBracket& other) : mTimestamp(other.mTimestamp), mEventIndex(other.mEventIndex) {}
	SectionBracket& operator=(const SectionBracket& other)
	{
		mTimestamp = other.mTimestamp;
		mEventIndex = other.mEventIndex;
		return *this;
	}
	bool isEventValid() const { return mEventIndex != (PxU64)-1; }
	bool isTimestampValid() const { return mTimestamp != (PxU64)-1; }
	// checked access to timestamp and event index
	PxU64 getTimestamp() const
	{
		PX_ASSERT(isTimestampValid());
		return mTimestamp;
	}
	PxU64 getEventIndex() const
	{
		PX_ASSERT(isEventValid());
		return mEventIndex;
	}
};

struct Section
{
	SectionBracket mBegin;
	SectionBracket mEnd;
	Section(const SectionBracket& beg) : mBegin(beg) {}
	Section() {}
};

struct SectionEventData
{
	PxU64 mStreamId;
	PxU64 mTimestamp;
	PxU32 mName;
	bool mIsBegin;
	SectionEventData(PxU64 sid, PxU32 nm, PxU64 ts, bool isBegin) : mStreamId(sid), mTimestamp(ts), mName(nm), mIsBegin(isBegin) {}
};

struct PvdProfileZoneEvent
{
	physx::pvdsdk::String mName;
	PxU16 mEventId;
	bool mIsCompiledTimeEnabled;
	PvdProfileZoneEvent(physx::pvdsdk::String nm, PxU16 eid, bool cte) : mName(nm), mEventId(eid), mIsCompiledTimeEnabled(cte) {}
	PvdProfileZoneEvent() : mName("") {}
};

struct PvdProfileZone
{
	PxU64 mStreamId;
	physx::pvdsdk::String mName;
	PxI32 mInstanceId;
	physx::pvdsdk::DataRef<PvdProfileZoneEvent> mEvents;

	PvdProfileZone(PxU64 sid, physx::pvdsdk::String nm, PxI32 iid, physx::pvdsdk::DataRef<PvdProfileZoneEvent> evts) : mStreamId(sid), mName(nm), mInstanceId(iid), mEvents(evts) {}
	PvdProfileZone() : mName("") {}
};

struct PvdMemoryEventEntry
{
	physx::pvdsdk::String mType;
	physx::pvdsdk::String mFile;
	PxU32 mLine;
	PxU32 mTotalsArrayIndex;

	PvdMemoryEventEntry(physx::pvdsdk::String type, physx::pvdsdk::String file, PxU32 line, PxU32 totalsIdx) : mType(type), mFile(file), mLine(line), mTotalsArrayIndex(totalsIdx) {}
	PvdMemoryEventEntry() : mType(""), mFile(""), mLine(0), mTotalsArrayIndex(0) {}
};

class PvdCommStreamMetaDataMutator;

class PvdCommStreamMetaData
{
  protected:
	virtual ~PvdCommStreamMetaData() {}

  public:
	virtual void addRef() = 0;
	virtual void release() = 0;

	virtual void getTimestampNanosecondConversion(PxU64& numerator, PxU64& denominator) = 0;
	virtual PxU32 getNbInstances() = 0;
	virtual PxU32 getInstances(PxI32* instanceBuf, PxU32 bufSize, PxU32 idx = 0) = 0;
	virtual PxU64 getStreamId(PxI32 instance) = 0;

	// Only works when the item with the streamid is live
	virtual PxI32 getInstanceId(PxU64 streamId) = 0;
	// Returns the first instance that matches this id, works whether the stream
	// id is live or not.
	virtual PxI32 reverseMap(PxU64 streamId) = 0;
	virtual bool isPickable(PxI32 instance) = 0;
	virtual physx::pvdsdk::Option<physx::pvdsdk::PvdColor> getColor(PxI32 instance) = 0;
	virtual bool isTopLevel(PxI32 instance) = 0;

	virtual PxU32 getNbCameras() = 0;
	virtual PxU32 getCameras(physx::pvdsdk::StringHandle* buffer, PxU32 bufSize, PxU32 idx = 0) = 0;
	virtual physx::pvdsdk::Option<CameraProps> getCameraData(physx::pvdsdk::StringHandle name) = 0;

	virtual PxU32 getNbSectionLists() = 0;
	virtual PxU32 getSectionLists(SectionListInfo* sectionBuffer, PxU32 bufSize, PxU32 idx = 0) = 0;

	virtual PxU32 getNbSections(const SectionListInfo& list) = 0;
	virtual PxU32 getSections(const SectionListInfo& list, Section* buffer, PxU32 bufSize, PxU32 idx = 0) = 0;

	virtual Option<SectionListInfo> getDefaultFrameMarker() = 0;
	virtual Option<SectionListInfo> getSectionListInfoForScene(const PxU64 sceneId) = 0;

	// Section indexes are 1 based, not 0 based!!!!
	// So list.sections[sectionCount] is a valid operation!!!
	virtual Option<Section> getSectionInfo(const SectionListInfo& list, PxU32 currentSectionIndex) = 0;
	// Section indexes are 1 based, but evId 0 *always* corresponds to idx 0.
	// Event zero corresponds to a 'no index' return.
	// Event EventCount corresponds to SectionCount return.
	virtual PxU32 eventToSectionIndex(const SectionListInfo& list, PxU64 evId) = 0;
	// Event 0 means none.
	// Events past lastSection.mEnd.mEventId also return none.
	Option<Section> getSectionInfoFromEvent(const SectionListInfo& list, PxU64 evId);
	PxU64 getInnerSectionTime(const SectionListInfo& list, PxU64 inStartEvent);
	PxU64 getTotalSectionTime(const SectionListInfo& list, PxU64 inStartEvent);
	PxU64 getSectionEndTimestamp(const SectionListInfo& list, PxU64 inStartEvent);
	// What event does the next frame correspond to.
	// If we are at the end, then we use PX_MAX_U64
	PxU64 findNextSectionEvent(const SectionListInfo& list, PxU64 inStartEvent);
	// What event does the previous frame correspond to
	PxU64 findPreviousSectionEvent(const SectionListInfo& list, PxU64 inStartEvent);
	// Get the total timespan of the DB capture.
	PxU64 getTotalSectionTimespan(const SectionListInfo& list);
	physx::pvdsdk::Option<Section> getFrameInfo(const PxU32 frameIndex, const PxU64 sceneId);
	PxU32 eventToFrame(PxU64 eventIdx, PxU64 sceneId);
	PxU64 frameToEvent(PxU32 frameIndex, PxU64 sceneId);
	PxU32 getFrameCount(PxU64 sceneId);
	PxU64 getInnerFrameTime(PxU64 inStartEvent);
	PxU64 getTotalFrameTime(PxU64 inStartEvent);
	PxU64 getFrameEndTimestamp(PxU64 inStartEvent);
	// What event does the next frame correspond to.
	// If we are at the end, then we use PX_MAX_U64
	PxU64 findNextFrameEvent(PxU64 inStartEvent, PxU64 sceneId);
	// What event does the previous frame correspond to
	PxU64 findPreviousFrameEvent(PxU64 inStartEvent, PxU64 sceneId);
	// Get the total timespan of the DB capture.
	PxU64 getTotalFrameTimespan();

	virtual PxU32 getNbProfileZones() = 0;
	virtual PxU32 getProfileZones(PvdProfileZone* buffer, PxU32 bufSize, PxU32 idx = 0) = 0;

	virtual PxU32 getNbMemoryEventEntries() = 0;
	virtual PxU32 getMemoryEventEntries(PvdMemoryEventEntry* buffer, PxU32 bufSize, PxU32 idx = 0) = 0;
	virtual PxI32 getMemoryEventBufferInstance() = 0;
	virtual PxI32 getMemoryTotalsInstance() = 0;

	static NamespacedName getUserRendererClassName() { return NamespacedName("_debugger_", "PvdUserRenderer"); }

	virtual PvdCommStreamMetaData& clone(PvdObjectModelMetaData& metaData, PvdObjectModelReader& modelReader) = 0;
	virtual void save(PvdOutputStream& outputStream) = 0;

	virtual PvdCommStreamMetaDataMutator& cloneForTransfer(PvdObjectModelMetaData& metaData, PvdObjectModelReader& modelReader, PvdObjectModelMutator& model,
	                                                       PvdObjectModelEventStreamUserEventListener& eventFactory) = 0;

	static PvdCommStreamMetaData& create(PxAllocatorCallback& callback, PvdObjectModelMetaData& metaData, PvdObjectModelReader& modelReader, PvdInputStream& inStream);
};

class PvdCommStreamMetaDataMutator
{
  protected:
	virtual ~PvdCommStreamMetaDataMutator() {}

  public:
	virtual void addRef() = 0;
	virtual void release() = 0;
	virtual PvdCommStreamMetaData& getMetaData() = 0;
	virtual PvdObjectModelEventStreamUserEventTransfer& getEventTransfer() = 0;

	virtual void setTimestampNanosecondConversion(PxU64 numerator, PxU64 denominator) = 0;
	virtual void setPickable(PxU64 streamId, bool pickable) = 0;
	virtual void setColor(PxU64 streamId, const PvdColor& color) = 0;
	virtual void setIsTopLevel(PxU64 streamId, bool isTopLevel) = 0;
	virtual void setInstanceId(PxU64 streamId, PxI32 instance) = 0;
	virtual void destroyInstance(PxU64 streamId) = 0;
	virtual void setCamera(StringHandle name, const PxVec3& pos, const PxVec3& up, const PxVec3& target) = 0;
	virtual void sendErrorMessage(PxU32 code, StringHandle message, StringHandle file, PxU32 name) = 0;
	// odd event streams are dealt with as follows.
	//
	// begin,begin - The previous section's end event will equal ts, and its eventId will be one less.
	//
	// end,end - The current event's begin ts will be the previous event's end ts, and its
	// begin eventId will be one more than the previous events end event id.
	virtual void addSectionEvent(PxU64 streamId, StringHandle name, bool isBegin, PxU64 timestamp) = 0;
	// Delete all section event information as it no longer applies.
	virtual void onRecordingStopped() = 0;
	virtual void onRecordingStarted() = 0;
	virtual void addProfileZone(PxU64 sid, String nm) = 0;
	virtual void addProfileZoneEvent(PxU64 sid, String nm, PxU16 evId, bool enabled) = 0;

	virtual void addMemoryEventEntry(String type, String file, PxU32 line, PxU32 totalsArrayIndex) = 0;
	// events on the event buffer instance, 'totals' on the totals instance
	virtual void setMemoryEventInstances(PxI32 eventBuffer, PxI32 totals) = 0;

	static PvdCommStreamMetaDataMutator& create(PxAllocatorCallback& callback, PvdObjectModelMetaData& metaData, PvdObjectModelReader& modelReader, PvdObjectModelMutator& model,
	                                            PvdObjectModelEventStreamUserEventListener& eventFactory);
};
} // namespace pvdapp

#endif
