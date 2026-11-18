/*
 * SPDX-FileCopyrightText: Copyright (c) 2008-2014 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Copyright (c) 2008-2014 NVIDIA Corporation. All rights reserved.
#ifndef PVDAPP_CLIENT_H
#define PVDAPP_CLIENT_H
#include "PvdSimpleFoundation.h"
#include "PxPvdObjectModelBaseTypes.h"

namespace physx
{

class PxAllocatorCallback;

namespace pvdsdk
{
class PvdObjectModelMetaData;
class PvdObjectModelReader;
class PvdObjectModelMutator;
} // namespace pvdsdk

namespace profile
{
class PxProfileBulkEventHandler;
class PxProfileBulkMemoryEventHandler;
} // namespace profile
} // namespace physx

namespace pvdapp
{
class PvdCommStreamEventReader;
class PvdCommStreamMetaData;
class PvdCommStreamParser;
class PvdObjectModelEventStreamFactory;
class PvdObjectModelEventStream;
class PvdCommStreamMetaDataMutator;
class PvdRandomAccessIOStream;

class PvdClient
{
  protected:
	virtual ~PvdClient() {}

  public:
	// These four bytes begin all of our files.
	static PxU32 getFileId() { return 907025328; }

	virtual void addRef() = 0;
	virtual void release() = 0;

	// Frame 0 corresponds to event 0.  Seeking to a frame
	// means seeking to the event recorded at its end section.
	// So frame 1 ends at section[0].mEnd.mEventId.
	// The current frame cannot equal frame count, it is at most
	// frame count - 1.
	virtual PxU32 getCurrentFrame(PxU64 sceneId) = 0;
	virtual void setCurrentFrame(PxU32 frame, PxU64 sceneId) = 0;

	virtual PxU32 getFrameCount(PxU64 sceneId) = 0;
	virtual PxU64 getEventCount() = 0;
	virtual PxI64 getFileSize() = 0;

	virtual PxI64 getCurrentEvent() = 0;
	virtual void setCurrentEvent(PxI64 val) = 0;

	virtual void parseProfileEvents(PxI64 startEvent, PxI64 endEvent, int profileZone, physx::profile::PxProfileBulkEventHandler& handler) = 0;
	virtual PxU32 getNbMemoryTotals() = 0;
	virtual PxU32 getMemoryTotals(PxI32* buffer, PxU32 bufSize) = 0;
	virtual void parseMemoryEvents(PxI64 startEvent, PxI64 endEvent, physx::profile::PxProfileBulkMemoryEventHandler& handler) = 0;

	virtual PvdCommStreamMetaData& getStreamMetaData() = 0;
	virtual physx::pvdsdk::PvdObjectModelMetaData& getMetaData() = 0;
	virtual physx::pvdsdk::PvdObjectModelReader& getReader() = 0;
	// Note that this object may or may not exist.  When we are recording,
	// or if this item is loaded from a file then we have an event stream.
	// When we are just capturing without recording, there may not be an
	// event stream.
	virtual PvdObjectModelEventStream* getEventStream() = 0;

	virtual PvdClient* transferEvents(PxU64 eventCount, PvdRandomAccessIOStream& fileBacking) = 0;

	// Create a distinct client sharing only the file backend and copying everything else.
	// It should be possible to add information to this client if it has a mutator while
	// accesssing the other client;i.e. no mutable objects can be shared.  The file stream,
	// for all intents and purposes, is the only immutable object and as such it is probably
	// the only thing shared.
	virtual PvdClient* branch() = 0;

	virtual void save(PvdRandomAccessIOStream& ioStream) const = 0;

	// If the data file isn't a valid pvd file, this function returns NULL.
	static PvdClient* load(physx::PxAllocatorCallback& alloc, PvdRandomAccessIOStream& data);
};

// The client mutator is either monitoring events
// or recording events.  If it is monitoring events
// then it is not using an event stream factory.
// If it is recording events then it is, and the
// mutation
class PvdClientMutator
{
  protected:
	virtual ~PvdClientMutator() {}

  public:
	virtual void addRef() = 0;
	virtual void release() = 0;

	virtual PvdClient& getClient() = 0;
	virtual PvdCommStreamMetaDataMutator& getCommStreamMetaDataMutator() = 0;
	virtual physx::pvdsdk::PvdObjectModelMutator& getObjectModelMutator() = 0;

	virtual void disableRecording() = 0;
	virtual void enableRecording(PvdRandomAccessIOStream& recordingBacking) = 0;
	virtual bool isRecording() const = 0;

	virtual physx::pvdsdk::PvdObjectModelMutator& getRawModelMutator() = 0;
	virtual PvdObjectModelEventStreamFactory* getEventStreamFactory() = 0;
	virtual PvdCommStreamParser& createParser(PvdCommStreamEventReader& reader) = 0;

	static PvdClientMutator& create(physx::PxAllocatorCallback& alloc, PvdRandomAccessIOStream& fileBacking, bool isCapture);
};
} // namespace pvdapp

#endif
