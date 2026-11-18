/*
 * SPDX-FileCopyrightText: Copyright (c) 2008-2014 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Copyright (c) 2008-2014 NVIDIA Corporation. All rights reserved.

#ifndef PVD_EVENT_STREAM_H
#define PVD_EVENT_STREAM_H
#include "PvdSimpleFoundation.h"
#include "PxPvdObjectModel.h"
#include "PxPvdObjectModelBaseTypes.h"

namespace pvdapp
{
class PvdRandomAccessIOStream;
class PvdObjectModelEventStreamMetaData;
class PvdObjectModelEventStreamFactory;
class PvdObjectModelEventStreamTransferDataProvider;

inline uint64_t toPaddedSize(uint64_t inOriginal, uint32_t inPageSize = 0x1000) { return (inOriginal + inPageSize) - inOriginal % inPageSize; }

template <typename TObjType>
struct PvdRefPtr
{
	mutable TObjType* mObj;
	~PvdRefPtr() { release(); }
	PvdRefPtr(TObjType* obj = NULL) : mObj(obj) { addRef(); }
	PvdRefPtr(const PvdRefPtr<TObjType>& other)
	{
		mObj = other.mObj;
		addRef();
	}
	PvdRefPtr<TObjType>& operator=(const PvdRefPtr<TObjType>& other)
	{
		if(mObj != other.mObj)
			release();
		mObj = other.mObj;
		addRef();
		return *this;
	}
	void addRef()
	{
		if(mObj)
			mObj->addRef();
	}
	void release()
	{
		if(mObj)
			mObj->release();
	}
	operator TObjType*() { return mObj; }
	operator const TObjType*() const { return mObj; }

	TObjType* operator->()
	{
		PX_ASSERT(mObj);
		return mObj;
	}
	const TObjType* operator->() const
	{
		PX_ASSERT(mObj);
		return mObj;
	}

	TObjType& operator*()
	{
		PX_ASSERT(mObj);
		return *mObj;
	}
	const TObjType& operator*() const
	{
		PX_ASSERT(mObj);
		return *mObj;
	}
};

// using namespace physx;
class PvdObjectModelEventStreamPropertyHandler
{
  protected:
	virtual ~PvdObjectModelEventStreamPropertyHandler() {}

  public:
	virtual void handleProperty(PxU64 eventIndex, PxI32 instanceId, PxI32 propertyId, const PxU8* data, PxU32 len) = 0;
};

class PvdObjectModelEventStreamUserEventTransfer
{
  protected:
	virtual ~PvdObjectModelEventStreamUserEventTransfer() {}

  public:
	virtual void handleUserEvent(PvdObjectModelEventStreamFactory& factory, PxI32 dtype, physx::pvdsdk::DataRef<const PxU8> data) = 0;
};

struct InstancePropertyId
{
	PxI32 mInstanceId;
	PxI32 mPropertyId;
	InstancePropertyId(PxI32 instId, PxI32 propId) : mInstanceId(instId), mPropertyId(propId) {}
	InstancePropertyId() {}
};

class PvdObjectModelEventStream
{
  protected:
	virtual ~PvdObjectModelEventStream() {}

  public:
	virtual void addRef() = 0;
	virtual void release() = 0;

	virtual physx::pvdsdk::PvdObjectModelReader& getModel() = 0;

	virtual PxU64 getEventCount() const = 0;
	virtual PxU64 getCurrentEvent() const = 0;

	virtual PxU64 getFileSize() const = 0;

	virtual void seekToEvent(PxU64 eventIndex) = 0;

	// seek to event, but forcefully by loading a snapshot
	virtual void loadSnapshot(PxU64 evtIdx) = 0;

	// Property values are iterated through in the order in which they were set.
	virtual bool getPropertyHistory(physx::pvdsdk::DataRef<InstancePropertyId> queries, PxU64 beginEvent, PxU64 endEvent, PvdObjectModelEventStreamPropertyHandler& handler) = 0;
	bool getPropertyHistory(InstancePropertyId query, PxU64 beginEvent, PxU64 endEvent, PvdObjectModelEventStreamPropertyHandler& handler)
	{
		physx::pvdsdk::DataRef<InstancePropertyId> queries(&query, 1);
		return getPropertyHistory(queries, beginEvent, endEvent, handler);
	}

	virtual PvdObjectModelEventStream& branch(physx::pvdsdk::PvdObjectModel& newModel) = 0;
	// Save this item out to a file.  Open will work on the result.
	virtual void save(PvdRandomAccessIOStream& stream) = 0;

	// Creates a new meta data object also
	virtual PvdObjectModelEventStreamFactory& createTransferFactory(PvdRandomAccessIOStream& stream, physx::pvdsdk::PvdObjectModelMetaData& meta) = 0;
	// Save off events in this range to a new stream.
	// Save a snapshot of the current DB as event 0.
	// Save next N events, including deleted objects another system
	virtual bool transfer(PxU64 eventCount, PvdObjectModelEventStreamFactory& transferFactory, PvdObjectModelEventStreamUserEventTransfer& userEventHandler) = 0;

	PvdObjectModelEventStreamFactory& transfer(PxU64 startEvent, PxU64 eventCount, PvdRandomAccessIOStream& stream, PvdObjectModelEventStreamUserEventTransfer& userEventHandler)
	{
		PxU64 current = getCurrentEvent();
		seekToEvent(startEvent);
		PvdObjectModelEventStreamFactory& factory(createTransferFactory(stream, getModel().getMetaData().clone()));
		transfer(eventCount, factory, userEventHandler);
		seekToEvent(current);
		return factory;
	}

	// croptoend
	// croptobeginning
	// splitatcurrent
	// Load the stream meta and such from this file and run from here
	static PvdObjectModelEventStream& load(physx::PxAllocatorCallback& callback, physx::pvdsdk::PvdObjectModel& modelAndInitialState, PvdRandomAccessIOStream& inStream, PxU64 initialOffset);
	static PvdObjectModelEventStream& create(physx::PxAllocatorCallback& callback, PvdObjectModelEventStreamMetaData& metaData, physx::pvdsdk::PvdObjectModel& modelAndInitialState,
	                                         PvdRandomAccessIOStream& inStream, PxU64 pageSectionStart);
};

class PvdObjectModelEventStreamUserEventListener
{
  protected:
	virtual ~PvdObjectModelEventStreamUserEventListener() {}

  public:
	virtual physx::pvdsdk::PvdObjectModelMutator& getMutator() = 0;
	// Add an event with arbitrary data.  Returns the event index where
	// this event was added.
	virtual PxU64 addUserEvent(PxI32 dtype, physx::pvdsdk::DataRef<const PxU8> userData) = 0;
	template <typename TDataType>
	PxU64 addUserEvent(const TDataType* dtype, PxU32 numItems)
	{
		const PxU8* dataStart = reinterpret_cast<const PxU8*>(dtype);
		PxI32 dtypeClass = getMutator().getMetaData().findClass(getPvdNamespacedNameForType<TDataType>())->mClassId;
		return addUserEvent(dtypeClass, physx::pvdsdk::DataRef<const PxU8>(dataStart, numItems * sizeof(TDataType)));
	}
	template <typename TDataType>
	PxU64 addUserEvent(const TDataType& dtype)
	{
		return addUserEvent(&dtype, 1U);
	}
};

class PvdObjectModelEventStreamFactory : public PvdObjectModelEventStreamUserEventListener
{
  protected:
	virtual ~PvdObjectModelEventStreamFactory() {}

  public:
	virtual void addRef() = 0;
	virtual void release() = 0;

	// Gets the event stream associated with this factory.
	// The event stream and the factory share the underlying object model
	// and event stream meta data.
	virtual PvdObjectModelEventStream& getEventStream() = 0;

	// Add an event that doesn't do anything but marks a spot
	// in the event stream.  Returns the event index in the current stream.
	virtual physx::pvdsdk::PvdObjectModelReader& getModel() = 0; // represents the result of the latest events.
	virtual PvdRandomAccessIOStream& getIOStream() = 0;
	virtual void flush() = 0;

	// For the cropping functionality
	virtual void transferEvents(physx::pvdsdk::DataRef<const PxU32> offsets, physx::pvdsdk::DataRef<const PxU8> data, PvdObjectModelEventStreamTransferDataProvider& dataSupplier,
	                            PvdObjectModelEventStreamUserEventTransfer& userEventHandler) = 0;

	static PvdObjectModelEventStreamFactory& create(physx::PxAllocatorCallback& callback, physx::pvdsdk::PvdObjectModelReader& initialState, physx::pvdsdk::PvdObjectModel& model,
	                                                PvdRandomAccessIOStream& inStream, PxU32 pageSizeCutoff = 0x200000);
	// Start a new page after cutoff has been reached.  Will save out data to inStream while writing information
	// Event 0 corresponds to the model's initial state.
	// This object does not clear the model's update list.
	static PvdObjectModelEventStreamFactory& create(physx::PxAllocatorCallback& callback, physx::pvdsdk::PvdObjectModel& modelAndInitialState, PvdRandomAccessIOStream& inStream,
	                                                PxU32 pageSizeCutoff = 0x200000);
};
} // namespace pvdapp

#endif
