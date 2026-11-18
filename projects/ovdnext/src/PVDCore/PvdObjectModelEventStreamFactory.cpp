/*
 * SPDX-FileCopyrightText: Copyright (c) 2008-2014 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Copyright (c) 2008-2014 NVIDIA Corporation. All rights reserved.

#include "PvdObjectModelEventStreamInternal.h"
#include "PvdObjectModelEventStreamPreviousEvents.h"

using namespace pvdapp;

namespace
{

class PvdObjectModelEventStreamFactoryImpl : public PvdObjectModelMutator, public PvdObjectModelEventStreamFactory
{
	typedef HashMap<PxI32, PreviousEvent*> TPreviousEventsMap; // map instance ids to previous events.

	PxAllocatorCallback& mAllocator;
	PvdRefPtr<PvdObjectModelReader> mReader;
	PvdObjectModel* mModel;
	PvdRefPtr<PvdRandomAccessIOStream> mStream;
	PxU64 mPageSizeCutoff;
	PxU32 mSnapshotLen;

	PvdRefPtr<PvdObjectModelEventStreamMetaData> mMetaData;

	Array<InstanceDescription> mTempDescriptionBuffer;
	Array<PxU64> mSnapshotLastWriteOffsets;

	// Offsets, relative to the start of the page.
	TPreviousEventsMap mPreviousEventsMap;
	Array<PxU32> mTempPreviousRelativeIds;
	TPreviousEventPool mPreviousEventPool;
	PvdObjectModelEventStream& mEventStream;
	HashMap<PxI32, InstanceFilePosAndLength> mInstanceWriteMap;
	PxU32 mRefCount;

  public:
	PvdObjectModelEventStreamFactoryImpl(PxAllocatorCallback& alloc, PvdObjectModelReader& initialState, PvdObjectModel& model, PvdRandomAccessIOStream& stream, PxU32 psc)
	: mAllocator(alloc)
	, mReader(&model)
	, mModel(&model)
	, mStream(&stream)
	, mPageSizeCutoff(psc)
	, mSnapshotLen(psc * 5)
	, mMetaData((PVD_NEW(PvdObjectModelEventStreamMetaData)(alloc)))
	, mTempDescriptionBuffer("PvdObjectModelEventStreamFactoryImpl::mSnapshots")
	, mSnapshotLastWriteOffsets("PvdObjectModelEventStreamFactoryImpl::mSnapshotLastWriteOffsets")
	, mPreviousEventsMap("PvdObjectModelEventStreamFactoryImpl::mPreviousEventsMap")
	, mTempPreviousRelativeIds("PvdObjectModelEventStreamFactoryImpl::mTempPreviousRelativeIds")
	, mPreviousEventPool("PvdObjectModelEventStreamFactoryImpl::mPreviousEventPool")
	, mEventStream(PvdObjectModelEventStream::create(alloc, *mMetaData, model, stream, 0))
	, mInstanceWriteMap("PvdObjectModelEventStreamFactoryImpl::mInstanceWriteMap")
	, mRefCount(0)
	{
		// Save out initial state.
		saveSnapshot(initialState);
		mEventStream.addRef();
		// Ensure that our event map records the state of any live objects in the snapshot
		// This is so further set property type events have accurate previous event maps.
		PvdObjectModelReader& reader(initialState);
		PxU32 numItems = reader.getNbInstances();
		mTempDescriptionBuffer.resize(numItems);
		reader.getInstances(mTempDescriptionBuffer.begin(), numItems);
		PVD_FOREACH(idx, numItems)
		{
			SnapshotPreviousEvent evt;
			PxU64 evtId = 0;
			PreviousEvent* previousEvent = mPreviousEventPool.construct(evtId, evt);
			mPreviousEventsMap[mTempDescriptionBuffer[idx].mId] = previousEvent;
		}
	}

	PageFileData& getWritePage()
	{
		if(mMetaData->mWritePage.hasValue() == false)
			mMetaData->mWritePage = Option<PageFileData>(PageFileData(0));
		return mMetaData->mWritePage.getValue();
	}

	Option<InstanceFilePosAndLength> getLastWriteInfo(PxI32 id)
	{
		const HashMap<PxI32, InstanceFilePosAndLength>::Entry* entry = mInstanceWriteMap.find(id);
		if(entry != NULL)
			return entry->second;
		return None();
	}

	void setLastWriteInfo(PxI32 id, InstanceFilePosAndLength info) { mInstanceWriteMap[id] = info; }

	void clearLastWriteInfo(PxI32 id) { mInstanceWriteMap.erase(id); }

	void clearLastWriteInfo(void* id) { mInstanceWriteMap.erase(mModel->ptrToId(id)); }

	void commitWritePage()
	{
		PX_ASSERT(mMetaData->mWritePage.hasValue());
		if(mMetaData->mWritePage.hasValue())
			mMetaData->mPageInfo.pushBack(mMetaData->mWritePage.getValue());
		mMetaData->mWritePage = Option<PageFileData>(PageFileData(mMetaData->mEventCount));
	}

	virtual ~PvdObjectModelEventStreamFactoryImpl()
	{
		clearPreviousEventsMap();
		mEventStream.release();
	}

	void deletePreviousEventsChain(PreviousEvent* current)
	{
		while(current)
		{
			PreviousEvent* next = current->getPreviousEvent();
			mPreviousEventPool.deallocate(current);
			current = next;
		}
	}

	void clearPreviousEventsMapEntry(PxI32 id)
	{
		const TPreviousEventsMap::Entry* mapEntry(mPreviousEventsMap.find(id));
		if(mapEntry)
		{
			deletePreviousEventsChain(mapEntry->second);
			mPreviousEventsMap.erase(id);
		}
	}

	void clearPreviousEventsMap()
	{
		for(TPreviousEventsMap::Iterator iter = mPreviousEventsMap.getIterator(); iter.done() == false; ++iter)
		{
			deletePreviousEventsChain(iter->second);
		}
		mPreviousEventsMap.clear();
	}
	PvdObjectModelReader& getReader() { return *mReader; }
	const PvdObjectModelReader& getReader() const { return *mReader; }
	virtual void* idToPtr(PxI32 instId) const { return getReader().idToPtr(instId); }
	virtual PxI32 ptrToId(void* instPtr) const { return getReader().ptrToId(instPtr); }
	virtual InstanceDescription idToDescriptor(PxI32 instId) const { return getReader().idToDescriptor(instId); }
	virtual InstanceDescription ptrToDescriptor(void* instId) const { return getReader().ptrToDescriptor(instId); }
	virtual Option<ClassDescription> getClassOf(void* instId) const { return getReader().getClassOf(instId); }
	virtual const PvdObjectModelMetaData& getMetaData() const { return getReader().getMetaData(); }

	void ensureStreamIsAtEnd()
	{
		if(mMetaData->mEventCount && mMetaData->mCurrentEvent < mMetaData->mEventCount - 1)
			mEventStream.seekToEvent(mMetaData->mEventCount - 1);
		mMetaData->mCurrentEvent = mMetaData->mEventCount ? mMetaData->mEventCount - 1 : 0;
	}

	// Used just before adding a new event to make sure that no matter what
	// happens it is clear that this event has just occurred.
	void incrementCurrentEvent()
	{
		PX_ASSERT(mMetaData->mEventCount);
		PxU64 maxEvent = mMetaData->mEventCount;
		PX_ASSERT(mMetaData->mCurrentEvent < maxEvent);
		if(mMetaData->mCurrentEvent < maxEvent)
			++mMetaData->mCurrentEvent;
	}

	void recordCompletedInstances() { mModel->recordCompletedInstances(); }

	void doCreateInstance(PxI32 clsId, PxI32 instId, void* instIdPtr)
	{
		// The reason we pin/unpin instances here is because someone could do this operation:
		// inst createInstance
		// eventStream->seekToEvent(0); //removing the instance from the alive list.
		// modelReader->clearUpdatedInstances(); //removing the instance from the updated list and thus nuking the inst
		// ptr.
		// setPropertyValue( inst );
		// These don't necessarily happen right away, or in the same context of code.
		// The debugger accesses the factory from one thread and the event stream from another
		// so through the magic of threading and users scrolling through data while recording,
		// that exact possible set of circumstances could easily happen.
		mModel->pinInstance(instIdPtr);
		PX_ASSERT(mPreviousEventsMap.find(instId) == nullptr);
		CreateInstancePreviousEvent eventData;
		PreviousEvent* previousEvent = mPreviousEventPool.construct(mMetaData->mEventCount, eventData);
		mPreviousEventsMap[instId] = previousEvent;
		incrementCurrentEvent();
		addEvent(CreateInstance_(instId, clsId));
	}

	virtual InstanceDescription createInstance(PxI32 clsId, PxI32 instId)
	{
		ensureStreamIsAtEnd();
		InstanceDescription retval(mModel->createInstance(clsId, instId));
		if(retval.mInstPtr)
			doCreateInstance(clsId, instId, retval.mInstPtr);
		return retval;
	}

	virtual InstanceDescription createInstance(PxI32 clsId)
	{
		ensureStreamIsAtEnd();
		InstanceDescription retval(mModel->createInstance(clsId));
		if(retval.mInstPtr)
			doCreateInstance(clsId, retval.mId, retval.mInstPtr);
		return retval;
	}
	virtual void pinInstance(void* instId) { mModel->pinInstance(instId); }
	virtual void unPinInstance(void* instId) { mModel->unPinInstance(instId); }
	void afterDestroyInstance(PxI32 id, PxU64 fileOffset)
	{
		PxU64 streamlength = mStream->getLength();
		PxU32 instanceFileLength = static_cast<PxU32>(streamlength - fileOffset);

		Option<InstanceFilePosAndLength> lastVal = getLastWriteInfo(id);
		if(lastVal.hasValue())
			instanceFileLength = lastVal.getValue().mLength;

		mMetaData->mInstanceDestructionMap.insert(id, InstanceFilePosAndLength(fileOffset, instanceFileLength));
		clearPreviousEventsMapEntry(id);
	}
	virtual void destroyInstance(void* instId)
	{
		ensureStreamIsAtEnd();
		InstanceDescription desc = getReader().ptrToDescriptor(instId);
		PxI32 id = desc.mId;
		if(id >= 0)
		{
			PxU64 offset = writeInstance(*mModel, desc);
			mModel->destroyInstance(instId);
			mModel->unPinInstance(instId);
			incrementCurrentEvent();
			addEvent(DestroyInstance_(id));
			afterDestroyInstance(id, offset);
		}
	}
	virtual PxI32 getNextInstanceHandleValue() const { return mModel->getNextInstanceHandleValue(); }
	// reserve a set of instance handle values by getting the current, adding an amount to it
	// and setting the value.  You can never set the value lower than it already is, it only climbs.
	virtual void setNextInstanceHandleValue(PxI32 hdlValue) { mModel->setNextInstanceHandleValue(hdlValue); }

	PxU64 getEventCount() { return mMetaData->mEventCount; }

	struct PreviousEventResolveOp : public PreviousEvent::IIntersectionOp
	{
		virtual ~PreviousEventResolveOp() {}
		virtual void resolve(PreviousEvent& evt, Array<PxU32>& ids, Pool<PreviousEvent>& pool) { evt.resolve(ids, pool, *this); }
	};

	DataRef<PxU32> resolvePreviousEvents(PreviousEvent& previousEvent, PxI32 id, PreviousEventResolveOp& resolveOp)
	{
		// Figure out the previous events that influence this dataset.
		// create the set property value event.
		const TPreviousEventsMap::Entry* entry(mPreviousEventsMap.find(id));
		DataRef<PxU32> previousIds;
		if(entry)
		{
			mTempPreviousRelativeIds.clear();
			previousEvent.setPreviousEvent(entry->second);

			resolveOp.resolve(previousEvent, mTempPreviousRelativeIds, mPreviousEventPool);
			PxU32 numEvents = mTempPreviousRelativeIds.size();
			PxU32 numSwaps = numEvents / 2;
			// reverse the vector.
			for(PxU32 idx = 0; idx < numSwaps; ++idx)
				physx::shdfnd::swap(mTempPreviousRelativeIds[idx], mTempPreviousRelativeIds[numEvents - idx - 1]);
			previousIds = DataRef<PxU32>(mTempPreviousRelativeIds.begin(), mTempPreviousRelativeIds.size());
		}
		mPreviousEventsMap[id] = &previousEvent;
		return previousIds;
	}

	struct SPVResolveOp : PreviousEventResolveOp
	{
		PxI32 instId;
		PxI32 propId;
		SPVResolveOp(PxI32 _instId, PxI32 _propId) : instId(_instId), propId(_propId) {}
		virtual const IntersectResult& handleEvent(PreviousEvent& evt) { return evt.intersectWith(instId, propId); }
	};

	virtual bool setPropertyValue(void* instId, PxI32 propId, const PxU8* data, PxU32 dataLen, PxI32 incomingType)
	{
		ensureStreamIsAtEnd();
		if(mModel->setPropertyValue(instId, propId, data, dataLen, incomingType) == false)
			return false;

		PxI32 id = getReader().ptrToId(instId);
		SetPropertyValuePreviousEvent current(propId);

		PreviousEvent* previousEvent = mPreviousEventPool.construct(mMetaData->mEventCount, current);
		SPVResolveOp resolveOp(id, propId);
		DataRef<PxU32> previousIds = resolvePreviousEvents(*previousEvent, id, resolveOp);
		addMutatingEvent(SetPropertyValue_(id, propId, DataRef<const PxU8>(data, dataLen), previousIds));
		return true;
	}

	struct SPMResolveOp : PreviousEventResolveOp
	{
		PxI32 instId;
		PxI32 msgId;
		const PropertyMessageDescription& msg;
		SPMResolveOp(PxI32 _instId, PxI32 _id, const PropertyMessageDescription& _msg) : instId(_instId), msgId(_id), msg(_msg) {}
		virtual const IntersectResult& handleEvent(PreviousEvent& evt) { return evt.intersectWith(instId, msgId, msg); }
	};

	// Set a set of properties defined by a property message
	virtual bool setPropertyMessage(void* instId, PxI32 msgId, const PxU8* data, PxU32 dataLen)
	{
		ensureStreamIsAtEnd();
		if(mModel->setPropertyMessage(instId, msgId, data, dataLen) == false)
			return false;
		PropertyMessageDescription msg = getReader().getMetaData().getPropertyMessage(msgId);
		PxI32 id = getReader().ptrToId(instId);

		SetPropertyMessagePreviousEvent evtData(&getReader().getMetaData(), msgId);
		PreviousEvent* previousEvent = mPreviousEventPool.construct(mMetaData->mEventCount, evtData);
		SPMResolveOp resolveOp(id, msgId, msg);
		DataRef<PxU32> previousIds = resolvePreviousEvents(*previousEvent, id, resolveOp);
		addMutatingEvent(SetPropertyMessage_(id, msgId, DataRef<const PxU8>(data, dataLen), previousIds));
		return true;
	}

	void doInsertArrayElement(void* instId, PxI32 propId, PxI32 propIdx, const PxU8* data, PxU32 dataLen)
	{
		PxI32 id = getReader().ptrToId(instId);
		InsertArrayElementPreviousEvent evtData(propId, propIdx);
		PreviousEvent* previousEvent = mPreviousEventPool.construct(mMetaData->mEventCount, evtData);
		const TPreviousEventsMap::Entry* entry(mPreviousEventsMap.find(id));
		if(entry)
			previousEvent->setPreviousEvent(entry->second);
		mPreviousEventsMap[id] = previousEvent;
		// This event doesn't care what the previous events were.  It has self-contained do and undo
		// Later events, (removeArrayElement and setArrayElementValue) do need to know about inserts
		// that happen on the same index, however.
		addMutatingEvent(InsertArrayElement(id, propId, propIdx, DataRef<const PxU8>(data, dataLen)));
	}

	virtual bool insertArrayElement(void* instId, PxI32 propId, PxI32 propIdx, const PxU8* data, PxU32 dataLen, PxI32 incomingType)
	{
		ensureStreamIsAtEnd();
		if(mModel->insertArrayElement(instId, propId, propIdx, data, dataLen, incomingType) == false)
			return false;
		clearLastWriteInfo(instId);
		doInsertArrayElement(instId, propId, propIdx, data, dataLen);
		return true;
	}

	struct ArrayElementResolveOp : PreviousEventResolveOp
	{
		PxI32 instId;
		PxI32 propId;
		PxI32 propIdx;
		ArrayElementResolveOp(PxI32 _instId, PxI32 _id, PxI32 _idx) : instId(_instId), propId(_id), propIdx(_idx) {}
		virtual const IntersectResult& handleEvent(PreviousEvent& evt)
		{
			// Intersection may change prop idx.
			return evt.intersectWithArrayElement(instId, propId, propIdx);
		}
	};

	void doRemoveArrayElement(void* instId, PxI32 propId, PxI32 propIdx)
	{
		PxI32 id = getReader().ptrToId(instId);
		RemoveArrayElementPreviousEvent evtData(propId, propIdx);
		PreviousEvent* previousEvent = mPreviousEventPool.construct(mMetaData->mEventCount, evtData);
		ArrayElementResolveOp resolveOp(id, propId, propIdx);
		DataRef<PxU32> previousIds = resolvePreviousEvents(*previousEvent, id, resolveOp);
		addMutatingEvent(RemoveArrayElement(id, propId, propIdx, previousIds));
	}

	virtual bool removeArrayElement(void* instId, PxI32 propId, PxI32 propIdx)
	{
		ensureStreamIsAtEnd();
		if(mModel->removeArrayElement(instId, propId, propIdx) == false)
			return false;
		doRemoveArrayElement(instId, propId, propIdx);
		return true;
	}

	virtual Option<PxI32> pushBackArrayElementIf(void* instId, PxI32 propId, const PxU8* data, PxU32 dataLen, PxI32 incomingType)
	{
		ensureStreamIsAtEnd();
		Option<PxI32> retval(mModel->pushBackArrayElementIf(instId, propId, data, dataLen, incomingType));
		if(retval.hasValue() == false)
			return retval; // there was an error with the function call itself
		PxI32 itemIdx = retval;
		if(itemIdx < 0)
			return retval; // The item is already in the array so we didn't do it.
		clearLastWriteInfo(instId);
		doInsertArrayElement(instId, propId, itemIdx, data, dataLen);
		return retval;
	}

	virtual Option<PxI32> removeArrayElementIf(void* instId, PxI32 pname, const PxU8* data, PxU32 dataLen, PxI32 incomingType)
	{
		ensureStreamIsAtEnd();
		Option<PxI32> retval(mModel->removeArrayElementIf(instId, pname, data, dataLen, incomingType));
		if(retval.hasValue() == false)
			return retval; // there was an error with the function call itself
		PxI32 itemIdx = retval;
		if(itemIdx < 0)
			return retval; // The item is already in the array so we didn't do it.
		clearLastWriteInfo(instId);
		doRemoveArrayElement(instId, pname, itemIdx);
		return retval;
	}

	virtual bool setArrayElementValue(void* instId, PxI32 propId, PxI32 propIdx, const PxU8* data, PxU32 dataLen, PxI32 incomingType)
	{
		ensureStreamIsAtEnd();
		// didn't work.
		if(mModel->setArrayElementValue(instId, propId, propIdx, data, dataLen, incomingType) == false)
		{
			return false;
		}
		PxI32 id = getReader().ptrToId(instId);
		SetArrayElementValuePreviousEvent evtData(propId, propIdx);
		PreviousEvent* previousEvent = mPreviousEventPool.construct(mMetaData->mEventCount, evtData);
		ArrayElementResolveOp resolveOp(id, propId, propIdx);
		DataRef<PxU32> previousIds = resolvePreviousEvents(*previousEvent, id, resolveOp);
		addMutatingEvent(SetArrayElementValue(id, propId, propIdx, DataRef<const PxU8>(data, dataLen), previousIds));
		return true;
	}

	virtual void originShift(void* instId, PxVec3 shift)
	{
		ensureStreamIsAtEnd();

		PxI32 id = getReader().ptrToId(instId);
		mModel->originShift(instId, shift);

		PxU32 nbShiftedPair = mModel->getNbShifted();
		if(nbShiftedPair == 0)
		{
			return;
		}
		Array<InstancePropertyPair> shiftedArray("originShift()::shiftedArray");
		shiftedArray.resizeUninitialized(nbShiftedPair);

		mModel->getShiftedPair(&shiftedArray[0], shiftedArray.size());

		DataRef<InstancePropertyPair> shiftedArrayDataRef(&shiftedArray[0], shiftedArray.size());

		// OriginShiftPreviousEvent eventData(shiftedArrayDataRef);
		// PreviousEvent* previousEvent = mPreviousEventPool.construct( mMetaData->mEventCount, eventData );
		typedef HashMap<PxI32, OriginShiftPreviousEvent> TOriginShiftPreviousEventsMap; // map instance ids to previous
		                                                                                // events.

		TOriginShiftPreviousEventsMap evtMap("originShift()::evtMap");

		for(PxU32 i = 0; i < shiftedArray.size(); i++)
		{
			const TOriginShiftPreviousEventsMap::Entry* entry = evtMap.find(shiftedArray[i].first);
			if(entry != NULL)
			{
				// Always overwrite the second property, we only trace 2 properties
				evtMap[entry->first].mPropId2 = shiftedArray[i].second;
			}
			else
			{
				OriginShiftPreviousEvent eventData(shiftedArray[i].first, shiftedArray[i].second, 0);
				evtMap[shiftedArray[i].first] = eventData;
			}
		}

		for(TOriginShiftPreviousEventsMap::Iterator iter = evtMap.getIterator(); iter.done() == false; ++iter)
		{
			OriginShiftPreviousEvent& eventData = iter->second;
			PreviousEvent* previousEvent = mPreviousEventPool.construct(mMetaData->mEventCount, eventData);

			const TPreviousEventsMap::Entry* entry = mPreviousEventsMap.find(iter->first);
			if(entry != NULL)
			{
				previousEvent->setPreviousEvent(mPreviousEventsMap[iter->first]);
				mPreviousEventsMap[iter->first] = previousEvent;
			}
			else
				mPreviousEventsMap[iter->first] = previousEvent;
		}

		/*for( PxU32 i = 0; i < shiftedArray.size(); i++ )
		{
		    OriginShiftPreviousEvent eventData( shiftedArray[i].first, shiftedArray[i].second, 0 );
		    PreviousEvent* previousEvent = mPreviousEventPool.construct( mMetaData->mEventCount, eventData );

		    const TPreviousEventsMap::Entry* entry = mPreviousEventsMap.find(shiftedArray[i].first);
		    if(entry != NULL)
		    {
		        PreviousEvent* currentPreviousEvt = entry->second;

		        //If duplicated events
		        if( currentPreviousEvt->getEventId() ==  mMetaData->mEventCount
		            && currentPreviousEvt->getType() == previousEvent->getType())
		        {
		            currentPreviousEvt->mEventData.get();
		        }
		        previousEvent->setPreviousEvent( mPreviousEventsMap[shiftedArray[i].first] );
		        mPreviousEventsMap[shiftedArray[i].first] = previousEvent;
		    }
		    else
		        mPreviousEventsMap[shiftedArray[i].first] = previousEvent;
		}
		*/

		addMutatingEvent(OriginShiftEvent(id, shift, shiftedArrayDataRef));
	}

	virtual PxU64 addUserEvent(PxI32 dtype, DataRef<const PxU8> data)
	{
		addEvent(UserEvent(dtype, data));
		return getEventCount() - 1;
	}

	PxU64 writeInstance(PvdObjectModelReader& model, InstanceDescription& desc)
	{
		Option<InstanceFilePosAndLength> lastVal = getLastWriteInfo(desc.mId);
		if(lastVal.hasValue())
			return lastVal.getValue().mOffset;

		PvdOutputStream& outStream(*mStream);
		PxU64 streamOffset = mStream->getLength();
		model.writeInstance(desc, outStream);

		InstanceFilePosAndLength info(streamOffset, static_cast<PxU32>(mStream->getLength() - streamOffset));
		setLastWriteInfo(desc.mId, info);

		// Regardless, we now add an event.
		incrementCurrentEvent();
		addEvent(SavedInstance(desc.mId, streamOffset));
		return streamOffset;
	}

	void saveSnapshot() { saveSnapshot(*mModel); }

	void saveSnapshot(PvdObjectModelReader& model)
	{
		ensureStreamIsAtEnd();
		mMetaData->mSnapshots.pushBack(SnapshotInfo(mMetaData->mEventCount, 0, 0));
		addEvent(Snapshot());
		mMetaData->mCurrentEvent = mMetaData->mEventCount - 1;

		// Ensure we don't write out any pages while this process is occurring.
		PxU64 pageSizeCutoff = mPageSizeCutoff;
		mPageSizeCutoff = (PxU64)-1;
		PxU32 itemCount = model.getNbInstances();
		mTempDescriptionBuffer.resize(itemCount);
		model.getInstances(mTempDescriptionBuffer.begin(), itemCount);
		mSnapshotLastWriteOffsets.clear();
		PvdOutputStream& outStream(*mStream);
		// Ensure all instances have valid offsets
		// If we aren't saving from our model, then we can't
		// use the last write offsets from some other model.
		PVD_FOREACH(idx, itemCount)
		{
			PxU64 offset = writeInstance(model, mTempDescriptionBuffer[idx]);
			mSnapshotLastWriteOffsets.pushBack(offset);
		}

		PxU64 snapshotBegin = mStream->getLength();
		outStream << itemCount;
		// One may wonder why snapshots do not participate in the previous events map.
		// This is because when we track back through the datatset trying to find all the
		// places where a particule property value's value changes, we use the previous events
		// to efficient track back through the dataset and find the events.
		// So I take an efficiency hit when scrubbing back in order to gain a bit of efficiency
		// during queries.
		// write out a header telling where each instance starts relative to the start of the
		// snapshot data.
		PX_ASSERT(mSnapshotLastWriteOffsets.size() == itemCount);
		for(PxU32 idx = 0; idx < itemCount; ++idx)
		{
			outStream << mTempDescriptionBuffer[idx].mId;
			outStream << mSnapshotLastWriteOffsets[idx];
		}
		mMetaData->mSnapshots.back().mLength = static_cast<PxU32>(mStream->getLength() - snapshotBegin);
		mMetaData->mSnapshots.back().mFileOffset = snapshotBegin;
		mPageSizeCutoff = pageSizeCutoff;
	}

	void beginAddEvent()
	{
		mMetaData->mCurrentPageEventOffsets.pushBack(mMetaData->mPageBuffer.size());
		++mMetaData->mEventCount;
	}

	void endAddEvent()
	{
		getWritePage().mEventCount = static_cast<PxU32>(mMetaData->mCurrentPageEventOffsets.size());
		if(mMetaData->mPageBuffer.size() >= mPageSizeCutoff)
		{
			PxU32 bufSize = mMetaData->mPageBuffer.size();
			ensureStreamIsAtEnd();
			writeOutPage();
			PX_ASSERT(mMetaData->mPageBuffer.size() == 0);
			if(bufSize > mSnapshotLen)
			{
				saveSnapshot();
				mSnapshotLen = static_cast<PxU32>(mPageSizeCutoff * 5);
			}
			else
				mSnapshotLen -= static_cast<PxU32>(mPageSizeCutoff);
		}
	}

	template <typename TEventType>
	void addMutatingEvent(const TEventType& evt)
	{
		clearLastWriteInfo(evt.mInstanceId);
		incrementCurrentEvent();
		addEvent(evt);
	}

	template <typename TEventType>
	void addEvent(TEventType evt)
	{
		beginAddEvent();
		PvdObjectModelEventStreamType::Enum evtType = getObjectModelEventTypeForDataType<TEventType>();
		PxU32 eventType = static_cast<PxU32>(evtType);
		mMetaData->mPageBuffer << eventType;
		EventWriter writer(mMetaData->mPageBuffer);
		evt.serialize(writer);
		endAddEvent();
	}

	void writeOutPage()
	{
		PxU32 bufSize = mMetaData->mPageBuffer.size();
		if(bufSize == 0)
			return;
		mStream->zeroPadToPageSize();
		PxU32 pageEventCount = static_cast<PxU32>(mMetaData->mCurrentPageEventOffsets.size());
		PxU64 fileOffset = mStream->getLength();
		getWritePage().beforePageWrite(pageEventCount, fileOffset);
		PvdOutputStream& outStream(*mStream);
		// Write out the number of events in the page
		outStream << pageEventCount;
		outStream.write(reinterpret_cast<PxU8*>(mMetaData->mCurrentPageEventOffsets.begin()), sizeof(mMetaData->mCurrentPageEventOffsets[0]) * pageEventCount);
		outStream.write(mMetaData->mPageBuffer.begin(), bufSize);

		mMetaData->mPageBuffer.clear();
		mMetaData->mCurrentPageEventOffsets.clear();
		getWritePage().afterPageWrite(static_cast<PxU32>(mStream->getLength() - getWritePage().mFileOffset));
		commitWritePage();
	}

	virtual PvdObjectModelMutator& getMutator() { return *this; }
	virtual PvdObjectModelEventStream& getEventStream() { return mEventStream; }

	virtual PvdObjectModelReader& getModel() { return *mModel; }
	virtual PvdRandomAccessIOStream& getIOStream() { return *mStream; }
	virtual void flush() { writeOutPage(); }

	template <typename TDataType>
	void applyTransferEvent(MemPvdInputStream& stream)
	{
		EventReader reader(stream);
		PvdObjectModelEventForwardMutatorImpl mutator(*this);
		TDataType evt;
		evt.serialize(reader);
		evt.doEvent(mutator);
	}

	virtual void transferEvents(DataRef<const PxU32> offsets, DataRef<const PxU8> data, PvdObjectModelEventStreamTransferDataProvider& dataSupplier,
	                            PvdObjectModelEventStreamUserEventTransfer& userEventHandler)
	{
		PX_UNUSED(dataSupplier);
		PxU32 evtCount = offsets.size();
		for(PxU32 idx = 0; idx < evtCount; ++idx)
		{
			PxU32 relativeOffset = offsets[idx];
			const PxU8* evtStart = data.begin() + relativeOffset;
			const PxU8* evtEnd = data.end();

			if(idx < evtCount - 1)
				evtEnd = data.begin() + offsets[idx + 1];

			MemPvdInputStream stream(evtStart, evtEnd);
			PxU32 evtType;
			stream >> evtType;
			switch(evtType)
			{
			// Snapshots are ignored
			case PvdObjectModelEventStreamType::Snapshot:
				break;
			case PvdObjectModelEventStreamType::SavedInstance:
				break;
			// User events are forwarded to something that can do
			// something about them.
			case PvdObjectModelEventStreamType::UserEvent:
			{
				EventReader reader(stream);
				UserEvent evt;
				evt.serialize(reader);
				userEventHandler.handleUserEvent(*this, evt.mType, evt.mData);
			}
			break;
#define DEFAULT_HANDLE_TRANSFER_EVENT(type)                                                                                                                                                            \
	case PvdObjectModelEventStreamType::type:                                                                                                                                                          \
		applyTransferEvent<type>(stream);                                                                                                                                                              \
		break

				DEFAULT_HANDLE_TRANSFER_EVENT(DestroyInstance_);
				DEFAULT_HANDLE_TRANSFER_EVENT(SetPropertyValue_);
				DEFAULT_HANDLE_TRANSFER_EVENT(SetPropertyMessage_);
				DEFAULT_HANDLE_TRANSFER_EVENT(InsertArrayElement);
				DEFAULT_HANDLE_TRANSFER_EVENT(RemoveArrayElement);
				DEFAULT_HANDLE_TRANSFER_EVENT(SetArrayElementValue);
				DEFAULT_HANDLE_TRANSFER_EVENT(CreateInstance_);
				DEFAULT_HANDLE_TRANSFER_EVENT(OriginShiftEvent);

#undef DEFAULT_HANDLE_TRANSFER_EVENT
			default:
				PX_ASSERT(false);
				break;
			}
		}
	}
	virtual void addRef() { ++mRefCount; }
	virtual void release()
	{
		if(mRefCount)
			--mRefCount;
		if(!mRefCount)
			PVD_DELETE(this);
	}
};
} // namespace

PvdObjectModelEventStreamFactory& PvdObjectModelEventStreamFactory::create(PxAllocatorCallback& callback, PvdObjectModelReader& initialState, PvdObjectModel& model, PvdRandomAccessIOStream& stream,
                                                                           PxU32 pageSize)
{
	return *PVD_NEW(PvdObjectModelEventStreamFactoryImpl)(callback, initialState, model, stream, pageSize);
}

PvdObjectModelEventStreamFactory& PvdObjectModelEventStreamFactory::create(PxAllocatorCallback& callback, PvdObjectModel& model, PvdRandomAccessIOStream& stream, PxU32 pageSize)
{
	return create(callback, model, model, stream, pageSize);
}
