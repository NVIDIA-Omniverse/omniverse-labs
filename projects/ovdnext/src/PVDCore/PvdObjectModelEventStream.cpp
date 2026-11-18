/*
 * SPDX-FileCopyrightText: Copyright (c) 2008-2014 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Copyright (c) 2008-2014 NVIDIA Corporation. All rights reserved.
#include "PvdObjectModelEventStreamInternal.h"
#include <algorithm>  //lower bound
#include <functional> //greater

#include "profiler.h"

using namespace pvdapp;

namespace
{

template <typename TOperator>
struct PageEventOperator
{
	TOperator& mOperator;
	EventReader mReader;

	PageEventOperator(TOperator& op, MemPvdInputStream& stream) : mOperator(op), mReader(stream) {}

	PageEventOperator& operator=(const PageEventOperator& other);
	template <typename TEventType>
	TEventType deserialize()
	{
		TEventType evt;
		evt.serialize(mReader);
		return evt;
	}

	void handleCreateInstance() { mOperator.handleCreateInstance(deserialize<CreateInstance_>()); }

	void handleDestroyInstance() { mOperator.handleDestroyInstance(deserialize<DestroyInstance_>()); }

	void handleSetPropertyValue() { mOperator.handleSetPropertyValue(deserialize<SetPropertyValue_>()); }

	void handleSetPropertyMessage() { mOperator.handleSetPropertyMessage(deserialize<SetPropertyMessage_>()); }

	void handleInsertArrayElement() { mOperator.handleInsertArrayElement(deserialize<InsertArrayElement>()); }

	void handleRemoveArrayElement() { mOperator.handleRemoveArrayElement(deserialize<RemoveArrayElement>()); }

	void handleSetArrayElementValue() { mOperator.handleSetArrayElementValue(deserialize<SetArrayElementValue>()); }

	void handleUserEvent() { mOperator.handleUserEvent(UserEvent()); }

	void handleSnapshot() { mOperator.handleSnapshot(Snapshot()); }

	void handleSavedInstance() { mOperator.handleSavedInstance(deserialize<SavedInstance>()); }

	void handleOriginShiftEvent() { mOperator.handleOriginShiftEvent(deserialize<OriginShiftEvent>()); }
};

struct LoadedPage
{
	PxU64 mFirstEvent;
	PxU64 mLastEvent;
	PxU64 mFileOffset;
	PxU32 mEventCount;
	PxU8* mPagePointer;
	PxU32* mRelativeEventOffsets;
	PxU8* mDataStart;
	PxU8* mDataEnd;
	PxU64 mLastUse;

	LoadedPage(const PxU64 startEvt = 0, const PxU64 lastEvent = 0, const PxU64 offset = 0, const PxU32 evtCount = 0, PxU8* pagePtr = 0, PxU32* offsets = 0, PxU8* ds = 0, PxU8* de = 0)
	: mFirstEvent(startEvt), mLastEvent(lastEvent), mFileOffset(offset), mEventCount(evtCount), mPagePointer(pagePtr), mRelativeEventOffsets(offsets), mDataStart(ds), mDataEnd(de), mLastUse(0)
	{
	}

	template <typename TOperator>
	inline void operateOnPageEvent(const PxU64 eventIndex, TOperator& op)
	{
		if(eventIndex < mFirstEvent || eventIndex > mLastEvent)
		{
			// PX_ASSERT(false);
			return;
		}
		const PxU32 relativeEvent = static_cast<PxU32>(eventIndex - mFirstEvent);
		if(relativeEvent >= mEventCount)
		{
			PX_ASSERT(false);
			return;
		}
		const PxU32 relativeOffset = mRelativeEventOffsets[relativeEvent];
		MemPvdInputStream inputStream(mDataStart + relativeOffset, mDataEnd);
		PxU32 eventType = 0;
		inputStream >> eventType;
		PageEventOperator<TOperator> evtOp(op, inputStream);
		visitPvdObjectModelEventType(static_cast<PvdObjectModelEventStreamType::Enum>(eventType), evtOp);
	}
};

struct EventDoOperator
{
	PvdObjectModelEventMutator& mMutator;
	EventDoOperator(PvdObjectModelEventMutator& mutator) : mMutator(mutator) {}

	void handleCreateInstance(CreateInstance_ evt) { evt.doEvent(mMutator); }

	void handleDestroyInstance(DestroyInstance_ evt) { evt.doEvent(mMutator); }

	void handleSetPropertyValue(SetPropertyValue_ evt) { evt.doEvent(mMutator); }
	void handleSetPropertyMessage(SetPropertyMessage_ evt) { evt.doEvent(mMutator); }

	void handleInsertArrayElement(InsertArrayElement evt) { evt.doEvent(mMutator); }

	void handleRemoveArrayElement(RemoveArrayElement evt) { evt.doEvent(mMutator); }

	void handleSetArrayElementValue(SetArrayElementValue evt) { evt.doEvent(mMutator); }

	void handleUserEvent(UserEvent) {}
	void handleSnapshot(Snapshot) {}
	void handleSavedInstance(SavedInstance) {}
	void handleOriginShiftEvent(OriginShiftEvent evt) { evt.doEvent(mMutator); }
};

struct EventUndoOperator
{
	PvdObjectModelEventMutator& mMutator;
	PxI64 mEventIdx;
	EventUndoOperator(PvdObjectModelEventMutator& mutator, PxI64 evtIdx) : mMutator(mutator), mEventIdx(evtIdx) {}

	void handleCreateInstance(CreateInstance_ evt) { evt.undoEvent(mMutator); }

	void handleDestroyInstance(DestroyInstance_ evt) { evt.undoEvent(mMutator); }

	void handleSetPropertyValue(SetPropertyValue_ evt) { evt.undoEvent(mEventIdx, mMutator); }
	void handleSetPropertyMessage(SetPropertyMessage_ evt) { evt.undoEvent(mEventIdx, mMutator); }

	void handleInsertArrayElement(InsertArrayElement evt) { evt.undoEvent(mEventIdx, mMutator); }

	void handleRemoveArrayElement(RemoveArrayElement evt) { evt.undoEvent(mEventIdx, mMutator); }

	void handleSetArrayElementValue(SetArrayElementValue evt) { evt.undoEvent(mEventIdx, mMutator); }
	void handleUserEvent(UserEvent) {}
	void handleSnapshot(Snapshot) {}
	void handleSavedInstance(SavedInstance) {}
	void handleOriginShiftEvent(OriginShiftEvent evt) { evt.undoEvent(mEventIdx, mMutator); }
};

class PvdObjectModelEventStreamImpl;

//
#pragma pack(push)
#pragma pack(4)
struct InstanceIdAndOffset
{
	PxI32 mInstanceId;
	PxU64 mOffset;
	InstanceIdAndOffset(PxI32 instId, PxU64 off) : mInstanceId(instId), mOffset(off) {}
	InstanceIdAndOffset() : mInstanceId(0), mOffset(0) {}
};
#pragma pack(pop)

struct LoadedSnapshot
{
	PxU64 mEventIndex;
	PxU32 mInstanceCount;
	Array<InstanceIdAndOffset> mSnapshotHeader;
	LoadedSnapshot() : mEventIndex((PxU64)-1), mInstanceCount(0), mSnapshotHeader("LoadedSnapshot::mSnapshotHeader") {}
	void clear()
	{
		mEventIndex = (PxU64)-1;
		mInstanceCount = 0;
		mSnapshotHeader.clear();
	}
	void load(PvdInputStream& snapshot, const PxU64 evtIndex)
	{
		mEventIndex = evtIndex;
		readArray(snapshot, mSnapshotHeader);
		mInstanceCount = mSnapshotHeader.size();
	}
};

struct PvdRandomAccessIOStreamWrapper
{
	PvdRandomAccessIOStream& mStream;
	PxU64 mOffset;

	PvdRandomAccessIOStreamWrapper(PvdRandomAccessIOStream& stream, PxU64 offset) : mStream(stream), mOffset(offset) { mStream.addRef(); }
	~PvdRandomAccessIOStreamWrapper() { mStream.release(); }
	Option<ErrorCode> read(PxU8* buffer, PxU32 inLen, PxU64 inOffset) { return mStream.read(buffer, inLen, inOffset + mOffset); }
};

class PvdObjectModelEventStreamImpl : public PvdObjectModelEventStream, public PvdObjectModelEventMutator, public PvdObjectModelEventStreamTransferDataProvider
{
	PxAllocatorCallback& mAllocator;
	PvdObjectModel& mModel;
	HashMap<PxU64, LoadedPage> mInMemoryPages;
	PvdObjectModelEventStreamMetaData& mMetaData;
	PvdRandomAccessIOStreamWrapper mStream;
	LoadedSnapshot mLoadedSnapshot;
	PxU64 mLastUsedPage;
	const PxU64 MAX_OFFSET = static_cast<PxU64>(-1);
	PxU64 mLockedPageOffset;
	PxU32 mMaxLoadedPages;
	Array<PxU8> mDestroyedInstanceBuffer;
	Option<LoadedPage> mLastLoadedPage;
	MemPvdOutputStream mSnapshotInstanceLoader;
	PxU64 mLastSnapshotInstanceOffset;

	PxU32 mRefCount;

  public:
	PvdObjectModelEventStreamImpl(PxAllocatorCallback& alloc, PvdObjectModel& model, PvdObjectModelEventStreamMetaData& meta, PvdRandomAccessIOStream& stream, PxU32 maxLoadedPages,
	                              PxU64 pageSectionStart)
	: mAllocator(alloc)
	, mModel(model)
	, mInMemoryPages("PvdObjectModelEventStreamImpl::mInMemoryPages")
	, mMetaData(meta)
	, mStream(stream, pageSectionStart)
	, mLoadedSnapshot()
	, mLastUsedPage(0)
	, mLockedPageOffset(MAX_OFFSET)
	, mMaxLoadedPages(maxLoadedPages)
	, mDestroyedInstanceBuffer("PvdObjectModelEventStreamImpl::mDestroyedInstanceBuffer")
	, mSnapshotInstanceLoader("PvdObjectModelEventStreamImpl::mSnapshotInstanceLoader")
	, mLastSnapshotInstanceOffset((PxU64)-1)
	, mRefCount(0)
	{
		mModel.addRef();
		mMetaData.addRef();
	}

	virtual ~PvdObjectModelEventStreamImpl()
	{
		mModel.release();
		for(HashMap<PxU64, LoadedPage>::Iterator iter = mInMemoryPages.getIterator(); iter.done() == false; ++iter)
		{
			mAllocator.deallocate(iter->second.mPagePointer);
		}
		mInMemoryPages.clear();
		mMetaData.release();
	}
	virtual void addRef() { ++mRefCount; }
	virtual void release()
	{
		if(mRefCount)
			--mRefCount;
		if(!mRefCount)
			PVD_DELETE(this);
	}

	virtual PvdObjectModelReader& getModel() { return mModel; }

	Option<LoadedPage> getPageForEvent(const PxU64 evt)
	{
		if(mLastLoadedPage.hasValue() && evt >= mLastLoadedPage->mFirstEvent && evt <= mLastLoadedPage->mLastEvent)
		{
			return mLastLoadedPage;
		}
		mLastLoadedPage = Option<LoadedPage>();
		Option<LoadedPage> retval = doGetPageForEvent(evt);
		if(retval.hasValue() && retval->mFileOffset != MAX_OFFSET)
			mLastLoadedPage = retval;
		return retval;
	}

	Option<LoadedPage> doGetPageForEvent(const PxU64 evt)
	{
		if(mMetaData.mWritePage.hasValue() && evt >= mMetaData.mWritePage->mFirstEventIndex)
		{
			PX_ASSERT(mMetaData.mPageBuffer.size());
			PX_ASSERT(mMetaData.mCurrentPageEventOffsets.size());
			if(mMetaData.mPageBuffer.size() && mMetaData.mCurrentPageEventOffsets.size())
			{
				const PxU32 numEvents = mMetaData.mCurrentPageEventOffsets.size();
				const PxU64 startEvt = mMetaData.mEventCount - numEvents;
				const PxU64 lastEvent = startEvt + numEvents - 1;
				const PxU64 offset = MAX_OFFSET;
				const PxU32 evtCount = numEvents;
				PxU8* pagePtr = const_cast<PxU8*>(mMetaData.mPageBuffer.begin());
				PxU32* offsets = mMetaData.mCurrentPageEventOffsets.begin();
				PxU8* ds = pagePtr;
				PxU8* de = pagePtr + mMetaData.mPageBuffer.size();
				LoadedPage retval(startEvt, lastEvent, offset, evtCount, pagePtr, offsets, ds, de);
				return retval;
			}
			return None();
		}
		if(mMetaData.mPageInfo.size() == 0)
		{
			PX_ASSERT(false);
			return None();
		}
		// first figure out where the page must sit.
		PageFileData* begin = mMetaData.mPageInfo.begin();
		PageFileData* end = begin + mMetaData.mPageInfo.size();
		auto PageFileDataFirstEventComparator = [](const PageFileData& lhs, const PageFileData& rhs) -> bool { return lhs.mFirstEventIndex < rhs.mFirstEventIndex; };
		PageFileData* pageData = std::lower_bound(begin, end, PageFileData(evt), PageFileDataFirstEventComparator);
		if(pageData == end)
			--pageData;
		else if(pageData != begin && pageData->mFirstEventIndex > evt)
			--pageData;

		PX_ASSERT(pageData->mFirstEventIndex <= evt);

		const HashMap<PxU64, LoadedPage>::Entry* lastEntry = mInMemoryPages.find(pageData->mFileOffset);
		if(lastEntry)
		{
			LoadedPage* thePage = const_cast<LoadedPage*>(&lastEntry->second);
			thePage->mLastUse = ++mLastUsedPage;
			return *thePage;
		}
		// else dump the page from memory
		if(static_cast<PxU32>(mInMemoryPages.size()) == mMaxLoadedPages)
		{
			LoadedPage* removePage = NULL;
			PxU64 minLastUse = MAX_OFFSET;
			PxU64 fileOffset = 0;
			for(HashMap<PxU64, LoadedPage>::Iterator iter = mInMemoryPages.getIterator(); iter.done() == false; ++iter)
			{
				if(iter->first != mLockedPageOffset && iter->second.mLastUse < minLastUse)
				{
					removePage = &iter->second;
					fileOffset = iter->first;
					minLastUse = removePage->mLastUse;
				}
			}
			if(removePage != NULL)
			{
				mAllocator.deallocate(reinterpret_cast<void*>(removePage->mPagePointer));
				mInMemoryPages.erase(fileOffset);
			}
		}

		// Now load the desired page into memory
		PxU8* pagePtr = static_cast<PxU8*>(mAllocator.allocate(pageData->mLength, "PvdObjectModelEventStreamImpl::PageData", __FILE__, __LINE__));
		Option<ErrorCode> error = mStream.read(pagePtr, pageData->mLength, pageData->mFileOffset);
		if(error.hasValue() == true)
		{
			PX_ASSERT(false);
			mAllocator.deallocate(pagePtr);
			return None();
		}
		PxU32* offsetsStart = reinterpret_cast<PxU32*>(pagePtr + 4);
		PxU8* eventDataStart = reinterpret_cast<PxU8*>(offsetsStart + pageData->mEventCount);
		PxU8* eventDataEnd = pagePtr + pageData->mLength;
		LoadedPage theLoadedPage(pageData->mFirstEventIndex, pageData->mFirstEventIndex + pageData->mEventCount - 1, pageData->mFileOffset, pageData->mEventCount, pagePtr, offsetsStart,
		                         eventDataStart, eventDataEnd);

		theLoadedPage.mLastUse = ++mLastUsedPage;
		mInMemoryPages[pageData->mFileOffset] = theLoadedPage;
		return theLoadedPage;
	}

	virtual PxU64 getEventCount() { return mMetaData.mEventCount; }
	virtual PxU64 getCurrentEvent() { return mMetaData.mCurrentEvent; }

	virtual PxU64 getFileSize() const { return mStream.mStream.getLength(); }

	virtual void seekToEvent(PxU64 eventIndex)
	{
		if(mMetaData.mEventCount == 0)
			return;
		eventIndex = static_cast<PxU64>(PxMin(eventIndex, mMetaData.mEventCount - 1));
		PX_ASSERT(mMetaData.mCurrentEvent < mMetaData.mEventCount);
		if(mMetaData.mCurrentEvent >= mMetaData.mEventCount)
			mMetaData.mCurrentEvent = mMetaData.mEventCount - 1;

		// We assume loading a snapshot takes around 500 events of time.
		// Probably should time this, but it would depend on the size of the
		// working set.
		const PxU64 snapshotPenalty = 500;
		const PxU64 distance = PxMax(mMetaData.mCurrentEvent, eventIndex) - PxMin(mMetaData.mCurrentEvent, eventIndex);
		if(distance > snapshotPenalty && mMetaData.mSnapshots.size())
		{
			// find the closest snapshot
			SnapshotInfo testInfo(eventIndex);
			auto SnapshotInfoEventComparator = [](const SnapshotInfo& lhs, const SnapshotInfo& rhs) -> bool { return lhs.mEventIndex < rhs.mEventIndex; };

			SnapshotInfo* nearest = std::lower_bound(mMetaData.mSnapshots.begin(), mMetaData.mSnapshots.end(), testInfo, SnapshotInfoEventComparator);
			if(nearest == mMetaData.mSnapshots.end())
				--nearest;

			else if(nearest != mMetaData.mSnapshots.begin())
			{
				SnapshotInfo* lesserShot = nearest - 1;
				nearest = lesserShot;
			}

			if(nearest != NULL)
			{
				PxU64 snapDistance = nearest->mEventIndex > eventIndex ? nearest->mEventIndex - eventIndex : eventIndex - nearest->mEventIndex;
				PxU32 reloadInstanceCost = nearest->mLength >> 3; // estimate the cost
				if(mMetaData.mCurrentEvent > eventIndex || (snapDistance + reloadInstanceCost) < distance)
					loadSnapshot(nearest->mEventIndex);
			}
		}

		Option<LoadedPage> thePage = getPageForEvent(mMetaData.mCurrentEvent);
		if(thePage.hasValue() == false)
		{
			PX_ASSERT(false);
			return;
		}
		while(mMetaData.mCurrentEvent > eventIndex)
		{
			mLockedPageOffset = thePage->mFileOffset;
			EventUndoOperator op(*this, mMetaData.mCurrentEvent);
			thePage->operateOnPageEvent(mMetaData.mCurrentEvent, op);
			--mMetaData.mCurrentEvent;
			if(thePage->mFirstEvent > mMetaData.mCurrentEvent)
			{
				thePage = getPageForEvent(mMetaData.mCurrentEvent);
				if(thePage.hasValue() == false)
				{
					mLockedPageOffset = MAX_OFFSET;
					PX_ASSERT(false);
					return;
				}
			}
		}
		while(mMetaData.mCurrentEvent < eventIndex)
		{
			++mMetaData.mCurrentEvent;
			if(thePage->mLastEvent < mMetaData.mCurrentEvent)
			{
				thePage = getPageForEvent(mMetaData.mCurrentEvent);
				if(thePage.hasValue() == false)
				{
					mLockedPageOffset = MAX_OFFSET;
					PX_ASSERT(false);
					return;
				}
			}
			mLockedPageOffset = thePage->mFileOffset;
			EventDoOperator op(*this);
			thePage->operateOnPageEvent(mMetaData.mCurrentEvent, op);
		}
		mLockedPageOffset = MAX_OFFSET;
	}

	struct PropertyHistoryRecorder
	{
	  private:
		PropertyHistoryRecorder(const PropertyHistoryRecorder& other);
		PropertyHistoryRecorder& operator=(const PropertyHistoryRecorder& other);

	  public:
		InstancePropertyId mQuery;
		const PvdObjectModelMetaData& mMetaData;
		Array<PxU64>& mPreviousEventsList;
		Array<PxU64>& mRecordedEventsList;
		PxU64 mEventId;
		PropertyHistoryRecorder(InstancePropertyId query, const PvdObjectModelMetaData& meta, Array<PxU64>& prevEvents, Array<PxU64>& recordedEvents)
		: mQuery(query), mMetaData(meta), mPreviousEventsList(prevEvents), mRecordedEventsList(recordedEvents), mEventId(0)
		{
		}
		bool hasFoundQuery(PxI32 instId, PxI32 propId)
		{
			const InstancePropertyId& id = mQuery;
			return id.mInstanceId == instId && id.mPropertyId == propId;
		}

		void addPreviousEvents(DataRef<PxU32> previousEvents)
		{
			if(mRecordedEventsList.size() == 0 || mRecordedEventsList.back() != mEventId)
				mRecordedEventsList.pushBack(mEventId);

			PVD_FOREACH(prevIdx, previousEvents.size())
			{
				PxU32 prevValue = previousEvents[prevIdx];
				if(prevValue <= mEventId)
				{
					PxU64 previousEventIdx = mEventId - prevValue;
					mPreviousEventsList.pushBack(previousEventIdx);
					std::push_heap(mPreviousEventsList.begin(), mPreviousEventsList.end(), std::greater<PxU64>());
				}
			}
		}

		void handleSetPropertyValue(SetPropertyValue_ evt)
		{
			// bool foundQuery = hasFoundQuery( evt.mInstanceId, evt.mPropertyId );
			if(hasFoundQuery(evt.mInstanceId, evt.mPropertyId))
				addPreviousEvents(evt.mRelativePreviousEvents);
		}

		void handleSetPropertyMessage(SetPropertyMessage_ evt)
		{
			const PropertyMessageDescription& msg(mMetaData.getPropertyMessage(evt.mMsgId));
			bool foundQuery = false;
			PVD_FOREACH(idx, msg.mProperties.size())
			{
				const PropertyMessageEntry& entry(msg.mProperties[idx]);
				foundQuery |= hasFoundQuery(evt.mInstanceId, entry.mProperty.mPropertyId);
			}
			if(foundQuery)
				addPreviousEvents(evt.mRelativePreviousEvents);
		}

		// nops for these events.
		void handleUserEvent(UserEvent) {}
		void handleDestroyInstance(DestroyInstance_) {}
		void handleCreateInstance(CreateInstance_) {}
		void handleSnapshot(Snapshot) {}
		void handleInsertArrayElement(InsertArrayElement) {}
		void handleRemoveArrayElement(RemoveArrayElement) {}
		void handleSetArrayElementValue(SetArrayElementValue) {}
		void handleSavedInstance(SavedInstance) {}
		void handleOriginShiftEvent(OriginShiftEvent) {}
	};

	struct PropertyHistoryApplier
	{
	  private:
		PropertyHistoryApplier(const PropertyHistoryApplier& other);
		PropertyHistoryApplier& operator=(const PropertyHistoryApplier& other);

	  public:
		InstancePropertyId mQuery;
		PvdObjectModelEventStreamPropertyHandler& mHandler;
		const PvdObjectModelMetaData& mMetaData;
		PxU64 mEventId;

		PropertyHistoryApplier(InstancePropertyId query, PvdObjectModelEventStreamPropertyHandler& handler, const PvdObjectModelMetaData& meta)
		: mQuery(query), mHandler(handler), mMetaData(meta), mEventId(0)
		{
		}

		bool hasFoundQuery(PxI32 instId, PxI32 propId)
		{
			const InstancePropertyId& id = mQuery;
			return id.mInstanceId == instId && id.mPropertyId == propId;
		}

		void doHandlePropertyValue(PxI32 instId, PxI32 propId, const PxU8* data, PxU32 dataLen)
		{
			if(hasFoundQuery(instId, propId))
				mHandler.handleProperty(mEventId, instId, propId, data, dataLen);
		}

		void handleSetPropertyValue(SetPropertyValue_ evt) { doHandlePropertyValue(evt.mInstanceId, evt.mPropertyId, evt.mData.begin(), evt.mData.size()); }

		void handleSetPropertyMessage(SetPropertyMessage_ evt)
		{
			const PropertyMessageDescription& msg(mMetaData.getPropertyMessage(evt.mMsgId));
			PVD_FOREACH(idx, msg.mProperties.size())
			{
				const PropertyMessageEntry& entry(msg.mProperties[idx]);
				doHandlePropertyValue(evt.mInstanceId, entry.mProperty.mPropertyId, evt.mData.begin() + entry.mMessageOffset, entry.mByteSize);
			}
		}

		// nops
		void handleUserEvent(UserEvent) {}
		void handleDestroyInstance(DestroyInstance_) {}
		void handleCreateInstance(CreateInstance_) {}
		void handleSnapshot(Snapshot) {}
		void handleInsertArrayElement(InsertArrayElement) {}
		void handleRemoveArrayElement(RemoveArrayElement) {}
		void handleSetArrayElementValue(SetArrayElementValue) {}
		void handleSavedInstance(SavedInstance) {}
		void handleOriginShiftEvent(OriginShiftEvent) {}
	};

	// Two step algorithm
	// 1.  Run from end of section to beginning of section finding all events that apply and recording them.
	//	These events will be in reverse order because for efficiency we jump backwards through the event stream
	//	in order to find the next event using the previous event's previous event list to find the answer.
	// 2.  Run forward through recorded events applying them to the handler.  This ensures that the handler
	//	gets events in order, which is very important for some handlers (like profile stream parsers).
	virtual bool getPropertyHistory(DataRef<InstancePropertyId> queries, PxU64 beginEvent, PxU64 endEvent, PvdObjectModelEventStreamPropertyHandler& handler)
	{
		if(queries.size() == 0)
		{
			PX_ASSERT(false);
			return false;
		}
		endEvent = PxMin(endEvent, getEventCount());
		PxU64 temp = PxMax(endEvent, beginEvent);
		beginEvent = PxMin(endEvent, beginEvent);
		endEvent = temp;
		PxU64 eventCount = endEvent - beginEvent;
		PropertyHistoryRecorder* theRecorders =
		    static_cast<PropertyHistoryRecorder*>(mAllocator.allocate(queries.size() * sizeof(PropertyHistoryRecorder), "PropertyHistoryRecorder", __FILE__, __LINE__));
		PropertyHistoryApplier* theAppliers = static_cast<PropertyHistoryApplier*>(mAllocator.allocate(queries.size() * sizeof(PropertyHistoryApplier), "PropertyHistoryApplier", __FILE__, __LINE__));
		Array<PxU64> previousEvents("getPropertyHistory::previousEvents");
		Array<PxU64> recordedEvents("getPropertyHistory::recordedEvents");

		PVD_FOREACH(idx, queries.size())
		{
			new(theRecorders + idx) PropertyHistoryRecorder(queries[idx], getModel().getMetaData(), previousEvents, recordedEvents);
			new(theAppliers + idx) PropertyHistoryApplier(queries[idx], handler, getModel().getMetaData());
		}
		PxU64 idx = 0;
		while(idx < eventCount)
		{
			PxU64 eventIdx = endEvent - idx - 1;
			Option<LoadedPage> thePage = getPageForEvent(eventIdx);
			if(thePage.hasValue() == false)
			{
				PX_ASSERT(false);
				return false;
			}
			PVD_FOREACH(queryIdx, queries.size())
			{
				PropertyHistoryRecorder& theOp(theRecorders[queryIdx]);
				theOp.mEventId = eventIdx;
				thePage->operateOnPageEvent(eventIdx, theOp);
			}
			if(previousEvents.empty())
				++idx;
			else
			{
				PxU64 nextEvent = 0;
				do
				{
					nextEvent = previousEvents.back();
					previousEvents.popBack();
				} while(previousEvents.empty() == false && previousEvents.back() == nextEvent);
				idx = endEvent - nextEvent - 1;
			}
		}
		// Now go through each event in recorded events, backwards, and apply it to
		// the handlers.
		PxU64 lastEvent = 0;
		PVD_FOREACH(ridx, recordedEvents.size())
		{
			PxU32 recordedIdx = recordedEvents.size() - ridx - 1;
			PxU64 currentEvent = recordedEvents[recordedIdx];
			PX_UNUSED(lastEvent);
			PX_ASSERT(currentEvent > lastEvent);
			lastEvent = currentEvent;
			Option<LoadedPage> thePage = getPageForEvent(currentEvent);
			if(thePage.hasValue() == false)
			{
				PX_ASSERT(false);
				return false;
			}
			PVD_FOREACH(queryIdx, queries.size())
			{
				PropertyHistoryApplier& theOp(theAppliers[queryIdx]);
				theOp.mEventId = currentEvent;
				thePage->operateOnPageEvent(currentEvent, theOp);
			}
		}
		PVD_FOREACH(idx, queries.size())
		{
			theRecorders[idx].~PropertyHistoryRecorder();
			theAppliers[idx].~PropertyHistoryApplier();
		}
		mAllocator.deallocate(theRecorders);
		mAllocator.deallocate(theAppliers);
		return true;
	}

	virtual PvdObjectModelEventStream& branch(PvdObjectModel& newModel)
	{
		PvdObjectModelEventStreamMetaData& newMeta(mMetaData.clone());
		PvdObjectModelEventStreamImpl& newStream(*PVD_NEW(PvdObjectModelEventStreamImpl)(mAllocator, newModel, newMeta, mStream.mStream, mMaxLoadedPages, mStream.mOffset));
		// Make sure we are really at the correct event.
		// All streams start with a snapshot, so this should always be valid.
		newStream.loadSnapshot(0);
		return newStream;
	}

	virtual void loadSnapshot(PxU64 evtIdx)
	{
#ifdef PROFILING
		profiler::ProfilerScope prof_scope("PvdObjectModelEventStreamImpl::loadSnapshot()", 0);
#endif
		applySnapshot(evtIdx);
		mMetaData.mCurrentEvent = evtIdx;
	}

	virtual PxU64 getEventCount() const { return mMetaData.mEventCount; }
	virtual PxU64 getCurrentEvent() const { return mMetaData.mCurrentEvent; }

	virtual void save(PvdRandomAccessIOStream& stream)
	{
		mMetaData.save(stream);
		stream.zeroPadToPageSize();
		const PxU64 totalLen = mStream.mStream.getLength();
		const PxU64 dataLen = totalLen - mStream.mOffset;
		IOStreamToInputStream transferStream(mStream.mOffset, mStream.mStream);
		stream.directCopy(transferStream, dataLen);
	}

	virtual PvdObjectModelEventStreamFactory& createTransferFactory(PvdRandomAccessIOStream& newStream, PvdObjectModelMetaData& meta)
	{
		PvdObjectModel& model(PvdObjectModel::create(mAllocator, meta));
		PvdObjectModelEventStreamFactory& retval = PvdObjectModelEventStreamFactory::create(mAllocator, mModel, model, newStream);
		// Set us at a valid event 0;
		retval.getEventStream().loadSnapshot(0);
		return retval;
	}

	virtual bool transfer(PxU64 eventCount, PvdObjectModelEventStreamFactory& factory, PvdObjectModelEventStreamUserEventTransfer& userEventHandler)
	{
		eventCount = PxMin(eventCount, mMetaData.mEventCount - mMetaData.mCurrentEvent);
		for(PxU64 idx = 1; idx < eventCount;)
		{
			PxU64 eventIdx = mMetaData.mCurrentEvent + idx;
			Option<LoadedPage> pageOpt = getPageForEvent(eventIdx);
			if(pageOpt.hasValue() == false)
			{
				PX_ASSERT(false);
				return false;
			}
			LoadedPage& page(pageOpt);
			PxU64 numPossible = page.mLastEvent - eventIdx + 1;
			numPossible = PxMin(numPossible, eventCount);
			PxU32 relativeStart = static_cast<PxU32>(eventIdx - page.mFirstEvent);
			PxU32 numPageEvents = static_cast<PxU32>(PxMin(numPossible, eventCount - idx));
			const PxU32* offsetStart(page.mRelativeEventOffsets + relativeStart);
			const PxU32* offsetStop = offsetStart + numPageEvents;
			factory.transferEvents(DataRef<const PxU32>(offsetStart, offsetStop), DataRef<const PxU8>(page.mDataStart, page.mDataEnd), *this, userEventHandler);
			idx += numPageEvents;
		}
		factory.flush();
		return true;
	}

	// PvdObjectModelEventMutator Implementation

	virtual InstanceDescription idToDescriptor(PxI32 instId) const { return mModel.idToDescriptor(instId); }

	virtual void createInstance(PxI32 clsId, PxI32 instId) { mModel.createInstance(clsId, instId); }

	virtual void destroyInstance(PxI32 instId) { mModel.destroyInstance(mModel.idToPtr(instId)); }

	virtual void loadDestroyedInstance(PxI32 instId)
	{
		DataRef<const PxU8> instData(getDeletedInstance(instId));
		if(instData.size())
			mModel.readInstance(instData);
	}

	virtual void setPropertyValue(void* instId, PxI32 propId, const PxU8* data, PxU32 dataLen, PxI32 incomingType) { mModel.setPropertyValue(instId, propId, data, dataLen, incomingType); }

	virtual void originShift(void* instId, PxVec3 shift) { mModel.originShift(instId, shift); }

	bool loadSnapshotData(PxU64 evtId)
	{
#ifdef PROFILING
		profiler::ProfilerScope prof_scope("PvdObjectModelEventStreamImpl::loadSnapshotData()", 0);
#endif
		if(mLoadedSnapshot.mEventIndex == evtId)
			return true;
		PxU32 numSnapshots = mMetaData.mSnapshots.size();
		mLoadedSnapshot.clear();
		mLastSnapshotInstanceOffset = MAX_OFFSET;
		mSnapshotInstanceLoader.clear();
		PVD_FOREACH(idx, numSnapshots)
		{
			if(mMetaData.mSnapshots[idx].mEventIndex == evtId)
			{
				SnapshotInfo& theInfo(mMetaData.mSnapshots[idx]);
				PxU64 theOffset = mStream.mOffset + theInfo.mFileOffset;
				IOStreamToInputStream stream(theOffset, mStream.mStream);
				mLoadedSnapshot.load(stream, evtId);
				return true;
			}
		}
		return false;
	}

	// Data ends up in mSnapshotInstanceLoader
	DataRef<const PxU8> loadSnapshotInstance(PxU32 snapshotInstanceIndex)
	{
		if(snapshotInstanceIndex == mLoadedSnapshot.mInstanceCount)
		{
			PX_ASSERT(false);
			return DataRef<const PxU8>();
		}
		PxU64 fileOffset = mLoadedSnapshot.mSnapshotHeader[snapshotInstanceIndex].mOffset + mStream.mOffset;
		if(mLastSnapshotInstanceOffset != fileOffset)
		{
			IOStreamToInputStream stream(fileOffset, mStream.mStream);
			mSnapshotInstanceLoader.clear();
			PvdObjectModel::readInstance(stream, mSnapshotInstanceLoader);
			mLastSnapshotInstanceOffset = fileOffset;
		}
		return mSnapshotInstanceLoader.toRef();
	}

	void applySnapshotToInstance(PxI32 instId, PxU64 evtId, PxI32 propId)
	{
		if(loadSnapshotData(evtId) == false)
		{
			PX_ASSERT(false);
			return;
		}

		PxU32 instIndex;
		for(instIndex = 0; instIndex < mLoadedSnapshot.mInstanceCount; ++instIndex)
			if(mLoadedSnapshot.mSnapshotHeader[instIndex].mInstanceId == instId)
				break;

		DataRef<const PxU8> instData(loadSnapshotInstance(instIndex));
		mModel.readInstanceProperty(instData, propId);
	}

	void applySnapshot(PxU64 evtId)
	{
		if(!loadSnapshotData(evtId))
		{
			PX_ASSERT(false);
			return;
		}
		mModel.destroyAllInstances();
		for(PxU32 idx = 0; idx < mLoadedSnapshot.mInstanceCount; ++idx)
		{
			DataRef<const PxU8> instance(loadSnapshotInstance(idx));
			mModel.readInstance(instance);
		}
	}

	virtual void setPropertyValue(void* instId, PxI32 propId, const PxU8* data, PxU32 dataLen)
	{
		bool success = mModel.setPropertyValue(instId, propId, data, dataLen, -1);
		PX_UNUSED(success);
		PX_ASSERT(success);
	}

	virtual void setPropertyMessage(void* instId, PxI32 msgId, const PxU8* data, PxU32 dataLen)
	{
		bool success = mModel.setPropertyMessage(instId, msgId, data, dataLen);
		PX_UNUSED(success);
		PX_ASSERT(success);
	}

	virtual void insertArrayElement(void* instId, PxI32 propId, PxI32 propIdx, const PxU8* data, PxU32 dataLen)
	{
		bool success = mModel.insertArrayElement(instId, propId, propIdx, data, dataLen);
		PX_UNUSED(success);
		PX_ASSERT(success);
	}

	virtual void removeArrayElement(void* instId, PxI32 propId, PxI32 propIdx)
	{
		bool success = mModel.removeArrayElement(instId, propId, propIdx);
		PX_UNUSED(success);
		PX_ASSERT(success);
	}

	virtual void setArrayElementValue(void* instId, PxI32 propId, PxI32 propIdx, const PxU8* data, PxU32 dataLen)
	{
		bool success = mModel.setArrayElementValue(instId, propId, propIdx, data, dataLen, -1);
		PX_UNUSED(success);
		PX_ASSERT(success);
	}

	void setPropertyValueToDefault(PxI32 instanceId, PxI32 propId)
	{
		void* ptr = mModel.idToPtr(instanceId);
		mModel.setPropertyValueToDefault(ptr, propId);
	}

	void applySetPropertyValueToInstance(SetPropertyValue_& evt, PxI32 instId)
	{
		void* ptr = mModel.idToPtr(instId);
		bool success = mModel.setPropertyValue(ptr, evt.mPropertyId, evt.mData.begin(), evt.mData.size(), -1);
		PX_UNUSED(success);
		PX_ASSERT(success);
	}

	void applySetPropertyMessageToInstance(SetPropertyMessage_& evt, PxI32 propId)
	{
		void* ptr = mModel.idToPtr(evt.mInstanceId);
		PropertyMessageDescription msg(mModel.getMetaData().getPropertyMessage(evt.mMsgId));
		PxU32 numProps = msg.mProperties.size();
		for(PxU32 idx = 0; idx < numProps; ++idx)
		{
			const PropertyMessageEntry& entry(msg.mProperties[idx]);
			if(entry.mProperty.mPropertyId == propId)
			{
				const PxU8* dataStart = evt.mData.begin() + entry.mMessageOffset;
				const PxU8* dataEnd = dataStart + entry.mByteSize;
				if(dataEnd > evt.mData.end())
				{
					PX_ASSERT(false);
					return;
				}
				mModel.setPropertyValue(ptr, propId, dataStart, entry.mByteSize, entry.mDatatypeId);
			}
		}
	}

	void applySetPropertyMessageToInstance(SetPropertyMessage_& evt, PxI32 msgId, const PropertyMessageDescription& msg)
	{
		void* ptr = mModel.idToPtr(evt.mInstanceId);
		if(evt.mMsgId == msgId)
		{
			mModel.setPropertyMessage(ptr, msgId, evt.mData.begin(), evt.mData.size());
			return;
		}
		// Uggh.  Perform intersection of old properties with new properties, set all results.
		PropertyMessageDescription earlierMsg(mModel.getMetaData().getPropertyMessage(evt.mMsgId));
		PxU32 numLater(msg.mProperties.size());
		PxU32 numEarlier(earlierMsg.mProperties.size());
		for(PxU32 later = 0; later < numLater; ++later)
		{
			for(PxU32 earlier = 0; earlier < numEarlier; ++earlier)
			{
				const PropertyMessageEntry& entry(earlierMsg.mProperties[earlier]);
				if(msg.mProperties[later].mProperty.mPropertyId == entry.mProperty.mPropertyId)
				{
					const PxU8* dataStart = evt.mData.begin() + entry.mMessageOffset;
					const PxU8* dataEnd = dataStart + entry.mByteSize;
					if(dataEnd > evt.mData.end())
					{
						PX_ASSERT(false);
						return;
					}
					mModel.setPropertyValue(ptr, entry.mProperty.mPropertyId, dataStart, entry.mByteSize, entry.mDatatypeId);
				}
			}
		}
	}

	void applyInsertArrayElement(const InsertArrayElement& elem, PxI32 propIdx)
	{
		if(propIdx < 0)
			propIdx = elem.mPropIdx;
		void* ptr = mModel.idToPtr(elem.mInstanceId);
		mModel.insertArrayElement(ptr, elem.mPropId, propIdx, elem.mData.begin(), elem.mData.size());
	}

	void applySetArrayElementValue(const SetArrayElementValue& evt, PxI32 propIdx)
	{
		void* ptr = mModel.idToPtr(evt.mInstanceId);
		if(propIdx < 0)
			propIdx = evt.mPropIdx;
		mModel.setArrayElementValue(ptr, evt.mPropId, propIdx, evt.mData.begin(), evt.mData.size(), -1);
	}

	void applyOriginShiftToInstance(const OriginShiftEvent& evt, PxI32 realInst, PxI32 prop)
	{
		void* ptr = mModel.idToPtr(realInst);
		PxVec3 shift(static_cast<PxF32>(evt.mShiftVec_x), static_cast<PxF32>(evt.mShiftVec_y), static_cast<PxF32>(evt.mShiftVec_z));
		mModel.shiftObject(ptr, prop, shift);
	}

	struct SetPropertyValuePreviousEventOperator
	{
		PvdObjectModelEventStreamImpl& mImpl;
		PxI32 mInstanceId;
		PxU64 mEventId;
		PxI32 mPropId;
		PxI32 mPropIdx;
		bool mInsertChangedToSetElementValue;

		SetPropertyValuePreviousEventOperator(PvdObjectModelEventStreamImpl& impl, PxI32 instId, PxU64 previousEvent, PxI32 propId, PxI32 propIdx, bool insertChangedToSetElementValue)
		: mImpl(impl), mInstanceId(instId), mEventId(previousEvent), mPropId(propId), mPropIdx(propIdx), mInsertChangedToSetElementValue(insertChangedToSetElementValue)
		{
		}
		SetPropertyValuePreviousEventOperator& operator=(const SetPropertyValuePreviousEventOperator&);

		void handleUserEvent(UserEvent) { PX_ASSERT(false); }
		void handleDestroyInstance(DestroyInstance_) { PX_ASSERT(false); }
		void handleCreateInstance(CreateInstance_) { mImpl.setPropertyValueToDefault(mInstanceId, mPropId); }

		void handleSnapshot(Snapshot) { mImpl.applySnapshotToInstance(mInstanceId, mEventId, mPropId); }

		void handleSetPropertyMessage(SetPropertyMessage_ evt)
		{
			PX_ASSERT(evt.mInstanceId == mInstanceId);
			mImpl.applySetPropertyMessageToInstance(evt, mPropId);
		}

		void handleSetPropertyValue(SetPropertyValue_ evt)
		{
			PX_ASSERT(evt.mInstanceId == mInstanceId);
			mImpl.applySetPropertyValueToInstance(evt, mInstanceId);
		}

		void handleInsertArrayElement(InsertArrayElement evt)
		{
			if(mInsertChangedToSetElementValue)
			{
				SetArrayElementValue fakeEvt(evt.mInstanceId, evt.mPropId, evt.mPropIdx, evt.mData, DataRef<PxU32>());
				mImpl.applySetArrayElementValue(fakeEvt, mPropIdx);
			}
			else
				mImpl.applyInsertArrayElement(evt, mPropIdx);
		}

		void handleRemoveArrayElement(RemoveArrayElement) { PX_ASSERT(false); }

		void handleSetArrayElementValue(SetArrayElementValue evt) { mImpl.applySetArrayElementValue(evt, mPropIdx); }
		void handleSavedInstance(SavedInstance) {}
		void handleOriginShiftEvent(OriginShiftEvent) {}
	};

	// Apply this property of this previous event to this object
	// callback from setpropertyvalue events.
	virtual void applyPreviousEvent(PxI32 instanceId, PxI64 previousEvent, PxI32 inPropertyId, PxI32 propIdx, bool insertChangedToSetElementValue)
	{
		if(previousEvent < 0)
		{
			PX_ASSERT(false);
			return;
		}
		PxU64 eventIndex = static_cast<PxU64>(previousEvent);
		// Note that the previous event could be a snapshot.
		Option<LoadedPage> thePage = getPageForEvent(eventIndex);
		if(thePage.hasValue() == false)
		{
			PX_ASSERT(false);
			return;
		}
		SetPropertyValuePreviousEventOperator theOp(*this, instanceId, eventIndex, inPropertyId, propIdx, insertChangedToSetElementValue);
		thePage->operateOnPageEvent(eventIndex, theOp);
	}

	struct SetPropertyMessagePreviousEventOperator
	{
		PvdObjectModelEventStreamImpl& mImpl;
		PxI32 mInstanceId;
		PxU64 mEventId;
		PxI32 mMsgId;
		const PropertyMessageDescription& mMsg;

		SetPropertyMessagePreviousEventOperator(PvdObjectModelEventStreamImpl& impl, PxI32 instId, PxU64 previousEvent, PxI32 msgId, const PropertyMessageDescription& desc)
		: mImpl(impl), mInstanceId(instId), mEventId(previousEvent), mMsgId(msgId), mMsg(desc)
		{
		}

		SetPropertyMessagePreviousEventOperator& operator=(const SetPropertyMessagePreviousEventOperator&);

		void handleUserEvent(UserEvent) { PX_ASSERT(false); }
		void handleDestroyInstance(DestroyInstance_) { PX_ASSERT(false); }
		void handleCreateInstance(CreateInstance_)
		{
			PxU32 numProps = mMsg.mProperties.size();
			for(PxU32 idx = 0; idx < numProps; ++idx)
				mImpl.setPropertyValueToDefault(mInstanceId, mMsg.mProperties[idx].mProperty.mPropertyId);
		}

		void handleSnapshot(Snapshot)
		{
			PxU32 numProps = mMsg.mProperties.size();
			for(PxU32 idx = 0; idx < numProps; ++idx)
				mImpl.applySnapshotToInstance(mInstanceId, mEventId, mMsg.mProperties[idx].mProperty.mPropertyId);
		}

		void handleSetPropertyMessage(SetPropertyMessage_ evt)
		{
			PX_ASSERT(evt.mInstanceId == mInstanceId);
			mImpl.applySetPropertyMessageToInstance(evt, mMsgId, mMsg);
		}

		void handleSetPropertyValue(SetPropertyValue_ evt)
		{
			PX_ASSERT(evt.mInstanceId == mInstanceId);
			mImpl.applySetPropertyValueToInstance(evt, mInstanceId);
		}

		void handleInsertArrayElement(InsertArrayElement) { PX_ASSERT(false); }

		void handleSetArrayElementValue(SetArrayElementValue) { PX_ASSERT(false); }

		void handleRemoveArrayElement(RemoveArrayElement) { PX_ASSERT(false); }
		void handleSavedInstance(SavedInstance) {}

		void handleOriginShiftEvent(OriginShiftEvent evt)
		{
			DataRef<InstancePropertyPair> instProps = evt.mRelativePairs;

			PxU32 numProps = mMsg.mProperties.size();
			for(PxU32 idx = 0; idx < numProps; ++idx)
			{
				for(const InstancePropertyPair* iter = instProps.begin(); iter != instProps.end(); ++iter)
				{
					if(iter->first == mInstanceId && iter->second == mMsg.mProperties[idx].mProperty.mPropertyId)
					{
						mImpl.applyOriginShiftToInstance(evt, iter->first, iter->second);
					}
				}
			}
		}
	};

	virtual void applyPreviousEventFromMessage(PxI32 instanceId, PxI64 previousEvent, PxI32 inMsgId)
	{
		if(previousEvent < 0)
		{
			PX_ASSERT(false);
			return;
		}
		PxU64 eventIndex = static_cast<PxU64>(previousEvent);
		// Note that the previous event could be a snapshot.
		Option<LoadedPage> thePage = getPageForEvent(eventIndex);
		if(thePage.hasValue() == false)
		{
			PX_ASSERT(false);
			return;
		}
		PropertyMessageDescription desc(mModel.getMetaData().getPropertyMessage(inMsgId));
		SetPropertyMessagePreviousEventOperator theOp(*this, instanceId, eventIndex, inMsgId, desc);
		thePage->operateOnPageEvent(eventIndex, theOp);
	}
	// End PvdObjectModelEventMutator Implementation

	DataRef<const PxU8> getDeletedInstance(PxI32 instId)
	{
		const HashMap<PxI32, InstanceFilePosAndLength>::Entry* entry = mMetaData.mInstanceDestructionMap.find(instId);
		if(entry == NULL)
		{
			PX_ASSERT(false);
			return DataRef<const PxU8>();
		}
		InstanceFilePosAndLength item(entry->second);
		if(mDestroyedInstanceBuffer.size() < item.mLength)
			mDestroyedInstanceBuffer.resize(item.mLength * 2);
		Option<ErrorCode> retval = mStream.read(mDestroyedInstanceBuffer.begin(), item.mLength, item.mOffset);
		if(retval.hasValue())
		{
			PX_ASSERT(false);
			return DataRef<const PxU8>();
		}
		return DataRef<const PxU8>(mDestroyedInstanceBuffer.begin(), mDestroyedInstanceBuffer.begin() + item.mLength);
	}

	DataRef<const PxU8> getSavedInstance(PxU64 fileOffset)
	{
		mSnapshotInstanceLoader.clear();
		PxU64 realOffset = fileOffset + mStream.mOffset;
		IOStreamToInputStream stream(realOffset, mStream.mStream);
		PvdObjectModel::readInstance(stream, mSnapshotInstanceLoader);
		return mSnapshotInstanceLoader.toRef();
	}
};
} // namespace

PvdObjectModelEventStream& PvdObjectModelEventStream::load(PxAllocatorCallback& callback, PvdObjectModel& model, PvdRandomAccessIOStream& inStream, PxU64 initialOffset)
{
	PvdObjectModelEventStreamMetaData& newData = *PVD_NEW(PvdObjectModelEventStreamMetaData)(callback);
	IOStreamToInputStream wrapper(initialOffset, inStream);
	newData.load(wrapper);
	PX_ASSERT(newData.mEventCount > 0);
	PxU64 dataStart = toPaddedSize(wrapper.mOffset, 0x1000);
	PvdObjectModelEventStreamImpl& retval = *PVD_NEW(PvdObjectModelEventStreamImpl)(callback, model, newData, inStream, 256, dataStart);
	// Force model to initial state
	retval.applySnapshot(0);
	newData.mCurrentEvent = 0;
	return retval;
}

PvdObjectModelEventStream& PvdObjectModelEventStream::create(PxAllocatorCallback& callback, PvdObjectModelEventStreamMetaData& metaData, PvdObjectModel& model, PvdRandomAccessIOStream& stream,
                                                             PxU64 pageSectionStart)
{
	return *PVD_NEW(PvdObjectModelEventStreamImpl)(callback, model, metaData, stream, 256, pageSectionStart);
}
