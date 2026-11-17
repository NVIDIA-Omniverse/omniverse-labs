/*
 * SPDX-FileCopyrightText: Copyright (c) 2008-2014 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Copyright (c) 2008-2014 NVIDIA Corporation. All rights reserved.
#include "PvdCommStreamMetaData.h"
#include "PvdObjectModelEventStream.h"
#include "PxMat44.h"
#include "PxPvdByteStreams.h"
#include "PxPvdCommStreamTypes.h"
#include "PxPvdFoundation.h"
#include "PxPvdObjectModelInternalTypes.h"
#include "PxVec2.h"
#include "PxVec4.h"
#include <algorithm>
#include <vector>


using namespace pvdapp;
using namespace shdfnd;

namespace physx
{
namespace pvdsdk
{
DEFINE_PVD_TYPE_NAME_MAP(CameraProps, "", "Camera");
DEFINE_PVD_TYPE_NAME_MAP(SectionEventData, "_debugger_", "SectionEventData");
} // namespace pvdsdk
} // namespace physx

namespace
{

struct InstanceMetaFlagTypes
{
	enum Enum
	{
		HasPickable = 1,
		HasColor = 1 << 1,
		HasTopLevel = 1 << 2,
	};
};

typedef PxFlags<InstanceMetaFlagTypes::Enum, PxU32> InstanceMetaFlags;

struct InstanceMetaData
{
	PxU64 mStreamInstanceId;
	ObjectRef mObject;
	InstanceMetaFlags mFlags;
	PvdColor mColor;
	bool mPickable;
	bool mIsTopLevel;
	InstanceMetaData(PxU64 streamId) : mStreamInstanceId(streamId), mFlags(0), mPickable(false), mIsTopLevel(false) {}

	InstanceMetaData() {}

	template <typename TSerializer>
	void serialize(TSerializer& s)
	{
		s.streamify(mStreamInstanceId);
		s.streamify(mObject);
		s.streamify(mFlags);
		s.streamify(mColor);
		s.streamify(mPickable);
		s.streamify(mIsTopLevel);
	}
};

struct ProfileZoneImpl : public PvdProfileZone
{
	Array<PvdProfileZoneEvent> mEventArray;

	ProfileZoneImpl() : mEventArray("ProfileZoneImpl::mEventArray") {}
	void addEvent(const PvdProfileZoneEvent& evt)
	{
		mEventArray.pushBack(evt);
		mEvents = DataRef<PvdProfileZoneEvent>(mEventArray.begin(), mEventArray.end());
	}
	template <typename TSerializer>
	void serialize(TSerializer& s)
	{
		s.streamify(mStreamId);
		s.streamify(mName);
		s.streamify(mInstanceId);
		s.streamify(mEventArray);
		mEvents = DataRef<PvdProfileZoneEvent>(mEventArray.begin(), mEventArray.end());
	}
};

template <typename TSerializer>
void serialize(TSerializer& s, SectionListInfo& evt)
{
	s.streamify(evt.mStreamId);
	s.streamify(evt.mName);
}

template <typename TSerializer>
void serialize(TSerializer& s, SectionBracket& value)
{
	s.streamify(value.mTimestamp);
	s.streamify(value.mEventIndex);
}

template <typename TSerializer>
void serialize(TSerializer& s, Section& value)
{
	serialize(s, value.mBegin);
	serialize(s, value.mEnd);
}

template <typename TSerializer>
void serialize(TSerializer& s, InstanceMetaData& evt)
{
	evt.serialize(s);
}

template <typename TSerializer>
void serialize(TSerializer& s, PvdProfileZoneEvent& evt)
{
	s.streamify(evt.mName);
	s.streamify(evt.mEventId);
	s.streamify(evt.mIsCompiledTimeEnabled);
}

template <typename TSerializer>
void serialize(TSerializer& s, PvdMemoryEventEntry& evt)
{
	s.streamify(evt.mType);
	s.streamify(evt.mFile);
	s.streamify(evt.mLine);
	s.streamify(evt.mTotalsArrayIndex);
}

struct InputSerializer
{
	PxAllocatorCallback& mAllocator;
	PvdInputStream& mStream;
	Pool<InstanceMetaData>& mPool;
	StringTable& mStringTable;
	Array<PxU8> mTempBuf;

	InputSerializer(PxAllocatorCallback& alloc, PvdInputStream& str, Pool<InstanceMetaData>& pool, StringTable& strTable)
	: mAllocator(alloc), mStream(str), mPool(pool), mStringTable(strTable), mTempBuf("InputSerializer::tempBuf")
	{
	}
	void streamify(PxU64& data) { mStream >> data; }
	void streamify(PxU32& data) { mStream >> data; }
	void streamify(PxI32& data) { mStream >> data; }
	void streamify(PxU16& data) { mStream >> data; }
	void streamify(bool& data)
	{
		PxU8 temp;
		mStream >> temp;
		data = temp ? true : false;
	}
	void streamify(String& data)
	{
		PxU32 len;
		mStream >> len;
		mTempBuf.resize(len);
		if(len)
		{
			mStream.read(mTempBuf.begin(), len);
			data = mStringTable.registerStr((const char*)mTempBuf.begin());
		}
		else
			data = "";
	}
	void streamify(ObjectRef& data) { mStream >> data; }
	void streamify(InstanceMetaFlags& flags) { mStream >> flags; }
	void streamify(PvdColor& color) { mStream >> color; }
	void streamify(InstanceMetaData*& instPtr)
	{
		instPtr = mPool.construct();
		instPtr->serialize(*this);
	}

	void streamify(ProfileZoneImpl*& implPtr)
	{
		implPtr = PVD_NEW(ProfileZoneImpl)();
		implPtr->serialize(*this);
	}

	template <typename TDType>
	void streamify(Array<TDType>& events)
	{
		PxU32 count;
		mStream >> count;
		events.resize(count);
		PVD_FOREACH(idx, count)
		serialize(*this, events[idx]);
	}
	template <typename TDType>
	void streamify(Array<TDType>*& events)
	{
		events = PVD_NEW(Array<TDType>)("Array");
		streamify(*events);
	}
	void streamify(SectionListInfo& info) { serialize(*this, info); }
	template <typename TKeyType, typename TValType, typename THashType>
	void streamify(HashMap<TKeyType, TValType, THashType>& map)
	{
		PxU32 count;
		mStream >> count;
		PVD_FOREACH(idx, count)
		{
			TKeyType key;
			TValType val;
			streamify(key);
			streamify(val);
			map.insert(key, val);
		}
	}
};

struct OutputSerializer
{
	PvdOutputStream& mStream;
	OutputSerializer(PvdOutputStream& str) : mStream(str) {}
	void streamify(PxU64& data) { mStream << data; }
	void streamify(PxU32& data) { mStream << data; }
	void streamify(PxU16& data) { mStream << data; }
	void streamify(PxI32& data) { mStream << data; }
	void streamify(ObjectRef& data) { mStream << data; }
	void streamify(InstanceMetaFlags& flags) { mStream << flags; }
	void streamify(PvdColor& color) { mStream << color; }
	void streamify(bool& pick)
	{
		PxU8 val = pick ? 1 : 0;
		mStream << val;
	}
	void streamify(String& str)
	{
		PxU32 len = safeStrLen(str);
		if(len)
		{
			++len;
			mStream << len;
			mStream.write(str, len);
		}
		else
			mStream << len;
	}
	void streamify(InstanceMetaData*& instPtr) { instPtr->serialize(*this); }
	void streamify(ProfileZoneImpl*& profPtr) { profPtr->serialize(*this); }
	template <typename TDType>
	void streamify(Array<TDType>& events)
	{
		PxU32 count = static_cast<PxU32>(events.size());
		mStream << count;
		PVD_FOREACH(idx, count)
		serialize(*this, events[idx]);
	}
	template <typename TDType>
	void streamify(Array<TDType>* events)
	{
		streamify(*events);
	}

	void streamify(SectionListInfo& info) { serialize(*this, info); }

	template <typename TKeyType, typename TValType, typename THashType>
	void streamify(HashMap<TKeyType, TValType, THashType>& map)
	{
		PxU32 count = static_cast<PxU32>(map.size());
		mStream << count;
		for(typename HashMap<TKeyType, TValType, THashType>::Iterator iter = map.getIterator(); iter.done() == false; ++iter)
		{
			streamify(const_cast<TKeyType&>(iter->first));
			streamify(const_cast<TValType&>(iter->second));
		}
	}
};

struct SectionListHasher
{
	PxU32 operator()(const SectionListInfo& list) { return Hash<PxU64>()(list.mStreamId) ^ Hash<PxU32>()(list.mName); }

	bool equal(const SectionListInfo& lhs, const SectionListInfo& rhs) { return lhs.mStreamId == rhs.mStreamId && lhs.mName == rhs.mName; }
};

class StreamMetaData : public PvdCommStreamMetaData, public PvdCommStreamMetaDataMutator, public PvdObjectModelEventStreamUserEventTransfer
{
	typedef HashMap<SectionListInfo, Array<Section>*, SectionListHasher> TSectionMapType;

	// Note that when adding any new member variables you need to take note
	// and make sure that cloneForTransfer and clone both still fullfill
	// their intents.
	PxAllocatorCallback& mAllocator;
	PvdObjectModelMetaData& mMetaData;
	PvdObjectModelReader& mModelReader;
	PvdObjectModelMutator* mModelMutator;
	PvdObjectModelEventStreamUserEventListener* mUserEventListener;
	PxU32 mRefCount;

	PxU64 mNumerator;
	PxU64 mDenominator;
	HashMap<PxU64, InstanceMetaData*> mStreamIdToInstance;
	HashMap<PxI32, InstanceMetaData*> mInstanceIdToInstance;
	Pool<InstanceMetaData> mInstancePool;
	HashMap<PxU32, PxI32> mCameraMap; // string handle to camera.
	HashMap<PxU64, ProfileZoneImpl*> mProfileZoneMap;
	Array<PvdMemoryEventEntry> mMemoryEventEntries;
	TSectionMapType mSectionMap;
	Array<PxU64> mStreamIds;
	bool mRecording;
	PxI32 mMemoryEventsInstance;
	PxI32 mMemoryTotalsInstance;
	PxI32 mUserCustomedEvents;

  public:
	static PxU32 getStaticMetaVersion() { return 1; }

	// Constructor for when creation of new meta data is disallowed.
	StreamMetaData(PxAllocatorCallback& alloc, PvdObjectModelMetaData& meta, PvdObjectModelReader& modelReader, PvdObjectModelMutator* mutator, PvdObjectModelEventStreamUserEventListener* factory)
	: mAllocator(alloc)
	, mMetaData(meta)
	, mModelReader(modelReader)
	, mModelMutator(mutator)
	, mUserEventListener(factory)
	, mRefCount(0)
	, mStreamIdToInstance("PvdCommStreamMetaDataData::mStreamIdToInstance")
	, mInstanceIdToInstance("PvdCommStreamMetaDataData::mInstanceToStreamId")
	, mInstancePool("PvdCommStreamMetaDataData::mInstancePool")
	, mCameraMap("PvdCommStreamMetaDataData::mCameraMap")
	, mProfileZoneMap("PvdCommStreamMetaDataData::mProfileZoneMap")
	, mMemoryEventEntries("PvdCommStreamMetaDataData::mMemoryEventEntries")
	, mSectionMap("PvdCommStreamMetaDataData::mSectionMap")
	, mRecording(true)
	, mMemoryEventsInstance(-1)
	, mMemoryTotalsInstance(-1)
	, mUserCustomedEvents(-1)
	{
	}

	PX_FORCE_INLINE NamespacedName getUserCustomedEventName() { return NamespacedName("_debugger_", "UserCustomedEvent"); }

	void clearSections()
	{
		for(TSectionMapType::Iterator iter = mSectionMap.getIterator(); iter.done() == false; ++iter)
		{
			PVD_DELETE(iter->second);
		}
		mSectionMap.clear();
	}

	virtual ~StreamMetaData()
	{
		clearSections();

		for(HashMap<PxU64, ProfileZoneImpl*>::Iterator iter = mProfileZoneMap.getIterator(); iter.done() == false; ++iter)
			PVD_DELETE(iter->second);
		mProfileZoneMap.clear();
	}

	virtual void addRef() { ++mRefCount; }
	virtual void release()
	{
		if(mRefCount)
			--mRefCount;
		if(!mRefCount)
			PVD_DELETE(this);
	}

	virtual void getTimestampNanosecondConversion(PxU64& numerator, PxU64& denominator)
	{
		numerator = mNumerator;
		denominator = mDenominator;
	}

	virtual void setTimestampNanosecondConversion(PxU64 numerator, PxU64 denominator)
	{
		mNumerator = numerator;
		mDenominator = denominator;
	}

	virtual PxU32 getNbInstances() { return static_cast<PxU32>(mInstanceIdToInstance.size()); }

	virtual PxU32 getInstances(PxI32* instanceBuf, PxU32 bufSize, PxU32 startIdx) { return getMapKeys(mInstanceIdToInstance, instanceBuf, bufSize, startIdx); }

	InstanceMetaData* getInstanceMetaData(PxI32 instance)
	{
		const HashMap<PxI32, InstanceMetaData*>::Entry* entry(mInstanceIdToInstance.find(instance));
		if(entry)
			return entry->second;
		return nullptr;
	}

	virtual PxU64 getStreamId(PxI32 instance)
	{
		InstanceMetaData* meta = getInstanceMetaData(instance);
		if(meta)
			return meta->mStreamInstanceId;
		return 0;
	}

	virtual PxI32 getInstanceId(PxU64 streamId)
	{
		const HashMap<PxU64, InstanceMetaData*>::Entry* entry(mStreamIdToInstance.find(streamId));
		if(entry)
			return entry->second->mObject;
		return -1;
	}

	virtual PxI32 reverseMap(PxU64 streamId)
	{
		for(HashMap<PxI32, InstanceMetaData*>::Iterator iter = mInstanceIdToInstance.getIterator(); iter.done() == false; ++iter)
		{
			const InstanceMetaData* inst = iter->second;
			if(inst->mStreamInstanceId == streamId)
				return inst->mObject.mInstanceId;
		}
		return -1;
	}

	virtual bool isPickable(PxI32 instance)
	{
		InstanceMetaData* meta(getInstanceMetaData(instance));
		if(meta && (meta->mFlags & InstanceMetaFlagTypes::HasPickable))
			return meta->mPickable;
		// By default, things are pickable.
		return true;
	}
	virtual Option<PvdColor> getColor(PxI32 instance)
	{
		InstanceMetaData* meta(getInstanceMetaData(instance));
		if(meta && (meta->mFlags & InstanceMetaFlagTypes::HasColor))
			return meta->mColor;
		return None();
	}

	virtual bool isTopLevel(PxI32 instance)
	{
		InstanceMetaData* meta(getInstanceMetaData(instance));
		if(meta && (meta->mFlags & InstanceMetaFlagTypes::HasTopLevel))
			return meta->mIsTopLevel;
		// By default, things are not top level.
		return false;
	}

	virtual PxU32 getNbCameras() { return static_cast<PxU32>(mCameraMap.size()); }
	virtual PxU32 getCameras(StringHandle* buffer, PxU32 bufSize, PxU32 startIdx)
	{
		PxU32* realbuf = reinterpret_cast<PxU32*>(buffer);
		return getMapKeys(mCameraMap, realbuf, bufSize, startIdx);
	}

	virtual Option<CameraProps> getCameraData(StringHandle name)
	{
		PxU32 handle(name);
		const HashMap<PxU32, PxI32>::Entry* entry(mCameraMap.find(handle));
		if(entry == NULL)
			return None();
		InstanceDescription desc(mModelReader.idToDescriptor(entry->second));
		if(desc.mAlive == false)
		{
			return None();
		}
		CameraProps retval;
		mModelReader.getPropertyMessage(desc, retval);
		return retval;
	}

	virtual PxU32 getNbSectionLists() { return mSectionMap.size(); }

	virtual PxU32 getSectionLists(SectionListInfo* sectionBuffer, PxU32 bufSize, PxU32 idx) { return getMapKeys(mSectionMap, sectionBuffer, bufSize, idx); }

	Array<Section>* getSectionList(SectionListInfo list)
	{
		const TSectionMapType::Entry* listEntry(mSectionMap.find(list));
		if(listEntry)
			return const_cast<Array<Section>*>(listEntry->second);
		return nullptr;
	}

	virtual PxU32 getNbSections(const SectionListInfo& list)
	{
		Array<Section>* sectionData = getSectionList(list);
		if(sectionData)
			return sectionData->size();
		return 0;
	}

	virtual PxU32 getSections(const SectionListInfo& list, Section* buffer, PxU32 bufSize, PxU32 idx = 0)
	{
		Array<Section>* sectionData = getSectionList(list);
		if(sectionData)
			return getArrayEntries(*sectionData, buffer, bufSize, idx);
		return 0;
	}

	// The section with the earliest timestamp becomes the default.
	// This has the effect of making the outermost section the default
	// section in a situation where sections are nested (like
	// Apex::Simulate logically wrapping PhysX::Simulate
	virtual Option<SectionListInfo> getDefaultFrameMarker()
	{
		Option<SectionListInfo> retval;
		PxU64 minTimestamp = static_cast<PxU64>(-1);
		const PxU32 sectionListCount = getNbSectionLists();
		// qDebug() << "Total section list count: " << sectionListCount;
		std::vector<SectionListInfo> sectionListBuffer(sectionListCount);
		PxU32 r = getSectionLists(sectionListBuffer.data(), sectionListCount, 0);
		PX_ASSERT(r == sectionListCount);

		for(TSectionMapType::Iterator iter = mSectionMap.getIterator(); iter.done() == false; ++iter)
		{

			// qDebug() << "Name: " << iter->first.mName << " StreamID: " << iter->first.mStreamId << " Iter size: " << iter->second->size() << endl;
			if(iter->second->size())
			{
				Section val = (*iter->second)[0];
				if(val.mBegin.mTimestamp < minTimestamp)
				{
					retval = Option<SectionListInfo>(iter->first);
					minTimestamp = val.mBegin.mTimestamp;
				}
			}
		}
		return retval;
	}

	virtual Option<SectionListInfo> getSectionListInfoForScene(const PxU64 sceneId)
	{
		Option<SectionListInfo> retval;
		// PxU32 sectionListCount = getNbSectionLists();
		// qDebug() << "Total section list count: " << sectionListCount;
		// std::vector<SectionListInfo> sectionListBuffer(sectionListCount);
		// PxU32 r = getSectionLists(sectionListBuffer.data(), sectionListCount, 0);
		// PX_ASSERT(r == sectionListCount);

		for(TSectionMapType::Iterator iter = mSectionMap.getIterator(); iter.done() == false; ++iter)
		{
			// qDebug() << "Name: " << iter->first.mName << " StreamID: " << iter->first.mStreamId << " Iter size: " << iter->second->size() << endl;
			if(iter->first.mStreamId == sceneId)
			{
				return Option<SectionListInfo>(iter->first);
			}
		}
		return retval;
	}

	// Section indexes are 1 based, not 0 based!!!!
	// So list.sections[sectionCount] is a valid operation!!!
	virtual Option<Section> getSectionInfo(const SectionListInfo& list, PxU32 currentSectionIndex)
	{
		if(currentSectionIndex == 0)
			return None();
		--currentSectionIndex;
		Array<Section>* sectionData = getSectionList(list);
		if(sectionData && currentSectionIndex < sectionData->size())
			return (*sectionData)[currentSectionIndex];
		return None();
	}

	// Section indexes are 1 based, but evId 0 *always* corresponds to idx 0.
	// Event zero corresponds to a 'no index' return.
	// Event EventCount corresponds to SectionCount return.
	virtual PxU32 eventToSectionIndex(const SectionListInfo& list, PxU64 evId)
	{
		if(evId == 0)
			return 0;
		Array<Section>* sectionData = getSectionList(list);
		if(sectionData)
		{
			if(sectionData->size() == 0)
				return 0;

			Section testData(SectionBracket(0, evId));
			testData.mEnd = SectionBracket(0, evId);

			auto sectionEndComparator = [](const Section& lhs, const Section& rhs) -> bool {
				PX_ASSERT(lhs.mEnd.isEventValid() || rhs.mEnd.isEventValid());
				return lhs.mEnd.mEventIndex < rhs.mEnd.mEventIndex;
			};
			Section* lowerBound = std::lower_bound(sectionData->begin(), sectionData->end(), testData, sectionEndComparator);
			if(lowerBound == sectionData->end())
				return sectionData->size();
			return static_cast<PxU32>(lowerBound - sectionData->begin()) + 1;
		}
		return 0;
	}

	virtual PxU32 getNbProfileZones() { return mProfileZoneMap.size(); }
	virtual PxU32 getProfileZones(PvdProfileZone* buffer, PxU32 bufSize, PxU32 idx = 0) { return getMapValues(mProfileZoneMap, buffer, bufSize, idx, DerefOp()); }

	virtual PxU32 getNbMemoryEventEntries() { return mMemoryEventEntries.size(); }
	virtual PxU32 getMemoryEventEntries(PvdMemoryEventEntry* buffer, PxU32 bufSize, PxU32 idx) { return getArrayEntries(mMemoryEventEntries, buffer, bufSize, idx); }
	virtual PxI32 getMemoryEventBufferInstance() { return mMemoryEventsInstance; }
	virtual PxI32 getMemoryTotalsInstance() { return mMemoryTotalsInstance; }

	virtual PvdCommStreamMetaData& getMetaData() { return *this; }
	virtual PvdObjectModelEventStreamUserEventTransfer& getEventTransfer() { return *this; }

	InstanceMetaData& getOrCreateInstanceMetaData(PxU64 streamId)
	{
		const HashMap<PxU64, InstanceMetaData*>::Entry* entry(mStreamIdToInstance.find(streamId));
		if(entry)
			return *entry->second;

		InstanceMetaData* newInst(mInstancePool.construct(streamId));
		PX_ASSERT(mStreamIdToInstance.find(streamId) == NULL);
		mStreamIdToInstance[streamId] = newInst;
		return *newInst;
	}

	InstanceMetaData& createInstanceMetaDataFromInstanceId(PxI32 iid)
	{
		const HashMap<PxI32, InstanceMetaData*>::Entry* entry(mInstanceIdToInstance.find(iid));
		if(entry)
		{
			PX_ASSERT(false);
			return *entry->second;
		}
		PxU64 sid = 0;
		InstanceMetaData* newInst(mInstancePool.construct(sid));
		newInst->mObject.mInstanceId = iid;
		mInstanceIdToInstance[iid] = newInst;
		return *newInst;
	}

	virtual void setPickable(PxU64 streamId, bool pickable)
	{
		InstanceMetaData& meta = getOrCreateInstanceMetaData(streamId);
		meta.mFlags |= InstanceMetaFlagTypes::HasPickable;
		meta.mPickable = pickable;
	}
	virtual void setColor(PxU64 streamId, const PvdColor& color)
	{
		InstanceMetaData& meta = getOrCreateInstanceMetaData(streamId);
		meta.mFlags |= InstanceMetaFlagTypes::HasColor;
		meta.mColor = color;
	}
	virtual void setIsTopLevel(PxU64 streamId, bool isTopLevel)
	{
		InstanceMetaData& meta = getOrCreateInstanceMetaData(streamId);
		meta.mFlags |= InstanceMetaFlagTypes::HasTopLevel;
		meta.mIsTopLevel = isTopLevel;
	}
	virtual void setInstanceId(PxU64 streamId, PxI32 instance)
	{
		InstanceMetaData& meta = getOrCreateInstanceMetaData(streamId);
		meta.mObject = ObjectRef(instance);
		mInstanceIdToInstance.insert(instance, &meta);
		if(mProfileZoneMap.find(streamId))
			getOrCreateProfileZone(streamId).mInstanceId = instance;
	}
	virtual void destroyInstance(PxU64 streamId)
	{
		const HashMap<PxU64, InstanceMetaData*>::Entry* entry(mStreamIdToInstance.find(streamId));
		if(entry)
			mStreamIdToInstance.erase(streamId);
	}
	virtual void setCamera(StringHandle name, const PxVec3& pos, const PxVec3& up, const PxVec3& target)
	{
		if(mModelMutator == NULL)
		{
			PX_ASSERT(false);
			return;
		}
		const HashMap<PxU32, PxI32>::Entry* entry = mCameraMap.find(name.mHandle);
		PxI32 instanceId;
		if(entry == NULL)
		{
			InstanceDescription cameraDesc = mModelMutator->createInstance(getPvdNamespacedNameForType<CameraProps>());
			mModelMutator->pinInstance(cameraDesc);
			instanceId = cameraDesc.mId;
			mCameraMap.insert(name.mHandle, instanceId);
			mModelMutator->setPropertyValue(cameraDesc, "Name", name);
		}
		else
			instanceId = entry->second;
		CameraProps props(pos, up, target);
		mModelMutator->setPropertyMessage(mModelMutator->idToDescriptor(instanceId), props);
	}

	virtual void sendErrorMessage(PxU32 code, StringHandle message, StringHandle file, PxU32 name)
	{
		if(mUserCustomedEvents == -1)
		{
			InstanceDescription customedEventDesc = mModelMutator->createInstance(getUserCustomedEventName());
			mModelMutator->pinInstance(customedEventDesc);
			mUserCustomedEvents = customedEventDesc.mId;
		}

		PxU32 arrayLen = mModelReader.getNbArrayElements(mModelMutator->idToDescriptor(mUserCustomedEvents), "Events");
		PvdErrorMessage errorEvent(code, message, file, name, arrayLen);
		mModelMutator->pushBackArrayElementIf(mModelMutator->idToDescriptor(mUserCustomedEvents), "Events", errorEvent);
	}

	// odd event streams are dealt with as follows.
	//
	// begin,begin - The previous section's end event will equal ts, and its eventId will be one less.
	//
	// end,end - The current event's begin ts will be the previous event's end ts, and its
	// begin eventId will be one more than the previous events end event id.
	inline void addSectionEvent(PxU64 streamId, StringHandle name, bool isBegin, PxU64 timestamp, PxU64 eventIndex)
	{
		if(mRecording)
		{
			Array<Section>* sectionData = getSectionList(SectionListInfo(streamId, name));
			if(sectionData == NULL)
			{
				sectionData = PVD_NEW(Array<Section>)("Array");
				mSectionMap.insert(SectionListInfo(streamId, name), sectionData);
				mStreamIds.pushBack(streamId);
			}
			PX_ASSERT(sectionData);
			if(isBegin)
			{
				sectionData->pushBack(Section(SectionBracket(timestamp, eventIndex)));
				if(sectionData->size() > 1)
				{
					Section& previous = (*sectionData)[sectionData->size() - 2];
					// Section& current = sectionData->back();
					// some minor error correction.
					// correct for begin,begin data stream
					if(previous.mEnd.isEventValid() == false)
					{
						previous.mEnd.mTimestamp = timestamp;
						previous.mEnd.mEventIndex = eventIndex - 1;
					}
				}
			}
			if(!isBegin)
			{
				// correct for two conditions.
				// end,end event stream
				// and an event stream beginning with an end event.
				if(sectionData->size() == 0 || sectionData->back().mEnd.isEventValid())
				{
					sectionData->pushBack(Section());
					sectionData->back().mEnd = SectionBracket(timestamp, eventIndex);
					if(sectionData->size() > 1)
					{
						Section& previous = (*sectionData)[sectionData->size() - 2];
						Section& current = sectionData->back();
						current.mBegin = previous.mEnd;
						current.mBegin.mEventIndex += 1;
					}
					else
						sectionData->back().mBegin = SectionBracket(timestamp, eventIndex);
				}
				else
				{
					sectionData->back().mEnd = SectionBracket(timestamp, eventIndex);
					if(mModelMutator)
						mModelMutator->recordCompletedInstances();
				}
			}
		}
	}
	virtual void addSectionEvent(PxU64 streamId, StringHandle name, bool isBegin, PxU64 timestamp)
	{
		if(mRecording)
		{
			// qDebug() << "StreamId: " << streamId << " name: " << name << " isBegin: " << isBegin << "Time: " << timestamp << endl;
			SectionEventData evtData(streamId, name, timestamp, isBegin);
			PxU64 eventIndex = mUserEventListener->addUserEvent(evtData);
			addSectionEvent(streamId, name, isBegin, timestamp, eventIndex);
		}
	}

	ProfileZoneImpl& getOrCreateProfileZone(PxU64 sid)
	{
		const HashMap<PxU64, ProfileZoneImpl*>::Entry* entry = mProfileZoneMap.find(sid);
		if(entry)
			return *entry->second;
		ProfileZoneImpl* newImpl = PVD_NEW(ProfileZoneImpl)();
		mProfileZoneMap.insert(sid, newImpl);
		PxI32 item = getInstanceId(sid);
		if(item >= 0)
			newImpl->mInstanceId = item;
		return *newImpl;
	}

	virtual void addProfileZone(PxU64 sid, String nm) { getOrCreateProfileZone(sid).mName = mMetaData.getStringTable().registerStr(nonNull(nm)); }

	void addProfileZone(PxU64 sid, String nm, PxI32 instanceId)
	{
		ProfileZoneImpl& zone(getOrCreateProfileZone(sid));
		zone.mName = mMetaData.getStringTable().registerStr(nm);
		zone.mInstanceId = instanceId;
	}
	virtual void addProfileZoneEvent(PxU64 sid, String nm, PxU16 evId, bool enabled)
	{
		getOrCreateProfileZone(sid).addEvent(PvdProfileZoneEvent(mMetaData.getStringTable().registerStr(nonNull(nm)), evId, enabled));
	}

	virtual void onRecordingStopped()
	{
		if(mRecording)
		{
			clearSections();
			// Cleanup deleted instances that we can no longer reach.
			Array<InstanceMetaData*> itemsToRemove("PvdCommStreamMetaData::onRecordingStopped::itemsToRemove");
			for(HashMap<PxI32, InstanceMetaData*>::Iterator iter = mInstanceIdToInstance.getIterator(); iter.done() == false; ++iter)
			{
				if(mStreamIdToInstance.find(iter->second->mStreamInstanceId) == NULL)
					itemsToRemove.pushBack(iter->second);
			}
			PVD_FOREACH(idx, itemsToRemove.size())
			{
				mInstanceIdToInstance.erase(itemsToRemove[idx]->mObject.mInstanceId);
				mStreamIdToInstance.erase(itemsToRemove[idx]->mStreamInstanceId);
				mInstancePool.destroy(itemsToRemove[idx]);
			}
			mRecording = false;
		}
	}

	virtual void onRecordingStarted() { mRecording = true; }

	virtual void addMemoryEventEntry(String type, String file, PxU32 line, PxU32 totalsArrayIndex)
	{
		// PxU32 nbEntries = getNbMemoryEventEntries();
		StringTable& table(mMetaData.getStringTable());
		mMemoryEventEntries.pushBack(PvdMemoryEventEntry(table.registerStr(type), table.registerStr(file), line, totalsArrayIndex));
	}
	// events on the event buffer instance, 'totals' on the totals instance
	virtual void setMemoryEventInstances(PxI32 eventBuffer, PxI32 totals)
	{
		mMemoryEventsInstance = eventBuffer;
		mMemoryTotalsInstance = totals;
	}

	void initializeMetaData()
	{
//***********************************compatible: add the Px types in old than physx 3.4 version
#define CREATE_COMPATIBLE_BASIC_PVD_CLASS(type)                                                                                                                                                        \
	{                                                                                                                                                                                                  \
		NamespacedName pvdName("physx3", #type);                                                                                                                                                       \
		int id = mMetaData.getOrCreateClass(pvdName).mClassId;                                                                                                                                         \
		ClassDescription* cls = mMetaData.getClassPtr(id);                                                                                                                                             \
		cls->get32BitSize() = sizeof(type);                                                                                                                                                            \
		cls->get32BitSizeInfo().mAlignment = sizeof(type);                                                                                                                                             \
		cls->get64BitSize() = sizeof(type);                                                                                                                                                            \
		cls->get64BitSizeInfo().mAlignment = sizeof(type);                                                                                                                                             \
		cls->mLocked = true;                                                                                                                                                                           \
		cls->mPackedUniformWidth = sizeof(type);                                                                                                                                                       \
		cls->mPackedClassType = getPvdTypeForType<type>();                                                                                                                                             \
	}

		CREATE_COMPATIBLE_BASIC_PVD_CLASS(PxU8)
		CREATE_COMPATIBLE_BASIC_PVD_CLASS(PxI8)
		CREATE_COMPATIBLE_BASIC_PVD_CLASS(PxU16)
		CREATE_COMPATIBLE_BASIC_PVD_CLASS(PxI16)
		CREATE_COMPATIBLE_BASIC_PVD_CLASS(PxU32)
		CREATE_COMPATIBLE_BASIC_PVD_CLASS(PxI32)
		CREATE_COMPATIBLE_BASIC_PVD_CLASS(PxU64)
		CREATE_COMPATIBLE_BASIC_PVD_CLASS(PxI64)
		CREATE_COMPATIBLE_BASIC_PVD_CLASS(PxF32)
		CREATE_COMPATIBLE_BASIC_PVD_CLASS(PxF64)
#undef CREATE_COMPATIBLE_BASIC_PVD_CLASS

		int32_t fltClassType = getPvdTypeForType<float>();
		int32_t v3ClassType = getPvdTypeForType<PxVec3>();
		int32_t qtClassType = getPvdTypeForType<PxQuat>();
		{
			NamespacedName pvdName("physx3", "PxVec2");
			if(!mMetaData.findClass(pvdName).hasValue())
			{
				int classId = mMetaData.getOrCreateClass(pvdName).mClassId;
				ClassDescription* cls = mMetaData.getClassPtr(classId);
				mMetaData.createProperty(classId, "x", "", fltClassType);
				mMetaData.createProperty(classId, "y", "", fltClassType);
				PX_ASSERT(cls->get32BitSizeInfo().mAlignment == 1);
				PX_ASSERT(cls->get32BitSize() == 4);
				PX_ASSERT(cls->get64BitSizeInfo().mAlignment == 1);
				PX_ASSERT(cls->get64BitSize() == 4);
				PX_ASSERT(cls->mPackedUniformWidth == 1);
				PX_ASSERT(cls->mPackedClassType == getPvdTypeForType<uint8_t>());
				cls->mLocked = true;
			}
		}
		{
			NamespacedName pvdName("physx3", "PxVec3");
			if(!mMetaData.findClass(pvdName).hasValue())
			{
				int classId = mMetaData.getOrCreateClass(pvdName).mClassId;
				ClassDescription* cls = mMetaData.getClassPtr(classId);
				mMetaData.createProperty(classId, "x", "", fltClassType);
				mMetaData.createProperty(classId, "y", "", fltClassType);
				mMetaData.createProperty(classId, "z", "", fltClassType);
				PX_ASSERT(cls->get32BitSizeInfo().mAlignment == 4);
				PX_ASSERT(cls->get32BitSize() == 12);
				PX_ASSERT(cls->get64BitSizeInfo().mAlignment == 4);
				PX_ASSERT(cls->get64BitSize() == 12);
				PX_ASSERT(cls->mPackedUniformWidth == 4);
				PX_ASSERT(cls->mPackedClassType == fltClassType);
				cls->mLocked = true;
			}
		}
		{
			NamespacedName pvdName("physx3", "PxVec4");
			if(!mMetaData.findClass(pvdName).hasValue())
			{
				int classId = mMetaData.getOrCreateClass(pvdName).mClassId;
				ClassDescription* cls = mMetaData.getClassPtr(classId);
				mMetaData.createProperty(classId, "x", "", fltClassType);
				mMetaData.createProperty(classId, "y", "", fltClassType);
				mMetaData.createProperty(classId, "z", "", fltClassType);
				mMetaData.createProperty(classId, "w", "", fltClassType);
				PX_ASSERT(cls->get32BitSizeInfo().mAlignment == 4);
				PX_ASSERT(cls->get32BitSize() == 16);
				PX_ASSERT(cls->get64BitSizeInfo().mAlignment == 4);
				PX_ASSERT(cls->get64BitSize() == 16);
				PX_ASSERT(cls->mPackedUniformWidth == 4);
				PX_ASSERT(cls->mPackedClassType == fltClassType);
				cls->mLocked = true;
			}
		}

		{
			NamespacedName pvdName("physx3", "PxQuat");
			if(!mMetaData.findClass(pvdName).hasValue())
			{
				int classId = mMetaData.getOrCreateClass(pvdName).mClassId;
				ClassDescription* cls = mMetaData.getClassPtr(classId);
				mMetaData.createProperty(classId, "x", "", fltClassType);
				mMetaData.createProperty(classId, "y", "", fltClassType);
				mMetaData.createProperty(classId, "z", "", fltClassType);
				mMetaData.createProperty(classId, "w", "", fltClassType);
				PX_ASSERT(cls->get32BitSizeInfo().mAlignment == 4);
				PX_ASSERT(cls->get32BitSize() == 16);
				PX_ASSERT(cls->get64BitSizeInfo().mAlignment == 4);
				PX_ASSERT(cls->get64BitSize() == 16);
				PX_ASSERT(cls->mPackedUniformWidth == 4);
				PX_ASSERT(cls->mPackedClassType == fltClassType);
				cls->mLocked = true;
			}
		}

		{
			NamespacedName pvdName("physx3", "PxBounds3");
			if(!mMetaData.findClass(pvdName).hasValue())
			{
				int classId = mMetaData.getOrCreateClass(pvdName).mClassId;
				ClassDescription* cls = mMetaData.getClassPtr(classId);
				mMetaData.createProperty(classId, "minimum", "", v3ClassType);
				mMetaData.createProperty(classId, "maximum", "", v3ClassType);
				PX_ASSERT(cls->get32BitSizeInfo().mAlignment == 4);
				PX_ASSERT(cls->get32BitSize() == 24);
				PX_ASSERT(cls->mPackedUniformWidth == 4);
				PX_ASSERT(cls->mPackedClassType == fltClassType);
				cls->mLocked = true;
			}
		}

		{
			NamespacedName pvdName("physx3", "PxTransform");
			if(!mMetaData.findClass(pvdName).hasValue())
			{
				int classId = mMetaData.getOrCreateClass(pvdName).mClassId;
				ClassDescription* cls = mMetaData.getClassPtr(classId);
				mMetaData.createProperty(classId, "q", "", qtClassType, PropertyType::Scalar);
				mMetaData.createProperty(classId, "p", "", v3ClassType, PropertyType::Scalar);
				PX_ASSERT(cls->get32BitSizeInfo().mAlignment == 4);
				PX_ASSERT(cls->get32BitSize() == 28);
				PX_ASSERT(cls->mPackedUniformWidth == 4);
				PX_ASSERT(cls->mPackedClassType == fltClassType);
				cls->mLocked = true;
			}
		}

		{
			NamespacedName pvdName("physx3", "PxMat33");
			if(!mMetaData.findClass(pvdName).hasValue())
			{
				int classId = mMetaData.getOrCreateClass(pvdName).mClassId;
				ClassDescription* cls = mMetaData.getClassPtr(classId);
				mMetaData.createProperty(classId, "column0", "", v3ClassType, PropertyType::Scalar);
				mMetaData.createProperty(classId, "column1", "", v3ClassType, PropertyType::Scalar);
				mMetaData.createProperty(classId, "column2", "", v3ClassType, PropertyType::Scalar);
				PX_ASSERT(cls->get32BitSizeInfo().mAlignment == 4);
				PX_ASSERT(cls->get32BitSize() == 36);
				PX_ASSERT(cls->mPackedUniformWidth == 4);
				PX_ASSERT(cls->mPackedClassType == fltClassType);
				cls->mLocked = true;
			}
		}

		{
			NamespacedName pvdName("physx3", "PxMat34Legacy");
			if(!mMetaData.findClass(pvdName).hasValue())
			{
				int classId = mMetaData.getOrCreateClass(pvdName).mClassId;
				ClassDescription* cls = mMetaData.getClassPtr(classId);
				mMetaData.createProperty(classId, "column0", "", v3ClassType, PropertyType::Scalar);
				mMetaData.createProperty(classId, "column1", "", v3ClassType, PropertyType::Scalar);
				mMetaData.createProperty(classId, "column2", "", v3ClassType, PropertyType::Scalar);
				mMetaData.createProperty(classId, "column3", "", v3ClassType, PropertyType::Scalar);
				PX_ASSERT(cls->get32BitSizeInfo().mAlignment == 4);
				PX_ASSERT(cls->get32BitSize() == 48);
				PX_ASSERT(cls->mPackedUniformWidth == 4);
				PX_ASSERT(cls->mPackedClassType == fltClassType);
				cls->mLocked = true;
			}
		}
		//*******************************  end for compatible

		// camera props
		{
			PxI32 classId = mMetaData.getOrCreateClass(getPvdNamespacedNameForType<CameraProps>()).mClassId;
			mMetaData.createProperty(classId, "Name", "", getPvdTypeForType<StringHandle>());
			mMetaData.createProperty(classId, "Origin", "", getPvdTypeForType<PxVec3>());
			mMetaData.createProperty(classId, "Up", "", getPvdTypeForType<PxVec3>());
			mMetaData.createProperty(classId, "Target", "", getPvdTypeForType<PxVec3>());
			PropertyMessageArg args[] = {
				PropertyMessageArg("Origin", getPvdNamespacedNameForType<PxVec3>(), offsetof(CameraProps, mPosition), sizeof(PxVec3)),
				PropertyMessageArg("Up", getPvdNamespacedNameForType<PxVec3>(), offsetof(CameraProps, mUp), sizeof(PxVec3)),
				PropertyMessageArg("Target", getPvdNamespacedNameForType<PxVec3>(), offsetof(CameraProps, mTarget), sizeof(PxVec3)),
			};
			mMetaData.createPropertyMessage(getPvdNamespacedNameForType<CameraProps>(), getPvdNamespacedNameForType<CameraProps>(), DataRef<PropertyMessageArg>(args, 3), sizeof(CameraProps));
		}

		{
			PxI32 classId = mMetaData.getOrCreateClass(getPvdNamespacedNameForType<SectionEventData>()).mClassId;
			mMetaData.createProperty(classId, "StreamId", getPvdTypeForType<PxU64>());
			mMetaData.createProperty(classId, "Timestamp", getPvdTypeForType<PxU64>());
			mMetaData.createProperty(classId, "Name", getPvdTypeForType<StringHandle>());
			mMetaData.createProperty(classId, "IsBegin", getPvdTypeForType<bool>());

			PropertyMessageArg args[] = {
				PropertyMessageArg("StreamId", getPvdNamespacedNameForType<PxU64>(), offsetof(SectionEventData, mStreamId), sizeof(PxU64)),
				PropertyMessageArg("Timestamp", getPvdNamespacedNameForType<PxU64>(), offsetof(SectionEventData, mTimestamp), sizeof(PxU64)),
				PropertyMessageArg("Name", getPvdNamespacedNameForType<StringHandle>(), offsetof(SectionEventData, mName), sizeof(void*)),
				PropertyMessageArg("IsBegin", getPvdNamespacedNameForType<bool>(), offsetof(SectionEventData, mIsBegin), sizeof(bool)),
			};

			mMetaData.createPropertyMessage(getPvdNamespacedNameForType<SectionEventData>(), getPvdNamespacedNameForType<SectionEventData>(), DataRef<PropertyMessageArg>(args, 4),
			                                sizeof(SectionEventData));
		}

		// Error Messsage
		{
#ifndef SDK_SIDE_DEFINE
			PxI32 classId = mMetaData.getOrCreateClass(getPvdNamespacedNameForType<PvdErrorMessage>()).mClassId;

			mMetaData.createProperty(classId, "Code", "user message", getPvdTypeForType<PxU32>());
			mMetaData.createProperty(classId, "Message", "user message", getPvdTypeForType<StringHandle>());
			mMetaData.createProperty(classId, "File", "user message", getPvdTypeForType<StringHandle>());
			mMetaData.createProperty(classId, "Line", "user message", getPvdTypeForType<PxU32>());
			mMetaData.createProperty(classId, "Index", "user message", getPvdTypeForType<PxU32>());
			PropertyMessageArg args[] = {
				PropertyMessageArg("Code", getPvdNamespacedNameForType<PxU32>(), offsetof(PvdErrorMessage, mCode), sizeof(PxU32)),
				PropertyMessageArg("Message", getPvdNamespacedNameForType<StringHandle>(), offsetof(PvdErrorMessage, mMessage), sizeof(StringHandle)),
				PropertyMessageArg("File", getPvdNamespacedNameForType<StringHandle>(), offsetof(PvdErrorMessage, mFile), sizeof(StringHandle)),
				PropertyMessageArg("Line", getPvdNamespacedNameForType<PxU32>(), offsetof(PvdErrorMessage, mLine), sizeof(PxU32)),
				PropertyMessageArg("Index", getPvdNamespacedNameForType<PxU32>(), offsetof(PvdErrorMessage, mIndex), sizeof(PxU32)),
			};
			mMetaData.createPropertyMessage(getPvdNamespacedNameForType<PvdErrorMessage>(), getPvdNamespacedNameForType<PvdErrorMessage>(), DataRef<PropertyMessageArg>(args, 4),
			                                sizeof(PvdErrorMessage));

			PxI32 parentId = mMetaData.getOrCreateClass(getUserCustomedEventName()).mClassId;
			mMetaData.createProperty(parentId, "Events", "user message", classId, PropertyType::Array);
#endif
		}
	}

	void load(PvdInputStream& stream)
	{
		PxU32 metaVersion;
		stream >> metaVersion;
		PX_ASSERT(metaVersion == getStaticMetaVersion());

		stream >> mNumerator;
		stream >> mDenominator;
		InputSerializer serializer(mAllocator, stream, mInstancePool, mMetaData.getStringTable());
		serializer.streamify(mInstanceIdToInstance);
		mStreamIdToInstance.clear();
		PxU32 numStreamIds = 0;
		serializer.streamify(numStreamIds);
		PVD_FOREACH(idx, numStreamIds)
		{
			PxU64 sid;
			PxI32 iid;
			serializer.streamify(sid);
			serializer.streamify(iid);
			mStreamIdToInstance[sid] = mInstanceIdToInstance[iid];
		}
		serializer.streamify(mCameraMap);
		serializer.streamify(mSectionMap);
		serializer.streamify(mProfileZoneMap);
		stream >> mMemoryEventsInstance;
		stream >> mMemoryTotalsInstance;
		serializer.streamify(mMemoryEventEntries);
	}

	virtual void save(PvdOutputStream& stream)
	{
		stream << getStaticMetaVersion();
		stream << mNumerator;
		stream << mDenominator;
		OutputSerializer serializer(stream);
		serializer.streamify(mInstanceIdToInstance);
		PxU32 numSids = static_cast<PxU32>(mStreamIdToInstance.size());
		serializer.streamify(numSids);
		for(HashMap<PxU64, InstanceMetaData*>::Iterator iter = mStreamIdToInstance.getIterator(); iter.done() == false; ++iter)
		{
			PxU64 sid = iter->first;
			PxI32 iid = iter->second->mObject.mInstanceId;
			serializer.streamify(sid);
			serializer.streamify(iid);
		}
		serializer.streamify(mCameraMap);
		serializer.streamify(mSectionMap);
		serializer.streamify(mProfileZoneMap);
		stream << mMemoryEventsInstance;
		stream << mMemoryTotalsInstance;
		serializer.streamify(mMemoryEventEntries);
	}

	inline void copyPersonalData(StreamMetaData& newData)
	{
		for(HashMap<PxI32, InstanceMetaData*>::Iterator iter = mInstanceIdToInstance.getIterator(); iter.done() == false; ++iter)
		{
			if(iter->second)
			{
				InstanceMetaData& instMeta(newData.createInstanceMetaDataFromInstanceId(iter->second->mObject));
				instMeta = *iter->second;
				PxU64 sid = iter->second->mStreamInstanceId;
				if(mStreamIdToInstance.find(sid))
					newData.mStreamIdToInstance.insert(sid, &instMeta);
			}
		}

		for(HashMap<PxU64, ProfileZoneImpl*>::Iterator iter = mProfileZoneMap.getIterator(); iter.done() == false; ++iter)
		{
			newData.addProfileZone(iter->first, iter->second->mName, iter->second->mInstanceId);
			PVD_FOREACH(idx, iter->second->mEventArray.size())
			{
				PvdProfileZoneEvent& evt(iter->second->mEventArray[idx]);
				newData.addProfileZoneEvent(iter->first, evt.mName, evt.mEventId, evt.mIsCompiledTimeEnabled);
			}
		}

		for(Array<PvdMemoryEventEntry>::Iterator iter = mMemoryEventEntries.begin(); iter != mMemoryEventEntries.end(); ++iter)
		{
			newData.addMemoryEventEntry(iter->mType, iter->mFile, iter->mLine, iter->mTotalsArrayIndex);
		}

		// newData.mCameraMap = mCameraMap;
		newData.mCameraMap.clear();
		for(HashMap<PxU32, PxI32>::Iterator iter = mCameraMap.getIterator(); iter.done() == false; ++iter)
			newData.mCameraMap.insert(iter->first, iter->second);

		newData.mMemoryEventsInstance = mMemoryEventsInstance;
		newData.mMemoryTotalsInstance = mMemoryTotalsInstance;
		newData.mNumerator = mNumerator;
		newData.mDenominator = mDenominator;
	}

	virtual PvdCommStreamMetaData& clone(PvdObjectModelMetaData& metaData, PvdObjectModelReader& modelReader)
	{
		StreamMetaData& newData = *PVD_NEW(StreamMetaData)(mAllocator, metaData, modelReader, NULL, NULL);
		copyPersonalData(newData);
		// Note that when we are cloning for transfer we don't copy section information.
		// When we are branching, however, it is important that we do.
		for(TSectionMapType::Iterator iter = mSectionMap.getIterator(); iter.done() == false; ++iter)
		{
			const SectionListInfo& info(iter->first);
			const Array<Section>& sections(*iter->second);
			// Snapshot-only timelines have no section events to recreate their key.
			newData.mSectionMap.insert(info, PVD_NEW(Array<Section>)("Array"));
			PVD_FOREACH(idx, sections.size())
			{
				const Section& section(sections[idx]);
				if(section.mBegin.isEventValid())
					newData.addSectionEvent(info.mStreamId, info.mName, true, section.mBegin.mTimestamp, section.mBegin.mEventIndex);
				if(section.mEnd.isEventValid())
					newData.addSectionEvent(info.mStreamId, info.mName, false, section.mEnd.mTimestamp, section.mEnd.mEventIndex);
			}
		}
		return newData;
	}

	virtual PvdCommStreamMetaDataMutator& cloneForTransfer(PvdObjectModelMetaData& metaData, PvdObjectModelReader& modelReader, PvdObjectModelMutator& model,
	                                                       PvdObjectModelEventStreamUserEventListener& eventFactory)
	{
		StreamMetaData& newData = *PVD_NEW(StreamMetaData)(mAllocator, metaData, modelReader, &model, &eventFactory);
		// We are assuming we don't need to initalized the meta data, but we need to copy over all of our datastructures
		//*other* than sections.
		// Sections are copied over during the transfer process
		copyPersonalData(newData);
        // Preserve timeline identities even for a one-frame clip containing only
        // its initial snapshot. Actual section events are rebuilt by transfer.
        for(TSectionMapType::Iterator iter = mSectionMap.getIterator(); !iter.done(); ++iter)
            newData.mSectionMap.insert(iter->first, PVD_NEW(Array<Section>)("Array"));
		return newData;
	}

	// PvdObjectModelEventStreamUserEventTransfer
	virtual void handleUserEvent(PvdObjectModelEventStreamFactory&, PxI32 dtype, DataRef<const PxU8> data)
	{
		PxI32 eventDataDtype = mMetaData.findClass(getPvdNamespacedNameForType<SectionEventData>())->mClassId;
		if(dtype != eventDataDtype)
		{
			PX_ASSERT(false);
			return;
		}
		const SectionEventData* dbData = reinterpret_cast<const SectionEventData*>(data.begin());
		addSectionEvent(dbData->mStreamId, dbData->mName, dbData->mIsBegin, dbData->mTimestamp);
	}
};
} // namespace

PvdCommStreamMetaData& PvdCommStreamMetaData::create(PxAllocatorCallback& callback, PvdObjectModelMetaData& metaData, PvdObjectModelReader& modelReader, PvdInputStream& inStream)
{
	StreamMetaData& retval(*PVD_NEW(StreamMetaData)(callback, metaData, modelReader, NULL, NULL));
	retval.load(inStream);
	return retval;
}

PvdCommStreamMetaDataMutator& PvdCommStreamMetaDataMutator::create(PxAllocatorCallback& callback, PvdObjectModelMetaData& metaData, PvdObjectModelReader& modelReader, PvdObjectModelMutator& mutator,
                                                                   PvdObjectModelEventStreamUserEventListener& factory)
{
	StreamMetaData& retval(*PVD_NEW(StreamMetaData)(callback, metaData, modelReader, &mutator, &factory));
	retval.initializeMetaData();
	return retval;
}

Option<Section> PvdCommStreamMetaData::getSectionInfoFromEvent(const SectionListInfo& list, PxU64 evId)
{
	PxU32 idx = eventToSectionIndex(list, evId);
	return getSectionInfo(list, idx);
}

PxU64 PvdCommStreamMetaData::getInnerSectionTime(const SectionListInfo& list, PxU64 inStartEvent)
{
	Option<Section> section = getSectionInfoFromEvent(list, inStartEvent);
	if(section.hasValue() && section->mBegin.isTimestampValid() && section->mEnd.isTimestampValid())
		return section->mEnd.mTimestamp - section->mBegin.mTimestamp;
	return 0;
}
PxU64 PvdCommStreamMetaData::getTotalSectionTime(const SectionListInfo& list, PxU64 inStartEvent)
{
	PxU32 idx = eventToSectionIndex(list, inStartEvent);
	if(idx == 0)
		return 0;
	Option<Section> section = getSectionInfo(list, idx);
	--idx;
	Option<SectionBracket> beginSection;
	if(idx)
	{
		Option<Section> begin = getSectionInfo(list, idx);
		if(begin->mEnd.isTimestampValid())
			beginSection = begin->mEnd;
	}
	else if(section->mBegin.isTimestampValid())
		beginSection = section->mBegin;
	if(beginSection.hasValue() && section->mEnd.isTimestampValid())
		return section->mEnd.mTimestamp - beginSection->mTimestamp;
	return 0;
}
PxU64 PvdCommStreamMetaData::getSectionEndTimestamp(const SectionListInfo& list, PxU64 inStartEvent)
{
	Option<Section> section = getSectionInfoFromEvent(list, inStartEvent);
	if(section.hasValue() && section->mEnd.isTimestampValid())
		return section->mEnd.mTimestamp;
	return 0;
}
PxU64 PvdCommStreamMetaData::findNextSectionEvent(const SectionListInfo& list, PxU64 inStartEvent)
{
	PxU32 idx = eventToSectionIndex(list, inStartEvent) + 1;
	Option<Section> section = getSectionInfo(list, idx);
	if(section.hasValue())
		return section->mEnd.mEventIndex;
	return static_cast<PxU64>(-1);
}
PxU64 PvdCommStreamMetaData::findPreviousSectionEvent(const SectionListInfo& list, PxU64 inStartEvent)
{
	if(inStartEvent == 0)
		return 0;
	PxU32 idx = eventToSectionIndex(list, inStartEvent);
	if(!idx)
		return 0;
	--idx;
	Option<Section> section = getSectionInfo(list, idx);
	if(section.hasValue())
		return section->mEnd.mEventIndex;
	return 0;
}
PxU64 PvdCommStreamMetaData::getTotalSectionTimespan(const SectionListInfo& list)
{
	PxU32 numSections = getNbSections(list);
	Option<Section> begin = getSectionInfo(list, 1);
	Option<Section> end = getSectionInfo(list, numSections);
	if(begin.hasValue() && end.hasValue())
	{
		PxU64 beginTs = begin->mBegin.isTimestampValid() ? begin->mBegin.mTimestamp : begin->mEnd.mTimestamp;
		PxU64 endTs = end->mEnd.isTimestampValid() ? end->mEnd.mTimestamp : end->mBegin.mTimestamp;
		if(endTs > beginTs)
			return endTs - beginTs;
	}
	return 0;
}
Option<Section> PvdCommStreamMetaData::getFrameInfo(const PxU32 frameIndex, const PxU64 sceneId)
{
	// Option<SectionListInfo> frameMarker(getDefaultFrameMarker());
	Option<SectionListInfo> frameMarker(getSectionListInfoForScene(sceneId));
	if(frameMarker.hasValue())
		return getSectionInfo(frameMarker, frameIndex);
	return None();
}

PxU32 PvdCommStreamMetaData::eventToFrame(PxU64 eventIdx, PxU64 sceneId)
{
	// Option<SectionListInfo> frameMarker(getDefaultFrameMarker());
	Option<SectionListInfo> frameMarker(getSectionListInfoForScene(sceneId));
	if(frameMarker.hasValue())
		return eventToSectionIndex(frameMarker, eventIdx);
	return 0;
}
PxU64 PvdCommStreamMetaData::frameToEvent(PxU32 frameIndex, PxU64 sceneId)
{
	Option<Section> frameSection = getFrameInfo(frameIndex, sceneId);
	if(frameSection.hasValue())
	{
		const Section& sectionRef(frameSection);
		if(sectionRef.mEnd.isEventValid())
			return sectionRef.mEnd.mEventIndex;
	}
	if(frameIndex)
		return static_cast<PxU64>(-1);
	else
		return 0;
}

PxU32 PvdCommStreamMetaData::getFrameCount(PxU64 sceneId)
{
	// Option<SectionListInfo> frameMarker(getDefaultFrameMarker());
	Option<SectionListInfo> frameMarker(getSectionListInfoForScene(sceneId));
	if(frameMarker.hasValue())
		return getNbSections(frameMarker) + 1;
	return 0;
}

PxU64 PvdCommStreamMetaData::getInnerFrameTime(PxU64 inStartEvent)
{
	Option<SectionListInfo> frameMarker(getDefaultFrameMarker());
	if(frameMarker.hasValue())
		return getInnerSectionTime(frameMarker, inStartEvent);
	return 0;
}
PxU64 PvdCommStreamMetaData::getTotalFrameTime(PxU64 inStartEvent)
{
	Option<SectionListInfo> frameMarker(getDefaultFrameMarker());
	if(frameMarker.hasValue())
		return getTotalSectionTime(frameMarker, inStartEvent);
	return 0;
}
PxU64 PvdCommStreamMetaData::getFrameEndTimestamp(PxU64 inStartEvent)
{
	Option<SectionListInfo> frameMarker(getDefaultFrameMarker());
	if(frameMarker.hasValue())
		return getSectionEndTimestamp(frameMarker, inStartEvent);
	return 0;
}
// What event does the next frame correspond to.
// If we are at the end, then we use PX_MAX_U64
PxU64 PvdCommStreamMetaData::findNextFrameEvent(PxU64 inStartEvent, PxU64 sceneId)
{
	// Option<SectionListInfo> frameMarker(getDefaultFrameMarker());
	Option<SectionListInfo> frameMarker(getSectionListInfoForScene(sceneId));
	if(frameMarker.hasValue())
		return findNextSectionEvent(frameMarker, inStartEvent);
	return 0;
}
// What event does the previous frame correspond to
PxU64 PvdCommStreamMetaData::findPreviousFrameEvent(PxU64 inStartEvent, PxU64 sceneId)
{
	// Option<SectionListInfo> frameMarker(getDefaultFrameMarker());
	Option<SectionListInfo> frameMarker(getSectionListInfoForScene(sceneId));
	if(frameMarker.hasValue())
		return findPreviousSectionEvent(frameMarker, inStartEvent);
	return 0;
}
// Get the total timespan of the DB capture.
PxU64 PvdCommStreamMetaData::getTotalFrameTimespan()
{
	Option<SectionListInfo> frameMarker(getDefaultFrameMarker());
	if(frameMarker.hasValue())
		return getTotalSectionTimespan(frameMarker);
	return 0;
}
