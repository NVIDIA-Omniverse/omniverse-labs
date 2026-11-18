/*
 * SPDX-FileCopyrightText: Copyright (c) 2008-2014 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Copyright (c) 2008-2014 NVIDIA Corporation. All rights reserved.
#include "PVDClient.h"
#include "PvdCommStreamMetaData.h"
#include "pvdcommstreamparsing.h"
#include "PvdObjectModelEventStream.h"
#include "PvdRandomAccessIOStream.h"
#include "PxProfileEventHandler.h"
#include "PxProfileEventParser.h"
#include "PxProfileMemoryEventTypes.h"
#include "PxPvdCommStreamTypes.h"
#include "PxPvdFoundation.h"
#include "PxPvdInternalByteStreams.h"
#include "PxPvdObjectModel.h"

using namespace pvdapp;
using namespace physx::profile;

namespace
{

struct ProfileEventHandler : public PvdObjectModelEventStreamPropertyHandler
{
  private:
	ProfileEventHandler(const ProfileEventHandler&);
	ProfileEventHandler& operator=(const ProfileEventHandler&);

  public:
	physx::profile::PxProfileBulkEventHandler& mHandler;
	ProfileEventHandler(physx::profile::PxProfileBulkEventHandler& handler) : mHandler(handler) {}
	virtual void handleProperty(PxU64 eventIndex, PxI32 instanceId, PxI32 propertyId, const PxU8* data, PxU32 len)
	{
		PX_UNUSED(eventIndex);
		PX_UNUSED(instanceId);
		PX_UNUSED(propertyId);
		physx::profile::PxProfileBulkEventHandler::parseEventBuffer(data, len, mHandler, false);
	}
};

struct MemoryEventHandler : public PvdObjectModelEventStreamPropertyHandler
{
  private:
	MemoryEventHandler(const MemoryEventHandler&);
	MemoryEventHandler& operator=(const MemoryEventHandler&);

  public:
	PxAllocatorCallback& mAllocator;
	physx::profile::PxProfileBulkMemoryEventHandler& mHandler;
	MemoryEventHandler(physx::profile::PxProfileBulkMemoryEventHandler& handler, PxAllocatorCallback& inAlloc) : mAllocator(inAlloc), mHandler(handler) {}
	virtual void handleProperty(PxU64 eventIndex, PxI32 instanceId, PxI32 propertyId, const PxU8* data, PxU32 len)
	{
		PX_UNUSED(eventIndex);
		PX_UNUSED(instanceId);
		PX_UNUSED(propertyId);
		physx::profile::PxProfileBulkMemoryEventHandler::parseEventBuffer(data, len, mHandler, false, &mAllocator);
	}
};

struct Client : public PvdClient
{
	PxAllocatorCallback& mAllocator;
	PvdCommStreamMetaData* mStreamMetaData;
	PvdObjectModelMetaData& mMetaData;
	PvdObjectModelReader& mReader;
	PvdObjectModelEventStream* mEventStream;
	PxU32 mRefCount;

	static PxU32 getClientVersion() { return 1; }

	Client(PxAllocatorCallback& alloc, PvdCommStreamMetaData* streamMeta, PvdObjectModelMetaData& objMeta, PvdObjectModelReader& reader, PvdObjectModelEventStream* eventStream)
	: mAllocator(alloc), mStreamMetaData(streamMeta), mMetaData(objMeta), mReader(reader), mEventStream(NULL), mRefCount(0)
	{
		mMetaData.addRef();
		if(streamMeta)
			streamMeta->addRef();
		mReader.addRef();
		setEventStream(eventStream);
	}
	virtual ~Client()
	{
		setEventStream(NULL);
		mReader.release();
		if(mStreamMetaData)
			mStreamMetaData->release();
		mMetaData.release();
	}

	void setEventStream(PvdObjectModelEventStream* eventStream)
	{
		if(mEventStream != eventStream)
		{
			if(mEventStream)
				mEventStream->release();
			mEventStream = eventStream;
			if(mEventStream)
				mEventStream->addRef();
		}
	}

	PxU32 getFrameCount(PxU64 sceneId)
	{
		PvdCommStreamMetaData& meta(getStreamMetaData());
		return meta.getFrameCount(sceneId);
	}

	PxU64 getEventCount()
	{
		PvdObjectModelEventStream* stream = getEventStream();
		if(stream)
			return stream->getEventCount();
		return 0;
	}

	PxI64 getFileSize()
	{
		PvdObjectModelEventStream* stream = getEventStream();
		if(stream)
			return stream->getFileSize();
		return 0;
	}

	PxI64 getCurrentEvent()
	{
		PvdObjectModelEventStream* stream = getEventStream();
		if(stream)
			return stream->getCurrentEvent();
		return 0;
	}

	void setCurrentEvent(PxI64 val)
	{
		PvdObjectModelEventStream* stream = getEventStream();
		if(val < 0)
			val = 0;
		if(stream)
			stream->seekToEvent(val);
	}

	virtual void addRef() { ++mRefCount; }
	virtual void release()
	{
		if(mRefCount)
			--mRefCount;
		if(!mRefCount)
			PVD_DELETE(this);
	}

	virtual PvdCommStreamMetaData& getStreamMetaData() { return *mStreamMetaData; }
	virtual PvdObjectModelMetaData& getMetaData() { return mMetaData; }
	virtual PvdObjectModelReader& getReader() { return mReader; }
	virtual PvdObjectModelEventStream* getEventStream() { return mEventStream; }

	virtual PxU32 getCurrentFrame(PxU64 sceneId)
	{
		if(mEventStream == NULL)
			return 0;
		return mStreamMetaData->eventToFrame(mEventStream->getCurrentEvent(), sceneId);
	}

	virtual void setCurrentFrame(PxU32 frame, PxU64 sceneId)
	{
		if(mEventStream == NULL)
			return;
		const PxU64 eventIdx = mStreamMetaData->frameToEvent(frame, sceneId);
		mEventStream->seekToEvent(eventIdx);
	}

	virtual void parseProfileEvents(PxI64 startEvent, PxI64 endEvent, int profileZone, physx::profile::PxProfileBulkEventHandler& handler)
	{
		if(mEventStream == NULL)
			return;
		Option<PropertyDescription> prop(mMetaData.findProperty(getPvdNamespacedNameForType<physx::profile::PxProfileZone>(), "events"));
		// compatible old version
		if(prop.hasValue() == false)
		{
			NamespacedName npn("_debugger_", "PxProfileZone");
			prop = mMetaData.findProperty(npn, "events");
		}
		PX_ASSERT(prop.hasValue());
		if(prop.hasValue() == false)
			return;
		InstancePropertyId queryItem(profileZone, prop->mPropertyId);
		ProfileEventHandler evtHandler(handler);
		mEventStream->getPropertyHistory(queryItem, startEvent, endEvent, evtHandler);
	}

	Option<PropertyDescription> getMemoryEventTotalsProperty()
	{
		NamespacedName totalsCls = getMemoryEventTotalsClassName();
		Option<PropertyDescription> prop(mMetaData.findProperty(totalsCls, "totals"));
		PX_ASSERT(prop.hasValue());
		return prop;
	}

	virtual PxU32 getNbMemoryTotals()
	{
		PxI32 instance = mStreamMetaData->getMemoryTotalsInstance();
		if(instance < 0)
			return 0;
		Option<PropertyDescription> prop(getMemoryEventTotalsProperty());
		if(prop.hasValue() == false)
			return 0;
		return mReader.getNbArrayElements(mReader.idToPtr(instance), prop->mPropertyId);
	}

	virtual PxU32 getMemoryTotals(PxI32* buffer, PxU32 bufSize)
	{
		PxI32 instance = mStreamMetaData->getMemoryTotalsInstance();
		if(instance < 0)
			return 0;
		Option<PropertyDescription> prop(getMemoryEventTotalsProperty());
		if(prop.hasValue() == false)
			return 0;
		return mReader.getPropertyValue(mReader.idToPtr(instance), prop->mPropertyId, reinterpret_cast<PxU8*>(buffer), bufSize * sizeof(PxI32));
	}

	virtual void parseMemoryEvents(PxI64 startEvent, PxI64 endEvent, physx::profile::PxProfileBulkMemoryEventHandler& handler)
	{
		if(mEventStream == nullptr)
			return;
		PxI32 instance = mStreamMetaData->getMemoryEventBufferInstance();
		if(instance < 0)
		{
			return;
		}
		Option<PropertyDescription> prop(mMetaData.findProperty(getPvdNamespacedNameForType<physx::profile::PxProfileMemoryEventBuffer>(), "events"));
		// compatible old version
		if(prop.hasValue() == false)
		{
			NamespacedName npn("_debugger_", "PxProfileMemoryEventBuffer");
			prop = mMetaData.findProperty(npn, "events");
			if(prop.hasValue() == false)
			{
				PX_ASSERT(false);
				return;
			}
		}
		InstancePropertyId queryItem(instance, prop->mPropertyId);
		MemoryEventHandler evtHandler(handler, mAllocator);
		mEventStream->getPropertyHistory(queryItem, startEvent, endEvent, evtHandler);
	}

	virtual PvdClient* branch()
	{
		if(mEventStream == NULL)
		{
			PX_ASSERT(false);
			return NULL;
		}
		PvdObjectModelMetaData& newMeta(mMetaData.clone());
		PvdObjectModel& newModel(PvdObjectModel::create(mAllocator, newMeta));
		PvdObjectModelEventStream& newStream(mEventStream->branch(newModel));
		PvdCommStreamMetaData& newStreamMeta(mStreamMetaData->clone(newMeta, newModel));
		return PVD_NEW(Client)(mAllocator, &newStreamMeta, newMeta, newModel, &newStream);
	}

	virtual PvdClient* transferEvents(PxU64 eventCount, PvdRandomAccessIOStream& fileBacking)
	{
		if(mEventStream == NULL)
		{
			PX_ASSERT(false);
			return NULL;
		}
		PvdObjectModelMetaData& newMeta(mMetaData.clone());
		PvdObjectModelEventStreamFactory& newFactory(mEventStream->createTransferFactory(fileBacking, newMeta));
		PvdCommStreamMetaDataMutator& mutator = mStreamMetaData->cloneForTransfer(newMeta, newFactory.getModel(), newFactory.getMutator(), newFactory);
		const bool transferred = mEventStream->transfer(eventCount, newFactory, mutator.getEventTransfer());
		PvdClient* retval = PVD_NEW(Client)(mAllocator, &mutator.getMetaData(), newMeta, newFactory.getModel(), &newFactory.getEventStream());
		newFactory.release();
		if(!transferred)
		{
			retval->release();
			return NULL;
		}
		return retval;
	}
	virtual void save(PvdRandomAccessIOStream& ioStream) const
	{
		if(mEventStream == NULL)
		{
			PX_ASSERT(false);
			return;
		}
		ioStream << getFileId();
		// Save the meta data, save the comm stream meta data,
		// then save the event stream.
		ioStream << getClientVersion();
		MemPvdOutputStream metaStream("PvdClient::save::metaStream");
		MemPvdOutputStream streamMetaStream("PvdClient::save::streamMetaStream");

		mMetaData.save(metaStream);
		mStreamMetaData->save(streamMetaStream);

		const PxU32 metaSize = metaStream.size();
		const PxU32 streamMetaSize = streamMetaStream.size();
		ioStream << metaSize;
		ioStream << streamMetaSize;
		MemPvdInputStream metaInput(metaStream);
		ioStream.directCopy(metaInput, metaSize);
		MemPvdInputStream streamMetaInput(streamMetaStream);
		ioStream.directCopy(streamMetaInput, streamMetaSize);
		ioStream.zeroPadToPageSize();
		mEventStream->save(ioStream);
	}

	static PvdClient* load(PxAllocatorCallback& alloc, PvdRandomAccessIOStream& ioStream)
	{
		if(ioStream.getLength() < 16) return NULL;
		IOStreamToInputStream stream(0, ioStream);
		PxU32 clientId;
		stream >> clientId;
		if(clientId != PvdClient::getFileId())
			return NULL;

		PxU32 clientVersion;
		stream >> clientVersion;
		if(clientVersion != getClientVersion())
			return NULL;
		PxU32 metaSize, streamMetaSize;

		stream >> metaSize;
		stream >> streamMetaSize;
		if(static_cast<PxU64>(metaSize) + streamMetaSize > ioStream.getLength() - 16)
			return NULL;

		MemPvdOutputStream metaStream("PvdClient::load::metaStream");
		MemPvdOutputStream streamMetaStream("PvdClient::load::streamMetaStream");

		metaStream.directCopy(stream, metaSize);
		streamMetaStream.directCopy(stream, streamMetaSize);

		MemPvdInputStream metaInput(metaStream);
		MemPvdInputStream streamMetaInput(streamMetaStream);

		PvdObjectModelMetaData& newMeta(PvdObjectModelMetaData::create(metaInput));
		PvdObjectModel& newModel(PvdObjectModel::create(alloc, newMeta));
		PvdCommStreamMetaData& streamMeta(PvdCommStreamMetaData::create(alloc, newMeta, newModel, streamMetaInput));

		PvdObjectModelEventStream& eventStream(PvdObjectModelEventStream::load(alloc, newModel, ioStream, toPaddedSize(stream.mOffset)));
		return PVD_NEW(Client)(alloc, &streamMeta, newMeta, eventStream.getModel(), &eventStream);
	}
};

struct ClientMutator : public Client, public PvdClientMutator, public PvdObjectModelMutator, public PvdObjectModelEventStreamUserEventListener
{
	PvdCommStreamMetaDataMutator* mCommStreamMetaDataMutator;
	PvdObjectModel& mModel;
	PvdObjectModelEventStreamFactory* mEventStreamFactory;
	PvdObjectModelMutator* mMutator;

	ClientMutator(PxAllocatorCallback& alloc, PvdObjectModel& model, PvdObjectModelEventStreamFactory& streamFactory)
	: Client(alloc, NULL, const_cast<PvdObjectModelMetaData&>(streamFactory.getModel().getMetaData()), model, &streamFactory.getEventStream())
	, mModel(model)
	, mEventStreamFactory(&streamFactory)
	, mMutator(&streamFactory.getMutator())
	{
		mCommStreamMetaDataMutator = &PvdCommStreamMetaDataMutator::create(alloc, mMetaData, streamFactory.getModel(), *this, *this);
		mCommStreamMetaDataMutator->addRef();
		mStreamMetaData = &mCommStreamMetaDataMutator->getMetaData();
		mStreamMetaData->addRef();
		mModel.addRef();
		mEventStreamFactory->addRef();
	}
	virtual ~ClientMutator()
	{
		if(mEventStreamFactory)
			mEventStreamFactory->release();
		mModel.release();
		mCommStreamMetaDataMutator->release();
	}

	virtual void addRef() { ++mRefCount; }

	virtual void release()
	{
		if(mRefCount)
			--mRefCount;
		if(!mRefCount)
			PVD_DELETE(this);
	}

	virtual PvdClient& getClient() { return *this; }
	virtual PvdCommStreamMetaDataMutator& getCommStreamMetaDataMutator() { return *mCommStreamMetaDataMutator; }
	virtual PvdObjectModelMutator& getObjectModelMutator() { return *this; }
	virtual PvdObjectModelMutator& getRawModelMutator() { return mModel; }
	virtual PvdObjectModelEventStreamFactory* getEventStreamFactory() { return mEventStreamFactory; }

	virtual void disableRecording()
	{
		if(mEventStreamFactory)
		{
			mEventStreamFactory->release();
			mEventStreamFactory = NULL;
			mMutator = &mModel;
			mCommStreamMetaDataMutator->onRecordingStopped();
			setEventStream(NULL);
		}
	}

	virtual void enableRecording(PvdRandomAccessIOStream& recordingBacking)
	{
		disableRecording();
		mEventStreamFactory = &PvdObjectModelEventStreamFactory::create(mAllocator, mModel, recordingBacking);
		mMutator = &mEventStreamFactory->getMutator();
		mCommStreamMetaDataMutator->onRecordingStarted();
		mEventStreamFactory->addRef();
		setEventStream(&mEventStreamFactory->getEventStream());
	}

	virtual bool isRecording() const { return mEventStreamFactory != NULL; }

	virtual PvdCommStreamParser& createParser(PvdCommStreamEventReader& reader) { return reader.createParser(mMetaData, *this, *mCommStreamMetaDataMutator); }

	// PvdObjectModelMutator
	virtual void recordCompletedInstances() { mMutator->recordCompletedInstances(); }

	virtual void* idToPtr(const PxI32 instId) const { return mMutator->idToPtr(instId); }
	virtual PxI32 ptrToId(void* instPtr) const { return mMutator->ptrToId(instPtr); }
	virtual InstanceDescription idToDescriptor(PxI32 instId) const { return mMutator->idToDescriptor(instId); }
	virtual InstanceDescription ptrToDescriptor(void* instPtr) const { return mMutator->ptrToDescriptor(instPtr); }
	virtual Option<ClassDescription> getClassOf(void* instId) const { return mMutator->getClassOf(instId); }
	virtual const PvdObjectModelMetaData& getMetaData() const { return mMutator->getMetaData(); }
	// if the instance is alive, this destroyes any arrays and sets the instance back to its initial state.
	virtual InstanceDescription createInstance(PxI32 clsId, PxI32 instId)
	{
		InstanceDescription retval(mMutator->createInstance(clsId, instId));
		return retval;
	}
	virtual InstanceDescription createInstance(PxI32 clsId) { return mMutator->createInstance(clsId); }
	virtual void pinInstance(void* instId) { mMutator->pinInstance(instId); }
	virtual void unPinInstance(void* instId) { mMutator->unPinInstance(instId); }
	virtual void destroyInstance(void* instId) { return mMutator->destroyInstance(instId); }
	virtual PxI32 getNextInstanceHandleValue() const { return mMutator->getNextInstanceHandleValue(); }
	virtual void setNextInstanceHandleValue(PxI32 hdlValue) { return mMutator->setNextInstanceHandleValue(hdlValue); }
	virtual bool setPropertyValue(void* instId, PxI32 propId, const PxU8* data, PxU32 dataLen, PxI32 incomingType) { return mMutator->setPropertyValue(instId, propId, data, dataLen, incomingType); }
	virtual bool setPropertyMessage(void* instId, PxI32 msgId, const PxU8* data, PxU32 dataLen) { return mMutator->setPropertyMessage(instId, msgId, data, dataLen); }
	virtual bool insertArrayElement(void* instId, PxI32 propId, PxI32 index, const PxU8* data, PxU32 dataLen, PxI32 incomingType)
	{
		return mMutator->insertArrayElement(instId, propId, index, data, dataLen, incomingType);
	}
	virtual bool removeArrayElement(void* instId, PxI32 propId, PxI32 index) { return mMutator->removeArrayElement(instId, propId, index); }
	virtual Option<PxI32> pushBackArrayElementIf(void* instId, PxI32 propId, const PxU8* data, PxU32 dataLen, PxI32 incomingType)
	{
		return mMutator->pushBackArrayElementIf(instId, propId, data, dataLen, incomingType);
	}
	virtual Option<PxI32> removeArrayElementIf(void* instId, PxI32 propId, const PxU8* data, PxU32 dataLen, PxI32 incomingType)
	{
		return mMutator->removeArrayElementIf(instId, propId, data, dataLen, incomingType);
	}
	virtual bool setArrayElementValue(void* instId, PxI32 propId, PxI32 propIdx, const PxU8* data, PxU32 dataLen, PxI32 incomingType)
	{
		return mMutator->setArrayElementValue(instId, propId, propIdx, data, dataLen, incomingType);
	}
	virtual void originShift(void* instId, PxVec3 shift) { return mMutator->originShift(instId, shift); }

	// PvdObjectModelEventStreamUserEventListener
	virtual PvdObjectModelMutator& getMutator() { return *this; }

	virtual PxU64 addUserEvent(PxI32 dtype, DataRef<const PxU8> userData)
	{
		if(mEventStreamFactory)
			return mEventStreamFactory->addUserEvent(dtype, userData);
		return 0;
	}
};
} // namespace

PvdClient* PvdClient::load(PxAllocatorCallback& alloc, PvdRandomAccessIOStream& stream) { return Client::load(alloc, stream); }

PvdClientMutator& PvdClientMutator::create(PxAllocatorCallback& alloc, PvdRandomAccessIOStream& fileBacking, bool isCapture)
{
	PvdObjectModelMetaData& meta(PvdObjectModelMetaData::create());
	PvdObjectModel& model(PvdObjectModel::create(alloc, meta, isCapture));
	PvdObjectModelEventStreamFactory& factory(PvdObjectModelEventStreamFactory::create(alloc, model, fileBacking));
	return *PVD_NEW(ClientMutator)(alloc, model, factory);
}
