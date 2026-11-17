/*
 * SPDX-FileCopyrightText: Copyright (c) 2008-2014 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Copyright (c) 2008-2014 NVIDIA Corporation. All rights reserved.
#include "pvdcommstreamparsing.h"
#include "PvdCommStreamMetaData.h"
#include "PvdCommStreamParsingConverters.h"
#include "PxProfileEventSerialization.h"
#include "PxProfileMemoryEventParser.h"
#include "PxProfileMemoryEventSummarizer.h"
#include "PxPvdByteStreams.h"
#include "PxPvdCommStreamTypes.h"
#include "PxPvdFoundation.h"
#include "PvdInputValidation.h"
#include "PxPvdObjectModel.h"
#include "PxPvdUserRenderImpl.h"

#include "profiler.h"

#include <cstring>

using namespace pvdapp;
using namespace pvdapp::comm;
using namespace physx::profile;
using namespace physx::shdfnd;

namespace
{

typedef ForwardingMemoryBuffer TMemoryBuffer;
typedef physx::profile::EventSerializer<ForwardingMemoryBuffer> TSerializerType;
typedef EventDeserializer<true> TDeserializerType;

struct ProfileEventPassthrough
{
	TSerializerType& mSerializer;
	TDeserializerType& mDeserializer;
	EventHeader& mHeader;

	ProfileEventPassthrough(TSerializerType& ser, TDeserializerType& deser, EventHeader& header) : mSerializer(ser), mDeserializer(deser), mHeader(header) {}

	template <typename TEventType>
	bool operator()(const TEventType&)
	{
		TEventType theEvt;
		// Take care of any endianness issues at parse time.
		theEvt.streamify(mDeserializer, mHeader);
		mHeader.streamify(mSerializer);
		theEvt.streamify(mSerializer, mHeader);
		return true;
	}

	bool operator()(PxU8)
	{
		PX_ASSERT(false);
		return false;
	}
};

typedef HashMap<PxU32, const char*> THandleToStringMap;

class MemoryEventParser
{
  protected:
	virtual ~MemoryEventParser() {}

  public:
	virtual DataRef<const PxU8> handleEventData(DataRef<const PxU8> data) = 0;
	virtual void release() = 0;
};

#pragma warning(disable : 4355)

template <bool TSwapBytes>
struct SMemoryEventContext : public MemoryEventParser
{
	typedef SMemoryEventContext<TSwapBytes> TThisType;
	typedef MemoryEventSummarizer<TSwapBytes, NewEntryOperatorForward<TThisType>, MemoryEventForward<TThisType> > TSummarizerType;
	typedef physx::profile::EventSerializer<TMemoryBuffer> TSerializerType;
	typedef EventDeserializer<TSwapBytes> TDeserializerType;

	TMemoryBuffer mMemoryBuffer;
	PvdCommStreamMetaDataMutator& mStreamMetaData;
	PvdObjectModelMetaData& mMetaData;
	PvdObjectModelMutator& mMutator;
	StringTable& mStringTable;
	HashMap<PxU32, PxU32> mHdlToHdlMap;

	InstanceDescription mEventStreamObject;
	InstanceDescription mTotalsArrayObject;

	TSerializerType mSerializer;
	TDeserializerType mDeserializer;
	TSummarizerType mSummarizer;

	SMemoryEventContext(PvdCommStreamMetaDataMutator& streamMeta, PvdObjectModelMutator& mutator, InstanceDescription eventStream)
	: mMemoryBuffer("SMemoryEventContext::mMemoryBuffer")
	, mStreamMetaData(streamMeta)
	, mMetaData(const_cast<PvdObjectModelMetaData&>(mutator.getMetaData()))
	, mMutator(mutator)
	, mStringTable(mMetaData.getStringTable())
	, mHdlToHdlMap("SMemoryEventContext::mHdlToHdlMap")
	, mEventStreamObject(eventStream)
	, mSerializer(&mMemoryBuffer)
	, mDeserializer(NULL, 0)
	, mSummarizer(*physx::pvdsdk::gPvdAllocatorCallback, NewEntryOperatorForward<TThisType>(*this), MemoryEventForward<TThisType>(*this))
	{
		NamespacedName typeName(getMemoryEventTotalsClassName());
		PxI32 clsId = mMetaData.getOrCreateClass(typeName).mClassId;
		mMetaData.createProperty(typeName, "totals", "", getPvdNamespacedNameForType<PxI32>(), PropertyType::Array);
		mTotalsArrayObject = mMutator.createInstance(clsId);
		mStreamMetaData.setMemoryEventInstances(eventStream, mTotalsArrayObject);
	}

	virtual ~SMemoryEventContext() {}

	DataRef<const PxU8> handleEventData(DataRef<const PxU8> inData)
	{
		PxU32 inLen = inData.size();
		mMemoryBuffer.reserve(inLen * 2);
		mMemoryBuffer.clear();
		mDeserializer = TDeserializerType(inData.begin(), inLen);
		mSummarizer.handleData(inData.begin(), inLen);
		// set the array data of the summarizer.
		PxU32 size = mSummarizer.mTotalsArray.size();
		if(size)
		{
			const PxI32* theData = mSummarizer.mTotalsArray.begin();
			PxU32 theLen = mSummarizer.mTotalsArray.size();
			mMutator.setPropertyValue(mTotalsArrayObject, "totals", theData, theLen);
		}
		return DataRef<const PxU8>(mMemoryBuffer.begin(), mMemoryBuffer.size());
	}
	void operator()(const MemoryEventSummarizerEntry& inEntry, const char* inTypeStr, const char* inFileStr, PxU32 inTotalsArrayIndex)
	{
		mStreamMetaData.addMemoryEventEntry(inTypeStr, inFileStr, inEntry.mLine, inTotalsArrayIndex);
	}

	void operator()(const MemoryEventHeader&, const StringTableEvent& evt) { mHdlToHdlMap.insert(evt.mHandle, mStringTable.strToHandle(evt.mString)); }

	PxU32 findName(PxU32 inSrc)
	{
		const HashMap<PxU32, PxU32>::Entry* entry(mHdlToHdlMap.find(inSrc));
		if(entry)
			return entry->second;
		return 0;
	}

	void operator()(const MemoryEventHeader& inHeader, const AllocationEvent& evt)
	{
		MemoryEventHeader newHeader(inHeader.getType());
		AllocationEvent newEvent(evt);
		newEvent.mType = findName(evt.mType);
		newEvent.mFile = findName(evt.mFile);
		newEvent.setup(newHeader);
		newHeader.streamify(mSerializer);
		newEvent.streamify(mSerializer, newHeader);
	}

	void operator()(const MemoryEventHeader& inHeader, const DeallocationEvent& evt)
	{
		MemoryEventHeader newHeader(inHeader);
		DeallocationEvent newEvent(evt);
		newHeader.streamify(mSerializer);
		newEvent.streamify(mSerializer, newHeader);
	}

	// will never happen.
	void operator()(const MemoryEventHeader&, const FullAllocationEvent&) { PX_ASSERT(false); }

	virtual void release() { PVD_DELETE(this); }
};

struct PvdUserRendererTransfer : public PvdUserRenderHandler, public RendererEventClient
{
	PvdCommStreamMetaData& mMetaData;
	ForwardingMemoryBuffer mBuffer;
	PvdUserRenderer& mRenderer;
	PvdUserRenderParser& mParser;

	PvdUserRendererTransfer(PvdCommStreamMetaData& meta, bool swapBytes)
	: mMetaData(meta), mBuffer("PvdUserRendererTransfer::buffer"), mRenderer(*PvdUserRenderer::create()), mParser(PvdUserRenderParser::create(swapBytes))
	{
		mRenderer.setClient(this);
	}
	virtual ~PvdUserRendererTransfer()
	{
		mRenderer.release();
		mParser.release();
	}

	DataRef<const PxU8> handleData(DataRef<const PxU8> data)
	{
		mBuffer.clear();
		mParser.parseData(data, *this);
		mRenderer.flushRenderEvents();
		return DataRef<const PxU8>(mBuffer.begin(), mBuffer.size());
	}

	void* fromStreamToVoidPtr(PxU64 iid)
	{
		if(iid)
		{
			PxI32 streamId = mMetaData.getInstanceId(iid);
			if(streamId >= 0)
				return reinterpret_cast<void*>(static_cast<size_t>(streamId));
		}
		return NULL;
	}

	virtual void handleRenderEvent(const SetInstanceIdRenderEvent& evt) { mRenderer.setInstanceId(fromStreamToVoidPtr(evt.mInstanceId)); }

	virtual void handleRenderEvent(const PointsRenderEvent& evt) { mRenderer.drawPoints(evt.mData.begin(), evt.mData.size()); }

	virtual void handleRenderEvent(const LinesRenderEvent& evt) { mRenderer.drawLines(evt.mData.begin(), evt.mData.size()); }

	virtual void handleRenderEvent(const TrianglesRenderEvent& evt) { mRenderer.drawTriangles(evt.mData.begin(), evt.mData.size()); }

	virtual void handleRenderEvent(const DebugRenderEvent& evt)
	{
		mRenderer.drawRenderbuffer(evt.mPointData.begin(), evt.mPointData.size(), evt.mLineData.begin(), evt.mLineData.size(), evt.mTriangleData.begin(), evt.mTriangleData.size());
	}

	virtual void handleRenderEvent(const TextRenderEvent& evt) { mRenderer.drawText(evt.mText); }

	virtual void handleRenderEvent(const JointFramesRenderEvent& evt) { mRenderer.visualizeJointFrames(evt.parent, evt.child); }
	virtual void handleRenderEvent(const LinearLimitRenderEvent& evt) { mRenderer.visualizeLinearLimit(evt.t0, evt.t1, evt.value, evt.active); }
	virtual void handleRenderEvent(const AngularLimitRenderEvent& evt) { mRenderer.visualizeAngularLimit(evt.t0, evt.lower, evt.upper, evt.active); }
	virtual void handleRenderEvent(const LimitConeRenderEvent& evt) { mRenderer.visualizeLimitCone(evt.t, evt.ySwing, evt.zSwing, evt.active); }
	virtual void handleRenderEvent(const DoubleConeRenderEvent& evt) { mRenderer.visualizeDoubleCone(evt.t, evt.angle, evt.active); }

	virtual void handleBufferFlush(const PxU8* inData, PxU32 inLength) { mBuffer.write(inData, inLength); }
};

struct ParserSessionBudget
{
	PxU32 mParserStreams;
	PxU32 mIncompletePropertyStates;
	PxU64 mRetainedPropertyBytes;

	ParserSessionBudget()
	: mParserStreams(0)
	, mIncompletePropertyStates(0)
	, mRetainedPropertyBytes(0)
	{
	}

	bool tryAddParserStream()
	{
		if(mParserStreams >= pvdapp::input::kMaxParserStreams)
			return false;
		++mParserStreams;
		return true;
	}

	void removeParserStream()
	{
		PX_ASSERT(mParserStreams > 0);
		if(mParserStreams > 0)
			--mParserStreams;
	}

	bool tryBeginPropertyState()
	{
		if(mIncompletePropertyStates >= pvdapp::input::kMaxIncompletePropertyStates)
			return false;
		++mIncompletePropertyStates;
		return true;
	}

	void endPropertyState()
	{
		PX_ASSERT(mIncompletePropertyStates > 0);
		if(mIncompletePropertyStates > 0)
			--mIncompletePropertyStates;
	}

	bool tryRetainPropertyBytes(PxU32 byteCount)
	{
		if(mRetainedPropertyBytes > pvdapp::input::kMaxRetainedPropertyBytes ||
		   static_cast<PxU64>(byteCount) >
		       pvdapp::input::kMaxRetainedPropertyBytes - mRetainedPropertyBytes)
			return false;
		mRetainedPropertyBytes += byteCount;
		return true;
	}

	void releasePropertyBytes(PxU32 byteCount)
	{
		PX_ASSERT(static_cast<PxU64>(byteCount) <= mRetainedPropertyBytes);
		if(static_cast<PxU64>(byteCount) <= mRetainedPropertyBytes)
			mRetainedPropertyBytes -= byteCount;
		else
			mRetainedPropertyBytes = 0;
	}
};

struct ParserStream : public ParserIdToDBHandle
{
	PvdObjectModelMetaData& mMetaData;
	PvdObjectModelMutator& mModel;
	PvdCommStreamMetaDataMutator& mCommStreamMeta;
	StringTable& mStringTable;
	THandleToStringMap& mStringMap;
	HashMap<PxI32, PxI32>& mPropertyMessageMap; // Maps property messages incoming to what we apply to the model.
	HashMap<PxU64, InstanceDescription>& mInstanceMap;
	PxU64& mTotalEventCount;
	ExtendedEventSerializer& mSerializer;
	ForwardingMemoryBuffer& mTempData; // Shared marshalling scratch.
	ForwardingMemoryBuffer* mSPVData; // Released when an unfinished value ends.
	PxU32 mSPVNumItems;
	ParserSessionBudget& mSessionBudget;
	DataStreamState::Enum mStreamState;
	bool mValid;
	bool mClosed;
	BeginSetPropertyValue mSPVEvent;
	// For SetPropertyMessage groups
	PropertyMessageDescription mGroupPropertyMessage;
	PxI32 mGroupMessageId;
	MemoryEventParser* mMemoryEventParser;
	PvdUserRendererTransfer* mRenderTransfer;
	Array<PropertyDescription> mPropertyArray;

	bool m64BitPointers;

	ParserStreamDataConverterFactory& mConverterFactory;
	Array<ConverterAndData> mCurrentConverters;

	ParserStream(const ParserStream& other);
	ParserStream& operator=(const ParserStream& other);

	ParserStream(PxAllocatorCallback& alloc, PvdObjectModelMetaData& meta, PvdObjectModelMutator& model, PvdCommStreamMetaDataMutator& commStreamMeta, THandleToStringMap& stringMap,
	             PxU64& totalEventCount, HashMap<PxI32, PxI32>& propMessageMap, HashMap<PxU64, InstanceDescription>& instMap, ExtendedEventSerializer& serializer, bool pointersAre64Bit,
	             ParserSessionBudget& sessionBudget, ForwardingMemoryBuffer& tempData)
	: mMetaData(meta)
	, mModel(model)
	, mCommStreamMeta(commStreamMeta)
	, mStringTable(meta.getStringTable())
	, mStringMap(stringMap)
	, mPropertyMessageMap(propMessageMap)
	, mInstanceMap(instMap)
	, mTotalEventCount(totalEventCount)
	, mSerializer(serializer)
	, mTempData(tempData)
	, mSPVData(NULL)
	, mSPVNumItems(0)
	, mSessionBudget(sessionBudget)
	, mStreamState(DataStreamState::Open)
	, mValid(true)
	, mClosed(false)
	, mMemoryEventParser(NULL)
	, mRenderTransfer(NULL)
	, mPropertyArray("ParserStream::mPropertyArray")
	, m64BitPointers(pointersAre64Bit)
	, mConverterFactory(ParserStreamDataConverterFactory::createConverterFactory(alloc, serializer.performsSwap(), pointersAre64Bit, *this, meta.getStringTable()))
	, mCurrentConverters("ParserStream::mCurrentConverters")
	{
	}
	virtual ~ParserStream()
	{
		discardIncompletePropertyState();
		if(mMemoryEventParser)
			mMemoryEventParser->release();
		mMemoryEventParser = NULL;
		if(mRenderTransfer)
			PVD_DELETE(mRenderTransfer);
		mRenderTransfer = NULL;
		mCurrentConverters.clear();
		mConverterFactory.release();
	}

	void discardIncompletePropertyState()
	{
		if(mSPVData)
		{
			mSessionBudget.releasePropertyBytes(mSPVData->size());
			PVD_DELETE(mSPVData);
			mSPVData = NULL;
		}
		mSPVNumItems = 0;
		if(mStreamState == DataStreamState::SetPropertyValue)
		{
			mSessionBudget.endPropertyState();
			mStreamState = DataStreamState::Open;
		}
	}

	PvdCommStreamParseResult::Enum failParsing()
	{
		mValid = false;
		discardIncompletePropertyState();
		return PvdCommStreamParseResult::Failed;
	}

	PvdCommStreamParseResult::Enum finalizeParsing()
	{
		if(!mValid || mStreamState != DataStreamState::Open)
			return failParsing();
		return PvdCommStreamParseResult::Success;
	}

	void visitCommEventType(PvdCommStreamEventTypes::Enum value)
	{
		switch(value)
		{
		case PvdCommStreamEventTypes::StringHandleEvent:
			handleStringHandleEvent();
			break;
		case PvdCommStreamEventTypes::CreateClass:
			handleCreateClass();
			break;
		case PvdCommStreamEventTypes::DeriveClass:
			handleDeriveClass();
			break;
		case PvdCommStreamEventTypes::CreateProperty:
			handleCreateProperty();
			break;
		case PvdCommStreamEventTypes::CreatePropertyMessage:
			handleCreatePropertyMessage();
			break;
		case PvdCommStreamEventTypes::CreateInstance:
			handleCreateInstance();
			break;
		case PvdCommStreamEventTypes::SetPropertyValue:
			handleSetPropertyValue();
			break;
		case PvdCommStreamEventTypes::BeginSetPropertyValue:
			handleBeginSetPropertyValue();
			break;
		case PvdCommStreamEventTypes::AppendPropertyValueData:
			handleAppendPropertyValueData();
			break;
		case PvdCommStreamEventTypes::EndSetPropertyValue:
			handleEndSetPropertyValue();
			break;
		case PvdCommStreamEventTypes::SetPropertyMessage:
			handleSetPropertyMessage();
			break;
		case PvdCommStreamEventTypes::BeginPropertyMessageGroup:
			handleBeginPropertyMessageGroup();
			break;
		case PvdCommStreamEventTypes::SendPropertyMessageFromGroup:
			handleSendPropertyMessageFromGroup();
			break;
		case PvdCommStreamEventTypes::EndPropertyMessageGroup:
			handleEndPropertyMessageGroup();
			break;
		case PvdCommStreamEventTypes::DestroyInstance:
			handleDestroyInstance();
			break;
		case PvdCommStreamEventTypes::PushBackObjectRef:
			handlePushBackObjectRef();
			break;
		case PvdCommStreamEventTypes::RemoveObjectRef:
			handleRemoveObjectRef();
			break;
		case PvdCommStreamEventTypes::BeginSection:
			handleBeginSection();
			break;
		case PvdCommStreamEventTypes::EndSection:
			handleEndSection();
			break;
		case PvdCommStreamEventTypes::SetPickable:
			handleSetPickable();
			break;
		case PvdCommStreamEventTypes::SetColor:
			handleSetColor();
			break;
		case PvdCommStreamEventTypes::SetIsTopLevel:
			handleSetIsTopLevel();
			break;
		case PvdCommStreamEventTypes::SetCamera:
			handleSetCamera();
			break;
		case PvdCommStreamEventTypes::AddProfileZone:
			handleAddProfileZone();
			break;
		case PvdCommStreamEventTypes::AddProfileZoneEvent:
			handleAddProfileZoneEvent();
			break;
		case PvdCommStreamEventTypes::StreamEndEvent:
			handleStreamEndEvent();
			break;
		case PvdCommStreamEventTypes::ErrorMessage:
			handleErrorMessage();
			break;
		case PvdCommStreamEventTypes::OriginShift:
			handleOriginShift();
			break;
		case PvdCommStreamEventTypes::Unknown:
		case PvdCommStreamEventTypes::Last:
		default:
			PX_ASSERT(false);
		}
	}

	bool isEventAllowedInCurrentState(PvdCommStreamEventTypes::Enum eventType) const
	{
		switch(mStreamState)
		{
		case DataStreamState::Open:
			return eventType != PvdCommStreamEventTypes::AppendPropertyValueData &&
			       eventType != PvdCommStreamEventTypes::EndSetPropertyValue &&
			       eventType != PvdCommStreamEventTypes::SendPropertyMessageFromGroup &&
			       eventType != PvdCommStreamEventTypes::EndPropertyMessageGroup;
		case DataStreamState::SetPropertyValue:
			return eventType == PvdCommStreamEventTypes::AppendPropertyValueData ||
			       eventType == PvdCommStreamEventTypes::EndSetPropertyValue;
		case DataStreamState::PropertyMessageGroup:
			return eventType == PvdCommStreamEventTypes::SendPropertyMessageFromGroup ||
			       eventType == PvdCommStreamEventTypes::EndPropertyMessageGroup;
		default:
			return false;
		}
	}

	PvdCommStreamParseResult::Enum handleEventBlock(PxU32 eventCount)
	{
		if(mClosed || !mValid)
			return failParsing();
		for(PxU32 idx = 0; idx < eventCount && mSerializer.isGood() && mValid; ++idx, ++mTotalEventCount)
		{
			PvdCommStreamEventTypes::Enum eventType = PvdCommStreamEventTypes::Unknown;
			mSerializer.streamify(eventType);
			if(!mSerializer.isGood() ||
			   eventType <= PvdCommStreamEventTypes::Unknown ||
			   eventType >= PvdCommStreamEventTypes::Last)
			{
				mValid = false;
				discardIncompletePropertyState();
				return PvdCommStreamParseResult::Failed;
			}
			if(!isEventAllowedInCurrentState(eventType))
				return failParsing();
			if(eventType == PvdCommStreamEventTypes::StreamEndEvent)
			{
				if(idx + 1u != eventCount)
				{
					mValid = false;
					discardIncompletePropertyState();
					return PvdCommStreamParseResult::Failed;
				}
				try
				{
					parseEvent<StreamEndEvent>();
					if(mStreamState != DataStreamState::Open)
						return failParsing();
					handleStreamEndEvent();
					mClosed = true;
				}
				catch(const InvalidSerializedEvent&)
				{
					discardIncompletePropertyState();
					return PvdCommStreamParseResult::Failed;
				}
				return PvdCommStreamParseResult::Success;
			}
			try
			{
				visitCommEventType(eventType);
			}
			catch(const InvalidSerializedEvent&)
			{
				discardIncompletePropertyState();
				return PvdCommStreamParseResult::Failed;
			}
		}
		if(!mValid || !mSerializer.isGood())
		{
			discardIncompletePropertyState();
			return PvdCommStreamParseResult::Failed;
		}
		return PvdCommStreamParseResult::Success;
	}

	// All events need to (possibly) be byte swapped.
	// Only SPV events may need to be byte swapped *and* marshalled.

	struct InvalidSerializedEvent
	{
	};

	template <typename TEventType>
	TEventType parseEvent()
	{
		TEventType evt = TEventType();
		evt.serialize(mSerializer);
		if(!mSerializer.isGood())
		{
			mValid = false;
			throw InvalidSerializedEvent();
		}
		return evt;
	}

	// meta data
	void handleStringHandleEvent()
	{
		StringHandleEvent evt(parseEvent<StringHandleEvent>());
		const char* newStr = mStringTable.registerStr(evt.mString);
		mStringMap[evt.mHandle] = newStr;
	}

	String hdlToString(StringHandle hdl)
	{
		if(hdl.mHandle == 0)
			return "";
		const THandleToStringMap::Entry* entry(mStringMap.find(hdl.mHandle));
		if(!entry)
		{
			mValid = false;
			throw InvalidSerializedEvent();
		}
		return entry->second;
	}

	PxU32 hdlToHdl(const StringHandle& hdl) { return mStringTable.strToHandle(hdlToString(hdl)); }

	String fromStream(const StringHandle& hdl) { return hdlToString(hdl); }

	NamespacedName fromStream(const StreamNamespacedName& nm) { return NamespacedName(fromStream(nm.mNamespace), fromStream(nm.mName)); }

	InstanceDescription fromStream(PxU64 streamHandle)
	{
		const HashMap<PxU64, InstanceDescription>::Entry* entry(mInstanceMap.find(streamHandle));
		if(entry == NULL)
		{
			// PX_ASSERT(false);
			return InstanceDescription();
		}
		return entry->second;
	}

	virtual PxI32 getDBHandle(PxU32 streamId) { return fromStream(static_cast<PxU64>(streamId)).mId; }

	virtual PxI32 getDBHandle(PxU64 streamId) { return fromStream(streamId).mId; }

	template <typename TDataType>
	TDataType* allocTemp(PxU32 numItems)
	{
		PxU32 numBytes = 0;
		if(!pvdapp::input::checkedMultiplyToLimit(numItems, sizeof(TDataType),
			pvdapp::input::kMaxPropertyArrayBytes, numBytes))
			return NULL;
		if(mTempData.size() < numBytes)
		{
			if(!mTempData.growBuf(numBytes - mTempData.size()))
				return NULL;
		}
		return reinterpret_cast<TDataType*>(mTempData.begin());
	}

	void handleCreateClass()
	{
		PX_ASSERT(mStreamState == DataStreamState::Open);
		CreateClass evt(parseEvent<CreateClass>());
		mMetaData.getOrCreateClass(fromStream(evt.mName));
	}

	void handleDeriveClass()
	{
		if(mStreamState != DataStreamState::Open)
		{
			mValid = false;
			return;
		}
		DeriveClass evt(parseEvent<DeriveClass>());
		if(!mMetaData.deriveClass(fromStream(evt.mParent), fromStream(evt.mChild)))
			mValid = false;
	}
	void handleCreateProperty()
	{
		if(mStreamState != DataStreamState::Open)
		{
			mValid = false;
			return;
		}
		CreateProperty evt(parseEvent<CreateProperty>());
		if(evt.mPropertyType != PropertyType::Scalar &&
		   evt.mPropertyType != PropertyType::Array)
		{
			mValid = false;
			return;
		}
		const NamespacedName className = fromStream(evt.mClass);
		const String propertyName = fromStream(evt.mName);
		const String semantic = fromStream(evt.mSemantic);
		const NamespacedName datatypeName = fromStream(evt.mDatatypeName);
		const Option<ClassDescription> ownerClass = mMetaData.findClass(className);
		const Option<ClassDescription> datatype = mMetaData.findClass(datatypeName);
		if(!ownerClass.hasValue() || ownerClass->mLocked || !datatype.hasValue() ||
		   datatype->mClassId == getPvdTypeForType<String>() ||
		   datatype->mRequiresDestruction ||
		   mMetaData.findProperty(ownerClass->mClassId, propertyName).hasValue())
		{
			mValid = false;
			return;
		}
		const Option<PropertyDescription> property = mMetaData.createProperty(
			ownerClass->mClassId, propertyName, semantic, datatype->mClassId,
			evt.mPropertyType);
		if(!property.hasValue())
		{
			mValid = false;
			return;
		}
		PxI32 propId = property->mPropertyId;
		if(evt.mValues.size())
		{
			NamedValue* items = allocTemp<NamedValue>(evt.mValues.size());
			if(!items)
			{
				mValid = false;
				return;
			}
			PVD_FOREACH(idx, evt.mValues.size())
			{
				items[idx].mName = fromStream(evt.mValues[idx].mName);
				items[idx].mValue = evt.mValues[idx].mValue;
			}
			mMetaData.setNamedPropertyValues(DataRef<NamedValue>(items, evt.mValues.size()), propId);
		}
	}

	void handleCreatePropertyMessage()
	{
		if(mStreamState != DataStreamState::Open)
		{
			mValid = false;
			return;
		}
		CreatePropertyMessage evt(parseEvent<CreatePropertyMessage>());
		const NamespacedName ownerName = fromStream(evt.mClass);
		const NamespacedName messageName = fromStream(evt.mMessageName);
		const Option<ClassDescription> ownerClass = mMetaData.findClass(ownerName);
		if(!ownerClass.hasValue() ||
		   mMetaData.findPropertyMessage(messageName).hasValue())
		{
			mValid = false;
			return;
		}
		PxU32 numArgs = evt.mMessageEntries.size();
		PropertyMessageArg* args = allocTemp<PropertyMessageArg>(numArgs);
		if(numArgs && !args)
		{
			mValid = false;
			return;
		}
		bool containsVoidPtr = false;
		for(PxU32 idx = 0; idx < numArgs; ++idx)
		{
			const StreamPropMessageArg& streamArg = evt.mMessageEntries[idx];
			args[idx].mByteSize = streamArg.mByteSize;
			args[idx].mDatatypeName = fromStream(streamArg.mDatatypeName);
			const Option<ClassDescription> datatype =
				mMetaData.findClass(args[idx].mDatatypeName);
			args[idx].mMessageOffset = streamArg.mMessageOffset;
			args[idx].mPropertyName = fromStream(streamArg.mPropertyName);
			const Option<PropertyDescription> property = mMetaData.findProperty(
				ownerClass->mClassId, args[idx].mPropertyName);
			if(!datatype.hasValue() || !property.hasValue() ||
			   !isMessageRangeValid(args[idx].mMessageOffset, args[idx].mByteSize,
			                        evt.mMessageByteSize))
			{
				mValid = false;
				return;
			}
			const PxU32 streamTypeSize =
				datatype->mSizeInfo[m64BitPointers ? 1 : 0].mByteSize;
			if(streamTypeSize == 0 || args[idx].mByteSize < streamTypeSize)
			{
				mValid = false;
				return;
			}
			if(datatype->mClassId == getPvdTypeForType<VoidPtr>())
				containsVoidPtr = true;
			const bool compatibleDatatype =
				datatype->mClassId == property->mDatatype ||
				(datatype->mClassId == getPvdTypeForType<String>() &&
				 property->mDatatype == getPvdTypeForType<StringHandle>()) ||
				(datatype->mClassId == getPvdTypeForType<VoidPtr>() &&
				 property->mDatatype == getPvdTypeForType<ObjectRef>());
			if(property->mPropertyType != PropertyType::Scalar || !compatibleDatatype)
			{
				mValid = false;
				return;
			}
		}
		DataRef<PropertyMessageArg> argRef(args, numArgs);
		const Option<PropertyMessageDescription> descOpt = mMetaData.createPropertyMessage(
			ownerName, messageName, argRef,
			evt.mMessageByteSize);
		if(!descOpt.hasValue())
		{
			mValid = false;
			return;
		}
		PropertyMessageDescription desc = descOpt.getValue();
		if(desc.mStringOffsets.size() || containsVoidPtr)
		{
			if(desc.mProperties.size() != numArgs)
			{
				mValid = false;
				return;
			}
			NamespacedName newName("__debugger__", fromStream(evt.mMessageName.mName));
			if(mMetaData.findPropertyMessage(newName).hasValue())
			{
				mValid = false;
				return;
			}
			for(PxU32 idx = 0; idx < numArgs; ++idx)
			{
				if(desc.mProperties[idx].mDatatypeId == getPvdTypeForType<String>())
					args[idx].mDatatypeName = getPvdNamespacedNameForType<StringHandle>();
				else if(desc.mProperties[idx].mDatatypeId == getPvdTypeForType<VoidPtr>())
					args[idx].mDatatypeName = getPvdNamespacedNameForType<ObjectRef>();
			}
			const Option<PropertyMessageDescription> newDescOpt =
				mMetaData.createPropertyMessage(ownerName, newName,
				                                argRef, evt.mMessageByteSize);
			if(!newDescOpt.hasValue())
			{
				mValid = false;
				return;
			}
			mPropertyMessageMap[desc.mMessageId] = newDescOpt->mMessageId;
		}
		else
		{
			mPropertyMessageMap[desc.mMessageId] = desc.mMessageId;
		}
	}

	// instance editing
	void handleCreateInstance()
	{
		if(mStreamState != DataStreamState::Open)
		{
			mValid = false;
			return;
		}
		CreateInstance evt(parseEvent<CreateInstance>());
		if(mInstanceMap.find(evt.mInstanceId))
		{
			mValid = false;
			return;
		}
		NamespacedName clsName(fromStream(evt.mClass));
		const Option<ClassDescription> cls = mMetaData.findClass(clsName);
		if(!cls.hasValue())
		{
			mValid = false;
			return;
		}
		InstanceDescription desc = mModel.createInstance(cls->mClassId);
		if(!desc.mInstPtr || desc.mId < 0)
		{
			mValid = false;
			return;
		}
		mInstanceMap[evt.mInstanceId] = desc;
		mCommStreamMeta.setInstanceId(evt.mInstanceId, desc.mId);
	}

	// Recursively get the property converters to convert a given piece of incoming data into
	// data we can put into the database.
	bool buildPropertyConverters(PxI32 incomingClassId, PxI32 outgoingClassId,
	                             PxU32 incomingOffset, PxU32 outgoingOffset,
	                             PxU32 depth = 0)
	{
		if(depth > 32)
			return false;
		const ClassDescription* incomingDescPtr = mMetaData.getClassPtr(incomingClassId);
		const ClassDescription* outgoingDescPtr = mMetaData.getClassPtr(outgoingClassId);
		if(!incomingDescPtr || !outgoingDescPtr)
			return false;
		ClassDescription incomingDesc = *incomingDescPtr;
		ClassDescription outgoingDesc = *outgoingDescPtr;
		ClassDescriptionSizeInfo incomingInfo(incomingDesc.get32BitSizeInfo());
		if(m64BitPointers)
			incomingInfo = incomingDesc.get64BitSizeInfo();
		// The database is always 32 bit
		ClassDescriptionSizeInfo outgoingInfo(outgoingDesc.get32BitSizeInfo());

		// In this case we know this isn't a string *and* this isn't an object reference.
		ParserStreamDataConverter* streamConverter = NULL;
		PxU32 localNumItems = 1;
		if(incomingDesc.mPackedClassType >= 0 && outgoingDesc.mPackedClassType >= 0)
		{
			if(incomingDesc.mPackedUniformWidth <= 0 || outgoingDesc.mPackedUniformWidth <= 0)
				return false;
			PxU32 numIncomingItems = incomingInfo.mByteSize / incomingDesc.mPackedUniformWidth;
			PxU32 numOutgoingItems = outgoingInfo.mByteSize / outgoingDesc.mPackedUniformWidth;
			if(numIncomingItems == 0 || numOutgoingItems == 0 ||
			   PxMax(numIncomingItems, numOutgoingItems) %
			       PxMin(numIncomingItems, numOutgoingItems) != 0)
				return false;
			streamConverter = mConverterFactory.getConverter(incomingDesc.mPackedClassType, outgoingDesc.mPackedClassType, incomingDesc.mPackedUniformWidth);
			localNumItems = numIncomingItems;
		}
		// Special treatment for stream ids
		else if(incomingClassId == getPvdTypeForType<ObjectRef>() || incomingClassId == getPvdTypeForType<VoidPtr>())
		{
			if(outgoingClassId != getPvdTypeForType<ObjectRef>() && outgoingClassId != getPvdTypeForType<VoidPtr>())
				return false;
			streamConverter = mConverterFactory.getConverter(getPvdTypeForType<VoidPtr>(), getPvdTypeForType<ObjectRef>(), m64BitPointers ? 8 : 4);
		}
		// Special treatment for strings
		else if(incomingClassId == getPvdTypeForType<String>() || incomingClassId == getPvdTypeForType<StringHandle>())
		{
			if(outgoingClassId != getPvdTypeForType<StringHandle>() && outgoingClassId != getPvdTypeForType<String>())
				return false;
			streamConverter = mConverterFactory.getConverter(getPvdTypeForType<String>(), getPvdTypeForType<StringHandle>(), m64BitPointers ? 8 : 4);
		}
		if(streamConverter != NULL)
		{
			mCurrentConverters.pushBack(streamConverter);
			ConverterAndData& converter = mCurrentConverters.back();
			converter.mData.incomingOffset = incomingOffset;
			converter.mData.outgoingOffset = outgoingOffset;
			converter.mData.numItems = localNumItems;
			return true;
		}
		else // go property by property and get the matching properties
		{
			PropertyDescription incomingProps[5];
			PropertyDescription outgoingProps[5];
			PxU32 numProps = mMetaData.getNbProperties(incomingClassId);
			if(outgoingClassId != incomingClassId)
			{
				PxU32 numOutgoing = mMetaData.getNbProperties(outgoingClassId);
				if(numProps != numOutgoing)
					return false;
			}
			PxU32 propBatchIdx = 0;
			while(propBatchIdx < numProps)
			{
				PxU32 received = mMetaData.getProperties(incomingClassId, incomingProps, 5, propBatchIdx);
				if(!received)
					return false;
				if(incomingClassId != outgoingClassId)
				{
					if(mMetaData.getProperties(outgoingClassId, outgoingProps, 5,
					                           propBatchIdx) != received)
						return false;
				}
				else
					physx::intrinsics::memCopy(outgoingProps, incomingProps, sizeof(outgoingProps));
				propBatchIdx += received;
				for(PxU32 propIdx = 0; propIdx < received; ++propIdx)
				{
					const PropertyDescription& incomingProp(incomingProps[propIdx]);
					const PropertyDescription& outgoingProp(outgoingProps[propIdx]);
					PxU32 newIncomingOffset = incomingProp.m32BitOffset;
					if(m64BitPointers)
						newIncomingOffset = incomingProp.m64BitOffset;
					if(newIncomingOffset > PX_MAX_U32 - incomingOffset ||
					   outgoingProp.m32BitOffset > PX_MAX_U32 - outgoingOffset ||
					   !buildPropertyConverters(
						   incomingProp.mDatatype, outgoingProp.mDatatype,
						   incomingOffset + newIncomingOffset,
						   outgoingOffset + outgoingProp.m32BitOffset, depth + 1))
						return false;
				}
			}
		}
		return true;
	}
	bool countStreamStrings(PxI32 classId, PxU32 depth, PxU32& count)
	{
		if(depth > 32)
			return false;
		if(classId == getPvdTypeForType<String>() ||
		   classId == getPvdTypeForType<StringHandle>())
		{
			if(count == PX_MAX_U32)
				return false;
			++count;
			return true;
		}
		const ClassDescription* cls = mMetaData.getClassPtr(classId);
		if(!cls)
			return false;
		PropertyDescription properties[16];
		const PxU32 propertyCount = mMetaData.getNbProperties(classId);
		for(PxU32 offset = 0; offset < propertyCount;)
		{
			const PxU32 received = mMetaData.getProperties(
				classId, properties, 16, offset);
			if(!received)
				return false;
			offset += received;
			for(PxU32 idx = 0; idx < received; ++idx)
			{
				if(!countStreamStrings(properties[idx].mDatatype, depth + 1, count))
					return false;
			}
		}
		return true;
	}

	bool validateIncomingPropertyData(PxI32 classId, PxU32 inputStride,
	                                  PxU32 numItems,
	                                  DataRef<const PxU8> propData)
	{
		if(inputStride == 0 || numItems > pvdapp::input::kMaxPropertyItems)
			return false;
		PxU32 fixedBytes = 0;
		if(!pvdapp::input::checkedMultiplyToLimit(
			numItems, inputStride, propData.size(), fixedBytes))
			return false;

		PxU32 stringsPerItem = 0;
		if(!countStreamStrings(classId, 0, stringsPerItem))
			return false;
		if(stringsPerItem == 0)
			return fixedBytes == propData.size();

		const PxU8* begin = propData.begin();
		const PxU8* end = propData.end();
		const PxU8* cursor = begin;
		for(PxU32 item = 0; item < numItems; ++item)
		{
			if(static_cast<size_t>(end - cursor) < inputStride)
				return false;
			cursor += inputStride;
			for(PxU32 stringIdx = 0; stringIdx < stringsPerItem; ++stringIdx)
			{
				const void* terminator = std::memchr(
					cursor, 0, static_cast<size_t>(end - cursor));
				if(!terminator)
					return false;
				cursor = static_cast<const PxU8*>(terminator) + 1;
			}
		}
		return cursor == end;
	}

	// Anything that sets a property value needs to do several things.
	// 1.  Endian convert all the data.
	// 2.  Ptr-to-db-handle conversion.  Streams send us void* ptrs
	//	and we need to convert those void* ptrs to db handles (if possible).  db handles are
	//	32 bit but streams can be 32 or 64 bit, so in some cases the size of the output data
	//	is different than the size of the input data.
	//
	// 3.  Convert const char* data into property string handles by registering
	//	it with the meta data system.
	//
	// 4.  Parse any special embedded data we know about, mainly to endian convert
	//	the entirety of it.
	//
	// 5.  Set the property value
	bool doSetPropertyValue(PxU64 instanceId, const StringHandle& propName, DataRef<const PxU8> propData, const StreamNamespacedName& incomingTypeName, PxU32 numItems)
	{
		InstanceDescription desc;
		const HashMap<PxU64, InstanceDescription>::Entry* entry(mInstanceMap.find(instanceId));
		if(entry == NULL)
		{
			PX_ASSERT(false);
			return false;
		}
		desc = entry->second;
		ClassDescription cls(mModel.getClassOf(desc));
		Option<PropertyDescription> propOpt(mMetaData.findProperty(cls.mClassId, fromStream(propName)));
		if(propOpt.hasValue() == false)
		{
			PX_ASSERT(false);
			return false;
		}
		PropertyDescription& prop(propOpt);
		const ClassDescription* propClsPtr = mMetaData.getClassPtr(prop.mDatatype);
		const Option<ClassDescription> incomingTypeOpt =
			mMetaData.findClass(fromStream(incomingTypeName));
		if(!propClsPtr || !incomingTypeOpt.hasValue())
			return false;
		ClassDescription propCls = *propClsPtr;
		ClassDescription incomingType = incomingTypeOpt.getValue();

		// PxU32 numProps = mMetaData.getNbProperties( propCls.mClassId );
		// bool swapBytes = mSerializer.performsSwap();
		ClassDescriptionSizeInfo incomingInfo(incomingType.get32BitSizeInfo());
		if(m64BitPointers)
			incomingInfo = incomingType.get64BitSizeInfo();
		ClassDescriptionSizeInfo outgoingInfo(propCls.get32BitSizeInfo());

		PxU32 inputStride = incomingInfo.mByteSize;
		if((prop.mPropertyType != PropertyType::Array && numItems != 1) ||
		   !validateIncomingPropertyData(incomingType.mClassId, inputStride,
		                                numItems, propData))
			return false;

		PxU32 numOutputBytes = 0;
		if(!pvdapp::input::checkedMultiplyToLimit(
			numItems, outgoingInfo.mByteSize,
			pvdapp::input::kMaxPropertyArrayBytes, numOutputBytes))
			return false;
		if(incomingType.mPackedUniformWidth >= 0 && propCls.mPackedUniformWidth >= 0)
		{
			if(incomingType.mPackedUniformWidth == 0 || propCls.mPackedUniformWidth == 0 ||
			   propData.size() % incomingType.mPackedUniformWidth)
				return false;
			PxU32 realNumItems = propData.size() / incomingType.mPackedUniformWidth;
			PxU32 packedOutputBytes = 0;
			if(!pvdapp::input::checkedMultiplyToLimit(
				realNumItems, propCls.mPackedUniformWidth,
				pvdapp::input::kMaxPropertyArrayBytes, packedOutputBytes) ||
			   packedOutputBytes != numOutputBytes)
				return false;
		}

		mCurrentConverters.clear();
		if(!buildPropertyConverters(incomingType.mClassId, propCls.mClassId, 0, 0))
			return false;
		if(numOutputBytes == 0)
			return mModel.setPropertyValue(desc, prop.mPropertyId, NULL, 0,
			                               propCls.mClassId);
		if(mCurrentConverters.empty() && numOutputBytes != propData.size())
			return false;
		PxU8* inputPtr = const_cast<PxU8*>(propData.begin());

		PxU32 numConverters = mCurrentConverters.size();
		DataRef<const PxU8> spvData(propData.begin(), propData.size());
		// Strings are stored after each item, so the first string start has to be
		// at the end of the item.
		// After all the converters have run, the next item must start where the string
		// ptr is after all strings have been read out of the dataset.
		ConverterSrcData srcData(inputPtr, const_cast<PxU8*>(propData.end()), inputStride);

		// The code below looks duplicated but it isn't.  In the case where
		// the output matches the input, we don't need to create a new buffer.  This can
		// significantly decrease memory traffic in the case of large buffers.
		// In the case where we had to create a new buffer, then we have different conversion
		// methods to call that will copy data into the newer buffer.
		// Also, if there are strings stored in the message then we need to copy
		// the data into a new buffer that fits exactly the message size.
		if(numOutputBytes != propData.size())
		{
			PxU8* outputPtr = allocTemp<PxU8>(numOutputBytes);
			if(numOutputBytes && !outputPtr)
				return false;
			spvData = DataRef<const PxU8>(outputPtr, numOutputBytes);
			if(numConverters == 1)
				mCurrentConverters[0].ConvertBlock(srcData, outputPtr);
			else
			{
				// Unless our output size different than our input size,
				// which happens when a 64 bit stream is sending us heterogeneous arrays
				// which contain void* members.
				PxU32 outputStride = outgoingInfo.mByteSize;
				for(PxU32 idx = 0; idx < numItems; ++idx, outputPtr += outputStride)
				{
					for(PxU32 convertIdx = 0; convertIdx < numConverters; ++convertIdx)
						mCurrentConverters[convertIdx].Convert(srcData, outputPtr);

					// After all the converters have run for the current item,
					// reset the src data such that is is valid again.
					srcData.src = srcData.srcStringStart;
					srcData.srcStringStart = srcData.src + inputStride;
				}
			}
		}
		else
		{
			if(numConverters == 1)
				mCurrentConverters[0].ConvertBlock(srcData);
			else
			{
				for(PxU32 idx = 0; idx < numItems; ++idx, inputPtr += inputStride)
				{
					for(PxU32 convertIdx = 0; convertIdx < numConverters; ++convertIdx)
						mCurrentConverters[convertIdx].Convert(srcData);

					// After all the converters have run for the current item,
					// reset the src data such that is is valid again.
					srcData.src = srcData.srcStringStart;
					srcData.srcStringStart = srcData.src + inputStride;
				}
			}
		}
		// Data may be tunneled through these arrays.
		// In this case we know that
		if(prop.mSemantic && prop.mPropertyType == PropertyType::Array)
		{
			if(physx::shdfnd::strcmp(PvdCommStreamEmbeddedTypes::getProfileEventStreamSemantic(), prop.mSemantic) == 0)
			{
				if(mSerializer.performsSwap())
				{
					mTempData.clear();
					mTempData.reserve(propData.size() * 2);

					TSerializerType theSerializer(&mTempData);
					TDeserializerType theDeserializer(propData.begin(), propData.size());
					EventHeader theHeader;
					Event::EventData theEventData;
					ProfileEventPassthrough theParseOp(theSerializer, theDeserializer, theHeader);
					while(theDeserializer.mFail == false && theDeserializer.mLength)
					{
						theHeader.streamify(theDeserializer);
						if(theDeserializer.mFail == false)
							visit<bool>(static_cast<EventTypes::Enum>(theHeader.mEventType), theEventData, theParseOp);
					}
					spvData = DataRef<const PxU8>(mTempData.begin(), mTempData.size());
				}
			}
			else if(physx::shdfnd::strcmp(PvdCommStreamEmbeddedTypes::getMemoryEventStreamSemantic(), prop.mSemantic) == 0)
			{
				if(mMemoryEventParser == NULL)
				{
					if(mSerializer.performsSwap())
						mMemoryEventParser = PVD_NEW(SMemoryEventContext<true>)(mCommStreamMeta, mModel, desc);
					else
						mMemoryEventParser = PVD_NEW(SMemoryEventContext<false>)(mCommStreamMeta, mModel, desc);
				}
				spvData = mMemoryEventParser->handleEventData(spvData);
			}
			else if(physx::shdfnd::strcmp(PvdCommStreamEmbeddedTypes::getRendererEventStreamSemantic(), prop.mSemantic) == 0)
			{
				if(mRenderTransfer == NULL)
				{
					mRenderTransfer = PVD_NEW(PvdUserRendererTransfer)(mCommStreamMeta.getMetaData(), mSerializer.performsSwap());
				}
				spvData = mRenderTransfer->handleData(spvData);
			}
		}
		return mModel.setPropertyValue(desc, prop.mPropertyId, spvData.begin(),
		                               spvData.size(), propCls.mClassId);
	}

	void handleSetPropertyValue()
	{
		if(mStreamState != DataStreamState::Open)
		{
			mValid = false;
			return;
		}
		SetPropertyValue evt(parseEvent<SetPropertyValue>());
		if(!doSetPropertyValue(evt.mInstanceId, evt.mPropertyName, evt.mData,
		                       evt.mIncomingTypeName, evt.mNumItems))
			mValid = false;
	}

	void handleBeginSetPropertyValue()
	{
		if(mStreamState != DataStreamState::Open)
		{
			mValid = false;
			return;
		}
		mSPVEvent = parseEvent<BeginSetPropertyValue>();
		if(!mSessionBudget.tryBeginPropertyState())
		{
			mValid = false;
			return;
		}
		mSPVData = PVD_NEW(ForwardingMemoryBuffer)("ParserStream::mSPVData");
		if(!mSPVData)
		{
			mSessionBudget.endPropertyState();
			mValid = false;
			return;
		}
		mStreamState = DataStreamState::SetPropertyValue;
		mSPVNumItems = 0;
	}

	void handleAppendPropertyValueData()
	{
		if(mStreamState != DataStreamState::SetPropertyValue)
		{
			mValid = false;
			return;
		}
		AppendPropertyValueData evt = parseEvent<AppendPropertyValueData>();
		const bool retained = mSessionBudget.tryRetainPropertyBytes(evt.mData.size());
		if(!mSPVData ||
		   !pvdapp::input::canAccumulatePropertyData(
			mSPVData ? mSPVData->size() : 0, mSPVNumItems,
			evt.mData.size(), evt.mNumItems) ||
		   !retained ||
		   mSPVData->write(evt.mData.begin(), evt.mData.size()) != evt.mData.size())
		{
			if(retained)
				mSessionBudget.releasePropertyBytes(evt.mData.size());
			discardIncompletePropertyState();
			mValid = false;
			return;
		}
		mSPVNumItems += evt.mNumItems;
	}

	void handleEndSetPropertyValue()
	{
		if(mStreamState != DataStreamState::SetPropertyValue)
		{
			mValid = false;
			return;
		}
		EndSetPropertyValue evt = parseEvent<EndSetPropertyValue>();
		const bool success = mSPVData && doSetPropertyValue(
			mSPVEvent.mInstanceId, mSPVEvent.mPropertyName,
			DataRef<const PxU8>(mSPVData->begin(), mSPVData->size()),
			mSPVEvent.mIncomingTypeName, mSPVNumItems);
		discardIncompletePropertyState();
		if(!success)
			mValid = false;
	}

	bool buildPropertyMessageConverters(const PropertyMessageDescription& incomingMsg)
	{
		mCurrentConverters.clear();
		PVD_FOREACH(idx, incomingMsg.mProperties.size())
		{
			const PropertyMessageEntry& entry(incomingMsg.mProperties[idx]);
			if(!buildPropertyConverters(entry.mDatatypeId, entry.mDatatypeId,
			                            entry.mMessageOffset, 0))
				return false;
		}
		return true;
	}

	static bool isMessageRangeValid(PxU32 offset, PxU32 length, PxU32 total)
	{
		return offset <= total && length <= total - offset;
	}

	bool validatePropertyMessageData(const PropertyMessageDescription& incomingMsg,
	                                 DataRef<const PxU8> data,
	                                 bool grouped) const
	{
		if(incomingMsg.mMessageId < 0 || data.size() < incomingMsg.mMessageByteSize)
			return false;
		if(grouped && (data.size() != incomingMsg.mMessageByteSize ||
		               incomingMsg.mStringOffsets.size() != 0))
			return false;

		PVD_FOREACH(idx, incomingMsg.mProperties.size())
		{
			const PropertyMessageEntry& entry = incomingMsg.mProperties[idx];
			const ClassDescription* datatype = mMetaData.getClassPtr(entry.mDatatypeId);
			if(!datatype)
				return false;
			const ClassDescriptionSizeInfo& streamInfo =
				datatype->mSizeInfo[m64BitPointers ? 1 : 0];
			if(streamInfo.mByteSize == 0 ||
			   !isMessageRangeValid(entry.mMessageOffset, streamInfo.mByteSize,
			                        incomingMsg.mMessageByteSize) ||
			   !isMessageRangeValid(entry.mMessageOffset, entry.mByteSize,
			                        incomingMsg.mMessageByteSize))
				return false;
		}

		if(incomingMsg.mStringOffsets.size() == 0)
			return true;
		if(!data.begin())
			return false;

		const PxU32 streamPointerSize = m64BitPointers ? 8u : 4u;
		const PxU8* cursor = data.begin() + incomingMsg.mMessageByteSize;
		const PxU8* end = data.end();
		PVD_FOREACH(idx, incomingMsg.mStringOffsets.size())
		{
			if(!isMessageRangeValid(incomingMsg.mStringOffsets[idx], streamPointerSize,
			                        incomingMsg.mMessageByteSize))
				return false;
			const void* terminator = std::memchr(
				cursor, 0, static_cast<size_t>(end - cursor));
			if(!terminator)
				return false;
			cursor = static_cast<const PxU8*>(terminator) + 1;
		}
		return cursor == end;
	}

	bool doSetPropertyMessage(const PropertyMessageDescription& incomingMsg,
	                          PxI32 msgId, PxU64 instanceId,
	                          DataRef<const PxU8> data, bool grouped)
	{
		if(!validatePropertyMessageData(incomingMsg, data, grouped))
			return false;
		InstanceDescription desc;
		{
			const HashMap<PxU64, InstanceDescription>::Entry* entry(mInstanceMap.find(instanceId));
			if(entry == NULL)
			{
				PX_ASSERT(false);
				return false;
			}
			desc = entry->second;
		}
		if(!buildPropertyMessageConverters(incomingMsg))
			return false;
		if(mCurrentConverters.size())
		{
			if(!data.begin())
				return false;
			PxU8* srcPtr = const_cast<PxU8*>(data.begin());
			PxU8* srcEnd = srcPtr + data.size();
			ConverterSrcData srcData(srcPtr, srcEnd, incomingMsg.mMessageByteSize);
			PVD_FOREACH(idx, mCurrentConverters.size())
				mCurrentConverters[idx].Convert(srcData);
		}
		return mModel.setPropertyMessage(desc, msgId, data.begin(), data.size());
	}

	void handleSetPropertyMessage()
	{
		if(mStreamState != DataStreamState::Open)
		{
			mValid = false;
			return;
		}
		SetPropertyMessage evt = parseEvent<SetPropertyMessage>();
		const Option<PropertyMessageDescription> incomingMsgOpt =
			mMetaData.findPropertyMessage(fromStream(evt.mMessageName));
		if(!incomingMsgOpt.hasValue())
		{
			mValid = false;
			return;
		}
		PropertyMessageDescription incomingMsg = incomingMsgOpt.getValue();
		PxI32 msgId = incomingMsg.mMessageId;
		const HashMap<PxI32, PxI32>::Entry* entry = mPropertyMessageMap.find(incomingMsg.mMessageId);
		if(entry)
			msgId = entry->second;
		if(!doSetPropertyMessage(incomingMsg, msgId, evt.mInstanceId, evt.mData, false))
			mValid = false;
	}

	void handleBeginPropertyMessageGroup()
	{
		if(mStreamState != DataStreamState::Open)
		{
			mValid = false;
			return;
		}
		BeginPropertyMessageGroup evt = parseEvent<BeginPropertyMessageGroup>();
		const Option<PropertyMessageDescription> incomingMsgOpt =
			mMetaData.findPropertyMessage(fromStream(evt.mMsgName));
		if(!incomingMsgOpt.hasValue() || incomingMsgOpt->mStringOffsets.size())
		{
			mValid = false;
			return;
		}
		PropertyMessageDescription incomingMsg = incomingMsgOpt.getValue();
		PxI32 msgId = incomingMsg.mMessageId;
		const HashMap<PxI32, PxI32>::Entry* entry =
			mPropertyMessageMap.find(incomingMsg.mMessageId);
		if(entry)
			msgId = entry->second;
		mGroupPropertyMessage = incomingMsg;
		mGroupMessageId = msgId;
		mStreamState = DataStreamState::PropertyMessageGroup;
	}
	void handleSendPropertyMessageFromGroup()
	{
		if(mStreamState != DataStreamState::PropertyMessageGroup)
		{
			mValid = false;
			return;
		}
		SendPropertyMessageFromGroup evt = parseEvent<SendPropertyMessageFromGroup>();
		if(!doSetPropertyMessage(mGroupPropertyMessage, mGroupMessageId,
		                         evt.mInstance, evt.mData, true))
			mValid = false;
	}
	void handleEndPropertyMessageGroup()
	{
		if(mStreamState != DataStreamState::PropertyMessageGroup)
		{
			mValid = false;
			return;
		}
		EndPropertyMessageGroup evt = parseEvent<EndPropertyMessageGroup>();
		mStreamState = DataStreamState::Open;
	}

	void handleDestroyInstance()
	{
		PX_ASSERT(mStreamState == DataStreamState::Open);
		DestroyInstance evt(parseEvent<DestroyInstance>());
		InstanceDescription desc;
		const HashMap<PxU64, InstanceDescription>::Entry* entry(mInstanceMap.find(evt.mInstanceId));
		if(entry == NULL)
		{
			PX_ASSERT(false);
			return;
		}
		desc = entry->second;
		mModel.destroyInstance(desc);
		mInstanceMap.erase(evt.mInstanceId);
		mCommStreamMeta.destroyInstance(evt.mInstanceId);
	}

	void handlePushBackObjectRef()
	{
		PX_ASSERT(mStreamState == DataStreamState::Open);
		PushBackObjectRef evt(parseEvent<PushBackObjectRef>());
		void* srcInst = fromStream(evt.mInstanceId);
		ObjectRef objRef = fromStream(evt.mObjectRef).mId;
		mModel.pushBackArrayElementIf(srcInst, hdlToString(evt.mProperty), objRef);
	}
	void handleRemoveObjectRef()
	{
		PX_ASSERT(mStreamState == DataStreamState::Open);
		RemoveObjectRef evt(parseEvent<RemoveObjectRef>());
		// It is not guarantee that msg received has the same order as sdk sent
		// if(mInstanceMap.find( evt.mInstanceId ) == NULL)
		//	return;

		void* srcInst = fromStream(evt.mInstanceId);
		ObjectRef objRef(fromStream(evt.mObjectRef).mId);
		mModel.removeArrayElementIf(srcInst, hdlToString(evt.mProperty), objRef);
	}

	// section system
	void handleBeginSection()
	{
		PX_ASSERT(mStreamState == DataStreamState::Open);
		BeginSection evt(parseEvent<BeginSection>());
		mCommStreamMeta.addSectionEvent(evt.mSectionId, hdlToHdl(evt.mName), true, evt.mTimestamp);
	}
	void handleEndSection()
	{
		PX_ASSERT(mStreamState == DataStreamState::Open);
		EndSection evt(parseEvent<EndSection>());
		mCommStreamMeta.addSectionEvent(evt.mSectionId, hdlToHdl(evt.mName), false, evt.mTimestamp);
	}
	void handleSetPickable()
	{
		PX_ASSERT(mStreamState == DataStreamState::Open);
		SetPickable evt(parseEvent<SetPickable>());
		mCommStreamMeta.setPickable(evt.mInstanceId, evt.mPickable);
	}
	void handleSetColor()
	{
		PX_ASSERT(mStreamState == DataStreamState::Open);
		SetColor evt(parseEvent<SetColor>());
		mCommStreamMeta.setColor(evt.mInstanceId, evt.mColor);
	}
	void handleSetIsTopLevel()
	{
		PX_ASSERT(mStreamState == DataStreamState::Open);
		SetIsTopLevel evt(parseEvent<SetIsTopLevel>());
		mCommStreamMeta.setIsTopLevel(evt.mInstanceId, evt.mIsTopLevel);
	}
	void handleSetCamera()
	{
		PX_ASSERT(mStreamState == DataStreamState::Open);
		SetCamera evt(parseEvent<SetCamera>());
		mCommStreamMeta.setCamera(mStringTable.strToHandle(evt.mName), evt.mPosition, evt.mUp, evt.mTarget);
	}
	void handleAddProfileZone()
	{
		PX_ASSERT(mStreamState == DataStreamState::Open);
		AddProfileZone evt(parseEvent<AddProfileZone>());
		mCommStreamMeta.addProfileZone(evt.mInstanceId, evt.mName);
	}
	void handleAddProfileZoneEvent()
	{
		PX_ASSERT(mStreamState == DataStreamState::Open);
		AddProfileZoneEvent evt(parseEvent<AddProfileZoneEvent>());
		mCommStreamMeta.addProfileZoneEvent(evt.mInstanceId, evt.mName, evt.mEventId, evt.mCompileTimeEnabled);
	}
	void handleStreamEndEvent() { PX_ASSERT(mStreamState == DataStreamState::Open); }

	void handleErrorMessage()
	{
		ErrorMessage evt(parseEvent<ErrorMessage>());

		mCommStreamMeta.sendErrorMessage(evt.mCode, mStringTable.strToHandle(evt.mMessage), mStringTable.strToHandle(evt.mFile), evt.mLine);
	}

	void handleOriginShift()
	{
		if(mStreamState != DataStreamState::Open)
		{
			mValid = false;
			return;
		}
		OriginShift evt(parseEvent<OriginShift>());
		// mCommStreamMeta.setCamera( mStringTable.strToHandle( evt.mName ), evt.mPosition, evt.mUp, evt.mTarget );
		void* srcInst = fromStream(evt.mInstanceId);
		if(!srcInst)
		{
			mValid = false;
			return;
		}
		mModel.originShift(srcInst, evt.mShift);
	}
};

// Performs demultiplexing data out of the stream,
// performs all the byte swapping and marshalling of
// the stream data.
struct Parser : public PvdCommStreamParser
{
	PxAllocatorCallback& mAllocator;

	PvdObjectModelMetaData& mMetaData;
	PvdObjectModelMutator& mModel;
	PvdCommStreamMetaDataMutator& mCommStreamMeta;
	ExtendedEventSerializer& mSerializer;
	bool m64BitPointers;

	THandleToStringMap mStringMap;
	PxU64 mTotalEventCount;
	HashMap<PxI32, PxI32> mPropMessageMap;

	HashMap<PxU64, InstanceDescription> mInstances;
	ForwardingMemoryBuffer mTempData;
	HashMap<PxU64, ParserStream*> mStreams;
	ParserSessionBudget mSessionBudget;

	Parser(PxAllocatorCallback& alloc, PvdObjectModelMetaData& metaData, PvdObjectModelMutator& model, PvdCommStreamMetaDataMutator& commStreamMeta, ExtendedEventSerializer& serializer,
	       bool _64BitPointers)
	: mAllocator(alloc)
	, mMetaData(metaData)
	, mModel(model)
	, mCommStreamMeta(commStreamMeta)
	, mSerializer(serializer)
	, m64BitPointers(_64BitPointers)
	, mStringMap("Parser::mStringMap")
	, mTotalEventCount(0)
	, mPropMessageMap("Parser::mPropMessageMap")
	, mInstances("Parser::mInstances")
	, mTempData("Parser::mTempData")
	, mStreams("Parser::mStreams")
	{
		mMetaData.addRef();
		mModel.addRef();
		mCommStreamMeta.addRef();
	}
	virtual ~Parser()
	{
		for(HashMap<PxU64, ParserStream*>::Iterator iter = mStreams.getIterator(); iter.done() == false; ++iter)
			PVD_DELETE(iter->second);
		mMetaData.release();
		mModel.release();
		mCommStreamMeta.release();
		mSerializer.release();
	}

	ParserStream* getOrCreateParser(PxU64 streamId)
	{
		const HashMap<PxU64, ParserStream*>::Entry* entry = mStreams.find(streamId);
		if(entry)
			return entry->second;
		if(!mSessionBudget.tryAddParserStream())
			return NULL;
		ParserStream* newStream = PVD_NEW(ParserStream)(mAllocator, mMetaData, mModel, mCommStreamMeta, mStringMap, mTotalEventCount, mPropMessageMap, mInstances, mSerializer, m64BitPointers, mSessionBudget, mTempData);
		if(!newStream)
		{
			mSessionBudget.removeParserStream();
			return NULL;
		}
		mStreams.insert(streamId, newStream);
		return newStream;
	}

	virtual PvdCommStreamParseResult::Enum parseNextEventGroupResult(DataRef<const PxU8> eventData)
	{
		//#ifdef PROFILING
		//    profiler::ProfilerScope prof_scope("Parser::parseNextEventGroup()", 1);
		//#endif
		EventGroup evt;
		mSerializer.setData(eventData);
		evt.serialize(mSerializer);
		if(!mSerializer.isGood())
			return PvdCommStreamParseResult::Failed;
		ParserStream* stream = getOrCreateParser(evt.mStreamId);
		if(!stream)
			return PvdCommStreamParseResult::Failed;
		const PvdCommStreamParseResult::Enum parseResult =
			stream->handleEventBlock(evt.mNumEvents);
		if(parseResult == PvdCommStreamParseResult::Failed)
			return parseResult;

		// A group is well formed only when the declared event count consumes the
		// complete payload. Probe one byte: exact exhaustion marks the serializer
		// bad, while any successful read proves undeclared trailing data exists.
		PxU8 trailingByte = 0;
		mSerializer.streamify(trailingByte);
		if(mSerializer.isGood())
			return stream->failParsing();
		return PvdCommStreamParseResult::Success;
	}
	virtual bool parseNextEventGroup(DataRef<const PxU8> eventData)
	{
		return parseNextEventGroupResult(eventData) == PvdCommStreamParseResult::Success;
	}
	virtual PvdCommStreamParseResult::Enum finalizeParsingResult()
	{
		bool valid = true;
		for(HashMap<PxU64, ParserStream*>::Iterator iter = mStreams.getIterator();
		    iter.done() == false; ++iter)
		{
			if(iter->second->finalizeParsing() == PvdCommStreamParseResult::Failed)
				valid = false;
		}
		return valid ? PvdCommStreamParseResult::Success
		             : PvdCommStreamParseResult::Failed;
	}

	virtual void release() { PVD_DELETE(this); }
};

struct EventBasiceDeserializer : public PvdEventSerializer
{
	virtual void streamify(String&) { PX_ASSERT(false); }
	virtual void streamify(DataRef<const PxU8>&) { PX_ASSERT(false); }
	virtual void streamify(DataRef<NameHandleValue>&) { PX_ASSERT(false); }
	virtual void streamify(DataRef<StreamPropMessageArg>&) { PX_ASSERT(false); }
	virtual void streamify(DataRef<StringHandle>&) { PX_ASSERT(false); }
};

struct EventBufferingDeserializer : public EventBasiceDeserializer
{
	PvdOutputStream& mResult;
	PvdInputStream& mInputStream;
	bool mSwapBytes;
	EventBufferingDeserializer(PvdInputStream& inStream, PvdOutputStream& outResult, bool swapBytes) : mResult(outResult), mInputStream(inStream), mSwapBytes(swapBytes) {}
	template <typename TDataType>
	void readWrite(TDataType& dt)
	{
		mInputStream >> dt;
		mResult << dt;
		if(mSwapBytes)
			swapBytes(dt);
	}
	virtual void streamify(PxU16& val) { readWrite(val); }
	virtual void streamify(PxU8& val) { readWrite(val); }
	virtual void streamify(PxU32& val) { readWrite(val); }
	virtual void streamify(PxF32& val) { readWrite(val); }
	virtual void streamify(PxU64& val) { readWrite(val); }
};

struct EventReader : public PvdCommStreamEventReader
{
	PxAllocatorCallback& mAllocator;
	StreamInitialization mStreamInit;
	bool mSwapBytes;
	PxU64 mEventCount;

	EventReader(PxAllocatorCallback& alloc, const StreamInitialization& init, bool swapBytes) : mAllocator(alloc), mStreamInit(init), mSwapBytes(swapBytes), mEventCount(0) {}

	// Read the next event and place the data, unchanged or endian converted, into result.
	virtual PvdCommStreamReadResult::Enum readNextEventGroupResult(PvdInputStream& inStream, PvdOutputStream& result)
	{
		EventGroup evt;
		EventBufferingDeserializer stream(inStream, result, mSwapBytes);
		evt.serialize(stream);
		if(!pvdapp::input::isEventGroupDataSizeAllowed(evt.mDataSize))
			return PvdCommStreamReadResult::Failed;
		bool retval = result.directCopy(inStream, evt.mDataSize);
		++mEventCount;
		return retval && evt.mNumEvents
			? PvdCommStreamReadResult::Success
			: PvdCommStreamReadResult::Failed;
	}
	virtual bool readNextEventGroup(PvdInputStream& inStream, PvdOutputStream& result)
	{
		return readNextEventGroupResult(inStream, result) == PvdCommStreamReadResult::Success;
	}
	virtual bool isSwappingBytes() { return mSwapBytes; }
	virtual void release() { PVD_DELETE(this); }
	virtual PvdCommStreamParser& createParser(PvdObjectModelMetaData& metaData, PvdObjectModelMutator& model, PvdCommStreamMetaDataMutator& commStreamMeta)
	{
		commStreamMeta.setTimestampNanosecondConversion(mStreamInit.mTimestampNumerator, mStreamInit.mTimestampDenominator);
		ExtendedEventSerializer& serializer = ExtendedEventSerializer::createInputSerializer(mSwapBytes);
		bool pointers = mStreamInit.mStreamFlags & CommStreamFlagTypes::Is64BitPtr;
		return *PVD_NEW(Parser)(mAllocator, metaData, model, commStreamMeta, serializer, pointers);
	}
};
} // namespace

struct StreamInitDeserializer : EventBasiceDeserializer
{
	PvdInputStream& mInputStream;
	PxU32 mFirstWord;
	bool mFirstWordRead;
	bool mSwapBytes;
	bool mGood;

	StreamInitDeserializer(PvdInputStream& inStream, PxU32 firstWord, bool swap)
	: mInputStream(inStream)
	, mFirstWord(firstWord)
	, mFirstWordRead(false)
	, mSwapBytes(swap)
	, mGood(true)
	{
	}

	template <typename TDataType>
	void tread(TDataType& dt)
	{
		if(!mGood)
		{
			dt = TDataType();
			return;
		}
		PxU32 readLength = static_cast<PxU32>(sizeof(TDataType));
		if(!mInputStream.read(reinterpret_cast<PxU8*>(&dt), readLength) ||
		   readLength != static_cast<PxU32>(sizeof(TDataType)))
		{
			dt = TDataType();
			mGood = false;
			return;
		}
		if(mSwapBytes)
			swapBytes(dt);
	}
	void read(PxU32& dt)
	{
		if(mFirstWordRead)
		{
			tread(dt);
		}
		else
		{
			dt = mFirstWord;
			mFirstWordRead = true;
			if(mSwapBytes)
				swapBytes(dt);
		}
	}
	bool isGood() const { return mGood; }
	virtual void streamify(PxU8& val) { tread(val); }
	virtual void streamify(PxU16& val) { tread(val); }
	virtual void streamify(PxU32& val) { read(val); }
	virtual void streamify(PxF32& val) { tread(val); }
	virtual void streamify(PxU64& val) { tread(val); }
};

struct IsPvdStreamQueryResult
{
	bool isPvdStream;
	bool swapBytes;
	IsPvdStreamQueryResult(bool s, bool b) : isPvdStream(s), swapBytes(b) {}
};

static inline IsPvdStreamQueryResult isPvdStreamQuery(PxU32 firstWord)
{
	if(firstWord == StreamInitialization::getStreamId())
		return IsPvdStreamQueryResult(true, false);
	swapBytes(firstWord);
	if(firstWord == StreamInitialization::getStreamId())
		return IsPvdStreamQueryResult(true, true);
	return IsPvdStreamQueryResult(false, false);
}

bool PvdCommStreamEventReader::isPvdStream(const PxU32 firstWord) { return isPvdStreamQuery(firstWord).isPvdStream; }

PvdCommStreamEventReader* PvdCommStreamEventReader::create(PxAllocatorCallback& alloc, PvdInputStream& inStream, const PxU32 inFirstWord)
{
	const IsPvdStreamQueryResult query = isPvdStreamQuery(inFirstWord);
	if(query.isPvdStream == false)
	{
		return nullptr;
	}
	StreamInitDeserializer stream(inStream, inFirstWord, query.swapBytes);
	StreamInitialization evt;
	evt.serialize(stream);
	if(!stream.isGood())
		return nullptr;
	const PxU32 supportedFlags =
		static_cast<PxU32>(CommStreamFlagTypes::Is64BitPtr);
	const PxU32 streamFlags = static_cast<PxU32>(evt.mStreamFlags);
	if(evt.mStreamVersion != StreamInitialization::getStreamVersion() ||
	   evt.mTimestampDenominator == 0 ||
	   (streamFlags & ~supportedFlags) != 0)
		return nullptr;
	return PVD_NEW(EventReader)(alloc, evt, query.swapBytes);
}
