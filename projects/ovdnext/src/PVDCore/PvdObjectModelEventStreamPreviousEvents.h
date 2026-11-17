/*
 * SPDX-FileCopyrightText: Copyright (c) 2008-2014 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Copyright (c) 2008-2014 NVIDIA Corporation. All rights reserved.

#ifndef PVD_OBJECT_MODEL_EVENT_STREAM_PREVIOUS_EVENTS_H
#define PVD_OBJECT_MODEL_EVENT_STREAM_PREVIOUS_EVENTS_H
#include "PxPvdFoundation.h"
#include "PxPvdObjectModelBaseTypes.h"

namespace pvdapp
{

template <PxU32 TSize>
struct Union
{
	PxU8 mData[TSize];
	Union() {}
	template <typename TDataType>
	void set(const TDataType& inValue)
	{
		PX_COMPILE_TIME_ASSERT(sizeof(TDataType) <= TSize);
		new(mData) TDataType(inValue);
	}
	template <typename TDataType>
	TDataType get() const
	{
		PX_COMPILE_TIME_ASSERT(sizeof(TDataType) <= TSize);
		return *reinterpret_cast<const TDataType*>(mData);
	}
};

struct IntersectResult
{
	bool mIntersects;
	bool mReplaces;
	bool mStopSearching;
	IntersectResult(bool is, bool rep, bool stopSearch) : mIntersects(is), mReplaces(rep), mStopSearching(stopSearch) {}
	IntersectResult() : mIntersects(false), mReplaces(false), mStopSearching(false) {}
};

static const IntersectResult intersectResultFFF(false, false, false);
static const IntersectResult intersectResultTTT(true, true, true);
static const IntersectResult intersectResultTFT(true, false, true);
static const IntersectResult intersectResultTTF(true, true, false);
static const IntersectResult intersectResultTFF(true, false, false);

struct SnapshotPreviousEvent
{
	const IntersectResult& intersectWith(PxI32, PxI32) { return intersectResultTFT; }
	const IntersectResult& intersectWith(PxI32, PxI32, const PropertyMessageDescription&) { return intersectResultTFT; }
	const IntersectResult& intersectWithArrayElement(PxI32, PxI32, PxI32) { return intersectResultTFT; }
};

struct CreateInstancePreviousEvent
{
	const IntersectResult& intersectWith(PxI32, PxI32) { return intersectResultTFT; }
	const IntersectResult& intersectWith(PxI32, PxI32, const PropertyMessageDescription&) { return intersectResultTFT; }
	const IntersectResult& intersectWithArrayElement(PxI32, PxI32, PxI32) { return intersectResultTFT; }
};

struct SetPropertyValuePreviousEvent
{
	PxI32 mPropId;
	SetPropertyValuePreviousEvent(PxI32 prop) : mPropId(prop) {}
	const IntersectResult& intersectWith(PxI32, PxI32 propId)
	{
		if(mPropId == propId)
			return intersectResultTTT;
		return intersectResultFFF;
	}
	const IntersectResult& intersectWith(PxI32, PxI32, const PropertyMessageDescription& msg)
	{
		PxU32 numIncoming(msg.mProperties.size());
		for(PxU32 idx = 0; idx < numIncoming; ++idx)
			if(msg.mProperties[idx].mProperty.mPropertyId == mPropId)
				return intersectResultTTF;
		return intersectResultFFF;
	}
	const IntersectResult& intersectWithArrayElement(PxI32, PxI32 propId, PxI32)
	{
		if(propId == mPropId)
			return intersectResultTFT;
		return intersectResultFFF;
	}
};

struct SetPropertyMessagePreviousEvent
{
	const PvdObjectModelMetaData* mMetaData;
	PxI32 mMsgId;
	SetPropertyMessagePreviousEvent(const PvdObjectModelMetaData* metaData, PxI32 msgId) : mMetaData(metaData), mMsgId(msgId) {}
	const IntersectResult& intersectWith(PxI32, PxI32 propId)
	{
		PropertyMessageDescription message = mMetaData->getPropertyMessage(mMsgId);
		PxU32 numProps = message.mProperties.size();
		for(PxU32 idx = 0; idx < numProps; ++idx)
			if(message.mProperties[idx].mProperty.mPropertyId == propId)
				return intersectResultTFT;
		return intersectResultFFF;
	}
	const IntersectResult& intersectWith(PxI32, PxI32 msgId, const PropertyMessageDescription& desc)
	{
		if(mMsgId == msgId)
			return intersectResultTTT;
		PropertyMessageDescription message = mMetaData->getPropertyMessage(mMsgId);
		// If any of the incoming properties overlap with one of our properties, then we return true, false.
		PxU32 numIncoming(desc.mProperties.size());
		PxU32 numCurrent(message.mProperties.size());
		for(PxU32 idx = 0; idx < numIncoming; ++idx)
		{
			for(PxU32 msgIdx = 0; msgIdx < numCurrent; ++msgIdx)
			{
				if(message.mProperties[msgIdx].mProperty.mPropertyId == desc.mProperties[idx].mProperty.mPropertyId)
					return intersectResultTFF;
			}
		}
		return intersectResultFFF;
	}
	const IntersectResult& intersectWithArrayElement(PxI32 instId, PxI32 propId, PxI32) { return intersectWith(instId, propId); }
};

struct ArrayElementPreviousEvent
{
	PxI32 mPropId;
	PxI32 mPropIdx;
	ArrayElementPreviousEvent(PxI32 propId, PxI32 propIdx) : mPropId(propId), mPropIdx(propIdx) {}
	const IntersectResult& intersectWith(PxI32, PxI32 propId)
	{
		if(propId == mPropId)
			return intersectResultTTF;
		return intersectResultFFF;
	}
	const IntersectResult& intersectWith(PxI32, PxI32, const PropertyMessageDescription& desc)
	{
		PxU32 numIncoming(desc.mProperties.size());
		for(PxU32 idx = 0; idx < numIncoming; ++idx)
		{
			if(desc.mProperties[idx].mProperty.mPropertyId == mPropId)
				return intersectResultTTF;
		}
		return intersectResultFFF;
	}
};

struct InsertArrayElementPreviousEvent : public ArrayElementPreviousEvent
{
	InsertArrayElementPreviousEvent(PxI32 propId, PxI32 propIdx) : ArrayElementPreviousEvent(propId, propIdx) {}
	const IntersectResult& intersectWithArrayElement(PxI32, PxI32 propId, PxI32& propIdx)
	{
		if(propId != mPropId)
			return intersectResultFFF;
		if(propId == mPropId && propIdx == mPropIdx)
			return intersectResultTTT; // we created the element
		// Going back in time, this element's index was one lower before the insert.
		if((PxI32)propIdx > mPropIdx)
			propIdx--;
		// We didn't influence that property value.
		return intersectResultFFF;
	}
};

struct SetArrayElementValuePreviousEvent : public ArrayElementPreviousEvent
{
	SetArrayElementValuePreviousEvent(PxI32 propId, PxI32 propIdx) : ArrayElementPreviousEvent(propId, propIdx) {}
	const IntersectResult& intersectWithArrayElement(PxI32, PxI32 propId, PxI32 propIdx)
	{
		if(propId == mPropId && propIdx == mPropIdx)
			return intersectResultTTF; // we created the element
		// We don't mangle the property indexes because we don't change them.
		return intersectResultFFF;
	}
};

struct RemoveArrayElementPreviousEvent : public ArrayElementPreviousEvent
{
	RemoveArrayElementPreviousEvent(PxI32 propId, PxI32 propIdx) : ArrayElementPreviousEvent(propId, propIdx) {}

	// It isn't clear how a remove array element could intersect with a later event.
	const IntersectResult& intersectWithArrayElement(PxI32, PxI32 propId, PxI32& propIdx)
	{
		if(propId == mPropId && propIdx >= mPropIdx)
		{
			propIdx++;
		}
		return intersectResultFFF;
	}
};

struct OriginShiftPreviousEvent
{
	PxI32 mInstId;
	// Suppose origin shift only affect at most 2 properties
	PxI32 mPropId1;
	PxI32 mPropId2;

	// Hashmap need default constructor
	OriginShiftPreviousEvent() : mInstId(-1), mPropId1(-1), mPropId2(-1) {}
	OriginShiftPreviousEvent(PxI32 instId, PxI32 propId1, PxI32 propId2) : mInstId(instId), mPropId1(propId1), mPropId2(propId2) {}

	const IntersectResult& intersectWith(PxI32, PxI32 propId)
	{
		if(propId == mPropId1 || propId == mPropId2)
			return intersectResultTTT;
		else
			return intersectResultFFF;
	}
	const IntersectResult& intersectWith(PxI32 instId, PxI32 msgId, const PropertyMessageDescription& msg)
	{
		PX_UNUSED(instId);
		PX_UNUSED(msgId);
		PxU32 numIncoming(msg.mProperties.size());
		for(PxU32 idx = 0; idx < numIncoming; ++idx)
			if((PxI32)msg.mProperties[idx].mProperty.mPropertyId == mPropId1 || (PxI32)msg.mProperties[idx].mProperty.mPropertyId == mPropId2)
				return intersectResultTTF;

		return intersectResultFFF;
	}
	const IntersectResult& intersectWithArrayElement(PxI32 instId, PxI32 propId, PxI32)
	{
		PX_UNUSED(instId);
		if(propId == mPropId1 || propId == mPropId2)
			return intersectResultTTT;
		else
			return intersectResultFFF;
	}
};

struct PreviousEventTypes
{
	enum Enum
	{
#define DECLARE_PREVIOUS_EVENT_TYPE(x) x,
#include "PvdObjectModelEventStreamPreviousEventTypes.h"
#undef DECLARE_PREVIOUS_EVENT_TYPE
	};
};

template <typename TDataType>
struct PreviousEventDatatypeToType
{
	bool compile_error;
};
template <PreviousEventTypes::Enum>
struct PreviousEventTypeToDatatype
{
	bool compile_error;
};

#define DECLARE_PREVIOUS_EVENT_TYPE(x)                                                                                                                                                                 \
	template <>                                                                                                                                                                                        \
	struct PreviousEventDatatypeToType<x##PreviousEvent>                                                                                                                                               \
	{                                                                                                                                                                                                  \
		enum Enum                                                                                                                                                                                      \
		{                                                                                                                                                                                              \
			EventType = PreviousEventTypes::x                                                                                                                                                          \
		};                                                                                                                                                                                             \
	};                                                                                                                                                                                                 \
	template <>                                                                                                                                                                                        \
	struct PreviousEventTypeToDatatype<PreviousEventTypes::x>                                                                                                                                          \
	{                                                                                                                                                                                                  \
		typedef x##PreviousEvent TEventType;                                                                                                                                                           \
	};
#include "PvdObjectModelEventStreamPreviousEventTypes.h"
#undef DECLARE_PREVIOUS_EVENT_TYPE

template <typename TDataType>
PreviousEventTypes::Enum getPreviousEventType()
{
	return static_cast<PreviousEventTypes::Enum>(PreviousEventDatatypeToType<TDataType>::EventType);
}
template <typename TVisitor>
const IntersectResult& visitPreviousEventType(PreviousEventTypes::Enum val, TVisitor& visitor)
{
	switch(val)
	{
#define DECLARE_PREVIOUS_EVENT_TYPE(x)                                                                                                                                                                 \
	case PreviousEventTypes::x:                                                                                                                                                                        \
		return visitor.template handle<x##PreviousEvent>();
#include "PvdObjectModelEventStreamPreviousEventTypes.h"
#undef DECLARE_PREVIOUS_EVENT_TYPE
	}
	PX_ASSERT(false);
	return intersectResultTFT;
}
class PreviousEvent;

struct IntersectsWithVisitor
{
	PxI32 mInstId;
	PxI32 mPropId;
	const Union<16>& mEventData;
	IntersectsWithVisitor(PxI32 instId, PxI32 pid, const Union<16>& evtData) : mInstId(instId), mPropId(pid), mEventData(evtData) {}

	template <typename TDataType>
	const IntersectResult& handle()
	{
		return mEventData.get<TDataType>().intersectWith(mInstId, mPropId);
	}
};

struct PropertyMessageIntersectsWithVisitor
{
	PxI32 mInstId;
	PxI32 mMsgId;
	const PropertyMessageDescription& mMsg;
	const Union<16>& mEventData;
	PropertyMessageIntersectsWithVisitor(PxI32 _instId, PxI32 _msg, const PropertyMessageDescription& desc, const Union<16>& evtData) : mInstId(_instId), mMsgId(_msg), mMsg(desc), mEventData(evtData)
	{
	}
	template <typename TDataType>
	const IntersectResult& handle()
	{
		return mEventData.get<TDataType>().intersectWith(mInstId, mMsgId, mMsg);
	}
};

struct ArrayElementIntersectsWithVisitor
{
	PxI32 mInstId;
	PxI32 mPropId;
	PxI32& mPropIdx;
	const Union<16>& mEventData;
	ArrayElementIntersectsWithVisitor(PxI32 _instId, PxI32 _pid, PxI32& _pidx, const Union<16>& evtData) : mInstId(_instId), mPropId(_pid), mPropIdx(_pidx), mEventData(evtData) {}
	template <typename TDataType>
	const IntersectResult& handle()
	{
		return mEventData.get<TDataType>().intersectWithArrayElement(mInstId, mPropId, mPropIdx);
	}
};

class PreviousEvent
{
	PxU64 mEventId;
	Union<16> mEventData;
	mutable PreviousEvent* mPreviousEvent;
	PreviousEventTypes::Enum mEventType;

  public:
	template <typename TDataType>
	PreviousEvent(PxU64 evid, const TDataType& inType) : mEventId(evid), mPreviousEvent(NULL), mEventType(getPreviousEventType<TDataType>())
	{
		mEventData.set(inType);
	}

	PxU64 getEventId() const { return mEventId; }
	PreviousEventTypes::Enum getType() const { return mEventType; }
	PreviousEvent* getPreviousEvent() const { return mPreviousEvent; }
	void setPreviousEvent(PreviousEvent* item) const { mPreviousEvent = item; }

	struct IIntersectionOp
	{
		virtual ~IIntersectionOp() {}
		virtual const IntersectResult& handleEvent(PreviousEvent& evt) = 0;
	};

	void resolve(Array<PxU32>& outIds, Pool<PreviousEvent>& ioPool, IIntersectionOp& op) const
	{
		PreviousEvent* currentEvent = mPreviousEvent;
		const PreviousEvent* nextEvent = this;
		PxU64 mId = getEventId();
		while(currentEvent)
		{
			const IntersectResult& intersection = op.handleEvent(*currentEvent);
			if(intersection.mIntersects)
			{
				outIds.pushBack(static_cast<PxU32>(mId - currentEvent->getEventId()));
				if(intersection.mReplaces) // then we completely superceded the previous event.
				{
					nextEvent->setPreviousEvent(currentEvent->getPreviousEvent());
					ioPool.deallocate(currentEvent);
					currentEvent = const_cast<PreviousEvent*>(nextEvent);
				}
				if(intersection.mStopSearching)
					return;
			}
			nextEvent = currentEvent;
			currentEvent = currentEvent->getPreviousEvent();
		}
	}

	const IntersectResult& intersectWith(PxI32 instId, PxI32 incomingDescription) const
	{
		IntersectsWithVisitor visitor(instId, incomingDescription, mEventData);
		return visitPreviousEventType(mEventType, visitor);
	}

	const IntersectResult& intersectWith(PxI32 instId, PxI32 msgId, const PropertyMessageDescription& msg) const
	{
		PropertyMessageIntersectsWithVisitor visitor(instId, msgId, msg, mEventData);
		return visitPreviousEventType(mEventType, visitor);
	}

	const IntersectResult& intersectWithArrayElement(PxI32 instId, PxI32 propId, PxI32& propIdx)
	{
		ArrayElementIntersectsWithVisitor visitor(instId, propId, propIdx, mEventData);
		return visitPreviousEventType(mEventType, visitor);
	}
};

typedef Pool<PreviousEvent> TPreviousEventPool;
} // namespace pvdapp

#endif
