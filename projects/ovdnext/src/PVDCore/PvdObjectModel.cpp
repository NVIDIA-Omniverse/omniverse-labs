/*
 * SPDX-FileCopyrightText: Copyright (c) 2008-2014 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Copyright (c) 2008-2014 NVIDIA Corporation. All rights reserved.

#include "PxPvdBits.h"
#include "PxPvdByteStreams.h"
#include "PxPvdFoundation.h"
#include "PvdInputValidation.h"
#include "PxPvdInternalByteStreams.h"
#include "PxPvdObjectModel.h"
#include "PxPvdObjectModelInternalTypes.h"

#include "PxBounds3.h"
#include "PxTransform.h"
#include "PxVec4.h"

#include "profiler.h"

using namespace physx;
using namespace physx::pvdsdk;
using namespace physx::shdfnd;

#define USE_PROPERTY_CACHE 0
#define ORIGIN_ARRAYDATA_SIZE 12
namespace
{

static const PxU32 MAX_INSTANCE_DATA_SIZE = 64u * 1024u * 1024u;

struct InstanceHeader
{
	PxI32 mId;
	PxI32 mClassId;
	NamespacedName mClassName;
	PxU32 mDataSize;
	PxU32 mAliveSetIndex;
	PxU32 mUpdateSetIndex;
	PxU32 mPinnedSetIndex;
	// The offset in the stream where this object was last saved
	// This value is set to an invalid number (PxU64)-1 every time
	// is changed.
	PxU64 mDataSectionStart;

	// Unfortunately, because classes can be modified after they are created
	// we may have less data in our data section than the classes published size.
	InstanceHeader(PxI32 id, PxI32 classId, const NamespacedName& nm, PxU32 dataSize)
	: mId(id), mClassId(classId), mClassName(nm), mDataSize(dataSize), mAliveSetIndex(PX_MAX_U32), mUpdateSetIndex(PX_MAX_U32), mPinnedSetIndex(PX_MAX_U32)
	{
		mDataSectionStart = PVD_POINTER_TO_U64(pointerAfterThis());
	}

	PxU8* pointerAfterThis() { return reinterpret_cast<PxU8*>(this) + sizeof(*this); }

	bool isAlive() const { return mAliveSetIndex != PX_MAX_U32; }
	bool isUpdated() const { return mUpdateSetIndex != PX_MAX_U32; }
	bool isPinned() const { return mPinnedSetIndex != PX_MAX_U32; }

	bool isReferenced() const { return isAlive() || isUpdated() || isPinned(); }

	PxU8* getDataSection() { return reinterpret_cast<PxU8*>(static_cast<size_t>(mDataSectionStart)); }

	bool isPartOf(PxU8* ptr) { return ptr == pointerAfterThis(); }

	operator InstanceDescription() const { return InstanceDescription(mId, mClassId, const_cast<void*>(reinterpret_cast<const void*>(this)), isAlive()); }
};

struct InstanceHeaderAliveSet
{
	void operator()(InstanceHeader& header, PxU32 idx) { header.mAliveSetIndex = idx; }
};
struct InstanceHeaderAliveGet
{
	PxU32 operator()(const InstanceHeader& header) { return header.mAliveSetIndex; }
};

struct InstanceHeaderUpdateSet
{
	void operator()(InstanceHeader& header, PxU32 idx) { header.mUpdateSetIndex = idx; }
};
struct InstanceHeaderUpdateGet
{
	PxU32 operator()(const InstanceHeader& header) { return header.mUpdateSetIndex; }
};

struct InstanceHeaderPinnedSet
{
	void operator()(InstanceHeader& header, PxU32 idx) { header.mPinnedSetIndex = idx; }
};
struct InstanceHeaderPinnedGet
{
	PxU32 operator()(const InstanceHeader& header) { return header.mPinnedSetIndex; }
};

class PvdObjectModelImpl : public PvdObjectModel
{
	struct PropertyPtrCache
	{
		PropertyPtrCache() : ptr(NULL), size(0) {}
		PropertyPtrCache(const PxU8* inPtr, size_t inSize) : ptr(inPtr), size(inSize) {}
		const PxU8* ptr;
		size_t size;
	};
	typedef InvasiveSet<InstanceHeader, InstanceHeaderAliveGet, InstanceHeaderAliveSet> TAliveSet;
	typedef InvasiveSet<InstanceHeader, InstanceHeaderUpdateGet, InstanceHeaderUpdateSet> TUpdateSet;
	typedef InvasiveSet<InstanceHeader, InstanceHeaderPinnedGet, InstanceHeaderPinnedSet> TPinnedSet;
	typedef HashSet<InstancePropertyPair> TShiftedInfoSet;
	typedef HashMap<PxI32, InstanceHeader*> TInstanceMap;
	typedef HashMap<PxI32, PropertyPtrCache> TPropertyPtrMap;
	typedef HashMap<PxI32, TPropertyPtrMap*> TInstancePropertyCacheMap;

	PxAllocatorCallback& mAlloc;
	PvdObjectModelMetaData& mMetaData;
	PxI32 mNextInstHandle;
	TAliveSet mAliveSet;
	TUpdateSet mUpdateSet;
	TPinnedSet mPinnedSet;
	TInstanceMap mInstances;
	MemPvdOutputStream mInstanceWriteBuffer;
	PxU32 mRefCount;
	mutable TInstancePropertyCacheMap mInstancePropertyPtrs;

	// This set stores shifted instances and affected properties, for seek back support
	TShiftedInfoSet mShiftedInstances;

	bool mIsCapture;
	PxU32 mNbCompletedEvt;

	// 32bit arrayData address to new 64bit arrayData point
	HashMap<PxU8*, ArrayData*> mReplacementArrayMap;

  public:
	PvdObjectModelImpl(PxAllocatorCallback& alloc, PvdObjectModelMetaData& meta, bool isCapture)
	: mAlloc(alloc)
	, mMetaData(meta)
	, mNextInstHandle(1)
	, mAliveSet("PvdObjectModelImpl::mAliveSet")
	, mUpdateSet("PvdObjectModelImpl::mUpdateSet")
	, mPinnedSet("PvdObjectModelImpl::mPinnedSet")
	, mInstances("PvdObjectModelImpl::mInstances")
	, mInstanceWriteBuffer("PvdObjectModelImpl::mInstanceWriteBuffer")
	, mShiftedInstances("PvdObjectModelImpl::mShiftedInstances")
	, mRefCount(0)
	, mInstancePropertyPtrs("PvdObjectModelImpl::mInstancePropertyPtrs")
	, mIsCapture(isCapture)
	, mNbCompletedEvt(0)
	, mReplacementArrayMap("PvdObjectModelImpl::mReplacementArrayMap")
	{
		mMetaData.addRef();
	}

	void destructInstance(PxU8* dataSection, PxU8* dataEnd, const ClassDescription& cls)
	{
		const PxU32 totalNum = mMetaData.getNbProperties(cls.mClassId);
		PxU32 offset = 0;
		PropertyDescription propDescs[16];
		while(offset < totalNum)
		{
			const PxU32 numProps = mMetaData.getProperties(cls.mClassId, propDescs, 16, offset);
			offset += numProps;
			PVD_FOREACH(idx, numProps)
			{
				// ClassDescription& propClass = mMetaData.getClass( propDescs[idx].mDatatype );
				const ClassDescription* propClass = mMetaData.getClassPtr(propDescs[idx].mDatatype);
				PxU8* propStart = dataSection + propDescs[idx].m32BitOffset;
				bool isArray = propDescs[idx].mPropertyType == PropertyType::Array;
				const PxU32 propSize = isArray ? ORIGIN_ARRAYDATA_SIZE : propClass->get32BitSize();
				PxU8* propEnd = propStart + propSize;
				if(propEnd <= dataEnd)
				{
					if(isArray)
					{
						ArrayData* item = (getOrReplaceArrayData(propStart));
						if(item->begin())
						{
							// run through the entire array and call destruct instance on every object..
							if(propClass->mRequiresDestruction)
							{
								for(PxU8 *beg = item->begin(), *end = item->end(); beg < end; beg += propClass->get32BitSize())
									destructInstance(beg, beg + propClass->get32BitSize(), propClass);
							}
							mAlloc.deallocate(item->begin());
						}
						deleteReplacedArrayData(propStart);
					}
					else if(propClass->mRequiresDestruction)
						destructInstance(propStart, propEnd, propClass);
				}
			}
		}
	}

	void destructInstance(PxU8* dataSection, PxU8* dataEnd, const ClassDescription* cls)
	{
		const PxU32 totalNum = mMetaData.getNbProperties(cls->mClassId);
		PxU32 offset = 0;
		PropertyDescription buff[16] = {};
		PropertyDescription* propDescs = &buff[0];
		while(offset < totalNum)
		{
			const PxU32 numProps = mMetaData.getProperties(cls->mClassId, propDescs, 16, offset);
			offset += numProps;
			PVD_FOREACH(idx, numProps)
			{
				const ClassDescription* propClass = mMetaData.getClassPtr(propDescs[idx].mDatatype);
				PxU8* propStart = dataSection + propDescs[idx].m32BitOffset;
				const bool isArray = propDescs[idx].mPropertyType == PropertyType::Array;
				const PxU32 propSize = isArray ? ORIGIN_ARRAYDATA_SIZE : propClass->get32BitSize();
				PxU8* propEnd = propStart + propSize;
				if(propEnd <= dataEnd)
				{
					if(isArray)
					{
						ArrayData* item = (getOrReplaceArrayData(propStart));
						if(item->begin())
						{
							// run through the entire array and call destruct instance on every object..
							if(propClass->mRequiresDestruction)
							{
								for(PxU8 *beg = item->begin(), *end = item->end(); beg < end; beg += propClass->get32BitSize())
									destructInstance(beg, beg + propClass->get32BitSize(), propClass);
							}
							mAlloc.deallocate(item->begin());
						}
						deleteReplacedArrayData(propStart);
					}
					else if(propClass->mRequiresDestruction)
					{
						destructInstance(propStart, propEnd, propClass);
					}
				}
			}
		}
	}

	ArrayData* getOrReplaceArrayData(PxU8* orignData)
	{
#if defined(WIN64) || defined(__APPLE__)
		// orign is 32bit size array data, we convert it as 64b size array
		// if exist
		const HashMap<PxU8*, ArrayData*>::Entry* e = mReplacementArrayMap.find(orignData);
		if(e)
			return e->second;

		ArrayData* newArray = static_cast<ArrayData*>(mAlloc.allocate(sizeof(ArrayData), "ArrayData", __FILE__, __LINE__));
		newArray->mBegin = 0;
		newArray->mEnd = 0;
		newArray->mCapacity = 0;

		mReplacementArrayMap.insert(orignData, newArray);

		return newArray;
#else
		return reinterpret_cast<ArrayData*>(orignData);
#endif
	}

	void deleteReplacedArrayData(PxU8* orignData)
	{
#if defined(WIN64) || defined(__APPLE__)
		const HashMap<PxU8*, ArrayData*>::Entry* e = mReplacementArrayMap.find(orignData);
		ArrayData* data = e->second;

		mReplacementArrayMap.erase(orignData);

		if(data)
		{
			mAlloc.deallocate(data);
		}
#endif
	}

	bool resizeArray(ArrayData& theArray, PxU32 objectSize, PxU32 desiredByteSize) const
	{
		PxU32 newSize = 0;
		if(!pvdapp::input::computeArrayAllocationSize(objectSize, desiredByteSize, newSize))
		{
			return false;
		}

		if(desiredByteSize != 0 && theArray.byteCapacity() < desiredByteSize) // we are replacing the entire array
		{
			PxU8* newArrayData = static_cast<PxU8*>(mAlloc.allocate(newSize, "InstanceArrayData", __FILE__, __LINE__));
			if(!newArrayData)
				return false;
			if(theArray.byteSize())
				physx::intrinsics::memCopy(newArrayData, theArray.begin(), theArray.byteSize());
			if(theArray.byteCapacity())
				mAlloc.deallocate(theArray.begin());
			physx::intrinsics::memZero(newArrayData + theArray.byteSize(), newSize - theArray.byteSize());
			theArray.mBegin = newArrayData;
			theArray.mCapacity = theArray.mBegin + newSize;
		}
		theArray.mEnd = theArray.mBegin ? theArray.mBegin + desiredByteSize : NULL;
		return true;
	}

	void copyInstanceArrays(PxU8* dataSection, PxU8* dataEnd, const PxU8* srcSection, const PxU8* srcEnd, const ClassDescription* cls)
	{
		PxU32 totalNum = mMetaData.getNbProperties(cls->mClassId);
		PxU32 offset = 0;
		PropertyDescription propDescs[16];
		while(offset < totalNum)
		{
			PxU32 numProps = mMetaData.getProperties(cls->mClassId, propDescs, 16, offset);
			offset += numProps;
			PVD_FOREACH(idx, numProps)
			{
				ClassDescription* propClass = mMetaData.getClassPtr(propDescs[idx].mDatatype);
				PxU8* propStart = dataSection + propDescs[idx].m32BitOffset;
				PxU8* propEnd = propStart + propClass->get32BitSize();
				const PxU8* propSrcSection = srcSection + propDescs[idx].m32BitOffset;
				const PxU8* propSrcEnd = propSrcSection + propClass->get32BitSize();
				if(propEnd <= dataEnd && propSrcEnd <= srcEnd)
				{
					if(propDescs[idx].mPropertyType == PropertyType::Array)
					{
						ArrayData* item = getOrReplaceArrayData(propStart);
						item->forgetData();
						const ArrayData* srcItem = getOrReplaceArrayData(const_cast<PxU8*>(propSrcSection));
						if(!resizeArray(*item, propClass->get32BitSize(), srcItem->byteSize()))
							return;
						physx::intrinsics::memCopy(item->mBegin, srcItem->mBegin, srcItem->byteSize());
					}
					else if(propClass->mRequiresDestruction)
						copyInstanceArrays(propStart, propEnd, propSrcSection, propSrcEnd, propClass);
				}
			}
		}
	}

	void destructInstanceData(InstanceHeader* instance, const ClassDescription& cls)
	{
		PxU8* dataSection = instance->getDataSection();
		if(cls.mRequiresDestruction == true)
		{
			PxU8* dataEnd = dataSection + instance->mDataSize;
			destructInstance(dataSection, dataEnd, cls);
		}
	}

	void destructInstanceData(InstanceHeader* instance, const ClassDescription* cls)
	{
		if(cls->mRequiresDestruction == true)
		{
			PxU8* dataSection = instance->getDataSection();
			PxU8* dataEnd = dataSection + instance->mDataSize;
			destructInstance(dataSection, dataEnd, cls);
		}
	}

	void destructInstance(InstanceHeader* instance, const ClassDescription& cls)
	{
		destructInstanceData(instance, cls);
		// If the data section does not immediately follow and thus is not part
		// of the object, destruct it separately.
		if(instance->isPartOf(instance->getDataSection()) == false)
			mAlloc.deallocate(instance->getDataSection());
		mAlloc.deallocate(instance);
	}

	void destructInstance(InstanceHeader* instance, const ClassDescription* cls)
	{
		destructInstanceData(instance, cls);
		// If the data section does not immediately follow and thus is not part
		// of the object, destruct it separately.
		if(instance->isPartOf(instance->getDataSection()) == false)
			mAlloc.deallocate(instance->getDataSection());
		mAlloc.deallocate(instance);
	}

	void destructInstance(InstanceHeader* instance)
	{
		// run through data section and release each array
		const ClassDescription* cls = mMetaData.getClassPtr(instance->mClassId);
		destructInstance(instance, cls);
	}

	virtual ~PvdObjectModelImpl()
	{
		mAliveSet.clear();
		mUpdateSet.clear();
		for(TInstanceMap::Iterator iter = mInstances.getIterator(); iter.done() == false; ++iter)
			destructInstance(iter->second);

#if USE_PROPERTY_CACHE
		for(TInstancePropertyCacheMap::Iterator iter = mInstancePropertyPtrs.getIterator(); iter.done() == false; ++iter)
			removeInstancePropertyCache(iter->first);
#endif

		mInstances.clear();
		mMetaData.release();
	}

	bool nextHandle(PxI32& handle)
	{
		if(mNextInstHandle < 0 || mNextInstHandle == PX_MAX_I32)
			return false;
		handle = mNextInstHandle++;
		return true;
	}

	virtual const PvdObjectModelMetaData& getMetaData() const { return mMetaData; }

	inline void setInstanceDirty(InstanceHeader& inst) { mUpdateSet.insert(inst); }

	virtual void recordCompletedInstances()
	{
		if(mIsCapture)
			mNbCompletedEvt = mUpdateSet.size();
	}

	InstanceDescription doCreateInstance(PxI32 clsId, PxI32 hdlValue)
	{
		if(hdlValue < 0 || hdlValue == PX_MAX_I32)
			return InstanceDescription();
		ClassDescription* cls = mMetaData.getClassPtr(clsId);
		if(!cls || cls->get32BitSize() > MAX_INSTANCE_DATA_SIZE)
			return InstanceDescription();
		const size_t instanceSize = sizeof(InstanceHeader) +
			static_cast<size_t>(cls->get32BitSize());
		InstanceHeader* newObj = reinterpret_cast<InstanceHeader*>(mAlloc.allocate(instanceSize, "PvdObjectModel::Instance", __FILE__, __LINE__));
		if(!newObj)
			return InstanceDescription();
		new(newObj) InstanceHeader(hdlValue, cls->mClassId, cls->mName, cls->get32BitSize());
		physx::intrinsics::memSet(newObj->getDataSection(), 0, newObj->mDataSize);
		mAliveSet.insert(*newObj);
		setInstanceDirty(*newObj);
		mInstances.insert(newObj->mId, newObj);
		mMetaData.lockClass(cls->mClassId);
		if(hdlValue >= mNextInstHandle)
			mNextInstHandle = hdlValue + 1;
		return *newObj;
	}

	virtual InstanceDescription createInstance(PxI32 clsId)
	{
		PxI32 handle = -1;
		if(!nextHandle(handle))
			return InstanceDescription();
		return doCreateInstance(clsId, handle);
	}
	virtual InstanceDescription createInstance(PxI32 clsId, PxI32 instId)
	{
		void* instData = idToPtr(instId);
		if(instData == NULL)
			return doCreateInstance(clsId, instId);

		const ClassDescription* cls = mMetaData.getClassPtr(clsId);
		InstanceHeader* inst = reinterpret_cast<InstanceHeader*>(instData);
		if(!cls || inst->mClassId != clsId)
			return InstanceDescription();
		// Grow before destroying the previous value so an allocation failure leaves
		// the existing instance intact.
		if(!resizeInstance(inst, cls))
			return InstanceDescription();
		// Destroy any arrays, but don't release the instance
		destructInstanceData(inst, cls);
		// zero it out to set it back at its original state
		physx::intrinsics::memZero(inst->getDataSection(), inst->mDataSize);
		// Add to the appropriate lists
		mAliveSet.insert(*inst);
		setInstanceDirty(*inst);
		return *inst;
	}

	virtual void pinInstance(void* instPtr)
	{
		PX_ASSERT(instPtr);
		if(instPtr == NULL)
			return;
		InstanceHeader* obj = reinterpret_cast<InstanceHeader*>(instPtr);
		mPinnedSet.insert(*obj);
	}

	virtual void unPinInstance(void* instPtr)
	{
		PX_ASSERT(instPtr);
		if(instPtr == NULL)
			return;
		InstanceHeader* obj = reinterpret_cast<InstanceHeader*>(instPtr);
		mPinnedSet.remove(*obj);
		maybeDeleteInstance(obj);
	}

	virtual void destroyInstance(void* instPtr)
	{
		//        PX_ASSERT(instPtr);
		if(instPtr == NULL)
			return;
		InstanceHeader* obj = reinterpret_cast<InstanceHeader*>(instPtr);

		// we don't clear the last save offset here
		// because we didn't change any internal data and because
		// we save out instances lazily and only if their last
		// save offset is bogus.
		setInstanceDirty(*obj);
		mAliveSet.remove(*obj);
	}

	virtual void destroyAllInstances()
	{
#ifdef PROFILING
		profiler::ProfilerScope prof_scope("PvdObjectModelImpl::destroyAllInstances()", 0);
#endif
		while(mAliveSet.size())
		{
			InstanceHeader* obj = mAliveSet[0];
			destroyInstance(obj);
		}
	}

	virtual Option<ClassDescription> getClassOf(void* instPtr) const
	{
		if(instPtr == nullptr)
			return None();
		return mMetaData.getClass(reinterpret_cast<InstanceHeader*>(instPtr)->mClassId);
	}

	virtual void* idToPtr(const PxI32 instId) const
	{
		const TInstanceMap::Entry* entry = mInstances.find(instId);
		if(entry)
			return reinterpret_cast<void*>(entry->second);
		return nullptr;
	}

	virtual PxI32 ptrToId(void* instPtr) const
	{
		if(instPtr)
			return reinterpret_cast<const InstanceHeader*>(instPtr)->mId;
		return -1;
	}

	virtual InstanceDescription idToDescriptor(PxI32 instId) const
	{
		void* ptr(idToPtr(instId));
		if(ptr)
			return *(reinterpret_cast<InstanceHeader*>(ptr));
		return InstanceDescription();
	}

	virtual InstanceDescription ptrToDescriptor(void* ptr) const
	{
		if(ptr)
			return *(reinterpret_cast<InstanceHeader*>(ptr));
		return InstanceDescription();
	}

	virtual PxI32 getNextInstanceHandleValue() const { return mNextInstHandle; }

	virtual void setNextInstanceHandleValue(PxI32 hdlValue)
	{
		if(hdlValue >= 0 && hdlValue < PX_MAX_I32 && hdlValue >= mNextInstHandle)
			mNextInstHandle = hdlValue + 1;
	}
	bool resizeInstance(InstanceHeader* inst, const ClassDescription& desc) const
	{
		if(!inst)
			return false;
		const PxU32 requiredSize = desc.get32BitSize();
		if(requiredSize > MAX_INSTANCE_DATA_SIZE)
			return false;
		if(inst->mDataSize >= requiredSize)
		{
			mMetaData.lockClass(desc.mClassId);
			return true;
		}

		PxU8* current = inst->getDataSection();
		if(inst->mDataSize && !current)
			return false;
		const PxU64 doubledSize = static_cast<PxU64>(requiredSize) * 2u;
		const PxU32 newSize = static_cast<PxU32>(
			PxMin<PxU64>(doubledSize, MAX_INSTANCE_DATA_SIZE));
		if(newSize < requiredSize)
			return false;
		PxU8* newData = static_cast<PxU8*>(mAlloc.allocate(newSize, "PvdObjectModelImpl::InstanceData", __FILE__, __LINE__));
		if(!newData)
			return false;
		physx::intrinsics::memZero(newData, newSize);
		if(inst->mDataSize)
			physx::intrinsics::memCopy(newData, current, inst->mDataSize);
		const bool currentIsSeparate = current && !inst->isPartOf(current);
		inst->mDataSectionStart = PVD_POINTER_TO_U64(newData);
		inst->mDataSize = newSize;
		if(currentIsSeparate)
			mAlloc.deallocate(current);
		mMetaData.lockClass(desc.mClassId);
		return true;
	}

	bool resizeInstance(InstanceHeader* inst, const ClassDescription* desc) const
	{
		return desc && resizeInstance(inst, *desc);
	}

	struct ResolvePropertyInstanceDataResult
	{
		PropertyDescription mProperty;
		DataRef<PxU8> mData;
		ArrayData* mArray;
		PxU32 mPropertyDatatypeSize;
		InstanceHeader* mInstance;
		ResolvePropertyInstanceDataResult(const PropertyDescription& prop, DataRef<PxU8> data, ArrayData* mArray, PxU32 propDatatypeSize, InstanceHeader* instance)
		: mProperty(prop), mData(data), mArray(mArray), mPropertyDatatypeSize(propDatatypeSize), mInstance(instance)
		{
		}
		ResolvePropertyInstanceDataResult() : mArray(NULL), mPropertyDatatypeSize(0) {}
	};

	// Run through the chain of properties and figure out exactly where the data for the final
	// property lies.
	inline Option<ResolvePropertyInstanceDataResult> resolvePropertyInstanceData(void* instId, PxI32 propId)
	{
		//        PX_ASSERT(instId);
		if(instId == nullptr)
		{
			//            PX_ASSERT(false);
			return None();
		}
		InstanceHeader* inst(reinterpret_cast<InstanceHeader*>(instId));
		const ClassDescription* instType = mMetaData.getClassPtr(inst->mClassId);
		if(!instType || !resizeInstance(inst, instType))
			return None();
		const Option<PropertyDescription>& resolvedProp = mMetaData.resolvePropertyPath(instType->mClassId, propId);
		if(resolvedProp.hasValue() == false)
		{
			PX_ASSERT(false);
			return None();
		}
		return resolvePropertyInstanceData(*inst, inst->getDataSection(), inst->getDataSection() + inst->mDataSize, resolvedProp.getValue(), -1);
	}

	inline Option<ResolvePropertyInstanceDataResult> resolvePropertyInstanceData(InstanceHeader& instance, PxU8* instData, PxU8* instEnd, const PropertyDescription& prop, PxI32 incomingType = -1)
	{
		if(incomingType >= 0 && prop.mDatatype != incomingType)
		{
			PX_ASSERT(false);
			return None();
		}
		ArrayData* theArray = NULL;

		instData = instData + prop.m32BitOffset;
		const PxI32 propType = prop.mDatatype;
		const ClassDescription* localType = mMetaData.getClassPtr(propType);
		const PxU32 propSize = prop.mPropertyType == PropertyType::Array ? ORIGIN_ARRAYDATA_SIZE : localType->get32BitSize();
		const PxU32 itemByteSize = localType->get32BitSize();
		if((instData + propSize) > instEnd)
		{
			PX_ASSERT(false);
			return None();
		}
		if(prop.mPropertyType == PropertyType::Array)
		{
			theArray = getOrReplaceArrayData(instData);
			instData = theArray->begin();
			instEnd = theArray->end();
		}
		else
			instEnd = instData + propSize;
		return ResolvePropertyInstanceDataResult(prop, DataRef<PxU8>(instData, instEnd), theArray, itemByteSize, &instance);
	}

	bool setScalarPropertyValue(const PxU8* src, const PxU8* srcEnd, PxU8* dest, PxU8* destEnd, const ClassDescription& incomingType)
	{
		PxU32 len = static_cast<PxU32>(srcEnd - src);
		PxU32 destLen = static_cast<PxU32>(destEnd - dest);
		if(incomingType.mRequiresDestruction || len != destLen)
			return false;
		if(len)
			physx::intrinsics::memCopy(dest, src, len);
		return true;
	}

	bool setScalarPropertyValue(const PxU8* src, const PxU8* srcEnd, PxU8* dest, PxU8* destEnd, const ClassDescription* incomingType)
	{
		return incomingType && setScalarPropertyValue(src, srcEnd, dest, destEnd, *incomingType);
	}

	virtual bool setPropertyValue(void* instId, PxI32 propId, const PxU8* data, PxU32 len, PxI32 incomingTypeId)
	{
		const Option<ResolvePropertyInstanceDataResult>& resolutionOpt(resolvePropertyInstanceData(instId, propId));
		if(resolutionOpt.hasValue() == false)
			return false;

		const ResolvePropertyInstanceDataResult& resolution(resolutionOpt);

		DataRef<PxU8> propData(resolution.mData);
		ArrayData* theArray(resolution.mArray);

		const PropertyDescription& prop(resolution.mProperty);
		const ClassDescription* propType = mMetaData.getClassPtr(prop.mDatatype);
		if(!propType)
			return false;
		PxU32 propByteSize = propType->get32BitSize();
		if(propByteSize == 0 || (len && !data))
			return false;
		if(prop.mPropertyType == PropertyType::Array)
		{
			if(len % propByteSize)
				return false;
		}
		else if(len != propByteSize)
			return false;
		// You can't set a property that has an array in it like this.
		if(propType->mRequiresDestruction)
		{
			PX_ASSERT(false);
			return false;
		}

		if(incomingTypeId >= 0)
		{
			const MarshalQueryResult& marshalQuery(mMetaData.checkMarshalling(incomingTypeId, propType->mClassId));
			if(marshalQuery.needsMarshalling == true)
			{
				PX_ASSERT(false);
				return false;
			}
		}
		PxU8* instData = const_cast<PxU8*>(propData.begin());
		PxU8* instEnd = instData + propData.size();
		if(prop.mPropertyType == PropertyType::Array) // len = 0 means clean the property
		{
			if(theArray == NULL)
			{
				PX_ASSERT(false);
				return false;
			}
			if(!resizeArray(*theArray, propType->get32BitSize(), len))
				return false;
			instData = theArray->mBegin;
			instEnd = theArray->mEnd;
			if(len)
				physx::intrinsics::memCopy(instData, data, len);
		}
		else if(!setScalarPropertyValue(data, data + len, instData, instEnd, propType))
			return false;
		setInstanceDirty(*reinterpret_cast<InstanceHeader*>(instId));
		return true;
	}

	virtual bool setPropertyValueToDefault(void* instId, PxI32 propId)
	{
		Option<ResolvePropertyInstanceDataResult> resolutionOpt(resolvePropertyInstanceData(instId, propId));
		if(resolutionOpt.hasValue() == false)
		{
			PX_ASSERT(false);
			return false;
		}
		if(resolutionOpt->mProperty.mPropertyType == PropertyType::Array)
		{
			if(!resizeArray(*resolutionOpt->mArray, resolutionOpt->mPropertyDatatypeSize, 0))
				return false;
		}
		else
			physx::intrinsics::memZero(const_cast<PxU8*>(resolutionOpt->mData.begin()), resolutionOpt->mData.size());
		setInstanceDirty(*reinterpret_cast<InstanceHeader*>(instId));
		return true;
	}

	// Return the byte size of a possible nested property
	virtual PxU32 getPropertyByteSize(void* instId, int32_t propId)
	{
		Option<ResolvePropertyInstanceDataResult> resolution(resolvePropertyInstanceData(instId, propId));
		if(resolution.hasValue())
			return resolution->mData.size();
		return 0;
	}

	PxU32 getScalarPropertyValue(const PxU8* src, const PxU8* srcEnd, PxU8* dest, PxU8* destEnd) const
	{
		PxU32 outDataLen = static_cast<PxU32>(destEnd - dest);
		PxU32 instDataSize = static_cast<PxU32>(srcEnd - src);
		PX_ASSERT(outDataLen >= instDataSize);
		PxU32 amountToCopy = PxMin(outDataLen, instDataSize);
		physx::intrinsics::memCopy(dest, src, amountToCopy);
		if(amountToCopy < outDataLen)
			physx::intrinsics::memZero(dest + amountToCopy, outDataLen - amountToCopy);
		return amountToCopy;
	}
	// Return the value of a property
	virtual PxU32 getPropertyValue(void* instId, PxI32 propId, PxU8* outData, PxU32 outDataLen)
	{
		PX_ASSERT(outData);
		PX_ASSERT(outDataLen);

		const PxU8* data = NULL;
		const PxU8* dataEnd = NULL;

#if USE_PROPERTY_CACHE
		InstanceHeader* instHd = (InstanceHeader*)instId;
		const TInstancePropertyCacheMap::Entry* entry = mInstancePropertyPtrs.find(instHd->mId);
		TPropertyPtrMap* proPtrMap = NULL;

		if(entry != NULL)
		{
			proPtrMap = entry->second;
			PX_ASSERT(proPtrMap);
			const TPropertyPtrMap::Entry* propertyPtr = proPtrMap->find(propId);
			if(propertyPtr != NULL)
			{
				data = propertyPtr->second.ptr;
				dataEnd = propertyPtr->second.ptr + propertyPtr->second.size;
			}
		}
		else
		{
			PxU8* newData = (PxU8*)mAlloc.allocate(sizeof(TPropertyPtrMap), "PvdObjectModelImpl::TPropertyPtrMap", __FILE__, __LINE__);
			PX_ASSERT(newData);

			proPtrMap = PX_PLACEMENT_NEW(newData, TPropertyPtrMap(mAlloc, "PvdObjectModelImpl::TPropertyPtrMap"));
			mInstancePropertyPtrs.insert(instHd->mId, proPtrMap);
		}
#endif
		if(data == NULL)
		{
			Option<ResolvePropertyInstanceDataResult> resolution = resolvePropertyInstanceData(instId, propId);
			PX_ASSERT(resolution.hasValue());
			if(resolution.hasValue())
			{
				data = resolution->mData.begin();
				dataEnd = resolution->mData.end();

#if USE_PROPERTY_CACHE
				// Array can't be cahced cause array data address maybe changed.
				if(resolution->mProperty.mDatatype != PropertyType::Array)
				{
					PropertyPtrCache cache(data, dataEnd - data);
					proPtrMap->insert(propId, cache);
				}
#endif
			}
		}

		if(data != NULL)
			return getScalarPropertyValue(data, dataEnd, outData, outData + outDataLen);

		return 0;
	}

	virtual DataRef<PxU8> getRawPropertyValue(void* instId, PxI32 propId)
	{
		const Option<ResolvePropertyInstanceDataResult> resolution = resolvePropertyInstanceData(instId, propId);
		PX_ASSERT(resolution.hasValue());
		if(resolution.hasValue())
			return resolution->mData;
		return DataRef<PxU8>();
	}

	virtual PxU32 getPropertyValue(void* instId, PxI32 propId, int arrayIndex, PxU8* outData, PxU32 outDataLen)
	{
		PX_ASSERT(outData);
		PX_ASSERT(outDataLen);
		Option<ResolvePropertyInstanceDataResult> resolution = resolvePropertyInstanceData(instId, propId);
		PX_ASSERT(resolution.hasValue());
		if(resolution.hasValue())
		{
			PX_ASSERT(resolution->mArray);
			if(resolution->mArray != NULL)
			{
				PxU32 numItems = resolution->mArray->numberOfItems(resolution->mPropertyDatatypeSize);
				PxU32 amountToCopy = 0;
				if(arrayIndex < (int)numItems)
				{
					amountToCopy = PxMin(resolution->mPropertyDatatypeSize, outDataLen);
					physx::intrinsics::memCopy(outData, resolution->mArray->mBegin + arrayIndex * resolution->mPropertyDatatypeSize, resolution->mPropertyDatatypeSize);
				}
				if(amountToCopy < outDataLen)
					physx::intrinsics::memZero(outData + amountToCopy, outDataLen - amountToCopy);
				return amountToCopy;
			}
		}
		return 0;
	}

	virtual PxU32 getPropertyValue(void* instId, PxI32 propId, int arrayIndex, int nestedProperty, PxU8* outData, PxU32 outDataLen)
	{
		PX_ASSERT(outData);
		PX_ASSERT(outDataLen);
		Option<ResolvePropertyInstanceDataResult> resolution = resolvePropertyInstanceData(instId, propId);
		PX_ASSERT(resolution.hasValue());
		Option<PropertyDescription> propDesc(mMetaData.getProperty(nestedProperty));
		PX_ASSERT(propDesc.hasValue());
		if(propDesc.hasValue() == false)
			return 0;
		ClassDescription* propDtypeDesc = mMetaData.getClassPtr(propDesc->mDatatype);
		PX_ASSERT(propDtypeDesc != NULL);
		if(resolution.hasValue())
		{
			PX_ASSERT(resolution->mArray);
			if(resolution->mArray != NULL)
			{
				PxU32 numItems = resolution->mArray->numberOfItems(resolution->mPropertyDatatypeSize);
				PxU32 amountToCopy = 0;
				if(arrayIndex < (int)numItems)
				{
					amountToCopy = PxMin(propDtypeDesc->get32BitSize(), outDataLen);
					const PxU8* srcData = resolution->mArray->begin() + resolution->mPropertyDatatypeSize * arrayIndex;
					srcData += propDesc->m32BitOffset;
					physx::intrinsics::memCopy(outData, srcData, amountToCopy);
				}
				if(amountToCopy < outDataLen)
					physx::intrinsics::memZero(outData + amountToCopy, outDataLen - amountToCopy);
				return amountToCopy;
			}
		}
		return 0;
	}

	struct PropertyMessagePrep
	{
		InstanceHeader* instance;
		PropertyMessageDescription msg;
		PropertyMessagePrep(InstanceHeader* _inst) : instance(_inst) {}
	};

	PropertyMessagePrep preparePropertyMessage(void* instId, PxI32 msgId, PxU32 dataLen) const
	{
		if(instId == NULL)
		{
			PX_ASSERT(false);
			return NULL;
		}

		const Option<PropertyMessageDescription>& msgOpt = mMetaData.getPropertyMessage(msgId);
		if(msgOpt.hasValue() == false)
		{
			PX_ASSERT(false);
			return NULL;
		}

		InstanceHeader* instance = reinterpret_cast<InstanceHeader*>(instId);
		if(mMetaData.isDerivedFrom(instance->mClassId, msgOpt->mClassId) == false)
		{
			PX_ASSERT(false);
			return NULL;
		}

		if(dataLen < msgOpt->mMessageByteSize)
		{
			PX_ASSERT(false);
			return NULL;
		}

		const ClassDescription* instCls = mMetaData.getClassPtr(instance->mClassId);
		if(!instCls || !resizeInstance(instance, instCls))
			return NULL;

		PropertyMessagePrep msgPrep(instance);
		msgPrep.msg = msgOpt.getValue();
		return msgPrep;
	}

	// Set a set of properties defined by a property message
	virtual bool setPropertyMessage(void* instId, PxI32 msgId, const PxU8* data, PxU32 dataLen)
	{
		const PropertyMessagePrep& msgPrep = preparePropertyMessage(instId, msgId, dataLen);
		const PropertyMessageDescription& msg = msgPrep.msg;
		InstanceHeader* instance = msgPrep.instance;

		if(instance == NULL)
		{
			PX_ASSERT(false);
			return false;
		}
		if(msg.mStringOffsets.size())
		{
			PX_ASSERT(false);
			return false;
		}

		PxU8* instData(instance->getDataSection());
		PxU8* instEnd(instance->getDataSection() + instance->mDataSize);
		PVD_FOREACH(entryIdx, msg.mProperties.size())
		{
			const PropertyMessageEntry& entry = msg.mProperties[entryIdx];
			PxU8* target = instData + entry.mProperty.m32BitOffset;
			const PxU8* src = data + entry.mMessageOffset;
			if(!setScalarPropertyValue(src, src + entry.mByteSize, target,
				PxMin(target + entry.mDestByteSize, instEnd),
				mMetaData.getClassPtr(entry.mDatatypeId)))
				return false;
		}
		setInstanceDirty(*instance);
		return true;
	}

	// Get a set of properties defined by a property message
	virtual bool getPropertyMessage(void* instId, PxI32 msgId, PxU8* data, PxU32 dataLen) const
	{
		const PropertyMessagePrep& msgPrep(preparePropertyMessage(instId, msgId, dataLen));
		const PropertyMessageDescription& msg(msgPrep.msg);
		InstanceHeader* instance(msgPrep.instance);

		if(instance == NULL)
		{
			PX_ASSERT(false);
			return false;
		}
		const PxU8* instData(instance->getDataSection());
		const PxU8* instEnd(instData + instance->mDataSize);
		PVD_FOREACH(entryIdx, msg.mProperties.size())
		{
			const PropertyMessageEntry& entry(msg.mProperties[entryIdx]);
			const PxU8* src(instData + entry.mProperty.m32BitOffset);
			PxU8* target(data + entry.mMessageOffset);
			getScalarPropertyValue(src, PxMin(src + entry.mDestByteSize, instEnd), target, target + entry.mByteSize);
		}
		return true;
	}

	bool doInsertElement(const PropertyDescription& propId, ArrayData& mArray, PxI32 idx, const PxU8* data, PxU32 dataLen, PxI32 incomingType, InstanceHeader& instance)
	{
		if(incomingType >= 0 && propId.mDatatype != incomingType)
		{
			PX_ASSERT(false);
			return false;
		}

		ClassDescription propType(mMetaData.getClass(propId.mDatatype));
		PxU32 mArrayByteSize = propType.get32BitSize();
		if(mArrayByteSize == 0 || mArrayByteSize > pvdapp::input::kMaxPropertyArrayBytes)
			return false;
		if(!data || dataLen != mArrayByteSize)
			return false;
		PxU32 numItems = mArray.byteSize() / mArrayByteSize;
		if(idx < 0 || numItems < (PxU32)(idx))
			idx = (PxI32)numItems;

		if(mArray.byteSize() > pvdapp::input::kMaxPropertyArrayBytes - mArrayByteSize ||
		   !resizeArray(mArray, mArrayByteSize, mArray.byteSize() + mArrayByteSize))
			return false;
		PxU8* instBegin = mArray.begin() + idx * mArrayByteSize;
		PxU8* instEnd = instBegin + mArrayByteSize;
		if(instEnd > mArray.mEnd)
		{
			PX_ASSERT(false);
			return false;
		}
		if(idx < (PxI32)numItems)
			physx::intrinsics::memMove(instEnd, instBegin, (numItems - idx) * mArrayByteSize);
		// Ensure this is recognized as a new instance.
		physx::intrinsics::memZero(instBegin, static_cast<PxU32>(instEnd - instBegin));
		if(!setScalarPropertyValue(data, data + dataLen, instBegin, instEnd, propType))
			return false;
		setInstanceDirty(instance);
		return true;
	}

	// insert an element into array index.  If index > numElements, element is appended.
	virtual bool insertArrayElement(void* instId, PxI32 propId, PxI32 index, const PxU8* data, PxU32 dataLen, PxI32 incomingType)
	{
		Option<ResolvePropertyInstanceDataResult> resolvedPath(resolvePropertyInstanceData(instId, propId));
		if(resolvedPath.hasValue() == false)
		{
			PX_ASSERT(false);
			return false;
		}
		if(resolvedPath->mArray == nullptr)
		{
			PX_ASSERT(false);
			return false;
		}

		return doInsertElement(resolvedPath->mProperty, *(resolvedPath->mArray), index, data, dataLen, incomingType, *resolvedPath->mInstance);
	}

	bool doRemoveArrayElement(const ResolvePropertyInstanceDataResult& resolvedPath, PxI32 idx)
	{
		if(!resolvedPath.mArray || idx < 0)
			return false;
		ArrayData& item(*resolvedPath.mArray);
		PxU32 objectByteSize = resolvedPath.mPropertyDatatypeSize;
		if(objectByteSize == 0 || static_cast<PxU32>(idx) >= item.numberOfItems(objectByteSize))
			return false;
		const PxU32 byteOffset = static_cast<PxU32>(idx) * objectByteSize;
		PxU8* instBegin(item.begin() + byteOffset);
		PxU8* instEnd(instBegin + objectByteSize);
		if(instEnd > item.end())
			return false;
		ClassDescription clsDesc(mMetaData.getClass(resolvedPath.mProperty.mDatatype));

		if(clsDesc.mRequiresDestruction)
			destructInstance(instBegin, instEnd, clsDesc);
		PxU32 moveAmount = static_cast<PxU32>(item.end() - instEnd);
		if(moveAmount)
			physx::intrinsics::memMove(instBegin, instEnd, moveAmount);
		item.mEnd = item.mEnd - resolvedPath.mPropertyDatatypeSize;
		setInstanceDirty(*resolvedPath.mInstance);
		return true;
	}

	virtual bool removeArrayElement(void* instId, PxI32 propId, PxI32 index)
	{
		Option<ResolvePropertyInstanceDataResult> resolvedPath(resolvePropertyInstanceData(instId, propId));
		if(resolvedPath.hasValue() && resolvedPath->mArray && index >= 0)
			return doRemoveArrayElement(resolvedPath, index);
		return false;
	}
	Option<PxI32> indexOfArrayElement(const ResolvePropertyInstanceDataResult& resolvedPath, const PxU8* cmpData, PxU32 dataLen, PxI32 incomingType)
	{
		if(incomingType >= 0 && incomingType != resolvedPath.mProperty.mDatatype)
		{
			PX_ASSERT(false);
			return None();
		}

		if(resolvedPath.mArray == NULL)
		{
			PX_ASSERT(false);
			return None();
		}
		PxU32 objectSize = mMetaData.getClass(resolvedPath.mProperty.mDatatype)->get32BitSize();
		if(dataLen < resolvedPath.mPropertyDatatypeSize)
		{
			PX_ASSERT(false);
			return None();
		}
		PxU32 idx = 0;
		PxU32 numElements = resolvedPath.mArray->numberOfItems(objectSize);
		PxU8* itemAddr = resolvedPath.mArray->begin();
		for(; idx < numElements; ++idx, itemAddr += objectSize)
		{
			if(memcmp(itemAddr, cmpData, objectSize) == 0)
				return PxI32(idx);
		}
		return -1;
	}

	virtual Option<PxI32> pushBackArrayElementIf(void* instId, PxI32 propId, const PxU8* cmpData, PxU32 dataLen, PxI32 incomingType)
	{
		Option<ResolvePropertyInstanceDataResult> resolvedPath(resolvePropertyInstanceData(instId, propId));
		if(resolvedPath.hasValue() == false)
		{
			PX_ASSERT(false);
			return None();
		}

		Option<PxI32> idxOpt(indexOfArrayElement(resolvedPath, cmpData, dataLen, incomingType));
		if(idxOpt.hasValue() == false)
			return idxOpt;
		PxI32 idx = idxOpt.getValue();
		if(idx >= 0)
		{
			return -1;
		}
		if(doInsertElement(resolvedPath->mProperty, *(resolvedPath->mArray), -1, cmpData, dataLen, incomingType, *resolvedPath->mInstance))
			return resolvedPath->mArray->numberOfItems(resolvedPath->mPropertyDatatypeSize) - 1;
		return None();
	}

	virtual Option<PxI32> removeArrayElementIf(void* instId, PxI32 propId, const PxU8* cmpData, PxU32 dataLen, PxI32 incomingType)
	{
		Option<ResolvePropertyInstanceDataResult> resolvedPath(resolvePropertyInstanceData(instId, propId));
		if(resolvedPath.hasValue() == false)
		{
			PX_ASSERT(false);
			return None();
		}

		Option<PxI32> idxOpt(indexOfArrayElement(resolvedPath, cmpData, dataLen, incomingType));
		if(idxOpt.hasValue() == false)
			return idxOpt;
		PxI32 idx(idxOpt);
		if(idx < 0)
			return idx;
		PxU32 numElements = resolvedPath->mArray->numberOfItems(resolvedPath->mPropertyDatatypeSize);
		if(idx >= (PxI32)numElements)
			return -1;
		doRemoveArrayElement(resolvedPath, idx);
		return idx;
	}

	virtual bool setArrayElementValue(void* instId, PxI32 propId, PxI32 propIdx, const PxU8* data, PxU32 dataLen, PxI32 incomingType)
	{
		Option<ResolvePropertyInstanceDataResult> resolvedPathOpt(resolvePropertyInstanceData(instId, propId));
		if(resolvedPathOpt.hasValue() == false)
		{
			PX_ASSERT(false);
			return false;
		}
		ResolvePropertyInstanceDataResult resolvedPath(resolvedPathOpt);

		if(resolvedPath.mArray == NULL)
		{
			PX_ASSERT(false);
			return false;
		}
		if(propIdx < 0)
		{
			PX_ASSERT(false);
			return false;
		}

		PxU32 numItems = resolvedPath.mArray->numberOfItems(resolvedPath.mPropertyDatatypeSize);
		PxU32 idx = (PxU32)propIdx;
		if(idx >= numItems)
		{
			PX_ASSERT(false);
			return false;
		}
		if(incomingType >= 0 && incomingType != resolvedPath.mProperty.mDatatype)
		{
			PX_ASSERT(false);
			return false;
		}
		if(dataLen != resolvedPath.mPropertyDatatypeSize)
		{
			PX_ASSERT(false);
			return false;
		}
		physx::intrinsics::memCopy(resolvedPath.mArray->begin() + idx * resolvedPath.mPropertyDatatypeSize, data, dataLen);
		setInstanceDirty(*(reinterpret_cast<InstanceHeader*>(instId)));

		return true;
	}

	virtual PxU32 getNbArrayElements(void* instId, PxI32 propId)
	{
		Option<ResolvePropertyInstanceDataResult> resolvedPath(resolvePropertyInstanceData(instId, propId));
		if(resolvedPath.hasValue() == false)
		{
			//            PX_ASSERT(false);
			return 0;
		}
		if(resolvedPath->mArray == NULL)
		{
			//            PX_ASSERT(false);
			return 0;
		}

		return resolvedPath->mArray->byteSize() / resolvedPath->mPropertyDatatypeSize;
	}

	virtual PxU32 getNbInstances() const { return static_cast<PxU32>(mAliveSet.size()); }
	virtual PxU32 getInstances(InstanceDescription* outBuffer, PxU32 count, PxU32 startIdx = 0) const
	{
		if(outBuffer == NULL || count == 0)
			return 0;
		PxU32 numPossible = mAliveSet.size();
		startIdx = PxMin(startIdx, numPossible);
		PxU32 numItems = PxMin(numPossible - startIdx, count);
		PVD_FOREACH(idx, numItems) { outBuffer[idx] = *mAliveSet[startIdx + idx]; }
		return numItems;
	}
	// Get the list of updated objects since the last time someone cleared the updated instance list.
	virtual PxU32 getNbUpdatedInstances() const { return mIsCapture ? mNbCompletedEvt : mUpdateSet.size(); }

	virtual PxU32 getUpdatedInstances(InstanceDescription* outBuffer, PxU32 count, PxU32 startIdx = 0)
	{
		if(outBuffer == NULL || count == 0)
			return 0;
		PxU32 numPossible = mIsCapture ? mNbCompletedEvt : mUpdateSet.size();
		startIdx = PxMin(startIdx, numPossible);
		PxU32 numItems = PxMin(numPossible - startIdx, count);
		PVD_FOREACH(idx, numItems) { outBuffer[idx] = *mUpdateSet[startIdx + idx]; }
		mNbCompletedEvt = 0;
		return numItems;
	}

#if USE_PROPERTY_CACHE
	bool removeInstancePropertyCache(PxI32 id)
	{
		const TInstancePropertyCacheMap::Entry* entry = mInstancePropertyPtrs.find(id);
		if(entry != NULL)
		{
			TPropertyPtrMap* map = entry->second;
			map->~TPropertyPtrMap();
			mAlloc.deallocate(map);
			return true;
		}
		return false;
	}
#endif

	void maybeDeleteInstance(InstanceHeader* instance)
	{
		if(instance->isReferenced() == false)
		{
#if USE_PROPERTY_CACHE
			if(removeInstancePropertyCache(instance->mId))
				mInstancePropertyPtrs.erase(instance->mId);
#endif
			mInstances.erase(instance->mId);
			destructInstance(instance);
		}
	}

	// The only way something can be destroyed
	virtual void clearUpdatedInstances()
	{
		PxU32 nb = mIsCapture ? mNbCompletedEvt : mUpdateSet.size();
		for(PxU32 i = 0; i < nb; i++)
		{
			InstanceHeader* instance(mUpdateSet[0]);
			mUpdateSet.remove(*instance);
			maybeDeleteInstance(instance);
		}
		mNbCompletedEvt = 0;
	}

	void writeInstanceData(PvdOutputStream& stream, PxU8* start, PxU8* stop, const ClassDescription& description)
	{
		PxU32 amountToWrite = static_cast<PxU32>(stop - start);
		stream << amountToWrite;
		stream.write(start, static_cast<PxU32>(stop - start));
		// Go through the properties writing out each array as we get to it.
		PxU32 totalNumProps = mMetaData.getNbProperties(description.mClassId);
		PxU32 offset = 0;
		PropertyDescription descs[16];
		for(PxU32 idx = 0; idx < totalNumProps; idx = offset)
		{
			PxU32 numProps = mMetaData.getProperties(description.mClassId, descs, 16, offset);
			offset += numProps;
			for(PxU32 propIdx = 0; propIdx < numProps; ++propIdx)
			{
				const PropertyDescription& prop(descs[propIdx]);
				PxU8* propStart = start + prop.m32BitOffset;
				Option<ClassDescription> propCls(mMetaData.getClass(prop.mDatatype));
				if(propCls.hasValue() == false)
				{
					PX_ASSERT(false);
					return;
				}
				const ClassDescription& propType(propCls);
				PxU32 propSize = prop.mPropertyType == PropertyType::Array ? ORIGIN_ARRAYDATA_SIZE : propType.get32BitSize();
				PxU8* propEnd = propStart + propSize;
				// we are done.
				if(propEnd > stop)
					return;
				if(prop.mPropertyType == PropertyType::Array)
				{
					ArrayData* propData = getOrReplaceArrayData(propStart);
					PxU32 arraySize = propData->byteSize();
					stream << arraySize;
					if(arraySize)
					{
						stream.write(propData->begin(), propData->byteSize());
						if(propType.mRequiresDestruction) // christ.
						{
							PxU8* item = propData->begin();
							PxU8* itemEnd = propData->end();
							for(; item < itemEnd; item += propCls->get32BitSize())
							{
								writeInstanceData(stream, item, item + propCls->get32BitSize(), propType);
							}
						}
					}
				}
				else if(propType.mRequiresDestruction)
				{
					writeInstanceData(stream, propStart, propStart + propType.get32BitSize(), propType);
				}
			}
		}
	}

	virtual void writeInstance(void* instId, PvdOutputStream& stream)
	{
		InstanceHeader* header(reinterpret_cast<InstanceHeader*>(instId));
		if(header == NULL)
		{
			PX_ASSERT(false);
			return;
		}
		Option<ClassDescription> clsDescOpt(mMetaData.getClass(header->mClassId));
		if(clsDescOpt.hasValue() == false)
		{
			PX_ASSERT(false);
			return;
		}
		if(!resizeInstance(header, clsDescOpt))
			return;
		const ClassDescription cls(clsDescOpt);
		stream << header->mClassId;
		stream << header->mId;
		mInstanceWriteBuffer.clear();
		writeInstanceData(mInstanceWriteBuffer, header->getDataSection(), header->getDataSection() + header->mDataSize, cls);
		DataRef<const PxU8> bufData = mInstanceWriteBuffer.toRef();
		stream << (PxU32)bufData.size();
		stream.write(bufData.begin(), bufData.size());
		// else we don't change the header at all.
	}

	void readInstanceData(PvdInputStream& stream, PxU8* instStart, PxU8* instEnd, const ClassDescription* cls)
	{
		PxU32 byteCount;
		stream >> byteCount;
		const PxU32 available = static_cast<PxU32>(instEnd - instStart);
		PX_ASSERT(byteCount <= cls->get32BitSize());
		PX_ASSERT(available <= cls->get32BitSize());
		PX_ASSERT(available >= byteCount);
		stream.read(instStart, available);
		PX_ASSERT(available == static_cast<PxU32>(instEnd - instStart));
		// Go through the properties writing out each array as we get to it.
		const PxU32 numProps = mMetaData.getNbProperties(cls->mClassId);
		PxU32 offset = 0;
		char buff[16 * sizeof(PropertyDescription)] = { 0 };
		for(PxU32 idx = 0; idx < numProps; idx = offset)
		{
			PropertyDescription* descs = (PropertyDescription*)(buff);
			PxU32 _numProps = mMetaData.getProperties(cls->mClassId, descs, 16, offset);
			offset += _numProps;
			for(PxU32 propIdx = 0; propIdx < _numProps; ++propIdx)
			{
				const PropertyDescription& prop = descs[propIdx];
				PxU8* propStart = instStart + prop.m32BitOffset;
				const ClassDescription* propCls = mMetaData.getClassPtr(prop.mDatatype);
				const PxU32 propSize = prop.mPropertyType == PropertyType::Array ? ORIGIN_ARRAYDATA_SIZE : propCls->get32BitSize();
				const PxU8* propEnd = propStart + propSize;
				// we are done.
				if(propEnd > instEnd)
					return;
				if(propCls->mRequiresDestruction)
				{
					PX_ASSERT(false);
					return;
				}
				if(prop.mPropertyType == PropertyType::Array)
				{
					PxU32 arraySize;
					stream >> arraySize;

					ArrayData* propData = getOrReplaceArrayData(propStart);
// Ensure it knows it was just read from file.
#ifndef WIN64
					propData->forgetData();
#endif

					if(propData->begin() == 0 || propData->byteSize() != arraySize)
					{
						if(!resizeArray(*propData, propCls->get32BitSize(), arraySize))
							return;
					}

					if(arraySize)
					{
						PxU32 amountRead = propData->byteSize();
						stream.read(propData->begin(), amountRead);
						PX_ASSERT(amountRead == propData->byteSize());
					}
				}
			}
		}
	}

	virtual InstanceDescription readInstance(DataRef<const PxU8> dataBuffer)
	{
		MemPvdInputStream stream(dataBuffer.begin(), dataBuffer.end());
		// Read three PxI32s in one go. Used to be an anonymous union around
		// a named struct and a PxI32[3] — MSVC extension, GCC requires
		// -fms-extensions. Flat array with explicit field aliases is portable.
		// read() takes `uint32_t& len` by non-const ref, so pass a local.
		PxI32 a[3];
		uint32_t l = (uint32_t)sizeof(a);
		stream.read((uint8_t*)a, l);
		const PxI32 clsId = a[0];
		const PxI32 instId = a[1];
		const PxU32 byteCount = (PxU32)a[2];
		//        stream >> clsId;
		//        stream >> instId;
		//        stream >> byteCount;
		if(stream.size() < byteCount)
		{
			PX_ASSERT(false);
			return InstanceDescription();
		}
		const InstanceDescription& retval(createInstance(clsId, instId));
		// failed to create the instance.
		if(retval.mInstPtr == NULL)
		{
			PX_ASSERT(false);
			return retval;
		}
		InstanceHeader* hdr(reinterpret_cast<InstanceHeader*>(retval.mInstPtr));
		PX_ASSERT(hdr->isAlive() && hdr->isUpdated());

		if(hdr->mDataSize != 0)
		{
			const ClassDescription* cls = mMetaData.getClassPtr(hdr->mClassId);
			readInstanceData(stream, hdr->getDataSection(), hdr->getDataSection() + hdr->mDataSize, cls);
		}

		setInstanceDirty(*hdr);
		return *hdr;
	}

	virtual bool readInstanceProperty(DataRef<const PxU8> dataBuffer, PxI32 propId)
	{
		MemPvdInputStream stream(dataBuffer.begin(), dataBuffer.end());
		//        PxI32 clsId;
		//        PxI32 instId;
		//        PxU32 dataSize;
		//        PxU32 byteCount;

		// Same MSVC-extension rewrite as above — four PxI32-sized fields read
		// at once, aliases extracted afterward. read() takes a non-const
		// lvalue reference for the length, so use a local variable.
		PxI32 a[4];
		uint32_t l = (uint32_t)sizeof(a);
		stream.read((uint8_t*)a, l);
		const PxI32 clsId = a[0];
		const PxI32 instId = a[1];
		const PxU32 dataSize = (PxU32)a[2];
		const PxU32 byteCount = (PxU32)a[3];
		(void)dataSize; // declared for byte-accurate layout; unused in this path

		//        stream >> clsId;
		//        stream >> instId;
		//        stream >> dataSize;
		//        stream >> byteCount;

		void* inst = idToPtr(instId);
		if(inst == NULL)
		{
			PX_ASSERT(false);
			return false;
		}
		Option<ResolvePropertyInstanceDataResult> resolvedPath(resolvePropertyInstanceData(inst, propId));
		if(resolvedPath.hasValue() == false)
		{
			PX_ASSERT(false);
			return false;
		}
		InstanceHeader* hdr(reinterpret_cast<InstanceHeader*>(inst));
		setInstanceDirty(*hdr);
		if(resolvedPath->mProperty.mPropertyType != PropertyType::Array)
		{
			ClassDescription* propType = mMetaData.getClassPtr(resolvedPath->mProperty.mDatatype);
			if(propType->mRequiresDestruction)
			{
				PX_ASSERT(false);
				return false;
			}
			const PxU8* dataStart = stream.mBegin + resolvedPath->mProperty.m32BitOffset;
			const PxU8* dataEnd = dataStart + propType->get32BitSize();
			if((PxU32)(dataEnd - stream.mBegin) > byteCount)
			{
				PX_ASSERT(false);
				return false;
			}
			return setScalarPropertyValue(dataStart, dataEnd,
				const_cast<PxU8*>(resolvedPath->mData.begin()),
				const_cast<PxU8*>(resolvedPath->mData.end()), propType);
		}
		else
		{
			stream.mBegin = stream.mBegin + byteCount;
			if(stream.mBegin >= stream.mEnd)
			{
				PX_ASSERT(false);
				return false;
			}
			PropertyDescription descs[16];
			PxU32 propOffset = 0;
			PxU32 numProps = mMetaData.getNbProperties(clsId);
			while(propOffset < numProps)
			{
				PxU32 numReturned = mMetaData.getProperties(clsId, descs, 16, propOffset);
				propOffset += numReturned;
				for(PxU32 idx = 0; idx < numReturned; ++idx)
				{
					const PropertyDescription& propDesc(descs[idx]);
					if(propDesc.mPropertyId != propId)
					{
						if(propDesc.mPropertyType == PropertyType::Array)
						{
							// the system is too complex for this operation.
							if(mMetaData.getClassPtr(propDesc.mDatatype)->mRequiresDestruction)
							{
								PX_ASSERT(false);
								return false;
							}
							// Skip the array in the stream.
							PxU32 arrayByteSize;
							stream >> arrayByteSize;
							// bump read ptr to next item.
							stream.mBegin += arrayByteSize;
							if(stream.mBegin >= stream.mEnd)
							{
								PX_ASSERT(false);
								return false;
							}
						}
					}
					else
					{
						if(propDesc.mPropertyType != PropertyType::Array)
						{
							PX_ASSERT(false);
							return false;
						}
						// we got the property.
						PxU32 arrayByteSize;
						stream >> arrayByteSize;
						const PxU8* dataStart = stream.mBegin;
						const PxU8* dataEnd = dataStart + arrayByteSize;
						if(dataEnd > stream.mEnd)
						{
							PX_ASSERT(false);
							return false;
						}
						return setPropertyValue(inst, propId, stream.mBegin, arrayByteSize, -1);
					}
				}
			}
		}
		return false;
	}
	virtual void addRef() { ++mRefCount; }
	virtual void release()
	{
		if(mRefCount)
			--mRefCount;
		if(!mRefCount)
			PVD_DELETE(this);
	}

	DataRef<ObjectRef> getObjectArrayProperty(InstanceHeader* instance, const String arrayProp, Array<PxU8>& dataBuffer)
	{
		Option<PropertyDescription> prop = getMetaData().findProperty(instance->mClassId, arrayProp);
		if(prop.hasValue())
		{
			PropertyDescription& propValue = prop.getValue();
			if(propValue.mDatatypeName == NamespacedName("physx3", "ObjectRef"))
			{
				DataRef<ObjectRef> objArray = getArray<ObjectRef>(dataBuffer, *this, (InstanceDescription)(*instance), arrayProp);

				return objArray;
			}
		}

		return DataRef<ObjectRef>();
	}

	template <typename TOperator>
	void visitObjectArrayProp(InstanceHeader* instance, const String arrayProp, TOperator& oper)
	{
		Array<PxU8> dataBuffer("visitArrayProp");
		DataRef<ObjectRef> items = getObjectArrayProperty(instance, arrayProp, dataBuffer);

		for(const ObjectRef* obj = items.begin(); obj != items.end(); obj++)
		{
			if(!obj->hasValue())
				continue;

			void* objInst = idToPtr((PxI32)(*obj));
			InstanceHeader* actorDesc = reinterpret_cast<InstanceHeader*>(objInst);
			oper.visit(actorDesc);
		}
	}

	// For articulation link
	void recruseShiftOrignin(InstanceHeader* instance, const String arrayProp, PxVec3 shift)
	{
		shiftObject(instance, 0, shift);
		setInstanceDirty(*instance);

		Option<PropertyDescription> prop = getMetaData().findProperty(instance->mClassId, arrayProp);
		if(!prop.hasValue())
			return;

		struct Visitor
		{
			const String mArrayProp;
			PxVec3 mShift;
			PvdObjectModelImpl* mModel;

			Visitor(const String inProp, PxVec3 inShift, PvdObjectModelImpl* inModel) : mArrayProp(inProp), mShift(inShift), mModel(inModel) {}

			void visit(InstanceHeader* instance) { mModel->recruseShiftOrignin(instance, mArrayProp, mShift); }
		} visitor(arrayProp, shift, this);

		visitObjectArrayProp(instance, arrayProp, visitor);
	}

	struct RecruseVisitor
	{
		const String mArrayProp;
		PxVec3 mShift;
		PvdObjectModelImpl* mModel;

		RecruseVisitor(const String inProp, PxVec3 inShift, PvdObjectModelImpl* inModel) : mArrayProp(inProp), mShift(inShift), mModel(inModel) {}

		void visit(InstanceHeader* instance) { mModel->recruseShiftOrignin(instance, mArrayProp, mShift); }
	};

	void originShiftArticulations(InstanceHeader* sceneInst, PxVec3 shift)
	{
		RecruseVisitor visitor("Links", shift, this);

		visitObjectArrayProp(sceneInst, "Articulations", visitor);
	}

	// For those have GlobalPose and WorldBounds properties
	void originShiftArrayItems(InstanceHeader* instance, const String arrayProp, PxVec3 shift)
	{
		struct Visitor
		{
			PxVec3 mShift;
			PvdObjectModelImpl* mModel;

			Visitor(PxVec3 inShift, PvdObjectModelImpl* inModel) : mShift(inShift), mModel(inModel) {}

			void visit(InstanceHeader* instance) { mModel->shiftObject(instance, 0, mShift); }
		} visitor(shift, this);

		visitObjectArrayProp(instance, arrayProp, visitor);
	}

	virtual void originShift(void* instId, PxVec3 shift)
	{
		// This set should only meanful for this shift
		mShiftedInstances.clear();
		if(!instId)
			return;
		InstanceHeader* instance = reinterpret_cast<InstanceHeader*>(instId);
		if(instance->mClassName == NamespacedName("physx3", "PxScene"))
		{
			originShiftArrayItems(instance, "RigidStatics", shift);
			originShiftArrayItems(instance, "RigidDynamics", shift);
			originShiftArticulations(instance, shift);
			originShiftArrayItems(instance, "Cloths", shift);
			originShiftArrayItems(instance, "ParticleFluids", shift);

			//"RigidDynamics"
			//"Articulations"
			//"ParticleSystems"
			//"ParticleFluids"
			//"Cloths"
			//"Joints"
			//"Constraints"
		}
		// this->getMetaData().findProperty(cls, prop);
	}

	virtual PxU32 getNbShifted() { return static_cast<PxU32>(mShiftedInstances.size()); }

	virtual void getShiftedPair(InstancePropertyPair* outData, PxU32 count)
	{
		PxU32 idx = 0;
		for(TShiftedInfoSet::Iterator theIter = mShiftedInstances.getIterator(); !theIter.done() && idx != count; ++theIter, ++idx)
		{
			outData[idx] = *theIter;
		}
	}

	virtual void clearShiftedPair() { mShiftedInstances.clear(); }

	virtual void shiftObject(void* instId, PxI32 propId, PxVec3 shift)
	{
		InstanceHeader* instance = reinterpret_cast<InstanceHeader*>(instId);

		const String poseProp = "GlobalPose";
		const String boundProp = "WorldBounds";
		const String kinemicTargetProp = "KinematicTarget";
		const String partilceProjectPlaneProp = "ProjectionPlane";

		PxU32 shiftedPropCount = 0;

		Option<PropertyDescription> posePropDesc(getMetaData().findProperty(instance->mClassId, poseProp));
		if(posePropDesc.hasValue() && (propId == 0 || posePropDesc->mPropertyId == propId))
		{
			Option<PxTransform> pose = PvdObjectModelReader::getPropertyValue<PxTransform>(instance, poseProp);
			;
			if(pose.hasValue())
			{
				pose.getValue().p -= shift;
				PvdObjectModelMutator::setPropertyValue(instance, poseProp, pose.getValue());
				mShiftedInstances.insert(InstancePropertyPair(instance->mId, posePropDesc->mPropertyId));
				shiftedPropCount++;
			}
		}

		Option<PropertyDescription> boundPropDesc(getMetaData().findProperty(instance->mClassId, boundProp));
		if(boundPropDesc.hasValue() && (propId == 0 || boundPropDesc->mPropertyId == propId))
		{
			Option<PxBounds3> bound = PvdObjectModelReader::getPropertyValue<PxBounds3>(instance, boundProp);
			;
			if(bound.hasValue())
			{
				bound.getValue().minimum -= shift;
				bound.getValue().maximum -= shift;
				PvdObjectModelMutator::setPropertyValue(instance, boundProp, bound.getValue());
				mShiftedInstances.insert(InstancePropertyPair(instance->mId, boundPropDesc->mPropertyId));
				shiftedPropCount++;
			}
		}

		Option<PropertyDescription> kinemicTargetPropDesc(getMetaData().findProperty(instance->mClassId, kinemicTargetProp));
		if(kinemicTargetPropDesc.hasValue() && (propId == 0 || kinemicTargetPropDesc->mPropertyId == propId))
		{
			Option<PxTransform> kinTarget = PvdObjectModelReader::getPropertyValue<PxTransform>(instance, kinemicTargetProp);
			;
			if(kinTarget.hasValue())
			{
				kinTarget.getValue().p -= shift;
				PvdObjectModelMutator::setPropertyValue(instance, kinemicTargetProp, kinTarget.getValue());
				mShiftedInstances.insert(InstancePropertyPair(instance->mId, kinemicTargetPropDesc->mPropertyId));
				shiftedPropCount++;
			}
		}

		Option<PropertyDescription> partilceProjectPlanePropDesc(getMetaData().findProperty(instance->mClassId, partilceProjectPlaneProp));
		if(partilceProjectPlanePropDesc.hasValue() && (propId == 0 || partilceProjectPlanePropDesc->mPropertyId == propId))
		{
			Option<PxVec4> plane = PvdObjectModelReader::getPropertyValue<PxVec4>(instance, partilceProjectPlaneProp);
			;
			if(plane.hasValue())
			{
				PxReal delta = plane.getValue().getXYZ().dot(shift);
				plane.getValue().w += delta;

				PvdObjectModelMutator::setPropertyValue(instance, partilceProjectPlaneProp, plane.getValue());
				mShiftedInstances.insert(InstancePropertyPair(instance->mId, partilceProjectPlanePropDesc->mPropertyId));
				shiftedPropCount++;
			}
		}

		setInstanceDirty(*instance);

		if(shiftedPropCount > 2)
		{
			PX_ASSERT("Shifted more than 2 properties, OriginShiftPreviousEvent can only handle 2 properties");
		}
	}
};
} // namespace

bool PvdObjectModel::readInstance(PvdInputStream& inStream, PvdOutputStream& outStream)
{
	//	PxI32 clsId;
	//	PxI32 instId;
	//	PxU32 instLen;
	//	inStream >> clsId;
	//	inStream >> instId;
	//	inStream >> instLen;
	PxI32 a[3];
	inStream.read(a, 3);
	//	outStream << clsId;
	//	outStream << instId;
	//	outStream << instLen;
	outStream.write(a, 3);
	PxU32 instLen = a[2];
	outStream.directCopy(inStream, instLen);
	return true;
}

PvdObjectModel& PvdObjectModel::create(physx::PxAllocatorCallback& callback, PvdObjectModelMetaData& metaData, bool isCapture) { return *PVD_NEW(PvdObjectModelImpl)(callback, metaData, isCapture); }
