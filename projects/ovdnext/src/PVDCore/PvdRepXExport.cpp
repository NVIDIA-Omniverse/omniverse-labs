/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "PvdRepXExport.h"
#include "PVDClient.h"
#include "PvdRandomAccessIOStream.h"
#include "PvdSimpleAllocator.h"
#include "PxPvdFoundation.h"
#include "PxPvdInternalByteStreams.h"
#include "PxPvdObjectModel.h"
#include "PxPvdObjectModelBaseTypes.h"
#include "PxPvdObjectModelMetaData.h"
#include "PxSimpleTypes.h"
#include "PxTransform.h"
#include "SimpleXmlWriter.h"
#include <stdio.h>
#include <stdlib.h>

using namespace pvdapp;
using namespace pvdapp::repx;
using namespace physx::pvdsdk;

namespace
{

/**
\brief Describe type of phase in cloth fabric.
\see PxClothFabric for an explanation of concepts on phase and set.
*/
struct PxClothFabricPhaseType
{
	enum Enum
	{
		eINVALID,    //!< invalid type
		eVERTICAL,   //!< resists stretching or compression, usually along the gravity
		eHORIZONTAL, //!< resists stretching or compression, perpendicular to the gravity
		eBENDING,    //!< resists out-of-plane bending in angle-based formulation
		eSHEARING,   //!< resists in-plane shearing along (typically) diagonal edges,
		eCOUNT       // internal use only
	};
};

struct PxClothFabricPhase
{
	PxClothFabricPhase(PxClothFabricPhaseType::Enum type = PxClothFabricPhaseType::eINVALID, PxU32 index = 0);

	/**
	\brief Type of constraints to solve.
	*/
	PxClothFabricPhaseType::Enum phaseType;

	/**
	\brief Index of the set that contains the particle indices.
	*/
	PxU32 setIndex;
};

inline void writeProperty_ClothFabricPhase(XmlWriter& inWriter, ForwardingMemoryBuffer& tempBuffer, const char* inPropName, DataRef<NamedValue> namedValues, const PxClothFabricPhase* phase,
                                           PxU32 numItems)
{
	tempBuffer.clear();
	PVD_FOREACH(idx, numItems)
	{
		if(idx)
			tempBuffer << " ";
		PxU32 realValue = phase[idx].phaseType;

		PVD_FOREACH(valIdx, namedValues.size())
		{
			if(realValue == namedValues[valIdx].mValue)
				tempBuffer << namedValues[valIdx].mName;
		}

		tempBuffer << " " << phase[idx].setIndex;
	}

	writeProperty(inWriter, tempBuffer, inPropName);
}

template <typename TDataType>
inline void writeProperty(XmlWriter& inWriter, physx::pvdsdk::ForwardingMemoryBuffer& tempBuffer, const char* inPropName, const TDataType* inStart, const TDataType* inStop, PxU32 inTabCount,
                          PxU32 inItemsPerLine = 6)
{
	PxU32 idx = 0;
	PxU32 itemCount = static_cast<PxU32>(inStop - inStart);
	if(itemCount > 9)
	{
		inWriter.beginTag(inPropName);

		for(const TDataType* theIter = inStart; theIter != inStop; ++theIter, ++idx)
		{
			if(idx && ((idx % inItemsPerLine) == 0))
			{
				inWriter.addContent(tempBuffer.cStr());
				tempBuffer.clear();
				inWriter.writeTabs(inTabCount, true);
			}
			else if(idx)
				tempBuffer << " ";
			tempBuffer << *theIter;
		}

		inWriter.addContent(tempBuffer.cStr());
		tempBuffer.clear();

		inWriter.endTag(false);
	}
	else
	{
		for(const TDataType* theIter = inStart; theIter != inStop; ++theIter, ++idx)
		{
			if(idx)
				tempBuffer << " ";
			tempBuffer << *theIter;
		}
		writeProperty(inWriter, tempBuffer, inPropName);
	}
}

template <typename TDataType>
inline void writeEnumOrBitflagType(XmlWriter& writer, physx::pvdsdk::ForwardingMemoryBuffer& tempBuffer, const char* inPropName, bool isEnumeration, DataRef<NamedValue> namedValues,
                                   const PxU8* propData, PxU32 propByteSize)
{
	tempBuffer.clear();
	const TDataType* dataPtr = reinterpret_cast<const TDataType*>(propData);
	PxU32 numItems = propByteSize / sizeof(TDataType);
	PVD_FOREACH(idx, numItems)
	{
		if(idx)
			tempBuffer << "\n";
		PxU32 realValue = static_cast<PxU32>(dataPtr[idx]);
		if(isEnumeration)
		{
			PVD_FOREACH(valIdx, namedValues.size())
			{
				if(realValue == namedValues[valIdx].mValue)
					tempBuffer << namedValues[valIdx].mName;
			}
		}
		else
		{
			bool written = false;
			PVD_FOREACH(valIdx, namedValues.size())
			{
				if((realValue & namedValues[valIdx].mValue) == namedValues[valIdx].mValue)
				{
					if(written)
						tempBuffer << "|";
					tempBuffer << namedValues[valIdx].mName;
					written = true;
				}
			}
		}
	}
	writeProperty(writer, tempBuffer, inPropName);
}

inline bool writeBaseProperty(XmlWriter& writer, physx::pvdsdk::ForwardingMemoryBuffer& tempBuffer, const PvdObjectModelReader& reader, const PropertyDescription& prop, const char* propName,
                              const PxU8* propData, PxU32 propByteSize, PxI32 propType)
{
	const PxU8* dtypeEnd = propData + propByteSize;
	bool isEnumeration = safeStrEq(prop.mSemantic, "Enumeration Value");
	bool isBitflag = safeStrEq(prop.mSemantic, "Bitflag");
	if(isEnumeration || isBitflag)
	{
		switch(propType)
		{
#define HANDLE_BASE_DATATYPE(type)                                                                                                                                                                     \
	case PvdBaseType::type:                                                                                                                                                                            \
		writeEnumOrBitflagType<type>(writer, tempBuffer, propName, isEnumeration, reader.getMetaData().getNamedPropertyValues(prop.mPropertyId), propData, propByteSize);                              \
		return true;

			HANDLE_BASE_DATATYPE(PvdI8)
			HANDLE_BASE_DATATYPE(PvdU8)
			HANDLE_BASE_DATATYPE(PvdI16)
			HANDLE_BASE_DATATYPE(PvdU16)
			HANDLE_BASE_DATATYPE(PvdI32)
			HANDLE_BASE_DATATYPE(PvdU32)
			HANDLE_BASE_DATATYPE(PvdI64)
			HANDLE_BASE_DATATYPE(PvdU64)

#undef HANDLE_BASE_DATATYPE
		}
	}
	else
	{
		switch(propType)
		{
#define HANDLE_BASE_DATATYPE(type)                                                                                                                                                                     \
	case PvdBaseType::type:                                                                                                                                                                            \
		writeProperty(writer, tempBuffer, propName, reinterpret_cast<const type*>(propData), reinterpret_cast<const type*>(dtypeEnd), 4);                                                              \
		return true;
			HANDLE_BASE_DATATYPE(PvdI8)
			HANDLE_BASE_DATATYPE(PvdU8)
			HANDLE_BASE_DATATYPE(PvdI16)
			HANDLE_BASE_DATATYPE(PvdU16)
			HANDLE_BASE_DATATYPE(PvdI32)
			HANDLE_BASE_DATATYPE(PvdU32)
			HANDLE_BASE_DATATYPE(PvdI64)
			HANDLE_BASE_DATATYPE(PvdU64)
			HANDLE_BASE_DATATYPE(PvdF32)
			HANDLE_BASE_DATATYPE(PvdF64)
			HANDLE_BASE_DATATYPE(PvdBool)
#undef HANDLE_BASE_DATATYPE
		case PvdBaseType::StringHandle:
		{
			PX_ASSERT(propByteSize == 4);
			if(propByteSize >= 4)
			{
				PxU32 strHandle;
				physx::intrinsics::memCopy(&strHandle, propData, 4);
				tempBuffer.clear();
				const char* strVal = reader.getMetaData().getStringTable().handleToStr(strHandle);
				tempBuffer.write(strVal, strLen(strVal));
				writeProperty(writer, tempBuffer, propName);
				return true;
			}
			break;
		}
		case PvdBaseType::ObjectRef:
		{
			MallocAllocator allocator;
			Array<PxI32> objRefData("writeBaseProperty::objectRef");
			const PxI32* dataStart = reinterpret_cast<const PxI32*>(propData);
			const PxI32* dataStop = reinterpret_cast<const PxI32*>(dtypeEnd);
			PxU32 numItems = static_cast<PxU32>(dataStop - dataStart);
			PVD_FOREACH(idx, numItems)
			{
				if(reader.idToDescriptor(dataStart[idx]).mAlive)
					objRefData.pushBack(dataStart[idx]);
				else
					objRefData.pushBack(0);
			}
			if(objRefData.size())
				writeProperty(writer, tempBuffer, propName, objRefData.begin(), objRefData.end(), 4);
			return true;
		}
		}
	}
	return false;
}

inline void writeGenericProperty(XmlWriter& writer, ForwardingMemoryBuffer& tempBuffer, const PvdObjectModelReader& reader, const PropertyDescription& prop, const char* propName, const PxU8* propData,
                                 PxU32 propByteSize)
{
	ClassDescription dtype(reader.getMetaData().getClass(prop.mDatatype));
	if(dtype.mPackedClassType >= 0)
		writeBaseProperty(writer, tempBuffer, reader, prop, propName, propData, propByteSize, dtype.mPackedClassType);
	else
		writeBaseProperty(writer, tempBuffer, reader, prop, propName, propData, propByteSize, prop.mDatatype);
}

inline void writeProperty(XmlWriter& writer, ForwardingMemoryBuffer& tempBuf, const char* propName, const InstanceDescription& desc) { writeProperty(writer, tempBuf, propName, desc.mId); }

static int compareProperty(const void* c0, const void* c1)
{
	PropertyDescription* p0 = (PropertyDescription*)c0;
	PropertyDescription* p1 = (PropertyDescription*)c1;
	return ::strcmp(p0->mName, p1->mName);
}

struct Exporter
{
	PxAllocatorCallback& mAllocator;
	XmlWriterImpl<PvdOutputStream> mBaseWriter;
	ForwardingMemoryBuffer mPropertyStream;
	ForwardingMemoryBuffer shapes;
	ForwardingMemoryBuffer scenes;
	ForwardingMemoryBuffer joints;
	// Saved resources go here.
	HashSet<PxI32> mResourceSet;
	XmlWriterImpl<ForwardingMemoryBuffer> mShapeWriter;
	XmlWriterImpl<ForwardingMemoryBuffer> mSceneWriter;
	XmlWriterImpl<ForwardingMemoryBuffer> mJointWriter;
	PxU32 mVersionMajor;
	PxU32 mVersionMinor;
	PxU32 mVersionBugfix;
	// Joint map, avoid export duplicated joints
	HashSet<PxI32> mExportedJoints;
	// shared shapes map, avoid export duplicated shapes
	HashSet<PxI32> mExportedShapes;
	const PvdObjectModelMetaData& mMetaData;
	PvdObjectModelReader& mReader;

	Array<PxI32> mCachedClassIdMap;

	Exporter(PxAllocatorCallback& alloc, PvdOutputStream& fileStream, const PvdObjectModelReader& reader)
	: mAllocator(alloc)
	, mBaseWriter(fileStream)
	, mPropertyStream("Exporter::mPropertyStream")
	, shapes("Exporter::shapes")
	, scenes("Exporter::scenes")
	, joints("Exporter::joints")
	, mShapeWriter(shapes, 1)
	, mSceneWriter(scenes, 1)
	, mJointWriter(joints, 1)
	, mVersionMajor(0)
	, mVersionMinor(0)
	, mVersionBugfix(0)
	, mMetaData(reader.getMetaData())
	, mResourceSet("Exporter::mResourceSet")
	, mReader(const_cast<PvdObjectModelReader&>(reader))
	, mExportedJoints("Exporter::mExportedJoints")
	, mExportedShapes("Exporter::mExportedShapes")
	{
		mCachedClassIdMap.pushBack(getClassId("PxRigidStatic"));
		mCachedClassIdMap.pushBack(getClassId("PxRigidDynamic"));
		mCachedClassIdMap.pushBack(getClassId("PxArticulation"));
		mCachedClassIdMap.pushBack(getClassId("PxCloth"));
		mCachedClassIdMap.pushBack(getClassId("PxAggregate"));
		mCachedClassIdMap.pushBack(getClassId("PxParticleSystem"));
		mCachedClassIdMap.pushBack(getClassId("PxParticleFluid"));
		mCachedClassIdMap.pushBack(getClassId("PxScene"));
	}

	PxI32 getClassId(const char* typeName, const char* ns = "physx3") { return mMetaData.findClass(NamespacedName(ns, typeName))->mClassId; }

	void serialize(physx::shdfnd::Array<uint64_t>& objects)
	{
		Array<InstanceDescription> descs("Exporter::serialize::descs");
		PxU32 numDescs = mReader.getNbInstances();
		descs.resize(numDescs);
		mReader.getInstances(descs.begin(), numDescs);
		PxI32 sdkClassId = getClassId("PxPhysics");
		InstanceDescription physicsDesc;
		PVD_FOREACH(idx, numDescs)
		{
			if(sdkClassId == descs[idx].mClassId)
			{
				physicsDesc = descs[idx];
				break;
			}
		}
		if(physicsDesc.mAlive == false)
			return;

		mVersionMajor = mReader.getPropertyValue<PxU32>(physicsDesc, "Version.Major");
		mVersionMinor = mReader.getPropertyValue<PxU32>(physicsDesc, "Version.Minor");
		mVersionBugfix = mReader.getPropertyValue<PxU32>(physicsDesc, "Version.Bugfix");

		char repxVersionStr[256];
#ifdef WIN32
		char itaBuf[256];
		_itoa_s(mVersionMinor, itaBuf, 10);

		strncpy_s(repxVersionStr, "3.", 256);
		strncat_s(repxVersionStr, itaBuf, 256);
		strncat_s(repxVersionStr, ".0", 256);
#endif
#ifndef WIN32
		::snprintf(repxVersionStr, 256, "%d.%d.%d", mVersionMajor, mVersionMinor, mVersionBugfix);
#endif
		mBaseWriter << BeginTag("PhysX30Collection") << Att("version", repxVersionStr);

		if(mVersionMajor > 3 || (mVersionMajor == 3 && mVersionMinor >= 3))
		{
			PxU32 numSharedShapes = mReader.getNbArrayElements(physicsDesc, "SharedShapes");
			if(numSharedShapes)
			{
				Array<PxI32> sharedShapes("SharedShapes");
				sharedShapes.resize(numSharedShapes);
				mReader.getPropertyValue(physicsDesc, "SharedShapes", sharedShapes.begin(), numSharedShapes);
				PVD_FOREACH(idx, numSharedShapes)
				{
					InstanceDescription shape(mReader.idToDescriptor(sharedShapes[idx]));
					if(shape.mAlive == false)
						continue;
					exportShape(shape);
				}
			}
		}

		PxU32 numScenes = mReader.getNbArrayElements(physicsDesc, "Scenes");
		if(numScenes == 0)
			return;
		Array<PxI32> sceneRefs("Scenes");
		sceneRefs.resize(numScenes);
		mReader.getPropertyValue(physicsDesc, "Scenes", sceneRefs.begin(), numScenes);
		InstanceDescription firstScene = mReader.idToDescriptor(sceneRefs[0]);

		PxVec3 gravity = mReader.getPropertyValue<PxVec3>(firstScene, "Gravity");
		PxVec3 upVec(0, 1, 0);
		if(gravity.normalize() > .01f)
			upVec = -gravity;

		writeProperty(mBaseWriter, mPropertyStream, "UpVector", upVec);

		PxF32 tolLen = mReader.getPropertyValue<PxF32>(physicsDesc, "TolerancesScale.Length");
		PxF32 tolMass = mReader.getPropertyValue<PxF32>(physicsDesc, "TolerancesScale.Mass");
		PxF32 tolSpeed = mReader.getPropertyValue<PxF32>(physicsDesc, "TolerancesScale.Speed");
		// Setup sane defaults if we can't get the property values.
		if(tolLen == 0 || tolMass == 0 || tolSpeed == 0)
		{
			tolLen = 1;
			tolMass = 1000;
			tolSpeed = 10;
		}

		mBaseWriter << BeginTag("Scale");
		writeProperty(mBaseWriter, mPropertyStream, "Length", tolLen);
		writeProperty(mBaseWriter, mPropertyStream, "Mass", tolMass);
		writeProperty(mBaseWriter, mPropertyStream, "Speed", tolSpeed);
		mBaseWriter << EndTag();

		uint32_t num = objects.size();
		if(num == 0)
		{
			PVD_FOREACH(idx, numScenes)
			{
				InstanceDescription scene(mReader.idToDescriptor(sceneRefs[idx]));
				if(scene.mAlive)
					exportScene(sceneRefs[idx]);
			}
		}
		else
		{
			PVD_FOREACH(idx, num)
			{
				uint32_t instId = static_cast<uint32_t>(objects[idx]);
				InstanceDescription Obj(mReader.idToDescriptor(instId));

				// should consisted with the mCachedClassIdMap pushback in class construct
				if(Obj.mClassId == mCachedClassIdMap[0])
				{
					exportBaseRigidActor(instId, "PxRigidStatic");
				}
				else if(Obj.mClassId == mCachedClassIdMap[1])
				{
					exportBaseRigidActor(instId, "PxRigidDynamic");
				}
				else if(Obj.mClassId == mCachedClassIdMap[2])
				{
					exportArticulation(instId);
				}
				else if(Obj.mClassId == mCachedClassIdMap[3])
				{
					exportCloth(instId);
				}
				else if(Obj.mClassId == mCachedClassIdMap[4])
				{
					exportAggregate(instId);
				}
				else if(Obj.mClassId == mCachedClassIdMap[5])
				{
					exportParticleSystem(instId);
				}
				else if(Obj.mClassId == mCachedClassIdMap[6])
				{
					exportParticleSystem(instId);
				}
			}
		}

		// mBaseWriter.insertXml(shapes.cStr());
		mBaseWriter.insertXml(scenes.cStr());
		mBaseWriter.insertXml(joints.cStr());
	}

  private:
	void PvdPropertySplit(const char* inStr, Array<const char*>& prefixes, char** propName)
	{
		*propName = NULL;

		if(inStr)
		{
			PxU32 inLen = strLen(inStr);
			char* prefix = (char*)mAllocator.allocate(inLen + 1, "char", __FILE__, __LINE__);
			PxU32 outLen = 0;
			for(PxU32 i = 0; i < inLen; i++)
			{
				if(inStr[i] == ']')
					continue;
				else if(inStr[i] == '[' || inStr[i] == '.')
				{
					prefix[outLen] = 0;
					prefixes.pushBack(prefix);
					prefix = (char*)mAllocator.allocate(inLen - i, "char", __FILE__, __LINE__);
					outLen = 0;
				}
				else
					prefix[outLen++] = inStr[i];
			}
			prefix[outLen] = 0;
			*propName = prefix;
		}
	}

	void exportProperty(XmlWriter& writer, InstanceDescription info, Array<const char*>& childStack, const char* pname, const PropertyDescription& prop)
	{
		// Transform from pvd's "." property notation (w.x.y, w.x.z)
		// to RepX's nested notation (<w><x> <y/><z/> </x></w>)
		DataRef<PxU8> propData = mReader.getRawPropertyValue(info, prop.mPropertyId);
		if(propData.size() == 0)
			return;
		const char* key = pname;

		PxU32 stackIdx = 0;
		Array<const char*> prefixes("prefixes");
		char* propName;
		PvdPropertySplit(key, prefixes, &propName);

		if(prefixes.size() > 0)
		{
			for(PxU32 i = 0; i < prefixes.size(); i++)
			{
				const char* item = prefixes[i];

				if(childStack.size() > stackIdx && (::strcmp(childStack[stackIdx], item) != 0))
				{
					while(childStack.size() > stackIdx)
					{
						writer.endTag();
						mAllocator.deallocate((void*)childStack.popBack());
					}
				}
				if(childStack.size() == stackIdx)
				{
					childStack.pushBack(item);
					writer << BeginTag(item);
				}
				else
					mAllocator.deallocate((void*)item);
				++stackIdx;
			}
		}
		if(childStack.size() > stackIdx)
		{
			PxU32 diff = static_cast<PxU32>(childStack.size()) - stackIdx;
			for(PxU32 idx = 0; idx < diff; ++idx)
			{
				writer << EndTag();
				const char* child = childStack.popBack();
				mAllocator.deallocate((void*)child);
			}
		}
		writeGenericProperty(writer, mPropertyStream, mReader, prop, propName, propData.begin(), propData.size());
		mAllocator.deallocate((void*)propName);
	}

	void exportProperty(XmlWriter& writer, InstanceDescription info, Array<const char*>& childStack, const char* pname)
	{
		exportProperty(writer, info, childStack, pname, mMetaData.findProperty(info.mClassId, pname));
	}

	void endExportProperty(XmlWriter& writer, Array<const char*>& childStack)
	{
		while(!childStack.empty())
		{
			writer << EndTag();
			mAllocator.deallocate((void*)childStack.popBack());
		}
	}

	void exportAllProperties(XmlWriter& writer, InstanceDescription info, bool ignoreObjRef = true)
	{
		ClassDescription classType = mMetaData.getClass(info.mClassId);

		Array<PropertyDescription> props("transferAllProperties::props");
		PxU32 numProps = mMetaData.getNbProperties(info.mClassId);
		props.resize(numProps);
		mMetaData.getProperties(info.mClassId, props.begin(), numProps);

		Array<PropertyDescription> sortedProps("sortedProps");

		PVD_FOREACH(idx, numProps)
		{
			const PropertyDescription& propData(props[idx]);
			const char* name = propData.mName;
			bool isObjRef = propData.mDatatype == getPvdTypeForType<ObjectRef>();
			bool isObjRefArray = isObjRef && propData.mPropertyType == PropertyType::Array;
			// Skip properties that we create ourselves.
			if(safeStrEq(name, "Stream Id") || safeStrEq(name, "ClassName") || safeStrEq(name, "ConcreteTypeName") || safeStrEq(name, "Type") || safeStrEq(name, "GeometryType") ||
			   safeStrEq(name, "IsValid") || safeStrEq(propData.mSemantic, "parents") || safeStrEq(name, "CollisionNormals") || safeStrEq(name, "IsExclusive") || safeStrEq(name, "IsSleeping") ||
			   (safeStrEq(name, "ParticleBuffer") && safeStrEq(classType.mName.mName, "PxCloth")) || (isObjRef && ignoreObjRef) || isObjRefArray)
				continue;
			sortedProps.pushBack(propData);
		}

		qsort(sortedProps.begin(), sortedProps.size(), sizeof(PropertyDescription), compareProperty);

		Array<const char*> childStack("childStack");
		for(PxU32 i = 0; i < sortedProps.size(); i++)
		{
			exportProperty(writer, info, childStack, sortedProps[i].mName, sortedProps[i]);
		}
		endExportProperty(writer, childStack);
	}

	void exportMaterial(InstanceDescription material)
	{
		if(material.mAlive == false || mResourceSet.contains(material.mId))
			return;
		mResourceSet.insert(material.mId);
		mBaseWriter << BeginTag("PxMaterial");
		writeProperty(mBaseWriter, mPropertyStream, "Id", material);
		exportAllProperties(mBaseWriter, material);
		mBaseWriter << EndTag();
	}

	void exportClothFabric(InstanceDescription clothFabric)
	{
		if(clothFabric.mAlive == false || mResourceSet.contains(clothFabric.mId))
			return;
		mResourceSet.insert(clothFabric.mId);

		mBaseWriter << BeginTag("PxClothFabric");
		writeProperty(mBaseWriter, mPropertyStream, "Id", clothFabric);
		if(mVersionMajor > 3 || (mVersionMajor == 3 && mVersionMinor >= 3))
		{
			PxU32 nbParticles = mReader.getPropertyValue<PxU32>(clothFabric, "NbParticles");
			writeProperty(mBaseWriter, mPropertyStream, "NbParticles", nbParticles);

			DataRef<PxClothFabricPhase> PhaseData = mReader.getTypedRawPropertyValue<PxClothFabricPhase>(clothFabric, "Phases");

			Option<ClassDescription> phaseTypeDesc = mMetaData.findClass(NamespacedName("physx3", "PxClothFabricPhase"));
			PropertyDescription& prop = mMetaData.findProperty(phaseTypeDesc->mClassId, "PhaseType");

			DataRef<NamedValue> namedValues = mReader.getMetaData().getNamedPropertyValues(prop.mPropertyId);
			writeProperty_ClothFabricPhase(mBaseWriter, mPropertyStream, "Phases", namedValues, PhaseData.begin(), PhaseData.size());

			DataRef<PxReal> RestData = mReader.getTypedRawPropertyValue<PxReal>(clothFabric, "Restvalues");
			writeProperty(mBaseWriter, mPropertyStream, "Restvalues", RestData.begin(), RestData.end(), mBaseWriter.tabCount() + 2);

			DataRef<PxU32> SetsData = mReader.getTypedRawPropertyValue<PxU32>(clothFabric, "Sets");
			writeProperty(mBaseWriter, mPropertyStream, "Sets", SetsData.begin(), SetsData.end(), mBaseWriter.tabCount() + 2);

			DataRef<PxU32> ParticleIndexData = mReader.getTypedRawPropertyValue<PxU32>(clothFabric, "ParticleIndices");
			writeProperty(mBaseWriter, mPropertyStream, "ParticleIndices", ParticleIndexData.begin(), ParticleIndexData.end(), mBaseWriter.tabCount() + 2);

			if(mReader.getMetaData().findProperty(clothFabric.mClassId, "TetherAnchors").hasValue())
			{
				DataRef<PxU32> TetherAnchorData = mReader.getTypedRawPropertyValue<PxU32>(clothFabric, "TetherAnchors");
				writeProperty(mBaseWriter, mPropertyStream, "TetherAnchors", TetherAnchorData.begin(), TetherAnchorData.end(), mBaseWriter.tabCount() + 2);
			}

			if(mReader.getMetaData().findProperty(clothFabric.mClassId, "TetherLengths").hasValue())
			{
				DataRef<PxReal> TetherLengthData = mReader.getTypedRawPropertyValue<PxReal>(clothFabric, "TetherLengths");
				writeProperty(mBaseWriter, mPropertyStream, "TetherLengths", TetherLengthData.begin(), TetherLengthData.end(), mBaseWriter.tabCount() + 2);
			}
		}
		else
		{
			exportAllProperties(mBaseWriter, clothFabric);
		}

		mBaseWriter << EndTag();
	}

	void exportConvexMesh(InstanceDescription mesh)
	{
		if(mesh.mAlive == false || mResourceSet.contains(mesh.mId))
			return;
		mResourceSet.insert(mesh.mId);
		ClassDescription meshClass = mMetaData.getClass(mesh.mClassId);
		mBaseWriter << BeginTag(meshClass.mName.mName);
		writeProperty(mBaseWriter, mPropertyStream, "Id", mesh);

		Array<const char*> childStack("childStack");
		endExportProperty(mBaseWriter, childStack);

		DataRef<PxF32> vertices(mReader.getTypedRawPropertyValue<PxF32>(mesh, "Points"));
		writeProperty(mBaseWriter, mPropertyStream, "Points", vertices.begin(), vertices.end(), mBaseWriter.tabCount() + 2);

		mBaseWriter << EndTag();
	}

	void exportTriangleMesh(InstanceDescription mesh)
	{
		if(mesh.mAlive == false || mResourceSet.contains(mesh.mId))
			return;
		mResourceSet.insert(mesh.mId);
		ClassDescription meshClass = mMetaData.getClass(mesh.mClassId);
		if(mVersionMajor == 3 && mVersionMinor == 3)
		{
			mBaseWriter << BeginTag("PxBVH33TriangleMesh");
		}
		else if(mVersionMajor > 3 || (mVersionMajor == 3 && mVersionMinor >= 4))
		{
			mBaseWriter << BeginTag("PxBVH34TriangleMesh");
		}
		else
		{
			mBaseWriter << BeginTag(meshClass.mName.mName);
		}
		writeProperty(mBaseWriter, mPropertyStream, "Id", mesh);

		Array<const char*> childStack("childStack");
		endExportProperty(mBaseWriter, childStack);

		DataRef<PxF32> vertices(mReader.getTypedRawPropertyValue<PxF32>(mesh, "Points"));
		writeProperty(mBaseWriter, mPropertyStream, "Points", vertices.begin(), vertices.end(), mBaseWriter.tabCount() + 2);
		DataRef<PxU32> triangles(mReader.getTypedRawPropertyValue<PxU32>(mesh, "Triangles"));
		writeProperty(mBaseWriter, mPropertyStream, "Triangles", triangles.begin(), triangles.end(), mBaseWriter.tabCount() + 2);
		DataRef<PxU16> materialData = mReader.getTypedRawPropertyValue<PxU16>(mesh, "MaterialIndices");
		writeProperty(mBaseWriter, mPropertyStream, "MaterialIndices", materialData.begin(), materialData.end(), mBaseWriter.tabCount() + 2);

		mBaseWriter << EndTag();
	}

	void exportHeightField(InstanceDescription info)
	{
		if(info.mAlive == false || mResourceSet.contains(info.mId))
			return;
		mResourceSet.insert(info.mId);
		mBaseWriter << BeginTag("PxHeightField");
		writeProperty(mBaseWriter, mPropertyStream, "Id", info);
		exportAllProperties(mBaseWriter, info);
		DataRef<PxU32> samples = mReader.getTypedRawPropertyValue<PxU32>(info, "Samples");
		writeProperty(mBaseWriter, mPropertyStream, "Samples", samples.begin(), samples.end(), mBaseWriter.tabCount() + 2);
		mBaseWriter << EndTag();
	}

	void exportGeometry(InstanceDescription geometry)
	{
		if(geometry.mAlive == false)
			return;
		mSceneWriter << BeginTag("Geometry");
		ClassDescription desc = mMetaData.getClass(geometry.mClassId);
		mSceneWriter << BeginTag(desc.mName.mName);
		exportAllProperties(mSceneWriter, geometry);
		if(safeStrEq(desc.mName.mName, "PxConvexMeshGeometry"))
		{
			PxI32 bufferId = mReader.getPropertyValue<PxI32>(geometry, "ConvexMesh");
			writeProperty(mSceneWriter, mPropertyStream, "ConvexMesh", bufferId);
			exportConvexMesh(mReader.idToDescriptor(bufferId));
		}
		else if(safeStrEq(desc.mName.mName, "PxTriangleMeshGeometry"))
		{
			PxI32 bufferId = mReader.getPropertyValue<PxI32>(geometry, "TriangleMesh");
			writeProperty(mSceneWriter, mPropertyStream, "TriangleMesh", bufferId);
			exportTriangleMesh(mReader.idToDescriptor(bufferId));
		}
		else if(safeStrEq(desc.mName.mName, "PxHeightFieldGeometry"))
		{
			PxI32 bufferId = mReader.getPropertyValue<PxI32>(geometry, "HeightField");
			writeProperty(mSceneWriter, mPropertyStream, "HeightField", bufferId);
			exportHeightField(mReader.idToDescriptor(bufferId));
		}
		mSceneWriter << EndTag();
		mSceneWriter << EndTag();
	}

#define GET_CHILDREN(arrayName, owner, propName)                                                                                                                                                       \
	PxU32 num##arrayName = mReader.getNbArrayElements(owner, propName);                                                                                                                                \
	Array<PxI32> arrayName("GET_CHILDREN::" #arrayName);                                                                                                                                               \
	if(num##arrayName)                                                                                                                                                                                 \
	{                                                                                                                                                                                                  \
		arrayName.resize(num##arrayName);                                                                                                                                                              \
		mReader.getPropertyValue(owner, propName, arrayName.begin(), num##arrayName);                                                                                                                  \
	}

	void exportScene(uint32_t sceneInstId)
	{
		InstanceDescription scene(mReader.idToDescriptor(sceneInstId));
		if(scene.mAlive == false)
			return;

		{
			GET_CHILDREN(items, scene, "RigidStatics");
			PVD_FOREACH(idx, numitems)
			exportBaseRigidActor(items[idx], "PxRigidStatic");
		}

		{
			GET_CHILDREN(items, scene, "RigidDynamics");
			PVD_FOREACH(idx, numitems)
			exportBaseRigidActor(items[idx], "PxRigidDynamic");
		}

		{
			GET_CHILDREN(items, scene, "Articulations");
			PVD_FOREACH(idx, numitems)
			exportArticulation(items[idx]);
		}

		{
			GET_CHILDREN(items, scene, "Cloths");
			PVD_FOREACH(idx, numitems)
			exportCloth(items[idx]);
		}

		{
			GET_CHILDREN(items, scene, "Aggregates");
			PVD_FOREACH(idx, numitems)
			exportAggregate(items[idx]);
		}

		{
			GET_CHILDREN(items, scene, "ParticleSystems");
			PVD_FOREACH(idx, numitems)
			exportParticleSystem(items[idx]);
		}

		{
			GET_CHILDREN(items, scene, "ParticleFluids");
			PVD_FOREACH(idx, numitems)
			exportFluid(items[idx]);
		}
	}

	bool isSharedShape(InstanceDescription& shape) { return !mReader.getPropertyValue<bool>(shape, "IsExclusive"); }

	void exportShape(InstanceDescription& shape)
	{
		if(mExportedShapes.contains(shape.mId))
			return;

		mExportedShapes.insert(shape.mId);

		mSceneWriter << BeginTag("PxShape");
		writeProperty(mSceneWriter, mPropertyStream, "Id", shape);
		exportAllProperties(mSceneWriter, shape);
		GET_CHILDREN(materials, shape, "Materials");
		if(nummaterials)
		{
			mSceneWriter << BeginTag("Materials");
			PVD_FOREACH(matIdx, nummaterials)
			{
				exportMaterial(mReader.idToDescriptor(materials[matIdx]));
				writeProperty(mSceneWriter, mPropertyStream, "PxMaterialRef", materials[matIdx]);
			}
			mSceneWriter << EndTag();
		}
		PxI32 geomId = mReader.getPropertyValue<PxI32>(shape, "Geometry");
		exportGeometry(mReader.idToDescriptor(geomId));
		mSceneWriter << EndTag();
	}

	void exportShapes(InstanceDescription shapeOwner)
	{
		GET_CHILDREN(items, shapeOwner, "Shapes");
		mSceneWriter << BeginTag("Shapes");
		PVD_FOREACH(idx, numitems)
		{
			InstanceDescription info(mReader.idToDescriptor(items[idx]));
			if(info.mAlive == false)
				continue;

			bool isSupportSharedShape = mVersionMajor > 3 || (mVersionMajor == 3 && mVersionMinor >= 3);
			if(isSupportSharedShape && isSharedShape(info))
			{
				writeProperty(mSceneWriter, mPropertyStream, "PxShapeRef", items[idx]);
			}
			else
			{
				exportShape(info);
			}
		}
		mSceneWriter << EndTag();
	}

	void transferProperty(XmlWriter& writer, InstanceDescription info, const char* xmlName, const char* dbName)
	{
		Array<PropertyDescription> props("transferAllProperties::props");
		PxU32 numProps = mMetaData.getNbProperties(info.mClassId);
		props.resize(numProps);
		mMetaData.getProperties(info.mClassId, props.begin(), numProps);

		PVD_FOREACH(idx, numProps)
		{
			const PropertyDescription& propData(props[idx]);
			DataRef<PxU8> propData1 = mReader.getRawPropertyValue(info, propData.mPropertyId);
			const char* name = propData.mName;
			// Skip properties that we create ourselves.
			if(safeStrEq(name, dbName))
			{
				writeGenericProperty(writer, mPropertyStream, mReader, propData, xmlName, propData1.begin(), propData1.size());
			}
		}
	}

	void exportCloth(uint32_t instId)
	{
		InstanceDescription info(mReader.idToDescriptor(instId));
		if(info.mAlive == false)
			return;

		mSceneWriter << BeginTag("PxCloth");
		writeProperty(mSceneWriter, mPropertyStream, "Id", info);
		transferProperty(mSceneWriter, info, "Particles", "ParticleBuffer");
		exportAllProperties(mSceneWriter, info);

		PxI32 fabId = mReader.getPropertyValue<PxI32>(info, "Fabric");
		writeProperty(mSceneWriter, mPropertyStream, "Fabric", fabId);
		exportClothFabric(mReader.idToDescriptor(fabId));
		mSceneWriter << EndTag();
	}

	void exportJoints(InstanceDescription actor)
	{
		GET_CHILDREN(items, actor, "Joints");
		PVD_FOREACH(idx, numitems)
		{
			if(mExportedJoints.contains((PxI32)items[idx]))
				continue;

			InstanceDescription joint = mReader.idToDescriptor(items[idx]);
			if(joint.mAlive == false)
				continue;
			ClassDescription jointClass = mMetaData.getClass(joint.mClassId);
			mJointWriter << BeginTag(jointClass.mName.mName);
			writeProperty(mJointWriter, mPropertyStream, "Id", joint);
			exportAllProperties(mJointWriter, joint, false);
			mJointWriter << EndTag();
			mExportedJoints.insert((PxI32)items[idx]);
		}
	}

	void exportParticleSystem(uint32_t instId)
	{
		InstanceDescription info(mReader.idToDescriptor(instId));
		if(info.mAlive == false)
			return;

		mSceneWriter << BeginTag("PxParticleSystem");
		writeProperty(mSceneWriter, mPropertyStream, "Id", info);
		exportAllProperties(mSceneWriter, info);

		mSceneWriter << EndTag();
	}

	void exportFluid(uint32_t instId)
	{
		InstanceDescription info(mReader.idToDescriptor(instId));
		if(info.mAlive == false)
			return;

		mSceneWriter << BeginTag("PxParticleFluid");
		writeProperty(mSceneWriter, mPropertyStream, "Id", info);
		exportAllProperties(mSceneWriter, info);

		mSceneWriter << EndTag();
	}

	void exportBaseRigidActor(uint32_t instId, const char* typeName)
	{
		InstanceDescription rigidActor(mReader.idToDescriptor(instId));
		if(rigidActor.mAlive == false)
			return;
		mSceneWriter << BeginTag(typeName);
		writeProperty(mSceneWriter, mPropertyStream, "Id", rigidActor);
		exportAllProperties(mSceneWriter, rigidActor);
		exportShapes(rigidActor);
		exportJoints(rigidActor);
		mSceneWriter << EndTag();
	}

	void exportArticulationLinks(InstanceDescription parent, bool writeParent)
	{
		GET_CHILDREN(items, parent, "Links");
		PVD_FOREACH(idx, numitems)
		{
			InstanceDescription link(mReader.idToDescriptor(items[idx]));
			if(link.mAlive == false)
				continue;
			mSceneWriter << BeginTag("PxArticulationLink");
			writeProperty(mSceneWriter, mPropertyStream, "Id", link);

			PxI32 parent = mReader.getPropertyValue<PxI32>(link, "Parent");
			InstanceDescription parentDesc(mReader.idToDescriptor(parent));
			if(parentDesc.mAlive && writeParent)
				writeProperty(mSceneWriter, mPropertyStream, "Parent", parent);

			PxI32 joint = mReader.getPropertyValue<PxI32>(link, "InboundJoint");
			InstanceDescription jointDesc(mReader.idToDescriptor(joint));
			if(jointDesc.mAlive)
			{
				mSceneWriter << BeginTag("Joint");
				exportAllProperties(mSceneWriter, jointDesc);
				mSceneWriter << EndTag();
			}

			exportAllProperties(mSceneWriter, link);
			exportShapes(link);
			exportJoints(link);
			mSceneWriter << EndTag();
		}
		PVD_FOREACH(idx, numitems)
		{
			InstanceDescription link(mReader.idToDescriptor(items[idx]));
			if(link.mAlive == false)
				continue;
			exportArticulationLinks(link, true);
		}
	}

	// Return number of articultionlinks
	PxU32 getArticulationLinkCount(InstanceDescription parent)
	{
		GET_CHILDREN(items, parent, "Links");
		PxU32 linkCount = numitems;
		PVD_FOREACH(idx, numitems)
		{
			InstanceDescription link(mReader.idToDescriptor(items[idx]));
			if(link.mAlive == false)
				continue;
			linkCount += getArticulationLinkCount(link);
		}

		return linkCount;
	}

	void exportAggregateActors(InstanceDescription parent)
	{
		mSceneWriter << BeginTag("Actors");
		GET_CHILDREN(items, parent, "Actors");
		PxU32 numActors = numitems;
		PVD_FOREACH(idx, numitems)
		{
			InstanceDescription actor(mReader.idToDescriptor(items[idx]));
			if(actor.mAlive)
			{
				writeProperty(mSceneWriter, mPropertyStream, "PxActorRef", actor);
			}
		}
		GET_CHILDREN(articulations, parent, "Articulations");
		// PVD_FOREACH(idx, numitems)
		PVD_FOREACH(idx, numarticulations)
		{
			InstanceDescription art(mReader.idToDescriptor(articulations[idx]));
			if(art.mAlive)
			{
				writeProperty(mSceneWriter, mPropertyStream, "PxArticulationRef", art);
			}
			numActors += getArticulationLinkCount(art);
		}
		mSceneWriter << EndTag();

		writeProperty(mSceneWriter, mPropertyStream, "numActors", numActors);
	}

	void exportAggregate(uint32_t instId)
	{
		InstanceDescription aggregate(mReader.idToDescriptor(instId));
		mSceneWriter << BeginTag("PxAggregate");
		writeProperty(mSceneWriter, mPropertyStream, "Id", aggregate);

		exportAggregateActors(aggregate);
		exportAllProperties(mSceneWriter, aggregate);
		mSceneWriter << EndTag();
	}

	void exportArticulation(uint32_t instId)
	{
		InstanceDescription articulation(mReader.idToDescriptor(instId));
		if(articulation.mAlive == false)
			return;
		mSceneWriter << BeginTag("PxArticulation");
		writeProperty(mSceneWriter, mPropertyStream, "Id", articulation);
		exportAllProperties(mSceneWriter, articulation);
		mSceneWriter << BeginTag("Links");
		exportArticulationLinks(articulation, false);
		mSceneWriter << EndTag();
		mSceneWriter << EndTag();
	}
};
} // namespace

bool PvdRepxExporter::canExport(PvdClient& client)
{
	// Ensure we can find at least one physx scene.
	NamespacedName sceneTypeName("physx3", "PxScene");
	PvdObjectModelReader& reader(client.getReader());
	const PvdObjectModelMetaData& meta(reader.getMetaData());
	// Exporter caches each of these class IDs in its constructor. A scene
	// name alone is insufficient for custom/partial PVD schemas.
	const char* requiredClasses[] = {"PxPhysics", "PxScene", "PxRigidStatic", "PxRigidDynamic",
		"PxArticulation", "PxCloth", "PxAggregate", "PxParticleSystem", "PxParticleFluid"};
	for(const char* name : requiredClasses)
		if(!meta.findClass(NamespacedName("physx3", name)).hasValue())
			return false;
	Option<ClassDescription> sceneType(meta.findClass(sceneTypeName));
	if(sceneType.hasValue() == false)
		return false;
	InstanceDescription descs[64];
	PxU32 numObjs = reader.getNbInstances();
	PxU32 idx = 0;
	while(idx < numObjs)
	{
		PxU32 numAcquired = reader.getInstances(descs, 64, idx);
		idx += numAcquired;
		PVD_FOREACH(descIdx, numAcquired)
		{
			InstanceDescription inst(descs[descIdx]);
			if(inst.mClassId == sceneType->mClassId && inst.mAlive)
				return true;
		}
	}
	return false;
}

void PvdRepxExporter::exportToRepX(physx::shdfnd::Array<uint64_t>& objects, PvdClient& client, PvdRandomAccessIOStream& stream, PxAllocatorCallback& alloc)
{
	Exporter theSerializer(alloc, stream, client.getReader());

	theSerializer.serialize(objects);
}
// void PvdRepxExporter::exportToRepX( physx::shdfnd::Array<uint64_t>& objects, PvdClient& client,
// pvdapp::PvdRandomAccessIOStream& stream, PxAllocatorCallback& alloc )
//{
//    Exporter theSerializer( alloc, stream, client.getReader() );

//	theSerializer.serialize(objects);
//}
// void PvdRepxExporter::exportToRepX( physx::shdfnd::Array<uint64_t>& objects, PvdClient& client, const char* fname,
// PxAllocatorCallback& alloc )
//{
//    PvdFileOutputStream output( fname );
//    Exporter theSerializer( alloc, output, client.getReader() );

//    theSerializer.serialize(objects);
//}
