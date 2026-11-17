/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifdef PVD_OVD_SUPPORT

#include "OvdDomAdapter.h"
#include "OvdLog.h"
#include "PvdDomQuery.h"  // ::isObjectAliveAtFrame canonical helper

#include <algorithm>
#include <set>
#include <cctype>
#include <cstring>

namespace pvdapp
{

// ---------------------------------------------------------------------------
// Internal helpers (file-local)
// ---------------------------------------------------------------------------

namespace
{

using OvdObjectSet = std::unordered_set<const OmniPvdObject*>;

OvdObjectSet collectDomObjects(const OmniPvdDOMState& dom)
{
    OvdObjectSet objects;
    objects.reserve(dom.mObjectCreations.size() + dom.mObjectHandleToObjectMap.size() + 2);
    if (dom.mSceneRoot) objects.insert(dom.mSceneRoot);
    if (dom.mSharedRoot) objects.insert(dom.mSharedRoot);
    for (const OmniPvdObject* object : dom.mObjectCreations)
        if (object) objects.insert(object);
    for (const OmniPvdObject* object : dom.mSceneCreations)
        if (object) objects.insert(object);
    for (const auto& entry : dom.mObjectHandleToObjectMap)
        if (entry.second) objects.insert(entry.second);
    return objects;
}

// Map OmniPvdDataType + nbrFields to PVD3 NamespacedName.
// Mirrors the logic in OvdFileLoader.cpp's omniDataTypeToPvd3().
NamespacedName domTypeToPvd3Name(uint16_t dataType, uint32_t nbrFields)
{
    switch (static_cast<OmniPvdDataType::Enum>(dataType))
    {
    case OmniPvdDataType::eINT8:   return NamespacedName("physx3", "PvdI8");
    case OmniPvdDataType::eINT16:  return NamespacedName("physx3", "PvdI16");
    case OmniPvdDataType::eINT32:  return NamespacedName("physx3", "PvdI32");
    case OmniPvdDataType::eINT64:  return NamespacedName("physx3", "PvdI64");
    case OmniPvdDataType::eUINT8:  return NamespacedName("physx3", "PvdU8");
    case OmniPvdDataType::eUINT16: return NamespacedName("physx3", "PvdU16");
    case OmniPvdDataType::eUINT32: return NamespacedName("physx3", "PvdU32");
    case OmniPvdDataType::eUINT64: return NamespacedName("physx3", "PvdU64");
    case OmniPvdDataType::eFLOAT32:
        if (nbrFields == 2) return NamespacedName("physx3", "PxVec2");
        if (nbrFields == 3) return NamespacedName("physx3", "PxVec3");
        if (nbrFields == 4) return NamespacedName("physx3", "PxVec4");
        if (nbrFields == 6) return NamespacedName("physx3", "PxBounds3");
        if (nbrFields == 7) return NamespacedName("physx3", "PxTransform");
        if (nbrFields == 9) return NamespacedName("physx3", "PxMat3");
        return NamespacedName("physx3", "PvdF32");
    case OmniPvdDataType::eFLOAT64:  return NamespacedName("physx3", "PvdF64");
    case OmniPvdDataType::eSTRING:   return NamespacedName("physx3", "StringHandle");
    case OmniPvdDataType::eOBJECT_HANDLE: return NamespacedName("physx3", "ObjectRef");
    case OmniPvdDataType::eENUM_VALUE:    return NamespacedName("physx3", "PvdU32");
    case OmniPvdDataType::eFLAGS_WORD:   return NamespacedName("physx3", "PvdU32");
    default:                              return NamespacedName("physx3", "PvdU8");
    }
}

} // anonymous namespace

bool validateOvdDomTopology(const OmniPvdDOMState& dom)
{
    const OvdObjectSet objects = collectDomObjects(dom);
    std::unordered_map<const OmniPvdObject*, uint8_t> visitState;
    visitState.reserve(objects.size());
    std::vector<const OmniPvdObject*> path;
    path.reserve(std::min<size_t>(objects.size(), 64));
    for (const OmniPvdObject* object : objects)
    {
        if (visitState[object] == 2)
            continue;

        path.clear();
        const OmniPvdObject* current = object;
        while (current)
        {
            // Never dereference a parent that is not owned by this DOM. The
            // size-derived hop bound is defense-in-depth for corrupted chains;
            // a valid acyclic chain cannot visit more objects than the set.
            if (objects.find(current) == objects.end() || path.size() >= objects.size())
                return false;

            const uint8_t state = visitState[current];
            if (state == 1)
                return false;
            if (state == 2)
                break;

            visitState[current] = 1;
            path.push_back(current);
            current = current->mAncestor;
        }
        for (const OmniPvdObject* visited : path)
            visitState[visited] = 2;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

OvdDomAdapter::OvdDomAdapter(std::unique_ptr<OmniPvdDOMState> dom,
                             std::vector<OmniPvdMessage>      msgs,
                             OvdEvidenceFrameScopes scopes)
    : mDom(std::move(dom))
    , mMessages(std::move(msgs))
    , mEvidenceScopes(std::move(scopes))
    , mCurrentFrame(mDom->mMaxFrame)
    , mInputError(OvdInputErrorCode::None)
    , mNextStringHandle(1)
{
    buildIndex();
}

OvdEvidenceDomain OvdDomAdapter::evidenceDomain() const {
    if(mDom->mRecordingSegmentMaxFrames.size()>OvdEvidencePageBytes/sizeof(uint64_t))
        throw std::length_error("Evidence domain exceeds 8 MiB");
    OvdEvidenceDomain result;result.supported=true;
    result.sawSegmentMetadata=mDom->mSawRecordingSegmentMetadata;
    result.frameScopesComplete=mEvidenceScopes.complete;
    result.minFrame=mDom->mMinFrame;result.maxFrame=mDom->mMaxFrame;
    result.segmentMaxima=mDom->mRecordingSegmentMaxFrames;result.scopes=mEvidenceScopes.scopes;
    return result;
}

OvdEvidenceIdentity OvdDomAdapter::evidenceIdentity(PxI32 instanceId) const {
    OvdEvidenceIdentity result;
    if(instanceId<=0 || size_t(instanceId)>mObjects.size())return result;
    const auto& entry=mObjects[size_t(instanceId-1)];const auto* object=entry.obj;
    const auto& name=mClasses[size_t(entry.classId)].name;
    if(name.size()>OvdEvidencePageBytes/2 || object->mLifeSpans.size()>OvdEvidencePageBytes/(2*sizeof(OvdEvidenceLifeSpan)))
        throw std::length_error("Evidence identity exceeds 8 MiB");
    result.supported=true;result.instanceId=instanceId;result.objectUid=object->mUID;
    result.apiHandle=object->mOmniAPIHandle;result.recordingSegment=object->mRecordingSegmentId;result.className=name;
    result.sceneInstanceId=name=="PxScene"?instanceId:getOwningSceneId(instanceId);
    if(result.sceneInstanceId>0 && size_t(result.sceneInstanceId)<=mObjects.size()) {
        const auto* scene=mObjects[size_t(result.sceneInstanceId-1)].obj;
        result.sceneContext=scene->mOmniAPIHandle;result.sceneObjectUid=scene->mUID;
    }
    for(const auto& life:object->mLifeSpans)result.lifespans.push_back({life.mFrameStart,life.mFrameStop});
    return result;
}

OvdEvidenceSampleStamp OvdDomAdapter::evidenceSampleStamp(void* instance,PxI32 property,uint64_t frame) const {
    OvdEvidenceSampleStamp result;
    const auto index=resolvePublishedInstanceKey(reinterpret_cast<PxU64>(instance));
    if(index<0 || size_t(index)>=mObjects.size())return result;
    const auto* object=mObjects[size_t(index)].obj;
    if(!isAliveAtFrame(object,frame))return result;
    const auto classIndex=property>>16,attributeIndex=property&0xffff;
    if(classIndex<0 || size_t(classIndex)>=mClasses.size() || !object->mOmniPvdClass)return result;
    const auto* cls=mClasses[size_t(classIndex)].cls;
    const auto& chain=object->mOmniPvdClass->mInheritanceChain;
    if(!cls || size_t(attributeIndex)>=cls->mAttributeDefinitions.size() ||
        std::find(chain.begin(),chain.end(),cls)==chain.end())return result;
    const auto* definition=cls->mAttributeDefinitions[size_t(attributeIndex)];
    if(!definition || definition->mIsUniqueList)return result;
    result.supported=true;
    auto* list=getAttrList(mObjects[size_t(index)].obj,property);
    if(!list)return result;
    if(const auto* sample=findSampleAtFrame(list,frame)){result.hasSample=true;result.frame=sample->mTimeStamp;}
    return result;
}

// ---------------------------------------------------------------------------
// buildIndex() — build mClasses, mObjects and all handle maps from the DOM
// ---------------------------------------------------------------------------

bool OvdDomAdapter::buildIndex()
{
    if (mInputError != OvdInputErrorCode::None || !validateOvdDomTopology(*mDom))
    {
        mInputError = OvdInputErrorCode::InvalidDomTopology;
        return false;
    }

    // -----------------------------------------------------------------------
    // Pass 1: Build class entries from dom.mClassHandleToClassMap
    // -----------------------------------------------------------------------
    for (auto& kv : mDom->mClassHandleToClassMap)
    {
        OmniPvdClass* cls = kv.second;
        if (cls->mIsEnumClass)
            continue;

        // Incremental (live) re-index: skip classes already indexed so existing
        // class ids stay stable (append-only). At construction the map is empty
        // so this never triggers.
        if (mClassHandleToId.find(cls->mOmniClassHandle) != mClassHandleToId.end())
            continue;

        ClassEntry entry;
        entry.cls  = cls;
        entry.id   = static_cast<PxI32>(mClasses.size());
        entry.name = cls->mClassName;
        entry.nsName = NamespacedName("physx3", cls->mClassName.c_str());

        // Build one PropertyDescription per attribute in this class's own list.
        // propId = (classEntry.id << 16) | attrIndex
        for (PxU32 ai = 0; ai < static_cast<PxU32>(cls->mAttributeDefinitions.size()); ++ai)
        {
            OmniPvdAttributeDef* attrDef = cls->mAttributeDefinitions[ai];

            PxI32 propId = (entry.id << 16) | static_cast<PxI32>(ai);

            NamespacedName datatypeName = domTypeToPvd3Name(attrDef->mDataType, attrDef->mNbrFields);

            PropertyType::Enum propType = attrDef->mIsUniqueList
                ? PropertyType::Array
                : PropertyType::Scalar;

            // Semantic determines how the tree builder handles ObjectRef properties:
            // "" or "children" → expanded as child nodes in the tree
            // anything else → shown as plain property values
            // Only unique-list ObjectRef properties are actual children (shapes, scenes, etc.)
            const char* semantic = "";
            switch (attrDef->mAttributeId)
            {
            case ePvdDomAttribTranslateOp: semantic = "translate"; break;
            case ePvdDomAttribScaleOp:     semantic = "scale";     break;
            case ePvdDomAttribRotationOp:  semantic = "rotate";    break;
            default:
                if (attrDef->mDataType == OmniPvdDataType::eOBJECT_HANDLE && !attrDef->mIsUniqueList)
                {
                    // "Scene" and "Physics" scalar ObjectRef → "parents" semantic
                    // so that getParentId() can walk up the tree for pick-to-select
                    const char* aName = attrDef->mAttributeName.c_str();
                    if (strcmp(aName, "Scene") == 0 || strcmp(aName, "Physics") == 0)
                        semantic = "parents";
                    else
                        semantic = "reference";
                }
                else if (attrDef->mDataType == OmniPvdDataType::eFLAGS_WORD)
                {
                    // Provisional — may be downgraded to "Enumeration Value"
                    // below once the named-value table is built and we can
                    // tell whether this is a real bitflag or a sequential
                    // enum stuffed through OMNI_PVD_ATTRIBUTE_FLAG (which is
                    // PhysX's only flag-attribute macro — both enums like
                    // PxActorType and real flags like PxActorFlags share it).
                    semantic = "Bitflag";
                }
                else if (attrDef->mDataType == OmniPvdDataType::eENUM_VALUE
                         && !cls->mIsEnumClass)
                {
                    // Single-enum-value attribute on a regular class (e.g.
                    // PxActor.type pointing into PxActorType). The named-value
                    // table is built from mDerivedFromClass below.
                    semantic = "Enumeration Value";
                }
                break;
            }

            // Pre-build named-value table for enum/bitflag attributes so the
            // property tree can render named bits (PxShapeFlag::eSCENE_QUERY_SHAPE
            // etc.) instead of raw ints. The OVD DOM stores enum bit names in a
            // *separate* OmniPvdClass linked via mDerivedFromClass; each of its
            // mAttributeDefinitions has the bit's name (mAttributeName) and the
            // bit value (stored in mNbrFields per the parser convention).
            if ((attrDef->mDataType == OmniPvdDataType::eFLAGS_WORD
                 || attrDef->mDataType == OmniPvdDataType::eENUM_VALUE)
                && attrDef->mDerivedFromClass != nullptr
                && attrDef->mDerivedFromClass->mIsEnumClass)
            {
                std::vector<NamedValue> nv;
                for (OmniPvdAttributeDef* bitDef : attrDef->mDerivedFromClass->mAttributeDefinitions)
                {
                    if (!bitDef) continue;
                    mNamedValueStringPool.push_back(bitDef->mAttributeName);
                    NamedValue n;
                    n.mName  = mNamedValueStringPool.back().c_str();
                    n.mValue = static_cast<PxU32>(bitDef->mNbrFields);
                    nv.push_back(n);
                }

                // Disambiguate enum-stuffed-as-flags: PhysX's
                // OMNI_PVD_ATTRIBUTE_FLAG macro registers PxActorType
                // (sequential 0,1,2,…) the same way as PxActorFlags
                // (powers of two). At the schema level both are eFLAGS_WORD;
                // only the named-value table can tell them apart. A real
                // bitflag has every named value as a non-zero single bit
                // (popcount == 1). Anything else — a 0 entry, or a value
                // with popcount > 1 — means we're looking at an enum, so
                // re-tag the semantic so the panel renders the value name
                // (eRIGID_STATIC, eDEFORMABLE_VOLUME, …) instead of empty.
                if (semantic && std::strcmp(semantic, "Bitflag") == 0 && !nv.empty())
                {
                    bool looksLikeRealBitflag = true;
                    for (const NamedValue& n : nv)
                    {
                        const PxU32 v = n.mValue;
                        // popcount via Hamming-weight trick
                        PxU32 t = v;
                        t = t - ((t >> 1) & 0x55555555);
                        t = (t & 0x33333333) + ((t >> 2) & 0x33333333);
                        const PxU32 popcount = (((t + (t >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
                        if (v == 0 || popcount != 1)
                        {
                            looksLikeRealBitflag = false;
                            break;
                        }
                    }
                    if (!looksLikeRealBitflag)
                        semantic = "Enumeration Value";
                }

                if (!nv.empty())
                    mNamedValuesByPropId[propId] = std::move(nv);
            }

            PropertyDescription pd;
            pd.mOwnerClassName = entry.nsName;
            pd.mOwnerClassId   = entry.id;
            pd.mName           = attrDef->mAttributeName.c_str(); // DOM owns the string
            pd.mSemantic       = semantic;
            pd.mDatatype       = -1;  // not used by adapter callers
            pd.mDatatypeName   = datatypeName;
            pd.mPropertyType   = propType;
            pd.mPropertyId     = propId;
            pd.m32BitOffset    = 0;
            pd.m64BitOffset    = 0;

            entry.props.push_back(pd);
        }

        mClassHandleToId[cls->mOmniClassHandle] = entry.id;
        mClassNameToId[cls->mClassName]          = entry.id;

        mClasses.push_back(std::move(entry));
    }

    // Assign each class its registration GENERATION: walking classes in handle
    // order (the writer hands out class handles monotonically), a class NAME
    // seen again means the schema was re-registered -- a new recording segment
    // started. Objects carry their class's generation as their segment identity.
    // Recomputed wholesale each (re)build: cheap, deterministic, append-safe.
    {
        std::vector<std::pair<OmniPvdClassHandle, PxI32>> byHandle;
        byHandle.reserve(mClasses.size());
        for (const ClassEntry& c : mClasses)
            byHandle.push_back({ c.cls->mOmniClassHandle, c.id });
        std::sort(byHandle.begin(), byHandle.end());

        mClassGeneration.assign(mClasses.size(), 0);
        std::set<std::string> seenNames;
        PxI32 generation = 0;
        for (const auto& hc : byHandle)
        {
            const std::string& name = mClasses[hc.second].name;
            if (!seenNames.insert(name).second)
            {
                ++generation;
                seenNames.clear();
                seenNames.insert(name);
            }
            mClassGeneration[hc.second] = generation;
        }
    }

    ovdLog("[OvdDomAdapter] buildIndex: %zu classes indexed\n", mClasses.size());

    // -----------------------------------------------------------------------
    // Pass 2: Build object entries from dom.mObjectHandleToObjectMap
    // -----------------------------------------------------------------------
    for (auto& kv : mDom->mObjectHandleToObjectMap)
    {
        OmniPvdObject* obj = kv.second;

        // Incremental (live) re-index: skip objects already indexed so existing
        // object ids stay stable (append-only). At construction the map is empty
        // so this never triggers.
        if (mObjHandleToId.find(kv.first) != mObjHandleToId.end())
            continue;

        // Resolve class id (may be -1 for objects with enum classes or unknown classes)
        PxI32 classId = -1;
        if (obj->mOmniPvdClass)
        {
            auto it = mClassHandleToId.find(obj->mOmniPvdClass->mOmniClassHandle);
            if (it != mClassHandleToId.end())
                classId = it->second;
        }

        ObjectEntry entry;
        entry.obj         = obj;
        entry.id          = static_cast<PxI32>(mObjects.size());
        entry.classId     = classId;
        entry.internalKey = static_cast<PxU64>(kv.first);
        entry.isTopLevel  = (obj->mAncestor == nullptr ||
                             obj->mAncestor == mDom->mSceneRoot ||
                             obj->mAncestor == mDom->mSharedRoot);

        // pvddom currently tags some sub-component objects as top-level
        // (mAncestor = sceneRoot/sharedRoot), but semantically they belong
        // under their owning parent. Force these off the top-level list so
        // the scene tree shows them nested under the right canonical home —
        // for the schema-referenced ones, they'll appear through the
        // ObjectRef property paths during loadChildren as IT_COMMON; the
        // schema-orphaned ones (PxOmniPvdMetaData) get a manual placement
        // under PxPhysics in InstanceView::loadChildren.
        //   PxGpuDynamicsMemoryConfig — child of PxScene (via gpuDynamicsConfig)
        //   PxD6JointDrive            — child of PxD6Joint (via drives[*])
        //   PxAggregate               — child of PxPhysics (via aggregates)
        //   PxOmniPvdMetaData         — no schema parent; bolted under PxPhysics
        if (entry.isTopLevel && obj->mOmniPvdClass)
        {
            const std::string& clsName = obj->mOmniPvdClass->mClassName;
            if (clsName == "PxGpuDynamicsMemoryConfig" ||
                clsName == "PxD6JointDrive"            ||
                clsName == "PxAggregate"               ||
                clsName == "PxOmniPvdMetaData"         ||
                clsName == "PxArticulationJointReducedCoordinate")
            {
                // PxArticulationJointReducedCoordinate is reachable only via
                // PxArticulationLink.inboundJoint (single ObjectRef); the link
                // tree node bolts it on as a child in InstanceView::loadChildren
                // so it shows up where the user expects it.
                entry.isTopLevel = false;
            }
        }

        mObjHandleToId[kv.first] = entry.id;

        // Map internal key → object index (used as instPtr in InstanceDescription)
        mInternalKeyToId[entry.internalKey] = entry.id;

        // Collect external API handle → object index candidates (for
        // eOBJECT_HANDLE resolution). One PhysX pointer maps to several DOM
        // objects when the writer re-registers it; resolveExternalReferenceAt()
        // picks per frame and referrer instead of collapsing to the last one.
        if (obj->mOmniAPIHandle != 0)
            mApiHandleCandidates[obj->mOmniAPIHandle].push_back(entry.id);

        mObjects.push_back(entry);
    }

    // Keep each handle's candidates in creation order (DOM mUID), so "nearest
    // the referrer" and "newest" picks in resolveExternalReferenceAt() are
    // deterministic regardless of the unordered_map iteration order above.
    for (auto& kv : mApiHandleCandidates)
    {
        std::sort(kv.second.begin(), kv.second.end(), [this](PxI32 a, PxI32 b)
                  { return mObjects[a].obj->mUID < mObjects[b].obj->mUID; });
    }

    ovdLog("[OvdDomAdapter] buildIndex: %zu objects indexed\n", mObjects.size());

    // Pass 3: per-object owning-PxScene, resolved from the DOM tree (mAncestor),
    // NOT from PxScene.actors external-handle resolution. The actors path maps
    // through external PhysX handles, which are many-to-one when a pointer is
    // reused (scene recreated mid-capture / actor moved scene-to-scene), so it
    // collapses re-registered objects onto the newest one. The appendChild tree
    // built at parse time keeps each distinct object under its correct scene
    // object (verified: VolumeVolumeCollision's two test iterations land under
    // distinct scene synthetic ids), which is exactly what the canonical
    // OmniPvdUsdWriter.cpp uses to place per-scene prims. mObjAncestorSceneId
    // lets the renderer gate per-scene Vis correctly for these captures.
    {
        struct OwnershipSummary
        {
            PxI32 sceneId = -1;
            PxI32 physicsId = -1;
            bool hasPhysics = false;
        };

        std::unordered_map<const OmniPvdObject*, PxI32> objToId;
        objToId.reserve(mObjects.size());
        for (const ObjectEntry& e : mObjects) objToId[e.obj] = e.id;

        // Cache the ownership summary for every ancestry node as it is first
        // encountered. Even a maximally deep valid chain is therefore walked
        // only once across the whole pass instead of once per indexed object.
        const OvdObjectSet knownObjects = collectDomObjects(*mDom);
        std::unordered_map<const OmniPvdObject*, OwnershipSummary> ownershipMemo;
        ownershipMemo.reserve(knownObjects.size());
        std::vector<const OmniPvdObject*> ancestryPath;
        ancestryPath.reserve(std::min<size_t>(knownObjects.size(), 64));

        for (ObjectEntry& e : mObjects)
        {
            // Normal captures expose only PxPhysics at the inspector root and
            // nest scenes beneath it. A rootless scene remains a fallback root
            // even when another recording segment has its own PxPhysics.
            const std::string className = e.obj->mOmniPvdClass
                                              ? e.obj->mOmniPvdClass->mClassName
                                              : std::string();

            OwnershipSummary ownership;
            ancestryPath.clear();
            const OmniPvdObject* ancestor = e.obj->mAncestor;
            while (ancestor)
            {
                if (knownObjects.find(ancestor) == knownObjects.end() ||
                    ancestryPath.size() >= knownObjects.size())
                {
                    mInputError = OvdInputErrorCode::InvalidDomTopology;
                    return false;
                }

                const auto memoIt = ownershipMemo.find(ancestor);
                if (memoIt != ownershipMemo.end())
                {
                    ownership = memoIt->second;
                    break;
                }

                ancestryPath.push_back(ancestor);
                ancestor = ancestor->mAncestor;
            }

            for (auto it = ancestryPath.rbegin(); it != ancestryPath.rend(); ++it)
            {
                const OmniPvdObject* node = *it;
                if (node->mOmniPvdClass)
                {
                    const std::string& name = node->mOmniPvdClass->mClassName;
                    if (name == "PxPhysics")
                    {
                        // Match the old walk's PxPhysics stop condition: any
                        // malformed ancestry above a physics root does not
                        // contribute scene ownership below that root.
                        ownership = OwnershipSummary();
                        ownership.hasPhysics = true;
                        const auto idIt = objToId.find(node);
                        if (idIt != objToId.end()) ownership.physicsId = idIt->second;
                    }
                    else if (name == "PxScene")
                    {
                        const auto idIt = objToId.find(node);
                        if (idIt != objToId.end()) ownership.sceneId = idIt->second;
                    }
                }
                ownershipMemo[node] = ownership;
            }

            e.isTopLevel = className == "PxPhysics" ||
                           (className == "PxScene" && !ownership.hasPhysics);
            mObjAncestorSceneId[e.id] = ownership.sceneId;
            mObjAncestorPhysicsId[e.id] = ownership.physicsId;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// rebuildIndex() — incremental re-scan of a growing (live) DOM
// ---------------------------------------------------------------------------

bool OvdDomAdapter::rebuildIndex()
{
    // buildIndex() is append-only (skips already-indexed class/object handles)
    // and recomputes Pass 3 owning-scene resolution for every object, so simply
    // re-running it picks up everything registered since the last pass while
    // keeping existing class/object ids — and the propIds/instance ids derived
    // from them — stable for any UI state already referencing them.
    if (!buildIndex())
        return false;

    // The current frame's sample lists may have grown (new keyframes appended)
    // and objects whose scene ancestor only just arrived now resolve, so the
    // frame-dependent read caches must be dropped. mPropFlexCache and the string
    // table are id-stable / monotonic and are kept.
    mUniqueListCache.invalidate();
    mSampleHintCache.clear();
    return true;
}

// ---------------------------------------------------------------------------
// Frame navigation
// ---------------------------------------------------------------------------

PxU32 OvdDomAdapter::getFrameCount() const
{
    if (mDom->mMaxFrame < mDom->mMinFrame)
        return 1;
    // +1 for the pre-section frame (frame before mMinFrame)
    return static_cast<PxU32>(mDom->mMaxFrame - mDom->mMinFrame + 2);
}

PxU32 OvdDomAdapter::getCurrentFrame() const
{
    return static_cast<PxU32>(mCurrentFrame);
}

void OvdDomAdapter::setCurrentFrame(PxU64 frame)
{
    if (mCurrentFrame != frame)
    {
        mCurrentFrame = frame;
        mUniqueListCache.invalidate();
        // mSampleHintCache persists across frame changes — forward scrubbing
        // resumes from cached positions instead of walking from the beginning.
    }
}

PxU64 OvdDomAdapter::getMinFrame() const
{
    return mDom->mMinFrame;
}

PxU64 OvdDomAdapter::getMaxFrame() const
{
    return mDom->mMaxFrame;
}

PxU32 OvdDomAdapter::getSegmentEpoch() const
{
    if (mClassGeneration.empty())
        return 0;
    return static_cast<PxU32>(*std::max_element(mClassGeneration.begin(), mClassGeneration.end()));
}

// ---------------------------------------------------------------------------
// Scenes
// ---------------------------------------------------------------------------

PxU32 OvdDomAdapter::getNbScenes() const
{
    return mDom->mSceneCreations.empty() ? 0u : 1u;
}

PxU64 OvdDomAdapter::getSceneStreamId(PxU32 /*index*/) const
{
    if (mDom->mSceneCreations.empty())
        return 0;

    OmniPvdObject* scene = mDom->mSceneCreations.front();
    return scene->mOmniAPIHandle ? scene->mOmniAPIHandle : scene->mOmniObjectHandle;
}

PxU32 OvdDomAdapter::getScenes(SectionListInfo* outBuffer, PxU32 count, PxU32 startIndex) const
{
    if (!outBuffer || count == 0 || startIndex >= getNbScenes())
        return 0;

    PxU32 written = 0;
    PxU32 nbScenes = getNbScenes();
    for (PxU32 i = startIndex; i < nbScenes && written < count; ++i, ++written)
    {
        outBuffer[written].mStreamId = getSceneStreamId(i);
        outBuffer[written].mName     = 0; // no string handle
    }
    return written;
}

// ---------------------------------------------------------------------------
// Class metadata
// ---------------------------------------------------------------------------

PxU32 OvdDomAdapter::getNbClasses() const
{
    return static_cast<PxU32>(mClasses.size());
}

PxU32 OvdDomAdapter::getClasses(ClassDescription* outBuffer, PxU32 count, PxU32 startIndex) const
{
    if (!outBuffer || count == 0)
        return 0;

    PxU32 nb = getNbClasses();
    PxU32 written = 0;
    for (PxU32 i = startIndex; i < nb && written < count; ++i, ++written)
    {
        const ClassEntry& e = mClasses[i];
        ClassDescription cd(e.nsName, e.id);
        outBuffer[written] = cd;
    }
    return written;
}

Option<ClassDescription> OvdDomAdapter::getClass(PxI32 classId) const
{
    if (classId < 0 || classId >= static_cast<PxI32>(mClasses.size()))
        return Option<ClassDescription>(None());

    const ClassEntry& e = mClasses[static_cast<size_t>(classId)];
    ClassDescription cd(e.nsName, e.id);
    return cd;
}

Option<ClassDescription> OvdDomAdapter::findClass(const char* name) const
{
    if (!name)
        return Option<ClassDescription>(None());

    auto it = mClassNameToId.find(name);
    if (it == mClassNameToId.end())
        return Option<ClassDescription>(None());

    return getClass(it->second);
}

// ---------------------------------------------------------------------------
// Property metadata
// ---------------------------------------------------------------------------

// Collect all properties from the inheritance chain (ancestors first, then own class)
void OvdDomAdapter::collectAllProperties(PxI32 classId, std::vector<PropertyDescription>& out) const
{
    if (classId < 0 || classId >= static_cast<PxI32>(mClasses.size()))
        return;

    const ClassEntry& entry = mClasses[static_cast<size_t>(classId)];
    if (!entry.cls)
        return;

    // Walk inheritance chain: [0]=root ancestor ... [n]=this class
    for (OmniPvdClass* ancestor : entry.cls->mInheritanceChain)
    {
        if (!ancestor || ancestor->mIsEnumClass)
            continue;
        auto it = mClassHandleToId.find(ancestor->mOmniClassHandle);
        if (it == mClassHandleToId.end())
            continue;
        const ClassEntry& ancEntry = mClasses[static_cast<size_t>(it->second)];
        for (const PropertyDescription& pd : ancEntry.props)
            out.push_back(pd);
    }
}

PxU32 OvdDomAdapter::getNbProperties(PxI32 classId) const
{
    std::vector<PropertyDescription> all;
    collectAllProperties(classId, all);
    return static_cast<PxU32>(all.size());
}

PxU32 OvdDomAdapter::getProperties(PxI32 classId, PropertyDescription* outBuffer,
                                    PxU32 count, PxU32 startIndex) const
{
    if (!outBuffer || count == 0)
        return 0;

    std::vector<PropertyDescription> all;
    collectAllProperties(classId, all);
    PxU32 nb = static_cast<PxU32>(all.size());
    PxU32 written = 0;
    for (PxU32 i = startIndex; i < nb && written < count; ++i, ++written)
        outBuffer[written] = all[i];
    return written;
}

Option<PropertyDescription> OvdDomAdapter::findProperty(PxI32 classId, const char* name) const
{
    if (!name || classId < 0 || classId >= static_cast<PxI32>(mClasses.size()))
        return Option<PropertyDescription>(None());

    // Search own properties first
    const ClassEntry& entry = mClasses[static_cast<size_t>(classId)];
    for (const PropertyDescription& pd : entry.props)
    {
        if (pd.mName && strcmp(pd.mName, name) == 0)
            return pd;
    }

    // Walk the inheritance chain and search parent class properties.
    // mInheritanceChain goes from root [0] toward the class itself [last].
    // The class itself is the last entry; search all entries except the last
    // (those are ancestors).
    if (entry.cls)
    {
        const std::vector<OmniPvdClass*>& chain = entry.cls->mInheritanceChain;
        // chain[last] is the class itself – skip it, we already searched
        for (int ci = static_cast<int>(chain.size()) - 2; ci >= 0; --ci)
        {
            OmniPvdClass* ancestor = chain[static_cast<size_t>(ci)];
            if (!ancestor || ancestor->mIsEnumClass)
                continue;
            auto ait = mClassHandleToId.find(ancestor->mOmniClassHandle);
            if (ait == mClassHandleToId.end())
                continue;
            PxI32 ancClassId = ait->second;
            if (ancClassId < 0 || ancClassId >= static_cast<PxI32>(mClasses.size()))
                continue;
            const ClassEntry& ancEntry = mClasses[static_cast<size_t>(ancClassId)];
            for (const PropertyDescription& pd : ancEntry.props)
            {
                if (pd.mName && strcmp(pd.mName, name) == 0)
                    return pd;
            }
        }
    }

    return Option<PropertyDescription>(None());
}

Option<PropertyDescription> OvdDomAdapter::findPropertyFlex(PxI32 classId, const char* name) const
{
    if (!name || name[0] == '\0')
        return Option<PropertyDescription>(None());

    // Cache key: classId in upper 32 bits, name hash in lower 32 bits.
    // Property metadata is frame-independent, so this cache is never invalidated.
    std::hash<std::string> hasher;
    uint64_t cacheKey = (static_cast<uint64_t>(static_cast<uint32_t>(classId)) << 32)
                      | static_cast<uint32_t>(hasher(name));
    auto cacheIt = mPropFlexCache.find(cacheKey);
    if (cacheIt != mPropFlexCache.end())
        return cacheIt->second;

    // Try exact name first
    Option<PropertyDescription> result = findProperty(classId, name);
    if (result.hasValue())
    {
        mPropFlexCache[cacheKey] = result;
        return result;
    }

    // Try camelCase variant: lowercase first letter (PascalCase → camelCase)
    if (isupper(static_cast<unsigned char>(name[0])))
    {
        std::string camel = name;
        camel[0] = static_cast<char>(tolower(static_cast<unsigned char>(camel[0])));
        result = findProperty(classId, camel.c_str());
        if (result.hasValue())
        {
            mPropFlexCache[cacheKey] = result;
            return result;
        }
    }

    // Try PascalCase variant: uppercase first letter (camelCase → PascalCase)
    if (islower(static_cast<unsigned char>(name[0])))
    {
        std::string pascal = name;
        pascal[0] = static_cast<char>(toupper(static_cast<unsigned char>(pascal[0])));
        result = findProperty(classId, pascal.c_str());
        if (result.hasValue())
        {
            mPropFlexCache[cacheKey] = result;
            return result;
        }
    }

    // Try special aliases: "Geometry" → "geom" and vice versa
    static const char* aliasFrom[] = { "Geometry", "geometry" };
    static const char* aliasTo[]   = { "geom",     "geom"     };
    for (int i = 0; i < 2; ++i)
    {
        if (strcmp(name, aliasFrom[i]) == 0)
        {
            result = findProperty(classId, aliasTo[i]);
            if (result.hasValue())
            {
                mPropFlexCache[cacheKey] = result;
                return result;
            }
        }
    }
    // reverse alias: "geom" → "Geometry"
    if (strcmp(name, "geom") == 0)
    {
        result = findProperty(classId, "Geometry");
        if (result.hasValue())
        {
            mPropFlexCache[cacheKey] = result;
            return result;
        }
    }

    // Cache the miss too, so we don't repeat the search
    mPropFlexCache[cacheKey] = Option<PropertyDescription>(None());
    return Option<PropertyDescription>(None());
}

// ---------------------------------------------------------------------------
// Instance queries
// ---------------------------------------------------------------------------

PxU32 OvdDomAdapter::getNbInstances() const
{
    // Return total object count (matching PXD2 convention).
    // Each instance carries an mAlive flag — callers filter by alive status.
    PxU32 count = 0;
    for (const ObjectEntry& e : mObjects)
    {
        if (e.classId >= 0)
            ++count;
    }
    return count;
}

PxU32 OvdDomAdapter::getInstances(InstanceDescription* outBuffer, PxU32 count,
                                   PxU32 startIndex) const
{
    if (!outBuffer || count == 0)
        return 0;

    PxU32 written   = 0;
    PxU32 skipped   = 0;
    for (const ObjectEntry& e : mObjects)
    {
        if (e.classId < 0)
            continue;

        bool alive = isAliveAtFrame(e.obj, mCurrentFrame);

        if (skipped < startIndex)
        {
            ++skipped;
            continue;
        }
        if (written >= count)
            break;

        InstanceDescription inst;
        // Use 1-based IDs so that 0 means "null" in PVD3
        inst.mId      = e.id + 1;
        inst.mClassId = e.classId;
        // Use the unique internal key as instPtr — guaranteed unique per object
        inst.mInstPtr = reinterpret_cast<void*>(e.internalKey);
        inst.mAlive   = alive;

        outBuffer[written++] = inst;
    }
    return written;
}

bool OvdDomAdapter::isTopLevel(PxI32 instId) const
{
    // instId is 1-based
    PxI32 idx = instId - 1;
    if (idx < 0 || idx >= static_cast<PxI32>(mObjects.size()))
        return false;
    return mObjects[static_cast<size_t>(idx)].isTopLevel;
}

PxU64 OvdDomAdapter::getRecordingSegment(PxI32 instId) const
{
    if (instId <= 0 || static_cast<size_t>(instId) > mObjects.size()) return 0;
    const auto& entry = mObjects[static_cast<size_t>(instId - 1)];
    if (mDom->mSawRecordingSegmentMetadata) return entry.obj->mRecordingSegmentId;
    return entry.classId >= 0 && static_cast<size_t>(entry.classId) < mClassGeneration.size()
        ? static_cast<PxU64>(mClassGeneration[entry.classId]) : 0;
}

PxI32 OvdDomAdapter::getOwningSceneId(PxI32 instId) const
{
    // instId is 1-based; mObjAncestorSceneId is keyed by and stores 0-based
    // synthetic ids (the e.id used in buildIndex). Convert both ends.
    const PxI32 idx = instId - 1;
    auto it = mObjAncestorSceneId.find(idx);
    if (it == mObjAncestorSceneId.end() || it->second < 0)
        return -1;
    return it->second + 1;  // back to 1-based
}

PxI32 OvdDomAdapter::getOwningPhysicsId(PxI32 instId) const
{
    // Nearest PxPhysics ancestor in the DOM tree, segment-consistent (each
    // segment's scene nests under that segment's re-created physics). Same
    // 1-based <-> 0-based convention as getOwningSceneId.
    const PxI32 idx = instId - 1;
    auto it = mObjAncestorPhysicsId.find(idx);
    if (it == mObjAncestorPhysicsId.end() || it->second < 0)
        return -1;
    return it->second + 1;
}

bool OvdDomAdapter::isSceneJoint(PxI32 instId) const
{
    const PxI32 idx = instId - 1;
    if (idx < 0 || idx >= static_cast<PxI32>(mObjects.size()))
        return false;
    const PxI32 classId = mObjects[static_cast<size_t>(idx)].classId;
    return classId >= 0 && mClasses[static_cast<size_t>(classId)].cls->mPhysXBaseProcessingClassId ==
        OmniPvdPhysXClassEnum::ePxJoint;
}

ClassDescription OvdDomAdapter::getClassOf(void* instPtr) const
{
    PxU64 handle = reinterpret_cast<PxU64>(instPtr);
    PxI32 objIdx = resolvePublishedInstanceKey(handle);
    if (objIdx < 0 || objIdx >= static_cast<PxI32>(mObjects.size()))
        return ClassDescription();

    PxI32 classId = mObjects[static_cast<size_t>(objIdx)].classId;
    Option<ClassDescription> cd = getClass(classId);
    return cd.hasValue() ? cd.getValue() : ClassDescription();
}

PxU64 OvdDomAdapter::getStreamId(PxI32 instId) const
{
    PxI32 idx = instId - 1;
    if (idx < 0 || idx >= static_cast<PxI32>(mObjects.size()))
        return 0;
    const OmniPvdObject* obj = mObjects[static_cast<size_t>(idx)].obj;
    return obj->mOmniAPIHandle ? obj->mOmniAPIHandle : obj->mOmniObjectHandle;
}

// ---------------------------------------------------------------------------
// Property value queries
// ---------------------------------------------------------------------------

PxU32 OvdDomAdapter::getPropertyByteSize(void* instPtr, PxI32 propId) const
{
    PxU64 handle = reinterpret_cast<PxU64>(instPtr);
    PxI32 objIdx = resolvePublishedInstanceKey(handle);
    if (objIdx < 0 || objIdx >= static_cast<PxI32>(mObjects.size()))
        return 0;

    OmniPvdObject* obj = mObjects[static_cast<size_t>(objIdx)].obj;
    OmniPvdAttributeInstList* attrList = getAttrList(obj, propId);
    if (!attrList)
        return 0;

	OmniPvdAttributeInst* inst = findSampleAtFrame(attrList, mCurrentFrame);
    if (!inst)
        return 0;

	if (attrList->mAttributeDef && !attrList->mAttributeDef->mIsUniqueList &&
	    attrList->mAttributeDef->mDataType == OmniPvdDataType::eSTRING)
		return sizeof(PxU32);

	if (attrList->mAttributeDef && !attrList->mAttributeDef->mIsUniqueList &&
	    attrList->mAttributeDef->mDataType == OmniPvdDataType::eOBJECT_HANDLE &&
	    attrList->mAttributeDef->mNbrFields == 1)
		return sizeof(PxI32);

	if (attrList->mAttributeDef && attrList->mAttributeDef->mIsUniqueList)
    {
        const auto& allElements = getAccumulatedUniqueList(objIdx, propId, attrList);
        PxU32 nElems = static_cast<PxU32>(allElements.size());
        if (attrList->mAttributeDef->mDataType == OmniPvdDataType::eOBJECT_HANDLE)
            return nElems * sizeof(PxI32);
        PxU32 total = 0;
        for (OmniPvdUniqueListElement* elem : allElements)
            total += elem->mDataLen;
        return total;
    }

    OmniPvdAttributeSample* sample = static_cast<OmniPvdAttributeSample*>(inst);
    return sample->mDataLen;
}

bool OvdDomAdapter::hasPropertySample(void* instPtr, PxI32 propId) const
{
    const PxU64 handle = reinterpret_cast<PxU64>(instPtr);
    const PxI32 objIdx = resolvePublishedInstanceKey(handle);
    if (objIdx < 0 || objIdx >= static_cast<PxI32>(mObjects.size()))
        return false;

    OmniPvdObject* obj = mObjects[static_cast<size_t>(objIdx)].obj;
    OmniPvdAttributeInstList* attrList = getAttrList(obj, propId);
    return attrList && findSampleAtFrame(attrList, mCurrentFrame) != nullptr;
}

PxU32 OvdDomAdapter::getPropertyValue(void* instPtr, PxI32 propId,
                                       void* outData, PxU32 outLen) const
{
    if (!outData || outLen == 0)
        return 0;

    PxU64 handle = reinterpret_cast<PxU64>(instPtr);
    PxI32 objIdx = resolvePublishedInstanceKey(handle);
    if (objIdx < 0 || objIdx >= static_cast<PxI32>(mObjects.size()))
        return 0;

    OmniPvdObject* obj = mObjects[static_cast<size_t>(objIdx)].obj;
    OmniPvdAttributeInstList* attrList = getAttrList(obj, propId);
    if (!attrList || !attrList->mAttributeDef)
        return 0;

    OmniPvdAttributeDef* attrDef = attrList->mAttributeDef;
    OmniPvdAttributeInst* inst   = findSampleAtFrame(attrList, mCurrentFrame);
    if (!inst)
        return 0;

    OmniPvdDataType::Enum dataType = static_cast<OmniPvdDataType::Enum>(attrDef->mDataType);

    // -----------------------------------------------------------------------
    // Unique list (Array property)
    // -----------------------------------------------------------------------
    if (attrDef->mIsUniqueList)
    {
        const auto& allElements = getAccumulatedUniqueList(objIdx, propId, attrList);
        PxU32 written = 0;
        uint8_t* dst  = static_cast<uint8_t*>(outData);

        for (OmniPvdUniqueListElement* elem : allElements)
        {
            if (!elem->mData || elem->mDataLen == 0)
                continue;

            if (dataType == OmniPvdDataType::eOBJECT_HANDLE)
            {
                // Resolve handle → instance ID (1-based), write as int32_t
                OmniPvdObjectHandle ovdHandle = 0;
                if (elem->mDataLen >= sizeof(OmniPvdObjectHandle))
                    memcpy(&ovdHandle, elem->mData, sizeof(OmniPvdObjectHandle));

                int32_t instIdResolved = -1;
                // The reading object is a known referrer, so external handles
                // must resolve within its recording segment or remain unresolved.
                PxI32 refIdx = resolveExternalReferenceAt(static_cast<PxU64>(ovdHandle), objIdx);
                if (refIdx >= 0)
                    instIdResolved = refIdx + 1;

                if (written + sizeof(int32_t) <= outLen)
                {
                    memcpy(dst + written, &instIdResolved, sizeof(int32_t));
                    written += sizeof(int32_t);
                }
            }
            else
            {
                PxU32 toCopy = std::min(elem->mDataLen, outLen - written);
                if (toCopy == 0)
                    break;
                memcpy(dst + written, elem->mData, toCopy);
                written += toCopy;
            }
        }
        return written;
    }

    // -----------------------------------------------------------------------
    // Scalar sample
    // -----------------------------------------------------------------------
    OmniPvdAttributeSample* sample = static_cast<OmniPvdAttributeSample*>(inst);
    if (!sample->mData || sample->mDataLen == 0)
        return 0;

    // eOBJECT_HANDLE scalar: resolve to instance ID
    if (dataType == OmniPvdDataType::eOBJECT_HANDLE)
    {
        OmniPvdObjectHandle ovdHandle = 0;
        if (sample->mDataLen >= sizeof(OmniPvdObjectHandle))
            memcpy(&ovdHandle, sample->mData, sizeof(OmniPvdObjectHandle));

        int32_t instIdResolved = -1;
        PxI32 refIdx = resolveExternalReferenceAt(static_cast<PxU64>(ovdHandle), objIdx);
        if (refIdx >= 0)
            instIdResolved = refIdx + 1;

        PxU32 toCopy = std::min(static_cast<PxU32>(sizeof(int32_t)), outLen);
        memcpy(outData, &instIdResolved, toCopy);
        return toCopy;
    }

    // eSTRING: register string, return handle
    if (dataType == OmniPvdDataType::eSTRING)
    {
		const char* strData = reinterpret_cast<const char*>(sample->mData);
		size_t strLength = strData ? sample->mDataLen : 0;
		if(strLength && strData[strLength - 1] == '\0')
			--strLength;
		const std::string str = strData ? std::string(strData, strLength) : std::string();

        // Look for existing handle
        PxU32 strHandle = 0;
        bool found = false;
        for (auto& kv : mStringTable)
        {
			if (kv.second == str)
            {
                strHandle = kv.first;
                found = true;
                break;
            }
        }
        if (!found)
        {
            strHandle = mNextStringHandle++;
			mStringTable[strHandle] = str;
        }

        PxU32 toCopy = std::min(static_cast<PxU32>(sizeof(PxU32)), outLen);
        memcpy(outData, &strHandle, toCopy);
        return toCopy;
    }

    // General case: memcpy raw sample data
    PxU32 toCopy = std::min(sample->mDataLen, outLen);
    memcpy(outData, sample->mData, toCopy);
    return toCopy;
}

DataRef<uint8_t> OvdDomAdapter::getRawPropertyValue(void* instPtr, PxI32 propId) const
{
    return getRawPropertyValueAtFrame(instPtr, propId, mCurrentFrame);
}

DataRef<uint8_t> OvdDomAdapter::getRawPropertyValueAtFrame(void* instPtr, PxI32 propId,
                                                            PxU64 frame) const
{
    // "Freshest known value" semantic: returns the latest sample with
    // mTimeStamp <= frame. No keyframe-range check — for typical attributes
    // (transforms, names, flags, …) the previous sample's value persists
    // until something else overwrites it, regardless of whether the writer
    // re-confirmed it at the requested frame. Use this for everything except
    // properties where "is data recorded at this frame?" is the correct
    // question (PxScene::pairsContactPoints, etc. — see ...RecordedAtFrame).
    PxU64 handle = reinterpret_cast<PxU64>(instPtr);
    PxI32 objIdx = resolvePublishedInstanceKey(handle);
    if (objIdx < 0 || objIdx >= static_cast<PxI32>(mObjects.size()))
        return DataRef<uint8_t>();

    OmniPvdObject* obj = mObjects[static_cast<size_t>(objIdx)].obj;
    OmniPvdAttributeInstList* attrList = getAttrList(obj, propId);
    if (!attrList || !attrList->mAttributeDef)
        return DataRef<uint8_t>();

    if (attrList->mAttributeDef->mIsUniqueList)
        return DataRef<uint8_t>();

    OmniPvdAttributeInst* inst = findSampleAtFrame(attrList, frame);
    if (!inst)
        return DataRef<uint8_t>();

    OmniPvdAttributeSample* sample = static_cast<OmniPvdAttributeSample*>(inst);
    if (!sample->mData || sample->mDataLen == 0)
        return DataRef<uint8_t>();

    const uint8_t* begin = static_cast<const uint8_t*>(sample->mData);
    return DataRef<uint8_t>(begin, begin + sample->mDataLen);
}

DataRef<uint8_t> OvdDomAdapter::getRawPropertyValueRecordedAtFrame(void* instPtr, PxI32 propId,
                                                                    PxU64 frame) const
{
    // "Was data recorded at this frame?" semantic: returns the sample iff
    // frame == mTimeStamp. pvddom doesn't dedup — each setAttribute is its
    // own keyframe with mTimeStamp == mEndTimeStamp — so the [mTimeStamp,
    // mEndTimeStamp] range check below is exact-frame equality in disguise.
    // After the writer stops emitting (sim end or contacts ended), out-of-
    // range queries return empty so the caller never sees arbitrarily stale
    // data leaking past the recorded window.
    PxU64 handle = reinterpret_cast<PxU64>(instPtr);
    PxI32 objIdx = resolvePublishedInstanceKey(handle);
    if (objIdx < 0 || objIdx >= static_cast<PxI32>(mObjects.size()))
        return DataRef<uint8_t>();

    OmniPvdObject* obj = mObjects[static_cast<size_t>(objIdx)].obj;
    OmniPvdAttributeInstList* attrList = getAttrList(obj, propId);
    if (!attrList || !attrList->mAttributeDef)
        return DataRef<uint8_t>();

    if (attrList->mAttributeDef->mIsUniqueList)
        return DataRef<uint8_t>();

    OmniPvdAttributeInst* inst = findSampleAtFrame(attrList, frame);
    if (!inst)
        return DataRef<uint8_t>();
    if (frame < inst->mTimeStamp || frame > inst->mEndTimeStamp)
        return DataRef<uint8_t>();

    OmniPvdAttributeSample* sample = static_cast<OmniPvdAttributeSample*>(inst);
    if (!sample->mData || sample->mDataLen == 0)
        return DataRef<uint8_t>();

    const uint8_t* begin = static_cast<const uint8_t*>(sample->mData);
    return DataRef<uint8_t>(begin, begin + sample->mDataLen);
}

PxU32 OvdDomAdapter::getNbArrayElements(void* instPtr, PxI32 propId) const
{
    PxU64 handle = reinterpret_cast<PxU64>(instPtr);
    PxI32 objIdx = resolvePublishedInstanceKey(handle);
    if (objIdx < 0 || objIdx >= static_cast<PxI32>(mObjects.size()))
        return 0;

    OmniPvdObject* obj = mObjects[static_cast<size_t>(objIdx)].obj;
    OmniPvdAttributeInstList* attrList = getAttrList(obj, propId);
    if (!attrList || !attrList->mAttributeDef)
        return 0;

    if (!attrList->mAttributeDef->mIsUniqueList)
        return 0;

    const auto& allElements = getAccumulatedUniqueList(objIdx, propId, attrList);
    return static_cast<PxU32>(allElements.size());
}

PxU32 OvdDomAdapter::getNbArrayElements(void* instPtr, const char* propName) const
{
    if (!propName)
        return 0;

    PxU64 handle = reinterpret_cast<PxU64>(instPtr);
    PxI32 objIdx = resolvePublishedInstanceKey(handle);
    if (objIdx < 0 || objIdx >= static_cast<PxI32>(mObjects.size()))
        return 0;

    PxI32 classId = mObjects[static_cast<size_t>(objIdx)].classId;
    Option<PropertyDescription> pd = findPropertyFlex(classId, propName);
    if (!pd.hasValue())
        return 0;

    return getNbArrayElements(instPtr, pd.getValue().mPropertyId);
}

// ---------------------------------------------------------------------------
// String table
// ---------------------------------------------------------------------------

const char* OvdDomAdapter::handleToString(PxU32 handle) const
{
    auto it = mStringTable.find(handle);
    if (it == mStringTable.end())
        return "";
    return it->second.c_str();
}

DataRef<NamedValue> OvdDomAdapter::getNamedPropertyValues(PxI32 propId) const
{
    auto it = mNamedValuesByPropId.find(propId);
    if (it == mNamedValuesByPropId.end() || it->second.empty())
        return DataRef<NamedValue>();
    return DataRef<NamedValue>(it->second.data(),
                               static_cast<PxU32>(it->second.size()));
}

int64_t OvdDomAdapter::getPropertyFieldCount(PxI32 propId) const
{
    const PxI32 classIdx=(propId >> 16) & 0xFFFF;
    const PxI32 attrIdx=propId & 0xFFFF;
    if(classIdx<0 || classIdx>=static_cast<PxI32>(mClasses.size())) return -1;
    const auto& ce=mClasses[classIdx];
    if(!ce.cls || attrIdx<0 || attrIdx>=static_cast<PxI32>(ce.cls->mAttributeDefinitions.size())) return -1;
    const auto* def=ce.cls->mAttributeDefinitions[attrIdx];
    if(!def) return -1;
    if(def->mIsUniqueList) return 0;
    if(def->mDataType==OmniPvdDataType::eFLAGS_WORD || def->mDataType==OmniPvdDataType::eENUM_VALUE) return 1;
    return int64_t(def->mNbrFields);
}

std::string OvdDomAdapter::getNativeTypeLabel(PxI32 propId) const
{
    const PxI32 classIdx = (propId >> 16) & 0xFFFF;
    const PxI32 attrIdx  = propId & 0xFFFF;
    if (classIdx < 0 || classIdx >= static_cast<PxI32>(mClasses.size()))
        return std::string();
    const ClassEntry& ce = mClasses[classIdx];
    if (!ce.cls || attrIdx < 0
        || attrIdx >= static_cast<PxI32>(ce.cls->mAttributeDefinitions.size()))
        return std::string();
    OmniPvdAttributeDef* def = ce.cls->mAttributeDefinitions[attrIdx];
    if (!def)
        return std::string();

    // Type column shows PhysX-shaped type names (PxVec3, PxShapeFlag, etc.) —
    // OVD-native names like eFLOAT32 / eFLAGS_WORD aren't useful to testers
    // who think in PhysX terms. We use the same PhysX shape mapping as
    // domTypeToPvd3Name (so PxVec3 / PxTransform / PxBounds3 surface correctly
    // for multi-field eFLOAT32) but override flags / enums with the source
    // enum class name (PxShapeFlag instead of PvdU32).
    std::string label;
    if (def->mDataType == OmniPvdDataType::eFLAGS_WORD
        || def->mDataType == OmniPvdDataType::eENUM_VALUE)
    {
        if (def->mDerivedFromClass)
            label = def->mDerivedFromClass->mClassName;
        else
            label = (def->mDataType == OmniPvdDataType::eFLAGS_WORD) ? "PxFlags" : "PxEnum";
    }
    else
    {
        label = domTypeToPvd3Name(def->mDataType, def->mNbrFields).mName;
    }
    if (def->mIsUniqueList)
        label += "[]";
    // domTypeToPvd3Name returns the PVD-SDK canonical name (PvdU8, PvdF32, …);
    // testers think in PhysX terms, so swap the leading "Pvd" prefix for "Px"
    // before showing the label.
    if (label.compare(0, 3, "Pvd") == 0)
        label.replace(0, 3, "Px");
    return label;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

OmniPvdAttributeInst* OvdDomAdapter::findSampleAtFrame(OmniPvdAttributeInstList* list,
                                                         PxU64 frame) const
{
    if (!list)
        return nullptr;

    auto hintIt = mSampleHintCache.find(list);
    if (hintIt != mSampleHintCache.end())
    {
        const SampleHint& hint = hintIt->second;

        // Same frame: return cached result directly (avoids double-walk for byteSize+getValue)
        if (hint.frame == frame)
            return hint.result;

        // Forward scrub: start from cached position and advance
        if (frame > hint.frame && hint.result)
        {
            OmniPvdAttributeInst* best = hint.result;
            OmniPvdAttributeInst* inst = best->mNextAttribute;
            while (inst != nullptr && inst->mTimeStamp <= frame)
            {
                best = inst;
                inst = inst->mNextAttribute;
            }
            hintIt->second = { frame, best };
            return best;
        }

        // Backward scrub: fall through to full walk from beginning
    }

    // Full walk from the beginning
    OmniPvdAttributeInst* best = nullptr;
    for (OmniPvdAttributeInst* inst = list->mFirst; inst != nullptr; inst = inst->mNextAttribute)
    {
        if (inst->mTimeStamp <= frame)
            best = inst;
        else
            break; // timestamps are ordered
    }

    mSampleHintCache[list] = { frame, best };
    return best;
}

/*static*/
bool OvdDomAdapter::isAliveAtFrame(const OmniPvdObject* obj, PxU64 frame)
{
    // Delegate to pvddom's canonical helper so the half-open semantic
    // (writer stamps mFrameStop with the FIRST dead frame) is shared with the
    // omni.physx.pvd kit reader instead of forking. See PvdDomUtils.h.
    return ::isObjectAliveAtFrame(obj, frame);
}

OmniPvdAttributeInstList* OvdDomAdapter::getAttrList(OmniPvdObject* obj, PxI32 propId) const
{
    if (!obj)
        return nullptr;

    // Decode the packed propId
    PxI32 classIdx = propId >> 16;
    PxI32 attrIdx  = propId & 0xFFFF;

    if (classIdx < 0 || classIdx >= static_cast<PxI32>(mClasses.size()))
        return nullptr;

    const ClassEntry& targetEntry = mClasses[static_cast<size_t>(classIdx)];
    if (!targetEntry.cls)
        return nullptr;

    // Walk the object's mInheritedClassInstances to find the slot matching classIdx.
    // The inheritance chain on the object's class maps 1:1 with mInheritedClassInstances.
    OmniPvdClass* objClass = obj->mOmniPvdClass;
    if (!objClass)
        return nullptr;

    const std::vector<OmniPvdClass*>& chain = objClass->mInheritanceChain;
    const std::vector<OmniPvdClassInstance>& instances = obj->mInheritedClassInstances;

    for (size_t ci = 0; ci < chain.size() && ci < instances.size(); ++ci)
    {
        OmniPvdClass* chainCls = chain[ci];
        if (!chainCls)
            continue;

        // Match by class handle
        if (chainCls->mOmniClassHandle != targetEntry.cls->mOmniClassHandle)
            continue;

        // Found the right class instance slot
        const OmniPvdClassInstance& classInst = instances[ci];
        if (attrIdx < 0 || attrIdx >= static_cast<PxI32>(classInst.mClassAttributeLists.size()))
            return nullptr;

        return classInst.mClassAttributeLists[static_cast<size_t>(attrIdx)];
    }

    return nullptr;
}

const std::vector<OmniPvdUniqueListElement*>& OvdDomAdapter::getAccumulatedUniqueList(
    PxI32 objIdx, PxI32 propId, OmniPvdAttributeInstList* attrList) const
{
    // Invalidate if frame changed since last cache use
    if (mUniqueListCache.frame != mCurrentFrame)
    {
        mUniqueListCache.entries.clear();
        mUniqueListCache.frame = mCurrentFrame;
    }

    uint64_t cacheKey = (static_cast<uint64_t>(objIdx) << 32) | static_cast<uint32_t>(propId);
    auto it = mUniqueListCache.entries.find(cacheKey);
    if (it != mUniqueListCache.entries.end())
        return it->second;

    // Cache miss: walk the linked list and accumulate elements
    auto& elems = mUniqueListCache.entries[cacheKey];
    for (OmniPvdAttributeInst* inst = attrList->mFirst; inst != nullptr; inst = inst->mNextAttribute)
    {
        if (inst->mTimeStamp > mCurrentFrame) break;
        OmniPvdUniqueList* ul = static_cast<OmniPvdUniqueList*>(inst);
        for (OmniPvdUniqueListElement* elem : ul->mElements)
            elems.push_back(elem);
    }
    return elems;
}

PxI32 OvdDomAdapter::resolvePublishedInstanceKey(PxU64 internalKey) const
{
    auto it = mInternalKeyToId.find(internalKey);
    return (it != mInternalKeyToId.end()) ? it->second : -1;
}

PxI32 OvdDomAdapter::resolveExternalReferenceAt(PxU64 apiHandle, PxI32 referrerIdx) const
{
    auto cit = mApiHandleCandidates.find(apiHandle);
    if (cit == mApiHandleCandidates.end() || cit->second.empty())
        return -1;
    const std::vector<PxI32>& cands = cit->second; // creation order (mUID ascending)

    // SEGMENT identity first: use pvddom's per-object segment id whenever the
    // parser observed recording-segment metadata. Older DOMs lack that marker,
    // so retain class registration generations as their compatibility fallback.
    // When exactly one candidate shares the referrer's segment, it wins even
    // over frame-aliveness: on the overlaid shared timeline the earlier
    // segment's copy is closed at the boundary frame (and a LIVE stream keeps
    // the newest segment's copy open forever), so aliveness at the newest
    // frame would leak the reference into the other segment. The frame-alive
    // filter below then keeps same-segment scene-recreate (disjoint lifespans)
    // semantics.
    std::vector<PxI32> pool;
    const std::vector<PxI32>* scan = &cands;
    if (referrerIdx >= 0 && referrerIdx < static_cast<PxI32>(mObjects.size()))
    {
        bool hasSegmentIdentity = false;
        if (mDom->mSawRecordingSegmentMetadata)
        {
            hasSegmentIdentity = true;
            const uint64_t refSegmentId = mObjects[referrerIdx].obj->mRecordingSegmentId;
            for (PxI32 id : cands)
            {
                if (mObjects[id].obj->mRecordingSegmentId == refSegmentId)
                    pool.push_back(id);
            }
        }
        else
        {
            const PxI32 refClassId = mObjects[referrerIdx].classId;
            const PxI32 refGen = (refClassId >= 0 && refClassId < static_cast<PxI32>(mClassGeneration.size()))
                                     ? mClassGeneration[refClassId] : -1;
            if (refGen >= 0)
            {
                hasSegmentIdentity = true;
                for (PxI32 id : cands)
                {
                    const PxI32 classId = mObjects[id].classId;
                    if (classId >= 0 && classId < static_cast<PxI32>(mClassGeneration.size())
                        && mClassGeneration[classId] == refGen)
                        pool.push_back(id);
                }
            }
        }
        if (hasSegmentIdentity)
        {
            if (pool.empty())
                return -1;
            if (pool.size() == 1)
                return pool[0];
            scan = &pool;
        }
    }
    if (scan == &cands && cands.size() == 1)
        return cands[0];

    // With no known referrer, or with several same-segment candidates (a single
    // segment reusing a pointer), resolve within the selected candidate set
    // by frame-aliveness -- a scene recreated mid-capture yields copies with
    // disjoint lifespans, so exactly one is alive at the scrubbed frame.
    PxI32 newestAlive = -1;
    int   aliveCount  = 0;
    for (PxI32 id : *scan) // creation-ordered, so the last alive hit is the newest
    {
        if (isAliveAtFrame(mObjects[id].obj, mCurrentFrame))
        {
            newestAlive = id;
            ++aliveCount;
        }
    }
    if (aliveCount > 0)
        return newestAlive; // the single alive copy, or the newest of several
    return scan->back();    // none alive: newest (deterministic)
}

PxI32 OvdDomAdapter::resolveInstanceId(PxU64 externalHandle, void* referrerInstPtr) const
{
    // Resolve in the external-reference namespace first. Without a known
    // referrer, retain the legacy DOM external→internal fallback; a known
    // referrer must never use it to escape segment filtering.
    const PxI32 referrerIdx = referrerInstPtr
        ? resolvePublishedInstanceKey(reinterpret_cast<PxU64>(referrerInstPtr))
        : -1;
    PxI32 idx = resolveExternalReferenceAt(externalHandle, referrerIdx);
    if (idx < 0 && referrerIdx < 0 && mDom)
    {
        auto extIt = mDom->mExternalToInternalHandleMap.find(externalHandle);
        if (extIt != mDom->mExternalToInternalHandleMap.end())
            idx = resolvePublishedInstanceKey(static_cast<PxU64>(extIt->second));
    }
    return idx >= 0 ? idx + 1 : -1;
}

} // namespace pvdapp

#endif // PVD_OVD_SUPPORT
