/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PVD_OVD_DOM_ADAPTER_H
#define PVD_OVD_DOM_ADAPTER_H

#ifdef PVD_OVD_SUPPORT

#include "PvdDom.h"
#include "PvdDomParser.h"   // OmniPvdMessage
#include "PxPvdObjectModelBaseTypes.h"
#include "PxPvdObjectModelMetaData.h"
#include "PvdCommStreamMetaData.h"
#include "OvdInputValidation.h"
#include "OvdEvidenceMetadata.h"

#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace pvdapp
{

using namespace physx;
using namespace physx::pvdsdk;

// Validate every known DOM object's parent chain without trusting or walking
// through pointers outside the DOM-owned object set. This is intentionally
// adapter-independent so live ingestion can reject a malformed completed frame
// before publishing it to the UI.
bool validateOvdDomTopology(const OmniPvdDOMState& dom);

// ---------------------------------------------------------------------------
// OvdDomAdapter
//
// Owns an OmniPvdDOMState and serves all UI/renderer queries directly from
// the DOM, bypassing PVD3's object model. The adapter builds an internal
// index on construction and then answers metadata and property-value queries
// without going through PvdObjectModelMetaData or PvdObjectModelReader.
//
// propId encoding: (classIndex << 16) | attrIndex, packed as int32_t.
// classIndex is the 0-based index of the ClassEntry in mClasses.
// attrIndex  is the 0-based index of the attribute in
//            OmniPvdClass::mAttributeDefinitions.
// ---------------------------------------------------------------------------

class OvdDomAdapter
{
public:
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    OvdDomAdapter(std::unique_ptr<OmniPvdDOMState> dom,
                  std::vector<OmniPvdMessage>      msgs,
                  OvdEvidenceFrameScopes scopes={});
    OvdEvidenceDomain evidenceDomain() const;
    OvdEvidenceIdentity evidenceIdentity(PxI32 instanceId) const;
    OvdEvidenceSampleStamp evidenceSampleStamp(void* instance,PxI32 property,uint64_t frame) const;

    // Non-copyable; move is fine if needed later.
    OvdDomAdapter(const OvdDomAdapter&) = delete;
    OvdDomAdapter& operator=(const OvdDomAdapter&) = delete;

    // -----------------------------------------------------------------------
    // Diagnostic messages (eRECORD_MESSAGE stream)
    //
    // Parser-global messages are drained into this engine-owned vector after
    // every isolated command, so file and live engines never share storage.
    // -----------------------------------------------------------------------
    const std::vector<OmniPvdMessage>& getMessages() const { return mMessages; }
    std::vector<OmniPvdMessage>& liveMessageSink() { return mMessages; }

    // -----------------------------------------------------------------------
    // Frame navigation
    // -----------------------------------------------------------------------

    PxU32   getFrameCount()                  const;
    PxU32   getCurrentFrame()                const;
    void    setCurrentFrame(PxU64 frame);
    PxU64   getMinFrame()                    const;
    PxU64   getMaxFrame()                    const;
    PxU32   getSegmentEpoch()                const;
    PxU64   getRecordingSegment(PxI32 instId) const;

    // -----------------------------------------------------------------------
    // Live ingestion
    //
    // Re-scan the (growing) DOM and index any classes/objects registered since
    // the last pass. The index is append-only: existing class/object ids stay
    // stable so propIds and instance ids the UI already holds remain valid as
    // the live stream grows. Pass 3 owning-scene resolution is recomputed and
    // the frame-dependent caches are invalidated (new samples may have arrived
    // for the current frame). The caller MUST hold the DOM mutex (the worker
    // thread is concurrently appending to the same OmniPvdDOMState).
    // -----------------------------------------------------------------------
    bool    rebuildIndex();
    bool    isValid() const { return mInputError == OvdInputErrorCode::None; }
    OvdInputErrorCode errorCode() const { return mInputError; }

    // The underlying DOM (borrowed by OvdLiveSource so the ingest worker can
    // grow it in place) and the recursive mutex that serializes those writes
    // against UI/renderer reads. The mutex is recursive so a UI read path that
    // nests adapter calls can re-lock on the same thread without deadlocking.
    // Both are unused for file-loaded captures (no worker, no contention).
    OmniPvdDOMState*      dom()       { return mDom.get(); }
    std::recursive_mutex& liveMutex() { return mLiveMutex; }

    // -----------------------------------------------------------------------
    // Scenes
    // -----------------------------------------------------------------------

    PxU32  getNbScenes()                     const;
    PxU64  getSceneStreamId(PxU32 index = 0) const;

    // Fills outBuffer with at most count SectionListInfo entries starting at
    // startIndex. Returns the number of entries written.
    PxU32  getScenes(SectionListInfo* outBuffer, PxU32 count, PxU32 startIndex = 0) const;

    // -----------------------------------------------------------------------
    // Class metadata
    // -----------------------------------------------------------------------

    PxU32  getNbClasses()                                                 const;
    PxU32  getClasses(ClassDescription* outBuffer, PxU32 count,
                      PxU32 startIndex = 0)                               const;
    Option<ClassDescription> getClass(PxI32 classId)                     const;
    Option<ClassDescription> findClass(const char* name)                  const;

    // -----------------------------------------------------------------------
    // Property metadata
    // -----------------------------------------------------------------------

    PxU32  getNbProperties(PxI32 classId)                                 const;
    PxU32  getProperties(PxI32 classId, PropertyDescription* outBuffer,
                         PxU32 count, PxU32 startIndex = 0)               const;

    // Exact name match within the class's own attribute list.
    Option<PropertyDescription> findProperty(PxI32 classId, const char* name) const;

    // Flexible lookup: tries name as-is, then camelCase variant (first letter
    // lowercased) so that OVD camelCase names match PascalCase UI queries.
    Option<PropertyDescription> findPropertyFlex(PxI32 classId, const char* name) const;

    // -----------------------------------------------------------------------
    // Instance queries
    // -----------------------------------------------------------------------

    PxU32  getNbInstances()                                               const;
    PxU32  getInstances(InstanceDescription* outBuffer, PxU32 count,
                        PxU32 startIndex = 0)                             const;

    bool              isTopLevel(PxI32 instId)    const;
    // Synthetic id of the PxScene that owns instId, resolved from the DOM
    // tree (mAncestor) rather than PxScene.actors handle resolution so it
    // stays correct when a PhysX pointer is reused across scenes. Returns
    // -1 for objects with no PxScene ancestor (PxPhysics factory children:
    // meshes, materials, the scene objects themselves, etc.).
    PxI32             getOwningSceneId(PxI32 instId) const;
    PxI32             getOwningPhysicsId(PxI32 instId) const;
    // True for extension objects processed by PvdDom as PxJoint descendants.
    bool              isSceneJoint(PxI32 instId) const;
    ClassDescription  getClassOf(void* instPtr)   const;
    PxU64             getStreamId(PxI32 instId)   const;

    // -----------------------------------------------------------------------
    // Property value queries
    //
    // instPtr here is the unique internal DOM key published by getInstances().
    // It deliberately occupies a separate namespace from external PhysX object
    // handles carried by eOBJECT_HANDLE properties.
    // -----------------------------------------------------------------------

    // Returns the byte size of the latest sample for (instPtr, propId) at
    // the current frame, or 0 if no sample exists.
    PxU32  getPropertyByteSize(void* instPtr, PxI32 propId)               const;

    // True when the DOM has a sample at or before the current frame. This is
    // independent of payload size, so a recorded zero-byte value is present.
    bool   hasPropertySample(void* instPtr, PxI32 propId)                 const;

    // Copies up to outLen bytes of the latest sample into outData.
    // Returns the number of bytes copied, or 0 on failure.
    PxU32  getPropertyValue(void* instPtr, PxI32 propId,
                            void* outData, PxU32 outLen)                  const;

    // Zero-copy view of the latest non-unique-list sample. Returns an empty
    // DataRef for unique-list properties (use getPropertyValue instead).
    DataRef<uint8_t> getRawPropertyValue(void* instPtr, PxI32 propId)     const;

    // "Freshest known value" — returns the latest sample with
    // mTimeStamp <= frame. No keyframe-range check; the previous sample's
    // value is implicitly assumed to persist until something else overwrites
    // it. Use this for transforms, names, flags, etc.
    DataRef<uint8_t> getRawPropertyValueAtFrame(void* instPtr, PxI32 propId,
                                                PxU64 frame)              const;

    // "Was data recorded at this frame?" — returns the sample iff
    // frame == mTimeStamp. pvddom doesn't dedup, so each setAttribute is its
    // own keyframe with mTimeStamp == mEndTimeStamp; the range check below
    // degenerates to exact-frame equality. Use this for properties where
    // stale data MUST NOT leak past the recorded window — collisions are
    // the canonical case.
    DataRef<uint8_t> getRawPropertyValueRecordedAtFrame(void* instPtr, PxI32 propId,
                                                       PxU64 frame)        const;


    // Number of elements in an array property (unique-list attribute).
    PxU32  getNbArrayElements(void* instPtr, PxI32 propId)                const;
    PxU32  getNbArrayElements(void* instPtr, const char* propName)        const;

    // Resolve an external object reference to a 1-based instance id (matching
    // InstanceDescription::mId). Published mInstPtr keys never enter this
    // external-reference namespace.
    PxI32 resolveInstanceId(PxU64 externalHandle, void* referrerInstPtr = nullptr) const;

    // -----------------------------------------------------------------------
    // String table
    // -----------------------------------------------------------------------

    // Resolve a string handle (from eSTRING property data) to a C string.
    // Returns "" for unknown handles.
    const char* handleToString(PxU32 handle)                              const;

    // Named values for an enum (eENUM_VALUE) or bitflag (eFLAGS_WORD)
    // attribute identified by propId. The OVD DOM stores these as a separate
    // enum class linked via OmniPvdAttributeDef::mDerivedFromClass; we copy
    // the (name, value) pairs at index time so the property panel can render
    // "eSCENE_QUERY_SHAPE | eSIMULATION_SHAPE" instead of "5".
    DataRef<NamedValue> getNamedPropertyValues(PxI32 propId)              const;

    int64_t getPropertyFieldCount(PxI32 propId) const;

    // OVD-native datatype label for a property, formatted for the property
    // panel's "Type" column. Examples: "eFLOAT32x3", "eOBJECT_HANDLE",
    // "eFLAGS_WORD (PxShapeFlag)", "eUINT32[]". Empty string if propId is
    // unknown. This deliberately surfaces OVD's native type info rather than
    // the PhysX-shaped translation in domTypeToPvd3Name.
    std::string getNativeTypeLabel(PxI32 propId)                          const;

private:
    // -----------------------------------------------------------------------
    // Internal index structures
    // -----------------------------------------------------------------------

    // One entry per non-enum OmniPvdClass registered in the DOM.
    struct ClassEntry
    {
        OmniPvdClass*              cls;
        PxI32                      id;       // == index in mClasses (0-based)
        std::string                name;     // cls->mClassName
        NamespacedName             nsName;   // ("physx3", name)

        // Cached PropertyDescription array, one per attribute in
        // cls->mAttributeDefinitions (same order / same index).
        // propId for attribute[i] == (id << 16) | i
        std::vector<PropertyDescription> props;
    };

    // One entry per OmniPvdObject in the DOM (all objects, all frames).
    struct ObjectEntry
    {
        OmniPvdObject* obj;
        PxI32          id;           // == index in mObjects (0-based)
        PxI32          classId;      // ClassEntry::id of the object's class
        PxU64          internalKey;  // unique key from mObjectHandleToObjectMap
        bool           isTopLevel;
    };

    // -----------------------------------------------------------------------
    // Owned data
    // -----------------------------------------------------------------------

    std::unique_ptr<OmniPvdDOMState> mDom;
    std::vector<OmniPvdMessage>      mMessages;
    OvdEvidenceFrameScopes mEvidenceScopes;
    PxU64           mCurrentFrame;
    OvdInputErrorCode mInputError;

    // Serializes OvdLiveSource's ingest-worker writes to mDom against UI/renderer
    // reads through this adapter. Only engaged for a live connection; idle (never
    // contended) for a file-loaded capture. Recursive so nested UI read paths can
    // re-lock on the UI thread. See dom()/liveMutex().
    std::recursive_mutex             mLiveMutex;

    std::vector<ClassEntry>  mClasses;
    std::vector<ObjectEntry> mObjects;

    // Registration generation per ClassEntry::id. Legacy DOMs do not expose
    // object recording-segment metadata, so repeated class names provide their
    // compatibility fallback for segment-consistent reference resolution.
    // Recomputed by buildIndex().
    std::vector<PxI32> mClassGeneration;

    // Handle → index maps (all populated by buildIndex())
    std::unordered_map<OmniPvdClassHandle,  PxI32> mClassHandleToId;
    std::unordered_map<std::string,         PxI32> mClassNameToId;

    // Internal DOM handle → ObjectEntry index
    std::unordered_map<OmniPvdObjectHandle, PxI32> mObjHandleToId;

    // Synthetic object id → owning PxScene synthetic id (or -1), resolved
    // from the DOM tree (mAncestor) in buildIndex Pass 3. Used for per-scene
    // visibility gating that survives PhysX-pointer reuse across scenes.
    std::unordered_map<PxI32, PxI32> mObjAncestorSceneId;
    std::unordered_map<PxI32, PxI32> mObjAncestorPhysicsId;

    // Unique internal key (the DOM's per-creation handle) → ObjectEntry index.
    // instPtr passed by the UI is reinterpret_cast<void*>(internalKey).
    std::unordered_map<PxU64, PxI32> mInternalKeyToId;

    // External API handle (PhysX pointer) → ALL ObjectEntry indices carrying it,
    // sorted by creation order (DOM mUID). A pointer maps to several objects when
    // the writer re-registers it: a scene recreated mid-capture (disjoint
    // lifespans) or a new recording segment re-creating the world onto the same
    // stream (overlapping lifespans on the shared, overlaid timeline). Reference
    // resolution first restricts candidates to the referrer's exact DOM segment,
    // or its legacy class-registration generation, then applies frame aliveness.
    // See resolveExternalReferenceAt().
    std::unordered_map<PxU64, std::vector<PxI32>> mApiHandleCandidates;

    // String table: handle → string, built lazily from eSTRING attribute samples.
    mutable std::unordered_map<PxU32, std::string> mStringTable;
    mutable PxU32                                  mNextStringHandle;

    // Per-property named-value tables (for eENUM_VALUE / eFLAGS_WORD attributes).
    // Keyed by propId. Strings are owned via mNamedValueStringPool so DataRef
    // can hand out raw const char* without copying. Deque (not vector) so
    // c_str() pointers stay valid across appends.
    std::unordered_map<PxI32, std::vector<NamedValue>> mNamedValuesByPropId;
    std::deque<std::string>                             mNamedValueStringPool;

    // Unique-list element cache: avoids re-walking the linked list on every query.
    // Key = (objInternalKey << 32) | propId, Value = accumulated elements at mCurrentFrame.
    // Invalidated on setCurrentFrame() when the frame changes.
    struct UniqueListCache
    {
        PxU64 frame = UINT64_MAX;
        std::unordered_map<uint64_t, std::vector<OmniPvdUniqueListElement*>> entries;
        void invalidate() { entries.clear(); frame = UINT64_MAX; }
    };
    mutable UniqueListCache mUniqueListCache;

    // findPropertyFlex cache: avoids repeated string comparisons.
    // Key = (classId << 32) | hash(name), Value = resolved PropertyDescription.
    // This is frame-independent (properties don't change), so never invalidated.
    mutable std::unordered_map<uint64_t, Option<PropertyDescription>> mPropFlexCache;

    // findSampleAtFrame hint cache: remembers last found position per attribute list.
    // Persists across frame changes for efficient sequential scrubbing:
    // - Same frame: return cached result directly (avoids double-walk for byteSize+getValue)
    // - Forward scrub (new frame > cached frame): start from cached position and advance
    // - Backward scrub (new frame < cached frame): must walk from the beginning
    struct SampleHint
    {
        PxU64 frame;
        OmniPvdAttributeInst* result;
    };
    mutable std::unordered_map<const void*, SampleHint> mSampleHintCache;

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    bool buildIndex();
    void collectAllProperties(PxI32 classId, std::vector<PropertyDescription>& out) const;

    // Find the latest OmniPvdAttributeInst at or before 'frame' in 'list'.
    // Returns nullptr if no sample exists at or before that frame.
    // Uses mSampleHintCache for fast sequential access.
    OmniPvdAttributeInst* findSampleAtFrame(OmniPvdAttributeInstList* list,
                                            PxU64 frame) const;

    // Returns true if 'obj' has a life span that covers 'frame', or if the
    // object carries no life-span information (treated as always alive).
    static bool isAliveAtFrame(const OmniPvdObject* obj, PxU64 frame);

    // Returns the OmniPvdAttributeInstList for attribute index 'propId' within
    // 'obj', or nullptr if not found.  propId must be a valid encoded id
    // produced by this adapter (classIndex << 16 | attrIndex).
    OmniPvdAttributeInstList* getAttrList(OmniPvdObject* obj, PxI32 propId) const;

    // Resolve a key previously published as InstanceDescription::mInstPtr.
    // This path never consults external PhysX object handles.
    PxI32 resolvePublishedInstanceKey(PxU64 internalKey) const;

    // Resolve an external PhysX object reference, using the referrer to make
    // segment-consistent picks when one pointer maps to several DOM objects.
    // referrerIdx -1 = no referrer (newest wins).
    PxI32 resolveExternalReferenceAt(PxU64 apiHandle, PxI32 referrerIdx) const;

    // Returns the accumulated unique-list elements for (objIdx, propId) at
    // mCurrentFrame. Uses mUniqueListCache; the first call per (obj, prop, frame)
    // walks the linked list, subsequent calls are O(1).
    const std::vector<OmniPvdUniqueListElement*>& getAccumulatedUniqueList(
        PxI32 objIdx, PxI32 propId, OmniPvdAttributeInstList* attrList) const;
};

} // namespace pvdapp

#endif // PVD_OVD_SUPPORT
#endif // PVD_OVD_DOM_ADAPTER_H
