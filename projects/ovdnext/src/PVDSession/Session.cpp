/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "Session.h"
#include "Presentation.h"
#include "../PVDCore/LoadControl.h"
#include "../PVDCore/PvdCheckedOutput.h"

#include "PvdEngine.h"
#include "fakeinstanceheader.h"
#include "PvdSimpleAllocator.h"
#include "PxErrorCallback.h"
#include "PxFoundation.h"
#include "PxFoundationVersion.h"
#include "OmniPvdWriteStream.h"

#include <filesystem>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <cmath>
#include <mutex>
#include <chrono>
#include <stdexcept>
#include <unordered_map>

namespace ovdnext {
namespace {
using namespace pvdapp;

struct ErrorCallback : physx::PxErrorCallback {
    void reportError(physx::PxErrorCode::Enum, const char*, const char*, int) override {}
};
struct Runtime {
    CheckingAllocator allocator;
    ErrorCallback errors;
    physx::PxFoundation* foundation;
    Runtime() : foundation(PxCreateFoundation(PX_FOUNDATION_VERSION, allocator, errors)) {
        if (!foundation) throw std::runtime_error("Unable to initialize PhysX foundation");
    }
    ~Runtime() { foundation->release(); }
};

std::shared_ptr<Runtime> acquireRuntime() {
    // Retain one foundation for the lifetime of the loaded native module.
    // Per-session weak ownership would race last-session destruction against
    // a simultaneous new session's PxCreateFoundation call.
    static const auto runtime = std::make_shared<Runtime>();
    return runtime;
}

std::string safeString(const char* value) { return value ? value : ""; }

class CheckedOutputWriteStream final : public OmniPvdWriteStream {
public:
    explicit CheckedOutputWriteStream(OutputFile& output) : mOutput(output) {}
    uint64_t OMNI_PVD_CALL writeBytes(const uint8_t* bytes,uint64_t count) override {
        if(!mOpen || count>std::numeric_limits<uint32_t>::max())return 0;
        return mOutput.write(bytes,static_cast<uint32_t>(count))?count:0;
    }
    bool OMNI_PVD_CALL flush() override { return mOpen && !mOutput.flush().hasValue(); }
    bool OMNI_PVD_CALL openStream() override { mOpen=true;return true; }
    bool OMNI_PVD_CALL closeStream() override {
        if(!mOpen)return true;
        const bool ok=flush();mOpen=false;return ok;
    }
private:
    OutputFile& mOutput;
    bool mOpen=false;
};

bool sameFile(const std::string& first,const std::string& second) {
    if(first.empty() || second.empty())return false;
    namespace fs = std::filesystem;
    const auto a = fs::absolute(fs::u8path(first)).lexically_normal();
    const auto b = fs::absolute(fs::u8path(second)).lexically_normal();
    if(a == b) return true;
    std::error_code error;
    // Resolve symlinks before '..' using the original filesystem paths.
    const bool equivalent = fs::equivalent(fs::u8path(first), fs::u8path(second), error);
    if(!error) return equivalent;
    if(error == std::errc::no_such_file_or_directory) return false;
    throw fs::filesystem_error("Cannot check capture file identity", a, b, error);
}
}

struct Session::Impl {
    std::shared_ptr<Runtime> runtime = acquireRuntime();
    std::unique_ptr<pvdapp::PvdEngine> engine;
    std::recursive_mutex mutex;
    CaptureRequestState captureRequests;
    uint64_t captureForwardBytes=0;
    std::chrono::steady_clock::time_point captureForwardAt=std::chrono::steady_clock::now();
    std::string sourcePath,temporaryDirectory;
    std::shared_ptr<CaptureIdentity> captureIdentity=std::make_shared<CaptureIdentity>();
    std::shared_ptr<PresentationData> lastPresentation;
    std::shared_ptr<PresentationSchema> presentationSchema;
    std::vector<uint8_t> presentationSchemaSignature;
    uint64_t presentationGeneration=0;
    std::vector<uint8_t> hierarchyContent;
    uint64_t hierarchyRevision=0,indexSerial=0,hierarchyIndexedSerial=~uint64_t(0);
    HierarchyChanges hierarchyDelta;
    std::unordered_map<int32_t,std::vector<uint8_t>> hierarchyInstances;
    std::vector<uint8_t> hierarchySchema,hierarchyQueries;
    GeometryExtractor geometry;
    std::shared_ptr<GeometrySnapshot> lastGeometry = std::make_shared<GeometrySnapshot>();
    uint64_t lastLiveIndexedSerial = 0;
    std::shared_ptr<OverlaySnapshot> lastOverlay = std::make_shared<OverlaySnapshot>();
    std::vector<pvdapp::InstanceDescription> instances;
    std::unordered_map<int32_t, size_t> instanceIndex;
    std::unordered_map<int32_t, std::vector<pvdapp::PropertyDescription>> propertyCache;

    ExportCapabilities capabilities() {
        ExportCapabilities result;result.format=engine->isOvdFile()?"OVD":"PXD2";
        const bool active=engine->isLive() && !engine->isLiveEnded();
        const std::string source=engine->isLive()?engine->getLiveRecordingPath():sourcePath;
        const bool retained=engine->isLive() && engine->isOvdFile() && engine->isLiveEnded()
            && engine->getLiveErrorCode()==OvdInputErrorCode::None && engine->getLiveRecordingSize()>0;
        result.save=!active && (!engine->isOvdFile() || !source.empty() || retained);
        if(!result.save) result.saveReason=active?"Detach the live producer before saving its recording":"This live capture has no recording file";
        result.clip=!engine->isOvdFile() && engine->getEventStream();
        if(!result.clip) result.clipReason="Frame cropping is implemented for native PXD2 event streams only";
        result.repx=!engine->isOvdFile() && engine->canExportRepX();
        if(!result.repx) result.repxReason=engine->isOvdFile()?"The original RepX exporter supports PXD2 only":"No PhysX scene is available at this frame";
        return result;
    }
    void index() {
        ++indexSerial;
        instances.resize(engine->getNbInstances());
        instances.resize(engine->getInstances(instances.data(), instances.size()));
        instanceIndex.clear();
        for (size_t i = 0; i < instances.size(); ++i) instanceIndex[instances[i].mId] = i;
        propertyCache.clear();
    }
    int32_t sceneInstanceId(uint64_t streamId) {
        for (const auto& i : instances) {
            const auto cls=engine->getClass(i.mClassId);
            if(cls.hasValue() && safeString(cls.getValue().mName.mName)=="PxScene" && engine->getStreamId(i.mId)==streamId)
                return i.mId;
        }
        return -1;
    }
    std::vector<SceneInfo> sceneInfos() {
        std::vector<SceneInfo> result;
        if(engine->isOvdFile()) {
            // The DOM engine shares one OVD timeline across scene creations.
            // Expose each creation's stable identity instead of collapsing
            // reused writer handles into that timeline's first scene.
            for(const auto& i:instances) {
                const auto cls=engine->getClass(i.mClassId);
                if(!cls.hasValue() || safeString(cls.getValue().mName.mName)!="PxScene") continue;
                const auto stream=engine->getStreamId(i.mId);
                std::string name;
                const auto nameProp=engine->findPropertyFlex(i.mClassId,"Name");
                if(nameProp.hasValue() && i.mAlive) {
                    uint32_t handle=0;
                    if(engine->getPropertyValue(i.mInstPtr,nameProp.getValue().mPropertyId,&handle,sizeof(handle))==sizeof(handle))
                        name=safeString(engine->handleToString(handle));
                }
                result.push_back({uint64_t(i.mId),name,engine->getFrameCount(stream),engine->getCurrentFrame(stream),i.mId,stream,i.mAlive,
                    engine->getOvdRecordingSegment(i.mId)});
            }
        } else {
            std::vector<pvdapp::SectionListInfo> native(engine->getNbScenes());
            native.resize(engine->getScenes(native.data(),native.size()));
            for(const auto& scene:native) result.push_back({scene.mStreamId,safeString(engine->handleToString(scene.mName)),
                engine->getFrameCount(scene.mStreamId),engine->getCurrentFrame(scene.mStreamId),sceneInstanceId(scene.mStreamId),scene.mStreamId,true});
        }
        return result;
    }
    pvdapp::InstanceDescription& instance(int32_t id) {
        auto it = instanceIndex.find(id);
        if (it == instanceIndex.end()) throw std::out_of_range("Unknown instance ID");
        auto& result = instances[it->second];
        if (!result.mAlive || !result.mInstPtr) throw std::out_of_range("Instance is not alive at this frame");
        return result;
    }
    const std::vector<pvdapp::PropertyDescription>& descriptors(int32_t classId) {
        auto found = propertyCache.find(classId);
        if (found != propertyCache.end()) return found->second;
        if (!engine->getClass(classId).hasValue()) throw std::out_of_range("Unknown class ID");
        std::vector<pvdapp::PropertyDescription> result(engine->getNbProperties(classId));
        result.resize(engine->getProperties(classId, result.data(), result.size()));
        return propertyCache.emplace(classId, std::move(result)).first->second;
    }
    pvdapp::DataRef<uint8_t> value(int32_t id, int32_t propertyId) {
        auto& inst = instance(id);
        const auto& props = descriptors(inst.mClassId);
        if (std::none_of(props.begin(), props.end(), [propertyId](const pvdapp::PropertyDescription& p) {
            return p.mPropertyId == propertyId;
        })) throw std::out_of_range("Property does not belong to this instance");
        return engine->getRawPropertyValue(inst.mInstPtr, propertyId);
    }
};

Session::Session(const std::string& path, const std::string& temporaryDirectory,
    std::shared_ptr<pvdapp::LoadControl> control) : m(new Impl) {
    if(control && control->cancelled())throw std::runtime_error("Capture load cancelled");
    std::ifstream input(std::filesystem::u8path(path), std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open capture: " + path);
    OvdInputErrorCode error = OvdInputErrorCode::None;
    m->engine.reset(pvdapp::PvdEngine::load(m->runtime->allocator, temporaryDirectory.c_str(), path.c_str(), &error, control.get()));
    if(control && control->cancelled())throw std::runtime_error("Capture load cancelled");
    if (!m->engine || error != OvdInputErrorCode::None)
        throw std::runtime_error("Capture load failed: " + std::string(ovdInputErrorMessage(error)));
    m->sourcePath=path;m->temporaryDirectory=temporaryDirectory;
    if(control)control->update("Indexing capture",0,0,false);
    m->index();
    if(control && control->cancelled())throw std::runtime_error("Capture load cancelled");
    if(control)control->update("Ready",1,1);
}

Session::Session(uint16_t port, const std::string& recordingPath) : m(new Impl) {
    if (!port) throw std::invalid_argument("Live port must be between 1 and 65535");
    m->engine.reset(pvdapp::PvdEngine::connectLive(m->runtime->allocator, port,
        recordingPath.empty() ? nullptr : recordingPath.c_str()));
    if (!m->engine) throw std::runtime_error("Unable to create live session or exclusive recording");
    m->index();
}
Session::~Session() = default;
void Session::withEngineLocked(const std::function<void(pvdapp::PvdEngine&)>& callback) {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);pvdapp::PvdEngine::LiveReadGuard guard(*m->engine);callback(*m->engine);
}
std::shared_ptr<Session> Session::listen(uint16_t port, const std::string& path) {
    return std::shared_ptr<Session>(new Session(port, path));
}
uint64_t Session::captureKey() {return m->captureIdentity->key;}
EvidenceMetadataPage Session::evidenceMetadata(uint32_t offset,uint32_t limit) {
    if(!limit || limit>512)throw std::length_error("Evidence page limit must be 1..512");
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    auto& e=*m->engine;pvdapp::PvdEngine::LiveReadGuard guard(e);
    EvidenceMetadataPage result;result.captureKey=captureKey();result.segmentEpoch=e.getOvdSegmentEpoch();
    result.domain=e.ovdEvidenceDomain();result.total=m->instances.size();
    uint64_t bytes=pvdapp::evidenceBytes(result.domain);
    if(bytes>pvdapp::OvdEvidencePageBytes)throw std::length_error("Evidence page exceeds 8 MiB");
    result.nextOffset=std::min(result.total,uint64_t(offset)+limit);
    for(uint64_t i=offset;i<result.nextOffset;++i) {
        auto identity=e.ovdEvidenceIdentity(m->instances[size_t(i)].mId);
        bytes+=pvdapp::evidenceBytes(identity);
        if(bytes>pvdapp::OvdEvidencePageBytes)throw std::length_error("Evidence page exceeds 8 MiB");
        result.rows.push_back(std::move(identity));
    }
    return result;
}
std::shared_ptr<Presentation> Session::presentation(uint64_t sceneId,int64_t frame,const std::vector<int32_t>& inspectIds,
    const std::vector<ReadRequest>& readRequests,const std::vector<RecordRequest>& recordRequests,
    uint32_t overlayMask,float gizmoScale,float toleranceSpeed,bool allowUnavailableRequests,bool capturedToleranceSpeed,const JointScaleRequest& jointScaling,const std::map<uint32_t,float>& familyScales,float masterScale,bool includeEvidence) {
    if(frame < -1 || frame > std::numeric_limits<uint32_t>::max())throw std::out_of_range("Presentation frame is outside uint32 range");
    if(inspectIds.size()>64 || readRequests.size()>1024 || recordRequests.size()>1024)
        throw std::length_error("Presentation requests exceed inspector bounds (64 objects, 1024 windows/records)");
    if(overlayMask&~uint32_t(OverlayAll) || !std::isfinite(gizmoScale) || gizmoScale<=0 || !std::isfinite(toleranceSpeed) || toleranceSpeed<=0)
        throw std::invalid_argument("Invalid presentation overlay settings");
    validateFamilyScales(familyScales);
    for(const auto& request:readRequests)if(request.limit>8*1024*1024)throw std::length_error("Presentation read window exceeds 8 MiB");
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    auto& e=*m->engine;pvdapp::PvdEngine::LiveReadGuard guard(e);
    validateJointScaleRequest(jointScaling,e.isOvdFile());
    validateMasterScale(masterScale,gizmoScale,jointScaling,e.isOvdFile(),familyScales);
    auto deferred=[&] {
        auto data=m->lastPresentation;
        if(!data) {data=std::make_shared<PresentationData>();data->owner=m->captureIdentity;data->info.ovd=e.isOvdFile();data->info.live=e.isLive();}
        return std::make_shared<Presentation>(data,false);
    };
    if(e.hasLiveUnpublishedIndexChanges() && !e.isLiveEnded())return deferred();
    if(e.isLive()) {
        const auto update=refreshLive();
        if(!update.ready && !e.isLiveEnded())return deferred();
        // A stopped source with an unpublished tail can scrub its previously
        // indexed history. It cannot invent an initial published adapter.
        if(!m->lastLiveIndexedSerial)return deferred();
    }
    const auto before=m->sceneInfos();
    if(before.empty())return deferred();
    if(!sceneId) {const auto alive=std::find_if(before.begin(),before.end(),[](const SceneInfo& s){return s.alive;});sceneId=(alive==before.end()?before.front():*alive).id;}
    const auto selected=std::find_if(before.begin(),before.end(),[sceneId](const SceneInfo& s){return s.id==sceneId;});
    if(selected==before.end())throw std::out_of_range("Unknown presentation scene ID");
    const uint32_t target=frame<0?selected->currentFrame:uint32_t(frame);
    if(target>=selected->frameCount)throw std::out_of_range("Presentation frame is outside this scene");
    const uint32_t previousFrame=selected->currentFrame;
    const uint64_t streamId=selected->streamId;
    const auto started=std::chrono::steady_clock::now();
    try {
        if(target!=previousFrame && !seek(sceneId,target))return deferred();
        hierarchyRevision();
        auto data=std::make_shared<PresentationData>();data->owner=m->captureIdentity;
        data->sceneId=sceneId;data->frame=target;data->info=info();data->scenes=scenes();
        data->serial=e.isLive()?m->lastLiveIndexedSerial:data->info.completedSerial;
        data->hierarchy=m->hierarchyDelta;
        if(!m->presentationSchema || m->presentationSchemaSignature!=m->hierarchySchema) {
            auto schema=std::make_shared<PresentationSchema>();schema->classes=classes();
            for(const auto& cls:schema->classes) {
                auto props=properties(cls.id);
                for(const auto& prop:props)if(schema->namedValues.find(prop.id)==schema->namedValues.end()) {
                    auto& values=schema->namedValues[prop.id];
                    const auto native=e.getNamedPropertyValues(prop.id);values.reserve(native.size());
                    for(const auto& value:native)values.emplace_back(safeString(value.mName),value.mValue);
                }
                schema->properties.emplace(cls.id,std::move(props));
            }
            m->presentationSchema=std::move(schema);m->presentationSchemaSignature=m->hierarchySchema;
        }
        data->schema=m->presentationSchema;
        for(const auto& native:m->instances) {
            data->instanceIndex[native.mId]=data->instances.size();
            data->instances.push_back({native.mId,native.mClassId,e.getOwningSceneId(native.mId),e.getOwningPhysicsId(native.mId),native.mAlive,e.isOvdSceneJoint(native.mId)});
        }
        data->info.instanceCount=uint32_t(data->instances.size());
        constexpr uint64_t MaxCopiedBytes=128*1024*1024;
        auto account=[&](uint64_t count){if(data->copiedBytes+count>MaxCopiedBytes)throw std::length_error("Presentation metadata/inspector data exceeds 128 MiB");data->copiedBytes+=count;};
        auto ensureValue=[&](int32_t id,int32_t prop)->PresentationValue& {
            const auto key=Presentation::valueKey(id,prop);const auto found=data->values.find(key);
            if(found!=data->values.end())return found->second;
            PresentationValue value;value.size=propertySize(id,prop);
            value.samplePresent=!data->info.ovd || e.hasPropertySample(m->instance(id).mInstPtr,prop);
            return data->values.emplace(key,std::move(value)).first->second;
        };
        auto addWindow=[&](int32_t id,int32_t prop,uint32_t offset,uint32_t limit) {
            auto& value=ensureValue(id,prop);const uint64_t end=std::min(uint64_t(value.size),uint64_t(offset)+limit);
            if(offset>=end)return;
            for(const auto& window:value.windows)if(offset>=window.offset && end<=uint64_t(window.offset)+window.bytes.size())return;
            account(end-offset);value.windows.push_back({offset,read(id,prop,offset,uint32_t(end-offset))});
        };
        // Derive the small metadata projection once per class. The full schema
        // can contain dozens of scalar fields per actor; those belong only to
        // explicitly inspected objects and need no repeated string dispatch.
        struct MetadataProperty {const PropertyInfo* property;uint8_t flags;};
        enum {Reference=1,Name=2,CameraField=4,SceneArray=8};
        std::unordered_map<int32_t,std::vector<MetadataProperty>> metadata;
        for(const auto& cls:data->schema->classes) {
            auto& fields=metadata[cls.id];
            const bool scene=cls.name=="PxScene",camera=cls.name=="Camera",physics=cls.name=="PxPhysics";
            for(const auto& prop:data->schema->properties.at(cls.id)) {
                uint8_t flags=0;
                if(prop.type=="ObjectRef")flags|=Reference;
                else if((prop.name=="name" || prop.name=="Name") && prop.type=="StringHandle")flags|=Name;
                else if(scene && prop.array)flags|=SceneArray;
                if((camera && (prop.name=="Origin" || prop.name=="Target" || prop.name=="Up")) ||
                   (scene && (prop.name=="gravity" || prop.name=="Gravity")) ||
                   ((scene || physics) && (prop.name=="tolerancesScale" || prop.name=="TolerancesScale.Length" || prop.name=="TolerancesScale.Speed")))flags|=CameraField;
                if(flags)fields.push_back({&prop,flags});
            }
        }
        // Only this projection uses validated reads: the current hierarchy
        // signature supplied the schema, and these native instances belong to
        // the index protected by this publication's mutex and LiveReadGuard.
        // Do not reuse either across frames/epochs or bypass public validation.
        auto metadataValue=[&](const pvdapp::InstanceDescription& instance,int32_t prop)->PresentationValue& {
            const auto key=Presentation::valueKey(instance.mId,prop);const auto found=data->values.find(key);
            if(found!=data->values.end())return found->second;
            // Raw bytes and converted reference-list bytes can have different
            // sizes (notably OVD unique lists). Preserve the raw size contract.
            PresentationValue value;value.size=e.getRawPropertyValue(instance.mInstPtr,prop).size();
            value.samplePresent=!data->info.ovd || e.hasPropertySample(instance.mInstPtr,prop);
            return data->values.emplace(key,std::move(value)).first->second;
        };
        auto metadataWindow=[&](const pvdapp::InstanceDescription& instance,int32_t prop,uint32_t limit) {
            auto& value=metadataValue(instance,prop);const uint32_t end=std::min(value.size,limit);
            if(!end)return;
            for(const auto& window:value.windows)if(!window.offset && end<=window.bytes.size())return;
            account(end);
            const auto raw=e.getRawPropertyValue(instance.mInstPtr,prop);
            const size_t count=std::min(size_t(end),size_t(raw.size()));
            std::vector<uint8_t> bytes;if(count)bytes.assign(raw.begin(),raw.begin()+count);
            value.windows.push_back({0,std::move(bytes)});
        };
        for(const auto& instance:m->instances) {
            if(!data->instances.at(data->instanceIndex.at(instance.mId)).alive)continue;
            const auto& fields=metadata.at(instance.mClassId);
            if(!fields.empty() && !instance.mInstPtr)throw std::out_of_range("Instance is not alive at this frame");
            for(const auto& field:fields) {
                const auto& prop=*field.property;
                if(field.flags&CameraField)metadataWindow(instance,prop.id,64);
                if(field.flags&Reference) {
                    auto& value=metadataValue(instance,prop.id);
                    const auto size=e.getPropertyByteSize(instance.mInstPtr,prop.id);
                    if(size>8*1024*1024)throw std::length_error("Reference list exceeds bounded transfer size");
                    if(size) {
                        value.references.resize((size+sizeof(int32_t)-1)/sizeof(int32_t));
                        const auto bytes=e.getPropertyValue(instance.mInstPtr,prop.id,value.references.data(),value.references.size()*sizeof(int32_t));
                        value.references.resize(bytes/sizeof(int32_t));
                    }
                    value.hasReferences=true;
                    account(value.references.size()*sizeof(int32_t));
                } else if(field.flags&Name) {
                    auto& value=metadataValue(instance,prop.id);uint32_t handle=0;
                    const auto size=e.getPropertyValue(instance.mInstPtr,prop.id,&handle,sizeof(handle));
                    value.string=size==sizeof(handle)?safeString(e.handleToString(handle)):"";
                    value.hasString=true;account(value.string.size());
                    metadataWindow(instance,prop.id,256);
                } else if(field.flags&SceneArray)metadataValue(instance,prop.id);
            }
        }
        data->queries=queryRecords();
        account(data->queries.size()*sizeof(QueryRecord));
        std::vector<RecordRequest> records=recordRequests;
        for(const int32_t id:inspectIds) {
            if(id<0) {
                const auto query=std::find_if(data->queries.begin(),data->queries.end(),[id](const QueryRecord& q){return q.queryId==id;});
                if(query!=data->queries.end()){records.push_back({query->sceneInstanceId,query->propertyId,query->elementIndex});data->inspectedIds.push_back(id);}
                else data->unavailableIds.push_back(id);
                continue;
            }
            const auto index=data->instanceIndex.find(id);
            if(index==data->instanceIndex.end() || !data->instances[index->second].alive){data->unavailableIds.push_back(id);continue;}
            const auto& instance=data->instances[index->second];data->inspectedIds.push_back(id);
            for(const auto& prop:data->schema->properties.at(instance.classId)) {
                auto& value=ensureValue(id,prop.id);
                if(prop.type=="StringHandle" && !value.hasString) {value.string=stringValue(id,prop.id);value.hasString=true;account(value.string.size());}
                // Native arrays are copied only as a bounded first-page preview;
                // later windows are explicit requests, never delayed live reads.
                addWindow(id,prop.id,0,65536);
            }
        }
        auto propertyExists=[&](int32_t id,int32_t propertyId) {
            const auto found=data->instanceIndex.find(id);
            if(found==data->instanceIndex.end() || !data->instances[found->second].alive)return false;
            const auto& props=data->schema->properties.at(data->instances[found->second].classId);
            return std::any_of(props.begin(),props.end(),[propertyId](const PropertyInfo& p){return p.id==propertyId;});
        };
        for(const auto& request:readRequests) {
            if(allowUnavailableRequests && !propertyExists(request.instanceId,request.propertyId)) {
                data->unavailableReads.push_back(request);continue;
            }
            addWindow(request.instanceId,request.propertyId,request.offset,request.limit);
            if(allowUnavailableRequests && request.limit && request.offset>=ensureValue(request.instanceId,request.propertyId).size)
                data->unavailableReads.push_back(request);
        }
        for(const auto& request:records) {
            const auto key=std::make_tuple(request.instanceId,request.propertyId,request.index);
            if(data->records.find(key)!=data->records.end())continue;
            if(allowUnavailableRequests && !propertyExists(request.instanceId,request.propertyId)) {
                data->unavailableRecords.push_back(request);continue;
            }
            RecordInfo record;
            try {record=arrayRecord(request.instanceId,request.propertyId,request.index);}
            catch(const std::out_of_range&) {
                if(!allowUnavailableRequests)throw;
                data->unavailableRecords.push_back(request);continue;
            }
            uint32_t stride=0;for(const auto& cls:data->schema->classes)if(cls.id==record.classId){stride=cls.byteSize;break;}
            const uint64_t offset=uint64_t(request.index)*stride;if(offset>std::numeric_limits<uint32_t>::max())throw std::length_error("Record byte offset exceeds uint32");
            addWindow(request.instanceId,request.propertyId,uint32_t(offset),stride);
            for(const auto& field:record.fields)account(field.data.size()+field.stringValue.size());
            data->records.emplace(key,std::move(record));
        }
        if(includeEvidence) {
            auto evidenceAccount=[&](uint64_t bytes) {
                if(bytes>MaxCopiedBytes-data->copiedBytes)throw std::length_error("Presentation evidence exceeds 128 MiB");
                data->copiedBytes+=bytes;
            };
            data->evidenceDomain=e.ovdEvidenceDomain();evidenceAccount(pvdapp::evidenceBytes(data->evidenceDomain));
            auto captureIdentity=[&](int32_t id) {
                if(data->evidenceIdentities.count(id))return;
                const auto found=data->instanceIndex.find(id);
                if(found==data->instanceIndex.end() || !data->instances[found->second].alive)return;
                auto identity=e.ovdEvidenceIdentity(id);evidenceAccount(pvdapp::evidenceBytes(identity));
                data->evidenceIdentities.emplace(id,std::move(identity));
            };
            captureIdentity(selected->instanceId);
            for(const auto id:data->inspectedIds)captureIdentity(id);
            for(const auto& request:readRequests) {
                if(!propertyExists(request.instanceId,request.propertyId))continue;
                auto stamp=e.ovdEvidenceSampleStamp(m->instance(request.instanceId).mInstPtr,request.propertyId,target);
                evidenceAccount(sizeof(stamp)+sizeof(uint64_t));
                data->evidenceStamps[Presentation::valueKey(request.instanceId,request.propertyId)]=stamp;
            }
        }
        data->toleranceSpeed=toleranceSpeed;
        data->jointScaling=jointScaling;
        const auto current=std::find_if(data->scenes.begin(),data->scenes.end(),[sceneId](const SceneInfo& s){return s.id==sceneId;});
        resolveCapturedTolerances(*data,current==data->scenes.end()?-1:current->instanceId,capturedToleranceSpeed);
        data->geometry=geometrySnapshot();
        if(overlayMask)data->overlays=overlaySnapshot(overlayMask,gizmoScale,data->toleranceSpeed,jointScaling,familyScales,masterScale);
        data->generation=++m->presentationGeneration;data->ready=true;
        data->extractionMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        m->lastPresentation=data;
        return std::make_shared<Presentation>(data,true);
    } catch(...) {
        if(target!=previousFrame){e.setCurrentFrame(previousFrame,streamId);m->index();}
        throw;
    }
}
SessionInfo Session::info() {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    auto& e = *m->engine;
    pvdapp::PvdEngine::LiveReadGuard guard(e);
    const auto error = e.getLiveErrorCode();
    return {e.isOvdFile(), e.getNbInstances(), e.getMaxFrame(), e.isLive(), e.isLiveConnected(),
        e.isLiveEnded(), e.getLiveCompletedFrameSerial(), e.getOvdSegmentEpoch(),
        error == OvdInputErrorCode::None ? "" : ovdInputErrorMessage(error),
        e.hasLiveUnpublishedIndexChanges(), e.getLiveMaxFrame(), e.getLiveCurrentFrame(),
        e.getLiveRecordingSize()};
}
std::shared_ptr<GeometrySnapshot> Session::geometrySnapshot() {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    pvdapp::PvdEngine::LiveReadGuard guard(*m->engine);
    if(m->engine->hasLiveUnpublishedIndexChanges() && !m->engine->isLiveEnded()) return m->lastGeometry;
    m->lastGeometry = m->geometry.extract(*m->engine);
    return m->lastGeometry;
}
std::shared_ptr<OverlaySnapshot> Session::overlaySnapshot(uint32_t mask,float gizmoScale,float toleranceSpeed,const JointScaleRequest& jointScaling,const std::map<uint32_t,float>& familyScales,float masterScale) {
    validateFamilyScales(familyScales);
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    pvdapp::PvdEngine::LiveReadGuard guard(*m->engine);
    validateJointScaleRequest(jointScaling,m->engine->isOvdFile());
    validateMasterScale(masterScale,gizmoScale,jointScaling,m->engine->isOvdFile(),familyScales);
    if(m->engine->hasLiveUnpublishedIndexChanges() && !m->engine->isLiveEnded()) return m->lastOverlay;
    m->lastOverlay=extractOverlays(*m->engine,mask,gizmoScale,toleranceSpeed,jointScaling,familyScales,masterScale);
    if(!m->lastOverlay->queryRequests.empty()) m->lastOverlay->queryGeometry=m->geometry.extractReferences(*m->engine,m->lastOverlay->queryRequests);
    return m->lastOverlay;
}
uint64_t Session::hierarchyRevision() {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    auto& e=*m->engine;pvdapp::PvdEngine::LiveReadGuard guard(e);
    if(e.hasLiveUnpublishedIndexChanges() && !e.isLiveEnded()) return m->hierarchyRevision;
    if(m->hierarchyIndexedSerial==m->indexSerial) return m->hierarchyRevision;
    std::vector<uint8_t> bytes;bytes.reserve(m->hierarchyContent.size());
    auto append=[&](const void* data,size_t size) {
        if(bytes.size()+size>128*1024*1024)throw std::length_error("Hierarchy revision content exceeds 128 MiB");
        if(size)bytes.insert(bytes.end(),static_cast<const uint8_t*>(data),static_cast<const uint8_t*>(data)+size);
    };
    auto number=[&](uint64_t value){append(&value,sizeof(value));};
    auto text=[&](const char* value){const size_t size=value?std::strlen(value):0;number(size);append(value,size);};
    number(e.getOvdSegmentEpoch());
    std::vector<ClassDescription> classes(e.getNbClasses());classes.resize(e.getClasses(classes.data(),classes.size()));
    number(classes.size());
    std::unordered_map<int32_t,std::vector<PropertyDescription>> relevant;
    for(const auto& cls:classes) {
        number(cls.mClassId);number(cls.mBaseClass);number(cls.get32BitSize());text(cls.mName.mNamespace);text(cls.mName.mName);
        const auto& props=m->descriptors(cls.mClassId);number(props.size());
        const bool scene=safeString(cls.mName.mName)=="PxScene";
        for(const auto& p:props) {
            number(p.mPropertyId);number(p.mDatatype);number(p.mPropertyType);text(p.mName);text(p.mSemantic);
            text(p.mDatatypeName.mName);number(e.getOvdPropertyFieldCount(p.mPropertyId));
            const std::string name=safeString(p.mName),type=safeString(p.mDatatypeName.mName);
            if(type=="ObjectRef" || ((name=="Name" || name=="name") && type=="StringHandle") ||
               (scene && p.mPropertyType==PropertyType::Array)) relevant[cls.mClassId].push_back(p);
        }
    }
    const std::vector<uint8_t> schema=bytes;
    std::unordered_map<int32_t,std::vector<uint8_t>> instanceBytes;
    for(const auto& i:m->instances) {
        if(!i.mAlive || !i.mInstPtr)continue;
        const auto start=bytes.size();
        number(i.mId);number(i.mClassId);number(e.getOwningSceneId(i.mId));number(e.getOwningPhysicsId(i.mId));
        const auto found=relevant.find(i.mClassId);if(found==relevant.end()){number(0);instanceBytes[i.mId]=std::vector<uint8_t>(bytes.begin()+start,bytes.end());continue;}
        number(found->second.size());
        for(const auto& p:found->second) {
            number(p.mPropertyId);
            const auto size=e.getPropertyByteSize(i.mInstPtr,p.mPropertyId);number(size);
            if(safeString(p.mDatatypeName.mName)=="ObjectRef") {
                if(size>8*1024*1024)throw std::length_error("Hierarchy reference list exceeds 8 MiB");
                std::vector<uint8_t> refs(size);if(size)refs.resize(e.getPropertyValue(i.mInstPtr,p.mPropertyId,refs.data(),size));
                number(refs.size());append(refs.data(),refs.size());
            } else if(safeString(p.mDatatypeName.mName)=="StringHandle") {
                uint32_t handle=0;e.getPropertyValue(i.mInstPtr,p.mPropertyId,&handle,sizeof(handle));text(e.handleToString(handle));
            }
        }
        instanceBytes[i.mId]=std::vector<uint8_t>(bytes.begin()+start,bytes.end());
    }
    const auto queryStart=bytes.size();
    const auto records=extractQueryRecords(e);number(records.size());
    for(const auto& r:records) {number(r.queryId);number(r.sceneInstanceId);number(r.propertyId);number(r.elementIndex);number(r.elementSize);number(r.classId);}
    // Compare the actual length-delimited content. A hash collision can never
    // hide a reference/name/lifetime/schema change from the tree cache.
    if(bytes!=m->hierarchyContent || !m->hierarchyRevision) {
        HierarchyChanges delta;delta.revision=++m->hierarchyRevision;delta.schemaChanged=schema!=m->hierarchySchema;
        const std::vector<uint8_t> queries(bytes.begin()+queryStart,bytes.end());delta.queryRecordsChanged=queries!=m->hierarchyQueries;
        for(const auto& item:instanceBytes) {const auto old=m->hierarchyInstances.find(item.first);if(old==m->hierarchyInstances.end() || item.second!=old->second)delta.changedIds.push_back(item.first);}
        for(const auto& item:m->hierarchyInstances)if(instanceBytes.find(item.first)==instanceBytes.end())delta.removedIds.push_back(item.first);
        std::sort(delta.changedIds.begin(),delta.changedIds.end());std::sort(delta.removedIds.begin(),delta.removedIds.end());
        m->hierarchyDelta=std::move(delta);m->hierarchySchema=schema;m->hierarchyQueries=queries;m->hierarchyInstances=std::move(instanceBytes);
        m->hierarchyContent=std::move(bytes);
    }
    m->hierarchyIndexedSerial=m->indexSerial;
    return m->hierarchyRevision;
}
HierarchyChanges Session::hierarchyChanges(uint64_t previousRevision) {
    hierarchyRevision();
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    if(previousRevision==m->hierarchyRevision) {HierarchyChanges result;result.revision=m->hierarchyRevision;return result;}
    auto result=m->hierarchyDelta;
    if(previousRevision+1!=m->hierarchyRevision) result.schemaChanged=true; // caller must rebuild after a skipped delta
    return result;
}
std::vector<QueryRecord> Session::queryRecords() {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    pvdapp::PvdEngine::LiveReadGuard guard(*m->engine);
    return extractQueryRecords(*m->engine);
}
RecordInfo Session::arrayRecord(int32_t instanceId,int32_t propertyId,uint32_t elementIndex) {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    auto& e=*m->engine;
    pvdapp::PvdEngine::LiveReadGuard guard(e);
    if(e.isOvdFile()) throw std::invalid_argument("Typed structure records are supported for PXD2 arrays only");
    const auto& inst=m->instance(instanceId);
    const auto& props=m->descriptors(inst.mClassId);
    const auto prop=std::find_if(props.begin(),props.end(),[propertyId](const PropertyDescription& p){return p.mPropertyId==propertyId;});
    if(prop==props.end() || prop->mPropertyType!=PropertyType::Array) throw std::invalid_argument("Property is not a structure array on this instance");
    const auto cls=e.getClass(prop->mDatatype);
    if(!cls.hasValue() || !cls.getValue().get32BitSize()) throw std::invalid_argument("Array element has no native structure layout");
    const uint32_t stride=cls.getValue().get32BitSize();
    const auto raw=e.getRawPropertyValue(inst.mInstPtr,propertyId);
    if(raw.size()%stride) throw std::invalid_argument("Malformed native structure array");
    if(elementIndex>=raw.size()/stride) throw std::out_of_range("Array record index out of range");
    FakeInstanceHeader header(1,prop->mDatatype,prop->mDatatypeName,stride);
    header.mDataSectionStart=reinterpret_cast<uint64_t>(raw.begin()+size_t(elementIndex)*stride);
    RecordInfo result;result.classId=prop->mDatatype;
    for(const auto& field:m->descriptors(result.classId)) {
        RecordField value;value.id=field.mPropertyId;value.name=safeString(field.mName);
        value.type=safeString(field.mDatatypeName.mName);value.semantic=safeString(field.mSemantic);
        value.array=field.mPropertyType==PropertyType::Array;
        // Legacy query/contact records contain fixed fields. Their offsets and
        // native value conversion belong to the object model, not Python.
        if(value.array) continue;
        const auto size=e.getPropertyByteSize(&header,field.mPropertyId);
        if(size>8*1024*1024) throw std::length_error("Record field exceeds bounded transfer size");
        value.data.resize(size);
        if(size) value.data.resize(e.getPropertyValue(&header,field.mPropertyId,value.data.data(),size));
        if(value.type=="StringHandle" && value.data.size()==4) {uint32_t handle;std::memcpy(&handle,value.data.data(),4);value.stringValue=safeString(e.handleToString(handle));}
        result.fields.push_back(std::move(value));
    }
    return result;
}
std::vector<std::pair<std::string,uint32_t>> Session::namedValues(int32_t propertyId) {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);pvdapp::PvdEngine::LiveReadGuard guard(*m->engine);
    bool known=false;
    if(propertyId>=0) {
        std::vector<ClassDescription> classes(m->engine->getNbClasses());classes.resize(m->engine->getClasses(classes.data(),classes.size()));
        for(const auto& cls:classes) {const auto& props=m->descriptors(cls.mClassId);if(std::any_of(props.begin(),props.end(),[propertyId](const PropertyDescription& p){return p.mPropertyId==propertyId;})){known=true;break;}}
    }
    if(!known)throw std::out_of_range("Unknown property ID");
    const auto native=m->engine->getNamedPropertyValues(propertyId);
    std::vector<std::pair<std::string,uint32_t>> result;result.reserve(native.size());
    for(const auto& value:native) result.emplace_back(safeString(value.mName),value.mValue);
    return result;
}
ExportCapabilities Session::exportCapabilities() {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);pvdapp::PvdEngine::LiveReadGuard guard(*m->engine);
    return m->capabilities();
}
uint64_t Session::save(const std::string& path,bool overwrite) {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    const auto cap=m->capabilities();if(!cap.save) throw std::invalid_argument(cap.saveReason);
    const std::string source=m->engine->isLive()?m->engine->getLiveRecordingPath():m->sourcePath;
    if(sameFile(path,source))
        throw std::invalid_argument("Save requires a destination different from the source capture");
    if(m->engine->isLive() && m->engine->isOvdFile() && source.empty()) {
        m->engine->stopLive();
        OutputFile output(path,overwrite);CheckedOutputWriteStream stream(output);stream.openStream();
        const bool copied=m->engine->writeLiveRecording(stream);const bool closed=stream.closeStream();
        if(!copied || !closed)throw std::runtime_error("Native live recording copy failed");
        return output.commit();
    }
    pvdapp::PvdEngine::LiveReadGuard guard(*m->engine);OutputFile output(path,overwrite);
    if(m->engine->isOvdFile()) {
        m->engine->flushLiveRecording();
        std::ifstream input(std::filesystem::u8path(source),std::ios::binary);if(!input)throw std::runtime_error("Cannot open original OVD bytes");
        std::vector<uint8_t> buffer(1024*1024);
        while(input) {input.read(reinterpret_cast<char*>(buffer.data()),buffer.size());const auto n=input.gcount();if(n)output.write(buffer.data(),uint32_t(n));}
        if(!input.eof())throw std::runtime_error("OVD source read failed");
    } else m->engine->save(&output);
    return output.commit();
}
uint64_t Session::saveClip(const std::string& path,uint64_t sceneId,uint32_t start,uint32_t stop,bool overwrite) {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);pvdapp::PvdEngine::LiveReadGuard guard(*m->engine);
    const auto cap=m->capabilities();if(!cap.clip)throw std::invalid_argument(cap.clipReason);
    const auto scenes=m->sceneInfos();
    const auto scene=std::find_if(scenes.begin(),scenes.end(),[sceneId](const SceneInfo& s){return s.id==sceneId;});
    if(scene==scenes.end())throw std::out_of_range("Unknown scene ID");
    if(stop<=start || stop>scene->frameCount)throw std::out_of_range("Clip requires a nonempty [start, stop) range within the scene");
    if(sameFile(path,m->sourcePath))
        throw std::invalid_argument("Clip destination must differ from the source capture");
    auto release=[](pvdapp::PvdClient* c){if(c)c->release();};
    std::unique_ptr<pvdapp::PvdClient,decltype(release)> client(m->engine->crop(scene->streamId,start,stop-1,m->temporaryDirectory.c_str()),release);
    if(!client)throw std::runtime_error("Native clip creation failed");
    OutputFile output(path,overwrite);client->save(output);return output.commit();
}
uint64_t Session::exportRepX(const std::string& path,const std::vector<int32_t>& ids,bool all,bool overwrite) {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);pvdapp::PvdEngine::LiveReadGuard guard(*m->engine);
    const auto cap=m->capabilities();if(!cap.repx)throw std::invalid_argument(cap.repxReason);
    if(!all && ids.empty())throw std::invalid_argument("RepX selection is empty");
    std::vector<uint64_t> selected;selected.reserve(ids.size());
    for(const auto id:ids) {m->instance(id);selected.push_back(uint64_t(id));}
    if(sameFile(path,m->sourcePath))
        throw std::invalid_argument("RepX destination must differ from the source capture");
    OutputFile output(path,overwrite);m->engine->exportRepX(m->runtime->allocator,&output,all?nullptr:&selected);return output.commit();
}
uint64_t Session::messageCount() {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    pvdapp::PvdEngine::LiveReadGuard guard(*m->engine);
    return m->engine->getMessages().size();
}
std::vector<MessageInfo> Session::messages(uint32_t offset, uint32_t limit) {
    if(limit>1000) throw std::invalid_argument("Message page limit exceeds 1000");
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    pvdapp::PvdEngine::LiveReadGuard guard(*m->engine);
    const auto& native=m->engine->getMessages();
    const size_t end=std::min(native.size(),size_t(offset)+limit);
    std::vector<MessageInfo> result;
    if(offset<end) result.reserve(end-offset);
    auto text=[](const char* source,size_t capacity) { return std::string(source,std::find(source,source+capacity,'\0')); };
    for(size_t i=offset;i<end;++i) {
        const auto& msg=native[i];
        result.push_back({msg.frameId,text(msg.typeName,sizeof(msg.typeName)),text(msg.message,sizeof(msg.message)),
            text(msg.file,sizeof(msg.file)),msg.line,msg.type,msg.handle});
    }
    return result;
}
FrameTiming Session::frameTiming() {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    pvdapp::PvdEngine::LiveReadGuard guard(*m->engine);
    if(m->engine->isOvdFile()) return {};
    uint64_t numerator=0,denominator=0;
    m->engine->getTimestampNanosecondConversion(numerator,denominator);
    if(!denominator) return {};
    const double factor=double(numerator)/double(denominator)/1e6;
    return {true,m->engine->getInnerFrameTime()*factor,m->engine->getTotalFrameTime()*factor};
}
std::shared_ptr<ScalarHistory> Session::scalarHistory(uint64_t sceneId,int32_t instanceId,int32_t propertyId,uint32_t start,uint32_t stop) {
    if(stop<start || uint64_t(stop)-start>4096) throw std::invalid_argument("Scalar history requires at most 4096 samples and stop >= start");
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    auto& engine=*m->engine;
    pvdapp::PvdEngine::LiveReadGuard guard(engine);
    if(engine.isLive()) throw std::invalid_argument("Scalar history requires a file session; freeze or copy the recording first");
    const auto scenes=m->sceneInfos();
    const auto selected=std::find_if(scenes.begin(),scenes.end(),[sceneId](const SceneInfo& s){return s.id==sceneId;});
    if(selected==scenes.end()) throw std::out_of_range("Unknown scene ID");
    if(stop>selected->frameCount) throw std::out_of_range("History range is outside this scene");
    const uint32_t previous=selected->currentFrame;
    auto restore=[&] { engine.setCurrentFrame(previous,selected->streamId); m->index(); };
    auto result=std::make_shared<ScalarHistory>();
    try {
        auto known=m->instanceIndex.find(instanceId);
        // Legacy metadata at frame zero contains no created instances. Discover
        // the target inside the requested range before resolving its class schema.
        if(known==m->instanceIndex.end() && !engine.isOvdFile()) {
            for(uint32_t frame=start;frame<stop && known==m->instanceIndex.end();++frame) {
                engine.setCurrentFrame(frame,selected->streamId);m->index();known=m->instanceIndex.find(instanceId);
            }
        }
        if(known==m->instanceIndex.end()) throw std::out_of_range("Unknown instance ID");
        const size_t originalIndex=known->second;
        const auto& properties=m->descriptors(m->instances[originalIndex].mClassId);
        const auto prop=std::find_if(properties.begin(),properties.end(),[propertyId](const pvdapp::PropertyDescription& p){return p.mPropertyId==propertyId;});
        if(prop==properties.end()) throw std::out_of_range("Property does not belong to this instance");
        const std::string type=safeString(prop->mDatatypeName.mName);
        // Older PXD2 schemas use PxF32/PxU32/etc. for the same recorded
        // scalar layouts. Normalize only for decoding; retain the schema's
        // original type in the result and keep arrays/composites excluded.
        const std::string numericType=type.compare(0,2,"Px")==0 ? "Pvd"+type.substr(2) : type;
        const bool floating=numericType=="PvdF32" || numericType=="PvdF64";
        const bool signedInteger=numericType=="PvdI8" || numericType=="PvdI16" || numericType=="PvdI32" || numericType=="PvdI64";
        const bool unsignedInteger=numericType=="PvdU8" || numericType=="PvdU16" || numericType=="PvdU32" || numericType=="PvdU64" || numericType=="PvdBool";
        if(prop->mPropertyType==pvdapp::PropertyType::Array || (engine.isOvdFile() && engine.getOvdPropertyFieldCount(propertyId)!=1) || !(floating || signedInteger || unsignedInteger))
            throw std::invalid_argument("Scalar history supports numeric scalar properties only");
        result->propertyName=safeString(prop->mName); result->type=type;
        result->frames.reserve(stop-start); result->values.reserve(stop-start); result->valid.reserve(stop-start);
        for(uint32_t frame=start;frame<stop;++frame) {
            engine.setCurrentFrame(frame,selected->streamId);
            pvdapp::InstanceDescription instance;
            bool exists=false;
            if(engine.isOvdFile()) {
                exists=engine.getInstances(&instance,1,originalIndex)==1 && instance.mId==instanceId;
            } else {
                m->index();
                const auto found=m->instanceIndex.find(instanceId);
                if(found!=m->instanceIndex.end()) {instance=m->instances[found->second];exists=true;}
            }
            double value=std::numeric_limits<double>::quiet_NaN(); bool valid=false;
            if(exists && instance.mAlive && instance.mInstPtr) {
                const auto raw=engine.getRawPropertyValue(instance.mInstPtr,propertyId);
                const size_t n=raw.size();
                if(floating) {
                    if(numericType=="PvdF32" && n==4) { float x; std::memcpy(&x,raw.begin(),4); value=x; valid=std::isfinite(value); }
                    if(numericType=="PvdF64" && n==8) { std::memcpy(&value,raw.begin(),8); valid=std::isfinite(value); }
                } else if(n==1 || n==2 || n==4 || n==8) {
                    // Enum declarations may expose PvdU32 with a one-byte
                    // recorded flag width. Decode the recorded scalar width.
                    if(signedInteger) {
                        int64_t x=0;
                        if(n==1) {int8_t v;std::memcpy(&v,raw.begin(),1);x=v;}
                        else if(n==2) {int16_t v;std::memcpy(&v,raw.begin(),2);x=v;}
                        else if(n==4) {int32_t v;std::memcpy(&v,raw.begin(),4);x=v;}
                        else std::memcpy(&x,raw.begin(),8);
                        value=double(x);
                    } else { uint64_t x=0;std::memcpy(&x,raw.begin(),n);value=double(x); }
                    valid=true;
                }
            }
            result->frames.push_back(frame);result->values.push_back(value);result->valid.push_back(valid);
        }
    } catch(...) { restore(); throw; }
    restore();
    return result;
}
std::vector<SceneInfo> Session::scenes() {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    pvdapp::PvdEngine::LiveReadGuard guard(*m->engine);
    return m->sceneInfos();
}
std::vector<ClassInfo> Session::classes() {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    pvdapp::PvdEngine::LiveReadGuard guard(*m->engine);
    std::vector<pvdapp::ClassDescription> native(m->engine->getNbClasses());
    native.resize(m->engine->getClasses(native.data(), native.size()));
    std::vector<ClassInfo> result;
    for (const auto& cls : native) result.push_back({cls.mClassId, cls.mBaseClass, cls.get32BitSize(), safeString(cls.mName.mName)});
    return result;
}
std::vector<InstanceInfo> Session::instances(uint32_t offset, uint32_t limit) {
    if (limit > 10000) throw std::invalid_argument("Instance page limit exceeds 10000");
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    pvdapp::PvdEngine::LiveReadGuard guard(*m->engine);
    std::vector<InstanceInfo> result;
    const size_t end = std::min(m->instances.size(), size_t(offset) + limit);
    for (size_t i = offset; i < end; ++i) {
        const auto& inst = m->instances[i];
        result.push_back({inst.mId, inst.mClassId, m->engine->getOwningSceneId(inst.mId),
            m->engine->getOwningPhysicsId(inst.mId), inst.mAlive, m->engine->isOvdSceneJoint(inst.mId)});
    }
    return result;
}
std::vector<PropertyInfo> Session::properties(int32_t classId) {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    pvdapp::PvdEngine::LiveReadGuard guard(*m->engine);
    std::vector<PropertyInfo> result;
    for (const auto& prop : m->descriptors(classId)) result.push_back({prop.mPropertyId,
        safeString(prop.mName), safeString(prop.mDatatypeName.mName),
        m->engine->getOvdNativeTypeLabel(prop.mPropertyId), safeString(prop.mSemantic),
        prop.mPropertyType == pvdapp::PropertyType::Array || (m->engine->isOvdFile() && m->engine->getOvdPropertyFieldCount(prop.mPropertyId)==0 && safeString(prop.mDatatypeName.mName)!="StringHandle"),
        m->engine->getOvdPropertyFieldCount(prop.mPropertyId)});
    return result;
}
std::vector<uint8_t> Session::read(int32_t id, int32_t prop, uint32_t offset, uint32_t limit) {
    if (limit > 8 * 1024 * 1024) throw std::invalid_argument("Property page limit exceeds 8 MiB");
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    pvdapp::PvdEngine::LiveReadGuard guard(*m->engine);
    auto value = m->value(id, prop);
    if (offset >= value.size() || !limit) return {};
    const size_t count = std::min(size_t(limit), size_t(value.size()) - offset);
    return std::vector<uint8_t>(value.begin() + offset, value.begin() + offset + count);
}
uint32_t Session::propertySize(int32_t id, int32_t prop) {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    pvdapp::PvdEngine::LiveReadGuard guard(*m->engine);
    return m->value(id, prop).size();
}
std::vector<int32_t> Session::references(int32_t id, int32_t prop) {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    pvdapp::PvdEngine::LiveReadGuard guard(*m->engine);
    auto& instance = m->instance(id);
    const auto& props = m->descriptors(instance.mClassId);
    const auto found = std::find_if(props.begin(), props.end(), [prop](const pvdapp::PropertyDescription& p) {
        return p.mPropertyId == prop && std::strcmp(p.mDatatypeName.mName, "ObjectRef") == 0;
    });
    if (found == props.end()) throw std::invalid_argument("Property is not an object reference");
    auto size = m->engine->getPropertyByteSize(instance.mInstPtr, prop);
    if (!size) return {};
    if (size > 8 * 1024 * 1024) throw std::length_error("Reference list exceeds bounded transfer size");
    std::vector<int32_t> result((size + sizeof(int32_t) - 1) / sizeof(int32_t));
    auto bytes = m->engine->getPropertyValue(instance.mInstPtr, prop, result.data(), result.size() * sizeof(int32_t));
    result.resize(bytes / sizeof(int32_t));
    return result;
}
std::string Session::stringValue(int32_t id, int32_t prop) {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    pvdapp::PvdEngine::LiveReadGuard guard(*m->engine);
    auto& instance = m->instance(id);
    const auto& props = m->descriptors(instance.mClassId);
    const auto found = std::find_if(props.begin(), props.end(), [prop](const pvdapp::PropertyDescription& p) {
        return p.mPropertyId == prop && std::strcmp(p.mDatatypeName.mName, "StringHandle") == 0;
    });
    if (found == props.end()) throw std::invalid_argument("Property is not a string");
    uint32_t handle = 0;
    auto size = m->engine->getPropertyValue(instance.mInstPtr, prop, &handle, sizeof(handle));
    return size == sizeof(handle) ? safeString(m->engine->handleToString(handle)) : "";
}
bool Session::seek(uint64_t sceneId, uint32_t frame) {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    pvdapp::PvdEngine::LiveReadGuard guard(*m->engine);
    const auto scenes=m->sceneInfos();
    const auto selected=std::find_if(scenes.begin(),scenes.end(),[sceneId](const SceneInfo& s){return s.id==sceneId;});
    if(selected==scenes.end()) throw std::out_of_range("Unknown scene ID");
    if(frame>=selected->frameCount) throw std::out_of_range("Frame is outside this scene");
    if(m->engine->hasLiveUnpublishedIndexChanges() && !m->engine->isLiveEnded()) return false;
    m->engine->setCurrentFrame(frame, selected->streamId);
    m->index();
    return true;
}
CaptureControlInfo Session::captureControlInfo() {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    auto& e=*m->engine;
    CaptureControlInfo result;
    result.connected=e.isLiveConnected() && !e.isLiveEnded();
    result.capabilities=e.captureCapabilities(); result.catalog=e.captureCatalogSnapshot();
    result.metadata=e.captureMetadata(); result.transport=e.captureTransport(); result.budget=e.captureBudget();
    const auto now=std::chrono::steady_clock::now();
    m->captureRequests.observe(result.transport,result.metadata,now,result.connected);
    result.request=m->captureRequests.current(); result.results=m->captureRequests.history();
    result.issuedHighWater=m->captureRequests.highWater(); result.ackClassification=m->captureRequests.ackClassification();
    result.forwardBytes=e.getLiveRecordingSize(); result.completedBoundarySerial=e.getLiveMetadataBoundarySerial();
    result.hasCompletedFrame=e.getLiveMaxFrame()>0;
    if(result.forwardBytes!=m->captureForwardBytes) {
        m->captureForwardBytes=result.forwardBytes; m->captureForwardAt=now;
    }
    result.forwardIdleSeconds=std::chrono::duration<double>(now-m->captureForwardAt).count();
    if(!result.connected) result.reason="No connected live producer";
    else if(result.transport.state==pvdapp::CaptureTransportState::Unavailable)
        result.reason="Control transfer stalled; recording may continue. Reconnect the producer to restore control.";
    else if(result.issuedHighWater==UINT64_MAX) result.reason="Capture request IDs exhausted; reconnect the producer";
    else if(result.request.state=="sending" || result.request.state=="awaiting_ack") result.reason="A capture request is pending";
    if(result.reason.empty()) {
        result.canApply=(result.capabilities&4) && pvdapp::canApplyCapture(result.budget.used,result.budget.previousSegment,result.budget.limits);
        result.canRefresh=(result.capabilities&8) && pvdapp::canRefreshCapture(result.budget.used,result.budget.previousCatalogBytes,result.budget.limits);
        if(!(result.capabilities&12)) result.reason="Peer does not advertise capture control";
        else if((result.capabilities&4) && !result.canApply)
            result.reason="Recording budget is low; save and reconnect before changing selection";
    }
    return result;
}
pvdapp::CaptureCatalogPage Session::captureCatalog(uint64_t generation,uint64_t offset,uint32_t limit,
                                                  uint64_t parent,const std::string& prefix) {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    return m->engine->captureCatalogPage(generation,offset,limit,parent,prefix);
}
uint64_t Session::requestCapture(const std::vector<uint64_t>& scenes,const std::vector<uint64_t>& objects,
                                bool secondary,bool all,uint64_t generation,bool supersede) {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    const auto info=captureControlInfo();
    if(!info.canApply) throw std::runtime_error(info.reason.empty()?"Peer does not support selection":info.reason);
    pvdapp::CaptureRequest request; request.requestId=m->captureRequests.nextId(supersede);
    request.scenes=scenes; request.objects=objects; request.secondary=secondary; request.all=all;
    m->engine->queueCaptureRequest(request,generation);
    m->captureRequests.submitted(request,std::chrono::steady_clock::now());
    return request.requestId;
}
uint64_t Session::requestCaptureCatalog(bool supersede) {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    const auto info=captureControlInfo();
    if(!info.canRefresh) throw std::runtime_error(info.reason.empty()?"Peer does not support catalog refresh":info.reason);
    pvdapp::CaptureRequest request; request.requestId=m->captureRequests.nextId(supersede); request.catalogOnly=true;
    m->engine->queueCaptureRequest(request,0);
    m->captureRequests.submitted(request,std::chrono::steady_clock::now());
    return request.requestId;
}
LiveUpdate Session::refreshLive() {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    auto& e = *m->engine;
    pvdapp::PvdEngine::LiveReadGuard guard(e);
    LiveUpdate result;
    result.serial=e.getLiveCompletedFrameSerial();
    result.segmentTransitionSerial=e.getLiveMetadataBoundarySerial();
    result.currentFrame=e.getLiveCurrentFrame();
    result.maxFrame=e.getLiveMaxFrame();
    result.segmentEpoch=e.getOvdSegmentEpoch();
    if(!result.maxFrame || e.hasLiveUnpublishedIndexChanges()) return result;
    e.refreshLive();
    result.segmentEpoch=e.getOvdSegmentEpoch();
    if(result.serial != m->lastLiveIndexedSerial) {
        m->index(); result.changed=true; m->lastLiveIndexedSerial=result.serial;
    }
    result.scenes=m->sceneInfos();
    result.ready=!result.scenes.empty();
    return result;
}
void Session::refresh() {
    std::lock_guard<std::recursive_mutex> lock(m->mutex);
    pvdapp::PvdEngine::LiveReadGuard guard(*m->engine);
    if(m->engine->hasLiveUnpublishedIndexChanges()) return;
    m->engine->refreshLive();
    m->index();
    m->lastLiveIndexedSerial=m->engine->getLiveCompletedFrameSerial();
}
void Session::stopLive() { std::lock_guard<std::recursive_mutex> lock(m->mutex); m->engine->stopLive(); }
void Session::flushRecording() { std::lock_guard<std::recursive_mutex> lock(m->mutex); m->engine->flushLiveRecording(); }
std::string Session::recordingPath() { std::lock_guard<std::recursive_mutex> lock(m->mutex); return m->engine->getLiveRecordingPath(); }
} // namespace ovdnext
